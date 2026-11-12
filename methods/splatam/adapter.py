"""SplaTAM (Keetha et al., CVPR 2024) on a MonoGS Replica-format dataset. Runs
on Kaggle with the baselines repo alone (no MonoGS repo, no MonoGS
environment: SplaTAM's rasterizer installs under the same module name as
MonoGS's) and ends with ONE zip per run, scored later on the local machine
by baselines/common/score_zip.py (baselines/PROTOCOL.md).

    python methods/splatam/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every <n, 0 = off> [--resume] [--no_map] [--splatam <source, default third_party/SplaTAM>]
    python methods/splatam/adapter.py export  --dataset <...> --work <dir> [--splatam <source>]
    python methods/splatam/adapter.py package --dataset <...> --work <dir> [--no_map]

`run` = stage the input, build the config, run SLAM (timed), `export`, `package`.
`export` = params.npz -> <work>/export/: traj_est.tum, keyframes.txt,
  renders/<idx:06d>.npy, figures/, timing.json, calibration.json, method.json,
  notes.txt.
`package` = <work>/../splatam__<scene>_<variant>__<start time>.zip: export/,
  run/ (configs, timing segments, log, final map params.npz unless --no_map)
  and summary.json.

What differs from SplaTAM's own Replica run (configs/replica/splatam.py),
all recorded in <work>/overrides.json and README.md:
- resolution, intrinsics, depth scale: from protocol.calibration() (a copy
  of MonoGS's Dataset.Calibration, checked against it at scoring), written
  into a copy of SplaTAM's configs/data/replica.yaml (other keys kept);
- paths, run name; W&B logging off (score_zip.py logs the scored run);
- traj.txt: SplaTAM's loader reads one pose per frame. The staged dataset
  gives it IDENTITY matrices, so no reference pose can reach the run (the
  real traj.txt is only read when scoring). SplaTAM's online ATE printouts
  are therefore meaningless; tracking itself never reads them
  (tracking.use_gt_poses is False, checked);
- SplaTAM's final self-evaluation is skipped (the scoring is score.py's);
  its call marks the end of the timed SLAM loop;
- optional periodic checkpoints (SplaTAM's own save_checkpoints), older ones
  pruned, for resuming after a Kaggle session ends.
"""
import argparse
import copy
import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "common"))
import protocol  # noqa: E402

METHOD = "splatam"
# The pinned SplaTAM copy of the baselines repo (third_party/README.md); its
# .vendor.json gives the commit.
VENDORED_SPLATAM = os.path.join(os.path.dirname(os.path.dirname(HERE)), "third_party", "SplaTAM")
SPLATAM_CONFIG = os.path.join("configs", "replica", "splatam.py")
SPLATAM_DATA_CONFIG = os.path.join("configs", "data", "replica.yaml")
# SplaTAM's top-level folders are namespace packages (their init files are
# named `_init_.py`), so any installed package of the same name, e.g. the
# Hugging Face `datasets` on Kaggle, would shadow them. import_splatam()
# registers them explicitly.
SPLATAM_PACKAGES = ("datasets", "utils", "scripts")
CKPT_RE = re.compile(r"^params(\d+)\.npz$")
KF_RE = re.compile(r"^keyframe_time_indices(\d+)\.npy$")
# Files of <work> that go into run/ of the zip (besides the log and the map).
RUN_FILES = ("config_resolved.json", "overrides.json", "data_replica.yaml", "timing_segments.json")
LOG_NAME = "run.log"  # common/kaggle_runner.run writes the adapter's output to <work>/run.log


# ---------------------------------------------------------------- helpers

def scene_variant(dataset):
    """(scene, variant) as score.py names a run: the dataset's parent folder
    and folder."""
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def sequence_name(dataset):
    return "_".join(scene_variant(dataset))


def import_splatam(splatam):
    splatam = os.path.abspath(splatam)
    for name in SPLATAM_PACKAGES:
        path = os.path.join(splatam, name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path}: not a SplaTAM source tree?")
        mod = sys.modules.get(name)
        if mod is not None:
            if list(getattr(mod, "__path__", [])) != [path]:
                raise ImportError(f"module {name!r} already imported from {getattr(mod, '__file__', mod)}")
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
    if splatam not in sys.path:
        sys.path.insert(0, splatam)


def load_py_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_dotted(cfg, key, value):
    """cfg['a']['b'] = value for key 'a.b'; the key must already exist in
    SplaTAM's config (a typo would otherwise add an ignored entry)."""
    *parents, last = key.split(".")
    d = cfg
    for p in parents:
        d = d[p]
    if last not in d:
        raise KeyError(f"SplaTAM config has no {key!r}")
    d[last] = value


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def checkpoints(run_dir):
    """Frame indices of the COMPLETE checkpoints (params<i>.npz together with
    keyframe_time_indices<i>.npy) and all checkpoint files by index."""
    params, kfs = {}, {}
    if os.path.isdir(run_dir):
        for f in os.listdir(run_dir):
            m = CKPT_RE.match(f)
            if m:
                params[int(m.group(1))] = os.path.join(run_dir, f)
            m = KF_RE.match(f)
            if m:
                kfs[int(m.group(1))] = os.path.join(run_dir, f)
    return sorted(set(params) & set(kfs)), params, kfs


def method_commit(splatam):
    """The SplaTAM commit: from the .vendor.json of the copy in
    baselines/third_party, or from git for a clone."""
    vendor = os.path.join(splatam, ".vendor.json")
    if os.path.isfile(vendor):
        return read_json(vendor)["commit"]
    if os.path.isdir(os.path.join(splatam, ".git")):
        return subprocess.check_output(["git", "-C", splatam, "rev-parse", "HEAD"], text=True).strip()
    raise FileNotFoundError(f"{splatam}: neither .vendor.json nor a git clone, the commit is unknown")


def run_dir_of(work, dataset):
    return os.path.join(work, "splatam", sequence_name(dataset))


# ---------------------------------------------------------------- staging

def stage(dataset, work, n_frames, cal):
    """<work>/data/<seq>/results -> the dataset's results/, and a traj.txt of
    identity matrices (see the module docstring)."""
    from PIL import Image

    colors, depths = protocol.dataset_frames(dataset)
    if len(depths) != len(colors):
        raise ValueError(f"{dataset}: {len(colors)} colour and {len(depths)} depth frames")
    for path in (colors[0], depths[0]):
        w, h = Image.open(path).size
        if (w, h) != (cal["width"], cal["height"]):
            raise ValueError(f"{path}: {w}x{h}, calibration says {cal['width']}x{cal['height']}")
    seq = sequence_name(dataset)
    root = os.path.join(work, "data")
    seq_dir = os.path.join(root, seq)
    os.makedirs(seq_dir, exist_ok=True)
    src = os.path.abspath(os.path.join(dataset, "results"))
    link = os.path.join(seq_dir, "results")
    if os.path.islink(link):
        if os.path.realpath(link) != os.path.realpath(src):
            raise FileExistsError(f"{link} points to {os.path.realpath(link)}, not {src}")
    elif os.path.exists(link):
        raise FileExistsError(f"{link} exists and is not a link")
    else:
        os.symlink(src, link, target_is_directory=True)
    eye = " ".join(f"{v:.1f}" for v in np.eye(4).ravel())
    with open(os.path.join(seq_dir, "traj.txt"), "w") as f:
        f.write((eye + "\n") * n_frames)
    return root, seq


def data_config(splatam, work, cal):
    """SplaTAM's configs/data/replica.yaml with the camera of the MonoGS
    dataset; every other key kept."""
    import yaml

    with open(os.path.join(splatam, SPLATAM_DATA_CONFIG), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cam = cfg["camera_params"]
    for key, value in (("image_height", cal["height"]), ("image_width", cal["width"]),
                       ("fx", cal["fx"]), ("fy", cal["fy"]), ("cx", cal["cx"]), ("cy", cal["cy"]),
                       ("png_depth_scale", cal["depth_scale"])):
        if key not in cam:
            raise KeyError(f"{SPLATAM_DATA_CONFIG}: camera_params has no {key!r}")
        cam[key] = value
    path = os.path.join(work, "data_replica.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def build_config(splatam, work, data_root, seq, data_cfg_path, cal, checkpoint_every, resume_idx):
    base = load_py_module(os.path.join(splatam, SPLATAM_CONFIG), "splatam_replica_config").config
    cfg = copy.deepcopy(base)
    overrides = {
        "workdir": os.path.join(work, "splatam"),
        "run_name": seq,
        "use_wandb": False,
        "data.basedir": data_root,
        "data.sequence": seq,
        "data.gradslam_data_cfg": data_cfg_path,
        "data.desired_image_height": cal["height"],
        "data.desired_image_width": cal["width"],
        "save_checkpoints": checkpoint_every > 0,
        "load_checkpoint": resume_idx is not None,
        "checkpoint_time_idx": resume_idx if resume_idx is not None else base["checkpoint_time_idx"],
    }
    if checkpoint_every > 0:
        overrides["checkpoint_interval"] = checkpoint_every
    for key, value in overrides.items():
        set_dotted(cfg, key, value)
    if cfg["tracking"]["use_gt_poses"]:
        raise ValueError("SplaTAM config tracks with ground-truth poses")
    if cfg["data"]["num_frames"] != -1 or cfg["data"]["start"] != 0 or cfg["data"]["end"] != -1 \
            or cfg["data"]["stride"] != 1:
        raise ValueError(f"SplaTAM config does not use every frame: {cfg['data']}")
    return cfg, overrides


# ---------------------------------------------------------------- SLAM

def run_slam(splatam, cfg, segments_path, segments):
    """SplaTAM's rgbd_slam() with its final eval replaced by a time mark and
    its checkpoint saver wrapped (time marks, pruning). Appends this session
    to the segments file: wall seconds from the call to each checkpoint and
    to the end of the last frame."""
    import_splatam(splatam)
    import torch

    import scripts.splatam as S  # SplaTAM's main script, as `python scripts/splatam.py` runs it

    run_dir = os.path.join(cfg["workdir"], cfg["run_name"])
    seg = {"start_frame": cfg["checkpoint_time_idx"] if cfg["load_checkpoint"] else 0,
           "started_at": datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
           "gpu": torch.cuda.get_device_name(0), "checkpoint_seconds": {}, "end_seconds": None}
    segments.append(seg)
    t0 = None
    save_ckpt = S.save_params_ckpt

    def timed_checkpoint(params, output_dir, time_idx):
        save_ckpt(params, output_dir, time_idx)
        seg["checkpoint_seconds"][str(time_idx)] = time.time() - t0
        write_json(segments_path, segments)
        # Keep the newest COMPLETE checkpoint before this one (this one's
        # keyframe list is written after this call returns).
        complete, params_files, kf_files = checkpoints(output_dir)
        keep = max([i for i in complete if i < time_idx], default=None)
        for files in (params_files, kf_files):
            for i, path in files.items():
                if i < time_idx and i != keep:
                    os.remove(path)

    def end_of_loop(*args, **kwargs):  # replaces SplaTAM's final eval()
        torch.cuda.synchronize()
        seg["end_seconds"] = time.time() - t0
        write_json(segments_path, segments)

    S.save_params_ckpt = timed_checkpoint
    S.eval = end_of_loop
    S.seed_everything(seed=cfg["seed"])
    write_json(segments_path, segments)
    t0 = time.time()
    S.rgbd_slam(cfg)
    if seg["end_seconds"] is None:
        raise RuntimeError("SplaTAM's loop ended without reaching its final eval")
    if not os.path.isfile(os.path.join(run_dir, "params.npz")):
        raise FileNotFoundError(f"no params.npz in {run_dir}")


def cmd_run(args):
    splatam, work = os.path.abspath(args.splatam), os.path.abspath(args.work)
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint_every: frames between checkpoints, 0 = off")
    cal = protocol.calibration()
    colors, _ = protocol.dataset_frames(args.dataset)
    run_dir = run_dir_of(work, args.dataset)
    segments_path = os.path.join(work, "timing_segments.json")
    complete, params_files, _ = checkpoints(run_dir)
    if args.resume:
        if not complete:
            raise SystemExit(f"--resume: no complete checkpoint in {run_dir}")
        resume_idx = complete[-1]
        segments = read_json(segments_path)
        if segments[-1]["end_seconds"] is not None:
            raise SystemExit(f"{work}: the run already finished; `export` and `package` it instead")
        print(f"resuming from the checkpoint of frame {resume_idx} (frame {resume_idx} is processed again)")
    else:
        if params_files or os.path.isfile(os.path.join(run_dir, "params.npz")) or os.path.isfile(segments_path):
            raise SystemExit(f"{work} already holds a run: --resume it or use a new --work")
        resume_idx, segments = None, []
    os.makedirs(work, exist_ok=True)
    data_root, seq = stage(args.dataset, work, len(colors), cal)
    data_cfg_path = data_config(splatam, work, cal)
    cfg, overrides = build_config(splatam, work, data_root, seq, data_cfg_path, cal,
                                  args.checkpoint_every, resume_idx)
    write_json(os.path.join(work, "overrides.json"), overrides)
    write_json(os.path.join(work, "config_resolved.json"), cfg)
    run_slam(splatam, cfg, segments_path, segments)
    cmd_export(args)
    cmd_package(args)


# ---------------------------------------------------------------- export

def timing(segments, n_frames):
    """timing.json for score.py from the sessions of one run: a session that
    was resumed from counts up to the checkpoint the next one started from."""
    seconds = 0.0
    for seg, nxt in zip(segments, segments[1:]):
        key = str(nxt["start_frame"])
        if key not in seg["checkpoint_seconds"]:
            raise ValueError(f"session from frame {seg['start_frame']} has no time for checkpoint {key}")
        seconds += seg["checkpoint_seconds"][key]
    if segments[-1]["end_seconds"] is None:
        raise ValueError("the last session did not finish")
    seconds += segments[-1]["end_seconds"]
    gpus = sorted({s["gpu"] for s in segments})
    what = ("wall clock of SplaTAM's rgbd_slam() from the call (dataset index, first-frame init) to the end "
            "of the last frame's mapping (cuda synchronized), before its final eval (skipped) and the "
            "params.npz save; includes its default progress reports every 500 frames and the checkpoint saves")
    if len(segments) > 1:
        what += (f"; RESUMED {len(segments) - 1}x: sessions summed up to the checkpoint each next one "
                 "started from, that frame processed twice")
    return {"frames": n_frames, "seconds": seconds, "gpu": " / ".join(gpus) + ", 1 GPU (cuda:0)",
            "what": what, "sessions": len(segments)}


def write_figures(renders_dir, colors, out):
    """figures/<idx>_render.png and _obs.jpg of protocol.FIGURE_FRAMES, as
    score.py writes them (for a look at the run before scoring)."""
    from PIL import Image

    os.makedirs(out, exist_ok=True)
    for i in protocol.FIGURE_FRAMES:
        if i >= len(colors):
            continue
        image = protocol.load_render(os.path.join(renders_dir, f"{i:06d}.npy"))
        Image.fromarray(protocol.to_png8(image)).save(os.path.join(out, f"{i:06d}_render.png"))
        shutil.copy(colors[i], os.path.join(out, f"{i:06d}_obs.jpg"))


def cmd_export(args):
    import torch
    import torch.nn.functional as F

    splatam, work = os.path.abspath(args.splatam), os.path.abspath(args.work)
    import_splatam(splatam)
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer
    from utils.recon_helpers import setup_camera
    from utils.slam_external import build_rotation
    from utils.slam_helpers import transform_to_frame, transformed_params2rendervar

    cal = protocol.calibration()
    colors, _ = protocol.dataset_frames(args.dataset)
    n = len(colors)
    scene, variant = scene_variant(args.dataset)
    run_dir = run_dir_of(work, args.dataset)
    out = os.path.join(work, "export")
    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir, exist_ok=True)

    with np.load(os.path.join(run_dir, "params.npz")) as z:
        npz = {k: z[k] for k in z.files}
    K = npz["intrinsics"]
    if (int(npz["org_width"]), int(npz["org_height"])) != (cal["width"], cal["height"]) or not np.allclose(
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], [cal["fx"], cal["fy"], cal["cx"], cal["cy"]]):
        raise ValueError(f"params.npz camera {npz['org_width']}x{npz['org_height']}, K={K.tolist()} "
                         f"differs from the calibration {cal}")
    keys = ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales",
            "cam_unnorm_rots", "cam_trans")
    P = {k: torch.tensor(npz[k]).cuda().float() for k in keys}
    if P["cam_trans"].shape[-1] != n:
        raise ValueError(f"params.npz has {P['cam_trans'].shape[-1]} poses for {n} frames")
    first_w2c = npz["w2c"].astype(np.float64)

    # Poses: SplaTAM renders frame t with the Gaussians moved by its camera
    # parameters (transform_to_frame) and then viewed through the first-frame
    # camera, so its world-to-camera is first_w2c @ rel_w2c(t).
    with torch.no_grad():
        q = F.normalize(P["cam_unnorm_rots"][0].T)  # (N, 4), w x y z
        rel = torch.eye(4, device="cuda").repeat(n, 1, 1)
        rel[:, :3, :3] = build_rotation(q)
        rel[:, :3, 3] = P["cam_trans"][0].T
    rel = rel.cpu().double().numpy()
    poses = {t: np.linalg.inv(first_w2c @ rel[t]) for t in range(n)}
    protocol.write_tum(os.path.join(out, "traj_est.tum"), poses)
    keyframes = sorted(int(k) for k in npz["keyframe_time_indices"])
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")

    # Renders of the final map at the estimated poses, op for op as SplaTAM's
    # eval() renders colour.
    cam = setup_camera(cal["width"], cal["height"], K, first_w2c)
    frames = protocol.required_renders(n, keyframes)
    with torch.no_grad():
        for i in frames:
            transformed = transform_to_frame(P, i, gaussians_grad=False, camera_grad=False)
            rendervar = transformed_params2rendervar(P, transformed)
            im, _, _ = Renderer(raster_settings=cam)(**rendervar)
            np.save(os.path.join(renders_dir, f"{i:06d}.npy"),
                    im.permute(1, 2, 0).cpu().numpy().astype(np.float16))
    write_figures(renders_dir, colors, os.path.join(out, "figures"))

    segments = read_json(os.path.join(work, "timing_segments.json"))
    write_json(os.path.join(out, "timing.json"), timing(segments, n))
    write_json(os.path.join(out, "calibration.json"), cal)
    overrides = read_json(os.path.join(work, "overrides.json"))
    from importlib.metadata import PackageNotFoundError, version

    try:
        raster = version("diff_gaussian_rasterization")
    except PackageNotFoundError:
        raster = "unknown"
    notes = ("SplaTAM configs/replica/splatam.py defaults; changed: resolution, intrinsics, depth scale, paths "
             "(MonoGS calibration), W&B off, identity traj.txt (no reference poses), final self-eval skipped"
             + (f", checkpoints every {overrides['checkpoint_interval']} frames" if overrides["save_checkpoints"]
                else "")
             + ". Maps EVERY frame (map_every=1): the rendering-metric frames were mapping targets.")
    method = {"method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
              "started_at": segments[0]["started_at"], "commit": method_commit(splatam),
              "torch": torch.__version__, "diff_gaussian_rasterization": raster,
              "frames": n, "keyframes": len(keyframes), "renders": len(frames),
              "gaussians": int(npz["means3D"].shape[0]), "overrides": overrides, "notes": notes}
    write_json(os.path.join(out, "method.json"), method)
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{n} poses, {len(keyframes)} keyframes, {len(frames)} renders, "
          f"{method['gaussians']} Gaussians -> {out}")


# ---------------------------------------------------------------- package

def cmd_package(args):
    """One zip per run, everything score_zip.py needs and the files to
    analyse it: export/ at the top level of the zip, run/ with the configs,
    timing segments, log and (unless --no_map) the final map."""
    work = os.path.abspath(args.work)
    out = os.path.join(work, "export")
    method = read_json(os.path.join(out, "method.json"))
    timing_ = read_json(os.path.join(out, "timing.json"))
    name = f"{METHOD}__{method['scene']}_{method['variant']}__{method['started_at']}"
    zip_path = os.path.join(os.path.dirname(work), name + ".zip")
    members = []  # (path on disk, path in the zip)
    for root, _, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(root, f)
            members.append((p, os.path.relpath(p, out).replace(os.sep, "/")))
    for f in RUN_FILES + (LOG_NAME,):
        p = os.path.join(work, f)
        if os.path.isfile(p):
            members.append((p, "run/" + f))
        elif f != LOG_NAME:
            raise FileNotFoundError(p)
    params = os.path.join(run_dir_of(work, args.dataset), "params.npz")
    if not args.no_map:
        members.append((params, "run/params.npz"))
    summary = {
        "method": METHOD, "scene": method["scene"], "variant": method["variant"],
        "started_at": method["started_at"], "commit": method["commit"],
        "frames": method["frames"], "keyframes": method["keyframes"], "renders": method["renders"],
        "gaussians": method["gaussians"], "fps": timing_["frames"] / timing_["seconds"],
        "seconds": timing_["seconds"], "sessions": timing_["sessions"], "gpu": timing_["gpu"],
        "map_included": not args.no_map,
        "score": "python baselines/common/score_zip.py --zip <this zip> --dataset <local dataset folder> --wandb",
        "files": sorted(arc for _, arc in members) + ["summary.json"],
    }
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in members:
            z.write(p, f"{name}/{arc}")
        z.writestr(f"{name}/summary.json", json.dumps(summary, indent=2))
    os.replace(tmp, zip_path)
    print(f"{len(members) + 1} files, {os.path.getsize(zip_path) / 2**20:.0f} MiB -> {zip_path}")
    print(json.dumps({k: summary[k] for k in ("frames", "keyframes", "gaussians", "fps", "sessions")}))
    return zip_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "export", "package"):
        p = sub.add_parser(name)
        if name != "package":
            p.add_argument("--splatam", default=VENDORED_SPLATAM,
                           help="SplaTAM source tree (default: the pinned copy in third_party/; or a git clone)")
        p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
        p.add_argument("--work", required=True, help="output folder of this run")
        if name == "run":
            p.add_argument("--checkpoint_every", type=int, required=True,
                           help="frames between checkpoints (SplaTAM's checkpoint_interval), 0 = off")
            p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --work")
        if name != "export":
            p.add_argument("--no_map", action="store_true", help="leave the final map (params.npz) out of the zip")
    args = ap.parse_args(argv)
    {"run": cmd_run, "export": cmd_export, "package": cmd_package}[args.cmd](args)


if __name__ == "__main__":
    main()
