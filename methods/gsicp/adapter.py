"""GS-ICP SLAM (Ha et al., ECCV 2024) on a MonoGS Replica-format dataset.
Runs on Kaggle with the baselines repo alone and ends with ONE zip per run,
scored later on the local machine by baselines/common/score_zip.py
(baselines/PROTOCOL.md). Contract of common/kaggle_runner.py.

    python methods/gsicp/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--gsicp <source, default third_party/GS_ICP_SLAM>]

GS-ICP SLAM has no checkpoints (a run takes minutes): --checkpoint_every
must be 0, --resume is refused.

What is run: GS-ICP's gs_icp_slam.py (tracking limited to 30 FPS, the
README's first "Run" mode, as replica.sh), with replica.sh's parameters for
Replica, read from that script. What differs, all recorded in
<work>/overrides.json and README.md:
- camera: protocol.calibration() written into a copy of GS-ICP's
  configs/Replica/caminfo.txt (depth_trunc and dataset type kept);
- downsample_rate, a pixel stride for the points GICP tracks and the map
  starts from, is scaled by the image width (published value at 1200 px
  wide -> same point spacing at the dataset width), so that the half
  resolution does not also cut the number of points by four;
- input folders: GS-ICP reads <dataset>/images/frame*.jpg and
  depth_images/depth*.png; the adapter stages per-file links to results/;
- traj.txt: GS-ICP takes its first pose from it and prints an ATE with it.
  The staged one holds IDENTITY matrices: no reference pose reaches the run
  (the first pose becomes the identity, which the SE(3) alignment absorbs);
- GS-ICP's own final evaluation over every frame is replaced by the export
  (hooks.py); its scene.ply is kept as the final map.
"""
import argparse
import datetime
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASELINES, "common"))
import protocol  # noqa: E402
import runzip  # noqa: E402

METHOD = "gsicp"
VENDORED = os.path.join(BASELINES, "third_party", "GS_ICP_SLAM")
CAMINFO = os.path.join("configs", "Replica", "caminfo.txt")
REPLICA_SCRIPT = "replica.sh"
# The SLAM parameters replica.sh sets for every Replica scene (shell
# variables there, passed as --<name>; knn_maxd is the CLI name of knn_max).
REPLICA_PARAMS = ("keyframe_th", "knn_maxd", "overlapped_th", "max_correspondence_distance",
                  "trackable_opacity_th", "overlapped_th2", "downsample_rate")
RUN_FILES = ("caminfo.txt", "overrides.json", "params.json")
OUT_NAME = "gsicp_out"  # GS-ICP's --output_path under <work>


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def vendor_commit(src):
    path = os.path.join(src, ".vendor.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: the commit of the GS-ICP source is unknown")
    return runzip.read_json(path)["commit"]


# ---------------------------------------------------------------- inputs

def stage(dataset, work, cal):
    """<work>/data/<seq>/{images,depth_images}/ with one link per frame into
    the dataset's results/, and a traj.txt of identity matrices."""
    from PIL import Image

    colors, depths = protocol.dataset_frames(dataset)
    if len(depths) != len(colors):
        raise ValueError(f"{dataset}: {len(colors)} colour and {len(depths)} depth frames")
    for path in (colors[0], depths[0]):
        w, h = Image.open(path).size
        if (w, h) != (cal["width"], cal["height"]):
            raise ValueError(f"{path}: {w}x{h}, calibration says {cal['width']}x{cal['height']}")
    seq_dir = os.path.join(work, "data", "_".join(scene_variant(dataset)))
    for sub, files, prefix in (("images", colors, "frame"), ("depth_images", depths, "depth")):
        d = os.path.join(seq_dir, sub)
        os.makedirs(d)
        for f in files:
            name = os.path.basename(f)
            if not re.fullmatch(prefix + r"\d{6}\.(jpg|png)", name):
                raise ValueError(f"{f}: unexpected name for GS-ICP's loader")
            os.symlink(os.path.abspath(f), os.path.join(d, name))
    eye = " ".join(f"{v:.1f}" for v in np.eye(4).ravel())
    with open(os.path.join(seq_dir, "traj.txt"), "w") as f:
        f.write((eye + "\n") * len(colors))
    return seq_dir, len(colors)


def caminfo(src, work, cal):
    """GS-ICP's configs/Replica/caminfo.txt with the dataset camera. Line 3:
    W H fx fy cx cy depth_scale depth_trunc dataset_type (gs_icp_slam.py reads
    [0] as W, [1] as H). Returns the path and the published values."""
    with open(os.path.join(src, CAMINFO), encoding="utf-8") as f:
        lines = f.read().splitlines()
    vals = lines[2].split()
    if len(vals) != 9 or vals[8] != "replica":
        raise ValueError(f"{CAMINFO}: unexpected line 3 {lines[2]!r}")
    published = {"width": int(vals[0]), "height": int(vals[1]), "depth_trunc": float(vals[7])}
    new = [str(cal["width"]), str(cal["height"]), str(cal["fx"]), str(cal["fy"]), str(cal["cx"]), str(cal["cy"]),
           str(cal["depth_scale"]), vals[7], vals[8]]
    lines[2] = " ".join(new)
    path = os.path.join(work, "caminfo.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path, published


def replica_params(src):
    """The parameter values replica.sh assigns (shell `name=value` lines)."""
    with open(os.path.join(src, REPLICA_SCRIPT), encoding="utf-8") as f:
        text = f.read()
    params = {}
    for name in REPLICA_PARAMS:
        m = re.findall(rf"^{name}=([0-9.eE+-]+)\s*$", text, flags=re.M)
        if len(m) != 1:
            raise ValueError(f"{REPLICA_SCRIPT}: expected one '{name}=<number>' line, found {m}")
        params[name] = m[0]
    if "gs_icp_slam.py" not in text or "--save_results" not in text:
        raise ValueError(f"{REPLICA_SCRIPT}: no longer runs gs_icp_slam.py with --save_results")
    return params


# ---------------------------------------------------------------- run

def cmd_run(args):
    src, work = os.path.abspath(args.gsicp), os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("GS-ICP SLAM has no checkpoints (one run takes minutes): "
                         "use --checkpoint_every 0 and no --resume")
    out_dir = os.path.join(work, OUT_NAME)
    if os.path.exists(out_dir) or os.path.exists(os.path.join(work, "export")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    os.makedirs(work, exist_ok=True)
    cal = protocol.calibration()
    commit = vendor_commit(src)
    scene, variant = scene_variant(args.dataset)
    print(f"GS-ICP SLAM {commit} on {args.dataset}", flush=True)

    seq_dir, n = stage(args.dataset, work, cal)
    cam_path, published = caminfo(src, work, cal)
    params = replica_params(src)
    rate_pub = int(params["downsample_rate"])
    rate = rate_pub * cal["width"] / published["width"]
    if rate != int(rate) or rate < 1:
        raise ValueError(f"downsample_rate {rate_pub} at {published['width']} px does not scale to an integer "
                         f"stride at {cal['width']} px ({rate})")
    params["downsample_rate"] = str(int(rate))
    overrides = {
        "camera": {k: cal[k] for k in ("width", "height", "fx", "fy", "cx", "cy", "depth_scale")},
        "camera_published": f"{CAMINFO}: {published}",
        "downsample_rate": f"{rate_pub} at {published['width']} px -> {int(rate)} at {cal['width']} px",
        "input": "per-file links images/, depth_images/ -> results/",
        "traj.txt": "identity matrices (no reference poses)",
        "final_evaluation": "GS-ICP's calc_2d_metric replaced by the export (hooks.ExportMapper)",
    }
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)
    runzip.write_json(os.path.join(work, "params.json"), {"replica.sh": params, "script": "gs_icp_slam.py"})

    # GS-ICP as `python gs_icp_slam.py <replica.sh arguments>`, with the
    # export subclasses in place of its Tracker and Mapper.
    for p in (HERE, src):
        if p not in sys.path:
            sys.path.insert(0, p)
    import gs_icp_slam as G
    import hooks

    G.Tracker, G.Mapper = hooks.ExportTracker, hooks.ExportMapper
    cli = argparse.Namespace(
        dataset_path=seq_dir, config=cam_path, output_path=out_dir, verbose=False, demo=False, test=None,
        save_results=True, rerun_viewer=False, **params)
    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    slam = G.GS_ICP_SLAM(cli)
    slam.run()

    hook = os.path.join(out_dir, hooks.HOOK_DIR)
    for f in ("tracker_timing.json", "mapper_state.json", "poses_c2w.npy"):
        if not os.path.isfile(os.path.join(hook, f)):
            raise RuntimeError(f"GS-ICP ended without {f} in {hook}: see the output above")
    print("SLAM done, exporting", flush=True)
    export(args, work, hook, n, cal, commit, scene, variant, started_at, params)
    ply = os.path.join(out_dir, "scene.ply")
    runzip.package(work, RUN_FILES + (os.path.join(OUT_NAME, hooks.HOOK_DIR, "tracker_timing.json"),
                                      os.path.join(OUT_NAME, hooks.HOOK_DIR, "mapper_state.json")),
                   map_file=None if args.no_map else ply)


def export(args, work, hook, n, cal, commit, scene, variant, started_at, params):
    import shutil

    import torch

    out = os.path.join(work, "export")
    os.makedirs(out)
    tt = runzip.read_json(os.path.join(hook, "tracker_timing.json"))
    ms = runzip.read_json(os.path.join(hook, "mapper_state.json"))
    if tt["frames"] != n or ms["image_size"] != [cal["width"], cal["height"]]:
        raise ValueError(f"GS-ICP saw {tt['frames']} frames at {ms['image_size']}, dataset has {n} at "
                         f"{cal['width']}x{cal['height']}")
    poses = np.load(os.path.join(hook, "poses_c2w.npy"))
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"poses: shape {poses.shape}, finite {np.isfinite(poses).all()}")
    protocol.write_tum(os.path.join(out, "traj_est.tum"), {i: poses[i] for i in range(n)})
    keyframes = ms["keyframes"]
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")
    frames = protocol.required_renders(n, keyframes)
    shutil.move(os.path.join(hook, "renders"), os.path.join(out, "renders"))
    missing = [i for i in frames if not os.path.isfile(os.path.join(out, "renders", f"{i:06d}.npy"))]
    if missing:
        raise FileNotFoundError(f"renders missing for frames {missing[:10]}...")
    colors, _ = protocol.dataset_frames(args.dataset)
    runzip.write_figures(os.path.join(out, "renders"), colors, os.path.join(out, "figures"))

    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": tt["loop_s"], "gpu": tt["gpu"] + ", 1 GPU (cuda:0; tracking with fast_gicp on CPU)",
        "what": ("GS-ICP's tracking loop, first to last frame (its own 'System FPS'), with mapping running "
                 "alongside in its own process and stopping at the end of tracking; tracking is capped at 30 FPS "
                 "(gs_icp_slam.py, as replica.sh). Excludes the preload of every frame into memory before the "
                 f"loop ({tt['preload_s']:.1f} s), the scene.ply save and the export"),
        "sessions": 1, "preload_s": tt["preload_s"]})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"GS-ICP SLAM gs_icp_slam.py (tracking capped at 30 FPS) with replica.sh parameters; changed: camera "
             f"(MonoGS calibration), downsample_rate scaled with the image width ({params['downsample_rate']}), "
             "identity traj.txt (no reference poses; first pose = identity), own final evaluation replaced by the "
             f"export. Mapping: {ms['mapping_iterations']} iterations during tracking, on its keyframes.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "torch": torch.__version__,
        "frames": n, "keyframes": len(keyframes), "renders": len(frames), "gaussians": ms["gaussians"],
        "mapping_iterations": ms["mapping_iterations"], "params": params,
        "overrides": runzip.read_json(os.path.join(work, "overrides.json")), "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{n} poses, {len(keyframes)} keyframes, {len(frames)} renders, {ms['gaussians']} Gaussians, "
          f"{n / tt['loop_s']:.1f} FPS -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0: GS-ICP has no checkpoints")
    p.add_argument("--resume", action="store_true", help="refused: GS-ICP has no checkpoints")
    p.add_argument("--no_map", action="store_true", help="leave the final map (scene.ply) out of the zip")
    p.add_argument("--gsicp", default=VENDORED, help="GS-ICP source tree (default: the pinned copy in third_party/)")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
