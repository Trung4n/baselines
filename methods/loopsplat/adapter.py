"""LoopSplat (Zhu et al., 3DV 2025) on a MonoGS Replica-format dataset. Runs
on Kaggle with the baselines repo alone and ends with ONE zip per run, scored
later on the local machine by baselines/common/score_zip.py (PROTOCOL.md).
Contract of common/kaggle_runner.py.

    python methods/loopsplat/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--loopsplat <source, default third_party/LoopSplat>]

LoopSplat's checkpoint_path option only reloads finished runs: no resume, so
--checkpoint_every must be 0 and --resume is refused.

What is run: GaussianSLAM(config).run() of LoopSplat's run_slam.py, in this
process, with configs/Replica/<scene>.yaml (inheriting replica.yaml); its
Evaluator is not run (scoring is score.py's). What differs, all recorded in
<work>/overrides.json and README.md:
- cam: H, W, fx, fy, cx, cy, depth_scale from protocol.calibration();
- data: input_path = a staged folder (results/ linked, traj.txt of IDENTITY
  matrices: LoopSplat takes frames 0 and 1 from traj.txt and logs ATEs with
  it; with identities no reference pose reaches the run, and frame 1 starts
  at frame 0's pose), output_path, scene_name; use_wandb off;
- faiss: the GPU build when installed; else faiss_shim (exact torch GPU
  search, same neighbours).
Exported: estimated_c2w (every frame; frame 0 = identity), keyframes = the
mapping frames of all submaps, renders of the final submaps (after loop
closure corrections, before LoopSplat's offline global refinement) at the
estimated poses: frame i is rendered with the submap it belongs to (the last
submap starting at or before i), as LoopSplat's Evaluator renders keyframes.

Render frames: protocol.required_renders, plus every 5th frame from frame 2
that is not a keyframe. LoopSplat maps frames 0, 5, 10, ... (map_every 5), so
the protocol's rendering frames (0, 5, 10, ... minus keyframes) are all
training views and drop out; PROTOCOL.md (2026-11-13) scores such a method on
the held-out grid protocol.HELDOUT_OFFSET, declared in method.json
(heldout_offset), which score_zip.py passes to score.py.
"""
import argparse
import datetime
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASELINES, "common"))
import protocol  # noqa: E402
import runzip  # noqa: E402

METHOD = "loopsplat"
VENDORED = os.path.join(BASELINES, "third_party", "LoopSplat")
SCENE_CONFIG = os.path.join("configs", "Replica", "{scene}.yaml")


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def stage(dataset, work, n_frames):
    """<work>/data/<seq>/results -> the dataset's results/, traj.txt of
    identity matrices."""
    seq_dir = os.path.join(work, "data", "_".join(scene_variant(dataset)))
    os.makedirs(seq_dir)
    os.symlink(os.path.abspath(os.path.join(dataset, "results")), os.path.join(seq_dir, "results"),
               target_is_directory=True)
    eye = " ".join(f"{v:.1f}" for v in np.eye(4).ravel())
    with open(os.path.join(seq_dir, "traj.txt"), "w") as f:
        f.write((eye + "\n") * n_frames)
    return seq_dir


def offset_frames(n, keyframes):
    """The frames score.py scores for this method (PROTOCOL.md, held-out grid)."""
    return protocol.rendering_frames(n, keyframes, protocol.HELDOUT_OFFSET)


def build_config(src, scene, seq_dir, out_dir, cal):
    """LoopSplat's scene config (inherit_from is relative to its root) with
    the overrides; refuses settings that would use reference poses."""
    from src.utils.io_utils import load_config

    path = SCENE_CONFIG.format(scene=scene)
    if not os.path.isfile(os.path.join(src, path)):
        raise FileNotFoundError(f"{os.path.join(src, path)}: LoopSplat has no Replica config for scene {scene!r}")
    cwd = os.getcwd()
    os.chdir(src)
    try:
        config = load_config(path)
    finally:
        os.chdir(cwd)
    overrides = {}
    for key, value in (("H", cal["height"]), ("W", cal["width"]), ("fx", cal["fx"]), ("fy", cal["fy"]),
                       ("cx", cal["cx"]), ("cy", cal["cy"]), ("depth_scale", cal["depth_scale"])):
        if key not in config["cam"]:
            raise KeyError(f"LoopSplat config has no cam.{key}")
        overrides[f"cam.{key}"] = f"{config['cam'][key]} -> {value}"
        config["cam"][key] = value
    for key, value in (("input_path", seq_dir), ("output_path", out_dir), ("scene_name", scene)):
        overrides[f"data.{key}"] = f"{config['data'].get(key)} -> {value}"
        config["data"][key] = value
    overrides["use_wandb"] = f"{config['use_wandb']} -> False"
    config["use_wandb"] = False
    if config["tracking"]["gt_camera"] or config["tracking"]["odometry_type"] == "gt" \
            or config["lc"]["registration"]["method"] == "gt" or config.get("frame_limit", -1) != -1:
        raise ValueError("LoopSplat config uses reference poses or a frame limit")
    return config, overrides


def render_submaps(out_dir, estimated_c2w, frames, renders_dir, cal):
    """Renders each frame with the submap it belongs to, as Evaluator.run_rendering_eval
    renders keyframes. Returns (keyframes, submap starts, Gaussian count of the last submap)."""
    import torch
    from argparse import ArgumentParser

    from src.entities.arguments import OptimizationParams
    from src.entities.gaussian_model import GaussianModel
    from src.utils.utils import get_render_settings, render_gaussian_model

    paths = sorted(glob.glob(os.path.join(out_dir, "submaps", "*.ckpt")))
    submaps = []
    for p in paths:
        ck = torch.load(p, map_location="cuda", weights_only=False)
        kfs = sorted(int(k) for k in ck["submap_keyframes"])
        submaps.append((min(kfs), p, kfs))
    submaps.sort()
    keyframes = sorted({k for _, _, kfs in submaps for k in kfs})
    starts = [s for s, _, _ in submaps]
    intr = np.array([[cal["fx"], 0, cal["cx"]], [0, cal["fy"], cal["cy"]], [0, 0, 1]])
    opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
    by_submap = {}
    for i in frames(keyframes):
        j = max(k for k, s in enumerate(starts) if s <= i)
        by_submap.setdefault(j, []).append(i)
    done, total, n_gauss = 0, sum(len(v) for v in by_submap.values()), 0
    for j, idxs in sorted(by_submap.items()):
        ck = torch.load(submaps[j][1], map_location="cuda", weights_only=False)
        model = GaussianModel()
        model.training_setup(opt)
        model.restore_from_params(ck["gaussian_params"], opt)
        n_gauss = max(n_gauss, int(model.get_xyz().shape[0]))
        with torch.no_grad():
            for i in idxs:
                w2c = np.linalg.inv(estimated_c2w[i])
                out = render_gaussian_model(model, get_render_settings(cal["width"], cal["height"], intr, w2c))
                np.save(os.path.join(renders_dir, f"{i:06d}.npy"),
                        out["color"].permute(1, 2, 0).cpu().numpy().astype(np.float16))
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"export: rendered {done}/{total}", flush=True)
        del model
        torch.cuda.empty_cache()
    return keyframes, starts, n_gauss


def cmd_run(args):
    src, work = os.path.abspath(args.loopsplat), os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("LoopSplat cannot resume a run: use --checkpoint_every 0 and no --resume")
    out_dir = os.path.join(work, "loopsplat_out")
    if os.path.exists(out_dir) or os.path.exists(os.path.join(work, "export")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    os.makedirs(work, exist_ok=True)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    if len(depths) != n:
        raise ValueError(f"{args.dataset}: {n} colour, {len(depths)} depth frames")
    scene, variant = scene_variant(args.dataset)
    commit = runzip.read_json(os.path.join(src, ".vendor.json"))["commit"]
    # LoopSplat (torch 2.1) torch.load()s its own submap checkpoints; torch >= 2.6
    # defaults to weights_only=True. These are files this run writes itself.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    for p in (src, os.path.join(src, "thirdparty", "Hierarchical-Localization")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import faiss
    import torch

    import faiss_shim
    faiss_backend = faiss_shim.install(faiss)
    print(f"LoopSplat {commit} on {args.dataset}: {n} frames; faiss backend: {faiss_backend}", flush=True)

    seq_dir = stage(args.dataset, work, n)
    config, overrides = build_config(src, scene, seq_dir, out_dir, cal)
    overrides["faiss"] = faiss_backend
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)
    runzip.write_json(os.path.join(work, "config_resolved.json"), config)

    from src.entities.gaussian_slam import GaussianSLAM
    from src.utils.utils import setup_seed

    setup_seed(config["seed"])
    t_init0 = time.time()
    slam = GaussianSLAM(config)  # dataset index, NetVLAD weights (downloaded once), output folders
    init_s = time.time() - t_init0
    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.cuda.synchronize()
    t0 = time.time()
    slam.run()
    torch.cuda.synchronize()
    loop_s = time.time() - t0
    print(f"SLAM done in {loop_s:.0f} s, exporting", flush=True)

    out = os.path.join(work, "export")
    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir)
    est = torch.load(os.path.join(out_dir, "estimated_c2w.ckpt"), map_location="cpu", weights_only=False)
    est = est.numpy().astype(np.float64)
    if est.shape != (n, 4, 4) or not np.isfinite(est).all():
        raise ValueError(f"estimated_c2w: shape {est.shape}, finite {np.isfinite(est).all()}")
    protocol.write_tum(os.path.join(out, "traj_est.tum"), {i: est[i] for i in range(n)})
    keyframes, starts, n_gauss = render_submaps(
        out_dir, est, lambda kfs: sorted(set(protocol.required_renders(n, kfs)) | set(offset_frames(n, kfs))),
        renders_dir, cal)
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")
    runzip.write_figures(renders_dir, colors, os.path.join(out, "figures"))
    n_protocol = len(protocol.rendering_frames(n, keyframes))
    n_offset = len(offset_frames(n, keyframes))
    gpu = torch.cuda.get_device_name(0)
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": loop_s, "gpu": f"{gpu}, 1 GPU",
        "what": ("GaussianSLAM.run(), first to last frame: frame reading (its dataset loader, inside the loop as MonoGS "
                 "reads its frames), tracking, mapping, submap starts and loop closures, cuda synchronized. Excludes "
                 f"construction ({init_s:.1f} s: dataset index, NetVLAD weights) and the export"),
        "sessions": 1, "init_s": init_s})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"LoopSplat configs/Replica/{scene}.yaml with the MonoGS camera, W&B off, identity traj.txt (frames 0-1 "
             f"start at the identity, no reference poses), faiss: {faiss_backend}. {len(starts)} submaps, "
             f"{len(keyframes)} keyframes (mapping frames, every 5th + submap starts). Renders: final submaps after "
             "loop closure, before the offline global refinement. Protocol rendering frames left after excluding "
             f"keyframes: {n_protocol}; scored instead on the held-out grid ({protocol.HELDOUT_OFFSET} mod "
             f"{protocol.RENDER_INTERVAL}, PROTOCOL.md): {n_offset} frames.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "torch": torch.__version__, "faiss": faiss_backend,
        "frames": n, "keyframes": len(keyframes), "submaps": len(starts), "submap_starts": starts,
        "renders": len(glob.glob(os.path.join(renders_dir, "*.npy"))), "gaussians": n_gauss,
        "rendering_frames_protocol": n_protocol, "rendering_frames_offset": n_offset,
        "heldout_offset": protocol.HELDOUT_OFFSET, "overrides": overrides, "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{n} poses, {len(keyframes)} keyframes in {len(starts)} submaps; protocol render frames {n_protocol}, "
          f"offset {n_offset}; {n / loop_s:.2f} FPS -> {out}", flush=True)
    map_file = None
    if not args.no_map:  # the submaps are the map: pack them as one archive
        import tarfile

        map_file = os.path.join(work, "submaps.tar")
        with tarfile.open(map_file, "w") as t:
            t.add(os.path.join(out_dir, "submaps"), arcname="submaps")
    runzip.package(work, ("overrides.json", "config_resolved.json",
                          os.path.join("loopsplat_out", "estimated_c2w.ckpt")), map_file=map_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0: LoopSplat cannot resume")
    p.add_argument("--resume", action="store_true", help="refused: LoopSplat cannot resume")
    p.add_argument("--no_map", action="store_true", help="leave the final submaps out of the zip")
    p.add_argument("--loopsplat", default=VENDORED, help="LoopSplat source (default: the pinned copy in third_party/)")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
