"""3DGS with a per-image bilateral grid (gsplat v1.5.3 simple_trainer,
--use_bilateral_grid) on a MonoGS Replica-format dataset, as a group C method
(PROTOCOL.md, Nhóm C): the most common GENERIC appearance compensation, no
light model. Offline at the REFERENCE poses; no ATE, no FPS; rendering scored
on the held-out grid. Ends with ONE zip per run, scored by score_zip.py.

    python methods/bilagrid/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--iterations <n>, local tests only]

kaggle.json "reference_poses": true (the runner stages traj.txt). What is run:
gsplat's own examples/simple_trainer.py, "default" config (third_party/
gsplat_examples, the v1.5.3 tag; the gsplat library is pip gsplat==1.5.3),
exactly as its CLI:
    simple_trainer.py default --data_dir <package> --data_factor 1
        --result_dir <dir> --use_bilateral_grid --disable_viewer
on the same COLMAP package as DarkGS and SpotlessGS (methods/darkgs/
colmap_pkg.py, plus COLMAP's binary model for gsplat's pycolmap reader): the 0 grid at the
reference poses, points from sensor depth. Defaults otherwise (30000 steps,
SH degree 3, normalised world space, grid 16 x 16 x 8 per training image).

Test-time rendering, as gsplat's own eval(): the bilateral grids belong to
the TRAINING images, so a held-out frame is rendered from the Gaussians alone
(rasterize_splats, no grid). gsplat's eval also reports cc_psnr after an
affine colour correction fitted to the ground-truth image; that uses the
answer, so it is not computed here (README).

Changed on a COPY of third_party/gsplat_examples in the work dir (overrides.json):
- datasets/colmap.py trains on every image (upstream held out every 8th,
  always image 0, for its own validation);
- datasets/__init__.py added (empty): Kaggle has the Hugging Face 'datasets'
  package, which would shadow gsplat's datasets/ namespace package.
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(BASELINES, "common"), os.path.join(BASELINES, "methods", "darkgs")]
import colmap_pkg  # noqa: E402  (methods/darkgs)
import protocol  # noqa: E402
import runzip  # noqa: E402

METHOD = "bilagrid"
VENDORED = os.path.join(BASELINES, "third_party", "gsplat_examples")
GSPLAT_VERSION = "1.5.3"
SPLIT_LINE = "            self.indices = indices[indices % self.parser.test_every != 0]\n"
CKPT_LINE = '                data = {"step": step, "splats": self.splats.state_dict()}\n'
CKPT_PATCH = CKPT_LINE + ('                if cfg.use_bilateral_grid:  # baselines: keep the per-image grids (not saved upstream)\n'
                          '                    data["bil_grids"] = self.bil_grids.state_dict()\n')
SPLIT_PATCH = "            self.indices = indices  # baselines: train on every image (was: every test_every-th held out)\n"


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def patched_copy(dst):
    shutil.copytree(VENDORED, dst)
    path = os.path.join(dst, "datasets", "colmap.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(SPLIT_LINE) != 1:
        raise RuntimeError(f"{path}: expected one train-split line, found {text.count(SPLIT_LINE)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(SPLIT_LINE, SPLIT_PATCH))
    path = os.path.join(dst, "simple_trainer.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(CKPT_LINE) != 1:
        raise RuntimeError(f"{path}: expected one checkpoint line, found {text.count(CKPT_LINE)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(CKPT_LINE, CKPT_PATCH))
    init = os.path.join(dst, "datasets", "__init__.py")
    if os.path.exists(init):
        raise RuntimeError(f"{init} exists upstream: check the shadowing patch")
    open(init, "w").close()
    return dst


def trainer_args(pkg, result_dir, iterations):
    args = ["default", "--data_dir", pkg, "--data_factor", "1", "--result_dir", result_dir,
            "--use_bilateral_grid", "--disable_viewer"]
    if iterations:  # local tests only: the same schedule, shorter
        args += ["--max_steps", str(iterations), "--eval_steps", str(iterations), "--save_steps", str(iterations)]
    return args


def train(src, pkg, result_dir, iterations):
    cmd = [sys.executable, "-u", os.path.join(src, "simple_trainer.py")] + trainer_args(pkg, result_dir, iterations)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=src)
    seconds = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"gsplat simple_trainer.py failed (exit code {proc.returncode})")
    return seconds


def final_ckpt(result_dir):
    ckpts = glob.glob(os.path.join(result_dir, "ckpts", "ckpt_*_rank0.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {result_dir}/ckpts")
    step = lambda p: int(re.search(r"ckpt_(\d+)_rank0", p).group(1))
    return max(ckpts, key=step), max(step(p) for p in ckpts)


def load_runner(src, pkg, result_dir, iterations, ckpt):
    """simple_trainer's Runner with the training run's configuration and the
    final splats; the bilateral-grid names are put in the module as its
    __main__ block does."""
    import torch

    sys.path.insert(0, src)
    import lib_bilagrid
    import simple_trainer as st
    from gsplat.strategy import DefaultStrategy

    for name in ("BilateralGrid", "color_correct", "slice", "total_variation_loss"):
        setattr(st, name, getattr(lib_bilagrid, name))
    cfg = st.Config(strategy=DefaultStrategy(verbose=True), data_dir=pkg, data_factor=1,
                    result_dir=os.path.join(result_dir, "export_runner"), use_bilateral_grid=True, disable_viewer=True)
    if iterations:
        cfg.max_steps, cfg.eval_steps, cfg.save_steps = iterations, [iterations], [iterations]
    cfg.adjust_steps(cfg.steps_scaler)
    runner = st.Runner(0, 0, 1, cfg)
    data = torch.load(ckpt, map_location=runner.device, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = data["splats"][k]
    if "bil_grids" not in data:
        raise KeyError(f"{ckpt}: no bil_grids (the checkpoint patch of patched_copy did not apply)")
    runner.bil_grids.load_state_dict(data["bil_grids"])
    return runner, cfg


def nearest_train(i, train_frames):
    """The training frame nearest to frame i (the earlier one on a tie)."""
    return min(train_frames, key=lambda t: (abs(t - i), t))


def render_frames(runner, cfg, frames, c2w, renders_dir, train_frames, grid=False):
    """rasterize_splats at the reference poses (moved into the trainer's
    normalised world with its own transform_cameras), clamped to [0, 1].
    grid False: no bilateral grid, as eval(). grid True: the grid of the
    nearest training frame applied as in training (slice with that image's id)."""
    import torch
    from datasets.normalize import transform_cameras
    from lib_bilagrid import slice as bil_slice

    p = runner.parser
    cams = sorted(set(p.camera_ids))
    if len(cams) != 1:
        raise ValueError(f"expected one camera, got {cams}")
    K = torch.from_numpy(p.Ks_dict[cams[0]]).float().to(runner.device)
    W, H = p.imsize_dict[cams[0]]
    # the trainer's poses of the training images must be the reference poses, moved the same way
    names = {f"frame{i:06d}.jpg": i for i in train_frames}
    idx = [k for k, nm in enumerate(p.image_names) if nm in names]
    ours = transform_cameras(p.transform, np.stack([c2w[names[p.image_names[k]]] for k in idx]))
    err = np.abs(ours - p.camtoworlds[idx]).max()
    if err > 1e-4:
        raise ValueError(f"reference poses moved by transform_cameras differ from the trainer's by {err}")
    if len(runner.trainset) != len(p.image_names) or len(p.image_names) != len(train_frames):
        raise ValueError(f"trainset {len(runner.trainset)}, parser {len(p.image_names)}, training frames {len(train_frames)}")
    image_id = {names[nm]: k for k, nm in enumerate(p.image_names)}  # trainset indices are all images (patched split)
    gy, gx = torch.meshgrid((torch.arange(H, device=runner.device) + 0.5) / H,
                            (torch.arange(W, device=runner.device) + 0.5) / W, indexing="ij")
    grid_xy = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    with torch.no_grad():
        for j, i in enumerate(frames):
            cw = torch.from_numpy(transform_cameras(p.transform, c2w[i][None].copy())).float().to(runner.device)
            colors, _, _ = runner.rasterize_splats(camtoworlds=cw, Ks=K[None], width=W, height=H, sh_degree=cfg.sh_degree,
                                                   near_plane=cfg.near_plane, far_plane=cfg.far_plane)
            if grid:
                ids = torch.tensor([[image_id[nearest_train(i, train_frames)]]], device=runner.device)
                colors = bil_slice(runner.bil_grids, grid_xy, colors, ids)["rgb"]
            np.save(os.path.join(renders_dir, f"{i:06d}.npy"),
                    colors.clamp(0.0, 1.0)[0].cpu().numpy().astype(np.float16))
            if (j + 1) % 50 == 0 or j + 1 == len(frames):
                print(f"export: rendered {j + 1}/{len(frames)}" + (" (nearest training grid)" if grid else ""), flush=True)
    return err


def cmd_run(args):
    work = os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("bilagrid is run in one go here: use --checkpoint_every 0 and no --resume")
    if os.path.exists(os.path.join(work, "export")) or os.path.exists(os.path.join(work, "data")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    # One GPU: gsplat's cli() spawns one process per visible GPU (Kaggle T4 x2), which is a
    # different run (twice the batch, its docs scale the steps) and fails here (the spawned
    # processes do not run simple_trainer's __main__ block, which defines BilateralGrid).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible = "0"
    elif len([d for d in visible.split(",") if d.strip()]) != 1:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={visible!r}: run bilagrid on one GPU (gsplat's single-GPU default)")
    import gsplat

    if gsplat.__version__ != GSPLAT_VERSION:
        raise SystemExit(f"gsplat {gsplat.__version__} installed, the vendored trainer is v{GSPLAT_VERSION}")
    scene, variant = scene_variant(args.dataset)
    traj = os.path.join(args.dataset, "traj.txt")
    if not os.path.isfile(traj):
        raise SystemExit(f"{traj}: missing. Group C needs the reference poses (kaggle.json reference_poses, PROTOCOL.md)")
    os.makedirs(work, exist_ok=True)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    if len(depths) != n:
        raise ValueError(f"{args.dataset}: {n} colour, {len(depths)} depth frames")
    c2w = protocol.gt_poses_c2w(args.dataset)
    if len(c2w) != n:
        raise ValueError(f"{traj}: {len(c2w)} poses for {n} frames")
    src = patched_copy(os.path.join(work, "gsplat_src"))
    train_frames = list(range(0, n, protocol.RENDER_INTERVAL))

    pkg = os.path.join(work, "data", f"{scene}_{variant}")
    print(f"gsplat {GSPLAT_VERSION} simple_trainer + bilateral grid on {args.dataset}: {len(train_frames)} training "
          f"frames of {n}; building the COLMAP package at the reference poses", flush=True)
    points = colmap_pkg.build(pkg, train_frames, colors, depths, c2w, cal, binary=True)
    print(f"package: {points['points']} points ({points['raw_points']} back-projected)", flush=True)
    overrides = {
        "poses": "reference poses (traj.txt), PROTOCOL.md Nhóm C",
        "training_frames": f"0 grid: {len(train_frames)} frames (0, {protocol.RENDER_INTERVAL}, ..., {train_frames[-1]})",
        "point_cloud": f"sensor depth of the training frames at the reference poses, voxel {points['voxel_m']} m "
                       "(methods/darkgs/colmap_pkg.py; gsplat initialises from COLMAP SfM points)",
        "simple_trainer.py": "default " + " ".join(trainer_args("<package>", "<dir>", args.iterations)[1:]),
        "gpus": f"CUDA_VISIBLE_DEVICES={visible}: one GPU (gsplat's cli() would spawn one process per GPU)",
        "datasets/colmap.py": "train on every image (upstream: every 8th held out, always image 0), on a copy",
        "datasets/__init__.py": "added (empty) on the copy: the Hugging Face 'datasets' package would shadow it",
        "simple_trainer.py checkpoint": "also saves the bilateral grids (bil_grids; upstream saves the splats only), on a copy",
        "test-time": "renders/: Gaussians alone (no bilateral grid: it belongs to the training images), as gsplat's eval(); "
                     "renders_nngrid/: held-out frames with the grid of the nearest training frame (2 frames away); its "
                     "colour-corrected cc_* metrics (fitted to the ground truth) not computed",
    }
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = os.path.join(work, "gsplat_out")
    train_s = train(src, pkg, result_dir, args.iterations)
    ckpt, step = final_ckpt(result_dir)
    print(f"final checkpoint: {ckpt} (step {step})", flush=True)

    import torch
    runner, cfg = load_runner(src, pkg, result_dir, args.iterations, ckpt)
    if step != cfg.max_steps - 1:
        raise ValueError(f"last checkpoint at step {step}, training ran {cfg.max_steps} steps")
    out = os.path.join(work, "export")
    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir)
    frames = sorted(set(protocol.required_renders(n, train_frames))
                    | set(protocol.rendering_frames(n, train_frames, protocol.HELDOUT_OFFSET)))
    pose_err = render_frames(runner, cfg, frames, c2w, renders_dir, train_frames)
    nn_dir = os.path.join(out, "renders_nngrid")
    os.makedirs(nn_dir)
    render_frames(runner, cfg, frames, c2w, nn_dir, train_frames, grid=True)  # figure frames are training frames: own grid
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in train_frames) + "\n")
    runzip.write_figures(renders_dir, colors, os.path.join(out, "figures"))
    n_gauss = int(runner.splats["means"].shape[0])
    n_heldout = len(protocol.rendering_frames(n, train_frames, protocol.HELDOUT_OFFSET))
    gpu = torch.cuda.get_device_name(0)
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": train_s, "gpu": f"{gpu}, 1 GPU",
        "what": (f"gsplat simple_trainer.py, {cfg.max_steps} steps on {len(train_frames)} images at the reference poses, "
                 "with a bilateral grid per image (process wall time, including its first-run CUDA compilation if "
                 "any). Not a SLAM loop: no FPS (PROTOCOL.md, Nhóm C)"),
        "sessions": 1})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"3DGS + bilateral grid (gsplat {GSPLAT_VERSION} simple_trainer default, --use_bilateral_grid) at the reference "
             f"poses, {len(train_frames)} training frames (0 grid), {cfg.max_steps} steps; {n_gauss} Gaussians. Held-out "
             f"frames rendered from the Gaussians alone (no grid), as gsplat's eval, in renders/ (scored on the held-out "
             f"grid, {n_heldout} frames), and with the nearest training frame's grid in renders_nngrid/ (score separately). Trainer poses vs reference poses after normalisation: max diff {pose_err:.1e}.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": f"gsplat v{GSPLAT_VERSION}", "torch": torch.__version__, "frames": n,
        "keyframes": len(train_frames), "renders": len(glob.glob(os.path.join(renders_dir, "*.npy"))),
        "gaussians": n_gauss, "given_poses": "reference", "heldout_offset": protocol.HELDOUT_OFFSET,
        "rendering_frames_offset": n_heldout, "iterations": cfg.max_steps, "point_cloud": points,
        "overrides": overrides, "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{len(train_frames)} training frames, {n_gauss} Gaussians; {n_heldout} held-out frames; training "
          f"{train_s:.0f} s -> {out}", flush=True)
    map_file = None
    if not args.no_map:
        map_file = os.path.join(work, os.path.basename(ckpt))
        shutil.copyfile(ckpt, map_file)
    runzip.package(work, ("overrides.json",), map_file=map_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder with results/ and traj.txt")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0")
    p.add_argument("--resume", action="store_true", help="refused")
    p.add_argument("--no_map", action="store_true", help="leave the final checkpoint out of the zip")
    p.add_argument("--iterations", type=int, default=None,
                   help="LOCAL TESTS ONLY: fewer training steps (default: gsplat's own, as on Kaggle)")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
