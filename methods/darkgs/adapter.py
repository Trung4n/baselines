"""DarkGS (Zhang et al., IROS 2024) on a MonoGS Replica-format dataset, as a
group C method (PROTOCOL.md, Nhóm C): a 3DGS map with a calibrated co-located
light, built offline at the REFERENCE poses; no ATE, no FPS, rendering scored
on the held-out grid. Runs on Kaggle with the baselines repo alone and ends
with ONE zip per run, scored by baselines/common/score_zip.py.

    python methods/darkgs/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--iterations <n>, local tests only]

kaggle.json "reference_poses": true makes the runner stage traj.txt next to
results/ (only group C). What is run: DarkGS's own train.py (the user's fork
in third_party/darkgs, unchanged), with
- a COLMAP package of the training frames (colmap_pkg.py): the 0 grid
  (0, 5, ..., 1995), reference poses, a point cloud from their sensor depth
  with normals (instead of COLMAP SfM);
- --linearize (the frames are sRGB 8-bit, DarkGS models linear intensity),
  --scaling_factor 1.0 (metric poses), --light_params
  methods/darkgs/light/<scene>_<variant>.pth: the lamp fitted from this
  project's light CALIBRATION (fit_light.py, D106), not Blender's values;
- lietorch replaced by lietorch_shim (the light rotation is frozen in
  training), the rest of DarkGS's defaults (30000 iterations, warm-up of the
  light pose, densification).
Then the final model (chkpnt<it>.pth, shader<it>.pth) is rendered through
DarkGS's renderer and shader at the reference poses of the held-out grid
(2, 7, 12, ...) and the figure frames, converted to sRGB (the observed
frames' encoding), and packed.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASELINES, "common"))
import protocol  # noqa: E402
import runzip  # noqa: E402

METHOD = "darkgs"
DARKGS = os.path.join(BASELINES, "third_party", "darkgs")
SHIM = os.path.join(HERE, "lietorch_shim")
LIGHT_DIR = os.path.join(HERE, "light")


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def darkgs_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([SHIM, DARKGS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def default_iterations():
    sys.path[:0] = [SHIM, DARKGS]
    from argparse import ArgumentParser

    from arguments import OptimizationParams

    parser = ArgumentParser()
    op = OptimizationParams(parser)
    return int(op.extract(parser.parse_args([])).iterations)


def train(pkg, model_dir, light, iterations, log_every_s=60):
    """DarkGS's train.py, unchanged, in a child process; returns the seconds."""
    cmd = [sys.executable, "-u", os.path.join(DARKGS, "train.py"), "-s", pkg, "-m", model_dir,
           "--linearize", "--scaling_factor", "1.0", "--light_params", light,
           "--iterations", str(iterations), "--save_iterations", str(iterations),
           "--checkpoint_iterations", str(iterations)]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=DARKGS, env=darkgs_env())
    seconds = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"DarkGS train.py failed (exit code {proc.returncode})")
    return seconds


def load_model(model_dir, iterations):
    """The trained Gaussians and shader, as train.py saved them."""
    import torch
    from argparse import ArgumentParser

    sys.path[:0] = [SHIM, DARKGS]
    from arguments import ModelParams, OptimizationParams
    from scene.gaussian_model import GaussianModel
    from scene.shading import ShadingModel

    parser = ArgumentParser()
    mp, op = ModelParams(parser), OptimizationParams(parser)
    args = parser.parse_args([])
    gaussians = GaussianModel(mp.extract(args).sh_degree)
    params, it = torch.load(os.path.join(model_dir, f"chkpnt{iterations}.pth"), weights_only=False)
    if it != iterations:
        raise ValueError(f"checkpoint at iteration {it}, expected {iterations}")
    gaussians.restore(params, op.extract(args))
    saved = torch.load(os.path.join(model_dir, f"shader{iterations}.pth"), weights_only=False)
    shader = ShadingModel(light="1DMLP").cuda()
    shader.load_state_dict(saved["model_state_dict"])
    r = saved["so3"].detach().reshape(-1).tolist()
    shader.light.set_r_vec((r[0], r[1], r[2]))
    return gaussians, shader


def render_frames(gaussians, shader, frames, c2w, cal, renders_dir):
    """DarkGS's renderer at the reference poses; linear -> sRGB (utils.image_utils.linear_to_srgb, the fork's
    inverse of its --linearize), clamped to [0, 1]."""
    import torch
    from argparse import ArgumentParser

    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    from utils.image_utils import linear_to_srgb

    parser = ArgumentParser()
    pp = PipelineParams(parser)
    pipe = pp.extract(parser.parse_args([]))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    W, H = cal["width"], cal["height"]
    fovx, fovy = focal2fov(cal["fx"], W), focal2fov(cal["fy"], H)
    blank = torch.zeros(3, H, W)
    with torch.no_grad():
        for j, i in enumerate(frames):
            R_w2c = c2w[i][:3, :3].T
            t_w2c = -R_w2c @ c2w[i][:3, 3]
            cam = Camera(colmap_id=i, R=R_w2c.T, T=t_w2c, FoVx=fovx, FoVy=fovy, image=blank, gt_alpha_mask=None,
                         image_name=f"frame{i:06d}", uid=i)
            lin = render(cam, gaussians, pipe, background, shader)["render"]
            srgb = linear_to_srgb(lin.clamp(0.0, 1.0))
            np.save(os.path.join(renders_dir, f"{i:06d}.npy"), srgb.permute(1, 2, 0).cpu().numpy().astype(np.float16))
            if (j + 1) % 50 == 0 or j + 1 == len(frames):
                print(f"export: rendered {j + 1}/{len(frames)}", flush=True)


def cmd_run(args):
    work = os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("DarkGS is run in one go here: use --checkpoint_every 0 and no --resume")
    if os.path.exists(os.path.join(work, "export")) or os.path.exists(os.path.join(work, "data")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    scene, variant = scene_variant(args.dataset)
    light = os.path.join(LIGHT_DIR, f"{scene}_{variant}.pth")
    if not os.path.isfile(light):
        raise SystemExit(f"{light}: no calibrated lamp for {scene}/{variant} (fit_light.py; the 'even' variants have "
                         "no co-located light, README)")
    traj = os.path.join(args.dataset, "traj.txt")
    if not os.path.isfile(traj):
        raise SystemExit(f"{traj}: missing. DarkGS needs the reference poses (kaggle.json reference_poses, PROTOCOL.md)")
    os.makedirs(work, exist_ok=True)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    if len(depths) != n:
        raise ValueError(f"{args.dataset}: {n} colour, {len(depths)} depth frames")
    c2w = protocol.gt_poses_c2w(args.dataset)
    if len(c2w) != n:
        raise ValueError(f"{traj}: {len(c2w)} poses for {n} frames")
    commit = runzip.read_json(os.path.join(DARKGS, ".vendor.json"))["commit"]
    train_frames = list(range(0, n, protocol.RENDER_INTERVAL))  # the 0 grid (PROTOCOL.md, Nhóm C)
    iterations = args.iterations or default_iterations()

    pkg = os.path.join(work, "data", f"{scene}_{variant}")
    print(f"DarkGS {commit[:8]} on {args.dataset}: {len(train_frames)} training frames of {n}, {iterations} iterations; "
          "building the COLMAP package at the reference poses", flush=True)
    import colmap_pkg
    points = colmap_pkg.build(pkg, train_frames, colors, depths, c2w, cal)
    print(f"package: {points['points']} points ({points['raw_points']} back-projected), "
          f"{points['geometric_normals']:.1%} with a geometric normal", flush=True)
    overrides = {
        "poses": "reference poses (traj.txt), PROTOCOL.md Nhóm C",
        "training_frames": f"0 grid: {len(train_frames)} frames (0, {protocol.RENDER_INTERVAL}, ..., {train_frames[-1]})",
        "point_cloud": f"sensor depth of the training frames at the reference poses, voxel {points['voxel_m']} m, normals "
                       "from depth (DarkGS uses COLMAP SfM points)",
        "light_params": f"methods/darkgs/light/{scene}_{variant}.pth: fitted from calib/light_calibrated.json (D106)",
        "train.py": f"--linearize --scaling_factor 1.0 --iterations {iterations} (DarkGS default otherwise)",
        "lietorch": "lietorch_shim (SO3 in torch; the light rotation is frozen in DarkGS's training)",
    }
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_dir = os.path.join(work, "darkgs_out")
    train_s = train(pkg, model_dir, light, iterations)

    import torch
    gaussians, shader = load_model(model_dir, iterations)
    out = os.path.join(work, "export")
    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir)
    frames = sorted(set(protocol.required_renders(n, train_frames))
                    | set(protocol.rendering_frames(n, train_frames, protocol.HELDOUT_OFFSET)))
    render_frames(gaussians, shader, frames, c2w, cal, renders_dir)
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in train_frames) + "\n")
    runzip.write_figures(renders_dir, colors, os.path.join(out, "figures"))
    n_gauss = int(gaussians.get_xyz.shape[0])
    n_heldout = len(protocol.rendering_frames(n, train_frames, protocol.HELDOUT_OFFSET))
    gpu = torch.cuda.get_device_name(0)
    learned = {"ambient_light": float(shader.ambient_light.item()), "scaling_factor": float(shader.scaling_factor.item())}
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": train_s, "gpu": f"{gpu}, 1 GPU",
        "what": (f"DarkGS train.py, {iterations} iterations on {len(train_frames)} images at the reference poses "
                 "(process wall time, including its image loading). Not a SLAM loop: no FPS (PROTOCOL.md, Nhóm C)"),
        "sessions": 1})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"DarkGS {commit[:8]} (user's fork) at the reference poses, {len(train_frames)} training frames (0 grid), "
             f"{iterations} iterations, lamp from this project's calibration (D106), --linearize, scaling 1; learned "
             f"ambient {learned['ambient_light']:.4f}, scaling {learned['scaling_factor']:.4f}; {n_gauss} Gaussians. "
             f"Renders at the reference poses, sRGB; scored on the held-out grid ({n_heldout} frames).")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "torch": torch.__version__, "frames": n,
        "keyframes": len(train_frames), "renders": len(glob.glob(os.path.join(renders_dir, "*.npy"))),
        "gaussians": n_gauss, "given_poses": "reference", "heldout_offset": protocol.HELDOUT_OFFSET,
        "rendering_frames_offset": n_heldout, "iterations": iterations, "point_cloud": points, "learned_shader": learned,
        "overrides": overrides, "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{len(train_frames)} training frames, {n_gauss} Gaussians, ambient {learned['ambient_light']:.4f}; "
          f"{n_heldout} held-out frames rendered; training {train_s:.0f} s -> {out}", flush=True)
    map_file = None
    if not args.no_map:
        import tarfile

        map_file = os.path.join(work, "darkgs_model.tar")
        with tarfile.open(map_file, "w") as t:
            for name in (f"chkpnt{iterations}.pth", f"shader{iterations}.pth", "cfg_args"):
                if os.path.isfile(os.path.join(model_dir, name)):
                    t.add(os.path.join(model_dir, name), arcname=name)
    runzip.package(work, ("overrides.json",), map_file=map_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder with results/ and traj.txt")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0")
    p.add_argument("--resume", action="store_true", help="refused")
    p.add_argument("--no_map", action="store_true", help="leave the trained model out of the zip")
    p.add_argument("--iterations", type=int, default=None,
                   help="LOCAL TESTS ONLY: fewer training iterations (default: DarkGS's own, as on Kaggle)")
    args = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    cmd_run(args)


if __name__ == "__main__":
    main()
