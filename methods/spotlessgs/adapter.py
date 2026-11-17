"""SpotlessGS (arXiv 2608.14713, code github.com/Liianne/SpotlessGS) on a MonoGS Replica-format
dataset, as a group C method (PROTOCOL.md, Nhóm C): DarkGS without the light
calibration, the light (position, rotation, falloff, emission MLP), a BRDF MLP
and an SH ambient learned with the map, offline at the REFERENCE poses; no
ATE, no FPS, rendering scored on the held-out grid. Ends with ONE zip per run,
scored by baselines/common/score_zip.py.

    python methods/spotlessgs/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--iterations <n>, local tests only]

kaggle.json "reference_poses": true (the runner stages traj.txt). What is run:
SpotlessGS's own train.py (third_party/SpotlessGS, official, unchanged
except one patch on a copy, below) on the same COLMAP package as DarkGS
(methods/darkgs/colmap_pkg.py: the 0 grid at the reference poses, points
from sensor depth with normals). Kept as published:
- no light calibration: the light starts from SpotlessGS's defaults and is
  learned (its point); its train.py reads ./model_parameters.pth only for
  the shape of so3 (the copy's own file);
- the images as they are (sRGB 8-bit): SpotlessGS has no linearisation;
  renders are in the same encoding;
- SCALE_FACTOR 1.0 in its train.py (metric poses), 30000 iterations.
Changed, in overrides.json and README.md:
- on a COPY of third_party/SpotlessGS in the work dir,
  scene/dataset_readers.py N_TEST = 29 -> 0: its COLMAP reader always held
  out the last 29 images, here frames 1855..1995 of the training set;
- lietorch replaced by methods/spotlessgs/lietorch_shim (a learnable SO3).
Then the final model (chkpnt<it>.pth, shader<it>.pth) is rendered with
SpotlessGS's renderer and shader at the reference poses of the held-out grid
(2, 7, 12, ...) and the figure frames, and packed.
"""
import argparse
import datetime
import glob
import os
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

METHOD = "spotlessgs"
VENDORED = os.path.join(BASELINES, "third_party", "SpotlessGS")
SHIM = os.path.join(HERE, "lietorch_shim")
N_TEST_LINE, N_TEST_PATCH = "    N_TEST = 29\n", "    N_TEST = 0  # baselines: was 29 (always held out the last images)\n"


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def patched_copy(dst):
    """third_party/SpotlessGS copied to dst with the N_TEST patch (exactly one line)."""
    shutil.copytree(VENDORED, dst)
    path = os.path.join(dst, "scene", "dataset_readers.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(N_TEST_LINE) != 1:
        raise RuntimeError(f"{path}: expected one {N_TEST_LINE.strip()!r}, found {text.count(N_TEST_LINE)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(N_TEST_LINE, N_TEST_PATCH))
    return dst


def env_for(src):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([SHIM, src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def default_iterations(src):
    sys.path[:0] = [SHIM, src]
    from argparse import ArgumentParser

    from arguments import OptimizationParams

    parser = ArgumentParser()
    op = OptimizationParams(parser)
    return int(op.extract(parser.parse_args([])).iterations)


KEEP_POSE_EVERY = 1000  # of SpotlessGS's per-iteration "r is" / "t is" lines


def train(src, pkg, model_dir, iterations):
    """SpotlessGS's train.py in a child process (cwd = the copy: it reads
    ./model_parameters.pth); returns the seconds. Its output is filtered, not
    its computation: ShadingModel.forward prints the whole normals tensor at
    every render (three per iteration) and train.py the light pose at every
    iteration, hundreds of thousands of lines over a run, which would flood
    the notebook. The tensor dumps are dropped, one "r is" / "t is" pair in
    KEEP_POSE_EVERY is kept, everything else (progress bar, reports, errors)
    passes through; the counts are printed at the end."""
    import re
    from collections import Counter

    cmd = [sys.executable, "-u", os.path.join(src, "train.py"), "-s", pkg, "-m", model_dir,
           "--iterations", str(iterations), "--save_iterations", str(iterations),
           "--checkpoint_iterations", str(iterations)]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=src, env=env_for(src), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    counts, in_dump, buf = Counter(), False, b""
    out = sys.stdout.buffer

    def handle(text, sep):
        nonlocal in_dump
        s = text.decode(errors="replace")
        if in_dump:
            counts["normals dump lines"] += 1
            in_dump = "device=" not in s and not s.rstrip().endswith(")")
            return
        if s.startswith("Normals:"):
            counts["normals dump lines"] += 1
            in_dump = "device=" not in s
            return
        if s.startswith("r is:") or s.startswith("t is:"):
            counts[s[:5]] += 1
            if counts[s[:5]] % KEEP_POSE_EVERY != 1:
                return
        out.write(text + sep)
        out.flush()

    for chunk in iter(lambda: proc.stdout.read1(65536), b""):
        parts = re.split(rb"(\r|\n)", buf + chunk)
        buf = parts.pop()
        for text, sep in zip(parts[0::2], parts[1::2]):
            handle(text, sep)
    if buf:
        handle(buf, b"\n")
    code = proc.wait()
    seconds = time.time() - t0
    print(f"train.py output filtered: {dict(counts)}", flush=True)
    if code != 0:
        raise SystemExit(f"SpotlessGS train.py failed (exit code {code})")
    return seconds


def load_model(model_dir, iterations):
    import torch
    from argparse import ArgumentParser

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
    shader = ShadingModel(light="1DMLP").cuda()  # as train.py builds it
    shader.load_state_dict(saved["model_state_dict"])  # light pose, falloff, MLPs, ambient, scale
    return gaussians, shader


def render_frames(gaussians, shader, frames, c2w, cal, renders_dir):
    """SpotlessGS's renderer at the reference poses, clamped to [0, 1] (its
    images are the sRGB inputs, so are its renders)."""
    import torch
    from argparse import ArgumentParser

    from arguments import PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    parser = ArgumentParser()
    pp = PipelineParams(parser)
    pipe = pp.extract(parser.parse_args([]))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    W, H = cal["width"], cal["height"]
    fovx, fovy = focal2fov(cal["fx"], W), focal2fov(cal["fy"], H)
    blank = torch.zeros(3, H, W)
    import contextlib
    import io

    with torch.no_grad():
        for j, i in enumerate(frames):
            R_w2c = c2w[i][:3, :3].T
            t_w2c = -R_w2c @ c2w[i][:3, 3]
            cam = Camera(colmap_id=i, R=R_w2c.T, T=t_w2c, FoVx=fovx, FoVy=fovy, image=blank, gt_alpha_mask=None,
                         image_name=f"frame{i:06d}", uid=i)
            with contextlib.redirect_stdout(io.StringIO()):  # ShadingModel.forward prints the normals tensor
                img = render(cam, gaussians, pipe, background, shader)["render"].clamp(0.0, 1.0)
            np.save(os.path.join(renders_dir, f"{i:06d}.npy"), img.permute(1, 2, 0).cpu().numpy().astype(np.float16))
            if (j + 1) % 50 == 0 or j + 1 == len(frames):
                print(f"export: rendered {j + 1}/{len(frames)}", flush=True)


def cmd_run(args):
    work = os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("SpotlessGS is run in one go here: use --checkpoint_every 0 and no --resume")
    if os.path.exists(os.path.join(work, "export")) or os.path.exists(os.path.join(work, "data")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    scene, variant = scene_variant(args.dataset)
    traj = os.path.join(args.dataset, "traj.txt")
    if not os.path.isfile(traj):
        raise SystemExit(f"{traj}: missing. SpotlessGS needs the reference poses (kaggle.json reference_poses, PROTOCOL.md)")
    os.makedirs(work, exist_ok=True)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    if len(depths) != n:
        raise ValueError(f"{args.dataset}: {n} colour, {len(depths)} depth frames")
    c2w = protocol.gt_poses_c2w(args.dataset)
    if len(c2w) != n:
        raise ValueError(f"{traj}: {len(c2w)} poses for {n} frames")
    commit = runzip.read_json(os.path.join(VENDORED, ".vendor.json"))["commit"]
    src = patched_copy(os.path.join(work, "spotless_src"))
    train_frames = list(range(0, n, protocol.RENDER_INTERVAL))
    iterations = args.iterations or default_iterations(src)

    pkg = os.path.join(work, "data", f"{scene}_{variant}")
    print(f"SpotlessGS {commit[:8]} on {args.dataset}: {len(train_frames)} training frames of {n}, {iterations} "
          "iterations; building the COLMAP package at the reference poses", flush=True)
    points = colmap_pkg.build(pkg, train_frames, colors, depths, c2w, cal)
    # SpotlessGS's reader takes sparse/0/points3D_with_normals.ply (x, y, z, red, green, blue, nx, ny, nz);
    # same file, same normal convention as DarkGS (away from the camera: its shader uses the same light_dir)
    shutil.copyfile(os.path.join(pkg, "sparse", "0", "points3D.ply"), os.path.join(pkg, "sparse", "0", "points3D_with_normals.ply"))
    print(f"package: {points['points']} points ({points['raw_points']} back-projected), "
          f"{points['geometric_normals']:.1%} with a geometric normal", flush=True)
    overrides = {
        "poses": "reference poses (traj.txt), PROTOCOL.md Nhóm C",
        "training_frames": f"0 grid: {len(train_frames)} frames (0, {protocol.RENDER_INTERVAL}, ..., {train_frames[-1]})",
        "point_cloud": f"sensor depth of the training frames at the reference poses, voxel {points['voxel_m']} m, normals "
                       "from depth (methods/darkgs/colmap_pkg.py; SpotlessGS uses COLMAP / OKVIS2 points)",
        "light": "not calibrated: SpotlessGS's defaults, learned (its train.py; no light file passed)",
        "images": "sRGB 8-bit as they are (SpotlessGS has no linearisation)",
        "scene/dataset_readers.py": "N_TEST 29 -> 0 on a copy: the last 29 images were always held out of training",
        "train.py": f"--iterations {iterations} (SpotlessGS default otherwise, SCALE_FACTOR 1.0)",
        "lietorch": "lietorch_shim: learnable SO3 (axis-angle nn.Parameter)",
    }
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_dir = os.path.join(work, "spotless_out")
    train_s = train(src, pkg, model_dir, iterations)

    sys.path[:0] = [SHIM, src]
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
    light = {"t_l2c_m": [float(x) for x in shader.light.get_t_vec()],
             "r_l2c_axis_angle": [float(x) for x in shader.light.get_r_vec()],
             "gamma": float(shader.light.gamma.item()), "tau": float(shader.light.tau.item()),
             "ambient_light": float(shader.ambient_light.item()), "scaling_factor": float(shader.scaling_factor.item())}
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": train_s, "gpu": f"{gpu}, 1 GPU",
        "what": (f"SpotlessGS train.py, {iterations} iterations on {len(train_frames)} images at the reference poses "
                 "(process wall time). Not a SLAM loop: no FPS (PROTOCOL.md, Nhóm C)"),
        "sessions": 1})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"SpotlessGS {commit[:8]} at the reference poses, {len(train_frames)} training frames (0 grid), "
             f"{iterations} iterations, light learned from SpotlessGS's defaults (no calibration), sRGB images; learned "
             f"light t_l2c {np.round(light['t_l2c_m'], 4).tolist()} m, gamma {light['gamma']:.3f}, scaling "
             f"{light['scaling_factor']:.3f}; {n_gauss} Gaussians. N_TEST patched to 0. Renders at the reference poses; "
             f"scored on the held-out grid ({n_heldout} frames).")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "torch": torch.__version__, "frames": n,
        "keyframes": len(train_frames), "renders": len(glob.glob(os.path.join(renders_dir, "*.npy"))),
        "gaussians": n_gauss, "given_poses": "reference", "heldout_offset": protocol.HELDOUT_OFFSET,
        "rendering_frames_offset": n_heldout, "iterations": iterations, "point_cloud": points, "learned_light": light,
        "overrides": overrides, "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{len(train_frames)} training frames, {n_gauss} Gaussians; light {light}; {n_heldout} held-out frames; "
          f"training {train_s:.0f} s -> {out}", flush=True)
    map_file = None
    if not args.no_map:
        import tarfile

        map_file = os.path.join(work, "spotless_model.tar")
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
                   help="LOCAL TESTS ONLY: fewer training iterations (default: SpotlessGS's own, as on Kaggle)")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
