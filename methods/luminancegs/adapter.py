"""Luminance-GS (CVPR 2025; per-view luminance curves learnt jointly with 3DGS,
built on gsplat 1.0.0) on a MonoGS Replica-format dataset, as a group C method
(PROTOCOL.md, Nhóm C): offline at the REFERENCE poses; no ATE, no FPS;
rendering scored on the held-out grid. Ends with ONE zip per run, scored by
score_zip.py.

    python methods/luminancegs/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--iterations <n>, local tests only]

kaggle.json "reference_poses": true (the runner stages traj.txt). What is run:
the authors' examples/simple_trainer_ours.py (third_party/Luminance-GS), as
their README's low-light command:
    simple_trainer_ours.py --data_dir <package> --exp_name low --result_dir <dir>
with its defaults (10000 steps, SH degree 3, normalised world space). Its CUDA
code is the repository's own modified gsplat 1.0.0 backend, JIT-compiled
(probe.py at setup); gsplat is not pip-installed (third_party/Luminance-GS/
.vendor.json).

The package is the LOM layout the trainer's default loader (datasets/colmap.py)
reads, built from the same COLMAP package as DarkGS, SpotlessGS and bilagrid
(methods/darkgs/colmap_pkg.py, binary model for the pycolmap fork; points from
the TRAINING frames' sensor depth at the reference poses):
    colmap_sparse/0/            cameras, images (training + held-out frames), points3D
    low/frameNNNNNN.jpg         every listed frame (the loader pairs the sorted COLMAP
                                names with the sorted files of this folder)
    high/frameNNNNNN.jpg        held-out frames: its validation images (LOM: the
                                normal-light ground truth). Here the OBSERVED frames
                                (the run sees nothing else); its own eval numbers are not used
    transforms_train.json       file_path "low/<name>": the 0 grid, training order
    transforms_test.json        file_path "high/<name>": the held-out grid
Only the training frames are read for training (trainset = the first lentrain).

Two renders of each frame, from the final checkpoint (rasterize_splats returns both):
- renders/: the "low" branch, per-Gaussian affine of the colours (adjust_k,
  adjust_b), fitted to the observed images: the model's prediction of the
  observation, scored against lit by score_zip.py like every other method;
- renders_enh/: the "enh" branch, the illumination-normalised image (what the
  authors evaluate against normal-light ground truth). Scored separately
  against the even variant (README): evaluation only, never an input.

Changed on a COPY (a fixed path, so that the JIT build of setup's probe is
reused), recorded in overrides.json:
- 'from turtle import color' removed from simple_trainer_ours.py (unused;
  turtle needs tkinter);
- examples/datasets/__init__.py added (empty): the Hugging Face 'datasets'
  package would shadow this namespace package;
- csrc/third_party/glm: gsplat 1.0.0's glm pin, fetched by setup_kaggle.sh.
"""
import argparse
import datetime
import glob
import json
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

METHOD = "luminancegs"
VENDORED = os.path.join(BASELINES, "third_party", "Luminance-GS", "Luminance-GS")
COMMIT = "f9b4b86692a3fa676e90f42236c6237f443e7e23"
GLM_URL = "https://github.com/g-truc/glm"
GLM_COMMIT = "45008b225e28eb700fa0f7d3ff69b7c1db94fadf"  # gsplat v1.0.0's submodule pin
TMP = os.environ.get("TMPDIR", "/tmp")
DEFAULT_GLM = os.path.join(TMP, "luminancegs_glm")
DEFAULT_SRC = os.path.join(TMP, "luminancegs_src")
EXP_NAME = "low"  # the README's low-light command; selects datasets/colmap.py and the histogram-prior weight
TURTLE_LINE = "from turtle import color\n"
NOISY = "the image train is 0"  # datasets/colmap.py prints the whole list once per training image


def cuda_arch_env():
    """TORCH_CUDA_ARCH_LIST as setup_kaggle.sh exports it for the probe's JIT
    build ("major.minor" of this GPU). The run is started by the runner, not
    by setup, so without it torch writes other -gencode flags and ninja
    rebuilds the backend in the run."""
    import torch

    arch = "%d.%d" % torch.cuda.get_device_capability(0)
    have = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if have and have != arch:
        raise SystemExit(f"TORCH_CUDA_ARCH_LIST={have}, but setup builds for {arch} (this GPU)")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch


def scene_variant(dataset):
    d = os.path.normpath(os.path.abspath(dataset))
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def prepare_src(dst=DEFAULT_SRC, glm=DEFAULT_GLM):
    """The patched copy (module docstring) at dst, replaced if present. copytree
    keeps the files' times, so the same path gives the same JIT build."""
    if not os.path.isfile(os.path.join(glm, "glm", "glm.hpp")):
        raise FileNotFoundError(f"{glm}/glm/glm.hpp: run setup_kaggle.sh (glm {GLM_COMMIT})")
    head = os.path.join(glm, ".git")
    if os.path.isdir(head):
        got = subprocess.run(["git", "-C", glm, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if got != GLM_COMMIT:
            raise RuntimeError(f"{glm} is at {got}, expected {GLM_COMMIT}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(VENDORED, dst)
    path = os.path.join(dst, "examples", "simple_trainer_ours.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.count(TURTLE_LINE) != 1:
        raise RuntimeError(f"{path}: expected one '{TURTLE_LINE.strip()}', found {text.count(TURTLE_LINE)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(TURTLE_LINE, "# baselines: 'from turtle import color' removed (unused, needs tkinter)\n"))
    shutil.copystat(os.path.join(VENDORED, "examples", "simple_trainer_ours.py"), path)
    init = os.path.join(dst, "examples", "datasets", "__init__.py")
    if os.path.exists(init):
        raise RuntimeError(f"{init} exists upstream: check the shadowing patch")
    open(init, "w").close()
    third = os.path.join(dst, "gsplat", "cuda", "csrc", "third_party")
    os.makedirs(third)
    shutil.copytree(os.path.join(glm, "glm"), os.path.join(third, "glm", "glm"))
    return dst


def build_package(pkg, train_frames, heldout, colors, depths, c2w, cal):
    """The LOM layout (module docstring); returns the point-cloud statistics."""
    tmp = pkg + "_colmap"
    points = colmap_pkg.build(tmp, train_frames, colors, depths, c2w, cal, binary=True,
                              cams=sorted(set(train_frames) | set(heldout)))
    os.makedirs(pkg)
    os.rename(os.path.join(tmp, "sparse"), os.path.join(pkg, "colmap_sparse"))
    os.rename(os.path.join(tmp, "images"), os.path.join(pkg, EXP_NAME))
    shutil.rmtree(tmp)
    os.makedirs(os.path.join(pkg, "high"))
    for i in heldout:
        os.symlink(os.path.abspath(colors[i]), os.path.join(pkg, "high", f"frame{i:06d}.jpg"))
    for split, folder, frames in (("train", EXP_NAME, train_frames), ("test", "high", heldout)):
        with open(os.path.join(pkg, f"transforms_{split}.json"), "w") as f:
            json.dump({"frames": [{"file_path": f"{folder}/frame{i:06d}.jpg"} for i in frames]}, f, indent=1)
    # the loader finds a validation image by replacing EXP_NAME with "high" anywhere in its path
    for i in heldout[:1]:
        p = os.path.join(pkg, EXP_NAME, f"frame{i:06d}.jpg")
        if p.count(EXP_NAME) != 1:
            raise ValueError(f"{p}: '{EXP_NAME}' must occur once in the path (the loader replaces it with 'high')")
    return points


def trainer_args(pkg, result_dir, iterations):
    args = ["--data_dir", pkg, "--exp_name", EXP_NAME, "--result_dir", result_dir]
    if iterations:  # local tests only: the same schedule, shorter
        args += ["--max_steps", str(iterations), "--eval_steps", str(iterations), "--save_steps", str(iterations)]
    return args


def train(src, pkg, result_dir, iterations):
    """simple_trainer_ours.py in a child process (cwd = examples/); returns the
    seconds. Output filtered, not the computation: the loader prints its list
    of training images once per training image (quadratic in the frames)."""
    cmd = [sys.executable, "-u", os.path.join(src, "examples", "simple_trainer_ours.py")] + trainer_args(pkg, result_dir, iterations)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=os.path.join(src, "examples"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dropped, buf, out = 0, b"", sys.stdout.buffer
    for chunk in iter(lambda: proc.stdout.read1(65536), b""):
        parts = re.split(rb"(\r|\n)", buf + chunk)
        buf = parts.pop()
        for text, sep in zip(parts[0::2], parts[1::2]):
            if text.startswith(NOISY.encode()):
                dropped += 1
                continue
            out.write(text + sep)
            out.flush()
    if buf:
        out.write(buf + b"\n")
    code = proc.wait()
    seconds = time.time() - t0
    print(f"simple_trainer_ours.py output filtered: {dropped} '{NOISY}' lines", flush=True)
    if code != 0:
        raise SystemExit(f"Luminance-GS simple_trainer_ours.py failed (exit code {code})")
    return seconds


def final_ckpt(result_dir):
    ckpts = glob.glob(os.path.join(result_dir, "ckpts", "ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {result_dir}/ckpts")
    step = lambda p: int(re.search(r"ckpt_(\d+)\.pt$", p).group(1))
    return max(ckpts, key=step), max(step(p) for p in ckpts)


def load_runner(src, pkg, result_dir, iterations, ckpt):
    """The trainer's Runner with the training run's configuration and the
    final splats. The module parses sys.argv at import (tyro.cli), so it is
    imported with the training command line."""
    import torch

    sys.path.insert(0, os.path.join(src, "examples"))
    argv = sys.argv
    sys.argv = ["simple_trainer_ours.py"] + trainer_args(pkg, os.path.join(result_dir, "export_runner"), iterations)
    try:
        import simple_trainer_ours as st
    finally:
        sys.argv = argv
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # the loader's per-image list prints, as train() filters them
        runner = st.Runner(st.cfg)
    kept = [ln for ln in buf.getvalue().splitlines() if not ln.startswith(NOISY)]
    print("\n".join(kept), flush=True)
    data = torch.load(ckpt, map_location=runner.device, weights_only=True)
    if set(data["splats"].keys()) != set(runner.splats.keys()):
        raise KeyError(f"{ckpt}: splats {sorted(data['splats'].keys())}, runner {sorted(runner.splats.keys())}")
    for k in runner.splats.keys():
        runner.splats[k].data = data["splats"][k]
    return runner, st.cfg


def render_frames(runner, cfg, frames, c2w, dirs, train_frames):
    """rasterize_splats at the reference poses (moved into the trainer's
    normalised world with its own transform_cameras): both branches, clamped
    to [0, 1], into dirs["low"] and dirs["enh"]. Returns the largest
    difference between the trainer's training poses and ours moved."""
    import torch
    from datasets.normalize import transform_cameras

    p = runner.parser
    cams = sorted(set(p.camera_ids))
    if len(cams) != 1:
        raise ValueError(f"expected one camera, got {cams}")
    K = torch.from_numpy(p.Ks_dict[cams[0]]).float().to(runner.device)
    W, H = p.imsize_dict[cams[0]]
    names = {nm: int(re.search(r"frame(\d+)\.jpg$", nm).group(1)) for nm in p.image_names}
    if [names[nm] for nm in p.image_names[:p.lentrain]] != list(train_frames):
        raise ValueError("the trainer's training images are not the 0 grid in order")
    ours = transform_cameras(p.transform, np.stack([c2w[names[nm]] for nm in p.image_names]))
    err = float(np.abs(ours - p.camtoworlds).max())
    if err > 1e-4:
        raise ValueError(f"reference poses moved by transform_cameras differ from the trainer's by {err}")
    with torch.no_grad():
        for j, i in enumerate(frames):
            cw = torch.from_numpy(transform_cameras(p.transform, c2w[i][None].copy())).float().to(runner.device)
            enh, low, _, _, _ = runner.rasterize_splats(camtoworlds=cw, Ks=K[None], width=W, height=H,
                                                        sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                                                        far_plane=cfg.far_plane)
            for key, img in (("low", low), ("enh", enh)):
                np.save(os.path.join(dirs[key], f"{i:06d}.npy"),
                        img[..., :3].clamp(0.0, 1.0)[0].cpu().numpy().astype(np.float16))
            if (j + 1) % 50 == 0 or j + 1 == len(frames):
                print(f"export: rendered {j + 1}/{len(frames)} (low and enh)", flush=True)
    return err


def cmd_run(args):
    work = os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("luminancegs is run in one go here: use --checkpoint_every 0 and no --resume")
    if os.path.exists(os.path.join(work, "export")) or os.path.exists(os.path.join(work, "data")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    try:
        import gsplat  # noqa: F401
        raise SystemExit(f"a gsplat package is installed ({gsplat.__file__}): the bundled backend would import its "
                         "kernels instead of compiling its own (third_party/Luminance-GS/.vendor.json)")
    except ImportError:
        pass
    cuda_arch_env()  # inherited by the trainer's process and used by the export below
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
    src = prepare_src(args.src, args.glm)
    train_frames = list(range(0, n, protocol.RENDER_INTERVAL))
    heldout = protocol.rendering_frames(n, train_frames, protocol.HELDOUT_OFFSET)

    pkg = os.path.join(work, "data", f"{scene}_{variant}")
    print(f"Luminance-GS {COMMIT[:8]} on {args.dataset}: {len(train_frames)} training frames of {n}, "
          f"{len(heldout)} held-out; building the package at the reference poses", flush=True)
    points = build_package(pkg, train_frames, heldout, colors, depths, c2w, cal)
    print(f"package: {points['points']} points ({points['raw_points']} back-projected)", flush=True)
    overrides = {
        "poses": "reference poses (traj.txt), PROTOCOL.md Nhóm C",
        "training_frames": f"0 grid: {len(train_frames)} frames (0, {protocol.RENDER_INTERVAL}, ..., {train_frames[-1]})",
        "validation_frames": f"held-out grid ({len(heldout)} frames) as the trainer's val set; its 'high' images are "
                             "the observed frames (LOM: normal-light ground truth); its eval numbers are not used",
        "point_cloud": f"sensor depth of the training frames at the reference poses, voxel {points['voxel_m']} m "
                       "(methods/darkgs/colmap_pkg.py; the authors initialise from COLMAP SfM points)",
        "simple_trainer_ours.py": " ".join(trainer_args("<package>", "<dir>", args.iterations)),
        "simple_trainer_ours.py import": "'from turtle import color' removed (unused; turtle needs tkinter), on a copy",
        "datasets/__init__.py": "added (empty) on the copy: the Hugging Face 'datasets' package would shadow it",
        "glm": f"{GLM_URL} @ {GLM_COMMIT} (gsplat 1.0.0's pin; the repository lost the submodule)",
        "gsplat": "the repository's bundled backend (JIT); no pip gsplat (README: gsplat 1.0.0 from git)",
        "numpy": "the notebook's numpy (README: numpy==1.24.3; not forced, it would break the installed stack)",
        "test-time": "renders/: 'low' branch (prediction of the observed images), scored against lit; renders_enh/: "
                     "'enh' branch (illumination-normalised), scored separately against even",
    }
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = os.path.join(work, "lumgs_out")
    train_s = train(src, pkg, result_dir, args.iterations)
    ckpt, step = final_ckpt(result_dir)
    print(f"final checkpoint: {ckpt} (step {step})", flush=True)

    import torch
    runner, cfg = load_runner(src, pkg, result_dir, args.iterations, ckpt)
    if step != cfg.max_steps - 1:
        raise ValueError(f"last checkpoint at step {step}, training ran {cfg.max_steps} steps")
    out = os.path.join(work, "export")
    dirs = {"low": os.path.join(out, "renders"), "enh": os.path.join(out, "renders_enh")}
    for d in dirs.values():
        os.makedirs(d)
    frames = sorted(set(protocol.required_renders(n, train_frames)) | set(heldout))
    pose_err = render_frames(runner, cfg, frames, c2w, dirs, train_frames)
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in train_frames) + "\n")
    runzip.write_figures(dirs["low"], colors, os.path.join(out, "figures"))
    n_gauss = int(runner.splats["means3d"].shape[0])
    gpu = torch.cuda.get_device_name(0)
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": train_s, "gpu": f"{gpu}, 1 GPU",
        "what": (f"Luminance-GS simple_trainer_ours.py, {cfg.max_steps} steps on {len(train_frames)} images at the reference "
                 "poses, including its own evaluations and trajectory videos (process wall time). Not a SLAM loop: no "
                 "FPS (PROTOCOL.md, Nhóm C)"),
        "sessions": 1})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"Luminance-GS ({COMMIT[:8]}, simple_trainer_ours.py --exp_name {EXP_NAME}) at the reference poses, "
             f"{len(train_frames)} training frames (0 grid), {cfg.max_steps} steps; {n_gauss} Gaussians. renders/: "
             f"'low' branch (prediction of the observation), scored on the held-out grid ({len(heldout)} frames); "
             f"renders_enh/: 'enh' branch (illumination-normalised), score separately against even. Trainer poses vs "
             f"reference poses after normalisation: max diff {pose_err:.1e}.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": COMMIT, "torch": torch.__version__, "frames": n,
        "keyframes": len(train_frames), "renders": len(glob.glob(os.path.join(dirs["low"], "*.npy"))),
        "gaussians": n_gauss, "given_poses": "reference", "heldout_offset": protocol.HELDOUT_OFFSET,
        "rendering_frames_offset": len(heldout), "iterations": cfg.max_steps, "point_cloud": points,
        "overrides": overrides, "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{len(train_frames)} training frames, {n_gauss} Gaussians; {len(heldout)} held-out frames; training "
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
    p.add_argument("--src", default=DEFAULT_SRC, help="where the patched copy is made (fixed: reuses setup's JIT build)")
    p.add_argument("--glm", default=DEFAULT_GLM, help=f"glm checkout at {GLM_COMMIT} (setup_kaggle.sh)")
    p.add_argument("--iterations", type=int, default=None,
                   help="LOCAL TESTS ONLY: fewer training steps (default: the authors' 10000, as on Kaggle)")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
