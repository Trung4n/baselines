"""NFL-BA (Dunn Beltran et al., NeurIPS 2025) on a MonoGS Replica-format
dataset. Runs on Kaggle with the baselines repo alone and ends with ONE zip per
run, scored later on the local machine by baselines/common/score_zip.py
(PROTOCOL.md). Contract of common/kaggle_runner.py.

    python methods/nflba/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--nflba <source, default $TMPDIR/nflba_src>]

NFL-BA is MonoGS with a near-field light model (a point light at the camera
centre: inverse-square falloff times a Lambert term on the Gaussians'
shortest-axis normals, applied to each Gaussian's colour before rasterizing,
pipeline_params.nfl), plus a highlight mask (grey >= 0.9 dropped) in the
tracking and mapping losses and camera-centre keyframe distances. Its
repository has no licence, so it is not vendored: setup_kaggle.sh fetches the
pinned commit to $TMPDIR/nflba_src. Its CUDA extensions are MonoGS's, which the
repo already carries at the same commits (third_party/LoopSplat/_deps). No
resume: --checkpoint_every must be 0 and --resume is refused.

What is run: the SLAM.__init__ of NFL-BA's slam.py (the whole MonoGS run:
frontend in this process, backend in a spawned one), with MonoGS's Replica
RGB-D config (methods/nflba/configs/replica, NFL-BA ships none) and nfl on. Its
evaluation (eval_ate, eval_rendering) is not run: it scores against the
reference poses of traj.txt, which the run does not see. What differs, all in
<work>/overrides.json and README.md:
- Dataset.Calibration from protocol.calibration() (half-resolution Replica);
- Dataset.dataset_path: a staged folder (results/ linked, traj.txt of IDENTITY
  matrices: MonoGS starts frame 0 at its reference pose; here at the
  identity, so no reference pose reaches the run);
- Results: save_results, use_gui, eval_rendering, use_wandb off (this adapter
  exports), pipeline_params.nfl on (slam.py --nfl);
- one GPU (CUDA_VISIBLE_DEVICES=0), as MonoGS runs.
Exported: every frame's final pose (keyframes as refined by the backend,
others as tracked, the frontend's cameras: as from_monogs_record.py takes
this repo's MonoGS runs), the keyframes, and renders of the final map (the
frontend's copy, as eval_rendering uses) at those poses through NFL-BA's
renderer with the light model, clamped to [0, 1], without the per-frame
exposure (as eval_rendering).
"""
import argparse
import datetime
import glob
import importlib.util
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

METHOD = "nflba"
UPSTREAM = "https://github.com/asdunnbe/NFL-BA"
COMMIT = "a01ae39d1aa914817d77248544259102b758d2b2"  # pinned; setup_kaggle.sh fetches it
DEFAULT_SRC = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nflba_src")
CONFIG_DIR = os.path.join(HERE, "configs", "replica")
MONOGS_CONFIG_COMMIT = "6c9254c319d8bff5caeef65259e6bb0941a9b9f6"


def _loopsplat():
    spec = importlib.util.spec_from_file_location(
        "loopsplat_adapter", os.path.join(BASELINES, "methods", "loopsplat", "adapter.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_commit(src):
    """The checked-out commit of the NFL-BA source; refuses another one."""
    head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        raise SystemExit(f"{src}: not a git checkout of NFL-BA (run setup_kaggle.sh): {head.stderr.strip()}")
    commit = head.stdout.strip()
    if commit != COMMIT:
        raise SystemExit(f"{src} is at {commit}, expected the pinned {COMMIT}")
    return commit


def merge(base, over):
    """MonoGS's utils/config_utils.update_recursive (scene file over base)."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def load_scene_config(scene):
    import yaml

    path = os.path.join(CONFIG_DIR, f"{scene}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: no MonoGS Replica config for scene {scene!r}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.pop("inherit_from", None)
    if inherit != "configs/rgbd/replica/base_config.yaml":
        raise ValueError(f"{path}: unexpected inherit_from {inherit!r}")
    with open(os.path.join(CONFIG_DIR, "base_config.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return merge(base, cfg)


def build_config(scene, seq_dir, cal):
    """MonoGS's Replica RGB-D config for the scene with the overrides; returns
    (config, overrides)."""
    config = load_scene_config(scene)
    overrides = {"config": f"MonoGS {MONOGS_CONFIG_COMMIT[:8]} configs/rgbd/replica/{scene}.yaml (NFL-BA has none)"}
    if config["Dataset"]["sensor_type"] != "depth" or config["Dataset"]["type"] != "replica":
        raise ValueError("expected an RGB-D Replica config")
    calib = config["Dataset"]["Calibration"]
    for key, value in (("width", cal["width"]), ("height", cal["height"]), ("fx", cal["fx"]), ("fy", cal["fy"]),
                       ("cx", cal["cx"]), ("cy", cal["cy"]), ("depth_scale", cal["depth_scale"])):
        if key not in calib:
            raise KeyError(f"MonoGS config has no Dataset.Calibration.{key}")
        overrides[f"Dataset.Calibration.{key}"] = f"{calib[key]} -> {value}"
        calib[key] = value
    if calib.get("distorted") or any(calib.get(k, 0.0) != 0.0 for k in ("k1", "k2", "p1", "p2", "k3")):
        raise ValueError("MonoGS Replica config has distortion; the dataset frames are undistorted")
    overrides["Dataset.dataset_path"] = f"{config['Dataset'].get('dataset_path')} -> {seq_dir}"
    config["Dataset"]["dataset_path"] = seq_dir
    for key in ("save_results", "use_gui", "eval_rendering", "use_wandb"):
        overrides[f"Results.{key}"] = f"{config['Results'].get(key)} -> False"
        config["Results"][key] = False
    overrides["pipeline_params.nfl"] = "absent -> True (NFL-BA's slam.py --nfl)"
    config["pipeline_params"]["nfl"] = True
    return config, overrides


def run_slam(config):
    """NFL-BA slam.py SLAM.__init__ up to the end of the frontend (same
    objects, queues and processes), without its evaluation. Returns
    (frontend, backend_queue, backend_process, pipeline_params, background,
    seconds) where seconds is MonoGS's "Total time" (CUDA events from the
    start of the set-up to the end of the frontend)."""
    import torch
    import torch.multiprocessing as mp
    from munch import munchify

    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from utils.dataset import load_dataset
    from utils.multiprocessing_utils import FakeQueue
    from utils.slam_backend import BackEnd
    from utils.slam_frontend import FrontEnd

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model_params = munchify(config["model_params"])
    opt_params = munchify(config["opt_params"])
    pipeline_params = munchify(config["pipeline_params"])
    monocular = config["Dataset"]["sensor_type"] in ("monocular", "mono")
    model_params.sh_degree = 3 if config["Training"]["spherical_harmonics"] else 0
    gaussians = GaussianModel(model_params.sh_degree, config=config)
    gaussians.init_lr(6.0)
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    gaussians.training_setup(opt_params)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    frontend_queue, backend_queue = mp.Queue(), mp.Queue()
    q_main2vis, q_vis2main = FakeQueue(), FakeQueue()
    config["Results"]["save_dir"] = None
    config["Training"]["monocular"] = monocular

    frontend = FrontEnd(config)
    backend = BackEnd(config)
    frontend.dataset = dataset
    frontend.background = background
    frontend.pipeline_params = pipeline_params
    frontend.frontend_queue = frontend_queue
    frontend.backend_queue = backend_queue
    frontend.q_main2vis = q_main2vis
    frontend.q_vis2main = q_vis2main
    frontend.set_hyperparams()
    backend.gaussians = gaussians
    backend.background = background
    backend.cameras_extent = 6.0
    backend.pipeline_params = pipeline_params
    backend.opt_params = opt_params
    backend.frontend_queue = frontend_queue
    backend.backend_queue = backend_queue
    backend.live_mode = False
    backend.set_hyperparams()

    backend_process = mp.Process(target=backend.run)
    backend_process.start()
    done = watchdog(frontend, backend_process, len(dataset))
    frontend.run()
    done.set()
    backend_queue.put(["pause"])
    end.record()
    torch.cuda.synchronize()
    return frontend, backend_queue, backend_process, pipeline_params, background, start.elapsed_time(end) * 1e-3


def watchdog(frontend, backend_process, n, every_s=60):
    """A daemon thread while the frontend runs: prints the progress every
    every_s seconds (MonoGS prints little with its results off; the runner
    stops a step that prints nothing for STALL_MIN), and ends the run at once
    if the backend process dies, since the frontend would wait for it forever
    (seen locally under WSL, where CUDA IPC fails: "invalid resource handle"
    in the backend, the frontend spinning). Returns the event to set when the
    frontend is done."""
    import threading

    done = threading.Event()
    t0 = time.time()

    def loop():
        while not done.wait(every_s):
            if not backend_process.is_alive():
                print(f"backend process died (exit code {backend_process.exitcode}) at frame "
                      f"{len(frontend.cameras)}/{n}; stopping the run", flush=True)
                os._exit(1)
            print(f"frame {len(frontend.cameras)}/{n}, {len(frontend.kf_indices)} keyframes, "
                  f"{time.time() - t0:.0f} s", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    return done


def stop_backend(backend_queue, backend_process, timeout_s=120):
    backend_queue.put(["stop"])
    backend_process.join(timeout=timeout_s)
    if backend_process.is_alive():
        print(f"backend did not stop in {timeout_s} s; terminating it", flush=True)
        backend_process.terminate()
        backend_process.join(timeout=30)


def final_poses(frontend, n):
    """c2w of every frame from the frontend's cameras (w2c R, T)."""
    from gaussian_splatting.utils.graphics_utils import getWorld2View2

    if sorted(frontend.cameras) != list(range(n)):
        raise RuntimeError(f"frontend has cameras for {len(frontend.cameras)} frames, expected {n}")
    poses = {}
    for i in range(n):
        cam = frontend.cameras[i]
        w2c = getWorld2View2(cam.R, cam.T).detach().double().cpu().numpy()
        c2w = np.linalg.inv(w2c)
        if not np.isfinite(c2w).all():
            raise ValueError(f"frame {i}: non-finite pose")
        poses[i] = c2w
    return poses


def render_frames(frontend, gaussians, pipeline_params, background, frames, renders_dir):
    import torch

    from gaussian_splatting.gaussian_renderer import render

    with torch.no_grad():
        for j, i in enumerate(frames):
            out = render(frontend.cameras[i], gaussians, pipeline_params, background)["render"]
            np.save(os.path.join(renders_dir, f"{i:06d}.npy"),
                    torch.clamp(out, 0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float16))
            if (j + 1) % 50 == 0 or j + 1 == len(frames):
                print(f"export: rendered {j + 1}/{len(frames)}", flush=True)


def cmd_run(args):
    src, work = os.path.abspath(args.nflba), os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("NFL-BA (MonoGS) cannot resume a run: use --checkpoint_every 0 and no --resume")
    if os.path.exists(os.path.join(work, "export")) or os.path.exists(os.path.join(work, "data")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    os.makedirs(work, exist_ok=True)
    commit = source_commit(src)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    if len(depths) != n:
        raise ValueError(f"{args.dataset}: {n} colour, {len(depths)} depth frames")
    ls = _loopsplat()
    scene, variant = ls.scene_variant(args.dataset)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible = "0"
    elif len([d for d in visible.split(",") if d.strip()]) != 1:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={visible!r}: run NFL-BA on one GPU, as MonoGS")
    if src not in sys.path:
        sys.path.insert(0, src)
    import torch

    seq_dir = ls.stage(args.dataset, work, n)
    config, overrides = build_config(scene, seq_dir, cal)
    overrides["gpus"] = f"CUDA_VISIBLE_DEVICES={visible}: one GPU, as MonoGS"
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)
    runzip.write_json(os.path.join(work, "config_resolved.json"), config)
    print(f"NFL-BA {commit} on {args.dataset}: {n} frames, nfl on", flush=True)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.cuda.reset_peak_memory_stats()
    frontend, backend_queue, backend_process, pipe, background, loop_s = run_slam(config)
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"SLAM done: {len(frontend.cameras)} frames, {len(frontend.kf_indices)} keyframes in {loop_s:.0f} s "
          f"({n / loop_s:.2f} FPS), frontend peak GPU memory {peak_gb:.2f} GB; exporting", flush=True)
    # The frontend's map arrived from the backend process through a CUDA IPC
    # queue, so the backend owns that memory: export (renders, map file)
    # BEFORE stopping it, as slam.py evaluates before its "stop".
    try:
        export(args, work, frontend, pipe, background, n, colors, cal, scene, variant, commit, started_at,
               loop_s, peak_gb, overrides, torch)
    finally:
        stop_backend(backend_queue, backend_process)
    runzip.package(work, ("overrides.json", "config_resolved.json"),
                   map_file=None if args.no_map else os.path.join(work, "point_cloud_final.ply"))


def export(args, work, frontend, pipe, background, n, colors, cal, scene, variant, commit, started_at, loop_s,
           peak_gb, overrides, torch):
    out = os.path.join(work, "export")
    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir)
    poses = final_poses(frontend, n)
    protocol.write_tum(os.path.join(out, "traj_est.tum"), poses)
    keyframes = sorted(set(int(k) for k in frontend.kf_indices))
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")
    gaussians = frontend.gaussians
    frames = protocol.required_renders(n, keyframes)
    render_frames(frontend, gaussians, pipe, background, frames, renders_dir)
    runzip.write_figures(renders_dir, colors, os.path.join(out, "figures"))
    n_gauss = int(gaussians.get_xyz.shape[0])
    n_protocol = len(protocol.rendering_frames(n, keyframes))
    gpu = torch.cuda.get_device_name(0)
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": loop_s, "gpu": f"{gpu}, 1 GPU",
        "what": ("MonoGS's 'Total time' (slam.py: CUDA events from the start of SLAM set-up to the end of the "
                 "frontend): dataset index, Gaussian model, frame reading inside the loop, tracking, keyframe mapping in "
                 "the backend process, MonoGS's own 3 FPS throttle after keyframes. Excludes this adapter's export"),
        "sessions": 1, "frontend_peak_gpu_memory_gb": peak_gb})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"NFL-BA {commit[:8]} (MonoGS + near-field light, nfl on) with MonoGS's Replica RGB-D config "
             f"configs/rgbd/replica/{scene}.yaml and the dataset camera, identity traj.txt (frame 0 at the identity, "
             f"no reference poses), one GPU. {len(keyframes)} keyframes, {n_gauss} Gaussians. Poses: frontend cameras "
             "(backend-refined keyframes, tracked poses elsewhere). Renders: the frontend's final map through NFL-BA's "
             f"renderer with the light model, no per-frame exposure (as eval_rendering). {n_protocol} protocol "
             "rendering frames.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "upstream": UPSTREAM, "torch": torch.__version__,
        "frames": n, "keyframes": len(keyframes), "renders": len(glob.glob(os.path.join(renders_dir, "*.npy"))),
        "gaussians": n_gauss, "rendering_frames_protocol": n_protocol, "overrides": overrides, "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{n} poses, {len(keyframes)} keyframes, {n_gauss} Gaussians; protocol render frames {n_protocol}; "
          f"{n / loop_s:.2f} FPS -> {out}", flush=True)
    if not args.no_map:
        gaussians.save_ply(os.path.join(work, "point_cloud_final.ply"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0: MonoGS cannot resume")
    p.add_argument("--resume", action="store_true", help="refused")
    p.add_argument("--no_map", action="store_true", help="leave the final map (.ply) out of the zip")
    p.add_argument("--nflba", default=DEFAULT_SRC, help="NFL-BA checkout at the pinned commit (setup_kaggle.sh)")
    args = ap.parse_args(argv)
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)  # as NFL-BA's slam.py (the backend is a separate process)
    cmd_run(args)


if __name__ == "__main__":
    main()
