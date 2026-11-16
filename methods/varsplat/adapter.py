"""VarSplat (Tran and Kosecka, CVPR 2026) on a MonoGS Replica-format dataset.
Runs on Kaggle with the baselines repo alone and ends with ONE zip per run,
scored later on the local machine by baselines/common/score_zip.py
(PROTOCOL.md). Contract of common/kaggle_runner.py.

    python methods/varsplat/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--varsplat <source, default third_party/VarSplat>]

VarSplat is LoopSplat's code base (same GaussianSLAM, submaps, gs_reg
registration, NetVLAD loop detection, configs) plus a learned colour variance
per Gaussian: a variance map rendered with the colour, an NLL term in
mapping, and the variance as a pixel weight in tracking, in submap
registration and (per submap) in loop detection. So this adapter is the
LoopSplat adapter (methods/loopsplat/adapter.py, imported, not copied) run on
VarSplat's source, with the same overrides, patches, renders and held-out
grid, plus one VarSplat patch (varsplat_patches). No resume:
--checkpoint_every must be 0 and --resume is refused.

What differs from VarSplat's run_slam.py with configs/Replica/<scene>.yaml,
all recorded in <work>/overrides.json and README.md:
- everything methods/loopsplat/adapter.py changes (camera, identity traj.txt,
  pixel counts of submap seeding, W&B off, faiss shim, one GPU, torch
  compatibility, analyse_pgo skipped, last submap saved);
- varsplat_patches: Loop_closure keeps, per finished submap, only the two
  tensors submap_reliability reads (colour variance and opacity) instead of
  the whole GaussianModel with its gradients and Adam state.
"""
import argparse
import datetime
import glob
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
LOOPSPLAT_DIR = os.path.join(BASELINES, "methods", "loopsplat")
sys.path.insert(0, os.path.join(BASELINES, "common"))
import protocol  # noqa: E402
import runzip  # noqa: E402


def _load_loopsplat_adapter():
    spec = importlib.util.spec_from_file_location("loopsplat_adapter", os.path.join(LOOPSPLAT_DIR, "adapter.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = _load_loopsplat_adapter()

METHOD = "varsplat"
VENDORED = os.path.join(BASELINES, "third_party", "VarSplat")
VAR_KEYS = ("use_nlldepth", "tau_track", "tau_registration", "tau_lc", "w_limit")


def build_config(src, scene, seq_dir, out_dir, cal):
    """LoopSplat's build_config on VarSplat's config, plus VarSplat's checks:
    the var block is present and complete, and the final loop closure is off
    (with lc.final, GaussianSLAM.run calls update_submaps_info without the
    submap's model and loop detection would read None)."""
    config, overrides = ls.build_config(src, scene, seq_dir, out_dir, cal)
    var = config.get("var")
    if not isinstance(var, dict) or any(k not in var for k in VAR_KEYS):
        raise KeyError(f"VarSplat config has no complete var block (need {VAR_KEYS}): {var}")
    if config["lc"]["final"]:
        raise ValueError("lc.final is on: VarSplat's final loop closure passes no submap model to loop detection")
    return config, overrides


def varsplat_patches():
    """Patch on VarSplat that changes no estimate; returns what was patched.

    Loop_closure.update_submaps_info(keyframes_info, gaussian_model) stores the
    finished submap's GaussianModel in submap_gaussians, and the only reader is
    submap_reliability(model), which calls model.get_color_variance() and
    model.get_opacity() (detect_closure, under torch.no_grad). Kept whole, every
    finished submap stays on the GPU with its gradients and Adam (amsgrad)
    state, about five times its parameters: several GB over a 40-submap run on
    a 16 GB T4 (the authors ran on an 80 GB A100). The patch stores a snapshot
    of those two tensors instead. The model is not changed after this call (it
    is replaced by a new one in start_new_submap, and loop-closure
    corrections rewrite the submap files on disk, and are rigid anyway, which
    leaves variance and opacity unchanged), so submap_reliability returns the
    same values."""
    import torch
    from src.entities.lc import Loop_closure

    class ReliabilityInputs:
        """What submap_reliability reads of a finished submap."""

        def __init__(self, model):
            with torch.no_grad():
                self._variance = model.get_color_variance().detach().clone()
                self._opacity = model.get_opacity().detach().clone()

        def get_color_variance(self):
            return self._variance

        def get_opacity(self):
            return self._opacity

    update = Loop_closure.update_submaps_info

    def update_submaps_info(self, keyframes_info, gaussian_model=None):
        if gaussian_model is None:
            raise RuntimeError("update_submaps_info without the submap's model (final loop closure): not supported")
        return update(self, keyframes_info, ReliabilityInputs(gaussian_model))

    Loop_closure.update_submaps_info = update_submaps_info
    return {"loop_closure.submap_gaussians": (
        "per finished submap, a snapshot of the colour variance and opacity (the inputs of submap_reliability) "
        "instead of the whole GaussianModel with gradients and optimizer state; same reliability values, less "
        "GPU memory")}


def cmd_run(args):
    src, work = os.path.abspath(args.varsplat), os.path.abspath(args.work)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("VarSplat cannot resume a run: use --checkpoint_every 0 and no --resume")
    out_dir = os.path.join(work, "varsplat_out")
    if os.path.exists(out_dir) or os.path.exists(os.path.join(work, "export")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    os.makedirs(work, exist_ok=True)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    if len(depths) != n:
        raise ValueError(f"{args.dataset}: {n} colour, {len(depths)} depth frames")
    scene, variant = ls.scene_variant(args.dataset)
    commit = runzip.read_json(os.path.join(src, ".vendor.json"))["commit"]
    # As methods/loopsplat/adapter.py: VarSplat torch.load()s its own submap
    # checkpoints (torch >= 2.6 defaults to weights_only=True), and runs on one
    # GPU (faiss.index_cpu_to_all_gpus in gsr/overlap.py). Set before torch
    # and faiss are imported.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible = "0"
    elif len([d for d in visible.split(",") if d.strip()]) != 1:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={visible!r}: VarSplat must see exactly one GPU (gsr/overlap.py)")
    for p in (LOOPSPLAT_DIR, src, os.path.join(src, "thirdparty", "Hierarchical-Localization")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import faiss
    import torch

    import faiss_shim
    faiss_backend = faiss_shim.install(faiss)
    compat = ls.torch_compat(torch)
    for k, v in compat.items():
        print(f"compat: {k}: {v}", flush=True)
    print(f"VarSplat {commit} on {args.dataset}: {n} frames; faiss backend: {faiss_backend}", flush=True)

    seq_dir = ls.stage(args.dataset, work, n)
    config, overrides = build_config(src, scene, seq_dir, out_dir, cal)
    overrides["faiss"] = faiss_backend
    overrides["gpus"] = (f"CUDA_VISIBLE_DEVICES={visible}: one GPU (faiss.index_cpu_to_all_gpus in gsr/overlap.py "
                         "needs a single-GPU index)")
    overrides.update(compat)
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)
    runzip.write_json(os.path.join(work, "config_resolved.json"), config)

    from src.entities.gaussian_slam import GaussianSLAM
    from src.utils.utils import setup_seed

    overrides.update(varsplat_patches())
    setup_seed(config["seed"])
    t_init0 = time.time()
    slam = GaussianSLAM(config)  # dataset index, NetVLAD weights (downloaded once), output folders
    init_s = time.time() - t_init0
    patched, pgo_skipped, save_last_submap = ls.loopsplat_patches(slam)
    overrides.update(patched)
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)
    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    slam.run()
    torch.cuda.synchronize()
    loop_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"SLAM done in {loop_s:.0f} s, {len(pgo_skipped)} pose graph optimizations, peak GPU memory "
          f"{peak_gb:.2f} GB; exporting", flush=True)
    save_last_submap(n)

    out = os.path.join(work, "export")
    renders_dir = os.path.join(out, "renders")
    os.makedirs(renders_dir)
    est = torch.load(os.path.join(out_dir, "estimated_c2w.ckpt"), map_location="cpu", weights_only=False)
    est = est.numpy().astype(np.float64)
    if est.shape != (n, 4, 4) or not np.isfinite(est).all():
        raise ValueError(f"estimated_c2w: shape {est.shape}, finite {np.isfinite(est).all()}")
    protocol.write_tum(os.path.join(out, "traj_est.tum"), {i: est[i] for i in range(n)})
    keyframes, starts, n_gauss = ls.render_submaps(
        out_dir, est, lambda kfs: sorted(set(protocol.required_renders(n, kfs)) | set(ls.offset_frames(n, kfs))),
        renders_dir, cal)
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")
    runzip.write_figures(renders_dir, colors, os.path.join(out, "figures"))
    n_protocol = len(protocol.rendering_frames(n, keyframes))
    n_offset = len(ls.offset_frames(n, keyframes))
    gpu = torch.cuda.get_device_name(0)
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": loop_s, "gpu": f"{gpu}, 1 GPU",
        "what": ("GaussianSLAM.run(), first to last frame: frame reading (its dataset loader, inside the loop as MonoGS "
                 "reads its frames), tracking, mapping, submap starts and loop closures, cuda synchronized. Excludes "
                 f"construction ({init_s:.1f} s: dataset index, NetVLAD weights) and the export"),
        "sessions": 1, "init_s": init_s, "peak_gpu_memory_gb": peak_gb})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"VarSplat configs/Replica/{scene}.yaml (var: {config['var']}) with the MonoGS camera, W&B off, identity "
             f"traj.txt (frames 0-1 start at the identity, no reference poses), faiss: {faiss_backend}. "
             f"{len(starts)} submaps, {len(keyframes)} keyframes (mapping frames, every 5th + submap starts). Renders: "
             "final submaps after loop closure, before the offline global refinement, each frame with its own "
             f"submap. Protocol rendering frames left after excluding keyframes: {n_protocol}; scored instead on the "
             f"held-out grid ({protocol.HELDOUT_OFFSET} mod {protocol.RENDER_INTERVAL}, PROTOCOL.md): {n_offset} "
             f"frames. Peak GPU memory of the SLAM loop {peak_gb:.2f} GB.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "torch": torch.__version__, "faiss": faiss_backend,
        "frames": n, "keyframes": len(keyframes), "submaps": len(starts), "submap_starts": starts,
        "pose_graph_optimizations": len(pgo_skipped), "var": config["var"],
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
    # VarSplat's own logs (timing_gpu_log.txt, lc.log, tracker.log, mapper.log)
    # repeat what it prints, which is in run.log.
    runzip.package(work, ("overrides.json", "config_resolved.json",
                          os.path.join("varsplat_out", "estimated_c2w.ckpt")), map_file=map_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0: VarSplat cannot resume")
    p.add_argument("--resume", action="store_true", help="refused: VarSplat cannot resume")
    p.add_argument("--no_map", action="store_true", help="leave the final submaps out of the zip")
    p.add_argument("--varsplat", default=VENDORED, help="VarSplat source (default: the pinned copy in third_party/)")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
