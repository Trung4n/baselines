"""Scores one run of a SLAM method with the protocol of this repo's MonoGS
runs (baselines/PROTOCOL.md) and writes MonoGS-shaped outputs.

    python baselines/common/score.py \
        --method splatam --dataset /path/office0/lit \
        --traj <est.tum> [--keyframes <kf.txt>] \
        [--renders <dir of 000005.npy ...> | --no_render] \
        --timing <timing.json> --out baselines/results/<method>__<scene>_<variant>/<time> \
        [--method_commit <hash>] [--notes "..."] [--wandb | --no_wandb]

Inputs (written by the method's adapter, see methods/<name>/README.md):
- --traj: TUM file, timestamp = frame index, camera-to-world;
- --keyframes: one frame index per line (the method's keyframes / mapping
  frames); used for the MonoGS-style keyframe ATE and to exclude training
  views from the rendering metrics. Omit for methods without keyframes;
- --renders: <idx:06d>.npy (float [0,1]) or .png per frame of
  protocol.required_renders: the metric frames and the 5 figure frames;
- --timing: {"frames": N, "seconds": s, "gpu": "...", "what": "..."}.

Outputs under --out (same names as a MonoGS run where one exists):
- plot/stats_final.json, plot/evo_2dplot_final.png: keyframe ATE (MonoGS);
- plot/stats_final_all.json, plot/evo_2dplot_final_all.png: all-frame ATE;
- psnr/final/final_result.json: mean_psnr, mean_ssim, mean_lpips;
- psnr/final/per_frame.csv, traj_est.tum, metrics.json (everything + provenance);
- figures/<idx:06d>_render.png and _obs.jpg (the observed frame) for protocol.FIGURE_FRAMES.
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol  # noqa: E402


def find_render(renders, i):
    for ext in (".npy", ".png"):
        p = os.path.join(renders, f"{i:06d}{ext}")
        if os.path.isfile(p):
            return p
    return None


def write_figures(renders, colors, out):
    """figures/<idx>_render.png (and the observed _obs.jpg) of the figure
    frames, the same frames for every method (PROTOCOL.md)."""
    from PIL import Image

    os.makedirs(out, exist_ok=True)
    for i in protocol.FIGURE_FRAMES:
        if i >= len(colors):
            continue
        path = find_render(renders, i)
        if path is None:
            raise FileNotFoundError(f"no render for figure frame {i} in {renders} "
                                    "(protocol.required_renders: render it even if it is a keyframe)")
        Image.fromarray(protocol.to_png8(protocol.load_render(path))).save(os.path.join(out, f"{i:06d}_render.png"))
        shutil.copy(colors[i], os.path.join(out, f"{i:06d}_obs.jpg"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--keyframes", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--renders", default=None)
    g.add_argument("--no_render", action="store_true", help="the method has no renderable map")
    ap.add_argument("--timing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--monocular", action="store_true", help="Sim(3) alignment (camera only)")
    ap.add_argument("--method_commit", default="")
    ap.add_argument("--notes", default="")
    w = ap.add_mutually_exclusive_group(required=True)
    w.add_argument("--wandb", action="store_true")
    w.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    colors, _ = protocol.dataset_frames(args.dataset)
    n = len(colors)
    gt_list = protocol.gt_poses_c2w(args.dataset)
    if len(gt_list) != n:
        raise ValueError(f"{len(gt_list)} reference poses for {n} frames")
    gt = {i: gt_list[i] for i in range(n)}
    est = protocol.read_tum(args.traj)
    shutil.copy(args.traj, os.path.join(args.out, "traj_est.tum"))
    keyframes = []
    if args.keyframes:
        keyframes = [int(x) for x in open(args.keyframes).read().split()]
        shutil.copy(args.keyframes, os.path.join(args.out, "keyframes.txt"))
    timing = json.load(open(args.timing))
    for key in ("frames", "seconds", "gpu", "what"):
        if key not in timing:
            raise KeyError(f"{args.timing}: missing {key!r} (PROTOCOL.md, FPS)")

    plot_dir = os.path.join(args.out, "plot")
    missing = [i for i in range(n) if i not in est]
    ate_all, n_all = protocol.ate(est, gt, list(range(n)), plot_dir, "final_all", args.monocular)
    ate_kf, n_kf = (protocol.ate(est, gt, keyframes, plot_dir, "final", args.monocular)
                    if keyframes else (float("nan"), 0))

    render = None
    if not args.no_render:
        frames = protocol.rendering_frames(n, keyframes)
        scorer = protocol.RenderScorer()
        rows, quantized = [], False
        for i in frames:
            path = find_render(args.renders, i)
            if path is None:
                raise FileNotFoundError(f"no render for frame {i} in {args.renders} (needed: {len(frames)} frames)")
            quantized |= path.endswith(".png")
            s = scorer(protocol.load_render(path), protocol.load_observed(colors[i]))
            rows.append({"frame": i, **s})
        mean = {f"mean_{k}": float(sum(r[k] for r in rows) / len(rows)) for k in ("psnr", "ssim", "lpips")}
        psnr_dir = os.path.join(args.out, "psnr", "final")
        os.makedirs(psnr_dir, exist_ok=True)
        json.dump(mean, open(os.path.join(psnr_dir, "final_result.json"), "w"), indent=4)
        with open(os.path.join(psnr_dir, "per_frame.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim", "lpips"])
            wr.writeheader()
            wr.writerows(rows)
        render = {**mean, "frames": len(rows), "quantized_8bit": quantized}
        write_figures(args.renders, colors, os.path.join(args.out, "figures"))

    fps = timing["frames"] / timing["seconds"]
    scene = os.path.basename(os.path.dirname(os.path.normpath(args.dataset)))
    variant = os.path.basename(os.path.normpath(args.dataset))
    metrics = {
        "method": args.method, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "frames_dataset": n, "frames_estimated": len(est), "frames_missing": len(missing),
        "ate_all_m": ate_all, "ate_all_frames": n_all,
        "ate_kf_m": ate_kf, "ate_kf_frames": n_kf,
        "alignment": "Sim3" if args.monocular else "SE3",
        "render": render, "fps": fps, "timing": timing,
        "method_commit": args.method_commit, "notes": args.notes,
        "scored_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps({k: metrics[k] for k in ("method", "scene", "variant", "ate_all_m", "ate_kf_m", "fps")}))
    if render:
        print(json.dumps(render))

    if args.wandb:
        import wandb

        name = f"{args.method}__{scene}_{variant}_{os.path.basename(os.path.normpath(args.out))}"
        run = wandb.init(project="MonoGS", name=name, config={k: metrics[k] for k in (
            "method", "scene", "variant", "dataset", "timing", "method_commit", "notes")}, tags=["baseline"])
        table = wandb.Table(columns=["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS", "ATE all frames"])
        table.add_data("Final", *(render[f"mean_{k}"] if render else None for k in ("psnr", "ssim", "lpips")),
                       ate_kf if keyframes else None, fps, ate_all)
        wandb.log({"Metrics": table})
        run.summary.update({"ate": ate_kf if keyframes else ate_all, "ate_all": ate_all, "fps": fps,
                            **({f"{k}": render[f"mean_{k}"] for k in ("psnr", "ssim", "lpips")} if render else {})})
        wandb.finish()


if __name__ == "__main__":
    main()
