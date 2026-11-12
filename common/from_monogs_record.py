"""This repo's own MonoGS runs, scored with the SAME all-frame protocol as the
other methods (baselines/PROTOCOL.md), from a run recording (slam.py --record).

MonoGS itself reports ATE on its keyframes only (utils/eval_utils.eval_ate).
For the comparison table every method also gets the ATE over all frames; this
script writes the TUM trajectory (record/frames_final.csv: final poses, BA-
refined keyframes, live poses elsewhere) and the keyframe list, ready for
score.py --no_render (rendering metrics: take MonoGS's own, eval_rendering).

With --figures it also renders the figure frames (protocol.FIGURE_FRAMES)
from the recorded map at the final poses, through the run's own light model
(tools/verify/eval_map.MapRenderer, the run's resolved config and learned
light parameters): figures/<idx>_render.png and _obs.jpg, as score.py writes
them for the other methods. --map final = the online map (before the colour
refinement, the main column); final_refined = after it.

    python baselines/common/from_monogs_record.py --record record/runs/<run> --out <dir> \
        [--figures --dataset <dataset folder> --map final|final_refined]
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol  # noqa: E402


def read_record(rec):
    poses, kfs = {}, []
    with open(os.path.join(rec, "frames_final.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = np.eye(4)
            m[:3] = np.array([float(r[f"est_{a}{b}"]) for a in range(3) for b in range(4)]).reshape(3, 4)
            i = int(r["frame_idx"])
            poses[i] = m
            if r["is_keyframe"] == "1":
                kfs.append(i)
    return poses, kfs


def render_figures(run, rec, dataset, poses, which, out):
    """figures/<idx>_render.png of the recorded map at the final poses."""
    import json

    import torch
    from PIL import Image

    sys.path[:0] = [protocol.REPO, os.path.join(protocol.REPO, "tools", "verify")]
    import eval_map
    import eval_records
    from light_models import build_shader
    from utils.config_utils import load_config

    cfg_path, _ = eval_records.resolved_config(run)
    config = load_config(cfg_path)
    config["Dataset"]["dataset_path"] = dataset
    shader = build_shader(config)
    csv_path = os.path.join(run, "light_learned.csv")
    if shader is not None and os.path.isfile(csv_path):
        shader.load_learnable_state(eval_map.learned_state(csv_path))
    with open(os.path.join(rec, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(os.path.join(rec, "maps", which + ".npz")) as z:
        snap = {k: z[k] for k in z.files}
    renderer = eval_map.MapRenderer(snap, shader, meta["background"], meta["intrinsics"])
    depth_scale = config["Dataset"]["Calibration"]["depth_scale"]
    colors, depths = protocol.dataset_frames(dataset)
    os.makedirs(out, exist_ok=True)
    for i in protocol.FIGURE_FRAMES:
        depth = np.asarray(Image.open(depths[i]), np.float64) / depth_scale  # sensor-depth normals
        image, _, _ = renderer(poses[i], depth)
        t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        Image.fromarray(protocol.to_png8(t)).save(os.path.join(out, f"{i:06d}_render.png"))
        shutil.copy(colors[i], os.path.join(out, f"{i:06d}_obs.jpg"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="run folder holding record/ (or record/ itself)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--dataset", default=None, help="the run's dataset folder (needed with --figures)")
    ap.add_argument("--map", default="final", choices=["final", "final_refined"])
    args = ap.parse_args(argv)
    rec = args.record if os.path.isfile(os.path.join(args.record, "frames_final.csv")) else os.path.join(args.record, "record")
    run = os.path.dirname(os.path.normpath(rec)) if os.path.basename(os.path.normpath(rec)) == "record" else args.record
    poses, kfs = read_record(rec)
    os.makedirs(args.out, exist_ok=True)
    protocol.write_tum(os.path.join(args.out, "traj_est.tum"), poses)
    with open(os.path.join(args.out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in kfs) + "\n")
    print(f"{len(poses)} poses, {len(kfs)} keyframes -> {args.out}")
    if args.figures:
        if not args.dataset:
            raise SystemExit("--figures needs --dataset")
        sub = "figures" if args.map == "final" else "figures_refined"
        render_figures(run, rec, args.dataset, poses, args.map, os.path.join(args.out, sub))
        print(f"figures ({args.map} map) -> {os.path.join(args.out, sub)}")


if __name__ == "__main__":
    main()
