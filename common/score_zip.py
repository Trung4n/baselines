"""Scores the zip of one run of another method (written by its adapter on
Kaggle, layout in baselines/PROTOCOL.md) on the local machine, inside the
MonoGS repo and environment.

    python baselines/common/score_zip.py --zip <method>__<scene>_<variant>__<start>.zip \
        --dataset <local dataset folder, e.g. .../monogs/office0/lit> (--wandb | --no_wandb)

1. Extracts the zip to baselines/runs/<zip name>/ (git-ignored: renders and
   the map are large). An existing complete extraction is reused.
2. Checks the run against the local dataset and MonoGS: calibration.json
   equals configs/rgbd/replica/base_config.yaml (protocol.check_calibration),
   same scene / variant and frame count.
3. Runs score.py into baselines/results/<method>__<scene>_<variant>/<start>/
   (refuses an existing folder), then copies the run's record (summary,
   method, calibration, configs, timing segments, log; not the renders or the
   map) into its run/ subfolder, with source.json (zip name, size, sha256).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

COMMON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMMON)
import protocol  # noqa: E402
import score  # noqa: E402

BASELINES = os.path.dirname(COMMON)
RECORD_FILES = ("summary.json", "method.json", "calibration.json", "timing.json", "notes.txt")
SKIP_RUN_FILES = ("params.npz",)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(zip_path, runs_dir):
    """baselines/runs/<top folder of the zip>/, refusing paths that leave it."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1:
            raise ValueError(f"{zip_path}: expected one top folder, got {sorted(tops)}")
        top = tops.pop()
        dest = os.path.join(runs_dir, top)
        root = os.path.realpath(runs_dir)
        for n in names:
            target = os.path.realpath(os.path.join(runs_dir, n))
            if not target.startswith(root + os.sep):
                raise ValueError(f"{zip_path}: unsafe path {n!r}")
        done = os.path.join(dest, ".extracted")
        if os.path.isfile(done):
            print(f"reusing {dest}")
            return dest
        if os.path.exists(dest):
            shutil.rmtree(dest)
        z.extractall(runs_dir)
        open(done, "w").close()
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dataset", required=True, help="local MonoGS dataset folder of the run's scene / variant")
    w = ap.add_mutually_exclusive_group(required=True)
    w.add_argument("--wandb", action="store_true")
    w.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--runs_dir", default=os.path.join(BASELINES, "runs"))
    ap.add_argument("--results_dir", default=os.path.join(BASELINES, "results"))
    args = ap.parse_args(argv)

    run = extract(args.zip, args.runs_dir)

    def load(name):
        with open(os.path.join(run, name), encoding="utf-8") as f:
            return json.load(f)

    summary, method, cal = load("summary.json"), load("method.json"), load("calibration.json")
    protocol.check_calibration(cal)
    scene = os.path.basename(os.path.dirname(os.path.normpath(args.dataset)))
    variant = os.path.basename(os.path.normpath(args.dataset))
    if (scene, variant) != (method["scene"], method["variant"]):
        raise ValueError(f"run is {method['scene']}/{method['variant']}, --dataset is {scene}/{variant}")
    colors, _ = protocol.dataset_frames(args.dataset)
    if len(colors) != method["frames"]:
        raise ValueError(f"run has {method['frames']} frames, {args.dataset} has {len(colors)}")

    out = os.path.join(args.results_dir, f"{method['method']}__{scene}_{variant}", method["started_at"])
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: results are never overwritten")
    renders = os.path.join(run, "renders")
    cmd = ["--method", method["method"], "--dataset", args.dataset,
           "--traj", os.path.join(run, "traj_est.tum"), "--timing", os.path.join(run, "timing.json"),
           "--out", out, "--method_commit", method["commit"], "--notes", method["notes"],
           "--wandb" if args.wandb else "--no_wandb"]
    if os.path.isfile(os.path.join(run, "keyframes.txt")):
        cmd += ["--keyframes", os.path.join(run, "keyframes.txt")]
    cmd += ["--renders", renders] if os.path.isdir(renders) else ["--no_render"]
    score.main(cmd)

    record = os.path.join(out, "run")
    os.makedirs(record)
    for f in RECORD_FILES:
        if os.path.isfile(os.path.join(run, f)):
            shutil.copy(os.path.join(run, f), record)
    if os.path.isdir(os.path.join(run, "run")):
        for f in sorted(os.listdir(os.path.join(run, "run"))):
            if f not in SKIP_RUN_FILES:
                shutil.copy(os.path.join(run, "run", f), record)
    with open(os.path.join(record, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"zip": os.path.basename(args.zip), "bytes": os.path.getsize(args.zip),
                   "sha256": sha256(args.zip), "extracted_to": os.path.relpath(run, BASELINES),
                   "map_included": summary.get("map_included")}, f, indent=2)
    print(f"scored -> {out}")


if __name__ == "__main__":
    main()
