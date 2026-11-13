"""Shared pieces of the adapters' output: figures of the common frames and
the one zip per run (PROTOCOL.md, "Gói kết quả của một lần chạy").

An adapter fills <work>/export/ with traj_est.tum, keyframes.txt (if any),
renders/<idx:06d>.npy (if it has a renderable map), timing.json,
calibration.json, method.json and notes.txt, then calls package(). Runs in
the method's own environment: no MonoGS import here.

(methods/splatam/adapter.py predates this module and has its own copy of
the same logic; same zip layout.)
"""
import json
import os
import shutil
import zipfile

import protocol

SUMMARY_KEYS = ("frames", "keyframes", "renders", "gaussians")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_figures(renders_dir, colors, out):
    """figures/<idx>_render.png and _obs.jpg of protocol.FIGURE_FRAMES, as
    score.py writes them (a look at the run before scoring)."""
    from PIL import Image

    os.makedirs(out, exist_ok=True)
    for i in protocol.FIGURE_FRAMES:
        if i >= len(colors):
            continue
        path = os.path.join(renders_dir, f"{i:06d}.npy")
        if not os.path.isfile(path):  # e.g. a frame without a pose: noted by the adapter, decided at scoring
            print(f"figures: no render of frame {i}, skipped", flush=True)
            continue
        image = protocol.load_render(path)
        Image.fromarray(protocol.to_png8(image)).save(os.path.join(out, f"{i:06d}_render.png"))
        shutil.copy(colors[i], os.path.join(out, f"{i:06d}_obs.jpg"))


def package(work, run_files, map_file=None):
    """<work>/../<method>__<scene>_<variant>__<started_at>.zip: <work>/export/
    at the top level, run/ with run_files (paths relative to <work>; the log
    <work>/run.log when present) and map_file (absolute path or None), and
    summary.json. Returns the zip path."""
    work = os.path.abspath(work)
    out = os.path.join(work, "export")
    method = read_json(os.path.join(out, "method.json"))
    timing = read_json(os.path.join(out, "timing.json"))
    name = f"{method['method']}__{method['scene']}_{method['variant']}__{method['started_at']}"
    zip_path = os.path.join(os.path.dirname(work), name + ".zip")
    members = []  # (path on disk, path in the zip)
    for root, _, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(root, f)
            members.append((p, os.path.relpath(p, out).replace(os.sep, "/")))
    for f in run_files:
        p = os.path.join(work, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        members.append((p, "run/" + f.replace(os.sep, "/")))
    log = os.path.join(work, "run.log")
    if os.path.isfile(log):
        members.append((log, "run/run.log"))
    if map_file is not None:
        members.append((map_file, "run/" + os.path.basename(map_file)))
    summary = {
        "method": method["method"], "scene": method["scene"], "variant": method["variant"],
        "started_at": method["started_at"], "commit": method["commit"],
        **{k: method[k] for k in SUMMARY_KEYS if k in method},
        "fps": timing["frames"] / timing["seconds"], "seconds": timing["seconds"],
        "sessions": timing.get("sessions", 1), "gpu": timing["gpu"],
        "map_included": map_file is not None,
        "score": "python baselines/common/score_zip.py --zip <this zip> --dataset <local dataset folder> --no_wandb",
        "files": sorted(arc for _, arc in members) + ["summary.json"],
    }
    tmp = zip_path + ".tmp"
    print(f"package: zipping {len(members)} files", flush=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for j, (p, arc) in enumerate(members):
            z.write(p, f"{name}/{arc}")
            if (j + 1) % 100 == 0:
                print(f"package: {j + 1}/{len(members)}", flush=True)
        z.writestr(f"{name}/summary.json", json.dumps(summary, indent=2))
    os.replace(tmp, zip_path)
    print(f"{len(members) + 1} files, {os.path.getsize(zip_path) / 2**20:.0f} MiB -> {zip_path}", flush=True)
    print(json.dumps({k: summary[k] for k in ("frames", "keyframes", "gaussians", "fps") if k in summary}), flush=True)
    return zip_path
