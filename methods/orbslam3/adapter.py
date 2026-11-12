"""ORB-SLAM3 RGB-D (Campos et al., T-RO 2021) on a MonoGS Replica-format
dataset. Runs on Kaggle with the baselines repo alone and ends with ONE zip
per run, scored later on the local machine by baselines/common/score_zip.py
with --no_render (ORB-SLAM3 has no renderable map). Contract of
common/kaggle_runner.py; built by setup_kaggle.sh into <tmp>/orbslam3_build.

    python methods/orbslam3/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--build <dir>]

ORB-SLAM3 has no checkpoints (a run takes minutes): --checkpoint_every must
be 0, --resume is refused; --no_map changes nothing (no map is packed).

What is run: rgbd_monogs.cc, the loop of ORB-SLAM3's Examples/RGB-D/rgbd_tum.cc
with its I/O adapted (see that file), on the settings of
Examples/RGB-D/TUM1.yaml with the dataset camera. What differs, all recorded
in <work>/overrides.json and README.md:
- camera: protocol.calibration() (fx, fy, cx, cy, width, height); no
  distortion keys (the dataset is undistorted); RGBD.DepthMapFactor =
  depth_scale;
- Camera.RGB = 0: frames are read with cv::imread, i.e. BGR (TUM1.yaml says
  1 while rgbd_tum also reads BGR; only the grey conversion uses it);
- every other key of TUM1.yaml kept: 1000 ORB features, 8 levels, scale 1.2,
  FAST 20/7, Camera.fps 30 (frame i gets timestamp i / 30, and the loop waits
  for that pace, as rgbd_tum), Stereo.ThDepth 40 and Stereo.b 0.07732 m
  (close/far threshold 3.1 m, in metres, so independent of the focal length);
- viewer off; loop closure and multi-map atlas as default.
The trajectory scored is the map with the most keyframes
(SaveTrajectoryEuRoC); frames tracked in another map of the atlas (after a
tracking loss) are in other coordinates and count as missing. The every-map
trajectory of rgbd_tum (SaveTrajectoryTUM) is kept in the zip.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASELINES, "common"))
import protocol  # noqa: E402
import runzip  # noqa: E402

METHOD = "orbslam3"
VENDORED = os.path.join(BASELINES, "third_party", "ORB_SLAM3")
PANGOLIN = os.path.join(BASELINES, "third_party", "Pangolin")
SETTINGS = os.path.join("Examples", "RGB-D", "TUM1.yaml")
# setup_kaggle.sh builds into ${TMPDIR:-/tmp}/orbslam3_build
DEFAULT_BUILD = os.path.join(tempfile.gettempdir(), "orbslam3_build")
DRIVER = os.path.join("ORB_SLAM3", "Examples", "RGB-D", "rgbd_monogs")
VOCABULARY = os.path.join("ORB_SLAM3", "Vocabulary", "ORBvoc.txt")
TRACKING_STATES = {-1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED", 2: "OK",
                   3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT"}
RAW_FILES = ("CameraTrajectory.txt", "KeyFrameTrajectory.txt", "CameraTrajectory_main_map.txt",
             "KeyFrameTrajectory_main_map.txt", "states.csv", "driver_timing.json")


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def vendor_commit(path):
    return runzip.read_json(os.path.join(path, ".vendor.json"))["commit"]


# ---------------------------------------------------------------- settings

def settings(src, work, cal):
    """TUM1.yaml with the dataset camera (line edits: it is an OpenCV
    FileStorage file, %YAML:1.0, not plain YAML). Returns (path, fps,
    changes)."""
    with open(os.path.join(src, SETTINGS), encoding="utf-8") as f:
        lines = f.read().splitlines()
    set_to = {"Camera1.fx": cal["fx"], "Camera1.fy": cal["fy"], "Camera1.cx": cal["cx"], "Camera1.cy": cal["cy"],
              "Camera.width": cal["width"], "Camera.height": cal["height"], "Camera.RGB": 0,
              "RGBD.DepthMapFactor": cal["depth_scale"]}
    drop = ("Camera1.k1", "Camera1.k2", "Camera1.p1", "Camera1.p2", "Camera1.k3")
    seen, changes, out, fps = set(), {}, [], None
    for line in lines:
        m = re.match(r"^([A-Za-z0-9_.]+):\s*([^#]*?)\s*(#.*)?$", line)
        key = m.group(1) if m else None
        if key in drop:
            changes[key] = f"{m.group(2)} -> removed (undistorted images)"
            seen.add(key)
            continue
        if key in set_to:
            changes[key] = f"{m.group(2)} -> {set_to[key]}"
            line = f"{key}: {set_to[key]}" + (f" {m.group(3)}" if m.group(3) else "")
            seen.add(key)
        if key == "Camera.fps":
            fps = float(m.group(2))
        out.append(line)
    missing = [k for k in (*set_to, *drop) if k not in seen]
    if missing or fps is None:
        raise KeyError(f"{SETTINGS}: keys not found {missing}, Camera.fps {fps}")
    path = os.path.join(work, "settings_rgbd.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path, fps, changes


# ---------------------------------------------------------------- outputs

def read_trajectory(path, fps):
    """{frame index: 4x4 camera-to-world} from a TUM-layout file whose time is
    seconds or nanoseconds (ORB-SLAM3's TUM and EuRoC savers); frame i was
    given timestamp i / fps."""
    from scipy.spatial.transform import Rotation

    rows = [line.split() for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#")]
    if not rows:
        return {}
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] != 8:
        raise ValueError(f"{path}: {arr.shape[1]} columns")
    t = arr[:, 0] * (1e-9 if arr[:, 0].max() > 1e6 else 1.0)
    idx = np.rint(t * fps)
    if np.abs(t * fps - idx).max() > 1e-3:
        raise ValueError(f"{path}: timestamps are not multiples of 1/{fps}")
    poses = {}
    for i, row in zip(idx.astype(int), arr):
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()  # qx qy qz qw
        m[:3, 3] = row[1:4]
        poses[int(i)] = m
    return poses


def run_driver(cmd):
    """Runs the driver with its output passed through (the runner shows and
    logs it) and returns that output for parsing."""
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"rgbd_monogs exited with {code}")
    return "".join(lines)


# ---------------------------------------------------------------- run

def cmd_run(args):
    work, build = os.path.abspath(args.work), os.path.abspath(args.build)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("ORB-SLAM3 has no checkpoints (one run takes minutes): "
                         "use --checkpoint_every 0 and no --resume")
    raw = os.path.join(work, "orbslam3_out")
    if os.path.exists(raw) or os.path.exists(os.path.join(work, "export")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    driver, vocab = os.path.join(build, DRIVER), os.path.join(build, VOCABULARY)
    for p in (driver, vocab):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p}: run methods/orbslam3/setup_kaggle.sh first (builds into {build})")
    os.makedirs(raw)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    expect = [os.path.join(args.dataset, "results", f"frame{i:06d}.jpg") for i in range(n)]
    if [os.path.normpath(c) for c in colors] != [os.path.normpath(e) for e in expect] or len(depths) != n:
        raise ValueError(f"{args.dataset}: frames are not frame000000.jpg ... with matching depth*.png")
    commit = vendor_commit(VENDORED)
    scene, variant = scene_variant(args.dataset)
    settings_path, fps, changes = settings(VENDORED, work, cal)
    overrides = {"settings": f"{SETTINGS} with: {changes}", "viewer": "off",
                 "input": f"results/frame%06d.jpg, depth%06d.png, frame i at timestamp i / {fps:g}",
                 "trajectory_scored": "map with the most keyframes (SaveTrajectoryEuRoC)"}
    runzip.write_json(os.path.join(work, "overrides.json"), overrides)
    print(f"ORB-SLAM3 {commit} on {args.dataset}: {n} frames at {fps:g} Hz timestamps", flush=True)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log = run_driver([driver, vocab, settings_path, os.path.abspath(args.dataset), str(n), f"{fps:g}", raw])
    for f in RAW_FILES + ("driver_done",):
        if not os.path.isfile(os.path.join(raw, f)):
            raise RuntimeError(f"rgbd_monogs ended without {f}: see the output above")
    export(args, work, raw, log, n, fps, cal, commit, scene, variant, started_at, changes)
    runzip.package(work, ("settings_rgbd.yaml", "overrides.json") + tuple(os.path.join("orbslam3_out", f)
                                                                          for f in RAW_FILES))


def export(args, work, raw, log, n, fps, cal, commit, scene, variant, started_at, changes):
    out = os.path.join(work, "export")
    os.makedirs(out)
    main = read_trajectory(os.path.join(raw, "CameraTrajectory_main_map.txt"), fps)
    every = read_trajectory(os.path.join(raw, "CameraTrajectory.txt"), fps)
    kfs = read_trajectory(os.path.join(raw, "KeyFrameTrajectory_main_map.txt"), fps)
    if not main or max(main) >= n:
        raise ValueError(f"main-map trajectory: {len(main)} frames, max index {max(main) if main else None}")
    protocol.write_tum(os.path.join(out, "traj_est.tum"), main)
    keyframes = sorted(kfs)
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")

    states = np.genfromtxt(os.path.join(raw, "states.csv"), delimiter=",", skip_header=1)
    counts = {TRACKING_STATES.get(int(s), str(int(s))): int((states[:, 1] == s).sum()) for s in np.unique(states[:, 1])}
    m = re.search(r"There are (\d+) maps in the atlas", log)
    n_maps = int(m.group(1)) if m else None
    dt = runzip.read_json(os.path.join(raw, "driver_timing.json"))
    if dt["frames"] != n:
        raise ValueError(f"driver ran {dt['frames']} frames, dataset has {n}")
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": dt["loop_s"],
        "gpu": f"CPU only (ORB-SLAM3 does not use the GPU), {os.cpu_count()} logical cores",
        "what": ("rgbd_monogs loop, first to last frame: frame reading (cv::imread, inside the loop as MonoGS reads "
                 f"its frames), TrackRGBD, and the wait for the next timestamp (frames at {fps:g} Hz, as rgbd_tum), "
                 "with local mapping and loop closing in their own threads. Excludes system start-up "
                 f"(vocabulary loading, {dt['init_s']:.1f} s), Shutdown and the trajectory saving"),
        "sessions": 1, "init_s": dt["init_s"], "loop_s": dt["loop_s"],
        "track_mean_s": dt["track_mean_s"], "track_median_s": dt["track_median_s"],
        "fps_cap": fps})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    notes = (f"ORB-SLAM3 v1.0 RGB-D, rgbd_tum loop (frames paced at {fps:g} Hz), TUM1.yaml with the MonoGS camera, "
             "no distortion, Camera.RGB 0 (BGR from imread), viewer off, loop closure on (MonoGS has none). "
             f"Atlas: {n_maps} map(s); scored trajectory = the largest map, {len(main)} of {n} frames "
             f"({len(every)} frames in all maps). Tracking states: {counts}. No renderable map.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "pangolin": vendor_commit(PANGOLIN),
        "frames": n, "keyframes": len(keyframes), "frames_in_main_map": len(main),
        "frames_in_all_maps": len(every), "maps": n_maps, "tracking_states": counts,
        "settings_changes": changes, "overrides": runzip.read_json(os.path.join(work, "overrides.json")),
        "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{len(main)}/{n} frames in the main map ({n_maps} maps), {len(keyframes)} keyframes, "
          f"{n / dt['loop_s']:.2f} FPS (frames paced at {fps:g} Hz; mean tracking "
          f"{dt['track_mean_s'] * 1000:.1f} ms) -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0: ORB-SLAM3 has no checkpoints")
    p.add_argument("--resume", action="store_true", help="refused: ORB-SLAM3 has no checkpoints")
    p.add_argument("--no_map", action="store_true", help="no effect: ORB-SLAM3 has no renderable map to pack")
    p.add_argument("--build", default=DEFAULT_BUILD, help="where setup_kaggle.sh built ORB-SLAM3")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
