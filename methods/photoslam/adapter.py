"""Photo-SLAM RGB-D (Huang et al., CVPR 2024) on a MonoGS Replica-format
dataset. Runs on Kaggle with the baselines repo alone and ends with ONE zip
per run, scored later on the local machine by baselines/common/score_zip.py.
Contract of common/kaggle_runner.py; built by setup_kaggle.sh into
<tmp>/photoslam_build.

    python methods/photoslam/adapter.py run --dataset <.../office0/lit> --work <dir> \
        --checkpoint_every 0 [--no_map] [--build <dir>]

Photo-SLAM has no checkpoints: --checkpoint_every must be 0, --resume is
refused.

What is run: photoslam_monogs.cpp, the loop of Photo-SLAM's
examples/replica_rgbd.cpp with its I/O adapted (see that file), on Photo-SLAM's
own Replica settings for the scene: cfg/ORB_SLAM3/RGB-D/Replica/<scene>.yaml
and cfg/gaussian_mapper/RGB-D/Replica/<scene>.yaml. What differs, all recorded
in <work>/overrides.json and README.md:
- ORB-SLAM3 camera: protocol.calibration() (fx, fy, cx, cy, width, height,
  RGBD.DepthMapFactor); the distortion keys of Photo-SLAM's Replica files are
  removed (the dataset is undistorted); every other key kept (1600 ORB
  features, Camera.RGB 1 as the example turns frames to RGB, ...);
- Gaussian mapper settings: unchanged;
- viewer off.
Scored: the map with the most keyframes, poses in that map's world frame
(where the Gaussians are); frames of other maps count as missing. Renders: the
final Gaussian map (after the mapper's own final training) at those poses,
full resolution (GaussianMapper::renderFromPose), for protocol.required_renders.
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASELINES, "common"))
import protocol  # noqa: E402
import runzip  # noqa: E402

METHOD = "photoslam"
VENDORED = os.path.join(BASELINES, "third_party", "Photo-SLAM")
ORB_SETTINGS = os.path.join("cfg", "ORB_SLAM3", "RGB-D", "Replica", "{scene}.yaml")
MAPPER_SETTINGS = os.path.join("cfg", "gaussian_mapper", "RGB-D", "Replica", "{scene}.yaml")
DEFAULT_BUILD = os.path.join(tempfile.gettempdir(), "photoslam_build")  # setup_kaggle.sh builds there
DRIVER = os.path.join("Photo-SLAM", "bin", "photoslam_monogs")
VOCABULARY = os.path.join("Photo-SLAM", "ORB-SLAM3", "Vocabulary", "ORBvoc.txt")
RUNTIME_LIBS = "runtime_ld_library_path"  # written by setup_kaggle.sh
TRACKING_STATES = {-1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED", 2: "OK",
                   3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT"}
RAW_FILES = ("trajectory_main_map_world.txt", "keyframes_main_map.txt", "states.csv", "driver_timing.json",
             "renders_without_pose.txt", "CameraTrajectory_TUM.txt", "KeyFrameTrajectory_TUM.txt",
             "CameraTrajectory_EuRoC.txt")


def scene_variant(dataset):
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def vendor_commit(path):
    return runzip.read_json(os.path.join(path, ".vendor.json"))["commit"]


def orb_settings(src, scene, work, cal):
    """Photo-SLAM's ORB-SLAM3 Replica settings of the scene with the dataset
    camera (line edits: OpenCV FileStorage file). Returns (path, changes)."""
    path_in = os.path.join(src, ORB_SETTINGS.format(scene=scene))
    if not os.path.isfile(path_in):
        raise FileNotFoundError(f"{path_in}: Photo-SLAM has no Replica settings for scene {scene!r}")
    with open(path_in, encoding="utf-8") as f:
        lines = f.read().splitlines()
    set_to = {"Camera1.fx": cal["fx"], "Camera1.fy": cal["fy"], "Camera1.cx": cal["cx"], "Camera1.cy": cal["cy"],
              "Camera.width": cal["width"], "Camera.height": cal["height"], "RGBD.DepthMapFactor": cal["depth_scale"]}
    drop = ("Camera1.k1", "Camera1.k2", "Camera1.k3", "Camera1.p1", "Camera1.p2")
    seen, changes, out = set(), {}, []
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
        out.append(line)
    missing = [k for k in (*set_to, *drop) if k not in seen]
    if missing:
        raise KeyError(f"{path_in}: keys not found {missing}")
    path = os.path.join(work, "orb_settings.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path, changes


def run_driver(cmd, build):
    """The driver with its output passed through, with the LD_LIBRARY_PATH
    setup_kaggle.sh tested it with (<build>/runtime_ld_library_path: the CUDA
    toolkit it was built with, the NVIDIA CUDA driver libraries; not the
    notebook's, whose NVIDIA GL libraries break the system ones, see
    methods/orbslam3/README.md)."""
    with open(os.path.join(build, RUNTIME_LIBS), encoding="utf-8") as f:
        libs = f.read().strip()
    print(f"LD_LIBRARY_PATH={libs}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env["LD_LIBRARY_PATH"] = libs
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            env=env)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if proc.wait() != 0:
        raise RuntimeError(f"photoslam_monogs exited with {proc.returncode}")
    return "".join(lines)


def cmd_run(args):
    work, build = os.path.abspath(args.work), os.path.abspath(args.build)
    if args.checkpoint_every != 0 or args.resume:
        raise SystemExit("Photo-SLAM has no checkpoints: use --checkpoint_every 0 and no --resume")
    raw = os.path.join(work, "photoslam_out")
    if os.path.exists(raw) or os.path.exists(os.path.join(work, "export")):
        raise SystemExit(f"{work} already holds a run: use a new --work")
    driver, vocab = os.path.join(build, DRIVER), os.path.join(build, VOCABULARY)
    for p in (driver, vocab, os.path.join(build, RUNTIME_LIBS)):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p}: run methods/photoslam/setup_kaggle.sh first (builds into {build})")
    os.makedirs(raw)
    cal = protocol.calibration()
    colors, depths = protocol.dataset_frames(args.dataset)
    n = len(colors)
    expect = [os.path.join(args.dataset, "results", f"frame{i:06d}.jpg") for i in range(n)]
    if [os.path.normpath(c) for c in colors] != [os.path.normpath(e) for e in expect] or len(depths) != n:
        raise ValueError(f"{args.dataset}: frames are not frame000000.jpg ... with matching depth*.png")
    scene, variant = scene_variant(args.dataset)
    commit = vendor_commit(VENDORED)
    orb_path, changes = orb_settings(VENDORED, scene, work, cal)
    mapper_src = os.path.join(VENDORED, MAPPER_SETTINGS.format(scene=scene))
    mapper_path = os.path.join(work, "mapper_settings.yaml")
    shutil.copyfile(mapper_src, mapper_path)
    rule = os.path.join(work, "render_rule.txt")
    with open(rule, "w", encoding="utf-8") as f:
        f.write(f"interval {protocol.RENDER_INTERVAL}\nfigures {' '.join(str(i) for i in protocol.FIGURE_FRAMES)}\n")
    runzip.write_json(os.path.join(work, "overrides.json"), {
        "orb_settings": f"{ORB_SETTINGS.format(scene=scene)} with: {changes}",
        "mapper_settings": f"{MAPPER_SETTINGS.format(scene=scene)}, unchanged",
        "viewer": "off", "input": "results/frame%06d.jpg, depth%06d.png, timestamp = frame index (as replica_rgbd)",
        "trajectory_scored": "map with the most keyframes, in its own world frame"})
    print(f"Photo-SLAM {commit} on {args.dataset}: {n} frames", flush=True)

    started_at = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log = run_driver([driver, vocab, orb_path, mapper_path, os.path.abspath(args.dataset), str(n), rule, raw], build)
    for f in RAW_FILES + ("driver_done",):
        if not os.path.isfile(os.path.join(raw, f)):
            raise RuntimeError(f"photoslam_monogs ended without {f}: see the output above")
    plys = sorted(glob.glob(os.path.join(raw, "mapper", "*_shutdown", "ply", "point_cloud", "iteration_*",
                                         "point_cloud.ply")))
    export(args, work, raw, log, n, cal, commit, scene, variant, started_at, changes, build)
    runzip.package(work, ("orb_settings.yaml", "mapper_settings.yaml", "render_rule.txt", "overrides.json")
                   + tuple(os.path.join("photoslam_out", f) for f in RAW_FILES),
                   map_file=None if args.no_map or not plys else plys[-1])


def export(args, work, raw, log, n, cal, commit, scene, variant, started_at, changes, build):
    out = os.path.join(work, "export")
    renders_out = os.path.join(out, "renders")
    os.makedirs(renders_out)
    dt = runzip.read_json(os.path.join(raw, "driver_timing.json"))
    if dt["frames"] != n or (dt["width"], dt["height"]) != (cal["width"], cal["height"]):
        raise ValueError(f"driver: {dt['frames']} frames at {dt['width']}x{dt['height']}, dataset {n} at "
                         f"{cal['width']}x{cal['height']}")
    main = protocol.read_tum(os.path.join(raw, "trajectory_main_map_world.txt"))
    protocol.write_tum(os.path.join(out, "traj_est.tum"), main)
    keyframes = [int(x) for x in open(os.path.join(raw, "keyframes_main_map.txt")).read().split()]
    with open(os.path.join(out, "keyframes.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keyframes) + "\n")
    # renders: float32 from the driver -> float16 (PROTOCOL.md allows either; halves the zip)
    for p in sorted(glob.glob(os.path.join(raw, "renders", "*.npy"))):
        np.save(os.path.join(renders_out, os.path.basename(p)), np.load(p).astype(np.float16))
    no_pose = [int(x) for x in open(os.path.join(raw, "renders_without_pose.txt")).read().split()]
    expected = protocol.required_renders(n, keyframes)
    have = {int(os.path.basename(p)[:6]) for p in glob.glob(os.path.join(renders_out, "*.npy"))}
    if sorted(have | set(no_pose)) != expected:
        raise ValueError(f"renders: {len(have)} written + {len(no_pose)} without pose, protocol expects {len(expected)}")
    colors, _ = protocol.dataset_frames(args.dataset)
    runzip.write_figures(renders_out, colors, os.path.join(out, "figures"))

    states = np.genfromtxt(os.path.join(raw, "states.csv"), delimiter=",", skip_header=1)
    counts = {TRACKING_STATES.get(int(s), str(int(s))): int((states[:, 1] == s).sum()) for s in np.unique(states[:, 1])}
    ply = glob.glob(os.path.join(raw, "mapper", "*_shutdown", "ply", "point_cloud", "iteration_*", "point_cloud.ply"))
    gaussians = None
    if ply:
        with open(ply[-1], "rb") as f:
            m = re.search(rb"element vertex (\d+)", f.read(4096))
            gaussians = int(m.group(1)) if m else None
    track = states[:, 2].sum()
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    runzip.write_json(os.path.join(out, "timing.json"), {
        "frames": n, "seconds": dt["loop_s"],
        "gpu": f"{gpu}, 1 GPU (Gaussian mapper; ORB-SLAM3 tracking on CPU, {os.cpu_count()} logical cores)",
        "what": ("photoslam_monogs loop, first to last frame: frame reading (cv::imread, inside the loop as MonoGS "
                 "reads its frames), TrackRGBD, with the Gaussian mapper training in its own thread (no pacing: "
                 "timestamps = frame index, as replica_rgbd). Excludes start-up "
                 f"({dt['init_s']:.1f} s), the mapper's own work after the last frame "
                 f"({dt['mapper_after_loop_s']:.1f} s: final training to {dt['mapper_iterations']} iterations and "
                 "saving), and the renders"),
        "sessions": 1, "init_s": dt["init_s"], "loop_s": dt["loop_s"], "mapper_after_loop_s": dt["mapper_after_loop_s"],
        "track_total_s": float(track), "fps_tracking_only": n / float(track) if track else None})
    runzip.write_json(os.path.join(out, "calibration.json"), cal)
    libtorch = ""
    bv = os.path.join(build, "deps", "libtorch", "build-version")
    if os.path.isfile(bv):
        libtorch = open(bv).read().strip()
    notes = (f"Photo-SLAM RGB-D, replica_rgbd loop, Photo-SLAM's Replica {scene} settings with the MonoGS camera, "
             "no distortion, viewer off, loop closure on (ORB-SLAM3; MonoGS has none). "
             f"Atlas: {dt['maps']} map(s); scored = the largest map, {len(main)} of {n} frames; "
             f"{len(no_pose)} render frames without a pose. Tracking states: {counts}. "
             f"Mapper: {dt['mapper_iterations']} iterations, {gaussians} Gaussians.")
    runzip.write_json(os.path.join(out, "method.json"), {
        "method": METHOD, "scene": scene, "variant": variant, "dataset": os.path.abspath(args.dataset),
        "started_at": started_at, "commit": commit, "libtorch": libtorch,
        "frames": n, "keyframes": len(keyframes), "renders": len(have), "renders_without_pose": no_pose,
        "gaussians": gaussians, "mapper_iterations": dt["mapper_iterations"], "maps": dt["maps"],
        "frames_in_main_map": len(main), "tracking_states": counts, "settings_changes": changes,
        "overrides": runzip.read_json(os.path.join(work, "overrides.json")), "notes": notes})
    with open(os.path.join(out, "notes.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"{len(main)}/{n} frames in the main map ({dt['maps']} maps), {len(keyframes)} keyframes, {len(have)} renders "
          f"({len(no_pose)} without pose), {gaussians} Gaussians, {n / dt['loop_s']:.2f} FPS -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--dataset", required=True, help="MonoGS dataset folder, e.g. .../office0/lit")
    p.add_argument("--work", required=True, help="output folder of this run")
    p.add_argument("--checkpoint_every", type=int, required=True, help="must be 0: Photo-SLAM has no checkpoints")
    p.add_argument("--resume", action="store_true", help="refused: Photo-SLAM has no checkpoints")
    p.add_argument("--no_map", action="store_true", help="leave the final map (point_cloud.ply) out of the zip")
    p.add_argument("--build", default=DEFAULT_BUILD, help="where setup_kaggle.sh built Photo-SLAM")
    args = ap.parse_args(argv)
    cmd_run(args)


if __name__ == "__main__":
    main()
