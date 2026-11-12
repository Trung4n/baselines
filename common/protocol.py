"""The evaluation protocol of this repo's MonoGS runs, as functions other SLAM
methods can be scored with (baselines/PROTOCOL.md).

Every number comes from the same code MonoGS uses:
- ATE: utils/eval_utils.evaluate_evo (evo, translation part, RMSE, SE(3)
  alignment for RGB-D, Sim(3) only for monocular);
- PSNR: gaussian_splatting/utils/image_utils.psnr over the pixels where the
  observed image is > 0 (per channel), as utils/eval_utils.eval_rendering;
- SSIM: gaussian_splatting/utils/loss_utils.ssim (11x11 Gaussian window);
- LPIPS: torchmetrics LearnedPerceptualImagePatchSimilarity, AlexNet,
  normalize=True;
- rendering frames: idx = 0, 5, 10, ... < N - 1, keyframes of the method
  excluded (they are training views), as eval_rendering.
"""
import glob
import os
import sys

import numpy as np

COMMON = os.path.dirname(os.path.abspath(__file__))
# The MonoGS repo, when baselines/ sits inside it (local scoring). On Kaggle
# the baselines repo is cloned alone and this folder does not exist.
REPO = os.path.dirname(os.path.dirname(COMMON))

# Camera intrinsics and depth scale of the MonoGS Replica datasets. The
# source is MonoGS's base config; adapters run without the MonoGS repo, so
# they read a verbatim copy of its Dataset.Calibration block, which
# score_zip.py checks against the source before scoring (check_calibration).
CALIB_CONFIG = os.path.join(REPO, "configs", "rgbd", "replica", "base_config.yaml")
CALIB_COPY = os.path.join(COMMON, "calibration.yaml")


def monogs_on_path():
    """Puts the MonoGS repo on sys.path, for the functions that import its
    evaluation code. Not done at import: adapters import this module inside
    another method's environment, whose own top-level packages (SplaTAM's
    `utils`, for one) MonoGS's would shadow."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

RENDER_INTERVAL = 5  # utils/eval_utils.eval_rendering: `interval = 5`

# The frames every method renders for side-by-side figures (PROTOCOL.md):
# spread over the 2000 frames, an early one (young map), and 825, 1125, 1750,
# the frames of the ambient figures of D127. Rendered whether or not they are
# keyframes of the method; figures only, not in the metrics.
FIGURE_FRAMES = (100, 500, 825, 1125, 1750)


# ---------------------------------------------------------------- dataset

def dataset_frames(dataset):
    """Sorted colour and depth paths of a MonoGS Replica-format dataset folder
    (<dataset>/results/frame*.jpg, depth*.png), as utils/dataset.py."""
    colors = sorted(glob.glob(os.path.join(dataset, "results", "frame*.jpg")))
    depths = sorted(glob.glob(os.path.join(dataset, "results", "depth*.png")))
    if not colors:
        raise FileNotFoundError(f"no results/frame*.jpg in {dataset}")
    return colors, depths


def calibration(path=CALIB_COPY):
    """{width, height, fx, fy, cx, cy, depth_scale} from a Dataset.Calibration
    block (metres = PNG value / depth_scale): by default the copy in
    baselines/common, CALIB_CONFIG for MonoGS's own. Undistorted images only:
    the adapters pass no distortion model on."""
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        cal = cfg["Dataset"]["Calibration"]
    except (KeyError, TypeError):
        raise KeyError(f"{path}: no Dataset.Calibration block") from None
    keys = ("width", "height", "fx", "fy", "cx", "cy", "depth_scale", "distorted")
    missing = [k for k in keys if k not in cal]
    if missing:
        raise KeyError(f"{path}: Dataset.Calibration lacks {missing}")
    if cal["distorted"] or any(cal.get(k, 0.0) != 0.0 for k in ("k1", "k2", "p1", "p2", "k3")):
        raise ValueError(f"{path}: distorted calibration; the adapters assume undistorted images")
    return {k: cal[k] for k in keys if k != "distorted"}


def check_calibration(used):
    """Raises unless `used` (the calibration an adapter ran with, as written
    to its calibration.json) equals MonoGS's base config. Needs the MonoGS
    repo, i.e. runs at scoring time."""
    if not os.path.isfile(CALIB_CONFIG):
        raise FileNotFoundError(f"{CALIB_CONFIG}: score inside the MonoGS repo")
    source = calibration(CALIB_CONFIG)
    if {k: float(v) for k, v in used.items()} != {k: float(v) for k, v in source.items()}:
        raise ValueError(f"run calibration {used} differs from {CALIB_CONFIG}: {source}")
    return source


def gt_poses_c2w(dataset):
    """(N, 4, 4) camera-to-world reference poses from <dataset>/traj.txt (one
    row-major 4x4 per line). MonoGS inverts them to world-to-camera for itself
    and back for evo, i.e. evo sees these matrices."""
    rows = np.loadtxt(os.path.join(dataset, "traj.txt"))
    return rows.reshape(-1, 4, 4)


def load_observed(path, device="cuda"):
    """(3, H, W) float32 in [0, 1] exactly as utils/dataset.py loads a frame."""
    import torch
    from PIL import Image

    image = np.array(Image.open(path))
    return torch.from_numpy(image / 255.0).clamp(0.0, 1.0).permute(2, 0, 1).to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------- trajectories

def read_tum(path):
    """{frame index: 4x4 camera-to-world} from a TUM file whose timestamp
    column is the FRAME INDEX (baselines convention, PROTOCOL.md)."""
    from scipy.spatial.transform import Rotation

    poses = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) != 8:
            raise ValueError(f"{path}: TUM lines have 8 numbers, got {len(v)}: {line!r}")
        idx = int(round(v[0]))
        if abs(v[0] - idx) > 1e-6:
            raise ValueError(f"{path}: timestamp {v[0]} is not a frame index (see PROTOCOL.md)")
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(v[4:8]).as_matrix()  # qx qy qz qw
        m[:3, 3] = v[1:4]
        poses[idx] = m
    return poses


def write_tum(path, poses):
    """Inverse of read_tum: {frame index: 4x4 camera-to-world}."""
    from scipy.spatial.transform import Rotation

    with open(path, "w", encoding="utf-8") as f:
        for idx in sorted(poses):
            m = np.asarray(poses[idx])
            q = Rotation.from_matrix(m[:3, :3]).as_quat()
            f.write(f"{idx} {m[0,3]:.9f} {m[1,3]:.9f} {m[2,3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def ate(est, gt, frames, plot_dir, label, monocular=False):
    """RMSE ATE [m] over `frames` with MonoGS's evaluate_evo (writes
    plot/stats_<label>.json and plot/evo_2dplot_<label>.png)."""
    monogs_on_path()
    from utils.eval_utils import evaluate_evo

    frames = [i for i in frames if i in est]
    if len(frames) < 3:
        return float("nan"), 0
    os.makedirs(plot_dir, exist_ok=True)
    value = evaluate_evo(poses_gt=[gt[i] for i in frames], poses_est=[est[i] for i in frames],
                         plot_dir=plot_dir, label=label, monocular=monocular)
    return float(value), len(frames)


# ---------------------------------------------------------------- rendering

def rendering_frames(n_frames, keyframes):
    """Frame indices eval_rendering scores: every 5th, keyframes excluded."""
    kf = set(int(k) for k in keyframes)
    return [i for i in range(0, n_frames - 1, RENDER_INTERVAL) if i not in kf]


def required_renders(n_frames, keyframes):
    """Every frame an adapter must render: the metric frames and the figure
    frames."""
    return sorted(set(rendering_frames(n_frames, keyframes)) | set(i for i in FIGURE_FRAMES if i < n_frames))


def to_png8(rendering):
    """(3,H,W) tensor in [0,1] -> (H,W,3) uint8, as the figures are saved."""
    a = rendering.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (a * 255.0 + 0.5).astype(np.uint8)


class RenderScorer:
    """PSNR, SSIM, LPIPS of one rendered frame against the observed image,
    op for op as utils/eval_utils.eval_rendering."""

    def __init__(self, device="cuda"):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        monogs_on_path()
        self.device = device
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def __call__(self, rendering, observed):
        """rendering, observed: (3, H, W) tensors; rendering is clamped to
        [0, 1] first, as MonoGS does with its render."""
        import torch

        from gaussian_splatting.utils.image_utils import psnr
        from gaussian_splatting.utils.loss_utils import ssim

        with torch.no_grad():
            image = torch.clamp(rendering.to(self.device, torch.float32), 0.0, 1.0)
            gt = observed.to(self.device, torch.float32)
            if image.shape != gt.shape:
                raise ValueError(f"render {tuple(image.shape)} vs observed {tuple(gt.shape)}: "
                                 "render at the dataset resolution")
            mask = gt > 0
            return {
                "psnr": float(psnr(image[mask].unsqueeze(0), gt[mask].unsqueeze(0)).item()),
                "ssim": float(ssim(image.unsqueeze(0), gt.unsqueeze(0)).item()),
                "lpips": float(self.lpips(image.unsqueeze(0), gt.unsqueeze(0)).item()),
            }


def load_render(path):
    """A rendered frame saved by an adapter: .npy float (H,W,3) or (3,H,W) in
    [0, 1] (preferred, no quantization), or an 8-bit .png (quantized: noted
    in metrics.json)."""
    import torch
    from PIL import Image

    if path.endswith(".npy"):
        a = np.load(path).astype(np.float32)
    else:
        a = np.array(Image.open(path)).astype(np.float32) / 255.0
    if a.ndim != 3:
        raise ValueError(f"{path}: expected an RGB image, got shape {a.shape}")
    t = torch.from_numpy(a)
    return t if t.shape[0] == 3 else t.permute(2, 0, 1)
