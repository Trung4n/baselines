"""Setup probe of VarSplat's CUDA extensions (run by setup_kaggle.sh after the
build). Renders a small synthetic Gaussian model with VarSplat's own code and
backpropagates the way its three users of the rasterizers do, so a broken
build, an ABI mismatch with the installed torch, or an API change shows in
setup, not an hour into a run:

1. mapping (src.utils.utils.render_gaussian_model, gaussian_rasterizer):
   colour, depth and the colour-variance map; L1 plus the NLL term of
   Mapper.optimize_submap; gradients reach the colour variance; Adam step;
2. tracking: means overridden by a translated copy (as Tracker.compute_losses
   overrides them), variance-weighted colour loss; gradient reaches the
   translation;
3. registration (src.gsr.solver.viewpoint_localizer with
   diff_gaussian_rasterization, the pose-gradient rasterizer): a camera
   displaced by 2 cm is localized against the model, called with the same
   positional arguments as gaussian_registration.

The scene is a textured plane 2 m in front of a 64 x 48 camera; none of these
numbers are VarSplat's or the dataset's. Exits non-zero on any failure.
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "third_party", "VarSplat")
sys.path.insert(0, SRC)

W, H, F = 64, 48, 60.0
CX, CY = (W - 1) / 2, (H - 1) / 2
PLANE_Z = 2.0


def check(cond, what):
    if not cond:
        raise SystemExit(f"probe FAILED: {what}")
    print(f"probe ok: {what}", flush=True)


def main():
    import open3d as o3d
    import torch

    from src.entities.arguments import OptimizationParams
    from src.entities.gaussian_model import GaussianModel
    from src.gsr.camera import Camera
    from src.gsr.solver import viewpoint_localizer
    from src.utils.graphics_utils import focal2fov, getProjectionMatrix2
    from src.utils.utils import get_render_settings, render_gaussian_model

    torch.manual_seed(0)
    intr = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]])
    xs, ys = np.meshgrid(np.linspace(-1.2, 1.2, 48), np.linspace(-0.9, 0.9, 36))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, PLANE_Z)], 1)
    cols = np.stack([0.5 + 0.4 * np.sin(6 * xs.ravel()), 0.5 + 0.4 * np.cos(5 * ys.ravel()),
                     0.5 + 0.4 * np.sin(4 * (xs + ys).ravel())], 1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    opt = OptimizationParams(ArgumentParser(description="probe"))
    model = GaussianModel(0)
    model.training_setup(opt)
    model.add_points(pcd)  # simple_knn: initial scales
    n = model.get_size() if hasattr(model, "get_size") else model.get_xyz().shape[0]
    check(n == pts.shape[0] and torch.isfinite(model.get_scaling()).all(), f"simple_knn, {n} Gaussians added")

    # 1. mapping
    eye = np.eye(4)
    settings = get_render_settings(W, H, intr, eye)
    out = render_gaussian_model(model, settings)
    color, depth, var = out["color"], out["depth"], out["variance"]
    check(color.shape == (3, H, W) and var.shape == (3, H, W) and depth.shape[-2:] == (H, W),
          f"render shapes colour {tuple(color.shape)}, depth {tuple(depth.shape)}, variance {tuple(var.shape)}")
    covered = out["alpha"] > 0.5
    check(covered.float().mean() > 0.5, f"plane covers {covered.float().mean().item():.0%} of the image")
    check(torch.isfinite(var).all() and (var[:, covered[0]] > 0).all(), "variance map finite and positive where covered")
    target = (color.detach() * 0.8 + 0.1).clamp(0, 1)
    mask = covered[0]
    resid2 = (color[:, mask].detach() - target[:, mask]) ** 2
    nll = (0.5 * resid2 / (var[:, mask] + 1e-6) + 0.5 * torch.log(var[:, mask] + 1e-6)).mean()
    loss = torch.abs(color[:, mask] - target[:, mask]).mean() + 1e-4 * nll
    loss.backward()
    g = model._color_variance.grad
    check(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "gradient reaches the colour variance")
    check(model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all(), "gradient reaches the means")
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)

    # 2. tracking: translation of the means, variance-weighted loss
    with torch.no_grad():
        ref = render_gaussian_model(model, settings)
    t = torch.zeros(3, device="cuda", requires_grad=True)
    shifted = render_gaussian_model(model, settings, override_means_3d=model.get_xyz() + t
                                    + torch.tensor([0.02, 0.0, 0.0], device="cuda"))
    w = torch.exp(-(torch.log(shifted["variance"].detach() + 1e-6)
                    - torch.log(shifted["variance"].detach() + 1e-6).median()) / 100.0).clamp(1 / 3, 3)
    loss = (w * torch.abs(shifted["color"] - ref["color"])).sum() + torch.abs(shifted["depth"] - ref["depth"]).sum()
    loss.backward()
    check(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0,
          f"tracking gradient reaches the pose, {t.grad.tolist()}")
    model.optimizer.zero_grad(set_to_none=True)

    # 3. registration with the pose-gradient rasterizer
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=F, fy=F, cx=CX, cy=CY, W=W, H=H).transpose(0, 1)
    with torch.no_grad():
        ref = render_gaussian_model(model, settings)
    vp = Camera(0, None, None, torch.eye(4, device="cuda"), proj, F, F, CX, CY,
                focal2fov(F, W), focal2fov(F, H), H, W)
    vp.config = {"Training": {"edge_threshold": 4.0}, "Dataset": {"type": "replica"}}
    vp.load_rgb(ref["color"].detach())
    vp.depth = ref["depth"].detach().squeeze().cpu().numpy()
    vp.T = torch.tensor([0.02, 0.0, 0.0], device="cuda")
    converged, rel, residual, log = viewpoint_localizer(vp, model, 1e-3, 100.0, 3.0)
    moved = vp.T.detach().cpu().numpy()
    check(np.isfinite(log).all() and len(log) > 0 and np.isfinite(rel.detach().cpu().numpy()).all(),
          f"registration ran {len(log)} iterations, loss {log[0]:.4g} -> {log[-1]:.4g}, "
          f"camera x 0.0200 -> {moved[0]:.4f}")
    # The pose, not the last loss value, is the check: the loss also refits the
    # exposure and its masks change with the pose.
    check(abs(moved[0]) < 0.5 * 0.02, "registration moved the camera at least halfway back to the reference")
    print("probe: VarSplat rasterizers ok", flush=True)


if __name__ == "__main__":
    main()
