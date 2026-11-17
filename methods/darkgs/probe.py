"""Setup probe of DarkGS (run by setup_kaggle.sh after the build): DarkGS's own
code, through lietorch_shim, on a synthetic plane, with a calibrated lamp file
of methods/darkgs/light/. A broken build, an ABI mismatch with the installed
torch or a shim problem shows in setup, not into a run.

1. GaussianModel.create_from_pcd (simple_knn) from points with normals
   pointing away from the camera (DarkGS's convention);
2. ShadingModel loaded from the lamp file (state dict + so3), render through
   diff_gaussian_rasterization: finite, the cone centre lit, the corners
   (outside the 60-degree cone) far darker;
3. a loss backward reaches the Gaussians' albedo and positions.

Scene: a plane 1.5 m in front of a 64 x 48 camera; not the dataset's numbers.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DARKGS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "third_party", "darkgs")
sys.path[:0] = [os.path.join(HERE, "lietorch_shim"), DARKGS]

W, H, F, Z = 64, 48, 32.0, 1.5


def check(cond, what):
    if not cond:
        raise SystemExit(f"probe FAILED: {what}")
    print(f"probe ok: {what}", flush=True)


def main():
    import torch
    from argparse import ArgumentParser

    from arguments import OptimizationParams, PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from scene.shading import ShadingModel
    from utils.graphics_utils import BasicPointCloud, focal2fov

    lamps = sorted(glob.glob(os.path.join(HERE, "light", "*.pth")))
    check(len(lamps) > 0, f"lamp files: {[os.path.basename(p) for p in lamps]}")
    xs, ys = np.meshgrid(np.linspace(-2.5, 2.5, 160), np.linspace(-1.8, 1.8, 120))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, Z)], 1)
    cols = np.full_like(pts, 0.6)
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))  # away from the camera
    parser = ArgumentParser()
    op, pp = OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    g = GaussianModel(3)
    g.create_from_pcd(BasicPointCloud(points=pts, colors=cols, normals=nrm), 1.0)
    g.training_setup(op.extract(args))
    check(int(g.get_xyz.shape[0]) == len(pts) and torch.isfinite(g.get_scaling).all(), f"{len(pts)} Gaussians (simple_knn)")

    saved = torch.load(lamps[0], weights_only=False)
    shader = ShadingModel(light="1DMLP").cuda()
    shader.load_state_dict(saved["model_state_dict"])
    r = saved["so3"].reshape(-1).tolist()
    shader.light.set_r_vec((r[0], r[1], r[2]))
    shader.set_scaling_factor(1.0)
    cam = Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), FoVx=focal2fov(F, W), FoVy=focal2fov(F, H),
                 image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name="probe", uid=0)
    img = render(cam, g, pp.extract(args), torch.zeros(3, device="cuda"), shader)["render"]
    check(img.shape == (3, H, W) and torch.isfinite(img).all(), f"render {tuple(img.shape)} finite")
    centre = img[:, H // 2 - 3:H // 2 + 3, W // 2 - 3:W // 2 + 3].mean().item()
    corner = img[:, :4, :4].mean().item()
    # corners: 51 degrees off axis at F = 32, outside the 60-degree cone
    check(centre > 0 and corner < 0.2 * centre, f"cone: centre {centre:.4f}, corner {corner:.4f} (linear)")
    img.mean().backward()
    ga, gx = g._albedo.grad, g._xyz.grad
    check(ga is not None and torch.isfinite(ga).all() and ga.abs().sum() > 0 and gx is not None and torch.isfinite(gx).all(),
          "gradient reaches albedo and positions")
    print("probe: DarkGS rasterizer, shader and lietorch shim ok", flush=True)


if __name__ == "__main__":
    main()
