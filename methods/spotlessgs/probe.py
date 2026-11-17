"""Setup probe of SpotlessGS (run by setup_kaggle.sh after the build):
SpotlessGS's own code, through lietorch_shim (learnable SO3), on a synthetic
plane. A broken build, an ABI mismatch with the installed torch or a shim
problem shows in setup, not into a run.

1. GaussianModel.create_from_pcd (simple_knn) from points with normals;
2. ShadingModel(light="1DMLP") as train.py builds it (default light, MLP
   BRDF, SH ambient), render through diff_gaussian_rasterization: finite;
3. a loss backward reaches the Gaussians and the learned light (position,
   rotation, emission MLP), and one optimizer step of the shader runs.

Scene: a plane 1.5 m in front of a 64 x 48 camera; not the dataset's numbers.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "third_party", "SpotlessGS")
sys.path[:0] = [os.path.join(HERE, "lietorch_shim"), SRC]

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

    xs, ys = np.meshgrid(np.linspace(-2.5, 2.5, 160), np.linspace(-1.8, 1.8, 120))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, Z)], 1)
    parser = ArgumentParser()
    op, pp = OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    g = GaussianModel(3)
    g.create_from_pcd(BasicPointCloud(points=pts, colors=np.full_like(pts, 0.6), normals=np.tile([0.0, 0.0, 1.0], (len(pts), 1))), 1.0)
    g.training_setup(op.extract(args))
    check(int(g.get_xyz.shape[0]) == len(pts) and torch.isfinite(g.get_scaling).all(), f"{len(pts)} Gaussians (simple_knn)")

    shader = ShadingModel(light="1DMLP").cuda()
    shader.set_scaling_factor(1.0)
    cam = Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), FoVx=focal2fov(F, W), FoVy=focal2fov(F, H),
                 image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name="probe", uid=0)
    img = render(cam, g, pp.extract(args), torch.zeros(3, device="cuda"), shader)["render"]
    check(img.shape == (3, H, W) and torch.isfinite(img).all(), f"render {tuple(img.shape)} finite, mean {img.mean().item():.4f}")
    (img - 0.3).abs().mean().backward()
    grads = {"albedo": g._albedo.grad, "xyz": g._xyz.grad, "light t": shader.light._t_vec.grad,
             "light rotation": shader.light._r_l2c_SO3.grad, "light MLP": next(shader.light.mlp.parameters()).grad}
    for k, v in grads.items():
        check(v is not None and torch.isfinite(v).all(), f"gradient reaches {k}")
    before = shader.light._t_vec.detach().clone()
    shader.optimizer.step()
    check(not torch.equal(before, shader.light._t_vec.detach()), "shader optimizer step moves the light position")
    print("probe: SpotlessGS rasterizer, shader and lietorch shim ok", flush=True)


if __name__ == "__main__":
    main()
