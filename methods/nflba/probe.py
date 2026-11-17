"""Setup probe of NFL-BA (run by setup_kaggle.sh after the build): NFL-BA's own
code on a small synthetic RGB-D frame, so a broken build, an ABI mismatch with
the installed torch or an API change shows in setup, not into a run.

1. Gaussians from the frame (GaussianModel.extend_from_pcd_seq: open3d,
   simple_knn), MonoGS's Replica config with a 64 x 48 camera and, for the
   probe only, one Gaussian per pixel as wide as the pixel spacing;
2. render with nfl off and on (diff_gaussian_rasterization w/ pose): the light
   model changes the image by the shading (its ratio follows the distance);
3. tracking: the camera displaced by 2 cm, MonoGS's tracking loop pieces
   (get_loss_tracking with exposure, Adam on the pose deltas, update_pose)
   bring it back at least halfway;
4. mapping: get_loss_mapping backward reaches the Gaussians.

The scene is a textured plane 2 m in front of the camera; the numbers are not
NFL-BA's or the dataset's. Exits non-zero on any failure.
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402

W, H, F = 64, 48, 60.0
CX, CY = (W - 1) / 2, (H - 1) / 2
PLANE_Z = 2.0


def check(cond, what):
    if not cond:
        raise SystemExit(f"probe FAILED: {what}")
    print(f"probe ok: {what}", flush=True)


def main(src):
    sys.path.insert(0, src)
    import torch
    from munch import munchify

    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix2
    from utils.camera_utils import Camera
    from utils.pose_utils import update_pose
    from utils.slam_utils import get_loss_mapping, get_loss_tracking

    torch.manual_seed(0)
    np.random.seed(0)
    config = adapter.load_scene_config("office0")
    calib = config["Dataset"]["Calibration"]
    calib.update(width=W, height=H, fx=F, fy=F, cx=CX, cy=CY)
    config["Training"]["monocular"] = False
    # Probe scene only: every pixel becomes a Gaussian as wide as the pixel spacing, so the untrained model
    # covers the image (a run grows its Gaussians in 1050 initial mapping iterations instead).
    config["Dataset"].update(pcd_downsample_init=1, point_size=1.0, adaptive_pointsize=False)
    ys, xs = np.mgrid[0:H, 0:W]
    color = np.stack([0.5 + 0.4 * np.sin(xs / 3.0), 0.5 + 0.4 * np.cos(ys / 4.0),
                      0.5 + 0.4 * np.sin((xs + ys) / 5.0)]).astype(np.float32)
    depth = np.full((H, W), PLANE_Z, dtype=np.float32)
    proj = getProjectionMatrix2(znear=0.0001, zfar=100.0, fx=F, fy=F, cx=CX, cy=CY, W=W, H=H).transpose(0, 1).cuda()
    cam = Camera(0, torch.from_numpy(color).cuda(), depth, torch.eye(4, device="cuda"), proj, F, F, CX, CY,
                 focal2fov(F, W), focal2fov(F, H), H, W)
    cam.compute_grad_mask(config)

    model = GaussianModel(0, config=config)
    model.init_lr(6.0)
    model.training_setup(munchify(config["opt_params"]))
    model.extend_from_pcd_seq(cam, kf_id=0, init=True, scale=2.0, depthmap=depth)
    n = int(model.get_xyz.shape[0])
    expect = W * H / config["Dataset"]["pcd_downsample_init"]  # MonoGS keeps 1 pixel in pcd_downsample_init
    check(n >= 0.8 * expect and torch.isfinite(model.get_scaling).all(),
          f"simple_knn, {n} Gaussians from the frame (expected about {expect:.0f})")

    bg = torch.zeros(3, device="cuda")
    pipe_off = munchify(dict(config["pipeline_params"], nfl=False))
    pipe_on = munchify(dict(config["pipeline_params"], nfl=True))
    off = render(cam, model, pipe_off, bg)
    on = render(cam, model, pipe_on, bg)
    img_off, img_on = off["render"], on["render"]
    check(img_on.shape == (3, H, W) and on["depth"].shape[-2:] == (H, W) and torch.isfinite(img_on).all(),
          f"render shapes {tuple(img_on.shape)}, depth {tuple(on['depth'].shape)}")
    covered = on["opacity"][0] > 0.5
    check(covered.float().mean() > 0.5, f"plane covers {covered.float().mean().item():.0%} of the image")
    r = img_on[:, covered] / img_off[:, covered].clamp(min=1e-3)
    # nfl shading = (cos(n, l) / d^2)^(1/2.2) <= (1 / 4)^(1/2.2) = 0.53 on a plane 2 m away. New Gaussians are
    # isotropic, so the "shortest axis" normal is an arbitrary axis and cos varies; only the bound is checked.
    check(torch.isfinite(r).all() and r.max().item() <= 0.53 + 0.02 and (img_off - img_on).abs().mean() > 1e-3,
          f"light model applied: nfl-on / nfl-off ratio median {r.median().item():.3f}, max {r.max().item():.3f} "
          "(bound 0.53)")

    # tracking: target = the nfl render at the identity; start 2 cm off
    vp = copy.copy(cam)
    vp.original_image = img_on.detach()
    vp.depth = on["depth"].detach().squeeze(0).cpu().numpy()
    vp.compute_grad_mask(config)
    vp.update_RT(torch.eye(3, device="cuda"), torch.tensor([0.02, 0.0, 0.0], device="cuda"))
    opt = torch.optim.Adam([
        {"params": [vp.cam_rot_delta], "lr": config["Training"]["lr"]["cam_rot_delta"]},
        {"params": [vp.cam_trans_delta], "lr": config["Training"]["lr"]["cam_trans_delta"]},
        {"params": [vp.exposure_a], "lr": 0.01}, {"params": [vp.exposure_b], "lr": 0.01}])
    for _ in range(config["Training"]["tracking_itr_num"]):
        pkg = render(vp, model, pipe_on, bg)
        opt.zero_grad()
        loss = get_loss_tracking(config, pkg["render"], pkg["depth"], pkg["opacity"], vp)
        loss.backward()
        with torch.no_grad():
            opt.step()
            update_pose(vp)
    x = vp.T[0].item()
    check(abs(x) < 0.5 * 0.02, f"tracking moved the camera from x 0.0200 to {x:.4f} (reference 0)")

    # mapping
    pkg = render(cam, model, pipe_on, bg)
    loss = get_loss_mapping(config, pkg["render"], pkg["depth"], cam, pkg["opacity"])
    loss.backward()
    g = model._xyz.grad
    check(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "mapping gradient reaches the Gaussians")
    print("probe: NFL-BA rasterizer and light model ok", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else adapter.DEFAULT_SRC)
