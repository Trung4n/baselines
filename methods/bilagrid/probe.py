"""Setup probe of the bilateral-grid baseline (run by setup_kaggle.sh): imports
gsplat's simple_trainer from a patched copy of third_party/gsplat_examples
(every module it needs, and gsplat's datasets/ ahead of Hugging Face's), then
compiles and runs gsplat's rasterizer (its CUDA code is compiled on first use),
fused_ssim, and lib_bilagrid's slice on the GPU, with a backward pass.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapter  # noqa: E402


def check(cond, what):
    if not cond:
        raise SystemExit(f"probe FAILED: {what}")
    print(f"probe ok: {what}", flush=True)


def main():
    import torch

    import gsplat
    check(gsplat.__version__ == adapter.GSPLAT_VERSION, f"gsplat {gsplat.__version__}")
    src = adapter.patched_copy(os.path.join(tempfile.mkdtemp(prefix="bilagrid_probe_"), "gsplat_src"))
    sys.path.insert(0, src)
    import datasets
    check(os.path.dirname(datasets.__file__) == os.path.join(src, "datasets"), f"datasets package from {datasets.__file__}")
    import lib_bilagrid
    import simple_trainer  # noqa: F401  (all of its imports)
    from fused_ssim import fused_ssim
    from gsplat.rendering import rasterization
    check(True, "simple_trainer and its imports")

    torch.manual_seed(0)
    n, W, H = 2000, 64, 48
    means = (torch.rand(n, 3, device="cuda") - 0.5) * torch.tensor([3.0, 2.0, 0.5], device="cuda") + torch.tensor([0, 0, 3.0], device="cuda")
    means.requires_grad_(True)
    quats = torch.randn(n, 4, device="cuda")
    scales = torch.full((n, 3), 0.05, device="cuda")
    opac = torch.full((n,), 0.8, device="cuda")
    colors = torch.rand(n, 3, device="cuda", requires_grad=True)
    viewmat = torch.eye(4, device="cuda")[None]
    K = torch.tensor([[[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1]]], device="cuda")
    img, alpha, _ = rasterization(means, quats, scales, opac, colors, viewmat, K, W, H)
    check(img.shape == (1, H, W, 3) and torch.isfinite(img).all() and alpha.mean() > 0.1,
          f"gsplat rasterization (CUDA compiled), coverage {alpha.mean().item():.0%}")
    grid = lib_bilagrid.BilateralGrid(1, grid_X=16, grid_Y=16, grid_W=8).cuda()
    gy, gx = torch.meshgrid((torch.arange(H, device="cuda") + 0.5) / H, (torch.arange(W, device="cuda") + 0.5) / W, indexing="ij")
    out = lib_bilagrid.slice(grid, torch.stack([gx, gy], -1)[None], img, torch.zeros(1, 1, dtype=torch.long, device="cuda"))["rgb"]
    loss = 1 - fused_ssim(out.permute(0, 3, 1, 2), torch.rand(1, 3, H, W, device="cuda"), padding="valid")
    loss.backward()
    check(torch.isfinite(means.grad).all() and colors.grad.abs().sum() > 0, "bilateral slice, fused_ssim, backward")
    print("probe: gsplat, bilateral grid and fused_ssim ok", flush=True)


if __name__ == "__main__":
    main()
