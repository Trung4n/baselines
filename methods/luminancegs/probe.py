"""Setup probe of Luminance-GS (run by setup_kaggle.sh): makes the adapter's
patched copy at its fixed path, compiles the repository's bundled gsplat
backend from it (JIT; the run then reuses this build), checks that the
kernels came from that build and not from an installed gsplat, runs
rasterization_dual (both colour branches, SH degree 3, as the trainer calls
it) forward and backward, and imports simple_trainer_ours with every module
it needs (it parses its command line at import: a dummy one is given).
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

    try:
        import gsplat
        check(False, f"no gsplat package installed (found {gsplat.__file__})")
    except ImportError:
        check(True, "no gsplat package installed (the bundled backend compiles its own kernels)")
    src = adapter.prepare_src()
    sys.path[:0] = [os.path.join(src, "examples"), os.path.join(src, "gsplat")]
    print("probe: compiling the bundled gsplat backend (a few minutes the first time)", flush=True)
    from cuda import _backend
    check(_backend._C is not None and _backend._C.__name__ == "gsplat_cuda",
          f"bundled backend JIT-compiled ({getattr(_backend._C, '__file__', None)})")
    from rendering_double import rasterization_dual

    torch.manual_seed(0)
    n, W, H, sh = 2000, 64, 48, 3
    means = (torch.rand(n, 3, device="cuda") - 0.5) * torch.tensor([3.0, 2.0, 0.5], device="cuda") + torch.tensor([0, 0, 3.0], device="cuda")
    means.requires_grad_(True)
    quats = torch.randn(n, 4, device="cuda")
    scales = torch.full((n, 3), 0.05, device="cuda")
    opac = torch.full((n,), 0.8, device="cuda")
    shc = torch.zeros(n, (sh + 1) ** 2, 3, device="cuda")
    shc[:, 0] = (torch.rand(n, 3, device="cuda") - 0.5) / 0.28209479177387814
    shc.requires_grad_(True)
    k = torch.ones(n, 1, 3, device="cuda", requires_grad=True)
    b = torch.zeros(n, 1, 3, device="cuda", requires_grad=True)
    viewmat = torch.eye(4, device="cuda")[None]
    K = torch.tensor([[[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1]]], device="cuda")
    for mode, ch in (("RGB", 3), ("RGB+ED", 4)):
        enh, low, a_enh, a_low, _ = rasterization_dual(means, quats, scales, opac, shc, shc * k + b, viewmat, K, W, H,
                                                       sh_degree=sh, packed=False, render_mode=mode)
        check(enh.shape == low.shape == (1, H, W, ch) and torch.isfinite(enh).all() and torch.isfinite(low).all()
              and a_enh.mean() > 0.1, f"rasterization_dual {mode}, coverage {a_enh.mean().item():.0%}")
    (enh.mean() + 2 * low.mean()).backward()
    check(torch.isfinite(means.grad).all() and shc.grad.abs().sum() > 0 and k.grad.abs().sum() > 0,
          "rasterization_dual backward (means, SH, per-Gaussian affine)")

    data = tempfile.mkdtemp(prefix="luminancegs_probe_")
    argv = sys.argv
    sys.argv = ["simple_trainer_ours.py"] + adapter.trainer_args(data, os.path.join(data, "out"), None)
    try:
        import datasets
        check(os.path.dirname(datasets.__file__) == os.path.join(src, "examples", "datasets"),
              f"datasets package from {datasets.__file__}")
        import simple_trainer_ours as st
    finally:
        sys.argv = argv
    check(st.cfg.exp_name == adapter.EXP_NAME and st.cfg.max_steps == 10000 and st.cfg.disable_viewer,
          f"simple_trainer_ours and its imports (exp_name {st.cfg.exp_name}, {st.cfg.max_steps} steps)")
    import imageio
    import imageio_ffmpeg  # noqa: F401  (the trainer writes trajectory videos at each evaluation)
    check(True, "imageio ffmpeg backend")
    print("probe: Luminance-GS backend and trainer ok", flush=True)


if __name__ == "__main__":
    main()
