"""Checks methods/spotlessgs/lietorch_shim: the learnable LieGroupParameter
acts like the frozen DarkGS shim (same rotation), is an nn.Parameter in a
module's state_dict and in an optimizer, gets a gradient through act(), and an
Adam step moves it; a module rebuilt from that state_dict acts the same.

    python methods/spotlessgs/test_lietorch_shim.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lietorch_shim"))
from lietorch import SO3, LieGroupParameter  # noqa: E402


class Light(torch.nn.Module):
    def __init__(self, r):
        super().__init__()
        self.rot = LieGroupParameter(SO3.exp(torch.tensor([[r]], dtype=torch.float32)))


def main():
    r = [0.02, -0.01, 0.3]
    m = Light(r)
    pts = torch.randn(1, 500, 3)
    ref = SO3.exp(torch.tensor([[r]])).act(pts)
    assert (m.rot.act(pts) - ref).abs().max() < 1e-6
    assert (m.rot.inv().act(m.rot.act(pts)) - pts).abs().max() < 1e-5
    assert list(m.state_dict()) == ["rot"] and isinstance(m.rot, torch.nn.Parameter)
    opt = torch.optim.Adam([m.rot], lr=1e-2)
    target = SO3.exp(torch.tensor([[[0.05, 0.0, 0.25]]])).act(pts)
    before = (m.rot.act(pts) - target).pow(2).mean().item()
    for _ in range(50):
        opt.zero_grad()
        loss = (m.rot.act(pts) - target).pow(2).mean()
        loss.backward()
        assert m.rot.grad is not None and torch.isfinite(m.rot.grad).all()
        opt.step()
    after = (m.rot.act(pts) - target).pow(2).mean().item()
    print(f"rotation fit loss {before:.2e} -> {after:.2e}, r = {m.rot.log().detach().reshape(-1).tolist()}")
    assert after < 0.1 * before
    m2 = Light([0.0, 0.0, 0.0])
    m2.load_state_dict(m.state_dict())
    assert (m2.rot.act(pts) - m.rot.act(pts)).abs().max() < 1e-6
    print("spotlessgs lietorch shim ok")


if __name__ == "__main__":
    main()
