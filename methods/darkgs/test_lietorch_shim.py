"""Checks lietorch_shim/lietorch.py against scipy.spatial.transform.Rotation:
exp, log (round trip, near 0 and near pi), act with DarkGS's broadcasting
((1, 1) group on (1, N, 3) points), inv, gradient through act, and that
LieGroupParameter stays out of a module's state_dict (as lietorch's).

    python methods/darkgs/test_lietorch_shim.py
"""
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lietorch_shim"))
from lietorch import SO3, LieGroupParameter  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(500, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.concatenate([rng.uniform(0, np.pi - 1e-3, 470), [0.0, 1e-9, 1e-7, 1e-5, 1e-3],
                             np.full(25, np.pi - 1e-6)])
    rv = axes * angles[:, None]
    R_ref = Rotation.from_rotvec(rv).as_matrix()
    R = SO3.exp(torch.tensor(rv, dtype=torch.float64)).matrix.numpy()
    e_exp = np.abs(R - R_ref).max()
    back = SO3.exp(torch.tensor(rv, dtype=torch.float64)).log().numpy()
    e_log = np.abs(Rotation.from_rotvec(back).as_matrix() - R_ref).max()  # compare rotations (pi is double-covered)
    print(f"exp max err {e_exp:.2e}, log round trip max err {e_log:.2e}")
    assert e_exp < 1e-9 and e_log < 1e-6, (e_exp, e_log)

    r = torch.tensor([[[0.01, -0.02, 0.3]]])  # DarkGS's (1, 1, 3) float32
    g = LieGroupParameter(SO3.exp(r))
    pts = torch.tensor(rng.normal(size=(1, 1000, 3)), dtype=torch.float32, requires_grad=True)
    ref = torch.tensor(Rotation.from_rotvec(r[0, 0].numpy().astype(np.float64)).apply(pts[0].detach().numpy()))
    e_act = (g.act(pts)[0].double() - ref).abs().max().item()
    e_inv = (g.inv().act(g.act(pts)) - pts).abs().max().item()
    e_logr = (g.log() - r).abs().max().item()
    print(f"act max err {e_act:.2e}, inv(act) max err {e_inv:.2e}, log max err {e_logr:.2e}")
    assert g.act(pts).shape == (1, 1000, 3) and e_act < 1e-5 and e_inv < 1e-5 and e_logr < 1e-6
    g.act(pts).sum().backward()
    assert pts.grad is not None and torch.isfinite(pts.grad).all()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t = torch.nn.Parameter(torch.zeros(1, 1, 3))
            self.rot = LieGroupParameter(SO3.exp(r))

    assert list(M().state_dict()) == ["t"], list(M().state_dict())
    print("lietorch shim ok")


if __name__ == "__main__":
    main()
