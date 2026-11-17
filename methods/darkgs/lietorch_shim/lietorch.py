"""Stand-in for the parts of lietorch (princeton-vl/lietorch) DarkGS uses, so
lietorch's CUDA extension need not be built on Kaggle (methods/darkgs/README.md).
Put this folder first on PYTHONPATH.

DarkGS (scene/light.py, scene/shading.py, train.py) uses only:
- SO3.exp(r) for an axis-angle r of shape (..., 3);
- LieGroupParameter(SO3) as the light-to-camera rotation, with .inv(), .act(p)
  (rotate points, broadcasting the group's batch shape against p's) and
  .log() (back to axis-angle, saved in shader<iter>.pth);
- nothing differentiates the rotation: it is not in ShadingModel's optimizer
  (the r_vec entry is commented out there), and set_r_vec rebuilds it from
  numbers during the warm-up. Gradients do flow through act() to the points.

As in lietorch, LieGroupParameter is not an nn.Parameter: it is not in a
module's state_dict (model_parameters.pth carries the rotation as "so3"
separately). Representation: rotation matrices (float32 like DarkGS's r_vec),
Rodrigues formula with a series near 0; log by the standard inverse with the
pi case handled. Checked against scipy.spatial.transform.Rotation
(methods/darkgs/test_lietorch_shim.py).
"""
import math

import torch

_EPS = 1e-6


def _hat(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _exp(r):
    theta = r.norm(dim=-1, keepdim=True)[..., None]  # (..., 1, 1)
    K = _hat(r)
    small = theta < _EPS
    t = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(t) / t)
    b = torch.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - torch.cos(t)) / t ** 2)
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(K.shape)
    return eye + a * K + b * (K @ K)


def _log(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sin = torch.sin(theta)
    small = theta < _EPS
    scale = torch.where(small, 0.5 + theta ** 2 / 12.0, theta / (2.0 * torch.where(small, torch.ones_like(sin), sin)))
    r = w * scale[..., None]
    near_pi = theta > math.pi - 1e-4
    if near_pi.any():  # sin ~ 0: axis from the symmetric part, R = 2 a a^T - I at theta = pi
        S = (R + R.transpose(-1, -2)) / 2.0 + torch.eye(3, dtype=R.dtype, device=R.device)
        k = torch.argmax(torch.diagonal(S, dim1=-2, dim2=-1), dim=-1)
        col = torch.gather(S, -1, k[..., None, None].expand(*S.shape[:-1], 1)).squeeze(-1)
        axis = col / col.norm(dim=-1, keepdim=True)
        sign = torch.where((axis * w).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        r = torch.where(near_pi[..., None], axis * sign * theta[..., None], r)
    return r


class SO3:
    """A batch of rotations stored as matrices (..., 3, 3)."""

    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def exp(cls, r):
        r = torch.as_tensor(r)
        if r.shape[-1] != 3:
            raise ValueError(f"SO3.exp: axis-angle of shape (..., 3), got {tuple(r.shape)}")
        return cls(_exp(r))

    def log(self):
        return _log(self.matrix)

    def inv(self):
        return SO3(self.matrix.transpose(-1, -2))

    def act(self, pts):
        """Rotate points (..., 3); the group's batch shape broadcasts against theirs."""
        return (self.matrix @ pts.to(self.matrix.dtype)[..., None]).squeeze(-1)

    def matrix4(self):
        M = torch.eye(4, dtype=self.matrix.dtype, device=self.matrix.device).expand(*self.matrix.shape[:-2], 4, 4).clone()
        M[..., :3, :3] = self.matrix
        return M

    @property
    def shape(self):
        return self.matrix.shape[:-2]

    def to(self, *a, **k):
        return SO3(self.matrix.to(*a, **k))

    def cuda(self):
        return SO3(self.matrix.cuda())


class LieGroupParameter(SO3):
    """The group as a module attribute; like lietorch's, not an nn.Parameter
    (so not in state_dict). Holds the group it is built from."""

    def __init__(self, group, requires_grad=True):
        if not isinstance(group, SO3):
            raise TypeError(f"LieGroupParameter: expected an SO3, got {type(group).__name__}")
        super().__init__(group.matrix)
        self.requires_grad = requires_grad
