"""Stand-in for the parts of lietorch SpotlessGS uses, so lietorch's CUDA
extension need not be built on Kaggle (methods/spotlessgs/README.md). Put this
folder first on PYTHONPATH.

Unlike DarkGS, SpotlessGS puts the light rotation in its optimizer
(ShadingModel.set_optimizer: {'params': [self.light._r_l2c_SO3], ...}), so
LieGroupParameter here is a real nn.Parameter: it holds the axis-angle r
(..., 3) and builds the rotation exp(r) with differentiable torch ops, so a
gradient reaches r and Adam can update it. SO3 and its exp / log / inv / act
are those of methods/darkgs/lietorch_shim (checked against scipy by
methods/darkgs/test_lietorch_shim.py).

Differences from lietorch, recorded in the README:
- lietorch steps the rotation on the group (a retraction in the tangent
  space); here Adam steps the axis-angle. The two agree to first order near
  the identity, where SpotlessGS starts the light (r = 0).
- lietorch's LieGroupParameter is not an nn.Parameter, so it is not in a
  module's state_dict; here it is (key light._r_l2c_SO3). SpotlessGS only
  loads state dicts it saved itself (shader_relit.load_state_dict(shader.state_dict())).
"""
import importlib.util
import os

import torch

_spec = importlib.util.spec_from_file_location(
    "darkgs_lietorch_shim",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "darkgs", "lietorch_shim", "lietorch.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

SO3 = _base.SO3
_exp = _base._exp


class LieGroupParameter(torch.nn.Parameter):
    """Learnable rotation: the parameter IS the axis-angle (..., 3)."""

    def __new__(cls, group, requires_grad=True):
        if not isinstance(group, SO3):
            raise TypeError(f"LieGroupParameter: expected an SO3, got {type(group).__name__}")
        return torch.Tensor._make_subclass(cls, group.log().detach().clone(), requires_grad)

    def _group(self):
        return SO3(_exp(self.as_subclass(torch.Tensor)))

    def inv(self):
        return self._group().inv()

    def act(self, pts):
        return self._group().act(pts)

    def log(self):
        return self.as_subclass(torch.Tensor)
