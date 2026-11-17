"""DarkGS's light file (model_parameters.pth: ShadingModel state with a 1D MLP
light, plus "so3") from this project's light CALIBRATION, the same lamp the
main model uses (PROTOCOL.md, Nhóm C; user decision 2026-11-17): not from the
Blender light parameters. Run locally, once per scene, in the MonoGS
environment (it reads the project's light_models and configs); the result is
committed under methods/darkgs/light/.

    python baselines/methods/darkgs/fit_light.py --dataset <.../office0/lit> \
        --ambient_config configs/light/rgbd_lambert_calib_ambgrid.yaml --out baselines/methods/darkgs/light/office0_lit.pth

Source: <dataset>/calib/light_calibrated.json (tools/calibrate/fit_light.py,
D106: 8-bit images and sensor depth of the flat-target clip). The calibrated
lamp is direct = K * Phi(theta) / d^2, d the distance to the lamp at t_CL,
theta the angle to its axis R_CL e_z, Phi a table linear in cos(theta)
(light_models.angular.Tabulated, linear_cos).

DarkGS's 1D MLP light (scene/light.py LightMLP1D) is, at a point x in the
light frame (x = R_l2c^T (p_cam - t_l2c)):
    incident = MLP(features(r)) / (tau + |x|^2)^gamma,  r = sqrt(x^2 + y^2) / z = tan(theta)
so: gamma = 1, tau = TAU_M2 (about 0: 1/d^2), t_l2c = t_CL, so3 = log(R_CL),
and the MLP is fitted to K * Phi(atan r). DarkGS's ambient is (incident +
ambient) * cos(n, l), not this project's c * A without the cosine (a model
difference, README); it starts at the main model's starting guess for c
(read from --ambient_config, Light.ambient.c.value) and DarkGS learns it.
scaling_factor 1 (reference poses are metric). The file keeps the key /
shape / dtype layout of DarkGS's own model_parameters.pth (checked).

Checks written to the .json next to the .pth: the MLP against K * Phi on r,
and DarkGS's ShadingModel itself (scene/light.py through
lietorch_shim) against K * Phi(theta) / d^2 computed in the camera frame
for random points in view (relative error, scaled by K / d^2).
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(BASELINES)
DARKGS = os.path.join(BASELINES, "third_party", "darkgs")
TAU_M2 = 1e-4  # DarkGS needs tau > 0; tau / d^2 <= 0.25% at d >= 0.2 m
MLP_WIDTH, MLP_DEPTH, MLP_IN = 20, 3, 5  # LightMLP1D.create_mlp(width=20, input_dim=5, output_dim=1, depth=3)
FIT_STEPS, FIT_SEED = 12000, 0


def features(r):
    """LightMLP1D.mlp_process: [r, cos r, cos 2r, cos 4r, cos 8r]."""
    return torch.stack([r, torch.cos(r), torch.cos(2 * r), torch.cos(4 * r), torch.cos(8 * r)], dim=-1)


def create_mlp():
    """LightMLPBase.create_mlp, same layers and initialisation."""
    layers = [nn.Linear(MLP_IN, MLP_WIDTH), nn.Softplus()]
    for _ in range(MLP_DEPTH - 1):
        layers += [nn.Linear(MLP_WIDTH, MLP_WIDTH), nn.Softplus()]
    layers += [nn.Linear(MLP_WIDTH, 1), nn.Softplus()]
    mlp = nn.Sequential(*layers)
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    return mlp


def load_calibration(dataset):
    path = os.path.join(dataset, "calib", "light_calibrated.json")
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    L = cal["light"]
    if L["falloff"]["type"] != "inverse_square" or L["angular"]["type"] not in ("spline", "tabulated"):
        raise ValueError(f"{path}: expected an inverse-square lamp with a tabulated / spline profile, got "
                         f"{L['falloff']['type']}, {L['angular']['type']}")
    return path, {"K": float(L["intensity"]), "t_CL": np.array(L["t_CL_m"], dtype=np.float64),
                  "R_CL": np.array(L["R_CL"], dtype=np.float64),
                  "knots_deg": [float(x) for x in L["angular"]["knots_deg"]],
                  "values": [float(x) for x in L["angular"]["knot_values"]]}


def ambient_guess(config_path):
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        return float(cfg["Light"]["ambient"]["c"]["value"])
    except (KeyError, TypeError):
        raise KeyError(f"{config_path}: no Light.ambient.c.value (the main model's starting guess for c)")


def phi_fn(lamp):
    """Phi(cos theta) exactly as the main model evaluates the table."""
    sys.path.insert(0, REPO)
    from light_models.angular import _linear_in_cos

    cos_knots = torch.cos(torch.tensor(lamp["knots_deg"], dtype=torch.float64) * (math.pi / 180.0))
    values = torch.tensor(lamp["values"], dtype=torch.float64)
    return lambda cos_theta: _linear_in_cos(cos_knots, values, cos_theta)


def fit_mlp(K, phi, log=print):
    """MLP(features(r)) ~ K * Phi(cos(atan r)), r = tan(theta). Samples dense
    around the cone edge (where Phi falls), up to r = 1.8 (the image corners
    reach about 1.2)."""
    torch.manual_seed(FIT_SEED)
    rng = np.random.default_rng(FIT_SEED)
    r = np.concatenate([rng.uniform(0.0, 0.4, 4000), rng.uniform(0.4, 0.8, 8000), rng.uniform(0.8, 1.8, 5000), [0.0]])
    rt = torch.tensor(r, dtype=torch.float32)
    y = K * phi(1.0 / torch.sqrt(1.0 + torch.tensor(r, dtype=torch.float64) ** 2)).float()
    x = features(rt)
    mlp = create_mlp()
    with torch.no_grad():  # start near the plateau: the last softplus need not climb to K
        mlp[-2].weight.mul_(0.1)
        mlp[-2].bias.fill_(K)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-3) ** (1.0 / FIT_STEPS))
    for it in range(FIT_STEPS):
        opt.zero_grad()
        loss = ((mlp(x).squeeze(-1) - y) / K).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if it % 3000 == 0 or it == FIT_STEPS - 1:
            log(f"  fit MLP step {it:5d} loss {loss.item():.3e}")
    rg = torch.linspace(0.0, 1.8, 20001)
    with torch.no_grad():
        out = mlp(features(rg)).squeeze(-1).double()
    tgt = K * phi(1.0 / torch.sqrt(1.0 + rg.double() ** 2))
    err = (out - tgt).abs() / K
    return mlp, {"mlp_max_err_over_K": float(err.max()), "mlp_rms_err_over_K": float(err.pow(2).mean().sqrt()),
                 "mlp_at_axis": float(out[0]), "K": K}


def rotvec(R):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def build(lamp, ambient, mlp, template):
    sd = collections.OrderedDict()
    sd["albedo_log"] = template["model_state_dict"]["albedo_log"].clone()  # not used in training
    sd["ambient_light_log"] = torch.tensor(math.log(ambient), dtype=torch.float32)
    sd["scaling_factor"] = torch.tensor(1.0, dtype=torch.float32)
    sd["light._t_vec"] = torch.tensor([[lamp["t_CL"].tolist()]], dtype=torch.float64)
    sd["light.gamma_log"] = torch.tensor(0.0, dtype=torch.float32)
    sd["light.tau_log"] = torch.tensor(math.log(TAU_M2), dtype=torch.float32)
    for k, v in mlp.state_dict().items():
        sd[f"light.mlp.{k}"] = v.detach().clone()
    out = {"model_state_dict": sd, "so3": torch.tensor([[rotvec(lamp["R_CL"]).tolist()]], dtype=torch.float64)}
    tsd = template["model_state_dict"]
    if list(sd) != list(tsd):
        raise ValueError(f"keys differ from DarkGS's model_parameters.pth: {list(sd)} vs {list(tsd)}")
    for k, v in sd.items():
        if tuple(v.shape) != tuple(tsd[k].shape) or v.dtype != tsd[k].dtype:
            raise ValueError(f"{k}: {tuple(v.shape)} {v.dtype}, template {tuple(tsd[k].shape)} {tsd[k].dtype}")
    if out["so3"].shape != template["so3"].shape or out["so3"].dtype != template["so3"].dtype:
        raise ValueError("so3 layout differs from the template")
    return out


def check_with_darkgs(state, lamp, phi, n=50000, seed=1):
    """DarkGS's own ShadingModel light (lietorch_shim) vs K Phi(theta) / d^2 at
    random points in view, in the camera frame; error scaled by K / d^2."""
    if not torch.cuda.is_available():
        return {"darkgs_check": "skipped: no CUDA (DarkGS builds its light on cuda:0)"}
    sys.path[:0] = [os.path.join(HERE, "lietorch_shim"), DARKGS]
    from scene.shading import ShadingModel

    shader = ShadingModel(light="1DMLP").cuda()
    shader.load_state_dict(state["model_state_dict"])
    r = state["so3"].squeeze()
    shader.light.set_r_vec((float(r[0]), float(r[1]), float(r[2])))
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.3, 8.0, n)
    p = np.stack([rng.uniform(-1.0, 1.0, n) * z, rng.uniform(-0.6, 0.6, n) * z, z], -1)  # in view (HFOV 90, VFOV ~61)
    with torch.no_grad():
        out = shader.light(torch.tensor(p, dtype=torch.float32, device="cuda")[None]).squeeze(0).double().cpu().numpy()
    q = p - lamp["t_CL"]
    d2 = (q ** 2).sum(-1)
    cos = (q @ lamp["R_CL"][:, 2]) / np.sqrt(d2)
    tgt = lamp["K"] * phi(torch.tensor(cos)).numpy() / d2
    e = np.abs(out - tgt) * d2 / lamp["K"]
    return {"darkgs_max_err_over_K": float(e.max()), "darkgs_p99_err_over_K": float(np.quantile(e, 0.99)),
            "darkgs_points": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="MonoGS dataset folder with calib/light_calibrated.json")
    ap.add_argument("--ambient_config", required=True, help="main model config: Light.ambient.c.value = starting guess")
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(DARKGS, "model_parameters.pth"))
    a = ap.parse_args()
    src, lamp = load_calibration(a.dataset)
    ambient = ambient_guess(a.ambient_config)
    phi = phi_fn(lamp)
    print(f"lamp from {src}: K {lamp['K']:.4f}, t_CL {lamp['t_CL'].round(5).tolist()} m, "
          f"axis tilt {np.degrees(np.linalg.norm(rotvec(lamp['R_CL']))):.4f} deg; ambient start {ambient}")
    mlp, stats = fit_mlp(lamp["K"], phi)
    template = torch.load(a.template, map_location="cpu", weights_only=False)
    state = build(lamp, ambient, mlp, template)
    stats.update(check_with_darkgs(state, lamp, phi))
    stats.update({"source": os.path.relpath(src, REPO) if src.startswith(REPO) else src, "ambient_start": ambient,
                  "ambient_config": a.ambient_config, "t_l2c": lamp["t_CL"].tolist(),
                  "so3": state["so3"].squeeze().tolist(), "tau_m2": TAU_M2, "gamma": 1.0, "scaling_factor": 1.0,
                  "fit_steps": FIT_STEPS, "fit_seed": FIT_SEED})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(state, a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
