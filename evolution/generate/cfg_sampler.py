"""Per-token dynamic classifier-free guidance for the PI-CFM DiT sampler.

The DiT uses patch_size=4 on a 128x64 grid -> 32x16 = 512 tokens, and each token
carries its own CFG schedule s_token(t):

    v(x,t) = v_null + S(t,x,y) * sum_i m_i (v_i - v_null)

S is a [1,1,128,64] field, nearest-neighbour upsampled 4x from a [1,1,32,16]
per-token field -- one guidance value per DiT token. The region masks m_i come
from the VORONOI condition: `voronoi_regions` decomposes the Voronoi RGB into
per-class blends (topopt / finray / graph / lattice) and their spatial masks.

There is NO CPPN anywhere in this pipeline. The research code called this file
`cppn_cfg_per_token_batch.py` and this function `build_cppn_regions`, but a CPPN
was only ever an earlier, abandoned attempt at injecting the condition -- it was
never adopted, and the misleading names are dropped here. (`cppn_num_hidden` in
the bundled BC npz is inert metadata from how the mechanism DATASET was
generated, not a model input.) The condition is the Voronoi diagram alone: site
positions plus per-site class.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "cfm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
DesignGenerator = importlib.import_module("generator").DesignGenerator

PATCH = 4                 # DiT patch size: 128x64 pixels -> 32x16 = 512 tokens

# condition-channel indices in the encoder's spatial condition tensor
CH_TOPOPT, CH_FINRAY, CH_GRAPH, CH_LATTICE = 6, 7, 8, 9

def rgb_to_compositional_regions(rgb: np.ndarray):
    """R -> topopt, B -> finray, G -> graph, lattice=0."""
    rgb01 = rgb.astype(np.float32) / 255.0
    R, G, B = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    s = R + G + B
    fallback = s < 1e-6
    R_m = np.where(fallback, 1.0, R / (s + 1e-8))
    B_m = np.where(fallback, 0.0, B / (s + 1e-8))
    G_m = np.where(fallback, 0.0, G / (s + 1e-8))
    blends = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    ]
    masks = [R_m, B_m, G_m]
    return blends, masks

def voronoi_regions(gen, rgb):
    H, W = gen.base_cond.shape[-2:]
    blends, masks_np = rgb_to_compositional_regions(rgb)
    conditions, masks = [], []
    for blend, m in zip(blends, masks_np):
        c = gen.base_cond.unsqueeze(0).clone().to(gen.device)
        c[:, CH_TOPOPT, :, :] = blend[0]
        c[:, CH_FINRAY, :, :] = blend[1]
        c[:, CH_GRAPH, :, :] = blend[2]
        c[:, CH_LATTICE, :, :] = blend[3]
        conditions.append(c)
        masks.append(torch.tensor(m, dtype=torch.float32,
                                  device=gen.device).view(1, 1, H, W))
    return conditions, masks, H, W

def vel(gen, t_scalar, x, cond, use_null=False):
    t = torch.full((x.shape[0],), float(t_scalar), device=gen.device)
    return gen.model(t, x, cond_spatial=cond, use_null=use_null)

def cosine_bell(t, t_peak, t_width, s_lo, s_hi):
    """Cosine² bell, value = s_hi at t_peak, = s_lo outside [t_peak±t_width/2]."""
    half = 0.5 * t_width
    if t_width <= 0:
        return float(s_hi if abs(t - t_peak) < 1e-9 else s_lo)
    if abs(t - t_peak) >= half:
        return float(s_lo)
    u = (t - t_peak) / half          # in (-1, 1)
    bell = math.cos(0.5 * math.pi * u) ** 2   # peak 1 at u=0
    return float(s_lo + (s_hi - s_lo) * bell)

def active_window_gate(t, t0, t1):
    """Raised-cosine gate active only on [t0, t1], peaking at the midpoint."""
    if t1 <= t0:
        return float(1.0 if t >= t0 else 0.0)
    t_peak = 0.5 * (t0 + t1)
    t_width = t1 - t0
    return cosine_bell(t, t_peak, t_width, 0.0, 1.0)

def upsample_token_field(token_field_HW, target_H, target_W):
    """token_field_HW: [1,1,gH,gW] → [1,1,H,W] by nearest-neighbour ×PATCH."""
    return F.interpolate(token_field_HW, size=(target_H, target_W), mode="nearest")

def sample_adaptive_v2(gen, conditions, masks, n_steps, seed, H, W,
                       s_lo, s_hi, gamma, ema, active_t0=0.3, active_t1=0.7):
    gH, gW = H // PATCH, W // PATCH
    g = torch.Generator(device=gen.device).manual_seed(seed)
    x = torch.randn((1, 1, H, W), device=gen.device, generator=g)
    dt = 1.0 / n_steps
    s_token = torch.full((1, 1, gH, gW), s_lo, device=gen.device)
    s_mean_log = np.zeros((n_steps,), dtype=np.float32)
    s_std_log = np.zeros((n_steps,), dtype=np.float32)

    for k in range(n_steps):
        t = k * dt

        # Forward passes (same as before)
        v_null = vel(gen, t, x, conditions[0], use_null=True)
        blend = torch.zeros_like(v_null)
        for c_i, m_i in zip(conditions, masks):
            v_i = vel(gen, t, x, c_i)
            blend = blend + m_i * (v_i - v_null)

        # Binary uncertainty per token
        x_clamp = x.clamp(0, 1)
        uncertainty = 1.0 - 2.0 * (x_clamp - 0.5).abs()
        u_tok = F.avg_pool2d(uncertainty, kernel_size=(PATCH, PATCH))

        # Windowed time gate: strongest inside [active_t0, active_t1]
        time_gate = active_window_gate(t, active_t0, active_t1)

        # Per-token CFG scale
        s_new = s_lo + (s_hi - s_lo) * u_tok.pow(gamma) * time_gate
        s_token = ema * s_token + (1.0 - ema) * s_new
        s_field = upsample_token_field(s_token, H, W)

        # Compose velocity
        v = v_null + s_field * blend
        x = x + dt * v
        s_mean_log[k] = float(s_token.mean().item())
        s_std_log[k] = float(s_token.std().item())

    return x.clamp(0, 1), s_mean_log, s_std_log
