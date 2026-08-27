"""Genome -> design mask.

    genome = [ theta (5K)  |  z_token (gH*gW) ]        D = 5K + 512 for K=12, 128x64

theta     the CONDITION. 2K Voronoi site positions (sigmoid -> [0,1]^2) followed by
          3K per-site class logits (argmax -> one-hot topopt / graph / finray). This
          becomes a hard, pure-class Voronoi RGB image, which the PI-CFM sampler
          decomposes into per-class region masks and guides on.

z_token   the flow's STARTING NOISE, not a condition. One value per DiT token
          (gH = H/PATCH, gW = W/PATCH -> 32 x 16 = 512), upsampled to the pixel grid,
          added to a frozen base white-noise field and re-standardised to ~N(0,1).
          z_token = 0 leaves the frozen base noise, standardised -- the same field
          rescaled by a fraction of a percent, NOT a bit-identical x0, so it gives a
          very similar but not identical design to the frozen-x0 path
          (thetas_to_masks): measured mask IoU 0.88 at seed 12345.

Decode: theta -> Voronoi RGB -> per-class conditions -> PI-CFM flow ODE with
per-token adaptive CFG -> threshold at 0.5 -> binary mask.

NO REPAIR STEP. The research pipeline post-processed every design with an NV-loop
rib repair that added material to close broken struts; this release omits it, so a
mask is exactly what the model produced. Designs whose ribs do not close stay open
and are scored as they are. The threshold itself costs nothing: the trained model's
output is fully saturated (measured over 4 random genomes, 0.00% of pixels land in
(0.1, 0.9)), so 0.5 is an arbitrary cut through empty space. What is gone is the rib
stitching -- random genomes decode to 2-7 disconnected pieces rather than 1.

Every function here is deterministic: the same (genome, x0_seed) gives the same
mask on any GPU, which is what keeps the evolutionary objective noise-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cfg_sampler import (PATCH, DesignGenerator, active_window_gate,
                         sample_adaptive_v2, voronoi_regions)
from voronoi import HardVoronoi

ROOT = Path(__file__).resolve().parents[2]

# Checkpoint + boundary-condition source. The BC npz pins the design space
# (128 x 64) and the fixed/loaded boundaries every design is generated against;
# it is bundled. The checkpoint is NOT -- see checkpoints/README.md.
DEFAULT_CKPT = "checkpoints/picfm_dit.pt"
DEFAULT_BC = str(Path(__file__).resolve().parent / "assets" / "bc_data_00000.npz")


def _resolve_ckpt(path):
    """Resolve a checkpoint path against the repo root and fail loudly if absent.

    The trained PI-CFM weights are not distributed with the source; see
    checkpoints/README.md for how to obtain or retrain them (training/)."""
    p = Path(path)
    if not p.is_absolute():
        # try it as given (relative to the CWD, which is what a --checkpoint flag
        # means to a user) before falling back to repo-root-relative
        p = p if p.exists() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"PI-CFM checkpoint not found: {p}\n"
            f"  Put the trained DiT weights there, or pass --checkpoint /path/to/weights.pt\n"
            f"  See checkpoints/README.md; training/run_training.sh reproduces them.")
    return str(p)


def make_generator(checkpoint=DEFAULT_CKPT, bc_npz=DEFAULT_BC, device="cuda"):
    ckpt = _resolve_ckpt(checkpoint)
    bc = Path(bc_npz)
    if not bc.is_absolute() and not bc.exists():
        bc = ROOT / bc
    bc = str(bc)
    gen = DesignGenerator(checkpoint_path=ckpt, bc_npz_path=bc, device=device)
    H, W = gen.base_cond.shape[-2:]
    return gen, H, W


def theta_to_rgb(theta: np.ndarray, K: int, H: int, W: int) -> np.ndarray:
    """theta in R^{5K} -> (H,W,3) uint8 HARD pure-class Voronoi RGB.
    First 2K = site positions (sigmoid->[0,1]); next 3K = class logits
    (argmax -> one-hot R/G/B = topopt/graph/finray)."""
    t = np.asarray(theta, dtype=np.float32)
    sites = 1.0 / (1.0 + np.exp(-t[:2 * K])).reshape(K, 2)
    cls = t[2 * K:].reshape(K, 3).argmax(axis=1)
    colors = np.zeros((K, 3), dtype=np.float32)
    colors[np.arange(K), cls] = 1.0
    return HardVoronoi(sites=sites, colors=colors).sample_rgb_uint8(H, W)


@torch.no_grad()
def generate_mask(gen, rgb_u8, x0_seed, H, W) -> np.ndarray:
    """RGB -> PI-CFM (frozen x0) -> threshold -> (H,W) binary mask. No repair."""
    conds, masks, _, _ = voronoi_regions(gen, rgb_u8)
    x, _, _ = sample_adaptive_v2(gen, conds, masks, 100, x0_seed, H, W,
                                 1.0, 4.0, 2.0, 0.8, 0.3, 0.7)
    design = x.squeeze().cpu().numpy()
    return (design >= 0.5).astype(np.uint8)                  # NO repair: threshold


@torch.no_grad()
def generate_masks_batched(gen, rgbs, x0_seed, H, W, ode_steps=100,
                           s_lo=1.0, s_hi=4.0, gamma=2.0, ema=0.8,
                           active_t0=0.3, active_t1=0.7, fwd_chunk=256, x0=None,
                           soft=False):
    """POPULATION-batched generation: run the WHOLE list of designs through ONE
    flow-matching ODE instead of one-at-a-time.

    Mathematically identical to running sample_adaptive_v2 per design with the
    *frozen* x0 (same seed for all -> common random numbers), but each ODE step
    fires a single batched forward over every design's [null + per-region]
    passes.  That collapses ~B*ode_steps*(1+regions) tiny batch-1 launches into
    ode_steps batched ones, which is what makes population-sized batches practical.

    Variable region count per design is handled by flattening all passes into
    one row-batch (one null row + K_b conditional rows per design) and scattering
    the composed velocity back per design.  Returns a list of (H,W) uint8 masks.

    soft=True returns the raw clamped DENSITY field as (H,W) float32 instead of the
    thresholded mask -- the same ODE output, left continuous.
    """
    device = gen.device
    B = len(rgbs)
    gH, gW = H // PATCH, W // PATCH

    # --- per-design Voronoi -> compositional regions; flatten to a row-batch ---
    row_cond, row_design, row_is_null, row_regmask = [], [], [], []
    for b, rgb in enumerate(rgbs):
        conds, rmasks, _, _ = voronoi_regions(gen, rgb)
        row_cond.append(conds[0]); row_design.append(b)          # null row (filler cond)
        row_is_null.append(True);  row_regmask.append(None)
        for c_i, m_i in zip(conds, rmasks):                       # one conditional row / region
            row_cond.append(c_i); row_design.append(b)
            row_is_null.append(False); row_regmask.append(m_i)

    R = len(row_cond)
    COND = torch.cat(row_cond, dim=0)                                          # (R,C,H,W)
    NULL = torch.tensor(row_is_null, device=device)                           # (R,) bool
    design_idx = torch.tensor(row_design, device=device, dtype=torch.long)    # row -> design
    null_pos = torch.tensor([r for r in range(R) if row_is_null[r]],          # design-ordered
                            device=device, dtype=torch.long)                  # (B,)
    REGM = torch.zeros((R, 1, H, W), device=device)                           # region mask / row
    for r, m in enumerate(row_regmask):
        if m is not None:
            REGM[r] = m[0]

    # --- x0: frozen shared noise (CRN), unless a per-design x0 is supplied ---
    if x0 is None:
        g = torch.Generator(device=device).manual_seed(int(x0_seed))
        x = torch.randn((1, 1, H, W), device=device, generator=g).expand(B, 1, H, W).clone()
    else:
        x = x0.to(device=device, dtype=torch.float32).reshape(B, 1, H, W).clone()
    s_token = torch.full((B, 1, gH, gW), s_lo, device=device)
    dt = 1.0 / ode_steps

    def forward_rows(t_val, X):
        """One batched DiT forward over all rows (chunked to bound memory)."""
        if X.shape[0] <= fwd_chunk:
            tv = torch.full((X.shape[0],), float(t_val), device=device)
            return gen.model(tv, X, cond_spatial=COND, null_mask=NULL)
        outs = []
        for s in range(0, X.shape[0], fwd_chunk):
            xb = X[s:s + fwd_chunk]
            tv = torch.full((xb.shape[0],), float(t_val), device=device)
            outs.append(gen.model(tv, xb, cond_spatial=COND[s:s + fwd_chunk],
                                  null_mask=NULL[s:s + fwd_chunk]))
        return torch.cat(outs, dim=0)

    for k in range(ode_steps):
        t = k * dt
        X = x.index_select(0, design_idx)                         # (R,1,H,W) each row's design x
        V = forward_rows(t, X)                                     # (R,1,H,W)
        v_null = V.index_select(0, null_pos)                      # (B,1,H,W) design-ordered
        vnull_row = v_null.index_select(0, design_idx)           # (R,1,H,W)
        contrib = REGM * (V - vnull_row)                         # null rows -> 0 (REGM=0)
        blend = torch.zeros((B, 1, H, W), device=device)
        blend.index_add_(0, design_idx, contrib)                 # sum_i m_i (v_i - v_null) / design
        # per-design adaptive CFG scale field (identical formula to sample_adaptive_v2)
        uncertainty = 1.0 - 2.0 * (x.clamp(0, 1) - 0.5).abs()
        u_tok = F.avg_pool2d(uncertainty, kernel_size=(PATCH, PATCH))
        s_new = s_lo + (s_hi - s_lo) * u_tok.pow(gamma) * active_window_gate(t, active_t0, active_t1)
        s_token = ema * s_token + (1.0 - ema) * s_new
        s_field = F.interpolate(s_token, size=(H, W), mode="nearest")
        v = v_null + s_field * blend
        x = x + dt * v

    designs = x.clamp(0, 1).squeeze(1).cpu().numpy()              # (B,H,W) float
    if soft:                                        # continuous density, not thresholded
        return [d.astype(np.float32) for d in designs]
    return [(d >= 0.5).astype(np.uint8) for d in designs]     # NO repair: threshold


def thetas_to_masks(gen, thetas, K, H, W, x0_seed):
    """Batch of theta vectors -> list of finger masks, generated in ONE batched
    ODE (see generate_masks_batched). Replaces the old per-design serial loop."""
    rgbs = [theta_to_rgb(t, K, H, W) for t in thetas]
    return generate_masks_batched(gen, rgbs, x0_seed, H, W)


def _token_noise_x0(z, x0_seed, H, W, device, noise_scale=1.0):
    """Map a per-design TOKEN noise latent z (B, gH*gW) -> per-design x0 (B,1,H,W).

    z carries ONE value per DiT token (gH=H/PATCH, gW=W/PATCH -> 32x16 = 512 for
    this model) — the resolution the model actually sees. It is upsampled to the
    pixel grid (each token fills its PATCH x PATCH block), added to the frozen
    base white noise (which keeps high-freq content so x0 stays ~in-distribution),
    then standardised to ~N(0,1).  z = 0 -> the frozen base noise, standardised
    (the raw base has mean ~-0.01, std ~0.99, so this is a sub-1% rescale, not a no-op).
    Token-upsampling is naturally ~unit-scale, so noise_scale ~ 1 already varies
    the design meaningfully (unlike the 128-d linear latent which needed ~10)."""
    gH, gW = H // PATCH, W // PATCH
    B = z.shape[0]
    gb = torch.Generator(device=device).manual_seed(int(x0_seed))
    base = torch.randn((1, 1, H, W), device=device, generator=gb)
    pert = F.interpolate(z.reshape(B, 1, gH, gW), size=(H, W), mode="nearest")
    x0 = base + noise_scale * pert
    m = x0.mean(dim=(1, 2, 3), keepdim=True)
    s = x0.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x0 - m) / s

@torch.no_grad()
def genomes_to_masks(gen, genomes, K, H, W, x0_seed, noise_dim=0, noise_scale=1.0,
                     ode_steps=100, soft=False):
    """Genome -> list of (H,W) uint8 masks, searching CONDITION and NOISE together.

    genome = [theta(5K)]                   if noise_dim == 0  (frozen x0)
           = [theta(5K), z_token(noise_dim)] if noise_dim > 0

    With noise_dim > 0 each design gets its own x0 from its own z_token, so the
    optimiser explores the condition and the flow's starting noise jointly while the
    objective stays deterministic. noise_dim == 0 reproduces thetas_to_masks.

    ode_steps is explicit rather than defaulted, because adaptive CFG carries a
    per-STEP EMA over a gate defined on continuous t: a different step count is a
    different generative process, not a finer integration of the same one. Every
    caller in a single run must use the same value.
    """
    g = np.asarray(genomes, dtype=np.float32)
    rgbs = [theta_to_rgb(g[i, :5 * K], K, H, W) for i in range(g.shape[0])]
    x0 = None
    if noise_dim > 0:
        z = torch.tensor(g[:, 5 * K:5 * K + noise_dim], dtype=torch.float32,
                         device=gen.device)
        x0 = _token_noise_x0(z, x0_seed, H, W, gen.device, noise_scale)
    return generate_masks_batched(gen, rgbs, x0_seed, H, W,
                                  ode_steps=ode_steps, x0=x0, soft=soft)


def genome_dim(K, H, W):
    """D = 5K (Voronoi condition) + one noise value per DiT token."""
    return 5 * K + (H // PATCH) * (W // PATCH)
