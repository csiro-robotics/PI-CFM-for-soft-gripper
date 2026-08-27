"""
Euler ODE sampler for conditional flow matching (DiT with CFG).

Includes:
  - euler_sample_dit: standard CFG sampling with a single condition
  - euler_sample_compositional: multi-condition CFG for spatial DS composition
"""

import torch
import torch.nn as nn


@torch.no_grad()
def euler_sample_dit(
    model: nn.Module,
    condition: torch.Tensor,
    shape: tuple,
    n_steps: int = 100,
    device: str = "cuda",
    cfg_scale: float = 1.0,
    disable_ds: bool = False,
) -> torch.Tensor:
    """
    Generate samples via Euler integration of the DiT velocity field.

    x_0 ~ N(0, I)
    x_{k+1} = x_k + (1/n_steps) * v_θ(t_k, x_k, cond)

    When cfg_scale > 1.0, classifier-free guidance is applied:
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

    For CFG unconditional pass:
      use_null=True → condition encoder returns learned null tokens.

    Args:
        model: DiT velocity field v_θ(t, x, cond_spatial)
        condition: (B, 10, H, W) spatial conditioning tensor
        shape: (B, 1, H, W) sample shape
        n_steps: Number of Euler steps
        device: Device
        cfg_scale: Guidance scale (1.0 = no guidance)
        disable_ds: If True, skip DS tokens (Stage 1 mode)

    Returns:
        Generated samples (B, 1, H, W) clamped to [0, 1]
    """
    dt = 1.0 / n_steps
    x = torch.randn(shape, device=device)
    condition = condition.to(device)
    use_cfg = cfg_scale != 1.0

    for k in range(n_steps):
        t = torch.full((shape[0],), k * dt, device=device)
        v_cond = model(t, x, cond_spatial=condition, use_null=False,
                       disable_ds=disable_ds)

        if use_cfg:
            v_uncond = model(t, x, cond_spatial=condition, use_null=True,
                            disable_ds=disable_ds)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        x = x + dt * v

    return x.clamp(0.0, 1.0)


@torch.no_grad()
def euler_sample_compositional(
    model: nn.Module,
    conditions: list,
    spatial_masks: list,
    shape: tuple,
    n_steps: int = 100,
    device: str = "cuda",
    cfg_scale: float = 3.0,
) -> torch.Tensor:
    """
    Multi-condition CFG for spatial design-space composition.

    Composes multiple DS conditions into a single output by running one
    unconditional pass + one conditional pass per region, then blending
    the condition deltas spatially:

        v = v_null + cfg_scale * sum_i[ M_i * (v_i - v_null) ]

    Each condition sees the FULL shared x_t (so self-attention connects
    structures across region boundaries), but only its delta contributes
    within its spatial mask.

    This costs (1 + N_regions) forward passes per ODE step. For N=2 regions
    this is 3 passes — 25% cheaper than naive compositional velocity (4).

    Args:
        model:         DiT model v_θ(t, x, cond_spatial)
        conditions:    List of (B, 10, H, W) condition tensors, one per region.
                       Each should have its own DS channels set (broadcast H×W).
        spatial_masks: List of (1, 1, H, W) or (B, 1, H, W) float masks in [0, 1].
                       Should sum to 1.0 at each pixel (mutually exclusive or soft).
        shape:         (B, 1, H, W) sample shape
        n_steps:       Number of Euler steps
        device:        Device string
        cfg_scale:     Guidance scale (1.0 = no guidance)

    Returns:
        Generated samples (B, 1, H, W) clamped to [0, 1]
    """
    assert len(conditions) == len(spatial_masks), \
        f"Need same number of conditions ({len(conditions)}) and masks ({len(spatial_masks)})"

    dt = 1.0 / n_steps
    x = torch.randn(shape, device=device)

    # Move all conditions and masks to device
    conditions = [c.to(device) for c in conditions]
    spatial_masks = [m.to(device).float() for m in spatial_masks]

    for k in range(n_steps):
        t = torch.full((shape[0],), k * dt, device=device)

        # One unconditional pass (shared)
        v_null = model(t, x, cond_spatial=conditions[0], use_null=True)

        # One conditional pass per region
        v = v_null.clone()
        if cfg_scale != 1.0:
            for cond_i, mask_i in zip(conditions, spatial_masks):
                v_i = model(t, x, cond_spatial=cond_i, use_null=False)
                delta_i = v_i - v_null
                v = v + cfg_scale * mask_i * delta_i
        else:
            for cond_i, mask_i in zip(conditions, spatial_masks):
                v_i = model(t, x, cond_spatial=cond_i, use_null=False)
                v = v + mask_i * (v_i - v_null)

        x = x + dt * v

    return x.clamp(0.0, 1.0)


def dopri5_sample_compositional(
    model: nn.Module,
    conditions: list,
    spatial_masks: list,
    shape: tuple,
    device: str = "cuda",
    cfg_scale: float = 3.0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> torch.Tensor:
    """
    Adaptive Dormand-Prince 5(4) sampler. Same composite velocity as
    `euler_sample_compositional`, integrated with `torchdiffeq.odeint`.
    Prints NFE on completion.
    """
    from torchdiffeq import odeint

    conditions    = [c.to(device) for c in conditions]
    spatial_masks = [m.to(device).float() for m in spatial_masks]

    nfe = [0]

    def f(t_scalar, x):
        nfe[0] += 1
        t = t_scalar.expand(shape[0]).to(x.dtype)
        v_null = model(t, x, cond_spatial=conditions[0], use_null=True)
        v = v_null.clone()
        if cfg_scale != 1.0:
            for cond_i, mask_i in zip(conditions, spatial_masks):
                v_i = model(t, x, cond_spatial=cond_i, use_null=False)
                v = v + cfg_scale * mask_i * (v_i - v_null)
        else:
            for cond_i, mask_i in zip(conditions, spatial_masks):
                v_i = model(t, x, cond_spatial=cond_i, use_null=False)
                v = v + mask_i * (v_i - v_null)
        return v

    x0 = torch.randn(shape, device=device)
    t_grid = torch.tensor([0.0, 1.0], device=device)
    with torch.no_grad():
        traj = odeint(f, x0, t_grid, method="dopri5", rtol=rtol, atol=atol)
    print(f"    [dopri5] NFE = {nfe[0]}")
    return traj[-1].clamp(0.0, 1.0)
