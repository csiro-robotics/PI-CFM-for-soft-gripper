"""
Training utilities for PICFM-DiT.

Separate from src/training_utils.py to avoid modifying the UNet training pipeline.

Contains:
  - EMA (reused from training_utils)
  - warmup_lr (reused from training_utils)
  - euler_sample_dit: Euler ODE sampling adapted for DiT forward signature
  - visualize_samples_dit: Same visualization but with DiT model calls
  - visualize_label_comparison_dit: Label diagnostic for DiT
"""

import copy
import numpy as np

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse EMA and warmup_lr from the UNet utilities
from src.training_utils import EMA, warmup_lr


# ---------------------------------------------------------------------------
# ODE sampling (Euler) for DiT
# ---------------------------------------------------------------------------

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

    For CFG unconditional pass (v9):
      - use_null=True → ConditionEncoder returns learned null tokens
      - No need to modify condition channels or ds_label

    Args:
        model: DiT velocity field (v9 — ConditionEncoder-based)
        condition: (B, 10, H, W) spatial conditioning tensor
        shape: (B, 1, H, W) sample shape
        n_steps: Number of Euler steps
        device: Device
        cfg_scale: Guidance scale

    Returns:
        Generated samples (B, 1, H, W)
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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _get_ds_label(conditions: torch.Tensor, idx: int) -> str:
    """Extract design-space label string from condition channels 6-9."""
    DS_NAMES = ['topopt', 'finray', 'graph', 'lattice']
    ds_vec = conditions[idx, 6:10, 0, 0].cpu()
    max_idx = ds_vec.argmax().item()
    if ds_vec[max_idx] > 0.5:
        return DS_NAMES[max_idx]
    return "unknown"


def visualize_samples_dit(
    samples: torch.Tensor,
    gt: torch.Tensor,
    conditions: torch.Tensor,
    save_path: str,
    n_show: int = 4,
):
    """Save a comparison grid: condition | prediction | ground truth."""
    n_show = min(n_show, samples.shape[0])
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        ds_label = _get_ds_label(conditions, i)

        cond_vis = conditions[i, 0].cpu().numpy()
        axes[i, 0].imshow(cond_vis, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"BC_x  [{ds_label}]", fontsize=10, fontweight="bold",
                              color="blue" if ds_label == "topopt" else "red")
        axes[i, 0].axis("off")

        force_vis = np.sqrt(
            conditions[i, 2].cpu().numpy() ** 2 + conditions[i, 3].cpu().numpy() ** 2
        )
        axes[i, 1].imshow(force_vis, cmap="hot", origin="lower")
        axes[i, 1].set_title(f"Force")
        axes[i, 1].axis("off")

        pred = samples[i, 0].cpu().numpy()
        pred_01 = np.clip(pred, 0.0, 1.0)
        axes[i, 2].imshow(pred_01, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicted ({ds_label})", fontsize=10, fontweight="bold",
                              color="blue" if ds_label == "topopt" else "red")
        axes[i, 2].axis("off")

        gt_vis = gt[i, 0].cpu().numpy()
        gt_01 = np.clip(gt_vis, 0.0, 1.0)
        axes[i, 3].imshow(gt_01, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        axes[i, 3].set_title(f"GT ({ds_label})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_label_comparison_dit(
    model: nn.Module,
    cond_sample: torch.Tensor,
    save_path: str,
    n_ode_steps: int = 100,
    cfg_scale: float = 1.0,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Diagnostic: same BCs, generate with topopt vs finray label side-by-side.
    """
    DS_NAMES = ['topopt', 'finray', 'graph', 'lattice']
    H, W = cond_sample.shape[1], cond_sample.shape[2]

    conds = []
    for ds_idx, ds_name in enumerate(DS_NAMES[:2]):
        c = cond_sample.clone()
        c[6:10, :, :] = 0.0
        c[6 + ds_idx, :, :] = 1.0
        conds.append(c)

    cond_batch = torch.stack(conds).to(device)

    torch.manual_seed(seed)
    with torch.no_grad():
        samples = euler_sample_dit(
            model, cond_batch, shape=(2, 1, H, W),
            n_steps=n_ode_steps, device=device, cfg_scale=cfg_scale,
        )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(cond_sample[0].cpu().numpy(), cmap="gray", origin="lower")
    axes[0].set_title("BC_x (shared)", fontsize=11)
    axes[0].axis("off")

    force_vis = np.sqrt(cond_sample[2].cpu().numpy()**2 + cond_sample[3].cpu().numpy()**2)
    axes[1].imshow(force_vis, cmap="hot", origin="lower")
    axes[1].set_title("Force (shared)", fontsize=11)
    axes[1].axis("off")

    img0 = samples[0, 0].cpu().numpy()
    img0 = np.clip(img0, 0.0, 1.0)
    axes[2].imshow(img0, cmap="gray_r", origin="lower", vmin=0, vmax=1)
    axes[2].set_title("topopt label", fontsize=12, fontweight="bold", color="blue")
    axes[2].axis("off")

    img1 = samples[1, 0].cpu().numpy()
    img1 = np.clip(img1, 0.0, 1.0)
    axes[3].imshow(img1, cmap="gray_r", origin="lower", vmin=0, vmax=1)
    axes[3].set_title("finray label", fontsize=12, fontweight="bold", color="red")
    axes[3].axis("off")

    corr = torch.corrcoef(torch.stack([
        samples[0].flatten().cpu(), samples[1].flatten().cpu()
    ]))[0, 1].item()
    fig.suptitle(f"Same BCs, Different Labels — pixel corr={corr:.3f}  (cfg={cfg_scale})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
