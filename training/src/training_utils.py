"""
Training utilities for PICFM.

EMA, learning-rate schedules, Euler ODE sampling, and visualisation helpers.
"""

import copy
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average for model parameters.
    
    Args:
        decay: EMA decay rate (default: 0.9999)
        exclude_prefix: if set, state_dict keys starting with this prefix
                        are copied verbatim (no averaging). Use to prevent
                        EMA drift on frozen sub-modules like a pretrained encoder.
    """

    def __init__(self, decay: float = 0.9999, exclude_prefix: str = None):
        self.decay = decay
        self.step = 0
        self.shadow = None
        self.backup = None
        self.exclude_prefix = exclude_prefix

    def register(self, model: nn.Module):
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module):
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for name, param in model.state_dict().items():
            if param.dtype in (torch.float32, torch.float16):
                # Skip EMA averaging for frozen/excluded params — copy directly
                if self.exclude_prefix and name.startswith(self.exclude_prefix):
                    self.shadow[name] = param.clone()
                else:
                    self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param

    def apply_shadow(self, model: nn.Module):
        """Switch model to EMA weights (backup current weights)."""
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """Restore original weights after EMA evaluation."""
        model.load_state_dict(self.backup)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def warmup_lr(step: int, warmup_steps: int) -> float:
    """Linear warmup schedule."""
    return min(step, warmup_steps) / max(warmup_steps, 1)


# ---------------------------------------------------------------------------
# ODE sampling (Euler)
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model: nn.Module,
    condition: torch.Tensor,
    shape: tuple,
    n_steps: int = 100,
    device: str = "cuda",
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate samples via Euler integration of the learned velocity field.

    x_0 ~ N(0, I)
    x_{k+1} = x_k + (1/n_steps) * v_θ(t_k, x_k, cond)

    When cfg_scale > 1.0, classifier-free guidance is applied:
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
    The unconditional pass uses ds_label=4 (null class) and zeroed ch 6-10.

    Args:
        model: UNet velocity field v_θ(t, x, cond_spatial, ds_label)
        condition: Spatial conditioning tensor (batch, C, H, W)
        shape: Sample shape (batch, 1, H, W)
        n_steps: Number of Euler steps
        device: Device
        cfg_scale: Guidance scale. 1.0 = no guidance, >1.0 = stronger conditioning.

    Returns:
        Generated samples (batch, 1, H, W)
    """
    dt = 1.0 / n_steps
    x = torch.randn(shape, device=device)
    use_cfg = cfg_scale != 1.0

    # Extract ds_label from condition channels 7-10
    ds_label = condition[:, 6:10, 0, 0].argmax(dim=1).to(device)  # (B,)

    if use_cfg:
        null_cond = condition.clone()
        null_cond[:, 6:, :, :] = 0.0
        null_ds = torch.full_like(ds_label, 4)  # null class index

    for k in range(n_steps):
        t = torch.full((shape[0],), k * dt, device=device)
        v_cond = model(t, x, cond_spatial=condition, ds_label=ds_label)

        if use_cfg:
            v_uncond = model(t, x, cond_spatial=null_cond, ds_label=null_ds)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        x = x + dt * v

    return x


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _get_ds_label(conditions: torch.Tensor, idx: int) -> str:
    """Extract design-space label string from condition channels 7-10."""
    DS_NAMES = ['topopt', 'finray', 'graph', 'lattice']
    ds_vec = conditions[idx, 6:10, 0, 0].cpu()
    max_idx = ds_vec.argmax().item()
    if ds_vec[max_idx] > 0.5:
        return DS_NAMES[max_idx]
    return "unknown"


def visualize_samples(
    samples: torch.Tensor,
    gt: torch.Tensor,
    conditions: torch.Tensor,
    save_path: str,
    n_show: int = 4,
):
    """Save a comparison grid: condition | prediction | ground truth.

    Each row now shows the design-space label (topopt/finray/...) so you
    can verify the model generates the correct type.
    """
    n_show = min(n_show, samples.shape[0])
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        ds_label = _get_ds_label(conditions, i)

        # Condition: show BCx overlay
        cond_vis = conditions[i, 0].cpu().numpy()  # BCx channel
        axes[i, 0].imshow(cond_vis, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"BC_x  [{ds_label}]", fontsize=10, fontweight="bold",
                              color="blue" if ds_label == "topopt" else "red")
        axes[i, 0].axis("off")

        # Input force
        force_vis = np.sqrt(
            conditions[i, 2].cpu().numpy() ** 2 + conditions[i, 3].cpu().numpy() ** 2
        )
        axes[i, 1].imshow(force_vis, cmap="hot", origin="lower")
        axes[i, 1].set_title(f"Force")
        axes[i, 1].axis("off")

        # Predicted topology
        pred = samples[i, 0].cpu().numpy()
        pred_01 = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        axes[i, 2].imshow(pred_01, cmap="gray_r", origin="lower")
        axes[i, 2].set_title(f"Predicted ({ds_label})", fontsize=10, fontweight="bold",
                              color="blue" if ds_label == "topopt" else "red")
        axes[i, 2].axis("off")

        # Ground truth topology
        gt_vis = gt[i, 0].cpu().numpy()
        gt_01 = (gt_vis - gt_vis.min()) / (gt_vis.max() - gt_vis.min() + 1e-8)
        axes[i, 3].imshow(gt_01, cmap="gray_r", origin="lower")
        axes[i, 3].set_title(f"GT ({ds_label})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_label_comparison(
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

    This directly shows whether the model can distinguish design spaces.
    Columns: BC_x | Force | topopt output | finray output
    """
    DS_NAMES = ['topopt', 'finray', 'graph', 'lattice']
    H, W = cond_sample.shape[1], cond_sample.shape[2]

    # Build conditions with same BCs but different labels
    conds = []
    for ds_idx, ds_name in enumerate(DS_NAMES[:2]):  # topopt and finray
        c = cond_sample.clone()
        c[6:10, :, :] = 0.0
        c[6 + ds_idx, :, :] = 1.0
        conds.append(c)

    cond_batch = torch.stack(conds).to(device)

    # Generate with same noise
    torch.manual_seed(seed)
    with torch.no_grad():
        samples = euler_sample(
            model, cond_batch, shape=(2, 1, H, W),
            n_steps=n_ode_steps, device=device, cfg_scale=cfg_scale,
        )

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # BC_x
    axes[0].imshow(cond_sample[0].cpu().numpy(), cmap="gray", origin="lower")
    axes[0].set_title("BC_x (shared)", fontsize=11)
    axes[0].axis("off")

    # Force
    force_vis = np.sqrt(cond_sample[2].cpu().numpy()**2 + cond_sample[3].cpu().numpy()**2)
    axes[1].imshow(force_vis, cmap="hot", origin="lower")
    axes[1].set_title("Force (shared)", fontsize=11)
    axes[1].axis("off")

    # topopt
    img0 = samples[0, 0].cpu().numpy()
    img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 1e-8)
    axes[2].imshow(img0, cmap="gray_r", origin="lower")
    axes[2].set_title("topopt label", fontsize=12, fontweight="bold", color="blue")
    axes[2].axis("off")

    # finray
    img1 = samples[1, 0].cpu().numpy()
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    axes[3].imshow(img1, cmap="gray_r", origin="lower")
    axes[3].set_title("finray label", fontsize=12, fontweight="bold", color="red")
    axes[3].axis("off")

    # Compute correlation
    corr = torch.corrcoef(torch.stack([
        samples[0].flatten().cpu(), samples[1].flatten().cpu()
    ]))[0, 1].item()
    fig.suptitle(f"Same BCs, Different Labels — pixel corr={corr:.3f}  (cfg={cfg_scale})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
