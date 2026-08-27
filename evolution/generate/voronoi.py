"""
Voronoi condition generator.

Two flavours, both producing a 3-channel RGB field of shape (3, H, W) in
[0, 1] — the same format consumed by the CFM sampler via
`build_cppn_regions(rgb_uint8)`.

  * `HardVoronoi`  : numpy, nearest-site rasterisation. Cheap, non-
                     differentiable.  Used by genetic-algorithm drivers
                     (mutate sites / colours, no gradient required).

  * `SoftVoronoi`  : torch.nn.Module.  Sites and per-site colours are
                     `nn.Parameter`s.  The hard `argmin` over distances
                     is replaced by `softmax(-β · d²)` so the whole
                     forward is differentiable.  As β → ∞ this recovers
                     the hard Voronoi.  Use this when an upstream
                     simulator / sampler can supply a gradient.

Per-site colours are kept in **full RGB** (continuous in [0, 1]^3).  This
is a strict generalisation of the categorical "R = topopt, B = finray,
G = graph" mapping used in the early prototype:

    one-hot colours -> identical to the categorical scheme
    soft  colours   -> per-cell *blends* between the three design spaces,
                       which is what `build_cppn_regions` expects anyway
                       (it normalises R/G/B to sum to 1 per pixel).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hard (numpy) Voronoi
# ---------------------------------------------------------------------------
@dataclass
class HardVoronoi:
    """Nearest-site Voronoi rasteriser.

    sites  : (K, 2) float in [0, 1], coords as (y, x)
    colors : (K, 3) float in [0, 1], per-site RGB
    """

    sites: np.ndarray
    colors: np.ndarray

    # ----- constructors ----------------------------------------------------
    @classmethod
    def random(
        cls,
        seed: int,
        n_sites: int,
        categorical: bool = False,
        class_set: Optional[list[int]] = None,
        class_probs: Optional[list[float]] = None,
        palette: Optional[np.ndarray] = None,
    ) -> "HardVoronoi":
        """Random sites with random colours.

        ``palette`` (K, 3) overrides ``categorical``/``class_set``: each
        site is assigned one row of the palette, chosen uniformly (or
        with ``class_probs`` weights of length K). This lets you build
        arbitrary mixture classes (e.g. 50% R + 50% B).

        If ``palette`` is None and ``categorical=True``, colours are
        drawn from the 3-class one-hot palette (R=0, G=1, B=2) optionally
        restricted by ``class_set``. Otherwise colours are continuous.
        """
        rng = np.random.default_rng(seed)
        sites = rng.random((n_sites, 2), dtype=np.float32)

        if palette is not None:
            pal = np.asarray(palette, dtype=np.float32)
            if pal.ndim != 2 or pal.shape[1] != 3:
                raise ValueError("palette must have shape (K, 3)")
            K = pal.shape[0]
            if class_probs is not None:
                probs = np.asarray(class_probs, dtype=np.float64)
                if probs.shape != (K,):
                    raise ValueError("class_probs length must match palette")
                probs = probs / probs.sum()
            else:
                probs = None
            idx = rng.choice(K, size=(n_sites,), p=probs)
            if n_sites >= K:
                idx[:K] = np.arange(K)
                rng.shuffle(idx)
            colors = pal[idx]
        elif categorical:
            base = np.eye(3, dtype=np.float32)
            allowed = list(range(3)) if class_set is None else list(class_set)
            if not allowed:
                raise ValueError("class_set must be non-empty")
            if class_probs is not None:
                probs = np.asarray(class_probs, dtype=np.float64)
                if probs.shape != (len(allowed),):
                    raise ValueError("class_probs length must match class_set")
                probs = probs / probs.sum()
            else:
                probs = None
            idx = rng.choice(allowed, size=(n_sites,), p=probs)
            if n_sites >= len(allowed):
                idx[: len(allowed)] = np.array(allowed)
                rng.shuffle(idx)
            colors = base[idx]
        else:
            colors = rng.random((n_sites, 3), dtype=np.float32)
        return cls(sites=sites, colors=colors)

    # ----- core operation --------------------------------------------------
    def sample_field(self, H: int, W: int) -> np.ndarray:
        """Return an (3, H, W) RGB field in [0, 1]."""
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, H, dtype=np.float32),
            np.linspace(0.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        coords = np.stack([yy, xx], axis=-1).reshape(-1, 2)            # (HW, 2)
        d2 = ((coords[:, None, :] - self.sites[None, :, :]) ** 2).sum(-1)
        nearest = d2.argmin(axis=1).reshape(H, W)                      # (H, W)
        rgb = self.colors[nearest]                                     # (H, W, 3)
        return np.clip(rgb, 0.0, 1.0).transpose(2, 0, 1).copy()

    def sample_rgb_uint8(self, H: int, W: int) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB image — handy for plotting."""
        rgb = self.sample_field(H, W)
        return (rgb.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Soft (torch) Voronoi — differentiable
# ---------------------------------------------------------------------------
class SoftVoronoi(nn.Module):
    """Differentiable Voronoi.

    Forward:
        weights = softmax(-β · ||grid - sites||²)   over sites
        field   = weights @ colors                  -> (H, W, 3)

    Both `sites_xy` and `color_logits` are learnable `nn.Parameter`s.
    Colours are kept in [0, 1] via a sigmoid.  As β → ∞ the softmax
    collapses to the hard `argmin`, recovering `HardVoronoi`.
    """

    def __init__(
        self,
        n_sites: int,
        beta: float = 200.0,
        seed: Optional[int] = None,
        categorical_init: bool = False,
        class_set: Optional[list[int]] = None,
        class_probs: Optional[list[float]] = None,
        palette: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_sites = n_sites
        # β is stored as a buffer (not a parameter) — use `set_beta` to
        # anneal it during training.
        self.register_buffer("beta", torch.tensor(float(beta)))

        g = torch.Generator().manual_seed(seed) if seed is not None else None
        sites = torch.rand((n_sites, 2), generator=g)                 # (K, 2)

        if palette is not None:
            pal = torch.as_tensor(np.asarray(palette), dtype=torch.float)
            if pal.ndim != 2 or pal.shape[1] != 3:
                raise ValueError("palette must have shape (K, 3)")
            K = pal.shape[0]
            if class_probs is not None:
                probs_t = torch.tensor(class_probs, dtype=torch.float)
                if probs_t.numel() != K:
                    raise ValueError("class_probs length must match palette")
                probs_t = probs_t / probs_t.sum()
                pick = torch.multinomial(probs_t, n_sites, replacement=True,
                                         generator=g)
            else:
                pick = torch.randint(0, K, (n_sites,), generator=g)
            if n_sites >= K:
                pick[:K] = torch.arange(K)
                perm = torch.randperm(n_sites, generator=g)
                pick = pick[perm]
            chosen = pal[pick].clamp(1e-4, 1.0 - 1e-4)
            # Invert sigmoid to recover logits so SoftVoronoi can train.
            color_logits = torch.log(chosen / (1.0 - chosen))
        elif categorical_init:
            allowed = list(range(3)) if class_set is None else list(class_set)
            if not allowed:
                raise ValueError("class_set must be non-empty")
            allowed_t = torch.tensor(allowed, dtype=torch.long)
            if class_probs is not None:
                probs_t = torch.tensor(class_probs, dtype=torch.float)
                if probs_t.numel() != len(allowed):
                    raise ValueError("class_probs length must match class_set")
                probs_t = probs_t / probs_t.sum()
                pick = torch.multinomial(probs_t, n_sites, replacement=True,
                                         generator=g)
            else:
                pick = torch.randint(0, len(allowed), (n_sites,), generator=g)
            idx = allowed_t[pick]
            if n_sites >= len(allowed):
                idx[: len(allowed)] = allowed_t
                perm = torch.randperm(n_sites, generator=g)
                idx = idx[perm]
            color_logits = -3.0 * torch.ones(n_sites, 3)
            color_logits[torch.arange(n_sites), idx] = 3.0
        else:
            color_logits = torch.randn((n_sites, 3), generator=g) * 0.5

        self.sites_xy = nn.Parameter(sites)            # (K, 2) free in R^2
        self.color_logits = nn.Parameter(color_logits) # (K, 3)

    # ----------------------------------------------------------------------
    def set_beta(self, beta: float) -> None:
        self.beta.fill_(float(beta))

    @property
    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits)        # (K, 3) in (0, 1)

    @property
    def sites(self) -> torch.Tensor:
        # Sites live conceptually in [0, 1]² but we don't hard-clamp so
        # gradients can push them around freely; only used for plotting.
        return self.sites_xy

    # ----------------------------------------------------------------------
    def _grid(self, H: int, W: int, device, dtype) -> torch.Tensor:
        yy = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        xx = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([gy, gx], dim=-1)                            # (H, W, 2)

    def forward(self, H: int, W: int) -> torch.Tensor:
        """Return a differentiable (3, H, W) RGB field in [0, 1]."""
        device = self.sites_xy.device
        dtype = self.sites_xy.dtype
        grid = self._grid(H, W, device, dtype).reshape(-1, 2)           # (HW, 2)

        diff = grid[:, None, :] - self.sites_xy[None, :, :]             # (HW, K, 2)
        d2 = (diff * diff).sum(-1)                                      # (HW, K)
        weights = torch.softmax(-self.beta * d2, dim=-1)                # (HW, K)

        field = weights @ self.colors                                   # (HW, 3)
        field = field.clamp(0.0, 1.0)
        return field.view(H, W, 3).permute(2, 0, 1).contiguous()

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def to_hard(self) -> HardVoronoi:
        """Snapshot current parameters as a `HardVoronoi` (categorical
        argmax → site colour kept as is)."""
        return HardVoronoi(
            sites=self.sites_xy.detach().cpu().numpy().astype(np.float32),
            colors=self.colors.detach().cpu().numpy().astype(np.float32),
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def make_voronoi(
    seed: int,
    n_sites: int,
    differentiable: bool = False,
    **kwargs,
) -> HardVoronoi | SoftVoronoi:
    """Factory mirroring `make_cppn`: pick numpy or torch backend."""
    if differentiable:
        return SoftVoronoi(n_sites=n_sites, seed=seed, **kwargs)
    return HardVoronoi.random(seed=seed, n_sites=n_sites, **kwargs)


def voronoi_field(
    seed: int,
    n_sites: int,
    H: int,
    W: int,
    categorical: bool = False,
) -> np.ndarray:
    """Quick numpy (3, H, W) RGB field from a fresh hard Voronoi."""
    return HardVoronoi.random(seed=seed, n_sites=n_sites,
                              categorical=categorical).sample_field(H, W)
