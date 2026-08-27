"""Behaviour descriptors — the 4-D CVT MAP-Elites axes.

    strut_complexity   fraction of material in THIN struts   (a.k.a. strut_thinness)
    branch_density     skeleton branch-points per unit length
    hole_count         normalised number of enclosed voids
    material_fraction  mean of the binary mask (defined inline where used)

All take the (H, W) binary design mask and return a scalar in [0, 1], so the
archive axes are directly comparable across designs. Extracted verbatim from the
research code so archives stay comparable with previously published runs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve, label as ndi_label
from skimage.morphology import skeletonize


def strut_complexity(mask) -> float:
    """STRUCTURAL fineness in [0,1]: fraction of the material that sits in THIN
    struts (local half-thickness <= 2 px ~ 2 mm).  0 = chunky/solid blob,
    1 = lacy/filigree network.  Computed from a distance transform, so it is
    smooth, robust, and roughly orthogonal to material_fraction (you can be
    thick or thin at any density) -> it spreads STRUCTURE across the map where
    material/wrap only spread amount.  Use as a MAP-Elites descriptor axis."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return 0.0
    dt = distance_transform_edt(m)            # per-pixel local half-thickness (px)
    return float((dt[m] <= 2.0).mean())


def _skel_neighbours(sk):
    """8-connected skeleton-neighbour count per skeleton pixel (self removed)."""
    return convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk.astype(np.uint8)


def branch_density(mask, prune: int = 3) -> float:
    """TOPOLOGICAL branchiness = skeleton branch-points / skeleton length.
    Low = simple strut / few junctions; high = bushy, networked.  Normalised by
    skeleton length so it is ~orthogonal to BOTH material_fraction and mean
    thickness (it measures how the centreline *connects*, not how long or thick
    it is).  Short spurs (<= `prune` px) are trimmed first so boundary roughness
    doesn't fake junctions.  Use as a MAP-Elites descriptor axis."""
    m = np.asarray(mask).astype(bool)
    if m.sum() < 5:
        return 0.0
    sk = skeletonize(m)
    for _ in range(prune):                       # erode spurs/tips by `prune` px
        sk = sk & (_skel_neighbours(sk) != 1)
    L = int(sk.sum())
    if L == 0:
        return 0.0
    branches = int((sk & (_skel_neighbours(sk) >= 3)).sum())
    return float(branches) / float(L)


def hole_count(mask) -> float:
    """TOPOLOGICAL genus proxy: number of enclosed voids (background regions that
    do NOT touch the image border).  0 = simple solid/curved finger; high = a
    lattice/ladder with many windows.  Captures connectivity variety the thinness
    and branch axes miss.  MAP-Elites descriptor axis (count, cap via the range)."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return 0.0
    lbl, n = ndi_label(~m)                              # label the BACKGROUND
    if n == 0:
        return 0.0
    border = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    border = set(int(x) for x in border)
    return float(sum(1 for i in range(1, n + 1) if i not in border))

def material_fraction(mask) -> float:
    """Fraction of the design space that is solid."""
    return float(np.asarray(mask).astype(bool).mean())


# The 4-D CVT axis set used by the released search, in archive order.
AXES = ("strut_complexity", "branch_density", "hole_count", "material_fraction")


def descriptors(mask) -> dict:
    """All four axes for one design."""
    return {
        "strut_complexity": strut_complexity(mask),
        "branch_density": branch_density(mask),
        "hole_count": hole_count(mask),
        "material_fraction": material_fraction(mask),
    }
