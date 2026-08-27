"""Mounting-socket add-on for the top of a soft-gripper design.

The socket is a fixed rigid mounting interface that lives *above* the
optimizable finger. The CFM still designs the full (H, W) finger region
unchanged; the socket is appended as extra rows on top so the gripper
can clip into a physical setup.

Layout (with two centred blocks and a gap):

    rows [H, H + socket_rows)  ->  TOP STRIP (socket)
        ┌──────────┬────┬──────────┐
        │  block_L │ gap│  block_R │     (top, +y)
        ├──────────┴────┴──────────┤
    rows [0, H)                ->  FINGER (CFM-designed, unchanged)
        │                          │
        │   ... finger ...         │
        │                          │
        └──────────────────────────┘  (bottom, -y)

Use ``add_socket_to_mask()`` to append the socket to a repaired finger
mask. Use ``socket_block_columns()`` to query where the blocks land so
BCs can be placed on top of them.
"""

from __future__ import annotations

import numpy as np


def socket_height_rows(socket_height_m: float, finger_height_m: float,
                        finger_rows: int) -> int:
    """Convert a physical socket height in metres to a number of pixel rows
    that matches the finger's pixel resolution.

    Pixel pitch ``dy = finger_height_m / finger_rows``. The returned
    socket-row count rounds to the nearest integer with a floor of 1.
    """
    if finger_rows <= 0 or finger_height_m <= 0.0:
        raise ValueError("finger_rows and finger_height_m must be positive")
    dy = finger_height_m / float(finger_rows)
    return max(1, int(round(socket_height_m / dy)))


def socket_block_columns(W: int, gap_frac: float,
                          n_blocks: int = 2) -> list[tuple[int, int]]:
    """Return ``[(col_lo, col_hi), ...]`` for the solid blocks in the
    socket strip (column indices in [0, W)).

    With ``n_blocks=2`` and a centred gap of width ``gap_frac * W``: two
    blocks of equal width on the left and right with the gap between them.

    With ``n_blocks=1``: a single full-width block (no gap), useful for a
    plain mount.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    if n_blocks == 1:
        return [(0, W)]
    if not (0.0 <= gap_frac < 1.0):
        raise ValueError(f"gap_frac must be in [0, 1); got {gap_frac}")
    if n_blocks != 2:
        # General case: equal blocks separated by equal-width gaps so the
        # total gap width = gap_frac * W. Not heavily used; kept for
        # flexibility.
        total_gap = int(round(gap_frac * W))
        n_gaps = n_blocks - 1
        gap_w = max(0, total_gap // n_gaps)
        block_w = (W - n_gaps * gap_w) // n_blocks
        out: list[tuple[int, int]] = []
        cursor = 0
        for i in range(n_blocks):
            out.append((cursor, cursor + block_w))
            cursor += block_w + gap_w
        # Snap the last block to the right edge to absorb rounding.
        out[-1] = (out[-1][0], W)
        return out
    # n_blocks == 2: symmetric left + right with a single centred gap.
    gap_w = int(round(gap_frac * W))
    block_w = (W - gap_w) // 2
    return [(0, block_w), (W - block_w, W)]


def neck_column_ranges(W: int, gap_frac: float, n_blocks: int,
                        neck_width_frac: float) -> list[tuple[int, int]]:
    """Per-block column range of the (narrower) neck region.

    Each neck is centred within its block and has width
    ``round(neck_width_frac * block_w)``.
    """
    out: list[tuple[int, int]] = []
    for c_lo, c_hi in socket_block_columns(W, gap_frac, n_blocks):
        block_w = c_hi - c_lo
        neck_w = max(1, int(round(neck_width_frac * block_w)))
        neck_lo = c_lo + (block_w - neck_w) // 2
        neck_hi = neck_lo + neck_w
        out.append((neck_lo, neck_hi))
    return out


def _block_notch_columns(
    block_cols: list[tuple[int, int]],
    notch_cols: int,
    notch_side: str,
) -> list[list[tuple[int, int]]]:
    """For each block, return the column ranges where the notch removes
    material at the bottom rows. ``notch_side`` ∈ {'inner', 'outer',
    'both', 'none'}. For ``n_blocks == 2``: 'inner' = facing the gap
    (left block → right corner; right block → left corner). For
    ``n_blocks == 1``: 'inner' has no meaning and produces no notch.

    Returns a list with one inner list per block, where each inner list
    contains 0–2 ``(col_lo, col_hi)`` tuples.
    """
    out: list[list[tuple[int, int]]] = []
    if notch_cols <= 0 or notch_side == "none":
        return [[] for _ in block_cols]
    n = len(block_cols)
    for i, (c_lo, c_hi) in enumerate(block_cols):
        per_block: list[tuple[int, int]] = []
        if n != 2:
            out.append(per_block)
            continue
        is_left = (i == 0)
        # Inner = side facing the gap.
        inner_lo, inner_hi = (
            (max(c_lo, c_hi - notch_cols), c_hi) if is_left
            else (c_lo, min(c_hi, c_lo + notch_cols))
        )
        outer_lo, outer_hi = (
            (c_lo, min(c_hi, c_lo + notch_cols)) if is_left
            else (max(c_lo, c_hi - notch_cols), c_hi)
        )
        if notch_side in ("inner", "both") and inner_hi > inner_lo:
            per_block.append((inner_lo, inner_hi))
        if notch_side in ("outer", "both") and outer_hi > outer_lo:
            per_block.append((outer_lo, outer_hi))
        out.append(per_block)
    return out


def make_socket_strip(
    W: int,
    block_rows: int,
    neck_rows: int,
    gap_frac: float,
    n_blocks: int = 2,
    neck_width_frac: float = 0.5,
    notch_rows: int = 0,
    notch_cols: int = 0,
    notch_side: str = "inner",
) -> np.ndarray:
    """Return ``(neck_rows + block_rows, W)`` uint8 socket strip.

    Layout (bottom → top in row order):
        rows [0, neck_rows)                       optional narrow necks
                                                  (centred, width =
                                                  neck_width_frac × block_w)
        rows [neck_rows, neck_rows + block_rows)  wide blocks separated by
                                                  the centred gap, with
                                                  optional notch cut from
                                                  the bottom-inner corner

    Notches sit at the BOTTOM of each block (rows
    [neck_rows, neck_rows + notch_rows)) on the side specified by
    ``notch_side``. They're the mounting lip that catches on a physical
    clip.
    """
    total = neck_rows + block_rows
    strip = np.zeros((total, W), dtype=np.uint8)
    block_cols = socket_block_columns(W, gap_frac, n_blocks)
    notch_specs = _block_notch_columns(block_cols, notch_cols, notch_side)

    # Wide blocks at the top — fill, then carve out notches.
    for (c_lo, c_hi), notches in zip(block_cols, notch_specs):
        strip[neck_rows:total, c_lo:c_hi] = 1
        if notch_rows > 0:
            for n_lo, n_hi in notches:
                strip[neck_rows:neck_rows + notch_rows, n_lo:n_hi] = 0

    # Optional necks (narrow) below the blocks.
    if neck_rows > 0:
        for neck_lo, neck_hi in neck_column_ranges(
            W, gap_frac, n_blocks, neck_width_frac,
        ):
            strip[0:neck_rows, neck_lo:neck_hi] = 1
    return strip


def add_socket_to_mask(
    finger_mask: np.ndarray,
    block_rows: int,
    neck_rows: int = 0,
    gap_frac: float = 1.0 / 3.0,
    n_blocks: int = 2,
    neck_width_frac: float = 0.5,
    notch_rows: int = 0,
    notch_cols: int = 0,
    notch_side: str = "inner",
    finger_overlap_rows: int = 2,
    enforce_connectivity: bool = True,   # DEPRECATED: no-op, kept for call compatibility
) -> tuple[np.ndarray, dict]:
    """Append a socket (necks + blocks) on top of a finger mask.

    Connection strategy:
      1. The socket strip itself includes narrow ``neck`` rows below the
         wide blocks; the necks are the "bottleneck" mounting transition.
      2. The top ``finger_overlap_rows`` rows of the finger, restricted to
         the neck column ranges, are forced to material so the necks land
         on solid pixels (the finger's top boundary often has voids in
         those columns).
      3. Nothing else. The original pipeline ran a cluster-repair pass here that
         bridged disconnected components; this release does not repair, so the
         combined mask is reported as-is (see ``n_components_*`` in ``info``).

    Parameters
    ----------
    finger_mask           : (H, W) uint8 / bool finger, row 0 = bottom.
    block_rows            : number of pixel rows for the wide blocks.
    neck_rows             : number of pixel rows for the narrow necks (0 disables).
    gap_frac              : centred-gap width as a fraction of W (n_blocks=2 only).
    n_blocks              : 1 = single full-width block, 2 = left + right + gap.
    neck_width_frac       : neck width as a fraction of its block width.
    finger_overlap_rows   : how many top finger rows to force material under
                            each neck so the socket attaches to the finger.
    enforce_connectivity  : DEPRECATED and IGNORED. This release performs no
                            repair; connectivity is reported, not enforced. Formerly: run the cluster
                            bridge + close + small-hole-fill to guarantee one
                            connected component.

    Returns
    -------
    combined_mask : (H + neck_rows + block_rows, W) uint8 — full mask.
    info          : dict with keys
        block_cols       : list of (c_lo, c_hi)
        neck_cols        : list of (c_lo, c_hi)
        block_rows       : echoed
        neck_rows        : echoed
        overlap_rows     : echoed
        n_components_before_repair : connected components of the combined mask (before the
                                       mask is used as-is).
        n_components_after_repair  : same value (no repair is applied); >1 means the
                                     finger is not fully attached to the socket.
    """
    if finger_mask.ndim != 2:
        raise ValueError(f"finger_mask must be 2D; got {finger_mask.shape}")
    H, W = finger_mask.shape

    block_cols = socket_block_columns(W, gap_frac, n_blocks)
    neck_cols = neck_column_ranges(W, gap_frac, n_blocks, neck_width_frac) \
        if neck_rows > 0 else []
    notch_specs = _block_notch_columns(block_cols, notch_cols, notch_side) \
        if notch_rows > 0 else [[] for _ in block_cols]

    # 1. Force material in the top of the finger so the socket lands on
    #    solid pixels. If a neck is present we use neck-column ranges;
    #    otherwise we use each block's non-notched bottom footprint.
    finger_modified = finger_mask.astype(np.uint8).copy()
    overlap = max(0, min(int(finger_overlap_rows), H))
    if overlap > 0:
        if neck_rows > 0:
            for neck_lo, neck_hi in neck_cols:
                finger_modified[H - overlap:H, neck_lo:neck_hi] = 1
        else:
            # No neck — force material under each block's bottom edge,
            # EXCLUDING the notch column ranges (the notch is meant to
            # be void). Forces a small "tongue" of material that the
            # notched block sits on.
            for (c_lo, c_hi), notches in zip(block_cols, notch_specs):
                # Build per-col mask of the block bottom minus notch.
                col_mask = np.ones(c_hi - c_lo, dtype=bool)
                for n_lo, n_hi in notches:
                    col_mask[max(0, n_lo - c_lo):n_hi - c_lo] = False
                if col_mask.any():
                    rel = np.where(col_mask)[0]
                    abs_cols = rel + c_lo
                    finger_modified[H - overlap:H, abs_cols] = 1

    # 2. Build the socket strip (necks + blocks with notches) and concatenate.
    strip = make_socket_strip(
        W, block_rows=block_rows, neck_rows=neck_rows,
        gap_frac=gap_frac, n_blocks=n_blocks,
        neck_width_frac=neck_width_frac,
        notch_rows=notch_rows, notch_cols=notch_cols, notch_side=notch_side,
    )
    combined = np.concatenate([finger_modified, strip], axis=0)

    # 3. Connectivity is REPORTED, never repaired.
    #
    # The original pipeline bridged floating clusters here (design_validation.
    # check_and_repair_clusters). This release runs WITHOUT any repair step, so a
    # design that leaves material disconnected from the socket keeps it: the
    # floating island carries no boundary condition, contributes no grasp force,
    # and the design simply scores badly. That is the honest signal for a search --
    # repairing it would silently hand fitness to a design that does not hold
    # together. `n_components` is returned in `info` so callers can gate on it.
    from scipy.ndimage import label as ndimage_label
    _, n_before = ndimage_label(combined)
    n_after = n_before

    info = dict(
        block_cols=block_cols,
        neck_cols=neck_cols,
        notch_specs=notch_specs,
        block_rows=int(block_rows),
        neck_rows=int(neck_rows),
        notch_rows=int(notch_rows),
        notch_cols=int(notch_cols),
        notch_side=str(notch_side),
        overlap_rows=int(overlap),
        n_components_before_repair=int(n_before),
        n_components_after_repair=int(n_after),
    )
    return combined.astype(np.uint8), info


def adjusted_physical_size_and_center(
    finger_width: float,
    finger_height: float,
    center_x: float,
    center_y: float,
    socket_height: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(physical_size, center)`` for the *combined* mesh
    (finger + socket on top) such that the *finger* occupies the same
    physical region it would without a socket.

    The finger's bottom y is preserved at ``center_y - finger_height/2``;
    the socket extends upward by ``socket_height``; the combined center
    shifts up by ``socket_height/2``.
    """
    total_h = finger_height + socket_height
    new_center_y = center_y + socket_height / 2.0
    return (finger_width, total_h), (center_x, new_center_y)
