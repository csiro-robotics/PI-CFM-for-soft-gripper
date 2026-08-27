"""Single-finger gripper geometry: socket, boundary conditions, multi-env mesh.

Extracted so the implicit solver has NO dependency on the explicit FEM path. In the
original project `build_multi_env` reached into `warp_grasp` (the explicit solver),
`run_voronoi_batched_sim` and `utils.socket_mask` at call time; those three are
folded in here.

The design variable is ONE finger (a 128x64 binary mask). A socket is appended
above the design space -- two blocks, left pinned / right driven -- and the finger
closes on a rigid disc. `build_multi_env` lays B independent fingers into one flat
mesh so a whole QD batch is evaluated in a single solve; every env keeps its own
node/triangle ranges (`node2env`, `tri2env`), so designs never interact.

`_socket_bc_nodes` here is AST-identical to the one it replaces. `_pick_top_strip`
differs only in the order of the dedupe/filter steps, which is immaterial (verified
equivalent over 2e5 random cases).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from scipy.ndimage import binary_erosion, label as ndi_label

from socket_mask import socket_height_rows, add_socket_to_mask
from mesh import build_mesh


# Socket rig geometry (metres / fractions). Matches the physical mount.
SOCKET_GEOMETRY = dict(
    block_height=0.013,        # wide block region height (m)
    neck_height=0.0,           # narrow neck below blocks; 0 = none
    notch_height=0.0065,       # "L" cut at the bottom-inner corner of each block
    notch_width=0.00525,
    notch_side="inner",
    gap_frac=1.0 / 3.0,
    n_blocks=2,
    neck_width_frac=0.5,
    finger_overlap_rows=2,
)


def _pick_top_strip(me, env_id, ix_lo, ix_hi, ny):
    """Top-strip corner node ids on the design space (walks down until >=2 are live)."""
    for iy in range(ny, -1, -1):
        ids = [me.node_index(env_id, ix, iy) for ix in range(ix_lo, ix_hi)]
        ids = [n for n in dict.fromkeys(ids) if n >= 0]
        if len(ids) >= 2:
            return ids
    return []


def _socket_bc_nodes(me, masks_xy, B, block_cols, H, block_rows, n_corner=5):
    """Fixed/driven BC = socket-CELL corner nodes + the top-strip corners.

    Selecting by socket-cell membership (rows [H, H+block_rows)) follows the socket
    material including the notch; a plain column rectangle would wrongly grab ~10
    finger/notch nodes per block on the design-space interface.
    Left block + left corners -> pinned, right block + right corners -> driven.
    """
    (l_lo, l_hi), (r_lo, r_hi) = block_cols
    nx = me.nx
    pinned, driven = [], []
    for b in range(B):
        m = masks_xy[b]
        pin, drv = set(), set()
        for ix in range(nx):
            for iy in range(H, H + block_rows):
                if not m[ix, iy]:
                    continue
                side = pin if ix < l_hi else (drv if ix >= r_lo else None)
                if side is None:
                    continue
                for ax_, ay_ in ((ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)):
                    n = me.node_index(b, ax_, ay_)
                    if n >= 0:
                        side.add(n)
        pin |= set(_pick_top_strip(me, b, 0, n_corner, H - 1))
        drv |= set(_pick_top_strip(me, b, nx + 1 - n_corner, nx + 1, H - 1))
        drv -= pin
        pinned += sorted(pin); driven += sorted(drv)
    return pinned, driven


def socket_block_cols(cfg, W):
    """Column spans of the socket blocks for a design W pixels wide."""
    from socket_mask import socket_block_columns
    G = SOCKET_GEOMETRY
    return socket_block_columns(W, G["gap_frac"], G["n_blocks"])


def anchor_valid(mask, block_cols, anchor_rows: int = 6, min_fill: float = 0.20) -> bool:
    """Reject ANCHOR-STARVED fingers (no material where the socket clamps).

    add_socket_to_mask() adds the fixed/driven socket on TOP of the finger and
    forces material in the top `finger_overlap_rows` under the necks so the socket
    lands on solid pixels. A finger with almost nothing under the blocks therefore
    still meshes -- it hangs off that forced tongue by a hairline -- and passes the
    inversion/contact guards while having no real material at the fixed nodes.
    Two cheap mask-only checks close that hole:

      (1) the anchor band (top `anchor_rows` finger rows, under the block
          footprints) must carry >= `min_fill` material in the ORIGINAL finger
          (not the socket-forced overlap rows);
      (2) after a 1px erosion (which snaps hairline bridges), the finger's
          largest connected component must span anchor band -> contact region
          (bottom half), i.e. a continuous load path exists.

    mask: (H, W) finger, row 0 = bottom (socket side = high rows).  block_cols:
    list of (c_lo, c_hi) socket-block column spans (from add_socket_to_mask info).
    """
    m = np.asarray(mask) > 0
    H, W = m.shape
    ar = max(1, min(int(anchor_rows), H))
    cols = np.zeros(W, dtype=bool)
    for c_lo, c_hi in block_cols:
        cols[int(c_lo):int(c_hi) + 1] = True
    if not cols.any():
        cols[:] = True
    if float(m[H - ar:H, cols].mean()) < min_fill:           # (1) loaded anchor band
        return False
    er = binary_erosion(m, iterations=1)                     # (2) continuous load path
    lbl, n = ndi_label(er)
    if n == 0:
        return False
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    comp = lbl == int(sizes.argmax())
    return bool(comp[H - ar:H, cols].any() and comp[:H // 2, :].any())

def keep_socket_component(comb):
    """Drop material not connected to the socket. REMOVAL ONLY -- never adds a pixel.

    This release runs the generator WITHOUT the NV-loop rib repair, so a design can
    decode into several disconnected pieces (2-7 is typical for a random genome).
    A piece that does not reach the socket carries no boundary condition: in the FEM
    it is a free body with mass and no gravity, so it neither transmits grasp force
    nor holds the object -- but it CAN be pushed by the disc and register contact
    arc, crediting a design for material that is not attached to the gripper.

    Deleting those pieces is not repair: it never invents material, it only refuses
    to simulate parts that would fall off a printed finger. Set cfg.drop_islands
    False to simulate the raw mask instead.

    comb : (H+socket_rows, W) mask, row 0 = bottom, socket occupies the TOP rows.
    """
    from scipy.ndimage import label as ndimage_label
    lab, n = ndimage_label(comb)
    if n <= 1:
        return comb, n
    keep = set(np.unique(lab[-1, :]))        # labels touching the socket's top row
    keep.discard(0)
    if not keep:                             # no material at the socket at all
        return np.zeros_like(comb), n
    return np.isin(lab, list(keep)).astype(comb.dtype), n


def _socketed(mask, cfg, H, W):
    """(mask+socket in xy order, info, block_rows, effective height/centre-y)."""
    G = SOCKET_GEOMETRY
    fw, fh = cfg.finger_width, cfg.finger_height
    block_rows = socket_height_rows(G["block_height"], fh, H)
    notch_rows = socket_height_rows(G["notch_height"], fh, H) if G["notch_height"] > 0 else 0
    notch_cols = max(1, int(round(G["notch_width"] / (fw / W)))) if G["notch_width"] > 0 else 0
    comb, info = add_socket_to_mask(
        mask, block_rows=block_rows, neck_rows=0, gap_frac=G["gap_frac"],
        n_blocks=G["n_blocks"], neck_width_frac=G["neck_width_frac"],
        notch_rows=notch_rows, notch_cols=notch_cols, notch_side=G["notch_side"],
        finger_overlap_rows=G["finger_overlap_rows"], enforce_connectivity=True)
    # No repair: a design that decodes into several pieces keeps them. By default we
    # simulate only the piece bolted to the socket (removal, never addition).
    if getattr(cfg, "drop_islands", True):
        comb, n_comp = keep_socket_component(comb)
        info["n_components_kept"] = 1 if comb.any() else 0
    else:
        n_comp = info["n_components_before_repair"]
        info["n_components_kept"] = n_comp
    info["n_components"] = n_comp
    soh = G["block_height"]
    return comb.T.copy(), info, block_rows, (fh + soh, cfg.center_y + soh / 2.0)


def build_multi_env(masks, cfg, H, W):
    """B single fingers in ONE flat multi-env mesh. Returns (mesh, pinned, driven, block_rows)."""
    masks_xy, infos = [], []
    eff_h = eff_cy = block_rows = None
    for m in masks:
        mxy, info, block_rows, (eff_h, eff_cy) = _socketed(m, cfg, H, W)
        masks_xy.append(mxy); infos.append(info)
    B = len(masks)
    me = build_mesh(masks_xy, W, H + block_rows, (cfg.finger_width, eff_h),
                    (cfg.center_x, eff_cy), cfg.young, cfg.nu, cfg.rho_mass)
    pinned, driven = _socket_bc_nodes(me, masks_xy, B, infos[0]["block_cols"], H, block_rows)
    me.n_components = np.array([i["n_components"] for i in infos], np.int32)
    return me, np.asarray(pinned, np.int64), np.asarray(driven, np.int64), block_rows


def build_single_finger(mask, cfg, H, W):
    """Convenience wrapper: one design, one env."""
    return build_multi_env([mask], cfg, H, W)
