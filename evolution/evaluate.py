"""Fitness: design mask -> implicit PNCG-IPC grasp sim -> composite score.

ONE soft finger closes on a rigid disc and then lifts it. The finger is bolted into
a two-block socket above the design space (left block pinned, right block driven);
`close` drives the socket down by a 10 mm cosine-eased stroke so the finger wraps
the disc, then `pull` lifts both blocks 30 mm and the peak reaction on the disc is
recorded. Everything runs on the implicit backward-Euler PNCG solver with IPC
log-barrier contact, self-collision and friction -- the same solver, at the same
operating point, that produced output/gallery_v3 in the research repo.

    score = w_force * tanh(pull_off / f_ref) + w_wrap * min(arc / circumference, 1)

with w_force = 0.2, w_wrap = 0.8: mostly "how much of the object do you wrap", with
a saturating force term so a single very strong design cannot dominate the map.

OPERATING POINT (pinned; do not drift from these without re-deriving f_ref):
    E    1.9 MPa            the printed material; scale_for_material rescales dt,
                            damping, the penalty stiffnesses and f_ref with it
    mu   0.9514042          sys-identified against the 1.0x-sphere close/pull rig
                            at dt = 2 ms (that fit's own E was 1.0019 MPa, so the
                            contact/damping constants are carried over, not refitted)
    dt   2 ms               the rate the material was identified at
    close 1.10 s, pull 3.03 s over 30 mm, disc r = 14 mm, finger gap 52 mm

The gap is the two-finger rig's finger-to-finger opening. One finger sees half of
it: the object-facing surface sits gap/2 = 26 mm from the disc centre, i.e. 12 mm
of clearance to a 14 mm disc. `object_x_from_gap` does that conversion, so a gap
quoted from a two-finger experiment transfers here unambiguously.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "sim"), str(_HERE / "generate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import SimCfg, scale_for_material                        # noqa: E402
from descriptors import descriptors                                  # noqa: E402

# ---------------------------------------------------------------- operating point
# Implicit sys-id best_params (output/sysid_sphere10_dt2ms), the material behind
# output/gallery_v3. damp_factor MULTIPLIES the material-scaled damping.
MATERIAL = dict(
    E_MPa=1.9,
    mu=0.9514042276941765,
    damp_factor=0.4096152938020379,
    d_hat=6.196980026582734e-4,
    epsv=1.2903984284627922e-4,
)
DT = 2.0e-3          # implicit step (s) -- the rate the material was identified at
CLOSE_T = 1.10       # close phase (s)
PULL_T = 3.03        # pull phase (s)
PULL_MM = 30.0       # lift distance (mm)
GAP_MM = 52.0        # two-finger equivalent opening (mm); one finger sees gap/2
OBJECT_R_MM = 14.0   # rigid disc radius (mm)
W_FORCE, W_WRAP = 0.2, 0.8
ITER_MAX = 150       # PNCG iterations per implicit step


def object_x_from_gap(cfg, gap_mm=GAP_MM):
    """Disc-centre x for a two-finger opening of `gap_mm`, with ONE finger.

    The finger's object-facing surface is at center_x - finger_width/2; in the
    two-finger rig that surface sits gap/2 from the disc centre. Returns the
    object_x that reproduces the same clearance with a single finger."""
    return cfg.center_x - 0.5 * cfg.finger_width - 0.5 * gap_mm * 1e-3


def make_cfg(gap_mm=GAP_MM, object_r_mm=OBJECT_R_MM, pull_mm=PULL_MM,
             drop_islands=True, material=None):
    """SimCfg pinned to the gallery_v3 operating point. Returns (cfg, info).

    scale_for_material rescales dt / damping / stiffnesses / caps / f_ref to the
    target E while holding the dimensionless dynamics fixed; the sys-id then
    overrides the four contact/damping parameters it actually fitted."""
    m = dict(MATERIAL) if material is None else dict(material)
    cfg, info = scale_for_material(SimCfg(), m["E_MPa"] * 1e6)
    cfg.obstacle_friction = float(m["mu"])
    cfg.damping *= float(m["damp_factor"])
    cfg.ipc_d_hat = float(m["d_hat"])
    cfg.obstacle_friction_epsv = float(m["epsv"])
    cfg.object_r = object_r_mm * 1e-3
    cfg.object_x = object_x_from_gap(cfg, gap_mm)
    cfg.pull_distance = pull_mm * 1e-3
    cfg.drop_islands = bool(drop_islands)
    info = dict(info, gap_mm=gap_mm, object_r_mm=object_r_mm, pull_mm=pull_mm,
                dt_ms=DT * 1e3, close_s=CLOSE_T, pull_s=PULL_T, **m)
    return cfg, info


# ---------------------------------------------------------------------- objective
def composite_score(m, cfg, w_force=W_FORCE, w_wrap=W_WRAP, f_ref=None):
    """Force + wrap composite in [0, w_force + w_wrap]; 0 for an invalid design.

    f_ref defaults to cfg.f_ref, which scale_for_material has already rescaled with
    E -- pull-off scales with stiffness, so a fixed f_ref would make the force term
    a proxy for stiffness rather than for grip."""
    if not m["valid"]:
        return 0.0
    po, arc = m["pull_off"], m["arc"]
    if not (math.isfinite(po) and math.isfinite(arc)):
        return 0.0
    f_ref = float(cfg.f_ref if f_ref is None else f_ref)
    circ = 2.0 * math.pi * cfg.object_r
    s = w_force * math.tanh(po / f_ref) + w_wrap * min(arc / circ, 1.0)
    return float(s) if math.isfinite(s) else 0.0


def effective_mask(mask, cfg):
    """The part of a design that the solver actually sees, as an (H, W) finger mask.

    Socketing is not the identity: it forces material into the top `finger_overlap_
    rows` under the socket blocks so the mount lands on solid pixels, and -- when
    cfg.drop_islands is on -- deletes every piece not connected to the socket. Both
    change the geometry that is meshed.

    Descriptors and novelty features are computed on THIS, not on the raw mask, so
    two genomes whose attached geometry is identical but whose floating debris
    differs land in the same archive cell instead of being told apart by material
    that is not simulated."""
    from finger import _socketed                                    # noqa: E402
    mask = np.asarray(mask).astype(np.uint8)
    if not mask.any():
        return mask
    mxy, _, _, _ = _socketed(mask, cfg, *mask.shape)
    return mxy.T[:mask.shape[0]].astype(np.uint8)      # xy -> row-major, finger rows only


def evaluate_masks(masks, cfg=None, device="cuda", dt=DT, close_t=CLOSE_T,
                   pull_t=PULL_T, iter_max=ITER_MAX, w_force=W_FORCE, w_wrap=W_WRAP,
                   verbose=False):
    """Simulate a batch of masks in ONE multi-env solve; return per-design dicts.

    Every design becomes its own independent env in a single flat mesh, so a whole
    QD population is one solver launch. Each dict carries the raw sim metrics, the
    four MAP-Elites descriptors and the composite score.

    A design is screened out BEFORE the solver, and returned invalid with score 0, if
    it has no material left once cfg.drop_islands has removed the pieces that are not
    bolted to the socket, or if it fails the anchor guard (no real material under the
    socket blocks, so it hangs off the forced mounting tongue by a hairline). Without
    the repair step both are common, and both would otherwise mesh and "succeed".
    """
    from pncg2d_solver import implicit_simulate_and_evaluate       # noqa: E402

    if cfg is None:
        cfg, _ = make_cfg()
    masks = [np.asarray(m).astype(np.uint8) for m in masks]
    H, W = masks[0].shape

    eff = [effective_mask(m, cfg) for m in masks]
    live = [i for i, m in enumerate(masks) if _meshable(m, eff[i], cfg)]
    out = [_dead(eff[i]) for i in range(len(masks))]
    if not live:
        return out

    _, mets = implicit_simulate_and_evaluate(
        [masks[i] for i in live], cfg, H, W, device=device, dt=dt,
        close_T=close_t, pull_T=pull_t, pull_dist=cfg.pull_distance,
        iter_max=iter_max, self_collision=True, friction=True, verbose=verbose)

    for j, i in enumerate(live):
        m = dict(mets[j])
        m.update(descriptors(eff[i]))
        m["effective_mask"] = eff[i]
        m["score"] = composite_score(m, cfg, w_force, w_wrap)
        m["force_term"] = w_force * math.tanh(max(m["pull_off"], 0.0) / cfg.f_ref) \
            if m["valid"] and math.isfinite(m["pull_off"]) else 0.0
        m["wrap_term"] = m["score"] - m["force_term"]
        out[i] = m
    return out


def _meshable(mask, eff, cfg):
    """Can this design be simulated as a finger at all?

    (1) something must survive socketing -- non-empty, and with drop_islands on, some
        of it must reach the socket; and
    (2) it must pass the anchor guard: real material under the socket blocks, plus a
        continuous load path from the anchor band down to the contact region that
        survives a 1 px erosion. Applied to the RAW mask, so the mounting tongue that
        socketing forces in cannot be what satisfies the guard."""
    from finger import socket_block_cols, anchor_valid              # noqa: E402
    if not eff.any():
        return False
    if getattr(cfg, "anchor_guard", True) and not anchor_valid(
            mask, socket_block_cols(cfg, mask.shape[1]),
            cfg.anchor_rows, cfg.anchor_min_fill):
        return False
    return True


def _dead(eff):
    """Metric dict for a design that cannot be simulated at all."""
    d = dict(valid=False, pull_off=0.0, arc=0.0, jmin=float("nan"), jmax=float("nan"),
             ncon=0, mean_disp=0.0, score=0.0, force_term=0.0, wrap_term=0.0,
             effective_mask=eff)
    d.update(descriptors(eff))
    d["mat"] = d["material_fraction"]
    return d
