"""Simulation configuration and material scaling.

`SimCfg` holds the rig geometry, the contact/IPC parameters and the material.
`scale_for_material` rescales a config to a different Young's modulus while
preserving the DIMENSIONLESS dynamics -- dt and the phase durations track the
elastic wave speed, so `n_steps` is invariant and a stiffer material costs no
extra runtime. Every force-like quantity (contact stiffnesses, force cap, the
fitness reference f_ref) scales linearly with E.

Extracted from the research code unchanged so the released physics matches the
published runs exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class SimCfg:
    """Physics — matches run_voronoi_batched_sim.py, with the dt fix."""
    finger_width = 0.032
    finger_height = 0.064
    center_x = 0.10
    center_y = 0.10
    # MATERIAL: sim2real-calibrated 2026-07-30 by 5-param Bayesian system-ID
    # against the 1.5x-sphere close_pull rig (rank 5, whole close->pull curve):
    # E=2.05 MPa, mu=0.618, damping ratio x2.806 (folded into REF_DAMPING),
    # d_hat=1.04 mm, epsv=7.19e-4. Fit RMSE 1.10 N, pull-off 5.45 N vs rig 5.89 N.
    # CAVEATS (verification 2026-07-30): the fit is a DEGENERATE RIDGE — damp and
    # d_hat are essentially unidentified and E sits in a shallow basin ~1% above
    # its lower bound; only mu(~0.6) and the pull-off outcome are well-determined.
    # E=2.05 also disagrees with the 28 mm calibration (~2.75); a real material E
    # shouldn't depend on sphere size, so treat these as an operating point, not a
    # measured material. dt/damping/stiffnesses/caps are REF_* scaled to E by
    # scale_for_material. (Prior E=3.8 MPa 28mm-calibration is in git history.)
    young = 2.05218e6
    nu = 0.40                # not 0.49: P1 triangles volumetrically lock near 0.5,
                             # over-stiffening thin members in a design-dependent way
    rho_mass = 1150.0        # = REF_RHO used by scale_for_material (idempotent); the
                             # measured 1.12 g/cm^3 moves forces <1% (rho-independent
                             # quasi-static equilibrium), only shifting the CFL dt.
    damping = 1385.77        # REF_DAMPING(701.58) * s  (s=1.976 P-wave ratio @ 2.05 MPa)
    dt = 2.53137e-6          # REF 5e-6 / s
    input_disp = 0.010
    ramp_fraction = 0.9
    contact_stiffness = 3.90147e5   # REF 1.0e5 * r
    obstacle_stiffness = 7802.95    # REF 2.0e3 * r   (r = 3.901 stiffness ratio)
    ipc_d_hat = 1.04097e-3          # sysid-tuned (weakly identified)
    obstacle_friction = 0.618284    # sysid-tuned (well identified ~0.6)
    obstacle_friction_epsv = 7.18778e-4  # sysid-tuned
    obstacle_ccd_eta = 0.9
    newton_iter = 8
    # object-facing gripper surface = center_x - finger_width/2 = 0.084 m.
    # Circle centre sits 24 mm from that surface (MEASURED on the rig 2026-07-29):
    # object_x = 0.084 - 0.024 = 0.060.  object_y keeps the object at finger height.
    object_x = 0.060
    object_y = 0.080
    object_r = 0.014
    # Validity guards. The IMPLICIT path reads only the two J bounds:
    #   valid = finite pull_off/arc, ncon > 0, arc > 0, buckle_J_min <= J <= J_max_thr
    buckle_J_min = 0.3          # reject if any element inverts below this
    J_max_thr = 5.0             # reject if any element stretches above this
    # The next two guarded the EXPLICIT solver, which this release does not ship.
    # Kept so a SimCfg still round-trips with the research one, but NOT read here.
    arc_explode_factor = 1.25   # (unused) reject if contact arc > circumference * this
    force_cap = 3.90147e4       # (unused as a guard; still rescaled with E) REF 1.0e4 * r
    # Anchor guard: an EXPLICIT-path screen (warp_grasp.py in the research repo) that
    # rejected fingers carrying no material where the socket clamps. The implicit path
    # never ran it, and neither does this release -- validity is the solver's own rule
    # above, so a design that does not reach the object is simply invalid. Kept as
    # inert fields for SimCfg compatibility.
    anchor_rows = 6             # (unused) top finger rows that must be loaded
    anchor_min_fill = 0.20      # (unused) min material fraction in that band
    anchor_guard = True         # (unused)
    # No repair step in this release, so a design can decode into several pieces.
    # False (default) -> simulate the mask EXACTLY as the model produced it, floating
    # pieces included: no repair means no modification of any kind.
    # True -> simulate only the piece attached to the socket (finger.keep_socket_
    # component; removal only, never adds material).
    drop_islands = False
    # phase time = n_steps*dt, shrinking with stiffness so n_steps stays fixed
    n_steps = 20000
    pull_n_steps = 20000
    pull_distance = 30.0e-3
    save_every = 200            # (unused) explicit-path trajectory sampling interval
    broadphase_every = 10       # (unused) the implicit solver rebuilds its own Warp
                                # HashGrid broadphase and never reads this
    f_ref = 390.147
REF_YOUNG = 5.26e5
REF_NU = 0.40
REF_RHO = 1150.0
REF_DT = 5.0e-6
REF_DAMPING = 701.581
REF_CONTACT_STIFFNESS = 1.0e5
REF_OBSTACLE_STIFFNESS = 2.0e3
REF_FORCE_CAP = 1.0e4
REF_F_REF = 100.0
def scale_for_material(cfg, young, rho=None):
    """Rescale a SimCfg to a different material, preserving the DIMENSIONLESS
    dynamics. Returns (cfg, info).

    Time is non-dimensionalised by the elastic wave transit time. With
    c = sqrt(E/rho) and s = c_new/c_ref:

        dt      /= s   CFL — the stable step tracks the wave speed
        phase_T /= s   same quasi-staticity (T*omega held constant)
        => n_steps = T/dt is INVARIANT, so a stiffer material costs NO extra
           runtime (this is why n_steps is not touched below)
        damping *= s   damping is a rate (Dfac = exp(-dt*damping)), so holding
                       the damping ratio zeta fixed means scaling with omega
        contact_stiffness, obstacle_stiffness, force_cap, f_ref *= E_new/E_ref
                       penalty stiffnesses must track the material, and every
                       force (hence the cap and the fitness reference) is linear in E

    Left alone because they are geometric or dimensionless: input_disp,
    pull_distance, ipc_d_hat, nu, friction, the J guards, newton_iter.

    Validated over E = 0.53 / 5 / 23 / 50 MPa: 4/4 valid at every stiffness,
    jmin/jmax identical to 3 decimals (same deformation state), pull-off linear
    in E to <0.2%, runtime flat at ~27 s.

        cfg, _ = scale_for_material(SimCfg(), 23.0e6)   # resin print, 23 MPa
    """
    import math as _math

    def _pwave(E, nu_, rho_):
        """Dilatational (P-wave) speed sqrt((lam+2mu)/rho) — the quantity that
        actually sets the CFL limit. Using E instead is wrong once nu moves: at
        nu=0.49 lam is ~16x larger than at 0.40, so an E-based dt is ~2.8x too
        big and the solve goes unstable."""
        mu_ = E / (2.0 * (1.0 + nu_))
        lam_ = E * nu_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_))
        return _math.sqrt((lam_ + 2.0 * mu_) / rho_)

    rho = REF_RHO if rho is None else rho
    nu = getattr(cfg, "nu", REF_NU)
    s = _pwave(young, nu, rho) / _pwave(REF_YOUNG, REF_NU, REF_RHO)
    r = young / REF_YOUNG                                    # stiffness ratio
    cfg.young = young
    cfg.rho_mass = rho
    cfg.dt = REF_DT / s
    cfg.damping = REF_DAMPING * s
    cfg.contact_stiffness = REF_CONTACT_STIFFNESS * r
    cfg.obstacle_stiffness = REF_OBSTACLE_STIFFNESS * r
    cfg.force_cap = REF_FORCE_CAP * r
    cfg.f_ref = REF_F_REF * r
    # n_steps / pull_n_steps deliberately untouched — see docstring
    return cfg, {"wave_ratio": s, "stiffness_ratio": r,
                 "phase_time": cfg.n_steps * cfg.dt}