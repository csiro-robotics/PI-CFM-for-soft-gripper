"""2D multi-env PNCG-IPC solver (Warp) — implicit backward Euler for the grasp sim.

Replaces the CFL-pinned explicit IMEX path. At the rig's real rates (1.1 s close,
9.9 mm/s pull) explicit needs ~2.2M steps per design, which makes MAP-Elites at
experimental conditions impossible. Backward Euler is unconditionally stable, so
dt is set by accuracy: the same 4.1 s becomes ~4000 steps at dt = 1 ms.

Each step minimises the incremental potential (IPC, Li 2020):

    E(x) = 1/2 (x - xhat)^T M (x - xhat) + h^2 [ Psi(x) + kappa sum_k b(d_k) ]
    xhat = x^t + h v^t exp(-h*damping)                 (same drag law as explicit)
    v^{t+1} = (x^{t+1} - x^t) / h

with preconditioned nonlinear CG (Dai-Kou + Jacobi), after
Warp_pncg_ipc_csiro / PNCG_IPC (SIGGRAPH'24). No linear system is ever assembled,
which is what makes batching many small designs cheap.

MULTI-ENV. Every scalar the algorithm reduces is PER-ENV, shape (n_envs,) and in
float64. The reference is single-env and reduces into index 0; ported literally
that silently couples designs — a global alpha lets the most-constrained design
throttle the whole population, a global beta shares one CG direction, and the
result still runs and still returns finite pull_off for every design. There is
deliberately not one `wp.atomic_*(..., 0, ...)` in this file.

float64 for the accumulators is not paranoia: order-dependent float32 summation
over ~10k nodes makes two envs holding the SAME design return different numbers,
which is indistinguishable from a coupling bug during the isolation test.

SCOPE: disc contact + inversion/CCD step caps + self-collision + friction. All
four are implemented and ON by default in this release (evaluate.py passes
self_collision=True, friction=True).

Friction was ORIGINALLY written as a gradient-only term, on the reasoning that in
2D its Hessian vanishes in the sliding regime. That was wrong in practice: all of
its curvature sits in a kink of half-width a = epsv*dt, and init_step sets xn = x,
so every free node starts every timestep sitting on the kink. Omitting the
curvature made the line search inconsistent with the gradient and produced force
chatter. grad_friction_disc_kernel now also writes diagH, and
pHp_friction_disc_kernel makes alpha = -gTp/pHp self-consistent. Purely numerical:
the converged solution is unchanged.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import warp as wp

# this package is a flat module set: make sibling modules importable whether the
# solver is used through evolution/evaluate.py or on its own.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from pncg2d_math import (vec6, ds_from_nodes, dF_from_dir, dF_from_vec6,      # noqa: E402
                         nh_dPsidF, nh_d2PsidF2_dir, nh_pHp_elem, cross2,
                         barrier_b, barrier_bp, barrier_bpp)

f64 = wp.float64
BIG = 1.0e30


# ================================================================ step setup
@wp.kernel
def init_step_kernel(x: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
                     xn: wp.array(dtype=wp.vec2), xhat: wp.array(dtype=wp.vec2),
                     dt: float, drag: float):
    """Snapshot x^t and form the inertial predictor. MUST run BEFORE the BC is
    applied: the reference samples xn after its demo teleports nodes
    (deformer_pncg.py:1012), which would make our driven ramp — the actual load —
    invisible to both v=(x-xn)/dt and xhat."""
    i = wp.tid()
    xn[i] = x[i]
    xhat[i] = x[i] + dt * drag * v[i]


@wp.kernel
def apply_bc_kernel(x: wp.array(dtype=wp.vec2), xhat: wp.array(dtype=wp.vec2),
                    bc_idx: wp.array(dtype=wp.int32), bc_target: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    n = bc_idx[b]
    x[n] = bc_target[b]
    xhat[n] = bc_target[b]           # so the inertia term does not fight the BC


@wp.kernel
def reset_step_kernel(dE0: wp.array(dtype=f64), active: wp.array(dtype=wp.int32),
                      iters: wp.array(dtype=wp.int32)):
    e = wp.tid()
    dE0[e] = f64(0.0)
    active[e] = 1
    iters[e] = 0


# ================================================================ gradient + Jacobi diagonal
@wp.kernel
def grad_diag_mass_kernel(x: wp.array(dtype=wp.vec2), xhat: wp.array(dtype=wp.vec2),
                          mass: wp.array(dtype=float), free: wp.array(dtype=wp.vec2),
                          grad: wp.array(dtype=wp.vec2), grad_prev: wp.array(dtype=wp.vec2),
                          diagH: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    grad_prev[i] = grad[i]
    m = mass[i]
    d = x[i] - xhat[i]
    grad[i] = wp.cw_mul(wp.vec2(m * d[0], m * d[1]), free[i])
    diagH[i] = wp.vec2(m, m)


@wp.kernel
def grad_diag_elastic_kernel(x: wp.array(dtype=wp.vec2), tri: wp.array(dtype=wp.vec3i),
                             B: wp.array(dtype=wp.mat22), area: wp.array(dtype=float),
                             mu: wp.array(dtype=float), lam: wp.array(dtype=float),
                             h2: float, clamp: int, grad: wp.array(dtype=wp.vec2),
                             diagH: wp.array(dtype=wp.vec2)):
    t = wp.tid()
    ids = tri[t]
    i0 = ids[0]; i1 = ids[1]; i2 = ids[2]
    p0 = x[i0]; p1 = x[i1]; p2 = x[i2]
    Bt = B[t]
    F = ds_from_nodes(p0, p1, p2) * Bt
    A = area[t]; m_ = mu[t]; l_ = lam[t]

    # gradient: dE/dx = +A * P * B^T applied to the edge stencil. This is exactly
    # MINUS warp_fem.elastic_force_kernel's force, which is the port's Layer-1 check.
    P = nh_dPsidF(F, m_, l_)
    HH = A * (P * wp.transpose(Bt))
    g1 = wp.vec2(HH[0, 0], HH[1, 0])
    g2 = wp.vec2(HH[0, 1], HH[1, 1])
    g0 = -(g1 + g2)
    wp.atomic_add(grad, i0, h2 * g0)
    wp.atomic_add(grad, i1, h2 * g1)
    wp.atomic_add(grad, i2, h2 * g2)

    # Jacobi diagonal: six directional second derivatives, one per element DOF.
    for n in range(3):
        dx = float(0.0); dy = float(0.0)
        dFx = dF_from_dir(n, 0, Bt)
        dx = A * wp.ddot(dFx, nh_d2PsidF2_dir(F, dFx, m_, l_))
        dFy = dF_from_dir(n, 1, Bt)
        dy = A * wp.ddot(dFy, nh_d2PsidF2_dir(F, dFy, m_, l_))
        # clamp at 0: a negative diagonal would flip the preconditioner's sign
        # (reference does the same, deformer_pncg.py:211-213). `clamp` is 0 only
        # in the self-test, where the raw value is compared against autograd.
        if clamp != 0:
            dx = wp.max(dx, 0.0)
            dy = wp.max(dy, 0.0)
        contrib = wp.vec2(h2 * dx, h2 * dy)
        if n == 0:
            wp.atomic_add(diagH, i0, contrib)
        elif n == 1:
            wp.atomic_add(diagH, i1, contrib)
        else:
            wp.atomic_add(diagH, i2, contrib)


@wp.kernel
def grad_diag_disc_kernel(x: wp.array(dtype=wp.vec2), bnd: wp.array(dtype=wp.int32),
                          c: wp.vec2, radius: float, dhat: float, kappa: float, h2: float,
                          clamp: int, grad: wp.array(dtype=wp.vec2),
                          diagH: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    n = bnd[b]
    diff = x[n] - c
    dc = wp.length(diff)
    d = dc - radius
    if d <= 0.0 or d >= dhat:
        return                                    # d<=0 would be log of a negative
    nh = diff / dc
    bp = barrier_bp(d, dhat)
    bpp = barrier_bpp(d, dhat)
    wp.atomic_add(grad, n, h2 * kappa * bp * nh)
    # H = kappa[ b'' n n^T + (b'/dc)(I - n n^T) ].  NOTE the tangential term is
    # divided by dc = ||x-c||, NOT by d. The 3D reference divides by d because its
    # primitives satisfy ||t|| == d; for a circle ||x-c|| = d + radius, and using
    # 1/d inflates this term by (d+r)/d (measured 173x at d = 0.29 dhat).
    tang = bp / dc
    dgx = kappa * (bpp * nh[0] * nh[0] + tang * (1.0 - nh[0] * nh[0]))
    dgy = kappa * (bpp * nh[1] * nh[1] + tang * (1.0 - nh[1] * nh[1]))
    if clamp != 0:
        dgx = wp.max(dgx, 0.0)
        dgy = wp.max(dgy, 0.0)
    wp.atomic_add(diagH, n, wp.vec2(h2 * dgx, h2 * dgy))


@wp.kernel
def mask_grad_kernel(grad: wp.array(dtype=wp.vec2), free: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    grad[i] = wp.cw_mul(grad[i], free[i])


# ================================================================ Dai-Kou direction
@wp.kernel
def dk_reduce_kernel(grad: wp.array(dtype=wp.vec2), grad_prev: wp.array(dtype=wp.vec2),
                     p: wp.array(dtype=wp.vec2), diagH: wp.array(dtype=wp.vec2),
                     free: wp.array(dtype=wp.vec2), node2env: wp.array(dtype=wp.int32),
                     gPg: wp.array(dtype=f64), g_p: wp.array(dtype=f64),
                     g_Py: wp.array(dtype=f64), y_p: wp.array(dtype=f64),
                     y_Py: wp.array(dtype=f64)):
    """Five per-env dot products for beta^DK. Every one segmented by node2env —
    the reference reduces all of them into index 0 (deformer_pncg.py:636-639)."""
    i = wp.tid()
    e = node2env[i]
    dh = diagH[i]
    fr = free[i]
    # P = free / diagH, floored: mesh.py clips lumped mass to 1e-30, so an
    # orphan node would otherwise give P ~ 1e30 and blow up its whole env.
    Px = fr[0] / wp.max(dh[0], 1.0e-12)
    Py_ = fr[1] / wp.max(dh[1], 1.0e-12)
    g = grad[i]
    pp = p[i]
    yy = g - grad_prev[i]
    Pg0 = Px * g[0]; Pg1 = Py_ * g[1]
    Py0 = Px * yy[0]; Py1 = Py_ * yy[1]
    wp.atomic_add(gPg, e, f64(g[0] * Pg0 + g[1] * Pg1))
    wp.atomic_add(g_p, e, f64(g[0] * pp[0] + g[1] * pp[1]))
    wp.atomic_add(g_Py, e, f64(g[0] * Py0 + g[1] * Py1))
    wp.atomic_add(y_p, e, f64(yy[0] * pp[0] + yy[1] * pp[1]))
    wp.atomic_add(y_Py, e, f64(yy[0] * Py0 + yy[1] * Py1))


@wp.kernel
def finalize_beta_kernel(gPg: wp.array(dtype=f64), g_p: wp.array(dtype=f64),
                         g_Py: wp.array(dtype=f64), y_p: wp.array(dtype=f64),
                         y_Py: wp.array(dtype=f64), it: int,
                         beta: wp.array(dtype=f64), gTp: wp.array(dtype=f64)):
    """beta^DK per env, with the guards the reference lacks (it divides by y_p
    twice with no check, deformer_pncg.py:642, and has no ascent restart).

    gTp is computed ANALYTICALLY here rather than reduced again:
        p_new = -P g + beta p_old  =>  g.p_new = -g.Pg + beta (g.p_old)
    """
    e = wp.tid()
    b = f64(0.0)
    if it > 0:
        ytp = y_p[e]
        if wp.abs(ytp) > f64(1.0e-30):
            b = g_Py[e] / ytp - y_Py[e] * g_p[e] / (ytp * ytp)
    t = -gPg[e] + b * g_p[e]
    if t >= f64(0.0):                    # not a descent direction -> steepest descent
        b = f64(0.0)
        t = -gPg[e]
    beta[e] = b
    gTp[e] = t


@wp.kernel
def update_p_kernel(grad: wp.array(dtype=wp.vec2), diagH: wp.array(dtype=wp.vec2),
                    free: wp.array(dtype=wp.vec2), node2env: wp.array(dtype=wp.int32),
                    beta: wp.array(dtype=f64), p: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    e = node2env[i]
    dh = diagH[i]; fr = free[i]; g = grad[i]
    bb = float(beta[e])
    px = -fr[0] / wp.max(dh[0], 1.0e-12) * g[0] + bb * p[i][0]
    py = -fr[1] / wp.max(dh[1], 1.0e-12) * g[1] + bb * p[i][1]
    p[i] = wp.cw_mul(wp.vec2(px, py), fr)


# ================================================================ curvature p^T H p
@wp.kernel
def pHp_reset_kernel(pHp: wp.array(dtype=f64), alpha: wp.array(dtype=f64),
                     p_inf: wp.array(dtype=f64), clr_min: wp.array(dtype=f64),
                     alpha_max: float):
    e = wp.tid()
    pHp[e] = f64(0.0)
    alpha[e] = f64(alpha_max)
    p_inf[e] = f64(0.0)
    clr_min[e] = f64(BIG)


@wp.kernel
def pHp_mass_kernel(p: wp.array(dtype=wp.vec2), mass: wp.array(dtype=float),
                    node2env: wp.array(dtype=wp.int32),
                    pHp: wp.array(dtype=f64), p_inf: wp.array(dtype=f64)):
    i = wp.tid()
    e = node2env[i]
    pp = p[i]
    wp.atomic_add(pHp, e, f64(mass[i] * (pp[0] * pp[0] + pp[1] * pp[1])))
    # per-node L2, not the inf-norm: the dhat/2 cap bounds nodal DISPLACEMENT,
    # and an inf-norm under-bounds it by up to sqrt(2).
    wp.atomic_max(p_inf, e, f64(wp.length(pp)))


@wp.kernel
def pHp_elastic_kernel(x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                       tri: wp.array(dtype=wp.vec3i), B: wp.array(dtype=wp.mat22),
                       area: wp.array(dtype=float), mu: wp.array(dtype=float),
                       lam: wp.array(dtype=float), h2: float, clamp: int,
                       tri2env: wp.array(dtype=wp.int32), pHp: wp.array(dtype=f64)):
    t = wp.tid()
    ids = tri[t]
    i0 = ids[0]; i1 = ids[1]; i2 = ids[2]
    Bt = B[t]
    F = ds_from_nodes(x[i0], x[i1], x[i2]) * Bt
    pe = vec6(p[i0][0], p[i0][1], p[i1][0], p[i1][1], p[i2][0], p[i2][1])
    val = area[t] * nh_pHp_elem(F, Bt, pe, mu[t], lam[t])
    if clamp != 0:
        val = wp.max(val, 0.0)
    wp.atomic_add(pHp, tri2env[t], f64(h2 * val))


@wp.kernel
def pHp_disc_kernel(x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                    bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                    c: wp.vec2, radius: float, dhat: float, kappa: float, h2: float,
                    clamp: int, pHp: wp.array(dtype=f64), clr_min: wp.array(dtype=f64)):
    b = wp.tid()
    n = bnd[b]
    e = b2e[b]
    diff = x[n] - c
    dc = wp.length(diff)
    d = dc - radius
    wp.atomic_min(clr_min, e, f64(d))
    if d <= 0.0 or d >= dhat:
        return
    nh = diff / dc
    pp = p[n]
    pn = wp.dot(pp, nh)
    pt = pp - pn * nh
    bp = barrier_bp(d, dhat)
    bpp = barrier_bpp(d, dhat)
    val = kappa * (bpp * pn * pn + (bp / dc) * wp.dot(pt, pt))
    if clamp != 0:
        val = wp.max(val, 0.0)
    wp.atomic_add(pHp, e, f64(h2 * val))


# ================================================================ step-size caps
@wp.kernel
def inversion_alpha_kernel(x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                           tri: wp.array(dtype=wp.vec3i), tri2env: wp.array(dtype=wp.int32),
                           shrink: float, alpha: wp.array(dtype=f64)):
    """Largest step before a triangle's signed area drops below `shrink` of its
    current value. In 2D area(alpha) is exactly quadratic in alpha, so this is a
    closed-form root — no iteration, unlike the 3D cubic."""
    t = wp.tid()
    ids = tri[t]
    i0 = ids[0]; i1 = ids[1]; i2 = ids[2]
    e1 = x[ids[1]] - x[ids[0]]
    e2 = x[ids[2]] - x[ids[0]]
    q1 = p[ids[1]] - p[ids[0]]
    q2 = p[ids[2]] - p[ids[0]]
    a0 = cross2(e1, e2)                       # 2x signed area now
    a1 = cross2(e1, q2) + cross2(q1, e2)
    a2 = cross2(q1, q2)
    target = shrink * a0
    # solve a2 s^2 + a1 s + (a0 - target) = 0 for the smallest positive root
    cc = a0 - target
    if wp.abs(a2) < 1.0e-24:
        if wp.abs(a1) > 1.0e-24:
            s = -cc / a1
            if s > 0.0:
                wp.atomic_min(alpha, tri2env[t], f64(s))
        return
    disc = a1 * a1 - 4.0 * a2 * cc
    if disc < 0.0:
        return
    sq = wp.sqrt(disc)
    r1 = (-a1 - sq) / (2.0 * a2)
    r2 = (-a1 + sq) / (2.0 * a2)
    s = float(BIG)
    if r1 > 0.0:
        s = wp.min(s, r1)
    if r2 > 0.0:
        s = wp.min(s, r2)
    if s < BIG:
        wp.atomic_min(alpha, tri2env[t], f64(s))


@wp.kernel
def disc_ccd_kernel(x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                    bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                    c: wp.vec2, radius: float, eta: float, alpha: wp.array(dtype=f64)):
    """Exact time of impact against a FIXED circle: ||x + s p - c|| = radius is a
    quadratic in s. No broadphase, no conservative bound — the obstacle is analytic."""
    b = wp.tid()
    n = bnd[b]
    o = x[n] - c
    d = p[n]
    a = wp.dot(d, d)
    if a < 1.0e-24:
        return
    bq = 2.0 * wp.dot(o, d)
    cq = wp.dot(o, o) - radius * radius
    disc = bq * bq - 4.0 * a * cq
    if disc < 0.0:
        return
    sq = wp.sqrt(disc)
    r1 = (-bq - sq) / (2.0 * a)
    r2 = (-bq + sq) / (2.0 * a)
    s = float(BIG)
    if r1 > 0.0:
        s = wp.min(s, r1)
    if r2 > 0.0:
        s = wp.min(s, r2)
    if s < BIG:
        wp.atomic_min(alpha, b2e[b], f64(eta * s))


@wp.kernel
def alpha_finalize_kernel(gTp: wp.array(dtype=f64), pHp: wp.array(dtype=f64),
                          p_inf: wp.array(dtype=f64), clr_min: wp.array(dtype=f64),
                          alpha: wp.array(dtype=f64), dE: wp.array(dtype=f64),
                          dE0: wp.array(dtype=f64), active: wp.array(dtype=wp.int32),
                          iters: wp.array(dtype=wp.int32),
                          dhat: float, near_factor: float, eps: float, tol_x: float, it: int):
    """NB two convergence tests (either can stop an env):

      ENERGY-DECREASE RATIO  d < eps*dE0   -- the original. `d` is built from gTp/pHp, which are
        atomic SUMS over all elements, so float32 error accumulates as sqrt(N)*1.2e-7 ~ 1.6e-5 at
        N=18k. Tolerances below that floor are UNREACHABLE: measured 0/810 steps converged at
        eps=1e-8 AND 1e-6, on BOTH the full-grid and the body-fitted mesh, at iter_max 320 and 1000.
        eps=1e-5 is the first that works (456/810 steps) -- right at the predicted floor.

      INCREMENT INF-NORM     alpha*p_inf < tol_x  -- this is what IPC uses (scaled by dt it is a
        velocity tolerance). p_inf is an atomic MAX, so it does NOT accumulate error with N: the
        sqrt(N) floor above simply does not apply. It also has physical units (metres of node
        motion), so the tolerance means something you can reason about.

    tol_x = 0.0 DISABLES the increment test and reproduces the original behaviour exactly."""
    e = wp.tid()
    if active[e] == 0:
        return
    ph = pHp[e]
    a = alpha[e]                                   # already min(inversion, CCD)
    if ph > f64(0.0):
        a = wp.min(a, -gTp[e] / ph)                # Newton estimate (paper eq 11)
    # dhat/2 displacement cap — ONLY near contact. Ungated it throttles the 12 mm
    # of free flight before the finger reaches the disc to 0.5 mm per iteration.
    if clr_min[e] < f64(near_factor * dhat) and p_inf[e] > f64(0.0):
        a = wp.min(a, f64(0.5 * dhat) / p_inf[e])
    if a <= f64(0.0):
        active[e] = 0
        alpha[e] = f64(0.0)
        return
    alpha[e] = a
    # INCREMENT INF-NORM test (IPC-style): the largest node displacement this iteration would make.
    # Immune to the sqrt(N) summation floor that caps the energy test -- see the docstring.
    if tol_x > 0.0 and a * p_inf[e] < f64(tol_x):
        active[e] = 0
        iters[e] = iters[e] + 1
        return
    d = -a * gTp[e] - f64(0.5) * a * a * ph        # quadratic model of the decrease
    dE[e] = d
    if it == 0:
        dE0[e] = wp.max(d, f64(1.0e-300))
    elif d < f64(eps) * dE0[e]:
        active[e] = 0
    iters[e] = iters[e] + 1


@wp.kernel
def update_x_kernel(x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                    node2env: wp.array(dtype=wp.int32), alpha: wp.array(dtype=f64),
                    active: wp.array(dtype=wp.int32)):
    i = wp.tid()
    e = node2env[i]
    if active[e] == 0:
        return
    x[i] = x[i] + float(alpha[e]) * p[i]


@wp.kernel
def commit_v_kernel(x: wp.array(dtype=wp.vec2), xn: wp.array(dtype=wp.vec2),
                    v: wp.array(dtype=wp.vec2), inv_dt: float):
    i = wp.tid()
    v[i] = (x[i] - xn[i]) * inv_dt


# ================================================================ diagnostics
@wp.kernel
def energy_kernel(x: wp.array(dtype=wp.vec2), xhat: wp.array(dtype=wp.vec2),
                  mass: wp.array(dtype=float), node2env: wp.array(dtype=wp.int32),
                  E: wp.array(dtype=f64)):
    i = wp.tid()
    d = x[i] - xhat[i]
    wp.atomic_add(E, node2env[i], f64(0.5 * mass[i] * (d[0] * d[0] + d[1] * d[1])))


@wp.kernel
def energy_elastic_kernel(x: wp.array(dtype=wp.vec2), tri: wp.array(dtype=wp.vec3i),
                          B: wp.array(dtype=wp.mat22), area: wp.array(dtype=float),
                          mu: wp.array(dtype=float), lam: wp.array(dtype=float),
                          h2: float, tri2env: wp.array(dtype=wp.int32),
                          E: wp.array(dtype=f64)):
    t = wp.tid()
    ids = tri[t]
    Bt = B[t]
    F = ds_from_nodes(x[ids[0]], x[ids[1]], x[ids[2]]) * Bt
    J = wp.determinant(F)
    logJ = wp.log(wp.max(J, 1.0e-6))
    I2 = wp.ddot(F, F)
    psi = 0.5 * mu[t] * (I2 - 2.0) - mu[t] * logJ + 0.5 * lam[t] * logJ * logJ
    wp.atomic_add(E, tri2env[t], f64(h2 * area[t] * psi))


@wp.kernel
def energy_disc_kernel(x: wp.array(dtype=wp.vec2), bnd: wp.array(dtype=wp.int32),
                       b2e: wp.array(dtype=wp.int32), c: wp.vec2, radius: float,
                       dhat: float, kappa: float, h2: float, E: wp.array(dtype=f64)):
    b = wp.tid()
    diff = x[bnd[b]] - c
    d = wp.length(diff) - radius
    if d <= 0.0 or d >= dhat:
        return
    wp.atomic_add(E, b2e[b], f64(h2 * kappa * barrier_b(d, dhat)))


@wp.kernel
def energy_friction_disc_kernel(x: wp.array(dtype=wp.vec2), xn: wp.array(dtype=wp.vec2),
                                bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                                c: wp.vec2, radius: float, dhat: float, kappa: float,
                                mu: float, epsv: float, inv_dt: float, h2: float,
                                free: wp.array(dtype=wp.vec2), E: wp.array(dtype=f64)):
    """Lagged (semi-implicit) friction DISSIPATION POTENTIAL: D = h2*mu*f_n*f0(|u_t|),
    where u_t is the tangential slip since xn and f0(y)=y^2/(2a) (y<a) else y-a/2,
    a=epsv/inv_dt. This is the EXACT potential whose gradient is
    grad_friction_disc_kernel's force (dD/dx = h2*mu*f_n*tdir), so adding it to the
    energy lets the line-search gate friction-driven steps at the calibrated epsv."""
    b = wp.tid()
    n = bnd[b]
    fr = free[n]
    if fr[0] <= 0.5 and fr[1] <= 0.5:
        return
    diff = x[n] - c
    dc = wp.length(diff)
    d = dc - radius
    if d <= 0.0 or d >= dhat:
        return
    nh = diff / dc
    f_n = kappa * wp.abs(barrier_bp(d, dhat))             # lagged normal force magnitude
    v = (x[n] - xn[n]) * inv_dt
    vt = v - wp.dot(v, nh) * nh                           # tangential velocity
    ut = wp.length(vt) / inv_dt                           # |tangential slip|
    a = epsv / inv_dt
    f0 = ut - a * 0.5
    if ut < a:
        f0 = ut * ut / (2.0 * a)
    wp.atomic_add(E, b2e[b], f64(h2 * mu * f_n * f0))


@wp.kernel
def disc_force_kernel(x: wp.array(dtype=wp.vec2), bnd: wp.array(dtype=wp.int32),
                      b2e: wp.array(dtype=wp.int32), c: wp.vec2, radius: float,
                      dhat: float, kappa: float, F: wp.array(dtype=wp.vec2)):
    """Reaction ON the disc — same definition as metrics.obstacle_force
    (normal/barrier only; friction is not yet in this sum)."""
    b = wp.tid()
    diff = x[bnd[b]] - c
    dc = wp.length(diff)
    d = dc - radius
    if d <= 0.0 or d >= dhat:
        return
    nh = diff / dc
    # force on node = -kappa b'(d) n ; on disc = +kappa b'(d) n
    wp.atomic_add(F, b2e[b], kappa * barrier_bp(d, dhat) * nh)


# ================================================================ self-collision (node-node)
# Quadratic penalty E = 1/2 k (r - d)^2 for boundary node pairs with d = |xi-xj| < r
# (skip mesh-adjacent pairs via excl). This is exactly warp_fem/contact.py's
# node-node penalty, so it "ports with zero physics change" (pncg2d.py scope note),
# but here it enters the incremental potential: grad = dE/dx, and its clamped
# Hessian feeds the Jacobi preconditioner (diagH) and the curvature p^T H p.
#
#   grad_i = -k (r-d) n           (force_i = -grad_i = +k(r-d) n, matches the kernel)
#   H_ii   =  k n n^T - k (r-d)/d (I - n n^T)      [indefinite -> clamp tangential]
# A pair is visited from both endpoints (tid=i loops j, tid=j loops i): the node
# diagonal is added once from each node's own tid (no double count); p^T H p uses
# the relative direction dp=pi-pj and is the SAME from either endpoint, so 0.5x it.
@wp.kernel
def grad_diag_self_kernel(x: wp.array(dtype=wp.vec2), boundary: wp.array(dtype=wp.int32),
                          boundary2env: wp.array(dtype=wp.int32),
                          boundary_start: wp.array(dtype=wp.int32),
                          boundary_per_env: wp.array(dtype=wp.int32),
                          excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                          radius: float, stiffness: float, h2: float, clamp: int,
                          grad: wp.array(dtype=wp.vec2), diagH: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    g = wp.vec2(0.0, 0.0)
    dgx = float(0.0); dgy = float(0.0)
    for jj in range(n_be):
        if jj != i and excl_flat[eoff + i * n_be + jj] == wp.uint8(0):
            diff = pa - x[boundary[start + jj]]
            dd = wp.sqrt(wp.dot(diff, diff) + 1.0e-12)
            if dd < radius:
                nh = diff / dd
                rd = radius - dd                     # > 0 in contact
                g = g - stiffness * rd * nh          # grad_i = -k(r-d) n
                tang = stiffness * rd / dd
                hx = stiffness * nh[0] * nh[0] - tang * (1.0 - nh[0] * nh[0])
                hy = stiffness * nh[1] * nh[1] - tang * (1.0 - nh[1] * nh[1])
                if clamp != 0:
                    hx = wp.max(hx, 0.0); hy = wp.max(hy, 0.0)
                dgx = dgx + hx; dgy = dgy + hy
    wp.atomic_add(grad, node, h2 * g)
    wp.atomic_add(diagH, node, wp.vec2(h2 * dgx, h2 * dgy))


@wp.kernel
def pHp_self_kernel(x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                    boundary: wp.array(dtype=wp.int32), boundary2env: wp.array(dtype=wp.int32),
                    boundary_start: wp.array(dtype=wp.int32),
                    boundary_per_env: wp.array(dtype=wp.int32),
                    excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                    radius: float, stiffness: float, h2: float, clamp: int,
                    pHp: wp.array(dtype=f64)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]; pp = p[node]
    val = float(0.0)
    for jj in range(n_be):
        if jj != i and excl_flat[eoff + i * n_be + jj] == wp.uint8(0):
            diff = pa - x[boundary[start + jj]]
            dd = wp.sqrt(wp.dot(diff, diff) + 1.0e-12)
            if dd < radius:
                nh = diff / dd
                rd = radius - dd
                dp = pp - p[boundary[start + jj]]     # relative search direction
                pn = wp.dot(dp, nh)
                pt = dp - pn * nh
                v = stiffness * pn * pn - stiffness * rd / dd * wp.dot(pt, pt)
                if clamp != 0:
                    v = wp.max(v, 0.0)
                val = val + 0.5 * v                   # each pair counted from both ends
    wp.atomic_add(pHp, e, f64(h2 * val))


@wp.kernel
def energy_self_kernel(x: wp.array(dtype=wp.vec2), boundary: wp.array(dtype=wp.int32),
                       boundary2env: wp.array(dtype=wp.int32),
                       boundary_start: wp.array(dtype=wp.int32),
                       boundary_per_env: wp.array(dtype=wp.int32),
                       excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                       radius: float, stiffness: float, h2: float, E: wp.array(dtype=f64)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    val = float(0.0)
    for jj in range(n_be):
        if jj != i and excl_flat[eoff + i * n_be + jj] == wp.uint8(0):
            diff = pa - x[boundary[start + jj]]
            dd = wp.sqrt(wp.dot(diff, diff) + 1.0e-12)
            if dd < radius:
                rd = radius - dd
                val = val + 0.5 * (0.5 * stiffness * rd * rd)   # 0.5 pair double-count
    wp.atomic_add(E, e, f64(h2 * val))


# ================================================================ friction (disc, gradient-only)
# Kinetic Coulomb friction on contacting boundary nodes, opposing the tangential
# relative slide of the incremental step. Force = -mu f_n tdir; grad += +mu f_n tdir.
#   f_n  = kappa |b'(d)|          (barrier normal force, same as disc_force)
#   v    = (x - xn)/dt ; v_t = v - (v.n) n ; tdir = v_t / max(|v_t|, epsv)  (smoothed)
#
# CURVATURE (added 2026-08; was previously a gradient-ONLY term -- see below).
# The old scope note claimed the Hessian vanishes "in the sliding regime (all our
# contacts slide, 0.3-0.8 m/s >> epsv)". That is true ONLY in the slip branch: in 2D
# the tangent space is 1-D, so d(tdir)/dv = 0 once |v_t| > epsv. But it means 100% of
# friction's curvature is concentrated in a KINK of half-width a = epsv*dt, and
# init_step_kernel sets xn = x, so EVERY free node starts EVERY step with u_t = 0 --
# sitting exactly on the kink. At the real rig speed (9.9 mm/s, 30-80x slower than the
# 0.3-0.8 m/s the note assumed) one friction gradient application moves a contact node
# ~3.5 um, ~10x the kink half-width (0.36 um at dt=0.5ms). With no curvature in diagH
# (preconditioner) or pHp (step length), the tangential DOF overshoots the kink every
# iteration, tdir flips sign, and the reported friction jumps by the full +-mu f_n
# quantum -> the per-step force chatter, and a residual the CG can never annihilate
# (predicted 2.2e-5 vs measured 2.08e-5 terminal ||grad||).
# In the kink the potential is quadratic with tangential stiffness h2*mu*f_n/a, i.e.
#   kt = h2*mu*f_n*inv_dt / max(|v_t|, epsv)
# which is EXACT inside the kink and decays correctly into the sliding branch. SPD, so
# it is a valid Gauss-Newton bound. This changes ONLY the model Hessian used to pick
# the direction and step length -- the converged solution of the incremental potential
# is unchanged, so no calibrated material parameter is affected.
@wp.kernel
def grad_friction_disc_kernel(x: wp.array(dtype=wp.vec2), xn: wp.array(dtype=wp.vec2),
                              bnd: wp.array(dtype=wp.int32), c: wp.vec2, radius: float,
                              dhat: float, kappa: float, mu: float, epsv: float,
                              inv_dt: float, h2: float, free: wp.array(dtype=wp.vec2),
                              grad: wp.array(dtype=wp.vec2),
                              diagH: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    n = bnd[b]
    fr = free[n]
    if fr[0] <= 0.5 and fr[1] <= 0.5:
        return
    diff = x[n] - c
    dc = wp.length(diff)
    d = dc - radius
    if d <= 0.0 or d >= dhat:
        return
    nh = diff / dc
    f_n = kappa * wp.abs(barrier_bp(d, dhat))         # normal force magnitude
    v = (x[n] - xn[n]) * inv_dt                        # incremental relative velocity
    vn = wp.dot(v, nh)
    vt = v - vn * nh
    vtmag = wp.length(vt)
    tdir = vt / wp.max(vtmag, epsv)                    # smoothed sliding direction
    fg = wp.cw_mul(h2 * mu * f_n * tdir, fr)           # grad += mu f_n tdir (opposes slide)
    wp.atomic_add(grad, n, fg)
    # kink curvature -> Jacobi preconditioner (tangential direction only)
    kt = h2 * mu * f_n * inv_dt / wp.max(vtmag, epsv)
    th = wp.vec2(-nh[1], nh[0])                        # unit tangent (2D)
    wp.atomic_add(diagH, n, wp.cw_mul(wp.vec2(kt * th[0] * th[0],
                                              kt * th[1] * th[1]), fr))


@wp.kernel
def pHp_friction_disc_kernel(x: wp.array(dtype=wp.vec2), xn: wp.array(dtype=wp.vec2),
                             bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                             p: wp.array(dtype=wp.vec2), c: wp.vec2, radius: float,
                             dhat: float, kappa: float, mu: float, epsv: float,
                             inv_dt: float, h2: float, free: wp.array(dtype=wp.vec2),
                             pHp: wp.array(dtype=f64)):
    """Friction's contribution to p^T H p, so the step length alpha = -gTp/pHp is
    self-consistent: gTp is built from `grad` which INCLUDES friction, so pHp must too.
    Same gating and same kt as grad_friction_disc_kernel."""
    b = wp.tid()
    n = bnd[b]
    fr = free[n]
    if fr[0] <= 0.5 and fr[1] <= 0.5:
        return
    diff = x[n] - c
    dc = wp.length(diff)
    d = dc - radius
    if d <= 0.0 or d >= dhat:
        return
    nh = diff / dc
    f_n = kappa * wp.abs(barrier_bp(d, dhat))
    v = (x[n] - xn[n]) * inv_dt
    vt = v - wp.dot(v, nh) * nh
    kt = h2 * mu * f_n * inv_dt / wp.max(wp.length(vt), epsv)
    th = wp.vec2(-nh[1], nh[0])
    pt = wp.dot(wp.cw_mul(p[n], fr), th)
    wp.atomic_add(pHp, b2e[b], f64(kt * pt * pt))


@wp.kernel
def disc_friction_force_kernel(x: wp.array(dtype=wp.vec2), xn: wp.array(dtype=wp.vec2),
                               bnd: wp.array(dtype=wp.int32), b2e: wp.array(dtype=wp.int32),
                               c: wp.vec2, radius: float, dhat: float, kappa: float,
                               mu: float, epsv: float, inv_dt: float,
                               F: wp.array(dtype=wp.vec2)):
    """Friction reaction ON the disc (Newton's 3rd law: +mu f_n tdir), so the pull
    metric can include friction the way the explicit path's F_total does."""
    b = wp.tid()
    n = bnd[b]
    diff = x[n] - c
    dc = wp.length(diff)
    d = dc - radius
    if d <= 0.0 or d >= dhat:
        return
    nh = diff / dc
    f_n = kappa * wp.abs(barrier_bp(d, dhat))
    v = (x[n] - xn[n]) * inv_dt
    vt = v - wp.dot(v, nh) * nh
    tdir = vt / wp.max(wp.length(vt), epsv)
    wp.atomic_add(F, b2e[b], mu * f_n * tdir)          # on disc: opposes node's slide reaction


# ============================================ self-collision BROADPHASE (Warp HashGrid)
# The brute-force self kernels above are O(n_b^2) PER PNCG iteration. Because every
# design in a batch is built at the SAME physical (cx,cy), a naive spatial hash would
# cross-pair designs. Fix: pack each boundary node as vec3 with z = env * SHIFT (SHIFT
# > radius), so distinct designs live in separate z-slabs and can never be neighbours.
# Same-env pairs share z, so the 3D query distance == the true 2D distance. Turns each
# self kernel O(n_b) expected. Results are bit-for-bit the brute-force set (same pairs,
# same excl gate, same formula) up to float32 atomic order.
@wp.kernel
def pack_boundary_pos_kernel(x: wp.array(dtype=wp.vec2), boundary: wp.array(dtype=wp.int32),
                             boundary2env: wp.array(dtype=wp.int32), shift: float,
                             hx3: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    p = x[boundary[b]]
    hx3[b] = wp.vec3(p[0], p[1], float(boundary2env[b]) * shift)


@wp.kernel
def grad_diag_self_bp_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                             x: wp.array(dtype=wp.vec2), boundary: wp.array(dtype=wp.int32),
                             boundary2env: wp.array(dtype=wp.int32),
                             boundary_start: wp.array(dtype=wp.int32),
                             boundary_per_env: wp.array(dtype=wp.int32),
                             excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                             radius: float, stiffness: float, h2: float, clamp: int,
                             grad: wp.array(dtype=wp.vec2), diagH: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    g = wp.vec2(0.0, 0.0)
    dgx = float(0.0); dgy = float(0.0)
    q = wp.hash_grid_query(grid, hx3[b], radius)
    for jb in q:                                   # query yields the point index directly
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                diff = pa - x[boundary[jb]]
                dd = wp.sqrt(wp.dot(diff, diff) + 1.0e-12)
                if dd < radius:
                    nh = diff / dd
                    rd = radius - dd
                    g = g - stiffness * rd * nh
                    tang = stiffness * rd / dd
                    hx = stiffness * nh[0] * nh[0] - tang * (1.0 - nh[0] * nh[0])
                    hy = stiffness * nh[1] * nh[1] - tang * (1.0 - nh[1] * nh[1])
                    if clamp != 0:
                        hx = wp.max(hx, 0.0); hy = wp.max(hy, 0.0)
                    dgx = dgx + hx; dgy = dgy + hy
    wp.atomic_add(grad, node, h2 * g)
    wp.atomic_add(diagH, node, wp.vec2(h2 * dgx, h2 * dgy))


@wp.kernel
def pHp_self_bp_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                       x: wp.array(dtype=wp.vec2), p: wp.array(dtype=wp.vec2),
                       boundary: wp.array(dtype=wp.int32), boundary2env: wp.array(dtype=wp.int32),
                       boundary_start: wp.array(dtype=wp.int32),
                       boundary_per_env: wp.array(dtype=wp.int32),
                       excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                       radius: float, stiffness: float, h2: float, clamp: int,
                       pHp: wp.array(dtype=f64)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]; pp = p[node]
    val = float(0.0)
    q = wp.hash_grid_query(grid, hx3[b], radius)
    for jb in q:                                   # query yields the point index directly
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                diff = pa - x[boundary[jb]]
                dd = wp.sqrt(wp.dot(diff, diff) + 1.0e-12)
                if dd < radius:
                    nh = diff / dd
                    rd = radius - dd
                    dp = pp - p[boundary[jb]]
                    pn = wp.dot(dp, nh)
                    pt = dp - pn * nh
                    v = stiffness * pn * pn - stiffness * rd / dd * wp.dot(pt, pt)
                    if clamp != 0:
                        v = wp.max(v, 0.0)
                    val = val + 0.5 * v
    wp.atomic_add(pHp, e, f64(h2 * val))


@wp.kernel
def energy_self_bp_kernel(grid: wp.uint64, hx3: wp.array(dtype=wp.vec3),
                          x: wp.array(dtype=wp.vec2), boundary: wp.array(dtype=wp.int32),
                          boundary2env: wp.array(dtype=wp.int32),
                          boundary_start: wp.array(dtype=wp.int32),
                          boundary_per_env: wp.array(dtype=wp.int32),
                          excl_flat: wp.array(dtype=wp.uint8), excl_offset: wp.array(dtype=wp.int32),
                          radius: float, stiffness: float, h2: float, E: wp.array(dtype=f64)):
    b = wp.tid()
    node = boundary[b]
    e = boundary2env[b]
    start = boundary_start[e]
    n_be = boundary_per_env[e]
    i = b - start
    eoff = excl_offset[e]
    pa = x[node]
    val = float(0.0)
    q = wp.hash_grid_query(grid, hx3[b], radius)
    for jb in q:                                   # query yields the point index directly
        if jb != b and boundary2env[jb] == e:
            jl = jb - start
            if excl_flat[eoff + i * n_be + jl] == wp.uint8(0):
                diff = pa - x[boundary[jb]]
                dd = wp.sqrt(wp.dot(diff, diff) + 1.0e-12)
                if dd < radius:
                    rd = radius - dd
                    val = val + 0.5 * (0.5 * stiffness * rd * rd)
    wp.atomic_add(E, e, f64(h2 * val))
