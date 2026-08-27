"""2D PNCG-IPC math core — Warp @wp.func primitives.

Ported from Warp_pncg_ipc_csiro (3D tets) to 2D triangles. The 3D reference
builds a 12-DOF element stencil from mat33 / vec12f; here it is 6-DOF from
mat22 / vec6f.

CONVENTIONS (these are the silent-bug surface — all three are asserted in
pncg2d_selftest.py):

  * Ds = mat22 with EDGE VECTORS AS COLUMNS:  Ds = [x1-x0 | x2-x0].
    F = Ds @ B with B = D_m_inv = (rest Ds)^-1, so F = I at rest.
    This matches warp_fem.py:51-62 and mesh.py exactly, and the port is
    validated by asserting grad == -warp_fem.elastic_force_kernel node-by-node.
  * Node ordering in vec6f is node-major: [n0.x, n0.y, n1.x, n1.y, n2.x, n2.y].
    Node 0 is the apex whose block is minus the sum of the other two.
  * kappa is a PHYSICAL stiffness and BOTH the elastic and barrier terms are
    scaled by h^2 in the incremental potential. The CSIRO reference scales only
    the elastic term and folds dt^2 into its kappa (deformer_pncg.py:203 vs :258)
    — copying that convention with cfg.obstacle_stiffness is wrong by ~12 orders
    of magnitude. Follow this file, not the reference.

SECOND DERIVATIVES. The reference hand-derives a 12x12 element Hessian via
I1/I2/I3 invariants. In 2D the directional form is short enough to use directly:
for Psi = mu/2(I2 - 2) - mu log J + la/2 log^2 J, with G = F^-T,

    dPsi/dF        = mu F + (la logJ - mu) G
    d2Psi/dF2 [dF] = mu dF + la (G:dF) G - (la logJ - mu) G dF^T G

so p^T H p is one directional derivative and diag(H) is six (one per element
DOF). Exact, no 6x6 assembly, and no invariant decomposition to get wrong.
"""
from __future__ import annotations
import warp as wp

vec6 = wp.types.vector(6, wp.float32)

LOG_EPS = wp.constant(1.0e-6)


# ---------------------------------------------------------------- elastic
@wp.func
def ds_from_nodes(p0: wp.vec2, p1: wp.vec2, p2: wp.vec2) -> wp.mat22:
    """Edge vectors as COLUMNS — matches warp_fem.py:51-53."""
    return wp.mat22(p1[0] - p0[0], p2[0] - p0[0],
                    p1[1] - p0[1], p2[1] - p0[1])


@wp.func
def dF_from_dir(n: int, c: int, B: wp.mat22) -> wp.mat22:
    """dF produced by a unit perturbation of node `n` component `c`.

    F = Ds B is linear in x, so dF = dDs B where dDs has the basis pattern:
      n==0 -> both edges shrink   -> dDs = [-e_c | -e_c]
      n==1 -> first edge only     -> dDs = [ e_c |   0 ]
      n==2 -> second edge only    -> dDs = [  0  |  e_c]
    """
    a = float(0.0)
    b = float(0.0)
    if n == 0:
        a = -1.0
        b = -1.0
    elif n == 1:
        a = 1.0
    else:
        b = 1.0
    dDs = wp.mat22(0.0, 0.0, 0.0, 0.0)
    if c == 0:
        dDs = wp.mat22(a, b, 0.0, 0.0)
    else:
        dDs = wp.mat22(0.0, 0.0, a, b)
    return dDs * B


@wp.func
def dF_from_vec6(p: vec6, B: wp.mat22) -> wp.mat22:
    """dF for an arbitrary element displacement vector (node-major vec6)."""
    dDs = wp.mat22(p[2] - p[0], p[4] - p[0],
                   p[3] - p[1], p[5] - p[1])
    return dDs * B


@wp.func
def nh_dPsidF(F: wp.mat22, mu: float, la: float) -> wp.mat22:
    """First Piola: mu F + (la logJ - mu) F^-T. Same expression as
    warp_fem.py:58, so the assembled gradient is exactly minus that force."""
    J = wp.determinant(F)
    logJ = wp.log(wp.max(J, LOG_EPS))
    FinvT = wp.transpose(wp.inverse(F))
    return mu * F + (la * logJ - mu) * FinvT


@wp.func
def nh_d2PsidF2_dir(F: wp.mat22, dF: wp.mat22, mu: float, la: float) -> wp.mat22:
    """Directional second derivative d2Psi/dF2 [dF] (see module docstring)."""
    J = wp.determinant(F)
    logJ = wp.log(wp.max(J, LOG_EPS))
    Finv = wp.inverse(F)
    G = wp.transpose(Finv)
    GdF = wp.ddot(G, dF)                       # G : dF
    GdFtG = G * wp.transpose(dF) * G
    return mu * dF + la * GdF * G - (la * logJ - mu) * GdFtG


@wp.func
def nh_pHp_elem(F: wp.mat22, B: wp.mat22, p: vec6, mu: float, la: float) -> float:
    """p^T (d2Psi/dx2) p for one element, per unit rest area."""
    dF = dF_from_vec6(p, B)
    return wp.ddot(dF, nh_d2PsidF2_dir(F, dF, mu, la))


# ---------------------------------------------------------------- IPC barrier (point vs circle)
@wp.func
def cross2(a: wp.vec2, b: wp.vec2) -> float:
    """Scalar 2D cross product (wp.cross has no vec2 overload). Twice the signed
    area of the triangle spanned by a and b."""
    return a[0] * b[1] - a[1] * b[0]


# ---------------------------------------------------------------- IPC barrier (point vs circle)
@wp.func
def barrier_b(d: float, dhat: float) -> float:
    u = d - dhat
    return -(u * u) * wp.log(d / dhat)


@wp.func
def barrier_bp(d: float, dhat: float) -> float:
    """db/dd — identical to metrics._barrier_Bp / contact.py's Bp."""
    u = d - dhat
    return -2.0 * u * wp.log(d / dhat) - (u * u) / d


@wp.func
def barrier_bpp(d: float, dhat: float) -> float:
    """d2b/dd2."""
    u = d - dhat
    return -2.0 * wp.log(d / dhat) - 4.0 * u / d + (u * u) / (d * d)
