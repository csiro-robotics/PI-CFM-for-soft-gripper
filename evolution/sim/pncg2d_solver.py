"""Driver for the 2D multi-env PNCG-IPC solver — sequences the kernels in pncg2d.py.

Mirrors warp_grasp.warp_simulate_and_evaluate's API so it is a drop-in for the
optimiser, but takes CFL-free implicit steps: at dt = 1 ms the rig's 1.1 s closure
is 1100 steps instead of 599,128.

Per timestep (see pncg2d.py for why the ordering matters):

    init_step        xn = x ; xhat = x + dt*drag*v      <- BEFORE the BC moves
    apply_bc         x[bc] = target(t+dt) ; xhat[bc] = same
    reset_step       per-env dE0 = 0, active = 1
    repeat iter_max:
        grad + diagH   (mass, elastic, disc barrier)
        5 per-env Dai-Kou reductions -> finalize_beta -> update_p
        pHp            (mass, elastic, disc barrier) + p_inf + clr_min
        alpha caps     (area inversion root, exact circle CCD) -> alpha_finalize
        update_x       (converged envs are no-ops)
    commit_v         v = (x - xn)/dt

There are ZERO host syncs inside the inner loop: `active` is a device mask and
converged envs simply stop moving. The reference does ~14 host round-trips per
iteration, which at n_envs x n_steps x iter_max would dominate everything.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "warp_2d_fem"))
sys.path.insert(0, str(ROOT / "optimisation_methods"))
import pncg2d as K                                                      # noqa: E402
import pncg2d_sdf as SDF                                                # noqa: E402  arbitrary-object SDF contact
from mesh import build_mesh                                             # noqa: E402
from metrics import obstacle_force, contact_arc                         # noqa: E402
# Geometry/BC construction lives in finger.py.

f64 = wp.float64


@wp.kernel
def set_bc_target_kernel(bc_rest: wp.array(dtype=wp.vec2), bc_drv: wp.array(dtype=wp.int32),
                         off_pin: wp.vec2, off_drv: wp.vec2,
                         bc_target: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    if bc_drv[b] != 0:
        bc_target[b] = bc_rest[b] + off_drv
    else:
        bc_target[b] = bc_rest[b] + off_pin


@wp.kernel
def snapshot_bc_kernel(x: wp.array(dtype=wp.vec2), bc_idx: wp.array(dtype=wp.int32),
                       bc_rest: wp.array(dtype=wp.vec2)):
    b = wp.tid()
    bc_rest[b] = x[bc_idx[b]]


@wp.kernel
def max_force_kernel(F: wp.array(dtype=wp.vec2), maxf: wp.array(dtype=f64),
                     maxfy: wp.array(dtype=f64)):
    e = wp.tid()
    wp.atomic_max(maxf, e, f64(wp.length(F[e])))
    wp.atomic_max(maxfy, e, f64(F[e][1]))


@wp.kernel
def zero_vec2_kernel(a: wp.array(dtype=wp.vec2)):
    a[wp.tid()] = wp.vec2(0.0, 0.0)


@wp.kernel
def set_x_from_x0_kernel(x: wp.array(dtype=wp.vec2), x0: wp.array(dtype=wp.vec2),
                         p: wp.array(dtype=wp.vec2), node2env: wp.array(dtype=wp.int32),
                         alpha: wp.array(dtype=f64), active: wp.array(dtype=wp.int32)):
    """x = x0 + alpha*p  (per env, active only) — the line-search trial position."""
    i = wp.tid()
    e = node2env[i]
    if active[e] == 0:
        return
    x[i] = x0[i] + float(alpha[e]) * p[i]


@wp.kernel
def ls_halve_kernel(Ener: wp.array(dtype=f64), Eprev: wp.array(dtype=f64),
                    alpha: wp.array(dtype=f64), active: wp.array(dtype=wp.int32), tol: f64):
    """Backtracking safe-gate: halve alpha for any env whose trial energy rose above the
    pre-step energy (beyond a small tolerance). Guarantees monotone energy decrease, so
    the step can never diverge (blow up a stiff/dense design)."""
    b = wp.tid()
    if active[b] == 0:
        return
    if Ener[b] > Eprev[b] + tol * (wp.abs(Eprev[b]) + f64(1.0)):
        alpha[b] = alpha[b] * f64(0.5)


# build_multi_env / build_single_finger now live in finger.py (self-contained:
# no warp_grasp / run_voronoi_batched_sim / utils.socket_mask). The two-finger
# builder is intentionally NOT carried over -- this release is single-finger only.
from finger import build_multi_env, build_single_finger   # noqa: E402,F401


class PNCG2DSolver:
    def __init__(self, masks, cfg, H, W, device="cuda", dt=1.0e-3,
                 iter_max=150, eps=1e-4, tol_x=0.0, conv_check=0, near_factor=4.0, alpha_max=3.0,
                 area_shrink=0.1, ccd_eta=0.9, self_stiffness=0.0, self_radius=None,
                 friction=False, line_search=False, n_backtrack=4, ls_tol=1e-7,
                 prebuilt=None,
                 object_kind=None, object_rx=None, object_ry=None, object_rref=None):
        self.cfg, self.dev, self.dt = cfg, device, float(dt)
        self.iter_max, self.eps = int(iter_max), float(eps)
        # tol_x: IPC-style INCREMENT INF-NORM convergence (metres of node motion per iteration).
        # 0.0 disables it -> identical to the original energy-decrease-only behaviour.
        # The energy test is capped by a sqrt(N)*float32 summation floor (~1.6e-5 at N=18k), which
        # is why eps=1e-8/1e-6 NEVER converge; alpha*p_inf is an atomic MAX and has no such floor.
        self.tol_x = float(tol_x)
        # conv_check: check every N iterations whether ALL envs are done and break out of _iterate.
        # Without this, converging early saves NOTHING -- the loop always ran all iter_max launches
        # (measured: median iters 320 -> 100 when eps was relaxed, with ZERO wall-clock change).
        # 0 disables. Costs one device->host sync per check, so keep N >= 8.
        self.conv_check = int(conv_check)
        self.near_factor, self.alpha_max = float(near_factor), float(alpha_max)
        self.area_shrink, self.ccd_eta = float(area_shrink), float(ccd_eta)
        # deferred physics (default OFF -> byte-identical to the validated v1)
        self.self_stiffness = float(self_stiffness)
        # node-node self-contact radius. 0.5*dx is INERT on this mesh: boundary edges
        # are ~1*dx, so a vertex slips THROUGH an edge staying >0.5*dx from both its
        # nodes. 1.5*dx keeps the vertex within range of the edge's endpoints before it
        # can cross -> clears pull-stage self-penetration (measured 0 on ranks 8/29,
        # clean designs unchanged). Full point-edge would be the exact fix.
        self.self_radius = float(self_radius) if self_radius is not None \
            else 1.5 * cfg.finger_width / W
        self.friction = bool(friction)
        # energy line-search safe-gate: after the model+CCD step is chosen, verify it
        # decreases the total energy; if not, halve it (n_backtrack times). Guarantees
        # the implicit step can't diverge on stiff/dense designs (topopt). OFF by default
        # (byte-identical to the validated path); ~2-3x cost when on.
        self.line_search = bool(line_search)
        self.n_bt, self.ls_tol = int(n_backtrack), float(ls_tol)
        self.B = len(masks)
        me, pinned, driven, block_rows = prebuilt if prebuilt is not None \
            else build_multi_env(masks, cfg, H, W)
        self.me, self.masks = me, masks
        self.N = me.rest_pos.shape[0]
        self.n_tri = len(me.rest_area)
        self.n_b = len(me.boundary_nodes)

        a32 = lambda v, d: wp.array(np.asarray(v, np.float32), dtype=d, device=device)
        ai = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=device)
        self.x = a32(me.rest_pos, wp.vec2)
        self.v = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.xn = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.xhat = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.grad = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.grad_prev = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.p = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.diagH = wp.zeros(self.N, dtype=wp.vec2, device=device)
        self.mass = a32(me.mass, wp.float32)
        self.tri = wp.array(np.asarray(me.tri, np.int32), dtype=wp.vec3i, device=device)
        self.Bm = a32(me.D_m_inv, wp.mat22)
        self.area = a32(me.rest_area, wp.float32)
        self.mu = a32(me.mu, wp.float32)
        self.lam = a32(me.lam, wp.float32)
        self.bnd = ai(me.boundary_nodes)
        self.b2e = ai(me.boundary2env)
        self.node2env = ai(me.node2env)
        self.tri2env = ai(me.tri2env)

        free = np.ones((self.N, 2), np.float32)
        bc = np.concatenate([pinned, driven])
        free[bc] = 0.0
        self.free = a32(free, wp.vec2)
        self.bc_idx = ai(bc)
        self.bc_is_drv = ai(np.concatenate([np.zeros(len(pinned), np.int32),
                                            np.ones(len(driven), np.int32)]))
        self.bc_rest = wp.zeros(len(bc), dtype=wp.vec2, device=device)
        self.bc_target = wp.zeros(len(bc), dtype=wp.vec2, device=device)
        self.n_bc = len(bc)

        z = lambda: wp.zeros(self.B, dtype=f64, device=device)
        self.gPg, self.g_p, self.g_Py = z(), z(), z()
        self.y_p, self.y_Py = z(), z()
        self.beta, self.gTp, self.pHp = z(), z(), z()
        self.alpha, self.p_inf, self.clr_min = z(), z(), z()
        self.dE, self.dE0 = z(), z()
        self.maxf, self.maxfy = z(), z()
        self.active = wp.zeros(self.B, dtype=wp.int32, device=device)
        self.iters = wp.zeros(self.B, dtype=wp.int32, device=device)
        self.Fdisc = wp.zeros(self.B, dtype=wp.vec2, device=device)
        self.Ener = z()
        self.Els_prev = z()                                # energy before the step (line-search)
        self.x0buf = wp.zeros(self.N, dtype=wp.vec2, device=device)   # pre-step x (line-search)

        self.c = wp.vec2(float(cfg.object_x), float(cfg.object_y))
        self.radius = float(cfg.object_r)
        self.dhat = float(cfg.ipc_d_hat)
        self.kappa = float(cfg.obstacle_stiffness)
        # arbitrary-object geometry (kind=0 -> exact disc / unchanged; kind>=1 -> SDF level-set).
        # explicit args win; else read from cfg (so the QD pipeline can pick the object); else disc.
        self.object_kind = int(object_kind if object_kind is not None else getattr(cfg, "object_kind", 0))
        self.obj_cx = float(cfg.object_x); self.obj_cy = float(cfg.object_y)
        self.obj_rx = float(object_rx if object_rx is not None else getattr(cfg, "object_rx", self.radius))
        self.obj_ry = float(object_ry if object_ry is not None else getattr(cfg, "object_ry", self.radius))
        self.obj_rref = float(object_rref if object_rref is not None else getattr(cfg, "object_rref", self.radius))
        self.h2 = self.dt * self.dt
        self.inv_dt = 1.0 / self.dt
        self.drag = math.exp(-self.dt * cfg.damping)
        self.mu_fric = float(cfg.obstacle_friction)
        self.epsv = float(cfg.obstacle_friction_epsv)

        # self-collision structure (boundary layout + mesh-adjacency exclusion mask)
        # + a Warp HashGrid broadphase: each boundary node is packed as vec3 with
        # z = env*shift (shift > radius) so distinct designs live in separate z-slabs
        # and never cross-pair -> O(n_b) instead of O(n_b^2) per iteration.
        if self.self_stiffness > 0.0:
            self.sc_bstart = ai(np.asarray(me.boundary_starts[:-1], np.int32))
            self.sc_bpe = ai(np.asarray(me.boundary_per_env, np.int32))
            self.sc_excl = wp.array(np.asarray(me.excl_flat, np.uint8), dtype=wp.uint8, device=device)
            self.sc_excl_off = ai(np.asarray(me.excl_offset, np.int32))
            self.hx3 = wp.zeros(self.n_b, dtype=wp.vec3, device=device)
            self.sc_grid = wp.HashGrid(128, 128, 128, device=device)
            self.sc_shift = 8.0 * self.self_radius      # z-slab gap between designs

    def _build_sc_grid(self):
        """Pack boundary nodes as z-slab-separated vec3 and (re)build the broadphase
        grid at the current x. Cell size = self_radius (query ring covers one cell)."""
        wp.launch(K.pack_boundary_pos_kernel, dim=self.n_b,
                  inputs=[self.x, self.bnd, self.b2e, self.sc_shift, self.hx3], device=self.dev)
        self.sc_grid.build(self.hx3, self.self_radius)

    # ------------------------------------------------------------------ inner solve
    def _iterate(self):
        d = self.dev
        for it in range(self.iter_max):
            if self.conv_check and it and it % self.conv_check == 0:
                if int(self.active.numpy().sum()) == 0:
                    break
            wp.launch(K.grad_diag_mass_kernel, dim=self.N,
                      inputs=[self.x, self.xhat, self.mass, self.free,
                              self.grad, self.grad_prev, self.diagH], device=d)
            wp.launch(K.grad_diag_elastic_kernel, dim=self.n_tri,
                      inputs=[self.x, self.tri, self.Bm, self.area, self.mu, self.lam,
                              self.h2, 1, self.grad, self.diagH], device=d)
            if self.object_kind == 0:
                wp.launch(K.grad_diag_disc_kernel, dim=self.n_b,
                          inputs=[self.x, self.bnd, self.c, self.radius, self.dhat,
                                  self.kappa, self.h2, 1, self.grad, self.diagH], device=d)
            else:
                wp.launch(SDF.grad_diag_sdf_kernel, dim=self.n_b,
                          inputs=[self.x, self.bnd, self.object_kind, self.obj_cx, self.obj_cy,
                                  self.obj_rx, self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                                  self.h2, 1, self.grad, self.diagH], device=d)
            if self.self_stiffness > 0.0:
                self._build_sc_grid()               # rebuild at current x (used by grad + pHp)
                wp.launch(K.grad_diag_self_bp_kernel, dim=self.n_b,
                          inputs=[self.sc_grid.id, self.hx3, self.x, self.bnd, self.b2e,
                                  self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                                  self.self_radius, self.self_stiffness, self.h2, 1,
                                  self.grad, self.diagH], device=d)
            if self.friction:
                if self.object_kind == 0:
                    wp.launch(K.grad_friction_disc_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.c, self.radius, self.dhat,
                                      self.kappa, self.mu_fric, self.epsv, self.inv_dt, self.h2,
                                      self.free, self.grad, self.diagH], device=d)
                else:
                    wp.launch(SDF.grad_friction_sdf_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.object_kind, self.obj_cx,
                                      self.obj_cy, self.obj_rx, self.obj_ry, self.obj_rref,
                                      self.dhat, self.kappa, self.mu_fric, self.epsv,
                                      self.inv_dt, self.h2, self.free, self.grad, self.diagH],
                              device=d)
            wp.launch(K.mask_grad_kernel, dim=self.N,
                      inputs=[self.grad, self.free], device=d)

            for arr in (self.gPg, self.g_p, self.g_Py, self.y_p, self.y_Py):
                arr.zero_()
            wp.launch(K.dk_reduce_kernel, dim=self.N,
                      inputs=[self.grad, self.grad_prev, self.p, self.diagH, self.free,
                              self.node2env, self.gPg, self.g_p, self.g_Py,
                              self.y_p, self.y_Py], device=d)
            wp.launch(K.finalize_beta_kernel, dim=self.B,
                      inputs=[self.gPg, self.g_p, self.g_Py, self.y_p, self.y_Py, it,
                              self.beta, self.gTp], device=d)
            wp.launch(K.update_p_kernel, dim=self.N,
                      inputs=[self.grad, self.diagH, self.free, self.node2env,
                              self.beta, self.p], device=d)

            wp.launch(K.pHp_reset_kernel, dim=self.B,
                      inputs=[self.pHp, self.alpha, self.p_inf, self.clr_min,
                              self.alpha_max], device=d)
            wp.launch(K.pHp_mass_kernel, dim=self.N,
                      inputs=[self.p, self.mass, self.node2env, self.pHp, self.p_inf],
                      device=d)
            wp.launch(K.pHp_elastic_kernel, dim=self.n_tri,
                      inputs=[self.x, self.p, self.tri, self.Bm, self.area, self.mu,
                              self.lam, self.h2, 1, self.tri2env, self.pHp], device=d)
            if self.object_kind == 0:
                wp.launch(K.pHp_disc_kernel, dim=self.n_b,
                          inputs=[self.x, self.p, self.bnd, self.b2e, self.c, self.radius,
                                  self.dhat, self.kappa, self.h2, 1, self.pHp, self.clr_min],
                          device=d)
            else:
                wp.launch(SDF.pHp_sdf_kernel, dim=self.n_b,
                          inputs=[self.x, self.p, self.bnd, self.b2e, self.object_kind, self.obj_cx,
                                  self.obj_cy, self.obj_rx, self.obj_ry, self.obj_rref, self.dhat,
                                  self.kappa, self.h2, 1, self.pHp, self.clr_min], device=d)
            if self.friction:
                # alpha = -gTp/pHp must stay self-consistent: gTp is built from `grad`,
                # which contains friction, so pHp has to contain it too. Dispatched on the
                # same object_kind as the friction gradient above.
                if self.object_kind == 0:
                    wp.launch(K.pHp_friction_disc_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.b2e, self.p, self.c,
                                      self.radius, self.dhat, self.kappa, self.mu_fric,
                                      self.epsv, self.inv_dt, self.h2, self.free, self.pHp],
                              device=d)
                else:
                    wp.launch(SDF.pHp_friction_sdf_kernel, dim=self.n_b,
                              inputs=[self.x, self.xn, self.bnd, self.b2e, self.p,
                                      self.object_kind, self.obj_cx, self.obj_cy, self.obj_rx,
                                      self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                                      self.mu_fric, self.epsv, self.inv_dt, self.h2,
                                      self.free, self.pHp], device=d)
            if self.self_stiffness > 0.0:                 # reuse grid from the grad step (x fixed)
                wp.launch(K.pHp_self_bp_kernel, dim=self.n_b,
                          inputs=[self.sc_grid.id, self.hx3, self.x, self.p, self.bnd, self.b2e,
                                  self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                                  self.self_radius, self.self_stiffness, self.h2, 1,
                                  self.pHp], device=d)

            wp.launch(K.inversion_alpha_kernel, dim=self.n_tri,
                      inputs=[self.x, self.p, self.tri, self.tri2env,
                              self.area_shrink, self.alpha], device=d)
            if self.object_kind == 0:
                wp.launch(K.disc_ccd_kernel, dim=self.n_b,
                          inputs=[self.x, self.p, self.bnd, self.b2e, self.c, self.radius,
                                  self.ccd_eta, self.alpha], device=d)
            else:
                wp.launch(SDF.sdf_ccd_kernel, dim=self.n_b,
                          inputs=[self.x, self.p, self.bnd, self.b2e, self.object_kind, self.obj_cx,
                                  self.obj_cy, self.obj_rx, self.obj_ry, self.obj_rref,
                                  self.ccd_eta, self.alpha], device=d)
            wp.launch(K.alpha_finalize_kernel, dim=self.B,
                      inputs=[self.gTp, self.pHp, self.p_inf, self.clr_min, self.alpha,
                              self.dE, self.dE0, self.active, self.iters,
                              self.dhat, self.near_factor, self.eps, self.tol_x, it], device=d)
            if not self.line_search:
                wp.launch(K.update_x_kernel, dim=self.N,
                          inputs=[self.x, self.p, self.node2env, self.alpha, self.active],
                          device=d)
            else:
                # ENERGY LINE-SEARCH GATE: x is still the pre-step position here; the grad
                # step just built the self-collision grid at it. Verify the model+CCD step
                # lowers the energy; halve it (per env) up to n_bt times if it doesn't.
                self._energy_into(self.Els_prev, rebuild_grid=False)     # E(x0)
                wp.copy(self.x0buf, self.x)                              # save x0
                for _bt in range(self.n_bt):
                    wp.launch(set_x_from_x0_kernel, dim=self.N,
                              inputs=[self.x, self.x0buf, self.p, self.node2env,
                                      self.alpha, self.active], device=d)
                    self._energy_into(self.Ener, rebuild_grid=False)     # E(x0 + alpha*p)
                    wp.launch(ls_halve_kernel, dim=self.B,
                              inputs=[self.Ener, self.Els_prev, self.alpha, self.active,
                                      f64(self.ls_tol)], device=d)
                wp.launch(set_x_from_x0_kernel, dim=self.N,               # commit final alpha
                          inputs=[self.x, self.x0buf, self.p, self.node2env,
                                  self.alpha, self.active], device=d)

    def step(self, off_pin, off_drv):
        d = self.dev
        wp.launch(K.init_step_kernel, dim=self.N,
                  inputs=[self.x, self.v, self.xn, self.xhat, self.dt, self.drag], device=d)
        wp.launch(set_bc_target_kernel, dim=self.n_bc,
                  inputs=[self.bc_rest, self.bc_is_drv, wp.vec2(*off_pin),
                          wp.vec2(*off_drv), self.bc_target], device=d)
        wp.launch(K.apply_bc_kernel, dim=self.n_bc,
                  inputs=[self.x, self.xhat, self.bc_idx, self.bc_target], device=d)
        wp.launch(K.reset_step_kernel, dim=self.B,
                  inputs=[self.dE0, self.active, self.iters], device=d)
        self.p.zero_(); self.grad.zero_()
        self._iterate()
        wp.launch(K.commit_v_kernel, dim=self.N,
                  inputs=[self.x, self.xn, self.v, 1.0 / self.dt], device=d)

    def snapshot_bc(self):
        wp.launch(snapshot_bc_kernel, dim=self.n_bc,
                  inputs=[self.x, self.bc_idx, self.bc_rest], device=self.dev)

    def _energy_into(self, buf, rebuild_grid=True):
        """Total energy (inertia + elastic + disc barrier + self barrier) per env into
        `buf`, GPU-resident (no host copy). rebuild_grid=False reuses the self-collision
        hash grid already built at the iteration-start x — used for line-search trials,
        where rebuilding every backtrack would be wasteful (the step is small)."""
        buf.zero_(); d = self.dev
        wp.launch(K.energy_kernel, dim=self.N,
                  inputs=[self.x, self.xhat, self.mass, self.node2env, buf], device=d)
        wp.launch(K.energy_elastic_kernel, dim=self.n_tri,
                  inputs=[self.x, self.tri, self.Bm, self.area, self.mu, self.lam,
                          self.h2, self.tri2env, buf], device=d)
        if self.object_kind == 0:
            wp.launch(K.energy_disc_kernel, dim=self.n_b,
                      inputs=[self.x, self.bnd, self.b2e, self.c, self.radius, self.dhat,
                              self.kappa, self.h2, buf], device=d)
        else:
            wp.launch(SDF.energy_sdf_kernel, dim=self.n_b,
                      inputs=[self.x, self.bnd, self.b2e, self.object_kind, self.obj_cx, self.obj_cy,
                              self.obj_rx, self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                              self.h2, buf], device=d)
        if self.self_stiffness > 0.0:
            if rebuild_grid:
                self._build_sc_grid()
            wp.launch(K.energy_self_bp_kernel, dim=self.n_b,
                      inputs=[self.sc_grid.id, self.hx3, self.x, self.bnd, self.b2e,
                              self.sc_bstart, self.sc_bpe, self.sc_excl, self.sc_excl_off,
                              self.self_radius, self.self_stiffness, self.h2, buf], device=d)
        if self.friction:                                  # lagged friction potential (gates friction)
            if self.object_kind == 0:
                wp.launch(K.energy_friction_disc_kernel, dim=self.n_b,
                          inputs=[self.x, self.xn, self.bnd, self.b2e, self.c, self.radius,
                                  self.dhat, self.kappa, self.mu_fric, self.epsv, self.inv_dt,
                                  self.h2, self.free, buf], device=d)
            else:
                wp.launch(SDF.energy_friction_sdf_kernel, dim=self.n_b,
                          inputs=[self.x, self.xn, self.bnd, self.b2e, self.object_kind,
                                  self.obj_cx, self.obj_cy, self.obj_rx, self.obj_ry,
                                  self.obj_rref, self.dhat, self.kappa, self.mu_fric,
                                  self.epsv, self.inv_dt, self.h2, self.free, buf], device=d)

    def energy(self):
        self._energy_into(self.Ener, rebuild_grid=True)
        return self.Ener.numpy().copy()

    def _fill_disc_force(self, include_friction):
        """Barrier reaction on the disc into self.Fdisc, + friction (Newton's 3rd
        law) when asked — matching the explicit path's F_total = barrier + friction."""
        wp.launch(zero_vec2_kernel, dim=self.B, inputs=[self.Fdisc], device=self.dev)
        if self.object_kind == 0:
            wp.launch(K.disc_force_kernel, dim=self.n_b,
                      inputs=[self.x, self.bnd, self.b2e, self.c, self.radius, self.dhat,
                              self.kappa, self.Fdisc], device=self.dev)
        else:
            wp.launch(SDF.sdf_force_kernel, dim=self.n_b,
                      inputs=[self.x, self.bnd, self.b2e, self.object_kind, self.obj_cx, self.obj_cy,
                              self.obj_rx, self.obj_ry, self.obj_rref, self.dhat, self.kappa,
                              self.Fdisc], device=self.dev)
        if include_friction and self.friction:
            if self.object_kind == 0:
                wp.launch(K.disc_friction_force_kernel, dim=self.n_b,
                          inputs=[self.x, self.xn, self.bnd, self.b2e, self.c, self.radius,
                                  self.dhat, self.kappa, self.mu_fric, self.epsv, self.inv_dt,
                                  self.Fdisc], device=self.dev)
            else:
                wp.launch(SDF.sdf_friction_force_kernel, dim=self.n_b,
                          inputs=[self.x, self.xn, self.bnd, self.b2e, self.object_kind,
                                  self.obj_cx, self.obj_cy, self.obj_rx, self.obj_ry,
                                  self.obj_rref, self.dhat, self.kappa, self.mu_fric,
                                  self.epsv, self.inv_dt, self.Fdisc], device=self.dev)

    def disc_force(self, include_friction=False):
        self._fill_disc_force(include_friction)
        return self.Fdisc.numpy().copy()

    # ------------------------------------------------------------------ phases
    def squeeze(self, T, log_every=0, verbose=True):
        n = max(int(round(T / self.dt)), 1)
        self.snapshot_bc()
        for k in range(1, n + 1):
            a = 0.5 * (1 - math.cos(math.pi * min(k * self.dt / (self.cfg.ramp_fraction * T), 1.0)))
            self.step((0.0, 0.0), (0.0, -self.cfg.input_disp * a))
            if verbose and log_every and (k % log_every == 0 or k == n):
                F = self.disc_force()
                print(f"  [sq {k:>5}/{n}] closure {self.cfg.input_disp*a*1e3:5.2f} mm  "
                      f"iters {self.iters.numpy().max():>4}  Fy {F[:,1]} "
                      f"clr {self.clr_min.numpy().min()*1e3:+.4f} mm", flush=True)
        return n

    def pull(self, dist, T, sample_every=1, log_every=0, verbose=True):
        n = max(int(round(T / self.dt)), 1)
        self.snapshot_bc()
        self.maxf.zero_(); self.maxfy.zero_()
        for k in range(1, n + 1):
            off = (0.0, dist * k / n)
            self.step(off, off)
            if k % sample_every == 0 or k == n:
                self._fill_disc_force(include_friction=True)      # barrier + friction
                wp.launch(max_force_kernel, dim=self.B,
                          inputs=[self.Fdisc, self.maxf, self.maxfy], device=self.dev)
            if verbose and log_every and (k % log_every == 0 or k == n):
                print(f"  [pl {k:>5}/{n}] {dist*k/n*1e3:5.1f} mm  "
                      f"maxFy {self.maxfy.numpy()}", flush=True)
        return n

    def host_metrics(self):
        """arc / ncon / clearance on the host, using the SAME functions the
        explicit path's metrics use, so a comparison is not confounded."""
        x = self.x.numpy().astype(np.float64)
        me, cfg = self.me, self.cfg
        dc = [cfg.object_x, cfg.object_y]
        F = obstacle_force(x[me.boundary_nodes], np.asarray(me.boundary2env), self.B,
                           dc, cfg.object_r, cfg.obstacle_stiffness, cfg.ipc_d_hat)
        arc = contact_arc(x, me.boundary_nodes, me.boundary_edges, me.node2env, self.B,
                          dc, cfg.object_r, cfg.ipc_d_hat)
        d = np.linalg.norm(x[me.boundary_nodes] - np.asarray(dc), axis=1) - cfg.object_r
        ncon = np.zeros(self.B, int)
        np.add.at(ncon, np.asarray(me.boundary2env), ((d > 0) & (d < cfg.ipc_d_hat)).astype(int))
        clr = np.full(self.B, np.inf)
        np.minimum.at(clr, np.asarray(me.boundary2env), d)
        return dict(F=F, arc=arc, ncon=ncon, clr=clr, x=x)


def implicit_simulate_and_evaluate(masks, cfg, H, W, device="cuda", dt=1.0e-3,
                                   close_T=None, pull_T=None, pull_dist=None, iter_max=150,
                                   self_collision=True, friction=True, verbose=False):
    """Drop-in for warp_grasp.warp_simulate_and_evaluate on the IMPLICIT PNCG-IPC
    path. Physical close/pull RATES match the explicit cfg (n_steps*dt), but the
    steps are CFL-free — dt=1 ms means ~50 steps where explicit needs 20000. Returns
    (solver, per-env metric dicts) with the same keys the optimiser reads.

    self_collision/friction default ON (the finalised physics); set False to
    reproduce the validated v1 (disc contact only)."""
    from descriptors import branch_density as _bd
    B = len(masks)
    close_T = close_T if close_T is not None else cfg.n_steps * cfg.dt
    pull_T = pull_T if pull_T is not None else cfg.pull_n_steps * cfg.dt
    pull_dist = pull_dist if pull_dist is not None else cfg.pull_distance
    s = PNCG2DSolver(masks, cfg, H, W, device=device, dt=dt, iter_max=iter_max,
                     self_stiffness=(cfg.contact_stiffness if self_collision else 0.0),
                     friction=friction)
    s.squeeze(close_T, verbose=verbose)
    hm = s.host_metrics()                                 # arc/ncon at end of close
    xc = hm["x"]
    tri = np.asarray(s.me.tri); dmi = np.asarray(s.me.D_m_inv)
    ds = np.stack([xc[tri[:, 1]] - xc[tri[:, 0]], xc[tri[:, 2]] - xc[tri[:, 0]]], -1)
    Fdet = np.linalg.det(ds @ dmi)                        # det F per triangle
    t2e = np.asarray(s.me.tri2env)
    jmin = np.full(B, np.inf); jmax = np.full(B, -np.inf)
    np.minimum.at(jmin, t2e, Fdet); np.maximum.at(jmax, t2e, Fdet)
    disp = np.linalg.norm(xc - s.me.rest_pos, axis=1)     # mean nodal displacement
    n2e = np.asarray(s.me.node2env)
    md = np.zeros(B); cnt = np.zeros(B)
    np.add.at(md, n2e, disp); np.add.at(cnt, n2e, 1.0); md /= np.maximum(cnt, 1)

    s.pull(pull_dist, pull_T, verbose=verbose)
    pull_off = s.maxf.numpy()                             # max |F on disc| (barrier+friction)
    out = []
    for b in range(B):
        po = float(pull_off[b]); arc = float(hm["arc"][b]); ncon = int(hm["ncon"][b])
        valid = (math.isfinite(po) and math.isfinite(arc) and ncon > 0 and arc > 0
                 and jmin[b] >= cfg.buckle_J_min and jmax[b] <= cfg.J_max_thr)
        out.append(dict(valid=valid, pull_off=po, arc=arc, jmin=float(jmin[b]),
                        jmax=float(jmax[b]), ncon=ncon, mat=float(masks[b].mean()),
                        mean_disp=float(md[b]), branch_density=_bd(masks[b])))
    return s, out
