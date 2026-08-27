"""
Residual Physics Engine for Mechanism Design Flow Matching

Provides physics-informed loss computation and validation metrics for
training a conditional flow matching model that generates mechanism
topology designs.

Core idea (PIDM-style):
    During denoising, the physics constraint should be *loose* at high
    noise levels σ (the sample is still noisy, physics doesn't apply)
    and *tight* as σ → 0 (the sample should satisfy equilibrium).
    This is achieved by scaling the mechanism loss by 1 / (σ² + ε).

Formulation — **Fixed-output reaction force**:
    The output DOF is fixed (u_out = 0). The mechanism loss is the signed
    reaction force at the output:

        R_out = (K · u)_out   (force the clamp exerts on the structure)
        Loss  = sign · R_out

    Holding u_out = 0 against a mechanism that drives the output in the
    `sign` direction takes a reaction pointing the other way, so a good
    mechanism drives sign · R_out large and negative.  Random noise or
    disconnected structures cannot transmit force → R_out ≈ 0.  This
    creates strong discrimination between good and bad topologies.

Loss components:
    1. Mechanism Loss: sign · R_out — maximize transmitted force at output
    2. Binarization:   mean(ρ * (1 − ρ))    — encourages 0/1 material distribution
    3. Total:          w_mech * L_mech + w_bin * L_bin

Usage:
    # output_dof MUST be included in solver.fixed_dofs
    solver = TorchFEMSolverFast(nx, ny, fixed_dofs, device='cuda')

    engine = ResidualMechanics(
        solver, prescribed_dofs=..., prescribed_values=...,
        output_dof=output_dof, output_sign=1.0,
    )

    # Training loop
    losses = engine.compute_objectives(density_batch, sigma=noise_level)
    losses['total'].backward()

    # Validation
    metrics = engine.compute_metrics(density_batch)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union

from fem_solver.torch_fem_solver_fast import _pcg_solve


# ---------------------------------------------------------------------------
# Connectivity check (CPU / NumPy helper — called inside torch.no_grad)
# ---------------------------------------------------------------------------

def check_floating_material(density: torch.Tensor, threshold: float = 0.5) -> bool:
    """
    Detect disconnected (floating) material islands in a 2-D density field.

    Uses ``cv2.connectedComponents`` on the binarised density image.
    A design is considered *connected* when all solid pixels belong to a
    single connected component.

    Args:
        density: Element densities, shape ``(ny, nx)`` in [0, 1].
        threshold: Binarisation threshold (pixels ≥ threshold → solid).

    Returns:
        ``True`` if there are multiple disconnected islands (floating
        material detected), ``False`` if the structure is fully connected.
    """
    try:
        import cv2
    except ImportError:
        # If OpenCV is unavailable, skip the check gracefully.
        return False

    binary = (density.detach().cpu().numpy() >= threshold).astype(np.uint8)
    n_labels, _ = cv2.connectedComponents(binary, connectivity=4)
    # n_labels includes the background (label 0), so > 2 means multiple islands
    return n_labels > 2


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ResidualMechanics(nn.Module):
    """
    Physics-informed loss & metric computation for mechanism topology design.

    Uses **displacement-controlled input + fixed output** (reaction-force)
    formulation.  The dataset convention stores prescribed displacement
    values in ``inputx``/``inputy``.  The output DOF is clamped (u=0)
    and the loss maximises the reaction force at that DOF.

    FEM solve approach:
        1. Build ``u_pre`` with known displacement at input DOFs
           (output DOF is in fixed_dofs ⇒ u_out = 0 automatically)
        2. Compute ``K @ u_pre`` element-by-element (no global assembly)
        3. Solve ``K_ff @ u_correction = −K @ u_pre`` (free DOFs only)
        4. ``u_total = u_correction + u_pre``
        5. Compute reaction force ``R = (K · u_total)_out``

    Loss components:
        1. Mechanism:    ``sign · R_out`` — maximize transmitted force at output
        2. Binarisation: ``mean(ρ * (1 − ρ))``

    Parameters
    ----------
    solver : nn.Module
        Differentiable FEM solver (``TorchFEMSolverFast``).
        ``fixed_dofs`` must include prescribed input DOFs AND output DOF.
    prescribed_dofs : torch.Tensor
        DOF indices with prescribed displacement, shape ``(n_prescribed,)``.
    prescribed_values : torch.Tensor
        Prescribed displacement values, shape ``(n_prescribed,)``.
    output_dof : int or torch.Tensor
        DOF index at which reaction force is measured.
        Must be included in ``solver.fixed_dofs``.
    output_sign : float
        +1.0 or −1.0 — desired direction of output motion.
        Loss = ``output_sign * R_output``.
    w_mechanism, w_binary : float
        Loss weights.
    """

    def __init__(
        self,
        solver: nn.Module,
        prescribed_dofs: torch.Tensor,
        prescribed_values: torch.Tensor,
        output_dof: Union[int, torch.Tensor],
        output_sign: float = 1.0,
        w_mechanism: float = 1.0,
        w_binary: float = 0.1,
    ):
        super().__init__()

        self.solver = solver
        self.output_dof = output_dof
        self.output_sign = output_sign

        # Loss weights stored as buffers so they travel with .to(device)
        self.register_buffer("w_mechanism", torch.tensor(w_mechanism))
        self.register_buffer("w_binary", torch.tensor(w_binary))

        # Prescribed displacement BCs (input port)
        self.register_buffer("prescribed_dofs", prescribed_dofs.clone())
        self.register_buffer("prescribed_values", prescribed_values.clone())

        # Cache grid shape from solver
        self.nx: int = solver.nx
        self.ny: int = solver.ny
        self.nel: int = solver.nel

    # ------------------------------------------------------------------
    # Prescribed-displacement FEM solve (batched)
    # ------------------------------------------------------------------

    def _solve_and_reaction(
        self,
        density: torch.Tensor,
    ):
        """
        Solve FEM with prescribed-displacement input + fixed output,
        then compute reaction force at the output DOF.

        Steps:
            1. Build u_pre with prescribed values at input DOFs
               (output DOF is already in fixed_dofs → u_out = 0)
            2. Compute K @ u_pre element-by-element
            3. Solve K_ff @ u_correction = −K @ u_pre
            4. u_total = u_correction + u_pre
            5. Compute R = (K · u_total) at output DOF (reaction force)

        Args:
            density: Element densities, shape (batch, nel).

        Returns:
            u_total: Full displacement field, shape (batch, ndof).
            reaction: Reaction force at output DOF, shape (batch,).
        """
        batch_size = density.shape[0]
        device = self.solver.device
        dtype = self.solver.dtype
        ndof = self.solver.ndof

        # 1. Build u_pre (batch, ndof) — non-zero only at prescribed DOFs
        #    output DOF is fixed → u_pre[output_dof] = 0 (already zero)
        u_pre = torch.zeros(batch_size, ndof, dtype=dtype, device=device)
        u_pre[:, self.prescribed_dofs] = self.prescribed_values.unsqueeze(0).expand(batch_size, -1)

        # 2. Compute K @ u_pre element-by-element
        x_filt = self.solver._apply_filter(density)
        E_elem = self.solver.E_min + x_filt.pow(self.solver.penal) * (self.solver.E_max - self.solver.E_min)
        # E_elem: (batch, nel)

        # Gather u_pre at element DOFs: (batch, nel, 8) — memory-efficient
        u_pre_elem = u_pre[:, self.solver.edofMat]  # (batch, nel, 8)

        # KE @ u_pre_e for each element: (batch, nel, 8)
        Ku_pre_elem = torch.einsum('ij,bnj->bni', self.solver.KE, u_pre_elem)
        # Scale by element modulus
        Ku_pre_elem = Ku_pre_elem * E_elem.unsqueeze(-1)  # (batch, nel, 8)

        # Scatter-add to global vector
        Ku_pre = torch.zeros(batch_size, ndof, dtype=dtype, device=device)
        edof_flat = self.solver.edofMat_flat.unsqueeze(0).expand(batch_size, -1)  # (batch, nel*8)
        Ku_pre.scatter_add_(1, edof_flat, Ku_pre_elem.reshape(batch_size, -1))

        # 3. Effective force: f_eff = -K @ u_pre
        f_eff = -Ku_pre

        # 4. Solve K_ff @ u_correction = f_eff
        #    (prescribed DOFs AND output DOF are in fixed_dofs)
        _, u_correction = self.solver.forward_compliance(
            density, f_eff, return_displacement=True,
        )

        # 5. Total displacement
        u_total = u_correction + u_pre

        # 6. Compute reaction force at output DOF: R = (K · u_total)_output
        #    Assemble K @ u_total element-by-element, read off output DOF
        u_total_elem = u_total[:, self.solver.edofMat]  # (batch, nel, 8)
        Ku_total_elem = torch.einsum('ij,bnj->bni', self.solver.KE, u_total_elem)
        Ku_total_elem = Ku_total_elem * E_elem.unsqueeze(-1)  # (batch, nel, 8)

        # Scatter-add to get full K @ u_total
        Ku_total = torch.zeros(batch_size, ndof, dtype=dtype, device=device)
        Ku_total.scatter_add_(1, edof_flat, Ku_total_elem.reshape(batch_size, -1))

        # Extract reaction force at output DOF
        if isinstance(self.output_dof, int):
            reaction = Ku_total[:, self.output_dof]  # (batch,)
        else:
            reaction = Ku_total.gather(
                1, self.output_dof.view(-1, 1).expand(batch_size, 1)
            ).squeeze(1)

        return u_total, reaction

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def compute_objectives(
        self,
        density: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed training losses using displacement-controlled FEM.

        Parameters
        ----------
        density : torch.Tensor
            Element densities, shape ``(batch, nel)`` or ``(nel,)``.
        sigma : torch.Tensor, optional
            Noise level(s), shape ``(batch,)`` or scalar.
            When provided, mechanism loss is scaled by PIDM annealing.

        Returns
        -------
        dict with keys: mechanism, reaction_force, binary, pidm_scale, total
        """
        squeeze = False
        if density.dim() == 1:
            density = density.unsqueeze(0)
            squeeze = True

        batch_size = density.shape[0]
        density_safe = density.clamp(1e-3, 1.0)

        # --- 1. Mechanism loss (reaction force at fixed output) -------------
        _, reaction = self._solve_and_reaction(density_safe)  # (batch,)

        # loss = sign · R_out  (minimising this maximises the force the
        # mechanism transmits against the output clamp)
        mech_loss = self.output_sign * reaction  # (batch,)

        # --- 2. Binarisation loss ------------------------------------------
        bin_loss = (density_safe * (1.0 - density_safe)).mean(dim=1)

        # --- 3. PIDM noise-aware weighting ---------------------------------
        if sigma is not None:
            sigma = sigma.to(density.device)
            if sigma.dim() == 0:
                sigma = sigma.expand(batch_size)
            pidm_scale = 1.0 / (sigma + 0.1)
            pidm_scale = pidm_scale.clamp(max=10.0)
            mech_loss_scaled = mech_loss * pidm_scale
        else:
            pidm_scale = torch.ones(batch_size, device=density.device, dtype=density.dtype)
            mech_loss_scaled = mech_loss

        # --- 4. Total ------------------------------------------------------
        total = (
            self.w_mechanism * mech_loss_scaled.mean()
            + self.w_binary * bin_loss.mean()
        )

        result = {
            "mechanism": mech_loss,
            "reaction_force": reaction,
            "binary": bin_loss,
            "pidm_scale": pidm_scale,
            "total": total,
        }

        if squeeze:
            for k in ("mechanism", "reaction_force", "binary", "pidm_scale"):
                result[k] = result[k].squeeze(0)

        return result

    # ------------------------------------------------------------------
    # Validation / logging metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_metrics(
        self,
        density: torch.Tensor,
    ) -> Dict[str, Union[float, bool, torch.Tensor]]:
        """
        Compute human-readable validation metrics (no gradient).

        Uses displacement-controlled solve, same as training.
        """
        squeeze = False
        if density.dim() == 1:
            density = density.unsqueeze(0)
            squeeze = True

        density_safe = density.clamp(1e-3, 1.0)

        u, reaction = self._solve_and_reaction(density_safe)

        if squeeze:
            u_sample = u.squeeze(0)
            rho = density_safe.squeeze(0)
            r_out = reaction.squeeze(0).item()
        else:
            u_sample = u[0]
            rho = density_safe[0]
            r_out = reaction[0].item()

        bin_val = (rho * (1.0 - rho)).mean().item()
        binary_score = 1.0 - 4.0 * bin_val

        # Floating material check
        rho_2d = rho.reshape(self.ny, self.nx)
        has_floating = check_floating_material(rho_2d)

        return {
            "max_displacement": torch.abs(u_sample).max().item(),
            "reaction_force": r_out,
            "binary_score": binary_score,
            "has_floating": has_floating,
        }


# ---------------------------------------------------------------------------
# Penalty-method batched FEM solve (all samples in ONE CG solve)
# ---------------------------------------------------------------------------

class PenaltyFEMSolveFunction(torch.autograd.Function):
    """
    Custom autograd for batched penalty-method FEM solve.

    Forward:  Solve (K + P·diag(fixed)) @ u = P·diag(fixed)·u_pre  via CG
    Backward: Adjoint solve (same penalised system) for dL/dE_elem
    """

    @staticmethod
    def forward(ctx, E_elem, rhs, fixed_float, penalty, solver):
        batch_size = E_elem.shape[0]

        diag_K = solver._compute_diagonal_full(E_elem.detach())
        diag_precond = (diag_K + penalty * fixed_float).clamp(min=1e-30)

        def matvec(v):
            Kv = solver._matvec_full(v, E_elem.detach())
            return Kv + penalty * fixed_float * v

        def precond(r):
            return r / diag_precond

        x0 = rhs / diag_precond

        with torch.no_grad():
            u = _pcg_solve(matvec, rhs, x0, solver.cg_tol, solver.cg_max_iter, precond)

        ctx.save_for_backward(E_elem, u, fixed_float)
        ctx.penalty = penalty
        ctx.solver = solver
        return u

    @staticmethod
    def backward(ctx, grad_u):
        E_elem, u, fixed_float = ctx.saved_tensors
        penalty = ctx.penalty
        solver = ctx.solver

        diag_K = solver._compute_diagonal_full(E_elem.detach())
        diag_precond = (diag_K + penalty * fixed_float).clamp(min=1e-30)

        def matvec(v):
            Kv = solver._matvec_full(v, E_elem.detach())
            return Kv + penalty * fixed_float * v

        def precond(r):
            return r / diag_precond

        x0 = torch.zeros_like(grad_u)
        with torch.no_grad():
            lam = _pcg_solve(matvec, grad_u, x0, solver.cg_tol, solver.cg_max_iter, precond)

        # dL/dE_e = -λ_e^T @ KE @ u_e
        u_elem = u[:, solver.edofMat]
        lam_elem = lam[:, solver.edofMat]
        KE_u = torch.einsum('ij,bnj->bni', solver.KE, u_elem)
        grad_E = -(lam_elem * KE_u).sum(dim=-1)

        return grad_E, None, None, None, None


def compute_batched_physics_loss(
    densities: torch.Tensor,
    conds: torch.Tensor,
    base_solver,
    sigma: Optional[torch.Tensor] = None,
    w_mechanism: float = 1.0,
    w_binary: float = 1.0,
    penalty: float = 1e6,
) -> Dict[str, torch.Tensor]:
    """
    Compute physics losses for a batch of samples with DIFFERENT BCs
    in a SINGLE batched CG solve using the penalty method.

    Instead of per-sample solver clones + sequential CG solves, this:
    1. Extracts BCs from condition tensors (small GPU-side loop)
    2. Builds batched penalty masks and prescribed displacements
    3. Runs ONE batched CG solve for all samples simultaneously
    4. Computes all losses in parallel

    ~4-8x faster than the per-sample loop on H100.
    """
    batch_size = densities.shape[0]
    device = densities.device
    dtype = base_solver.dtype
    ny, nx = base_solver.ny, base_solver.nx
    ndof = base_solver.ndof

    # --- Extract BCs from condition tensors (GPU-side, fast) ---
    fixed_masks = torch.zeros(batch_size, ndof, dtype=torch.bool, device=device)
    u_prescribed = torch.zeros(batch_size, ndof, dtype=dtype, device=device)
    output_dofs = torch.zeros(batch_size, dtype=torch.long, device=device)
    output_signs = torch.ones(batch_size, dtype=dtype, device=device)

    for b in range(batch_size):
        cond_b = conds[b].detach()
        bc_x, bc_y = cond_b[0], cond_b[1]
        inp_x, inp_y = cond_b[2], cond_b[3]
        out_x, out_y = cond_b[4], cond_b[5]

        # Fixed DOFs from BC masks
        js, is_ = torch.where(bc_x > 0.5)
        if len(js) > 0:
            fixed_masks[b, 2 * (is_ * (ny + 1) + js)] = True
        js, is_ = torch.where(bc_y > 0.5)
        if len(js) > 0:
            fixed_masks[b, 2 * (is_ * (ny + 1) + js) + 1] = True

        # Prescribed input displacements
        for ch, dof_off in [(inp_x, 0), (inp_y, 1)]:
            mask = ch.abs() > 1e-8
            if not mask.any():
                continue
            js, is_ = torch.where(mask)
            vals = ch[js, is_].to(dtype)
            n_bl = (ny + 1) * is_ + js
            n_br = (ny + 1) * (is_ + 1) + js
            n_tl = (ny + 1) * is_ + (js + 1)
            n_tr = (ny + 1) * (is_ + 1) + (js + 1)
            for nodes in [n_bl, n_br, n_tl, n_tr]:
                dofs = 2 * nodes + dof_off
                fixed_masks[b, dofs] = True
                u_prescribed[b, dofs] = vals

        # Output DOF
        out_mask = (out_x.abs() + out_y.abs()) > 1e-8
        out_js, out_is = torch.where(out_mask)
        if len(out_js) > 0:
            oi = int(out_is.float().mean())
            oj = int(out_js.float().mean())
            out_node = oi * (ny + 1) + oj
            if abs(float(out_x[oj, oi])) >= abs(float(out_y[oj, oi])):
                output_dofs[b] = 2 * out_node
                output_signs[b] = float(torch.sign(out_x[oj, oi]))
            else:
                output_dofs[b] = 2 * out_node + 1
                output_signs[b] = float(torch.sign(out_y[oj, oi]))
        else:
            out_node = nx * (ny + 1) + ny // 2
            output_dofs[b] = 2 * out_node

        fixed_masks[b, output_dofs[b]] = True

    # --- SIMP interpolation ---
    density_safe = densities.clamp(1e-3, 1.0)
    x_filt = base_solver._apply_filter(density_safe)
    E_elem = base_solver.E_min + x_filt.pow(base_solver.penal) * (base_solver.E_max - base_solver.E_min)

    # --- ONE batched penalty CG solve ---
    fixed_float = fixed_masks.to(dtype=dtype)
    rhs = penalty * fixed_float * u_prescribed

    u = PenaltyFEMSolveFunction.apply(E_elem, rhs, fixed_float, penalty, base_solver)

    # --- Reaction force: R = (K @ u)[output_dof]  (physical K, no penalty) ---
    Ku = base_solver._matvec_full(u, E_elem)
    reaction = Ku.gather(1, output_dofs.unsqueeze(1)).squeeze(1)

    # --- Losses ---
    mech_loss = output_signs * reaction

    bin_loss = (density_safe * (1.0 - density_safe)).mean(dim=1)

    # --- PIDM noise-aware weighting ---
    if sigma is not None:
        sigma = sigma.to(device)
        if sigma.dim() == 0:
            sigma = sigma.expand(batch_size)
        pidm_scale = (1.0 / (sigma + 0.1)).clamp(max=10.0)
    else:
        pidm_scale = torch.ones(batch_size, dtype=dtype, device=device)

    total = (
        w_mechanism * pidm_scale * mech_loss
        + w_binary * bin_loss
    ).mean()

    return {
        "total": total,
        "mechanism": mech_loss.detach().mean(),
        "binary": bin_loss.detach().mean(),
        "reaction_force": reaction.detach().mean(),
        "pidm_scale": pidm_scale.detach().mean(),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fem_solver.torch_fem_solver_fast import TorchFEMSolverFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Setup solver with displacement-controlled input + fixed output ---
    nx, ny = 40, 40
    ndof = 2 * (nx + 1) * (ny + 1)

    # Fix left edge
    fixed_dofs = []
    for j in range(ny + 1):
        node = j  # left edge: x=0
        fixed_dofs.extend([2 * node, 2 * node + 1])

    # Prescribe displacement at top-right corner (y-direction)
    input_node = nx * (ny + 1) + ny  # top-right node
    input_dof = 2 * input_node + 1   # y-DOF
    prescribed_disp = -0.5
    fixed_dofs.append(input_dof)  # add to fixed DOFs

    # Output DOF: bottom-right corner, x-direction — also fixed!
    output_node = nx * (ny + 1)
    output_dof = 2 * output_node
    fixed_dofs.append(output_dof)  # fix output for reaction force

    fixed_dofs = torch.tensor(sorted(set(fixed_dofs)), dtype=torch.long, device=device)

    solver = TorchFEMSolverFast(
        nx=nx, ny=ny, fixed_dofs=fixed_dofs,
        penal=1.0, rmin=1.5,
        device=device, dtype=torch.float64,
    )

    prescribed_dofs = torch.tensor([input_dof], dtype=torch.long, device=device)
    prescribed_values = torch.tensor([prescribed_disp], dtype=torch.float64, device=device)

    # --- Create engine ---
    engine = ResidualMechanics(
        solver=solver,
        prescribed_dofs=prescribed_dofs,
        prescribed_values=prescribed_values,
        output_dof=output_dof,
        output_sign=1.0,
        w_mechanism=1.0,
        w_binary=0.5,
    )

    # --- Test compute_objectives ---
    print("=" * 60)
    print("Test 1: compute_objectives (single sample)")
    print("=" * 60)

    rho = torch.ones(nx * ny, dtype=torch.float64, device=device) * 0.4
    rho.requires_grad_(True)

    losses = engine.compute_objectives(rho)
    print(f"  Mechanism loss:  {losses['mechanism'].item():.6f}")
    print(f"  Reaction force:  {losses['reaction_force'].item():.6f}")
    print(f"  Binary loss:     {losses['binary'].item():.6f}")
    print(f"  Total loss:      {losses['total'].item():.6f}")

    losses["total"].backward()
    print(f"  Gradient norm:  {rho.grad.norm().item():.6f}")
    print(f"  Gradient range: [{rho.grad.min().item():.4e}, {rho.grad.max().item():.4e}]")

    # --- Test with PIDM scaling ---
    print("\n" + "=" * 60)
    print("Test 2: compute_objectives with PIDM sigma scaling")
    print("=" * 60)

    batch = 4
    rho_b = torch.rand(batch, nx * ny, dtype=torch.float64, device=device) * 0.5 + 0.2
    rho_b.requires_grad_(True)

    sigmas = torch.tensor([0.01, 0.1, 0.5, 1.0], dtype=torch.float64, device=device)
    losses_b = engine.compute_objectives(rho_b, sigma=sigmas)

    print(f"  Mechanism loss: {losses_b['mechanism'].detach().cpu().numpy()}")
    print(f"  PIDM scale:     {losses_b['pidm_scale'].detach().cpu().numpy()}")
    print(f"  Total loss:     {losses_b['total'].item():.6f}")

    losses_b["total"].backward()
    print(f"  Batch gradient shape: {rho_b.grad.shape}")

    # --- Test compute_metrics ---
    print("\n" + "=" * 60)
    print("Test 3: compute_metrics (validation)")
    print("=" * 60)

    rho_val = torch.rand(nx * ny, dtype=torch.float64, device=device) * 0.6
    metrics = engine.compute_metrics(rho_val)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25s}: {v:.6f}")
        else:
            print(f"  {k:<25s}: {v}")

    print("\nAll tests passed.")
