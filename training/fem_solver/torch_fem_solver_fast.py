"""
Optimized Pure PyTorch GPU-Differentiable FEM Solver for Mechanism Design

Key Optimizations over TorchFEMSolver:
1. Matrix-free CG solver: Never assembles the dense global K matrix
   - O(nnz) per CG iteration instead of O(ndof³) dense solve
   - Memory: O(nel) instead of O(ndof²)
2. Element-by-element matrix-vector product for K @ v
3. Jacobi (diagonal) preconditioning for faster CG convergence
4. Vectorized operations — no Python loops in hot paths
5. Optional float32 mode for ~2x GPU speedup
6. Sparse Cholesky fallback for small grids

For a 64×128 grid:
  - Old: 2.2 GB for dense K, O(ndof³) = O(16770³) solve
  - New: ~50 MB, O(nel × n_cg_iters) solve

Usage:
    solver = TorchFEMSolverFast(nx, ny, fixed_dofs, device='cuda')
    loss = solver(density, f_input, output_dof, target_disp)
    loss.backward()
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import math


class FEMSolveFunction(torch.autograd.Function):
    """
    Custom autograd for matrix-free FEM solve with proper adjoint gradients.
    
    Forward:  Solve K(E) @ u = f via CG  →  u
    Backward: Solve K(E) @ λ = dL/du via CG (adjoint),
              then dL/dE_e = -λ_e^T @ KE @ u_e  for each element
    """
    
    @staticmethod
    def forward(ctx, E_elem, f_free, solver):
        """
        Args:
            E_elem: Element moduli (batch, nel) — requires_grad
            f_free: Free-DOF force vector (batch, n_free)  
            solver: Reference to the TorchFEMSolverFast instance
        """
        batch_size = E_elem.shape[0]
        
        # Compute preconditioner
        diag_K = solver._compute_diagonal_precond(E_elem.detach())
        
        def matvec(v):
            return solver._matvec_free(v, E_elem.detach())
        def precond(r):
            return r / diag_K
        
        x0 = torch.zeros_like(f_free)
        
        with torch.no_grad():
            u_free = _pcg_solve(matvec, f_free, x0, solver.cg_tol, solver.cg_max_iter, precond)
        
        # Save for backward
        ctx.save_for_backward(E_elem, u_free)
        ctx.solver = solver
        
        return u_free
    
    @staticmethod
    def backward(ctx, grad_u_free):
        """
        Adjoint method:
        1. Solve K @ λ = dL/du  (adjoint CG solve)
        2. dL/dE_e = -λ_e^T @ KE @ u_e  (element-level sensitivity)
        """
        E_elem, u_free = ctx.saved_tensors
        solver = ctx.solver
        batch_size = E_elem.shape[0]
        
        # Compute preconditioner for adjoint solve
        diag_K = solver._compute_diagonal_precond(E_elem.detach())
        
        def matvec(v):
            return solver._matvec_free(v, E_elem.detach())
        def precond(r):
            return r / diag_K
        
        # Adjoint solve: K @ λ = grad_u_free
        x0 = torch.zeros_like(grad_u_free)
        with torch.no_grad():
            lambda_free = _pcg_solve(
                matvec, grad_u_free, x0,
                solver.cg_tol, solver.cg_max_iter, precond
            )
        
        # Expand u and λ to full DOF space for element gather
        u_full = torch.zeros(batch_size, solver.ndof, dtype=solver.dtype, device=solver.device)
        u_full[:, solver.free_dofs] = u_free
        
        lam_full = torch.zeros(batch_size, solver.ndof, dtype=solver.dtype, device=solver.device)
        lam_full[:, solver.free_dofs] = lambda_free
        
        # Gather element DOFs: (batch, nel, 8)
        u_elem = u_full[:, solver.edofMat]
        lam_elem = lam_full[:, solver.edofMat]
        
        # Element sensitivity: dL/dE_e = -λ_e^T @ KE @ u_e
        # KE: (8, 8), u_elem: (batch, nel, 8)
        KE_u = torch.einsum('ij,bnj->bni', solver.KE, u_elem)  # (batch, nel, 8)
        grad_E = -(lam_elem * KE_u).sum(dim=-1)  # (batch, nel)
        
        # grad for f_free = lambda (adjoint variable)
        grad_f = lambda_free
        
        return grad_E, grad_f, None


def _pcg_solve(matvec_fn, b, x0, tol, max_iter, precond_fn=None):
    """
    Preconditioned Conjugate Gradient solver.
    
    Solves A @ x = b where A is accessed only via matvec_fn.
    
    Args:
        matvec_fn: Function computing A @ v
        b: RHS vector, shape (batch, n) or (n,)
        x0: Initial guess
        tol: Relative tolerance on the PRECONDITIONED residual,
             sqrt(r^T M^-1 r) / sqrt(b^T M^-1 b)
        max_iter: Maximum iterations
        precond_fn: Optional preconditioner M^{-1} @ v
    
    Returns:
        x: Solution vector
    """
    x = x0.clone()
    r = b - matvec_fn(x)
    
    if precond_fn is not None:
        z = precond_fn(r)
        bz = (b * precond_fn(b)).sum(dim=-1, keepdim=True)
    else:
        z = r.clone()
        bz = (b * b).sum(dim=-1, keepdim=True)

    # Convergence is measured in the PRECONDITIONED norm,
    #     sqrt(r^T M^-1 r) / sqrt(b^T M^-1 b),
    # never the plain ||r|| / ||b||.  Under the penalty method the RHS is
    # b = P * fixed * u_pre with P ~ 1e6, so ||b|| is dominated by the
    # constraint rows and ||r||/||b|| drops below tol while the interior
    # equilibrium equations are still unsolved: the displacement field never
    # propagates away from the input, the output reaction comes out exactly
    # zero, and the adjoint still emits a ghost gradient.  Jacobi M^-1 divides
    # the penalty rows back down by their own diagonal, so every equation is
    # weighed in the same units and the interior actually has to converge.
    bz_norm = bz.clamp(min=1e-30).sqrt()

    p = z.clone()
    rz = (r * z).sum(dim=-1, keepdim=True)  # r^T M^{-1} r
    
    for i in range(max_iter):
        Ap = matvec_fn(p)
        pAp = (p * Ap).sum(dim=-1, keepdim=True)
        alpha = rz / pAp.clamp(min=1e-30)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        if precond_fn is not None:
            z = precond_fn(r)
        else:
            z = r.clone()

        rz_new = (r * z).sum(dim=-1, keepdim=True)

        # Check convergence in the M^-1 norm (see above)
        if (rz_new.clamp(min=0).sqrt() / bz_norm).max() < tol:
            break

        beta = rz_new / rz.clamp(min=1e-30)
        p = z + beta * p
        rz = rz_new
    
    return x


class TorchFEMSolverFast(nn.Module):
    """
    Optimized Pure PyTorch FEM Solver for Mechanism Design on GPU.
    
    Key differences from TorchFEMSolver:
    - Matrix-free: Never assembles the global K matrix (saves GBs of memory)
    - CG solver: O(nel × n_iter) instead of O(ndof³) dense solve
    - Element-by-element matvec: Fully vectorized K @ v products
    - Jacobi preconditioner: Extracted cheaply from diagonal
    - Optional sparse direct solve for small problems
    
    Args:
        nx, ny: Grid dimensions
        fixed_dofs: Fixed DOF indices
        young_modulus, poisson_ratio, penal, E_min: Material parameters
        rmin: Filter radius
        cg_tol: CG convergence tolerance (relative PRECONDITIONED residual)
        cg_max_iter: Maximum CG iterations
        use_direct_solve: If True, use dense solve (for small grids < 40×40)
        device, dtype: Device and precision
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        fixed_dofs: torch.Tensor,
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        penal: float = 3.0,
        E_min: float = 1e-9,
        rmin: float = 1.5,
        cg_tol: float = 1e-6,
        cg_max_iter: int = 2000,
        use_direct_solve: bool = False,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        
        self.nx = nx
        self.ny = ny
        self.nel = nx * ny
        self.n_nodes = (nx + 1) * (ny + 1)
        self.ndof = 2 * self.n_nodes
        self.penal = penal
        self.E_min = E_min
        self.E_max = young_modulus
        self.device = device
        self.dtype = dtype
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.use_direct_solve = use_direct_solve
        
        # Element stiffness matrix (8×8)
        KE = self._compute_element_stiffness(poisson_ratio)
        self.register_buffer('KE', torch.as_tensor(KE, dtype=dtype, device=device))
        
        # Element DOF mapping: (nel, 8)
        edofMat = self._compute_edof_map()
        self.register_buffer('edofMat', torch.as_tensor(edofMat, dtype=torch.long, device=device))
        
        # For matrix-free matvec: precompute scatter indices
        # edofMat_flat: (nel * 8,) — all DOF indices
        self.register_buffer('edofMat_flat', self.edofMat.reshape(-1))
        
        # Density filter (sparse COO)
        H_indices, H_values, Hs = self._compute_filter(rmin)
        self.register_buffer('H_indices', torch.as_tensor(H_indices, dtype=torch.long, device=device))
        self.register_buffer('H_values', torch.as_tensor(H_values, dtype=dtype, device=device))
        self.register_buffer('Hs', torch.as_tensor(Hs, dtype=dtype, device=device))
        
        # DOF classification
        self.register_buffer('fixed_dofs_buf', fixed_dofs.to(dtype=torch.long, device=device))
        all_dofs = torch.arange(self.ndof, device=device)
        free_mask = ~torch.isin(all_dofs, self.fixed_dofs_buf)
        self.register_buffer('free_dofs', all_dofs[free_mask])
        self.register_buffer('free_mask', free_mask)
        self.n_free = self.free_dofs.shape[0]
        
        # Mapping: full DOF index → free DOF index (-1 if fixed)
        dof_to_free = torch.full((self.ndof,), -1, dtype=torch.long, device=device)
        dof_to_free[self.free_dofs] = torch.arange(self.n_free, device=device)
        self.register_buffer('dof_to_free', dof_to_free)
        
        # Precompute: for each element, which of its 8 DOFs are free
        # and what are their indices in the reduced system
        # edofMat_free: (nel, 8) — maps to free DOF indices, -1 if fixed
        edof_free = dof_to_free[self.edofMat]  # (nel, 8)
        self.register_buffer('edofMat_free', edof_free)
        
        # Precompute diagonal of K for Jacobi preconditioner
        # diag(K) = sum_e E_e * diag(KE) scattered to global DOFs
        # We store the KE diagonal for fast recomputation
        self.register_buffer('KE_diag', torch.diag(self.KE))  # (8,)
        
        # For direct solve mode: precompute assembly indices
        if use_direct_solve:
            iK, jK = self._compute_assembly_indices(self.edofMat)
            self.register_buffer('iK', torch.as_tensor(iK, dtype=torch.long, device=device))
            self.register_buffer('jK', torch.as_tensor(jK, dtype=torch.long, device=device))

    def clone_with_dofs(self, fixed_dofs: torch.Tensor) -> 'TorchFEMSolverFast':
        """
        Create a lightweight solver clone with different fixed DOFs.

        SHARES heavy buffers (KE, edofMat, filter) with self — only
        recomputes DOF classification (~0.1 ms).  Each clone is a
        fully independent instance suitable for use in separate
        autograd graphs.

        Args:
            fixed_dofs: New fixed DOF indices (1-D LongTensor on device).

        Returns:
            New TorchFEMSolverFast with same grid/material but new BCs.
        """
        clone = TorchFEMSolverFast.__new__(TorchFEMSolverFast)
        nn.Module.__init__(clone)

        # Copy scalar attributes
        clone.nx = self.nx
        clone.ny = self.ny
        clone.nel = self.nel
        clone.n_nodes = self.n_nodes
        clone.ndof = self.ndof
        clone.penal = self.penal
        clone.E_min = self.E_min
        clone.E_max = self.E_max
        clone.device = self.device
        clone.dtype = self.dtype
        clone.cg_tol = self.cg_tol
        clone.cg_max_iter = self.cg_max_iter
        clone.use_direct_solve = self.use_direct_solve

        # Share heavy buffers (no copy — same GPU memory)
        clone.register_buffer('KE', self.KE)
        clone.register_buffer('edofMat', self.edofMat)
        clone.register_buffer('edofMat_flat', self.edofMat_flat)
        clone.register_buffer('H_indices', self.H_indices)
        clone.register_buffer('H_values', self.H_values)
        clone.register_buffer('Hs', self.Hs)
        clone.register_buffer('KE_diag', self.KE_diag)

        # Recompute DOF classification for new fixed_dofs
        fixed_dofs = fixed_dofs.to(dtype=torch.long, device=self.device)
        clone.register_buffer('fixed_dofs_buf', fixed_dofs)

        all_dofs = torch.arange(clone.ndof, device=self.device)
        free_mask = ~torch.isin(all_dofs, fixed_dofs)
        clone.register_buffer('free_dofs', all_dofs[free_mask])
        clone.register_buffer('free_mask', free_mask)
        clone.n_free = clone.free_dofs.shape[0]

        dof_to_free = torch.full((clone.ndof,), -1, dtype=torch.long, device=self.device)
        dof_to_free[clone.free_dofs] = torch.arange(clone.n_free, device=self.device)
        clone.register_buffer('dof_to_free', dof_to_free)
        clone.register_buffer('edofMat_free', dof_to_free[clone.edofMat])

        return clone
    
    def forward(
        self,
        density: torch.Tensor,
        force_vector: torch.Tensor,
        output_dof: Union[int, torch.Tensor],
        target_displacement: float = 1.0,
        return_displacement: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: Solve FEM and compute mechanism design loss.
        
        Args:
            density: Element densities, shape (batch, nel) or (nel,)
            force_vector: Applied force vector, shape (batch, ndof) or (ndof,)
            output_dof: DOF index for output displacement
            target_displacement: Target displacement value
            return_displacement: If True, also return displacement field
            
        Returns:
            loss: MSE loss (u_output - u_target)^2
            u (optional): Full displacement field
        """
        squeeze_output = False
        if density.dim() == 1:
            density = density.unsqueeze(0)
            force_vector = force_vector.unsqueeze(0)
            squeeze_output = True
        
        batch_size = density.shape[0]
        
        # 1. Apply density filter
        x_filt = self._apply_filter(density)
        
        # 2. SIMP interpolation
        E_elem = self.E_min + x_filt.pow(self.penal) * (self.E_max - self.E_min)
        
        # 3. Solve
        if self.use_direct_solve:
            u = self._solve_direct(E_elem, force_vector, batch_size)
        else:
            u = self._solve_cg(E_elem, force_vector, batch_size)
        
        # 4. Extract output displacement and compute loss
        if isinstance(output_dof, int):
            u_output = u[:, output_dof]
        else:
            u_output = u.gather(1, output_dof.view(-1, 1)).squeeze(1)
        
        loss = (u_output - target_displacement).pow(2)
        
        if squeeze_output:
            loss = loss.squeeze(0)
            u = u.squeeze(0)
        
        if return_displacement:
            return loss, u
        return loss
    
    def forward_compliance(
        self,
        density: torch.Tensor,
        force_vector: torch.Tensor,
        return_displacement: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute compliance c = u^T @ f (for standard topology optimization).
        """
        squeeze_output = False
        if density.dim() == 1:
            density = density.unsqueeze(0)
            force_vector = force_vector.unsqueeze(0)
            squeeze_output = True
        
        batch_size = density.shape[0]
        
        x_filt = self._apply_filter(density)
        E_elem = self.E_min + x_filt.pow(self.penal) * (self.E_max - self.E_min)
        
        if self.use_direct_solve:
            u = self._solve_direct(E_elem, force_vector, batch_size)
        else:
            u = self._solve_cg(E_elem, force_vector, batch_size)
        
        compliance = (u * force_vector).sum(dim=1)
        
        if squeeze_output:
            compliance = compliance.squeeze(0)
            u = u.squeeze(0)
        
        if return_displacement:
            return compliance, u
        return compliance
    
    # ==================== Core Compute ====================
    
    def _matvec_free(self, v: torch.Tensor, E_elem: torch.Tensor) -> torch.Tensor:
        """
        Matrix-free computation of K_free @ v (element-by-element).
        
        Instead of assembling the global K matrix, we:
        1. Gather element DOF values from v (using edofMat)
        2. Multiply by element stiffness: KE @ v_e (for each element)
        3. Scale by element modulus: E_e * (KE @ v_e)
        4. Scatter-add results back to global vector
        
        All operations are batched and vectorized.
        
        Args:
            v: Vector in free-DOF space, shape (batch, n_free)
            E_elem: Element moduli, shape (batch, nel)
            
        Returns:
            result: K_free @ v, shape (batch, n_free)
        """
        batch_size = v.shape[0]
        
        # Expand v to full DOF space (fixed DOFs = 0)
        v_full = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        v_full[:, self.free_dofs] = v  # (batch, ndof)
        
        # Gather element DOF values: v_e for each element
        # edofMat: (nel, 8) -> v_elem: (batch, nel, 8)
        v_elem = v_full[:, self.edofMat]  # (batch, nel, 8)
        
        # Element matrix-vector product: KE @ v_e for each element
        # KE: (8, 8), v_elem: (batch, nel, 8) -> (batch, nel, 8)
        Kv_elem = torch.einsum('ij,bnj->bni', self.KE, v_elem)  # (batch, nel, 8)
        
        # Scale by element modulus
        Kv_elem = Kv_elem * E_elem.unsqueeze(-1)  # (batch, nel, 8)
        
        # Scatter-add back to global DOF vector
        result_full = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        
        # Flatten: (batch, nel, 8) -> (batch, nel*8)
        Kv_flat = Kv_elem.reshape(batch_size, -1)
        idx = self.edofMat_flat.unsqueeze(0).expand(batch_size, -1)
        result_full.scatter_add_(1, idx, Kv_flat)
        
        # Extract free DOFs only
        result = result_full[:, self.free_dofs]  # (batch, n_free)
        
        return result
    
    def _compute_diagonal_precond(self, E_elem: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal of K_free for Jacobi preconditioning.
        
        diag(K)[dof_i] = sum over elements containing dof_i of E_e * KE[local_i, local_i]
        
        This is very cheap: O(nel) instead of O(ndof²).
        
        Args:
            E_elem: Element moduli, shape (batch, nel)
            
        Returns:
            diag_K_free: Diagonal of K restricted to free DOFs, shape (batch, n_free)
        """
        batch_size = E_elem.shape[0]
        
        # KE_diag: (8,) -> E_elem * KE_diag for each element: (batch, nel, 8)
        diag_contrib = E_elem.unsqueeze(-1) * self.KE_diag.unsqueeze(0).unsqueeze(0)
        # Flatten to (batch, nel*8)
        diag_contrib_flat = diag_contrib.reshape(batch_size, -1)
        
        # Scatter-add to global diagonal
        diag_full = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        # edofMat_flat: (nel*8,) -> expand to (batch, nel*8)
        idx = self.edofMat_flat.unsqueeze(0).expand(batch_size, -1)
        diag_full.scatter_add_(1, idx, diag_contrib_flat)
        
        # Extract free DOFs
        diag_free = diag_full[:, self.free_dofs]  # (batch, n_free)
        
        # Clamp to avoid division by zero
        diag_free = diag_free.clamp(min=1e-30)
        
        return diag_free
    
    def _matvec_full(self, v: torch.Tensor, E_elem: torch.Tensor) -> torch.Tensor:
        """
        Matrix-free K @ v on the FULL DOF space (no DOF elimination).
        
        Used by the penalty method for batched solves where different
        samples have different fixed DOFs.
        
        Args:
            v: (batch, ndof) vector in full DOF space
            E_elem: (batch, nel) element moduli
            
        Returns:
            Kv: (batch, ndof)
        """
        batch_size = v.shape[0]
        v_elem = v[:, self.edofMat]  # (batch, nel, 8)
        Kv_elem = torch.einsum('ij,bnj->bni', self.KE, v_elem)
        Kv_elem = Kv_elem * E_elem.unsqueeze(-1)
        result = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        idx = self.edofMat_flat.unsqueeze(0).expand(batch_size, -1)
        result.scatter_add_(1, idx, Kv_elem.reshape(batch_size, -1))
        return result
    
    def _compute_diagonal_full(self, E_elem: torch.Tensor) -> torch.Tensor:
        """
        Diagonal of K on the full DOF space (for penalty method preconditioning).
        
        Args:
            E_elem: (batch, nel) element moduli
            
        Returns:
            diag_K: (batch, ndof) diagonal of global stiffness matrix
        """
        batch_size = E_elem.shape[0]
        diag_contrib = E_elem.unsqueeze(-1) * self.KE_diag.unsqueeze(0).unsqueeze(0)
        diag_contrib_flat = diag_contrib.reshape(batch_size, -1)
        diag_full = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        idx = self.edofMat_flat.unsqueeze(0).expand(batch_size, -1)
        diag_full.scatter_add_(1, idx, diag_contrib_flat)
        return diag_full
    
    def _solve_cg(
        self, E_elem: torch.Tensor, force_vector: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Solve K @ u = f using matrix-free Preconditioned CG with adjoint gradients.
        
        Args:
            E_elem: Element moduli (batch, nel) — must have requires_grad for training
            force_vector: Force vector (batch, ndof)
            batch_size: Number of samples
            
        Returns:
            u: Full displacement vector (batch, ndof)
        """
        f_free = force_vector[:, self.free_dofs]  # (batch, n_free)
        
        # Use custom autograd function for proper adjoint gradients
        u_free = FEMSolveFunction.apply(E_elem, f_free, self)
        
        # Reconstruct full displacement
        u = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        u[:, self.free_dofs] = u_free
        
        return u
    
    def _solve_direct(
        self, E_elem: torch.Tensor, force_vector: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Direct solve using sparse assembly + dense Cholesky.
        Fallback for small grids where CG overhead dominates.
        """
        K_batch = self._assemble_stiffness_batch(E_elem)
        K_free = K_batch[:, self.free_dofs][:, :, self.free_dofs]
        f_free = force_vector[:, self.free_dofs]
        
        u_free = torch.linalg.solve(K_free, f_free)
        
        u = torch.zeros(batch_size, self.ndof, dtype=self.dtype, device=self.device)
        u[:, self.free_dofs] = u_free
        
        return u
    
    def _assemble_stiffness_batch(self, E_elem: torch.Tensor) -> torch.Tensor:
        """Dense assembly (only used when use_direct_solve=True)."""
        batch_size = E_elem.shape[0]
        KE_scaled = self.KE.unsqueeze(0).unsqueeze(0) * E_elem.unsqueeze(-1).unsqueeze(-1)
        sK = KE_scaled.reshape(batch_size, -1)
        
        K_flat = torch.zeros(batch_size, self.ndof * self.ndof, dtype=self.dtype, device=self.device)
        flat_idx = self.iK * self.ndof + self.jK
        flat_idx = flat_idx.unsqueeze(0).expand(batch_size, -1)
        K_flat.scatter_add_(1, flat_idx, sK)
        
        return K_flat.view(batch_size, self.ndof, self.ndof)
    
    # ==================== Filter ====================
    
    def _apply_filter(self, density: torch.Tensor) -> torch.Tensor:
        """Apply density filter using sparse matrix multiplication."""
        H_sparse = torch.sparse_coo_tensor(
            self.H_indices, self.H_values,
            size=(self.nel, self.nel),
            device=self.device, dtype=self.dtype
        )
        x_filt = torch.sparse.mm(H_sparse, density.T).T
        x_filt = x_filt / self.Hs.unsqueeze(0)
        return x_filt
    
    # ==================== Pre-computation (identical to original) ====================
    
    def _compute_element_stiffness(self, nu: float) -> torch.Tensor:
        k = [
            1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
            -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8,
        ]
        KE = 1.0 / (1 - nu**2) * torch.tensor([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ], dtype=torch.float64)
        return KE
    
    def _compute_edof_map(self) -> torch.Tensor:
        nx, ny = self.nx, self.ny
        # Vectorized edof computation (no Python loop)
        elx = torch.arange(nx).unsqueeze(1).expand(nx, ny)  # (nx, ny)
        ely = torch.arange(ny).unsqueeze(0).expand(nx, ny)  # (nx, ny)
        
        n1 = ((ny + 1) * elx + ely).reshape(-1)  # Bottom-left node
        n2 = ((ny + 1) * (elx + 1) + ely).reshape(-1)  # Bottom-right node
        
        edofMat = torch.stack([
            2*n1, 2*n1+1, 2*n2, 2*n2+1,
            2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
        ], dim=1)  # (nel, 8)
        
        return edofMat
    
    def _compute_assembly_indices(self, edofMat):
        iK = edofMat.unsqueeze(2).expand(-1, -1, 8).reshape(-1)
        jK = edofMat.unsqueeze(1).expand(-1, 8, -1).reshape(-1)
        return iK, jK
    
    def _compute_filter(self, rmin: float):
        nx, ny = self.nx, self.ny
        nel = nx * ny
        r_ceil = int(math.ceil(rmin))
        
        rows, cols, vals = [], [], []
        
        for i in range(nx):
            for j in range(ny):
                row = i * ny + j
                i_min = max(i - r_ceil + 1, 0)
                i_max = min(i + r_ceil, nx)
                j_min = max(j - r_ceil + 1, 0)
                j_max = min(j + r_ceil, ny)
                
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        col = ii * ny + jj
                        dist = math.sqrt((i - ii)**2 + (j - jj)**2)
                        weight = rmin - dist
                        if weight > 0:
                            rows.append(row)
                            cols.append(col)
                            vals.append(weight)
        
        H_indices = torch.tensor([rows, cols], dtype=torch.long)
        H_values = torch.tensor(vals, dtype=torch.float64)
        Hs = torch.zeros(nel, dtype=torch.float64)
        Hs.scatter_add_(0, torch.tensor(rows, dtype=torch.long), H_values)
        
        return H_indices, H_values, Hs


# ==================== Convenience Functions ====================

def create_cantilever_solver_fast(
    nx: int = 64, ny: int = 32,
    device: str = 'cuda', dtype: torch.dtype = torch.float64,
    **kwargs,
) -> Tuple[TorchFEMSolverFast, torch.Tensor, int]:
    """Create a fast solver for the cantilever beam problem."""
    fixed_dofs = []
    for j in range(ny + 1):
        node = j
        fixed_dofs.extend([2 * node, 2 * node + 1])
    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device)
    
    solver = TorchFEMSolverFast(nx, ny, fixed_dofs, device=device, dtype=dtype, **kwargs)
    
    ndof = 2 * (nx + 1) * (ny + 1)
    force = torch.zeros(ndof, dtype=dtype, device=device)
    load_node = (ny + 1) * nx + ny // 2
    load_dof = 2 * load_node + 1
    force[load_dof] = -1.0
    
    return solver, force, load_dof


def create_mechanism_solver_fast(
    nx: int = 64, ny: int = 64,
    fixed_nodes=None, input_node=None, input_direction='x',
    output_node=None, output_direction='x',
    device: str = 'cuda', dtype: torch.dtype = torch.float64,
    **kwargs,
) -> Tuple[TorchFEMSolverFast, torch.Tensor, int, int]:
    """Create a fast solver for mechanism design problems."""
    if fixed_nodes is None:
        fixed_nodes = [(0, 0), (0, ny)]
    if input_node is None:
        input_node = (0, ny // 2)
    if output_node is None:
        output_node = (nx, ny // 2)
    
    fixed_dofs = []
    for (x, y) in fixed_nodes:
        node = (ny + 1) * x + y
        fixed_dofs.extend([2 * node, 2 * node + 1])
    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device)
    
    solver = TorchFEMSolverFast(nx, ny, fixed_dofs, device=device, dtype=dtype, **kwargs)
    
    ndof = 2 * (nx + 1) * (ny + 1)
    force = torch.zeros(ndof, dtype=dtype, device=device)
    input_node_idx = (ny + 1) * input_node[0] + input_node[1]
    input_dof = 2 * input_node_idx + (0 if input_direction == 'x' else 1)
    force[input_dof] = 1.0
    
    output_node_idx = (ny + 1) * output_node[0] + output_node[1]
    output_dof = 2 * output_node_idx + (0 if output_direction == 'x' else 1)
    
    return solver, force, input_dof, output_dof


# ==================== Testing ====================

def quick_example():
    """Run one simple cantilever FEM forward/backward example."""
    import matplotlib.pyplot as plt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if device == 'cuda' else torch.float64

    nx, ny = 64, 32
    solver, force, load_dof = create_cantilever_solver_fast(nx, ny, device=device, dtype=dtype)

    density = torch.full((nx * ny,), 0.5, dtype=dtype, device=device, requires_grad=True)

    compliance, u = solver.forward_compliance(density, force, return_displacement=True)
    compliance.backward()

    print("=" * 70)
    print("TorchFEMSolverFast quick example")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Grid: {nx}x{ny}, elements: {nx * ny}, dofs: {solver.ndof}")
    print(f"Compliance: {compliance.item():.8f}")
    print(f"Load DOF displacement: {u[load_dof].item():.8e}")
    print(f"Density grad norm: {density.grad.norm().item():.8e}")
    print("=" * 70)

    # One figure with two views: density field and displacement magnitude.
    density_img = density.detach().reshape(nx, ny).T.cpu().numpy()
    ux = u[0::2].detach().reshape(nx + 1, ny + 1).T.cpu().numpy()
    uy = u[1::2].detach().reshape(nx + 1, ny + 1).T.cpu().numpy()
    u_mag = (ux ** 2 + uy ** 2) ** 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(density_img, origin='lower', cmap='gray', aspect='auto')
    axes[0].set_title('Density (cantilever)')
    axes[0].set_xlabel('x element index')
    axes[0].set_ylabel('y element index')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(u_mag, origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title('Displacement magnitude |u|')
    axes[1].set_xlabel('x node index')
    axes[1].set_ylabel('y node index')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()


if __name__ == "__main__":
    quick_example()
