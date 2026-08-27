"""
Physics engine construction for PICFM training.

Two modes:
  1. setup_physics_engine()        — from a dataset .npz file (one-time setup)
  2. setup_physics_from_condition() — from a condition tensor (per-sample, fast)
"""

import numpy as np
import torch

from flows.residual_physics import ResidualMechanics
from fem_solver.torch_fem_solver_fast import TorchFEMSolverFast


# ---------------------------------------------------------------------------
# Build physics from a .npz file (used for initial / standalone tests)
# ---------------------------------------------------------------------------

def setup_physics_engine(
    sample_data_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    w_mechanism: float = 1.0,
    w_binary: float = 0.5,
    E_min: float = 1e-4,
    penal: float = 5.0,
    rmin: float = 0.5,
) -> ResidualMechanics:
    """
    Create a ResidualMechanics engine from a representative dataset sample.

    Uses **displacement-controlled** input matching the dataset convention:
    ``inputx``/``inputy`` store prescribed displacement values (= disp_mag),
    NOT forces.  The input DOFs are added to the solver's fixed_dofs and
    treated as non-zero Dirichlet BCs.
    """
    data = np.load(sample_data_path)

    # Grid dimensions — dataset stores (ny, nx) images
    ny, nx = data["geometry"].shape  # (128, 64) → nx=64, ny=128

    # --- Extract fixed DOFs from BC masks ---
    bc_x = data["BCx"]
    bc_y = data["BCy"]

    fixed_dofs = []
    for j in range(ny):
        for i in range(nx):
            node = i * (ny + 1) + j
            if bc_x[j, i] > 0.5:
                fixed_dofs.append(2 * node)
            if bc_y[j, i] > 0.5:
                fixed_dofs.append(2 * node + 1)

    # --- Extract prescribed input displacements ---
    input_x = data["inputx"]
    input_y = data["inputy"]

    prescribed_dofs = []
    prescribed_values = []

    for j in range(ny):
        for i in range(nx):
            if abs(input_x[j, i]) > 1e-8 or abs(input_y[j, i]) > 1e-8:
                n_bl = (ny + 1) * i + j
                n_br = (ny + 1) * (i + 1) + j
                n_tl = (ny + 1) * i + j + 1
                n_tr = (ny + 1) * (i + 1) + j + 1
                for node in [n_bl, n_br, n_tl, n_tr]:
                    if abs(input_x[j, i]) > 1e-8:
                        dof = 2 * node
                        if dof not in fixed_dofs:
                            fixed_dofs.append(dof)
                            prescribed_dofs.append(dof)
                            prescribed_values.append(float(input_x[j, i]))
                    if abs(input_y[j, i]) > 1e-8:
                        dof = 2 * node + 1
                        if dof not in fixed_dofs:
                            fixed_dofs.append(dof)
                            prescribed_dofs.append(dof)
                            prescribed_values.append(float(input_y[j, i]))

    # Deduplicate prescribed DOFs
    seen = {}
    for d, v in zip(prescribed_dofs, prescribed_values):
        if d not in seen:
            seen[d] = v
    prescribed_dofs = list(seen.keys())
    prescribed_values = list(seen.values())

    if len(fixed_dofs) == 0:
        node_bl = 0
        node_tl = ny
        for n in [node_bl, node_tl]:
            fixed_dofs.extend([2 * n, 2 * n + 1])

    # --- Extract output DOF ---
    output_x = data["outputx"]
    output_y = data["outputy"]

    out_mask = (np.abs(output_x) + np.abs(output_y)) > 1e-8
    out_ys, out_xs = np.where(out_mask)
    if len(out_xs) > 0:
        oi = int(np.mean(out_xs))
        oj = int(np.mean(out_ys))
        out_node = oi * (ny + 1) + oj
        if abs(output_x[oj, oi]) >= abs(output_y[oj, oi]):
            output_dof = 2 * out_node
            output_sign = float(np.sign(output_x[oj, oi]))
        else:
            output_dof = 2 * out_node + 1
            output_sign = float(np.sign(output_y[oj, oi]))
    else:
        out_node = nx * (ny + 1) + ny // 2
        output_dof = 2 * out_node
        output_sign = 1.0

    if output_dof not in fixed_dofs:
        fixed_dofs.append(output_dof)

    fixed_dofs = torch.tensor(sorted(set(fixed_dofs)), dtype=torch.long, device=device)
    prescribed_dofs_t = torch.tensor(prescribed_dofs, dtype=torch.long, device=device)
    prescribed_values_t = torch.tensor(prescribed_values, dtype=dtype, device=device)

    vol_frac = float(data.get("volume_fraction", 0.3))
    disp_mag = float(data.get("disp_mag", 1.0))
    ndof = 2 * (nx + 1) * (ny + 1)

    solver = TorchFEMSolverFast(
        nx=nx, ny=ny,
        fixed_dofs=fixed_dofs,
        penal=penal,
        rmin=rmin if rmin is not None else float(data.get("rmin", 1.5)),
        E_min=E_min,
        device=device, dtype=dtype,
    )

    engine = ResidualMechanics(
        solver=solver,
        prescribed_dofs=prescribed_dofs_t,
        prescribed_values=prescribed_values_t,
        output_dof=output_dof,
        output_sign=output_sign,
        w_mechanism=w_mechanism,
        w_binary=w_binary,
    )

    actual_rmin = rmin if rmin is not None else float(data.get("rmin", 1.5))

    print(f"[Physics Engine] Grid: {nx}×{ny}  ({nx * ny} elements, {ndof} DOFs)")
    print(f"  Fixed DOFs (total): {len(fixed_dofs)}")
    print(f"  Prescribed DOFs (input): {len(prescribed_dofs)}")
    print(f"  Prescribed disp value: {prescribed_values[0]:.4f}  (disp_mag={disp_mag:.4f})")
    print(f"  Output DOF: {output_dof}  output_sign: {output_sign:+.0f}")
    print(f"  penal: {penal} (SIMP),  rmin: {actual_rmin:.1f},  E_min: {E_min:.1e}")

    return engine


# ---------------------------------------------------------------------------
# Per-sample physics from condition tensor (no file I/O, ~0.2 ms)
# ---------------------------------------------------------------------------

def setup_physics_from_condition(
    cond: torch.Tensor,
    base_solver: TorchFEMSolverFast,
    device: str = "cuda",
    w_mechanism: float = 1.0,
    w_binary: float = 1.0,
) -> ResidualMechanics:
    """
    Build a per-sample physics engine from the condition tensor.

    Extracts BCs from the first 6 channels of the condition tensor
    (BCx, BCy, inputx, inputy, outputx, outputy).  Channels 6+ (design-space
    blend weights) are ignored by physics — they only affect generation style.
    Creates a lightweight solver clone via ``base_solver.clone_with_dofs()``
    and wraps it in a ``ResidualMechanics`` engine.

    Args:
        cond: (C, ny, nx) condition tensor — channels 0-5 are BCs,
              channels 6+ are design blend weights (ignored by physics).
        base_solver: Pre-built solver whose grid/material/filter are reused.
        device: Target device string.
        w_mechanism, w_binary: Loss sub-weights.

    Returns:
        ResidualMechanics engine configured for this sample's BCs.
    """
    # Stay on GPU — avoid costly CPU round-trips
    cond = cond.detach().to(device)
    ny, nx = cond.shape[1], cond.shape[2]

    bc_x  = cond[0]
    bc_y  = cond[1]
    inp_x = cond[2]
    inp_y = cond[3]
    out_x = cond[4]
    out_y = cond[5]

    # --- Fixed DOFs from BC masks (all on GPU) ---
    js, is_ = torch.where(bc_x > 0.5)
    fixed_x = 2 * (is_ * (ny + 1) + js)
    js, is_ = torch.where(bc_y > 0.5)
    fixed_y = 2 * (is_ * (ny + 1) + js) + 1
    fixed_dofs_parts = [fixed_x, fixed_y]

    # --- Prescribed input displacements ---
    prescribed_dofs_parts = []
    prescribed_vals_parts = []

    for ch, dof_offset in [(inp_x, 0), (inp_y, 1)]:
        mask = ch.abs() > 1e-8
        if not mask.any():
            continue
        js, is_ = torch.where(mask)
        vals = ch[js, is_]
        n_bl = (ny + 1) * is_ + js
        n_br = (ny + 1) * (is_ + 1) + js
        n_tl = (ny + 1) * is_ + (js + 1)
        n_tr = (ny + 1) * (is_ + 1) + (js + 1)
        all_nodes = torch.stack([n_bl, n_br, n_tl, n_tr], dim=1).reshape(-1)
        all_vals  = vals.unsqueeze(1).expand(-1, 4).reshape(-1)
        all_dofs  = 2 * all_nodes + dof_offset
        prescribed_dofs_parts.append(all_dofs)
        prescribed_vals_parts.append(all_vals)
        fixed_dofs_parts.append(all_dofs)

    # --- Output DOF ---
    out_mask = (out_x.abs() + out_y.abs()) > 1e-8
    out_js, out_is = torch.where(out_mask)
    if len(out_js) > 0:
        oi = int(out_is.float().mean())
        oj = int(out_js.float().mean())
        out_node = oi * (ny + 1) + oj
        if abs(float(out_x[oj, oi])) >= abs(float(out_y[oj, oi])):
            output_dof = 2 * out_node
            output_sign = float(torch.sign(out_x[oj, oi]))
        else:
            output_dof = 2 * out_node + 1
            output_sign = float(torch.sign(out_y[oj, oi]))
    else:
        out_node = nx * (ny + 1) + ny // 2
        output_dof = 2 * out_node
        output_sign = 1.0

    fixed_dofs_parts.append(torch.tensor([output_dof], dtype=torch.long, device=device))

    # --- Combine & deduplicate (stay on GPU) ---
    all_fixed = torch.unique(torch.cat(fixed_dofs_parts)).to(
        dtype=torch.long, device=device
    )

    if prescribed_dofs_parts:
        p_dofs = torch.cat(prescribed_dofs_parts)
        p_vals = torch.cat(prescribed_vals_parts)
        uniq, inv = torch.unique(p_dofs, return_inverse=True)
        uniq_vals = torch.zeros(uniq.shape[0], dtype=p_vals.dtype, device=device)
        uniq_vals.scatter_(0, inv, p_vals)
        p_dofs_final = uniq.to(dtype=torch.long, device=device)
        p_vals_final = uniq_vals.to(dtype=base_solver.dtype, device=device)
    else:
        p_dofs_final = torch.tensor([], dtype=torch.long, device=device)
        p_vals_final = torch.tensor([], dtype=base_solver.dtype, device=device)

    solver = base_solver.clone_with_dofs(all_fixed)

    return ResidualMechanics(
        solver=solver,
        prescribed_dofs=p_dofs_final,
        prescribed_values=p_vals_final,
        output_dof=output_dof,
        output_sign=output_sign,
        w_mechanism=w_mechanism,
        w_binary=w_binary,
    )
