"""Grasp metrics (numpy) — the values the optimizer reads.

Ports `_instant_obstacle_force` (the pull-off / contact force on the disc) and
the contact-arc length from torch_2D_FEM. These are per-saved-frame reductions
over the boundary nodes (not per-substep), so numpy is fine. Single shared disc.

Force on disc = -sum_node f_barrier(node)  (Newton's 3rd law). The metric-only
`d_report_floor` caps the 1/d barrier singularity (matches the optimizer fix).
"""
from __future__ import annotations
import numpy as np


def _barrier_Bp(d, d_hat):
    ds = np.maximum(d, 1e-9)
    u = ds - d_hat
    logr = np.log(np.maximum(ds / d_hat, 1e-12))
    return -2.0 * u * logr - (u * u) / ds


def obstacle_force(b_pos, boundary2env, n_envs, center, radius, stiffness, d_hat,
                   d_report_floor=None):
    """Per-env reaction force ON the disc, (n_envs, 2)."""
    diff = b_pos - np.asarray(center)                       # (n_b, 2)
    dc = np.sqrt((diff * diff).sum(-1) + 1e-24)
    d = dc - radius
    active = (d > 0.0) & (d < d_hat)
    Bp = np.where(active, _barrier_Bp(d, d_hat), 0.0)
    nh = diff / dc[:, None]
    f_node = -stiffness * Bp[:, None] * nh                  # force ON node
    on_disc = -f_node                                       # force ON disc
    if d_report_floor is not None:                          # cap barrier singularity
        fcap = stiffness * abs(_barrier_Bp(np.array(d_report_floor), d_hat))
        mag = np.linalg.norm(on_disc, axis=-1, keepdims=True)
        on_disc = on_disc * np.minimum(fcap / np.maximum(mag, 1e-30), 1.0)
    forces = np.zeros((n_envs, 2))
    np.add.at(forces, np.asarray(boundary2env), on_disc)
    return forces


def contact_arc(pos, boundary_nodes, boundary_edges, node2env, n_envs,
                center, radius, d_hat):
    """Per-env contact arc length (m), (n_envs,). Sum of boundary-edge lengths
    in contact: full L if both endpoints active, 0.5 L if one — matches torch
    obstacle_contact_summary. 'Active' = clearance d in (0, d_hat)."""
    pos = np.asarray(pos)
    bn = np.asarray(boundary_nodes)
    diff = pos[bn] - np.asarray(center)
    d = np.sqrt((diff * diff).sum(-1) + 1e-24) - radius
    active = (d > 0.0) & (d < d_hat)
    active_g = np.zeros(pos.shape[0], bool)
    active_g[bn[active]] = True
    e = np.asarray(boundary_edges)
    arcs = np.zeros(n_envs)
    if len(e) == 0:
        return arcs
    a_in = active_g[e[:, 0]]; b_in = active_g[e[:, 1]]
    L = np.linalg.norm(pos[e[:, 1]] - pos[e[:, 0]], axis=1)
    contrib = np.where(a_in & b_in, L, np.where(a_in | b_in, 0.5 * L, 0.0))
    np.add.at(arcs, np.asarray(node2env)[e[:, 0]], contrib)
    return arcs
