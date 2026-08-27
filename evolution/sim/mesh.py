"""Self-contained triangle-mesh build (numpy) — no torch dependency.

Ports torch_2D_FEM/dynamic/mesh.py (build_triangle_grid + build_multi_env_mesh)
+ the lumped mass / Lame material from multi_env.py. Each quad on the (nx, ny)
grid splits into 2 triangles; node id = (ny+1)*ix + iy. Void quads (mask False)
are dropped and unused nodes pruned. Output is a flat multi-env mesh ready for
the Warp kernels.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np


def lame(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


@dataclass
class Mesh:
    n_envs: int
    nx: int
    ny: int
    rest_pos: np.ndarray         # (sum_N, 2)
    tri: np.ndarray              # (sum_n_tri, 3) global node ids
    D_m_inv: np.ndarray          # (sum_n_tri, 2, 2)
    rest_area: np.ndarray        # (sum_n_tri,)
    mu: np.ndarray               # (sum_n_tri,)
    lam: np.ndarray              # (sum_n_tri,)
    mass: np.ndarray             # (sum_N,)
    node2env: np.ndarray         # (sum_N,)
    tri2env: np.ndarray          # (sum_n_tri,)
    boundary_nodes: np.ndarray   # (sum_n_b,) global
    boundary2env: np.ndarray     # (sum_n_b,)
    boundary_edges: np.ndarray   # (sum_n_e, 2) global node ids
    node_starts: np.ndarray      # (n_envs+1,)
    tri_starts: np.ndarray       # (n_envs+1,)
    boundary_starts: np.ndarray  # (n_envs+1,)
    boundary_per_env: np.ndarray # (n_envs,) n_b per env
    excl_flat: np.ndarray        # concatenated per-env (n_b_e^2,) uint8 masks
    excl_offset: np.ndarray      # (n_envs+1,) offsets into excl_flat
    excluded_pairs_per_env: list # list of (n_b_e, n_b_e) bool
    grid_to_live_node_per_env: list
    # connected components of each env's socketed mask, filled in by
    # finger.build_multi_env. >1 means the design decoded in pieces.
    n_components: np.ndarray | None = None
    tri_quad: np.ndarray = None  # (sum_n_tri,) per-env-LOCAL grid-quad idx of each surviving tri (decode: ix=q//ny, iy=q%ny)

    @property
    def sum_N(self): return self.rest_pos.shape[0]
    @property
    def sum_n_tri(self): return self.tri.shape[0]

    def node_index(self, env, ix, iy):
        flat = (self.ny + 1) * ix + iy
        loc = int(self.grid_to_live_node_per_env[env][flat])
        return -1 if loc < 0 else int(self.node_starts[env]) + loc


def _build_one(nx, ny, Lx, Ly, center, mask, k_excl):
    n_nodes = (nx + 1) * (ny + 1)
    n_quads = nx * ny
    xs = np.linspace(0.0, Lx, nx + 1) + (center[0] - 0.5 * Lx)
    ys = np.linspace(0.0, Ly, ny + 1) + (center[1] - 0.5 * Ly)
    gx = np.broadcast_to(xs[:, None], (nx + 1, ny + 1))
    gy = np.broadcast_to(ys[None, :], (nx + 1, ny + 1))
    rest_pos = np.stack([gx, gy], -1).reshape(-1, 2).astype(np.float64)

    ix = np.repeat(np.arange(nx), ny)          # quad idx = ix*ny + iy
    iy = np.tile(np.arange(ny), nx)
    bl = (ny + 1) * ix + iy; tl = bl + 1; br = bl + (ny + 1); tr = br + 1
    tri1 = np.stack([bl, tl, tr], 1); tri2 = np.stack([bl, tr, br], 1)
    tri = np.stack([tri1, tri2], 1).reshape(-1, 3)       # (2*n_quads, 3)
    quad_of_tri = np.repeat(np.arange(n_quads), 2)

    if mask is None:
        active = np.ones(n_quads, bool)
    else:
        mask = np.asarray(mask, bool)
        if mask.shape != (nx, ny):
            raise ValueError(f"mask must be ({nx},{ny}); got {mask.shape}")
        active = mask.reshape(-1)
    active_tri = active[quad_of_tri]
    tri = tri[active_tri]
    tri_quad = quad_of_tri[active_tri]           # per-env-local grid-quad idx of each surviving tri

    used = np.zeros(n_nodes, bool); used[tri.reshape(-1)] = True
    used_ids = np.nonzero(used)[0]
    remap = np.full(n_nodes, -1, np.int64)
    remap[used_ids] = np.arange(len(used_ids))
    tri = remap[tri]
    rest_pos = rest_pos[used_ids]

    p = rest_pos[tri]                                     # (n_tri,3,2)
    e1 = p[:, 1] - p[:, 0]; e2 = p[:, 2] - p[:, 0]
    D_m = np.stack([e1, e2], -1)                          # (n_tri,2,2) cols e1,e2
    D_m_inv = np.linalg.inv(D_m)
    rest_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    # boundary edges = edges in exactly one triangle
    ec = {}
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(int(a), int(b)), max(int(a), int(b)))
            ec[k] = ec.get(k, 0) + 1
    bpairs = [k for k, v in ec.items() if v == 1]
    bset = sorted({n for e in bpairs for n in e})
    perim = np.array(bset, np.int64)

    # excluded self-collision pairs via mesh-graph BFS up to k_excl hops
    n_live = len(used_ids)
    adj = {i: set() for i in range(n_live)}
    for a, b, c in tri:
        a, b, c = int(a), int(b), int(c)
        adj[a] |= {b, c}; adj[b] |= {a, c}; adj[c] |= {a, b}
    nb = len(bset); idx_of = {n: k for k, n in enumerate(bset)}
    excl = np.eye(nb, dtype=bool) if nb else np.zeros((0, 0), bool)
    for src in bset:
        seen = {src: 0}; q = deque([src])
        while q:
            u = q.popleft(); d = seen[u]
            if d == k_excl:
                continue
            for w in adj[u]:
                if w not in seen:
                    seen[w] = d + 1; q.append(w)
        si = idx_of[src]
        for w in seen:
            if w in idx_of:
                excl[si, idx_of[w]] = True
    if nb:
        excl = excl | excl.T
    bedges = np.array(bpairs, np.int64) if bpairs else np.zeros((0, 2), np.int64)
    return dict(rest_pos=rest_pos, tri=tri, D_m_inv=D_m_inv, rest_area=rest_area,
                boundary=perim, bedges=bedges, excl=excl, grid_to_live=remap,
                n_nodes=n_live, tri_quad=tri_quad)


def build_mesh(masks, nx, ny, physical_size, center, young, nu, rho, k_excl=2):
    Lx, Ly = physical_size
    per = [_build_one(nx, ny, Lx, Ly, center, m, k_excl) for m in masks]
    nodes_pe = np.array([p["n_nodes"] for p in per])
    tri_pe = np.array([len(p["rest_area"]) for p in per])
    b_pe = np.array([len(p["boundary"]) for p in per])
    cz = lambda a: np.concatenate([[0], np.cumsum(a)])
    node_starts, tri_starts, b_starts = cz(nodes_pe), cz(tri_pe), cz(b_pe)

    rest_pos = np.concatenate([p["rest_pos"] for p in per], 0)
    D_m_inv = np.concatenate([p["D_m_inv"] for p in per], 0)
    rest_area = np.concatenate([p["rest_area"] for p in per], 0)
    tri = np.concatenate([p["tri"] + node_starts[b] for b, p in enumerate(per)], 0)
    boundary = np.concatenate([p["boundary"] + node_starts[b] for b, p in enumerate(per)], 0)
    be_list = [p["bedges"] + node_starts[b] for b, p in enumerate(per) if len(p["bedges"])]
    boundary_edges = np.concatenate(be_list, 0) if be_list else np.zeros((0, 2), np.int64)
    node2env = np.concatenate([np.full(p["n_nodes"], b) for b, p in enumerate(per)])
    tri2env = np.concatenate([np.full(len(p["rest_area"]), b) for b, p in enumerate(per)])
    tri_quad = np.concatenate([p["tri_quad"] for p in per], 0)   # per-env-local grid-quad idx / surviving tri
    boundary2env = np.concatenate([np.full(len(p["boundary"]), b) for b, p in enumerate(per)])

    excls = [p["excl"] for p in per]
    excl_flat = (np.concatenate([e.reshape(-1).astype(np.uint8) for e in excls])
                 if excls else np.zeros(0, np.uint8))
    excl_offset = np.concatenate([[0], np.cumsum([e.size for e in excls])]).astype(np.int64)

    mu_s, lam_s = lame(young, nu)
    mu = np.full(len(rest_area), mu_s)
    lam = np.full(len(rest_area), lam_s)
    mass = np.zeros(rest_pos.shape[0])
    contrib = (rho * rest_area / 3.0)
    np.add.at(mass, tri.reshape(-1), np.repeat(contrib, 3))
    mass = np.clip(mass, 1e-30, None)

    return Mesh(n_envs=len(masks), nx=nx, ny=ny, rest_pos=rest_pos, tri=tri,
                D_m_inv=D_m_inv, rest_area=rest_area, mu=mu, lam=lam, mass=mass,
                node2env=node2env, tri2env=tri2env, boundary_nodes=boundary,
                boundary2env=boundary2env, boundary_edges=boundary_edges,
                node_starts=node_starts,
                tri_starts=tri_starts, boundary_starts=b_starts,
                boundary_per_env=b_pe, excl_flat=excl_flat, excl_offset=excl_offset,
                excluded_pairs_per_env=excls,
                grid_to_live_node_per_env=[p["grid_to_live"] for p in per],
                tri_quad=tri_quad)
