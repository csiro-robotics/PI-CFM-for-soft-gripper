"""Novelty Search + local competition behaviour archive for NSLC.

Two scores per design, both in ~[0,1], over a growing behaviour archive whose
features are downsampled masks (default 32x16, so the k-NN stays cheap):

  * NOVELTY      = how visually DIFFERENT a gripper looks: mean distance from its
                   feature to its k nearest neighbours (Lehman & Stanley 2011).
                   Drives EXPLORATION.
  * LOCAL COMP.  = fraction of its k behavioural neighbours whose grasp quality
                   it EXCEEDS.  Drives QUALITY *locally* (be the best among
                   look-alikes) without a global pressure that collapses variety.

NSLC objective = novelty + w * local_competition.  This is the proper two-axis
NSLC (Lehman & Stanley 2011): unlike summing novelty with a raw quality term —
where novelty (~1 for any fresh cell) drowns quality — local competition stays
informative everywhere, so good grippers spread across the whole map.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


class NoveltyArchive:
    def __init__(self, gh=32, gw=16, k=15, cap=5000, seed=0):
        self.gh, self.gw, self.k, self.cap = gh, gw, k, cap
        self.dim = gh * gw
        self.feats = None                       # (N, gh*gw) float32 in [0,1]
        self.quals = None                       # (N,) float32 grasp quality per stored design
        self.rng = np.random.default_rng(seed)

    def _feat(self, masks):
        """Each (H,W) mask -> (gh*gw) block-mean feature in [0,1]."""
        out = np.empty((len(masks), self.dim), np.float32)
        for i, m in enumerate(masks):
            m = np.asarray(m, np.float32)
            H, W = m.shape
            ph, pw = H // self.gh, W // self.gw
            d = m[:self.gh * ph, :self.gw * pw].reshape(self.gh, ph, self.gw, pw).mean((1, 3))
            out[i] = d.ravel()
        return out

    def score(self, masks):
        """Novelty per mask in ~[0,1] (RMS per-cell distance to k nearest seen)."""
        return self.score_feats(self._feat(masks))

    def score_feats(self, f):
        """Same as score() but on precomputed features (gh*gw). Lets a multi-GPU
        worker ship the compact 512-d feature instead of the full mask."""
        f = np.asarray(f, np.float32)
        if self.feats is None or self.feats.shape[0] < 2:
            return np.ones(len(f), np.float32)              # nothing to compare -> all novel
        D = cdist(f, self.feats)                            # (B, N) euclidean
        D.sort(axis=1)
        kk = min(self.k, self.feats.shape[0])
        return (D[:, :kk].mean(axis=1) / np.sqrt(self.dim)).astype(np.float32)   # normalise to ~[0,1]

    def local_competition(self, masks, quals):
        """Fraction of each design's k behavioural-NN whose stored grasp quality
        it EXCEEDS, in [0,1] (NSLC's local competition).  1.0 until the archive
        has anyone to compete against."""
        if self.feats is None or self.feats.shape[0] < 1 or self.quals is None:
            return np.ones(len(masks), np.float32)
        f = self._feat(masks)
        q = np.asarray(quals, np.float32)
        kk = min(self.k, self.feats.shape[0])
        idx = np.argpartition(cdist(f, self.feats), kk - 1, axis=1)[:, :kk]  # k nearest
        neigh_q = self.quals[idx]                                           # (B, kk)
        return (q[:, None] > neigh_q).mean(axis=1).astype(np.float32)

    def add(self, masks, quals=None):
        if not len(masks):
            return
        self.add_feats(self._feat(masks), quals)

    def add_feats(self, f, quals=None):
        """Same as add() but on precomputed features (for the multi-GPU worker)."""
        if not len(f):
            return
        f = np.asarray(f, np.float32)
        q = (np.asarray(quals, np.float32) if quals is not None
             else np.zeros(len(f), np.float32))
        self.feats = f if self.feats is None else np.vstack([self.feats, f])
        self.quals = q if self.quals is None else np.concatenate([self.quals, q])
        if self.feats.shape[0] > self.cap:                  # bound cost: random subsample to cap
            idx = self.rng.choice(self.feats.shape[0], self.cap, replace=False)
            self.feats = self.feats[idx]; self.quals = self.quals[idx]

    def __len__(self):
        return 0 if self.feats is None else self.feats.shape[0]
