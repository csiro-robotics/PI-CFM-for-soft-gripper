"""CVT-MAP-Elites over the PI-CFM genome, LM-MA-ES emitters, implicit grasp physics.

    genome --PI-CFM--> mask --implicit PNCG-IPC--> (score, 4 descriptors) --> archive

The archive is a 4-D CVT tessellation of the DESCRIPTOR space

    [strut_complexity, branch_density, hole_count, material_fraction]

and the objective a novelty-dominant blend

    objective = 1 + novelty_weight * novelty + w_grasp * composite      (valid)
              = 0                                                       (invalid)

Novelty is the mean distance from a design's 32x16 block-mean image to its k
nearest neighbours in a growing behaviour archive, so it rewards looking unlike
anything seen so far; the +1 keeps every valid design above the archive's
threshold_min. Weighting novelty above grasp is deliberate: the point of the run
is to ILLUMINATE the space of gripper morphologies, and a pure grasp objective
collapses onto one family.

This is the single-GPU form of the 4-GPU run in the paper. The algorithm is
identical -- genome -> mask is a pure function of (genome, x0_seed), so which GPU
evaluates a design cannot change it. Only the batching differs: the population is
evaluated in `--chunk`-sized solver launches instead of one shard per GPU, which
is what lets a desktop card run the paper's 64 x 32 configuration.

    python evolution/map_elites.py --iterations 600 --cells 2000
    python evolution/map_elites.py --smoke          # 2 tiny iterations, minutes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "sim"), str(_HERE / "generate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ribs.archives import CVTArchive                       # noqa: E402
from ribs.emitters import EvolutionStrategyEmitter          # noqa: E402
from ribs.schedulers import Scheduler                       # noqa: E402

from novelty import NoveltyArchive                          # noqa: E402

# 4-D descriptor space. Ranges hug the observed manifold rather than boxing the
# theoretical extremes: a tight range spreads designs over more distinct cells, so
# coverage reads as "fraction of the REACHABLE space illuminated".
AXES = ["strut_complexity", "branch_density", "hole_count", "material_fraction"]
RANGES = [(0.12, 0.71), (0.02, 0.22), (0.0, 28.0), (0.30, 0.66)]

GH, GW = 32, 16          # novelty feature grid
EXTRA_FIELDS = {
    "novelty": ((), np.float32), "grasp": ((), np.float32),
    "force_term": ((), np.float32), "wrap_term": ((), np.float32),
    "pull_off": ((), np.float32), "arc": ((), np.float32), "ncon": ((), np.int32),
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- search ---
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--n-emitters", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32,
                   help="per emitter; population per iteration = n_emitters * this")
    p.add_argument("--cells", type=int, default=2000, help="CVT cells over the 4-D space")
    p.add_argument("--cvt-seed", type=int, default=0,
                   help="seed for the CVT tessellation (k-means). Same seed + same "
                        "--cells => identical cells, so coverage is comparable across runs")
    p.add_argument("--sigma0", type=float, default=0.8)
    p.add_argument("--es", choices=["lm_ma_es", "cma_es", "sep_cma_es", "openai_es"],
                   default="lm_ma_es",
                   help="LM-MA-ES is the default: at D=572 a full CMA covariance is "
                        "O(D^2) per emitter, LM-MA-ES keeps a low-rank approximation")
    p.add_argument("--learning-rate", type=float, default=1.0,
                   help="CVTArchive threshold learning rate. 1.0 = plain CMA-ME (greedy); "
                        "<1 (e.g. 0.05) = CMA-MAE, each cell's admission bar anneals upward")
    # --- objective ---
    p.add_argument("--novelty-k", type=int, default=15)
    p.add_argument("--novelty-weight", type=float, default=1.0,
                   help="0.0 makes the run QUALITY-driven: objective = 1 + w_grasp*composite")
    p.add_argument("--w-grasp", type=float, default=0.4)
    p.add_argument("--w-force", type=float, default=0.2)
    p.add_argument("--w-wrap", type=float, default=0.8)
    # --- genome / generation ---
    p.add_argument("--n-sites", type=int, default=12, help="Voronoi sites K (theta is 5K long)")
    p.add_argument("--x0-seed", type=int, default=12345,
                   help="seed of the FROZEN base noise. Fixed across the run so the "
                        "genome->design map stays deterministic")
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--ode-steps", type=int, default=100,
                   help="flow-ODE steps. Adaptive CFG carries a per-step EMA, so changing "
                        "this changes the generative process, not just its accuracy")
    p.add_argument("--checkpoint", default=None, help="PI-CFM weights (default: checkpoints/picfm_dit.pt)")
    # --- physics ---
    p.add_argument("--gap-mm", type=float, default=None, help="two-finger-equivalent opening")
    p.add_argument("--object-r-mm", type=float, default=None)
    p.add_argument("--object-y-mm", type=float, default=None,
                   help="disc-centre height (mm); gallery_v3 used 83")
    p.add_argument("--pull-mm", type=float, default=None)
    p.add_argument("--drop-islands", action="store_true",
                   help="simulate ONLY the piece of a design attached to the socket. Off "
                        "by default: no repair step means the mask is simulated exactly "
                        "as the model produced it")
    p.add_argument("--chunk", type=int, default=256,
                   help="designs per solver launch. Lower it if the GPU runs out of memory; "
                        "it changes throughput only, never the result")
    # --- run management ---
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="save a full archive snapshot every N iterations (1 tracks every "
                        "iteration, which is what the evolution animations replay)")
    p.add_argument("--resume-from", default=None,
                   help="path to a prior archive.npz to warm-start from. Its elites are "
                        "re-added and each emitter restarts near a random elite. Emitter "
                        "covariance and novelty history do NOT carry over")
    p.add_argument("--smoke", action="store_true",
                   help="2 iterations, 4 emitters x 2, 50 cells -- an end-to-end check")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.iterations, args.n_emitters, args.batch_size, args.cells = 2, 4, 2, 50
        args.snapshot_every = 1

    import evaluate
    import genome as gnm

    cfg, cfg_info = evaluate.make_cfg(
        gap_mm=evaluate.GAP_MM if args.gap_mm is None else args.gap_mm,
        object_r_mm=evaluate.OBJECT_R_MM if args.object_r_mm is None else args.object_r_mm,
        object_y_mm=evaluate.OBJECT_Y_MM if args.object_y_mm is None else args.object_y_mm,
        pull_mm=evaluate.PULL_MM if args.pull_mm is None else args.pull_mm,
        drop_islands=args.drop_islands)

    ckpt = args.checkpoint or gnm.DEFAULT_CKPT
    gen, H, W = gnm.make_generator(checkpoint=ckpt, device=args.device)
    K = args.n_sites
    noise_dim = (H // gnm.PATCH) * (W // gnm.PATCH)
    D = 5 * K + noise_dim

    run_dir = Path(args.run_dir or f"runs/me_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    pop = args.n_emitters * args.batch_size

    print(f"[me] design space {H}x{W} | genome D={D} (theta {5*K} + z_token {noise_dim})")
    print(f"[me] CVT {args.cells} cells over {AXES}")
    print(f"[me] {args.es} x {args.n_emitters} emitters x {args.batch_size} = {pop}/iter, "
          f"{args.iterations} iters, chunk {args.chunk}")
    print(f"[me] objective = 1 + {args.novelty_weight}*novelty(k={args.novelty_k}) "
          f"+ {args.w_grasp}*({args.w_force}*force + {args.w_wrap}*wrap)")
    print(f"[me] physics E={cfg.young/1e6:.4f}MPa mu={cfg.obstacle_friction:.4f} "
          f"dt={evaluate.DT*1e3:g}ms close={evaluate.CLOSE_T}s pull={evaluate.PULL_T}s/"
          f"{cfg.pull_distance*1e3:g}mm disc r={cfg.object_r*1e3:g}mm@y={cfg.object_y*1e3:g}mm gap={cfg_info['gap_mm']:g}mm "
          f"islands={'dropped' if args.drop_islands else 'kept'}")
    print(f"[me] -> {run_dir}", flush=True)
    (run_dir / "config.json").write_text(json.dumps(
        dict(vars(args), D=D, H=H, W=W, noise_dim=noise_dim, axes=AXES, ranges=RANGES,
             **{k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in cfg_info.items()}), indent=2))

    resume = np.load(args.resume_from) if args.resume_from else None
    # `centroids` as an int asks pyribs to grow that many CVT centroids by k-means;
    # `seed` pins them, so two runs with the same --cells/--cvt-seed tessellate the
    # descriptor space identically and their coverage numbers are comparable.
    archive = CVTArchive(solution_dim=D, centroids=args.cells, ranges=RANGES,
                         seed=args.cvt_seed, threshold_min=0.5,
                         learning_rate=args.learning_rate, extra_fields=EXTRA_FIELDS)
    x0s = [np.zeros(D) for _ in range(args.n_emitters)]
    if resume is not None:
        x0s = _warm_start(archive, resume, D, args)
    emitters = [EvolutionStrategyEmitter(archive, x0=x0s[i], sigma0=args.sigma0,
                                         es=args.es, batch_size=args.batch_size)
                for i in range(args.n_emitters)]
    scheduler = Scheduler(archive, emitters)
    narch = NoveltyArchive(k=args.novelty_k, gh=GH, gw=GW)

    history, t0 = [], time.time()
    for it in range(args.iterations):
        genomes = scheduler.ask()
        mets = []
        for s in range(0, len(genomes), args.chunk):
            mk = gnm.genomes_to_masks(gen, genomes[s:s + args.chunk], K, H, W, args.x0_seed,
                                      noise_dim=noise_dim, noise_scale=args.noise_scale,
                                      ode_steps=args.ode_steps)
            mets += evaluate.evaluate_masks(mk, cfg, device=args.device,
                                            w_force=args.w_force, w_wrap=args.w_wrap)

        valid = np.array([m["valid"] for m in mets], bool)
        grasp = np.array([m["score"] for m in mets], np.float32)
        # novelty is scored on the EFFECTIVE mask -- what the solver saw -- so a design
        # cannot buy novelty with floating debris that never touches the object
        feats = narch._feat([m["effective_mask"] for m in mets]).astype(np.float32)
        nov = narch.score_feats(feats).astype(np.float32)
        objective = np.where(valid, 1.0 + args.novelty_weight * nov + args.w_grasp * grasp, 0.0)
        measures = np.stack([[m[a] for a in AXES] for m in mets]).astype(np.float32)
        narch.add_feats(feats[valid], quals=grasp[valid])
        scheduler.tell(objective, measures, novelty=nov, grasp=grasp,
                       force_term=np.array([m["force_term"] for m in mets], np.float32),
                       wrap_term=np.array([m["wrap_term"] for m in mets], np.float32),
                       pull_off=np.array([m["pull_off"] for m in mets], np.float32),
                       arc=np.array([m["arc"] for m in mets], np.float32),
                       ncon=np.array([m["ncon"] for m in mets], np.int32))

        st = archive.stats
        best = float(st.obj_max) if st.num_elites else 0.0   # obj_max raises on an empty archive
        rate = (it + 1) / max(time.time() - t0, 1e-9)
        print(f"[iter {it:>3}] elites={st.num_elites:>4} coverage={st.coverage*100:>5.1f}% "
              f"QD={st.qd_score:>8.2f} valid={int(valid.sum())}/{len(mets)} "
              f"best={best:.3f} ({rate*60:.2f} it/min)", flush=True)
        history.append(dict(iter=it, num_elites=int(st.num_elites), coverage=float(st.coverage),
                            qd_score=float(st.qd_score), best_fit=best,
                            n_valid=int(valid.sum()), n_eval=int(len(mets))))
        np.savez_compressed(run_dir / "history.npz",
                            **{k: np.array([h[k] for h in history]) for k in history[0]})
        _save(archive, run_dir / "archive.npz", K, noise_dim)
        if it % args.snapshot_every == 0:
            snap = run_dir / "snapshots"; snap.mkdir(exist_ok=True)
            _save(archive, snap / f"archive_{it:04d}.npz", K, noise_dim)

    print(f"[me] {archive.stats.num_elites} elites -> {run_dir}/archive.npz", flush=True)
    return archive, run_dir


def _save(archive, path, K, noise_dim):
    d = archive.data(return_type="dict")
    np.savez_compressed(path, solution=d["solution"], objective=d["objective"],
                        measures=d["measures"], cells=int(archive.cells),
                        ranges=np.array(RANGES), measure_names=np.array(AXES),
                        centroids=np.asarray(getattr(archive, "centroids", np.empty(0))),
                        n_sites=K, noise_dim=int(noise_dim),
                        **{f: d[f] for f in EXTRA_FIELDS if f in d})


def _warm_start(archive, resume, D, args):
    """Re-add a prior run's elites and seed each emitter near one of them."""
    rs, ro, rm = (np.asarray(resume[k]) for k in ("solution", "objective", "measures"))
    keep = ro > 0
    if rs.shape[1] != D:
        raise SystemExit(f"[me] resume genome dim {rs.shape[1]} != current D={D} "
                         f"(--n-sites or model mismatch) -- cannot warm-start")
    extra = {f: (np.asarray(resume[f])[keep].astype(np.int32 if f == "ncon" else np.float32)
                 if f in resume.files
                 else np.zeros(int(keep.sum()), np.int32 if f == "ncon" else np.float32))
             for f in EXTRA_FIELDS}
    archive.add(rs[keep], ro[keep], rm[keep], **extra)
    st = archive.stats
    print(f"[me] warm-started: {int(keep.sum())} elites re-added -> {st.num_elites} in archive, "
          f"QD={st.qd_score:.2f}, coverage={st.coverage*100:.1f}%", flush=True)
    rng = np.random.default_rng(args.x0_seed)
    pick = rng.integers(0, int(keep.sum()), size=args.n_emitters)
    return [rs[keep][j] for j in pick]


if __name__ == "__main__":
    main()
