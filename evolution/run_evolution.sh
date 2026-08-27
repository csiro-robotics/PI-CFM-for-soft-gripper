#!/usr/bin/env bash
# =============================================================================
# CVT-MAP-Elites over the PI-CFM genome — LOCAL DESKTOP (single GPU).
# =============================================================================
# Reproduces the paper's search configuration: LM-MA-ES emitters, 64 x 32 = 2048
# designs per iteration, a 4-D CVT over
#     [strut_complexity, branch_density, hole_count, material_fraction]
# and the implicit PNCG-IPC grasp simulation at the gallery_v3 operating point.
#
# The published run sharded each iteration across 4 GPUs; here the same
# population is evaluated in CHUNK-sized solver launches on one card. genome ->
# design is a pure function of (genome, x0_seed), so the search is identical --
# only slower. Expect the full 600 iterations to take days on a desktop; the
# archive is checkpointed every iteration and --resume-from picks it back up.
#
#   ./run_evolution.sh                  # the full run
#   ./run_evolution.sh smoke            # 2 tiny iterations, end-to-end check
#   ITERATIONS=50 ./run_evolution.sh    # a short run
#
# Requires the trained PI-CFM weights at ../checkpoints/picfm_dit.pt
# (see ../checkpoints/README.md, or train them with ../training/run_training.sh).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CHECKPOINT="${CHECKPOINT:-../checkpoints/picfm_dit.pt}"
RUN_DIR="${RUN_DIR:-../runs/me_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"

# --- search (paper configuration) ---
ITERATIONS="${ITERATIONS:-600}"
N_EMITTERS="${N_EMITTERS:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CELLS="${CELLS:-2000}"

# --- desktop batching -------------------------------------------------------
# Designs per implicit solver launch. Throughput only: the archive is identical
# whatever you set. 256 fits a 24 GB card comfortably; drop it if you hit OOM,
# raise it if `nvidia-smi` shows headroom.
CHUNK="${CHUNK:-256}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: no PI-CFM checkpoint at ${CHECKPOINT}." >&2
  echo "       See ../checkpoints/README.md, or train one:" >&2
  echo "         cd ../training && ./run_training.sh" >&2
  exit 1
fi

if [ "${1:-run}" = "smoke" ]; then
  echo "=== SMOKE: 2 iterations, 4 emitters x 2, 50 cells ==="
  exec python map_elites.py --smoke --checkpoint "${CHECKPOINT}" \
       --device "${DEVICE}" --run-dir "${RUN_DIR}_smoke" --chunk 8
fi

echo "=============================================================="
echo " CVT-MAP-Elites  ${N_EMITTERS} x ${BATCH_SIZE} x ${ITERATIONS} iters"
echo " -> ${RUN_DIR}"
echo "=============================================================="
mkdir -p "${RUN_DIR}"
python map_elites.py \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --es lm_ma_es \
    --n-emitters "${N_EMITTERS}" \
    --batch-size "${BATCH_SIZE}" \
    --iterations "${ITERATIONS}" \
    --cells "${CELLS}" \
    --chunk "${CHUNK}" \
    --snapshot-every 1 \
    --novelty-k 15 \
    --w-grasp 0.4 --w-force 0.2 --w-wrap 0.8 \
    --noise-scale 1.0 \
    --run-dir "${RUN_DIR}" \
  2>&1 | tee -a "${RUN_DIR}/train.log"

echo "DONE -> ${RUN_DIR}/archive.npz"
