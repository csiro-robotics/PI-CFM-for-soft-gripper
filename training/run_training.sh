#!/usr/bin/env bash
# =============================================================================
# PI-CFM training — LOCAL DESKTOP (single GPU). No SLURM, no module load.
# =============================================================================
# Two stages, in order. Stage 2 REQUIRES stage 1: the DiT is trained with the
# condition encoder pre-trained here and frozen (training.freeze_encoder: true),
# so running stage 2 first would train against a randomly-initialised encoder.
#
#   stage 1  pretrain_encoder.py  -> $ENC_DIR/encoder_best.pt
#   stage 2  train_picfm_dit.py   -> $OUT_DIR/<run_name>/
#
#   ./run_training.sh              # both stages
#   ./run_training.sh encoder      # stage 1 only
#   ./run_training.sh dit          # stage 2 only (encoder must already exist)
#
# Data layout expected (see README):
#   data/mechanism_dataset/  data/finray_dataset/  data/graph_dataset/
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- paths (override from the environment) ----------------------------------
DATA_ROOT="${DATA_ROOT:-../data}"
ENC_DIR="${ENC_DIR:-../checkpoints/design_encoder}"
OUT_DIR="${OUT_DIR:-../checkpoints/picfm}"
CONFIG="${CONFIG:-configs/train_dit.yaml}"
DEVICE="${DEVICE:-cuda}"

# Desktop-scale defaults. The published run used batch 128 on an H100; 32 fits a
# 24 GB card. Raise if you have the memory — it does not change the method.
ENC_EPOCHS="${ENC_EPOCHS:-150}"
ENC_BATCH="${ENC_BATCH:-32}"
DIT_BATCH="${DIT_BATCH:-32}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

DATA_DIRS=(
  "${DATA_ROOT}/mechanism_dataset:topopt"
  "${DATA_ROOT}/finray_dataset:finray"
  "${DATA_ROOT}/graph_dataset:graph"
)

stage="${1:-all}"

run_encoder() {
  echo "=============================================================="
  echo " STAGE 1/2  condition encoder  ->  ${ENC_DIR}"
  echo "=============================================================="
  mkdir -p "${ENC_DIR}"
  python pretrain_encoder.py \
      --data_dirs "${DATA_DIRS[@]}" \
      --output_dir "${ENC_DIR}" \
      --epochs "${ENC_EPOCHS}" \
      --batch_size "${ENC_BATCH}" \
      --device "${DEVICE}"
}

run_dit() {
  local enc="${ENC_DIR}/encoder_best.pt"
  [ -f "${enc}" ] || enc="${ENC_DIR}/encoder_final.pt"
  if [ ! -f "${enc}" ]; then
    echo "ERROR: no encoder checkpoint in ${ENC_DIR}." >&2
    echo "       Run stage 1 first:  ./run_training.sh encoder" >&2
    exit 1
  fi
  echo "=============================================================="
  echo " STAGE 2/2  PI-CFM DiT  (encoder: ${enc})  ->  ${OUT_DIR}"
  echo "=============================================================="
  mkdir -p "${OUT_DIR}"
  python train_picfm_dit.py \
      --config "${CONFIG}" \
      --data_dirs "${DATA_DIRS[@]}" \
      --pretrained_encoder "${enc}" \
      --batch_size "${DIT_BATCH}" \
      --output_dir "${OUT_DIR}"
}

case "${stage}" in
  encoder) run_encoder ;;
  dit)     run_dit ;;
  all)     run_encoder; run_dit ;;
  *) echo "usage: $0 [all|encoder|dit]" >&2; exit 2 ;;
esac
echo "DONE."
