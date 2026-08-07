#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/train_lbm_latents.sh /path/to/latent_manifest.csv runs/lbm_source_seg" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MANIFEST="$1"
OUTPUT_DIR="$2"
INCLUDE_DATASETS_VALUE="${INCLUDE_DATASETS:-DUKE ISPY2}"
INCLUDE_ARGS=()
if [ -n "${INCLUDE_DATASETS_VALUE}" ]; then
  read -r -a INCLUDE_ARGS_ARRAY <<< "${INCLUDE_DATASETS_VALUE}"
  INCLUDE_ARGS=(--include-datasets "${INCLUDE_ARGS_ARRAY[@]}")
fi

python -m mama_synth_lbm.train \
  --latent-manifest-path "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --pretrained-model-name-or-path "${PRETRAINED_MODEL:-runwayml/stable-diffusion-v1-5}" \
  --variant "${SD_VARIANT:-fp16}" \
  --unet-init pretrained_text \
  --bridge-variant direct_peak \
  --prediction-target remaining_target \
  --conditioning-mode "${CONDITIONING_MODE:-source_segmentation}" \
  --target-kind peak \
  --timestep-sampling deterministic_sigmas \
  --deterministic-sigma-count "${DETERMINISTIC_SIGMA_COUNT:-501}" \
  --deterministic-sigma-sampling uniform \
  --validation-inference-steps "${VALIDATION_INFERENCE_STEPS:-500}" \
  --bridge-noise-sigma "${BRIDGE_NOISE_SIGMA:-0.005}" \
  --latent-loss-type "${LATENT_LOSS_TYPE:-l1_mse}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --lr-scheduler constant \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-200}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-16}" \
  --max-train-steps "${MAX_TRAIN_STEPS:-8000}" \
  --validation-steps "${VALIDATION_STEPS:-1000}" \
  --checkpointing-steps "${CHECKPOINTING_STEPS:-1000}" \
  --validation-loss-batches "${VALIDATION_LOSS_BATCHES:-4}" \
  --mixed-precision "${MIXED_PRECISION:-fp16}" \
  --seed "${SEED:-42}" \
  "${INCLUDE_ARGS[@]}"
