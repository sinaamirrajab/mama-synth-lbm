#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/infer_slice.sh /path/to/patient.mha /path/to/output_dir" >&2
  echo "   or: bash scripts/infer_slice.sh /path/to/staged_input_dir /path/to/output_dir" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$1"
OUTPUT="$2"
INPUT_SLUG="${MAMA_INPUT_SLUG:-pre-contrast-dce-mri-slice-breast}"

if [ ! -f "${REPO_ROOT}/submission_config.json" ]; then
  echo "Missing ${REPO_ROOT}/submission_config.json. Copy/edit submission_config.example.json after downloading resources." >&2
  exit 2
fi

if [ -f "${INPUT}" ]; then
  STAGED_INPUT="$(mktemp -d)"
  mkdir -p "${STAGED_INPUT}/images/${INPUT_SLUG}"
  cp "${INPUT}" "${STAGED_INPUT}/images/${INPUT_SLUG}/patient.mha"
else
  STAGED_INPUT="${INPUT}"
fi

MAMA_INPUT_DIR="${STAGED_INPUT}" \
MAMA_OUTPUT_DIR="${OUTPUT}" \
python "${REPO_ROOT}/inference.py"
