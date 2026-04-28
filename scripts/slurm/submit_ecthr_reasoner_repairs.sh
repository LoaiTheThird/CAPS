#!/bin/bash
#
# Submit repair jobs for existing ECtHR reasoner feature files.
#
# Usage from repo root:
#   scripts/slurm/submit_ecthr_reasoner_repairs.sh
#
# Common overrides:
#   ECTHR_OUT_DIR=outputs/ecthr_b REPAIR_TAG=repair2 LR_MAX_CANDIDATES=6 ECTHR_MAX_CASE_CHARS=8000 scripts/slurm/submit_ecthr_reasoner_repairs.sh

set -euo pipefail

ECTHR_OUT_DIR="${ECTHR_OUT_DIR:-outputs/ecthr_b}"
REPAIR_TAG="${REPAIR_TAG:-repair_next}"
LR_MAX_CANDIDATES="${LR_MAX_CANDIDATES:-6}"
ECTHR_MAX_CASE_CHARS="${ECTHR_MAX_CASE_CHARS:-${MAX_CASE_CHARS:-}}"
VLLM_MAX_OUTPUT_TOKENS="${VLLM_MAX_OUTPUT_TOKENS:-4096}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/slurm/run_ecthr_reasoner_features.sbatch}"
VENV_PATH="${VENV_PATH:-}"

mkdir -p "${ECTHR_OUT_DIR}/repairs"

for split in train validation test; do
  base="${ECTHR_OUT_DIR}/reasoner_features_${split}.jsonl"
  out="${ECTHR_OUT_DIR}/repairs/reasoner_features_${split}_${REPAIR_TAG}.jsonl"

  if [[ ! -f "${base}" ]]; then
    echo "Skipping ${split}: missing ${base}" >&2
    continue
  fi

  export_args=(
    "ALL"
    "ECTHR_SPLIT=${split}"
    "ECTHR_FAILED_FROM=${base}"
    "ECTHR_OUT=${out}"
    "LR_MAX_CANDIDATES=${LR_MAX_CANDIDATES}"
    "VLLM_MAX_OUTPUT_TOKENS=${VLLM_MAX_OUTPUT_TOKENS}"
  )

  if [[ -n "${ECTHR_MAX_CASE_CHARS}" ]]; then
    export_args+=("ECTHR_MAX_CASE_CHARS=${ECTHR_MAX_CASE_CHARS}")
  fi

  if [[ -n "${VENV_PATH}" ]]; then
    export_args+=("VENV_PATH=${VENV_PATH}")
  fi

  export_joined=$(IFS=,; echo "${export_args[*]}")
  echo "Submitting repair for ${split} -> ${out}"
  sbatch --export="${export_joined}" "${SBATCH_SCRIPT}"
done
