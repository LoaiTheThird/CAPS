#!/bin/bash
#
# Start a vLLM OpenAI-compatible server inside this Slurm allocation, wait until
# it is ready, then run Step 3 feature generation against localhost.
#
# Submit from the repository root:
#   sbatch scripts/slurm/run_ecthr_reasoner_features.sbatch
#
# Common overrides:
#   sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_N_EXAMPLES=25 scripts/slurm/run_ecthr_reasoner_features.sbatch
#   sbatch --export=ALL,ECTHR_SPLIT=test,LR_MAX_CANDIDATES=8 scripts/slurm/run_ecthr_reasoner_features.sbatch
#   sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_N_EXAMPLES=100,ECTHR_OUT=outputs/ecthr_b/pilot_top6_n100/reasoner_features_validation.jsonl,ECTHR_BASE_SCORES=outputs/ecthr_b/base_scores_validation.jsonl,LR_MAX_CANDIDATES=6 scripts/slurm/run_ecthr_reasoner_features.sbatch
#   sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_FAILED_FROM=outputs/ecthr_b/reasoner_features_validation.jsonl,ECTHR_OUT=outputs/ecthr_b/repairs/reasoner_features_validation_repair.jsonl,LR_MAX_CANDIDATES=6 scripts/slurm/run_ecthr_reasoner_features.sbatch

#SBATCH --job-name=ecthr-reasoner
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=35:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=hendrixgpu03fl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p slurm_logs

module load python/3.11.3

VENV_PATH="${VENV_PATH:-.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-$PWD}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_OUTPUT_TOKENS="${VLLM_MAX_OUTPUT_TOKENS:-2048}"
export VLLM_SEED="${VLLM_SEED:-0}"

ECTHR_SPLIT="${ECTHR_SPLIT:-validation}"
ECTHR_N_EXAMPLES="${ECTHR_N_EXAMPLES:-}"
ECTHR_OFFSET="${ECTHR_OFFSET:-0}"
ECTHR_OUT_DIR="${ECTHR_OUT_DIR:-outputs/ecthr_b}"
ECTHR_OUT="${ECTHR_OUT:-}"
ECTHR_CANDIDATE_SOURCE="${ECTHR_CANDIDATE_SOURCE:-base}"
ECTHR_BASE_SCORES="${ECTHR_BASE_SCORES:-}"
ECTHR_TOP_K="${ECTHR_TOP_K:-}"
ECTHR_IDS="${ECTHR_IDS:-}"
ECTHR_IDS_FILE="${ECTHR_IDS_FILE:-}"
ECTHR_FAILED_FROM="${ECTHR_FAILED_FROM:-}"
LR_MAX_CANDIDATES="${LR_MAX_CANDIDATES:-4}"
LR_MAX_STEPS="${LR_MAX_STEPS:-4}"
LR_MIN_VERIFIED_SUPPORT_STEPS="${LR_MIN_VERIFIED_SUPPORT_STEPS:-2}"

VLLM_LOG_DIR="${VLLM_LOG_DIR:-outputs/ecthr_b/logs}"
mkdir -p "${VLLM_LOG_DIR}"
VLLM_LOG="${VLLM_LOG_DIR}/vllm-${SLURM_JOB_ID:-local}.log"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Running on node:"
hostname

echo "Current directory:"
pwd

echo "Python:"
which python
python --version

echo "GPU:"
nvidia-smi

echo "Starting vLLM model: ${VLLM_MODEL}"
echo "vLLM URL: ${VLLM_BASE_URL}"
echo "vLLM log: ${VLLM_LOG}"

if command -v vllm >/dev/null 2>&1; then
  vllm serve "${VLLM_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --trust-remote-code \
    >"${VLLM_LOG}" 2>&1 &
else
  python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --trust-remote-code \
    >"${VLLM_LOG}" 2>&1 &
fi

VLLM_PID=$!

echo "Waiting for vLLM to become ready..."
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
until curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM process exited early. Last log lines:"
    tail -n 80 "${VLLM_LOG}" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM. Last log lines:"
    tail -n 80 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep 10
done

echo "vLLM is ready."
curl -fsS "${VLLM_BASE_URL}/models" || true
echo

cmd=(
  python -m legal.run_ecthr_reasoner_features
  --split "${ECTHR_SPLIT}"
  --offset "${ECTHR_OFFSET}"
  --out_dir "${ECTHR_OUT_DIR}"
  --candidate_source "${ECTHR_CANDIDATE_SOURCE}"
  --max_candidates "${LR_MAX_CANDIDATES}"
  --max_steps "${LR_MAX_STEPS}"
  --min_verified_support_steps "${LR_MIN_VERIFIED_SUPPORT_STEPS}"
)

if [[ -n "${ECTHR_OUT}" ]]; then
  cmd+=(--out "${ECTHR_OUT}")
fi

if [[ -n "${ECTHR_N_EXAMPLES}" ]]; then
  cmd+=(--n_examples "${ECTHR_N_EXAMPLES}")
fi

if [[ -n "${ECTHR_BASE_SCORES}" ]]; then
  cmd+=(--base_scores "${ECTHR_BASE_SCORES}")
fi

if [[ -n "${ECTHR_TOP_K}" ]]; then
  cmd+=(--top_k "${ECTHR_TOP_K}")
fi

if [[ -n "${ECTHR_IDS}" ]]; then
  cmd+=(--ids "${ECTHR_IDS}")
fi

if [[ -n "${ECTHR_IDS_FILE}" ]]; then
  cmd+=(--ids_file "${ECTHR_IDS_FILE}")
fi

if [[ -n "${ECTHR_FAILED_FROM}" ]]; then
  cmd+=(--failed_from "${ECTHR_FAILED_FROM}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Done."
