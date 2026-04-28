#!/bin/bash
#
# CPU-only ECtHR ablation/evaluation lane. Run after reasoner feature files are
# present and repaired.
#
# Submit from the repository root:
#   sbatch scripts/slurm/run_ecthr_ablation_eval.sbatch
#
# Common overrides:
#   sbatch --export=ALL,ECTHR_OUT_DIR=outputs/ecthr_b,ECTHR_RESTRICT_TO_REASONER=1 scripts/slurm/run_ecthr_ablation_eval.sbatch

#SBATCH --job-name=ecthr-ablate
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

ECTHR_OUT_DIR="${ECTHR_OUT_DIR:-outputs/ecthr_b}"
ECTHR_SPLITS="${ECTHR_SPLITS:-train,validation,test}"
ECTHR_ALPHAS="${ECTHR_ALPHAS:-0.05,0.1,0.2}"
ECTHR_RESTRICT_TO_REASONER="${ECTHR_RESTRICT_TO_REASONER:-1}"
ECTHR_METHODS="${ECTHR_METHODS:-base_only,base_rank,base_prelim,base_verdict,base_step_counts,base_full_no_rank,reasoner_only,base_plus_reasoner}"
ECTHR_METRICS_OUT="${ECTHR_METRICS_OUT:-${ECTHR_OUT_DIR}/conformal_metrics_ablations.json}"
ECTHR_SUMMARY_OUT="${ECTHR_SUMMARY_OUT:-${ECTHR_OUT_DIR}/compare_summary_ablations.json}"

echo "Running ECtHR ablations"
echo "out_dir=${ECTHR_OUT_DIR}"
echo "methods=${ECTHR_METHODS}"
echo "alphas=${ECTHR_ALPHAS}"

build_cmd=(
  python -m legal.build_ecthr_feature_table
  --out_dir "${ECTHR_OUT_DIR}"
  --splits "${ECTHR_SPLITS}"
)

if [[ "${ECTHR_RESTRICT_TO_REASONER}" == "1" ]]; then
  build_cmd+=(--restrict_to_reasoner_cases)
fi

echo "Running: ${build_cmd[*]}"
"${build_cmd[@]}"

train_cmd=(
  python -m legal.train_ecthr_meta_scorer
  --out_dir "${ECTHR_OUT_DIR}"
  --methods "${ECTHR_METHODS}"
)
echo "Running: ${train_cmd[*]}"
"${train_cmd[@]}"

eval_cmd=(
  python -m legal.eval_ecthr_conformal
  --out_dir "${ECTHR_OUT_DIR}"
  --methods "${ECTHR_METHODS}"
  --alphas "${ECTHR_ALPHAS}"
  --out "${ECTHR_METRICS_OUT}"
)
echo "Running: ${eval_cmd[*]}"
"${eval_cmd[@]}"

report_cmd=(
  python -m legal.report_ecthr_compare
  --metrics "${ECTHR_METRICS_OUT}"
  --out "${ECTHR_SUMMARY_OUT}"
)
echo "Running: ${report_cmd[*]}"
"${report_cmd[@]}"

echo "Done."
