# CAPS: Conformal Prediction for Legal Labels

CAPS is a research prototype for calibrated legal issue prediction. The current
main experiment asks a deliberately modest question:

> Can noisy LLM reasoning and verification signals help a conformal legal
> predictor return useful sets of ECtHR articles at a target coverage level?

The project uses the LexGLUE ECtHR-B task. Instead of forcing a single article
prediction, it returns a set of Convention articles, for example `{3, 6, 8}`.
The conformal goal is set coverage: at roughly `1 - alpha`, the returned set
should contain all gold labels for a case.

The LLM is not treated as a source of legal truth. It is used as a structured
feature generator: for top candidate articles, it writes short reasoning steps
and verifies whether those steps are supported by the case facts. Those verifier
signals are then combined with a base classifier and calibrated with split
conformal prediction.

## What Is In This Repo

- `legal/run_ecthr_base_scores.py`: trains the base ECtHR-B classifier.
- `legal/run_ecthr_reasoner_features.py`: generates LegalReasoner-style LLM
  features with a vLLM/OpenAI-compatible endpoint.
- `legal/build_ecthr_feature_table.py`: merges base scores and reasoner
  features into one row per `(case, label)`.
- `legal/train_ecthr_meta_scorer.py`: trains per-label meta-scorers.
- `legal/eval_ecthr_conformal.py`: calibrates and evaluates conformal sets.
- `legal/report_ecthr_compare.py`: prints a compact results table.
- `conformal/ecthr_multilabel.py`: conformal thresholding utilities.
- `scripts/slurm/`: cluster scripts for the GPU reasoner pass and CPU eval lane.
- `outputs/ecthr_b/`: checked-in artifacts for the current ECtHR-B run.
- `docs/ecthr_experiment_overview.md`: a more narrative experiment overview.
- `paper/ecthr_reasoner_conformal.tex`: draft paper text.

There are also older LEXam-style trace generation and scoring utilities in
`legal/run_generate_legal.py`, `legal/score_legal.py`, and related files. The
README focuses on the newer ECtHR-B conformal article-prediction experiment.

## Models And Data

Dataset:

- `coastalcph/lex_glue`, configuration `ecthr_b`, loaded through Hugging Face
  `datasets`.
- Splits: `train` for supervised fitting, `validation` for conformal
  calibration, and `test` for held-out evaluation.
- Current label set: `2`, `3`, `5`, `6`, `8`, `9`, `10`, `11`, `14`, `P1-1`.

Models:

- Base scorer: TF-IDF unigram/bigram features plus one-vs-rest logistic
  regression.
- Reasoner/verifier model: `Qwen/Qwen2.5-7B-Instruct`, served with vLLM.
- Meta-scorer: dictionary-vectorized logistic regression over per-label
  features.
- Conformal predictor: split conformal multi-label set prediction using
  validation as the calibration split.

Current checked-in run:

- Base scores cover `9000` train, `1000` validation, and `1000` test cases.
- LLM reasoner features cover the first `500` cases of each split.
- Reported conformal results below use the `500` validation and `500` test
  cases with reasoner features.
- Top candidate labels sent to the LLM: `K = 6`.
- Maximum reasoning steps per candidate: `4`.

## Current Results

Primary comparison: global conformal calibration at `alpha = 0.1`, calibrated on
`validation` and evaluated on `test`.

| Method | Coverage | Avg set size | Median set size | Micro-F1 | Macro-F1 | Micro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base only | 0.894 | 2.616 | 2.0 | 0.632 | 0.637 | 0.916 |
| Reasoner only | 0.900 | 2.794 | 3.0 | 0.609 | 0.586 | 0.922 |
| Base + reasoner | 0.898 | 2.668 | 3.0 | 0.628 | 0.622 | 0.922 |

Reading this honestly: the current run is promising but not a final empirical
claim. The reasoner-only model reaches nominal 90% coverage but has larger sets.
The combined model is more efficient and has better F1 than reasoner-only. The
base-only model is slightly under 90% coverage on this 500-case subset and has
the smallest average set size. At `alpha = 0.2`, the combined model matches
base-only coverage while using smaller sets and getting higher F1.

All global conformal results in `outputs/ecthr_b/conformal_metrics.json`:

| Method | Alpha | Coverage | Avg set size | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base only | 0.05 | 0.924 | 3.400 | 0.544 | 0.547 |
| Base only | 0.10 | 0.894 | 2.616 | 0.632 | 0.637 |
| Base only | 0.20 | 0.782 | 1.780 | 0.711 | 0.666 |
| Reasoner only | 0.05 | 0.946 | 3.850 | 0.507 | 0.496 |
| Reasoner only | 0.10 | 0.900 | 2.794 | 0.609 | 0.586 |
| Reasoner only | 0.20 | 0.778 | 1.830 | 0.693 | 0.660 |
| Base + reasoner | 0.05 | 0.926 | 3.462 | 0.539 | 0.539 |
| Base + reasoner | 0.10 | 0.898 | 2.668 | 0.628 | 0.622 |
| Base + reasoner | 0.20 | 0.782 | 1.690 | 0.731 | 0.694 |

To regenerate the compact comparison table from the checked-in metrics:

```bash
python -m legal.report_ecthr_compare \
  --metrics outputs/ecthr_b/conformal_metrics.json \
  --alpha 0.1 \
  --mode global \
  --out outputs/ecthr_b/compare_summary.json
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- The CPU stages need `datasets`, `scikit-learn`, `joblib`, `numpy`, and `tqdm`.
- The reasoner stage needs a Linux GPU environment that can run `vllm`.
- On CPU-only or non-Linux machines, `vllm` may be the dependency that fails.
  For artifact inspection and CPU-only evaluation, install the non-vLLM
  packages directly:

```bash
pip install clingo datasets joblib numpy "pydantic[dotenv]" orjson requests scikit-learn tqdm
```

- Install `torch` and `vllm` in the GPU environment that will run the reasoner.
- The first dataset run downloads LexGLUE from Hugging Face.
- vLLM will download the configured model unless it is already cached.

## Reproduce The Checked-In ECtHR-B Run

Run commands from the repository root.

### 1. Train Base Scores

```bash
python -m legal.run_ecthr_base_scores \
  --out_dir outputs/ecthr_b \
  --splits train,validation,test \
  --crossfit_train
```

This writes:

- `outputs/ecthr_b/base_scores_train.jsonl`
- `outputs/ecthr_b/base_scores_validation.jsonl`
- `outputs/ecthr_b/base_scores_test.jsonl`
- `outputs/ecthr_b/base_tfidf_ovr.joblib`
- `outputs/ecthr_b/base_scores_meta.json`

### 2. Generate Reasoner Features

This is the expensive GPU step. On Slurm:

```bash
sbatch --export=ALL,ECTHR_SPLIT=train,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6 \
  scripts/slurm/run_ecthr_reasoner_features.sbatch

sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6 \
  scripts/slurm/run_ecthr_reasoner_features.sbatch

sbatch --export=ALL,ECTHR_SPLIT=test,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6 \
  scripts/slurm/run_ecthr_reasoner_features.sbatch
```

The Slurm script starts a vLLM server for `Qwen/Qwen2.5-7B-Instruct`, waits for
it to become ready, then runs `legal.run_ecthr_reasoner_features`.

Useful environment overrides:

```bash
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_MAX_MODEL_LEN=32768
LR_MAX_CANDIDATES=6
LR_MAX_STEPS=4
MAX_CASE_CHARS=full
```

For a small local or interactive smoke test, start vLLM yourself:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --trust-remote-code
```

Then, in another shell:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

python -m legal.run_ecthr_reasoner_features \
  --split validation \
  --n_examples 25 \
  --candidate_source base \
  --max_candidates 6 \
  --top_k 6
```

### 3. Repair Failed LLM Rows

Structured LLM generation can fail or produce incomplete rows. Submit repair
jobs for only the failed case IDs:

```bash
REPAIR_TAG=repair1 LR_MAX_CANDIDATES=6 \
  scripts/slurm/submit_ecthr_reasoner_repairs.sh
```

Merge successful repairs back into the main files:

```bash
scripts/merge_ecthr_reasoner_repairs.sh repair1
```

You can also merge a single split manually:

```bash
python -m legal.merge_ecthr_reasoner_features \
  --base outputs/ecthr_b/reasoner_features_validation.jsonl \
  --repairs outputs/ecthr_b/repairs/reasoner_features_validation_repair1.jsonl \
  --out outputs/ecthr_b/reasoner_features_validation.jsonl
```

### 4. Build Feature Tables

The current checked-in result restricts evaluation to cases with reasoner
features:

```bash
python -m legal.build_ecthr_feature_table \
  --out_dir outputs/ecthr_b \
  --splits train,validation,test \
  --restrict_to_reasoner_cases
```

This creates one row per `(case, label)`:

- `outputs/ecthr_b/feature_table_train.jsonl`
- `outputs/ecthr_b/feature_table_validation.jsonl`
- `outputs/ecthr_b/feature_table_test.jsonl`

### 5. Train Meta-Scorers

```bash
python -m legal.train_ecthr_meta_scorer \
  --out_dir outputs/ecthr_b \
  --methods base_only,reasoner_only,base_plus_reasoner
```

This writes meta-score files for validation/test and model artifacts under
`outputs/ecthr_b/models/`.

### 6. Calibrate And Evaluate

```bash
python -m legal.eval_ecthr_conformal \
  --out_dir outputs/ecthr_b \
  --methods base_only,reasoner_only,base_plus_reasoner \
  --alphas 0.05,0.1,0.2 \
  --out outputs/ecthr_b/conformal_metrics.json
```

Then print the paper-style summary:

```bash
python -m legal.report_ecthr_compare \
  --metrics outputs/ecthr_b/conformal_metrics.json \
  --alpha 0.1 \
  --mode global \
  --out outputs/ecthr_b/compare_summary.json
```

## CPU-Only Evaluation Lane

Once reasoner feature files already exist, the remaining steps are CPU-only. On
Slurm you can run the ablation/evaluation script:

```bash
sbatch --export=ALL,ECTHR_OUT_DIR=outputs/ecthr_b,ECTHR_RESTRICT_TO_REASONER=1 \
  scripts/slurm/run_ecthr_ablation_eval.sbatch
```

By default this evaluates additional ablations:

- `base_rank`
- `base_prelim`
- `base_verdict`
- `base_step_counts`
- `base_full_no_rank`
- `reasoner_only`
- `base_plus_reasoner`

Those ablations are useful for checking whether verifier features add signal
beyond the base model rank.

## Artifact Guide

The main ECtHR-B artifacts are:

- `base_scores_*.jsonl`: one row per case with base classifier probabilities.
- `reasoner_features_*.jsonl`: one row per case with candidates, generated
  reasoning, verifier judgments, and flattened per-label features.
- `feature_table_*.jsonl`: one row per `(case, label)` for meta-scorer training.
- `meta_scores_*_validation.jsonl` and `meta_scores_*_test.jsonl`: per-case
  score dictionaries from each meta-scorer.
- `conformal_metrics.json`: full conformal evaluation output.
- `compare_summary.json`: compact table for one alpha/mode.
- `models/*.joblib`: trained meta-scorers.

Example reasoner feature fields:

- `candidate_present`
- `candidate_rank_norm`
- `prelim_support`
- `verdict_supported`
- `n_supported_steps`
- `supported_fraction`
- `hard_support_gate`

Labels outside the top `K` candidates are still scored by the base classifier.
Their reasoner features are simply set to defaults, so the LLM cannot hard-delete
a legal article from the final conformal set.

## Troubleshooting

- If `datasets` cannot find ECtHR-B, check network access and Hugging Face cache
  settings.
- If vLLM fails to start, verify CUDA, GPU memory, `VLLM_MAX_MODEL_LEN`, and that
  `VLLM_MODEL` is available.
- If all reasoner rows fail, check `VLLM_BASE_URL` and the vLLM log written under
  `outputs/ecthr_b/logs/`.
- If feature-table building warns about missing reasoner rows, use
  `--restrict_to_reasoner_cases` for pilot-style runs or omit it to fill missing
  reasoner features with defaults.
- If reproducibility matters, keep `VLLM_SEED=0` and use the checked-in command
  pattern above.

## Research Status

This is an unpolished but working research codebase. The strongest current use
is to inspect and reproduce the ECtHR-B conformal pipeline, then extend it with
stronger base encoders, more complete reasoner coverage, and deeper ablations.

The outputs are not legal advice, and the generated reasoning traces should not
be interpreted as faithful legal explanations. They are noisy signals used inside
a calibrated prediction-set pipeline.
