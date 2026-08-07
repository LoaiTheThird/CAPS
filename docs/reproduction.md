# Reproducing the ECtHR-B run

Run every command from the repository root. Generated files are written to `outputs/ecthr_b` and are not tracked by Git.

## Environment

The CPU stages target Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the GPU dependencies only on the machine that will run vLLM.

```bash
pip install -r requirements-gpu.txt
```

The first run downloads the ECtHR-B dataset from Hugging Face. The reasoner stage also downloads the configured instruction model if it is not already cached.

## 1. Train the base scorer

```bash
python -m legal.run_ecthr_base_scores \
  --out_dir outputs/ecthr_b \
  --splits train,validation,test \
  --crossfit_train
```

This trains a TF-IDF one-vs-rest classifier. Training scores are produced with five-fold cross-fitting so the downstream meta-scorer does not train on in-sample base probabilities.

## 2. Generate evidence features

The checked-in run uses the top six base labels and processes 500 cases from each split. On Slurm:

```bash
sbatch --export=ALL,ECTHR_SPLIT=train,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6 \
  scripts/slurm/run_ecthr_reasoner_features.sbatch

sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6 \
  scripts/slurm/run_ecthr_reasoner_features.sbatch

sbatch --export=ALL,ECTHR_SPLIT=test,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6 \
  scripts/slurm/run_ecthr_reasoner_features.sbatch
```

The Slurm job starts a local vLLM server, waits for it to become ready, and then runs `legal.run_ecthr_reasoner_features`.

Useful overrides:

```bash
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_MAX_MODEL_LEN=32768
VLLM_MAX_OUTPUT_TOKENS=2048
ECTHR_MAX_CASE_CHARS=full
LR_MAX_CANDIDATES=6
LR_MAX_STEPS=4
```

For an interactive smoke test, start the server yourself:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --trust-remote-code
```

Then run a small validation subset in another shell:

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

## 3. Repair incomplete rows

Structured generation can return incomplete rows. The repair helper submits jobs only for affected case IDs.

```bash
REPAIR_TAG=repair1 LR_MAX_CANDIDATES=6 \
  scripts/slurm/submit_ecthr_reasoner_repairs.sh
```

Merge successful repairs into the main feature files:

```bash
scripts/merge_ecthr_reasoner_repairs.sh repair1
```

## 4. Build features and train meta-scorers

```bash
python -m legal.build_ecthr_feature_table \
  --out_dir outputs/ecthr_b \
  --splits train,validation,test \
  --restrict_to_reasoner_cases

python -m legal.train_ecthr_meta_scorer \
  --out_dir outputs/ecthr_b \
  --methods base_only,reasoner_only,base_plus_reasoner
```

The feature table contains one row per case and label. Labels outside the reasoner candidate set receive default reasoner features and retain their base score.

## 5. Calibrate and evaluate

```bash
python -m legal.eval_ecthr_conformal \
  --out_dir outputs/ecthr_b \
  --methods base_only,reasoner_only,base_plus_reasoner \
  --alphas 0.05,0.1,0.2
```

Print a compact comparison:

```bash
python -m legal.report_ecthr_compare \
  --metrics outputs/ecthr_b/conformal_metrics.json \
  --alpha 0.1 \
  --out outputs/ecthr_b/compare_summary.json
```

Once the reasoner files exist, steps 4 and 5 can run as one CPU-only Slurm job:

```bash
sbatch --export=ALL,ECTHR_OUT_DIR=outputs/ecthr_b,ECTHR_RESTRICT_TO_REASONER=1 \
  scripts/slurm/run_ecthr_ablation_eval.sbatch
```

## Run settings

- Dataset splits: official train, validation, and test splits
- Base training cases: 9,000
- Base validation and test cases: 1,000 each
- Reasoner and meta-scorer subset: first 500 cases from each split
- Candidate labels per case: 6
- Maximum reasoning steps per label: 4
- vLLM seed: 0

The compact results copied from this run live in `results/ecthr_b`.
