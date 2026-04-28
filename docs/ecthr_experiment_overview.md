# ECtHR Conformal Reasoner Experiment Overview

This document explains the project idea in intuitive terms and maps it to the current repository.

## The Core Idea

The project is not trying to prove that an LLM's legal reasoning steps are true.

The project is asking a more defensible question:

> Can noisy reasoning-step verification signals help a calibrated legal predictor return smaller legal issue sets at the same target coverage?

In legal prediction, a model often needs to predict several relevant legal articles. A single top label is too brittle. A conformal predictor instead returns a set of labels:

```text
{3, 6, 8}
```

The desired guarantee is:

```text
With about 90% coverage, the set contains all gold LexGLUE labels.
```

The paper lives in the tradeoff:

```text
same coverage, smaller prediction sets
```

If two systems both cover the gold legal issues 90% of the time, the system that returns fewer labels is more useful.

## Why Reasoning Features Might Help

The base classifier gives a probability for each article:

```text
Article 3: 0.72
Article 6: 0.64
Article 8: 0.31
...
```

That is useful, but it does not say whether the article is actually supported by the case facts.

The LegalReasoner-style pipeline adds structured signals:

```text
candidate label -> reasoning steps -> verifier checks
```

For example:

```text
Candidate: Article 8
Reasoning steps:
1. The case concerns family contact.
2. The state decision affected the applicant's relationship with a family member.
3. This may interfere with private or family life.

Verifier:
step 1: supported
step 2: supported
step 3: supported
label verdict: supported
```

These signals become numeric features:

```text
candidate_present = 1
candidate_rank_norm = 0.0
prelim_support = 1
verdict_supported = 1
n_supported_steps = 3
supported_fraction = 1.0
```

The key move: these are features, not final truth.

The final conformal predictor can still include labels that the LLM did not analyze or support, because every label still has a base classifier score.

## Current Pipeline

### Step 1: Load ECtHR-B Splits

File:

```text
legal/gen_common.py
```

The loader supports:

```text
train
validation
test
```

These are used as:

```text
train       -> supervised training
validation  -> conformal calibration
test        -> final evaluation
```

### Step 2: Base Classifier Scores

File:

```text
legal/run_ecthr_base_scores.py
```

Current base model:

```text
TF-IDF vectorizer
One-vs-rest logistic regression
```

Outputs:

```text
outputs/ecthr_b/base_scores_train.jsonl
outputs/ecthr_b/base_scores_validation.jsonl
outputs/ecthr_b/base_scores_test.jsonl
```

Each row has:

```json
{
  "id": 0,
  "split": "validation",
  "gold_labels": ["8"],
  "scores": {
    "2": 0.08,
    "3": 0.40,
    "8": 0.74
  }
}
```

### Step 3: LegalReasoner Feature Generation

File:

```text
legal/run_ecthr_reasoner_features.py
```

This is the GPU/vLLM step.

For each case:

1. Read base classifier scores.
2. Pick top-K labels, currently `K=6`.
3. Ask the LLM to write reasoning steps for those labels.
4. Ask the LLM verifier to check the steps.
5. Convert the result into per-label numeric features.

Outputs:

```text
outputs/ecthr_b/reasoner_features_train.jsonl
outputs/ecthr_b/reasoner_features_validation.jsonl
outputs/ecthr_b/reasoner_features_test.jsonl
```

This step runs on Slurm:

```bash
sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_N_EXAMPLES=100,LR_MAX_CANDIDATES=6 scripts/slurm/run_ecthr_reasoner_features.sbatch
```

### Step 4: Repair Failed LLM Rows

Structured JSON generation sometimes fails. The current repair lane is:

1. Detect failed or incomplete rows.
2. Rerun only those case IDs on Slurm.
3. Merge successful repairs back into the main split file.

Rerun failed validation rows:

```bash
sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_FAILED_FROM=outputs/ecthr_b/reasoner_features_validation.jsonl,ECTHR_OUT=outputs/ecthr_b/repairs/reasoner_features_validation_repair1.jsonl,LR_MAX_CANDIDATES=6 scripts/slurm/run_ecthr_reasoner_features.sbatch
```

Merge repairs locally:

```bash
python -m legal.merge_ecthr_reasoner_features \
  --base outputs/ecthr_b/reasoner_features_validation.jsonl \
  --repairs outputs/ecthr_b/repairs/reasoner_features_validation_repair1.jsonl \
  --out outputs/ecthr_b/reasoner_features_validation.jsonl
```

Rule of thumb:

```text
run_ecthr_reasoner_features -> Slurm/GPU
merge/build/train/eval/report -> local CPU
```

### Step 5: Build Per-Label Feature Tables

File:

```text
legal/build_ecthr_feature_table.py
```

This merges:

```text
base classifier probability
reasoner/verifier features
gold target
```

One row per `(case, label)`:

```json
{
  "id": 0,
  "label": "8",
  "target": 1,
  "features": {
    "base_prob": 0.74,
    "candidate_present": 1,
    "prelim_support": 1,
    "verdict_supported": 1,
    "supported_fraction": 1.0
  }
}
```

For a 100-case pilot:

```bash
python -m legal.build_ecthr_feature_table --splits train,validation,test --restrict_to_reasoner_cases
```

For full complete splits:

```bash
python -m legal.build_ecthr_feature_table --splits train,validation,test --require_complete_reasoner
```

### Step 6: Train Meta-Scorers

File:

```text
legal/train_ecthr_meta_scorer.py
```

Trains three methods:

```text
base_only
reasoner_only
base_plus_reasoner
```

The main method is:

```text
base_plus_reasoner
```

It asks:

```text
Does adding verifier features improve the score used by conformal prediction?
```

Run:

```bash
python -m legal.train_ecthr_meta_scorer
```

### Step 7: Conformal Calibration and Evaluation

Files:

```text
conformal/ecthr_multilabel.py
legal/eval_ecthr_conformal.py
legal/report_ecthr_compare.py
```

The conformal score for a calibration case is:

```text
S_i = max over true labels y of (1 - p_i,y)
```

Then the prediction set is:

```text
C_alpha(x) = labels whose score is high enough under the calibrated threshold
```

Run:

```bash
python -m legal.eval_ecthr_conformal --alphas 0.05,0.1,0.2
python -m legal.report_ecthr_compare
```

## Current Pilot Result

The current repaired 100/100/100 pilot at `alpha=0.1` gives:

```text
Method              Coverage   Avg |C|   Micro-F1   Macro-F1
Base+Reasoner CP    0.910      2.27      0.696      0.706
Base CP             0.910      2.39      0.668      0.682
Reasoner-only CP    0.870      2.56      0.619      0.608
```

Interpretation:

```text
Base+Reasoner matches Base coverage.
Base+Reasoner returns smaller sets.
Base+Reasoner improves F1.
Reasoner-only is not enough by itself.
```

This is the right qualitative pattern.

But it is not paper-final because:

```text
only 100 test cases
small effect size
rare-label counts are tiny
no strong encoder baseline yet
no ablations yet
```

## What Would Make This Paper-Grade

### Full-Scale Evidence

Paper-final evaluation should use:

```text
full validation split for calibration
full test split for evaluation
large or full train split for meta-scorer training
```

For compute reasons, train can be sharded:

```bash
sbatch --export=ALL,ECTHR_SPLIT=train,ECTHR_OFFSET=0,ECTHR_N_EXAMPLES=1000,MAX_CASE_CHARS=8000,LR_MAX_CANDIDATES=6,ECTHR_OUT=outputs/ecthr_b/shards/reasoner_features_train_0000_0999.jsonl scripts/slurm/run_ecthr_reasoner_features.sbatch
```

### Stronger Base Encoder

The current base model is TF-IDF logistic regression.

For a stronger paper, add:

```text
Legal-BERT
Longformer
or another long-document legal encoder
```

Then ask whether reasoner features still help when the base classifier is stronger.

### Ablations

The key reviewer question:

```text
Is the gain from reasoning/verifier features, or just candidate rank?
```

Needed ablations:

```text
base only
base + candidate rank only
base + preliminary decision
base + verifier verdict only
base + step counts only
base + full reasoner features
```

The central claim is strongest if:

```text
base + full reasoner features > base + candidate rank only
```

### Rare-Label and Subgroup Analysis

Report:

```text
per-label recall
rare-label recall
per-label coverage
coverage by number of gold labels
coverage by case length bucket
false negative rate
```

This matters because legal datasets are imbalanced and rare articles are important.

## How To Think About Success

The ideal result is not:

```text
LegalReasoner has higher F1.
```

The ideal result is:

```text
At the same target coverage, Base+Reasoner CP returns smaller sets than Base CP.
```

For example, a strong final result might look like:

```text
Base CP             coverage 0.90, avg |C| 2.60
Base+Reasoner CP    coverage 0.90, avg |C| 2.25
```

That would mean the verifier features improve conformal efficiency.

## Current Status

Rough status:

```text
idea and framing:                      strong
TF-IDF base conformal pipeline:         working
reasoner feature pipeline:              working
repair lane:                            working
100-case pilot:                         promising
500/full-scale evidence:                missing
strong encoder baseline:                missing
ablations:                              missing
rare-label analysis:                    partial/missing
paper-grade reproducibility:            in progress
```

Overall, the project is past the toy stage, but not yet paper-final.

## Next Practical Step

Run a 500-case scale-up:

```bash
sbatch --export=ALL,ECTHR_SPLIT=train,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6,MAX_CASE_CHARS=8000,VLLM_MAX_OUTPUT_TOKENS=4096 scripts/slurm/run_ecthr_reasoner_features.sbatch
sbatch --export=ALL,ECTHR_SPLIT=validation,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6,MAX_CASE_CHARS=8000,VLLM_MAX_OUTPUT_TOKENS=4096 scripts/slurm/run_ecthr_reasoner_features.sbatch
sbatch --export=ALL,ECTHR_SPLIT=test,ECTHR_N_EXAMPLES=500,LR_MAX_CANDIDATES=6,MAX_CASE_CHARS=8000,VLLM_MAX_OUTPUT_TOKENS=4096 scripts/slurm/run_ecthr_reasoner_features.sbatch
```

Then:

```bash
python -m legal.build_ecthr_feature_table --splits train,validation,test --restrict_to_reasoner_cases
python -m legal.train_ecthr_meta_scorer
python -m legal.eval_ecthr_conformal --alphas 0.05,0.1,0.2
python -m legal.report_ecthr_compare
```

If the 500-case result preserves the pilot pattern, then move to full validation/test and sharded train.
