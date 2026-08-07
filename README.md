# CAPS

Calibrated multi-label prediction for European Court of Human Rights cases.

CAPS combines a lightweight text classifier with structured evidence signals from a local language model, then uses split conformal prediction to return a set of plausible Convention articles. The repository covers the full path from cross-fitted base scores to calibration and evaluation.

```text
case text → base scores → LLM evidence features → meta-scorer → conformal label set
```

## Results

The checked-in run uses 500 ECtHR-B validation cases for calibration and 500 test cases for evaluation.

| Method | Alpha | Coverage | Avg labels | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Base only | 0.10 | 0.894 | 2.616 | 0.632 |
| Base + reasoner | 0.10 | 0.898 | 2.668 | 0.628 |
| Base only | 0.20 | 0.782 | 1.780 | 0.711 |
| Base + reasoner | 0.20 | 0.782 | 1.690 | 0.731 |

At alpha 0.20, the combined scorer reduced average set size by 5.1% at the same observed coverage. At alpha 0.10, it matched coverage but did not improve efficiency. These are pilot results, not a claim about legal reliability.

The full metrics, including labelwise calibration and the reasoner-only ablation, are in [`results/ecthr_b/metrics.json`](results/ecthr_b/metrics.json).

## How it works

1. A TF-IDF logistic regression model produces out-of-fold label scores for the training split.
2. A local instruction model reviews the top candidate articles and returns structured reasoning and verification fields.
3. Logistic regression meta-scorers combine the base probabilities with the derived evidence features.
4. Split conformal calibration turns the scores into article sets at several target error levels.

The language model is used as a feature generator. It cannot remove labels outside its top candidates, and its text is not presented as a legal explanation.

## Repository

| Path | Purpose |
| --- | --- |
| `legal/` | ECtHR-B scoring, feature generation, training, and evaluation |
| `conformal/` | Conformal calibration and set construction |
| `scripts/` | Local helpers and Slurm jobs |
| `tests/` | Unit and pipeline contract tests |
| `results/` | Compact checked-in metrics |
| `docs/reproduction.md` | Full run instructions |

Generated models, score tables, and raw LLM traces are written to `outputs/`, which is ignored by Git.

## Quick start

CAPS targets Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Inspect the checked-in comparison:

```bash
python -m legal.report_ecthr_compare --alpha 0.2
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The reasoner stage needs a Linux GPU environment and separate dependencies:

```bash
pip install -r requirements-gpu.txt
```

See [`docs/reproduction.md`](docs/reproduction.md) for the complete pipeline and Slurm commands.

## Data and models

- Dataset: [LexGLUE ECtHR-B](https://huggingface.co/datasets/coastalcph/lex_glue)
- Base model: TF-IDF with one-vs-rest logistic regression
- Reasoner model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) served through vLLM
- Calibration split: the official ECtHR-B validation split


