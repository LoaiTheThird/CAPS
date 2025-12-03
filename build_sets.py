# build_sets.py

"""
Build conformal prediction sets and compute coverage / set size.

Given:
  - a JSONL file of scored candidates (e.g. eval split) with fields
      - example_id
      - candidate_id
      - valid (bool)
      - score (float)
  - a JSON file of quantiles from conformal/split.py

we build, for each alpha:
  S_alpha(x) = { candidates with score <= q_alpha }

and compute:
  - Coverage@alpha: fraction of examples where S_alpha(x)
                    contains at least one valid candidate.
  - Avg |S_alpha(x)|: average set size across examples.

Outputs:
  - JSONL with, per example, the candidate IDs in each S_alpha(x)
  - JSON with global metrics per alpha
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INPUT = Path("outputs/eb_task1_dev_eval_scored.jsonl")
DEFAULT_QUANTILES = Path("outputs/eb_task1_dev_calib_quantiles.json")
DEFAULT_SETS_OUT = Path("outputs/eb_task1_dev_eval_sets.jsonl")
DEFAULT_METRICS_OUT = Path("outputs/eb_task1_dev_eval_metrics.json")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_by_example(
    path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group candidate records by example_id.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in read_jsonl(path):
        eid = str(rec.get("example_id", "UNKNOWN_ID"))
        grouped[eid].append(rec)
    return grouped


def load_quantiles(path: Path) -> Dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    q = data.get("quantiles", {})
    # keys are strings of alpha, values are floats
    return {str(k): float(v) for k, v in q.items()}


def build_sets_and_metrics(
    examples: Dict[str, List[Dict[str, Any]]],
    quantiles: Dict[str, float],
) -> (List[Dict[str, Any]], Dict[str, Any]):
    """
    Build sets S_alpha(x) and compute metrics.

    Returns:
      - list of per-example records for JSONL output
      - metrics dict for JSON output
    """
    alpha_keys = sorted(quantiles.keys(), key=float)

    sets_records: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}

    n_examples = len(examples)
    if n_examples == 0:
        raise ValueError("No examples provided.")

    # Prepare accumulators
    cov_counts = {a: 0 for a in alpha_keys}
    total_sizes = {a: 0 for a in alpha_keys}

    for eid, cands in examples.items():
        # Per-example record
        ex_record: Dict[str, Any] = {
            "example_id": eid,
            "candidates": [
                {
                    "candidate_id": c.get("candidate_id"),
                    "score": c.get("score"),
                    "valid": c.get("valid", False),
                }
                for c in cands
            ],
            "sets": {},
        }

        # For each alpha, build S_alpha(x) and update metrics
        for a in alpha_keys:
            q_alpha = quantiles[a]
            S_ids = [
                c["candidate_id"]
                for c in cands
                if float(c.get("score", float("inf"))) <= q_alpha
            ]

            ex_record["sets"][a] = S_ids

            total_sizes[a] += len(S_ids)

            # coverage: does S_alpha(x) contain at least one valid candidate?
            covered = any(
                (c.get("candidate_id") in S_ids) and bool(c.get("valid", False))
                for c in cands
            )
            if covered:
                cov_counts[a] += 1

        sets_records.append(ex_record)

    # Compute global metrics
    for a in alpha_keys:
        coverage = cov_counts[a] / n_examples
        avg_size = total_sizes[a] / n_examples
        metrics[a] = {
            "coverage": coverage,
            "avg_set_size": avg_size,
            "n_examples": n_examples,
            "q_alpha": quantiles[a],
        }

    return sets_records, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conformal prediction sets and compute coverage/cardinality."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Scored eval JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--quantiles",
        type=Path,
        default=DEFAULT_QUANTILES,
        help=f"Quantiles JSON from conformal.split (default: {DEFAULT_QUANTILES})",
    )
    parser.add_argument(
        "--sets_out",
        type=Path,
        default=DEFAULT_SETS_OUT,
        help=f"Output JSONL with sets per example (default: {DEFAULT_SETS_OUT})",
    )
    parser.add_argument(
        "--metrics_out",
        type=Path,
        default=DEFAULT_METRICS_OUT,
        help=f"Output JSON with global metrics (default: {DEFAULT_METRICS_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input scored file not found: {args.input}")
    if not args.quantiles.exists():
        raise SystemExit(f"Quantiles file not found: {args.quantiles}")

    examples = group_by_example(args.input)
    print(f"Loaded {len(examples)} examples from {args.input}")

    quantiles = load_quantiles(args.quantiles)
    print("Using quantiles:")
    for a, q in quantiles.items():
        print(f"  alpha={a}: q_alpha={q}")

    sets_records, metrics = build_sets_and_metrics(examples, quantiles)

    # Write sets JSONL
    args.sets_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sets_out.open("w") as f:
        for rec in sets_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote per-example sets to {args.sets_out}")

    # Write metrics JSON
    with args.metrics_out.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote global metrics to {args.metrics_out}")

    print("Metrics summary:")
    for a, m in metrics.items():
        print(
            f"  alpha={a}: coverage={m['coverage']:.3f}, "
            f"avg_set_size={m['avg_set_size']:.2f}, "
            f"q_alpha={m['q_alpha']}"
        )


if __name__ == "__main__":
    main()
