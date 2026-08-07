from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score, recall_score

try:
    from .ecthr_features import read_jsonl
except ImportError:  # pragma: no cover
    from ecthr_features import read_jsonl

from conformal.ecthr_multilabel import (
    calibrate_global_threshold,
    calibrate_label_thresholds,
    prediction_set,
    prediction_set_labelwise,
    set_covers_gold,
)


def parse_alphas(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def rows_to_metric_arrays(rows: List[Dict[str, Any]], label_names: List[str], predicted_sets: List[List[str]]):
    if len(rows) != len(predicted_sets):
        raise ValueError(
            f"Expected one prediction set per case, got {len(predicted_sets)} sets "
            f"for {len(rows)} cases."
        )

    label_to_idx = {label: i for i, label in enumerate(label_names)}
    y_true = np.zeros((len(rows), len(label_names)), dtype=int)
    y_pred = np.zeros((len(rows), len(label_names)), dtype=int)
    for i, (row, pred_labels) in enumerate(zip(rows, predicted_sets)):
        for label in row.get("gold_labels", []):
            if label in label_to_idx:
                y_true[i, label_to_idx[label]] = 1
        for label in pred_labels:
            if label in label_to_idx:
                y_pred[i, label_to_idx[label]] = 1
    return y_true, y_pred


def evaluate_sets(
    rows: List[Dict[str, Any]],
    label_names: List[str],
    predicted_sets: List[List[str]],
) -> Dict[str, Any]:
    if len(rows) != len(predicted_sets):
        raise ValueError(
            f"Expected one prediction set per case, got {len(predicted_sets)} sets "
            f"for {len(rows)} cases."
        )

    covered = [
        set_covers_gold(pred, row.get("gold_labels", []))
        for row, pred in zip(rows, predicted_sets)
    ]
    y_true, y_pred = rows_to_metric_arrays(rows, label_names, predicted_sets)

    per_label = {}
    for j, label in enumerate(label_names):
        positives = int(y_true[:, j].sum())
        if positives:
            per_label[label] = {
                "positives": positives,
                "recall": float(((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum() / positives),
            }
        else:
            per_label[label] = {"positives": 0, "recall": None}

    return {
        "n_cases": len(rows),
        "coverage": float(sum(covered) / len(covered)) if covered else 0.0,
        "avg_set_size": float(np.mean([len(s) for s in predicted_sets])) if predicted_sets else 0.0,
        "median_set_size": float(np.median([len(s) for s in predicted_sets])) if predicted_sets else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label_recall": per_label,
    }


def label_names_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        raise ValueError("No score rows found.")
    return list((rows[0].get("scores") or {}).keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate and evaluate conformal ECtHR multi-label predictors.")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/ecthr_b"))
    p.add_argument("--methods", default="base_only,reasoner_only,base_plus_reasoner")
    p.add_argument("--alphas", default="0.05,0.1,0.2")
    p.add_argument("--calib_split", default="validation")
    p.add_argument("--test_split", default="test")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Metrics path. Defaults to <out_dir>/conformal_metrics.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (args.out_dir / "conformal_metrics.json")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    alphas = parse_alphas(args.alphas)
    report: Dict[str, Any] = {
        "calib_split": args.calib_split,
        "test_split": args.test_split,
        "alphas": alphas,
        "methods": {},
    }

    for method in methods:
        calib_rows = read_jsonl(args.out_dir / f"meta_scores_{method}_{args.calib_split}.jsonl")
        test_rows = read_jsonl(args.out_dir / f"meta_scores_{method}_{args.test_split}.jsonl")
        label_names = label_names_from_rows(test_rows)

        method_report: Dict[str, Any] = {"global": {}, "labelwise": {}}
        for alpha in alphas:
            q_global = calibrate_global_threshold(calib_rows, alpha=alpha)
            pred_global = [prediction_set(row["scores"], q_global) for row in test_rows]
            method_report["global"][str(alpha)] = {
                "q_alpha": q_global,
                **evaluate_sets(test_rows, label_names, pred_global),
            }

            q_by_label = calibrate_label_thresholds(
                calib_rows,
                label_names=label_names,
                alpha=alpha,
                fallback_q=q_global,
            )
            pred_labelwise = [
                prediction_set_labelwise(row["scores"], q_by_label, fallback_q=q_global)
                for row in test_rows
            ]
            method_report["labelwise"][str(alpha)] = {
                "fallback_q_alpha": q_global,
                "q_by_label": q_by_label,
                **evaluate_sets(test_rows, label_names, pred_labelwise),
            }

        report["methods"][method] = method_report

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for method, method_report in report["methods"].items():
        row = method_report["global"].get("0.1") or next(iter(method_report["global"].values()))
        print(
            f"{method:20s} global: coverage={row['coverage']:.3f} "
            f"avg|C|={row['avg_set_size']:.2f} microF1={row['micro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
