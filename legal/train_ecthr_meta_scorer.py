from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from .ecthr_features import read_jsonl, write_jsonl
except ImportError:  # pragma: no cover
    from ecthr_features import read_jsonl, write_jsonl


RANK_FEATURES = (
    "candidate_present",
    "candidate_rank",
    "candidate_rank_norm",
)

PRELIM_FEATURES = (
    *RANK_FEATURES,
    "prelim_support",
    "prelim_weak",
    "prelim_none",
    "n_reasoning_steps",
)

VERDICT_FEATURES = (
    *RANK_FEATURES,
    "verdict_supported",
    "verdict_partial",
    "verdict_unsupported",
    "hard_support_gate",
)

STEP_FEATURES = (
    *RANK_FEATURES,
    "n_step_checks",
    "n_supported_steps",
    "n_partial_steps",
    "n_unsupported_steps",
    "supported_fraction",
    "partial_fraction",
    "unsupported_fraction",
)

METHOD_FEATURES = {
    "base_only": ("base_prob",),
    "reasoner_only": None,
    "base_plus_reasoner": None,
    "base_rank": ("base_prob", *RANK_FEATURES),
    "base_prelim": ("base_prob", *PRELIM_FEATURES),
    "base_verdict": ("base_prob", *VERDICT_FEATURES),
    "base_step_counts": ("base_prob", *STEP_FEATURES),
    "base_full_no_rank": (
        "base_prob",
        "candidate_reason_chars",
        "prelim_support",
        "prelim_weak",
        "prelim_none",
        "n_reasoning_steps",
        "verdict_supported",
        "verdict_partial",
        "verdict_unsupported",
        "label_evidence_chars",
        "n_step_checks",
        "n_supported_steps",
        "n_partial_steps",
        "n_unsupported_steps",
        "supported_fraction",
        "partial_fraction",
        "unsupported_fraction",
        "hard_support_gate",
    ),
}

METHODS = tuple(METHOD_FEATURES.keys())
DEFAULT_METHODS = ("base_only", "reasoner_only", "base_plus_reasoner")


def feature_dict(row: Dict[str, Any], method: str) -> Dict[str, Any]:
    features = dict(row.get("features", {}))
    out: Dict[str, Any] = {"label": str(row["label"])}

    if method == "base_only":
        out["base_prob"] = float(features.get("base_prob", 0.0))
    elif method == "reasoner_only":
        for key, value in features.items():
            if key != "base_prob":
                out[key] = float(value)
    elif method == "base_plus_reasoner":
        for key, value in features.items():
            out[key] = float(value)
    elif method in METHOD_FEATURES:
        for key in METHOD_FEATURES[method] or ():
            out[key] = float(features.get(key, 0.0))
    else:
        raise ValueError(f"Unknown method: {method}")

    return out


def make_xy(rows: List[Dict[str, Any]], method: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    return [feature_dict(row, method) for row in rows], np.asarray([int(row["target"]) for row in rows])


def train_model(rows: List[Dict[str, Any]], method: str) -> Pipeline:
    x, y = make_xy(rows, method)
    model = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def grouped_score_rows(
    rows: List[Dict[str, Any]],
    scores: Iterable[float],
    *,
    method: str,
) -> List[Dict[str, Any]]:
    score_values = list(scores)
    if len(rows) != len(score_values):
        raise ValueError(
            f"Expected one score per feature row, got {len(score_values)} scores "
            f"for {len(rows)} rows."
        )

    by_id: Dict[int, Dict[str, Any]] = {}
    for row, score in zip(rows, score_values):
        case_id = int(row["id"])
        rec = by_id.setdefault(
            case_id,
            {
                "id": case_id,
                "split": row["split"],
                "gold_labels": row.get("gold_labels", []),
                "scores": {},
                "method": method,
            },
        )
        rec["scores"][str(row["label"])] = float(score)

    return [by_id[k] for k in sorted(by_id)]


def predict_scores(model: Pipeline, rows: List[Dict[str, Any]], method: str) -> List[float]:
    x = [feature_dict(row, method) for row in rows]
    proba = model.predict_proba(x)
    return [float(p) for p in proba[:, 1]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ECtHR meta-scorers over base + LegalReasoner features.")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/ecthr_b"))
    p.add_argument("--train_table", type=Path, default=None)
    p.add_argument("--predict_splits", default="validation,test")
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated methods. Available: {','.join(METHODS)}",
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Model directory. Defaults to <out_dir>/models.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_table = args.train_table or (args.out_dir / "feature_table_train.jsonl")
    model_dir = args.model_dir or (args.out_dir / "models")
    train_rows = read_jsonl(train_table)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    predict_splits = [s.strip() for s in args.predict_splits.split(",") if s.strip()]

    model_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"methods": {}, "train_table": str(train_table)}

    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

        model = train_model(train_rows, method)
        model_path = model_dir / f"{method}.joblib"
        joblib.dump(model, model_path)
        meta["methods"][method] = {"model_path": str(model_path)}

        for split in predict_splits:
            table_path = args.out_dir / f"feature_table_{split}.jsonl"
            rows = read_jsonl(table_path)
            scores = predict_scores(model, rows, method)
            grouped = grouped_score_rows(rows, scores, method=method)
            out = args.out_dir / f"meta_scores_{method}_{split}.jsonl"
            write_jsonl(out, grouped)
            print(f"Wrote {len(grouped)} {method} rows to {out}")

    (args.out_dir / "meta_scorer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
