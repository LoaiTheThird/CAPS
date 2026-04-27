from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

try:
    from .gen_common import DEFAULT_MAX_CASE_CHARS, build_case_text, load_split, to_multihot_from_ids
    from .ecthr_features import write_jsonl
except ImportError:  # pragma: no cover
    from gen_common import DEFAULT_MAX_CASE_CHARS, build_case_text, load_split, to_multihot_from_ids
    from ecthr_features import write_jsonl


def build_xy(split: str, n_examples: int | None) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]], List[str]]:
    ds, label_names = load_split(split, n_examples)
    texts: List[str] = []
    y_rows: List[List[int]] = []
    meta: List[Dict[str, Any]] = []

    for idx, ex in enumerate(ds):
        text = build_case_text(ex["text"], max_chars=DEFAULT_MAX_CASE_CHARS)
        labels = [int(i) for i in ex["labels"]]
        texts.append(text)
        y_rows.append(to_multihot_from_ids(labels, len(label_names)))
        meta.append(
            {
                "id": idx,
                "split": split,
                "gold_label_ids": labels,
                "gold_labels": [label_names[i] for i in labels],
                "case_chars": len(text),
            }
        )

    return texts, np.asarray(y_rows, dtype=int), meta, label_names


def make_model(max_features: int) -> Tuple[TfidfVectorizer, OneVsRestClassifier]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
    )
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
        )
    )
    return vectorizer, clf


def fit_predict(
    train_texts: List[str],
    train_y: np.ndarray,
    texts: List[str],
    *,
    max_features: int,
) -> np.ndarray:
    vectorizer, clf = make_model(max_features)
    x_train = vectorizer.fit_transform(train_texts)
    clf.fit(x_train, train_y)
    return clf.predict_proba(vectorizer.transform(texts))


def crossfit_train_scores(
    texts: List[str],
    y: np.ndarray,
    *,
    max_features: int,
    folds: int,
    seed: int,
) -> np.ndarray:
    scores = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, held_idx in kf.split(texts):
        train_texts = [texts[i] for i in train_idx]
        held_texts = [texts[i] for i in held_idx]
        held_scores = fit_predict(train_texts, y[train_idx], held_texts, max_features=max_features)
        scores[held_idx] = held_scores
    return scores


def rows_from_scores(
    meta: List[Dict[str, Any]],
    label_names: List[str],
    scores: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for rec, score_row in zip(meta, scores):
        rows.append(
            {
                **rec,
                "scores": {
                    label: float(score_row[i])
                    for i, label in enumerate(label_names)
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a TF-IDF legal classifier and emit ECtHR-B per-label scores.")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/ecthr_b"))
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_eval", type=int, default=None)
    p.add_argument("--max_features", type=int, default=200_000)
    p.add_argument("--crossfit_train", action="store_true", help="Emit out-of-fold train scores for meta-scorer training.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_out", type=Path, default=Path("outputs/ecthr_b/base_tfidf_ovr.joblib"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_texts, train_y, train_meta, label_names = build_xy("train", args.n_train)
    vectorizer, clf = make_model(args.max_features)
    x_train = vectorizer.fit_transform(train_texts)
    clf.fit(x_train, train_y)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf, "label_names": label_names}, args.model_out)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split == "train":
            if args.crossfit_train:
                scores = crossfit_train_scores(
                    train_texts,
                    train_y,
                    max_features=args.max_features,
                    folds=args.folds,
                    seed=args.seed,
                )
            else:
                scores = clf.predict_proba(x_train)
            rows = rows_from_scores(train_meta, label_names, scores)
        else:
            texts, _y, meta, split_labels = build_xy(split, args.n_eval)
            if split_labels != label_names:
                raise RuntimeError("Label schema changed across splits.")
            scores = clf.predict_proba(vectorizer.transform(texts))
            rows = rows_from_scores(meta, label_names, scores)

        out = args.out_dir / f"base_scores_{split}.jsonl"
        write_jsonl(out, rows)
        print(f"Wrote {len(rows)} rows to {out}")

    meta = {
        "model": "tfidf_ovr_logreg",
        "label_names": label_names,
        "max_features": args.max_features,
        "crossfit_train": bool(args.crossfit_train),
        "folds": args.folds if args.crossfit_train else None,
    }
    (args.out_dir / "base_scores_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
