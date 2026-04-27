from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from .ecthr_features import default_reasoner_features, read_jsonl, write_jsonl
except ImportError:  # pragma: no cover
    from ecthr_features import default_reasoner_features, read_jsonl, write_jsonl


def build_rows_for_split(
    *,
    base_path: Path,
    reasoner_path: Path | None,
    split: str,
    allow_missing_reasoner: bool,
) -> List[Dict[str, Any]]:
    base_rows = read_jsonl(base_path)
    if not base_rows:
        raise ValueError(f"No base rows found in {base_path}")

    label_names = list((base_rows[0].get("scores") or {}).keys())
    if not label_names:
        raise ValueError(f"Base scores missing label scores: {base_path}")

    reasoner_by_id: Dict[int, Dict[str, Any]] = {}
    if reasoner_path and reasoner_path.exists():
        reasoner_by_id = {int(row["id"]): row for row in read_jsonl(reasoner_path)}
    elif reasoner_path and not allow_missing_reasoner:
        raise FileNotFoundError(reasoner_path)

    default_features = default_reasoner_features(label_names)
    rows: List[Dict[str, Any]] = []

    for base in base_rows:
        case_id = int(base["id"])
        scores = base["scores"]
        gold_labels = [str(label) for label in base.get("gold_labels", [])]
        gold_set = set(gold_labels)
        reasoner = reasoner_by_id.get(case_id)
        per_label = (
            reasoner.get("per_label_features", {})
            if reasoner is not None
            else default_features
        )

        for label in label_names:
            reasoner_features = dict(default_features.get(label, {}))
            reasoner_features.update(per_label.get(label, {}))
            features = {
                "base_prob": float(scores.get(label, 0.0)),
                "case_chars": float(base.get("case_chars", 0)),
                **{k: float(v) for k, v in reasoner_features.items()},
            }
            rows.append(
                {
                    "id": case_id,
                    "split": split,
                    "label": label,
                    "target": int(label in gold_set),
                    "gold_labels": gold_labels,
                    "features": features,
                }
            )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge ECtHR base scores and LegalReasoner features into per-label rows.")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/ecthr_b"))
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--allow_missing_reasoner", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        base_path = args.out_dir / f"base_scores_{split}.jsonl"
        reasoner_path = args.out_dir / f"reasoner_features_{split}.jsonl"
        rows = build_rows_for_split(
            base_path=base_path,
            reasoner_path=reasoner_path,
            split=split,
            allow_missing_reasoner=args.allow_missing_reasoner,
        )
        out = args.out_dir / f"feature_table_{split}.jsonl"
        write_jsonl(out, rows)
        print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
