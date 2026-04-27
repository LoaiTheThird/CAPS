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
    require_complete_reasoner: bool,
    restrict_to_reasoner_cases: bool,
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

    base_ids = {int(row["id"]) for row in base_rows}
    if reasoner_by_id:
        missing_reasoner_ids = sorted(base_ids - set(reasoner_by_id))
        if missing_reasoner_ids:
            msg = (
                f"{split}: reasoner features cover {len(reasoner_by_id)}/{len(base_ids)} "
                f"base cases; {len(missing_reasoner_ids)} cases will use default reasoner features."
            )
            if require_complete_reasoner:
                preview = ", ".join(str(case_id) for case_id in missing_reasoner_ids[:10])
                raise ValueError(f"{msg} First missing ids: {preview}")
            print(f"[WARN] {msg}")

    default_features = default_reasoner_features(label_names)
    rows: List[Dict[str, Any]] = []

    for base in base_rows:
        case_id = int(base["id"])
        if restrict_to_reasoner_cases and reasoner_by_id and case_id not in reasoner_by_id:
            continue

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
    p.add_argument(
        "--require_complete_reasoner",
        action="store_true",
        help="Fail if a reasoner file exists but does not cover every base-score case.",
    )
    p.add_argument(
        "--restrict_to_reasoner_cases",
        action="store_true",
        help="For pilots, only emit cases that have a reasoner feature row.",
    )
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
            require_complete_reasoner=args.require_complete_reasoner,
            restrict_to_reasoner_cases=args.restrict_to_reasoner_cases,
        )
        out = args.out_dir / f"feature_table_{split}.jsonl"
        write_jsonl(out, rows)
        print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
