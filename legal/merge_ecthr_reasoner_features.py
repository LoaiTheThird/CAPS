from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from .ecthr_features import read_jsonl, write_jsonl
    from .run_ecthr_reasoner_features import record_needs_rerun
except ImportError:  # pragma: no cover
    from ecthr_features import read_jsonl, write_jsonl
    from run_ecthr_reasoner_features import record_needs_rerun


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge ECtHR reasoner repair rows back into a split JSONL."
    )
    p.add_argument("--base", type=Path, required=True, help="Original reasoner_features_<split>.jsonl")
    p.add_argument(
        "--repairs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more rerun/repair JSONL files.",
    )
    p.add_argument("--out", type=Path, required=True, help="Merged output path.")
    p.add_argument(
        "--accept_incomplete",
        action="store_true",
        help="Replace rows even if the repair row still has errors or incomplete stages.",
    )
    p.add_argument(
        "--allow_new_ids",
        action="store_true",
        help="Append repair rows whose ids do not appear in --base.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base)
    base_by_id = {int(row["id"]): row for row in base_rows}
    base_order = [int(row["id"]) for row in base_rows]

    repair_by_id: Dict[int, Dict[str, Any]] = {}
    n_repair_rows = 0
    n_skipped_incomplete = 0
    n_new_ids = 0

    for repair_path in args.repairs:
        for row in read_jsonl(repair_path):
            n_repair_rows += 1
            case_id = int(row["id"])
            if case_id not in base_by_id and not args.allow_new_ids:
                n_new_ids += 1
                continue
            if not args.accept_incomplete and record_needs_rerun(row):
                n_skipped_incomplete += 1
                continue
            repair_by_id[case_id] = row

    merged: List[Dict[str, Any]] = []
    n_replaced = 0
    for case_id in base_order:
        if case_id in repair_by_id:
            merged.append(repair_by_id[case_id])
            n_replaced += 1
        else:
            merged.append(base_by_id[case_id])

    if args.allow_new_ids:
        for case_id in sorted(set(repair_by_id) - set(base_by_id)):
            merged.append(repair_by_id[case_id])

    write_jsonl(args.out, merged)
    print(f"Base rows: {len(base_rows)}")
    print(f"Repair rows read: {n_repair_rows}")
    print(f"Rows replaced: {n_replaced}")
    print(f"Repair rows skipped as incomplete: {n_skipped_incomplete}")
    print(f"Repair rows skipped because id was not in base: {n_new_ids}")
    print(f"Wrote {len(merged)} rows to {args.out}")


if __name__ == "__main__":
    main()
