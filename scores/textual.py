# scores/textual.py

"""
Nonconformity scores for textual proofs.

Given verified candidate records (with fields:
  - valid (bool)
  - unsupported (int)
  - leaf_mismatch (int)
  - invalid_nodes (int)

we compute a scalar nonconformity score s(π) as:

  s(π) = 0                               if valid
         10 + 3*unsupported
              + 2*leaf_mismatch
              + 1*invalid_nodes          otherwise

This script reads a verified JSONL file and writes another JSONL file
with an added 'score' field.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

DEFAULT_INPUT = Path("outputs/eb_task1_dev_calib_verified.jsonl")
DEFAULT_OUTPUT = Path("outputs/eb_task1_dev_calib_scored.jsonl")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def nonconformity_score(cand: Dict[str, Any]) -> float:
    """
    Compute s(π) given a verified candidate record.
    """
    valid = bool(cand.get("valid", False))

    if valid:
        return 0.0

    unsupported = int(cand.get("unsupported", 0))
    leaf_mismatch = int(cand.get("leaf_mismatch", 0))
    invalid_nodes = int(cand.get("invalid_nodes", 0))

    score = 10 + 3 * unsupported + 2 * leaf_mismatch + 1 * invalid_nodes
    return float(score)


def process(input_path: Path, output_path: Path) -> None:
    count = 0
    scores = []

    out_records = []

    for cand in read_jsonl(input_path):
        count += 1
        s_pi = nonconformity_score(cand)
        cand_out = dict(cand)
        cand_out["score"] = s_pi
        out_records.append(cand_out)
        scores.append(s_pi)

    write_jsonl(out_records, output_path)

    if count == 0:
        print(f"No candidates found in {input_path}")
        return

    # Simple summary
    num_zero = sum(1 for s in scores if s == 0.0)
    print(f"Read {count} candidates from {input_path}")
    print(f"Scores: min={min(scores)}, max={max(scores)}")
    print(f"{num_zero} candidates have score 0 (valid).")
    print(f"Wrote scored candidates to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute nonconformity scores for textual proofs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Verified candidates JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL with scores (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    process(args.input, args.output)


if __name__ == "__main__":
    main()
