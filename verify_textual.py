# verify_textual.py

"""
Textual proof verification (Day 4, version 0.1 – stubbed).

This script:
  - reads candidate textual proofs from a JSONL file
  - parses each into a minimal ProofTree (stub)
  - applies a stub "verifier" that marks proofs as valid/invalid
  - records simple error counts that will later be computed by the ASP checker

Later:
  - replace `verify_proof_stub` with a real ASP-based checker that uses
    EntailmentBank facts & rules.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from proofs.schema import ProofNode, ProofTree

# Default paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DEFAULT_INPUT = OUTPUT_DIR / "eb_task1_dev_calib_candidates.jsonl"
DEFAULT_OUTPUT = OUTPUT_DIR / "eb_task1_dev_calib_verified.jsonl"


# ---------- IO helpers ----------

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


# ---------- Stub parsing / verification ----------

def build_stub_proof_tree(example_id: Any, candidate_id: Any, proof_text: str) -> ProofTree:
    """
    For now we wrap the whole proof_text into a single-node ProofTree.

    Later:
      - this function will parse structured EB-style proofs
        into multiple nodes with explicit premises and rules.
    """
    node_id = f"{example_id}_{candidate_id}_h1"
    node = ProofNode(
        id=node_id,
        text=proof_text,
        premises=[],
        rule="stub-root",
    )
    tree = ProofTree(
        id=str(example_id),
        hypothesis_id=node_id,
        nodes=[node],
    )
    return tree


def verify_proof_stub(tree: ProofTree) -> Tuple[bool, Dict[str, int]]:
    """
    Very simple heuristic verifier.

    Right now:
      - we mark a proof as 'valid' if it contains words like 'therefore' or 'thus'.
      - otherwise it's 'invalid'.

    We also return counts for:
      - unsupported
      - leaf_mismatch
      - invalid_nodes

    These are placeholders so that later, the ASP-based checker can compute
    real values but keep the same interface.
    """
    text = tree.nodes[0].text.lower()

    valid_markers = ("therefore", "thus", "so ", "hence")
    is_valid = any(marker in text for marker in valid_markers)

    if is_valid:
        metrics = {"unsupported": 0, "leaf_mismatch": 0, "invalid_nodes": 0}
    else:
        # Stub: treat "bad proofs" as having one unsupported step and one invalid node.
        metrics = {"unsupported": 1, "leaf_mismatch": 0, "invalid_nodes": 1}

    return is_valid, metrics


# ---------- Main pipeline ----------

def process_candidates(
    input_path: Path,
    output_path: Path,
) -> None:
    records_out = []

    num_in = 0
    num_valid = 0

    for cand in read_jsonl(input_path):
        num_in += 1
        example_id = cand.get("example_id", f"ex-{num_in}")
        candidate_id = cand.get("candidate_id", 0)
        proof_text = cand.get("proof_text", "")

        tree = build_stub_proof_tree(example_id, candidate_id, proof_text)
        is_valid, metrics = verify_proof_stub(tree)

        if is_valid:
            num_valid += 1

        # Attach verification results to the original record
        cand_out = dict(cand)  # shallow copy
        cand_out["valid"] = is_valid
        cand_out["unsupported"] = int(metrics["unsupported"])
        cand_out["leaf_mismatch"] = int(metrics["leaf_mismatch"])
        cand_out["invalid_nodes"] = int(metrics["invalid_nodes"])

        records_out.append(cand_out)

    write_jsonl(records_out, output_path)

    print(f"Read {num_in} candidates from {input_path}")
    print(f"Marked {num_valid} as valid (stub heuristic).")
    print(f"Wrote verified candidates to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify textual proof candidates (stub).")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input candidates JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL with verification fields (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    process_candidates(args.input, args.output)


if __name__ == "__main__":
    main()
