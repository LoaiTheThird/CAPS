# data/make_lexam_splits.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

from datasets import load_dataset

from legal.schema import LegalExample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create calib/eval splits for LEXam.")
    p.add_argument("--config", type=str, default="mcq_4_choices",
                   help="LEXam config name (e.g., mcq_4_choices, open_question).")
    p.add_argument("--split", type=str, default="test",
                   help="HF split to use (LEXam mcq configs often only have 'test').")
    p.add_argument("--calib_size", type=int, default=500,
                   help="Number of examples to put in calib.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=Path("data"))
    return p.parse_args()


def to_legal_example(rec: Dict[str, Any]) -> LegalExample:
    """
    Map a LEXam mcq record into our LegalExample schema.

    We pack the multiple choice structure into a single 'context' string
    so the generator/judge can see it easily.
    """
    qid = str(rec["id"])
    question = rec["question"]

    # Choices is typically a list of strings
    choices = rec.get("choices", [])
    gold = rec.get("gold", None)

    # Build a compact context string the LLM can use
    # Example:
    # Choices:
    # A) ...
    # B) ...
    # C) ...
    # D) ...
    # Gold: B  (optional, we store separately too)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_lines = []
    for i, ch in enumerate(choices):
        label = letters[i] if i < len(letters) else str(i)
        choice_lines.append(f"{label}) {ch}")

    context = ""
    if choice_lines:
        context = "Choices:\n" + "\n".join(choice_lines)

    meta = {
        "course": rec.get("course"),
        "language": rec.get("language"),
        "area": rec.get("area"),
        "jurisdiction": rec.get("jurisdiction"),
        "year": rec.get("year"),
        "n_statements": rec.get("n_statements"),
        "none_as_an_option": rec.get("none_as_an_option"),
        "negative_question": rec.get("negative_question"),
    }

    # gold_answer: store gold label or value (depends on dataset encoding)
    gold_answer = None
    if gold is not None:
        gold_answer = str(gold)

    return LegalExample(
        question_id=qid,
        question=question,
        context=context if context else None,
        gold_answer=gold_answer,
        metadata=meta,
    )


def write_jsonl(path: Path, items: List[LegalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in items:
            f.write(ex.model_dump_json() + "\n")


def main() -> None:
    args = parse_args()

    ds = load_dataset("LEXam-Benchmark/LEXam", args.config)
    if args.split not in ds:
        raise SystemExit(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    rows = list(ds[args.split])
    print(f"Loaded {len(rows)} examples from LEXam config={args.config} split={args.split}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    calib_size = min(args.calib_size, len(rows))
    calib_rows = rows[:calib_size]
    eval_rows = rows[calib_size:]

    calib = [to_legal_example(r) for r in calib_rows]
    eval_ = [to_legal_example(r) for r in eval_rows]

    out_calib = args.out_dir / f"lexam_{args.config}_calib.jsonl"
    out_eval = args.out_dir / f"lexam_{args.config}_eval.jsonl"

    write_jsonl(out_calib, calib)
    write_jsonl(out_eval, eval_)

    print(f"Wrote calib: {out_calib}  (n={len(calib)})")
    print(f"Wrote eval : {out_eval}  (n={len(eval_)})")


if __name__ == "__main__":
    main()
