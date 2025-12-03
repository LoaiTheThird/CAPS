# run_generate_textual.py

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from conformal.generator_stub import generate_candidates

DATA_DIR = Path("data")
INPUT_SPLIT = DATA_DIR / "eb_task1_dev_calib.jsonl"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "eb_task1_dev_calib_candidates.jsonl"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_prompt(example: Dict[str, Any]) -> str:
    """
    Very simple prompt builder for now.

    We don't rely on specific field names yet; we just show the whole
    example JSON and ask for a proof. Later you can switch this to use
    e.g. 'hypothesis', 'facts', etc. explicitly.
    """
    eid = example.get("id", "UNKNOWN_ID")
    pretty = json.dumps(example, indent=2, ensure_ascii=False)
    prompt = (
        f"EntailmentBank example ID: {eid}\n"
        f"Below is a JSON description of the example.\n"
        f"Write a clear, step-by-step natural language proof that explains "
        f"why the hypothesis follows from the facts.\n\n"
        f"{pretty}\n\n"
        f"Proof:"
    )
    return prompt


def main() -> None:
    if not INPUT_SPLIT.exists():
        raise SystemExit(f"Input split not found: {INPUT_SPLIT}")

    num_examples = 0
    num_per_example = 16  # can change to 32 later

    with OUT_PATH.open("w") as out_f:
        for example in read_jsonl(INPUT_SPLIT):
            num_examples += 1
            example_id = example.get("id", f"ex-{num_examples}")
            prompt = build_prompt(example)

            candidates = generate_candidates(prompt, num_samples=num_per_example)

            for cid, proof_text in enumerate(candidates):
                record: Dict[str, Any] = {
                    "example_id": example_id,
                    "candidate_id": cid,
                    "prompt": prompt,
                    "proof_text": proof_text,
                    # Later you'll add parsed ProofTrees, scores, etc.
                }
                out_f.write(json.dumps(record) + "\n")

    print(f"Processed {num_examples} examples from {INPUT_SPLIT}")
    print(f"Wrote candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()
