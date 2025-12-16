# critic/dataset.py

"""
Dataset utilities for the Tiny Recursive Proof Critic.

We treat each (example_id, candidate_id, proof_text, oracle_label)
as one training example for the critic.

For now, 'oracle_label' comes from the 'valid' field produced by
verify_textual.py. Later, you can swap this to use a real ASP/Lean
oracle without changing this file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
from torch.utils.data import Dataset


@dataclass
class ProofCriticExample:
    example_id: str
    candidate_id: int
    proof_text: str
    oracle_label: int  # 1 if oracle accepts, 0 otherwise
    meta: Dict[str, Any]


class ProofCriticDataset(Dataset):
    """
    Simple in-memory Dataset over verified candidates JSONL.

    Each line in the input JSONL is expected to have at least:
      - "example_id"
      - "candidate_id"
      - "proof_text"
      - "valid" (bool)  # from verify_textual.py

    All other fields are stored under 'meta'.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.examples: List[ProofCriticExample] = []
        self._load()

    # ---------- internal helpers ----------

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Critic dataset file not found: {self.path}")

        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                example_id = str(rec.get("example_id", "UNKNOWN_ID"))
                candidate_id = int(rec.get("candidate_id", 0))
                proof_text = rec.get("proof_text", "")

                # For now we take 'valid' as the oracle label.
                # Later you can replace this with a real ASP/Lean label.
                oracle_label = 1 if rec.get("valid", False) else 0

                # Keep all extra fields as metadata (scores, etc.)
                meta = {
                    k: v
                    for k, v in rec.items()
                    if k not in ("example_id", "candidate_id", "proof_text", "valid")
                }

                self.examples.append(
                    ProofCriticExample(
                        example_id=example_id,
                        candidate_id=candidate_id,
                        proof_text=proof_text,
                        oracle_label=oracle_label,
                        meta=meta,
                    )
                )

        print(
            f"Loaded {len(self.examples)} critic examples "
            f"from {self.path}"
        )

    # ---------- Dataset API ----------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "example_id": ex.example_id,
            "candidate_id": ex.candidate_id,
            "proof_text": ex.proof_text,
            "label": ex.oracle_label,
            "meta": ex.meta,
        }


# ---------- tiny sanity check ----------

if __name__ == "__main__":
    # Default to your calib verified file
    default_path = Path("outputs/eb_task1_dev_calib_verified.jsonl")
    ds = ProofCriticDataset(default_path)

    print(f"Dataset size: {len(ds)} examples")
    # Print one random-ish example (index 0)
    if len(ds) > 0:
        sample = ds[0]
        print("--- Sample example ---")
        print(f"example_id : {sample['example_id']}")
        print(f"candidate_id: {sample['candidate_id']}")
        print(f"label      : {sample['label']}")
        print(f"text       : {sample['proof_text'][:120]}...")
