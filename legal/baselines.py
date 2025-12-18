# legal/baselines.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Baselines for LEXam: Top-k selection by score.")
    p.add_argument("--eval_scored", type=Path, required=True, help="outputs/lexam_eval_scored.jsonl")
    p.add_argument("--k", type=int, default=4, help="Top-k by lowest nonconformity score")
    p.add_argument("--out", type=Path, default=Path("outputs/lexam_baselines.json"))
    p.add_argument("--good", choices=["correct", "acceptable"], default="correct",
                   help="Coverage condition: at least one good in selected set")
    args = p.parse_args()

    rows = read_jsonl(args.eval_scored)

    # group by question_id
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        qid = str(r["question_id"])
        by_q.setdefault(qid, []).append(r)

    n_q = len(by_q)
    covered = 0
    total_size = 0

    for qid, cands in by_q.items():
        # sort ascending by score (lower is "better")
        cands_sorted = sorted(cands, key=lambda x: float(x.get("score", 1.0)))
        selected = cands_sorted[: args.k]
        total_size += len(selected)

        if args.good == "correct":
            is_cov = any(c.get("correct") is True for c in selected)
        else:
            is_cov = any(c.get("acceptable") is True for c in selected)

        if is_cov:
            covered += 1

    coverage = covered / n_q if n_q else 0.0
    avg_set_size = total_size / n_q if n_q else 0.0

    summary = {
        "baseline": "top_k",
        "k": args.k,
        "good": args.good,
        "n_questions": n_q,
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "source_eval_scored": str(args.eval_scored),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("Top-k baseline summary:")
    print(f"  k={args.k}, good={args.good}")
    print(f"  coverage={coverage:.3f}, avg_set_size={avg_set_size:.2f} (n_questions={n_q})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
