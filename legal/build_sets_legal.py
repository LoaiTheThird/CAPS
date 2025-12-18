# legal/build_sets_legal.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build CP trace sets for LEXam and compute metrics.")
    p.add_argument("--eval_scored", type=Path, required=True, help="Scored eval candidates JSONL")
    p.add_argument("--qalphas", type=Path, required=True, help="Thresholds JSON from conformal.legal_split")
    p.add_argument("--sets_out", type=Path, required=True, help="Per-question sets JSONL output")
    p.add_argument("--metrics_out", type=Path, required=True, help="Global metrics JSON output")
    p.add_argument("--use_good", choices=["correct", "acceptable"], default="correct",
                   help="Coverage condition: does the selected set contain at least one good trace?")
    return_args = p.parse_args()

    eval_rows = read_jsonl(return_args.eval_scored)
    qobj = json.loads(return_args.qalphas.read_text())
    alphas = {float(k): float(v["q_alpha"]) for k, v in qobj["alphas"].items()}

    # Group candidates by question_id
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in eval_rows:
        qid = str(r["question_id"])
        by_q.setdefault(qid, []).append(r)

    per_q_sets: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "source_eval_scored": str(return_args.eval_scored),
        "source_qalphas": str(return_args.qalphas),
        "use_good": return_args.use_good,
        "alphas": {},
        "n_questions": len(by_q),
    }

    # Compute sets and metrics for each alpha
    for alpha, q_alpha in sorted(alphas.items()):
        covered = 0
        total_size = 0

        # Build per-question set membership
        # We also store minimal info so you can inspect later.
        for qid, cands in by_q.items():
            selected = [c for c in cands if float(c.get("score", 1.0)) <= q_alpha]
            total_size += len(selected)

            # Coverage = selected contains at least one good trace
            if return_args.use_good == "correct":
                is_cov = any(c.get("correct") is True for c in selected)
            else:
                is_cov = any(c.get("acceptable") is True for c in selected)

            if is_cov:
                covered += 1

            per_q_sets.append({
                "question_id": qid,
                "alpha": alpha,
                "q_alpha": q_alpha,
                "set_size": len(selected),
                "covered": bool(is_cov),
                "selected": [
                    {
                        "trace_id": c.get("trace_id"),
                        "answer": c.get("answer"),
                        "q": c.get("q"),
                        "score": c.get("score"),
                        "correct": c.get("correct"),
                        "acceptable": c.get("acceptable"),
                    }
                    for c in selected
                ],
            })

        coverage = covered / max(1, len(by_q))
        avg_set_size = total_size / max(1, len(by_q))

        metrics["alphas"][str(alpha)] = {
            "q_alpha": q_alpha,
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "n_questions": len(by_q),
        }

    # Write outputs
    write_jsonl(return_args.sets_out, per_q_sets)
    return_args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    return_args.metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote sets to {return_args.sets_out}")
    print(f"Wrote metrics to {return_args.metrics_out}")
    print("Metrics summary:")
    for a_str, m in metrics["alphas"].items():
        print(f"  alpha={a_str}: coverage={m['coverage']:.3f}, avg_set_size={m['avg_set_size']:.2f}, q_alpha={m['q_alpha']}")


if __name__ == "__main__":
    main()
