# conformal/legal_split.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def quantile_conformal(values: List[float], alpha: float) -> float:
    """
    Split-conformal quantile for scores.
    Standard choice: q = k-th order statistic where k = ceil((n+1)*(1-alpha)).
    Uses 1-indexing in definition.
    """
    if not values:
        raise ValueError("No values provided to quantile_conformal.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    xs = sorted(values)
    n = len(xs)
    k = math.ceil((n + 1) * (1.0 - alpha))  # 1..n+1
    k = max(1, min(k, n))  # clamp to 1..n
    return float(xs[k - 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute legal CP thresholds from calib scored candidates.")
    p.add_argument("--calib_scored", type=Path, required=True,
                   help="JSONL of scored candidates on calib split (output of legal.score_legal).")
    p.add_argument("--out", type=Path, required=True, help="Where to write thresholds JSON.")
    p.add_argument("--alphas", type=str, default="0.1,0.2",
                   help="Comma-separated alphas, e.g. '0.1,0.2'.")
    p.add_argument(
        "--good_def",
        choices=["correct", "acceptable"],
        default="correct",
        help="What counts as a 'good' candidate when forming per-question min score.",
    )
    p.add_argument(
        "--min_good_per_q",
        type=int,
        default=1,
        help="Require at least this many good candidates per question; otherwise skip question.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    rows = read_jsonl(args.calib_scored)
    # group by question_id, collect scores for "good" candidates
    good_scores_by_q: Dict[str, List[float]] = {}

    for r in rows:
        qid = str(r.get("question_id"))
        score = r.get("score", None)
        if score is None:
            continue

        is_good = False
        if args.good_def == "correct":
            is_good = bool(r.get("correct") is True)
        else:  # acceptable
            is_good = bool(r.get("acceptable") is True)

        if not is_good:
            continue

        good_scores_by_q.setdefault(qid, []).append(float(score))

    # For each question, take min score among good candidates
    per_q_min_scores: List[float] = []
    skipped = 0
    for qid, scores in good_scores_by_q.items():
        if len(scores) < args.min_good_per_q:
            skipped += 1
            continue
        per_q_min_scores.append(min(scores))

    if not per_q_min_scores:
        raise SystemExit(
            f"No per-question min scores computed. "
            f"Maybe good_def={args.good_def} yielded no good candidates?"
        )

    thresholds: Dict[str, Any] = {
        "source": str(args.calib_scored),
        "good_def": args.good_def,
        "min_good_per_q": args.min_good_per_q,
        "n_questions_total_with_any_good": len(good_scores_by_q),
        "n_questions_used": len(per_q_min_scores),
        "n_questions_skipped_min_good": skipped,
        "alphas": {},
    }

    for a in alphas:
        q_a = quantile_conformal(per_q_min_scores, a)
        thresholds["alphas"][str(a)] = {
            "q_alpha": q_a,
            "n": len(per_q_min_scores),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(thresholds, indent=2))
    print(f"Wrote thresholds to {args.out}")
    print(f"Used {len(per_q_min_scores)} questions for calibration (good_def={args.good_def})")
    for a in alphas:
        print(f"  alpha={a}: q_alpha={thresholds['alphas'][str(a)]['q_alpha']}")


if __name__ == "__main__":
    main()
