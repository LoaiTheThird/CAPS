# legal/report_compare.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare CP metrics vs Top-k baseline for LEXam.")
    p.add_argument("--cp_metrics", type=Path, required=True, help="outputs/lexam_eval_metrics.json")
    p.add_argument("--topk", type=Path, required=True, help="outputs/lexam_topk4_baseline.json")
    p.add_argument("--alpha", type=float, default=0.1, help="Which CP alpha to report (must exist in cp_metrics).")
    p.add_argument("--out", type=Path, default=Path("outputs/lexam_compare.json"))
    args = p.parse_args()

    cp = load_json(args.cp_metrics)
    topk = load_json(args.topk)

    akey = str(args.alpha)
    if akey not in cp.get("alphas", {}):
        raise SystemExit(f"alpha={args.alpha} not found in {args.cp_metrics}. Available: {list(cp.get('alphas', {}).keys())}")

    cp_m = cp["alphas"][akey]
    topk_m = topk

    # Basic comparison + derived efficiency
    cp_cov = float(cp_m["coverage"])
    cp_sz = float(cp_m["avg_set_size"])
    top_cov = float(topk_m["coverage"])
    top_sz = float(topk_m["avg_set_size"])

    shrink = (1.0 - (cp_sz / top_sz)) if top_sz > 0 else 0.0

    report = {
        "n_questions": int(cp.get("n_questions", topk_m.get("n_questions", 0))),
        "cp": {
            "alpha": args.alpha,
            "q_alpha": float(cp_m["q_alpha"]),
            "coverage": cp_cov,
            "avg_set_size": cp_sz,
        },
        "topk": {
            "k": int(topk_m["k"]),
            "coverage": top_cov,
            "avg_set_size": top_sz,
        },
        "derived": {
            "set_size_reduction_fraction": shrink,  # e.g., 0.49 means 49% smaller sets
            "same_coverage": abs(cp_cov - top_cov) < 1e-9,
        }
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Print a paper-friendly block
    print("LEXam comparison (eval)")
    print(f"Questions: {report['n_questions']}")
    print("")
    print("Method               Coverage    Avg |S|")
    print("----------------------------------------")
    print(f"CP (alpha={args.alpha})         {fmt(cp_cov)}      {cp_sz:.2f}")
    print(f"Top-k (k={topk_m['k']})          {fmt(top_cov)}      {top_sz:.2f}")
    print("----------------------------------------")
    print(f"Set size reduction: {fmt(100*shrink, 1)}% (CP vs Top-k)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
