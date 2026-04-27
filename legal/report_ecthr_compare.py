from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print a compact paper-style ECtHR conformal comparison table.")
    p.add_argument("--metrics", type=Path, default=Path("outputs/ecthr_b/conformal_metrics.json"))
    p.add_argument("--alpha", default="0.1")
    p.add_argument("--mode", choices=["global", "labelwise"], default="global")
    p.add_argument("--out", type=Path, default=Path("outputs/ecthr_b/compare_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    obj: Dict[str, Any] = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = []

    for method, method_report in obj["methods"].items():
        metrics = method_report[args.mode][args.alpha]
        rows.append(
            {
                "method": method,
                "coverage": float(metrics["coverage"]),
                "avg_set_size": float(metrics["avg_set_size"]),
                "median_set_size": float(metrics["median_set_size"]),
                "micro_f1": float(metrics["micro_f1"]),
                "macro_f1": float(metrics["macro_f1"]),
                "micro_recall": float(metrics["micro_recall"]),
            }
        )

    rows.sort(key=lambda r: (-r["coverage"], r["avg_set_size"]))
    summary = {
        "source": str(args.metrics),
        "alpha": args.alpha,
        "mode": args.mode,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"ECtHR-B conformal comparison ({args.mode}, alpha={args.alpha})")
    print("Method                  Coverage   Avg |C|   Med |C|   Micro-F1   Macro-F1   Micro-R")
    print("--------------------------------------------------------------------------------")
    for row in rows:
        print(
            f"{row['method']:<22s}"
            f"{fmt(row['coverage']):>9s}"
            f"{row['avg_set_size']:>10.2f}"
            f"{row['median_set_size']:>10.2f}"
            f"{fmt(row['micro_f1']):>11s}"
            f"{fmt(row['macro_f1']):>11s}"
            f"{fmt(row['micro_recall']):>10s}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
