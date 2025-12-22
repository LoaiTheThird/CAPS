# # conformal/split.py
#
# """
# Split conformal thresholds on calibration scores.
#
# Given a JSONL file of scored candidates (with a 'score' field),
# compute quantile thresholds q_alpha for alpha in {0.1, 0.2} using
# the standard split-conformal formula:
#
#   q_alpha = sorted_scores[ceil((n+1)*(1 - alpha)) - 1]
#
# and save them to a JSON file.
#
# This is done on the *calibration* split.
# """
#
# import argparse
# import json
# import math
# from pathlib import Path
# from typing import Any, Dict, Iterable, List
#
# DEFAULT_INPUT = Path("outputs/eb_task1_dev_calib_scored.jsonl")
# DEFAULT_OUTPUT = Path("outputs/eb_task1_dev_calib_quantiles.json")
#
#
# def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
#     with path.open() as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             yield json.loads(line)
#
#
# def load_scores(path: Path) -> List[float]:
#     scores: List[float] = []
#     for rec in read_jsonl(path):
#         if "score" not in rec:
#             continue
#         s = float(rec["score"])
#         scores.append(s)
#     return scores
#
#
# def compute_q_alpha(scores: List[float], alphas: List[float]) -> Dict[str, float]:
#     """
#     Compute split-conformal thresholds for each alpha.
#
#     scores: list of nonconformity scores s_1, ..., s_n
#     alphas: list of significance levels (e.g., [0.1, 0.2])
#
#     Returns dict mapping string(alpha) -> q_alpha.
#     """
#     if not scores:
#         raise ValueError("No scores provided to compute quantiles.")
#
#     n = len(scores)
#     sorted_scores = sorted(scores)
#
#     q: Dict[str, float] = {}
#     for alpha in alphas:
#         # Index per standard split-CP:
#         # k = ceil((n+1)*(1 - alpha)) - 1, clamped to [0, n-1]
#         k = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
#         k = max(0, min(k, n - 1))
#         q_alpha = float(sorted_scores[k])
#         q[str(alpha)] = q_alpha
#     return q
#
#
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Compute split-CP quantiles on calibration scores.")
#     parser.add_argument(
#         "--input",
#         type=Path,
#         default=DEFAULT_INPUT,
#         help=f"Scored calibration JSONL (default: {DEFAULT_INPUT})",
#     )
#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=DEFAULT_OUTPUT,
#         help=f"Output JSON file with quantiles (default: {DEFAULT_OUTPUT})",
#     )
#     parser.add_argument(
#         "--alphas",
#         type=float,
#         nargs="+",
#         default=[0.1, 0.2],
#         help="Significance levels α (default: 0.1 0.2)",
#     )
#     return parser.parse_args()
#
#
# def main() -> None:
#     args = parse_args()
#     if not args.input.exists():
#         raise SystemExit(f"Input file not found: {args.input}")
#
#     scores = load_scores(args.input)
#     n = len(scores)
#     if n == 0:
#         raise SystemExit(f"No scores found in {args.input}")
#
#     print(f"Loaded {n} scores from {args.input}")
#
#     q = compute_q_alpha(scores, args.alphas)
#
#     out = {
#         "n_scores": n,
#         "alphas": args.alphas,
#         "quantiles": q,  # keys are strings of alpha
#     }
#
#     args.output.parent.mkdir(parents=True, exist_ok=True)
#     with args.output.open("w") as f:
#         json.dump(out, f, indent=2)
#
#     print("Computed quantiles:")
#     for alpha in args.alphas:
#         print(f"  alpha={alpha}: q_alpha = {q[str(alpha)]}")
#     print(f"Saved to {args.output}")
#
#
# if __name__ == "__main__":
#     main()
