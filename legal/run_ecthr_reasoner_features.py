from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

try:
    from .ecthr_features import label_features_from_reasoner_record, read_jsonl, write_jsonl
    from .gen_common import (
        DEFAULT_MAX_CASE_CHARS,
        VLLM_MODEL,
        build_case_text,
        call_vllm_chat_legalreasoner,
        get_candidate_schema,
        get_legalreasoner_reasoning_schema,
        get_legalreasoner_verification_schema,
        labels_from_verified_supports,
        load_split,
    )
    from .legalreasoner_prompts import (
        build_candidate_prompt,
        build_reasoning_prompt,
        build_verifier_prompt,
        sanitize_candidates,
        sanitize_reasoning_articles,
        sanitize_verification_articles,
    )
except ImportError:  # pragma: no cover
    from ecthr_features import label_features_from_reasoner_record, read_jsonl, write_jsonl
    from gen_common import (
        DEFAULT_MAX_CASE_CHARS,
        VLLM_MODEL,
        build_case_text,
        call_vllm_chat_legalreasoner,
        get_candidate_schema,
        get_legalreasoner_reasoning_schema,
        get_legalreasoner_verification_schema,
        labels_from_verified_supports,
        load_split,
    )
    from legalreasoner_prompts import (
        build_candidate_prompt,
        build_reasoning_prompt,
        build_verifier_prompt,
        sanitize_candidates,
        sanitize_reasoning_articles,
        sanitize_verification_articles,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LegalReasoner-lite stages and emit per-label verifier features.")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--n_examples", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=Path("outputs/ecthr_b"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--ids",
        default=None,
        help="Comma-separated case ids to run, e.g. 4,17,25. Overrides --offset/--n_examples selection.",
    )
    p.add_argument(
        "--ids_file",
        type=Path,
        default=None,
        help="Text file with one case id per line, or comma-separated ids.",
    )
    p.add_argument(
        "--failed_from",
        type=Path,
        default=None,
        help=(
            "Existing reasoner JSONL. Rerun rows with errors or incomplete "
            "candidate/reasoning/verification stages."
        ),
    )
    p.add_argument(
        "--candidate_source",
        choices=["base", "llm"],
        default="base",
        help="Use top-K base classifier labels by default; use llm for the older LLM candidate stage.",
    )
    p.add_argument(
        "--base_scores",
        type=Path,
        default=None,
        help="JSONL base score file. Defaults to outputs/ecthr_b/base_scores_<split>.jsonl.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of base classifier labels to send to the reasoner. Defaults to --max_candidates.",
    )
    p.add_argument("--max_candidates", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=4)
    p.add_argument("--min_verified_support_steps", type=int, default=2)
    return p.parse_args()


def parse_case_ids(value: str | None) -> List[int]:
    if not value:
        return []
    ids: List[int] = []
    for part in value.replace("\n", ",").split(","):
        part = part.strip()
        if not part or part.startswith("#"):
            continue
        ids.append(int(part))
    return ids


def read_case_ids_file(path: Path | None) -> List[int]:
    if path is None:
        return []
    return parse_case_ids(path.read_text(encoding="utf-8"))


def record_needs_rerun(row: Dict[str, Any]) -> bool:
    if row.get("error"):
        return True

    candidate_articles = row.get("candidate_articles") or []
    reasoning_articles = row.get("reasoning_articles") or []
    verification_articles = row.get("verification_articles") or []

    if len(reasoning_articles) < len(candidate_articles):
        return True
    if len(verification_articles) < len(reasoning_articles):
        return True

    return False


def failed_case_ids(path: Path | None) -> List[int]:
    if path is None:
        return []
    rows = read_jsonl(path)
    return [int(row["id"]) for row in rows if record_needs_rerun(row)]


def unique_sorted_ids(ids: Sequence[int]) -> List[int]:
    return sorted(dict.fromkeys(int(case_id) for case_id in ids))


def load_base_scores_by_id(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {int(row["id"]): row for row in rows}


def candidate_articles_from_base_scores(
    *,
    base_row: Dict[str, Any],
    label_names: List[str],
    top_k: int,
) -> List[Dict[str, str]]:
    scores = base_row.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"Base score row {base_row.get('id')} is missing a scores object.")

    ranked = sorted(
        label_names,
        key=lambda label: (-float(scores.get(label, 0.0)), label),
    )
    candidates = []
    for rank, label in enumerate(ranked[: max(0, min(top_k, len(ranked)))]):
        prob = float(scores.get(label, 0.0))
        candidates.append(
            {
                "label": label,
                "reason": f"Base classifier rank {rank + 1} with probability {prob:.4f}.",
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    candidate_limit = (
        args.top_k
        if args.candidate_source == "base" and args.top_k is not None
        else args.max_candidates
    )
    if candidate_limit <= 0:
        raise ValueError("Candidate limit must be positive.")

    selected_ids = unique_sorted_ids(
        [
            *parse_case_ids(args.ids),
            *read_case_ids_file(args.ids_file),
            *failed_case_ids(args.failed_from),
        ]
    )
    subset_mode = bool(args.ids or args.ids_file or args.failed_from)

    if selected_ids:
        load_n = max(selected_ids) + 1
    elif subset_mode:
        load_n = 0
    else:
        load_n = None if args.n_examples is None else args.offset + args.n_examples

    ds, label_names = load_split(args.split, load_n)
    if selected_ids:
        examples: List[Tuple[int, Dict[str, Any]]] = [
            (case_id, ds[int(case_id)])
            for case_id in selected_ids
        ]
    elif subset_mode:
        examples = []
    else:
        raw_examples = list(ds)[args.offset:]
        if args.n_examples is not None:
            raw_examples = raw_examples[: args.n_examples]
        examples = [
            (args.offset + local_idx, ex)
            for local_idx, ex in enumerate(raw_examples)
        ]

    out = args.out or (
        args.out_dir / f"reasoner_features_{args.split}_rerun.jsonl"
        if subset_mode
        else args.out_dir / f"reasoner_features_{args.split}.jsonl"
    )
    rows: List[Dict[str, Any]] = []

    base_scores_path = args.base_scores or (args.out_dir / f"base_scores_{args.split}.jsonl")
    base_scores_by_id: Dict[int, Dict[str, Any]] = {}
    if args.candidate_source == "base":
        if not base_scores_path.exists():
            raise FileNotFoundError(
                f"Base score file not found: {base_scores_path}. "
                "Run legal.run_ecthr_base_scores first or pass --base_scores."
            )
        base_scores_by_id = load_base_scores_by_id(base_scores_path)
        requested_ids = [case_id for case_id, _ex in examples]
        missing_ids = [case_id for case_id in requested_ids if case_id not in base_scores_by_id]
        if missing_ids:
            preview = ", ".join(str(case_id) for case_id in missing_ids[:10])
            raise ValueError(
                f"Base score file {base_scores_path} is missing {len(missing_ids)} "
                f"requested case ids. First missing ids: {preview}"
            )

    candidate_schema = get_candidate_schema(label_names, max_candidates=candidate_limit)
    reasoning_schema = get_legalreasoner_reasoning_schema(
        label_names,
        max_candidates=candidate_limit,
        max_steps=args.max_steps,
    )
    verification_schema = get_legalreasoner_verification_schema(
        label_names,
        max_candidates=candidate_limit,
        max_steps=args.max_steps,
    )

    if subset_mode:
        print(f"Selected {len(examples)} case ids for rerun/subset output: {out}")

    for idx, ex in tqdm(examples, desc=f"Reasoner features ({args.split})"):
        case_text = build_case_text(ex["text"], max_chars=DEFAULT_MAX_CASE_CHARS)
        gold_ids = [int(i) for i in ex["labels"]]
        gold_labels = [label_names[i] for i in gold_ids]

        candidate_articles: List[Dict[str, Any]] = []
        reasoning_articles: List[Dict[str, Any]] = []
        verification_articles: List[Dict[str, Any]] = []
        support_only_labels: List[str] = []
        stage_error = None

        try:
            if args.candidate_source == "base":
                base_row = base_scores_by_id.get(idx)
                if base_row is None:
                    raise KeyError(f"No base score row found for case id {idx}.")
                candidate_articles = candidate_articles_from_base_scores(
                    base_row=base_row,
                    label_names=label_names,
                    top_k=candidate_limit,
                )
            else:
                candidate_response = call_vllm_chat_legalreasoner(
                    build_candidate_prompt(case_text, label_names, max_candidates=candidate_limit),
                    candidate_schema,
                )
                candidate_articles = sanitize_candidates(
                    candidate_response.get("candidates", []),
                    label_names,
                    max_candidates=candidate_limit,
                )
            candidate_labels = [item["label"] for item in candidate_articles]

            if candidate_labels:
                reasoning_response = call_vllm_chat_legalreasoner(
                    build_reasoning_prompt(
                        case_text,
                        candidate_articles,
                        label_names,
                        max_candidates=candidate_limit,
                        max_steps=args.max_steps,
                    ),
                    reasoning_schema,
                )
                reasoning_articles = sanitize_reasoning_articles(
                    reasoning_response.get("articles", []),
                    candidate_labels,
                    max_steps=args.max_steps,
                )

            reasoning_labels = [item["label"] for item in reasoning_articles]
            if reasoning_articles and reasoning_labels:
                verification_response = call_vllm_chat_legalreasoner(
                    build_verifier_prompt(
                        case_text,
                        reasoning_articles,
                        label_names,
                        max_candidates=candidate_limit,
                        max_steps=args.max_steps,
                    ),
                    verification_schema,
                )
                verification_articles = sanitize_verification_articles(
                    verification_response.get("articles", []),
                    reasoning_labels,
                    max_steps=args.max_steps,
                )
                support_only_labels = labels_from_verified_supports(
                    verification_articles,
                    label_names,
                    reasoning_articles=reasoning_articles,
                    min_supported_steps=args.min_verified_support_steps,
                    require_label_verdict=True,
                )

        except Exception as exc:
            stage_error = str(exc)

        per_label_features = label_features_from_reasoner_record(
            label_names=label_names,
            candidate_articles=candidate_articles,
            reasoning_articles=reasoning_articles,
            verification_articles=verification_articles,
            support_only_labels=support_only_labels,
        )

        rows.append(
            {
                "id": idx,
                "split": args.split,
                "gold_label_ids": gold_ids,
                "gold_labels": gold_labels,
                "candidate_articles": candidate_articles,
                "reasoning_articles": reasoning_articles,
                "verification_articles": verification_articles,
                "support_only_labels": support_only_labels,
                "per_label_features": per_label_features,
                "error": stage_error,
                "case_chars": len(case_text),
                "provider": "vllm",
                "model": VLLM_MODEL,
                "candidate_source": args.candidate_source,
                "base_scores_path": str(base_scores_path) if args.candidate_source == "base" else None,
                "top_k": candidate_limit if args.candidate_source == "base" else None,
                "max_candidates": candidate_limit,
                "max_steps": args.max_steps,
                "min_verified_support_steps": args.min_verified_support_steps,
            }
        )

    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows to {out}")
    n_errors = sum(1 for row in rows if row.get("error"))
    n_reasoning = sum(1 for row in rows if row.get("reasoning_articles"))
    n_verification = sum(1 for row in rows if row.get("verification_articles"))
    print(
        "Reasoner summary: "
        f"errors={n_errors}/{len(rows)}, "
        f"reasoning_nonempty={n_reasoning}/{len(rows)}, "
        f"verification_nonempty={n_verification}/{len(rows)}"
    )
    if rows and n_errors == len(rows):
        raise SystemExit(
            "All examples failed during LegalReasoner feature generation. "
            "Check that the vLLM/OpenAI-compatible server is running and VLLM_BASE_URL is correct."
        )


if __name__ == "__main__":
    main()
