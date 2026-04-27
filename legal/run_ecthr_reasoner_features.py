from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

try:
    from .ecthr_features import label_features_from_reasoner_record, write_jsonl
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
    from ecthr_features import label_features_from_reasoner_record, write_jsonl
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
    p.add_argument("--max_candidates", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=4)
    p.add_argument("--min_verified_support_steps", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_n = None if args.n_examples is None else args.offset + args.n_examples
    ds, label_names = load_split(args.split, load_n)
    examples = list(ds)[args.offset:]
    if args.n_examples is not None:
        examples = examples[: args.n_examples]

    out = args.out or (args.out_dir / f"reasoner_features_{args.split}.jsonl")
    rows: List[Dict[str, Any]] = []

    candidate_schema = get_candidate_schema(label_names, max_candidates=args.max_candidates)
    reasoning_schema = get_legalreasoner_reasoning_schema(
        label_names,
        max_candidates=args.max_candidates,
        max_steps=args.max_steps,
    )
    verification_schema = get_legalreasoner_verification_schema(
        label_names,
        max_candidates=args.max_candidates,
        max_steps=args.max_steps,
    )

    for local_idx, ex in enumerate(tqdm(examples, desc=f"Reasoner features ({args.split})")):
        idx = args.offset + local_idx
        case_text = build_case_text(ex["text"], max_chars=DEFAULT_MAX_CASE_CHARS)
        gold_ids = [int(i) for i in ex["labels"]]
        gold_labels = [label_names[i] for i in gold_ids]

        candidate_articles: List[Dict[str, Any]] = []
        reasoning_articles: List[Dict[str, Any]] = []
        verification_articles: List[Dict[str, Any]] = []
        support_only_labels: List[str] = []
        stage_error = None

        try:
            candidate_response = call_vllm_chat_legalreasoner(
                build_candidate_prompt(case_text, label_names, max_candidates=args.max_candidates),
                candidate_schema,
            )
            candidate_articles = sanitize_candidates(
                candidate_response.get("candidates", []),
                label_names,
                max_candidates=args.max_candidates,
            )
            candidate_labels = [item["label"] for item in candidate_articles]

            if candidate_labels:
                reasoning_response = call_vllm_chat_legalreasoner(
                    build_reasoning_prompt(
                        case_text,
                        candidate_articles,
                        label_names,
                        max_candidates=args.max_candidates,
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
                        max_candidates=args.max_candidates,
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
                "max_candidates": args.max_candidates,
                "max_steps": args.max_steps,
                "min_verified_support_steps": args.min_verified_support_steps,
            }
        )

    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
