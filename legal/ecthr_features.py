from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_features_from_reasoner_record(
    *,
    label_names: List[str],
    candidate_articles: List[Dict[str, Any]],
    reasoning_articles: List[Dict[str, Any]],
    verification_articles: List[Dict[str, Any]],
    support_only_labels: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Flatten a LegalReasoner-lite case record into numeric per-label features."""
    support_set = set(support_only_labels or [])
    out: Dict[str, Dict[str, float]] = {}

    candidate_rank = {
        str(item.get("label")): i for i, item in enumerate(candidate_articles)
    }
    candidate_reason_chars = {
        str(item.get("label")): len(str(item.get("reason", "")))
        for item in candidate_articles
    }
    reasoning_by_label = {
        str(item.get("label")): item for item in reasoning_articles
    }
    verification_by_label = {
        str(item.get("label")): item for item in verification_articles
    }

    max_rank = max(1, len(candidate_articles) - 1)

    for label in label_names:
        reasoning = reasoning_by_label.get(label, {})
        verification = verification_by_label.get(label, {})
        steps = reasoning.get("reasoning_steps", []) or []
        checks = verification.get("step_checks", []) or []

        n_supported = sum(check.get("verdict") == "supported" for check in checks)
        n_partial = sum(check.get("verdict") == "partial" for check in checks)
        n_unsupported = sum(check.get("verdict") == "unsupported" for check in checks)
        n_checks = len(checks)

        prelim = reasoning.get("preliminary_decision")
        verdict = verification.get("label_verdict")
        rank = candidate_rank.get(label, -1)

        out[label] = {
            "candidate_present": float(rank >= 0),
            "candidate_rank": float(rank),
            "candidate_rank_norm": float(rank / max_rank) if rank >= 0 else 1.0,
            "candidate_reason_chars": float(candidate_reason_chars.get(label, 0)),
            "prelim_support": float(prelim == "support"),
            "prelim_weak": float(prelim == "weak"),
            "prelim_none": float(prelim == "none"),
            "n_reasoning_steps": float(len(steps)),
            "verdict_supported": float(verdict == "supported"),
            "verdict_partial": float(verdict == "partial"),
            "verdict_unsupported": float(verdict == "unsupported"),
            "label_evidence_chars": float(len(str(verification.get("label_evidence", "")))),
            "n_step_checks": float(n_checks),
            "n_supported_steps": float(n_supported),
            "n_partial_steps": float(n_partial),
            "n_unsupported_steps": float(n_unsupported),
            "supported_fraction": float(n_supported / n_checks) if n_checks else 0.0,
            "partial_fraction": float(n_partial / n_checks) if n_checks else 0.0,
            "unsupported_fraction": float(n_unsupported / n_checks) if n_checks else 0.0,
            "hard_support_gate": float(label in support_set),
        }

    return out


def default_reasoner_features(label_names: List[str]) -> Dict[str, Dict[str, float]]:
    return label_features_from_reasoner_record(
        label_names=label_names,
        candidate_articles=[],
        reasoning_articles=[],
        verification_articles=[],
        support_only_labels=[],
    )


def multihot_label_names(label_names: List[str], label_ids: List[int]) -> List[str]:
    return [label_names[i] for i in label_ids]
