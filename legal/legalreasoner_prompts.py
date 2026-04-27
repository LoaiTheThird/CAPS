from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    from .gen_common import (
        build_label_guardrails,
        build_label_legend,
        get_candidate_schema,
        get_label_schema,
        get_legalreasoner_reasoning_schema,
        get_legalreasoner_verification_schema,
        label_for_prompt,
    )
except ImportError:  # pragma: no cover - allows running as python legal/foo.py
    from gen_common import (
        build_label_guardrails,
        build_label_legend,
        get_candidate_schema,
        get_label_schema,
        get_legalreasoner_reasoning_schema,
        get_legalreasoner_verification_schema,
        label_for_prompt,
    )


def build_candidate_prompt(
    case_text: str,
    label_names: List[str],
    *,
    max_candidates: int = 4,
) -> str:
    label_legend = build_label_legend(label_names)
    label_guardrails = build_label_guardrails(label_names)
    schema_str = json.dumps(
        get_candidate_schema(label_names, max_candidates=max_candidates),
        ensure_ascii=False,
    )

    return f"""
You are Stage 1 of a LegalReasoner-inspired ECtHR classification pipeline.

Possible labels:
{label_legend}

Select up to {max_candidates} labels that seem most relevant to the case facts.
For each selected label, give one short reason why it may be implicated.
Rank labels from strongest to weakest support.

Article guardrails:
{label_guardrails}

Rules:
- Use only the provided case facts.
- Choose only genuinely plausible labels whose protected right is directly implicated.
- Prefer no candidate over a label supported only by generic litigation, arrest, or background facts.
- Do not invent labels outside the list.
- Keep each reason to one short sentence.
- Return JSON only.
- The output must follow this JSON schema exactly:
{schema_str}

Case facts:
{case_text}
""".strip()


def build_reasoning_prompt(
    case_text: str,
    candidate_articles: List[Dict[str, str]],
    label_names: List[str],
    *,
    max_candidates: int = 4,
    max_steps: int = 4,
) -> str:
    schema_str = json.dumps(
        get_legalreasoner_reasoning_schema(
            label_names,
            max_candidates=max_candidates,
            max_steps=max_steps,
        ),
        ensure_ascii=False,
    )
    candidate_view = [
        {
            "label": item["label"],
            "label_description": label_for_prompt(item["label"]),
            "reason": item["reason"],
        }
        for item in candidate_articles
    ]
    candidate_json = json.dumps(candidate_view, ensure_ascii=False, indent=2)
    label_guardrails = build_label_guardrails([item["label"] for item in candidate_articles])

    return f"""
You are Stage 2 of a LegalReasoner-inspired ECtHR classification pipeline.

For each candidate label below, write 2 to {max_steps} short reasoning steps grounded in the case facts.
Then assign a preliminary_decision of:
- support
- weak
- none

Article guardrails for these candidates:
{label_guardrails}

Rules:
- Analyze only the candidate labels provided below.
- Keep each reasoning step short and fact-grounded.
- If support is weak or absent, the steps should make that clear.
- Use preliminary_decision "support" only when the facts directly implicate the label description.
- Use "weak" or "none" when facts are true but only loosely related to the label.
- Return JSON only.
- The output must follow this JSON schema exactly:
{schema_str}

Candidate labels:
{candidate_json}

Case facts:
{case_text}
""".strip()


def build_verifier_prompt(
    case_text: str,
    reasoning_articles: List[Dict[str, Any]],
    label_names: List[str],
    *,
    max_candidates: int = 4,
    max_steps: int = 4,
) -> str:
    schema_str = json.dumps(
        get_legalreasoner_verification_schema(
            label_names,
            max_candidates=max_candidates,
            max_steps=max_steps,
        ),
        ensure_ascii=False,
    )
    reasoning_view = [
        {
            "label": item["label"],
            "label_description": label_for_prompt(item["label"]),
            "preliminary_decision": item["preliminary_decision"],
            "reasoning_steps": item["reasoning_steps"],
        }
        for item in reasoning_articles
    ]
    reasoning_json = json.dumps(reasoning_view, ensure_ascii=False, indent=2)
    label_guardrails = build_label_guardrails([item["label"] for item in reasoning_articles])

    return f"""
You are Stage 3, the verifier, in a LegalReasoner-inspired ECtHR classification pipeline.

Check whether each reasoning step is supported by the case facts.
For each step, set verdict to exactly one of:
- supported
- partial
- unsupported

Also set label_verdict for the whole candidate:
- supported: the supported reasoning steps directly justify this article/protocol label.
- partial: some steps are true but the article connection is incomplete or speculative.
- unsupported: facts may be true, but they do not justify this label.

Article guardrails:
{label_guardrails}

Guidance:
- Be skeptical: do not mark a label supported just because individual facts are true.
- A supported step must connect case facts to the protected right, not just mention courts, detention, or general unfairness.
- Keep label_evidence to one short phrase; use an empty string for unsupported labels.

Rules:
- Preserve the original labels and step text.
- Judge support only from the provided case facts.
- Use a short evidence string for supported or partial steps.
- Use an empty evidence string for unsupported steps when needed.
- Return JSON only.
- The output must follow this JSON schema exactly:
{schema_str}

Reasoning draft:
{reasoning_json}

Case facts:
{case_text}
""".strip()


def build_final_prompt(
    verification_articles: List[Dict[str, Any]],
    label_names: List[str],
    support_only_labels: List[str],
    *,
    max_final_labels: int = 3,
) -> str:
    label_legend = build_label_legend(label_names)
    allowed_str = ", ".join(support_only_labels) if support_only_labels else "(none)"
    schema_str = json.dumps(get_label_schema(label_names), ensure_ascii=False)
    verification_json = json.dumps(verification_articles, ensure_ascii=False, indent=2)

    return f"""
You are Stage 4, the final decision stage, in a LegalReasoner-inspired ECtHR classification pipeline.

Possible labels:
{label_legend}

Labels that passed the reasoning and verifier gates: {allowed_str}

Predict final labels using only candidates that passed both the reasoning and verifier gates.

Rules:
- Use only the verification report below.
- Ignore steps marked partial or unsupported.
- Do not output any label outside the gated label set listed above.
- Output at most {max_final_labels} labels, ranked from strongest to weakest support.
- Prefer fewer labels when the verified evidence is sparse.
- Do not include a label merely because its facts are true; the facts must justify the protected right.
- Return JSON only.
- The output must follow this JSON schema exactly:
{schema_str}

Verification report:
{verification_json}
""".strip()


def sanitize_candidates(
    candidates: List[Dict[str, Any]],
    label_names: List[str],
    *,
    max_candidates: int = 4,
) -> List[Dict[str, str]]:
    allowed = set(label_names)
    cleaned = []
    seen = set()

    for item in candidates:
        label = item.get("label")
        reason = str(item.get("reason", "")).strip()
        if label not in allowed or label in seen:
            continue
        cleaned.append({"label": label, "reason": reason})
        seen.add(label)
        if len(cleaned) >= max_candidates:
            break

    return cleaned


def sanitize_reasoning_articles(
    articles: List[Dict[str, Any]],
    allowed_labels: List[str],
    *,
    max_steps: int = 4,
) -> List[Dict[str, Any]]:
    allowed = set(allowed_labels)
    cleaned = []
    seen = set()

    for item in articles:
        label = item.get("label")
        if label not in allowed or label in seen:
            continue

        preliminary_decision = item.get("preliminary_decision", "weak")
        if preliminary_decision not in {"support", "weak", "none"}:
            preliminary_decision = "weak"

        reasoning_steps = []
        for step in item.get("reasoning_steps", []):
            step_text = str(step).strip()
            if step_text:
                reasoning_steps.append(step_text)
            if len(reasoning_steps) >= max_steps:
                break

        cleaned.append(
            {
                "label": label,
                "preliminary_decision": preliminary_decision,
                "reasoning_steps": reasoning_steps,
            }
        )
        seen.add(label)

    return cleaned


def sanitize_verification_articles(
    articles: List[Dict[str, Any]],
    allowed_labels: List[str],
    *,
    max_steps: int = 4,
) -> List[Dict[str, Any]]:
    allowed = set(allowed_labels)
    cleaned = []
    seen = set()

    for item in articles:
        label = item.get("label")
        if label not in allowed or label in seen:
            continue

        label_verdict = item.get("label_verdict", "unsupported")
        if label_verdict not in {"supported", "partial", "unsupported"}:
            label_verdict = "unsupported"
        label_evidence = str(item.get("label_evidence", "")).strip()

        step_checks = []
        for check in item.get("step_checks", []):
            step = str(check.get("step", "")).strip()
            verdict = check.get("verdict")
            evidence = str(check.get("evidence", "")).strip()

            if not step or verdict not in {"supported", "partial", "unsupported"}:
                continue

            step_checks.append(
                {
                    "step": step,
                    "verdict": verdict,
                    "evidence": evidence,
                }
            )
            if len(step_checks) >= max_steps:
                break

        cleaned.append(
            {
                "label": label,
                "label_verdict": label_verdict,
                "label_evidence": label_evidence,
                "step_checks": step_checks,
            }
        )
        seen.add(label)

    return cleaned
