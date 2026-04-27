import json
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from gen_common import (
    DEFAULT_MAX_CASE_CHARS,
    DEFAULT_N_EXAMPLES,
    VLLM_MODEL,
    build_case_text,
    build_label_guardrails,
    build_label_legend,
    call_vllm_chat_direct,
    call_vllm_chat_legalreasoner,
    compute_metrics,
    get_candidate_schema,
    get_label_schema,
    get_legalreasoner_reasoning_schema,
    get_legalreasoner_verification_schema,
    label_for_prompt,
    labels_from_verified_supports,
    load_split,
    sanitize_predicted_labels,
    to_multihot_from_ids,
    to_multihot_from_names,
)

N_EXAMPLES = DEFAULT_N_EXAMPLES
MAX_CANDIDATES = max(1, int(os.environ.get("LR_MAX_CANDIDATES", "4")))
MAX_REASONING_STEPS = max(2, int(os.environ.get("LR_MAX_STEPS", "4")))
MAX_FINAL_LABELS = max(1, int(os.environ.get("LR_MAX_FINAL_LABELS", "3")))
MIN_VERIFIED_SUPPORT_STEPS = max(1, int(os.environ.get("LR_MIN_VERIFIED_SUPPORT_STEPS", "2")))
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS_PATH = RESULTS_DIR / "legalreasoner_lite_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "legalreasoner_lite_predictions.json"

test_ds, label_names = load_split(N_EXAMPLES)


def build_candidate_prompt(case_text: str, label_names: List[str]) -> str:
    label_legend = build_label_legend(label_names)
    label_guardrails = build_label_guardrails(label_names)
    schema_str = json.dumps(get_candidate_schema(label_names, max_candidates=MAX_CANDIDATES), ensure_ascii=False)

    return f"""
You are Stage 1 of a LegalReasoner-inspired ECtHR classification pipeline.

Possible labels:
{label_legend}

Select up to {MAX_CANDIDATES} labels that seem most relevant to the case facts.
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
) -> str:
    schema_str = json.dumps(
        get_legalreasoner_reasoning_schema(
            label_names,
            max_candidates=MAX_CANDIDATES,
            max_steps=MAX_REASONING_STEPS,
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

For each candidate label below, write 2 to {MAX_REASONING_STEPS} short reasoning steps grounded in the case facts.
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


def build_verifier_prompt(case_text: str, reasoning_articles: List[Dict[str, Any]], label_names: List[str]) -> str:
    schema_str = json.dumps(
        get_legalreasoner_verification_schema(
            label_names,
            max_candidates=MAX_CANDIDATES,
            max_steps=MAX_REASONING_STEPS,
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


def build_final_prompt(verification_articles: List[Dict[str, Any]], label_names: List[str], support_only_labels: List[str]) -> str:
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
- Output at most {MAX_FINAL_LABELS} labels, ranked from strongest to weakest support.
- Prefer fewer labels when the verified evidence is sparse.
- Do not include a label merely because its facts are true; the facts must justify the protected right.
- Return JSON only.
- The output must follow this JSON schema exactly:
{schema_str}

Verification report:
{verification_json}
""".strip()


def sanitize_candidates(candidates: List[Dict[str, Any]], label_names: List[str]) -> List[Dict[str, str]]:
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
        if len(cleaned) >= MAX_CANDIDATES:
            break

    return cleaned


def sanitize_reasoning_articles(articles: List[Dict[str, Any]], allowed_labels: List[str]) -> List[Dict[str, Any]]:
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
            if len(reasoning_steps) >= MAX_REASONING_STEPS:
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


def sanitize_verification_articles(articles: List[Dict[str, Any]], allowed_labels: List[str]) -> List[Dict[str, Any]]:
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
            if len(step_checks) >= MAX_REASONING_STEPS:
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


gold = []
pred = []
records = []

candidate_schema = get_candidate_schema(label_names, max_candidates=MAX_CANDIDATES)
final_schema = get_label_schema(label_names)

for idx, ex in enumerate(tqdm(test_ds, desc="LegalReasoner-lite baseline")):
    case_text = build_case_text(ex["text"], max_chars=DEFAULT_MAX_CASE_CHARS)
    candidate_articles = []
    reasoning_articles = []
    verification_articles = []
    support_only_labels = []
    model_labels = []
    predicted_labels = []
    stage_error = None

    try:
        candidate_response = call_vllm_chat_legalreasoner(
            build_candidate_prompt(case_text, label_names),
            candidate_schema,
        )
        candidate_articles = sanitize_candidates(candidate_response.get("candidates", []), label_names)
        candidate_labels = [item["label"] for item in candidate_articles]

        if candidate_labels:
            reasoning_response = call_vllm_chat_legalreasoner(
                build_reasoning_prompt(case_text, candidate_articles, label_names),
                get_legalreasoner_reasoning_schema(
                    label_names,
                    max_candidates=MAX_CANDIDATES,
                    max_steps=MAX_REASONING_STEPS,
                ),
            )
            reasoning_articles = sanitize_reasoning_articles(reasoning_response.get("articles", []), candidate_labels)

        reasoning_labels = [item["label"] for item in reasoning_articles]
        if reasoning_articles and reasoning_labels:
            verification_response = call_vllm_chat_legalreasoner(
                build_verifier_prompt(case_text, reasoning_articles, label_names),
                get_legalreasoner_verification_schema(
                    label_names,
                    max_candidates=MAX_CANDIDATES,
                    max_steps=MAX_REASONING_STEPS,
                ),
            )
            verification_articles = sanitize_verification_articles(
                verification_response.get("articles", []),
                reasoning_labels,
            )
            support_only_labels = labels_from_verified_supports(
                verification_articles,
                label_names,
                reasoning_articles=reasoning_articles,
                min_supported_steps=MIN_VERIFIED_SUPPORT_STEPS,
                require_label_verdict=True,
            )

        if support_only_labels:
            final_response = call_vllm_chat_direct(
                build_final_prompt(verification_articles, label_names, support_only_labels),
                final_schema,
            )
            model_labels = sanitize_predicted_labels(
                label_names,
                final_response.get("labels", []),
                max_labels=MAX_FINAL_LABELS,
                allowed_labels=support_only_labels,
            )
            predicted_labels = model_labels or sanitize_predicted_labels(
                label_names,
                support_only_labels,
                max_labels=MAX_FINAL_LABELS,
            )
        else:
            predicted_labels = []

    except Exception as e:
        stage_error = str(e)
        print(f"[WARN] example {idx} failed: {e}")
        # Fall back to the gated labels if earlier stages succeeded.
        predicted_labels = sanitize_predicted_labels(
            label_names,
            support_only_labels,
            max_labels=MAX_FINAL_LABELS,
        )

    gold_vec = to_multihot_from_ids(ex["labels"], len(label_names))
    pred_vec = to_multihot_from_names(label_names, predicted_labels)

    gold.append(gold_vec)
    pred.append(pred_vec)

    records.append(
        {
            "id": idx,
            "gold_ids": ex["labels"],
            "gold_labels": [label_names[i] for i in ex["labels"]],
            "candidate_articles": candidate_articles,
            "reasoning_articles": reasoning_articles,
            "verification_articles": verification_articles,
            "support_only_labels": support_only_labels,
            "model_labels": model_labels,
            "predicted_labels": predicted_labels,
            "error": stage_error,
            "case_chars": len(case_text),
        }
    )

metrics = compute_metrics(gold, pred)
metrics["n_examples"] = N_EXAMPLES
metrics["provider"] = "vllm"
metrics["model"] = VLLM_MODEL
metrics["baseline"] = "legalreasoner_lite"
metrics["max_candidates"] = MAX_CANDIDATES
metrics["max_reasoning_steps"] = MAX_REASONING_STEPS
metrics["max_final_labels"] = MAX_FINAL_LABELS
metrics["min_verified_support_steps"] = MIN_VERIFIED_SUPPORT_STEPS
metrics["max_case_chars"] = DEFAULT_MAX_CASE_CHARS

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with METRICS_PATH.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with PREDICTIONS_PATH.open("w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print("METRICS:", metrics)
print(f"Saved to {METRICS_PATH}")
print(f"Saved to {PREDICTIONS_PATH}")
