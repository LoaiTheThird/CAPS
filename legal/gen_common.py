import json
import os
from typing import Any, Dict, Iterable, List

import numpy as np
import requests
from sklearn.metrics import f1_score

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
DATASET_CONFIG = "ecthr_b"
DEFAULT_N_EXAMPLES = int(os.environ.get("N_EXAMPLES", "1000"))

_raw_max_case_chars = os.environ.get("MAX_CASE_CHARS", "full").strip().lower()

if _raw_max_case_chars in {"0", "-1", "none", "full"}:
    DEFAULT_MAX_CASE_CHARS = None
else:
    DEFAULT_MAX_CASE_CHARS = int(_raw_max_case_chars)

VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "131072"))
VLLM_MAX_OUTPUT_TOKENS = int(os.environ.get("VLLM_MAX_OUTPUT_TOKENS", "2048"))
VLLM_SEED = int(os.environ.get("VLLM_SEED", "0"))

DEFAULT_CASE_TEXT_STRATEGY = os.environ.get("CASE_TEXT_STRATEGY", "head_tail").strip().lower()

# Backwards-compatible aliases for older scripts in this repo.
OLLAMA_BASE_URL = VLLM_BASE_URL
OLLAMA_MODEL = VLLM_MODEL
OLLAMA_NUM_CTX = VLLM_MAX_MODEL_LEN

LEGAL_LABEL_DESCRIPTIONS = {
    "2": "Article 2: right to life; deaths, lethal force, or concrete life-threatening risks",
    "3": "Article 3: prohibition of torture, inhuman or degrading treatment; police violence, detention conditions, medical neglect in custody",
    "5": "Article 5: right to liberty and security; arrest, custody, detention lawfulness, release, remand duration",
    "6": "Article 6: fair trial; access to court, hearing fairness, length of proceedings, defence rights, impartial tribunal",
    "8": "Article 8: private/family life, home, correspondence, searches, personal data, reputation, environmental interference",
    "9": "Article 9: thought, conscience and religion; worship, religious practice, beliefs",
    "10": "Article 10: freedom of expression; journalism, speech, publications, source protection",
    "11": "Article 11: assembly and association; demonstrations, parties, trade unions, NGOs",
    "14": "Article 14: discrimination in enjoyment of Convention rights; differential treatment based on status",
    "18": "Article 18: limitation on rights for improper purpose; bad faith restriction of Convention rights",
    "P1-1": "Protocol 1 Article 1: property and possessions; land, compensation, licenses, assets, pensions",
    "P1-2": "Protocol 1 Article 2: right to education; school access, instruction, educational exclusion",
    "P1-3": "Protocol 1 Article 3: free elections; voting, candidacy, electoral process",
    "P4-2": "Protocol 4 Article 2: freedom of movement; residence, travel bans, leaving a country",
}

LABEL_NEGATIVE_RULES = {
    "2": "Do not choose it for ordinary civil, property, or environmental disputes unless life or death risk is concrete.",
    "3": "Do not choose it for mere detention, conviction, or procedural unfairness without ill-treatment or degrading conditions.",
    "5": "Do not choose it for prison conditions after lawful custody; focus on deprivation-of-liberty lawfulness or duration.",
    "6": "Do not choose it merely because domestic courts, appeals, or criminal proceedings are mentioned.",
    "8": "Do not choose it merely because a person was convicted or sued; identify private/family/home/correspondence interference.",
    "10": "Do not choose it for every newspaper or publication mention; identify expression, speech, journalism, or source-protection interference.",
    "14": "Do not choose it for general unfairness unless facts suggest discrimination or differential treatment.",
    "P1-1": "Do not choose it for every damages claim; identify possessions, land, assets, benefits, or compensation.",
}


def load_split(split: str | int = "test", n_examples: int | None = None):
    """
    Load a LexGLUE ECtHR-B split.

    Standard split names for this experiment are:
    - train: supervised scorer/meta-scorer training
    - validation: conformal calibration
    - test: final evaluation

    The older load_split(N) form is kept for run_legalreasoner_baseline.py
    and means "first N examples from test".
    """
    from datasets import load_dataset

    # Backwards-compatible form used by older scripts:
    #   load_split(1000) -> first 1000 examples from test
    if isinstance(split, int):
        n_examples = split
        split = "test"

    split = str(split).strip().lower()
    if split == "dev":
        split = "validation"

    ds = load_dataset("coastalcph/lex_glue", DATASET_CONFIG)
    label_names = ds["train"].features["labels"].feature.names
    if split not in ds:
        raise ValueError(f"Split {split!r} not found. Available: {list(ds.keys())}")

    out = ds[split]
    if n_examples is not None:
        out = out.select(range(min(int(n_examples), len(out))))
    return out, label_names


def to_multihot_from_ids(label_ids: List[int], num_labels: int) -> List[int]:
    y = [0] * num_labels
    for i in label_ids:
        y[i] = 1
    return y


def to_multihot_from_names(label_names: List[str], predicted_names: List[str]) -> List[int]:
    y = [0] * len(label_names)
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    for name in predicted_names:
        if name in label_to_idx:
            y[label_to_idx[name]] = 1
    return y


def compute_metrics(gold: List[List[int]], pred: List[List[int]]) -> Dict[str, float]:
    gold = np.array(gold)
    pred = np.array(pred)
    return {
        "micro_f1": float(f1_score(gold, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def label_for_prompt(label: str) -> str:
    label_text = clean_text(label)
    if not label_text:
        return ""
    if label_text in LEGAL_LABEL_DESCRIPTIONS:
        return LEGAL_LABEL_DESCRIPTIONS[label_text]
    if label_text.lower().startswith("article"):
        return label_text
    if label_text.replace(".", "", 1).isdigit():
        return f"Article {label_text}"
    return label_text


def build_label_legend(label_names: List[str]) -> str:
    return "\n".join(f"- {label} = {label_for_prompt(label)}" for label in label_names)


def build_label_guardrails(label_names: List[str]) -> str:
    lines = []
    for label in label_names:
        rule = LABEL_NEGATIVE_RULES.get(label)
        if rule:
            lines.append(f"- {label}: {rule}")
    return "\n".join(lines) if lines else "- Use the article descriptions above to avoid loose matches."


def sanitize_predicted_labels(
    label_names: List[str],
    predicted_names: Iterable[Any] | None,
    max_labels: int | None = None,
    allowed_labels: Iterable[str] | None = None,
) -> List[str]:
    allowed = set(label_names)
    if allowed_labels is not None:
        allowed &= {clean_text(label) for label in allowed_labels}

    if predicted_names is None:
        items: List[Any] = []
    elif isinstance(predicted_names, str):
        items = [predicted_names]
    else:
        items = list(predicted_names)

    cleaned = []
    seen = set()
    for item in items:
        label = clean_text(item)
        if label not in allowed or label in seen:
            continue
        cleaned.append(label)
        seen.add(label)
        if max_labels is not None and len(cleaned) >= max_labels:
            break

    return cleaned


def _collect_indices_from_start(paragraphs: List[str], budget: int, exclude: set[int] | None = None) -> List[int]:
    exclude = exclude or set()
    selected = []
    current_len = 0

    for idx, paragraph in enumerate(paragraphs):
        if idx in exclude:
            continue
        add_len = len(paragraph) + 2
        if selected and current_len + add_len > budget:
            break
        selected.append(idx)
        current_len += add_len
        if current_len >= budget:
            break

    return selected


def _collect_indices_from_end(paragraphs: List[str], budget: int, exclude: set[int] | None = None) -> List[int]:
    exclude = exclude or set()
    selected = []
    current_len = 0

    for idx in range(len(paragraphs) - 1, -1, -1):
        if idx in exclude:
            continue
        add_len = len(paragraphs[idx]) + 2
        if selected and current_len + add_len > budget:
            break
        selected.append(idx)
        current_len += add_len
        if current_len >= budget:
            break

    selected.reverse()
    return selected


def _collect_indices_from_middle(paragraphs: List[str], budget: int, exclude: set[int] | None = None) -> List[int]:
    exclude = exclude or set()
    if not paragraphs or budget <= 0:
        return []

    midpoint = len(paragraphs) // 2
    order = []
    for offset in range(len(paragraphs)):
        left = midpoint - offset
        right = midpoint + offset
        if 0 <= left < len(paragraphs):
            order.append(left)
        if right != left and 0 <= right < len(paragraphs):
            order.append(right)

    selected = []
    current_len = 0
    seen = set()
    for idx in order:
        if idx in exclude or idx in seen:
            continue
        add_len = len(paragraphs[idx]) + 2
        if selected and current_len + add_len > budget:
            break
        selected.append(idx)
        seen.add(idx)
        current_len += add_len
        if current_len >= budget:
            break

    return sorted(selected)


def _slice_long_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    marker = "\n\n[... MIDDLE OF CASE OMITTED FOR LENGTH ...]\n\n"
    remaining = max_chars - len(marker)
    if remaining <= 80:
        return text[:max_chars]

    head_budget = remaining // 2
    tail_budget = remaining - head_budget
    return text[:head_budget].rstrip() + marker + text[-tail_budget:].lstrip()


def _shorten_paragraph(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 8:
        return text[:max_chars]

    truncated = text[: max_chars - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "..."


def _render_selected_paragraphs(paragraphs: List[str], selected_indices: List[int], max_chars: int) -> str:
    if not selected_indices:
        return _slice_long_text("\n\n".join(paragraphs), max_chars=max_chars)

    chunks = []
    paragraph_chunk_positions = []
    previous_idx = None
    for idx in selected_indices:
        if previous_idx is not None and idx > previous_idx + 1:
            chunks.append("[... ADDITIONAL CASE FACTS OMITTED ...]")
        paragraph_chunk_positions.append(len(chunks))
        chunks.append(paragraphs[idx])
        previous_idx = idx

    rendered = "\n\n".join(chunks)
    if len(rendered) <= max_chars:
        return rendered

    marker_positions = [idx for idx in range(len(chunks)) if idx not in paragraph_chunk_positions]
    separator_overhead = 2 * max(0, len(chunks) - 1)
    marker_overhead = sum(len(chunks[idx]) for idx in marker_positions) + separator_overhead
    remaining_chars = max(40 * len(paragraph_chunk_positions), max_chars - marker_overhead)
    paragraph_budget = max(40, remaining_chars // max(1, len(paragraph_chunk_positions)))

    shortened_chunks = list(chunks)
    for idx in paragraph_chunk_positions:
        shortened_chunks[idx] = _shorten_paragraph(shortened_chunks[idx], paragraph_budget)

    rendered = "\n\n".join(shortened_chunks)
    if len(rendered) <= max_chars:
        return rendered

    return _slice_long_text(rendered, max_chars=max_chars)


def build_case_text(
    paragraphs: List[str],
    max_chars: int | None = 4000,
    strategy: str | None = None,
) -> str:
    """
    Build a case view for prompting.

    By default, this returns the full case text because DEFAULT_MAX_CASE_CHARS
    is None. If max_chars is set to an integer, the case is truncated using
    either head+tail or head+middle+tail paragraph selection.
    """
    cleaned = [clean_text(p) for p in paragraphs if clean_text(p)]
    if not cleaned:
        return ""

    joined = "\n\n".join(cleaned)

    if max_chars is None or max_chars <= 0:
        return joined
    
    if len(joined) <= max_chars:
        return joined

    mode = (strategy or DEFAULT_CASE_TEXT_STRATEGY or "head_tail").strip().lower()
    if mode not in {"head_tail", "head_middle_tail"}:
        mode = "head_tail"

    if mode == "head_tail":
        head_budget = max_chars // 2
        tail_budget = max_chars - head_budget

        head_indices = _collect_indices_from_start(cleaned, head_budget)
        tail_indices = _collect_indices_from_end(cleaned, tail_budget, exclude=set(head_indices))
        selected_indices = sorted(set(head_indices + tail_indices))
        return _render_selected_paragraphs(cleaned, selected_indices, max_chars=max_chars)

    usable_budget = max(240, max_chars - 120)
    head_budget = int(usable_budget * 0.4)
    middle_budget = int(usable_budget * 0.2)
    tail_budget = usable_budget - head_budget - middle_budget

    head_indices = _collect_indices_from_start(cleaned, head_budget)
    used_indices = set(head_indices)

    tail_indices = _collect_indices_from_end(cleaned, tail_budget, exclude=used_indices)
    used_indices.update(tail_indices)

    middle_indices = _collect_indices_from_middle(cleaned, middle_budget, exclude=used_indices)
    selected_indices = sorted(set(head_indices + middle_indices + tail_indices))

    return _render_selected_paragraphs(cleaned, selected_indices, max_chars=max_chars)


def get_label_schema(label_names: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": label_names},
                "default": []
            }
        },
        "required": ["labels"],
        "additionalProperties": False
    }


def get_reasoning_schema(label_names: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": label_names},
                        "decision": {
                            "type": "string",
                            "enum": ["support", "weak", "none"]
                        },
                        "reason": {"type": "string"}
                    },
                    "required": ["label", "decision", "reason"],
                    "additionalProperties": False
                },
                "default": []
            },
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": label_names},
                "default": []
            }
        },
        "required": ["analysis", "labels"],
        "additionalProperties": False
    }


def get_candidate_schema(label_names: List[str], max_candidates: int = 4) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": label_names},
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "reason"],
                    "additionalProperties": False,
                },
                "default": [],
                "maxItems": max_candidates,
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def get_legalreasoner_reasoning_schema(
    label_names: List[str],
    max_candidates: int = 4,
    max_steps: int = 4,
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": label_names},
                        "preliminary_decision": {
                            "type": "string",
                            "enum": ["support", "weak", "none"],
                        },
                        "reasoning_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                            "maxItems": max_steps,
                        },
                    },
                    "required": ["label", "preliminary_decision", "reasoning_steps"],
                    "additionalProperties": False,
                },
                "default": [],
                "maxItems": max_candidates,
            }
        },
        "required": ["articles"],
        "additionalProperties": False,
    }


def get_legalreasoner_verification_schema(
    label_names: List[str],
    max_candidates: int = 4,
    max_steps: int = 4,
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": label_names},
                        "label_verdict": {
                            "type": "string",
                            "enum": ["supported", "partial", "unsupported"],
                        },
                        "label_evidence": {"type": "string"},
                        "step_checks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string"},
                                    "verdict": {
                                        "type": "string",
                                        "enum": ["supported", "partial", "unsupported"],
                                    },
                                    "evidence": {"type": "string"},
                                },
                                "required": ["step", "verdict", "evidence"],
                                "additionalProperties": False,
                            },
                            "default": [],
                            "maxItems": max_steps,
                        },
                    },
                    "required": ["label", "label_verdict", "label_evidence", "step_checks"],
                    "additionalProperties": False,
                },
                "default": [],
                "maxItems": max_candidates,
            }
        },
        "required": ["articles"],
        "additionalProperties": False,
    }


def dedupe_preserve_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def labels_from_reasoning_analysis(
    analysis: List[Dict[str, Any]],
    label_names: List[str],
) -> List[str]:
    allowed = set(label_names)
    labels = []

    for item in analysis:
        label = item.get("label")
        decision = item.get("decision")
        if label in allowed and decision == "support":
            labels.append(label)

    # deduplicate while preserving order
    return list(dict.fromkeys(labels))


def labels_from_verified_supports(
    articles: List[Dict[str, Any]],
    label_names: List[str],
    reasoning_articles: List[Dict[str, Any]] | None = None,
    min_supported_steps: int = 1,
    require_label_verdict: bool = False,
) -> List[str]:
    allowed = set(label_names)
    preliminary_by_label = {
        item.get("label"): item.get("preliminary_decision")
        for item in (reasoning_articles or [])
    }
    labels = []

    for article in articles:
        label = article.get("label")
        if label not in allowed:
            continue
        if preliminary_by_label and preliminary_by_label.get(label) != "support":
            continue

        label_verdict = article.get("label_verdict")
        if require_label_verdict and label_verdict != "supported":
            continue

        step_checks = article.get("step_checks", [])
        supported_steps = sum(item.get("verdict") == "supported" for item in step_checks)
        if supported_steps >= min_supported_steps:
            labels.append(label)

    return dedupe_preserve_order(labels)


def enforce_support_only_predictions(
    label_names: List[str],
    model_labels: Iterable[Any] | None,
    support_only_labels: Iterable[str] | None,
    max_labels: int | None = None,
) -> List[str]:
    """Keep final predictions within the labels backed by support evidence."""

    supported_labels = sanitize_predicted_labels(
        label_names,
        support_only_labels,
        max_labels=max_labels,
    )
    if not supported_labels:
        return []

    filtered_model_labels = sanitize_predicted_labels(
        label_names,
        model_labels,
        max_labels=max_labels,
        allowed_labels=supported_labels,
    )
    return filtered_model_labels or supported_labels


def _extract_json_candidate(text: str) -> str:
    """
    Try to isolate the main JSON object if the model adds stray text.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _debug_response(data: Dict[str, Any], content: str) -> None:
    choice = {}
    if isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0] or {}
    usage = data.get("usage", {})

    print("\n[VLLM JSON PARSE ERROR]")
    print("model:", data.get("model"))
    print("finish_reason:", choice.get("finish_reason"))
    print("prompt_tokens:", usage.get("prompt_tokens"))
    print("completion_tokens:", usage.get("completion_tokens"))
    print("raw content preview:")
    print(content[:1200])
    print("-" * 60)


def _build_vllm_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    return headers


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
        if parts:
            return "".join(parts)

    raise ValueError("vLLM response did not include string message content.")


def call_vllm_chat(
    prompt: str,
    schema: Dict[str, Any],
    max_tokens: int = 64,
    timeout: int = 300,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Shared vLLM structured-output caller.

    Retries with a larger max_tokens budget if the model returns malformed/truncated JSON.
    """
    current_max_tokens = max_tokens
    last_error = None
    endpoint = f"{VLLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = _build_vllm_headers()

    for attempt in range(max_retries + 1):
        payload = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON that matches the provided schema exactly. "
                        "Do not include markdown, explanations, or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": current_max_tokens,
            "seed": VLLM_SEED,
            "guided_json": schema,
        }

        try:
            r = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()

            data = r.json()
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("vLLM response did not include choices.")

            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ValueError("vLLM response did not include a message object.")

            content = _message_content_to_text(message.get("content")).strip()
            content = _extract_json_candidate(content)

            return json.loads(content)

        except json.JSONDecodeError as e:
            last_error = e

            # Try again with a larger output budget in case the JSON was truncated.
            if attempt < max_retries:
                current_max_tokens = min(current_max_tokens * 2, VLLM_MAX_OUTPUT_TOKENS)
                continue

            try:
                _debug_response(data, content)
            except Exception:
                pass
            raise

        except (TypeError, ValueError) as e:
            last_error = e
            if attempt < max_retries:
                continue
            raise

        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries:
                continue
            raise

    raise RuntimeError(f"vLLM call failed after retries: {last_error}")


def call_vllm_chat_direct(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return call_vllm_chat(
        prompt=prompt,
        schema=schema,
        max_tokens=128,
        timeout=900,
        max_retries=2,
    )


def call_vllm_chat_reasoning(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return call_vllm_chat(
        prompt=prompt,
        schema=schema,
        max_tokens=256,
        timeout=900,
        max_retries=2,
    )


def call_vllm_chat_legalreasoner(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return call_vllm_chat(
        prompt=prompt,
        schema=schema,
        max_tokens=min(768, VLLM_MAX_OUTPUT_TOKENS),
        timeout=1200,
        max_retries=3,
    )


def call_ollama_chat(
    prompt: str,
    schema: Dict[str, Any],
    num_predict: int = 64,
    timeout: int = 300,
    max_retries: int = 2,
) -> Dict[str, Any]:
    return call_vllm_chat(
        prompt=prompt,
        schema=schema,
        max_tokens=num_predict,
        timeout=timeout,
        max_retries=max_retries,
    )


def call_ollama_chat_direct(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return call_vllm_chat_direct(prompt, schema)


def call_ollama_chat_reasoning(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return call_vllm_chat_reasoning(prompt, schema)


def call_ollama_chat_legalreasoner(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return call_vllm_chat_legalreasoner(prompt, schema)
