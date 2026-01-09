# legal/judge/ollama_judge.py
from __future__ import annotations

import urllib.request
import urllib.error


import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    q: float
    rubric: Dict[str, float]
    verdict: str
    notes: str
    raw: str


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    t = t.replace("```json", "").replace("```", "").strip()
    m = _JSON_RE.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# def _extract_json(text: str) -> Optional[Dict[str, Any]]:
#     t = (text or "").strip()
#
#     # take first non-empty line
#     line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
#     if not line:
#         return None
#
#     # remove any stray backticks if they appear
#     line = line.replace("```json", "").replace("```", "").strip()
#
#     # must start with '{'
#     if not line.startswith("{"):
#         return None
#
#     try:
#         return json.loads(line)
#     except Exception:
#         return None


# Prev version:
# def call_ollama(prompt: str, model: str, timeout_s: int) -> str:
#     cmd = [
#         "ollama", "run", model,
#         "-o", "temperature=0",
#         "-o", "num_predict=256",
#         "-o", "top_p=0.9",
#         # stop tokens (best effort; some models still ignore)
#         "-o", "stop=}\n",
#         "-o", "stop=}\r\n",
#     ]
#
#     proc = subprocess.run(
#         cmd,
#         input=prompt,
#         text=True,
#         capture_output=True,
#         timeout=timeout_s,
#     )
#     if proc.returncode != 0:
#         raise RuntimeError(f"Ollama failed (rc={proc.returncode}): {proc.stderr.strip()}")
#     return proc.stdout.strip()


def call_ollama(prompt: str, model: str, timeout_s: int) -> str:
    """
    Call local Ollama server via HTTP API.
    This supports options (num_predict, temperature, stop, etc.) reliably.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
            "top_p": 0.9,
            "stop": ["\n"],
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama HTTP call failed: {e}") from e

    obj = json.loads(out)
    return (obj.get("response") or "").strip()


def load_template(path: Path) -> Optional[str]:
    return path.read_text() if path.exists() else None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def fill_template(template: str, mapping: Dict[str, str]) -> str:
    """
    Safe template filling: only replaces {question}/{context}/{trace_text}/{answer}.
    Leaves any other braces (e.g., JSON examples) untouched.
    """
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def judge_trace_with_ollama(
    *,
    question: str,
    context: Optional[str],
    trace_text: str,
    answer: str,
    model: str,
    timeout_s: int = 180,
    template_path: Path = Path("legal/prompts/judge_trace.txt"),
    max_retries: int = 2,
) -> JudgeResult:
    """
    Returns continuous q in [0,1] plus rubric sub-scores.
    Fully local + free if you use Ollama.
    """
    template = load_template(template_path)
    if template is None:
        template = (
            "Return ONLY JSON with keys q,rubric,verdict,notes.\n\n"
            "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
            "TRACE:\n{trace_text}\n\nANSWER: {answer}\n"
        )

    prompt = fill_template(
        template,
        {
            "question": question,
            "context": (context or ""),
            "trace_text": trace_text,
            "answer": answer,
        },
    )

    last_raw = ""
    for _ in range(max_retries + 1):
        # raw = call_ollama(prompt, model=model, timeout_s=timeout_s)
        try:
            raw = call_ollama(prompt, model=model, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            return JudgeResult(
                q=0.0,
                rubric={"issue": 0.0, "rule": 0.0, "application": 0.0, "coherence": 0.0},
                verdict="unknown",
                notes="timeout",
                raw="",
            )
        except Exception as e:
            return JudgeResult(
                q=0.0,
                rubric={"issue": 0.0, "rule": 0.0, "application": 0.0, "coherence": 0.0},
                verdict="unknown",
                notes=f"ollama_error:{e}",
                raw="",
            )

        last_raw = raw
        obj = _extract_json(raw)
        if obj is None:
            continue

        q_raw = _safe_float(obj.get("q", 0.0), 0.0)
        rubric_in = obj.get("rubric", {}) or {}

        issue_raw = _safe_float(rubric_in.get("issue", 0.0), 0.0)
        rule_raw = _safe_float(rubric_in.get("rule", 0.0), 0.0)
        app_raw = _safe_float(rubric_in.get("application", 0.0), 0.0)
        coh_raw = _safe_float(rubric_in.get("coherence", 0.0), 0.0)

        # If any value > 1, assume the judge is using 0..100 integers
        uses_100 = any(v > 1.0 for v in [q_raw, issue_raw, rule_raw, app_raw, coh_raw])

        if uses_100:
            q = max(0.0, min(100.0, q_raw))
            rubric = {
                "issue": max(0.0, min(100.0, issue_raw)),
                "rule": max(0.0, min(100.0, rule_raw)),
                "application": max(0.0, min(100.0, app_raw)),
                "coherence": max(0.0, min(100.0, coh_raw)),
            }
        else:
            q = _clamp01(q_raw)
            rubric = {
                "issue": _clamp01(issue_raw),
                "rule": _clamp01(rule_raw),
                "application": _clamp01(app_raw),
                "coherence": _clamp01(coh_raw),
            }

        verdict = str(obj.get("verdict", "unknown")).strip().lower()
        if verdict not in {"pass", "fail", "unknown"}:
            verdict = "unknown"

        notes = str(obj.get("notes", "")).strip()

        return JudgeResult(q=q, rubric=rubric, verdict=verdict, notes=notes, raw=raw)

    # Conservative fallback if parsing keeps failing
    return JudgeResult(
        q=0.0,
        rubric={"issue": 0.0, "rule": 0.0, "application": 0.0, "coherence": 0.0},
        verdict="unknown",
        notes="failed_to_parse_json",
        raw=last_raw,
    )
