# legal/judge/ollama_judge.py
from __future__ import annotations

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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from text and parse it.
    Many local models sometimes add pre/post text; this tolerates that.
    """
    m = _JSON_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


# def call_ollama(prompt: str, model: str, timeout_s: int) -> str:
#     proc = subprocess.run(
#         ["ollama", "run", model],
#         input=prompt,
#         text=True,
#         capture_output=True,
#         timeout=timeout_s,
#     )
#     if proc.returncode != 0:
#         raise RuntimeError(f"Ollama failed (rc={proc.returncode}): {proc.stderr.strip()}")
#     return proc.stdout.strip()

def call_ollama(prompt: str, model: str, timeout_s: int) -> str:
    cmd = [
        "ollama", "run", model,
        "-o", "temperature=0",
        "-o", "num_predict=256",
        "-o", "top_p=0.9",
        # stop tokens (best effort; some models still ignore)
        "-o", "stop=}\n",
        "-o", "stop=}\r\n",
    ]

    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def load_template(path: Path) -> Optional[str]:
    return path.read_text() if path.exists() else None


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

    prompt = template.format(
        question=question,
        context=(context or ""),
        trace_text=trace_text,
        answer=answer,
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

        q = _clamp01(_safe_float(obj.get("q", 0.0), 0.0))
        rubric_in = obj.get("rubric", {}) or {}

        rubric = {
            "issue": _clamp01(_safe_float(rubric_in.get("issue", 0.0), 0.0)),
            "rule": _clamp01(_safe_float(rubric_in.get("rule", 0.0), 0.0)),
            "application": _clamp01(_safe_float(rubric_in.get("application", 0.0), 0.0)),
            "coherence": _clamp01(_safe_float(rubric_in.get("coherence", 0.0), 0.0)),
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
