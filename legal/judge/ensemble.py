# legal/judge/ensemble.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

from legal.judge.ollama_judge import JudgeResult, judge_trace_with_ollama


@dataclass
class EnsembleMember:
    name: str
    template_path: Path
    weight: float = 1.0


def _norm_rubric(r: Dict[str, float]) -> Dict[str, float]:
    # keep rubric in 0..100 space (as produced by judge)
    out = {}
    for k, v in (r or {}).items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = 0.0
    return out


def judge_trace_ensemble_with_ollama(
    *,
    question: str,
    context: Optional[str],
    trace_text: str,
    answer: str,
    model: str,
    members: List[EnsembleMember],
    timeout_s: int = 180,
) -> JudgeResult:
    """
    Runs multiple judge prompts and aggregates into a single JudgeResult.
    - q: weighted mean of member q's (0..1 in JudgeResult output)
    - rubric: merged mean rubric (scaled to 0..100 in the raw rubric dict)
    - verdict: pass if average q >= 0.5 else fail
    """
    results: List[JudgeResult] = []
    wsum = sum(m.weight for m in members) or 1.0

    for m in members:
        jr = judge_trace_with_ollama(
            question=question,
            context=context,
            trace_text=trace_text,
            answer=answer,
            model=model,
            timeout_s=timeout_s,
            template_path=m.template_path,
        )
        results.append(jr)

    # Weighted mean q (jr.q is 0..1)
    q = 0.0
    for m, jr in zip(members, results):
        q += m.weight * float(jr.q)
    q = q / wsum

    # Merge rubrics by averaging overlapping keys (keep 0..100 style values)
    keys = set()
    for jr in results:
        keys |= set((jr.rubric or {}).keys())

    rubric_out: Dict[str, float] = {}
    for k in sorted(keys):
        s = 0.0
        for m, jr in zip(members, results):
            rr = _norm_rubric(jr.rubric or {})
            s += m.weight * float(rr.get(k, 0.0))
        rubric_out[k] = s / wsum

    verdict = "pass" if q >= 0.5 else "fail"
    notes = " | ".join([f"{m.name}:{(jr.notes or '').strip()}" for m, jr in zip(members, results)])[:500]
    raw = " || ".join([jr.raw or "" for jr in results])[:4000]

    return JudgeResult(q=q, rubric=rubric_out, verdict=verdict, notes=notes, raw=raw)
