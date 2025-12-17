# legal/schema.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


# ----------------------------
# Core record types (JSONL-friendly)
# ----------------------------

class LegalExample(BaseModel):
    """
    A single legal question / case instance x.

    This is the unit you split into calib/eval.
    """
    model_config = ConfigDict(extra="allow")

    question_id: str = Field(..., description="Unique ID for the question/case.")
    question: str = Field(..., description="The user question (prompt) to answer.")
    context: Optional[str] = Field(
        None,
        description="Optional case facts / statute excerpts / background text.",
    )
    gold_answer: Optional[str] = Field(
        None,
        description="Optional gold label (e.g., correct option, ruling).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalTraceCandidate(BaseModel):
    """
    One generated reasoning trace r_k for a given example x.

    This is what your generator produces (K per question_id).
    """
    model_config = ConfigDict(extra="allow")

    question_id: str
    trace_id: str = Field(..., description="Unique candidate ID within question_id.")
    answer: str = Field(..., description="Final answer / decision / option.")
    trace_text: str = Field(..., description="Full reasoning trace text.")
    generator: Dict[str, Any] = Field(
        default_factory=dict,
        description="Info about model/prompt/temperature/seed/etc.",
    )


JudgeVerdict = Literal["pass", "fail", "unknown"]


class LegalJudgeScore(BaseModel):
    """
    Output of a judge/verifier for a candidate trace.

    q in [0,1] is the main scalar we conformalize.
    """
    model_config = ConfigDict(extra="allow")

    question_id: str
    trace_id: str

    # Scalar quality score from judge
    q: float = Field(..., ge=0.0, le=1.0, description="Judge quality score in [0,1].")

    # Optional discrete verdict (if you have a hard accept/reject rule)
    verdict: JudgeVerdict = Field(
        "unknown",
        description="Optional coarse label from judge rubric.",
    )

    # Optional rubric dimensions (also in [0,1])
    rubric: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional sub-scores, e.g. {'issue':0.8,'rule':0.6,'application':0.7}",
    )

    # Judge metadata (model name, prompt version, etc.)
    judge: Dict[str, Any] = Field(default_factory=dict)


class LegalScoredCandidate(BaseModel):
    """
    The record after scoring: includes candidate + judge + nonconformity.

    This is the input to conformal/split.py and build_sets.py.
    """
    model_config = ConfigDict(extra="allow")

    question_id: str
    trace_id: str
    answer: str
    trace_text: str

    # Judge score
    q: float = Field(..., ge=0.0, le=1.0)

    # Nonconformity score (lower = better)
    score: float = Field(..., ge=0.0, le=1.0, description="Typically score = 1 - q")

    # Optional flags for downstream metrics
    # - correct: matches gold (if you have gold_answer)
    # - acceptable: judged pass (if you define a threshold rule)
    correct: Optional[bool] = None
    acceptable: Optional[bool] = None

    # Keep rich metadata
    generator: Dict[str, Any] = Field(default_factory=dict)
    rubric: Dict[str, float] = Field(default_factory=dict)
    judge: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# Helper utilities
# ----------------------------

def compute_nonconformity(q: float) -> float:
    """
    Default LEXAM-style nonconformity: s = 1 - q.
    """
    q = float(q)
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0
    return 1.0 - q


def make_scored_candidate(
    cand: LegalTraceCandidate,
    judge_score: LegalJudgeScore,
    *,
    correct: Optional[bool] = None,
    acceptable: Optional[bool] = None,
) -> LegalScoredCandidate:
    """
    Merge a candidate + judge output into a scored candidate record.
    """
    if cand.question_id != judge_score.question_id or cand.trace_id != judge_score.trace_id:
        raise ValueError("Candidate and judge_score IDs do not match.")

    score = compute_nonconformity(judge_score.q)

    return LegalScoredCandidate(
        question_id=cand.question_id,
        trace_id=cand.trace_id,
        answer=cand.answer,
        trace_text=cand.trace_text,
        q=judge_score.q,
        score=score,
        correct=correct,
        acceptable=acceptable,
        generator=cand.generator,
        rubric=judge_score.rubric,
        judge=judge_score.judge,
    )
