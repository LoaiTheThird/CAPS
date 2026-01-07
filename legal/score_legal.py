# legal/score_legal.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from legal.schema import LegalExample, LegalTraceCandidate, LegalScoredCandidate, compute_nonconformity
from legal.judge.ollama_judge import judge_trace_with_ollama

CHOICE_LABELS = ["A", "B", "C", "D"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, recs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_choices_from_context(context: Optional[str]) -> Dict[str, str]:
    """
    Parse lines like "A) ..." from context into {label -> text}.
    """
    if not context:
        return {}
    m: Dict[str, str] = {}
    for line in context.splitlines():
        line = line.strip()
        mm = re.match(r"^([A-D])\)\s*(.*)$", line)
        if mm:
            m[mm.group(1)] = mm.group(2).strip()
    return m


def normalize_gold(gold_answer: Optional[str], ex: LegalExample) -> Optional[str]:
    """
    Convert gold into a letter A/B/C/D if possible.
    Handles:
      - "A"/"B"/...
      - "0"/"1"/"2"/"3"
      - gold as full choice text (rare) by matching context options
    """
    if gold_answer is None:
        return None

    g = str(gold_answer).strip()

    if g in CHOICE_LABELS:
        return g

    if re.fullmatch(r"\d+", g):
        idx = int(g)
        if 0 <= idx < len(CHOICE_LABELS):
            return CHOICE_LABELS[idx]

    choices = parse_choices_from_context(ex.context)
    if choices:
        g_norm = g.lower().strip()
        for label, text in choices.items():
            if text.lower().strip() == g_norm:
                return label

    return None


def normalize_answer(ans: str) -> Optional[str]:
    """
    Convert model answer into A/B/C/D if possible.
    """
    a = (ans or "").strip().upper()
    if a in CHOICE_LABELS:
        return a
    # tolerant: allow "Answer: B" or "B." etc
    m = re.search(r"\b([A-D])\b", a)
    if m:
        return m.group(1)
    return None


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score LEXam candidates with continuous q via local Ollama judge.")
    p.add_argument("--examples", type=Path, required=True, help="JSONL from make_lexam_splits.py")
    p.add_argument("--candidates", type=Path, required=True, help="Candidates JSONL from run_generate_legal.py")
    p.add_argument("--output", type=Path, required=True, help="Scored JSONL output")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--ollama_model", type=str, default="llama3.2")
    p.add_argument("--timeout_s", type=int, default=180)
    p.add_argument("--max_candidates", type=int, default=None, help="Cap for quick tests")

    # Optional: lightly mix in correctness so q reflects both reasoning quality and correct choice
    p.add_argument("--mix_gold", type=float, default=0.2,
                   help="q_final = (1-mix_gold)*q_judge + mix_gold*correct (0..1). Set 0 to ignore gold.")
    # Optional: acceptance threshold for later metrics
    p.add_argument("--accept_threshold", type=float, default=0.5,
                   help="acceptable := q_final >= threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure output directory exists and (optionally) clear output file immediately
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.write_text("")  # create/clear file right away so wc -l works immediately
    else:
        if args.output.exists():
            raise SystemExit(f"Output exists: {args.output}. Pass --overwrite to replace.")

    ex_rows = read_jsonl(args.examples)
    ex_by_id: Dict[str, LegalExample] = {}
    for r in ex_rows:
        ex = LegalExample(**r)
        ex_by_id[ex.question_id] = ex

    cand_rows = read_jsonl(args.candidates)
    if args.max_candidates is not None:
        cand_rows = cand_rows[: args.max_candidates]

    mix = clamp01(float(args.mix_gold))
    thr = clamp01(float(args.accept_threshold))

    out_batch: List[Dict[str, Any]] = []
    n_scored = 0
    n_parsed_gold = 0
    n_correct = 0

    for r in cand_rows:
        cand = LegalTraceCandidate(**r)
        ex = ex_by_id.get(cand.question_id)
        if ex is None:
            # candidate not in this split (e.g., calib scorer pointed at eval candidates)
            continue

        # Gold correctness (for coverage evaluation later)
        gold_letter = normalize_gold(ex.gold_answer, ex)
        ans_letter = normalize_answer(cand.answer)
        correct: Optional[bool] = None
        if gold_letter is not None and ans_letter is not None:
            correct = (ans_letter == gold_letter)
            n_parsed_gold += 1
            if correct:
                n_correct += 1

        # Judge quality score q in [0,1]
        jr = judge_trace_with_ollama(
            question=ex.question,
            context=ex.context,
            trace_text=cand.trace_text,
            answer=cand.answer,
            model=args.ollama_model,
            timeout_s=args.timeout_s,
        )
        # Use rubric mean as the primary continuous quality score (more stable than jr.q)
        rub = jr.rubric or {}
        vals = [
            float(rub.get("issue", 0.0)),
            float(rub.get("rule", 0.0)),
            float(rub.get("application", 0.0)),
            float(rub.get("coherence", 0.0)),
        ]
        q_rubric = clamp01((sum(vals) / 4.0) / 100.0)
        q = q_rubric

        # Mix correctness into q if available (keeps q continuous)
        if mix > 0.0 and correct is not None:
            q = clamp01((1.0 - mix) * q + mix * (1.0 if correct else 0.0))

        score = compute_nonconformity(q)
        acceptable = (q >= thr)

        scored = LegalScoredCandidate(
            question_id=cand.question_id,
            trace_id=cand.trace_id,
            answer=cand.answer,
            trace_text=cand.trace_text,
            q=q,
            score=score,
            correct=correct,
            acceptable=acceptable,
            generator=cand.generator,
            rubric=jr.rubric,
            judge={
                "mode": "ollama_judge",
                "model": args.ollama_model,
                "verdict": jr.verdict,
                "notes": jr.notes,
                "q_source": "rubric_mean",
            },
        )

        out_batch.append(scored.model_dump())
        n_scored += 1

        # Flush frequently so you see progress while Ollama judging is slow
        if len(out_batch) >= 20:
            write_jsonl(args.output, out_batch)
            out_batch = []
            if n_scored % 100 == 0:
                print(f"Scored {n_scored} candidates...", flush=True)

    if out_batch:
        write_jsonl(args.output, out_batch)

    acc = (n_correct / n_parsed_gold) if n_parsed_gold else 0.0
    print(f"Done. Scored {n_scored} candidates -> {args.output}")
    print(f"Gold comparable for {n_parsed_gold}/{n_scored} candidates; raw accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
