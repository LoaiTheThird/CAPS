# legal/score_legal.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from legal.schema import (
    LegalExample,
    LegalTraceCandidate,
    LegalScoredCandidate,
    compute_nonconformity,
)
from legal.judge.ollama_judge import judge_trace_with_ollama

CHOICE_LABELS = ["A", "B", "C", "D"]


# ----------------------------
# IO
# ----------------------------
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


# ----------------------------
# Helpers: choices + answers
# ----------------------------
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


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Judge aggregation (IMPORTANT)
# ----------------------------
def _infer_scale(vals: List[float]) -> float:
    """
    If rubric values look like 0..100 -> return 100
    If they look like 0..1 -> return 1
    Otherwise: if max>1.5 assume 100, else assume 1
    """
    if not vals:
        return 1.0
    mx = max(vals)
    if mx <= 1.0:
        return 1.0
    if mx <= 1.5:
        return 1.0
    return 100.0

#
# def q_from_rubric(rubric: Dict[str, Any]) -> float:
#     """
#     Compute a stable continuous q in [0,1] from rubric fields.
#
#     Supports:
#     - quality rubric: issue/rule/application/coherence (0..1 or 0..100)
#     - correctness rubric: supported/contradictions/choice_alignment/missing_rule (0..1 or 0..100)
#     - fallback: mean of all numeric rubric values
#     """
#     r = rubric or {}
#     # prefer known sets of keys
#     quality_keys = ["issue", "rule", "application", "coherence"]
#     corr_keys = ["supported", "contradictions", "choice_alignment", "missing_rule"]
#
#     def pull(keys: List[str]) -> List[float]:
#         out: List[float] = []
#         for k in keys:
#             if k in r:
#                 out.append(_to_float(r.get(k, 0.0), 0.0))
#         return out
#
#     vals = pull(quality_keys)
#     if vals:
#         scale = _infer_scale(vals)
#         return clamp01(sum(vals) / (len(vals) * scale))
#
#     vals = pull(corr_keys)
#     if vals:
#         scale = _infer_scale(vals)
#         return clamp01(sum(vals) / (len(vals) * scale))
#
#     # fallback: average all numeric rubric values
#     all_vals: List[float] = []
#     for v in r.values():
#         fv = _to_float(v, None)  # type: ignore[arg-type]
#         if fv is None:
#             continue
#         all_vals.append(fv)
#     if not all_vals:
#         return 0.0
#     scale = _infer_scale(all_vals)
#     return clamp01(sum(all_vals) / (len(all_vals) * scale))
#
def q_from_jr(jr) -> float:
    q_raw = _to_float(getattr(jr, "q", 0.0), 0.0)
    if q_raw > 1.5:  # assume 0..100
        q_raw = q_raw / 100.0
    return clamp01(q_raw)


def q_from_rubric(rubric: Dict[str, Any]) -> float:
    """
    Compute stable continuous q in [0,1] from rubric fields.

    Supports:
    - quality rubric (all positive): issue/rule/application/coherence
    - correctness rubric (some negative): supported, choice_alignment (positive),
      contradictions, missing_rule (negative -> invert)
    """
    r = rubric or {}

    # --- 1) quality rubric (all positive)
    quality_keys = ["issue", "rule", "application", "coherence"]
    q_vals = []
    for k in quality_keys:
        if k in r:
            q_vals.append(_to_float(r.get(k, 0.0), 0.0))
    if q_vals:
        scale = _infer_scale(q_vals)
        return clamp01(sum(q_vals) / (len(q_vals) * scale))

    # --- 2) correctness rubric (mixed signs)
    pos_keys = ["supported", "choice_alignment"]
    neg_keys = ["contradictions", "missing_rule"]

    pos = []
    for k in pos_keys:
        if k in r:
            pos.append(_to_float(r.get(k, 0.0), 0.0))

    neg = []
    for k in neg_keys:
        if k in r:
            neg.append(_to_float(r.get(k, 0.0), 0.0))

    if pos or neg:
        vals = pos + neg
        scale = _infer_scale(vals)

        pos_mean = (sum(pos) / (len(pos) * scale)) if pos else 0.0
        neg_mean = (sum(neg) / (len(neg) * scale)) if neg else 0.0

        # Negatives reduce quality
        q = 0.5 * pos_mean + 0.5 * (1.0 - neg_mean)
        return clamp01(q)

    # --- 3) fallback: average all numeric
    all_vals = []
    for v in r.values():
        try:
            all_vals.append(float(v))
        except Exception:
            pass
    if not all_vals:
        return 0.0
    scale = _infer_scale(all_vals)
    return clamp01(sum(all_vals) / (len(all_vals) * scale))

def merge_rubrics_weighted(rubrics: List[Dict[str, Any]], weights: List[float]) -> Dict[str, float]:
    """
    Weighted mean per key across rubric dicts.
    Keeps raw numeric scale (0..100 or 0..1) as-is; this is mainly for logging/inspection.
    """
    keys = set()
    for r in rubrics:
        keys |= set((r or {}).keys())

    wsum = sum(weights) if sum(weights) > 0 else 1.0
    out: Dict[str, float] = {}
    for k in sorted(keys):
        s = 0.0
        for r, w in zip(rubrics, weights):
            s += w * _to_float((r or {}).get(k, 0.0), 0.0)
        out[k] = s / wsum
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score LEXam candidates with continuous q via local Ollama judge(s).")
    p.add_argument("--examples", type=Path, required=True, help="JSONL from make_lexam_splits.py")
    p.add_argument("--candidates", type=Path, required=True, help="Candidates JSONL from run_generate_legal.py")
    p.add_argument("--output", type=Path, required=True, help="Scored JSONL output")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--ollama_model", type=str, default="llama3.2")
    p.add_argument("--timeout_s", type=int, default=180)
    p.add_argument("--max_candidates", type=int, default=None, help="Cap for quick tests")

    # Judge mode
    p.add_argument("--judge_mode", choices=["single", "ensemble"], default="single")
    p.add_argument(
        "--single_template",
        type=Path,
        default=Path("legal/prompts/judge_trace_quality.txt"),
        help="Template to use in single mode.",
    )
    p.add_argument(
        "--ensemble_templates",
        type=str,
        default="legal/prompts/judge_trace_quality.txt,legal/prompts/judge_trace_correctness.txt",
        help="Comma-separated templates for ensemble mode.",
    )
    p.add_argument(
        "--ensemble_weights",
        type=str,
        default="0.7,0.3",
        help="Comma-separated weights matching --ensemble_templates.",
    )

    # Optional: lightly mix in correctness so q reflects both reasoning quality and correct choice
    p.add_argument(
        "--mix_gold",
        type=float,
        default=0.2,
        help="q_final = (1-mix_gold)*q_judge + mix_gold*correct (0..1). Set 0 to ignore gold.",
    )
    # Optional: acceptance threshold for later metrics
    p.add_argument("--accept_threshold", type=float, default=0.5, help="acceptable := q_final >= threshold")

    # flush / logging
    p.add_argument("--flush_every", type=int, default=20, help="Write batch to disk every N scored rows.")
    p.add_argument("--print_every", type=int, default=200, help="Print progress every N scored rows.")

    return p.parse_args()


def _parse_templates_and_weights(args: argparse.Namespace) -> Tuple[List[Path], List[float]]:
    tps = [Path(s.strip()) for s in (args.ensemble_templates or "").split(",") if s.strip()]
    ws = [float(s.strip()) for s in (args.ensemble_weights or "").split(",") if s.strip()]
    if not tps:
        # default fallback (shouldn't happen)
        tps = [Path("legal/prompts/judge_trace_quality.txt"), Path("legal/prompts/judge_trace_correctness.txt")]
    if len(ws) != len(tps):
        # auto-fix: pad/truncate to match
        if len(ws) < len(tps):
            ws = ws + [1.0] * (len(tps) - len(ws))
        else:
            ws = ws[: len(tps)]
    # avoid all-zero
    if sum(ws) <= 0:
        ws = [1.0] * len(tps)
    return tps, ws


# ----------------------------
# Main
# ----------------------------
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

    if args.judge_mode == "ensemble":
        templates, weights = _parse_templates_and_weights(args)
    else:
        templates, weights = [], []

    out_batch: List[Dict[str, Any]] = []
    n_scored = 0
    n_parsed_gold = 0
    n_correct = 0

    for r in cand_rows:
        cand = LegalTraceCandidate(**r)
        ex = ex_by_id.get(cand.question_id)
        if ex is None:
            # candidate not in this split
            continue

        # Gold correctness (for later coverage evaluation)
        gold_letter = normalize_gold(ex.gold_answer, ex)
        ans_letter = normalize_answer(cand.answer)
        correct: Optional[bool] = None
        if gold_letter is not None and ans_letter is not None:
            correct = (ans_letter == gold_letter)
            n_parsed_gold += 1
            if correct:
                n_correct += 1

        # -------------------------
        # Judge q in [0,1]
        # -------------------------
        if args.judge_mode == "single":
            jr = judge_trace_with_ollama(
                question=ex.question,
                context=ex.context,
                trace_text=cand.trace_text,
                answer=cand.answer,
                model=args.ollama_model,
                timeout_s=args.timeout_s,
                template_path=args.single_template,
            )
            ##Single
            # q = q_from_rubric(jr.rubric or {})
            q = q_from_rubric(jr.rubric or {})
            ##
            merged_rubric = jr.rubric
            judge_payload: Dict[str, Any] = {
                "mode": "single",
                "model": args.ollama_model,
                "template": str(args.single_template),
                "verdict": jr.verdict,
                "notes": jr.notes,
                "q_source": "computed_from_rubric",
            }
        else:
            member_results = []
            member_qs: List[float] = []
            member_rubrics: List[Dict[str, Any]] = []
            member_notes: List[str] = []
            member_verdicts: List[str] = []

            for tp in templates:
                jr_m = judge_trace_with_ollama(
                    question=ex.question,
                    context=ex.context,
                    trace_text=cand.trace_text,
                    answer=cand.answer,
                    model=args.ollama_model,
                    timeout_s=args.timeout_s,
                    template_path=tp,
                )
                member_results.append(jr_m)
                ##ensemble
                # rq = q_from_rubric(jr_m.rubric or {})
                # member_qs.append(rq)
                member_qs.append(q_from_rubric(jr_m.rubric or {}))
                ##
                member_rubrics.append(jr_m.rubric or {})
                member_notes.append((jr_m.notes or "").strip())
                member_verdicts.append((jr_m.verdict or "unknown").strip())

            wsum = sum(weights) if sum(weights) > 0 else 1.0
            q = 0.0
            for w, qq in zip(weights, member_qs):
                q += w * qq
            q = q / wsum
            q = clamp01(q)

            merged_rubric = merge_rubrics_weighted(member_rubrics, weights)

            judge_payload = {
                "mode": "ensemble",
                "model": args.ollama_model,
                "templates": [str(t) for t in templates],
                "weights": weights,
                "member_qs": member_qs,
                "member_verdicts": member_verdicts,
                "member_notes": member_notes[:4],  # keep small
                "q_source": "weighted_mean_of_member_rubric_scores",
            }

        # Optional: mix correctness into q
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
            rubric=merged_rubric,
            judge=judge_payload,
        )

        out_batch.append(scored.model_dump())
        n_scored += 1

        # Flush frequently so you see progress while Ollama judging is slow
        if len(out_batch) >= int(args.flush_every):
            write_jsonl(args.output, out_batch)
            out_batch = []
            if n_scored % int(args.print_every) == 0:
                print(f"Scored {n_scored} candidates...", flush=True)

    if out_batch:
        write_jsonl(args.output, out_batch)

    acc = (n_correct / n_parsed_gold) if n_parsed_gold else 0.0
    print(f"Done. Scored {n_scored} candidates -> {args.output}")
    print(f"Gold comparable for {n_parsed_gold}/{n_scored} candidates; raw accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
