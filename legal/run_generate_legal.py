# legal/run_generate_legal.py
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from legal.schema import LegalExample, LegalTraceCandidate


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


def load_prompt_template(path: Path) -> Optional[str]:
    if path.exists():
        return path.read_text()
    return None


def extract_choice_block(context: Optional[str]) -> str:
    # Your make_lexam_splits puts choices into context as:
    # "Choices:\nA) ...\nB) ...\n..."
    return context or ""


def parse_mcq_answer(text: str) -> Optional[str]:
    """
    Robustly extract A/B/C/D from model output.
    Prefer lines like "Answer: B" but fall back to first standalone A-D.
    """
    if not text:
        return None

    m = re.search(r"(?i)\banswer\s*:\s*([ABCD])\b", text)
    if m:
        return m.group(1).upper()

    # Try "Final: C" etc.
    m = re.search(r"(?i)\b(final|choice)\s*:\s*([ABCD])\b", text)
    if m:
        return m.group(2).upper()

    # Fallback: first standalone A/B/C/D token (avoid matching inside words)
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1).upper()

    return None


def heuristic_generate(
    ex: LegalExample,
    k: int,
    rng: random.Random,
    style: str,
) -> Tuple[str, str]:
    """
    Free baseline generator: produces a structured trace + a choice.
    It's not "smart", but it's consistent and gives you a runnable pipeline.
    """
    choice_block = extract_choice_block(ex.context)
    # Pick answer with slight "keyword" bias: if question contains "not"/"except", try C/D more often.
    q = ex.question.lower()
    bias_pool = ["C", "D"] if (" not " in q or " except " in q or " least " in q) else ["A", "B", "C", "D"]
    answer = rng.choice(bias_pool)

    trace_lines = []
    trace_lines.append(f"Strategy: {style}")
    trace_lines.append("1) Identify the legal issue(s) asked by the question.")
    trace_lines.append("2) Recall the relevant doctrine/rule from general principles (no citations provided).")
    trace_lines.append("3) Compare each option to the rule and eliminate inconsistent choices.")
    trace_lines.append("4) Select the option that best matches the rule and the question wording.")
    if choice_block:
        trace_lines.append("")
        trace_lines.append("Options given:")
        trace_lines.append(choice_block.strip())
    trace_lines.append("")
    trace_lines.append(f"Answer: {answer}")

    return answer, "\n".join(trace_lines)


# def ollama_generate_once(
#     prompt: str,
#     model: str,
#     timeout_s: int = 180,
# ) -> str:
#     """
#     Calls local Ollama (free) via CLI:
#       ollama run <model>
#     with prompt via stdin.
#
#     Requires: `ollama` installed + model pulled.
#     """
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

def ollama_generate_once(
    prompt: str,
    model: str,
    timeout_s: int = 180,
) -> str:
    """
    Calls local Ollama with hard caps so generation can't ramble forever.
    """
    cmd = [
        "ollama", "run", model,
        "-o", "temperature=0.7",
        "-o", "num_predict=256",   # cap output tokens
        "-o", "top_p=0.9",
        "-o", "stop=Answer:",
        "-o", "stop=ANSWER:",
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


def build_prompt(
    ex: LegalExample,
    style: str,
    template: Optional[str],
) -> str:
    """
    Prompt asks for MCQ answer + step-by-step reasoning.
    We do NOT force JSON (small local models often fail strict JSON).
    """
    choice_block = extract_choice_block(ex.context)

    if template:
        return template.format(
            question=ex.question,
            context=choice_block,
            style=style,
            labels=", ".join(CHOICE_LABELS),
        )

    # Fallback built-in template
    return f"""Choose the best answer among [A,B,C,D].

    QUESTION:
    {ex.question}

    {choice_block}

    Write:
    Reasoning: (max 5 short bullet points)
    Answer: <A|B|C|D>
    """


def parse_ollama_output_to_trace(out: str) -> Tuple[str, str]:
    answer = parse_mcq_answer(out)
    if answer is None:
        answer = "A"  # last-resort; caller may override
    # Keep full output as trace text
    return answer, out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate K legal reasoning traces per LEXam question.")
    p.add_argument("--input", type=Path, required=True, help="JSONL of LegalExample (calib or eval).")
    p.add_argument("--output", type=Path, required=True, help="Where to write candidates JSONL.")

    p.add_argument("--k", type=int, default=8, help="Candidates per question.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--backend",
        choices=["heuristic", "ollama"],
        default="heuristic",
        help="Free backends: heuristic baseline or local Ollama LLM.",
    )
    p.add_argument("--ollama_model", type=str, default="llama3.2", help="Ollama model name (if backend=ollama).")
    p.add_argument("--timeout_s", type=int, default=180)

    p.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap for quick tests.",
    )

    args = p.parse_args()
    rng = random.Random(args.seed)

    rows = read_jsonl(args.input)
    if args.max_questions is not None:
        rows = rows[: args.max_questions]

    # start fresh
    if args.output.exists():
        args.output.unlink()

    template = load_prompt_template(Path("legal/prompts/generate_trace.txt"))

    styles = [
        "IRAC (Issue-Rule-Application-Conclusion)",
        "Elimination-first (reject wrong options explicitly)",
        "Textualist reading (focus on wording/negations)",
        "Policy/teleology (consider purpose)",
    ]

    total_written = 0
    t0 = time.time()

    for i, rec in enumerate(rows, start=1):
        ex = LegalExample(**rec)
        out_recs: List[Dict[str, Any]] = []

        for j in range(args.k):
            style = styles[j % len(styles)]
            trace_id = f"{j:03d}"

            if args.backend == "heuristic":
                answer, trace_text = heuristic_generate(ex, j, rng, style)
            else:
                prompt = build_prompt(ex, style, template)
                try:
                    raw = ollama_generate_once(prompt, model=args.ollama_model, timeout_s=args.timeout_s)
                    answer, trace_text = parse_ollama_output_to_trace(raw)
                    # If parsing failed, diversify by random fallback
                    if answer not in CHOICE_LABELS:
                        answer = rng.choice(CHOICE_LABELS)
                except Exception as e:
                    # Fallback to heuristic if Ollama fails for this sample
                    answer, trace_text = heuristic_generate(ex, j, rng, style)
                    trace_text = f"[OLLAMA_ERROR fallback]\n{e}\n\n{trace_text}"

            cand = LegalTraceCandidate(
                question_id=ex.question_id,
                trace_id=trace_id,
                answer=answer,
                trace_text=trace_text,
                generator={
                    "backend": args.backend,
                    "ollama_model": args.ollama_model if args.backend == "ollama" else None,
                    "style": style,
                    "seed": args.seed,
                    "k": args.k,
                },
            )
            out_recs.append(cand.model_dump())

        write_jsonl(args.output, out_recs)
        total_written += len(out_recs)

        if i % 50 == 0:
            dt = time.time() - t0
            print(f"[{i}/{len(rows)}] wrote {total_written} candidates ({total_written/dt:.1f} cand/s)")

    dt = time.time() - t0
    print(f"Done. Wrote {total_written} candidates to {args.output}")
    print(f"Throughput: {total_written/dt:.1f} cand/s (backend={args.backend})")


if __name__ == "__main__":
    main()
