# # critic/run_trm_score.py
#
# """
# Score proof candidates with a trained TRM critic and write a new scored JSONL.
#
# Input JSONL should contain at least:
#   - example_id
#   - candidate_id
#   - proof_text
# Optionally:
#   - valid (bool) for later coverage evaluation in build_sets.py
#
# Output JSONL will add:
#   - trm_logit         (float)
#   - trm_p_accept      (float in [0,1])
#   - trm_steps_used    (int)
#   - score             (float)  <-- nonconformity = 1 - trm_p_accept (lower is better)
#
# If the input already has 'score', we keep it as 'score_handcrafted'.
# """
#
# import argparse
# import json
# import re
# from pathlib import Path
# from typing import Any, Dict, List, Tuple
#
# import torch
#
# from critic.trm_model import TRMCritic, TRMConfig
#
# TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
#
#
# def tokenize(text: str) -> List[str]:
#     return TOKEN_RE.findall(text.lower().strip())
#
#
# def encode_one(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], List[int]]:
#     toks = tokenize(text)
#     ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:max_len]
#     if len(ids) == 0:
#         # Avoid empty sequences (embedding/GRU will choke on length 0)
#         ids = [vocab["<unk>"]]
#     mask = [1] * len(ids)
#     return ids, mask
#
#
# def batch_encode(texts: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     encoded = [encode_one(t, vocab, max_len) for t in texts]
#     ids_list = [x[0] for x in encoded]
#     mask_list = [x[1] for x in encoded]
#
#     L = min(max(len(x) for x in ids_list), max_len)
#     L = max(L, 1)
#
#     def pad(seq: List[int], pad_id: int) -> List[int]:
#         seq = seq[:L]
#         return seq + [pad_id] * (L - len(seq))
#
#     input_ids = torch.tensor([pad(x, vocab["<pad>"]) for x in ids_list], dtype=torch.long)
#     attention_mask = torch.tensor([pad(x, 0) for x in mask_list], dtype=torch.float)
#     return input_ids, attention_mask
#
#
# def load_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[TRMCritic, Dict[str, int], Dict[str, Any]]:
#     ckpt = torch.load(ckpt_path, map_location=device)
#
#     config_dict = ckpt["config"]
#     vocab = ckpt["vocab"]
#
#     config = TRMConfig(**config_dict)
#     model = TRMCritic(config)
#     model.load_state_dict(ckpt["state_dict"])
#     model.to(device)
#     model.eval()
#
#     meta = {
#         "best_val_loss": ckpt.get("best_val_loss", None),
#         "config": config_dict,
#     }
#     return model, vocab, meta
#
#
# def iter_jsonl(path: Path):
#     with path.open() as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             yield json.loads(line)
#
#
# def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("a") as f:
#         for rec in records:
#             f.write(json.dumps(rec) + "\n")
#
#
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Score proofs with TRM critic.")
#     p.add_argument("--ckpt", type=Path, default=Path("outputs/trm_critic/model.pt"))
#     p.add_argument("--input", type=Path, default=Path("outputs/eb_task1_dev_calib_verified.jsonl"))
#     p.add_argument("--output", type=Path, default=Path("outputs/eb_task1_dev_calib_trm_scored.jsonl"))
#
#     p.add_argument("--max_len", type=int, default=128)
#     p.add_argument("--batch_size", type=int, default=64)
#
#     # Halting behaviour
#     p.add_argument("--max_steps", type=int, default=None, help="Max refinement steps (default: from checkpoint config)")
#     p.add_argument("--halt_threshold", type=float, default=0.5)
#
#     return p.parse_args()
#
#
# def main() -> None:
#     args = parse_args()
#
#     if not args.ckpt.exists():
#         raise SystemExit(f"Checkpoint not found: {args.ckpt}")
#     if not args.input.exists():
#         raise SystemExit(f"Input not found: {args.input}")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model, vocab, meta = load_checkpoint(args.ckpt, device=device)
#
#     # default max_steps from model config if not provided
#     max_steps = args.max_steps if args.max_steps is not None else model.config.num_steps
#
#     # overwrite output (start fresh)
#     if args.output.exists():
#         args.output.unlink()
#
#     n = 0
#     sum_p = 0.0
#     min_score = float("inf")
#     max_score = float("-inf")
#
#     batch_recs: List[Dict[str, Any]] = []
#     batch_texts: List[str] = []
#
#     for rec in iter_jsonl(args.input):
#         batch_recs.append(rec)
#         batch_texts.append(rec.get("proof_text", ""))
#
#         if len(batch_recs) >= args.batch_size:
#             n, sum_p, min_score, max_score = score_and_write_batch(
#                 model=model,
#                 vocab=vocab,
#                 max_len=args.max_len,
#                 device=device,
#                 max_steps=max_steps,
#                 halt_threshold=args.halt_threshold,
#                 output_path=args.output,
#                 batch_recs=batch_recs,
#                 batch_texts=batch_texts,
#                 n=n,
#                 sum_p=sum_p,
#                 min_score=min_score,
#                 max_score=max_score,
#             )
#             batch_recs, batch_texts = [], []
#
#     # flush last batch
#     if batch_recs:
#         n, sum_p, min_score, max_score = score_and_write_batch(
#             model=model,
#             vocab=vocab,
#             max_len=args.max_len,
#             device=device,
#             max_steps=max_steps,
#             halt_threshold=args.halt_threshold,
#             output_path=args.output,
#             batch_recs=batch_recs,
#             batch_texts=batch_texts,
#             n=n,
#             sum_p=sum_p,
#             min_score=min_score,
#             max_score=max_score,
#         )
#
#     avg_p = sum_p / max(1, n)
#     print(f"Scored {n} candidates with TRM critic.")
#     print(f"Avg trm_p_accept: {avg_p:.4f}")
#     print(f"Nonconformity score range: [{min_score:.6f}, {max_score:.6f}]")
#     print(f"Wrote: {args.output}")
#     print(f"Checkpoint meta: best_val_loss={meta.get('best_val_loss')}  num_steps={model.config.num_steps}")
#
#
# @torch.no_grad()
# def score_and_write_batch(
#     model: TRMCritic,
#     vocab: Dict[str, int],
#     max_len: int,
#     device: torch.device,
#     max_steps: int,
#     halt_threshold: float,
#     output_path: Path,
#     batch_recs: List[Dict[str, Any]],
#     batch_texts: List[str],
#     n: int,
#     sum_p: float,
#     min_score: float,
#     max_score: float,
# ) -> Tuple[int, float, float, float]:
#     input_ids, attention_mask = batch_encode(batch_texts, vocab=vocab, max_len=max_len)
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#
#     final_logits, steps_used = model.predict_with_halting(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_steps=max_steps,
#         halt_threshold=halt_threshold,
#     )  # (B,), (B,)
#
#     p_accept = torch.sigmoid(final_logits).detach().cpu().tolist()
#     logits = final_logits.detach().cpu().tolist()
#     steps_used_list = steps_used.detach().cpu().tolist()
#
#     out_batch: List[Dict[str, Any]] = []
#     for rec, logit, p, steps in zip(batch_recs, logits, p_accept, steps_used_list):
#         rec_out = dict(rec)
#
#         # preserve any previous score if present
#         if "score" in rec_out and "score_handcrafted" not in rec_out:
#             rec_out["score_handcrafted"] = rec_out["score"]
#
#         rec_out["trm_logit"] = float(logit)
#         rec_out["trm_p_accept"] = float(p)
#         rec_out["trm_steps_used"] = int(steps)
#
#         # Nonconformity: lower is better, 0 means "very confident accept"
#         rec_out["score"] = float(1.0 - p)
#
#         out_batch.append(rec_out)
#
#         n += 1
#         sum_p += float(p)
#         min_score = min(min_score, float(1.0 - p))
#         max_score = max(max_score, float(1.0 - p))
#
#     write_jsonl(output_path, out_batch)
#     return n, sum_p, min_score, max_score
#
#
# if __name__ == "__main__":
#     main()
