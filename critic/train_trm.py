# # critic/train_trm.py
#
# """
# Train the Tiny Recursive Proof Critic (TRM-style) on oracle labels.
#
# Input: verified candidates JSONL, e.g.
#   outputs/eb_task1_dev_calib_verified.jsonl
#
# Expected fields per line:
#   - example_id
#   - candidate_id
#   - proof_text
#   - valid (bool)  # oracle label for now (later: ASP/Lean)
#
# This script:
#   - builds a small vocab from proof_text
#   - tokenizes via a simple regex tokenizer
#   - trains TRMCritic with deep supervision across steps
#   - trains a halting head with a simple ACT-lite target:
#       * if label==1: halt target is 1 at all steps (halt early)
#       * if label==0: halt target is 0 until last step, 1 at last step (eventually halt)
#   - saves model + vocab + config into out_dir
# """
#
# import argparse
# import json
# import math
# import random
# import re
# from collections import Counter, defaultdict
# from dataclasses import asdict
# from pathlib import Path
# from typing import Any, Dict, List, Tuple
#
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset, Subset
#
# from critic.dataset import ProofCriticDataset
# from critic.trm_model import TRMCritic, TRMConfig
#
#
# TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
#
#
# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# # ----------------------------
# # Tokenizer / Vocab
# # ----------------------------
#
# def tokenize(text: str) -> List[str]:
#     # Lowercase + regex tokenization (words + punctuation)
#     text = text.lower().strip()
#     return TOKEN_RE.findall(text)
#
#
# def build_vocab(
#     dataset: ProofCriticDataset,
#     vocab_size: int = 5000,
#     min_freq: int = 1,
# ) -> Dict[str, int]:
#     """
#     Build a vocab dict token->id.
#     Reserve:
#       0 = <pad>
#       1 = <unk>
#     """
#     counter = Counter()
#     for ex in dataset.examples:
#         toks = tokenize(ex.proof_text)
#         counter.update(toks)
#
#     # Apply min_freq and cap vocab size
#     items = [(tok, c) for tok, c in counter.items() if c >= min_freq]
#     items.sort(key=lambda x: x[1], reverse=True)
#
#     # Reserve 2 spots for pad/unk
#     max_tokens = max(0, vocab_size - 2)
#     items = items[:max_tokens]
#
#     vocab = {"<pad>": 0, "<unk>": 1}
#     for i, (tok, _) in enumerate(items, start=2):
#         vocab[tok] = i
#
#     return vocab
#
#
# def encode(
#     text: str,
#     vocab: Dict[str, int],
#     max_len: int = 128,
# ) -> Tuple[List[int], List[int]]:
#     """
#     Returns:
#       input_ids: List[int] length <= max_len
#       attention_mask: List[int] same length (1s)
#     """
#     toks = tokenize(text)
#     ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:max_len]
#     mask = [1] * len(ids)
#     return ids, mask
#
#
# # ----------------------------
# # Collate
# # ----------------------------
#
# def collate_batch(
#     batch: List[Dict[str, Any]],
#     vocab: Dict[str, int],
#     max_len: int,
#     num_steps: int,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Convert a list of dataset items into padded tensors.
#
#     Outputs:
#       input_ids: (B, L)
#       attention_mask: (B, L)
#       labels: (B,) float
#       halt_targets: (B, T) float
#     """
#     input_ids_list: List[List[int]] = []
#     mask_list: List[List[int]] = []
#     labels: List[int] = []
#
#     for item in batch:
#         ids, m = encode(item["proof_text"], vocab=vocab, max_len=max_len)
#         input_ids_list.append(ids)
#         mask_list.append(m)
#         labels.append(int(item["label"]))
#
#     # pad to max length in batch (cap at max_len)
#     L = min(max(len(x) for x in input_ids_list), max_len)
#
#     def pad(seq: List[int], pad_id: int) -> List[int]:
#         seq = seq[:L]
#         return seq + [pad_id] * (L - len(seq))
#
#     input_ids = torch.tensor([pad(x, vocab["<pad>"]) for x in input_ids_list], dtype=torch.long)
#     attention_mask = torch.tensor([pad(x, 0) for x in mask_list], dtype=torch.float)
#
#     y = torch.tensor(labels, dtype=torch.float)  # (B,)
#
#     # Halting targets (ACT-lite):
#     #  - positives: [1,1,...,1]
#     #  - negatives: [0,0,...,1]  (only last step is 1)
#     halt_targets = torch.zeros((len(batch), num_steps), dtype=torch.float)
#     for i, lab in enumerate(labels):
#         if lab == 1:
#             halt_targets[i, :] = 1.0
#         else:
#             halt_targets[i, :-1] = 0.0
#             halt_targets[i, -1] = 1.0
#
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": y,
#         "halt_targets": halt_targets,
#     }
#
#
# # ----------------------------
# # Splitting (by example_id)
# # ----------------------------
#
# def split_by_example_id(
#     dataset: ProofCriticDataset,
#     val_frac: float = 0.1,
#     seed: int = 0,
# ) -> Tuple[List[int], List[int]]:
#     """
#     Split candidate indices by grouping on example_id (avoid leakage).
#     """
#     rng = random.Random(seed)
#     by_eid: Dict[str, List[int]] = defaultdict(list)
#     for idx, ex in enumerate(dataset.examples):
#         by_eid[ex.example_id].append(idx)
#
#     eids = list(by_eid.keys())
#     rng.shuffle(eids)
#
#     n_val = max(1, int(len(eids) * val_frac))
#     val_eids = set(eids[:n_val])
#
#     train_idx: List[int] = []
#     val_idx: List[int] = []
#     for eid, idxs in by_eid.items():
#         if eid in val_eids:
#             val_idx.extend(idxs)
#         else:
#             train_idx.extend(idxs)
#
#     return train_idx, val_idx
#
#
# # ----------------------------
# # Training / Eval
# # ----------------------------
#
# def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     return F.binary_cross_entropy_with_logits(logits, targets)
#
#
# def train_one_epoch(
#     model: TRMCritic,
#     loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     lambda_halt: float,
# ) -> Dict[str, float]:
#     model.train()
#     loss_sum = 0.0
#     n_batches = 0
#
#     acc_sum = 0.0
#     n_items = 0
#
#     bce = nn.BCEWithLogitsLoss()
#
#     for batch in loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)                 # (B,)
#         halt_targets = batch["halt_targets"].to(device)     # (B,T)
#
#         optimizer.zero_grad()
#
#         answer_logits, halt_logits = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_all=True,
#         )  # (B,T), (B,T)
#
#         # Deep supervision: average BCE over all steps
#         # answer_logits[:, t] vs labels
#         T = answer_logits.size(1)
#         labels_T = labels.unsqueeze(1).expand(-1, T)  # (B,T)
#         loss_answer = bce(answer_logits, labels_T)
#
#         # Halt loss
#         loss_halt = bce(halt_logits, halt_targets)
#
#         loss = loss_answer + lambda_halt * loss_halt
#         loss.backward()
#         optimizer.step()
#
#         loss_sum += float(loss.item())
#         n_batches += 1
#
#         # Accuracy on final step only
#         final_logits = answer_logits[:, -1]
#         preds = (torch.sigmoid(final_logits) >= 0.5).float()
#         acc_sum += float((preds == labels).sum().item())
#         n_items += labels.size(0)
#
#     return {
#         "loss": loss_sum / max(1, n_batches),
#         "acc_final": acc_sum / max(1, n_items),
#     }
#
#
# @torch.no_grad()
# def eval_model(
#     model: TRMCritic,
#     loader: DataLoader,
#     device: torch.device,
#     lambda_halt: float,
# ) -> Dict[str, float]:
#     model.eval()
#     loss_sum = 0.0
#     n_batches = 0
#
#     acc_sum = 0.0
#     n_items = 0
#
#     bce = nn.BCEWithLogitsLoss()
#
#     for batch in loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#         halt_targets = batch["halt_targets"].to(device)
#
#         answer_logits, halt_logits = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_all=True,
#         )
#
#         T = answer_logits.size(1)
#         labels_T = labels.unsqueeze(1).expand(-1, T)
#
#         loss_answer = bce(answer_logits, labels_T)
#         loss_halt = bce(halt_logits, halt_targets)
#         loss = loss_answer + lambda_halt * loss_halt
#
#         loss_sum += float(loss.item())
#         n_batches += 1
#
#         final_logits = answer_logits[:, -1]
#         preds = (torch.sigmoid(final_logits) >= 0.5).float()
#         acc_sum += float((preds == labels).sum().item())
#         n_items += labels.size(0)
#
#     return {
#         "loss": loss_sum / max(1, n_batches),
#         "acc_final": acc_sum / max(1, n_items),
#     }
#
#
# # ----------------------------
# # Main
# # ----------------------------
#
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Train TRM-style proof critic.")
#     p.add_argument(
#         "--input",
#         type=Path,
#         default=Path("outputs/eb_task1_dev_calib_verified.jsonl"),
#         help="Verified candidates JSONL",
#     )
#     p.add_argument(
#         "--out_dir",
#         type=Path,
#         default=Path("outputs/trm_critic"),
#         help="Where to save model/vocab/config",
#     )
#
#     # Data
#     p.add_argument("--max_len", type=int, default=128)
#     p.add_argument("--vocab_size", type=int, default=5000)
#     p.add_argument("--min_freq", type=int, default=1)
#     p.add_argument("--val_frac", type=float, default=0.1)
#
#     # Model
#     p.add_argument("--embed_dim", type=int, default=128)
#     p.add_argument("--hidden_dim", type=int, default=128)
#     p.add_argument("--latent_dim", type=int, default=128)
#     p.add_argument("--num_steps", type=int, default=3)
#     p.add_argument("--dropout", type=float, default=0.1)
#
#     # Train
#     p.add_argument("--epochs", type=int, default=5)
#     p.add_argument("--batch_size", type=int, default=32)
#     p.add_argument("--lr", type=float, default=2e-3)
#     p.add_argument("--weight_decay", type=float, default=1e-2)
#     p.add_argument("--lambda_halt", type=float, default=0.1)
#     p.add_argument("--seed", type=int, default=0)
#
#     return p.parse_args()
#
#
# def main() -> None:
#     args = parse_args()
#     set_seed(args.seed)
#
#     if not args.input.exists():
#         raise SystemExit(f"Input not found: {args.input}")
#
#     # Load dataset
#     ds = ProofCriticDataset(args.input)
#
#     # Build vocab
#     vocab = build_vocab(ds, vocab_size=args.vocab_size, min_freq=args.min_freq)
#     print(f"Built vocab size: {len(vocab)}")
#
#     # Split by example_id
#     train_idx, val_idx = split_by_example_id(ds, val_frac=args.val_frac, seed=args.seed)
#     train_ds: Dataset = Subset(ds, train_idx)
#     val_ds: Dataset = Subset(ds, val_idx)
#
#     print(f"Train examples: {len(train_ds)}   Val examples: {len(val_ds)}")
#
#     # Dataloaders
#     def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         return collate_batch(batch, vocab=vocab, max_len=args.max_len, num_steps=args.num_steps)
#
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
#
#     # Model
#     config = TRMConfig(
#         vocab_size=len(vocab),
#         embed_dim=args.embed_dim,
#         hidden_dim=args.hidden_dim,
#         latent_dim=args.latent_dim,
#         num_steps=args.num_steps,
#         dropout=args.dropout,
#     )
#     model = TRMCritic(config)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     print(f"Using device: {device}")
#
#     # Optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#     # Train loop
#     best_val_loss = float("inf")
#
#     args.out_dir.mkdir(parents=True, exist_ok=True)
#
#     # Save vocab + config early
#     (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
#     (args.out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
#
#     for epoch in range(1, args.epochs + 1):
#         tr = train_one_epoch(model, train_loader, optimizer, device, lambda_halt=args.lambda_halt)
#         va = eval_model(model, val_loader, device, lambda_halt=args.lambda_halt)
#
#         print(
#             f"Epoch {epoch:02d}/{args.epochs} | "
#             f"train loss={tr['loss']:.4f} acc={tr['acc_final']:.3f} | "
#             f"val loss={va['loss']:.4f} acc={va['acc_final']:.3f}"
#         )
#
#         # Save best
#         if va["loss"] < best_val_loss:
#             best_val_loss = va["loss"]
#             ckpt = {
#                 "state_dict": model.state_dict(),
#                 "config": asdict(config),
#                 "vocab": vocab,
#                 "best_val_loss": best_val_loss,
#             }
#             torch.save(ckpt, args.out_dir / "model.pt")
#             print(f"  ✅ Saved new best model to {args.out_dir / 'model.pt'} (val loss={best_val_loss:.4f})")
#
#     print("Done.")
#
#
# if __name__ == "__main__":
#     main()
