# # critic/trm_model.py
#
# """
# Tiny Recursive Proof Critic (TRM-style) for CAPS.
#
# This module defines:
#   - SimpleTextEncoder: token ids -> proof embedding
#   - TRMCritic: a tiny recursive network that refines a belief about
#     "oracle accepts this proof" over multiple steps, plus a halting head.
#
# We *don't* train anything here, we just define the model.
# Training will be done in critic/train_trm.py (next step).
# """
#
# from dataclasses import dataclass
# from typing import Optional, Tuple, List
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # ---------- Text encoder ----------
#
# class SimpleTextEncoder(nn.Module):
#     """
#     Very small text encoder:
#       tokens (B, L) -> embedding (B, D)
#
#     - token embedding
#     - single-layer GRU
#     - final hidden state as sentence embedding
#     """
#
#     def __init__(
#         self,
#         vocab_size: int,
#         embed_dim: int = 128,
#         hidden_dim: int = 128,
#     ) -> None:
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.gru = nn.GRU(
#             input_size=embed_dim,
#             hidden_size=hidden_dim,
#             batch_first=True,
#         )
#         self.hidden_dim = hidden_dim
#
#     def forward(
#         self,
#         input_ids: torch.Tensor,        # (B, L)
#         attention_mask: Optional[torch.Tensor] = None,  # (B, L) 1 = keep, 0 = pad
#     ) -> torch.Tensor:
#         """
#         Returns:
#           proof_repr: (B, hidden_dim)
#         """
#         # input_ids: (B, L)
#         x = self.embedding(input_ids)  # (B, L, E)
#
#         if attention_mask is not None:
#             # GRU doesn't use mask directly; we can zero out padded positions.
#             # attention_mask: (B, L) in {0,1}
#             mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
#             x = x * mask
#
#         # GRU: output (B, L, H), h_n (1, B, H)
#         _, h_n = self.gru(x)
#         # Take final hidden state
#         proof_repr = h_n.squeeze(0)  # (B, H)
#         return proof_repr
#
#
# # ---------- TRM-style critic ----------
#
# @dataclass
# class TRMConfig:
#     vocab_size: int
#     embed_dim: int = 128
#     hidden_dim: int = 128
#     latent_dim: int = 128
#     num_steps: int = 3   # default number of refinement steps
#     dropout: float = 0.1
#
#
# class TRMCritic(nn.Module):
#     """
#     Tiny recursive proof critic.
#
#     High-level:
#
#       - Encode proof text into a vector h (B, H).
#       - Initialize latent z_0 = tanh(W_init * h).
#       - Initialize answer logit s_0 = 0.
#       - For t = 0..T-1:
#           concat_t = [h, z_t, s_t] -> core MLP -> [z_{t+1}, s_{t+1}]
#           halt_logit_t = halt_head(z_{t+1})
#       - Return all step-wise answer logits s_t and halting logits q_t.
#
#     Training (later):
#       - Deep supervision: each s_t is trained against oracle label y.
#       - Halting head q_t trained to mimic when we "should stop".
#     """
#
#     def __init__(self, config: TRMConfig) -> None:
#         super().__init__()
#         self.config = config
#
#         # Text encoder
#         self.encoder = SimpleTextEncoder(
#             vocab_size=config.vocab_size,
#             embed_dim=config.embed_dim,
#             hidden_dim=config.hidden_dim,
#         )
#
#         # Project encoded proof -> initial latent z_0
#         self.init_proj = nn.Linear(config.hidden_dim, config.latent_dim)
#
#         # Core recursive update MLP
#         # Input: [h, z_t, s_t]   (dim = H + latent_dim + 1)
#         # Output: [z_{t+1}, s_{t+1}] (dim = latent_dim + 1)
#         core_in_dim = config.hidden_dim + config.latent_dim + 1
#         core_out_dim = config.latent_dim + 1
#
#         self.core_mlp = nn.Sequential(
#             nn.Linear(core_in_dim, 2 * core_in_dim),
#             nn.ReLU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(2 * core_in_dim, core_out_dim),
#         )
#
#         # Halting head q_t from z_t
#         self.halt_head = nn.Linear(config.latent_dim, 1)
#
#     # ---------------------------------------------------------
#     # Forward API
#     # ---------------------------------------------------------
#
#     def forward(
#         self,
#         input_ids: torch.Tensor,          # (B, L)
#         attention_mask: Optional[torch.Tensor] = None,  # (B, L)
#         num_steps: Optional[int] = None,
#         return_all: bool = True,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Run the critic for a fixed number of refinement steps.
#
#         Args:
#           input_ids: (B, L) token ids of proof text
#           attention_mask: (B, L) binary mask (1 = real token, 0 = pad)
#           num_steps: how many refinement steps to run.
#                      If None, uses config.num_steps.
#           return_all: if True, return all step logits; otherwise only last.
#
#         Returns:
#           answer_logits: (B, T) if return_all else (B,)
#           halt_logits:   (B, T) if return_all else (B,)
#         """
#         if num_steps is None:
#             num_steps = self.config.num_steps
#         T = num_steps
#
#         batch_size = input_ids.size(0)
#
#         # 1) Encode proof to static representation h
#         h = self.encoder(input_ids, attention_mask)  # (B, H)
#
#         # 2) Initialize latent z_0 and answer logit s_0
#         z_t = torch.tanh(self.init_proj(h))          # (B, latent_dim)
#         s_t = torch.zeros(batch_size, 1, device=input_ids.device)  # (B, 1)
#
#         # 3) Recursive refinement
#         answer_logits_steps: List[torch.Tensor] = []
#         halt_logits_steps: List[torch.Tensor] = []
#
#         for _ in range(T):
#             # concat: [h, z_t, s_t]
#             concat_t = torch.cat([h, z_t, s_t], dim=-1)  # (B, H + latent + 1)
#
#             updated = self.core_mlp(concat_t)  # (B, latent_dim + 1)
#
#             # Split back into z_{t+1} and s_{t+1}
#             z_t = torch.tanh(updated[:, : self.config.latent_dim])  # (B, latent_dim)
#             s_t = updated[:, self.config.latent_dim:].view(-1, 1)  # (B, 1)
#
#             # Halting logit from new latent
#             q_t = self.halt_head(z_t)  # (B, 1)
#
#             answer_logits_steps.append(s_t.squeeze(-1))  # (B,)
#             halt_logits_steps.append(q_t.squeeze(-1))  # (B,)
#
#             # Stack into (B, T)
#             answer_logits = torch.stack(answer_logits_steps, dim=-1)  # (B, T)
#             halt_logits = torch.stack(halt_logits_steps, dim=-1)      # (B, T)
#
#         if not return_all:
#             # Just return final step
#             return answer_logits[:, -1], halt_logits[:, -1]
#
#         return answer_logits, halt_logits
#
#     # ---------------------------------------------------------
#     # Convenience method for inference with halting
#     # ---------------------------------------------------------
#
#     @torch.no_grad()
#     def predict_with_halting(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         max_steps: Optional[int] = None,
#         halt_threshold: float = 0.5,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Run the critic step-by-step and stop early when halting probability
#         exceeds 'halt_threshold', or when max_steps is reached.
#
#         Args:
#           input_ids: (B, L)
#           attention_mask: (B, L)
#           max_steps: maximum number of refinement steps
#                      (defaults to config.num_steps)
#           halt_threshold: threshold on sigmoid(q_t) to halt.
#
#         Returns:
#           final_answer_logits: (B,)
#           steps_used:          (B,) int tensor with number of steps each
#                                example actually ran.
#         """
#         if max_steps is None:
#             max_steps = self.config.num_steps
#
#         batch_size = input_ids.size(0)
#         device = input_ids.device
#
#         # Encode once
#         h = self.encoder(input_ids, attention_mask)  # (B, H)
#         z_t = torch.tanh(self.init_proj(h))          # (B, latent_dim)
#         s_t = torch.zeros(batch_size, 1, device=device)  # (B, 1)
#
#         finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
#         steps_used = torch.zeros(batch_size, dtype=torch.long, device=device)
#         final_logits = torch.zeros(batch_size, device=device)
#
#         for step in range(1, max_steps + 1):
#             concat_t = torch.cat([h, z_t, s_t], dim=-1)
#             updated = self.core_mlp(concat_t)
#
#             z_t = torch.tanh(updated[:, : self.config.latent_dim])  # (B, latent_dim)
#             s_t = updated[:, self.config.latent_dim:].view(-1, 1)  # (B, 1)
#
#             q_t = self.halt_head(z_t)  # (B, 1)
#             halt_prob = torch.sigmoid(q_t.squeeze(-1))  # (B,)
#
#             # Decide which examples halt now
#             newly_finished = (~finished) & (halt_prob >= halt_threshold)
#
#             # For newly finished examples, record answer + steps
#             final_logits[newly_finished] = s_t.squeeze(-1)[newly_finished]
#             steps_used[newly_finished] = step
#             finished = finished | newly_finished
#
#             if finished.all():
#                 break
#
#         # For any that never halted, just use last s_t
#         final_logits[~finished] = s_t.squeeze(-1)[~finished]
#         steps_used[~finished] = max_steps
#
#         return final_logits, steps_used
#
#
# # ---------- Tiny sanity check ----------
#
# if __name__ == "__main__":
#     # Fake vocab size and random token ids
#     vocab_size = 100
#     config = TRMConfig(vocab_size=vocab_size, num_steps=3)
#
#     model = TRMCritic(config)
#
#     batch_size = 4
#     seq_len = 10
#     input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
#     attention_mask = torch.ones_like(input_ids)
#
#     answer_logits, halt_logits = model(
#         input_ids, attention_mask, num_steps=3, return_all=True
#     )
#
#     print("answer_logits shape:", answer_logits.shape)  # (B, T)
#     print("halt_logits shape  :", halt_logits.shape)    # (B, T)
#
#     final_logits, steps_used = model.predict_with_halting(
#         input_ids, attention_mask, max_steps=3
#     )
#     print("final_logits shape :", final_logits.shape)   # (B,)
#     print("steps_used         :", steps_used)
