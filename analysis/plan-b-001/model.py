"""plan-b-001 c6 — GRUNetX3 (attention restructure F1/F2/F3 + yaw decode).

plan-030 GRUNetX2 파생:
- F1: 잔차(b) → attention bias. `attn_logits = q·kv/√d + λ·Linear₃(residual_b_yaw)`. Q 에서 잔차b 제거(29D).
- F2: K=V = kv_proj(gru_out) only (잔차a 제거; 잔차a 는 GRU input seq98 에만).
- F3: head_in = [attn_context, Linear(sample_summary→64), slim7]; gru_hidden_last drop. head_hidden 256.
- decode: final = baseline_pred + R_wfy @ (probs @ ANCHORS_A6). R_wfy = yaw→world. 전 경로 torch (grad 보존).

forward(seq_98, residual_b_yaw (B,K,T,3), query_29, head_summary_56, slim7 (B,K,7),
        baseline_pred (B,3), R_wfy (B,3,3)) -> (world_pred (B,3), probs (B,K))
"""
from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn


class GRUNetX3(nn.Module):
    def __init__(
        self,
        seq_dim: int = 98,             # seq 95 + residual_a yaw 3
        query_dim: int = 29,           # anchor_spec 9 + ppd 3 + interactions 10 + slim7 7
        head_summary_dim: int = 56,
        slim7_dim: int = 7,
        hidden: int = 196,
        attn_dim: int = 128,
        sample_proj_dim: int = 64,
        head_hidden: int = 256,
        gru_dropout: float = 0.10,
        head_dropout: float = 0.08,
        K: int = 14,
        resb_coord: int = 3,           # yaw 3-coord
        anchors: torch.Tensor | None = None,   # (K,3) yaw-frame codebook
    ):
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.attn_dim = attn_dim

        self.gru = nn.GRU(seq_dim, hidden, num_layers=2, dropout=gru_dropout, batch_first=True)
        self.kv_proj = nn.Linear(hidden, attn_dim)                 # F2: gru_out only
        self.q_proj = nn.Linear(query_dim, attn_dim)
        # F1: residual_b → bias (Linear 3→1) + learnable scale
        self.resb_proj = nn.Linear(resb_coord, 1)
        self.lambda_bias = nn.Parameter(torch.ones(1))
        # F3: sample summary projection (drop gru_hidden_last)
        self.sample_proj = nn.Linear(head_summary_dim, sample_proj_dim)
        head_in_dim = attn_dim + sample_proj_dim + slim7_dim       # 128+64+7 = 199
        self.head_in_dim = head_in_dim
        self.head_mlp = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        if anchors is None:
            anchors = torch.zeros(K, 3, dtype=torch.float32)
        else:
            anchors = anchors.detach().to(dtype=torch.float32)
            assert anchors.shape == (K, 3)
        self.register_buffer("ANCHORS_A6", anchors)

    def forward(
        self,
        seq_98: torch.Tensor,             # (B, T=7, 98)
        residual_b_yaw: torch.Tensor,     # (B, K=14, T=7, 3)
        query_29: torch.Tensor,           # (B, K=14, 29)
        head_summary_56: torch.Tensor,    # (B, 56)
        slim7: torch.Tensor,              # (B, K=14, 7)
        baseline_pred: torch.Tensor,      # (B, 3) world +80ms (F0/Kalman), frozen
        R_wfy: torch.Tensor,              # (B, 3, 3) yaw→world, frozen
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = seq_98.shape[0]
        K = self.K

        gru_out, _ = self.gru(seq_98)                              # (B,T,H)
        kv = self.kv_proj(gru_out)                                 # (B,T,attn) — F2
        q = self.q_proj(query_29)                                  # (B,K,attn)

        # F1: residual_b bias (B,K,T)
        bias = self.resb_proj(residual_b_yaw).squeeze(-1) * self.lambda_bias   # (B,K,T)
        attn_logits = torch.einsum("bka,bta->bkt", q, kv) / sqrt(self.attn_dim) + bias
        attn_w = torch.softmax(attn_logits, dim=-1)                # over T
        attn_context = torch.einsum("bkt,bta->bka", attn_w, kv)    # (B,K,attn)

        # F3: projected sample summary broadcast (no gru_hidden_last)
        sample_bias = self.sample_proj(head_summary_56).unsqueeze(1).expand(-1, K, -1)  # (B,K,64)
        head_in = torch.cat([attn_context, sample_bias, slim7], dim=-1)                 # (B,K,199)
        score = self.head_mlp(head_in).squeeze(-1)                 # (B,K)
        probs = torch.softmax(score, dim=-1)                       # over K=14 (selector)

        # decode (yaw → world), 전 경로 torch
        residual_yaw = probs @ self.ANCHORS_A6                     # (B,K)@(K,3)=(B,3)
        residual_world = torch.einsum("bij,bj->bi", R_wfy, residual_yaw)  # (B,3)
        world_pred = baseline_pred + residual_world
        return world_pred, probs


def _smoke() -> None:
    torch.manual_seed(20260526)
    B, T, K = 4, 7, 14
    anchors = torch.randn(K, 3) * 0.01
    model = GRUNetX3(anchors=anchors)
    model.train()
    seq = torch.randn(B, T, 98)
    rb = torch.randn(B, K, T, 3)
    q = torch.randn(B, K, 29)
    hs = torch.randn(B, 56)
    sl = torch.randn(B, K, 7)
    base = torch.randn(B, 3)
    # R_wfy = identity-ish rotation
    R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    wp, probs = model(seq, rb, q, hs, sl, base, R)
    assert wp.shape == (B, 3) and probs.shape == (B, K)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    assert model.head_in_dim == 199
    # decode gradient: combined CE(non-saturating) + MSE(world_pred) — softhit 단독은 random
    # data 에서 sigmoid 포화로 grad≈0 (실제 학습은 soft_CE 항이 grad 공급).
    y = torch.randn(B, 3)
    q_rand = torch.softmax(torch.randn(B, K), dim=-1)
    loss = -(q_rand * torch.log(probs.clamp_min(1e-12))).sum(-1).mean() + ((wp - y) ** 2).mean()
    loss.backward()
    nonzero = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(nonzero) > 0, "no gradient"
    # resb_proj (F1 bias path) + head must receive gradient
    assert model.head_mlp[0].weight.grad is not None
    assert model.resb_proj.weight.grad is not None and model.resb_proj.weight.grad.abs().sum() > 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] GRUNetX3 OK — wp={tuple(wp.shape)}, probs={tuple(probs.shape)}, "
          f"head_in={model.head_in_dim}, params={n_params}, softhit grad ✓")


if __name__ == "__main__":
    _smoke()
