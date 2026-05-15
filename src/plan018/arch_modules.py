"""plan-018 4 ablation arch (A1/A2/A3/A6).

A5 Neural CDE 제외 (torchcde 미설치). A4 Vector Neurons 제외 (rotation_term EDA 모순).

All arch return (B, n_coeffs=8) coefficient. p0 + sum(coeff × basis_terms) → 3D pred.

Common contract:
- forward(encoder_input) → (B, n_coeffs)
- aux_loss attribute (None or scalar) for MoLE load-balancing 등.
- global_init: n_coeffs-d numpy array → final layer bias init (plan-007 step 3 best 8-vec).
  weight=0 으로 두면 학습 0 step 에서 plan-007 step 3 동작 reproduce.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_COEFFS_DEFAULT = 8


# ─────────────────────────────────────────────────────────────────────────
# A0 baseline (plan-007 step 4 reproduce) — for unified runner use
# ─────────────────────────────────────────────────────────────────────────

class BaselineA0(nn.Module):
    """plan-007 step 4 MLP — 13-d stats → 8 coefficient. params ~ 300."""

    def __init__(self, *, n_coeffs: int = N_COEFFS_DEFAULT, feat_dim: int = 13,
                 global_init: Optional[np.ndarray] = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
                self.mlp[-1].weight.zero_()
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────
# A1 Set Transformer (ISAB-based) — Lee et al. 2019
# ─────────────────────────────────────────────────────────────────────────

class MAB(nn.Module):
    """Multihead attention block. Q ← Q + MHA(Q, K, V); output ← Q + FF(Q)."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, k):
        h, _ = self.mha(q, k, k)
        q = self.ln1(q + h)
        q = self.ln2(q + self.ff(q))
        return q


class ISAB(nn.Module):
    """Induced Set Attention Block (Lee 2019)."""

    def __init__(self, dim: int, num_inducing: int = 16, num_heads: int = 4):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = MAB(dim, num_heads)
        self.mab2 = MAB(dim, num_heads)

    def forward(self, x):
        B = x.shape[0]
        ind = self.inducing.expand(B, -1, -1)
        h = self.mab1(ind, x)
        return self.mab2(x, h)


class SetTransformerCoeff(nn.Module):
    """A1: trajectory (B, 6, 3) → ISAB → mean pool → per-coeff query attn → (B, 8)."""

    def __init__(self, *, n_coeffs: int = N_COEFFS_DEFAULT, dim: int = 32,
                 num_inducing: int = 16, num_heads: int = 4,
                 global_init: Optional[np.ndarray] = None):
        super().__init__()
        self.input_proj = nn.Linear(3, dim)
        self.isab1 = ISAB(dim, num_inducing, num_heads)
        self.isab2 = ISAB(dim, num_inducing, num_heads)
        self.coeff_queries = nn.Parameter(torch.randn(1, n_coeffs, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.bias_param = nn.Parameter(torch.zeros(n_coeffs))
        if global_init is not None:
            with torch.no_grad():
                self.bias_param.copy_(torch.tensor(global_init, dtype=torch.float32))
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 3)
        h = self.input_proj(x)            # (B, 6, dim)
        h = self.isab1(h)
        h = self.isab2(h)
        B = h.shape[0]
        q = self.coeff_queries.expand(B, -1, -1)   # (B, n_coeffs, dim)
        out, _ = self.cross_attn(q, h, h)          # (B, n_coeffs, dim)
        coeffs = self.head(out).squeeze(-1)        # (B, n_coeffs)
        return coeffs + self.bias_param


# ─────────────────────────────────────────────────────────────────────────
# A2 Path Signature (depth-3 truncated, manual implementation — signatory 미설치)
# ─────────────────────────────────────────────────────────────────────────

def truncated_signature_d3(path: torch.Tensor, d: int = 3) -> torch.Tensor:
    """Compute truncated signature up to depth d=3.

    path: (B, T, channels). Returns flat signature (B, 1 + ch + ch² + ch³).

    For depth-3 with ch=4: dim = 1 + 4 + 16 + 64 = 85.

    Signature recurrence: ΔX_t = X_{t+1} - X_t.
    S^{(k)}_t = S^{(k)}_{t-1} + (1/k) (S^{(k-1)}_{t-1} ⊗ ΔX_t).
    Final S_T 가 sample 의 signature.
    """
    B, T, C = path.shape
    deltas = path[:, 1:] - path[:, :-1]                # (B, T-1, C)

    s1 = torch.zeros(B, C, device=path.device, dtype=path.dtype)
    s2 = torch.zeros(B, C * C, device=path.device, dtype=path.dtype)
    s3 = torch.zeros(B, C * C * C, device=path.device, dtype=path.dtype)

    for t in range(T - 1):
        dx = deltas[:, t]                              # (B, C)
        # S^(1) update: S^(1) += dx
        new_s1 = s1 + dx
        # S^(2) update: S^(2) += S^(1) ⊗ dx
        # outer = (B, C, C) — old s1 already contains contribution up to t-1
        outer1 = (s1.unsqueeze(-1) * dx.unsqueeze(-2)).reshape(B, C * C)   # (B, C*C)
        new_s2 = s2 + outer1
        # S^(3) update: S^(3) += S^(2) ⊗ dx
        outer2 = (s2.unsqueeze(-1) * dx.unsqueeze(-2)).reshape(B, C * C * C)
        new_s3 = s3 + outer2
        s1, s2, s3 = new_s1, new_s2, new_s3

    # Concatenate: [1, S^(1), S^(2), S^(3)]
    s0 = torch.ones(B, 1, device=path.device, dtype=path.dtype)
    return torch.cat([s0, s1, s2, s3], dim=-1)         # (B, 1 + C + C² + C³)


class PathSignatureCoeff(nn.Module):
    """A2: trajectory (B, 6, 3) → add time channel → depth-3 signature (85d) → MLP → (B, 8)."""

    def __init__(self, *, n_coeffs: int = N_COEFFS_DEFAULT,
                 global_init: Optional[np.ndarray] = None):
        super().__init__()
        # signature dim with time channel: 1 + 4 + 16 + 64 = 85
        self.sig_dim = 1 + 4 + 16 + 64
        self.mlp = nn.Sequential(
            nn.Linear(self.sig_dim, 64), nn.SiLU(),
            nn.Linear(64, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
                self.mlp[-1].weight.zero_()
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 3) → add time channel → (B, 6, 4)
        B, T, _ = x.shape
        t = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).unsqueeze(-1).expand(B, T, 1)
        path = torch.cat([t, x], dim=-1)               # (B, T, 4)
        sig = truncated_signature_d3(path, d=3)         # (B, 85)
        return self.mlp(sig)


# ─────────────────────────────────────────────────────────────────────────
# A3 Sparse MoLE head (Shazeer 2017) — A0 encoder reuse
# ─────────────────────────────────────────────────────────────────────────

class MoLECoeffHead(nn.Module):
    """A3: 13-d stats → gating MLP → top-k=2 experts (out of K=16) → mixture (B, 8)."""

    def __init__(self, *, n_coeffs: int = N_COEFFS_DEFAULT, feat_dim: int = 13,
                 n_experts: int = 16, top_k: int = 2, aux_weight: float = 0.01,
                 global_init: Optional[np.ndarray] = None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_weight = aux_weight
        # Encoder = A0 의 첫 layer (13 → 32 SiLU)
        self.encoder = nn.Sequential(nn.Linear(feat_dim, 32), nn.SiLU())
        # Gating
        self.gate = nn.Linear(32, n_experts)
        # Experts: K shared (32, n_coeffs) — each expert = Linear(32 → 8)
        self.experts = nn.ModuleList([nn.Linear(32, n_coeffs) for _ in range(n_experts)])
        if global_init is not None:
            with torch.no_grad():
                # Init all experts' bias to global_init
                for ex in self.experts:
                    ex.bias.copy_(torch.tensor(global_init, dtype=torch.float32))
                    ex.weight.zero_()
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                           # (B, 32)
        gate_logits = self.gate(h)                     # (B, K)
        gate_probs = F.softmax(gate_logits, dim=-1)
        # top-k
        top_vals, top_idx = torch.topk(gate_probs, self.top_k, dim=-1)   # (B, k)
        top_probs = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute all experts (B, K, n_coeffs)
        expert_outs = torch.stack([ex(h) for ex in self.experts], dim=1)
        # Gather top-k experts per sample
        B = h.shape[0]
        n_coeffs = expert_outs.shape[-1]
        gathered = torch.gather(
            expert_outs, 1,
            top_idx.unsqueeze(-1).expand(-1, -1, n_coeffs)
        )                                              # (B, k, n_coeffs)
        # Weighted sum
        out = (top_probs.unsqueeze(-1) * gathered).sum(dim=1)   # (B, n_coeffs)

        # Aux load-balancing loss (Shazeer 2017)
        importance = gate_probs.sum(dim=0)            # (K,)
        load = (gate_probs > 0).float().sum(dim=0)    # (K,)
        cv_importance = importance.std() / (importance.mean() + 1e-9)
        cv_load = load.std() / (load.mean() + 1e-9)
        self.aux_loss = self.aux_weight * (cv_importance.pow(2) + cv_load.pow(2))

        return out


# ─────────────────────────────────────────────────────────────────────────
# A6 GRU + per-coeff attention
# ─────────────────────────────────────────────────────────────────────────

class GRUAttnCoeff(nn.Module):
    """A6: trajectory (B, 6, 3) → 2-layer GRU → per-coeff cross-attn → (B, 8).

    Δ-parametrization: c = c_init + Δ. c_init = global_init in head bias.
    """

    def __init__(self, *, n_coeffs: int = N_COEFFS_DEFAULT, hidden: int = 32,
                 num_heads: int = 4, l2_prior_weight: float = 1e-4,
                 global_init: Optional[np.ndarray] = None):
        super().__init__()
        self.gru = nn.GRU(3, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.coeff_queries = nn.Parameter(torch.randn(1, n_coeffs, hidden) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.bias_param = nn.Parameter(torch.zeros(n_coeffs))
        if global_init is not None:
            with torch.no_grad():
                self.bias_param.copy_(torch.tensor(global_init, dtype=torch.float32))
        self.l2_prior_weight = l2_prior_weight
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 3)
        h, _ = self.gru(x)                            # (B, 6, hidden)
        B = h.shape[0]
        q = self.coeff_queries.expand(B, -1, -1)      # (B, n_coeffs, hidden)
        out, _ = self.cross_attn(q, h, h)             # (B, n_coeffs, hidden)
        delta = self.head(out).squeeze(-1)            # (B, n_coeffs)
        # L2 prior on delta
        self.aux_loss = self.l2_prior_weight * delta.pow(2).mean()
        return delta + self.bias_param


# ─────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────

ARCH_REGISTRY = {
    "A0": {"class": BaselineA0,             "input_type": "stats_13d"},
    "A1": {"class": SetTransformerCoeff,    "input_type": "traj_6x3"},
    "A2": {"class": PathSignatureCoeff,     "input_type": "traj_6x3"},
    "A3": {"class": MoLECoeffHead,          "input_type": "stats_13d"},
    "A6": {"class": GRUAttnCoeff,           "input_type": "traj_6x3"},
    # A4 Vector Neurons 제외 (EDA 모순, plan-018 §5.5)
    # A5 Neural CDE 제외 (torchcde 미설치, v1.2 executor patch)
}
