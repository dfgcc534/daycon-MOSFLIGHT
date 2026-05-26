"""plan-a-004 §4.1 — Multi-Hypothesis 출력 head (GRUMultiHead).

plan-a-001 GRUModelMultiAux trunk(GRU+MLP→z) 구조 재사용 + 출력 head 만 K-way 확장:
  z(fc//2) → heads: K×Linear(→3) (각 tanh×2cm) + selector Linear(→K).
gen='mcl'(emergent)/'hybrid'(cand_k=cand_0.detach()+Δ_k)/'supervised'/'motion'/'mdn'(K*7).
**n_heads=1 → GRUModelMultiAux 와 param 생성순서·forward 동일 = bit-identical.**
plan-a-001 model.py 는 수정 X (신규 모듈).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUMultiHead(nn.Module):
    def __init__(
        self, n_channels=15, scal_dim=44, hidden_size=64, num_layers=1, bidirectional=False,
        fc_hidden=128, p=0.3, n_heads=2, gen="mcl", selector=True,
        aux_dims=(3, 3), main_out_scale_cm=2.0,
    ):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden_size, num_layers, bidirectional=bidirectional,
                          batch_first=True, dropout=p if num_layers > 1 else 0)
        gru_out = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(gru_out + scal_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.n_heads = n_heads
        self.gen = gen
        self.scale = main_out_scale_cm / 100.0
        h = fc_hidden // 2
        if gen == "mdn":
            self.mdn = nn.Linear(h, n_heads * 7)   # (μ3, logvar3, logit_w1)/mode
            self.heads = None
            self.selector = None
        else:
            self.heads = nn.ModuleList([nn.Linear(h, 3) for _ in range(n_heads)])
            self.selector = nn.Linear(h, n_heads) if (selector and n_heads > 1) else None
        self.aux_heads = nn.ModuleList([nn.Linear(h, d) for d in aux_dims]) if aux_dims else nn.ModuleList()

    def _trunk(self, seq, scal):
        out, _ = self.gru(seq)
        z = self.act(self.fc1(torch.cat([out[:, -1, :], scal], dim=1)))
        z = self.drop(z)
        return self.act(self.fc2(z))

    def forward(self, seq, scal):
        z = self._trunk(seq, scal)
        aux = [hd(z) for hd in self.aux_heads]
        if self.gen == "mdn":
            o = self.mdn(z).view(-1, self.n_heads, 7)
            return {"mode": "mdn", "mu": torch.tanh(o[..., :3]) * self.scale,
                    "logvar": o[..., 3:6], "logit_w": o[..., 6], "aux": aux}
        preds = torch.stack([torch.tanh(hd(z)) * self.scale for hd in self.heads], dim=1)  # (B,K,3)
        if self.gen == "hybrid" and self.n_heads > 1:
            base = preds[:, 0:1, :].detach()                 # cand_0 (default) detached
            preds = torch.cat([preds[:, 0:1, :], base + preds[:, 1:, :]], dim=1)  # cand_k = cand_0.detach()+Δ_k
        sel = self.selector(z) if self.selector is not None else None
        return {"mode": "mcl", "preds": preds, "selector": sel, "aux": aux}
