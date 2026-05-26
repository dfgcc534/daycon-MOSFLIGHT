"""GRU multi-aux 모델 (plan-a-001 §4.3).

notes/LB_0.6780 코드공유.ipynb cell 24 그대로 이식.
- GRU 인코더(9ch) → 마지막 hidden + 40D scalar concat → 2-layer MLP(GELU+dropout).
- main head: tanh × (main_out_scale_cm/100) — 출력 clamp(±2cm).
- aux F/W head: bare Linear (OOF runner 는 clip=None).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUModelMultiAux(nn.Module):
    def __init__(
        self,
        n_channels: int = 9,
        scal_dim: int = 40,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        fc_hidden: int = 128,
        p: float = 0.2,
        aux_dims: list[int] | None = None,
        aux_clips: list[str | None] | None = None,
        main_out_scale_cm: float = 2.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=p if num_layers > 1 else 0,
        )
        gru_out = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(gru_out + scal_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
        self.head_main = nn.Linear(fc_hidden // 2, 3)
        self.aux_heads = (
            nn.ModuleList([nn.Linear(fc_hidden // 2, d) for d in aux_dims])
            if aux_dims
            else nn.ModuleList([])
        )
        self.aux_clips = aux_clips if aux_clips else [None] * len(aux_dims or [])
        self.main_out_scale_cm = main_out_scale_cm

    def forward(self, seq: torch.Tensor, scal: torch.Tensor):
        out, _ = self.gru(seq)
        x = out[:, -1, :]
        z = torch.cat([x, scal], dim=1)
        z = self.act(self.fc1(z))
        z = self.drop(z)
        z = self.act(self.fc2(z))
        out_main = torch.tanh(self.head_main(z)) * (self.main_out_scale_cm / 100.0)
        outs_aux = []
        for h, clip in zip(self.aux_heads, self.aux_clips):
            o = h(z)
            if clip == "tanh_2cm":
                o = torch.tanh(o) * 0.02
            outs_aux.append(o)
        return out_main, outs_aux
