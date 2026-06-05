"""plan-019 §0.5 spec-amendment — TrajectoryCNNEncoder (신규 작성).

plan-011 P1.ID spec 만 carry (코드 import X, §10 정책).
1-D CNN over (B, T, 3) window — 3 layers, kernel=3, channels=[32,64,64], GELU, global maxpool.
output dim = 64. 학습 가능 (learnable, frozen X).

본 file 의 input dim 은 plan-011 의 9-d SEQ feature 가 아닌 *raw 3-axis window*:
  - plan-019 의 conditioning = traj_features_13d (handcrafted stats) ⊕ cnn(window_6x3) (64-d)
  - 즉 raw trajectory 위에 CNN — handcrafted 가 못 잡는 미세 패턴 학습 의도.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryCNNEncoder(nn.Module):
    """1-D CNN encoder over (B, T, 3) window → (B, 64).

    plan-011 P1.ID spec: 3 layers, kernel=3, channels=[32, 64, 64], GELU, global max-pool.
    """
    def __init__(self, *, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, hidden, kernel_size=3, padding=1), nn.GELU(),
        )
        self.hidden = hidden

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (B, T, 3). Returns (B, hidden=64) — global max-pool over T."""
        x = window.transpose(1, 2)            # (B, 3, T)
        h = self.net(x)                       # (B, hidden, T)
        return h.max(dim=-1).values           # (B, hidden)
