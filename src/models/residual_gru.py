"""ResidualGRU — lean GRU that predicts a residual (Δx, Δy, Δz)
relative to a closed-form baseline extrapolation.

Per plan-003 §4.1.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.dropout if self.layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hidden, 3)

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, T, input_dim) → out: (B, 3) residual Δ w.r.t. baseline extrapolation
        out, _ = self.gru(X)
        return self.fc(out[:, -1, :])
