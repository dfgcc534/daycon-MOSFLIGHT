"""Loss 함수 (plan-a-001 §4.3).

notes/LB_0.6780 코드공유.ipynb cell 18/20 그대로 이식.
- main: combo = euclid + 0.3·softhit(beta=0.002, thr=1cm).
- aux F/W: euclid, λ=0.3 each.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SOFTHIT_BETA = 0.002
HIT_THR = 0.01  # 1 cm
SOFTHIT_W = 0.3
LAMBDA_AUX = 0.3  # λ_F = λ_W


def loss_euclid(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - true) ** 2).sum(dim=-1) + 1e-12).mean()


def loss_softhit(pred: torch.Tensor, true: torch.Tensor, beta: float = SOFTHIT_BETA) -> torch.Tensor:
    d = torch.sqrt(((pred - true) ** 2).sum(dim=-1) + 1e-12)
    return torch.sigmoid((d - HIT_THR) / beta).mean()


def loss_combo(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return loss_euclid(pred, true) + SOFTHIT_W * loss_softhit(pred, true)


def loss_aux_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def loss_aux_euclid(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1) + 1e-12).mean()
