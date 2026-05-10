"""Residual-GRU per-fold training loop + feature_fn factory.

Per plan-003 §4.4.

Targets are residuals: target = y_true - baseline_extrap(X).
Inference: pred = baseline + model(feature_fn(X)).
All features RAW (no normalization) — lean baseline 원칙.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.oscillation import wingbeat_fft
from src.features.physics import acceleration, curvature, jerk, velocity


# ---------- feature functions (R001~R005) ----------

def relative_coords_feature(X: np.ndarray) -> np.ndarray:
    """X: (n, T, 3) → relative to last frame (n, T, 3). R001/R003/R005 default."""
    return X - X[:, -1:, :]


def physics_feature(X: np.ndarray) -> np.ndarray:
    """R002 feature: relative + velocity + acceleration + jerk + curvature.
    output dim = 3 + 3 + 3 + 3 + 1 = 13."""
    rel = relative_coords_feature(X)
    v = velocity(X)
    a = acceleration(X)
    j = jerk(X)
    k = curvature(X)
    return np.concatenate([rel, v, a, j, k], axis=-1)


def wingbeat_feature(X: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """R004 feature: relative + per-axis FFT magnitude (n_bins).
    output dim = 3 + 3*n_bins. default n_bins=3 → 12."""
    rel = relative_coords_feature(X)
    fft_feat = wingbeat_fft(rel, n_bins=n_bins)
    return np.concatenate([rel, fft_feat], axis=-1)


# ---------- factory (R006 combined) ----------

def make_feature_fn(
    components: list[str],
    wingbeat_n_bins: int = 3,
) -> Callable[[np.ndarray], np.ndarray]:
    """components: list[str], must contain "relative"; optionally "physics", "wingbeat".

    output dim = 3
                 + (10 if "physics" else 0)     # vel(3) + acc(3) + jerk(3) + κ(1)
                 + (3*wingbeat_n_bins if "wingbeat" else 0)

    Examples (default n_bins=3):
      ["relative"]                          → 3   (R001/R003/R005)
      ["relative", "physics"]               → 13  (R002)
      ["relative", "wingbeat"]              → 12  (R004)
      ["relative", "physics", "wingbeat"]   → 22  (R006 둘 다 winning)
    """
    if "relative" not in components:
        raise ValueError("'relative' must be in components")
    use_physics = "physics" in components
    use_wingbeat = "wingbeat" in components

    def fn(X: np.ndarray) -> np.ndarray:
        parts = [relative_coords_feature(X)]
        if use_physics:
            parts.extend([velocity(X), acceleration(X), jerk(X), curvature(X)])
        if use_wingbeat:
            parts.append(wingbeat_fft(relative_coords_feature(X), n_bins=wingbeat_n_bins))
        return np.concatenate(parts, axis=-1)

    return fn


# ---------- training loop ----------

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _val_mean_eucl(
    model: nn.Module,
    X_val_t: Tensor,
    baseline_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
) -> float:
    model.eval()
    with torch.no_grad():
        delta = model(X_val_t).cpu().numpy()
    pred = baseline_val + delta
    return float(np.linalg.norm(pred - y_val, axis=-1).mean())


def train_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    baseline_train: np.ndarray,
    baseline_val: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    loss_type: str = "huber",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch: int = 64,
    epochs: int = 100,
    early_stop_patience: int = 10,
    device: str | None = None,
    seed: int = 42,
) -> dict:
    """Train one fold of residual-GRU. Returns:
      {"best_state_dict": dict[Tensor on cpu],
       "best_val_mean_eucl": float, "best_epoch": int, "history": [...]}
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _set_seed(seed)

    # Pre-compute features (numpy) once before epoch loop — caveat: avoid per-batch numpy.
    feat_train = feature_fn(X_train).astype(np.float32, copy=False)
    feat_val = feature_fn(X_val).astype(np.float32, copy=False)
    target_train = (y_train - baseline_train).astype(np.float32, copy=False)

    Xt = torch.from_numpy(feat_train)
    yt = torch.from_numpy(target_train)
    Xv = torch.from_numpy(feat_val).to(device)

    pin = device.startswith("cuda")
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=pin,
    )

    model = model.to(device)
    if loss_type == "huber":
        loss_fn = nn.HuberLoss(delta=1.0)  # PyTorch default — caveat #7
    elif loss_type == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_epoch = -1
    best_state_dict: dict[str, Tensor] = {}
    no_improve = 0
    history: list[dict] = []
    t0 = time.monotonic()

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss at epoch {epoch}: {loss.item()}"
                )
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.detach().item()))
        train_loss = float(np.mean(epoch_losses))

        val_me = _val_mean_eucl(model, Xv, baseline_val, y_val, device)
        if not np.isfinite(val_me):
            raise RuntimeError(f"non-finite val_mean_eucl at epoch {epoch}: {val_me}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_eucl": val_me})

        if val_me < best_val - 1e-12:
            best_val = val_me
            best_epoch = epoch
            # deep-copy detached cpu copy — protect from in-place weight overwrite
            best_state_dict = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

    # restore best weights for downstream use
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    return {
        "best_state_dict": best_state_dict,
        "best_val_mean_eucl": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "n_epochs_run": len(history),
        "train_duration_sec": round(time.monotonic() - t0, 3),
    }
