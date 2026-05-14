"""plan-014 paradigm module v4 — F0 frozen prior + corrector from-scratch (재사용 끊은).

v4 narrative (plan-014 c1.v4 spec replacement):
  - **F0 = plan-006 frenet_par120_perp_neg020 frozen prior**
    (d1=1.98 / par=1.20 / perp=−0.20 constants, plain function, NOT nn.Module).
    학습 안 함 — requires_grad 개념 자체 없음.
  - **Corrector (encoder + cls + reg head + anchor codebook)** 만 from-scratch + learnable.
    plan-004 `CandidateAttentionGRUSelector` + plan-012 `ring_classifier.py` corrector
    코드 import 0. 본 module 안 직접 재구현.

§2.1.A baseline (4 컴포넌트) + §2.1.B.1 11 ablation lever interface 일괄 구현.

**재사용 끊김 (G1 self-enforced)**:
  - `src.pb_0_6822.selector` / `.ring_classifier` / `.boundary` import 0
  - plan-006 numpy F0 함수 (`f0_predict_frenet_par120_perp_neg020`) import 0
  - 외부 import = `torch`, `torch.nn`, `numpy`, `sklearn`, `src.io` 만
  - F0 산식 / Frenet basis / anchor / loss / hybrid_predict 전부 본 모듈 안 재구현

§ Architecture (from-scratch corrector + frozen F0):
  - Plan014F0Function: plan-006 산식, plain function, constants only — frozen prior
  - Plan014BiGRUEncoder: 2-layer BiGRU input=9 hidden=128 → last-step concat (B, 256)
  - Plan014HybridHead: shared encoder + cls head (Linear → K) + reg head (Linear → K*3, tanh × 0.005)
                       forward (training) / hybrid_predict (inference) 분리
  - hybrid_combined_loss: α·CE_soft(logits, w_k) + β·Huber(reg_offset, residual_k)
                          + opt 0.5·hinge_pred (E4)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Constants — §2.1.A baseline + plan-006 F0 frozen carry
# ─────────────────────────────────────────────────────────────────────────────

# F0 frozen constants (plan-006 frenet_par120_perp_neg020, hard evidence)
F0_D1 = 1.98
F0_PAR = 1.20
F0_PERP = -0.20

ANCHOR_RADIUS = 0.01
REG_SCALE = 0.005          # reg head bound
SOFT_SIGMA = 0.01          # Gaussian soft label kernel
HUBER_DELTA = 0.005
HINGE_R_HIT = 0.01
HINGE_SMOOTH = 0.005

EPS = 1e-12
EPS_BASIS = 1e-6

# K-Means (§2.1.B.1)
KMEANS_K = 7
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 20260606
KMEANS_RADIUS_CLIP = 0.020
KMEANS_MIN_CLUSTER_SIZE_THRESHOLD = 100

# Training defaults (§2.1.A)
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 256
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 20260514
N_FOLDS = 5


def stable_hash_fold(sample_id: str, n_folds: int = N_FOLDS, salt: str = "plan-014-v1") -> int:
    """§3.1: SHA256(f'{salt}::{sample_id}') → int.from_bytes([:8]) % n_folds."""
    digest = hashlib.sha256(f"{salt}::{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


# ─────────────────────────────────────────────────────────────────────────────
# Finite-diff + Frenet basis (§2.1.A.1) — numpy
# ─────────────────────────────────────────────────────────────────────────────


def finite_diff_at(X: np.ndarray, end_idx: int) -> dict[str, np.ndarray]:
    v_last = X[:, end_idx] - X[:, end_idx - 1]
    v_prev = X[:, end_idx - 1] - X[:, end_idx - 2]
    v_prev2 = X[:, end_idx - 2] - X[:, end_idx - 3]
    acc = v_last - v_prev
    prev_acc = v_prev - v_prev2
    jerk = acc - prev_acc
    return {"v_last": v_last, "v_prev": v_prev, "acc": acc, "prev_acc": prev_acc, "jerk": jerk}


def build_frenet_basis_3d(X: np.ndarray, end_idx: int = 10) -> np.ndarray:
    """X: (N, T, 3). Returns R: (N, 3, 3), columns [t̂ | n̂ | b̂]."""
    fd = finite_diff_at(X, end_idx)
    v_last, acc = fd["v_last"], fd["acc"]
    N = X.shape[0]

    t_hat = v_last / (np.linalg.norm(v_last, axis=1, keepdims=True) + EPS)
    acc_par_scalar = np.sum(acc * t_hat, axis=1, keepdims=True)
    acc_perp_vec = acc - acc_par_scalar * t_hat
    perp_norm = np.linalg.norm(acc_perp_vec, axis=1, keepdims=True)

    degenerate = (perp_norm < EPS_BASIS).squeeze(-1)
    n_hat = np.where(
        perp_norm < EPS_BASIS,
        np.tile(np.array([[0.0, 0.0, 1.0]]), (N, 1)),
        acc_perp_vec / (perp_norm + EPS),
    )
    if degenerate.any():
        proj = np.sum(n_hat[degenerate] * t_hat[degenerate], axis=1, keepdims=True)
        n_hat[degenerate] = n_hat[degenerate] - proj * t_hat[degenerate]
        n_norm = np.linalg.norm(n_hat[degenerate], axis=1, keepdims=True)
        n_hat[degenerate] = n_hat[degenerate] / (n_norm + EPS)

    b_hat = np.cross(t_hat, n_hat)
    return np.stack([t_hat, n_hat, b_hat], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# F0 FROZEN function — plan-006 frenet_par120_perp_neg020 산식 본 module 재구현
# ─────────────────────────────────────────────────────────────────────────────


class Plan014F0Function:
    """**Frozen F0 prior** — plan-006 frenet_par120_perp_neg020 (d1=1.98 / par=1.20 / perp=−0.20).

    NOT nn.Module — plain class (constants only, requires_grad 개념 없음).
    __call__ 은 numpy ndarray 또는 torch.Tensor (with `torch.no_grad()`) 양쪽 지원.

    산식: `F0 = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec`
      where `acc_par_vec = (acc · t̂)·t̂`, `acc_perp_vec = acc − acc_par_vec`,
      `v_last = p[end_idx] − p[end_idx−1]`, `acc = p[end_idx] − 2·p[end_idx−1] + p[end_idx−2]`.

    constants = plan-006 hard evidence (frozen, 학습 안 함). v_last 단위 m/step, acc 단위 m/step².
    """

    def __init__(self, end_idx: int = 10, d1: float = F0_D1, par: float = F0_PAR, perp: float = F0_PERP):
        self.end_idx = end_idx
        self.d1 = d1
        self.par = par
        self.perp = perp

    def __call__(self, X: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """X: (B, T, 3) or (N, T, 3). Returns F0_pred (B, 3) — same backend as input."""
        if isinstance(X, torch.Tensor):
            return self._forward_torch(X)
        return self._forward_numpy(X)

    def _forward_numpy(self, X: np.ndarray) -> np.ndarray:
        e = self.end_idx
        v_last = X[:, e] - X[:, e - 1]
        acc = X[:, e] - 2 * X[:, e - 1] + X[:, e - 2]
        t_hat = v_last / (np.linalg.norm(v_last, axis=1, keepdims=True) + EPS)
        acc_par_scalar = np.sum(acc * t_hat, axis=1, keepdims=True)
        acc_par_vec = acc_par_scalar * t_hat
        acc_perp_vec = acc - acc_par_vec
        p0 = X[:, e]
        return p0 + self.d1 * v_last + self.par * acc_par_vec + self.perp * acc_perp_vec

    def _forward_torch(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            e = self.end_idx
            v_last = X[:, e] - X[:, e - 1]
            acc = X[:, e] - 2 * X[:, e - 1] + X[:, e - 2]
            t_hat = v_last / (torch.linalg.norm(v_last, dim=1, keepdim=True) + EPS)
            acc_par_scalar = (acc * t_hat).sum(dim=1, keepdim=True)
            acc_par_vec = acc_par_scalar * t_hat
            acc_perp_vec = acc - acc_par_vec
            p0 = X[:, e]
            return p0 + self.d1 * v_last + self.par * acc_par_vec + self.perp * acc_perp_vec


# ─────────────────────────────────────────────────────────────────────────────
# Input features — make_seq_features (§2.1.A Input pipeline)
# ─────────────────────────────────────────────────────────────────────────────


def _turn_features_per_step(X: np.ndarray, s: int) -> np.ndarray:
    """8 step-local features at step s. Returns (N, 8). per-step Frenet basis (t̂_s = v[s]/‖v[s]‖)."""
    v = X[:, s] - X[:, s - 1]
    v_prev = X[:, s - 1] - X[:, s - 2]
    v_prev2 = X[:, s - 2] - X[:, s - 3]
    acc = v - v_prev
    prev_acc = v_prev - v_prev2
    jerk = acc - prev_acc

    speed = np.linalg.norm(v, axis=1, keepdims=True)
    prev_speed = np.linalg.norm(v_prev, axis=1, keepdims=True)
    t_hat_s = v / (speed + EPS)
    acc_par_scalar = np.sum(acc * t_hat_s, axis=1, keepdims=True)
    acc_perp = acc - acc_par_scalar * t_hat_s
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    perp_norm = np.linalg.norm(acc_perp, axis=1, keepdims=True)
    jerk_norm = np.linalg.norm(jerk, axis=1, keepdims=True)
    turn_cos = np.sum(v * v_prev, axis=1, keepdims=True) / ((speed * prev_speed) + EPS)
    curvature = perp_norm / (speed + EPS)

    feat = np.concatenate(
        [
            speed,
            prev_speed / (speed + EPS),
            acc_norm / (speed + EPS),
            acc_par_scalar / (speed + EPS),
            perp_norm / (speed + EPS),
            jerk_norm / (speed + EPS),
            turn_cos,
            curvature,
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def make_seq_features(X: np.ndarray, end_idx: int = 10, direction: float = 1.0) -> np.ndarray:
    """X: (N, T, 3). Returns (N, 6, 9)."""
    indices = list(range(max(3, end_idx - 5), end_idx + 1))
    if len(indices) < 6:
        indices = [indices[0]] * (6 - len(indices)) + indices
    feats = []
    N = X.shape[0]
    for s in indices:
        f8 = _turn_features_per_step(X, s)
        dir_col = np.full((N, 1), direction, dtype=np.float32)
        feats.append(np.concatenate([f8, dir_col], axis=1))
    return np.stack(feats, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Anchor functions (§2.1.B.1)
# ─────────────────────────────────────────────────────────────────────────────


def compute_anchors_absolute(radius_m: float = ANCHOR_RADIUS) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [+radius_m, 0.0, 0.0], [-radius_m, 0.0, 0.0],
            [0.0, +radius_m, 0.0], [0.0, -radius_m, 0.0],
            [0.0, 0.0, +radius_m], [0.0, 0.0, -radius_m],
        ],
        dtype=np.float32,
    )


def compute_anchors_frenet_orthogonal(radius_m: float = ANCHOR_RADIUS) -> np.ndarray:
    return compute_anchors_absolute(radius_m=radius_m)


def compute_anchors_kmeans(train_residuals_world: np.ndarray, R_world_from_frenet: np.ndarray,
                            fold_id: int = 0, K: int = KMEANS_K,
                            radius_clip_m: float = KMEANS_RADIUS_CLIP,
                            n_init: int = KMEANS_N_INIT,
                            random_state: int = KMEANS_RANDOM_STATE) -> np.ndarray:
    """Returns (K, 3) — anchor[0] = center prepend, anchor[1..K-1] = sklearn KMeans cluster order."""
    from sklearn.cluster import KMeans

    residuals_local = np.einsum("nji,nj->ni", R_world_from_frenet, train_residuals_world)
    residuals_local = np.clip(residuals_local, -radius_clip_m, radius_clip_m)

    km = KMeans(n_clusters=K - 1, n_init=n_init, random_state=random_state + fold_id)
    km.fit(residuals_local)
    centers = km.cluster_centers_.astype(np.float32)
    return np.concatenate([np.zeros((1, 3), dtype=np.float32), centers], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Torch model components — corrector (encoder + cls + reg head)
# ─────────────────────────────────────────────────────────────────────────────


class Plan014BiGRUEncoder(nn.Module):
    """2-layer BiGRU (input=9, hidden=128). forward(seq: (B, 6, 9)) → (B, 256)."""
    def __init__(self, input_dim: int = 9, hidden: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = hidden * 2

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        return out[:, -1, :]  # last-step bidir concat


class Plan014LastStepMLPEncoder(nn.Module):
    """E7 ablation. forward(seq) → (B, hidden=64)."""
    def __init__(self, input_dim: int = 9, hidden: int = 64):
        super().__init__()
        self.mlp_seq = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # separate MLP_cand for anchor encoding (3 → hidden), no final GELU
        self.mlp_cand = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.out_dim = hidden

    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        return self.mlp_seq(seq[:, -1, :])

    def forward_cand(self, anchors: torch.Tensor) -> torch.Tensor:
        return self.mlp_cand(anchors)


class Plan014HybridHead(nn.Module):
    """Encoder + cls head + reg head. Two-method design (§5.2):
      - forward(seq, anchors) — training loop 호출, raw logits + reg_offset.
      - hybrid_predict(seq, anchors, R_wfn, F0_pred_detached, ...) — inference 후처리.

    F0_pred detach 책임 = **caller** (train_one_fold 가 F0_function(X).detach() 후 전달).
    """
    def __init__(self, K: int = 7, encoder_name: str = "bigru",
                  bigru_hidden: int = 128, mlp_hidden: int = 64):
        super().__init__()
        self.K = K
        self.encoder_name = encoder_name
        if encoder_name == "bigru":
            self.encoder = Plan014BiGRUEncoder(input_dim=9, hidden=bigru_hidden, num_layers=2)
            h = self.encoder.out_dim
            self.cls_head = nn.Linear(h, K)
            self.reg_head = nn.Linear(h, K * 3)
        elif encoder_name == "laststep_mlp":
            self.encoder = Plan014LastStepMLPEncoder(input_dim=9, hidden=mlp_hidden)
            h = self.encoder.out_dim
            self.cls_head = None  # dot-product replaces linear cls
            self.reg_head = nn.Linear(h, K * 3)
        else:
            raise ValueError(f"unknown encoder_name={encoder_name}")

    def forward(self, seq: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = seq.shape[0]
        if self.encoder_name == "bigru":
            h = self.encoder(seq)
            logits = self.cls_head(h)
        else:
            h = self.encoder.forward_seq(seq)        # (B, hidden)
            cand_h = self.encoder.forward_cand(anchors)  # (K, hidden)
            logits = torch.einsum("bh,kh->bk", h, cand_h)
        reg_flat = self.reg_head(h)
        reg_offset = torch.tanh(reg_flat.view(B, self.K, 3)) * REG_SCALE
        return logits, reg_offset

    def hybrid_predict(self, seq: torch.Tensor, anchors: torch.Tensor,
                       R_wfn: torch.Tensor | None, F0_pred_detached: torch.Tensor,
                       temperature: float = 0.03, use_reg_head: bool = True,
                       r0_logit_prior: float = 0.0) -> torch.Tensor:
        """Returns hybrid_pred (B, 3) in world frame. F0_pred_detached = caller-side detached."""
        logits, reg_offset = self.forward(seq, anchors)
        if r0_logit_prior != 0.0:
            prior = torch.zeros_like(logits)
            prior[:, 0] = r0_logit_prior
            logits = logits + prior

        B = seq.shape[0]
        if R_wfn is not None:
            a_world = torch.einsum("bij,kj->bki", R_wfn, anchors)
        else:
            a_world = anchors.unsqueeze(0).expand(B, -1, -1)

        if temperature <= 1e-8:
            k_star = logits.argmax(dim=1)
            sel_anchor = a_world[torch.arange(B), k_star]
            if use_reg_head:
                sel_reg = reg_offset[torch.arange(B), k_star]
                return F0_pred_detached + sel_anchor + sel_reg
            return F0_pred_detached + sel_anchor

        prob = torch.softmax(logits / temperature, dim=1)
        if use_reg_head:
            weighted = (prob.unsqueeze(-1) * (a_world + reg_offset)).sum(dim=1)
        else:
            weighted = (prob.unsqueeze(-1) * a_world).sum(dim=1)
        return F0_pred_detached + weighted


# ─────────────────────────────────────────────────────────────────────────────
# Loss (§2.1.A.2 + §2.1.B.1 E4)
# ─────────────────────────────────────────────────────────────────────────────


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    abs_x = torch.abs(x)
    quad = 0.5 * x ** 2
    lin = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quad, lin)


def hit_aware_hinge(pred: torch.Tensor, target: torch.Tensor,
                     R_HIT: float = HINGE_R_HIT, smooth: float = HINGE_SMOOTH) -> torch.Tensor:
    err = torch.linalg.norm(pred - target, dim=1)
    excess = err - R_HIT
    return (torch.nn.functional.softplus(excess / smooth) * smooth) ** 2


def gaussian_soft_label(F0_pred_detached: torch.Tensor, anchors_world: torch.Tensor,
                         target: torch.Tensor, sigma: float = SOFT_SIGMA) -> torch.Tensor:
    """F0_pred_detached: (B, 3), anchors_world: (B, K, 3), target: (B, 3). Returns (B, K) w_k."""
    cand_pos = F0_pred_detached.unsqueeze(1) + anchors_world
    d = torch.linalg.norm(cand_pos - target.unsqueeze(1), dim=-1)
    log_w = -(d ** 2) / (2 * sigma ** 2)
    log_w = log_w - log_w.max(dim=1, keepdim=True).values
    w = torch.exp(log_w)
    return w / (w.sum(dim=1, keepdim=True) + EPS)


def hybrid_combined_loss(
    logits: torch.Tensor,                    # (B, K)
    reg_offset: torch.Tensor,                # (B, K, 3)
    F0_pred_detached: torch.Tensor,          # (B, 3) — caller-side detached
    anchors_world: torch.Tensor,             # (B, K, 3)
    target: torch.Tensor,                    # (B, 3)
    sample_weight: torch.Tensor | None = None,
    use_hinge: bool = False,
    use_reg_head: bool = True,
    temperature: float = 0.03,
    alpha: float = 1.0, beta: float = 1.0,
    huber_delta: float = HUBER_DELTA,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (scalar loss, diag dict). Batch reduction = mean over batch (Σw/N or Σ(w·l)/Σw if sample_weight)."""
    w_k = gaussian_soft_label(F0_pred_detached, anchors_world, target, sigma=SOFT_SIGMA)
    log_prob = torch.log_softmax(logits, dim=1)
    ce_loss = -(w_k * log_prob).sum(dim=1)  # (B,)

    residual_k = target.unsqueeze(1) - F0_pred_detached.unsqueeze(1) - anchors_world
    huber_pa = _huber(reg_offset - residual_k, delta=huber_delta)
    huber_offset = (w_k.unsqueeze(-1) * huber_pa).sum(dim=(1, 2))

    if use_hinge:
        prob = torch.softmax(logits / temperature, dim=1)
        if use_reg_head:
            blend = (prob.unsqueeze(-1) * (anchors_world + reg_offset)).sum(dim=1)
        else:
            blend = (prob.unsqueeze(-1) * anchors_world).sum(dim=1)
        hybrid_pred = F0_pred_detached + blend
        hinge_pred = hit_aware_hinge(hybrid_pred, target)
        if use_reg_head:
            loss_per_sample = alpha * ce_loss + 0.5 * (beta * huber_offset) + 0.5 * hinge_pred
        else:
            # E4+E5 동시: Huber 완전 제거, hinge 만 reg term 자리에
            loss_per_sample = alpha * ce_loss + 0.5 * hinge_pred
        diag = {
            "ce_loss": float(ce_loss.mean().item()),
            "huber_offset": float(huber_offset.mean().item()) if use_reg_head else 0.0,
            "hinge_pred": float(hinge_pred.mean().item()),
        }
    else:
        if not use_reg_head:
            huber_offset = huber_offset * 0.0
        loss_per_sample = alpha * ce_loss + beta * huber_offset
        diag = {
            "ce_loss": float(ce_loss.mean().item()),
            "huber_offset": float(huber_offset.mean().item()),
        }

    if sample_weight is not None:
        # weighted mean: Σ(w·loss) / Σw (plan-012 convention)
        loss = (loss_per_sample * sample_weight).sum() / (sample_weight.sum() + EPS)
    else:
        loss = loss_per_sample.mean()
    return loss, diag


# ─────────────────────────────────────────────────────────────────────────────
# Training config + train_one_fold + run_kfold_oof
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    name: str = "anchor"
    K: int = 7
    encoder_name: str = "bigru"            # E7 lever
    codebook: str = "absolute"             # E0: absolute / frenet_orthogonal / kmeans
    frame: str = "world"                   # E1 lever (world / frenet)
    use_reg_head: bool = True              # E5
    use_hinge: bool = False                # E4
    temperature: float = 0.03              # E3
    r0_logit_prior: float = 0.0            # E8
    boundary_weight_on: bool = False       # E6
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    seed: int = DEFAULT_SEED
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    end_idx: int = 10
    monitor: str = "val_hit"               # plan-016 Path B: "val_hit" | "val_loss"


def _hit_rate_torch(pred: torch.Tensor, true: torch.Tensor, threshold_m: float = 0.01) -> float:
    err = torch.linalg.norm(pred - true, dim=-1)
    return float((err <= threshold_m).float().mean().item())


def _boundary_weight(F0_pred_init: np.ndarray, target: np.ndarray) -> np.ndarray:
    """E6: boundary_mask = (0.005 < ‖F0−y‖ < 0.015), sw = where(mask, 3.0, 1.0)."""
    err = np.linalg.norm(F0_pred_init - target, axis=1)
    mask = (err > 0.005) & (err < 0.015)
    return np.where(mask, 3.0, 1.0).astype(np.float32)


def get_anchors_for_fold(codebook: str, X_train: np.ndarray, Y_train: np.ndarray | None,
                          fold_id: int, K: int = 7,
                          radius_m: float = ANCHOR_RADIUS,
                          f0_function: Plan014F0Function | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (anchors_local (K, 3), R_world_from_frenet (N, 3, 3) or None for absolute)."""
    if codebook == "absolute":
        anchors = compute_anchors_absolute(radius_m=radius_m)
        return anchors[:K] if K < 7 else anchors, None
    elif codebook == "frenet_orthogonal":
        anchors = compute_anchors_frenet_orthogonal(radius_m=radius_m)
        return anchors[:K] if K < 7 else anchors, build_frenet_basis_3d(X_train, end_idx=10)
    elif codebook == "kmeans":
        assert Y_train is not None and f0_function is not None, "kmeans needs Y_train + f0_function"
        F0_pred = f0_function(X_train)  # frozen
        R_train = build_frenet_basis_3d(X_train, end_idx=10)
        residuals_world = Y_train - F0_pred
        anchors = compute_anchors_kmeans(residuals_world, R_train, fold_id=fold_id, K=K)
        return anchors, R_train
    else:
        raise ValueError(f"unknown codebook={codebook}")


def train_one_fold(
    config: dict | TrainConfig,
    fold_id: int,
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    f0_function: Plan014F0Function,
    anchors_local: np.ndarray | None = None,
    R_train: np.ndarray | None = None,
    R_val: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train 1 fold. F0 frozen (f0_function plain class). Returns dict with best_val_hit etc."""
    if isinstance(config, dict):
        cfg = TrainConfig(**{k: v for k, v in config.items() if k in TrainConfig.__dataclass_fields__})
    else:
        cfg = config

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # Anchors (if not provided)
    if anchors_local is None:
        anchors_local, R_train = get_anchors_for_fold(cfg.codebook, X_train, Y_train, fold_id,
                                                       K=cfg.K, f0_function=f0_function)
        if cfg.codebook != "absolute":
            R_val = build_frenet_basis_3d(X_val, end_idx=cfg.end_idx)

    # Sequence features
    seq_train = make_seq_features(X_train, end_idx=cfg.end_idx)
    seq_val = make_seq_features(X_val, end_idx=cfg.end_idx)

    # F0 frozen prediction (numpy → use later as detached tensor)
    F0_train_np = f0_function(X_train)
    F0_val_np = f0_function(X_val)

    # Boundary weight (E6)
    sw_train = _boundary_weight(F0_train_np, Y_train) if cfg.boundary_weight_on else None

    # To torch
    seq_train_t = torch.from_numpy(seq_train).to(device)
    seq_val_t = torch.from_numpy(seq_val).to(device)
    Y_train_t = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    Y_val_t = torch.from_numpy(Y_val.astype(np.float32)).to(device)
    F0_train_t = torch.from_numpy(F0_train_np.astype(np.float32)).to(device)  # frozen, no grad
    F0_val_t = torch.from_numpy(F0_val_np.astype(np.float32)).to(device)
    anchors_t = torch.from_numpy(anchors_local.astype(np.float32)).to(device)
    R_train_t = torch.from_numpy(R_train.astype(np.float32)).to(device) if R_train is not None else None
    R_val_t = torch.from_numpy(R_val.astype(np.float32)).to(device) if R_val is not None else None
    sw_t = torch.from_numpy(sw_train).to(device) if sw_train is not None else None

    # Build corrector (F0 frozen separately)
    model = Plan014HybridHead(K=anchors_local.shape[0], encoder_name=cfg.encoder_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_train = X_train.shape[0]

    # Initial val_hit (model.eval, Adam.step()=0 random-init)
    model.eval()
    with torch.no_grad():
        hybrid_pred_init = model.hybrid_predict(
            seq_val_t, anchors_t, R_val_t, F0_val_t,
            temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
            r0_logit_prior=cfg.r0_logit_prior,
        )
        initial_val_hit = _hit_rate_torch(hybrid_pred_init, Y_val_t, threshold_m=0.01)

    best_val_hit = -1.0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    patience_left = cfg.patience
    epoch_log: list[dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses = []
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            seq_b = seq_train_t[idx]
            Y_b = Y_train_t[idx]
            F0_b = F0_train_t[idx]  # already detached (no grad source)
            anchors_world_b = (anchors_t.unsqueeze(0).expand(idx.shape[0], -1, -1)
                                if R_train_t is None
                                else torch.einsum("bij,kj->bki", R_train_t[idx], anchors_t))
            sw_b = sw_t[idx] if sw_t is not None else None

            opt.zero_grad()
            logits, reg_offset = model.forward(seq_b, anchors_t)
            loss, _ = hybrid_combined_loss(
                logits, reg_offset, F0_b, anchors_world_b, Y_b,
                sample_weight=sw_b,
                use_hinge=cfg.use_hinge, use_reg_head=cfg.use_reg_head,
                temperature=cfg.temperature,
            )
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            anchors_val_world = (anchors_t.unsqueeze(0).expand(seq_val_t.shape[0], -1, -1)
                                  if R_val_t is None
                                  else torch.einsum("bij,kj->bki", R_val_t, anchors_t))
            logits_v, reg_v = model.forward(seq_val_t, anchors_t)
            val_loss, _ = hybrid_combined_loss(
                logits_v, reg_v, F0_val_t, anchors_val_world, Y_val_t,
                use_hinge=cfg.use_hinge, use_reg_head=cfg.use_reg_head,
                temperature=cfg.temperature,
            )
            hybrid_pred_val = model.hybrid_predict(
                seq_val_t, anchors_t, R_val_t, F0_val_t,
                temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
                r0_logit_prior=cfg.r0_logit_prior,
            )
            val_hit = _hit_rate_torch(hybrid_pred_val, Y_val_t, threshold_m=0.01)
            dcm = float(torch.linalg.norm(hybrid_pred_val - F0_val_t, dim=1).mean().item())

        val_loss_f = float(val_loss.item())
        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss_f,
            "val_hit": val_hit,
            "dcm": dcm,
        })

        # monitor = cfg.monitor ("val_hit" ascending / "val_loss" descending)
        if cfg.monitor == "val_hit":
            improved = val_hit > best_val_hit
        elif cfg.monitor == "val_loss":
            improved = val_loss_f < best_val_loss
        else:
            raise ValueError(f"Unknown monitor: {cfg.monitor!r} (expected 'val_hit' or 'val_loss')")
        if improved:
            best_val_loss = val_loss_f
            best_val_hit = val_hit
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        hybrid_pred_val = model.hybrid_predict(
            seq_val_t, anchors_t, R_val_t, F0_val_t,
            temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
            r0_logit_prior=cfg.r0_logit_prior,
        )
        oof_pred = hybrid_pred_val.cpu().numpy()
        final_dcm = float(torch.linalg.norm(hybrid_pred_val - F0_val_t, dim=1).mean().item())

    return {
        "fold_id": fold_id,
        "n_train": int(n_train),
        "n_val": int(X_val.shape[0]),
        "best_val_loss": best_val_loss,
        "best_val_hit": best_val_hit,
        "best_epoch": best_epoch,
        "initial_val_hit": initial_val_hit,
        "dcm": final_dcm,
        "oof_pred": oof_pred,
        "epoch_log": epoch_log,
        "anchors_local": anchors_local.tolist(),
    }


def run_kfold_oof(
    ids: list[str],
    X: np.ndarray,
    Y: np.ndarray,
    config: dict | TrainConfig,
    f0_function: Plan014F0Function | None = None,
    folds: list[int] | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """Run K-fold OOF. F0 frozen (provided externally or default plan-006 carry)."""
    if f0_function is None:
        f0_function = Plan014F0Function()  # plan-006 frozen defaults

    fold_of = np.array([stable_hash_fold(sid) for sid in ids])
    if folds is None:
        folds = list(range(N_FOLDS))

    oof_pred = np.full_like(Y, fill_value=np.nan, dtype=np.float32)
    fold_results = []
    for f in folds:
        train_mask = (fold_of != f)
        val_mask = (fold_of == f)
        res = train_one_fold(
            config, f,
            X[train_mask], Y[train_mask],
            X[val_mask], Y[val_mask],
            f0_function=f0_function,
        )
        oof_pred[val_mask] = res["oof_pred"]
        fold_results.append({k: v for k, v in res.items() if k != "oof_pred"})
        if progress_cb is not None:
            progress_cb(f, res)

    completed_mask = ~np.isnan(oof_pred).any(axis=1)
    if completed_mask.sum() > 0:
        overall_oof_hit = float(np.mean(
            np.linalg.norm(oof_pred[completed_mask] - Y[completed_mask], axis=-1) <= 0.01
        ))
    else:
        overall_oof_hit = float("nan")

    return {
        "overall_oof_hit_1cm": overall_oof_hit,
        "fold_results": fold_results,
        "oof_pred": oof_pred,
        "fold_of": fold_of.tolist(),
    }
