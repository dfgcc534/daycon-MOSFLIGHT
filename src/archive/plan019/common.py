"""plan-019 공통 utilities — 신규 작성 (§10 정책).

plan-007 §6.3.1 / §7.1 / §7.2 spec 재구현. plan-018 §3.1 / §5.0 carry.
trainable module 의 import 없음. 외부 lookup = JSON 만:
  - analysis/archive/plan-007/basis_ablation.json  (best_basis_vars, best_basis_params)
  - analysis/archive/plan-007/cma_es_step2.json    (global_mean_speed)
  - analysis/archive/plan-007/sliding_validity.json (aug_usable)
  - data/train_labels.csv + data/{train,test}/*.csv (load_all_samples, load_labels)

stable_fold_id 만 `src/pb_0_6822/selector.py` 의 lock-in helper 를 import.

Exports:
  - compute_basis_terms(x, end_idx, horizon, global_mean_speed) → dict (8 vars + p0)
  - compute_traj_features(window) → (N, 13)
  - build_pool(...) → samples dict (50K pool, 8 vars stacked, fold_id, target)
  - kfold_split_ids(ids, k=5) → list of (train_idx, val_idx)
  - soft_hit_loss(pred, target, threshold=0.01, sharpness=200)
  - hit_rate(pred, target, r=0.01)
  - load_artifacts() → bundle dict
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import selector  # noqa: E402  (stable_fold_id only)


R_HIT = 0.01
EPS = 1e-9
BEST_BASIS_VARS = ["d1", "acc_par", "acc_perp", "d2", "jerk", "ts_term",
                   "speed_slope_d1", "rotation_term"]   # plan-007 best_basis.json carry


# ───────────────────────────────────────────────────────────────────────────
# §6.3.1 — basis terms (8 vars). plan-007 compute_all_terms 재구현 + horizon arg.
# §7.3 spec-amendment: horizon parameterization.
# ───────────────────────────────────────────────────────────────────────────

def compute_basis_terms(x: np.ndarray, end_idx: int, horizon: int,
                         global_mean_speed: float) -> dict[str, np.ndarray]:
    """plan-007 §6.3.1 식 (재구현). 8 basis vars + p0.

    horizon ∈ {1, 2} — t (h step) 의 scaling 효과:
      - t¹ (속도 / d1) 는 horizon 배수 (= horizon · d1).
      - t² (가속도 / d2) 는 horizon² 배수.
      - t³ (jerk) 는 horizon³ 배수.

    plan-007 의 default 는 horizon=2 (+80ms target = end_idx +2 step).
    plan-019 §7.3 의 amendment 로 inner-loop 의 +40ms self-supervised pretext 용
    horizon=1 도 별도 사용.

    Args:
        x:                  (N, T, 3) trajectory
        end_idx:            anchor 시점 (10 for original, 5..8 for sliding aug)
        horizon:            target horizon (1 or 2)
        global_mean_speed:  ts_term scaling 용 (plan-007 step 2 결과)

    Returns:
        dict with keys: p0, d1, acc_par, acc_perp, d2, jerk, ts_term,
                        speed_slope_d1, rotation_term.
        Each (N, 3) except p0 also (N, 3). Multiplied by horizon-dependent scale
        such that anchor = p0 + Σ c_i · term_i 가 +horizon-step prediction 의미.
    """
    p0 = x[:, end_idx]                                                          # (N, 3)
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = d1 - d2                                                                # (N, 3)
    d1_norm = np.linalg.norm(d1, axis=1, keepdims=True) + EPS                     # (N, 1)
    tangent = d1 / d1_norm
    acc_par = (acc * tangent).sum(axis=1, keepdims=True) * tangent
    acc_perp = acc - acc_par
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    jerk_raw = acc - prev_acc
    time_scale_factor = d1_norm / (global_mean_speed + EPS)
    ts_term_raw = time_scale_factor * d1

    # speed_slope_d1 (plan-007 §6.1 new var)
    steps_for_mean = np.stack([
        x[:, end_idx - 3] - x[:, end_idx - 4],
        x[:, end_idx - 2] - x[:, end_idx - 3],
        x[:, end_idx - 1] - x[:, end_idx - 2],
        x[:, end_idx]     - x[:, end_idx - 1],
    ], axis=1)                                                                   # (N, 4, 3)
    step_norms = np.linalg.norm(steps_for_mean, axis=2)                          # (N, 4)
    mean_speed = step_norms.mean(axis=1, keepdims=True) + EPS                    # (N, 1)
    older_step = x[:, end_idx - 4] - x[:, end_idx - 5]
    older_step_norm = np.linalg.norm(older_step, axis=1, keepdims=True)
    speed_slope_scalar = (d1_norm - older_step_norm) / mean_speed                # (N, 1)
    speed_slope_d1_raw = speed_slope_scalar * d1                                  # (N, 3)

    # rotation_term (rotate d1 by omega·horizon in xy-plane). horizon dependency
    # 은 *내장* — plan-007 §6.1 이 이미 horizon=2 기준 정의.
    cross_z = d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0]
    dot_xy = (d2 * d1).sum(axis=1)
    omega = np.arctan2(cross_z, dot_xy)
    theta = omega * horizon
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_x = cos_t * d1[:, 0] - sin_t * d1[:, 1]
    rot_y = sin_t * d1[:, 0] + cos_t * d1[:, 1]
    rot_z = d1[:, 2]
    rot_d1 = np.stack([rot_x, rot_y, rot_z], axis=1)
    rotation_term_raw = rot_d1 - d1                                                # (N, 3)

    # horizon scaling per term order:
    # - d1 (t¹): · horizon
    # - acc_par, acc_perp, d2 (t²) — plan-007 식이 이미 1-step base, 본 식 변경 X
    # - jerk (t³) — 동일하게 1-step base. plan-007 default 가 horizon=2 일 때
    #   CMA-ES 가 *결과 weighting 으로 보정* — 본 plan 도 *각 horizon 별 c_i*
    #   coefficient 가 보정 담당 (즉 식 자체는 동일, c_i 가 학습됨).
    # - 단, horizon=1 (inner loop pretext) 의 경우 *동일 식 + 다른 c* 가 정확.
    # → 본 구현: rotation_term 만 horizon 직접 의존 (rotate by omega·horizon).
    #   나머지 6 vars 는 horizon-invariant (1-step diff). horizon 효과는 학습된 c 가 흡수.
    return {
        "p0": p0,
        "d1": d1,
        "acc_par": acc_par,
        "acc_perp": acc_perp,
        "d2": d2,
        "jerk": jerk_raw,
        "ts_term": ts_term_raw,
        "speed_slope_d1": speed_slope_d1_raw,
        "rotation_term": rotation_term_raw,
    }


# ───────────────────────────────────────────────────────────────────────────
# §7.1 traj_features (13d). plan-007 compute_trajectory_features 재구현.
# ───────────────────────────────────────────────────────────────────────────

def compute_traj_features(window: np.ndarray) -> np.ndarray:
    """window: (N, 6, 3) — last 6 steps ending at anchor. Returns (N, 13) float32."""
    pos_mean = window.mean(axis=1)                                                # (N, 3)
    pos_std = window.std(axis=1)                                                  # (N, 3)
    pos_range = window.max(axis=1) - window.min(axis=1)                           # (N, 3)
    deltas = np.diff(window, axis=1)                                              # (N, 5, 3)
    speed_norms = np.linalg.norm(deltas, axis=2)                                  # (N, 5)
    speed_mean = speed_norms.mean(axis=1, keepdims=True)
    speed_std = speed_norms.std(axis=1, keepdims=True)
    speed_max = speed_norms.max(axis=1, keepdims=True)
    speed_last = speed_norms[:, -1:]
    feats = np.concatenate(
        [pos_mean, pos_std, pos_range, speed_mean, speed_std, speed_max, speed_last],
        axis=1,
    )
    assert feats.shape[1] == 13, feats.shape
    return feats.astype(np.float32)


def window_for_end_idx(x: np.ndarray, end_idx: int) -> np.ndarray:
    """(N, 6, 3) window ending at end_idx (inclusive). plan-018 §3.1 carry."""
    return x[:, end_idx - 5 : end_idx + 1]


# ───────────────────────────────────────────────────────────────────────────
# Pool builder — 50K (original 10K + sliding 40K). plan-018 §3.1 carry, 재구현.
# fold_id = stable_fold_id (md5-based, original sample_id 의 5-fold).
# ───────────────────────────────────────────────────────────────────────────

def build_pool(train_x: np.ndarray, train_y: np.ndarray, ids: list[str],
               aug_usable: bool, global_mean_speed: float,
               include_window: bool = True,
               horizon_for_basis: int = 2,
               include_prev_basis_h1: bool = False) -> dict:
    """50K pool builder. plan-007 stack_train_full + build_all_samples 재구현.

    Args:
        train_x:                (N, 11, 3) original trajectories
        train_y:                (N, 3) original targets
        ids:                    list of N sample ids
        aug_usable:             plan-007 sliding_validity.json key (True for 50K).
        global_mean_speed:      ts_term scaling.
        include_window:         (B, 6, 3) window per sample 도 stack (plan-018 §3.1 carry).
        horizon_for_basis:      basis_terms horizon (default 2 = +80ms = target=train_y).
        include_prev_basis_h1:  S3 spec-amendment — basis_terms_prev (horizon=1) 도 별도 계산.

    Returns:
        dict:
          p0, target,
          traj_features (N, 13),
          basis_terms (N, 8, 3),                      ← horizon=horizon_for_basis
          basis_terms_prev (N, 8, 3) (optional)       ← horizon=1, window[:, :-1] 기준
          window (N, 6, 3) (optional),
          fold_id (N,) int8,
          is_orig_end10 (N,) bool.
    """
    n_orig = train_x.shape[0]
    fold_orig = np.asarray([selector.stable_fold_id(s, 5) for s in ids], dtype=np.int8)

    block_specs = [10]
    if aug_usable:
        block_specs += [5, 6, 7, 8]

    feats_list, basis_list, basis_prev_list = [], [], []
    p0_list, tgt_list, fold_list, win_list = [], [], [], []
    is_orig_list = []

    for end_idx in block_specs:
        win = window_for_end_idx(train_x, end_idx)                                # (N, 6, 3)
        feats = compute_traj_features(win)                                        # (N, 13)
        terms = compute_basis_terms(train_x, end_idx, horizon=horizon_for_basis,
                                     global_mean_speed=global_mean_speed)
        basis = np.stack([terms[v] for v in BEST_BASIS_VARS], axis=1)             # (N, 8, 3)

        if end_idx == 10:
            target = train_y.astype(np.float32)
            is_orig = np.ones(n_orig, dtype=bool)
        else:
            target = train_x[:, end_idx + 2].astype(np.float32)
            is_orig = np.zeros(n_orig, dtype=bool)

        feats_list.append(feats.astype(np.float32))
        basis_list.append(basis.astype(np.float32))
        p0_list.append(terms["p0"].astype(np.float32))
        tgt_list.append(target)
        fold_list.append(fold_orig)
        win_list.append(win.astype(np.float32))
        is_orig_list.append(is_orig)

        if include_prev_basis_h1:
            # window[:, :-1] = last 5 steps. fake "end_idx_prev = end_idx - 1" 의
            # basis_terms (horizon=1). slicing 으로 end_idx 자체를 -1 한 view.
            terms_prev = compute_basis_terms(train_x, end_idx - 1, horizon=1,
                                              global_mean_speed=global_mean_speed)
            basis_prev = np.stack([terms_prev[v] for v in BEST_BASIS_VARS], axis=1)
            basis_prev_list.append(basis_prev.astype(np.float32))

    out = {
        "traj_features": np.concatenate(feats_list, axis=0),
        "basis_terms":   np.concatenate(basis_list, axis=0),
        "p0":            np.concatenate(p0_list, axis=0),
        "target":        np.concatenate(tgt_list, axis=0),
        "fold_id":       np.concatenate(fold_list, axis=0),
        "is_orig_end10": np.concatenate(is_orig_list, axis=0),
    }
    if include_window:
        out["window"] = np.concatenate(win_list, axis=0)
    if include_prev_basis_h1:
        out["basis_terms_prev"] = np.concatenate(basis_prev_list, axis=0)
    return out


# ───────────────────────────────────────────────────────────────────────────
# §7.2 loss / metric — plan-007 식 재구현.
# ───────────────────────────────────────────────────────────────────────────

def soft_hit_loss(pred: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.01, sharpness: float = 200.0) -> torch.Tensor:
    err = torch.norm(pred - target, dim=1)
    return torch.sigmoid(sharpness * (err - threshold)).mean()


def hit_rate(pred: torch.Tensor, target: torch.Tensor, r: float = R_HIT) -> float:
    err = torch.norm(pred - target, dim=1)
    return float((err <= r).float().mean())


def compute_pred_from_anchor(basis_terms: torch.Tensor, coeffs: torch.Tensor,
                              p0: torch.Tensor) -> torch.Tensor:
    """anchor 식: p0 + Σ c · B. (B, n_coeffs, 3), (B, n_coeffs), (B, 3) → (B, 3)."""
    return p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)


# ───────────────────────────────────────────────────────────────────────────
# Artifact bundle (plan-007 dependencies — JSON only).
# ───────────────────────────────────────────────────────────────────────────

def load_artifacts(repo_root: Path | None = None) -> dict:
    """plan-007 JSON 파일에서 best_basis / global_mean_speed / aug_usable 등을 load."""
    root = repo_root if repo_root is not None else REPO_ROOT
    sliding = json.loads((root / "analysis/archive/plan-007/sliding_validity.json").read_text())
    stage2 = json.loads((root / "analysis/archive/plan-007/cma_es_step2.json").read_text())
    stage3 = json.loads((root / "analysis/archive/plan-007/basis_ablation.json").read_text())
    return {
        "aug_usable": bool(sliding["aug_usable"]),
        "global_mean_speed": float(stage2["global_mean_speed"]),
        "best_basis_vars": stage3["best_basis_vars"],
        "stage3_best_params": np.asarray(stage3["best_basis_params"], dtype=np.float32),
    }


def load_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """(ids, train_x (10K, 11, 3), train_y (10K, 3)). float32."""
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids, "id ordering mismatch"
    return ids, X.astype(np.float32), Y.astype(np.float32)
