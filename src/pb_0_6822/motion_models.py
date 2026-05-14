"""IMM feasibility — CV / CA / CT motion model fit per sample.

산식 / parameter count / fallback rule: `/home/ahn/.claude/plans/glimmering-popping-river.md` §Approach.

- CV (6 params): axis-wise linear x(t) = c0 + c1·t.
- CA (9 params): axis-wise quadratic x(t) = c0 + c1·t + c2·t².
- CT (8 params): PCA-plane + Kasa 2D circle LS + perp linear.
  centroid 자유도 모든 model 공통 → 카운트 안 함.

BIC = n·log(RSS/n) + k·log(n), n=33 (11 timesteps × 3 axes).
posterior ∝ exp(-0.5·BIC), normalized.

Pure numpy. scipy / sklearn 의존 없음.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

N_STEPS: int = 11
N_AXES: int = 3
N_RES: int = N_STEPS * N_AXES   # 33
T_PRED: float = 0.08            # +80ms forward prediction target

# Time grid (s). end_idx=10 ↔ t=0.0. Sample i 의 step k 시간 = T_GRID[k].
T_GRID: np.ndarray = np.arange(-10, 1, 1, dtype=np.float64) * 0.04   # [-0.40, -0.36, ..., 0.0]

K_CV: int = 6
K_CA: int = 9
K_CT: int = 8

EPS_STATIONARY: float = 1e-6   # X.std() threshold
EPS_DEGEN_PLANE: float = 0.5   # singular[2] / singular[0] threshold
EPS_NUM: float = 1e-30         # RSS log clipping

FALLBACK_NONE: int = 0
FALLBACK_STATIONARY: int = 1
FALLBACK_DEGEN_PLANE: int = 2
FALLBACK_NON_MONOTONE_ANGLE: int = 4


# ─────────────────────────────────────────────────────────────────────────
# CV / CA — axis-wise polynomial fit (closed-form)
# ─────────────────────────────────────────────────────────────────────────

def fit_cv(traj: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit constant-velocity (linear) per axis.

    traj: (11, 3). Returns (coeffs (3, 2), RSS_total, pred_t80 (3,)).
    coeffs[axis] = [c0, c1] such that x(t) = c0 + c1·t.
    """
    coeffs = np.empty((3, 2), dtype=np.float64)
    rss = 0.0
    pred_t80 = np.empty(3, dtype=np.float64)
    for axis in range(3):
        c = np.polynomial.polynomial.polyfit(T_GRID, traj[:, axis], 1)
        coeffs[axis] = c
        pred = np.polynomial.polynomial.polyval(T_GRID, c)
        rss += float(np.sum((traj[:, axis] - pred) ** 2))
        pred_t80[axis] = float(np.polynomial.polynomial.polyval(T_PRED, c))
    return coeffs, rss, pred_t80


def fit_ca(traj: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit constant-acceleration (quadratic) per axis.

    traj: (11, 3). Returns (coeffs (3, 3), RSS_total, pred_t80 (3,)).
    """
    coeffs = np.empty((3, 3), dtype=np.float64)
    rss = 0.0
    pred_t80 = np.empty(3, dtype=np.float64)
    for axis in range(3):
        c = np.polynomial.polynomial.polyfit(T_GRID, traj[:, axis], 2)
        coeffs[axis] = c
        pred = np.polynomial.polynomial.polyval(T_GRID, c)
        rss += float(np.sum((traj[:, axis] - pred) ** 2))
        pred_t80[axis] = float(np.polynomial.polynomial.polyval(T_PRED, c))
    return coeffs, rss, pred_t80


# ─────────────────────────────────────────────────────────────────────────
# CT — PCA-plane + Kasa 2D circle LS + perp linear
# ─────────────────────────────────────────────────────────────────────────

def _pca_plane(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """centered: (11, 3) (= traj - traj.mean(0)).

    Returns (e1, e2, n_hat, singular) where:
        e1, e2: plane basis (orthonormal). e1 along largest spread.
        n_hat: plane normal (orthogonal to e1, e2).
        singular: (3,) singular values descending.
    """
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    e1 = Vt[0]
    e2 = Vt[1]
    n_hat = Vt[2]
    return e1, e2, n_hat, S


def _kasa_circle_ls(p2: np.ndarray) -> tuple[float, float, float]:
    """Kasa method: linear LS circle fit on 2D points.

    Circle (x-cx)² + (y-cy)² = r² 를 전개해 `x² + y² = 2x·cx + 2y·cy + (r² - cx² - cy²)`.
    `[2u, 2v, 1] · [a, b, c]^T = u² + v²` where a=cx, b=cy, c = r² - cx² - cy².
    p2: (11, 2). Returns (cx, cy, r). r² = c + a² + b². r clipped to ≥ 0.
    """
    u = p2[:, 0]
    v = p2[:, 1]
    A = np.stack([2.0 * u, 2.0 * v, np.ones_like(u)], axis=1)
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c_const = sol
    r2 = c_const + cx * cx + cy * cy
    r = math.sqrt(max(r2, 0.0))
    return float(cx), float(cy), float(r)


def fit_ct(traj: np.ndarray) -> tuple[dict[str, Any], float, np.ndarray, int]:
    """Fit 3D coordinated turn (PCA-plane + 2D arc + perp linear).

    traj: (11, 3). Returns (params_dict, RSS_total, pred_t80 (3,), fallback_flag).

    fallback_flag bitmask: FALLBACK_STATIONARY | FALLBACK_DEGEN_PLANE | FALLBACK_NON_MONOTONE_ANGLE.

    params_dict keys:
        cx, cy, r, theta_0, omega, q0, qv: scalar
        e1, e2, n_hat, centroid: (3,) arrays
    """
    fallback = FALLBACK_NONE
    centroid = traj.mean(axis=0)
    centered = traj - centroid

    e1, e2, n_hat, S = _pca_plane(centered)

    # degenerate plane check
    if S[0] > 1e-12 and S[2] / S[0] > EPS_DEGEN_PLANE:
        fallback |= FALLBACK_DEGEN_PLANE

    # 2D projection + perp coord
    p2 = centered @ np.stack([e1, e2], axis=1)   # (11, 2)
    q = centered @ n_hat                          # (11,)

    cx, cy, r = _kasa_circle_ls(p2)

    # angular θ_k
    theta_raw = np.arctan2(p2[:, 1] - cy, p2[:, 0] - cx)
    theta = np.unwrap(theta_raw)

    # non-monotone angular check
    dtheta = np.diff(theta)
    if dtheta.size > 0 and np.max(np.abs(dtheta)) > math.pi:
        fallback |= FALLBACK_NON_MONOTONE_ANGLE

    if fallback & FALLBACK_NON_MONOTONE_ANGLE:
        # no-fit baseline: RSS = sum((p2 - mean(p2))²) + perp linear RSS
        p2_mean = p2.mean(axis=0)
        rss_inplane = float(np.sum((p2 - p2_mean) ** 2))
        q_coeffs = np.polynomial.polynomial.polyfit(T_GRID, q, 1)
        q_pred = np.polynomial.polynomial.polyval(T_GRID, q_coeffs)
        rss_perp = float(np.sum((q - q_pred) ** 2))
        rss_total = rss_inplane + rss_perp
        pred_t80 = centroid.copy()   # no-fit pred = centroid (degenerate)
        params = {
            "cx": cx, "cy": cy, "r": r,
            "theta_0": float(theta[-1]), "omega": 0.0,
            "q0": float(q_coeffs[0]), "qv": float(q_coeffs[1]),
            "e1": e1, "e2": e2, "n_hat": n_hat,
            "centroid": centroid,
        }
        return params, rss_total, pred_t80, fallback

    # θ(t) = θ_0 + ω·t (linear LS via polyfit deg=1)
    theta_coeffs = np.polynomial.polynomial.polyfit(T_GRID, theta, 1)
    theta_0 = float(theta_coeffs[0])
    omega = float(theta_coeffs[1])

    # q(t) = q0 + qv·t
    q_coeffs = np.polynomial.polynomial.polyfit(T_GRID, q, 1)
    q0 = float(q_coeffs[0])
    qv = float(q_coeffs[1])

    # Reconstruction at each step
    theta_pred = theta_0 + omega * T_GRID
    p2_pred = np.stack([cx + r * np.cos(theta_pred), cy + r * np.sin(theta_pred)], axis=1)
    q_pred = q0 + qv * T_GRID
    pos_pred = centroid[None, :] + p2_pred[:, 0:1] * e1[None, :] + p2_pred[:, 1:2] * e2[None, :] + q_pred[:, None] * n_hat[None, :]
    rss_total = float(np.sum((traj - pos_pred) ** 2))

    # Forward prediction at t=+0.08
    theta_t80 = theta_0 + omega * T_PRED
    p2_t80 = np.array([cx + r * math.cos(theta_t80), cy + r * math.sin(theta_t80)])
    q_t80 = q0 + qv * T_PRED
    pred_t80 = centroid + p2_t80[0] * e1 + p2_t80[1] * e2 + q_t80 * n_hat

    params = {
        "cx": cx, "cy": cy, "r": r,
        "theta_0": theta_0, "omega": omega,
        "q0": q0, "qv": qv,
        "e1": e1, "e2": e2, "n_hat": n_hat,
        "centroid": centroid,
    }
    return params, rss_total, pred_t80, fallback


# ─────────────────────────────────────────────────────────────────────────
# BIC / posterior
# ─────────────────────────────────────────────────────────────────────────

def bic_from_rss(rss: float, n: int = N_RES, k: int = 0) -> float:
    """BIC = n·log(RSS/n) + k·log(n). RSS clipped to EPS_NUM."""
    rss_safe = max(rss, EPS_NUM)
    return n * math.log(rss_safe / n) + k * math.log(n)


def posterior_from_bic(bic_triple: np.ndarray) -> np.ndarray:
    """Schwarz approximation: log_evidence ≈ -0.5·BIC. Normalize via logsumexp.

    bic_triple: (3,). Returns posterior (3,) sum=1.
    """
    log_p = -0.5 * np.asarray(bic_triple, dtype=np.float64)
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()
    return p.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Per-sample driver
# ─────────────────────────────────────────────────────────────────────────

def fit_all_modes(traj: np.ndarray) -> dict[str, Any]:
    """Fit CV/CA/CT to single sample. Returns dict with rss/bic/posterior/label/delta_bic/pred_t80/fallback.

    pred_t80 shape: (3, 3) — [CV, CA, CT] × [x, y, z].
    """
    # Stationary fallback (per-axis std across 11 timesteps)
    if traj.std(axis=0).max() < EPS_STATIONARY:
        # All modes degenerate to constant. Force CV label.
        pred = np.tile(traj.mean(axis=0), (3, 1))   # (3, 3)
        bic_arr = np.array([0.0, 0.0, 0.0])
        return {
            "rss": np.array([0.0, 0.0, 0.0]),
            "bic": bic_arr,
            "posterior": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "mode_label": 0,
            "delta_bic": 0.0,
            "pred_t80": pred,
            "fallback_flag": FALLBACK_STATIONARY,
            "ct_params": None,
        }

    cv_coeffs, rss_cv, pred_cv = fit_cv(traj)
    ca_coeffs, rss_ca, pred_ca = fit_ca(traj)
    ct_params, rss_ct, pred_ct, ct_fallback = fit_ct(traj)

    bic_cv = bic_from_rss(rss_cv, k=K_CV)
    bic_ca = bic_from_rss(rss_ca, k=K_CA)
    bic_ct = bic_from_rss(rss_ct, k=K_CT)
    bic_arr = np.array([bic_cv, bic_ca, bic_ct], dtype=np.float64)
    rss_arr = np.array([rss_cv, rss_ca, rss_ct], dtype=np.float64)

    label = int(np.argmin(bic_arr))
    sorted_bic = np.sort(bic_arr)
    delta_bic = float(sorted_bic[1] - sorted_bic[0])
    post = posterior_from_bic(bic_arr)

    pred_t80 = np.stack([pred_cv, pred_ca, pred_ct], axis=0)   # (3, 3)

    return {
        "rss": rss_arr,
        "bic": bic_arr,
        "posterior": post,
        "mode_label": label,
        "delta_bic": delta_bic,
        "pred_t80": pred_t80,
        "fallback_flag": ct_fallback,
        "ct_params": ct_params,
    }


def fit_all_samples(X: np.ndarray) -> dict[str, np.ndarray]:
    """Fit CV/CA/CT to N samples. Sample-loop (closed-form, ~30s for 10K).

    X: (N, 11, 3). Returns dict of numpy arrays:
        rss: (N, 3) float64
        bic: (N, 3) float64
        posterior: (N, 3) float32
        labels: (N,) int8 ∈ {0:CV, 1:CA, 2:CT}
        delta_bic: (N,) float64
        pred_t80: (N, 3, 3) float64
        fallback_flags: (N,) int8
    """
    N = X.shape[0]
    rss = np.empty((N, 3), dtype=np.float64)
    bic = np.empty((N, 3), dtype=np.float64)
    posterior = np.empty((N, 3), dtype=np.float32)
    labels = np.empty(N, dtype=np.int8)
    delta_bic = np.empty(N, dtype=np.float64)
    pred_t80 = np.empty((N, 3, 3), dtype=np.float64)
    fallback_flags = np.empty(N, dtype=np.int8)

    for i in range(N):
        res = fit_all_modes(X[i])
        rss[i] = res["rss"]
        bic[i] = res["bic"]
        posterior[i] = res["posterior"]
        labels[i] = res["mode_label"]
        delta_bic[i] = res["delta_bic"]
        pred_t80[i] = res["pred_t80"]
        fallback_flags[i] = res["fallback_flag"]

    return {
        "rss": rss,
        "bic": bic,
        "posterior": posterior,
        "labels": labels,
        "delta_bic": delta_bic,
        "pred_t80": pred_t80,
        "fallback_flags": fallback_flags,
    }
