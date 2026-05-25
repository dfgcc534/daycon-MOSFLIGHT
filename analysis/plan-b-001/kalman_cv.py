"""plan-b-001 c1 — Kalman CV baseline (notebook LB_0.6780 cell 7 port).

Constant-Velocity Kalman filter, per-axis independent, vectorized over samples.
σ_obs=0.3mm, σ_proc=1.0 (notebook grid best), +80ms (2 step) extrapolation.

Two entry points:
- `kalman_predict(X, ...)`      : run filter over the full window, extrapolate +t_pred → (N, 3).
- `kalman_baseline_at(X, t_idx)`: +80ms prediction using observations up to step t_idx (inclusive).
                                  mirrors plan-020 `f0_baseline(x, end_idx)` role (per-step baseline).
"""
from __future__ import annotations

import numpy as np

DT = 0.040       # 40 ms
T_PRED = 0.080   # 80 ms (2 step) horizon — matches plan-b-001 HORIZON=2
SIGMA_OBS = 0.30e-3
SIGMA_PROC = 1.0


def kalman_predict(
    X: np.ndarray,
    sigma_obs: float = SIGMA_OBS,
    sigma_proc: float = SIGMA_PROC,
    dt: float = DT,
    t_pred: float = T_PRED,
    P0: float = 1.0,
) -> np.ndarray:
    """CV Kalman filter per axis + t_pred extrapolation. X (N, T, 3) → (N, 3) world XYZ."""
    X = np.asarray(X, dtype=np.float64)
    N, T, _ = X.shape
    F = np.array([[1.0, dt], [0.0, 1.0]])
    F_pred = np.array([[1.0, t_pred], [0.0, 1.0]])
    Q = sigma_proc ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    R = sigma_obs ** 2
    pred = np.zeros((N, 3))
    for j in range(3):
        z_all = X[:, :, j]
        state = np.zeros((N, 2))
        state[:, 0] = z_all[:, 0]
        P = np.eye(2) * P0
        for t in range(1, T):
            state = state @ F.T
            P = F @ P @ F.T + Q
            innov = z_all[:, t] - state[:, 0]
            S = P[0, 0] + R
            K = P[:, 0] / S
            state = state + np.outer(innov, K)
            P = P - np.outer(K, P[0, :])
        pred[:, j] = (state @ F_pred.T)[:, 0]
    return pred.astype(np.float32)


def kalman_baseline_at(X: np.ndarray, end_idx: int) -> np.ndarray:
    """+80ms Kalman prediction from observations up to step end_idx (inclusive).

    Interface mirrors plan-020 f0_baseline(x, end_idx) so residual_builder can swap arms.
    Uses full history X[:, :end_idx+1] (Kalman benefits from longer window, unlike F0's 3-pt).
    """
    return kalman_predict(X[:, : end_idx + 1, :])


if __name__ == "__main__":
    rng = np.random.default_rng(20260526)
    N = 32
    X = np.cumsum(rng.standard_normal((N, 11, 3)) * 0.01, axis=1).astype(np.float32)
    p = kalman_predict(X)
    assert p.shape == (N, 3) and not np.isnan(p).any()
    p_at = kalman_baseline_at(X, 10)
    assert p_at.shape == (N, 3)
    # baseline_at(X, 10) == kalman_predict(X) (full window)
    assert np.allclose(p, p_at, atol=1e-5)
    print(f"[smoke] kalman_cv OK — pred {p.shape}, baseline_at==full ✓")
