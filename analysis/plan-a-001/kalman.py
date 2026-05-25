"""Kalman CV/CA 필터 + t_pred 외삽 (plan-a-001 §4.1).

notes/LB_0.6780 코드공유.ipynb cell 6~8 그대로 이식.
- 각 축(x,y,z) 독립 벡터화 필터, 마지막 state 를 t_pred 만큼 외삽 → (N,3).
- MAIN config: sigma_obs=0.30e-3, sigma_proc=1.0 (노트북 SO_BEST/SP_BEST, train R-Hit≈0.5964).
- aux W alt config (cell 19): sigma_obs=1.0e-3, sigma_proc=1.0 (best σo 의 3.3배).
"""
from __future__ import annotations

import numpy as np

DT = 0.040       # 40 ms (노트북 cell 6)
T_PRED = 0.080   # 80 ms 외삽

# 노트북 cell 8 / 19 의 Kalman σ 박제
SIGMA_OBS_MAIN, SIGMA_PROC_MAIN = 0.30e-3, 1.0   # SO_BEST, SP_BEST
SIGMA_OBS_ALT, SIGMA_PROC_ALT = 1.0e-3, 1.0      # SO_ALT, SP_ALT (aux W)


def kalman_predict(
    X: np.ndarray,
    model: str = "CV",
    dt: float = DT,
    t_pred: float = T_PRED,
    sigma_obs: float = SIGMA_OBS_MAIN,
    sigma_proc: float = SIGMA_PROC_MAIN,
    P0: float = 1.0,
) -> np.ndarray:
    """각 축 독립 벡터화 칼만 필터 + t_pred 외삽.

    X: (N, T, 3) 관측 시계열. return: (N, 3) +t_pred 예측 위치.
    """
    N, T, _ = X.shape
    if model == "CV":
        F = np.array([[1, dt], [0, 1]], dtype=np.float64)
        F_pred = np.array([[1, t_pred], [0, 1]], dtype=np.float64)
        Q = sigma_proc**2 * np.array(
            [[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]], dtype=np.float64
        )
        n_state = 2
    elif model == "CA":
        F = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]], dtype=np.float64)
        F_pred = np.array(
            [[1, t_pred, t_pred**2 / 2], [0, 1, t_pred], [0, 0, 1]], dtype=np.float64
        )
        Q = sigma_proc**2 * np.array(
            [
                [dt**4 / 4, dt**3 / 2, dt**2 / 2],
                [dt**3 / 2, dt**2, dt],
                [dt**2 / 2, dt, 1],
            ],
            dtype=np.float64,
        )
        n_state = 3
    else:
        raise ValueError(f"unknown model {model!r}")

    R = sigma_obs**2
    pred = np.zeros((N, 3), dtype=np.float64)
    for j in range(3):
        z_all = X[:, :, j]
        state = np.zeros((N, n_state), dtype=np.float64)
        state[:, 0] = z_all[:, 0]
        P = np.eye(n_state) * P0
        for t in range(1, T):
            state = state @ F.T
            P = F @ P @ F.T + Q
            innov = z_all[:, t] - state[:, 0]
            S = P[0, 0] + R
            K = P[:, 0] / S
            state = state + np.outer(innov, K)
            P = P - np.outer(K, P[0, :])
        pred[:, j] = (state @ F_pred.T)[:, 0]
    return pred
