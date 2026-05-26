"""plan-a-002 §4.1 — Kalman 부산물 (innovation·filtered velocity·CV/CA 불일치).

plan-a-001 kalman.py 의 CV/CA 필터를 import 재사용 + 내부 신호 노출 변형.
- kalman_with_internals: CV 필터를 per-step 돌며 innov_seq·filtered_v 캡처.
  pred 는 kalman_predict 재사용(canonical); innovation/filtered velocity 는
  kalman.py CV loop 과 동일 행렬로 캡처 (t=0 zero-pad).
- cv_ca_disagreement: kalman_predict('CA') − kalman_predict('CV').
  (실행 확인: kalman.py 가 model='CA'/'CV' 둘 다 이미 지원 → 직접 재사용.)

plan-a-001 kalman.py 는 **수정 없음** (KR001/KR002 repro 불변).
leakage: 모든 산출은 관측창 X[:, :T] 만의 함수 — t_pred 외삽은 pred 에만 반영, internals 미참조.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_A001 = _THIS.parent / "plan-a-001"


def _load_a001(name: str):
    spec = importlib.util.spec_from_file_location(f"pa001_{name}", _A001 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kalman = _load_a001("kalman")
kalman_predict = _kalman.kalman_predict
DT, T_PRED = _kalman.DT, _kalman.T_PRED
SIGMA_OBS_MAIN, SIGMA_PROC_MAIN = _kalman.SIGMA_OBS_MAIN, _kalman.SIGMA_PROC_MAIN


def kalman_with_internals(
    X: np.ndarray,
    model: str = "CV",
    dt: float = DT,
    t_pred: float = T_PRED,
    sigma_obs: float = SIGMA_OBS_MAIN,
    sigma_proc: float = SIGMA_PROC_MAIN,
    P0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CV 필터 per-step → (pred (N,3), innov_seq (N,T,3), filtered_v (N,T,3)).

    행렬 setup 은 kalman.py CV branch 와 동일 (F/Q/R/P0). loop 는 축 독립 t=1..T-1:
      predict x̂(t|t-1)=F·x̂(t-1|t-1); innov(t)=z(t)−H·x̂(t|t-1) (H=[1,0]);
      update K=P·Hᵀ/S, x̂(t|t)=x̂(t|t-1)+K·innov, P=(I−KH)P.
    innov_seq[:,t]=innov, filtered_v[:,t]=state[:,1] (update 후 속도). t=0 zero-pad.
    pred 는 kalman_predict 재사용 (canonical CV +t_pred 외삽).
    """
    if model != "CV":
        raise ValueError("kalman_with_internals 는 CV 만 (filtered velocity = state[:,1])")
    N, T, _ = X.shape
    F = np.array([[1, dt], [0, 1]], dtype=np.float64)
    Q = sigma_proc**2 * np.array(
        [[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]], dtype=np.float64
    )
    R = sigma_obs**2
    innov_seq = np.zeros((N, T, 3), dtype=np.float64)
    filtered_v = np.zeros((N, T, 3), dtype=np.float64)
    for j in range(3):
        z = X[:, :, j]
        state = np.zeros((N, 2), dtype=np.float64)
        state[:, 0] = z[:, 0]
        P = np.eye(2) * P0
        for t in range(1, T):
            state = state @ F.T          # predict
            P = F @ P @ F.T + Q
            innov = z[:, t] - state[:, 0]
            innov_seq[:, t, j] = innov
            S = P[0, 0] + R
            K = P[:, 0] / S
            state = state + np.outer(innov, K)  # update
            P = P - np.outer(K, P[0, :])
            filtered_v[:, t, j] = state[:, 1]
    # pred: kalman_predict 재사용 (동일 σ/P0 → 동일 결과, canonical)
    pred = kalman_predict(
        X, model="CV", dt=dt, t_pred=t_pred,
        sigma_obs=sigma_obs, sigma_proc=sigma_proc, P0=P0,
    )
    return pred, innov_seq.astype(np.float32), filtered_v.astype(np.float32)


def cv_ca_disagreement(
    X: np.ndarray,
    dt: float = DT,
    t_pred: float = T_PRED,
    sigma_obs: float = SIGMA_OBS_MAIN,
    sigma_proc: float = SIGMA_PROC_MAIN,
    P0: float = 1.0,
) -> np.ndarray:
    """+80ms 외삽 차 CA−CV → (N,3). kalman.py 가 CA/CV 둘 다 지원 → 직접 재사용."""
    kw = dict(dt=dt, t_pred=t_pred, sigma_obs=sigma_obs, sigma_proc=sigma_proc, P0=P0)
    ca = kalman_predict(X, model="CA", **kw)
    cv = kalman_predict(X, model="CV", **kw)
    return (ca - cv).astype(np.float64)
