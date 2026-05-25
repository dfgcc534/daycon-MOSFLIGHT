"""plan-b-001 c5a — noise feature (C3, notebook LB_0.6780 cell 10 port).

관측 궤적의 "지저분함" 추정 2종 (poly2 / savgol), log1p 적용 → (N, 2).
LOO spline 미도입 (cost, §6).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def noise_poly2(X: np.ndarray) -> np.ndarray:
    """각 축 위치를 시간축에 2차 다항 fit → residual std, 3축 평균. X (N,11,3) → (N,)."""
    X = np.asarray(X, dtype=np.float64)
    N, T, _ = X.shape
    t = np.arange(T, dtype=np.float64)
    V = np.vander(t, 3, increasing=False)          # (T, 3) deg=2
    out = np.zeros(N)
    for j in range(3):
        coef, *_ = np.linalg.lstsq(V, X[:, :, j].T, rcond=None)   # (3, N)
        fit = (V @ coef).T                          # (N, T)
        out += (X[:, :, j] - fit).std(axis=1)
    return (out / 3.0).astype(np.float32)


def noise_savgol(X: np.ndarray, w: int = 5, p: int = 2) -> np.ndarray:
    """savgol smooth residual std (3축 평균). X (N,11,3) → (N,)."""
    X = np.asarray(X, dtype=np.float64)
    Xs = savgol_filter(X, window_length=w, polyorder=p, axis=1)
    return (X - Xs).std(axis=1).mean(axis=-1).astype(np.float32)


def build_noise(X: np.ndarray) -> np.ndarray:
    """(N, 2) = [log1p(poly2), log1p(savgol)]."""
    npoly = noise_poly2(X)
    nsav = noise_savgol(X)
    out = np.stack([np.log1p(npoly), np.log1p(nsav)], axis=-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)


if __name__ == "__main__":
    rng = np.random.default_rng(20260526)
    X = np.cumsum(rng.standard_normal((16, 11, 3)) * 0.01, axis=1).astype(np.float32)
    out = build_noise(X)
    assert out.shape == (16, 2) and not np.isnan(out).any()
    print(f"[smoke] noise_estimator OK — {out.shape}, mean={out.mean(0)}")
