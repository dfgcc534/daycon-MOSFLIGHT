"""plan-b-001 c2 — yaw frame (Frenet 교체, C1).

xy 평면만 heading θ 만큼 회전, z=world-vertical 보존. degenerate 없음 (움직이는 한 정의).
- yaw_from_X(X)        : θ = atan2(v_last_y, v_last_x), ‖v_xy‖<eps → θ=0 fallback.
- to_yaw(vec, θ)       : world → yaw  [forward, lateral, vertical].
- from_yaw(vec, θ)     : yaw → world (역).
- build_R_wfy(θ)       : yaw→world 회전행렬 (N,3,3) — model decode 의 einsum("nij,nj->ni") 용.

θ 는 vec[...,0].shape 로 broadcast 가능해야 함 (caller 가 reshape; 예: vec (N,K,3) → θ (N,1)).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def yaw_from_X(X: np.ndarray, end_idx: int = 10) -> np.ndarray:
    """θ from last velocity v_last = X[end] - X[end-1]. (N,) float64. degenerate → 0."""
    X = np.asarray(X, dtype=np.float64)
    v_last = X[:, end_idx, :] - X[:, end_idx - 1, :]
    speed_xy = np.linalg.norm(v_last[:, :2], axis=1)
    theta = np.arctan2(v_last[:, 1], v_last[:, 0])
    return np.where(speed_xy < EPS, 0.0, theta)


def to_yaw(vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """world → yaw. vec (..., 3), theta broadcastable to vec[...,0]."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.stack(
        [vec[..., 0] * c + vec[..., 1] * s,
         -vec[..., 0] * s + vec[..., 1] * c,
         vec[..., 2]],
        axis=-1,
    )


def from_yaw(vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """yaw → world (inverse of to_yaw). vec (..., 3), theta broadcastable to vec[...,0]."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.stack(
        [vec[..., 0] * c - vec[..., 1] * s,
         vec[..., 0] * s + vec[..., 1] * c,
         vec[..., 2]],
        axis=-1,
    )


def build_R_wfy(theta: np.ndarray) -> np.ndarray:
    """yaw→world 회전행렬 (N,3,3): R_wfy @ vec_yaw = vec_world. model decode einsum 용."""
    theta = np.asarray(theta, dtype=np.float64)
    c = np.cos(theta)
    s = np.sin(theta)
    N = theta.shape[0]
    R = np.zeros((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


if __name__ == "__main__":
    rng = np.random.default_rng(20260526)
    N = 100
    vec = rng.standard_normal((N, 3))
    theta = rng.uniform(-np.pi, np.pi, N)
    # 항등성: from_yaw(to_yaw(v)) == v
    err = np.abs(from_yaw(to_yaw(vec, theta), theta) - vec).max()
    assert err < 1e-12, f"yaw identity fail: {err}"
    # R_wfy @ to_yaw(v) == v  (R_wfy = yaw→world)
    vyaw = to_yaw(vec, theta)
    R = build_R_wfy(theta)
    vworld = np.einsum("nij,nj->ni", R, vyaw)
    err2 = np.abs(vworld - vec).max()
    assert err2 < 1e-12, f"R_wfy decode mismatch: {err2}"
    # degenerate
    Xd = np.zeros((4, 11, 3))
    assert np.all(yaw_from_X(Xd) == 0.0)
    print(f"[smoke] yaw_frame OK — identity err={err:.1e}, R_wfy err={err2:.1e}, degenerate→0 ✓")
