"""yaw 좌표계 회전 유틸 (plan-a-001 §4.1).

notes/LB_0.6780 코드공유.ipynb cell 13 그대로 이식.
- yaw_angle: 마지막 속도 벡터의 xy 헤딩.
- rotate_xy: v 를 +x 축 정렬(−θ 회전), z 보존. inverse 는 역회전.
- 항등성 inverse_rotate_xy(rotate_xy(v,θ),θ) == v (atol 1e-9) 보장.
"""
from __future__ import annotations

import numpy as np

DT = 0.040  # 40 ms — kalman.py 와 동일 상수 (dash-dir flat-module 컨벤션상 sibling import 회피)


def yaw_angle(v: np.ndarray) -> np.ndarray:
    """v: (N,3) 속도 → (N,) xy yaw = atan2(v_y, v_x)."""
    return np.arctan2(v[:, 1], v[:, 0])


def yaw_from_last_step(X: np.ndarray) -> np.ndarray:
    """X: (N,T,3) → 마지막 step 속도 (X[-1]-X[-2])/DT 의 yaw (N,).

    /DT 는 atan2 에 무영향이나 노트북 정의(cell 14) 그대로 유지.
    """
    v_last = (X[:, -1] - X[:, -2]) / DT
    return yaw_angle(v_last)


def rotate_xy(vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """vec: (N,3), theta: (N,) → xy 평면 −θ 회전(+x 정렬), z 보존."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.stack(
        [
            vec[:, 0] * c + vec[:, 1] * s,
            -vec[:, 0] * s + vec[:, 1] * c,
            vec[:, 2],
        ],
        axis=-1,
    )


def inverse_rotate_xy(vec: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """rotate_xy 의 역 — rotated frame → world frame."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.stack(
        [
            vec[:, 0] * c - vec[:, 1] * s,
            vec[:, 0] * s + vec[:, 1] * c,
            vec[:, 2],
        ],
        axis=-1,
    )


def assert_rotation_identity(n: int = 256, seed: int = 0, atol: float = 1e-9) -> None:
    """round-trip 항등성 검증 (smoke 용)."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((n, 3))
    theta = rng.uniform(-np.pi, np.pi, n)
    back = inverse_rotate_xy(rotate_xy(vec, theta), theta)
    if not np.allclose(back, vec, atol=atol):
        raise AssertionError(f"rotation identity broken: max|Δ|={np.abs(back - vec).max():.2e}")
