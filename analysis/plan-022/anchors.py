"""plan-022 §3.4 — 7 anchor layout codebook (Frenet, 단위 m).

각 layout 의 invariant:
  - dtype = np.float32
  - shape = (K, 3), K ∈ {7, 8, 13, 13, 13, 14, 13}
  - max ‖a‖ ≤ 0.005m + 1e-7 (= 0.5cm radius bound)

Anchor index 일관성 (§3.4 명시):
  - center 가 있는 layout (A1/A2/A3/A4/A7): index 0 = center (0,0,0)
  - center 없는 layout (A5/A6): index 0 부터 첫 anchor (layout-specific)

Post-G_final ablation (NOT in 7-layout sweep, NOT in LAYOUT_NAMES):
  - ANCHORS_A8 = (center, *ANCHORS_A6) — K=15, A6 의 center bias 직접 검증용
    (diag_center_bias_a6_a8.py 에서 import). LAYOUT_NAMES 에는 미등록하여
    21-cell sweep invariant (7 layout × 3 τ_cls) 보존.
"""
from __future__ import annotations

import numpy as np


# ── A1: octahedron + center (7) — plan-021 baseline ──────────────────

ANCHORS_A1 = np.array(
    [
        (0.0, 0.0, 0.0),       # 0: center
        (+0.005, 0.0, 0.0),    # 1: +t̂
        (-0.005, 0.0, 0.0),    # 2: -t̂
        (0.0, +0.005, 0.0),    # 3: +n̂
        (0.0, -0.005, 0.0),    # 4: -n̂
        (0.0, 0.0, +0.005),    # 5: +b̂
        (0.0, 0.0, -0.005),    # 6: -b̂
    ],
    dtype=np.float32,
)


# ── A2: icosahedron + center (13) ────────────────────────────────────

_PHI = (1.0 + np.sqrt(5.0)) / 2.0                  # ≈ 1.6180339
_S_A2 = 0.005 / np.sqrt(1.0 + _PHI**2)              # scale, ≈ 0.002629
_ICO12 = np.array(
    [
        (0.0, +1.0, +_PHI), (0.0, +1.0, -_PHI),
        (0.0, -1.0, +_PHI), (0.0, -1.0, -_PHI),
        (+1.0, +_PHI, 0.0), (+1.0, -_PHI, 0.0),
        (-1.0, +_PHI, 0.0), (-1.0, -_PHI, 0.0),
        (+_PHI, 0.0, +1.0), (+_PHI, 0.0, -1.0),
        (-_PHI, 0.0, +1.0), (-_PHI, 0.0, -1.0),
    ],
    dtype=np.float64,
) * _S_A2
ANCHORS_A2 = np.vstack([[(0.0, 0.0, 0.0)], _ICO12]).astype(np.float32)


# ── A3: cuboctahedron + center (13) — FCC neighbor, uniform edge ─────

_A_A3 = 0.005 / np.sqrt(2.0)                        # ≈ 0.003536
_CUB12 = np.array(
    [
        (+_A_A3, +_A_A3, 0.0), (+_A_A3, -_A_A3, 0.0),
        (-_A_A3, +_A_A3, 0.0), (-_A_A3, -_A_A3, 0.0),
        (+_A_A3, 0.0, +_A_A3), (+_A_A3, 0.0, -_A_A3),
        (-_A_A3, 0.0, +_A_A3), (-_A_A3, 0.0, -_A_A3),
        (0.0, +_A_A3, +_A_A3), (0.0, +_A_A3, -_A_A3),
        (0.0, -_A_A3, +_A_A3), (0.0, -_A_A3, -_A_A3),
    ],
    dtype=np.float64,
)
ANCHORS_A3 = np.vstack([[(0.0, 0.0, 0.0)], _CUB12]).astype(np.float32)


# ── A4: 2-shell octahedron (13) — center + inner 0.25cm + outer 0.5cm ──

ANCHORS_A4 = np.array(
    [
        (0.0, 0.0, 0.0),
        # inner shell ±0.0025m
        (+0.0025, 0.0, 0.0), (-0.0025, 0.0, 0.0),
        (0.0, +0.0025, 0.0), (0.0, -0.0025, 0.0),
        (0.0, 0.0, +0.0025), (0.0, 0.0, -0.0025),
        # outer shell ±0.005m
        (+0.005, 0.0, 0.0), (-0.005, 0.0, 0.0),
        (0.0, +0.005, 0.0), (0.0, -0.005, 0.0),
        (0.0, 0.0, +0.005), (0.0, 0.0, -0.005),
    ],
    dtype=np.float32,
)


# ── A5: cube corners (8) — no center ──────────────────────────────────

_S_A5 = 0.005 / np.sqrt(3.0)                        # ≈ 0.002887
ANCHORS_A5 = np.array(
    [
        (+_S_A5, +_S_A5, +_S_A5), (+_S_A5, +_S_A5, -_S_A5),
        (+_S_A5, -_S_A5, +_S_A5), (+_S_A5, -_S_A5, -_S_A5),
        (-_S_A5, +_S_A5, +_S_A5), (-_S_A5, +_S_A5, -_S_A5),
        (-_S_A5, -_S_A5, +_S_A5), (-_S_A5, -_S_A5, -_S_A5),
    ],
    dtype=np.float32,
)


# ── A6: BCC 14 (axis 6 + corner 8) — no center ────────────────────────

ANCHORS_A6 = np.vstack(
    [
        np.array(
            [
                (+0.005, 0.0, 0.0), (-0.005, 0.0, 0.0),
                (0.0, +0.005, 0.0), (0.0, -0.005, 0.0),
                (0.0, 0.0, +0.005), (0.0, 0.0, -0.005),
            ],
            dtype=np.float32,
        ),
        ANCHORS_A5,
    ]
)  # (14, 3)


# ── A7: Fibonacci spiral 12 + center (13) ─────────────────────────────

def _fib_sphere(N: int, r: float) -> np.ndarray:
    """N 점 Fibonacci spiral on sphere(radius=r). returns (N, 3)."""
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(N)
    theta = 2.0 * np.pi * i / phi_g                  # azimuth
    z = 1.0 - 2.0 * (i + 0.5) / N                    # latitude (uniform in z)
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1) * r


_FIB12 = _fib_sphere(12, 0.005)
ANCHORS_A7 = np.vstack([[(0.0, 0.0, 0.0)], _FIB12]).astype(np.float32)


# ── A8: A6 + center (15) — post-G_final ablation, NOT in LAYOUT_NAMES ──
#
# A6 의 "center bias = mode collapse 원인" 가설 직접 검증용. sweep 의
# 21-cell invariant (7 layout × 3 τ) 를 깨뜨리지 않기 위해 LAYOUT_NAMES
# 에는 등록하지 않음. diag_center_bias_a6_a8.py 에서 직접 import.

ANCHORS_A8 = np.vstack([[(0.0, 0.0, 0.0)], ANCHORS_A6]).astype(np.float32)
assert ANCHORS_A8.shape == (15, 3)
assert float(np.linalg.norm(ANCHORS_A8, axis=1).max()) <= 0.005 + 1e-7


# ── LAYOUT_NAMES dict (export) ────────────────────────────────────────

LAYOUT_NAMES = {
    "A1_octa7":      ANCHORS_A1,
    "A2_ico13":      ANCHORS_A2,
    "A3_cubocta13":  ANCHORS_A3,
    "A4_2shell13":   ANCHORS_A4,
    "A5_cube8":      ANCHORS_A5,
    "A6_bcc14":      ANCHORS_A6,
    "A7_fib13":      ANCHORS_A7,
}


# ── smoke (모듈 import 시 실행되는 invariant 검증) ────────────────────

def _smoke() -> None:
    """plan-022 §3.4 / §4.5 T2 invariants — 호출하지 않고 import 만 해도 OK."""
    expected_K = {
        "A1_octa7": 7, "A2_ico13": 13, "A3_cubocta13": 13, "A4_2shell13": 13,
        "A5_cube8": 8, "A6_bcc14": 14, "A7_fib13": 13,
    }
    for name, arr in LAYOUT_NAMES.items():
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
        assert arr.shape == (expected_K[name], 3), \
            f"{name}: shape {arr.shape} != ({expected_K[name]}, 3)"
        norms = np.linalg.norm(arr, axis=1)
        assert norms.max() <= 0.005 + 1e-7, \
            f"{name}: max ‖a‖ = {norms.max():.7f} > 0.005 + 1e-7"
        assert np.isfinite(arr).all(), f"{name}: non-finite entry"


if __name__ == "__main__":
    _smoke()
    for name, arr in LAYOUT_NAMES.items():
        norms = np.linalg.norm(arr, axis=1)
        print(f"{name:14s} K={arr.shape[0]:2d}  "
              f"‖a‖ min/max = {norms.min():.6f} / {norms.max():.6f}  "
              f"non-zero norms = {np.sum(norms > 1e-9):2d}")
