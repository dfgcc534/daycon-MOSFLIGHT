"""plan-b-001 c5c — head_summary (Bz/Tz drop + log1p C2 + noise/tier3 통합).

plan-030 51D 에서:
  - Bz_Tz (2) **drop** (yaw 에선 z-row=[0,0,1] 상수, 정보 0).
  - long-tail magnitude 블록(macro8/A6/A10_pct/plan021_macro9)에 **signed-log1p (C2)**.
  - **noise(2) + tier3(5)** 추가.
→ 56D = macro8(8) + A1(3) + A6(3) + A10_pct(9) + A12(3) + macro9(9) + L4(14) + noise(2) + tier3(5).
(A1=STA/LTA ratio, A12=autocorr, L4=soft-hit 0~1 은 bounded → log1p 미적용. noise/tier3 는 builder 에서 이미 log1p.)
model 에서 Linear(56→64) projection (F3 compact).
"""
from __future__ import annotations

import numpy as np


def _slog1p(x: np.ndarray) -> np.ndarray:
    """signed log1p — 부호 보존 long-tail 압축 (음수 안전)."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_head_summary(
    cand_feat_150: np.ndarray,
    plan021_macro9: np.ndarray,
    soft_hit_L4: np.ndarray,
    noise2: np.ndarray,
    tier3_5: np.ndarray,
) -> np.ndarray:
    """→ (N, 56) float32."""
    cand = np.asarray(cand_feat_150)
    m9 = np.asarray(plan021_macro9)
    L4 = np.asarray(soft_hit_L4)
    nz = np.asarray(noise2)
    t3 = np.asarray(tier3_5)
    N = cand.shape[0]
    assert cand.shape[1:] == (14, 150), f"cand_feat_150 mismatch: {cand.shape}"
    assert m9.shape == (N, 9) and L4.shape == (N, 14)
    assert nz.shape == (N, 2) and t3.shape == (N, 5)

    ctx0 = cand[:, 0, :]                              # (N,150) broadcast ctx
    macro_8 = _slog1p(ctx0[:, 24:32])                 # 8 (C2)
    A1 = ctx0[:, 52:55]                               # 3 (ratio, no log)
    A6 = _slog1p(ctx0[:, 120:123])                    # 3 (jitter std, C2)
    A10_pct = _slog1p(ctx0[:, 125:134])               # 9 (rolling std pct, C2)
    A12 = ctx0[:, 137:140]                            # 3 (autocorr, no log)
    m9_log = _slog1p(m9)                              # 9 (C2)

    head = np.concatenate([macro_8, A1, A6, A10_pct, A12, m9_log, L4, nz, t3], axis=-1).astype(np.float32)
    assert head.shape == (N, 56), f"head_summary dim mismatch: {head.shape}"
    return np.nan_to_num(head, nan=0.0, posinf=1e3, neginf=-1e3)


if __name__ == "__main__":
    rng = np.random.default_rng(20260526)
    N = 8
    cand = rng.standard_normal((N, 14, 150)).astype(np.float32)
    m9 = rng.standard_normal((N, 9)).astype(np.float32)
    L4 = rng.uniform(0, 1, (N, 14)).astype(np.float32)
    nz = rng.uniform(0, 1, (N, 2)).astype(np.float32)
    t3 = rng.uniform(0, 1, (N, 5)).astype(np.float32)
    hs = build_head_summary(cand, m9, L4, nz, t3)
    assert hs.shape == (N, 56) and not np.isnan(hs).any()
    print(f"[smoke] head_summary OK — {hs.shape} (Bz/Tz drop, +noise/tier3, log1p)")
