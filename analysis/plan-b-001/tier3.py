"""plan-b-001 c5b — Tier3 feature (③, notebook LB_0.6780 cell 16 port).

world frame, m 단위 통일. (N, 5):
  cum_path  = Σ‖disp‖ (누적 경로장)
  roll_mean = ‖v‖ window=3 rolling mean × DT 의 mean
  roll_std  = 동 rolling 의 std
  a_unit    = ‖a·DT²‖ mean
  j_unit    = ‖j·DT³‖ mean
모두 비음수 magnitude → log1p.
"""
from __future__ import annotations

import numpy as np

DT = 0.040


def build_tier3(X: np.ndarray) -> np.ndarray:
    """X (N, 11, 3) world → (N, 5) float32 (log1p)."""
    X = np.asarray(X, dtype=np.float64)
    disp = np.diff(X, axis=1)                       # (N,10,3)
    v = disp / DT
    a = np.diff(v, axis=1) / DT                     # (N,9,3)
    j = np.diff(a, axis=1) / DT                     # (N,8,3)

    cum_path = np.linalg.norm(disp, axis=-1).sum(axis=1)            # (N,)
    speed = np.linalg.norm(v, axis=-1)                              # (N,10)
    roll = np.stack(
        [speed[:, i:i + 3].mean(axis=1) for i in range(speed.shape[1] - 2)], axis=1
    ) * DT                                                          # (N,8) m
    roll_mean = roll.mean(axis=1)
    roll_std = roll.std(axis=1)
    a_unit = np.linalg.norm(a * DT ** 2, axis=-1).mean(axis=1)      # (N,) m
    j_unit = np.linalg.norm(j * DT ** 3, axis=-1).mean(axis=1)

    feats = np.stack([cum_path, roll_mean, roll_std, a_unit, j_unit], axis=-1)
    out = np.log1p(feats).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)


if __name__ == "__main__":
    rng = np.random.default_rng(20260526)
    X = np.cumsum(rng.standard_normal((16, 11, 3)) * 0.01, axis=1).astype(np.float32)
    out = build_tier3(X)
    assert out.shape == (16, 5) and not np.isnan(out).any()
    print(f"[smoke] tier3 OK — {out.shape}, mean={out.mean(0)}")
