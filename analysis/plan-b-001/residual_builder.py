"""plan-b-001 c3 — residual_builder (yaw frame, C1).

plan-030 잔차 골격 개작:
  (a) raw(t+2) − baseline(t)            → yaw 3-coord (N,7,3)   [GRU input concat]
  (b) raw(t+2) − anchor_world_k(t)      → yaw 3-coord (N,K,7,3) [attention bias]
      anchor_world_k(t) = baseline(t) + R_wfy @ anchors[k]

frame = yaw (단일 v_last θ, step-invariant). 5-coord Frenet → 직교 3축 [forward,lateral,vertical].
baseline = baseline_at_fn(X, t_idx) → (N,3) +80ms pred from data up to step t_idx (F0/Kalman arm).
step align: i=0..6 ↔ t_wall=-6..0; i=5,6 (raw +1/+2 미관측) → zero-pad.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_THIS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p_b001_yaw", _THIS / "yaw_frame.py")
_yaw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_yaw)


def build_residuals(
    X: np.ndarray,
    theta: np.ndarray,
    R_wfy: np.ndarray,
    anchors: np.ndarray,
    baseline_at_fn: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, np.ndarray]:
    """yaw-frame residual builder.

    Args:
        X:              (N, 11, 3) world raw (t_wall=-10..0).
        theta:          (N,) yaw angle (single v_last, step-invariant).
        R_wfy:          (N, 3, 3) yaw→world rotation (for anchor_world_k decode).
        anchors:        (K=14, 3) ANCHORS_A6 (yaw-frame codebook, frame-agnostic values).
        baseline_at_fn: (X, t_idx) -> (N, 3) +80ms world pred from data up to t_idx.

    Returns dict:
        "residual_a":     (N, 7, 3) float32 — yaw [fwd,lat,vert], GRU input concat.
        "residual_b":     (N, K=14, 7, 3) float32 — yaw, attention bias.
    """
    X = np.asarray(X, dtype=np.float64)
    R_wfy_f = np.asarray(R_wfy, dtype=np.float64)
    anchors_f = np.asarray(anchors, dtype=np.float64)
    N, T_obs, _ = X.shape
    K = anchors_f.shape[0]
    assert T_obs == 11, f"X must have T=11, got {T_obs}"

    th_a = np.asarray(theta, dtype=np.float64)        # (N,) for (N,3)
    th_b = th_a[:, None]                               # (N,1) for (N,K,3)

    residual_a = np.zeros((N, 7, 3), dtype=np.float32)
    residual_b = np.zeros((N, K, 7, 3), dtype=np.float32)

    for i in range(7):
        t_wall = -6 + i
        t_idx = t_wall + 10                            # {4..10}
        t_target_idx = t_idx + 2

        base_t = np.asarray(baseline_at_fn(X, t_idx), dtype=np.float64)   # (N,3) world

        if t_target_idx <= 10:
            raw_t2 = X[:, t_target_idx, :]
            delta_a = raw_t2 - base_t                                     # (N,3)
            anchor_world_k = (
                base_t[:, None, :] + np.einsum("nij,kj->nki", R_wfy_f, anchors_f)
            )                                                             # (N,K,3)
            delta_b = raw_t2[:, None, :] - anchor_world_k                 # (N,K,3)
        else:
            delta_a = np.zeros((N, 3))
            delta_b = np.zeros((N, K, 3))

        residual_a[:, i, :] = _yaw.to_yaw(delta_a, th_a).astype(np.float32)
        residual_b[:, :, i, :] = _yaw.to_yaw(delta_b, th_b).astype(np.float32)

    residual_a = np.nan_to_num(residual_a, nan=0.0, posinf=1e3, neginf=-1e3)
    residual_b = np.nan_to_num(residual_b, nan=0.0, posinf=1e3, neginf=-1e3)
    return {"residual_a": residual_a, "residual_b": residual_b}


if __name__ == "__main__":
    sys.path.insert(0, str(_THIS.parent.parent))
    rng = np.random.default_rng(20260526)
    N = 8
    X = np.cumsum(rng.standard_normal((N, 11, 3)) * 0.01, axis=1).astype(np.float32)
    theta = _yaw.yaw_from_X(X)
    R_wfy = _yaw.build_R_wfy(theta)
    anchors = rng.standard_normal((14, 3)).astype(np.float32) * 0.01

    def baseline_at(Xf, t_idx):  # dummy: last observed position
        return Xf[:, t_idx, :]

    out = build_residuals(X, theta, R_wfy, anchors, baseline_at)
    assert out["residual_a"].shape == (N, 7, 3)
    assert out["residual_b"].shape == (N, 14, 7, 3)
    assert np.all(out["residual_a"][:, 5:7, :] == 0.0)
    assert np.all(out["residual_b"][:, :, 5:7, :] == 0.0)
    for k, v in out.items():
        assert not np.isnan(v).any() and not np.isinf(v).any()
    print(f"[smoke] residual_builder(yaw) OK — a={out['residual_a'].shape}, b={out['residual_b'].shape}, zero-pad ✓")
