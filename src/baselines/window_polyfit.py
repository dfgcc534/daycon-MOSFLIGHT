"""Window polyfit predictor: fit per-axis polynomial on last `window` points,
extrapolate to t = t_target.

Per plan-001 §5, §6.
"""
from __future__ import annotations

import numpy as np

from src.io import TIMESTEPS_MS, T_TARGET_MS


def predict(
    X: np.ndarray,
    window: int,
    degree: int,
    t_target: int = T_TARGET_MS,
    timesteps: np.ndarray = TIMESTEPS_MS,
) -> np.ndarray:
    """Predict (x, y, z) at t = t_target for each sample.

    X: (n, n_t, 3) input trajectories.
    Fits a degree-`degree` polynomial through the last `window` points
    of each axis (least squares), evaluates at t_target.
    """
    if window <= degree:
        raise ValueError(f"window {window} must be > degree {degree}")
    if window > timesteps.size:
        raise ValueError(f"window {window} > available timesteps {timesteps.size}")

    n, _, n_axes = X.shape
    t = timesteps[-window:].astype(np.float64)
    Xw = X[:, -window:, :]

    V_fit = np.vander(t, degree + 1, increasing=False)
    V_eval = np.vander(np.array([float(t_target)]), degree + 1, increasing=False)

    Xw_flat = Xw.transpose(1, 0, 2).reshape(window, n * n_axes)
    coefs, *_ = np.linalg.lstsq(V_fit, Xw_flat, rcond=None)
    return (V_eval @ coefs).reshape(n, n_axes)


def predict_per_axis(
    X: np.ndarray,
    configs_per_axis: list[tuple[int, int]],
    t_target: int = T_TARGET_MS,
    timesteps: np.ndarray = TIMESTEPS_MS,
) -> np.ndarray:
    """configs_per_axis: list of (window, degree) of length 3, one per axis."""
    if len(configs_per_axis) != 3:
        raise ValueError(f"need 3 configs (one per axis), got {len(configs_per_axis)}")
    out = np.empty((X.shape[0], 3), dtype=np.float64)
    for axis, (w, d) in enumerate(configs_per_axis):
        single_axis = X[:, :, axis : axis + 1]
        pred_axis = predict(single_axis, w, d, t_target, timesteps)
        out[:, axis] = pred_axis[:, 0]
    return out
