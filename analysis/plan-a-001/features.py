"""Feature builders (plan-a-001 §4.2).

notes/LB_0.6780 코드공유.ipynb cell 10/11/16/22 그대로 이식.
- build_seq_t3: (N,11,9) 시계열 (rel/v/a, world frame, ch0-5 directional).
- build_scalar_feats: 21D + build_tier3_extra: 19D = 40D (전부 회전불변 magnitude/cos).
- noise: poly2 / savgol / loo_spline(캐시).
- log1p: 15 base magnitude 컬럼에만.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

DT = 0.040
T_OBS = np.arange(-400, 1, 40) / 1000.0  # (11,) sec

# log1p 적용 base 컬럼 (cell 11). 이진 flag / straightness / turn_cos / log_max_acc 제외.
LOG_COLS = [
    "mean_speed", "max_speed", "speed_std", "mean_acc", "max_acc", "max_jerk",
    "net_disp", "|v_last|", "|a_last|", "|a_recent|", "jerk_last", "jerk_recent",
    "noise_poly2", "noise_savgol", "noise_loo",
]


def cos_safe(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """행별 코사인 유사도 (a,b: (N,3))."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return num / den


# ---------------------------------------------------------------- noise (cell 10)
def noise_poly2(X: np.ndarray, t_obs: np.ndarray = T_OBS) -> np.ndarray:
    V = np.vander(t_obs, 3, increasing=False)
    out = np.zeros(X.shape[0])
    for j in range(3):
        coef = np.linalg.lstsq(V, X[:, :, j].T, rcond=None)[0]
        fit = (V @ coef).T
        out += (X[:, :, j] - fit).std(axis=1)
    return out / 3


def noise_savgol(X: np.ndarray, w: int = 5, p: int = 2) -> np.ndarray:
    Xs = savgol_filter(X, window_length=w, polyorder=p, axis=1)
    return (X - Xs).std(axis=1).mean(axis=-1)


def noise_loo_spline(X: np.ndarray, t_obs: np.ndarray = T_OBS) -> np.ndarray:
    N, T, _ = X.shape
    out = np.zeros(N)
    idx_all = np.arange(T)
    for i in range(N):
        s = 0.0
        for k in range(1, T - 1):
            mask = idx_all != k
            for j in range(3):
                cs = CubicSpline(t_obs[mask], X[i, mask, j])
                s += (X[i, k, j] - cs(t_obs[k])) ** 2
        out[i] = np.sqrt(s / ((T - 2) * 3))
    return out


def compute_noise(
    X: np.ndarray, cache_path: Path | None = None, key: str = "train", with_loo: bool = True
) -> dict[str, np.ndarray]:
    """poly2/savgol/loo 추정. with_loo=False(test) 시 loo=savgol fallback (cell 11).

    cache_path 지정 시 {key}_{poly2,savgol,loo} 를 npz 캐시 (loo 비용 회피).
    """
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path)
        if all(f"{key}_{n}" in z for n in ("poly2", "savgol", "loo")):
            return {n: z[f"{key}_{n}"] for n in ("poly2", "savgol", "loo")}
    poly2 = noise_poly2(X)
    savgol = noise_savgol(X)
    loo = noise_loo_spline(X) if with_loo else savgol.copy()
    res = {"poly2": poly2, "savgol": savgol, "loo": loo}
    if cache_path is not None:
        prev = dict(np.load(cache_path)) if Path(cache_path).exists() else {}
        prev.update({f"{key}_{n}": v for n, v in res.items()})
        np.savez(cache_path, **prev)
    return res


# ---------------------------------------------------------------- scalar (cell 11/16)
def build_scalar_feats(
    X: np.ndarray, noise_p: np.ndarray, noise_s: np.ndarray, noise_loo: np.ndarray | None = None
) -> pd.DataFrame:
    """21D 스칼라. 속도/가속/저크는 물리단위(Δ/DT). 전부 회전불변."""
    delta_ = np.diff(X, axis=1)
    v_ = delta_ / DT
    a_ = np.diff(v_, axis=1) / DT
    jerk_ = np.diff(a_, axis=1) / DT
    sp_ = np.linalg.norm(v_, axis=-1)
    ac_ = np.linalg.norm(a_, axis=-1)
    jk_ = np.linalg.norm(jerk_, axis=-1)
    v_l = v_[:, -1, :]
    a_l = a_[:, -1, :]
    a_r = a_[:, -3:, :].mean(axis=1)
    nd_vec = X[:, -1] - X[:, 0]
    nd = np.linalg.norm(nd_vec, axis=-1)
    pl = np.linalg.norm(delta_, axis=-1).sum(axis=1)
    straight = np.where(pl > 1e-12, nd / np.maximum(pl, 1e-12), 0.0)
    turn = cos_safe(v_l, v_[:, :-1, :].mean(axis=1))
    if noise_loo is None:
        noise_loo = noise_s
    return pd.DataFrame({
        "mean_speed": sp_.mean(1), "max_speed": sp_.max(1),
        "speed_std": sp_.std(1), "mean_acc": ac_.mean(1),
        "max_acc": ac_.max(1), "max_jerk": jk_.max(1),
        "straightness": straight, "net_disp": nd,
        "turn_cos": turn, "|v_last|": np.linalg.norm(v_l, axis=-1),
        "|a_last|": np.linalg.norm(a_l, axis=-1),
        "|a_recent|": np.linalg.norm(a_r, axis=-1),
        "jerk_last": jk_[:, -1], "jerk_recent": jk_[:, -3:].mean(1),
        "noise_poly2": noise_p, "noise_savgol": noise_s, "noise_loo": noise_loo,
        "hard_turn": (turn < 0.5).astype(np.float32),
        "high_speed": (np.linalg.norm(v_l, axis=-1) > 1.0).astype(np.float32),
        "high_acc": (ac_.max(axis=1) > 15).astype(np.float32),
        "log_max_acc": np.log1p(ac_.max(1)),
    })


def build_tier3_extra(X: np.ndarray) -> np.ndarray:
    """19D 추가 스칼라 = rolling_speed_mean×DT (8) + cumulative_path_length (11). 전부 회전불변."""
    disp = np.diff(X, axis=1)
    v = disp / DT
    speed = np.linalg.norm(v, axis=-1)  # (N,10)
    speed_roll = np.stack([speed[:, i:i + 3].mean(axis=1) for i in range(8)], axis=1) * DT  # (N,8)
    disp_norm = np.linalg.norm(disp, axis=-1)
    cum_path = np.concatenate(
        [np.zeros((X.shape[0], 1)), np.cumsum(disp_norm, axis=1)], axis=1
    )  # (N,11)
    return np.concatenate([speed_roll, cum_path], axis=1).astype(np.float32)  # (N,19)


def build_scalar_40d(
    X: np.ndarray, noise_p: np.ndarray, noise_s: np.ndarray, noise_loo: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """최종 40D 스칼라 (21 base + log1p, + 19 tier3). 전부 회전불변. names 동봉."""
    df = build_scalar_feats(X, noise_p, noise_s, noise_loo)
    for c in LOG_COLS:
        df[c] = np.log1p(df[c].to_numpy())
    base = df.to_numpy(dtype=np.float32)  # (N,21)
    tier3 = build_tier3_extra(X)          # (N,19)
    names = list(df.columns) + [f"tier3_{i}" for i in range(tier3.shape[1])]
    return np.concatenate([base, tier3], axis=-1), names  # (N,40)


SCALAR_ROTATION_CLASS = "invariant"  # 40D 전부 magnitude/cos → KR002 회전 비대상


# ---------------------------------------------------------------- seq (cell 22)
SEQ_CHANNELS = ["rel_x", "rel_y", "rel_z", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z"]


def build_seq_t3(X: np.ndarray) -> np.ndarray:
    """진짜 시계열 (N,11,9): ch0-2 rel(X−last), 3-5 v(front zero-pad 1), 6-8 a(front zero-pad 2)."""
    N = X.shape[0]
    rel = X - X[:, -1:, :]
    disp = np.diff(X, axis=1)
    v = disp / DT
    v_padded = np.concatenate([np.zeros((N, 1, 3)), v], axis=1)
    a = np.diff(v, axis=1) / DT
    a_padded = np.concatenate([np.zeros((N, 2, 3)), a], axis=1)
    seq = np.concatenate([rel, v_padded, a_padded], axis=-1)
    return seq.astype(np.float32)


def normalize_seq(arr: np.ndarray, scaler) -> np.ndarray:
    N, T, C = arr.shape
    flat = arr.reshape(-1, C)
    return scaler.transform(flat).astype(np.float32).reshape(N, T, C)
