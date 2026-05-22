"""plan-025 c3 — build_feat_1080 + compress_seq_8stat (spec §4.2 + §6.1).

block ① plan-022 170D carry +
block ② cand_builder ctx 128D (sample-level) +
block ③ cand_builder per-anchor 22D +
block ④ seq_builder 95×7 → per-channel 8-stat 압축 760D
= **1080D per row** (sample × anchor row-expand).

row order = sample-major (row i*K + k = sample i, anchor k).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# ── importlib carry (spec §4.3) ───────────────────────────────────────
_THIS = Path(__file__).resolve().parent              # analysis/plan-025/
_REPO = _THIS.parent.parent                           # repo root
_PLAN020 = _THIS.parent / "plan-020"
_PLAN021 = _THIS.parent / "plan-021"
_PLAN022 = _THIS.parent / "plan-022"
_PLAN024 = _THIS.parent / "plan-024"

# transitive import 보조 (plan-021/024 module 내부의 `from src.io ...` 등 대비)
for p in (_REPO, _PLAN021, _PLAN024):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bf = _load(_PLAN020 / "baseline_f0.py", "p025_bf")
_p021_build = _load(_PLAN021 / "build_input.py", "p025_p021_build")
_p022_anchors = _load(_PLAN022 / "anchors.py", "p025_p022_anchors")
_cand_mod = _load(_PLAN024 / "cand_builder.py", "p025_cand")
_seq_mod = _load(_PLAN024 / "seq_builder.py", "p025_seq")
_anchor_vocab_mod = _load(_PLAN024 / "anchor_vocab.py", "p025_av")
_quantile_mod = _load(_PLAN024 / "quantile_carry.py", "p025_qc")

from src.pb_0_6822.selector import fit_regime_bins, assign_regimes  # noqa: E402


# ── constants (spec §4.2 lock-in) ─────────────────────────────────────
BLOCK_DIMS: dict[str, int] = {
    "block1_p022": 170,
    "block2_ctx": 128,
    "block3_per_anchor": 22,
    "block4_seq_stat": 760,
    "total_per_row": 1080,
}
STAT_NAMES: list[str] = ["last", "first", "mean", "std", "slope", "max", "min", "range"]

K_ANCHORS = 14
T_SEQ = 7                              # past step count (t∈{4..10})
N_SEQ_CHANNELS = 95
N_STATS = 8
MULTIWINDOW_TRIM_PATH = str(_PLAN024 / "multiwindow_trim.json")


# ── §6.1 block ④ — per-channel 8-stat compression ─────────────────────
def compress_seq_8stat(seq: np.ndarray) -> np.ndarray:
    """seq (N, T=7, C=95) → (N, 760) per-channel 8-stat stack.

    stat order (stat-major): [last, first, mean, std, slope, max, min, range].
    slope = closed-form linear regression coefficient over t-grid = np.arange(7).
    """
    assert seq.ndim == 3, f"seq must be 3D, got shape={seq.shape}"
    N, T, C = seq.shape
    assert T == T_SEQ and C == N_SEQ_CHANNELS, f"expected (N, {T_SEQ}, {N_SEQ_CHANNELS}), got {seq.shape}"
    seq = seq.astype(np.float32)

    last_v = seq[:, -1, :]                                   # (N, C)
    first_v = seq[:, 0, :]                                   # (N, C)
    mean_v = seq.mean(axis=1)                                # (N, C)
    std_v = seq.std(axis=1)                                  # (N, C)

    # slope: closed-form linear regression coefficient
    t = np.arange(T, dtype=np.float32)
    t_dev = t - t.mean()                                     # (T,)
    denom = float((t_dev ** 2).sum())                        # scalar (T=7 → 28.0)
    seq_dev = seq - mean_v[:, None, :]                       # (N, T, C)
    slope_v = (t_dev[None, :, None] * seq_dev).sum(axis=1) / denom    # (N, C)

    max_v = seq.max(axis=1)                                  # (N, C)
    min_v = seq.min(axis=1)                                  # (N, C)
    range_v = max_v - min_v                                  # (N, C)

    return np.concatenate(
        [last_v, first_v, mean_v, std_v, slope_v, max_v, min_v, range_v],
        axis=1,
    ).astype(np.float32)                                     # (N, C*8 = 760)


# ── §4.2 main builder ────────────────────────────────────────────────
def build_feat_1080(
    X: np.ndarray,
    anchors: np.ndarray,
    f0_baseline_fn: Callable,
    quantiles: dict,
    regimes: np.ndarray | None = None,
) -> np.ndarray:
    """Returns (N*K=14, 1080) row-expanded LGBM input.

    Args:
        X: (N, 11, 3) float32, world frame.
        anchors: (K=14, 3) float32, Frenet coord (= plan-022 ANCHORS_A6).
        f0_baseline_fn: callable (X, end_idx) -> (N, 3) world frame (= plan-020 carry).
        quantiles: QuantileCarry dict (= plan-024 quantile_carry.build output).
        regimes: optional (N,) int64. None → 자체 계산 (fit_regime_bins + assign_regimes).

    row order: sample-major. row i*K + k = sample i, anchor k.
    """
    N = X.shape[0]
    K = anchors.shape[0]
    assert K == K_ANCHORS, f"K={K} != {K_ANCHORS}"
    X = X.astype(np.float32)

    # ── block ① plan-022 170D (sample-level) ────────────────────────
    common = _p021_build.build_input_common(X, f0_baseline_fn)
    extra = _p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    block1 = np.concatenate(
        [
            common["L1"].reshape(N, 99),
            common["L2"].reshape(N, 21),
            common["L4"].reshape(N, 14),
            extra,
        ],
        axis=1,
    ).astype(np.float32)                                     # (N, 170)
    assert block1.shape == (N, 170), f"block1 shape mismatch: {block1.shape}"

    R_wfn = common["R_wfn"]
    pred_F0_world = common["pred_F0_world"]

    # regimes (caller 가 안 주면 자체 fit + assign)
    if regimes is None:
        regime_bins = fit_regime_bins(X, end_idx=10)
        regimes = assign_regimes(X, end_idx=10, bins=regime_bins)
    assert regimes.shape == (N,)

    # ── block ②③ — cand_builder (N, 14, 150) slice ────────────────
    cand_feat = _cand_mod.build(
        X, R_wfn, pred_F0_world, anchors, f0_baseline_fn, regimes, quantiles,
        multiwindow_trim_path=MULTIWINDOW_TRIM_PATH, regime_count=18,
    )
    assert cand_feat.shape == (N, K, 150), f"cand_feat shape mismatch: {cand_feat.shape}"
    block2 = cand_feat[:, 0, 12:140].astype(np.float32)      # (N, 128) ctx broadcast
    block3 = np.concatenate(
        [cand_feat[:, :, 0:3], cand_feat[:, :, 3:12], cand_feat[:, :, 140:150]],
        axis=2,
    ).astype(np.float32)                                     # (N, 14, 22) per-anchor

    # ── block ④ — seq_builder (N, 7, 95) → 8-stat 760 ─────────────
    seq_feat = _seq_mod.build(X, R_wfn, anchors, f0_baseline_fn, quantiles)
    assert seq_feat.shape == (N, T_SEQ, N_SEQ_CHANNELS), f"seq_feat shape mismatch: {seq_feat.shape}"
    block4 = compress_seq_8stat(seq_feat)                    # (N, 760)

    # ── row-expand: sample-major ───────────────────────────────────
    block1_exp = np.repeat(block1, K, axis=0)                # (N*K, 170)
    block2_exp = np.repeat(block2, K, axis=0)                # (N*K, 128)
    block3_exp = block3.reshape(N * K, 22)                   # (N*K, 22)
    block4_exp = np.repeat(block4, K, axis=0)                # (N*K, 760)

    out = np.concatenate([block1_exp, block2_exp, block3_exp, block4_exp], axis=1)
    assert out.shape == (N * K, 1080), f"out shape mismatch: {out.shape}"
    return out.astype(np.float32)


# ── smoke ───────────────────────────────────────────────────────────
def _smoke() -> None:
    rng = np.random.default_rng(20260522)
    N = 8
    X = rng.standard_normal((N, 11, 3)).astype(np.float32)
    R_wfn = _p021_build.build_frenet_basis_3d(X, end_idx=10)
    qc = _quantile_mod.build(X, R_wfn)
    feat = build_feat_1080(X, _p022_anchors.ANCHORS_A6, _bf.f0_baseline, qc)
    assert feat.shape == (N * K_ANCHORS, 1080), f"smoke failed: {feat.shape}"
    print(f"smoke OK: feat.shape={feat.shape}, dtype={feat.dtype}, nan={np.isnan(feat).any()}, inf={np.isinf(feat).any()}")


if __name__ == "__main__":
    _smoke()
