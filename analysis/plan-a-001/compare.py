"""KR001/KR002 paired permutation 비교 (plan-a-001 §5).

- KR001 vs F0 (plan-020 baseline): G_repro 유의성.
- KR002 vs KR001: G_yaw lift (입력 yaw 회전 순수 기여).
paired sign-flip permutation 10000 resample, p threshold 0.05.

Usage: python analysis/plan-a-001/compare.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels  # noqa: E402

_spec = importlib.util.spec_from_file_location("bf0", _REPO / "analysis" / "plan-020" / "baseline_f0.py")
_bf0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf0)

R_HIT = 0.01


def hit_mask(pred, y, thr=R_HIT):
    return (np.linalg.norm(pred - y, axis=-1) <= thr)


def paired_perm(hit_b, hit_a, n_resample=10000, seed=0):
    """paired sign-flip permutation. stat = mean(hit_b - hit_a). return (delta, p)."""
    d = hit_b.astype(np.float64) - hit_a.astype(np.float64)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_resample, d.shape[0]))
    null = (signs * d[None, :]).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return float(obs), p


def main():
    z1 = np.load(_THIS / "results_kr001.npz")
    z2 = np.load(_THIS / "results_kr002.npz")
    y = z1["y"]
    hit_kr001 = z1["per_sample_hit"]
    hit_kr002 = z2["per_sample_hit"]
    assert np.allclose(y, z2["y"]), "KR001/KR002 y 정렬 불일치"

    # F0 baseline per-sample hit (동일 X)
    _, X = load_all_samples("train")
    X = X[: y.shape[0]]
    f0_pred = _bf0.f0_baseline(X, end_idx=X.shape[1] - 1)
    hit_f0 = hit_mask(f0_pred, y)

    # KR001 vs F0 (G_repro 유의성)
    d_kr001_f0, p_kr001_f0 = paired_perm(hit_kr001, hit_f0)
    # KR002 vs KR001 (G_yaw)
    d_yaw, p_yaw = paired_perm(hit_kr002, hit_kr001)

    # plan §5: positive/negative 는 Δ 임계 + p<0.05 둘 다 필요. 비유의는 neutral.
    if d_yaw >= 0.002 and p_yaw < 0.05:
        g_yaw = "positive"
    elif d_yaw <= -0.002 and p_yaw < 0.05:
        g_yaw = "negative"
    else:
        g_yaw = "neutral"  # 유의한 lift 없음 (Δ<임계 또는 p≥0.05)

    res = dict(
        hit_f0=float(hit_f0.mean()),
        hit_kr001=float(hit_kr001.mean()),
        hit_kr002=float(hit_kr002.mean()),
        kr001_vs_f0=dict(delta=d_kr001_f0, p=p_kr001_f0),
        kr002_vs_kr001=dict(delta=d_yaw, p=p_yaw),
        g_yaw_band=g_yaw,
        n=int(y.shape[0]),
    )
    print(json.dumps(res, indent=2))
    (_THIS / "compare_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[G_repro] KR001 {res['hit_kr001']:.4f} vs F0 {res['hit_f0']:.4f}: "
          f"Δ={d_kr001_f0:+.4f} p={p_kr001_f0:.4g}")
    print(f"[G_yaw]  KR002 {res['hit_kr002']:.4f} vs KR001 {res['hit_kr001']:.4f}: "
          f"Δ={d_yaw:+.4f} p={p_yaw:.4g} → {g_yaw}")
    return res


if __name__ == "__main__":
    main()
