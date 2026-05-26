"""plan-c-001 §4.1 — F0(perp=0.0) baseline predictor + floor 산출.

f0_perp0 = p0 + 1.98·v_last + 1.20·acc_par_vec  (plan-020 F0 에서 PERP −0.20 → 0.0).
plan-020 baseline_f0 와 *perp 계수만* 다르도록 ε=1e-9·분해로직 동일 박제
(plan-020 baseline_f0.py 원본 미수정 — 본 모듈 별도).

Exports:
  D1, PAR              — F0 계수 (PERP=0.0 고정, 항 제거)
  f0_perp0(X, end_idx) — (N,3) world +80ms 예측 (numpy 비학습 baseline)
  compute_floor(...)   — f0_perp0 의 hit_1cm/hit_1p5cm floor (pooled + per-fold)

__main__: train 전체 floor → analysis/plan-c-001/f0_perp0_floor.json 박제.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

D1: float = 1.98
PAR: float = 1.20
# PERP = 0.0 (perpendicular 가속 보정항 제거 — 본 plan 의 단일 계수 변경)
R_HIT: float = 0.01
R_HIT_LOOSE: float = 0.015


def f0_perp0(x: np.ndarray, end_idx: int | None = None) -> np.ndarray:
    """F0(perp=0.0). x (N,T,3), end_idx 기본 T-1. returns (N,3).

    식 (§4.1): p0 + 1.98·v_last + 1.20·a_par. a_par = (acc·v̂)v̂, v̂=v_last/(‖v_last‖+ε).
    ε=1e-9 (plan-020 baseline_f0 동일 — perp 항만 0 으로 차이).
    """
    e = (x.shape[1] - 1) if end_idx is None else end_idx
    p0 = x[:, e]
    v_last = x[:, e] - x[:, e - 1]
    v_prev = x[:, e - 1] - x[:, e - 2]
    acc = v_last - v_prev

    speed = np.linalg.norm(v_last, axis=1, keepdims=True)
    tangent = v_last / (speed + 1e-9)
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par_vec = acc_par_scalar * tangent
    # acc_perp_vec = acc - acc_par_vec  # PERP=0.0 → 미사용
    return p0 + D1 * v_last + PAR * acc_par_vec


def _hit(pred: np.ndarray, y: np.ndarray, thr: float = R_HIT) -> float:
    return float((np.linalg.norm(pred - y, axis=-1) <= thr).mean())


def compute_floor(X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray | None = None) -> dict:
    """f0_perp0 의 hit floor. deterministic 이라 fold 무관하나 band 용 per-fold 도 산출."""
    pred = f0_perp0(X)
    out = dict(
        n=int(X.shape[0]),
        hit_1cm_pooled=_hit(pred, y, R_HIT),
        hit_1p5cm_pooled=_hit(pred, y, R_HIT_LOOSE),
    )
    if fold_ids is not None:
        folds = sorted(set(int(f) for f in fold_ids))
        per = [_hit(pred[fold_ids == f], y[fold_ids == f], R_HIT) for f in folds]
        out["hit_1cm_per_fold"] = {int(f): h for f, h in zip(folds, per)}
        out["hit_1cm_fold_mean"] = float(np.mean(per))
        out["hit_1cm_fold_std"] = float(np.std(per))
    return out


def main() -> None:
    from src.io import load_all_samples, load_labels  # noqa: E402
    from src.pb_0_6822.selector import stable_fold_id  # noqa: E402

    ids, X = load_all_samples("train")
    lab_ids, y = load_labels()
    assert ids == lab_ids, "id 정렬 불일치"
    fold_ids = np.array([stable_fold_id(i, 5) for i in ids])
    floor = compute_floor(X, y, fold_ids)
    (_THIS / "f0_perp0_floor.json").write_text(json.dumps(floor, indent=2, ensure_ascii=False))
    print(f"[f0_perp0 floor] hit_1cm(pooled)={floor['hit_1cm_pooled']:.4f} "
          f"fold_mean={floor['hit_1cm_fold_mean']:.4f}±{floor['hit_1cm_fold_std']:.4f} "
          f"hit_1p5cm={floor['hit_1p5cm_pooled']:.4f}", flush=True)


if __name__ == "__main__":
    main()
