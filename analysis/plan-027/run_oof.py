"""plan-027 — 3-way ensemble (p022 + p023 + p026_A2).

각 base predictor 의 5-fold OOF sample-level final_pred 박제 후 weight 조합.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
_PLAN020 = _THIS.parent / "plan-020"
_PLAN021 = _THIS.parent / "plan-021"
_PLAN022 = _THIS.parent / "plan-022"
_PLAN023 = _THIS.parent / "plan-023"
_PLAN024 = _THIS.parent / "plan-024"
_PLAN025 = _THIS.parent / "plan-025"
_PLAN026 = _THIS.parent / "plan-026"

for p in (_REPO, _PLAN021, _PLAN022, _PLAN023, _PLAN024, _PLAN025, _PLAN026):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bf = _load(_PLAN020 / "baseline_f0.py", "p027_bf")
_p021_build = _load(_PLAN021 / "build_input.py", "p027_p021")
_p022_anchors = _load(_PLAN022 / "anchors.py", "p027_p022_a")
_p023_anchors = _load(_PLAN023 / "anchors_largeN.py", "p027_p023_a")
_som = _load(_PLAN022 / "selector_only_model.py", "p027_som")
_p025_run = _load(_PLAN025 / "run_oof.py", "p027_p025_run")
_p026_bmb = _load(_PLAN026 / "block_mask_builder.py", "p027_p026_bmb")
_qc_mod = _load(_PLAN024 / "quantile_carry.py", "p027_qc")

LgbmSelectorRowExpanded = _p025_run.LgbmSelectorRowExpanded

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402

N_FOLDS = 5
TAU_CLS = 0.001
LGBM_SEED_P022 = 20260519
LGBM_SEED_P026_A2 = 20260522


def _predict_oof_p022_style(X, gt, ids, anchors, seed, verbose=True, tag="p022"):
    """plan-022 LgbmSelectorOnly (sample-level X 170D + 내부 row-expand) 패턴 OOF."""
    N = X.shape[0]
    K = anchors.shape[0]
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)

    # 170D LGBM input (plan-022 동일)
    common = _p021_build.build_input_common(X, _bf.f0_baseline)
    extra = _p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    X_lgbm = np.concatenate([
        common["L1"].reshape(N, 99),
        common["L2"].reshape(N, 21),
        common["L4"].reshape(N, 14),
        extra,
    ], axis=1).astype(np.float32)
    R_wfn = common["R_wfn"]
    F0 = common["pred_F0_world"].astype(np.float32)

    oof_pred = np.zeros((N, 3), dtype=np.float32)
    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr = np.where(folds != fold)[0]
        te = np.where(folds == fold)[0]
        X_tr_lgbm = X_lgbm[tr]
        X_te_lgbm = X_lgbm[te]
        q_tr = _som.build_soft_label_with_tau(gt[tr], R_wfn[tr], F0[tr], anchors, TAU_CLS)
        model = _som.LgbmSelectorOnly(K=K)
        try:
            model.clf.set_params(random_state=seed)
        except Exception:
            pass
        model.fit(X_tr_lgbm, q_tr)
        # plan-022 LgbmSelectorOnly.fit 가 내부 row-expand. predict_proba 는 sample-level (N, K)
        probs_te = model.clf.predict_proba(X_te_lgbm)  # (N_te, K)
        residual_frenet = (probs_te[:, :, None] * anchors[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn[te], residual_frenet)
        oof_pred[te] = F0[te] + residual_world
        if verbose:
            print(f"[{tag}] fold {fold} runtime={time.time()-t_fold:.1f}s", flush=True)

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    if verbose:
        print(f"[{tag}] FINAL hit_1cm={hit_1cm:.4f} hit_1p5cm={hit_1p5cm:.4f}", flush=True)
    return oof_pred, hit_1cm, hit_1p5cm


def _predict_oof_p026_A2_style(X, gt, ids, seed=LGBM_SEED_P026_A2, verbose=True):
    """plan-026 A2 (build_feat_masked 1058D + LgbmSelectorRowExpanded) 패턴 OOF."""
    N = X.shape[0]
    anchors = _p022_anchors.ANCHORS_A6
    K = anchors.shape[0]
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)

    oof_pred = np.zeros((N, 3), dtype=np.float32)
    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr = np.where(folds != fold)[0]
        te = np.where(folds == fold)[0]
        X_tr, X_te = X[tr], X[te]
        gt_tr = gt[tr]
        R_wfn_tr = _p021_build.build_frenet_basis_3d(X_tr, end_idx=10)
        R_wfn_te = _p021_build.build_frenet_basis_3d(X_te, end_idx=10)
        F0_tr = _bf.f0_baseline(X_tr, end_idx=10).astype(np.float32)
        F0_te = _bf.f0_baseline(X_te, end_idx=10).astype(np.float32)
        qc = _qc_mod.build(X_tr, R_wfn_tr)
        feat_tr = _p026_bmb.build_feat_masked(X_tr, anchors, _bf.f0_baseline, qc, "A2")
        feat_te = _p026_bmb.build_feat_masked(X_te, anchors, _bf.f0_baseline, qc, "A2")
        q_tr = _som.build_soft_label_with_tau(gt_tr, R_wfn_tr, F0_tr, anchors, TAU_CLS)
        model = LgbmSelectorRowExpanded(K=K)
        try:
            model.clf.set_params(random_state=seed)
        except Exception:
            pass
        model.fit(feat_tr, q_tr)
        N_te = len(te)
        probs_te_exp = model.clf.predict_proba(feat_te)  # (N_te*K, K)
        anchor_idx = np.tile(np.arange(K), N_te)
        probs_sel = probs_te_exp[np.arange(N_te*K), anchor_idx].reshape(N_te, K)
        probs_sel = probs_sel / probs_sel.sum(axis=1, keepdims=True)
        residual_frenet = (probs_sel[:, :, None] * anchors[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_te, residual_frenet)
        oof_pred[te] = F0_te + residual_world
        if verbose:
            print(f"[p026_A2] fold {fold} runtime={time.time()-t_fold:.1f}s", flush=True)

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    if verbose:
        print(f"[p026_A2] FINAL hit_1cm={hit_1cm:.4f} hit_1p5cm={hit_1p5cm:.4f}", flush=True)
    return oof_pred, hit_1cm, hit_1p5cm


def run_ensemble(verbose=True) -> dict:
    """전체 ensemble run — base predictor 3개 OOF + weight sweep."""
    t0 = time.time()
    ids, X = load_all_samples()
    ids_gt, gt = load_labels()
    assert ids == ids_gt
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)

    # ── G1: base predictor reproduce ──
    print("[G1] p022 OOF reproduce...", flush=True)
    pred_p022, hit_p022, hit_p022_1p5 = _predict_oof_p022_style(
        X, gt, ids, _p022_anchors.ANCHORS_A6, LGBM_SEED_P022, verbose=verbose, tag="p022")
    assert 0.6523 <= hit_p022 <= 0.6533, f"p022 reproduce drift: {hit_p022}"

    print("[G1] p023 OOF reproduce...", flush=True)
    pred_p023, hit_p023, hit_p023_1p5 = _predict_oof_p022_style(
        X, gt, ids, _p023_anchors.ANCHORS_B4, LGBM_SEED_P022, verbose=verbose, tag="p023")
    assert 0.6527 <= hit_p023 <= 0.6537, f"p023 reproduce drift: {hit_p023}"

    print("[G1] p026_A2 OOF reproduce (1058D)...", flush=True)
    pred_p026, hit_p026, hit_p026_1p5 = _predict_oof_p026_A2_style(
        X, gt, ids, LGBM_SEED_P026_A2, verbose=verbose)
    # plan-026 A2 결과 0.6509 carry — tight band
    assert 0.6499 <= hit_p026 <= 0.6519, f"p026_A2 reproduce drift: {hit_p026}"

    base_summary = {
        "p022": {"hit_1cm": hit_p022, "hit_1p5cm": hit_p022_1p5},
        "p023": {"hit_1cm": hit_p023, "hit_1p5cm": hit_p023_1p5},
        "p026_A2": {"hit_1cm": hit_p026, "hit_1p5cm": hit_p026_1p5},
    }
    base_max = max(hit_p022, hit_p023, hit_p026)
    pass_threshold = base_max + 0.002
    print(f"[G1] base_max={base_max:.4f}, PASS threshold={pass_threshold:.4f}", flush=True)

    # ── G2: ensemble cells ──
    def _hit(pred):
        err = np.linalg.norm(pred - gt, axis=1)
        return float((err <= 0.01).mean()), float((err <= 0.015).mean())

    # E1: equal 3-way
    pred_E1 = (pred_p022 + pred_p023 + pred_p026) / 3.0
    hit_E1_1cm, hit_E1_1p5 = _hit(pred_E1)
    print(f"[E1] equal-3way: hit_1cm={hit_E1_1cm:.4f} hit_1p5cm={hit_E1_1p5:.4f}", flush=True)

    # E2: equal 2-way (p022 + p023)
    pred_E2 = (pred_p022 + pred_p023) / 2.0
    hit_E2_1cm, hit_E2_1p5 = _hit(pred_E2)
    print(f"[E2] equal-2way(p022+p023): hit_1cm={hit_E2_1cm:.4f} hit_1p5cm={hit_E2_1p5:.4f}", flush=True)

    # E3: weighted grid sweep (3-way simplex 5-point + 2-way edge)
    grid = [
        (0.5, 0.5, 0.0),
        (0.4, 0.4, 0.2),
        (0.4, 0.3, 0.3),
        (0.3, 0.3, 0.4),
        (0.34, 0.33, 0.33),
        (0.6, 0.4, 0.0),
        (0.4, 0.6, 0.0),
        (0.3, 0.5, 0.2),
        (0.5, 0.3, 0.2),
    ]
    E3_results = []
    best_E3_hit = -1.0
    best_E3_weight = None
    best_E3_1p5 = -1.0
    for w in grid:
        wa, wb, wc = w
        assert abs(wa + wb + wc - 1.0) < 1e-6
        pred_w = wa * pred_p022 + wb * pred_p023 + wc * pred_p026
        h, h15 = _hit(pred_w)
        E3_results.append({"weights": list(w), "hit_1cm": h, "hit_1p5cm": h15})
        if h > best_E3_hit:
            best_E3_hit = h
            best_E3_weight = list(w)
            best_E3_1p5 = h15
    print(f"[E3] best weight={best_E3_weight}, hit_1cm={best_E3_hit:.4f} hit_1p5cm={best_E3_1p5:.4f}", flush=True)

    # ── G3 verdict ──
    cells = [("E1", hit_E1_1cm), ("E2", hit_E2_1cm), ("E3", best_E3_hit)]
    best_cell, best_hit = max(cells, key=lambda x: x[1])
    if best_hit > pass_threshold:
        verdict = "PASS"
        band = "positive"
    elif best_hit >= base_max:
        verdict = "marginal"
        band = "partial"
    else:
        verdict = "negative_ensemble"
        band = "negative"

    runtime_s = time.time() - t0
    return {
        "base_predictors": base_summary,
        "base_max_hit_1cm": base_max,
        "pass_threshold": pass_threshold,
        "cells": {
            "E1": {"hit_1cm": hit_E1_1cm, "hit_1p5cm": hit_E1_1p5, "weights": [1/3, 1/3, 1/3]},
            "E2": {"hit_1cm": hit_E2_1cm, "hit_1p5cm": hit_E2_1p5, "weights": [0.5, 0.5, 0.0]},
            "E3": {"hit_1cm": best_E3_hit, "hit_1p5cm": best_E3_1p5, "best_weights": best_E3_weight, "grid": E3_results},
        },
        "best_cell": best_cell,
        "best_hit_1cm": best_hit,
        "G3_verdict": verdict,
        "G3_band": band,
        "runtime_s": runtime_s,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(_THIS / "results_ensemble.json"))
    args = p.parse_args()
    result = run_ensemble(verbose=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
    print(f"saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
