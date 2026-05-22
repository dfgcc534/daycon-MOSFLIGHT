"""plan-026 c3 — 3 ablation cell 5-fold OOF runner.

Carry plan-025 LgbmSelectorRowExpanded + run_oof_plan025 loop, swap build_feat
→ build_feat_masked. C1 hparam carry (default).

CLI: python analysis/plan-026/run_oof.py --cell {A1,A2,A3}
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
_PLAN024 = _THIS.parent / "plan-024"
_PLAN025 = _THIS.parent / "plan-025"

for p in (_REPO, _PLAN021, _PLAN022, _PLAN024, _PLAN025):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_bf = _load(_PLAN020 / "baseline_f0.py", "p026_bf")
_p021_build = _load(_PLAN021 / "build_input.py", "p026_p021")
_p022_anchors = _load(_PLAN022 / "anchors.py", "p026_p022_a")
_som = _load(_PLAN022 / "selector_only_model.py", "p026_som")
_qc_mod = _load(_PLAN024 / "quantile_carry.py", "p026_qc")
_p025_run_oof = _load(_PLAN025 / "run_oof.py", "p026_p025_run")

LgbmSelectorRowExpanded = _p025_run_oof.LgbmSelectorRowExpanded

# build_feat_masked
_bmb = _load(_THIS / "block_mask_builder.py", "p026_bmb")
build_feat_masked = _bmb.build_feat_masked
EXCLUSION_MAP = _bmb.EXCLUSION_MAP
EXPECTED_D_MASKED = _bmb.EXPECTED_D_MASKED

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


LGBM_RANDOM_STATE = 20260522
TAU_CLS = 0.001
K_ANCHORS = 14
N_FOLDS = 5
PREREQ_JSON = _PLAN025 / "results_C1.json"


def check_prereq() -> dict:
    """plan-025 G2.C1 결과 carry."""
    assert PREREQ_JSON.exists(), f"prereq_p025_g2_missing: {PREREQ_JSON}"
    with open(PREREQ_JSON) as f:
        d = json.load(f)
    required = ["hit_1cm", "hit_1p5cm", "max_class_ratio"]
    for k in required:
        assert k in d, f"baseline carry missing key: {k}"
    print(f"[prereq] plan-025 C1 baseline: hit_1cm={d['hit_1cm']:.4f} hit_1p5cm={d['hit_1p5cm']:.4f}", flush=True)
    return {
        "hit_1cm": float(d["hit_1cm"]),
        "hit_1p5cm": float(d["hit_1p5cm"]),
        "max_class_ratio": float(d["max_class_ratio"]),
        "runtime_s": float(d.get("runtime_s", 0.0)),
    }


def run_oof_plan026(cell_id: str, n_folds: int = N_FOLDS, seed: int = LGBM_RANDOM_STATE,
                     verbose: bool = True) -> dict:
    assert cell_id in EXCLUSION_MAP, f"unsupported cell: {cell_id}"
    expected_d = EXPECTED_D_MASKED[cell_id]

    t0 = time.time()
    _ids, X = load_all_samples()
    _ids_gt, gt = load_labels()
    assert _ids == _ids_gt, "id mismatch"
    X = X.astype(np.float32)
    gt = gt.astype(np.float32)
    N = X.shape[0]
    anchors = _p022_anchors.ANCHORS_A6
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)

    oof_pred = np.zeros((N, 3), dtype=np.float32)
    oof_probs_sel = np.zeros((N, K_ANCHORS), dtype=np.float32)
    per_fold = []

    for fold in range(n_folds):
        t_fold = time.time()
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        X_train, X_test = X[train_idx], X[test_idx]
        gt_train = gt[train_idx]
        N_te = len(test_idx)

        R_wfn_train = _p021_build.build_frenet_basis_3d(X_train, end_idx=10)
        R_wfn_test = _p021_build.build_frenet_basis_3d(X_test, end_idx=10)
        F0_train = _bf.f0_baseline(X_train, end_idx=10).astype(np.float32)
        F0_test = _bf.f0_baseline(X_test, end_idx=10).astype(np.float32)

        qc = _qc_mod.build(X_train, R_wfn_train)

        feat_train = build_feat_masked(X_train, anchors, _bf.f0_baseline, qc, cell_id)
        feat_test = build_feat_masked(X_test, anchors, _bf.f0_baseline, qc, cell_id)
        assert feat_train.shape[1] == expected_d, f"feat_train dim mismatch: {feat_train.shape[1]} != {expected_d}"

        q_train = _som.build_soft_label_with_tau(gt_train, R_wfn_train, F0_train, anchors, TAU_CLS)

        model = LgbmSelectorRowExpanded(K=K_ANCHORS)
        try:
            model.clf.set_params(random_state=seed)
        except Exception:
            pass
        model.fit(feat_train, q_train)

        probs_test_expanded = model.clf.predict_proba(feat_test)
        anchor_idx = np.tile(np.arange(K_ANCHORS), N_te)
        probs_sel = probs_test_expanded[np.arange(N_te * K_ANCHORS), anchor_idx].reshape(N_te, K_ANCHORS)
        probs_sel = probs_sel / probs_sel.sum(axis=1, keepdims=True)

        residual_frenet = (probs_sel[:, :, None] * anchors[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_test, residual_frenet)
        final_pred = F0_test + residual_world

        oof_pred[test_idx] = final_pred
        oof_probs_sel[test_idx] = probs_sel

        err = np.linalg.norm(final_pred - gt[test_idx], axis=1)
        hit_1cm_fold = float((err <= 0.01).mean())
        hit_1p5cm_fold = float((err <= 0.015).mean())
        runtime_fold = time.time() - t_fold
        per_fold.append({
            "fold": fold, "n_test": int(N_te),
            "hit_1cm": hit_1cm_fold, "hit_1p5cm": hit_1p5cm_fold,
            "max_class_ratio_fold": float(probs_sel.mean(axis=0).max()),
            "runtime_s_fold": runtime_fold,
        })
        if verbose:
            print(f"[{cell_id}] fold {fold} hit_1cm={hit_1cm_fold:.4f} hit_1p5cm={hit_1p5cm_fold:.4f} runtime={runtime_fold:.1f}s", flush=True)

    err = np.linalg.norm(oof_pred - gt, axis=1)
    hit_1cm = float((err <= 0.01).mean())
    hit_1p5cm = float((err <= 0.015).mean())
    max_class_ratio = float(oof_probs_sel.mean(axis=0).max())
    runtime_s = time.time() - t0

    if verbose:
        print(f"[{cell_id}] FINAL hit_1cm={hit_1cm:.4f} hit_1p5cm={hit_1p5cm:.4f} "
              f"max_class_ratio={max_class_ratio:.3f} runtime={runtime_s:.1f}s", flush=True)

    return {
        "cell_id": cell_id,
        "D_masked": expected_d,
        "hit_1cm": hit_1cm,
        "hit_1p5cm": hit_1p5cm,
        "max_class_ratio": max_class_ratio,
        "per_fold": per_fold,
        "runtime_s": runtime_s,
        "seed": seed,
        "n_folds": n_folds,
        "tau_cls": TAU_CLS,
        "K": K_ANCHORS,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, choices=["A1", "A2", "A3", "prereq"])
    p.add_argument("--seed", type=int, default=LGBM_RANDOM_STATE)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.cell == "prereq":
        result = check_prereq()
        out_path = args.out or str(_THIS / "baseline_carry.json")
    else:
        result = run_oof_plan026(args.cell, seed=args.seed, verbose=True)
        out_path = args.out or str(_THIS / f"results_{args.cell}.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
    print(f"saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
