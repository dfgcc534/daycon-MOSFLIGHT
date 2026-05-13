"""plan-013 c8 — Phase 3 best stack (autonomous recovery fallback).

spec @ plan-013 §8 + §0.5 L95 phase2_no_positive_lever recovery (a).

상황: Phase 2 의 3 sub-exp 모두 DEFERRED → autonomous recovery 진입.
→ Phase 3 = best Phase 1 baseline (= c4 결과 G1 0.6381) 단독 5-fold + submission.

수행:
1. c4 와 동일 config (use_in_ic=True, use_step4="off", use_25_cand=False) 5-fold
2. 각 fold 의 model 로 test 예측 산출 → 5-fold ensemble (mean)
3. analysis/plan-013/submission.csv 박제

산출:
- analysis/plan-013/phase3_best_stack.json (config + oof + submission path)
- analysis/plan-013/submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.pb_0_6822 import integrated_v3 as iv3
from src.pb_0_6822 import selector as base_sel


def main() -> int:
    parser = argparse.ArgumentParser(description="plan-013 c8 Phase 3 best stack")
    parser.add_argument("--root", default="data")
    parser.add_argument("--p001-dir", default="runs/baseline/P001_pb-0-6822-fullrun")
    parser.add_argument("--out", default="analysis/plan-013/phase3_best_stack.json")
    parser.add_argument("--submission", default="analysis/plan-013/submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = REPO / args.root
    device = (
        ("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    print(f"[phase3] device={device} folds={args.folds} epochs={args.epochs}", flush=True)

    t0 = time.time()
    ids, train_y = base_sel.read_labels(root / "train_labels.csv")
    train_x = base_sel.load_stack(root / "train", ids)
    sample_ids = np.asarray(ids)

    test_ids = base_sel.read_submission_ids(root / "sample_submission.csv")
    test_x = base_sel.load_stack(root / "test", test_ids)
    test_sample_ids = np.asarray(test_ids)
    print(
        f"[phase3] data loaded: N_train={len(train_y)}, N_test={len(test_ids)}, T={train_x.shape[1]}, "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )

    fold_ids = np.array([base_sel.stable_fold_id(sid, args.folds) for sid in sample_ids])
    oof_preds = np.zeros((len(train_y), 3), dtype=np.float32)
    test_preds_per_fold = np.zeros((args.folds, len(test_ids), 3), dtype=np.float32)
    fold_hits: dict[int, float] = {}
    in_ic_drift_flags: dict[int, bool] = {}

    for fold in range(args.folds):
        t_fold = time.time()
        config = {
            "use_in_ic": True,
            "use_step4": "off",
            "use_25_cand": False,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": device,
            "p001_dir": str(REPO / args.p001_dir),
        }
        result = iv3.run_integrated_v3(
            config, fold, train_x, train_y, sample_ids,
            test_x=test_x, test_sample_ids=test_sample_ids,
        )
        val_mask = fold_ids == fold
        oof_preds[val_mask] = result["val_preds"]
        test_preds_per_fold[fold] = result["test_preds"]
        fold_hits[fold] = result["oof_metric"]["hit"]
        in_ic_drift_flags[fold] = (
            result["in_ic_final_hash"] != result["in_ic_init_hash"]
            and result["in_ic_init_hash"] != 0
        )
        print(
            f"[phase3] fold {fold}: hit={fold_hits[fold]:.4f}, "
            f"n_epochs={result['n_epochs_trained']}, "
            f"in_ic_drift={in_ic_drift_flags[fold]}, elapsed={time.time()-t_fold:.1f}s",
            flush=True,
        )

    # OOF concat metric
    oof_hit = float(np.mean(np.linalg.norm(oof_preds - train_y, axis=1) <= 0.01))

    # Test ensemble (mean over 5 folds)
    test_preds_ensemble = test_preds_per_fold.mean(axis=0)

    # submission.csv 박제
    sub_path = REPO / args.submission
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    with sub_path.open("w") as f:
        f.write("id,x,y,z\n")
        for sid, pred in zip(test_ids, test_preds_ensemble):
            f.write(f"{sid},{pred[0]:.6f},{pred[1]:.6f},{pred[2]:.6f}\n")

    # G3 합격: best stack 5-fold OOF ≥ G1 + 0.005 — fallback path 라 G1 자체와 비교 (super-additive X)
    # G1 baseline (c4) = 0.6381 → G3 합격선 (fallback) = G1 + 0.005 = 0.6431
    g1_baseline_oof = 0.6381  # c4 결과 박제
    g3_threshold_fallback = g1_baseline_oof + 0.005
    g3_pass_fallback = oof_hit >= g3_threshold_fallback
    # 단, fallback 경로에서는 super-additive 검증 자체가 의미 X (lever 0개) — informational

    elapsed = time.time() - t0
    summary = {
        "exp_id": "H035_phase3-best-stack-5fold",
        "mode": "fallback (autonomous recovery — phase2_no_positive_lever)",
        "config": {
            "use_in_ic": True,
            "use_step4": "off",
            "use_25_cand": False,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": device,
        },
        "folds_run": list(range(args.folds)),
        "fold_hits": fold_hits,
        "oof_hit_concat": oof_hit,
        "g1_baseline_oof": g1_baseline_oof,
        "g3_threshold_fallback": g3_threshold_fallback,
        "g3_pass_fallback": g3_pass_fallback,
        "in_ic_drift_flags": in_ic_drift_flags,
        "frozen_gru_drift_severe": any(in_ic_drift_flags.values()),
        "elapsed_sec": elapsed,
        "submission_path": str(sub_path.relative_to(REPO)),
        "submission_n_rows": len(test_ids),
        "test_ensemble_mode": "5-fold mean",
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez(
        out_path.with_suffix(".npz"),
        oof_preds=oof_preds, test_preds_ensemble=test_preds_ensemble,
        test_preds_per_fold=test_preds_per_fold, train_y=train_y,
    )
    print(f"\n[phase3] saved: {out_path.relative_to(REPO)}")
    print(f"[phase3] submission: {sub_path.relative_to(REPO)} ({len(test_ids)} rows)")
    print(f"[phase3] OOF concat = {oof_hit:.4f} (fallback G3 threshold = {g3_threshold_fallback:.4f}, "
          f"pass = {g3_pass_fallback})")
    return 0 if g3_pass_fallback else 1


if __name__ == "__main__":
    sys.exit(main())
