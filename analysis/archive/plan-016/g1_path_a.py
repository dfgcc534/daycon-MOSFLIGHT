"""plan-016 c3 (STAGE 1, G1, Path A) — multi-seed × multi-fold ensemble.

§5 spec:
  - seeds = [20260514, 20260515, 20260516, 20260517, 20260518] (5 seed) × 5 fold = 25 models.
  - baseline config: E0c K-Means K=9 + boundary_weight_on, F0 frozen plan-006, monitor=val_hit
    (= plan-014/015 best_stack carry).
  - OOF aggregation: 좌표 mean over seeds → 5-fold concat → hit@1cm.
  - Test ensemble: 25 model 의 test prediction 좌표 mean → single submission.
  - dacon-submit 1회.

G1 합격:
  - OOF Δ ≥ +0.005 vs baseline 0.6425 → OOF Δ pass
  - LB Δ ≥ +0.005 vs baseline 0.6628 → LB Δ pass (사용자 dacon-submit 후 측정)
  - 둘 다 pass → positive

Usage:
    python analysis/plan-016/g1_path_a.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import plan014_paradigm as pp  # noqa: E402
from src.pb_0_6822 import plan016_ensemble as pe  # noqa: E402


# plan-016 §5.2
PATH_A_SEEDS = [20260514, 20260515, 20260516, 20260517, 20260518]
BASELINE_OOF = 0.6425
BASELINE_LB = 0.6628
G1_DELTA_THRESHOLD = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-016/g1_path_a.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan016_g1_path_a"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-016 G1] loading data ...", flush=True)
    ids_train, X_train = load_all_samples("train")
    ids_test, X_test = load_all_samples("test")
    label_ids, Y_train = load_labels()
    assert ids_train == label_ids
    X_train = X_train.astype(np.float32); Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print(f"[plan-016 G1] N_train={X_train.shape[0]} N_test={X_test.shape[0]}", flush=True)

    config_base = pp.TrainConfig(
        name="path_a_multiseed", K=9, encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,  # plan-014/015 best_stack
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=PATH_A_SEEDS[0],  # placeholder, overridden per seed
        monitor="val_hit",  # G1 Path A: monitor 미변경 (G2 에서 val_loss)
    )

    def progress(si, f, seed, res, elapsed):
        print(f"  seed={seed} fold={f}: val_hit={res['best_val_hit']:.4f} "
              f"dcm={res['dcm']:.4f} epoch={res['best_epoch']}/{args.epochs} "
              f"elapsed={elapsed:.1f}s", flush=True)

    print(f"\n[plan-016 G1] === 5-seed × 5-fold = 25 models ===", flush=True)
    print(f"  seeds={PATH_A_SEEDS}", flush=True)
    print(f"  config={asdict(config_base)}", flush=True)
    ensemble_result = pe.run_multiseed_kfold(
        ids_train, X_train, Y_train, ids_test, X_test,
        config_base=config_base, seeds=PATH_A_SEEDS,
        f0_function=pp.Plan014F0Function(),
        progress_cb=progress,
    )

    overall_oof = ensemble_result["overall_oof_hit_1cm"]
    delta_oof = overall_oof - BASELINE_OOF
    oof_pass = delta_oof >= G1_DELTA_THRESHOLD
    per_seed_oof = ensemble_result["per_seed_oof_hit_1cm"]
    fold_oof = ensemble_result["fold_oof_hit_per_fold"]

    print(f"\n[plan-016 G1] === G1 final ===", flush=True)
    print(f"  per-seed concat OOF hit@1cm = {per_seed_oof}", flush=True)
    print(f"  per-fold (seed-mean) OOF hit = {fold_oof}", flush=True)
    print(f"  multi-seed concat OOF hit@1cm = {overall_oof:.4f}", flush=True)
    print(f"  Δ vs baseline 0.6425 = {delta_oof:+.4f} (threshold +{G1_DELTA_THRESHOLD})", flush=True)
    print(f"  OOF Δ pass = {oof_pass} (LB Δ pass deferred to dacon-submit)", flush=True)

    # ── Submission write ─────────────────────────────────────────────────
    args.run_dir.mkdir(parents=True, exist_ok=True)
    sample_sub = pd.read_csv("data/sample_submission.csv")
    sample_ids = sample_sub["id"].tolist()
    id_to_idx = {sid: i for i, sid in enumerate(ids_test)}
    test_pred = ensemble_result["test_pred"]
    ordered = np.array([test_pred[id_to_idx[sid]] for sid in sample_ids], dtype=np.float64)
    submission_path = args.run_dir / "submission.csv"
    df = pd.DataFrame({
        "id": sample_ids,
        "x": [f"{v:.6f}" for v in ordered[:, 0]],
        "y": [f"{v:.6f}" for v in ordered[:, 1]],
        "z": [f"{v:.6f}" for v in ordered[:, 2]],
    })
    df.to_csv(submission_path, index=False)
    print(f"\n  submission -> {submission_path}", flush=True)

    elapsed_total = time.time() - t_start
    artifact = {
        "exp_id": "H050_g1_path_a_multiseed",
        "plan_version": "v1.5",
        "config_base": asdict(config_base),
        "seeds": PATH_A_SEEDS,
        "n_seeds": 5,
        "n_folds": pp.N_FOLDS,
        "n_models": 25,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "per_seed_oof_hit_1cm": per_seed_oof,
        "fold_oof_hit_per_fold": fold_oof,
        "overall_oof_hit_1cm": overall_oof,
        "baseline_oof": BASELINE_OOF,
        "baseline_lb": BASELINE_LB,
        "delta_oof": delta_oof,
        "delta_threshold": G1_DELTA_THRESHOLD,
        "oof_pass": oof_pass,
        "lb_pass": None,  # post dacon-submit
        "status": None,   # post dacon-submit (둘 다 pass = positive)
        "fold_results": ensemble_result["fold_results"],
        "submission_path": str(submission_path),
        "elapsed_total_seconds": elapsed_total,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\n[plan-016 G1] elapsed_total={elapsed_total:.1f}s ({elapsed_total/60:.2f} min)", flush=True)
    print(f"[plan-016 G1] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
