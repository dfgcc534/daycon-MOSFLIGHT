"""plan-016 c4 (STAGE 2, G2, Path B) — monitor=val_loss cumulative.

§6 spec:
  - seeds = G1 carry [20260514, 20260515, 20260516, 20260517, 20260518] × 5 fold = 25 models.
  - baseline: G1 carry (E0c K-Means K=9 + boundary_weight_on, F0 frozen) +
    **monitor=val_loss** (patience=5, plan-014 v3.10 c3.10 spec carry).
  - OOF aggregation: §5.2 carry (좌표 mean over seeds → 5-fold concat → hit@1cm).
  - Test ensemble: 25 model 좌표 mean.
  - dacon-submit 1회.

G2 합격 (§6.3):
  - OOF Δ ≥ +0.005 vs G1 OOF
  - LB Δ ≥ +0.005 vs G1 LB
  - 둘 다 pass → positive, G3 진행
  - 한 쪽 만 pass → marginal, G3 진행 + warn
  - 둘 다 fail (= 둘 다 Δ < 0, §3.2 footnote) → drop Path B → G3-G5 base=G1

Usage:
    python analysis/plan-016/g2_path_b.py
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


PATH_A_SEEDS = [20260514, 20260515, 20260516, 20260517, 20260518]
G2_DELTA_THRESHOLD = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-json", type=Path, default=Path("analysis/plan-016/g1_path_a.json"))
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-016/g2_path_b.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan016_g2_path_b"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    # G1 baseline carry
    g1 = json.loads(args.g1_json.read_text())
    g1_oof = g1["overall_oof_hit_1cm"]
    g1_lb = g1.get("lb_score")
    print(f"[plan-016 G2] G1 baseline: OOF={g1_oof:.4f} LB={g1_lb}", flush=True)

    t_start = time.time()
    print("[plan-016 G2] loading data ...", flush=True)
    ids_train, X_train = load_all_samples("train")
    ids_test, X_test = load_all_samples("test")
    label_ids, Y_train = load_labels()
    assert ids_train == label_ids
    X_train = X_train.astype(np.float32); Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print(f"[plan-016 G2] N_train={X_train.shape[0]} N_test={X_test.shape[0]}", flush=True)

    config_base = pp.TrainConfig(
        name="path_b_val_loss", K=9, encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=PATH_A_SEEDS[0],
        monitor="val_loss",  # ★ Path B 의 single 변경 lever
    )

    def progress(si, f, seed, res, elapsed):
        print(f"  seed={seed} fold={f}: val_loss={res['best_val_loss']:.4f} "
              f"val_hit={res['best_val_hit']:.4f} dcm={res['dcm']:.4f} "
              f"epoch={res['best_epoch']}/{args.epochs} elapsed={elapsed:.1f}s", flush=True)

    print(f"\n[plan-016 G2] === 5-seed × 5-fold = 25 models (monitor=val_loss) ===", flush=True)
    print(f"  seeds={PATH_A_SEEDS}", flush=True)
    ensemble_result = pe.run_multiseed_kfold(
        ids_train, X_train, Y_train, ids_test, X_test,
        config_base=config_base, seeds=PATH_A_SEEDS,
        f0_function=pp.Plan014F0Function(),
        progress_cb=progress,
    )

    overall_oof = ensemble_result["overall_oof_hit_1cm"]
    delta_oof = overall_oof - g1_oof
    oof_pass = delta_oof >= G2_DELTA_THRESHOLD
    per_seed_oof = ensemble_result["per_seed_oof_hit_1cm"]
    fold_oof = ensemble_result["fold_oof_hit_per_fold"]

    # best_epoch std (L3 closure target ≤ 3)
    best_epochs = [r["best_epoch"] for r in ensemble_result["fold_results"]]
    best_epoch_std = float(np.std(best_epochs))

    print(f"\n[plan-016 G2] === G2 final ===", flush=True)
    print(f"  per-seed concat OOF hit@1cm = {per_seed_oof}", flush=True)
    print(f"  per-fold (seed-mean) OOF hit = {fold_oof}", flush=True)
    print(f"  multi-seed concat OOF hit@1cm = {overall_oof:.4f}", flush=True)
    print(f"  Δ vs G1 0.6452 = {delta_oof:+.4f} (threshold +{G2_DELTA_THRESHOLD})", flush=True)
    print(f"  best_epoch std = {best_epoch_std:.2f} (L3 closure target ≤ 3)", flush=True)
    print(f"  OOF Δ pass = {oof_pass} (LB Δ pass deferred to dacon-submit)", flush=True)

    # Submission
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
        "exp_id": "H051_g2_path_b_val_loss",
        "plan_version": "v1.5",
        "config_base": asdict(config_base),
        "seeds": PATH_A_SEEDS,
        "n_models": 25,
        "per_seed_oof_hit_1cm": per_seed_oof,
        "fold_oof_hit_per_fold": fold_oof,
        "overall_oof_hit_1cm": overall_oof,
        "g1_oof": g1_oof,
        "g1_lb": g1_lb,
        "delta_oof_vs_g1": delta_oof,
        "delta_threshold": G2_DELTA_THRESHOLD,
        "oof_pass": oof_pass,
        "lb_pass": None,
        "status": None,
        "best_epoch_std": best_epoch_std,
        "l3_closure_target_3": best_epoch_std <= 3,
        "fold_results": ensemble_result["fold_results"],
        "submission_path": str(submission_path),
        "elapsed_total_seconds": elapsed_total,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\n[plan-016 G2] elapsed_total={elapsed_total:.1f}s", flush=True)
    print(f"[plan-016 G2] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
