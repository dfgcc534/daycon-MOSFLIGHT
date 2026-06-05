"""plan-017 c4 (STAGE 2, G2) — 7×7×7 voxel CE head 5-seed × 5-fold = 25 models.

§6 spec carry: plan-016 G1 config 위 voxel CE head 만 교체 (anchor codebook 무력화).
- seeds = [20260514..20260518]
- baseline config: K=9 (unused), boundary_weight_on=True, F0 frozen plan-006, monitor=val_hit, 5-fold
- OOF aggregation: 좌표 mean over seeds → 5-fold concat → hit@1cm (plan-016 §5.2 carry)
- Test ensemble: 25 model 좌표 mean

Usage:
    python analysis/plan-017/g2_voxel_ce.py
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
from src.pb_0_6822 import plan017_voxel_ce as v17  # noqa: E402


PATH_A_SEEDS = [20260514, 20260515, 20260516, 20260517, 20260518]
BASELINE_OOF = 0.6452     # plan-016 G1
BASELINE_LB = 0.6638      # plan-016 G1
G2_DELTA_THRESHOLD = 0.003


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-017/g2_voxel_ce.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan017_g2_voxel_ce"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-017 G2] loading data ...", flush=True)
    ids_train, X_train = load_all_samples("train")
    ids_test, X_test = load_all_samples("test")
    label_ids, Y_train = load_labels()
    assert ids_train == label_ids
    X_train = X_train.astype(np.float32); Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print(f"[plan-017 G2] N_train={X_train.shape[0]} N_test={X_test.shape[0]}", flush=True)
    print(f"[plan-017 G2] Voxel grid: {v17.VOXEL_DEPTH}^3 = {v17.VOXEL_TOTAL} classes, ±{v17.HALF * v17.VOXEL_WIDTH * 100:.0f}cm window", flush=True)

    config_base = pp.TrainConfig(
        name="voxel_ce", K=9,  # K unused in voxel paradigm but required by TrainConfig
        encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=PATH_A_SEEDS[0],
        monitor="val_hit",  # plan-016 G1 carry
    )

    def progress(si, f, seed, res, elapsed):
        print(f"  seed={seed} fold={f}: val_hit={res['best_val_hit']:.4f} "
              f"val_loss={res['best_val_loss']:.4f} epoch={res['best_epoch']}/{args.epochs} "
              f"elapsed={elapsed:.1f}s", flush=True)

    print(f"\n[plan-017 G2] === 5-seed × 5-fold = 25 models ===", flush=True)
    print(f"  seeds={PATH_A_SEEDS}", flush=True)
    result = v17.run_multiseed_kfold_voxel(
        ids_train, X_train, Y_train, ids_test, X_test,
        config_base=config_base, seeds=PATH_A_SEEDS,
        f0_function=pp.Plan014F0Function(),
        progress_cb=progress,
    )

    overall_oof = result["overall_oof_hit_1cm"]
    delta_oof = overall_oof - BASELINE_OOF
    oof_pass = delta_oof >= G2_DELTA_THRESHOLD
    per_seed = result["per_seed_oof_hit_1cm"]
    fold_oof = result["fold_oof_hit_per_fold"]

    print(f"\n[plan-017 G2] === G2 final ===", flush=True)
    print(f"  per-seed concat OOF = {per_seed}", flush=True)
    print(f"  per-fold (seed-mean) OOF = {fold_oof}", flush=True)
    print(f"  multi-seed concat OOF = {overall_oof:.4f}", flush=True)
    print(f"  Δ vs G1 baseline {BASELINE_OOF} = {delta_oof:+.4f} (threshold +{G2_DELTA_THRESHOLD})", flush=True)
    print(f"  OOF Δ pass = {oof_pass} (LB Δ pending dacon-submit)", flush=True)

    # Submission
    args.run_dir.mkdir(parents=True, exist_ok=True)
    sample_sub = pd.read_csv("data/sample_submission.csv")
    sample_ids = sample_sub["id"].tolist()
    id_to_idx = {sid: i for i, sid in enumerate(ids_test)}
    test_pred = result["test_pred"]
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
        "exp_id": "H059_g2_voxel_ce",
        "plan_version": "v1.4",
        "voxel_depth": v17.VOXEL_DEPTH,
        "voxel_total": v17.VOXEL_TOTAL,
        "voxel_width_m": v17.VOXEL_WIDTH,
        "voxel_window_half_m": v17.HALF * v17.VOXEL_WIDTH,
        "config_base": asdict(config_base),
        "seeds": PATH_A_SEEDS,
        "n_models": 25,
        "per_seed_oof_hit_1cm": per_seed,
        "fold_oof_hit_per_fold": fold_oof,
        "overall_oof_hit_1cm": overall_oof,
        "baseline_oof": BASELINE_OOF,
        "baseline_lb": BASELINE_LB,
        "delta_oof_vs_g1": delta_oof,
        "delta_threshold": G2_DELTA_THRESHOLD,
        "oof_pass": oof_pass,
        "lb_pass": None,
        "lb_score": None,
        "status": None,
        "fold_results": result["fold_results"],
        "submission_path": str(submission_path),
        "elapsed_total_seconds": elapsed_total,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\n[plan-017 G2] elapsed_total={elapsed_total:.1f}s ({elapsed_total/60:.2f} min)", flush=True)
    print(f"[plan-017 G2] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
