"""plan-015 c4 (G1, E1) — Feature A only (F0 residual direct, 12D).

baseline = plan-015 G0 reproduce OOF = 0.6425 (plan-014 best_stack carry).
E1 = baseline + A feature. 합격: ΔOOF ≥ +0.005.

Usage:
    python analysis/plan-015/g1_e1_feature_A.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import plan014_paradigm as pp  # noqa: E402
from src.pb_0_6822.plan015_train import run_kfold_oof_v2  # noqa: E402


BASELINE_OOF = 0.6425  # plan-015 G0 reproduce (= plan-014 G5 best_stack)
DELTA_THRESHOLD = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-015/g1_e1.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan015_e1"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-015 G1] loading data ...", flush=True)
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32); Y = Y.astype(np.float32)

    # plan-014 best_stack config + Feature A
    cfg = pp.TrainConfig(
        name="E1_A", K=9, encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
    )
    feature_flags = {"A": True, "B": False, "C": False, "D": False}
    print(f"[plan-015 G1] E1 config: feature_flags={feature_flags}, input_dim=12", flush=True)

    f0_function = pp.Plan014F0Function()

    def progress(fold_id, res):
        print(f"  fold {fold_id}: val_hit={res['best_val_hit']:.4f} dcm={res['dcm']:.4f} "
              f"epoch={res['best_epoch']}", flush=True)

    kfold_res = run_kfold_oof_v2(ids, X, Y, cfg, feature_flags, f0_function=f0_function,
                                   progress_cb=progress)
    oof = kfold_res["overall_oof_hit_1cm"]
    delta = oof - BASELINE_OOF
    mean_dcm = float(np.mean([fr["dcm"] for fr in kfold_res["fold_results"]]))

    # 합격 분기
    if delta >= DELTA_THRESHOLD:
        status = "positive"
    elif delta >= 0:
        status = "marginal"
    else:
        status = "negative"
    next_anchor_inheritance = oof if status != "negative" else BASELINE_OOF

    elapsed = time.time() - t_start
    artifact = {
        "exp_id": "H043_g1_e1_feature_A",
        "feature_flags": feature_flags,
        "input_dim": 12,
        "baseline_oof": BASELINE_OOF,
        "e1_oof": oof,
        "delta_oof_vs_baseline": delta,
        "mean_dcm": mean_dcm,
        "status": status,
        "g1_passed": status in {"positive", "marginal"},
        "g2_anchor": next_anchor_inheritance,  # G2 anchor inheritance
        "fold_results": kfold_res["fold_results"],
        "elapsed_seconds": elapsed,
        "plan_version": "v2.4",
    }

    # Save run dir
    args.run_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.run_dir / "oof_pred.npy", kfold_res["oof_pred"])
    (args.run_dir / "fold_log.json").write_text(json.dumps({
        "exp_id": artifact["exp_id"],
        "feature_flags": feature_flags,
        "e1_oof": oof,
        "fold_results": kfold_res["fold_results"],
    }, indent=2, ensure_ascii=False))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

    print(f"\n[plan-015 G1] === E1 (A) result ===", flush=True)
    print(f"  baseline_oof = {BASELINE_OOF:.4f}", flush=True)
    print(f"  e1_oof       = {oof:.4f}", flush=True)
    print(f"  delta_oof    = {delta:+.4f} (threshold +{DELTA_THRESHOLD})", flush=True)
    print(f"  status       = **{status}**", flush=True)
    print(f"  g1_passed    = {artifact['g1_passed']}, G2 anchor = {next_anchor_inheritance:.4f}",
          flush=True)
    print(f"  elapsed      = {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"[plan-015 G1] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
