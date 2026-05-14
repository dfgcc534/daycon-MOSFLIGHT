"""plan-015 c3 (STAGE 0, G0) — preflight.

2 task:
  (a) plan-014 baseline (E0c K-Means K=9 + boundary_weight_on, F0 frozen)
      5-fold OOF reproduce → 0.6425 ± 0.005 일치 확인
  (b) 4 feature (A/B/C/D) single-apply shape sanity (12/10/18/15D)

§3.3 G0 schema 박제.

Usage:
    python analysis/plan-015/preflight.py
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
from src.pb_0_6822.plan015_features import make_seq_features_v2  # noqa: E402


# G0 합격 spec
BASELINE_OOF = 0.6425
BASELINE_TOL = 0.005

# Feature single-apply expected dim (§3.3 G0(b))
EXPECTED_DIM_SINGLE = {"A": 12, "B": 10, "C": 18, "D": 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-015/preflight.json"))
    ap.add_argument("--epochs", type=int, default=20)  # G0 reproduce 가볍게
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-015 G0] loading data ...", flush=True)
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    print(f"[plan-015 G0] N_train={X.shape[0]}", flush=True)

    # ── task (b) feature single-apply shape sanity ────────────────────────
    print(f"\n[plan-015 G0] (b) feature single-apply shape sanity ...", flush=True)
    X_sub = X[:100]  # quick check
    sanity_results = {}
    for feat in ["A", "B", "C", "D"]:
        flags = {f: (f == feat) for f in "ABCD"}
        out = make_seq_features_v2(X_sub, end_idx=10, direction=1.0, feature_flags=flags)
        expected = EXPECTED_DIM_SINGLE[feat]
        shape_ok = out.shape == (100, 6, expected)
        nan_ok = not np.isnan(out).any() and not np.isinf(out).any()
        sanity_results[feat] = {
            "shape": list(out.shape),
            "expected_dim": expected,
            "shape_ok": shape_ok,
            "nan_inf_ok": bool(nan_ok),
        }
        print(f"  {feat} 단독: shape={out.shape} expected=(100, 6, {expected}) "
              f"shape_ok={shape_ok} nan_ok={nan_ok}", flush=True)

    # cumulative dim sanity (additional verify)
    print(f"\n[plan-015 G0] (b+) cumulative dim sanity ...", flush=True)
    cumulative_results = {}
    for stage, flags_str in [("A", "A"), ("A+B", "AB"), ("A+B+C", "ABC"), ("A+B+C+D", "ABCD")]:
        flags = {f: (f in flags_str) for f in "ABCD"}
        out = make_seq_features_v2(X_sub, end_idx=10, direction=1.0, feature_flags=flags)
        expected_cum = {"A": 12, "A+B": 13, "A+B+C": 26, "A+B+C+D": 32}[stage]
        ok = out.shape == (100, 6, expected_cum)
        cumulative_results[stage] = {"shape": list(out.shape), "expected": expected_cum, "ok": ok}
        print(f"  {stage}: shape={out.shape} expected=(100, 6, {expected_cum}) ok={ok}", flush=True)

    all_sanity_ok = (all(r["shape_ok"] and r["nan_inf_ok"] for r in sanity_results.values())
                      and all(r["ok"] for r in cumulative_results.values()))

    # ── task (a) plan-014 baseline 5-fold OOF reproduce ───────────────────
    print(f"\n[plan-015 G0] (a) plan-014 baseline 5-fold reproduce ...", flush=True)
    cfg = pp.TrainConfig(
        name="baseline_reproduce", K=9, encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,  # plan-014 best_stack
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
    )
    f0_function = pp.Plan014F0Function()

    def progress(fold_id, res):
        print(f"  fold {fold_id}: val_hit={res['best_val_hit']:.4f} "
              f"dcm={res['dcm']:.4f} epoch={res['best_epoch']}", flush=True)

    kfold_res = pp.run_kfold_oof(ids, X, Y, cfg, f0_function=f0_function, progress_cb=progress)
    reproduce_oof = kfold_res["overall_oof_hit_1cm"]
    in_range = abs(reproduce_oof - BASELINE_OOF) <= BASELINE_TOL
    print(f"  reproduce 5-fold concat OOF = {reproduce_oof:.4f} (target {BASELINE_OOF} ± {BASELINE_TOL}, "
          f"in_range={in_range})", flush=True)

    # ── G0 합격 ────────────────────────────────────────────────────────
    g0_checks = {
        "reproduce_in_range": in_range,
        "feature_dim_sanity": all_sanity_ok,
    }
    g0_passed = all(g0_checks.values())
    print(f"\n[plan-015 G0] g0_checks={g0_checks} → g0_passed={g0_passed}", flush=True)

    elapsed = time.time() - t_start
    artifact = {
        "exp_id": "H042_g0_preflight",
        "reproduce_5fold_oof": reproduce_oof,
        "baseline_target": BASELINE_OOF,
        "baseline_tolerance": BASELINE_TOL,
        "in_range": in_range,
        "feature_single_apply_sanity": sanity_results,
        "feature_cumulative_sanity": cumulative_results,
        "all_sanity_ok": all_sanity_ok,
        "g0_checks": g0_checks,
        "g0_passed": g0_passed,
        "elapsed_seconds": elapsed,
        "plan_version": "v2.4",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"[plan-015 G0] artifact -> {args.out_json} ({elapsed:.2f}s)", flush=True)
    return 0 if g0_passed else 1


if __name__ == "__main__":
    sys.exit(main())
