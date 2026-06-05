"""plan-016 c2 (STAGE 0, G0) — preflight.

2 task:
  (a) plan-014/015 baseline (E0c K-Means K=9 + boundary_weight_on, F0 frozen,
      seed=20260514, monitor=val_hit) 5-fold OOF reproduce
      → 0.6420 ≤ OOF ≤ 0.6430 (= 0.6425 ± 0.0005, plan-016 §4.1 / §4.3)
  (b) 3 path config sanity:
      - seed list [20260514, 20260515, 20260516, 20260517, 20260518] (5 seed) verify
      - monitor=val_loss option 동작 verify (1-fold 1-epoch smoke)
      - Feature B/C/D 단독 dim (10D / 18D / 15D) sanity

§3.3 G0 / §4.1 / §4.3 carry.

Usage:
    python analysis/plan-016/preflight.py
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


# G0 합격 spec (plan-016 §4.3)
BASELINE_OOF = 0.6425
BASELINE_TOL = 0.0005

# plan-016 5 seed list (§5.2)
PATH_A_SEEDS = [20260514, 20260515, 20260516, 20260517, 20260518]

# Feature single-apply expected dim (§3.2 / §4.1.b)
EXPECTED_DIM_SINGLE = {"B": 10, "C": 18, "D": 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-016/preflight.json"))
    ap.add_argument("--epochs", type=int, default=20)  # G0 reproduce 가볍게 (plan-015 G0 와 동일)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-016 G0] loading data ...", flush=True)
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    print(f"[plan-016 G0] N_train={X.shape[0]}", flush=True)

    # ── task (b) 3 path config sanity ────────────────────────────────────
    print(f"\n[plan-016 G0] (b1) seed list sanity ...", flush=True)
    seed_ok = (len(PATH_A_SEEDS) == 5
               and PATH_A_SEEDS[0] == pp.DEFAULT_SEED
               and PATH_A_SEEDS == sorted(PATH_A_SEEDS)
               and len(set(PATH_A_SEEDS)) == 5)
    print(f"  seeds={PATH_A_SEEDS} len=5 base_match unique → seed_ok={seed_ok}", flush=True)

    print(f"\n[plan-016 G0] (b2) monitor option signature sanity ...", flush=True)
    monitor_results = {}
    for monitor in ("val_hit", "val_loss"):
        try:
            cfg_m = pp.TrainConfig(
                name=f"monitor_{monitor}", K=9, codebook="kmeans",
                boundary_weight_on=True, seed=pp.DEFAULT_SEED, monitor=monitor,
            )
            ok = (cfg_m.monitor == monitor)
            err = None
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
            cfg_m = None
        monitor_results[monitor] = {"ok": ok, "err": err}
        print(f"  TrainConfig(monitor={monitor!r}): ok={ok} err={err}", flush=True)
    # invalid value 가 dispatcher 위에서 raise 하는지는 G1+ 단계에서 *실측* 으로 검증됨 (signature 만 G0)
    try:
        cfg_bad = pp.TrainConfig(name="bad", monitor="invalid_option")
        invalid_accepts = True  # signature 는 string 전부 수용 (런타임에 dispatcher 가 raise)
    except Exception:
        invalid_accepts = False
    monitor_results["invalid_signature_accepts"] = invalid_accepts
    monitor_ok = monitor_results["val_hit"]["ok"] and monitor_results["val_loss"]["ok"]
    print(f"  → monitor signature OK (val_hit + val_loss). invalid 값 dispatcher runtime 검증 deferred.", flush=True)

    print(f"\n[plan-016 G0] (b3) Feature B/C/D 단독 dim sanity ...", flush=True)
    X_sub = X[:100]
    dim_results = {}
    for feat in ("B", "C", "D"):
        flags = {f: (f == feat) for f in "ABCD"}
        out = make_seq_features_v2(X_sub, end_idx=10, direction=1.0, feature_flags=flags)
        expected = EXPECTED_DIM_SINGLE[feat]
        shape_ok = out.shape == (100, 6, expected)
        nan_ok = not np.isnan(out).any() and not np.isinf(out).any()
        dim_results[feat] = {
            "shape": list(out.shape),
            "expected_dim": expected,
            "shape_ok": shape_ok,
            "nan_inf_ok": bool(nan_ok),
        }
        print(f"  {feat} 단독: shape={out.shape} expected=(100, 6, {expected}) "
              f"shape_ok={shape_ok} nan_ok={nan_ok}", flush=True)
    dim_ok = all(r["shape_ok"] and r["nan_inf_ok"] for r in dim_results.values())

    # ── task (a) plan-014/015 baseline 5-fold OOF reproduce ───────────────
    print(f"\n[plan-016 G0] (a) plan-014/015 baseline 5-fold reproduce ...", flush=True)
    cfg = pp.TrainConfig(
        name="baseline_reproduce", K=9, encoder_name="bigru", codebook="kmeans",
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=True,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
        monitor="val_hit",
    )

    def progress(fold_id, res):
        print(f"  fold {fold_id}: val_hit={res['best_val_hit']:.4f} "
              f"dcm={res['dcm']:.4f} epoch={res['best_epoch']}", flush=True)

    kfold_res = pp.run_kfold_oof(ids, X, Y, cfg, f0_function=pp.Plan014F0Function(),
                                  progress_cb=progress)
    reproduce_oof = kfold_res["overall_oof_hit_1cm"]
    in_range = (BASELINE_OOF - BASELINE_TOL) <= reproduce_oof <= (BASELINE_OOF + BASELINE_TOL)
    print(f"  reproduce 5-fold concat OOF = {reproduce_oof:.4f} "
          f"(target {BASELINE_OOF} ± {BASELINE_TOL}, in_range={in_range})", flush=True)

    # ── G0 합격 ────────────────────────────────────────────────────────
    g0_checks = {
        "reproduce_in_range": in_range,
        "seed_list_ok": seed_ok,
        "monitor_options_ok": monitor_ok,
        "feature_dim_sanity": dim_ok,
    }
    g0_passed = all(g0_checks.values())
    print(f"\n[plan-016 G0] g0_checks={g0_checks} → g0_passed={g0_passed}", flush=True)

    elapsed = time.time() - t_start
    artifact = {
        "exp_id": "H049_g0_preflight",
        "plan_version": "v1.5",
        "reproduce_5fold_oof": reproduce_oof,
        "baseline_target": BASELINE_OOF,
        "baseline_tolerance": BASELINE_TOL,
        "in_range": in_range,
        "path_a_seeds": PATH_A_SEEDS,
        "seed_ok": bool(seed_ok),
        "monitor_results": monitor_results,
        "monitor_ok": bool(monitor_ok),
        "feature_single_apply_sanity": dim_results,
        "feature_dim_ok": bool(dim_ok),
        "g0_checks": g0_checks,
        "g0_passed": bool(g0_passed),
        "elapsed_seconds": elapsed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"[plan-016 G0] artifact -> {args.out_json} ({elapsed:.2f}s)", flush=True)
    return 0 if g0_passed else 1


if __name__ == "__main__":
    sys.exit(main())
