"""plan-014 c7 (STAGE 3, G3) — Phase 2 axis ablation 5 (E1~E5) on G2 winner.

각 sub-exp = G2 winner config (E0c K-Means/Frenet) 위 *지정 axis 1 변경*. fold=0.
F0 frozen 공통 (Plan014F0Function, plan-006 carry).

E1 frame swap conditional: winner=E0c → E1a Frenet (anchor) / E1b world (= K-Means
   centroid 그대로, train_residuals_world 이미 world)
E2 K density: E2a K=5 / E2c K=9 / E2d K=13 (E2b K=7 = anchor)
E3 τ scan: argmax + 0.01 + 0.1 + 0.3 + 1.0 (E3c τ=0.03 = anchor)
E4 loss swap: E4b = + L7 hinge (anchor = E4a baseline)
E5 reg head on/off: E5a = reg off (anchor = E5b on)

G3 합격: 5 axis 중 1+ max(ΔOOF) ≥ +0.005.

Usage:
    python analysis/plan-014/g3_phase2_axis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import plan014_paradigm as pp  # noqa: E402


def fold0_split(ids: list[str], X: np.ndarray, Y: np.ndarray):
    fold_of = np.array([pp.stable_hash_fold(s) for s in ids])
    val_mask = fold_of == 0
    train_mask = ~val_mask
    return X[train_mask], Y[train_mask], X[val_mask], Y[val_mask], train_mask, val_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-014/g3_phase2.json"))
    ap.add_argument("--epochs", type=int, default=pp.DEFAULT_EPOCHS)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-014 G3 v4] loading data ...", flush=True)
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    X_tr, Y_tr, X_va, Y_va, train_mask, val_mask = fold0_split(ids, X, Y)
    print(f"[plan-014 G3] N_train_fold0={X_tr.shape[0]}, N_val_fold0={X_va.shape[0]}", flush=True)

    # G2 winner config carry
    g2 = json.loads(Path("analysis/plan-014/g2_phase1.json").read_text())
    winner_id = g2["winner_id"]
    winner_codebook = g2["winner_codebook"]
    print(f"[plan-014 G3] G2 winner = {winner_id} ({winner_codebook})", flush=True)

    f0_function = pp.Plan014F0Function()

    anchor_cfg = pp.TrainConfig(
        name="anchor", K=7, encoder_name="bigru", codebook=winner_codebook,
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=False,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
    )

    def run_sub_exp(sub_id: str, cfg: pp.TrainConfig, anchors_override=None,
                    R_train_override=None, R_val_override=None):
        t = time.time()
        res = pp.train_one_fold(
            cfg, fold_id=0,
            X_train=X_tr, Y_train=Y_tr,
            X_val=X_va, Y_val=Y_va,
            f0_function=f0_function,
            anchors_local=anchors_override,
            R_train=R_train_override,
            R_val=R_val_override,
        )
        elapsed = time.time() - t
        return res, elapsed

    # ── anchor (G2 winner fold-0 re-run for reference) ─────────────────────
    print(f"\n[plan-014 G3] === anchor (winner {winner_id}, fold-0) ===", flush=True)
    anchor_res, anchor_elapsed = run_sub_exp("anchor", anchor_cfg)
    anchor_oof = anchor_res["best_val_hit"]
    anchor_dcm = anchor_res["dcm"]
    print(f"  anchor: val_hit={anchor_oof:.4f} dcm={anchor_dcm:.4f} epoch={anchor_res['best_epoch']} "
          f"elapsed={anchor_elapsed:.1f}s", flush=True)
    anchor_anchors_local = np.array(anchor_res["anchors_local"], dtype=np.float32)

    # ── sub-exp 정의 (anchor 제외 net new training) ────────────────────────
    sub_exps_to_run = []

    # E1 frame swap conditional (winner=E0c → world variant only)
    if winner_id == "E0a":
        sub_exps_to_run.append(("E1_skip", "skipped_frame_n/a", "E1", None))
    else:
        # world variant = absolute codebook 으로 swap (= K-Means centroid 를 world frame 그대로)
        cfg_e1b = replace(anchor_cfg, name="E1b", codebook="absolute")
        sub_exps_to_run.append(("E1b", "world_frame_swap", "E1", cfg_e1b))

    # E2 K density: K=5/9/13 (anchor = K=7)
    for sub_id_var, K in [("E2a", 5), ("E2c", 9), ("E2d", 13)]:
        cfg_e2 = replace(anchor_cfg, name=sub_id_var, K=K)
        sub_exps_to_run.append((sub_id_var, f"K_density_K={K}", "E2", cfg_e2))

    # E4 loss swap (E4b = + L7 hinge)
    cfg_e4b = replace(anchor_cfg, name="E4b", use_hinge=True)
    sub_exps_to_run.append(("E4b", "loss_swap_+L7_hinge", "E4", cfg_e4b))

    # E5 reg head off (E5a)
    cfg_e5a = replace(anchor_cfg, name="E5a", use_reg_head=False)
    sub_exps_to_run.append(("E5a", "reg_head_off", "E5", cfg_e5a))

    # ── Run training sub-exps ─────────────────────────────────────────────
    results_per_sub_exp: list[dict] = []
    for sub_id_var, name, axis, cfg in sub_exps_to_run:
        if cfg is None:
            results_per_sub_exp.append({
                "sub_exp_id": sub_id_var, "name": name, "axis": axis,
                "skipped": True, "skip_reason": "frame_axis_n/a (winner=E0a)",
            })
            continue
        print(f"\n[plan-014 G3] === {sub_id_var} ({name}) ===", flush=True)
        try:
            res, elapsed = run_sub_exp(sub_id_var, cfg)
            sub_oof = res["best_val_hit"]
            sub_dcm = res["dcm"]
            delta = sub_oof - anchor_oof
            print(f"  {sub_id_var}: val_hit={sub_oof:.4f} (Δ={delta:+.4f}) dcm={sub_dcm:.4f} "
                  f"epoch={res['best_epoch']} elapsed={elapsed:.1f}s", flush=True)
            results_per_sub_exp.append({
                "sub_exp_id": sub_id_var, "name": name, "axis": axis,
                "val_hit": sub_oof, "dcm": sub_dcm, "delta_oof_vs_anchor": delta,
                "best_epoch": res["best_epoch"],
                "elapsed_seconds": elapsed,
            })
        except Exception as e:
            print(f"  {sub_id_var} ERROR: {e}", flush=True)
            results_per_sub_exp.append({
                "sub_exp_id": sub_id_var, "name": name, "axis": axis,
                "error": str(e),
            })

    # ── E3 τ scan (separate per-τ training for fairness) ──────────────────
    print(f"\n[plan-014 G3] === E3 τ scan ===", flush=True)
    e3_taus = [("E3a", 1e-9), ("E3b", 0.01), ("E3d", 0.1), ("E3e", 0.3), ("E3f", 1.0)]
    for sub_id_var, tau in e3_taus:
        cfg_e3 = replace(anchor_cfg, name=sub_id_var, temperature=tau if tau > 0 else 1e-9)
        try:
            res, elapsed = run_sub_exp(sub_id_var, cfg_e3)
            sub_oof = res["best_val_hit"]
            sub_dcm = res["dcm"]
            delta = sub_oof - anchor_oof
            dilution_warn = (tau >= 0.3) and (sub_dcm < 0.001)
            print(f"  {sub_id_var} τ={tau}: val_hit={sub_oof:.4f} (Δ={delta:+.4f}) "
                  f"dcm={sub_dcm:.4f} epoch={res['best_epoch']} "
                  f"elapsed={elapsed:.1f}s dilution_collapse={dilution_warn}", flush=True)
            results_per_sub_exp.append({
                "sub_exp_id": sub_id_var, "name": f"tau_scan_{tau}", "axis": "E3",
                "tau": tau, "val_hit": sub_oof, "dcm": sub_dcm,
                "delta_oof_vs_anchor": delta, "best_epoch": res["best_epoch"],
                "dilution_collapse_warn": dilution_warn,
                "excluded_from_max_delta": dilution_warn,
                "elapsed_seconds": elapsed,
            })
        except Exception as e:
            print(f"  {sub_id_var} ERROR: {e}", flush=True)
            results_per_sub_exp.append({
                "sub_exp_id": sub_id_var, "name": f"tau_scan_{tau}", "axis": "E3",
                "error": str(e),
            })

    # ── axis summary ──────────────────────────────────────────────────────
    print(f"\n[plan-014 G3] === axis summary ===", flush=True)
    axis_summary: dict[str, dict] = {}
    for axis in ["E1", "E2", "E3", "E4", "E5"]:
        axis_subs = [r for r in results_per_sub_exp
                      if r.get("axis") == axis and "val_hit" in r
                      and not r.get("excluded_from_max_delta", False)]
        if not axis_subs:
            axis_summary[axis] = {"n_sub_exp": 0, "skipped": True}
            continue
        deltas = {r["sub_exp_id"]: r["delta_oof_vs_anchor"] for r in axis_subs}
        max_delta = max(deltas.values())
        best_sub_id = max(deltas, key=deltas.get)
        positive = max_delta >= 0.005
        axis_summary[axis] = {
            "n_sub_exp": len(axis_subs),
            "deltas": deltas,
            "max_delta": max_delta,
            "best_sub_id": best_sub_id,
            "best_val_hit": next(r["val_hit"] for r in axis_subs if r["sub_exp_id"] == best_sub_id),
            "positive_lever": positive,
        }
        print(f"  {axis}: n={len(axis_subs)}, max_delta={max_delta:+.4f}, "
              f"best={best_sub_id}, positive={positive}", flush=True)

    positive_axes = [a for a, info in axis_summary.items() if info.get("positive_lever")]
    G3_passed = len(positive_axes) >= 1
    G3_warn = None if G3_passed else "g3_marginal_only"

    elapsed_total = time.time() - t_start
    artifact = {
        "exp_id": "H039_g3_phase2_axis5",
        "winner_id": winner_id,
        "anchor_oof_fold0": anchor_oof,
        "anchor_dcm_fold0": anchor_dcm,
        "n_sub_exp": len([r for r in results_per_sub_exp if "val_hit" in r]),
        "axis_summary": axis_summary,
        "positive_axes": positive_axes,
        "G3_passed": G3_passed,
        "G3_warn": G3_warn,
        "results_per_sub_exp": results_per_sub_exp,
        "elapsed_total_seconds": elapsed_total,
        "plan_version": "v4.5",
        "f0_frozen_baseline": "plan-006_frenet_par120_perp_neg020",
        "training_config": {
            "anchor_codebook": winner_codebook,
            "encoder_name": "bigru",
            "K": 7,
            "lr": pp.DEFAULT_LR,
            "batch_size": pp.DEFAULT_BATCH,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": pp.DEFAULT_SEED,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

    print(f"\n[plan-014 G3] === final ===", flush=True)
    print(f"  positive_axes={positive_axes}, G3_passed={G3_passed}, warn={G3_warn}", flush=True)
    print(f"  elapsed_total={elapsed_total:.1f}s ({elapsed_total/60:.2f} min)", flush=True)
    print(f"[plan-014 G3] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
