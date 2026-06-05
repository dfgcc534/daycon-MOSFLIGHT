"""plan-012 c5~c10 (G2) — Phase 2 Core Ablation on winner E0a (5 axis × ~10 sub-exp).

Winner = E0a (Absolute, K=7, τ=0.03, reg head on, CE+huber+hinge).
E1 Frame swap = SKIP (winner=E0a → frame_axis_n/a).

E2 K density (3 추가 sub-exp on K=5/9/13)
E3 τ scan (5 추가 sub-exp on τ ∈ {0.0, 0.01, 0.1, 0.3, 1.0})
E4 Loss swap (1 추가 sub-exp: huber only, no hinge)
E5 Reg head off (1 추가 sub-exp)

각 sub-exp 의 ΔOOF = OOF − OOF(E0a anchor 0.6416). winner E0a 의 hyperparam 외 변경 변수만 차이.
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

from src.pb_0_6822 import ring_classifier as rc                       # noqa: E402
from src.pb_0_6822 import ring_classifier_train as rct                # noqa: E402
from src.pb_0_6822 import selector as base                            # noqa: E402


def absolute_anchors_for_K(K: int, ranking_abs: list[str], radius_m: float = 0.005) -> np.ndarray:
    """E2 K density swap: Absolute codebook 의 K 별 anchor.

    ranking_abs = G0 산출 axis family ranking (예: ["x", "y", "z"]).
    """
    axis_vec = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    dom, second, third = ranking_abs

    if K == 5:
        # center + ±dom + ±second
        vs = [axis_vec[dom], tuple(-c for c in axis_vec[dom]),
              axis_vec[second], tuple(-c for c in axis_vec[second])]
        anchors = np.array([[0, 0, 0]] + [tuple(c * radius_m for c in v) for v in vs], dtype=np.float64)
    elif K == 7:
        anchors = rc.compute_anchors_absolute(radius_m=radius_m)
    elif K == 9:
        # K=7 + ±(dom+second)/√2
        d, s = np.array(axis_vec[dom]), np.array(axis_vec[second])
        diag = (d + s) / np.sqrt(2.0)
        anchors_base = rc.compute_anchors_absolute(radius_m=radius_m)
        diag_anchors = np.array([diag, -diag], dtype=np.float64) * radius_m
        anchors = np.vstack([anchors_base, diag_anchors])
    elif K == 13:
        # K=7 + ±(dom+second)/√2, ±(dom+third)/√2, ±(second+third)/√2
        d, s, t = np.array(axis_vec[dom]), np.array(axis_vec[second]), np.array(axis_vec[third])
        diags = [
            (d + s) / np.sqrt(2.0),
            (d + t) / np.sqrt(2.0),
            (s + t) / np.sqrt(2.0),
        ]
        anchors_base = rc.compute_anchors_absolute(radius_m=radius_m)
        diag_anchors = []
        for dv in diags:
            diag_anchors.append(dv)
            diag_anchors.append(-dv)
        diag_anchors = np.array(diag_anchors, dtype=np.float64) * radius_m
        anchors = np.vstack([anchors_base, diag_anchors])
    else:
        raise ValueError(f"unsupported K: {K}")
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--preflight", type=str, default="analysis/plan-012/preflight.json")
    parser.add_argument("--winner", type=str, default="analysis/plan-012/phase1_winner.json")
    parser.add_argument("--out", type=str, default="analysis/plan-012/phase2_results.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    # ─── load preflight + winner ───
    with open(args.preflight) as f:
        preflight = json.load(f)
    with open(args.winner) as f:
        winner_data = json.load(f)
    ranking_abs = preflight["codebook_oracle_ceilings"]["per_axis_marginal_hit_1cm"]["axis_family_ranking_absolute"]
    anchor_oof = winner_data["winner_oof"]
    print(f"[phase2] winner anchor OOF = {anchor_oof:.4f}, axis ranking = {ranking_abs}", flush=True)

    # ─── data load ───
    root = Path(args.root)
    print(f"[phase2] loading data ...", flush=True)
    train_ids, train_y = base.read_labels(root / "train_labels.csv")
    train_x = base.load_stack(root / "train", train_ids).astype(np.float64)
    train_y = train_y.astype(np.float64)
    N, T, _ = train_x.shape
    end_idx = T - 1
    fold_id = np.array([base.stable_fold_id(sid, args.n_folds) for sid in train_ids], dtype=np.int64)

    F0_pred = rc.f0_predict_frenet_par120_perp_neg020(train_x, end_idx=end_idx)
    R_wfn = rc.build_frenet_basis_3d(train_x, end_idx=end_idx)
    seq_feat = base.make_seq_features(train_x, end_idx=end_idx, direction=1.0).astype(np.float32)

    # ─── sub-exp matrix ───
    sub_exps: list[dict] = []
    # E2: K density swap (3 추가 — K=5/9/13)
    for K in [5, 9, 13]:
        anchors = absolute_anchors_for_K(K, ranking_abs)
        sub_exps.append({
            "id": f"E2.K{K}",
            "axis": "E2_K_density",
            "K": K,
            "anchors_static": anchors,
            "temperature": 0.03, "use_reg_head": True, "use_hinge": True,
            "scorer_arch": "attn_gru",
        })

    # E3: τ scan (5 추가 — τ ∈ {0.0, 0.01, 0.1, 0.3, 1.0})
    anchors_default = rc.compute_anchors_absolute()
    for tau in [0.0, 0.01, 0.1, 0.3, 1.0]:
        sub_exps.append({
            "id": f"E3.tau{tau:.2f}",
            "axis": "E3_temperature",
            "K": 7,
            "anchors_static": anchors_default,
            "temperature": tau, "use_reg_head": True, "use_hinge": True,
            "scorer_arch": "attn_gru",
        })

    # E4: Loss swap (1 추가 — no hinge, huber only)
    sub_exps.append({
        "id": "E4.no_hinge",
        "axis": "E4_loss_swap",
        "K": 7,
        "anchors_static": anchors_default,
        "temperature": 0.03, "use_reg_head": True, "use_hinge": False,
        "scorer_arch": "attn_gru",
    })

    # E5: Reg head off (1 추가)
    sub_exps.append({
        "id": "E5.reg_head_off",
        "axis": "E5_reg_head",
        "K": 7,
        "anchors_static": anchors_default,
        "temperature": 0.03, "use_reg_head": False, "use_hinge": True,
        "scorer_arch": "attn_gru",
    })

    # ─── 학습 loop ───
    results: list[dict] = []
    t0 = time.time()
    for spec in sub_exps:
        ts = time.time()
        r = rct.run_sub_exp(
            sub_exp_id=spec["id"],
            codebook_id="absolute",
            anchors_local_per_fold=None,
            anchors_local_static=spec["anchors_static"],
            R_wfn=None,
            F0_pred=F0_pred,
            train_y=train_y,
            seq_feat=seq_feat,
            fold_id=fold_id,
            val_fold=args.val_fold,
            K=spec["K"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            temperature=spec["temperature"],
            use_reg_head=spec["use_reg_head"],
            use_hinge=spec["use_hinge"],
            scorer_arch=spec["scorer_arch"],
        )
        r["axis"] = spec["axis"]
        r["temperature"] = spec["temperature"]
        r["use_reg_head"] = spec["use_reg_head"]
        r["use_hinge"] = spec["use_hinge"]
        r["delta_oof_vs_anchor"] = r["best_val_hit"] - anchor_oof
        r["elapsed_seconds"] = round(time.time() - ts, 1)
        results.append(r)
        print(f"[{spec['id']}] best={r['best_val_hit']:.4f} ΔOOF={r['delta_oof_vs_anchor']:+.4f} DCM={r['best_dcm']:.5f} ({r['elapsed_seconds']}s)", flush=True)

    # ─── ΔOOF per axis ───
    axis_summary = {}
    for axis in ["E1_frame", "E2_K_density", "E3_temperature", "E4_loss_swap", "E5_reg_head"]:
        if axis == "E1_frame":
            axis_summary[axis] = {"skipped": True, "reason": "winner=E0a → frame_axis_n/a"}
            continue
        axis_subs = [r for r in results if r["axis"] == axis]
        if not axis_subs:
            continue
        deltas = [r["delta_oof_vs_anchor"] for r in axis_subs]
        max_delta = max(deltas)
        best_sub = max(axis_subs, key=lambda r: r["delta_oof_vs_anchor"])
        axis_summary[axis] = {
            "n_sub_exp": len(axis_subs),
            "deltas": {r["sub_exp_id"]: r["delta_oof_vs_anchor"] for r in axis_subs},
            "max_delta": max_delta,
            "best_sub_id": best_sub["sub_exp_id"],
            "best_val_hit": best_sub["best_val_hit"],
            "positive_lever": max_delta >= 0.005,
        }

    # ─── G2 합격 ───
    positive_axes = [ax for ax, summ in axis_summary.items()
                     if isinstance(summ, dict) and summ.get("positive_lever")]
    g2_passed = len(positive_axes) >= 1
    g2_warn = "phase2_no_positive_lever" if not g2_passed else None

    # ─── output ───
    out = {
        "exp_ids": ["H021_phase2-frame_SKIP", "H022_phase2-codebook-K", "H023_phase2-temperature",
                    "H024_phase2-loss", "H025_phase2-reg-head"],
        "winner_id": winner_data["winner_id"],
        "anchor_oof": anchor_oof,
        "n_sub_exp": len(sub_exps),
        "axis_summary": axis_summary,
        "results_per_sub_exp": results,
        "G2_passed": g2_passed,
        "G2_warn": g2_warn,
        "positive_axes": positive_axes,
        "elapsed_total_seconds": round(time.time() - t0, 1),
        "training_config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience,
            "val_fold": args.val_fold, "lr": 3e-4, "weight_decay": 1e-4,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[phase2] wrote {args.out}", flush=True)
    print(f"[phase2] G2 passed: {g2_passed} (positive axes: {positive_axes})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
