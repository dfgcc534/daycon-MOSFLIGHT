"""plan-014 c6 (STAGE 2, G2) — Phase 1 codebook bake-off v4 (E0a/E0b/E0c).

3 sub-exp × 5-fold OOF on same corrector arch + loss + τ + seed.
F0 frozen 공통 (plan-006 frenet_par120_perp_neg020 = d1=1.98/par=1.20/perp=−0.20).
Only anchor codebook varies — E0a Absolute / E0b Frenet-Orthogonal / E0c K-Means.

Winner = argmax OOF, tie-break (gap < 0.005): priority E0a > E0b > E0c.

G2 합격:
  - winner_oof ≥ 0.60 (encoder forward path 망가지지 않음)
  - winner DCM ≥ 0.002 (encoder mode 신호 살아있음)

Usage:
    python analysis/plan-014/g2_phase1_bakeoff.py \
        --out-json analysis/plan-014/g2_phase1.json \
        --run-dir  runs/baseline
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


SUB_EXPS = [
    ("E0a", "absolute"),
    ("E0b", "frenet_orthogonal"),
    ("E0c", "kmeans"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-014/g2_phase1.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline"))
    ap.add_argument("--epochs", type=int, default=pp.DEFAULT_EPOCHS)
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print(f"[plan-014 G2 v4] loading train data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    print(f"[plan-014 G2] N_train={X.shape[0]}. F0=Plan014F0Function (frozen, plan-006 constants)",
          flush=True)

    f0_function = pp.Plan014F0Function()  # frozen plan-006 carry

    results_per_sub_exp = []
    all_oof = {}
    all_dcm = {}

    for sub_id, codebook in SUB_EXPS:
        t_sub = time.time()
        print(f"\n[plan-014 G2] === {sub_id} ({codebook}) ===", flush=True)
        cfg = pp.TrainConfig(
            name=sub_id, K=7, encoder_name="bigru", codebook=codebook,
            use_reg_head=True, use_hinge=False,
            temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=False,
            lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
            epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
        )
        sub_run_dir = args.run_dir / f"plan014_g2_{sub_id}"
        sub_run_dir.mkdir(parents=True, exist_ok=True)

        def progress(fold_id, res):
            print(f"  [{sub_id}] fold {fold_id}: val_hit={res['best_val_hit']:.4f} "
                  f"(init {res['initial_val_hit']:.4f}, Δ={res['best_val_hit']-res['initial_val_hit']:+.4f}) "
                  f"epoch={res['best_epoch']}/{args.epochs} dcm={res['dcm']:.4f}", flush=True)

        kfold_res = pp.run_kfold_oof(ids, X, Y, cfg, f0_function=f0_function, progress_cb=progress)
        sub_elapsed = time.time() - t_sub

        # save fold-level log
        (sub_run_dir / "fold_log.json").write_text(json.dumps({
            "sub_id": sub_id,
            "codebook": codebook,
            "overall_oof_hit_1cm": kfold_res["overall_oof_hit_1cm"],
            "fold_results": kfold_res["fold_results"],
            "elapsed_seconds": sub_elapsed,
        }, indent=2, ensure_ascii=False))

        # save OOF predictions
        np.save(sub_run_dir / "oof_pred.npy", kfold_res["oof_pred"])

        # aggregate DCM
        mean_dcm = float(np.mean([fr["dcm"] for fr in kfold_res["fold_results"]]))
        oof = kfold_res["overall_oof_hit_1cm"]
        all_oof[sub_id] = oof
        all_dcm[sub_id] = mean_dcm

        results_per_sub_exp.append({
            "sub_exp_id": sub_id,
            "codebook_id": codebook,
            "K": 7,
            "n_train": int(X.shape[0]),
            "overall_oof_hit_1cm": oof,
            "mean_dcm": mean_dcm,
            "fold_results": kfold_res["fold_results"],
            "elapsed_seconds": sub_elapsed,
        })

        print(f"  [{sub_id}] OOF={oof:.4f}, mean DCM={mean_dcm:.4f}, elapsed={sub_elapsed:.1f}s",
              flush=True)

    # Winner + tie-break
    print(f"\n[plan-014 G2] === winner decision ===", flush=True)
    winner_id = max(all_oof, key=all_oof.get)
    sorted_vals = sorted(all_oof.values(), reverse=True)
    winner_oof = sorted_vals[0]
    second_oof = sorted_vals[1]
    gap = winner_oof - second_oof
    tie_break_applied = False
    if gap < 0.005:
        priority = ["E0a", "E0b", "E0c"]
        tied = [k for k, v in all_oof.items() if v >= winner_oof - 0.005]
        winner_id_new = next(k for k in priority if k in tied)
        if winner_id_new != winner_id:
            print(f"  tie-break: winner {winner_id} → {winner_id_new} (gap={gap:.4f}, "
                  f"priority=Absolute>Frenet>KMeans)", flush=True)
            winner_id = winner_id_new
            tie_break_applied = True
        winner_oof = all_oof[winner_id]

    winner_dcm = all_dcm[winner_id]
    winner_codebook = next(c for s, c in SUB_EXPS if s == winner_id)
    winner_frame = {
        "absolute": "world",
        "frenet_orthogonal": "frenet",
        "kmeans": "frenet",
    }[winner_codebook]
    second_id = next(k for k in all_oof if k != winner_id and all_oof[k] == second_oof)

    G1_BASELINE_ANCHOR = 0.60
    winner_above_anchor = winner_oof >= G1_BASELINE_ANCHOR
    winner_dcm_ok = winner_dcm >= 0.002
    G2_passed = winner_above_anchor and winner_dcm_ok
    G2_warn = None
    if not winner_above_anchor:
        G2_warn = "g2_severe_underperform"
    elif not winner_dcm_ok:
        G2_warn = "dcm_collapse"

    elapsed_total = time.time() - t_start
    artifact = {
        "exp_id": "H038_g2_phase1_bakeoff",
        "winner_id": winner_id,
        "winner_anchor_source": {
            "absolute": "compute_anchors_absolute",
            "frenet_orthogonal": "compute_anchors_frenet_orthogonal",
            "kmeans": "compute_anchors_kmeans",
        }[winner_codebook],
        "winner_codebook": winner_codebook,
        "winner_frame": winner_frame,
        "winner_K": 7,
        "second_id": second_id,
        "winner_oof": winner_oof,
        "winner_dcm": winner_dcm,
        "second_oof": second_oof,
        "gap": gap,
        "tie_break_applied": tie_break_applied,
        "all_sub_exp_oof": all_oof,
        "directional_commit_magnitudes": all_dcm,
        "winner_above_anchor": winner_above_anchor,
        "winner_dcm_ok": winner_dcm_ok,
        "G2_passed": G2_passed,
        "G2_warn": G2_warn,
        "G1_baseline_anchor": G1_BASELINE_ANCHOR,
        "elapsed_total_seconds": elapsed_total,
        "plan_version": "v4.5",
        "f0_frozen_baseline": "plan-006_frenet_par120_perp_neg020 (d1=1.98, par=1.20, perp=-0.20)",
        "training_config": {
            "encoder_name": "bigru",
            "K": 7,
            "lr": pp.DEFAULT_LR,
            "batch_size": pp.DEFAULT_BATCH,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": pp.DEFAULT_SEED,
            "temperature": 0.03,
            "use_reg_head": True,
            "use_hinge": False,
        },
        "results_per_sub_exp": results_per_sub_exp,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

    print(f"\n[plan-014 G2] === summary ===", flush=True)
    for sid in ["E0a", "E0b", "E0c"]:
        marker = " ★" if sid == winner_id else ""
        print(f"  {sid}: OOF={all_oof[sid]:.4f}, DCM={all_dcm[sid]:.4f}{marker}", flush=True)
    print(f"  winner={winner_id} ({winner_codebook}/{winner_frame}), gap={gap:.4f}, "
          f"tie_break={tie_break_applied}", flush=True)
    print(f"  G2_passed={G2_passed} (winner_oof>={G1_BASELINE_ANCHOR}: {winner_above_anchor}; "
          f"winner_dcm>=0.002: {winner_dcm_ok}), warn={G2_warn}", flush=True)
    print(f"  elapsed_total={elapsed_total:.1f}s ({elapsed_total/60:.2f} min)", flush=True)
    print(f"[plan-014 G2] artifact -> {args.out_json}", flush=True)

    return 0 if G2_passed else 1


if __name__ == "__main__":
    sys.exit(main())
