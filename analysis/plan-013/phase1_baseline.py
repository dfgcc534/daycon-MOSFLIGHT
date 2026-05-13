"""plan-013 c4 — Phase 1 Baseline Lock-in (plan-004 + In/IC, 5-fold).

spec @ plan-013 §6.

수행: 5-fold concat OOF (use_in_ic=True, use_step4="off", use_25_cand=False).
G1 합격: 5-fold concat OOF hit ≥ 0.65.

산출: analysis/plan-013/phase1_baseline.json + concat OOF hit.

decision-note: integrated_v3._run_phase1_simplified 의 standalone training (P001 frozen selector
scores + residual corrector). plan-004 의 regime/env/pretrain/finetune complexity 제외 —
Δ measurement 위한 minimal pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.pb_0_6822 import integrated_v3 as iv3
from src.pb_0_6822 import selector as base_sel


def main() -> int:
    parser = argparse.ArgumentParser(description="plan-013 c4 Phase 1 baseline (plan-004 + In/IC, 5-fold)")
    parser.add_argument("--root", default="data")
    parser.add_argument("--p001-dir", default="runs/baseline/P001_pb-0-6822-fullrun")
    parser.add_argument("--out", default="analysis/plan-013/phase1_baseline.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-fold", type=int, default=None, help="single-fold smoke test (0-4)")
    args = parser.parse_args()

    root = REPO / args.root
    device = (
        ("cuda:0" if (REPO / ".").exists() and __import__("torch").cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    print(f"[phase1] device={device} folds={args.folds} epochs={args.epochs}", flush=True)

    t0 = time.time()
    # data load
    ids, train_y = base_sel.read_labels(root / "train_labels.csv")
    train_x = base_sel.load_stack(root / "train", ids)
    sample_ids = np.asarray(ids)
    print(f"[phase1] data loaded: N={len(train_y)}, T={train_x.shape[1]}, elapsed={time.time()-t0:.1f}s", flush=True)

    # 5-fold loop
    folds_to_run = [args.smoke_fold] if args.smoke_fold is not None else list(range(args.folds))
    fold_ids = np.array([base_sel.stable_fold_id(sid, args.folds) for sid in sample_ids])
    oof_preds = np.zeros((len(train_y), 3), dtype=np.float32)
    oof_scores = np.zeros((len(train_y), 27), dtype=np.float32)
    fold_hits: dict[int, float] = {}
    in_ic_drift_flags: dict[int, bool] = {}

    for fold in folds_to_run:
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
        result = iv3.run_integrated_v3(config, fold, train_x, train_y, sample_ids)
        val_mask = fold_ids == fold
        oof_preds[val_mask] = result["val_preds"]
        oof_scores[val_mask] = result["val_scores"]
        fold_hits[fold] = result["oof_metric"]["hit"]
        # frozen GRU drift check
        in_ic_drift_flags[fold] = (
            result["in_ic_final_hash"] != result["in_ic_init_hash"]
            and result["in_ic_init_hash"] != 0
        )
        print(
            f"[phase1] fold {fold}: hit={fold_hits[fold]:.4f}, "
            f"n_val={int(val_mask.sum())}, n_epochs={result['n_epochs_trained']}, "
            f"in_ic_drift={in_ic_drift_flags[fold]}, elapsed={time.time()-t_fold:.1f}s",
            flush=True,
        )

    # concat OOF metric via search_temperature (§3.3)
    if args.smoke_fold is None:
        # full 5-fold concat
        # search_temperature signature: (candidates: (N, K, 3), scores: (N, K), true: (N, 3)) — but
        # oof_preds is (N, 3) already corrected. We treat *corrected predictions* as a single
        # candidate (K=1) — the soft-weighted aggregate is already in oof_preds.
        # Direct raw hit metric:
        oof_hit = float(np.mean(np.linalg.norm(oof_preds - train_y, axis=1) <= 0.01))
    else:
        # smoke fold only
        oof_hit = fold_hits[args.smoke_fold]

    elapsed = time.time() - t0
    summary = {
        "exp_id": "H031_phase1-baseline-plan004-inIC",
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
        "folds_run": folds_to_run,
        "fold_hits": fold_hits,
        "oof_hit_concat": oof_hit,
        "g1_pass": oof_hit >= 0.65,
        "g1_threshold": 0.65,
        "in_ic_drift_flags": in_ic_drift_flags,
        "frozen_gru_drift_severe": any(in_ic_drift_flags.values()),
        "elapsed_sec": elapsed,
        "smoke_fold": args.smoke_fold,
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez(
        out_path.with_suffix(".npz"),
        oof_preds=oof_preds, oof_scores=oof_scores, train_y=train_y,
    )
    print(f"\n[phase1] saved: {out_path.relative_to(REPO)}")
    print(f"[phase1] oof_hit_concat = {oof_hit:.4f} (G1 threshold 0.65) → pass = {summary['g1_pass']}")
    return 0 if summary["g1_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
