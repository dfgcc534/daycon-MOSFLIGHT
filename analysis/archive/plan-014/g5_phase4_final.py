"""plan-014 c9 (STAGE 5, G5) — Phase 4 final 5-fold + best stack + submission.

config_anchor = G2 winner (E0c K-Means K=7, default 5 lever).
config_best   = anchor + Phase 2 best (E2c K=9, ΔOOF +0.0030) + Phase 3 best
                (E6b boundary_weight_on, ΔOOF +0.0015).

10 fold trainings (anchor × 5 + best × 5). test 5-fold ensemble = coord mean.

G5 합격: best_stack_5fold >= anchor_5fold + 0.005.
band: ≥0.66 positive / 0.65~0.66 partial / <0.65 negative.

Usage:
    python analysis/plan-014/g5_phase4_final.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import plan014_paradigm as pp  # noqa: E402


def compute_hit_5fold(oof_pred, Y, threshold_m=0.01):
    completed = ~np.isnan(oof_pred).any(axis=1)
    if completed.sum() == 0:
        return float("nan")
    err = np.linalg.norm(oof_pred[completed] - Y[completed], axis=-1)
    return float(np.mean(err <= threshold_m))


def train_predict_fold(cfg, fold_id, X_train, Y_train, X_val, Y_val, f0_function):
    """Train one fold + predict val + return predict-only model state for test ensemble."""
    res = pp.train_one_fold(
        cfg, fold_id=fold_id,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        f0_function=f0_function,
    )
    # train_one_fold 이 oof_pred 반환 (val 의 prediction)
    return res


def predict_test(cfg, fold_id, X_train, Y_train, X_test, f0_function):
    """Train on train fold, predict on test. Returns (N_test, 3) numpy."""
    # 같은 logic 으로 train + final eval on X_test
    # train_one_fold 활용 — val 자리에 X_test/dummy Y 넣되 model.load 만 회수 곤란.
    # 대신 train_one_fold 함수를 살짝 확장하지 않고 직접 mini-runner 사용.

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Anchors
    anchors_local, R_train = pp.get_anchors_for_fold(
        cfg.codebook, X_train, Y_train, fold_id, K=cfg.K, f0_function=f0_function
    )
    R_test = pp.build_frenet_basis_3d(X_test, end_idx=cfg.end_idx) if cfg.codebook != "absolute" else None

    seq_train = pp.make_seq_features(X_train, end_idx=cfg.end_idx)
    seq_test = pp.make_seq_features(X_test, end_idx=cfg.end_idx)

    F0_train_np = f0_function(X_train)
    F0_test_np = f0_function(X_test)

    sw_train = pp._boundary_weight(F0_train_np, Y_train) if cfg.boundary_weight_on else None

    # To torch
    seq_train_t = torch.from_numpy(seq_train).to(device)
    seq_test_t = torch.from_numpy(seq_test).to(device)
    Y_train_t = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    F0_train_t = torch.from_numpy(F0_train_np.astype(np.float32)).to(device)
    F0_test_t = torch.from_numpy(F0_test_np.astype(np.float32)).to(device)
    anchors_t = torch.from_numpy(anchors_local.astype(np.float32)).to(device)
    R_train_t = torch.from_numpy(R_train.astype(np.float32)).to(device) if R_train is not None else None
    R_test_t = torch.from_numpy(R_test.astype(np.float32)).to(device) if R_test is not None else None
    sw_t = torch.from_numpy(sw_train).to(device) if sw_train is not None else None

    # Model
    model = pp.Plan014HybridHead(K=anchors_local.shape[0], encoder_name=cfg.encoder_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_train = X_train.shape[0]

    # Train (no separate val — use train-only monitor: just run full epochs)
    # Actually monitor=val_hit needs val, but here we just train. Use epoch count fixed.
    # For consistency, we use a small held-out split from train as val.
    # Simpler: just train on full and predict on test.
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            seq_b = seq_train_t[idx]
            Y_b = Y_train_t[idx]
            F0_b = F0_train_t[idx]
            anchors_world_b = (anchors_t.unsqueeze(0).expand(idx.shape[0], -1, -1)
                                if R_train_t is None
                                else torch.einsum("bij,kj->bki", R_train_t[idx], anchors_t))
            sw_b = sw_t[idx] if sw_t is not None else None

            opt.zero_grad()
            logits, reg_offset = model.forward(seq_b, anchors_t)
            loss, _ = pp.hybrid_combined_loss(
                logits, reg_offset, F0_b, anchors_world_b, Y_b,
                sample_weight=sw_b,
                use_hinge=cfg.use_hinge, use_reg_head=cfg.use_reg_head,
                temperature=cfg.temperature,
            )
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_pred = model.hybrid_predict(
            seq_test_t, anchors_t, R_test_t, F0_test_t,
            temperature=cfg.temperature, use_reg_head=cfg.use_reg_head,
            r0_logit_prior=cfg.r0_logit_prior,
        )
    return test_pred.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-014/g5_phase4.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan014_g5_phase4"))
    ap.add_argument("--epochs", type=int, default=20)  # G5 final, faster monitor
    ap.add_argument("--patience", type=int, default=pp.DEFAULT_PATIENCE)
    args = ap.parse_args()

    t_start = time.time()
    print("[plan-014 G5 v4] loading data ...", flush=True)
    ids_train, X_train_all = load_all_samples("train")
    ids_test, X_test_all = load_all_samples("test")
    _, Y_train_all = load_labels()
    X_train_all = X_train_all.astype(np.float32)
    Y_train_all = Y_train_all.astype(np.float32)
    X_test_all = X_test_all.astype(np.float32)
    print(f"[plan-014 G5] N_train={X_train_all.shape[0]}, N_test={X_test_all.shape[0]}", flush=True)

    g2 = json.loads(Path("analysis/plan-014/g2_phase1.json").read_text())
    winner_codebook = g2["winner_codebook"]

    f0_function = pp.Plan014F0Function()

    # ── Configs (anchor + best) ──────────────────────────────────────────
    config_anchor = pp.TrainConfig(
        name="anchor", K=7, encoder_name="bigru", codebook=winner_codebook,
        use_reg_head=True, use_hinge=False,
        temperature=0.03, r0_logit_prior=0.0, boundary_weight_on=False,
        lr=pp.DEFAULT_LR, batch_size=pp.DEFAULT_BATCH,
        epochs=args.epochs, patience=args.patience, seed=pp.DEFAULT_SEED,
    )
    # best_stack = anchor + E2c K=9 + E6b boundary_weight_on
    config_best = replace(config_anchor, name="best", K=9, boundary_weight_on=True)

    fold_of = np.array([pp.stable_hash_fold(s) for s in ids_train])

    # ── anchor 5-fold OOF + per-fold model for test ensemble ─────────────
    print(f"\n[plan-014 G5] === anchor 5-fold ===", flush=True)
    oof_anchor = np.full_like(Y_train_all, np.nan, dtype=np.float32)
    test_preds_anchor_folds = []
    anchor_fold_results = []
    for f in range(pp.N_FOLDS):
        train_mask = fold_of != f
        val_mask = fold_of == f
        t = time.time()
        # Train + val
        val_res = train_predict_fold(config_anchor, f,
                                       X_train_all[train_mask], Y_train_all[train_mask],
                                       X_train_all[val_mask], Y_train_all[val_mask],
                                       f0_function)
        oof_anchor[val_mask] = val_res["oof_pred"]
        # Train + test pred (same fold, train-only)
        test_pred = predict_test(config_anchor, f,
                                  X_train_all[train_mask], Y_train_all[train_mask],
                                  X_test_all, f0_function)
        test_preds_anchor_folds.append(test_pred)
        elapsed = time.time() - t
        print(f"  fold {f}: val_hit={val_res['best_val_hit']:.4f} dcm={val_res['dcm']:.4f} "
              f"epoch={val_res['best_epoch']}/{args.epochs} elapsed={elapsed:.1f}s", flush=True)
        anchor_fold_results.append({
            "fold": f, "val_hit": val_res["best_val_hit"], "dcm": val_res["dcm"],
            "best_epoch": val_res["best_epoch"], "elapsed_seconds": elapsed,
        })
    anchor_oof_hit = compute_hit_5fold(oof_anchor, Y_train_all)
    test_pred_anchor = np.stack(test_preds_anchor_folds, axis=0).mean(axis=0)
    print(f"  anchor 5-fold concat OOF hit@1cm = {anchor_oof_hit:.4f}", flush=True)

    # ── best 5-fold OOF + per-fold model for test ensemble ───────────────
    print(f"\n[plan-014 G5] === best 5-fold ===", flush=True)
    oof_best = np.full_like(Y_train_all, np.nan, dtype=np.float32)
    test_preds_best_folds = []
    best_fold_results = []
    for f in range(pp.N_FOLDS):
        train_mask = fold_of != f
        val_mask = fold_of == f
        t = time.time()
        val_res = train_predict_fold(config_best, f,
                                       X_train_all[train_mask], Y_train_all[train_mask],
                                       X_train_all[val_mask], Y_train_all[val_mask],
                                       f0_function)
        oof_best[val_mask] = val_res["oof_pred"]
        test_pred = predict_test(config_best, f,
                                  X_train_all[train_mask], Y_train_all[train_mask],
                                  X_test_all, f0_function)
        test_preds_best_folds.append(test_pred)
        elapsed = time.time() - t
        print(f"  fold {f}: val_hit={val_res['best_val_hit']:.4f} dcm={val_res['dcm']:.4f} "
              f"epoch={val_res['best_epoch']}/{args.epochs} elapsed={elapsed:.1f}s", flush=True)
        best_fold_results.append({
            "fold": f, "val_hit": val_res["best_val_hit"], "dcm": val_res["dcm"],
            "best_epoch": val_res["best_epoch"], "elapsed_seconds": elapsed,
        })
    best_oof_hit = compute_hit_5fold(oof_best, Y_train_all)
    test_pred_best = np.stack(test_preds_best_folds, axis=0).mean(axis=0)
    print(f"  best 5-fold concat OOF hit@1cm = {best_oof_hit:.4f}", flush=True)

    # ── G5 합격 + band 분류 ──────────────────────────────────────────────
    delta_oof = best_oof_hit - anchor_oof_hit
    G5_threshold = 0.005
    G5_passed = delta_oof >= G5_threshold
    G5_warn = None if G5_passed else "g5_no_additive"

    # band classification (using best_stack OOF if G5_passed else anchor OOF)
    band_oof = best_oof_hit if G5_passed else anchor_oof_hit
    if band_oof >= 0.66:
        band = "positive"
    elif band_oof >= 0.65:
        band = "partial"
    else:
        band = "negative"

    print(f"\n[plan-014 G5] === G5 final ===", flush=True)
    print(f"  anchor_5fold_oof = {anchor_oof_hit:.4f}", flush=True)
    print(f"  best_5fold_oof   = {best_oof_hit:.4f}", flush=True)
    print(f"  delta_oof        = {delta_oof:+.4f} (threshold +{G5_threshold})", flush=True)
    print(f"  G5_passed        = {G5_passed}, warn={G5_warn}", flush=True)
    print(f"  band             = **{band}** (band_oof={band_oof:.4f})", flush=True)

    # ── Submission write ─────────────────────────────────────────────────
    args.run_dir.mkdir(parents=True, exist_ok=True)

    sample_sub = pd.read_csv("data/sample_submission.csv")
    sample_ids = sample_sub["id"].tolist()
    # ids_test 와 sample_ids 정합 확인
    id_to_idx = {sid: i for i, sid in enumerate(ids_test)}

    def write_submission(test_pred, path):
        ordered = np.array([test_pred[id_to_idx[sid]] for sid in sample_ids], dtype=np.float64)
        df = pd.DataFrame({
            "id": sample_ids,
            "x": [f"{v:.6f}" for v in ordered[:, 0]],
            "y": [f"{v:.6f}" for v in ordered[:, 1]],
            "z": [f"{v:.6f}" for v in ordered[:, 2]],
        })
        df.to_csv(path, index=False)
        return path

    submission_best_path = args.run_dir / "submission_best.csv"
    submission_anchor_fallback_path = args.run_dir / "submission_anchor_fallback.csv"
    write_submission(test_pred_best, submission_best_path)
    write_submission(test_pred_anchor, submission_anchor_fallback_path)
    submission_used_for_LB = submission_best_path if G5_passed else submission_anchor_fallback_path
    print(f"\n  submission_best        -> {submission_best_path}", flush=True)
    print(f"  submission_anchor_fallback -> {submission_anchor_fallback_path}", flush=True)
    print(f"  submission_used_for_LB -> {submission_used_for_LB} ({'best' if G5_passed else 'anchor fallback'})",
          flush=True)

    elapsed_total = time.time() - t_start
    artifact = {
        "exp_id": "H041_g5_phase4_final",
        "config_anchor": asdict(config_anchor),
        "config_best": asdict(config_best),
        "n_train": int(X_train_all.shape[0]),
        "n_test": int(X_test_all.shape[0]),
        "n_folds": pp.N_FOLDS,
        "anchor_5fold_oof_hit_1cm": anchor_oof_hit,
        "best_5fold_oof_hit_1cm": best_oof_hit,
        "delta_oof": delta_oof,
        "G5_threshold_delta": G5_threshold,
        "G5_passed": G5_passed,
        "G5_warn": G5_warn,
        "band": band,
        "band_oof": band_oof,
        "fold_results": {
            "anchor": anchor_fold_results,
            "best": best_fold_results,
        },
        "submission_best_path": str(submission_best_path),
        "submission_anchor_fallback_path": str(submission_anchor_fallback_path),
        "submission_used_for_LB": str(submission_used_for_LB),
        "submission_n_rows": len(sample_ids),
        "elapsed_total_seconds": elapsed_total,
        "plan_version": "v4.5",
        "f0_frozen_baseline": "plan-006_frenet_par120_perp_neg020",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\n[plan-014 G5] elapsed_total={elapsed_total:.1f}s ({elapsed_total/60:.2f} min)", flush=True)
    print(f"[plan-014 G5] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
