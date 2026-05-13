"""plan-012 c15 (G4) — Phase 4 Best Stack 5-fold + Submission.

Best stack: E0a Absolute winner + Phase 2 best lever (E3 τ=0.01) + Phase 3 best lever (E8 r=0 +0.5).
Anchor 5-fold baseline (= E0a config 그대로) 도 동일 코드 path 로 산출 — fairness 비교.

G4 합격: best_stack 5-fold concat OOF ≥ anchor_5fold_oof + 0.005.
위반 시 final_no_additive warn + fallback = anchor 5-fold submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pb_0_6822 import ring_classifier as rc                       # noqa: E402
from src.pb_0_6822 import ring_classifier_train as rct                # noqa: E402
from src.pb_0_6822 import selector as base                            # noqa: E402

DEVICE = torch.device("cpu")


def train_one_fold(
    config: dict,
    val_fold: int,
    n_folds: int,
    F0_pred: np.ndarray, train_y: np.ndarray, seq_feat: np.ndarray, fold_id: np.ndarray,
    anchors_local: np.ndarray, R_wfn: np.ndarray | None,
    epochs: int, batch_size: int, patience: int,
    seed_base: int = 20260513,
) -> tuple[torch.nn.Module, dict]:
    """Train HybridScorerHead on fold-`val_fold`. Returns trained head + result dict.

    config keys: temperature, use_reg_head, use_hinge, r0_logit_prior, sample_weight (or None).
    """
    K = anchors_local.shape[0]
    N = F0_pred.shape[0]

    # anchors_world (N, K, 3)
    if R_wfn is None:
        anchors_world_all = rc.anchors_to_world(anchors_local, None, N=N).astype(np.float32)
    else:
        anchors_world_all = rc.anchors_to_world(anchors_local, R_wfn, N=N).astype(np.float32)
    cand_feat_all = rc.make_codebook_candidate_features(None, anchors_local, "absolute", R_wfn, F0_pred)

    val_mask = fold_id == val_fold
    train_mask = ~val_mask
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    seq_t = torch.from_numpy(seq_feat.astype(np.float32))
    cand_t = torch.from_numpy(cand_feat_all)
    aw_t = torch.from_numpy(anchors_world_all)
    F0_t = torch.from_numpy(F0_pred.astype(np.float32))
    tgt_t = torch.from_numpy(train_y.astype(np.float32))

    torch.manual_seed(seed_base + val_fold)
    np.random.seed(seed_base + val_fold)
    head = rc.HybridScorerHead(K=K, hidden=64, cand_dim=11).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_hit = -1.0
    best_epoch = -1
    best_state = None
    patience_left = patience

    for epoch in range(epochs):
        head.train()
        rng = np.random.default_rng(seed_base + val_fold + epoch)
        shuffled = rng.permutation(train_idx)
        for s in range(0, len(shuffled), batch_size):
            b_idx = shuffled[s:s + batch_size]
            b_idx_t = torch.from_numpy(b_idx).long()
            seq_b = seq_t[b_idx_t].to(DEVICE)
            cand_b = cand_t[b_idx_t].to(DEVICE)
            aw_b = aw_t[b_idx_t].to(DEVICE)
            F0_b = F0_t[b_idx_t].to(DEVICE)
            tgt_b = tgt_t[b_idx_t].to(DEVICE)
            logits, reg = head(seq_b, cand_b)
            loss = rc.hybrid_combined_loss(
                logits, reg, aw_b, F0_b, tgt_b,
                temperature=config["temperature"],
                use_reg_head=config["use_reg_head"],
                use_hinge=config["use_hinge"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()

        # val
        head.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(val_idx), batch_size):
                b_idx = val_idx[s:s + batch_size]
                b_idx_t = torch.from_numpy(b_idx).long()
                seq_b = seq_t[b_idx_t].to(DEVICE)
                cand_b = cand_t[b_idx_t].to(DEVICE)
                aw_b = aw_t[b_idx_t].to(DEVICE)
                F0_b = F0_t[b_idx_t].to(DEVICE)
                logits, reg = head(seq_b, cand_b)
                pred = rc.hybrid_predict(
                    logits, reg, aw_b, F0_b,
                    temperature=config["temperature"],
                    use_reg_head=config["use_reg_head"],
                    r0_logit_prior=config["r0_logit_prior"],
                )
                preds.append(pred.cpu().numpy())
            val_pred = np.concatenate(preds, axis=0)
        val_true = train_y[val_idx]
        val_hit = float(np.mean(np.linalg.norm(val_pred - val_true, axis=-1) <= 0.01))

        if val_hit > best_val_hit:
            best_val_hit = val_hit
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    head.load_state_dict(best_state)
    return head, {"val_fold": val_fold, "best_val_hit": best_val_hit, "best_epoch": best_epoch,
                  "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum())}


def predict_set(
    head: torch.nn.Module,
    config: dict,
    anchors_local: np.ndarray,
    F0_pred: np.ndarray,
    R_wfn: np.ndarray | None,
    seq_feat: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Run hybrid inference on a set. Returns (N, 3) predictions."""
    K = anchors_local.shape[0]
    N = F0_pred.shape[0]
    if R_wfn is None:
        anchors_world = rc.anchors_to_world(anchors_local, None, N=N).astype(np.float32)
    else:
        anchors_world = rc.anchors_to_world(anchors_local, R_wfn, N=N).astype(np.float32)
    cand_feat = rc.make_codebook_candidate_features(None, anchors_local, "absolute", R_wfn, F0_pred)

    seq_t = torch.from_numpy(seq_feat.astype(np.float32))
    cand_t = torch.from_numpy(cand_feat)
    aw_t = torch.from_numpy(anchors_world)
    F0_t = torch.from_numpy(F0_pred.astype(np.float32))

    head.eval()
    preds_list = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            seq_b = seq_t[s:s + batch_size].to(DEVICE)
            cand_b = cand_t[s:s + batch_size].to(DEVICE)
            aw_b = aw_t[s:s + batch_size].to(DEVICE)
            F0_b = F0_t[s:s + batch_size].to(DEVICE)
            logits, reg = head(seq_b, cand_b)
            pred = rc.hybrid_predict(
                logits, reg, aw_b, F0_b,
                temperature=config["temperature"],
                use_reg_head=config["use_reg_head"],
                r0_logit_prior=config["r0_logit_prior"],
            )
            preds_list.append(pred.cpu().numpy())
    return np.concatenate(preds_list, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--winner", type=str, default="analysis/plan-012/phase1_winner.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out", type=str, default="analysis/plan-012/phase4_results.json")
    parser.add_argument("--submission-best", type=str,
                        default="runs/baseline/H029_phase4-final-5fold/submission.csv")
    parser.add_argument("--submission-anchor", type=str,
                        default="runs/baseline/H029_phase4-final-5fold/submission_anchor_fallback.csv")
    args = parser.parse_args()

    Path(args.submission_best).parent.mkdir(parents=True, exist_ok=True)

    # data
    root = Path(args.root)
    print("[phase4] loading train ...", flush=True)
    train_ids, train_y = base.read_labels(root / "train_labels.csv")
    train_x = base.load_stack(root / "train", train_ids).astype(np.float64)
    train_y = train_y.astype(np.float64)
    N, T, _ = train_x.shape
    end_idx = T - 1
    fold_id = np.array([base.stable_fold_id(sid, args.n_folds) for sid in train_ids], dtype=np.int64)

    F0_train = rc.f0_predict_frenet_par120_perp_neg020(train_x, end_idx=end_idx)
    R_wfn_train = rc.build_frenet_basis_3d(train_x, end_idx=end_idx)
    seq_train = base.make_seq_features(train_x, end_idx=end_idx, direction=1.0).astype(np.float32)
    anchors_abs = rc.compute_anchors_absolute()

    # test data load
    print("[phase4] loading test ...", flush=True)
    submission_ids = base.read_submission_ids(root / "sample_submission.csv")
    test_x = base.load_stack(root / "test", submission_ids).astype(np.float64)
    T_test = test_x.shape[1]
    end_idx_test = T_test - 1
    F0_test = rc.f0_predict_frenet_par120_perp_neg020(test_x, end_idx=end_idx_test)
    R_wfn_test = rc.build_frenet_basis_3d(test_x, end_idx=end_idx_test)
    seq_test = base.make_seq_features(test_x, end_idx=end_idx_test, direction=1.0).astype(np.float32)
    print(f"[phase4] train: N={N} T={T}, test: N={test_x.shape[0]} T={T_test}", flush=True)

    # configs
    config_anchor = {
        "name": "anchor_E0a",
        "temperature": 0.03, "use_reg_head": True, "use_hinge": True,
        "r0_logit_prior": 0.0,
    }
    config_best = {
        "name": "best_stack_E0a+tau0.01+r0_0.5",
        "temperature": 0.01,       # E3 best lever
        "use_reg_head": True, "use_hinge": True,
        "r0_logit_prior": 0.5,     # E8 best lever
    }

    # 5-fold loop
    t0 = time.time()
    fold_results = {"anchor": [], "best": []}
    oof_preds = {"anchor": np.zeros((N, 3), dtype=np.float64),
                 "best":   np.zeros((N, 3), dtype=np.float64)}
    test_preds = {"anchor": [], "best": []}

    for cfg_key, cfg in [("anchor", config_anchor), ("best", config_best)]:
        print(f"\n[phase4] === config: {cfg['name']} ===", flush=True)
        for fold in range(args.n_folds):
            ts = time.time()
            head, meta = train_one_fold(
                cfg, val_fold=fold, n_folds=args.n_folds,
                F0_pred=F0_train, train_y=train_y, seq_feat=seq_train, fold_id=fold_id,
                anchors_local=anchors_abs, R_wfn=None,
                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            )
            # val OOF predictions (using best state)
            val_idx = np.where(fold_id == fold)[0]
            val_F0 = F0_train[val_idx]
            val_seq = seq_train[val_idx]
            val_pred = predict_set(
                head, cfg, anchors_abs, val_F0, None, val_seq,
                batch_size=args.batch_size,
            )
            oof_preds[cfg_key][val_idx] = val_pred

            # test predictions
            t_pred = predict_set(
                head, cfg, anchors_abs, F0_test, None, seq_test,
                batch_size=args.batch_size,
            )
            test_preds[cfg_key].append(t_pred)

            meta["elapsed_seconds"] = round(time.time() - ts, 1)
            fold_results[cfg_key].append(meta)
            print(f"[phase4 {cfg_key} fold {fold}] val_hit={meta['best_val_hit']:.4f} @ epoch {meta['best_epoch']} ({meta['elapsed_seconds']}s)", flush=True)

    # 5-fold concat OOF
    oof_hit_anchor = float(np.mean(np.linalg.norm(oof_preds["anchor"] - train_y, axis=-1) <= 0.01))
    oof_hit_best = float(np.mean(np.linalg.norm(oof_preds["best"] - train_y, axis=-1) <= 0.01))
    delta = oof_hit_best - oof_hit_anchor
    print(f"\n[phase4] anchor 5-fold OOF hit@1cm = {oof_hit_anchor:.4f}", flush=True)
    print(f"[phase4] best   5-fold OOF hit@1cm = {oof_hit_best:.4f}  (Δ = {delta:+.4f})", flush=True)

    # G4 합격
    g4_passed = delta >= 0.005
    g4_warn = None if g4_passed else "final_no_additive"

    # Test ensemble (5-fold 좌표 평균)
    test_ensemble_anchor = np.stack(test_preds["anchor"], axis=0).mean(axis=0)
    test_ensemble_best = np.stack(test_preds["best"], axis=0).mean(axis=0)

    def write_submission(path: str, preds: np.ndarray) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "x", "y", "z"])
            for sid, p in zip(submission_ids, preds):
                w.writerow([sid, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])

    write_submission(args.submission_best, test_ensemble_best)
    write_submission(args.submission_anchor, test_ensemble_anchor)
    print(f"[phase4] wrote best   submission → {args.submission_best}", flush=True)
    print(f"[phase4] wrote anchor submission → {args.submission_anchor}", flush=True)

    summary = {
        "exp_id": "H029_phase4-final-5fold",
        "config_anchor": config_anchor,
        "config_best": config_best,
        "n_train": int(N),
        "n_test": int(test_x.shape[0]),
        "n_folds": args.n_folds,
        "anchor_5fold_oof_hit_1cm": oof_hit_anchor,
        "best_5fold_oof_hit_1cm": oof_hit_best,
        "delta_oof": delta,
        "G4_threshold_delta": 0.005,
        "G4_passed": g4_passed,
        "G4_warn": g4_warn,
        "fold_results": fold_results,
        "submission_best": args.submission_best,
        "submission_anchor_fallback": args.submission_anchor,
        "submission_used_for_LB": args.submission_best if g4_passed else args.submission_anchor,
        "elapsed_total_seconds": round(time.time() - t0, 1),
        "training_config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "patience": args.patience,
            "lr": 3e-4, "weight_decay": 1e-4,
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[phase4] wrote {args.out}", flush=True)
    print(f"[phase4] G4 passed: {g4_passed} (delta={delta:+.4f}, threshold=+0.005)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
