"""plan-012 c4 (G1) — Phase 1 3-Way Codebook Bake-off.

3 sub-exp (E0a Absolute / E0b Frenet-Ortho / E0c K-Means) fold-0 학습 + winner 결정.
공통 (paradigm-clean): HybridScorerHead(K=7, hidden=64), AdamW(3e-4, wd=1e-4),
hybrid_combined_loss(λ_ce=1.0, λ_hinge=1.0, use_reg_head=True), τ=0.03 inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pb_0_6822 import ring_classifier as rc       # noqa: E402
from src.pb_0_6822 import selector as base            # noqa: E402


DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 학습 1 sub-exp
# ──────────────────────────────────────────────────────────────────────────────


def run_sub_exp(
    sub_exp_id: str,
    codebook_id: str,
    anchors_local_per_fold: np.ndarray | None,           # K-Means: (5, K, 3); else None
    anchors_local_static: np.ndarray | None,             # Absolute/Frenet-ortho: (K, 3); else None
    R_wfn: np.ndarray | None,                            # Frenet/K-Means: (N, 3, 3); Absolute: None
    F0_pred: np.ndarray,                                 # (N, 3)
    train_y: np.ndarray,                                 # (N, 3)
    seq_feat: np.ndarray,                                # (N, 6, 9)
    fold_id: np.ndarray,
    val_fold: int = 0,
    K: int = 7,
    hidden: int = 64,
    cand_dim: int = 11,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 15,
    batch_size: int = 512,
    patience: int = 3,
    temperature: float = 0.03,
    seed: int = 20260513,
    verbose: bool = True,
) -> dict:
    """Train 1 sub-exp on fold-0 val, return metrics + best epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = F0_pred.shape[0]
    val_mask = fold_id == val_fold
    train_mask = ~val_mask
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    if verbose:
        print(f"[{sub_exp_id}] n_train={n_train}, n_val={n_val}, K={K}", flush=True)

    # anchors_world (N, K, 3) — sample-wise (K-Means) or static (Absolute/Frenet-ortho)
    anchors_world_all = np.zeros((N, K, 3), dtype=np.float32)
    if codebook_id == "kmeans":
        assert anchors_local_per_fold is not None
        # For each sample, use anchors of its assigned fold
        for k in range(anchors_local_per_fold.shape[0]):
            mask_k = fold_id == k
            n_k = int(mask_k.sum())
            if n_k == 0:
                continue
            anchors_world_all[mask_k] = rc.anchors_to_world(
                anchors_local_per_fold[k], R_wfn[mask_k], N=n_k
            ).astype(np.float32)
        # anchors_local for feature builder uses fold-0 (val) anchors as canonical (sample-invariant feature OK)
        anchors_local_for_feat = anchors_local_per_fold[val_fold]
    elif codebook_id == "absolute":
        assert anchors_local_static is not None
        anchors_world_all = rc.anchors_to_world(anchors_local_static, None, N=N).astype(np.float32)
        anchors_local_for_feat = anchors_local_static
    elif codebook_id == "frenet_orthogonal":
        assert anchors_local_static is not None
        anchors_world_all = rc.anchors_to_world(anchors_local_static, R_wfn, N=N).astype(np.float32)
        anchors_local_for_feat = anchors_local_static
    else:
        raise ValueError(f"unknown codebook_id: {codebook_id}")

    # cand_feat (N, K, 11) — sample-invariant
    cand_feat_all = rc.make_codebook_candidate_features(
        None, anchors_local_for_feat, codebook_id, R_wfn, F0_pred
    )                                                                                       # (N, K, 11)

    # torch tensors
    seq_t = torch.from_numpy(seq_feat.astype(np.float32))
    cand_t = torch.from_numpy(cand_feat_all)
    anchors_world_t = torch.from_numpy(anchors_world_all)
    F0_t = torch.from_numpy(F0_pred.astype(np.float32))
    target_t = torch.from_numpy(train_y.astype(np.float32))

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # model
    head = rc.HybridScorerHead(K=K, hidden=hidden, cand_dim=cand_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_hit = -1.0
    best_epoch = -1
    best_state = None
    best_dcm = 0.0
    patience_left = patience
    epoch_log = []

    for epoch in range(epochs):
        head.train()
        rng = np.random.default_rng(seed + epoch)
        shuffled = rng.permutation(train_idx)
        total_loss = 0.0
        n_steps = 0

        for s in range(0, len(shuffled), batch_size):
            b_idx = shuffled[s:s + batch_size]
            b_idx_t = torch.from_numpy(b_idx).long()
            seq_b = seq_t[b_idx_t].to(DEVICE)
            cand_b = cand_t[b_idx_t].to(DEVICE)
            aw_b = anchors_world_t[b_idx_t].to(DEVICE)
            F0_b = F0_t[b_idx_t].to(DEVICE)
            tgt_b = target_t[b_idx_t].to(DEVICE)

            logits, reg = head(seq_b, cand_b)
            loss = rc.hybrid_combined_loss(
                logits, reg, aw_b, F0_b, tgt_b,
                temperature=temperature,
                use_reg_head=True,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1

        avg_loss = total_loss / max(1, n_steps)

        # val OOF
        head.eval()
        with torch.no_grad():
            val_preds = []
            for s in range(0, len(val_idx), batch_size):
                b_idx = val_idx[s:s + batch_size]
                b_idx_t = torch.from_numpy(b_idx).long()
                seq_b = seq_t[b_idx_t].to(DEVICE)
                cand_b = cand_t[b_idx_t].to(DEVICE)
                aw_b = anchors_world_t[b_idx_t].to(DEVICE)
                F0_b = F0_t[b_idx_t].to(DEVICE)
                logits, reg = head(seq_b, cand_b)
                pred = rc.hybrid_predict(
                    logits, reg, aw_b, F0_b,
                    temperature=temperature, use_reg_head=True, r0_logit_prior=0.0,
                )
                val_preds.append(pred.cpu().numpy())
            val_pred_arr = np.concatenate(val_preds, axis=0)

        val_true = train_y[val_idx]
        val_F0 = F0_pred[val_idx]
        err = np.linalg.norm(val_pred_arr - val_true, axis=-1)
        val_hit = float(np.mean(err <= 0.01))
        dcm = float(np.mean(np.linalg.norm(val_pred_arr - val_F0, axis=-1)))

        if verbose:
            print(f"[{sub_exp_id}] epoch {epoch+1:02d}/{epochs} loss={avg_loss:.4f} val_hit@1cm={val_hit:.4f} DCM={dcm:.5f}", flush=True)
        epoch_log.append({"epoch": epoch + 1, "loss": avg_loss, "val_hit": val_hit, "dcm": dcm})

        if val_hit > best_val_hit:
            best_val_hit = val_hit
            best_epoch = epoch + 1
            best_dcm = dcm
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[{sub_exp_id}] early stop @ epoch {epoch+1}", flush=True)
                break

    return {
        "sub_exp_id": sub_exp_id,
        "codebook_id": codebook_id,
        "K": K,
        "n_train": n_train,
        "n_val": n_val,
        "epochs_run": len(epoch_log),
        "best_epoch": best_epoch,
        "best_val_hit": best_val_hit,
        "best_dcm": best_dcm,
        "epoch_log": epoch_log,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--preflight", type=str, default="analysis/plan-012/preflight.json")
    parser.add_argument("--out", type=str, default="analysis/plan-012/phase1_winner.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    print(f"[bakeoff] loading data ...", flush=True)
    train_ids, train_y = base.read_labels(root / "train_labels.csv")
    train_x = base.load_stack(root / "train", train_ids).astype(np.float64)
    train_y = train_y.astype(np.float64)
    N, T, _ = train_x.shape
    end_idx = T - 1
    print(f"[bakeoff] N={N}, T={T}, end_idx={end_idx}", flush=True)

    fold_id = np.array([base.stable_fold_id(sid, args.n_folds) for sid in train_ids], dtype=np.int64)

    F0_pred = rc.f0_predict_frenet_par120_perp_neg020(train_x, end_idx=end_idx)
    R_wfn = rc.build_frenet_basis_3d(train_x, end_idx=end_idx)
    residuals_world = train_y - F0_pred

    print(f"[bakeoff] building seq features (end_idx={end_idx}) ...", flush=True)
    seq_feat = base.make_seq_features(train_x, end_idx=end_idx, direction=1.0).astype(np.float32)
    print(f"[bakeoff] seq_feat.shape = {seq_feat.shape}", flush=True)

    # codebooks
    anchors_abs = rc.compute_anchors_absolute(radius_m=0.005)
    anchors_fr = rc.compute_anchors_frenet_orthogonal(radius_m=0.005)
    centers_per_fold, sizes_per_fold, km_meta = rc.compute_anchors_kmeans(
        residuals_world, R_wfn, fold_id, K=7, radius_clip_m=0.020,
    )
    print(f"[bakeoff] codebooks ready (kmeans min cluster={sizes_per_fold[:, 1:].min()})", flush=True)

    # 3 sub-exp
    results = {}
    t0 = time.time()
    for sub_id, codebook_id, anchors_static, anchors_pf, R_arg in [
        ("E0a", "absolute", anchors_abs, None, None),
        ("E0b", "frenet_orthogonal", anchors_fr, None, R_wfn),
        ("E0c", "kmeans", None, centers_per_fold, R_wfn),
    ]:
        ts = time.time()
        r = run_sub_exp(
            sub_exp_id=sub_id,
            codebook_id=codebook_id,
            anchors_local_per_fold=anchors_pf,
            anchors_local_static=anchors_static,
            R_wfn=R_arg,
            F0_pred=F0_pred,
            train_y=train_y,
            seq_feat=seq_feat,
            fold_id=fold_id,
            val_fold=args.val_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        r["elapsed_seconds"] = round(time.time() - ts, 1)
        results[sub_id] = r
        print(f"[{sub_id}] best val_hit={r['best_val_hit']:.4f} @ epoch {r['best_epoch']} (DCM={r['best_dcm']:.5f}, {r['elapsed_seconds']}s)", flush=True)

    # Winner 결정
    oofs = {k: v["best_val_hit"] for k, v in results.items()}
    dcms = {k: v["best_dcm"] for k, v in results.items()}

    sorted_oof = sorted(oofs.items(), key=lambda x: x[1], reverse=True)
    winner_id, winner_oof = sorted_oof[0]
    second_id, second_oof = sorted_oof[1]
    gap = winner_oof - second_oof
    tie_break_applied = False

    if gap < 0.005:
        priority = ["E0a", "E0b", "E0c"]
        tied = [k for k, v in oofs.items() if v >= winner_oof - 0.005]
        winner_id = next(k for k in priority if k in tied)
        winner_oof = oofs[winner_id]
        tie_break_applied = True
        print(f"[bakeoff] tie-break applied: winner={winner_id} (gap={gap:.4f} < 0.005)", flush=True)
        # recompute second
        second_candidates = [(k, v) for k, v in oofs.items() if k != winner_id]
        second_id, second_oof = max(second_candidates, key=lambda x: x[1])
        gap = winner_oof - second_oof

    g1_baseline_anchor = 0.6450
    winner_above_anchor = winner_oof >= g1_baseline_anchor
    dcm_ok = dcms[winner_id] >= 0.002

    # anchor source label
    anchor_source = {
        "E0a": "compute_anchors_absolute",
        "E0b": "compute_anchors_frenet_orthogonal",
        "E0c": "compute_anchors_kmeans",
    }[winner_id]
    frame = {"E0a": "world", "E0b": "frenet", "E0c": "frenet"}[winner_id]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "exp_id": "H020_phase1-codebook-bakeoff",
        "winner_id": winner_id,
        "winner_oof": winner_oof,
        "winner_anchor_source": anchor_source,
        "winner_K": 7,
        "winner_frame": frame,
        "second_id": second_id,
        "second_oof": second_oof,
        "gap": gap,
        "tie_break_applied": tie_break_applied,
        "all_sub_exp_oof": oofs,
        "directional_commit_magnitudes": dcms,
        "G1_baseline_anchor": g1_baseline_anchor,
        "winner_above_anchor": winner_above_anchor,
        "winner_dcm_ok": dcm_ok,
        "G1_passed": winner_above_anchor and dcm_ok,
        "elapsed_total_seconds": round(time.time() - t0, 1),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "val_fold": args.val_fold,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "temperature": 0.03,
        },
        "results_per_sub_exp": results,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bakeoff] wrote {out_path}", flush=True)
    print(f"[bakeoff] winner = {winner_id} ({winner_oof:.4f}), gap={gap:.4f}, G1 passed={summary['G1_passed']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
