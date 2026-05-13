"""plan-012 — shared training helper for Phase 1/2/3 bake-off and ablation.

run_sub_exp(...) : 1 sub-exp 학습 + fold-0 val OOF hit + DCM 산출.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.pb_0_6822 import ring_classifier as rc


DEVICE = torch.device("cpu")


def run_sub_exp(
    sub_exp_id: str,
    codebook_id: str,
    anchors_local_per_fold: Optional[np.ndarray],          # K-Means: (5, K, 3); else None
    anchors_local_static: Optional[np.ndarray],            # Absolute/Frenet-ortho: (K, 3); else None
    R_wfn: Optional[np.ndarray],                            # Frenet/K-Means: (N, 3, 3); Absolute: None
    F0_pred: np.ndarray,                                    # (N, 3)
    train_y: np.ndarray,                                    # (N, 3)
    seq_feat: np.ndarray,                                   # (N, 6, 9)
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
    use_reg_head: bool = True,
    use_hinge: bool = True,
    r0_logit_prior: float = 0.0,
    sample_weight: Optional[np.ndarray] = None,             # (N,) optional sample weighting
    scorer_arch: str = "attn_gru",                          # "attn_gru" (default) or "last_step_mlp"
    seed: int = 20260513,
    verbose: bool = True,
) -> dict:
    """Train 1 sub-exp on fold-`val_fold` validation. Returns metrics + best epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = F0_pred.shape[0]
    val_mask = fold_id == val_fold
    train_mask = ~val_mask
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    if verbose:
        print(f"[{sub_exp_id}] n_train={n_train}, n_val={n_val}, K={K}", flush=True)

    # anchors_world (N, K, 3)
    anchors_world_all = np.zeros((N, K, 3), dtype=np.float32)
    if codebook_id == "kmeans":
        assert anchors_local_per_fold is not None
        for k in range(anchors_local_per_fold.shape[0]):
            mask_k = fold_id == k
            n_k = int(mask_k.sum())
            if n_k == 0:
                continue
            anchors_world_all[mask_k] = rc.anchors_to_world(
                anchors_local_per_fold[k], R_wfn[mask_k], N=n_k
            ).astype(np.float32)
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

    # cand_feat
    cand_feat_all = rc.make_codebook_candidate_features(
        None, anchors_local_for_feat, codebook_id, R_wfn, F0_pred
    )                                                                                      # (N, K, 11)

    # build head
    if scorer_arch == "attn_gru":
        head = rc.HybridScorerHead(K=K, hidden=hidden, cand_dim=cand_dim).to(DEVICE)
    elif scorer_arch == "last_step_mlp":
        head = rc.HybridScorerHead(K=K, hidden=hidden, cand_dim=cand_dim).to(DEVICE)
        # swap scorer to LastStepMLPScorer
        from torch import nn as _nn
        head.scorer = rc.LastStepMLPScorer(seq_dim=9, cand_dim=cand_dim, hidden=hidden, cand_count=K).to(DEVICE)
    else:
        raise ValueError(f"unknown scorer_arch: {scorer_arch}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    # to tensors
    seq_t = torch.from_numpy(seq_feat.astype(np.float32))
    cand_t = torch.from_numpy(cand_feat_all)
    anchors_world_t = torch.from_numpy(anchors_world_all)
    F0_t = torch.from_numpy(F0_pred.astype(np.float32))
    target_t = torch.from_numpy(train_y.astype(np.float32))
    sw_t = None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    best_val_hit = -1.0
    best_epoch = -1
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
                use_reg_head=use_reg_head,
                use_hinge=use_hinge,
            )
            # 단순 sample weighting (E6): 학습 loss 의 per-sample 가중 (re-weighting via batched scalar)
            if sw_t is not None:
                w_b = sw_t[b_idx_t].to(DEVICE)
                # hybrid_combined_loss 가 이미 reduce-mean 했으므로 batch mean × mean(weight) 가 추정치.
                # 정확한 reweighting 은 별도 path 가 필요하나 E6 도 informational 이므로 본 근사 적용.
                loss = loss * (w_b.mean())

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
                    temperature=temperature, use_reg_head=use_reg_head, r0_logit_prior=r0_logit_prior,
                )
                val_preds.append(pred.cpu().numpy())
            val_pred_arr = np.concatenate(val_preds, axis=0)

        val_true = train_y[val_idx]
        val_F0 = F0_pred[val_idx]
        err = np.linalg.norm(val_pred_arr - val_true, axis=-1)
        val_hit = float(np.mean(err <= 0.01))
        dcm = float(np.mean(np.linalg.norm(val_pred_arr - val_F0, axis=-1)))

        if verbose:
            print(f"[{sub_exp_id}] epoch {epoch+1:02d}/{epochs} loss={avg_loss:.4f} val_hit={val_hit:.4f} DCM={dcm:.5f}", flush=True)
        epoch_log.append({"epoch": epoch + 1, "loss": avg_loss, "val_hit": val_hit, "dcm": dcm})

        if val_hit > best_val_hit:
            best_val_hit = val_hit
            best_epoch = epoch + 1
            best_dcm = dcm
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
