"""plan-024 v8 OPT-A2 — 학습 중 두 bias (physics + regime) logits 에 추가.

사용자 통찰 follow:
  plan-004 의 physics_bias (K-vec, sample-invariant) + regime_bias (18 × K matrix, EB shrunk)
  → plan-024 의 cross-attn 에 OPT-A2 적용: 학습 중 *둘 다* logits 에 더함.

수학적 의미: Bayesian posterior
  q_pred ∝ exp(model_logit + s1 × physics_bias + s2 × regime_bias[regime])

config:
  - plan-022 anchors (14 BCC), TAU_CLS=0.001
  - plan-024 v1 spec: cross-attn hidden=384, drop 0.3/0.2, lr 7e-4, wd 0.02
  - bias: physics + regime, prior_strength=1.0, regime_prior_strength=1.0
  - epoch 50 + constant lr + best epoch tracking (long-diag carry, §5.10)
  - 5-fold OOF
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import (
    stable_fold_id, fit_regime_bins, assign_regimes,
    candidate_physics_bias, candidate_regime_bias,
)

_spec = importlib.util.spec_from_file_location("p020_bf", _REPO / "analysis" / "plan-020" / "baseline_f0.py")
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)
_spec = importlib.util.spec_from_file_location("p022_anchors", _REPO / "analysis" / "plan-022" / "anchors.py")
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)
_spec = importlib.util.spec_from_file_location("p022_som", _REPO / "analysis" / "plan-022" / "selector_only_model.py")
som = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(som)
_spec = importlib.util.spec_from_file_location("p021_build", _REPO / "analysis" / "plan-021" / "build_input.py")
p021_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p021_build)
_spec = importlib.util.spec_from_file_location("p024_qc", _THIS / "quantile_carry.py")
qc_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(qc_mod)
_spec = importlib.util.spec_from_file_location("p024_mw", _THIS / "multiwindow_trim_build.py")
mw_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mw_build)
_spec = importlib.util.spec_from_file_location("p024_seq", _THIS / "seq_builder.py")
seq_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(seq_builder)
_spec = importlib.util.spec_from_file_location("p024_cand", _THIS / "cand_builder.py")
cand_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cand_builder)
_spec = importlib.util.spec_from_file_location("p024_model", _THIS / "model.py")
model_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(model_mod)

N_FOLDS = 5
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_CLS = 0.001
TAU_PAST = 0.003
SEED = 20260521
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRIOR_STRENGTH = 1.0
REGIME_PRIOR_STRENGTH = 1.0
EPOCHS = 50
BATCH = 256
LR = 7e-4
WD = 0.02


def compute_biases(anchors, R_wfn_tr, pred_F0_tr, Y_tr, regimes_tr, regime_count=18):
    """train fold 위 physics_bias (K,) + regime_table (18, K) 계산."""
    anchors_world = (
        np.einsum("nij,kj->nki", R_wfn_tr, anchors.astype(np.float32))
        + pred_F0_tr[:, None, :].astype(np.float32)
    )
    physics_bias = candidate_physics_bias(anchors_world, Y_tr.astype(np.float32))
    regime_table = candidate_regime_bias(
        anchors_world, Y_tr.astype(np.float32),
        regimes_tr.astype(np.int64), regime_count=regime_count,
    )
    return physics_bias.astype(np.float32), regime_table.astype(np.float32)


def train_one_fold_opt_a2(
    seq_tr, cand_tr, q_true_tr, regimes_tr,
    seq_va, cand_va, regimes_va,
    physics_bias, regime_table,
    *, epochs=EPOCHS, batch_size=BATCH, lr=LR, weight_decay=WD,
    prior_strength=PRIOR_STRENGTH, regime_prior_strength=REGIME_PRIOR_STRENGTH,
    val_frac=0.2, seed=SEED, device=DEVICE,
):
    torch.manual_seed(seed); np.random.seed(seed)
    n_tr = seq_tr.shape[0]
    val_split = int(n_tr * (1.0 - val_frac))

    seq_train = torch.from_numpy(seq_tr[:val_split]).float()
    cand_train = torch.from_numpy(cand_tr[:val_split]).float()
    q_train = torch.from_numpy(q_true_tr[:val_split]).float()
    regs_train = torch.from_numpy(regimes_tr[:val_split]).long()
    seq_val = torch.from_numpy(seq_tr[val_split:]).float().to(device)
    cand_val = torch.from_numpy(cand_tr[val_split:]).float().to(device)
    q_val = torch.from_numpy(q_true_tr[val_split:]).float().to(device)
    regs_val = torch.from_numpy(regimes_tr[val_split:]).long().to(device)

    physics_t = torch.from_numpy(physics_bias).float().to(device)
    regime_t = torch.from_numpy(regime_table).float().to(device)

    model = model_mod.CrossAttentionAnchorSelector().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf"); best_state = None; best_epoch = -1
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(val_split)
        for s_idx in range(0, val_split, batch_size):
            idx = perm[s_idx:s_idx + batch_size]
            seq_b = seq_train[idx].to(device, non_blocking=True)
            cand_b = cand_train[idx].to(device, non_blocking=True)
            q_b = q_train[idx].to(device, non_blocking=True)
            regs_b = regs_train[idx].to(device, non_blocking=True)
            _, score = model(seq_b, cand_b)
            bias_b = prior_strength * physics_t[None, :] + regime_prior_strength * regime_t[regs_b]
            q_pred = F.softmax(score + bias_b, dim=-1)
            loss = -(q_b * torch.log(q_pred + 1e-12)).sum(-1).mean()
            optim.zero_grad(); loss.backward(); optim.step()

        model.eval()
        with torch.no_grad():
            _, score_v = model(seq_val, cand_val)
            bias_v = prior_strength * physics_t[None, :] + regime_prior_strength * regime_t[regs_val]
            q_v = F.softmax(score_v + bias_v, dim=-1)
            val_loss = -(q_val * torch.log(q_v + 1e-12)).sum(-1).mean().item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    seq_va_t = torch.from_numpy(seq_va).float().to(device)
    cand_va_t = torch.from_numpy(cand_va).float().to(device)
    regs_va_t = torch.from_numpy(regimes_va).long().to(device)
    with torch.no_grad():
        _, score_va = model(seq_va_t, cand_va_t)
        bias_va = prior_strength * physics_t[None, :] + regime_prior_strength * regime_t[regs_va_t]
        q_pred_va = F.softmax(score_va + bias_va, dim=-1).cpu().numpy().astype(np.float32)
    return q_pred_va, best_epoch, best_val_loss


def main():
    t0 = time.time()
    print(f"[v8 OPT-A2] PyTorch={torch.__version__} threads={torch.get_num_threads()} device={DEVICE}", flush=True)
    print(f"[v8] EPOCHS={EPOCHS} prior_strength={PRIOR_STRENGTH} regime_prior_strength={REGIME_PRIOR_STRENGTH}", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    N = X.shape[0]
    folds = np.asarray([stable_fold_id(str(s), N_FOLDS) for s in ids], dtype=int)

    common = p021_build.build_input_common(X, bf.f0_baseline)
    R_wfn_all = common["R_wfn"]; pred_F0_world_all = common["pred_F0_world"]
    L1_frenet_all = common["L1"]
    mw_path = _THIS / "multiwindow_trim.json"
    if not mw_path.exists():
        mw_build.build_and_save(L1_frenet_all, output_path=mw_path)
    anchors = anchors_mod.ANCHORS_A6
    K = anchors.shape[0]
    q_true_all = som.build_soft_label_with_tau(Y, R_wfn_all, pred_F0_world_all, anchors, tau_cls=TAU_CLS)

    probs_all = np.zeros((N, K), dtype=np.float32)
    fold_meta = []

    for k_fold in range(N_FOLDS):
        t_f = time.time()
        tr = np.where(folds != k_fold)[0]
        va = np.where(folds == k_fold)[0]
        tr_sorted = tr[np.argsort([ids[i] for i in tr])]

        qc = qc_mod.build(X[tr_sorted], R_wfn_all[tr_sorted])
        regime_bins = fit_regime_bins(X[tr_sorted], end_idx=10)
        regimes_tr = assign_regimes(X[tr_sorted], end_idx=10, bins=regime_bins)
        regimes_va = assign_regimes(X[va], end_idx=10, bins=regime_bins)

        physics_bias, regime_table = compute_biases(
            anchors, R_wfn_all[tr_sorted], pred_F0_world_all[tr_sorted],
            Y[tr_sorted], regimes_tr, regime_count=18,
        )
        print(f"  fold {k_fold}: physics_bias [{physics_bias.min():.3f}, {physics_bias.max():.3f}] "
              f"regime_table mean={regime_table.mean():.3f}", flush=True)

        seq_tr = seq_builder.build(X[tr_sorted], R_wfn_all[tr_sorted], anchors, bf.f0_baseline,
                                    quantile_carry=qc, tau_past=TAU_PAST)
        cand_tr = cand_builder.build(X[tr_sorted], R_wfn_all[tr_sorted], pred_F0_world_all[tr_sorted],
                                      anchors, bf.f0_baseline, regimes=regimes_tr,
                                      quantile_carry=qc, multiwindow_trim_path=mw_path)
        seq_va = seq_builder.build(X[va], R_wfn_all[va], anchors, bf.f0_baseline,
                                    quantile_carry=qc, tau_past=TAU_PAST)
        cand_va = cand_builder.build(X[va], R_wfn_all[va], pred_F0_world_all[va], anchors, bf.f0_baseline,
                                      regimes=regimes_va, quantile_carry=qc,
                                      multiwindow_trim_path=mw_path)
        q_true_tr = q_true_all[tr_sorted]

        q_pred_va, best_epoch, best_val_loss = train_one_fold_opt_a2(
            seq_tr, cand_tr, q_true_tr, regimes_tr,
            seq_va, cand_va, regimes_va,
            physics_bias, regime_table,
        )
        probs_all[va] = q_pred_va

        anchors_world_va = (
            np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
            + pred_F0_world_all[va, None, :]
        )
        final_va = (q_pred_va[:, :, None] * anchors_world_va).sum(axis=1)
        hit_1cm_va = float((np.linalg.norm(final_va - Y[va], axis=1) <= R_HIT).mean())
        print(f"    fold {k_fold}: hit@1cm={hit_1cm_va:.4f} best_epoch={best_epoch} "
              f"best_val_loss={best_val_loss:.4f} time={time.time()-t_f:.0f}s", flush=True)
        fold_meta.append({"fold": k_fold, "hit_1cm": hit_1cm_va,
                          "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                          "time_s": time.time() - t_f})

    anchors_world_all = (
        np.einsum("nij,kj->nki", R_wfn_all, anchors.astype(np.float32))
        + pred_F0_world_all[:, None, :]
    )
    final_world_all = (probs_all[:, :, None] * anchors_world_all).sum(axis=1)
    d_cell = np.linalg.norm(final_world_all - Y, axis=1)
    d_F0 = np.linalg.norm(pred_F0_world_all - Y, axis=1)
    hit_1cm = float((d_cell <= R_HIT).mean())
    hit_15cm = float((d_cell <= R_HIT_LOOSE).mean())
    f0_hit_1cm = float((d_F0 <= R_HIT).mean())
    f0_hit_15cm = float((d_F0 <= R_HIT_LOOSE).mean())

    oracle_dist = np.linalg.norm(anchors_world_all - Y[:, None, :], axis=2).min(axis=1)
    oracle_1cm = float((oracle_dist <= R_HIT).mean())
    argmax_idx = probs_all.argmax(axis=1)
    argmax_pos = anchors_world_all[np.arange(N), argmax_idx]
    argmax_hit = float((np.linalg.norm(argmax_pos - Y, axis=1) <= R_HIT).mean())
    gap_ranking = oracle_1cm - argmax_hit
    pred_top1 = probs_all.argmax(axis=1)
    true_top1 = q_true_all.argmax(axis=1)
    top1_acc = float((pred_top1 == true_top1).mean())
    eps = 1e-12
    soft_CE = float(-(q_true_all.astype(np.float64) * np.log(probs_all + eps)).sum(1).mean())

    out = {
        "config": {
            "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD,
            "prior_strength": PRIOR_STRENGTH, "regime_prior_strength": REGIME_PRIOR_STRENGTH,
            "scheduler": "constant_no_decay", "best_epoch_tracking": True,
        },
        "N": N,
        "hit_1cm": hit_1cm, "hit_1.5cm": hit_15cm,
        "delta_1cm": hit_1cm - f0_hit_1cm, "delta_1.5cm": hit_15cm - f0_hit_15cm,
        "f0_hit_1cm": f0_hit_1cm, "f0_hit_1.5cm": f0_hit_15cm,
        "max_class_ratio": float(probs_all.mean(0).max()),
        "oracle_1cm": oracle_1cm, "argmax_hit": argmax_hit, "gap_ranking": gap_ranking,
        "top1_acc": top1_acc, "soft_CE": soft_CE,
        "fold_meta": fold_meta,
        "elapsed_sec": float(time.time() - t0),
    }
    out_path = _THIS / "results_v8_opt_a2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[v8] hit_1cm={hit_1cm:.4f} hit_1.5cm={hit_15cm:.4f} Δ_1cm={out['delta_1cm']:+.4f} "
          f"gap_ranking={gap_ranking:.4f}", flush=True)
    print(f"[v8] vs plan-022 winner 0.6528: Δ = {hit_1cm - 0.6528:+.4f}", flush=True)
    print(f"[v8] vs plan-024 v1 0.6370: Δ = {hit_1cm - 0.6370:+.4f}", flush=True)
    print(f"[v8] total {out['elapsed_sec']:.1f}s → {out_path}", flush=True)


if __name__ == "__main__":
    main()
