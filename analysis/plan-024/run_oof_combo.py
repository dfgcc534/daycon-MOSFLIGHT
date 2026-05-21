"""plan-024 combo 5-fold OOF — hidden 128 + input Gaussian noise aug + epoch 100.

1-fold best (epoch 70): 0.6515 → 5-fold OOF 평균 확인.
- fold 별 best val_loss epoch state 자동 저장
- 5-fold concat hit_1cm 측정

config:
  - hidden 128, lr 5e-4 constant, wd 0.02, batch 256, epoch 100
  - input Gaussian noise σ=0.05×std (train only)
  - early stop: patience 10 (best val_loss tracking)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes

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
_spec = importlib.util.spec_from_file_location("p024_seq", _THIS / "seq_builder.py")
seq_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(seq_builder)
_spec = importlib.util.spec_from_file_location("p024_cand", _THIS / "cand_builder.py")
cand_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cand_builder)
_spec = importlib.util.spec_from_file_location("p024_model", _THIS / "model.py")
model_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(model_mod)

N_FOLDS = 5
HIDDEN = 128
EPOCHS = 100
LR = 5e-4
WD = 0.02
BATCH = 256
NOISE_SIGMA_RATIO = 0.05
PATIENCE = 10
TAU_CLS = 0.001
R_HIT = 0.01
R_HIT_LOOSE = 0.015


def train_fold(seq_tr_full, cand_tr_full, q_tr_full, seq_va, cand_va,
               R_wfn_va, pred_F0_va, Y_va, anchors, seed=20260521):
    torch.manual_seed(seed); np.random.seed(seed)
    n_tr = seq_tr_full.shape[0]
    val_split = int(n_tr * 0.8)

    seq_train = torch.from_numpy(seq_tr_full[:val_split]).float()
    cand_train = torch.from_numpy(cand_tr_full[:val_split]).float()
    q_train = torch.from_numpy(q_tr_full[:val_split]).float()
    seq_val_t = torch.from_numpy(seq_tr_full[val_split:]).float()
    cand_val_t = torch.from_numpy(cand_tr_full[val_split:]).float()
    q_val_t = torch.from_numpy(q_tr_full[val_split:]).float()
    seq_va_t = torch.from_numpy(seq_va).float()
    cand_va_t = torch.from_numpy(cand_va).float()

    seq_std = torch.from_numpy(seq_tr_full.std(axis=(0, 1)).astype(np.float32))
    cand_std = torch.from_numpy(cand_tr_full.std(axis=(0, 1)).astype(np.float32))

    model = model_mod.CrossAttentionAnchorSelector(hidden=HIDDEN)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(val_split)
        for s_idx in range(0, val_split, BATCH):
            idx = perm[s_idx:s_idx + BATCH]
            seq_b = seq_train[idx]
            cand_b = cand_train[idx]
            seq_b = seq_b + torch.randn_like(seq_b) * (NOISE_SIGMA_RATIO * seq_std[None, None, :])
            cand_b = cand_b + torch.randn_like(cand_b) * (NOISE_SIGMA_RATIO * cand_std[None, None, :])
            q_pred, _ = model(seq_b, cand_b)
            loss = -(q_train[idx] * torch.log(q_pred + 1e-12)).sum(-1).mean()
            optim.zero_grad(); loss.backward(); optim.step()

        model.eval()
        with torch.no_grad():
            q_v, _ = model(seq_val_t, cand_val_t)
            val_loss = -(q_val_t * torch.log(q_v + 1e-12)).sum(-1).mean().item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        q_va, _ = model(seq_va_t, cand_va_t)
    return q_va.cpu().numpy().astype(np.float32), best_val_loss


def main():
    t0 = time.time()
    print(f"[combo 5fold] hidden={HIDDEN}, aug σ={NOISE_SIGMA_RATIO}×std, "
          f"epoch={EPOCHS} max, patience={PATIENCE}", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    N = X.shape[0]
    folds = np.asarray([stable_fold_id(str(s), 5) for s in ids], dtype=int)
    common = p021_build.build_input_common(X, bf.f0_baseline)
    R_wfn_all = common["R_wfn"]; pred_F0_world_all = common["pred_F0_world"]
    L1_frenet_all = common["L1"]
    regime_bins = fit_regime_bins(X, end_idx=10)
    regimes_all = assign_regimes(X, end_idx=10, bins=regime_bins)
    anchors = anchors_mod.ANCHORS_A6
    K = anchors.shape[0]
    q_true_all = som.build_soft_label_with_tau(Y, R_wfn_all, pred_F0_world_all, anchors, tau_cls=TAU_CLS)

    mw_path = _THIS / "multiwindow_trim.json"
    if not mw_path.exists():
        from importlib.util import spec_from_file_location, module_from_spec
        _s = spec_from_file_location("p024_mw", _THIS / "multiwindow_trim_build.py")
        mw_build = module_from_spec(_s); _s.loader.exec_module(mw_build)
        mw_build.build_and_save(L1_frenet_all, output_path=mw_path)

    probs_all = np.zeros((N, K), dtype=np.float32)

    for k_fold in range(N_FOLDS):
        t_f = time.time()
        tr = np.where(folds != k_fold)[0]
        va = np.where(folds == k_fold)[0]
        tr_sorted = tr[np.argsort([ids[i] for i in tr])]
        qc = qc_mod.build(X[tr_sorted], R_wfn_all[tr_sorted])

        seq_tr = seq_builder.build(X[tr_sorted], R_wfn_all[tr_sorted], anchors, bf.f0_baseline,
                                    quantile_carry=qc, tau_past=0.003)
        cand_tr = cand_builder.build(X[tr_sorted], R_wfn_all[tr_sorted], pred_F0_world_all[tr_sorted],
                                      anchors, bf.f0_baseline, regimes=regimes_all[tr_sorted],
                                      quantile_carry=qc, multiwindow_trim_path=mw_path)
        seq_va = seq_builder.build(X[va], R_wfn_all[va], anchors, bf.f0_baseline,
                                    quantile_carry=qc, tau_past=0.003)
        cand_va = cand_builder.build(X[va], R_wfn_all[va], pred_F0_world_all[va], anchors, bf.f0_baseline,
                                      regimes=regimes_all[va], quantile_carry=qc,
                                      multiwindow_trim_path=mw_path)
        q_true_tr = q_true_all[tr_sorted]

        q_pred_va, best_val = train_fold(seq_tr, cand_tr, q_true_tr, seq_va, cand_va,
                                         R_wfn_all[va], pred_F0_world_all[va], Y[va], anchors)
        probs_all[va] = q_pred_va

        # per-fold hit rate
        anchors_world_va = (
            np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
            + pred_F0_world_all[va, None, :]
        )
        final = (q_pred_va[:, :, None] * anchors_world_va).sum(axis=1)
        d = np.linalg.norm(final - Y[va], axis=1)
        print(f"  fold {k_fold}: hit@1cm={(d<=R_HIT).mean():.4f} val_loss={best_val:.4f} "
              f"time={time.time()-t_f:.0f}s", flush=True)

    # OOF concat metric
    anchors_world_all = (
        np.einsum("nij,kj->nki", R_wfn_all, anchors.astype(np.float32))
        + pred_F0_world_all[:, None, :]
    )
    final_all = (probs_all[:, :, None] * anchors_world_all).sum(axis=1)
    d_cell = np.linalg.norm(final_all - Y, axis=1)
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
    top1_acc = float((probs_all.argmax(1) == q_true_all.argmax(1)).mean())

    out = {
        "config": {"hidden": HIDDEN, "epochs": EPOCHS, "lr": LR, "wd": WD,
                   "noise_sigma_ratio": NOISE_SIGMA_RATIO, "patience": PATIENCE},
        "N": N,
        "hit_1cm": hit_1cm, "hit_1.5cm": hit_15cm,
        "delta_1cm": hit_1cm - f0_hit_1cm, "delta_1.5cm": hit_15cm - f0_hit_15cm,
        "f0_hit_1cm": f0_hit_1cm,
        "oracle_1cm": oracle_1cm, "argmax_hit": argmax_hit,
        "gap_ranking": oracle_1cm - argmax_hit,
        "top1_acc": top1_acc,
        "elapsed_sec": float(time.time() - t0),
    }
    out_path = _THIS / "results_combo_5fold.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[combo 5fold] hit_1cm={hit_1cm:.4f} hit_1.5cm={hit_15cm:.4f} "
          f"Δ_F0={out['delta_1cm']:+.4f} gap_ranking={out['gap_ranking']:.4f}", flush=True)
    print(f"[combo 5fold] vs plan-022 0.6528: {hit_1cm - 0.6528:+.4f}", flush=True)
    print(f"[combo 5fold] total {out['elapsed_sec']:.0f}s → {out_path}", flush=True)


if __name__ == "__main__":
    main()
