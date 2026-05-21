"""plan-024 combo + val_hit criterion — best state = val 의 hit_1cm 기준.

핵심 변화: train_fold 의 best state 선택 criterion을
  val_loss (decoupled) → val_hit_1cm (fold-internal val 의 hit rate)
로 변경. fold-internal val 의 Y 만 사용 — test fold 정보 안 씀 (no leakage).

config: hidden 128 + aug σ=0.05 + epoch 100 + patience 10
가설: val_loss-best 의 5-fold OOF 0.6377 보다 ↑ (val_hit 가 hit metric 직접 추적)
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


def train_fold(seq_tr_full, cand_tr_full, q_tr_full, Y_tr_full, R_wfn_tr_full, pred_F0_tr_full,
               seq_va, cand_va, anchors, seed=20260521):
    """best state = val_hit_1cm 최대 (fold-internal val 의 hit rate)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_tr = seq_tr_full.shape[0]
    val_split = int(n_tr * 0.8)

    seq_train = torch.from_numpy(seq_tr_full[:val_split]).float()
    cand_train = torch.from_numpy(cand_tr_full[:val_split]).float()
    q_train = torch.from_numpy(q_tr_full[:val_split]).float()
    seq_val_t = torch.from_numpy(seq_tr_full[val_split:]).float()
    cand_val_t = torch.from_numpy(cand_tr_full[val_split:]).float()
    seq_va_t = torch.from_numpy(seq_va).float()
    cand_va_t = torch.from_numpy(cand_va).float()

    # fold-internal val 의 ground truth — for val_hit_1cm
    Y_val_internal = Y_tr_full[val_split:]
    R_wfn_val_internal = R_wfn_tr_full[val_split:]
    pred_F0_val_internal = pred_F0_tr_full[val_split:]

    seq_std = torch.from_numpy(seq_tr_full.std(axis=(0, 1)).astype(np.float32))
    cand_std = torch.from_numpy(cand_tr_full.std(axis=(0, 1)).astype(np.float32))

    model = model_mod.CrossAttentionAnchorSelector(hidden=HIDDEN)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_hit = 0.0
    best_state = None
    no_improve = 0
    best_epoch = -1
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
            q_v_np = q_v.cpu().numpy()
            anchors_world_val = (
                np.einsum("nij,kj->nki", R_wfn_val_internal, anchors.astype(np.float32))
                + pred_F0_val_internal[:, None, :]
            )
            final_val = (q_v_np[:, :, None] * anchors_world_val).sum(axis=1)
            val_hit_1cm = float((np.linalg.norm(final_val - Y_val_internal, axis=1) <= R_HIT).mean())

        if val_hit_1cm > best_val_hit + 1e-5:
            best_val_hit = val_hit_1cm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        q_va, _ = model(seq_va_t, cand_va_t)
    return q_va.cpu().numpy().astype(np.float32), best_val_hit, best_epoch


def main():
    t0 = time.time()
    print(f"[combo + val_hit criterion] hidden={HIDDEN}, aug σ={NOISE_SIGMA_RATIO}", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    N = X.shape[0]
    folds = np.asarray([stable_fold_id(str(s), 5) for s in ids], dtype=int)
    common = p021_build.build_input_common(X, bf.f0_baseline)
    R_wfn_all = common["R_wfn"]; pred_F0_world_all = common["pred_F0_world"]
    regime_bins = fit_regime_bins(X, end_idx=10)
    regimes_all = assign_regimes(X, end_idx=10, bins=regime_bins)
    anchors = anchors_mod.ANCHORS_A6
    K = anchors.shape[0]
    q_true_all = som.build_soft_label_with_tau(Y, R_wfn_all, pred_F0_world_all, anchors, tau_cls=TAU_CLS)

    mw_path = _THIS / "multiwindow_trim.json"

    probs_all = np.zeros((N, K), dtype=np.float32)
    fold_meta = []

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

        q_pred_va, best_val_hit, best_ep = train_fold(
            seq_tr, cand_tr, q_true_tr,
            Y[tr_sorted], R_wfn_all[tr_sorted], pred_F0_world_all[tr_sorted],
            seq_va, cand_va, anchors,
        )
        probs_all[va] = q_pred_va

        # per-fold test hit
        anchors_world_va = (
            np.einsum("nij,kj->nki", R_wfn_all[va], anchors.astype(np.float32))
            + pred_F0_world_all[va, None, :]
        )
        final = (q_pred_va[:, :, None] * anchors_world_va).sum(axis=1)
        d = np.linalg.norm(final - Y[va], axis=1)
        hit = float((d <= R_HIT).mean())
        fold_meta.append({"fold": k_fold, "hit_1cm": hit, "best_val_hit": best_val_hit,
                           "best_epoch": best_ep, "time_s": time.time() - t_f})
        print(f"  fold {k_fold}: hit={hit:.4f} best_val_hit={best_val_hit:.4f} "
              f"best_ep={best_ep} {time.time()-t_f:.0f}s", flush=True)

    anchors_world_all = (
        np.einsum("nij,kj->nki", R_wfn_all, anchors.astype(np.float32))
        + pred_F0_world_all[:, None, :]
    )
    final_all = (probs_all[:, :, None] * anchors_world_all).sum(axis=1)
    d_cell = np.linalg.norm(final_all - Y, axis=1)
    f0_dist = np.linalg.norm(pred_F0_world_all - Y, axis=1)
    hit_1cm = float((d_cell <= R_HIT).mean())
    hit_15cm = float((d_cell <= 0.015).mean())

    out = {
        "config": {"hidden": HIDDEN, "aug": NOISE_SIGMA_RATIO, "criterion": "val_hit_1cm",
                   "patience": PATIENCE, "epochs": EPOCHS},
        "hit_1cm": hit_1cm, "hit_1.5cm": hit_15cm,
        "delta_1cm": hit_1cm - float((f0_dist <= R_HIT).mean()),
        "fold_meta": fold_meta,
        "elapsed_sec": time.time() - t0,
    }
    out_path = _THIS / "results_combo_valhit_5fold.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[val_hit] OOF hit_1cm={hit_1cm:.4f} hit_1.5cm={hit_15cm:.4f}", flush=True)
    print(f"[val_hit] vs plan-022 0.6528: {hit_1cm - 0.6528:+.4f}", flush=True)
    print(f"[val_hit] vs combo val_loss 0.6377: {hit_1cm - 0.6377:+.4f}", flush=True)
    print(f"[val_hit] total {out['elapsed_sec']:.0f}s → {out_path}", flush=True)


if __name__ == "__main__":
    main()
