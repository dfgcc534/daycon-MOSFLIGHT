"""plan-b-001 c7 — train.py (yaw soft-label + softhit loss + 3-seed + 2-arm baseline).

per-fold pipeline (plan-030 carry + plan-b-001 개작):
  Frenet R_wfn (carry: seq_builder/cand_ext/quantile/regime) + yaw θ/R_wfy (개작: residual/decode/soft_label).
  baseline_at_fn ∈ {f0, kalman} (2-arm). 3-seed/fold 평균.

loss = soft_CE(probs, q_τ) + 0.3·softhit(world_pred, gt).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
for p in (_REPO, _THIS.parent / "plan-020", _THIS.parent / "plan-021",
          _THIS.parent / "plan-022", _THIS.parent / "plan-024",
          _THIS.parent / "plan-029"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# carry modules
_bf = _load(_THIS.parent / "plan-020" / "baseline_f0.py", "pb_bf")
_p021 = _load(_THIS.parent / "plan-021" / "build_input.py", "pb_p021")
_anchors = _load(_THIS.parent / "plan-022" / "anchors.py", "pb_anchors")
_seq_mod = _load(_THIS.parent / "plan-024" / "seq_builder.py", "pb_seq")
_qc_mod = _load(_THIS.parent / "plan-024" / "quantile_carry.py", "pb_qc")
_aqe = _load(_THIS.parent / "plan-029" / "anchor_query_extend.py", "pb_aqe")
# plan-b-001 modules
_yaw = _load(_THIS / "yaw_frame.py", "pb_yaw")
_kalman = _load(_THIS / "kalman_cv.py", "pb_kalman")
_residual = _load(_THIS / "residual_builder.py", "pb_residual")
_query = _load(_THIS / "query_builder.py", "pb_query")
_head = _load(_THIS / "head_summary.py", "pb_head")
_noise = _load(_THIS / "noise_estimator.py", "pb_noise")
_tier3 = _load(_THIS / "tier3.py", "pb_tier3")
_model_mod = _load(_THIS / "model.py", "pb_model")

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes  # noqa: E402

N_FOLDS = 5
K_ANCHORS = 14
HIDDEN = 196
ATTN_DIM = 128
HEAD_HIDDEN = 256
EPOCHS = 50
WARMUP_EP = 5
LR = 7e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 64
EVAL_BATCH = 256
TAU_CLS = 0.001
REGIME_COUNT = 18
N_SEEDS = 3
SEED = 20260526
LAMBDA_SOFTHIT = 0.3
SOFTHIT_BETA = 0.002
R_HIT = 0.01


def baseline_at_factory(name: str) -> Callable[[np.ndarray, int], np.ndarray]:
    if name == "f0":
        return lambda X, t_idx: _bf.f0_baseline(X[:, t_idx - 2:t_idx + 1, :], 2)
    if name == "kalman":
        return _kalman.kalman_baseline_at
    raise ValueError(f"unknown baseline {name}")


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.nan_to_num(torch.from_numpy(np.ascontiguousarray(arr)).float(),
                            nan=0.0, posinf=1e3, neginf=-1e3)


def _yaw_soft_label(gt: np.ndarray, baseline_final: np.ndarray, theta: np.ndarray,
                    anchors: np.ndarray, tau: float) -> np.ndarray:
    """yaw-frame soft label: q = softmax(-d_k/τ), d_k = ‖to_yaw(gt-baseline) - anchor_k‖. → (N,K)."""
    res_world = gt.astype(np.float64) - baseline_final.astype(np.float64)
    res_yaw = _yaw.to_yaw(res_world, theta.astype(np.float64))          # (N,3)
    d = np.linalg.norm(res_yaw[:, None, :] - anchors[None, :, :].astype(np.float64), axis=-1)  # (N,K)
    logits = -d / tau
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _build_fold_artifacts(X, gt, baseline_at_fn, baseline_name):
    """모든 per-sample artifact 산출 (train/test 공통 호출). gt=None 이면 soft label skip."""
    R_wfn = _p021.build_frenet_basis_3d(X, end_idx=10).astype(np.float32)   # carry (seq/cand)
    F0 = _bf.f0_baseline(X, end_idx=10).astype(np.float32)                   # carry input 용
    theta = _yaw.yaw_from_X(X, end_idx=10)                                   # (N,) yaw
    R_wfy = _yaw.build_R_wfy(theta).astype(np.float32)                       # yaw→world
    baseline_final = np.asarray(baseline_at_fn(X, 10), dtype=np.float32)     # (N,3) decode baseline
    return {"R_wfn": R_wfn, "F0": F0, "theta": theta, "R_wfy": R_wfy,
            "baseline_final": baseline_final, "X": X}


def _build_inputs(X, art, qc, regimes, regime_anchor_table, baseline_at_fn):
    """seq98 / residual_b / query29 / head56 / slim7 산출."""
    R_wfn, F0, theta, R_wfy = art["R_wfn"], art["F0"], art["theta"], art["R_wfy"]
    cand_ext = _aqe.build(
        X, R_wfn, F0, _anchors.ANCHORS_A6, _bf.f0_baseline,
        regimes=regimes, quantile_carry=qc,
        regime_count=REGIME_COUNT, regime_anchor_table=regime_anchor_table,
    )
    seq95 = _seq_mod.build(X, R_wfn, _anchors.ANCHORS_A6, _bf.f0_baseline, qc)   # (N,7,95)
    res = _residual.build_residuals(X, theta, R_wfy, _anchors.ANCHORS_A6, baseline_at_fn)
    seq98 = np.concatenate([seq95, res["residual_a"]], axis=-1).astype(np.float32)  # (N,7,98)
    slim7 = _query.extract_slim7_from_cand_ext_165(cand_ext)
    cand_feat_150 = cand_ext[:, :, :150]
    query29 = _query.build_query(cand_feat_150, slim7)
    macro9 = _p021._macro_stat_9d(X, end_idx=10)
    _L2, L4 = _p021._build_L2_L4(X, R_wfn, _bf.f0_baseline)
    L4_flat = L4.reshape(L4.shape[0], -1).astype(np.float32)
    noise2 = _noise.build_noise(X)
    tier3_5 = _tier3.build_tier3(X)
    head56 = _head.build_head_summary(cand_feat_150, macro9, L4_flat, noise2, tier3_5)
    return {"seq98": seq98, "residual_b": res["residual_b"], "query29": query29,
            "head56": head56, "slim7": slim7}


def _make_model(seed: int):
    torch.manual_seed(seed)
    anchors_t = torch.from_numpy(_anchors.ANCHORS_A6.astype(np.float32))
    return _model_mod.GRUNetX3(
        seq_dim=98, query_dim=29, head_summary_dim=56, slim7_dim=7,
        hidden=HIDDEN, attn_dim=ATTN_DIM, head_hidden=HEAD_HIDDEN,
        K=K_ANCHORS, anchors=anchors_t,
    )


def _train_seed(seed, inp_tr, art_tr, q_tr, gt_tr, inp_te, art_te, verbose, fold):
    model = _make_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_EP),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EP),
        ],
        milestones=[WARMUP_EP],
    )
    N_tr = inp_tr["seq98"].shape[0]
    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1000)
        perm = rng.permutation(N_tr)
        last = 0.0
        for bs in range(0, N_tr, BATCH_SIZE):
            idx = perm[bs:bs + BATCH_SIZE]
            seq_b = _to_torch(inp_tr["seq98"][idx])
            rb_b = _to_torch(inp_tr["residual_b"][idx])
            q29_b = _to_torch(inp_tr["query29"][idx])
            hs_b = _to_torch(inp_tr["head56"][idx])
            sl_b = _to_torch(inp_tr["slim7"][idx])
            base_b = _to_torch(art_tr["baseline_final"][idx])
            R_b = _to_torch(art_tr["R_wfy"][idx])
            qsoft_b = torch.from_numpy(q_tr[idx]).float()
            gt_b = _to_torch(gt_tr[idx])

            opt.zero_grad()
            world_pred, probs = model(seq_b, rb_b, q29_b, hs_b, sl_b, base_b, R_b)
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss_ce = -(qsoft_b * log_probs).sum(dim=-1).mean()
            d = torch.sqrt(((world_pred - gt_b) ** 2).sum(dim=-1) + 1e-12)
            loss_sh = torch.sigmoid((d - R_HIT) / SOFTHIT_BETA).mean()
            loss = loss_ce + LAMBDA_SOFTHIT * loss_sh
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            last = loss.item()
        sched.step()
        if verbose and (epoch + 1) % 25 == 0:
            print(f"    fold{fold} seed{seed % 100} ep{epoch+1}/{EPOCHS} loss={last:.4f}")

    # eval
    model.eval()
    preds, probs_l = [], []
    Nte = inp_te["seq98"].shape[0]
    with torch.no_grad():
        for es in range(0, Nte, EVAL_BATCH):
            sl_ = slice(es, es + EVAL_BATCH)
            wp, pb = model(
                _to_torch(inp_te["seq98"][sl_]), _to_torch(inp_te["residual_b"][sl_]),
                _to_torch(inp_te["query29"][sl_]), _to_torch(inp_te["head56"][sl_]),
                _to_torch(inp_te["slim7"][sl_]), _to_torch(art_te["baseline_final"][sl_]),
                _to_torch(art_te["R_wfy"][sl_]),
            )
            preds.append(wp.numpy()); probs_l.append(pb.numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(probs_l).astype(np.float32)


def train_one_fold(fold, X_tr, X_te, gt_tr, baseline_name, verbose=True):
    t0 = time.perf_counter()
    baseline_at_fn = baseline_at_factory(baseline_name)

    art_tr = _build_fold_artifacts(X_tr, gt_tr, baseline_at_fn, baseline_name)
    art_te = _build_fold_artifacts(X_te, None, baseline_at_fn, baseline_name)

    qc = _qc_mod.build(X_tr, art_tr["R_wfn"])
    bins = fit_regime_bins(X_tr, end_idx=10)
    reg_tr = assign_regimes(X_tr, end_idx=10, bins=bins)
    reg_te = assign_regimes(X_te, end_idx=10, bins=bins)
    rat = _aqe.build_regime_anchor_lookup(
        gt_train=gt_tr, regimes_train=reg_tr, ANCHORS_A6=_anchors.ANCHORS_A6,
        R_wfn_train=art_tr["R_wfn"], F0_train=art_tr["F0"],
        regime_count=REGIME_COUNT, laplace=1.0,
    )
    inp_tr = _build_inputs(X_tr, art_tr, qc, reg_tr, rat, baseline_at_fn)
    inp_te = _build_inputs(X_te, art_te, qc, reg_te, rat, baseline_at_fn)

    q_tr = _yaw_soft_label(gt_tr, art_tr["baseline_final"], art_tr["theta"],
                           _anchors.ANCHORS_A6, TAU_CLS)

    seed_preds, seed_probs = [], []
    for s in range(N_SEEDS):
        wp, pb = _train_seed(SEED + fold * 10 + s, inp_tr, art_tr, q_tr, gt_tr,
                             inp_te, art_te, verbose, fold)
        seed_preds.append(wp); seed_probs.append(pb)
    world_pred = np.mean(seed_preds, axis=0).astype(np.float32)
    probs = np.mean(seed_probs, axis=0).astype(np.float32)

    return {
        "fold": fold, "world_pred_te": world_pred, "probs_te": probs,
        "baseline_final_te": art_te["baseline_final"],
        "elapsed_s": time.perf_counter() - t0,
    }


def run_5fold_oof(baseline_name, verbose=True):
    t_total = time.perf_counter()
    _ids, X = load_all_samples()
    _ids2, gt = load_labels()
    assert _ids == _ids2
    X = X.astype(np.float32); gt = gt.astype(np.float32)
    N = X.shape[0]
    folds = np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in _ids], dtype=int)

    oof_pred = np.zeros((N, 3), np.float32)
    oof_base = np.zeros((N, 3), np.float32)
    fold_logs = []
    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]; te = np.where(folds == fold)[0]
        log = train_one_fold(fold, X[tr], X[te], gt[tr], baseline_name, verbose)
        oof_pred[te] = log["world_pred_te"]
        oof_base[te] = log["baseline_final_te"]
        fold_logs.append({"fold": fold, "elapsed_s": log["elapsed_s"], "N_te": int(len(te))})
        if verbose:
            print(f"[fold {fold}] done {log['elapsed_s']:.1f}s")

    err = np.linalg.norm(oof_pred - gt, axis=1)
    err_base = np.linalg.norm(oof_base - gt, axis=1)
    return {
        "baseline": baseline_name,
        "hit_1cm": float((err <= 0.01).mean()),
        "hit_1p5cm": float((err <= 0.015).mean()),
        "baseline_hit_1cm": float((err_base <= 0.01).mean()),   # G0.5 standalone
        "baseline_hit_1p5cm": float((err_base <= 0.015).mean()),
        "N_total": int(N), "K": K_ANCHORS, "n_seeds": N_SEEDS,
        "elapsed_total_s": time.perf_counter() - t_total,
        "fold_logs": fold_logs,
        "oof_pred": oof_pred,
    }


def _smoke():
    global EPOCHS, WARMUP_EP, N_SEEDS
    EPOCHS, WARMUP_EP, N_SEEDS = 2, 1, 1
    rng = np.random.default_rng(SEED)
    X_tr = np.cumsum(rng.standard_normal((24, 11, 3)) * 0.01, axis=1).astype(np.float32)
    X_te = np.cumsum(rng.standard_normal((8, 11, 3)) * 0.01, axis=1).astype(np.float32)
    gt_tr = X_tr[:, 10, :] + rng.standard_normal((24, 3)).astype(np.float32) * 0.005
    for bn in ("f0", "kalman"):
        log = train_one_fold(0, X_tr, X_te, gt_tr, bn, verbose=False)
        assert log["world_pred_te"].shape == (8, 3)
        assert log["probs_te"].shape == (8, K_ANCHORS)
        assert not np.isnan(log["world_pred_te"]).any()
        print(f"[smoke] train({bn}) OK — pred {log['world_pred_te'].shape}, {log['elapsed_s']:.1f}s")


if __name__ == "__main__":
    _smoke()
