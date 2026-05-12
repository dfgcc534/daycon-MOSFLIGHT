"""plan-011 c5 — Phase 1.L L1~L7 training (1-fold approx, fold=0).

L2 자동 skip (D001=0.6570 < 0.66 per G0). L1, L3, L4, L5, L6, L7 학습.

전략:
  - 데이터 prep 1회 (trajectory + 단일공식 candidate + cf features + Frenet basis).
  - 각 sub-exp 별로 corrector_redesign_v2.RedesignedCorrectionNet (또는 변형) build → loss-specific train → eval.
  - 산출 = runs/baseline/H011_phase1-loss-ablation/sub_L{N}/report_sub_L{N}.json + boundary_val_predictions.npz.

decision-note (자율 결정):
  - In-A anchor → dim_cf=32 (plan-008 v2.2 schema), encoder_emb=None (frozen GRU 미사용 anchor).
  - epochs=50, patience=10, batch=512, lr=1e-3, weight_decay=1e-4 (z1 minimum spec 적용).
  - L1 = Z1 minimum (huber + weight schedule + cap6mm).
  - L3 = L1 + asymmetric loss (gate=1, λ=8 replacement multiplier).
  - L4 = L1 + λ_aniso=1.0 * frenet_anisotropic (Frenet basis @ end_idx).
  - L5 = L1 + 0.5 * physics_conservation (delta/horizon vs recent_acc).
  - L6 = bell_shape_weight * huber (대체, not additive).
  - L7 = 0.5 * huber + 0.5 * hit_aware_hinge.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.pb_0_6822 import selector as sel
from src.pb_0_6822 import corrector_redesign_v2 as v2

REPO = Path(__file__).resolve().parents[2]
KST = timezone(timedelta(hours=9))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SINGLE_IDX = 17  # frenet_par120_perp_neg020
SINGLE_SPEC = sel.CANDIDATES[SINGLE_IDX]
N_FOLDS = 5
FOLD_VAL = 0
END_IDX = 10            # trajectories T=11, last index
HORIZON = 2
R_HIT = 0.01
CAP_6MM = 0.006
SEED_BASE = 20260513

BAND_EDGES_M = [(0.0, 0.005), (0.005, 0.010), (0.010, 0.015), (0.015, 0.020), (0.020, float("inf"))]
BAND_NAMES = ["[0,0.5cm)", "[0.5,1cm)", "[1,1.5cm)", "[1.5,2cm)", "[2cm,inf)"]


def weight_schedule(err: torch.Tensor, R: float = R_HIT) -> torch.Tensor:
    """plan-010 §5.1 C1+C2: easy=0, boundary=1.0, far=0.5."""
    easy = err < R * 0.7
    far = err >= R * 1.7
    return torch.where(easy, 0.0, torch.where(far, 0.5, 1.0))


def cap_6mm_np(delta: np.ndarray, cap: float = CAP_6MM) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = np.minimum(cap / (norm + 1e-6), 1.0)
    return delta * scale


def load_data(data_root: Path):
    """trajectory + truth + cf + targets + Frenet basis 일괄 로딩 (10000 samples)."""
    df = pd.read_csv(data_root / "train_labels.csv")
    sample_ids = df["id"].astype(str).tolist()
    truth = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    print(f"[data] loading {len(sample_ids)} trajectories...")
    x_seq = sel.load_stack(data_root / "train", sample_ids)        # (N, T=11, 3) float32

    # 단일공식 candidate: K=1, CANDIDATES[17]
    print(f"[data] generating single-formula candidates ({SINGLE_SPEC.name})...")
    all_cands = sel.make_candidates(x_seq, end_idx=END_IDX, horizon=HORIZON)  # (N, K=27, 3)
    cands = all_cands[:, SINGLE_IDX:SINGLE_IDX + 1, :]              # (N, 1, 3)

    # cf features (32-dim)
    print("[data] computing cf features (32-dim)...")
    cf_all = sel.make_candidate_features(
        x_seq, end_idx=END_IDX, candidates=cands, horizon=HORIZON,
        candidates_list=[SINGLE_SPEC],
    )                                                              # (N, 1, 32)
    cf = cf_all[:, 0, :]                                           # (N, 32)
    cand = cands[:, 0, :]                                          # (N, 3)

    # target: uncapped residual = truth − cand (plan-010 B1)
    target = (truth - cand).astype(np.float32)                     # (N, 3)
    err = np.linalg.norm(truth - cand, axis=1).astype(np.float32)  # (N,)

    # Frenet basis @ end_idx for L4
    print("[data] computing Frenet basis @ end_idx...")
    R, valid = v2.build_frenet_basis(
        torch.from_numpy(x_seq), torch.tensor([END_IDX] * len(x_seq))
    )
    R_np = R.cpu().numpy()                                          # (N, 3, 3)

    # recent_acc for L5 (step-domain)
    v_last = x_seq[:, END_IDX] - x_seq[:, END_IDX - 1]              # (N, 3) m/step
    v_prev = x_seq[:, END_IDX - 1] - x_seq[:, END_IDX - 2]
    recent_acc = (v_last - v_prev).astype(np.float32)               # (N, 3) m/step²

    # fold ids
    fold_ids = np.asarray([sel.stable_fold_id(sid, N_FOLDS) for sid in sample_ids])
    print(f"[data] fold={FOLD_VAL} val n={int((fold_ids==FOLD_VAL).sum())}")

    return {
        "sample_ids": sample_ids,
        "cf": cf,
        "cand": cand,
        "target": target,
        "truth": truth,
        "err": err,
        "R": R_np,
        "recent_acc": recent_acc,
        "fold_ids": fold_ids,
    }


def split_fold(data: dict) -> tuple[dict, dict]:
    tr_mask = data["fold_ids"] != FOLD_VAL
    va_mask = data["fold_ids"] == FOLD_VAL
    def slice_(m): return {k: v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v
                            for k, v in data.items()}
    return slice_(tr_mask), slice_(va_mask)


def compute_per_band(raw_err, corrected_err) -> dict:
    band = {}
    for name, (lo, hi) in zip(BAND_NAMES, BAND_EDGES_M):
        mask = (raw_err >= lo) & (raw_err < hi)
        n = int(mask.sum())
        hb = float((raw_err[mask] <= R_HIT).mean()) if n else 0.0
        ha = float((corrected_err[mask] <= R_HIT).mean()) if n else 0.0
        band[name] = {"n": n, "hit_before": hb, "hit_after": ha, "delta": float(ha - hb)}
    return band


def make_model(sub_exp: str, dim_cf: int = 32, hidden: int = 64) -> nn.Module:
    """sub_exp 별 corrector arch."""
    if sub_exp == "L2":
        return v2.GateHeadCorrector(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)
    return v2.RedesignedCorrectionNet(dim_cf=dim_cf, hidden=hidden, dim_encoder=0)


def compute_loss(
    sub_exp: str,
    delta: torch.Tensor,
    aux: dict,
    target: torch.Tensor,
    err_raw: torch.Tensor,
    R_basis: torch.Tensor,
    recent_acc: torch.Tensor,
    cand: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    """sub_exp 별 wrapper-level total loss (per plan-011 §5.2 표)."""
    # weight schedule (C1+C2) — L0 도 동일 schedule, L1~L7 all
    w = weight_schedule(err_raw)
    corrected_pos = cand + v2.cap_6mm(delta)

    if sub_exp == "L1":
        per = v2.huber_loss(delta, target)
        return (per * w).sum() / (w.sum() + 1e-8)

    if sub_exp == "L2":  # gate + asymmetric — DISABLED per G0 (D001 < 0.66)
        raw_delta = aux["raw_delta"]
        raw_hit = err_raw <= R_HIT
        per = v2.asymmetric_loss(raw_delta, target, raw_hit, corrected_pos, lambda_destructive=8.0)
        return (per * w).sum() / (w.sum() + 1e-8)

    if sub_exp == "L3":  # asymmetric only, gate ≡ 1
        raw_hit = err_raw <= R_HIT
        per = v2.asymmetric_loss(delta, target, raw_hit, corrected_pos, lambda_destructive=8.0)
        return (per * w).sum() / (w.sum() + 1e-8)

    if sub_exp == "L4":  # huber + λ_aniso × Frenet anisotropic
        pred_local = v2.world_to_local(delta, R_basis)
        target_local = v2.world_to_local(target, R_basis)
        per = v2.huber_loss(delta, target) + 1.0 * v2.frenet_anisotropic_loss(pred_local, target_local)
        return (per * w).sum() / (w.sum() + 1e-8)

    if sub_exp == "L5":  # huber + 0.5 × physics_conservation
        per = v2.huber_loss(delta, target) + 0.5 * v2.physics_conservation_loss(delta / HORIZON, recent_acc)
        return (per * w).sum() / (w.sum() + 1e-8)

    if sub_exp == "L6":  # bell_shape_weight × huber (replacement weighting)
        err = torch.norm(corrected_pos - truth, dim=1)
        bell = v2.bell_shape_weight(err)
        per = v2.huber_loss(delta, target)
        return (bell * per).mean()

    if sub_exp == "L7":  # 0.5 × huber + 0.5 × hit_aware_hinge
        per = 0.5 * v2.huber_loss(delta, target) + 0.5 * v2.hit_aware_hinge(corrected_pos, truth)
        return (per * w).sum() / (w.sum() + 1e-8)

    raise ValueError(f"unknown sub_exp {sub_exp}")


def train_one_sub_exp(
    sub_exp: str, data_tr: dict, data_va: dict, args: argparse.Namespace
) -> dict:
    """단일 sub-exp 학습 + fold-0 eval. 학습 device 자동 (CUDA available 시 CUDA, else CPU)."""
    t0 = time.time()
    torch.manual_seed(SEED_BASE + int(sub_exp[1]))
    np.random.seed(SEED_BASE + int(sub_exp[1]))

    model = make_model(sub_exp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # tensors
    def to_t(d, k): return torch.from_numpy(d[k]).to(DEVICE)
    cf_tr, target_tr, err_tr = to_t(data_tr, "cf"), to_t(data_tr, "target"), to_t(data_tr, "err")
    R_tr, racc_tr, cand_tr, truth_tr = to_t(data_tr, "R"), to_t(data_tr, "recent_acc"), \
        to_t(data_tr, "cand"), to_t(data_tr, "truth")

    cf_va, target_va, err_va = to_t(data_va, "cf"), to_t(data_va, "target"), to_t(data_va, "err")
    R_va, racc_va, cand_va, truth_va = to_t(data_va, "R"), to_t(data_va, "recent_acc"), \
        to_t(data_va, "cand"), to_t(data_va, "truth")

    n_tr = cf_tr.shape[0]
    best_hit, best_state, wait = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(n_tr, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, n_tr, args.batch):
            sel_ = idx[start:start + args.batch]
            cf_b = cf_tr[sel_]; tg_b = target_tr[sel_]
            er_b = err_tr[sel_]; R_b = R_tr[sel_]; ra_b = racc_tr[sel_]
            cd_b = cand_tr[sel_]; tr_b = truth_tr[sel_]

            opt.zero_grad(set_to_none=True)
            delta, aux = model(cf_b)
            loss = compute_loss(sub_exp, delta, aux, tg_b, er_b, R_b, ra_b, cd_b, tr_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(cf_b)
            n += len(cf_b)

        # eval
        model.eval()
        with torch.no_grad():
            delta_va, _ = model(cf_va)
            corrected_pos_va = cand_va + v2.cap_6mm(delta_va)
            err_va_after = torch.norm(corrected_pos_va - truth_va, dim=1)
            hit = float((err_va_after <= R_HIT).float().mean())

        if hit > best_hit:
            best_hit = hit; wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch % 5 == 0 or wait >= args.patience:
            print(f"  [{sub_exp}] ep{epoch:3d} loss={total/max(n,1):.4f} val_hit={hit:.4f} best={best_hit:.4f} wait={wait}")
        if wait >= args.patience and epoch >= args.min_epochs:
            break

    # load best + final eval
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        delta_va, _ = model(cf_va)
        delta_va_np = delta_va.cpu().numpy()
    corrected_pos_va_np = data_va["cand"] + cap_6mm_np(delta_va_np)
    err_after = np.linalg.norm(corrected_pos_va_np - data_va["truth"], axis=1)
    err_before = data_va["err"]
    oof_soft = float((err_after <= R_HIT).mean())
    oof_raw = float((err_before <= R_HIT).mean())
    per_band = compute_per_band(err_before, err_after)
    elapsed = time.time() - t0

    return {
        "sub_exp": f"P1.{sub_exp}",
        "n_val": int(len(err_after)),
        "fold": FOLD_VAL,
        "oof_soft_hit": oof_soft,
        "oof_raw_hit": oof_raw,
        "corrector_gain": float(oof_soft - oof_raw),
        "per_band_hit_after": per_band,
        "elapsed_sec": elapsed,
        "best_val_hit": best_hit,
        "device": str(DEVICE),
        "predictions_npz": True,  # caller saves it
    }, corrected_pos_va_np, delta_va_np


def main():
    parser = argparse.ArgumentParser(description="plan-011 c5 Phase 1.L training")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs/baseline/H011_phase1-loss-ablation")
    parser.add_argument("--summary-dir", type=Path, default=REPO / "analysis/plan-011")
    parser.add_argument("--sub-exps", type=str, default="L1,L3,L4,L5,L6,L7",
                        help="L2 자동 skip per G0; default = all valid sub-exp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = load_data(args.data_root)
    data_tr, data_va = split_fold(data)
    print(f"[fold-0] train n={len(data_tr['cf'])}, val n={len(data_va['cf'])}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_exps = [s.strip() for s in args.sub_exps.split(",") if s.strip()]
    reports = {}
    for sub_exp in sub_exps:
        print(f"\n=== {sub_exp} ===")
        report, corrected, delta = train_one_sub_exp(sub_exp, data_tr, data_va, args)
        reports[f"P1.{sub_exp}"] = report
        sub_dir = args.out_dir / f"sub_{sub_exp}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / f"report_sub_{sub_exp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        np.savez(
            sub_dir / "boundary_val_predictions.npz",
            corrected_pos=corrected.astype(np.float32),
            delta=delta.astype(np.float32),
            sample_ids=np.array([sid for sid, m in zip(data["sample_ids"], data["fold_ids"] == FOLD_VAL) if m]),
        )
        print(f"  → oof_soft_hit={report['oof_soft_hit']:.4f}, corrector_gain={report['corrector_gain']:+.4f}, "
              f"elapsed={report['elapsed_sec']:.1f}s")

    # merge with existing summary
    summary_path = args.summary_dir / "phase1_loss_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"sub_exps": {}}
    summary["sub_exps"].update(reports)

    # L1 anchor 박제 (delta vs anchor + z1) calculation
    anchor_oof = summary["sub_exps"].get("P1.L0", {}).get("oof_soft_hit")
    z1_oof = summary["sub_exps"].get("P1.L1", {}).get("oof_soft_hit")
    for k, r in summary["sub_exps"].items():
        if anchor_oof is not None:
            r["delta_vs_anchor"] = r["oof_soft_hit"] - anchor_oof
        if z1_oof is not None and k != "P1.L0":
            r["delta_vs_z1"] = r["oof_soft_hit"] - z1_oof

    # axis-level aggregation (max ΔOOF over L1~L7 vs anchor)
    non_anchor = [v for k, v in summary["sub_exps"].items() if k != "P1.L0" and "delta_vs_anchor" in v]
    max_delta = max((v["delta_vs_anchor"] for v in non_anchor), default=0.0)
    summary["axis_positive_threshold_0p005"] = bool(max_delta >= 0.005)
    summary["max_delta_vs_anchor"] = float(max_delta)
    best = max(non_anchor, key=lambda v: v["delta_vs_anchor"], default=None)
    summary["best_lever"] = best["sub_exp"] if best else None
    summary["generated_at"] = datetime.now(KST).isoformat()

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✓ summary updated → {summary_path.relative_to(REPO)}")
    print(f"  best_lever={summary['best_lever']}, max_delta_vs_anchor={max_delta:+.4f}, "
          f"axis_positive={summary['axis_positive_threshold_0p005']}")


if __name__ == "__main__":
    main()
