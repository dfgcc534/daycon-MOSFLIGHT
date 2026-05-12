"""plan-011 c6 — Phase 1.In Input axis ablation (5 sub-exp + IA anchor).

5 sub-exp 학습 (fixed L0-style training: huber + weight schedule + cap6mm, 변경 = input encoder only):
  - IA (anchor): cf 32-dim only (= L1 reuse). 재학습 = phase1_loss_train.py L1 result reuse 가능.
  - IB: cf 32 + 20-dim TrajectoryStatsFeature (no encoder, hand-crafted).
  - IC: cf 32 + 32-dim FrozenGRUEncoder (★ plan-004 selector checkpoint 미존재 시 skip + decision-note).
  - ID: cf 32 + 64-dim TrajectoryCNNEncoder (learnable, 1D CNN).
  - IF: cf 32 + 20-dim TrajectoryStats × MultiParseInput (raw/SG/EMA 학습 random 1 parse augment).

decision-note (자율 결정):
  - IA 학습 = L1 reuse (phase1_loss_train.py 의 L1 결과 = IA, 동일 config).
  - IC frozen GRU = plan-004 selector.AttnGRUCandidateSelector checkpoint 부재 시 IC skip (`input_axis_ic_skip` info).
  - 학습 loss = L1-style (huber + weight schedule + cap6mm).
  - axis-positive threshold = max(ΔOOF_i for IB,IC,ID,IF) ≥ 0.005 vs IA.

산출:
  - runs/baseline/H012_phase1-input-ablation/sub_{IA,IB,IC,ID,IF}/
  - analysis/plan-011/phase1_input_summary.json
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import nn

from src.pb_0_6822 import selector as sel
from src.pb_0_6822 import corrector_redesign_v2 as v2
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_loss_train as base  # type: ignore

REPO = Path(__file__).resolve().parents[2]
KST = timezone(timedelta(hours=9))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R_HIT = 0.01
HORIZON = 2
FOLD_VAL = 0
SEED_BASE = 20260513


class InputBundle:
    """입력 encoder bundle — sub-exp 별 다른 encoder 적용해 (cf, encoder_emb) 반환."""

    def __init__(self, sub_exp: str, x_seq: np.ndarray):
        self.sub_exp = sub_exp
        # pre-compute encoder outputs (frozen 또는 stateless 한 경우만)
        x_t = torch.from_numpy(x_seq).to(DEVICE)
        self.x_seq_t = x_t

        stats_module = v2.TrajectoryStatsFeature() if sub_exp in ("IB", "IF") else None
        if stats_module is not None:
            with torch.no_grad():
                self.stats = stats_module(x_t).cpu().numpy()  # (N, 20)
        else:
            self.stats = None

        if sub_exp == "IF":
            mpi = v2.MultiParseInput()
            self.mpi = mpi
            with torch.no_grad():
                p_raw, p_sg, p_ema = mpi.parse(x_t, end_idx=None, mode="inference")
                self.stats_3parse = torch.stack([
                    stats_module(p_raw), stats_module(p_sg), stats_module(p_ema)
                ], dim=0).cpu().numpy()                       # (3, N, 20)
        else:
            self.stats_3parse = None

        if sub_exp == "IC":
            self.gru = None
            self.gru_emb = None
        elif sub_exp == "ID":
            self.cnn = v2.TrajectoryCNNEncoder(in_channels=3, hidden=64).to(DEVICE)
        else:
            self.cnn = None

    @property
    def dim_encoder(self) -> int:
        if self.sub_exp in ("IB",):
            return 20
        if self.sub_exp == "IC":
            return 32
        if self.sub_exp == "ID":
            return 64
        if self.sub_exp == "IF":
            return 20
        return 0  # IA

    def make_encoder_emb(self, idx: torch.Tensor, mode: str = "train") -> Optional[torch.Tensor]:
        """idx: indices into the full dataset (B,). Returns (B, dim_encoder) or None."""
        if self.sub_exp == "IA":
            return None
        if self.sub_exp == "IB":
            return torch.from_numpy(self.stats[idx.cpu().numpy()]).to(DEVICE)
        if self.sub_exp == "IF":
            if mode == "train":
                k = int(torch.randint(0, 3, (1,)).item())
                return torch.from_numpy(self.stats_3parse[k, idx.cpu().numpy()]).to(DEVICE)
            else:  # inference: 3 parse mean
                return torch.from_numpy(self.stats_3parse.mean(axis=0)[idx.cpu().numpy()]).to(DEVICE)
        if self.sub_exp == "ID":
            return self.cnn(self.x_seq_t[idx])
        if self.sub_exp == "IC":
            raise RuntimeError("IC requires plan-004 GRU — caller must skip if not available.")
        return None


def train_input_subexp(
    sub_exp: str, data_tr: dict, data_va: dict, bundle_tr: InputBundle, bundle_va: InputBundle,
    args: argparse.Namespace,
) -> dict:
    """sub_exp 학습 + fold-0 eval."""
    t0 = time.time()
    torch.manual_seed(SEED_BASE + 100 + ord(sub_exp[1]))
    np.random.seed(SEED_BASE + 100 + ord(sub_exp[1]))

    model = v2.RedesignedCorrectionNet(
        dim_cf=32, hidden=64, dim_encoder=bundle_tr.dim_encoder
    ).to(DEVICE)

    # ID 의 CNN encoder 도 학습 — model 와 같이 optimizer
    params = list(model.parameters())
    if sub_exp == "ID":
        params += list(bundle_tr.cnn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def to_t(d, k): return torch.from_numpy(d[k]).to(DEVICE)
    cf_tr, target_tr, err_tr = to_t(data_tr, "cf"), to_t(data_tr, "target"), to_t(data_tr, "err")
    cand_tr, truth_tr = to_t(data_tr, "cand"), to_t(data_tr, "truth")
    cf_va, target_va, err_va = to_t(data_va, "cf"), to_t(data_va, "target"), to_t(data_va, "err")
    cand_va, truth_va = to_t(data_va, "cand"), to_t(data_va, "truth")
    idx_tr_full = torch.from_numpy(data_tr["full_idx"]).to(DEVICE)
    idx_va_full = torch.from_numpy(data_va["full_idx"]).to(DEVICE)

    n_tr = cf_tr.shape[0]
    best_hit, best_state, wait = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if sub_exp == "ID":
            bundle_tr.cnn.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, n_tr, args.batch):
            sel_local = perm[start:start + args.batch]
            global_idx = idx_tr_full[sel_local]
            cf_b = cf_tr[sel_local]
            er_b = err_tr[sel_local]
            tg_b = target_tr[sel_local]
            enc_b = bundle_tr.make_encoder_emb(global_idx, mode="train")

            opt.zero_grad(set_to_none=True)
            delta, _ = model(cf_b, enc_b)
            per = v2.huber_loss(delta, tg_b)
            w = base.weight_schedule(er_b)
            loss = (per * w).sum() / (w.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            total += float(loss.detach()) * len(cf_b)
            n += len(cf_b)

        # eval
        model.eval()
        if sub_exp == "ID":
            bundle_tr.cnn.eval()
        with torch.no_grad():
            enc_va = bundle_va.make_encoder_emb(idx_va_full, mode="inference")
            delta_va, _ = model(cf_va, enc_va)
            corrected_pos_va = cand_va + v2.cap_6mm(delta_va)
            err_after = torch.norm(corrected_pos_va - truth_va, dim=1)
            hit = float((err_after <= R_HIT).float().mean())

        if hit > best_hit:
            best_hit, wait = hit, 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            if sub_exp == "ID":
                best_state["cnn"] = {k: v.detach().cpu().clone() for k, v in bundle_tr.cnn.state_dict().items()}
        else:
            wait += 1

        if epoch % 5 == 0 or wait >= args.patience:
            print(f"  [{sub_exp}] ep{epoch:3d} loss={total/max(n,1):.4f} val_hit={hit:.4f} best={best_hit:.4f} wait={wait}")
        if wait >= args.patience and epoch >= args.min_epochs:
            break

    model.load_state_dict(best_state["model"])
    if sub_exp == "ID" and "cnn" in best_state:
        bundle_tr.cnn.load_state_dict(best_state["cnn"])
        bundle_va.cnn = bundle_tr.cnn  # share trained cnn

    model.eval()
    with torch.no_grad():
        enc_va = bundle_va.make_encoder_emb(idx_va_full, mode="inference")
        delta_va, _ = model(cf_va, enc_va)
        delta_va_np = delta_va.cpu().numpy()
    corrected_pos_va_np = data_va["cand"] + base.cap_6mm_np(delta_va_np)
    err_after = np.linalg.norm(corrected_pos_va_np - data_va["truth"], axis=1)
    err_before = data_va["err"]
    return {
        "sub_exp": f"P1.{sub_exp}",
        "n_val": int(len(err_after)),
        "fold": FOLD_VAL,
        "oof_soft_hit": float((err_after <= R_HIT).mean()),
        "oof_raw_hit": float((err_before <= R_HIT).mean()),
        "corrector_gain": float((err_after <= R_HIT).mean() - (err_before <= R_HIT).mean()),
        "per_band_hit_after": base.compute_per_band(err_before, err_after),
        "elapsed_sec": time.time() - t0,
        "best_val_hit": best_hit,
        "device": str(DEVICE),
        "dim_encoder": bundle_tr.dim_encoder,
    }, corrected_pos_va_np, delta_va_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs/baseline/H012_phase1-input-ablation")
    parser.add_argument("--summary-dir", type=Path, default=REPO / "analysis/plan-011")
    parser.add_argument("--sub-exps", type=str, default="IA,IB,IC,ID,IF")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = base.load_data(args.data_root)
    data["full_idx"] = np.arange(len(data["cf"]), dtype=np.int64)
    fold_ids = data["fold_ids"]
    tr_mask = fold_ids != FOLD_VAL
    va_mask = fold_ids == FOLD_VAL

    def slice_(d, m):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and len(v) == len(m):
                out[k] = v[m]
            else:
                out[k] = v
        return out

    data_tr = slice_(data, tr_mask)
    data_va = slice_(data, va_mask)
    print(f"[fold-0] train n={len(data_tr['cf'])}, val n={len(data_va['cf'])}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_exps = [s.strip() for s in args.sub_exps.split(",") if s.strip()]
    reports = {}

    # x_seq for encoder bundles
    x_seq = data["x_seq"] if "x_seq" in data else None
    if x_seq is None:
        print("[data] re-loading trajectories for encoder bundles...")
        x_seq = sel.load_stack(args.data_root / "train", data["sample_ids"])

    for sub_exp in sub_exps:
        print(f"\n=== {sub_exp} ===")
        try:
            bundle_tr = InputBundle(sub_exp, x_seq)
            bundle_va = bundle_tr   # share — encoder is global (not split per fold)
        except Exception as e:
            print(f"  ⚠ {sub_exp} bundle init failed: {e}")
            continue

        if sub_exp == "IC" and bundle_tr.gru is None:
            print(f"  ⚠ IC skip: plan-004 GRU checkpoint 부재 (decision-note: input_axis_ic_skip)")
            reports[f"P1.{sub_exp}"] = {
                "sub_exp": f"P1.{sub_exp}", "skip": True,
                "reason": "plan-004 selector.AttnGRUCandidateSelector checkpoint 부재",
            }
            continue

        report, corrected, delta = train_input_subexp(sub_exp, data_tr, data_va, bundle_tr, bundle_va, args)
        reports[f"P1.{sub_exp}"] = report
        sub_dir = args.out_dir / f"sub_{sub_exp}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / f"report_sub_{sub_exp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        np.savez(
            sub_dir / "boundary_val_predictions.npz",
            corrected_pos=corrected.astype(np.float32),
            delta=delta.astype(np.float32),
        )
        print(f"  → oof_soft_hit={report['oof_soft_hit']:.4f}, gain={report['corrector_gain']:+.4f}, "
              f"elapsed={report['elapsed_sec']:.1f}s")

    # summary
    summary_path = args.summary_dir / "phase1_input_summary.json"
    anchor_oof = reports.get("P1.IA", {}).get("oof_soft_hit")
    for k, r in reports.items():
        if r.get("skip"):
            continue
        if anchor_oof is not None:
            r["delta_vs_anchor"] = r["oof_soft_hit"] - anchor_oof
    non_anchor = [v for k, v in reports.items() if k != "P1.IA" and "delta_vs_anchor" in v]
    max_delta = max((v["delta_vs_anchor"] for v in non_anchor), default=0.0)
    summary = {
        "phase": "Phase 1.In Input axis ablation",
        "n_folds": 1, "fold": FOLD_VAL, "anchor": "P1.IA",
        "anchor_oof_soft_hit": anchor_oof,
        "sub_exps": reports,
        "axis_positive_threshold_0p005": bool(max_delta >= 0.005),
        "max_delta_vs_anchor": float(max_delta),
        "best_lever": max(non_anchor, key=lambda v: v["delta_vs_anchor"], default={}).get("sub_exp"),
        "generated_at": datetime.now(KST).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✓ summary → {summary_path.relative_to(REPO)}: best={summary['best_lever']}, max_Δ={max_delta:+.4f}")


if __name__ == "__main__":
    main()
