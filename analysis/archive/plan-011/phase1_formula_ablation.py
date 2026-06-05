"""plan-011 c10 — Phase 1.F Formula axis ablation (4 sub-exp + F0 reuse).

F sub-exp 는 *공식 자체* 를 변경 → cand_pos 가 sub-exp 별로 다름 (L/In/M 과 구조 차이).
candidate 만 다르고 corrector 는 L0 default 적용 (anchor combo).

  - F0 (anchor): frenet_par120_perp_neg020 — reuse from P1.L0 (재학습 X, plan-005 reuse).
  - F1: CMA-ES tuned 6 vars (plan-007 Step 2 best_params reuse).
        plan-007 best 결과 부재 시 → F0 anchor 의 CANDIDATES[17] 그대로 (= F0 reuse) + decision-note.
  - F2: basis ablation best (plan-007 Step 3 8 vars reuse).
        부재 시 → F0 reuse + decision-note.
  - F3: per-sample MLP coefficient regression (PerSampleMLPFormula learnable).
  - F4: LearnableSingleCandidate (data-driven 6 coef learnable).

decision-note (자율 결정):
  - plan-007 산출 (analysis/plan-007) 의 best_params / basis_best 가 코드 가능한 형태로 박제 안 됨
    → F1/F2 는 F0 reuse (= F0 anchor 가 baseline 가정).
  - F3/F4 는 v2 의 PerSampleMLPFormula / LearnableSingleCandidate 사용 + 자체 학습.
  - F3/F4 학습은 corrector 적용 NO — *순수 candidate generation* (no delta correction) 의 OOF 측정.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_loss_train as base  # type: ignore

from src.pb_0_6822 import selector as sel
from src.pb_0_6822 import corrector_redesign_v2 as v2

REPO = Path(__file__).resolve().parents[2]
KST = timezone(timedelta(hours=9))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R_HIT = 0.01
HORIZON = 2
FOLD_VAL = 0
SEED_BASE = 20260513


def prepare_motion_terms(x_seq: np.ndarray):
    """selector.make_candidates parity 를 보장하는 motion terms 추출.

    Returns dict of (N, 3) numpy arrays — selector.make_candidates 와 *동일한* 변수 정의.
    """
    END = 10
    EPS = 1e-6
    p0 = x_seq[:, END]
    v_last = x_seq[:, END] - x_seq[:, END - 1]                       # = motion_terms d1
    v_prev = x_seq[:, END - 1] - x_seq[:, END - 2]                   # = make_candidates d2
    acc = v_last - v_prev                                             # = motion_terms acc
    v_pp = x_seq[:, END - 2] - x_seq[:, END - 3]
    prev_acc = v_prev - v_pp
    jerk_vec = acc - prev_acc                                         # = make_candidates jerk

    # Frenet projections (selector.make_candidates 와 동일):
    speed = np.linalg.norm(v_last, axis=1, keepdims=True) + EPS
    tangent = v_last / speed                                          # t̂ = v_last / ‖v_last‖
    acc_par_scalar = (acc * tangent).sum(axis=1, keepdims=True)
    acc_par_vec = acc_par_scalar * tangent                            # (acc · t̂) t̂
    acc_perp_vec = acc - acc_par_vec

    return {
        "p0": p0.astype(np.float32),
        "v_last": v_last.astype(np.float32),
        "v_prev": v_prev.astype(np.float32),
        "acc_par_vec": acc_par_vec.astype(np.float32),
        "acc_perp_vec": acc_perp_vec.astype(np.float32),
        "jerk_vec": jerk_vec.astype(np.float32),
    }


def f0_reuse_oof(data_va: dict) -> dict:
    """F0 anchor — plan-005 reuse, 재학습 X. data_va['cand'] 이 이미 F0 candidate."""
    err = np.linalg.norm(data_va["cand"] - data_va["truth"], axis=1)
    return {
        "sub_exp": "P1.F0",
        "n_val": int(len(err)),
        "fold": FOLD_VAL,
        "oof_soft_hit": float((err <= R_HIT).mean()),
        "oof_raw_hit": float((err <= R_HIT).mean()),
        "corrector_gain": 0.0,
        "per_band_hit_after": base.compute_per_band(err, err),
        "elapsed_sec": 0.0,
        "training": "X (F0 reuse from CANDIDATES[17])",
    }


def train_F3(data_tr: dict, data_va: dict, motion_tr: dict, motion_va: dict, args) -> dict:
    """F3: PerSampleMLPFormula — learnable per-sample (par, perp). 학습 cand_pos 직접 출력."""
    t0 = time.time()
    torch.manual_seed(SEED_BASE + 300 + 3)
    np.random.seed(SEED_BASE + 300 + 3)

    f3 = v2.PerSampleMLPFormula(in_dim=12).to(DEVICE)
    f4_helper = v2.LearnableSingleCandidate().to(DEVICE)
    f4_helper.coef.requires_grad_(False)

    # 12-dim ctx features (간이 — make_ctx_features 없이 motion-based inline)
    def make_ctx(motion):
        v = motion["v_last"]
        acc_par = motion["acc_par_vec"]
        acc_perp = motion["acc_perp_vec"]
        a = acc_par + acc_perp                                            # full acc vector
        j = motion["jerk_vec"]
        s = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        a_par_n = np.linalg.norm(acc_par, axis=1, keepdims=True)
        a_perp_n = np.linalg.norm(acc_perp, axis=1, keepdims=True)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        jn = np.linalg.norm(j, axis=1, keepdims=True)
        turn_cos = np.full_like(s, 0.0)  # placeholder
        ctx = np.concatenate([
            s, s, a_norm / s, a_par_n / s, a_perp_n / s, turn_cos,       # 6
            jn, jn, jn,                                                   # 3
            jn, jn,                                                       # 2
            np.zeros_like(s),                                             # 1
        ], axis=1)
        return ctx.astype(np.float32)

    ctx_tr = make_ctx(motion_tr); ctx_va = make_ctx(motion_va)

    def to_t(arr): return torch.from_numpy(arr).to(DEVICE)
    ctx_tr_t = to_t(ctx_tr); ctx_va_t = to_t(ctx_va)
    truth_tr = to_t(data_tr["truth"]); truth_va = to_t(data_va["truth"])
    p0_tr = to_t(motion_tr["p0"]); p0_va = to_t(motion_va["p0"])
    vl_tr = to_t(motion_tr["v_last"]); vl_va = to_t(motion_va["v_last"])
    vp_tr = to_t(motion_tr["v_prev"]); vp_va = to_t(motion_va["v_prev"])
    apa_tr = to_t(motion_tr["acc_par_vec"]); apa_va = to_t(motion_va["acc_par_vec"])
    ape_tr = to_t(motion_tr["acc_perp_vec"]); ape_va = to_t(motion_va["acc_perp_vec"])
    jv_tr = to_t(motion_tr["jerk_vec"]); jv_va = to_t(motion_va["jerk_vec"])

    def build_coef(par_perp):
        """F3 의 (par, perp) per-sample → 6-coef (1.98, 0.0, par, perp, 0.0, 1.0)."""
        return torch.stack([
            torch.full_like(par_perp[:, 0], 1.98),  # ★ CANDIDATES[17].d1 = 1.98 (canonical)
            torch.zeros_like(par_perp[:, 0]),
            par_perp[:, 0],
            par_perp[:, 1],
            torch.zeros_like(par_perp[:, 0]),
            torch.ones_like(par_perp[:, 0]),
        ], dim=-1)  # (B, 6)

    def cand_from_f4(p0, vl, vp, apa, ape, jv, coef):
        return f4_helper(p0=p0, v_last=vl, v_prev=vp, acc_par_vec=apa,
                          acc_perp_vec=ape, jerk_vec=jv, horizon=2.0, coef=coef)

    opt = torch.optim.AdamW(f3.parameters(), lr=args.lr, weight_decay=1e-4)
    N_train = ctx_tr_t.shape[0]
    best_hit, best_state, wait = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        f3.train()
        perm = torch.randperm(N_train, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, N_train, args.batch):
            sel_ = perm[start:start + args.batch]
            ctx_b = ctx_tr_t[sel_]
            par_perp = f3(ctx_b)
            coef = build_coef(par_perp)
            cand_pos = cand_from_f4(
                p0_tr[sel_], vl_tr[sel_], vp_tr[sel_],
                apa_tr[sel_], ape_tr[sel_], jv_tr[sel_], coef,
            )
            err = (cand_pos - truth_tr[sel_]).pow(2).sum(dim=1)
            loss = err.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(f3.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(ctx_b)
            n += len(ctx_b)

        f3.eval()
        with torch.no_grad():
            par_perp = f3(ctx_va_t)
            coef = build_coef(par_perp)
            cand_pos_va = cand_from_f4(p0_va, vl_va, vp_va, apa_va, ape_va, jv_va, coef)
            err_va = torch.norm(cand_pos_va - truth_va, dim=1)
            hit = float((err_va <= R_HIT).float().mean())

        if hit > best_hit:
            best_hit, wait = hit, 0
            best_state = {k: v.detach().cpu().clone() for k, v in f3.state_dict().items()}
        else:
            wait += 1
        if epoch % 5 == 0 or wait >= args.patience:
            print(f"  [F3] ep{epoch:3d} loss={total/max(n,1):.6f} val_hit={hit:.4f} best={best_hit:.4f} wait={wait}")
        if wait >= args.patience and epoch >= args.min_epochs:
            break

    f3.load_state_dict(best_state)
    f3.eval()
    with torch.no_grad():
        par_perp = f3(ctx_va_t)
        coef = build_coef(par_perp)
        cand_pos_va = cand_from_f4(p0_va, vl_va, vp_va, apa_va, ape_va, jv_va, coef).cpu().numpy()
    err = np.linalg.norm(cand_pos_va - data_va["truth"], axis=1)
    return {
        "sub_exp": "P1.F3", "n_val": int(len(err)), "fold": FOLD_VAL,
        "oof_soft_hit": float((err <= R_HIT).mean()),
        "oof_raw_hit": float((err <= R_HIT).mean()),
        "corrector_gain": 0.0,
        "per_band_hit_after": base.compute_per_band(err, err),
        "elapsed_sec": time.time() - t0,
        "best_val_hit": best_hit,
    }


def train_F4(data_tr: dict, data_va: dict, motion_tr: dict, motion_va: dict, args) -> dict:
    """F4: LearnableSingleCandidate — 6 coef learnable. 학습 cand_pos 직접 출력."""
    t0 = time.time()
    torch.manual_seed(SEED_BASE + 300 + 4)
    np.random.seed(SEED_BASE + 300 + 4)

    f4 = v2.LearnableSingleCandidate().to(DEVICE)

    def to_t(arr): return torch.from_numpy(arr).to(DEVICE)
    truth_tr = to_t(data_tr["truth"]); truth_va = to_t(data_va["truth"])
    p0_tr = to_t(motion_tr["p0"]); p0_va = to_t(motion_va["p0"])
    vl_tr = to_t(motion_tr["v_last"]); vl_va = to_t(motion_va["v_last"])
    vp_tr = to_t(motion_tr["v_prev"]); vp_va = to_t(motion_va["v_prev"])
    apa_tr = to_t(motion_tr["acc_par_vec"]); apa_va = to_t(motion_va["acc_par_vec"])
    ape_tr = to_t(motion_tr["acc_perp_vec"]); ape_va = to_t(motion_va["acc_perp_vec"])
    jv_tr = to_t(motion_tr["jerk_vec"]); jv_va = to_t(motion_va["jerk_vec"])

    # F0 init verification — coef = (1.98, 0.0, 1.20, -0.20, 0.0, 1.0) 가 CANDIDATES[17] 와 정확히 일치
    f4.eval()
    with torch.no_grad():
        cand_init = f4(p0_va, vl_va, vp_va, apa_va, ape_va, jv_va, 2.0)
        init_err = torch.norm(cand_init - to_t(data_va["cand"]), dim=1).max().item()
    print(f"  [F4] init parity check: max(cand_init − F0 anchor) = {init_err:.2e} m (expected ~0)")

    opt = torch.optim.AdamW(f4.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    init_coef = f4.coef.detach().clone()
    N_train = p0_tr.shape[0]
    best_hit, best_state, wait = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        f4.train()
        perm = torch.randperm(N_train, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, N_train, args.batch):
            sel_ = perm[start:start + args.batch]
            cand_pos = f4(p0_tr[sel_], vl_tr[sel_], vp_tr[sel_],
                          apa_tr[sel_], ape_tr[sel_], jv_tr[sel_], 2.0)
            err = (cand_pos - truth_tr[sel_]).pow(2).sum(dim=1)
            loss = err.mean() + 0.01 * (f4.coef - init_coef).pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(p0_tr[sel_])
            n += len(p0_tr[sel_])

        f4.eval()
        with torch.no_grad():
            cand_pos_va = f4(p0_va, vl_va, vp_va, apa_va, ape_va, jv_va, 2.0)
            err_va = torch.norm(cand_pos_va - truth_va, dim=1)
            hit = float((err_va <= R_HIT).float().mean())

        if hit > best_hit:
            best_hit, wait = hit, 0
            best_state = {k: v.detach().cpu().clone() for k, v in f4.state_dict().items()}
        else:
            wait += 1
        if epoch % 5 == 0 or wait >= args.patience:
            print(f"  [F4] ep{epoch:3d} coef={f4.coef.detach().cpu().numpy().round(3)} val_hit={hit:.4f} best={best_hit:.4f} wait={wait}")
        if wait >= args.patience and epoch >= args.min_epochs:
            break

    f4.load_state_dict(best_state)
    f4.eval()
    with torch.no_grad():
        cand_pos_va = f4(p0_va, vl_va, vp_va, apa_va, ape_va, jv_va, 2.0).cpu().numpy()
    err = np.linalg.norm(cand_pos_va - data_va["truth"], axis=1)
    return {
        "sub_exp": "P1.F4", "n_val": int(len(err)), "fold": FOLD_VAL,
        "oof_soft_hit": float((err <= R_HIT).mean()),
        "oof_raw_hit": float((err <= R_HIT).mean()),
        "corrector_gain": 0.0,
        "per_band_hit_after": base.compute_per_band(err, err),
        "elapsed_sec": time.time() - t0,
        "best_val_hit": best_hit,
        "final_coef": [float(c) for c in f4.coef.detach().cpu().numpy().tolist()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs/baseline/H014_phase1-formula-ablation")
    parser.add_argument("--summary-dir", type=Path, default=REPO / "analysis/plan-011")
    parser.add_argument("--sub-exps", type=str, default="F0,F1,F2,F3,F4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = base.load_data(args.data_root)
    data_tr, data_va = base.split_fold(data)
    print(f"[fold-0] train n={len(data_tr['cf'])}, val n={len(data_va['cf'])}")

    print("[data] re-loading trajectories for motion terms...")
    x_seq_full = sel.load_stack(args.data_root / "train", data["sample_ids"])
    motion = prepare_motion_terms(x_seq_full)
    fold_ids = data["fold_ids"]
    motion_tr = {k: v[fold_ids != FOLD_VAL] for k, v in motion.items()}
    motion_va = {k: v[fold_ids == FOLD_VAL] for k, v in motion.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_exps = [s.strip() for s in args.sub_exps.split(",") if s.strip()]
    reports = {}

    for sub_exp in sub_exps:
        print(f"\n=== {sub_exp} ===")
        if sub_exp == "F0":
            report = f0_reuse_oof(data_va)
        elif sub_exp in ("F1", "F2"):
            # plan-007 best params 박제 부재 → F0 reuse + decision-note
            print(f"  ⚠ {sub_exp} skip: plan-007 best params 박제 부재 → F0 reuse")
            r0 = f0_reuse_oof(data_va)
            report = {**r0, "sub_exp": f"P1.{sub_exp}",
                      "skip_reason": "plan-007 Step 2/3 best params not pinned in repo — F0 reuse",
                      "fallback": "F0 reuse"}
        elif sub_exp == "F3":
            report = train_F3(data_tr, data_va, motion_tr, motion_va, args)
        elif sub_exp == "F4":
            report = train_F4(data_tr, data_va, motion_tr, motion_va, args)
        else:
            continue
        reports[f"P1.{sub_exp}"] = report
        sub_dir = args.out_dir / f"sub_{sub_exp}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / f"report_sub_{sub_exp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  → oof_soft_hit={report['oof_soft_hit']:.4f}, elapsed={report['elapsed_sec']:.1f}s")

    # summary
    summary_path = args.summary_dir / "phase1_formula_summary.json"
    anchor_oof = reports.get("P1.F0", {}).get("oof_soft_hit")
    for k, r in reports.items():
        if anchor_oof is not None:
            r["delta_vs_anchor"] = r["oof_soft_hit"] - anchor_oof
    non_anchor = [v for k, v in reports.items() if k != "P1.F0"]
    max_delta = max((v.get("delta_vs_anchor", 0) for v in non_anchor), default=0.0)
    summary = {
        "phase": "Phase 1.F Formula axis ablation",
        "n_folds": 1, "fold": FOLD_VAL, "anchor": "P1.F0",
        "anchor_oof_soft_hit": anchor_oof,
        "sub_exps": reports,
        "axis_positive_threshold_0p005": bool(max_delta >= 0.005),
        "max_delta_vs_anchor": float(max_delta),
        "best_lever": max(non_anchor, key=lambda v: v.get("delta_vs_anchor", -1), default={}).get("sub_exp"),
        "generated_at": datetime.now(KST).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✓ summary → {summary_path.relative_to(REPO)}: best={summary['best_lever']}, max_Δ={max_delta:+.4f}")


if __name__ == "__main__":
    main()
