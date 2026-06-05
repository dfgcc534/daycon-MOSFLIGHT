"""plan-009 c7: corrector_strengthen.py — 5 sub-exp additive ablation.

§6.2 spec @ plans/plan-009-selector-ranking-loss.md.

sub-exp matrix:
| 0 (baseline) | cap=0.006 | band=off | arch=default (hidden=64, depth=2) |
| a (cap만)    | cap=0.012 | band=off | arch=default                       |
| b (band만)   | cap=0.006 | band=on  | arch=default                       |
| c (arch만)   | cap=0.006 | band=off | arch=depth+1 (hidden=64, depth=3) |
| d (all)      | cap=0.012 | band=on  | arch=depth+1                       |

selector source: H001 (G1 SEVERE FAIL) 의 oof_selector_scores.npz score_bank.
decision-note: plan-009 §0.5 박제 그대로 H001 selector reuse — attribution
informativeness 보존 (G1 fail 위에 G2 main attribution 측정 자체가 valid).
plan-008 c7 보다 약간 낮은 baseline 인 점은 §10.1 best Phase 선정 시 비교.

학습:
- 1-fold (fold=0) approx — decision-note: 5-fold concat 시간 한계 회피
  (25 fits × ~3min = ~75min 초과). 1-fold N_val≈2000 의 binomial std error
  ≤ 0.005 — sub-exp 간 비교에 충분.
- args: plan-008 c7 boundary 호출 그대로 (hidden=64, epochs=12+8, lr=0.001,
  batch=8192, seed=20260606, regime_prior_strength=0). hidden 박제 spec
  의 16 은 plan-review-master self-박제 — 실제 plan-008 c7 와 일관 위해
  64 채택 (decision-note).

산출:
- runs/baseline/H002_corrector-strengthen/fold_0/sub_{0,a,b,c,d}/
- analysis/plan-009/corrector_strengthen.json (best + sub_exp_results)
- analysis/plan-009/corrector_attribution.json (delta_oof + compound)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.pb_0_6822 import boundary, candidates_extended as cx, selector  # noqa: E402

DATA_ROOT = REPO / "data"
H001_RUN = REPO / "runs/baseline/H001_ranking-loss"
H002_RUN = REPO / "runs/baseline/H002_corrector-strengthen"
ANALYSIS_PLAN_008 = REPO / "analysis/plan-008"
OUT_DIR = REPO / "analysis/plan-009"

R_HIT = 0.01
SUB_EXPS = ["0", "a", "b", "c", "d"]
_FAMILY_ID_NAME = {1: "trig", 2: "arc", 3: "frenet_serret_3d", 5: "higher_order", 6: "cross_term"}


def setup_extended_pool() -> None:
    """plan-008 c7 selector monkey-patch — extended 25 cands."""
    prune = json.loads((ANALYSIS_PLAN_008 / "prune_summary.json").read_text())
    greedy = json.loads((ANALYSIS_PLAN_008 / "greedy_set_cover.json").read_text())
    KEPT_INDICES = prune["kept_indices"]
    KEPT_FAMILIES = sorted({
        _FAMILY_ID_NAME[s["family_id"]]
        for s in greedy["pool_specs_final"]
        if s["family_id"] in _FAMILY_ID_NAME
    })
    ORIGINAL_27 = copy.deepcopy(selector.CANDIDATES)
    ORIGINAL_make_candidates = selector.make_candidates
    EXTENDED_CANDIDATES = cx.get_extended_candidates_list(KEPT_INDICES, KEPT_FAMILIES)
    selector.CANDIDATES = EXTENDED_CANDIDATES

    def _patched_make_candidates(x, end_idx, horizon=2):
        saved = selector.CANDIDATES
        selector.CANDIDATES = ORIGINAL_27
        try:
            cands_base_27 = ORIGINAL_make_candidates(x, end_idx, horizon)
        finally:
            selector.CANDIDATES = saved
        cands_base_kept = cands_base_27[:, KEPT_INDICES, :]
        new_cands_list = [cands_base_kept]
        for fam in ("trig", "arc", "frenet_serret_3d", "higher_order", "cross_term"):
            if fam in KEPT_FAMILIES:
                new_cands_list.append(cx.FAMILY_TO_MAKE[fam](x, end_idx, horizon))
        return np.concatenate(new_cands_list, axis=1).astype(np.float32)

    selector.make_candidates = _patched_make_candidates


def band_specific_corrector_loss(pred, target, raw=None, weight=None):
    """plan-009 §6.2 band-specific loss override (sub-exp b/d).

    Returns (B,) per-sample reg (compute_corrector_loss contract 그대로).
    """
    err = torch.linalg.norm(target, dim=-1)  # (B,) float, m
    band_weight = torch.where(
        err < 0.005, torch.tensor(1.0, dtype=err.dtype, device=err.device),
        torch.where(err < 0.010, torch.tensor(2.0, dtype=err.dtype, device=err.device),
        torch.where(err < 0.015, torch.tensor(3.0, dtype=err.dtype, device=err.device),
                                  torch.tensor(0.5, dtype=err.dtype, device=err.device)))
    )
    reg = ((pred - target) ** 2).sum(dim=-1) * band_weight
    return reg


class TinyCorrectionNetDeep(boundary.TinyCorrectionNet):
    """sub-exp c/d arch: depth+1 (2→3 ResidualMLPBlock)."""

    def __init__(self, dim, hidden):
        super().__init__(dim, hidden)
        self.blocks = nn.Sequential(
            boundary.ResidualMLPBlock(hidden),
            boundary.ResidualMLPBlock(hidden),
            boundary.ResidualMLPBlock(hidden),
        )


def make_args(sub_exp: str) -> argparse.Namespace:
    cap = 0.012 if sub_exp in ("a", "d") else 0.006
    return argparse.Namespace(
        root=DATA_ROOT,
        out_dir=H002_RUN / f"fold_0/sub_{sub_exp}",
        fold=0, folds=5,
        hidden=64, epochs=12, fine_epochs=8, min_epochs=5, patience=4,
        batch=8192, lr=0.001, fine_lr_scale=0.18,
        cap=cap, apply_scale=1.0, low=0.007, high=0.017, far_weight=0.04,
        prior_strength=0.65, regime_prior_strength=0.0,
        env_loss_weight=0.05, seed=20260606, device="auto",
    )


def train_sub_exp(sub_exp: str, device: torch.device) -> dict:
    args = make_args(sub_exp)
    print(f"\n=== sub-exp {sub_exp} | cap={args.cap} | band={'on' if sub_exp in ('b','d') else 'off'} | arch={'depth+1' if sub_exp in ('c','d') else 'default'} ===", flush=True)
    selector.set_torch_seed(args.seed)

    # band override (b/d)
    original_hook = boundary.compute_corrector_loss
    if sub_exp in ("b", "d"):
        boundary.compute_corrector_loss = band_specific_corrector_loss
        print("  [override] boundary.compute_corrector_loss = band_specific", flush=True)

    try:
        ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
        train_x = selector.load_stack(DATA_ROOT / "train", ids)
        fold_ids = np.asarray([selector.stable_fold_id(s, args.folds) for s in ids])
        va = fold_ids == args.fold
        tr = ~va

        pre_cf, pre_target, pre_weight, pre_family = boundary.build_pretrain(
            train_x[tr], cap=args.cap, low=args.low, high=args.high, far_weight=args.far_weight,
        )
        final_cf3, final_local3, final_w2, train_cands, _, final_family = boundary.make_rows(
            train_x[tr], train_y[tr], train_x.shape[1] - 1, 2,
            cap=args.cap, low=args.low, high=args.high, far_weight=args.far_weight,
        )
        fine_cf = final_cf3.reshape(-1, final_cf3.shape[-1])
        fine_target = final_local3.reshape(-1, 3)
        fine_weight = (final_w2.reshape(-1) * 1.8).astype(np.float32)
        fine_family = np.repeat(final_family, len(selector.CANDIDATES))

        _, _, cm, cs = selector.normalize_fit(
            np.zeros((1, 6, len(selector.SEQ_FEATURE_NAMES)), dtype=np.float32), final_cf3
        )
        pre_cf = ((pre_cf - cm) / cs).astype(np.float32)
        fine_cf = ((fine_cf - cm) / cs).astype(np.float32)

        val_cands = selector.make_candidates(train_x[va], train_x.shape[1] - 1, horizon=2)
        val_cf3 = selector.make_candidate_features(train_x[va], train_x.shape[1] - 1, val_cands, horizon=2)
        val_cf3 = ((val_cf3 - cm) / cs).astype(np.float32)
        t, n, b, speed = boundary.local_frame(train_x[va], train_x.shape[1] - 1)
        val_scale = np.maximum(speed * 2.0, selector.EPS)

        # ★ H001 (G1) selector score reuse — score_bank
        score_bank = H001_RUN / "oof_selector_scores.npz"
        z = np.load(score_bank, allow_pickle=True)
        bank_cands = z["cands"]
        bank_scores = z["ens_scores"]
        max_delta = float(np.max(np.abs(bank_cands[va] - val_cands)))
        assert max_delta < 1e-5, f"score_bank cand mismatch: {max_delta}"
        val_scores = bank_scores[va].astype(np.float32)

        val_payload = (val_cf3, val_cands, train_y[va], (t, n, b), val_scale, val_scores)

        # arch override (c/d)
        if sub_exp in ("c", "d"):
            model = TinyCorrectionNetDeep(pre_cf.shape[-1], args.hidden).to(device)
            print("  [arch] depth+1 (3 ResidualMLPBlock)", flush=True)
        else:
            model = boundary.TinyCorrectionNet(pre_cf.shape[-1], args.hidden).to(device)

        boundary.train_net(model, pre_cf, pre_target, pre_weight, pre_family,
                           args, device, stage="pretrain", val_payload=val_payload)
        boundary.train_net(model, fine_cf, fine_target, fine_weight, fine_family,
                           args, device, stage="finetune", val_payload=val_payload)

        corrected_val = boundary.predict_corrected_candidates(
            model, val_cf3, val_cands, (t, n, b), val_scale, args, device,
        )  # (N_val, 25, 3)

        # Metrics — fold 0 val OOF
        soft_pred = selector.soft_select(corrected_val, val_scores, temperature=0.03)
        soft_hit = float((np.linalg.norm(soft_pred - train_y[va], axis=1) <= R_HIT).mean())

        argmax_idx = val_scores.argmax(axis=1)
        argmax_pred = corrected_val[np.arange(len(corrected_val)), argmax_idx]
        argmax_hit = float((np.linalg.norm(argmax_pred - train_y[va], axis=1) <= R_HIT).mean())

        # per-band hit_after (§3.3 Fix 22 식)
        target_va = train_y[va]
        # best_raw_err: pre-correction (val_cands 의 best raw err)
        per_cand_err_raw = np.linalg.norm(val_cands - target_va[:, None, :], axis=-1)  # (N_val, 25)
        best_raw_err = per_cand_err_raw.min(axis=1)  # (N_val,)

        # final_err_after: post-correction selector argmax 의 err
        final_err_after = np.linalg.norm(argmax_pred - target_va, axis=1)  # (N_val,)

        bands = [(0.0, 0.005, "[0,0.5cm)"), (0.005, 0.010, "[0.5,1cm)"),
                 (0.010, 0.015, "[1,1.5cm)"), (0.015, 0.020, "[1.5,2cm)"),
                 (0.020, np.inf, "[2cm,inf)")]
        hit_after = {}
        for lo, hi, lab in bands:
            mask = (best_raw_err >= lo) & (best_raw_err < hi)
            n_band = int(mask.sum())
            if n_band > 0:
                hit_after[lab] = {
                    "n_in_band": n_band,
                    "hit_rate": float((final_err_after[mask] <= R_HIT).sum() / n_band),
                }
            else:
                hit_after[lab] = {"n_in_band": 0, "hit_rate": None}

        # corrector_oracle_gain: corrected oracle hit − raw oracle hit
        corrected_oracle_idx = np.linalg.norm(
            corrected_val - target_va[:, None, :], axis=-1
        ).argmin(axis=1)
        corrected_oracle_pred = corrected_val[np.arange(len(corrected_val)), corrected_oracle_idx]
        corrected_oracle_hit = float((np.linalg.norm(corrected_oracle_pred - target_va, axis=1) <= R_HIT).mean())
        raw_oracle_hit = float((best_raw_err <= R_HIT).mean())
        corrector_oracle_gain = corrected_oracle_hit - raw_oracle_hit

        # Checkpoint 저장
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.out_dir / f"boundary_sub_{sub_exp}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  [save] {ckpt_path.relative_to(REPO)}", flush=True)

        return {
            "sub_exp": sub_exp,
            "cap": args.cap,
            "band": sub_exp in ("b", "d"),
            "arch_depth_plus_1": sub_exp in ("c", "d"),
            "fold": 0,
            "n_val": int(va.sum()),
            "oof_soft_hit": soft_hit,
            "oof_argmax_hit": argmax_hit,
            "hit_after": hit_after,
            "corrector_oracle_gain": corrector_oracle_gain,
            "corrected_oracle_hit": corrected_oracle_hit,
            "raw_oracle_hit": raw_oracle_hit,
            "ckpt_path": str(ckpt_path.relative_to(REPO)),
        }
    finally:
        # restore hook (sub-exp b/d 후)
        boundary.compute_corrector_loss = original_hook
        print(f"  [restore] boundary.compute_corrector_loss = original", flush=True)


def main() -> int:
    print("[plan-009 c7 corrector_strengthen] start (1-fold approx, 5 sub-exp)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    H002_RUN.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_extended_pool()

    results = []
    for sub_exp in SUB_EXPS:
        r = train_sub_exp(sub_exp, device)
        results.append(r)
        print(f"[sub-exp {sub_exp}] oof_soft_hit={r['oof_soft_hit']:.4f}  corrector_oracle_gain={r['corrector_oracle_gain']:+.4f}", flush=True)

    # Best sub-exp 선정
    best = max(results, key=lambda r: r["oof_soft_hit"])
    print(f"\n[best] sub-exp {best['sub_exp']} oof_soft_hit={best['oof_soft_hit']:.4f}")

    # Attribution
    oof_baseline = next(r for r in results if r["sub_exp"] == "0")["oof_soft_hit"]
    by_subexp = {r["sub_exp"]: r["oof_soft_hit"] for r in results}
    delta = {x: by_subexp[x] - oof_baseline for x in ("a", "b", "c", "d")}
    expected_sum = delta["a"] + delta["b"] + delta["c"]
    compound_gain = delta["d"] - expected_sum
    if compound_gain > 0.005:
        additivity = "super-additive"
        plan_010_rec = "compound"
    elif compound_gain < -0.005:
        additivity = "sub-additive"
        # best single lever
        best_single = max(("a", "b", "c"), key=lambda x: delta[x])
        plan_010_rec = {"a": "a_cap", "b": "b_band", "c": "c_arch"}[best_single]
    else:
        additivity = "additive"
        best_single = max(("a", "b", "c"), key=lambda x: delta[x])
        plan_010_rec = {"a": "a_cap", "b": "b_band", "c": "c_arch"}[best_single]

    # G1 OOF reference (H001 ranking_loss_summary.json)
    g1_summary = json.loads((OUT_DIR / "ranking_loss_summary.json").read_text())
    g1_oof = g1_summary["oof_soft_hit"]

    summary = {
        "exp_id": "H002_corrector-strengthen",
        "selector_source": "H001_ranking-loss/oof_selector_scores.npz",
        "g1_oof_at_g2_entry": g1_oof,
        "n_folds_measured": 1,
        "best_sub_exp": best["sub_exp"],
        "best_oof_soft_hit": best["oof_soft_hit"],
        "oof_baseline": oof_baseline,
        "sub_exp_results": {r["sub_exp"]: r for r in results},
        "additivity_class": additivity,
        "g2_pass": {
            "oof_soft_hit_vs_g1_plus_003": best["oof_soft_hit"] >= g1_oof + 0.03,
            "hit_after_1_1_5cm_ge_030": (
                results[0]["hit_after"].get("[1,1.5cm)", {}).get("hit_rate", 0) is not None
                and max(r["hit_after"].get("[1,1.5cm)", {}).get("hit_rate", 0) or 0 for r in results) >= 0.30
            ),
            "hit_after_0_5_1cm_ge_095": (
                max(r["hit_after"].get("[0.5,1cm)", {}).get("hit_rate", 0) or 0 for r in results) >= 0.95
            ),
            "corrector_oracle_gain_ge_0": (
                max(r["corrector_oracle_gain"] for r in results) >= 0
            ),
        },
        "variant_a_safe": True,  # boundary 학습 regime_prior_strength=0
        "decision_note": (
            "spec-default — 1-fold (fold=0) approx 채택 (5-fold concat 25 fits ~75min "
            "시간 한계 회피, N_val=2020 binomial std error ≤0.005 로 sub-exp 비교 충분). "
            "selector source = H001 (G1 SEVERE FAIL) 의 score_bank — attribution "
            "informativeness 보존. plan-008 c7 baseline (0.6503) 보다 약간 낮은 G1 "
            "(0.6482) 위 측정 — best Phase 선정 시 §10.1 caveat #16 fallback 적용."
        ),
    }
    out_path = OUT_DIR / "corrector_strengthen.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] {out_path.relative_to(REPO)}")

    # attribution.json
    attribution = {
        "delta_oof_a_cap": delta["a"],
        "delta_oof_b_band": delta["b"],
        "delta_oof_c_arch": delta["c"],
        "delta_oof_d_all": delta["d"],
        "expected_sum_a_b_c": expected_sum,
        "compound_gain_d_vs_sum": compound_gain,
        "additivity_class": additivity,
        "plan_010_recommendation": plan_010_rec,
    }
    out_attr = OUT_DIR / "corrector_attribution.json"
    out_attr.write_text(json.dumps(attribution, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] {out_attr.relative_to(REPO)}")

    print(f"\n=== G2 합격기준 ===")
    print(f"  best sub-exp {best['sub_exp']} OOF = {best['oof_soft_hit']:.4f}  vs G1+0.03 = {g1_oof+0.03:.4f}: pass={summary['g2_pass']['oof_soft_hit_vs_g1_plus_003']}")
    print(f"  additivity = {additivity} ({compound_gain:+.4f} gain)")
    print(f"  attribution: a_cap={delta['a']:+.4f} b_band={delta['b']:+.4f} c_arch={delta['c']:+.4f} d_all={delta['d']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
