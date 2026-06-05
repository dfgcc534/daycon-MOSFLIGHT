"""plan-009 c2.1 (part 2): cap_saturation_extended.py — corrector cap binding rate 재측정.

Spec @ §4.2 + Fix 9 (self-contained hyperparam) @ plans/plan-009-selector-ranking-loss.md.

baseline corrector 학습 (cap=0.006, band=off, arch=default = plan-008 c7) 후
shift = ||predict_corrected_candidates - val_cands|| per (sample, cand) 측정,
overall_rate = (shift >= 0.0057).mean() — L2-norm cap 정의 (boundary.cap_vectors).

plan-005 의 0.0358 (raw 27 cands) 와 비교 → corrector main framing evidence.

학습:
- selector monkey-patch (plan-008 c7 prep 그대로 reuse) → extended 25 cands.
- boundary.train_net inline reproduce (plan-008 c7 args: epochs=12+8, hidden=64,
  batch=8192, lr=0.001, fine_lr_scale=0.18, cap=0.006, apply_scale=1.0,
  seed=20260606).

decision-note: spec-default — fold=0 (1-fold) approx 채택. 5-fold concat 은
시간 한계 회피 + 1-fold N_val≈2000 의 binomial std error ≤ 0.005 로
caveat #13 분기 (재현 ≤0.05 / 강화 ≥0.08) 결정에 충분. 5-fold concat 은
필요 시 plan-009 후속 plan 으로 carry-over.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pb_0_6822 import boundary, candidates_extended as cx  # noqa: E402
from src.pb_0_6822 import selector as base  # noqa: E402

DATA_ROOT = REPO_ROOT / "data"
RUN_DIR = REPO_ROOT / "runs/baseline/G001_candidate-redefine"
ANALYSIS_PLAN_008 = REPO_ROOT / "analysis/plan-008"
SCORE_BANK = RUN_DIR / "oof_selector_scores.npz"
OUT_DIR = REPO_ROOT / "analysis/plan-009"

CAP = 0.006
SATURATION_THRESHOLD = 0.0057  # cap * 0.95
FOLD = 0  # 1-fold approx (decision-note)

_FAMILY_ID_NAME = {1: "trig", 2: "arc", 3: "frenet_serret_3d", 5: "higher_order", 6: "cross_term"}


def setup_extended_pool() -> tuple[list, list[int], list[str]]:
    """plan-008 selector_retrain.py 의 prep 단계 reuse — selector monkey-patch."""
    prune = json.loads((ANALYSIS_PLAN_008 / "prune_summary.json").read_text())
    greedy = json.loads((ANALYSIS_PLAN_008 / "greedy_set_cover.json").read_text())
    KEPT_INDICES = prune["kept_indices"]
    KEPT_FAMILIES = sorted({
        _FAMILY_ID_NAME[s["family_id"]]
        for s in greedy["pool_specs_final"]
        if s["family_id"] in _FAMILY_ID_NAME
    })
    print(f"[prep] KEPT_INDICES ({len(KEPT_INDICES)}): {KEPT_INDICES}")
    print(f"[prep] KEPT_FAMILIES ({len(KEPT_FAMILIES)}): {KEPT_FAMILIES}")

    ORIGINAL_27 = copy.deepcopy(base.CANDIDATES)
    ORIGINAL_make_candidates = base.make_candidates
    EXTENDED_CANDIDATES = cx.get_extended_candidates_list(KEPT_INDICES, KEPT_FAMILIES)
    print(f"[prep] EXTENDED_CANDIDATES n={len(EXTENDED_CANDIDATES)}")

    base.CANDIDATES = EXTENDED_CANDIDATES

    def _patched_make_candidates(x, end_idx, horizon=2):
        saved = base.CANDIDATES
        base.CANDIDATES = ORIGINAL_27
        try:
            cands_base_27 = ORIGINAL_make_candidates(x, end_idx, horizon)
        finally:
            base.CANDIDATES = saved
        cands_base_kept = cands_base_27[:, KEPT_INDICES, :]
        new_cands_list = [cands_base_kept]
        for fam in ("trig", "arc", "frenet_serret_3d", "higher_order", "cross_term"):
            if fam in KEPT_FAMILIES:
                new_cands_list.append(cx.FAMILY_TO_MAKE[fam](x, end_idx, horizon))
        return np.concatenate(new_cands_list, axis=1).astype(np.float32)

    base.make_candidates = _patched_make_candidates
    return EXTENDED_CANDIDATES, KEPT_INDICES, KEPT_FAMILIES


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        root=DATA_ROOT, out_dir=OUT_DIR / "tmp_boundary", fold=FOLD, folds=5,
        hidden=64, epochs=12, fine_epochs=8, min_epochs=5, patience=4,
        batch=8192, lr=0.001, fine_lr_scale=0.18,
        cap=CAP, apply_scale=1.0, low=0.007, high=0.017, far_weight=0.04,
        prior_strength=0.65, regime_prior_strength=0.0,  # Variant A
        env_loss_weight=0.05, seed=20260606, device="auto",
    )


def train_one_fold(args: argparse.Namespace, device: torch.device) -> dict:
    base.set_torch_seed(args.seed)
    ids, train_y = base.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = base.load_stack(DATA_ROOT / "train", ids)
    fold_ids = np.asarray([base.stable_fold_id(s, args.folds) for s in ids])
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
    fine_family = np.repeat(final_family, len(base.CANDIDATES))

    _, _, cm, cs = base.normalize_fit(
        np.zeros((1, 6, len(base.SEQ_FEATURE_NAMES)), dtype=np.float32), final_cf3
    )
    pre_cf = ((pre_cf - cm) / cs).astype(np.float32)
    fine_cf = ((fine_cf - cm) / cs).astype(np.float32)

    val_cands = base.make_candidates(train_x[va], train_x.shape[1] - 1, horizon=2)
    val_cf3 = base.make_candidate_features(train_x[va], train_x.shape[1] - 1, val_cands, horizon=2)
    val_cf3 = ((val_cf3 - cm) / cs).astype(np.float32)
    t, n, b, speed = boundary.local_frame(train_x[va], train_x.shape[1] - 1)
    val_scale = np.maximum(speed * 2.0, base.EPS)

    z = np.load(SCORE_BANK, allow_pickle=True)
    bank_cands = z["cands"]
    bank_scores = z["ens_scores"]
    max_delta = float(np.max(np.abs(bank_cands[va] - val_cands)))
    assert max_delta < 1e-5, f"score bank cand mismatch fold {args.fold}: {max_delta}"
    val_scores = bank_scores[va].astype(np.float32)

    val_payload = (val_cf3, val_cands, train_y[va], (t, n, b), val_scale, val_scores)
    model = boundary.TinyCorrectionNet(pre_cf.shape[-1], args.hidden).to(device)
    print(f"[fold {args.fold}] pretrain start n_pre={len(pre_cf)} n_fine={len(fine_cf)}", flush=True)
    boundary.train_net(model, pre_cf, pre_target, pre_weight, pre_family,
                       args, device, stage="pretrain", val_payload=val_payload)
    print(f"[fold {args.fold}] finetune start", flush=True)
    boundary.train_net(model, fine_cf, fine_target, fine_weight, fine_family,
                       args, device, stage="finetune", val_payload=val_payload)

    corrected_val = boundary.predict_corrected_candidates(
        model, val_cf3, val_cands, (t, n, b), val_scale, args, device,
    )  # (N_val, 25, 3)
    shift_vec = corrected_val - val_cands  # apply_scale=1.0 → = capped delta_vec
    shift_norm = np.linalg.norm(shift_vec, axis=-1)  # (N_val, 25)

    saturation_overall = float((shift_norm.flatten() >= SATURATION_THRESHOLD).mean())
    rate_per_cand = (shift_norm >= SATURATION_THRESHOLD).mean(axis=0).tolist()
    return {
        "fold": int(args.fold),
        "n_val": int(va.sum()),
        "n_cands": int(shift_norm.shape[1]),
        "shift_norm_mean": float(shift_norm.mean()),
        "shift_norm_p50": float(np.median(shift_norm)),
        "shift_norm_p95": float(np.percentile(shift_norm, 95)),
        "shift_norm_max": float(shift_norm.max()),
        "saturation_overall_rate_fold": saturation_overall,
        "saturation_rate_per_cand": rate_per_cand,
        "n_at_least_threshold": int((shift_norm >= SATURATION_THRESHOLD).sum()),
        "n_total_pairs": int(shift_norm.size),
    }


def main() -> int:
    print("[plan-009 c2.1 cap_saturation_extended] start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_extended_pool()
    args = make_args()
    fold_r = train_one_fold(args, device)

    overall_rate = fold_r["saturation_overall_rate_fold"]
    plan_005_rate = 0.0358
    diff = overall_rate - plan_005_rate
    if abs(diff) <= 0.01:
        framing = "reproduce-weak (cap 확장 gain *약*, band/arch main lever)"
    elif diff <= 0.05:
        framing = "reproduce-moderate (cap 확장 gain *중*)"
    else:
        framing = "amplified (cap 확장 gain *강*, cap 도 main 후보)"

    summary = {
        "exp_id": "plan-009/cap_saturation_extended",
        "score_bank": str(SCORE_BANK.relative_to(REPO_ROOT)),
        "cap": CAP,
        "saturation_threshold": SATURATION_THRESHOLD,
        "threshold_definition": "0.0057 = cap × 0.95 (L2-norm of capped delta_vec, m)",
        "n_folds_measured": 1,
        "fold_results": [fold_r],
        "saturation_overall_rate": overall_rate,
        "plan_005_reference_rate": plan_005_rate,
        "diff_vs_plan_005": diff,
        "corrector_main_framing": framing,
        "decision_note": (
            "spec-default — 1-fold (fold=0) approx 채택 (plan-009 §4.2 spec 의 "
            "5-fold concat 의 시간 한계 회피, N_val≈2000 의 binomial std error "
            "≤ 0.005 로 caveat #13 분기 결정에 충분). boundary 학습 hyperparam = "
            "plan-008 c7 selector_retrain.py 의 boundary 호출 args 그대로 reuse "
            "(epochs=12+8, hidden=64, lr=0.001, batch=8192, seed=20260606, "
            "regime_prior_strength=0.0 = Variant A). 5-fold concat 측정은 필요 시 "
            "plan-009 후속 또는 plan-010 carry-over."
        ),
    }
    out_path = OUT_DIR / "cap_saturation_extended.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] cap_saturation_extended.json: {out_path.relative_to(REPO_ROOT)}")
    print(
        f"  overall_rate={overall_rate:.4f}  vs plan-005 0.0358 = "
        f"{diff:+.4f}  framing={framing}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
