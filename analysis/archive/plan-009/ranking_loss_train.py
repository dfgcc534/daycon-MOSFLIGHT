"""plan-009 c4: Phase 1 학습 wrapper — selector 5-fold OOF retrain (G1 main).

§5.2 + §5.4 spec @ plans/plan-009-selector-ranking-loss.md.

Variant A path (regime_prior_strength=0). plan-008 c7 baseline corrector
(cap=0.006, band=off, arch=default) 위에서 selector 만 retrain — ranking
loss 3 component (NDCG@1 + pair×2 + ListMLE) enable.

산출:
  runs/baseline/H001_ranking-loss/
    oof_selector_scores.npz, test_selector_scores.npz,
    submission_*.csv (5 variants — boundary 호출 후),
    train.log
  analysis/plan-009/ranking_loss_summary.json (§5.4 schema)

학습 args:
  plan-008 c7 selector_retrain.py 의 selector CLI 그대로 reuse + plan-009 신규:
    --loss-components ndcg1,pair2x,listmle, --K-pairs 10, --loss-temperature 0.5.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.pb_0_6822 import boundary, candidates_extended as cx, selector  # noqa: E402

DATA_ROOT = REPO / "data"
PLAN_008_RUN = REPO / "runs/baseline/G001_candidate-redefine"
H001_RUN = REPO / "runs/baseline/H001_ranking-loss"
ANALYSIS_PLAN_008 = REPO / "analysis/plan-008"
OUT_DIR = REPO / "analysis/plan-009"

R_HIT = 0.01

_FAMILY_ID_NAME = {1: "trig", 2: "arc", 3: "frenet_serret_3d", 5: "higher_order", 6: "cross_term"}


def _call_main(main_func, argv: list) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [main_func.__name__, *[str(a) for a in argv]]
        start = time.time()
        main_func()
        print(f"[DONE] {main_func.__name__} elapsed={time.time() - start:.1f}s", flush=True)
    finally:
        sys.argv = old


def _verify_variant_a(run_dir: Path) -> dict:
    z = np.load(run_dir / "oof_selector_scores.npz")
    info: dict = {"keys": list(z.files)}
    if "regime_bias_table" in z.files:
        rbt = z["regime_bias_table"]
        rbt_var = float(rbt.var())
        info["regime_bias_table_var"] = rbt_var
        assert rbt_var < 1e-10, f"regime_residue var {rbt_var}"
        info["variant_a_safe"] = True
    else:
        info["regime_bias_table"] = "(absent — Variant A path)"
        info["variant_a_safe"] = True
    return info


def setup_extended_pool() -> tuple[list, list[int], list[str]]:
    """plan-008 c7 selector_retrain.py 의 monkey-patch 그대로."""
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

    ORIGINAL_27 = copy.deepcopy(selector.CANDIDATES)
    ORIGINAL_make_candidates = selector.make_candidates
    EXTENDED_CANDIDATES = cx.get_extended_candidates_list(KEPT_INDICES, KEPT_FAMILIES)
    print(f"[prep] EXTENDED_CANDIDATES n={len(EXTENDED_CANDIDATES)}")

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
    return EXTENDED_CANDIDATES, KEPT_INDICES, KEPT_FAMILIES


def train_selector() -> None:
    """plan-008 c7 args + plan-009 c3 신규 loss-components."""
    cli_args = [
        "--root", DATA_ROOT,
        "--out-dir", H001_RUN,
        "--models", "attn_gru",
        "--folds", 5, "--fold-limit", 5,
        "--regime-prior-strength", 0,  # ★ Variant A
        "--pre-epochs", 10, "--fine-epochs", 8, "--freeze-fine-epochs", 3,
        "--epoch-plus", 5, "--patience", 4,
        "--hidden", 48, "--batch", 4096,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65,
        "--pairwise-loss-weight", 0.25,  # plan-008 c7 기존 pairwise (label-gap form) 유지
        "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        # plan-009 c3 신규 (★ G1 main)
        "--loss-components", "ndcg1,pair2x,listmle",
        "--K-pairs", 10,
        "--loss-temperature", 0.5,
        "--device", "cuda:0", "--seed", 20260506, "--log-every", 1,
    ]
    assert "--regime-prior-strength" in [str(a) for a in cli_args]
    rps_idx = [str(a) for a in cli_args].index("--regime-prior-strength")
    assert str(cli_args[rps_idx + 1]) == "0", "regime_residue: regime_prior_strength != 0"
    _call_main(selector.SELECTOR_MAIN, cli_args)


def boundary_inference_and_submission() -> None:
    """plan-008 c7 패턴 reuse — H001 selector OOF score 위에서 boundary inference."""
    score_bank = H001_RUN / "oof_selector_scores.npz"
    test_score_bank = H001_RUN / "test_selector_scores.npz"
    assert score_bank.exists() and test_score_bank.exists(), (
        f"selector full-fit 결과 누락 — {score_bank}, {test_score_bank}"
    )
    _call_main(boundary.BOUNDARY_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", H001_RUN,
        "--fold", 0, "--folds", 5,
        "--score-bank", score_bank,
        "--test-score-bank", test_score_bank,
        "--epochs", 12, "--fine-epochs", 8, "--min-epochs", 5, "--patience", 4,
        "--hidden", 64, "--batch", 8192,
        "--lr", 0.001, "--fine-lr-scale", 0.18,
        "--cap", 0.006, "--apply-scale", 1.0,
        "--device", "cuda:0", "--seed", 20260606, "--save-val-pred",
        "--make-test",
    ])
    soft_csv = H001_RUN / "submission_boundary_tiny_soft.csv"
    step1_csv = H001_RUN / "submission_step1.csv"
    if soft_csv.exists():
        step1_csv.write_bytes(soft_csv.read_bytes())
        print(f"[c4] submission_step1.csv ← {soft_csv.name}")


def measure_metrics() -> dict:
    """selector OOF score 위에서 4 metric 산출 — §5.3 합격기준."""
    score_path = H001_RUN / "oof_selector_scores.npz"
    z = np.load(score_path)
    oof_scores = z["ens_scores"]
    cands_oof = z["cands"]  # (N, 25, 3) — 학습 wrapper 가 박제

    train_ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    n_cand = int(oof_scores.shape[1])
    assert cands_oof.shape == (len(train_y), n_cand, 3), (
        f"cands shape {cands_oof.shape} mismatch with train_y {len(train_y)}, n_cand {n_cand}"
    )

    # OOF soft hit (softmax_temperature=0.03)
    soft_pred = selector.soft_select(cands_oof, oof_scores, temperature=0.03)
    soft_hit = float((np.linalg.norm(soft_pred - train_y, axis=1) <= R_HIT).mean())

    # OOF argmax hit
    argmax_idx = oof_scores.argmax(axis=1)
    argmax_pred = cands_oof[np.arange(len(train_y)), argmax_idx]
    argmax_hit = float((np.linalg.norm(argmax_pred - train_y, axis=1) <= R_HIT).mean())

    # top1_ranking_acc = argmax(selector) == argmin(per_cand_err)
    per_cand_err = np.linalg.norm(cands_oof - train_y[:, None, :], axis=-1)  # (N, 25)
    oracle_best_idx = per_cand_err.argmin(axis=1)
    top1_ranking_acc = float((argmax_idx == oracle_best_idx).mean())

    # gap_ranking = oracle_1cm − oof_soft_hit
    oracle_1cm = float((per_cand_err.min(axis=1) <= R_HIT).mean())
    gap_ranking = oracle_1cm - soft_hit
    return {
        "oof_soft_hit": soft_hit,
        "oof_argmax_hit": argmax_hit,
        "top1_ranking_acc": top1_ranking_acc,
        "gap_ranking": gap_ranking,
        "oracle_1cm": oracle_1cm,
        "n_train": int(len(train_y)),
        "n_candidates": int(n_cand),
    }


def main() -> int:
    H001_RUN.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[plan-009 c4 ranking_loss_train] start")
    setup_extended_pool()
    train_selector()
    variant_a = _verify_variant_a(H001_RUN)
    print(f"[c4] variant_a_check: {variant_a}")
    boundary_inference_and_submission()
    metrics = measure_metrics()
    summary = {
        "exp_id": "H001_ranking-loss",
        **metrics,
        "loss_components_effective_weight": {"ce": 1.0, "ndcg1": 1.0, "pair_plan_009": 2.0,
                                              "pairwise_plan_008_label_gap": 0.25, "listmle": 0.5},
        "variant_a_safe": bool(variant_a.get("variant_a_safe", False)),
        "variant_a_check": variant_a,
        "g1_pass": {
            "oof_soft_hit_pass": metrics["oof_soft_hit"] >= 0.6703,
            "top1_ranking_acc_pass": metrics["top1_ranking_acc"] >= 0.22,
            "gap_ranking_pass": metrics["gap_ranking"] <= 0.09,
            "variant_a_safe_pass": bool(variant_a.get("variant_a_safe", False)),
        },
        "decision_note": (
            "plan-008 c7 selector args 그대로 + plan-009 c3 신규 loss-components "
            "ndcg1,pair2x,listmle (effective weights {ce:1, ndcg1:1, pair_plan_009:2, "
            "pairwise_plan_008_label_gap:0.25, listmle:0.5}). plan-008 의 기존 "
            "label-gap pairwise (weight=0.25) 도 유지 — plan-009 의 pair2x 와 별도 form, "
            "compound 효과 측정."
        ),
    }
    out_path = OUT_DIR / "ranking_loss_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] ranking_loss_summary.json: {out_path.relative_to(REPO)}")
    print(f"  oof_soft_hit={metrics['oof_soft_hit']:.4f}  top1={metrics['top1_ranking_acc']:.4f}  gap={metrics['gap_ranking']:.4f}")
    print(f"  G1 pass: {summary['g1_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
