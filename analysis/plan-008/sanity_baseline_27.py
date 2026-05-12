"""plan-008 c5.5: Sanity baseline — 27 후보 + Variant A 새 hyperparam (family 효과 분리, v2.6).

spec @ plans/plan-008-candidate-redefine-corrector-redesign.md §6.0.

목적:
  Step 3 G2 OOF 임계 (0.70) 가 *family 추가 효과* 인지 *hyperparam 변경 효과* 인지
  분리. plan-005 STAGE 6 의 Variant A 측정 hyperparam 과 *다른* (재현 측정).
  → family_effect = oof_extended_pool − sanity_baseline_27_oof (Step 3 에서 derive).

Outputs:
  - runs/baseline/G001_sanity-27/oof_selector_scores.npz (selector 산출)
  - analysis/plan-008/sanity_baseline_27.json (OOF metrics)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
RUN_DIR = REPO / "runs/baseline/G001_sanity-27"
ANALYSIS_DIR = REPO / "analysis/plan-008"


def _call_main(main_func, argv: list) -> None:
    """Mirror run_full._call_main — set sys.argv around argparse-based main()."""
    old = sys.argv[:]
    try:
        sys.argv = [main_func.__name__, *[str(a) for a in argv]]
        start = time.time()
        main_func()
        print(f"[DONE] {main_func.__name__} elapsed={time.time() - start:.1f}s", flush=True)
    finally:
        sys.argv = old


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    from src.pb_0_6822 import selector

    # ORIGINAL_27_CANDIDATES + ORIGINAL_make_candidates — Sanity baseline 진입 시 monkey-patch 없음.
    # (이 시점에서 selector.CANDIDATES 는 기본 27, selector.make_candidates 는 기본 구현)
    assert len(selector.CANDIDATES) == 27, (
        f"expected 27 base candidates, got {len(selector.CANDIDATES)}"
    )
    for c in selector.CANDIDATES:
        assert c.family_id == 0, f"base candidate {c.name} family_id != 0"

    # v2.6 Variant A path hyperparam (§6.0 spec)
    _call_main(selector.SELECTOR_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", RUN_DIR,
        "--models", "attn_gru",
        "--folds", 5, "--fold-limit", 5,
        "--regime-prior-strength", 0,         # Variant A 핵심
        "--pre-epochs", 10, "--fine-epochs", 8, "--freeze-fine-epochs", 3,
        "--epoch-plus", 5, "--patience", 4,
        "--hidden", 48, "--batch", 4096,
        # plan-004 production hyperparams (그 외 default 유지)
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "cuda:1", "--seed", 20260506, "--log-every", 1,
        "--skip-full",                          # OOF만 — full-fit 불필요
    ])

    # OOF metrics 산출
    score_path = RUN_DIR / "oof_selector_scores.npz"
    assert score_path.exists(), f"missing {score_path}"
    z = np.load(score_path)
    oof_scores = z["ens_scores"]                  # (N, 27)
    oof_cands = z["oof_cands"] if "oof_cands" in z.files else None

    # Compute OOF hit (argmax + soft@0.03)
    train_ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", train_ids)
    end_idx = train_x.shape[1] - 1
    cands = selector.make_candidates(train_x, end_idx, horizon=2)

    argmax_idx = oof_scores.argmax(axis=1)
    argmax_pred = cands[np.arange(len(train_y)), argmax_idx]
    argmax_hit = float((np.linalg.norm(argmax_pred - train_y, axis=1) <= 0.01).mean())

    soft_pred = selector.soft_select(cands, oof_scores, temperature=0.03)
    soft_hit = float((np.linalg.norm(soft_pred - train_y, axis=1) <= 0.01).mean())

    oracle = float(
        (np.linalg.norm(cands - train_y[:, None, :], axis=2).min(axis=1) <= 0.01).mean()
    )

    summary = {
        "exp_id": "G001-sanity-27",
        "sanity_baseline_27_oof_argmax": argmax_hit,
        "sanity_baseline_27_oof_soft": soft_hit,
        "oracle_27": oracle,
        "hyperparam_set": {
            "models": "attn_gru",
            "folds": 5,
            "regime_prior_strength": 0,
            "pre_epochs": 10, "fine_epochs": 8, "freeze_fine_epochs": 3,
            "epoch_plus": 5, "patience": 4,
            "hidden": 48, "batch": 4096,
            "lr": 0.001, "fine_lr_scale": 0.12,
            "prior_strength": 0.65,
            "pairwise_loss_weight": 0.25,
            "fine_distill_weight": 0.55,
        },
        "n_train": int(len(train_y)),
        "n_candidates": int(oof_scores.shape[1]),
        # Variant A baseline 0.6570 ± 0.005 재현 검증 (§6.4)
        "baseline_band": [0.652, 0.662],
        "in_baseline_band": 0.652 <= soft_hit <= 0.662,
    }
    (ANALYSIS_DIR / "sanity_baseline_27.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[c5.5] sanity_baseline_27_oof: argmax={argmax_hit:.4f} soft={soft_hit:.4f} "
        f"oracle={oracle:.4f} "
        f"in_band={summary['in_baseline_band']} (target Variant A 0.6570 ± 0.005)"
    )


if __name__ == "__main__":
    main()
