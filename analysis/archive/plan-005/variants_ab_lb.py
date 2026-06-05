"""plan-005 follow-up: Variant A (GRU + physics, no regime) + Variant B (physics + regime, no GRU)
test submission CSV 생성. plan-006 §6 패턴 답습 (corrected_test + soft_select temp=0.03).

- Variant A: ens_scores from variant_A_no_regime/test_selector_scores.npz (retrain output)
- Variant B: physics_bias_full + 0.45 * regime_bias_table[test_regimes]  (no GRU = bias-only)
- 둘 다 corrected_test (plan-005/corrected_test.npz) 위에 soft_select.

Run: python -c "import sys; sys.path.insert(0,'analysis/plan-005'); import variants_ab_lb; variants_ab_lb.main()"
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN005_DIR = REPO / "analysis/plan-005"
RUN_A_DIR = REPO / "runs/baseline/E002_variant-a-gru-physics"
RUN_B_DIR = REPO / "runs/baseline/E003_variant-b-physics-regime"

PHYSICS_WEIGHT = 0.65
REGIME_WEIGHT = 0.45
SOFT_TEMP = 0.03
N_REGIMES = 18
N_CANDIDATES = 27


def _assert_schema(out_csv: Path) -> None:
    """5-line assert (plan-006 §6 답습)."""
    ref = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    sub = pd.read_csv(out_csv)
    assert sub.shape == ref.shape, f"shape: sub={sub.shape} ref={ref.shape}"
    assert list(sub.columns) == list(ref.columns), "columns mismatch"
    assert (sub.iloc[:, 0].astype(str).values == ref.iloc[:, 0].astype(str).values).all(), "id order"
    coord_numeric = sub[sub.columns[1:]].apply(pd.to_numeric, errors="coerce")
    assert coord_numeric.notna().all().all(), "non-numeric coord"
    assert np.isfinite(coord_numeric.to_numpy()).all(), "non-finite coord"


def _write_submission(pred: np.ndarray, out_csv: Path) -> None:
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    ref = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    assert pred.shape == (len(test_ids), 3), pred.shape
    id_col = ref.columns[0]
    coord_cols = list(ref.columns[1:])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        id_col: test_ids,
        coord_cols[0]: pred[:, 0],
        coord_cols[1]: pred[:, 1],
        coord_cols[2]: pred[:, 2],
    }).to_csv(out_csv, index=False)
    _assert_schema(out_csv)


def build_variant_A() -> Path:
    """Variant A = ens_scores_A (GRU + physics, regime_prior_strength=0) → soft_select on corrected_test."""
    test_scores_npz = PLAN005_DIR / "variant_A_no_regime/test_selector_scores.npz"
    z_a = np.load(test_scores_npz, allow_pickle=True)
    ens_scores_a = z_a["ens_scores"].astype(np.float32)
    assert ens_scores_a.shape == (10000, N_CANDIDATES), ens_scores_a.shape

    z_corr = np.load(PLAN005_DIR / "corrected_test.npz")
    corrected_test = z_corr["corrected"].astype(np.float32)
    assert corrected_test.shape == (10000, N_CANDIDATES, 3)

    pred = selector.soft_select(corrected_test, ens_scores_a, temperature=SOFT_TEMP)
    out_csv = RUN_A_DIR / "submission.csv"
    _write_submission(pred, out_csv)
    return out_csv


def build_variant_B() -> Path:
    """Variant B = physics_bias × 0.65 + 0.45 × regime_bias_table[test_regimes] → soft_select on corrected_test.

    physics_bias / regime_bias_table 은 *full train* 으로 fit (plan-005 bundle.physics_bias_full /
    regime_bias_table_full 와 동일 방식, line 196-200).
    """
    # Train 으로부터 bias 산정
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx_train = train_x.shape[1] - 1
    train_cands = selector.make_candidates(train_x, end_idx_train, horizon=2).astype(np.float32)

    physics_bias = selector.candidate_physics_bias(train_cands, train_y).astype(np.float32)  # [27]
    regime_bins = selector.fit_regime_bins(train_x, end_idx_train)
    train_regimes = selector.assign_regimes(train_x, end_idx_train, regime_bins).astype(np.int64)
    regime_bias_table = selector.candidate_regime_bias(
        train_cands, train_y, train_regimes, regime_count=N_REGIMES
    ).astype(np.float32)  # [18, 27]
    assert regime_bias_table.shape == (N_REGIMES, N_CANDIDATES)

    # Test regime 할당 (동일 bins)
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)
    end_idx_test = test_x.shape[1] - 1
    test_regimes = selector.assign_regimes(test_x, end_idx_test, regime_bins).astype(np.int64)
    assert test_regimes.shape == (len(test_ids),)

    # Variant B score: physics_bias × 0.65 + 0.45 × regime_bias[test_regimes]
    score_b = (
        PHYSICS_WEIGHT * physics_bias[None, :]
        + REGIME_WEIGHT * regime_bias_table[test_regimes]
    ).astype(np.float32)
    assert score_b.shape == (len(test_ids), N_CANDIDATES)

    z_corr = np.load(PLAN005_DIR / "corrected_test.npz")
    corrected_test = z_corr["corrected"].astype(np.float32)

    pred = selector.soft_select(corrected_test, score_b, temperature=SOFT_TEMP)
    out_csv = RUN_B_DIR / "submission.csv"
    _write_submission(pred, out_csv)
    return out_csv


def main() -> dict[str, Path]:
    a = build_variant_A()
    b = build_variant_B()
    return {"variant_a": a, "variant_b": b}


if __name__ == "__main__":
    out = main()
    for k, v in out.items():
        print(f"{k}: {v}")
