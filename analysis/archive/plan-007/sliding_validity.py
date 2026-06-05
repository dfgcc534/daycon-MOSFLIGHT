"""plan-007 STAGE 1 — sliding window validity check.

Spec: plans/plan-007-formula-tuning.md §4.1.

Compares:
  - Original residuals: end_idx = train_x.shape[1] - 1 = 10, horizon=2, target = train_y.
  - Sliding residuals: end_idx ∈ [5, 8] (4 stages × 10K = 40K), horizon=2,
                       target = train_x[:, end_idx + 2] (within-trajectory).
Single formula = CANDIDATES[17] = frenet_par120_perp_neg020 (plan-006 best).

Outputs:
  - analysis/plan-007/sliding_validity.json
  - analysis/plan-007/sliding_validity.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from src.pb_0_6822 import selector

DATA_ROOT = Path("data")
OUT_DIR = Path("analysis/plan-007")
BEST_IDX = 17  # CANDIDATES[17] = frenet_par120_perp_neg020 (plan-006 §5.5)


def stage1_sliding_validity() -> dict:
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    assert train_x.shape[1] == 11, f"expected T=11, got {train_x.shape}"

    # ── 1. Original residuals (end_idx = 10, horizon=2) ──
    cands_orig = selector.make_candidates(train_x, train_x.shape[1] - 1, horizon=2)
    pred_orig = cands_orig[:, BEST_IDX, :]
    err_orig = np.linalg.norm(pred_orig - train_y, axis=1)  # [N=10K]

    # ── 2. Sliding residuals (end_idx ∈ [5, 8], horizon=2, target = train_x[:, end+2]) ──
    err_slide_list = []
    for end_idx in range(5, 9):
        cands_sub = selector.make_candidates(train_x, end_idx, horizon=2)
        target_sub = train_x[:, end_idx + 2]
        pred_sub = cands_sub[:, BEST_IDX, :]
        err_sub = np.linalg.norm(pred_sub - target_sub, axis=1)
        err_slide_list.append(err_sub)
    err_slide = np.concatenate(err_slide_list)  # [40K]

    # ── 3. KS test (two-sample) ──
    ks_stat, ks_pvalue = stats.ks_2samp(err_orig, err_slide)

    # ── 4. Quantile-by-quantile RMSE ──
    quantiles = np.linspace(0.05, 0.95, 19)
    q_orig = np.quantile(err_orig, quantiles)
    q_slide = np.quantile(err_slide, quantiles)
    quantile_rmse = float(np.sqrt(((q_orig - q_slide) ** 2).mean()))

    # ── 5. Histogram comparison (informational) ──
    bins = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100, np.inf]
    hist_orig, _ = np.histogram(err_orig, bins=bins)
    hist_slide, _ = np.histogram(err_slide, bins=bins)

    # ── 6. Decision ──
    aug_usable = bool((ks_pvalue > 0.075) or (quantile_rmse < 0.0015))

    return {
        "n_orig": int(len(err_orig)),
        "n_slide": int(len(err_slide)),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "quantile_rmse": quantile_rmse,
        "threshold_ks_p": 0.075,
        "threshold_quantile_rmse": 0.0015,
        "aug_usable": aug_usable,
        "best_idx": BEST_IDX,
        "best_candidate_name": selector.CANDIDATES[BEST_IDX].name,
        "sliding_end_idx_range": [5, 8],
        "sliding_horizon": 2,
        "histogram_bins": [float(b) for b in bins[:-1]] + ["inf"],
        "histogram_orig_counts": [int(h) for h in hist_orig],
        "histogram_slide_counts": [int(h) for h in hist_slide],
        "histogram_orig_pct": [float(h / len(err_orig)) for h in hist_orig],
        "histogram_slide_pct": [float(h / len(err_slide)) for h in hist_slide],
    }


def _format_markdown(result: dict) -> str:
    lines = []
    aug = "TRUE (Step 2~4 use sliding aug pool, total 50K)" if result["aug_usable"] else "FALSE (Step 2~4 original 10K only)"
    lines.append("# plan-007 STAGE 1 — sliding window validity")
    lines.append("")
    lines.append(f"**aug_usable = {aug}**")
    lines.append("")
    lines.append(f"- single formula: `{result['best_candidate_name']}` (CANDIDATES[{result['best_idx']}])")
    lines.append(f"- N original = {result['n_orig']:,}; N sliding = {result['n_slide']:,} "
                 f"(end_idx ∈ [5,8], horizon=2)")
    lines.append("")
    lines.append("## Test results")
    lines.append("")
    lines.append(f"| metric | value | threshold | pass? |")
    lines.append(f"|---|---|---|---|")
    ks_pass = result["ks_pvalue"] > result["threshold_ks_p"]
    qr_pass = result["quantile_rmse"] < result["threshold_quantile_rmse"]
    lines.append(f"| KS p-value | {result['ks_pvalue']:.6f} | > {result['threshold_ks_p']} | {'✓' if ks_pass else '✗'} |")
    lines.append(f"| KS statistic | {result['ks_statistic']:.6f} | — | — |")
    lines.append(f"| quantile-by-quantile RMSE | {result['quantile_rmse']:.6f} m | < {result['threshold_quantile_rmse']} m | {'✓' if qr_pass else '✗'} |")
    lines.append("")
    lines.append("Decision: `aug_usable = (KS p > 0.075) OR (quantile RMSE < 0.0015)` "
                 f"→ **{result['aug_usable']}**")
    lines.append("")
    lines.append("## Histogram (residual norm)")
    lines.append("")
    lines.append("| bin (m) | original % | sliding % | orig count | slide count |")
    lines.append("|---|---|---|---|---|")
    bins = result["histogram_bins"]
    n_buckets = len(result["histogram_orig_pct"])
    for i in range(n_buckets):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else "inf"
        lines.append(
            f"| [{lo}, {hi}) "
            f"| {result['histogram_orig_pct'][i] * 100:.2f}% "
            f"| {result['histogram_slide_pct'][i] * 100:.2f}% "
            f"| {result['histogram_orig_counts'][i]:,} "
            f"| {result['histogram_slide_counts'][i]:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = stage1_sliding_validity()
    (OUT_DIR / "sliding_validity.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (OUT_DIR / "sliding_validity.md").write_text(_format_markdown(result))
    print(f"aug_usable={result['aug_usable']} "
          f"ks_p={result['ks_pvalue']:.6f} quantile_rmse={result['quantile_rmse']:.6f}")


if __name__ == "__main__":
    main()
