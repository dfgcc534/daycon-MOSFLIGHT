# plan-007 STAGE 1 — sliding window validity

**aug_usable = TRUE (Step 2~4 use sliding aug pool, total 50K)**

- single formula: `frenet_par120_perp_neg020` (CANDIDATES[17])
- N original = 10,000; N sliding = 40,000 (end_idx ∈ [5,8], horizon=2)

## Test results

| metric | value | threshold | pass? |
|---|---|---|---|
| KS p-value | 0.000000 | > 0.075 | ✗ |
| KS statistic | 0.045150 | — | — |
| quantile-by-quantile RMSE | 0.001252 m | < 0.0015 m | ✓ |

Decision: `aug_usable = (KS p > 0.075) OR (quantile RMSE < 0.0015)` → **True**

## Histogram (residual norm)

| bin (m) | original % | sliding % | orig count | slide count |
|---|---|---|---|---|
| [0.0, 0.005) | 33.68% | 34.25% | 3,368 | 13,699 |
| [0.005, 0.01) | 29.52% | 28.48% | 2,952 | 11,393 |
| [0.01, 0.015) | 17.13% | 13.10% | 1,713 | 5,239 |
| [0.015, 0.02) | 5.06% | 6.73% | 506 | 2,691 |
| [0.02, 0.03) | 4.85% | 6.64% | 485 | 2,656 |
| [0.03, 0.05) | 4.50% | 5.47% | 450 | 2,189 |
| [0.05, 0.1) | 4.27% | 4.40% | 427 | 1,761 |
| [0.1, inf) | 0.99% | 0.93% | 99 | 372 |
