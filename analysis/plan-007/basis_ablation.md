# plan-007 STAGE 3 — basis ablation

- baseline (Step 2 single fit) hit = **0.6342**
- final best basis hit = **0.6387** (Δ = +0.0044)
- best basis variables = `['d1', 'acc_par', 'acc_perp', 'd2', 'jerk', 'ts_term', 'speed_slope_d1', 'rotation_term']` (size = 8)
- elapsed = 33.2s

## Ablation steps

| step | added | best_hit | marginal_gain | kept? |
|---|---|---|---|---|
| 1 | `speed_slope_d1` | 0.6356 | +0.0014 | ✓ kept |
| 2 | `rotation_term` | 0.6387 | +0.0031 | ✓ kept |
| 3 | `speed_norm_acc_par` | 0.6391 | +0.0004 | ✗ dropped |
| 4 | `v_mean3_minus_d1` | 0.6387 | +0.0000 | ✗ dropped |

## Best basis coefficients

| var | coeff |
|---|---|
| `d1` | +1.8279 |
| `acc_par` | +1.4429 |
| `acc_perp` | -0.3243 |
| `d2` | +0.0658 |
| `jerk` | +0.0841 |
| `ts_term` | +0.0388 |
| `speed_slope_d1` | +0.1047 |
| `rotation_term` | +0.3087 |
