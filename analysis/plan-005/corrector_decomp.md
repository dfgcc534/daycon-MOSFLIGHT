# plan-005 STAGE 3 — Corrector Decomposition

- N_train = 10000
- cap = 0.0060, saturation threshold = 0.0057
- **overall cap saturation rate** : 0.0358

## Per-candidate cap saturation (top-5 most saturated)

| cand_idx | rate |
|---:|---:|
| 24 | 0.2732 |
| 21 | 0.1589 |
| 23 | 0.0635 |
| 25 | 0.0530 |
| 26 | 0.0497 |

## Direction breakdown (Frenet local frame; |delta|/scale)

- parallel  mean ± std = 0.0451 ± 0.0872
- perp      mean ± std = 0.0214 ± 0.0602
- binormal  mean ± std = 0.0064 ± 0.0272

## Best-raw-cand error histogram

| band | n |
|:--|---:|
| [0.000, 0.005) | 4594 |
| [0.005, 0.010) | 2594 |
| [0.010, 0.015) | 1290 |
| [0.015, 0.020) | 384 |
| [0.020, 0.030) | 419 |
| [0.030, 0.050) | 371 |
| [0.050, 0.100) | 297 |
| [0.100, inf) | 51 |

## Per-error-band corrector effectiveness

| band | n | hit_before | hit_after | delta |
|:--|---:|---:|---:|---:|
| [0.000, 0.005) | 4594 | 1.0000 | 1.0000 | +0.0000 |
| [0.005, 0.010) | 2594 | 1.0000 | 0.9217 | -0.0783 |
| [0.010, 0.015) | 1290 | 0.0000 | 0.0977 | +0.0977 |
| [0.015, 0.020) | 384 | 0.0000 | 0.0000 | +0.0000 |
| [0.020, 0.030) | 419 | 0.0000 | 0.0000 | +0.0000 |
| [0.030, 0.050) | 371 | 0.0000 | 0.0000 | +0.0000 |
| [0.050, 0.100) | 297 | 0.0000 | 0.0000 | +0.0000 |
| [0.100, inf) | 51 | 0.0000 | 0.0000 | +0.0000 |
