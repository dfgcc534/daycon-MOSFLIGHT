# IMM feasibility — mode distribution (strong_collapse)

- N = 10000 train samples
- Fit time: 5.8s

## Distribution

| mode | count | fraction |
|---|---|---|
| CV | 631 | 6.31% |
| CA | 8683 | 86.83% |
| CT | 686 | 6.86% |

**Verdict: `strong_collapse`**

> IMM 무가치 — single KF (top mode) 충분.

## ΔBIC (decisiveness)

| stat | value |
|---|---|
| p5 | 1.250 |
| p25 | 6.787 |
| p50 | 18.749 |
| p75 | 44.374 |
| p95 | 86.538 |
| mean | 29.128 |
| frac_gt_10 | 65.71% |
| frac_gt_2 | 92.09% |

> Kass-Raftery: ΔBIC > 2 = weak evidence, > 6 = positive, > 10 = strong.

## Posterior entropy (nats; max = log(3) ≈ 1.099)

| stat | value |
|---|---|
| p5 | 0.0000 |
| p25 | 0.0000 |
| p50 | 0.0009 |
| p75 | 0.1469 |
| p95 | 0.6576 |
| mean | 0.1255 |

> 0 = one-hot (decisive collapse), log(3)≈1.099 = uniform (mode ambiguous).

## RSS quantiles per mode (m²)

| mode | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| CV | 4.82e-05 | 2.90e-04 | 9.67e-04 | 3.21e-03 | 1.39e-02 |
| CA | 9.78e-06 | 5.90e-05 | 1.85e-04 | 6.10e-04 | 3.51e-03 |
| CT | 3.55e-05 | 2.11e-04 | 6.61e-04 | 2.12e-03 | 1.23e-02 |

## Fallback flag counts

| flag | count |
|---|---|
| stationary | 0 |
| degen_plane | 3 |
| non_monotone_angle | 0 |

- stationary: 모든 timestep 위치 동일 (std < 1e-06).
- degen_plane: PCA singular[2]/singular[0] > 0.5 (3D scatter, plane fit unreliable).
- non_monotone_angle: angular θ unwrap 후 |Δθ| > π (CT 부적합 fallback).