# plan-008 STAGE 1 — Diagnostic (c2)

**결론 (1 줄)**: dominant_causes=[(none)], prune_count=24 (tier=strict_v2.4), main_bottleneck=ranking (gap_ranking=0.0516, gap_drift=-0.0004).

## Selector gap decomposition

| metric | value |
|---|---|
| oracle_hit_raw (best of 27, §1.1) | 0.7188 |
| oracle_hit_corrected | 0.7111 |
| selector_argmax_hit | 0.6595 |
| selector_soft_hit (temp=0.03) | 0.6599 |
| top1_ranking_accuracy (= argmax == best) | 0.1602 |
| **gap_ranking** (oracle − argmax) | **0.0516** |
| **gap_drift** (argmax − soft) | **-0.0004** |
| main_bottleneck | **ranking** |

*top-1 ranking 의 정확한 의미*: 27 후보 중 *raw best (oracle best)* 정확 픽 비율. 1cm 안 들어가는 비율 (= argmax_hit) 과는 다름. 진짜 main metric = `gap_ranking` (selector 가 hit zone 의 후보를 *놓치는* 비율).

## Oracle miss residual breakdown (v2.5)

- mask: `err_raw.min(axis=1) > 0.01`  →  n_oracle_miss = 2812 (miss_rate = 0.2812)

| 항목 | 값 |
|---|---|
| par_pct (tangent 분산) | 0.425 |
| perp_pct (xy 수직 분산) | 0.393 |
| z_pct | 0.182 |
| corr_rotation (\|omega_z\| vs err) | 0.062 |
| corr_curvature (kinematic K vs err) | -0.000 |
| corr_jerk (\|jerk\| vs err) | 0.172 |

**dominant_causes**: []

## 가지치기 후보 (structural containment, v2.4 + v2.7 auto-relax)

tier = `strict_v2.4`, soft_thr = 0.95, dist_thr = 5mm, count = 24

| i | name | dom_idx | dom_name | rule | cont_soft | coord_dist (m) | hr_i | hr_j | oracle_δ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | p0_2d1 | 1 | acc_2d1_040 | soft | 0.975 | 0.0017 | 0.637 | 0.645 | 0.00090 |
| 1 | acc_2d1_040 | 5 | frenet_best | soft | 0.972 | 0.0020 | 0.645 | 0.651 | 0.00010 |
| 2 | acc_2d1_050 | 1 | acc_2d1_040 | soft | 0.995 | 0.0004 | 0.643 | 0.645 | 0.00000 |
| 3 | acc_2d1_056 | 1 | acc_2d1_040 | soft | 0.992 | 0.0007 | 0.640 | 0.645 | 0.00030 |
| 4 | acc_2d1_060 | 1 | acc_2d1_040 | soft | 0.990 | 0.0009 | 0.640 | 0.645 | 0.00070 |
| 5 | frenet_best | 6 | frenet_par090_perp000 | soft | 0.995 | 0.0003 | 0.651 | 0.651 | 0.00000 |
| 6 | frenet_par090_perp000 | 7 | frenet_par100_perp000 | soft | 0.997 | 0.0003 | 0.651 | 0.651 | 0.00000 |
| 7 | frenet_par100_perp000 | 9 | frenet_par090_perp020 | soft | 0.988 | 0.0007 | 0.651 | 0.653 | 0.00000 |
| 8 | frenet_par100_perp_neg010 | 6 | frenet_par090_perp000 | soft | 0.992 | 0.0005 | 0.651 | 0.651 | 0.00000 |
| 9 | frenet_par090_perp020 | 18 | frenet_par120_perp020 | soft | 0.990 | 0.0008 | 0.653 | 0.653 | 0.00000 |
| 10 | frenet_par080_perp020 | 5 | frenet_best | soft | 0.984 | 0.0010 | 0.650 | 0.651 | 0.00000 |
| 11 | frenet_par110_perp_neg020 | 5 | frenet_best | soft | 0.990 | 0.0008 | 0.649 | 0.651 | 0.00020 |
| 12 | frenet_fast_par100 | 5 | frenet_best | soft | 0.991 | 0.0008 | 0.648 | 0.651 | 0.00000 |
| 13 | frenet_slow_par100 | 5 | frenet_best | soft | 0.990 | 0.0010 | 0.648 | 0.651 | 0.00000 |
| 14 | jerk_small_pos | 9 | frenet_par090_perp020 | soft | 0.986 | 0.0009 | 0.652 | 0.653 | 0.00040 |
| 15 | jerk_small_neg | 5 | frenet_best | soft | 0.986 | 0.0009 | 0.647 | 0.651 | 0.00050 |
| 16 | frenet_par070_perp_neg020 | 5 | frenet_best | soft | 0.992 | 0.0008 | 0.646 | 0.651 | 0.00000 |
| 17 | frenet_par120_perp_neg020 | 5 | frenet_best | soft | 0.988 | 0.0008 | 0.649 | 0.651 | 0.00010 |
| 19 | frenet_fast_par120_perp_neg020 | 1 | acc_2d1_040 | soft | 0.955 | 0.0031 | 0.643 | 0.645 | 0.00040 |
| 20 | frenet_slow_par070_perp020 | 0 | p0_2d1 | soft | 0.965 | 0.0024 | 0.637 | 0.637 | 0.00020 |
| 22 | latency_short_frenet_best_092 | 0 | p0_2d1 | soft | 0.970 | 0.0025 | 0.629 | 0.637 | 0.00010 |
| 23 | latency_long_frenet_best_108 | 0 | p0_2d1 | soft | 0.960 | 0.0033 | 0.636 | 0.637 | 0.00030 |
| 25 | latency_long_turn_neg_110 | 0 | p0_2d1 | soft | 0.955 | 0.0046 | 0.615 | 0.637 | 0.00070 |
| 26 | latency_short_turn_pos_090 | 0 | p0_2d1 | soft | 0.961 | 0.0034 | 0.604 | 0.637 | 0.00050 |

## margin (top1 − top2) 분포 (logit 단위)

| pct | value |
|---|---|
| p10 | 0.0011 |
| p25 | 0.0032 |
| p50 | 0.0080 |
| p75 | 0.0201 |
| p90 | 0.0559 |

softmax_diffusion_signal (p50 < 0.1) = `True`

## Per-regime oracle (sanity only, decision 무관)

| regime | n | current_oracle | gap_to_0.85 |
|---|---|---|---|
| 0 | 661 | 0.900 | -0.050 |
| 1 | 629 | 0.941 | -0.091 |
| 2 | 663 | 0.884 | -0.034 |
| 3 | 458 | 0.904 | -0.054 |
| 4 | 615 | 0.673 | +0.177 |
| 5 | 274 | 0.774 | +0.076 |
| 6 | 544 | 0.833 | +0.017 |
| 7 | 701 | 0.815 | +0.035 |
| 8 | 592 | 0.801 | +0.049 |
| 9 | 562 | 0.754 | +0.096 |
| 10 | 546 | 0.526 | +0.324 |
| 11 | 355 | 0.566 | +0.284 |
| 12 | 549 | 0.643 | +0.207 |
| 13 | 916 | 0.622 | +0.228 |
| 14 | 476 | 0.582 | +0.268 |
| 15 | 749 | 0.546 | +0.304 |
| 16 | 354 | 0.345 | +0.505 |
| 17 | 356 | 0.441 | +0.409 |

## Oracle miss regime distribution (sanity only)

| regime | n_in_miss | miss_rate |
|---|---|---|
| 0 | 58 | 0.088 |
| 1 | 32 | 0.051 |
| 2 | 68 | 0.103 |
| 3 | 40 | 0.087 |
| 4 | 204 | 0.332 |
| 5 | 60 | 0.219 |
| 6 | 82 | 0.151 |
| 7 | 126 | 0.180 |
| 8 | 106 | 0.179 |
| 9 | 126 | 0.224 |
| 10 | 259 | 0.474 |
| 11 | 151 | 0.425 |
| 12 | 186 | 0.339 |
| 13 | 353 | 0.385 |
| 14 | 189 | 0.397 |
| 15 | 335 | 0.447 |
| 16 | 236 | 0.667 |
| 17 | 201 | 0.565 |
