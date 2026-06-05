# plan-008 STAGE 2 — Pruning + Greedy Set-Cover (c4 + c5)

**Step 2a 결과**: 27 → 12 candidates (pruned 15). oracle 0.7188 → 0.7173 (Δ=-0.00150, safe=True). post-hoc soft_hit_pruned=0.6486.

**Step 2b 결과**: pool 12 + 6 new = 18. oracle_final = **0.7543** (target 0.85 minimum, 0.90 stretch). stop_reason = `delta_below_threshold`.

## Step 2a — Pruned candidates

kept names: p0_2d1, acc_2d1_060, frenet_par110_perp_neg020, jerk_small_pos, jerk_small_neg, frenet_par120_perp020, frenet_fast_par120_perp_neg020, latency_short_frenet_best_085, latency_long_frenet_best_108, latency_long_frenet_best_115, latency_long_turn_neg_110, latency_short_turn_pos_090

| pruned_idx | name | dominator | rule | cont_soft | coord_dist | hr_i | hr_j |
|---|---|---|---|---|---|---|---|
| 2 | acc_2d1_050 | acc_2d1_040 | soft | 0.995 | 0.0004 | 0.643 | 0.645 |
| 5 | frenet_best | frenet_par090_perp000 | soft | 0.995 | 0.0003 | 0.651 | 0.651 |
| 6 | frenet_par090_perp000 | frenet_par100_perp000 | soft | 0.997 | 0.0003 | 0.651 | 0.651 |
| 7 | frenet_par100_perp000 | frenet_par090_perp020 | soft | 0.988 | 0.0007 | 0.651 | 0.653 |
| 8 | frenet_par100_perp_neg010 | frenet_par090_perp000 | soft | 0.992 | 0.0005 | 0.651 | 0.651 |
| 9 | frenet_par090_perp020 | frenet_par120_perp020 | soft | 0.990 | 0.0008 | 0.653 | 0.653 |
| 10 | frenet_par080_perp020 | frenet_best | soft | 0.984 | 0.0010 | 0.650 | 0.651 |
| 12 | frenet_fast_par100 | frenet_best | soft | 0.991 | 0.0008 | 0.648 | 0.651 |
| 13 | frenet_slow_par100 | frenet_best | soft | 0.990 | 0.0010 | 0.648 | 0.651 |
| 16 | frenet_par070_perp_neg020 | frenet_best | soft | 0.992 | 0.0008 | 0.646 | 0.651 |
| 1 | acc_2d1_040 | frenet_best | soft | 0.972 | 0.0020 | 0.645 | 0.651 |
| 17 | frenet_par120_perp_neg020 | frenet_best | soft | 0.988 | 0.0008 | 0.649 | 0.651 |
| 22 | latency_short_frenet_best_092 | p0_2d1 | soft | 0.970 | 0.0025 | 0.629 | 0.637 |
| 20 | frenet_slow_par070_perp020 | p0_2d1 | soft | 0.965 | 0.0024 | 0.637 | 0.637 |
| 3 | acc_2d1_056 | acc_2d1_040 | soft | 0.992 | 0.0007 | 0.640 | 0.645 |

## Step 2b — Greedy iteration log

| iter | added_template | family_id | oracle | delta | pool_size |
|---|---|---|---|---|---|
| 0 | (start) | - | 0.7173 | +0.0000 | 12 |
| 1 | arc_decel | 2 | 0.7419 | +0.0246 | 13 |
| 2 | rot_high_150 | 1 | 0.7481 | +0.0062 | 14 |
| 3 | speed_slope_d1_120 | 6 | 0.7509 | +0.0028 | 15 |
| 4 | rot_low_080 | 1 | 0.7522 | +0.0013 | 16 |
| 5 | omega_speed | 6 | 0.7533 | +0.0011 | 17 |
| 6 | fs_3d_low_torsion | 3 | 0.7543 | +0.0010 | 18 |

## Final pool — family composition

| family_id | family_name | count |
|---|---|---|
| 0 | base | 12 |
| 1 | trig | 2 |
| 2 | arc | 1 |
| 3 | frenet_serret_3d | 1 |
| 6 | cross_term | 2 |

## Per-regime worst (sanity only, regime infra 폐기)

| regime | n | oracle_after_greedy |
|---|---|---|
| 10 | 546 | 0.551 |
| 16 | 354 | 0.350 |
| 17 | 356 | 0.472 |
