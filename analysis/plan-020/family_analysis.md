# plan-020 STAGE 4 G3 — Family-level analysis

## F0 baseline: hit@1cm = 0.6320, hit@1.5cm = 0.8033

## 17 후보 × 2 metric × 5-fold concat OOF

| # | candidate | family | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | fold_var_1cm | pass 둘 다 ≥ +0.005 |
|---|---|---|---|---|---|---|---|---|
| C05_per_regime_f0 | F2_data_driven | 0.6503 | +0.0183 | 0.8086 | +0.0053 | 0.0056 | ✓ |
| N01_mlp_coef | F2_data_driven | 0.6389 | +0.0069 | 0.8023 | -0.0010 | 0.0053 | ✗ |
| N02_tcn_coef | F2_data_driven | 0.6372 | +0.0052 | 0.8036 | +0.0003 | 0.0059 | ✗ |
| N05_moe | F2_data_driven | 0.6327 | +0.0007 | 0.8065 | +0.0032 | 0.0069 | ✗ |
| C10_bishop_frame | F5_기하학 | 0.6320 | +0.0000 | 0.8033 | +0.0000 | 0.0052 | ✗ |
| C13_levy_prior | F6_도메인_정보 | 0.6320 | +0.0000 | 0.8033 | +0.0000 | 0.0052 | ✗ |
| C04_imm | F1_회전 | 0.5980 | -0.0340 | 0.7974 | -0.0059 | 0.0104 | ✗ |
| C08_singer | F4_noise_adaptive | 0.5951 | -0.0369 | 0.7851 | -0.0182 | 0.0089 | ✗ |
| C01_helix | F1_회전 | 0.5874 | -0.0446 | 0.7912 | -0.0121 | 0.0102 | ✗ |
| C03_ctrv | F1_회전 | 0.5207 | -0.1113 | 0.7187 | -0.0846 | 0.0120 | ✗ |
| C02_ctra | F1_회전 | 0.5070 | -0.1250 | 0.6898 | -0.1135 | 0.0090 | ✗ |
| C07_jerk_quartic | F3_고차_미분 | 0.3929 | -0.2391 | 0.5847 | -0.2186 | 0.0080 | ✗ |
| C11_se3_twist | F5_기하학 | 0.3450 | -0.2870 | 0.5323 | -0.2710 | 0.0070 | ✗ |
| C14_trajectory_knn | F7_비모수 | 0.3404 | -0.2916 | 0.5336 | -0.2697 | 0.0090 | ✗ |
| C09_kalman_smoother | F4_noise_adaptive | 0.2374 | -0.3946 | 0.3846 | -0.4187 | 0.0037 | ✗ |
| C06_quintic_hermite | F3_고차_미분 | 0.0096 | -0.6224 | 0.0260 | -0.7773 | 0.0009 | ✗ |
| C12_wingbeat_corrected | F6_도메인_정보 | 0.0008 | -0.6312 | 0.0015 | -0.8018 | 0.0002 | ✗ |

## Family-level winner (§8.2 2-단계: pass 우선, Δ_combined tie-break)

| family | winner | Δ_1cm | Δ_1.5cm | Δ_combined |
|---|---|---|---|---|
| F1_회전 | 없음 | — | — | — |
| F2_data_driven | C05_per_regime_f0 | +0.0183 | +0.0053 | +0.0209 |
| F3_고차_미분 | 없음 | — | — | — |
| F4_noise_adaptive | 없음 | — | — | — |
| F5_기하학 | 없음 | — | — | — |
| F6_도메인_정보 | 없음 | — | — | — |
| F7_비모수 | 없음 | — | — | — |

## Overall best_candidate (§9.1.1 단수 선정)

- **best_candidate**: `C05_per_regime_f0`
- **best_family**: `F2_data_driven`
- **best_hit_1cm**: 0.6503 (Δ +0.0183)
- **best_hit_1.5cm**: 0.8086 (Δ +0.0053)
- **band**: `positive` (positive if Δ_1cm ≥ +0.01)

## G3 합격 기준

- 17 × 2 metric table 박제 ✓
- 7 family winner 박제 ✓
- ≥ 1 후보 paired Δ ≥ +0.005 *둘 다*: ✓ (n_pass = 1)
- G3 **PASS**.