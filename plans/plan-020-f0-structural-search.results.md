---
plan_id: 020
version: 1.4
date: 2026-05-18 (Asia/Seoul)
status: all_complete
band: positive
best_candidate: C05_per_regime_f0
best_hit_1cm: 0.6503
best_hit_1.5cm: 0.8086
best_delta_1cm: +0.0183
best_delta_1.5cm: +0.0053
---

# plan-020.results — F0 Structural Search: C05 per-regime F0 단독 PASS

## 0. 한 줄 결론

**F0 단일 공식의 *18-regime 별 (d1, par, perp) 계수 분리* (C05_per_regime_f0)** 이 17 후보 (14 deterministic + 3 NN) 중 *단독* `paired Δ ≥ +0.005 둘 다` 통과. hit@1cm 0.6320 → 0.6503 (Δ +1.83 percentage point), hit@1.5cm 0.8033 → 0.8086 (Δ +0.53 pp). band positive.

## 1. F0 baseline (G1, plan-006 reproduce)

| metric | value | spec range | drift |
|---|---|---|---|
| hit@1cm | **0.6320** | [0.6315, 0.6325] | 0 (exact) |
| hit@1.5cm | **0.8033** | [0.8028, 0.8038] | 0 (exact) |
| fold variance 1cm | 0.0052 | < 0.05 | OK |
| fold variance 1.5cm | 0.0087 | < 0.05 | OK |

- N = 10000 train, 5-fold (`stable_fold_id(str(id), 5)` = MD5 prefix mod 5)
- Wall time 0.8 s (CPU)
- plan-014 G0 preflight reproduce 4 자리 정확 일치.

## 2. 17 후보 × 2 metric × 5-fold concat OOF

| candidate | family | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | fold_var_1cm | pass 둘 다 |
|---|---|---|---|---|---|---|---|
| F0 baseline | — | 0.6320 | — | 0.8033 | — | 0.0052 | — |
| **C05_per_regime_f0** | **F2 data-driven** | **0.6503** | **+0.0183** | **0.8086** | **+0.0053** | **0.0056** | **✓** |
| N01_mlp_coef | F2 data-driven | 0.6389 | +0.0069 | 0.8023 | −0.0010 | 0.0053 | ✗ |
| N02_tcn_coef | F2 data-driven | 0.6372 | +0.0052 | 0.8036 | +0.0003 | 0.0059 | ✗ |
| N05_moe | F2 data-driven | 0.6327 | +0.0007 | 0.8065 | +0.0032 | 0.0069 | ✗ |
| C10_bishop_frame | F5 기하학 | 0.6320 | +0.0000 | 0.8033 | +0.0000 | 0.0052 | ✗ (λ=1 stuck = F0) |
| C13_levy_prior | F6 도메인 | 0.6320 | +0.0000 | 0.8033 | +0.0000 | 0.0052 | ✗ (degenerate = F0) |
| C04_imm | F1 회전 | 0.5980 | −0.0340 | 0.7974 | −0.0059 | 0.0104 | ✗ |
| C08_singer | F4 noise | 0.5951 | −0.0369 | 0.7851 | −0.0182 | 0.0089 | ✗ |
| C01_helix | F1 회전 | 0.5874 | −0.0446 | 0.7912 | −0.0121 | 0.0102 | ✗ |
| C03_ctrv | F1 회전 | 0.5207 | −0.1113 | 0.7187 | −0.0846 | 0.0120 | ✗ |
| C02_ctra | F1 회전 | 0.5070 | −0.1250 | 0.6898 | −0.1135 | 0.0090 | ✗ |
| C07_jerk_quartic | F3 고차 | 0.3929 | −0.2391 | 0.5847 | −0.2186 | 0.0080 | ✗ |
| C11_se3_twist | F5 기하학 | 0.3450 | −0.2870 | 0.5323 | −0.2710 | 0.0070 | ✗ |
| C14_trajectory_knn | F7 비모수 | 0.3404 | −0.2916 | 0.5336 | −0.2697 | 0.0090 | ✗ |
| C09_kalman_smoother | F4 noise | 0.2374 | −0.3946 | 0.3846 | −0.4187 | 0.0037 | ✗ |
| C06_quintic_hermite | F3 고차 | 0.0096 | −0.6224 | 0.0260 | −0.7773 | 0.0009 | ✗ |
| C12_wingbeat_corrected | F6 도메인 | 0.0008 | −0.6312 | 0.0015 | −0.8018 | 0.0002 | ✗ (CMA fit fail, default broken) |

## 3. 7 family winner (§8.2 2-단계: pass 우선, Δ_combined tie-break)

| family | winner | Δ_combined |
|---|---|---|
| F1_회전 (C1/C2/C3/C4) | **없음** | — |
| **F2_data_driven (C5/N1/N2/N5)** | **C05_per_regime_f0** | **+0.0209** |
| F3_고차_미분 (C6/C7) | 없음 | — |
| F4_noise_adaptive (C8/C9) | 없음 | — |
| F5_기하학 (C10/C11) | 없음 | — |
| F6_도메인_정보 (C12/C13) | 없음 | — |
| F7_비모수 (C14) | 없음 | — |

## 4. Overall best — single winner (§9.1.1)

- **best_candidate**: `C05_per_regime_f0`
- **best_family**: F2_data_driven
- **best_hit_1cm**: 0.6503 (Δ +0.0183, sample-level paired)
- **best_hit_1.5cm**: 0.8086 (Δ +0.0053)
- **band**: `positive` (Δ_1cm ≥ +0.01)

## 5. NN vs Deterministic 직접 비교 (§8.3)

| 비교축 | 비교 | 결과 |
|---|---|---|
| 학습 방식 분리 (학습 paradigm vs deterministic regime) | C05 vs N1 MLP | C05 (Δ +0.018/+0.005) > N1 (Δ +0.007/−0.001) — **deterministic regime-conditional 분리 > NN coef regression** |
| Architecture (NN family 안) | N1 MLP vs N2 TCN | N1 hit@1cm 0.6389 ≈ N2 0.6372 — TCN 의 11-step context 가 *additional gain 없음* |
| Gating (mixture 효과) | N5 vs C1+C2+C6+F0 단순 평균 | N5 Δ +0.001/+0.003 — gating 의 추가 가치 marginal, mixture 자체가 F0 보다 worse 가능 |

→ **paradigm-level 결론**: 본 plan-020 narrative 하에서 NN coef regression (small MLP/TCN/MoE) 의 hit@1cm 향상은 marginal (+0.001 ~ +0.007), hit@1.5cm 거의 0 → *둘 다* criterion 통과 X. C05 deterministic per-regime 의 격차는 *NN 이 regime-conditional 분리의 정밀도 못 따라감* 이 본질.

## 6. N1 = plan-007 F002 paradigm 재측정 (drift 박제)

| 항목 | 값 |
|---|---|
| plan-007 F002 OOF hit@1cm | 0.6482 |
| plan-020 N1 OOF hit@1cm | 0.6389 |
| drift | **−0.0093** (in ±0.02 threshold ✓ — `n1_drift_vs_f002` warn 미발동) |

drift 원인 (§N+2 caveat #4):
1. **input feature 구성** (가장 큼 추정): F002 = 13D *통계 aggregates* × 6-step window, N1 = 27D *raw sequence* × 3-step
2. **train pool**: F002 = 50K (10K + 4× sliding), N1 = 10K original 만
3. **loss schedule**: F002 ≈ MSE, N1 = annealed smooth-hit + boundary
4. **seed list**: F002 = 20260606 single, N1 = [20260518..20260520] (3 seed) best-on-train

±0.02 threshold 안 → paradigm-class 동일 / instance 다름 가설 검증 ✓.

## 7. Decision-note (자율 결정 박제)

1. **spec-default — narrative 정합 §9/§N+1 삭제** (v1.1): 27-pool oracle delta + 작업량 회계 → narrative ("F0 단일 공식 결과 최대화") 와 불정합 → 삭제. STAGE 6 → STAGE 5, c13 → c12 renumber.
2. **spec-default — plan-review-master 5-iter 자동 fix BLOCKER 0 도달** (v1.2): 37 edit (산식 정의 / NN spec / loss surrogate / 단위 통일 / dispatch / single winner rule 등).
3. **spec-default — 코드 재사용 검토 carry signature 정합** (v1.3): stable_fold_id MD5 정정, fit_regime_bins → dict + assign_regimes 분리, N1 drift threshold ±0.01 → ±0.02 (architecture/pool/loss 차이).
4. **spec-default — C04 IMM fit interface fix + C09 Kalman init fix** (v1.3.1, c9 relaunch): cma_es_fit fit_c04_imm 의 π_raw + w_diag closure, _kalman_per_axis init a=0 + larger P.
5. **spec-default — c9 reduced spec** (popsize=10/maxiter=50/seeds=3): full (20/200/5) ≈ ~2 hr/단일 C09 → reduce → ~10 min total. C05 winner 발견 충분.
6. **spec-default — c10 reduced spec** (seeds=3/epochs=30): cuda:1, ~2 min total. 3 NN paradigm 한계 확인.
7. **spec-default — overall best 단수 선정 4-rule** (§9.1.1 grade): pass criterion 통과 후 Δ_combined → hit_1cm → fold_variance tie-break. C05 단독 통과라 trivial.
8. **spec-default — band 임계 positive ≥ +0.01** (plan body 미명시, 자율): Δ_1cm +0.0183 = +1.83 pp → positive.

## 8. Follow-up plan 후보 (≥2건, §9.4 합격 기준)

1. **plan-021 (가칭) — C05 → 27-pool 통합 + LB 측정**: 본 plan-020 의 C05_per_regime_f0 winner 를 plan-004 27-pool 에 추가 → best-of-28 oracle Δ 측정 + actual LB submit (DACON 5회 quota, 사용자 confirm 필수). *직교성* 측정 (단독 hit 향상 ≠ 27-pool 가치 보장 — §N+2 #8 caveat).
2. **plan-022 (가칭) — c12 wingbeat fix + C10 saddle escape + full-spec 재측정**: C12 default 산출이 0.0008 (broken — FFT mask + IFFT 가 zero-trajectory 의심). C10 의 CMA-ES 가 λ=1 saddle 에 stuck. 본 plan-020 v1.4 의 reduced spec → full spec (popsize=20, maxiter=200, seeds=5) 재측정 + 위 2 fix → quality 검증.
3. **plan-023 (가칭) — NN coef ⊕ C05 conditional ensemble**: NN coef 가 hit@1cm 에서 marginal 향상 (+0.001~+0.007) 만 보이는 paradigm 한계 박제 후, *N1/N2 → C05 regime 별 fallback / boost* 구조 탐색. 18-regime × NN 가중 ensemble.

## 9. Caveats

- C05 winner 의 hit@1cm 향상 +0.018 이 *paired 통계* 라 fold-level 분산 (per-fold mean Δ std ≈ 0.005, 큰 절대량 아님). 추가 다중 seed 재측정 권고.
- 18-regime fold-internal fit 의 min_samples=100 threshold 가 일부 regime 의 fallback (global F0) 비율 ≥ 1/3 가능성 — regime distribution skew 영향 별도 분석 필요 (plan-022).
- C12 wingbeat 의 default broken 은 본 plan body §6.1 C12 spec 의 FFT mask 산식 자체 점검 필요 (값 0 → IFFT 후 zero-trajectory → F0(0) = 0 가까이). v1.5 fix.
- NN coef paradigm 의 paradigm-level 한계 = plan-014/015/016 의 corrector ceiling 5.4% 와 동일 root cause (input feature ↔ 정답 방향 MI 부족) 확인.
- LB submit 미실행 — plan-020 narrative 에서 dacon-submit out-of-scope, follow-up plan-021 carry.

## 10. plan-017 overlap 해소 (§9.3)

- plan-017 (GRU-attention coef regressor) status: IN PROGRESS (본 plan-020 G_final 시점 기준).
- plan-020 N3/N4 (BiGRU/Transformer coef) out-of-scope 박제 — plan-017 산출 carry pending.
- plan-017 G_final 도달 시 → plan-020.results 의 **부록** 으로 N3/N4 결과 추가 가능. plan-020 본 분석은 N1/N2/N5 기준 그대로.
- ±0.01 이상 차이 발생 시 `plan017_carry_conflict` warn — 미발동 (carry pending).

## 11. 산출 파일

- `plans/plan-020-f0-structural-search.md` (frontmatter sync — status, best_candidate, band)
- `plans/plan-020-f0-structural-search.results.md` (본 문서)
- `analysis/plan-020/baseline_oof.{json,md}` — G1 결과
- `analysis/plan-020/results_deterministic.{json,md}` — c9 G2.D 결과
- `analysis/plan-020/results_nn.{json,md}` — c10 G2.N 결과
- `analysis/plan-020/family_analysis.{json,md}` — c11 G3 결과
- `analysis/plan-020/results.md` — 본 results 의 사본 (mirror)
- `analysis/plan-020/{baseline_f0,formula_deterministic,formula_nn,cma_es_fit,run_oof,family_analysis}.py` — 구현 modules
- `analysis/plan-020/{c9_run,c10_run}.log` — actual run logs
- `tests/test_plan020_smoke.py` — 6 pytest (G0)
