---
plan_id: 021
version: 1.3
date: 2026-05-18 (Asia/Seoul)
status: all_complete
band: positive
best_sub_exp: B_gru
best_hit_1cm: 0.6408
best_hit_1.5cm: 0.8100
best_delta_1cm: +0.0088
best_delta_1.5cm: +0.0067
---

# plan-021.results — Frenet Corrector with Input Augment

## 0. 한 줄 결론

4 lever input augment (Frenet trajectory + F0 잔차 seq + F0 soft hit seq + soft label) + dual head (7-anchor classifier + 7×3 reg) 의 **B GRU (단일방향 sequence)** 가 *pass_both* PASS (Δ_1cm +0.0088 / Δ_1.5cm +0.0067). plan-014/016/017/020 4 plan NN paradigm 의 corrector ceiling (단일 metric 향상만 가능) 을 **양쪽 metric 통과로 첫 돌파**. A LGBM 은 1cm 단독 PASS (partial). band positive.

## 1. F0 baseline (G1, plan-020 carry)

| metric | value | drift |
|---|---|---|
| hit@1cm | **0.6320** | 0 (plan-020 carry exact) |
| hit@1.5cm | **0.8033** | 0 |

## 2. 결과 표

| sub-exp | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | pass_both | wall time |
|---|---|---|---|---|---|---|
| F0 baseline | 0.6320 | — | 0.8033 | — | — | 0.8s CPU |
| **A LGBM v1.3** | **0.6488** | **+0.0168** | 0.8070 | +0.0037 | partial (1cm only) | 334s CPU |
| **B GRU v1.3** ⭐ | 0.6408 | +0.0088 | **0.8100** | **+0.0067** | **✓ 둘 다 PASS** | 60s cuda:1 |

## 3. Best sub-exp 단수 선정

- pass criterion (둘 다 ≥ +0.005) 통과 candidates = {**B_gru**} (LGBM partial, GRU full)
- **best_sub_exp = B_gru** (단수 후보, tie-break 불필요)
- best_hit_1cm = 0.6408, best_hit_1.5cm = 0.8100
- band = positive (양쪽 metric 통과의 paradigm value 우선, Δ_1cm 0.0088 marginal-positive 경계)

## 4. A vs B 직접 비교 (paradigm-level)

| axis | A LGBM (170D, tree) | B GRU (134D, sequence) | 차이 |
|---|---|---|---|
| 1cm tight zone Δ | **+0.0168** | +0.0088 | LGBM 1.9× |
| 1.5cm loose zone Δ | +0.0037 | **+0.0067** | GRU 1.8× |
| **pass_both** | ✗ partial | ✓ | **GRU win** |
| fold variance 1cm | 0.0066 | 0.0057 | GRU 안정 |
| wall time | 334s CPU | 60s cuda:1 | GRU 5.5× 빠름 |

→ LGBM 의 macro stat + EWMA 추가 (170D vs 134D) 가 1cm tight zone 의 nonlinear F0-residual signal 흡수에 강함. GRU 의 sequence learning 이 1.5cm graded distance 학습에 강함. 둘은 본질적으로 다른 lever.

## 5. 4 lever 의 paradigm-level 효과 (§8.3)

본 plan 의 4 lever 동시 적용 측정 (lever 별 ablation X — out-of-scope). G3 PASS 결과로 **4 lever 의 조합 paradigm-level 효과 입증**:

- ① Frenet trajectory (invariance) — sample efficiency ↑
- ② F0 residual sequence (Frenet, 21D) — sample-conditional accuracy signal
- ③ F0 soft hit sequence (14D) — sample-conditional graded accuracy
- ④ soft label CE (vs hard) — classifier collapse 회피

**input augment 의 marginal 가치 (vs plan-020 N1 MLP coef)**: B GRU vs N1 = +0.0019 (1cm) + 0.0077 (1.5cm) → input MI 부족 root cause 의 실질 해소.

## 6. plan-020 winner 와 비교

| candidate | Δ_1cm | Δ_1.5cm | pass_both | paradigm |
|---|---|---|---|---|
| plan-020 C05 per-regime F0 | +0.0183 | +0.0053 | ✓ | F0 산식 + 18-regime discrete coef partition |
| **plan-021 B GRU** | **+0.0088** | **+0.0067** | **✓** | **F0 잔차 + input augment + NN dual head** |
| plan-021 A LGBM | +0.0168 | +0.0037 | partial | F0 잔차 + input augment + tree |
| plan-020 N1 MLP coef | +0.0069 | -0.0010 | ✗ | NN coef (input augment 없음) |

→ F0 향상은 두 본질 lever — (a) C05 의 *discrete coef partition*, (b) plan-021 의 *input augment + NN corrector*. 둘은 직교 paradigm — ensemble 잠재력 (follow-up plan-023).

## 7. v1.3 conceptual fix lessons

c7 G2.A 첫 actual run 까지 spec conceptual error catch 안 됨:
- v1.2 (plan-review BLOCKER 0): anchor reference = `x[end_idx]` → A LGBM hit@1cm = 0.0600 (-57pp)
- 원인: final pred = `x[end_idx] + Frenet ±0.5cm` → F0 의 80ms 외삽 (~2cm) 누락
- **v1.3 fix**: anchor reference → `pred_F0_world`. final = `pred_F0_world + R_wfn @ Σ_k π_k · (anchor + offset)`
- 회복: +0.589 (0.0600 → 0.6488)

**plan-review-master 한계**: spec-implementation mode 가 식·시그너처·경계·단위 자족성 점검에 집중 → narrative 와 spec 의 *의도 정합* 의 semantic-level 점검 약함. follow-up = "spec-simulate vs §1 outcome 정합" 강화.

## 8. Decision-note 박제

1. plan-review-master 5-iter BLOCKER 0 (v1.2) — 37 fix, A/B/C 모두 잘 드러남.
2. 자율 spec fix — anchor reference x[end_idx] → pred_F0_world (v1.3, 사용자 자율 진행 승인).
3. LGBM single-class fallback augment classifier-only (v1.3.1) — X length mismatch fix.
4. LGBM regression target clip ±0.005m (§6.3) — tanh-bounded reg_offset 정합.
5. B GRU epochs=30 reduced (full 50) — 60s wall time, train-hit plateau 도달.
6. band positive — Δ_1cm 0.0088 marginal-positive 경계지만 양쪽 metric 통과 paradigm value 우선.

## 9. Caveats

1. Δ_1cm +0.0088 = marginal-positive 경계. 통계적 의미 검증 (fold-별 분산) follow-up.
2. A LGBM 1.5cm 미달 (Δ +0.0037) — anchor radius 0.5cm × reg_offset ±0.005m max ±1cm Frenet 영역 한계.
3. 4 lever 동시 적용 → lever 별 marginal 가치 분리 X (§2.2 out-of-scope).
4. LGBM soft CE 가 hard-argmax + sample_weight 우회 (§6.2) — paradigm 비교 confound.
5. F0 잔차 sub-window 가 plan-020 bf.f0_baseline 의 3pt 호환 가정 (c2 smoke verify, plan-020 spec drift 시 silent bug).
6. v1.3 conceptual fix = plan-review-master 5-iter 가 catch 못 한 semantic-level error.
7. G3 PASS 의 paradigm value 가 27-pool 통합 후 LB 향상 보장 X — follow-up plan-024 carry.

## 10. Follow-up plan 후보

1. **plan-022 (가칭)**: A LGBM 의 1.5cm fail mode 진단 + anchor radius / reg_offset bound 완화 ablation.
2. **plan-023 (가칭)**: A LGBM ⊕ B GRU + plan-020 C05 3 paradigm 직교성 측정 (per-sample correlation, ensemble 잠재력 정량).
3. **plan-024 (가칭)**: B GRU 의 27-pool 통합 + LB 측정 (DACON 5회 quota, 사용자 confirm 필수).

## 11. 산출 파일

- `plans/plan-021-frenet-corrector-input-augment.md` (frontmatter sync v1.3)
- `plans/plan-021-frenet-corrector-input-augment.results.md` (본 문서)
- `analysis/plan-021/results.md` (mirror)
- `analysis/plan-021/baseline_carry.json` (G1)
- `analysis/plan-021/results_lgbm.{json,md}` (c7)
- `analysis/plan-021/results_gru.{json,md}` (c8)
- `analysis/plan-021/{build_input,dual_head_model,run_oof}.py`
- `analysis/plan-021/{c7,c8}_run.log`
- `tests/test_plan021_smoke.py` (11/11 pytest)
