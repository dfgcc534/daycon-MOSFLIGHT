---
plan_id: 027
version: 1.1
date: 2026-05-22 (Asia/Seoul)
status: abandoned
abandoned_reason: "user intent mismatch — 본 spec 은 LGBM 3-way ensemble 으로 작성됐으나 사용자의 plan-027 의도 = GRU-attention paradigm 의 후속 (plan-026 GRU-attention 위 ensemble 또는 추가 lever). 사용자 명시 결정 (2026-05-22 turn) 으로 abandoned, plan-030 으로 재발행 예정."
superseded_by: plan-030 (가칭, GRU-attention paradigm 위 plan-029 결과 후속)
band: negative_ensemble
best_cell: E3_weighted
best_hit_1cm: 0.6529
best_delta_1cm: -0.0001
---

# plan-027 — 3-way Ensemble (ABANDONED — user intent mismatch)

> **본 spec abandoned**. main agent 가 plan-025 의 followed_by 후보 (ensemble) 를 사용자 의도 재확인 없이 carry. 사용자의 plan-027 의도 = **GRU-attention paradigm 의 후속** — paradigm 자체가 다름.

## 본 LGBM 실험 finding (참고 only)

- 3 base predictor (p022 + p023 + p026_A2) 모두 LGBM paradigm. base_max = 0.6530.
- best E3 weighted [0.3, 0.3, 0.4] = **0.6529** (-0.0001 vs base_max, effectively tie).
- finding: anchor codebook 차이 (K=14 BCC vs K=50 Fib vs K=14 1058D) 만으로는 prediction error pattern 이 *correlated* → ensemble lift X.

## paradigm-level 해석

본 finding 도 **LGBM 한정 lesson**. paradigm-shared assumption (F0 + LGBM softmax + sphere-mean-zero anchor) 가 3 base 공통 → diversity 부재. 진짜 ensemble diversity = architecture / paradigm 차이 필요.

→ plan-030 에서 GRU-attention 의 ensemble / hyperparameter sweep 으로 paradigm-distinct ensemble 가능성 검증.

## Cross-refs

- 폐기 결정: 2026-05-22 user turn
- spec/code archive: git history (a3f648f, e7ed15a, 16e1397, 6e12de3)
- analysis 산출물: 본 commit 으로 git rm
