---
plan_id: 027
finished_at: 2026-05-22 (Asia/Seoul)
status: abandoned
abandoned_reason: "user intent mismatch (LGBM 으로 실행됐으나 사용자 의도는 GRU-attention paradigm)"
superseded_by: plan-029
best_cell: E3_weighted
best_hit_1cm: 0.6529
best_hit_1p5cm: 0.8118
band: negative_ensemble
exp_ids_completed:
  - Z027_E1_eq_2way
  - Z027_E2_eq_3way
  - Z027_E3_weighted_optim
---

# plan-027.results — ABANDONED (user intent mismatch)

본 LGBM 3-way ensemble results 는 사용자 의도와 다른 paradigm 결과. git history (commit 6e12de3) 에 보존, plan-029 재발행 예정.

## LGBM ensemble finding (참고)

- base_max (p022 = p023) = 0.6530
- best E3 [0.3, 0.3, 0.4] = 0.6529 = effectively tie
- root cause: 3 base 의 paradigm-shared assumption (F0 + LGBM + sphere-mean-zero anchor) → prediction error correlated
- hit_1p5cm 만 +0.0010 미세 lift (0.8108 → 0.8118)

paradigm-level 해석: anchor codebook 차이만으로는 ensemble diversity 부족. architecture / paradigm 차이 필요.
