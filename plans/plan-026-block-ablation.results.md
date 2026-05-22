---
plan_id: 026
finished_at: 2026-05-22 (Asia/Seoul)
status: abandoned
abandoned_reason: "user intent mismatch (LGBM 으로 실행됐으나 사용자 의도는 GRU-attention)"
superseded_by: plan-029
best_cell: A2_no_block3
best_hit_1cm: 0.6509
best_delta_1cm: 0.0189
band: paradigm_reversal
exp_ids_completed:
  - Z026_A1_no_block2
  - Z026_A2_no_block3
  - Z026_A3_no_block4
---

# plan-026.results — ABANDONED (user intent mismatch)

본 LGBM block ablation results 는 사용자 의도와 다른 paradigm 의 결과 (사용자 의도 = GRU-attention). git history (commit d9daaf8 + 3b0b4e8) 에 보존, plan-029 재발행 예정.

## LGBM finding (참고)

- A2 (no block ③): hit_1cm=0.6509 (+0.0189) — mode collapse 해소
- A1 / A3: 0.6320 (no effect)
- root cause: block ③ 22D per-anchor = LGBM row-expand 의 trivial self-prediction trigger

paradigm-level 해석: LGBM 한정 lesson. GRU-attention 위에서는 block ③ 가 query identity 로 정상 작동 예상.
