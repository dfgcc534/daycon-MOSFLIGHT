---
plan_id: 026
version: 1.1
date: 2026-05-22 (Asia/Seoul)
status: abandoned
abandoned_reason: "user intent mismatch — 본 spec 은 LGBM block ablation 으로 작성됐으나 사용자의 plan-026 의도 = GRU-attention paradigm. 사용자 명시 결정 (2026-05-22 turn) 으로 abandoned 표기, GRU-attention plan-026 의도는 plan-029 로 재발행 예정."
superseded_by: plan-029 (가칭, GRU-attention paradigm 위 plan-025 1080D input 검증)
band: paradigm_reversal
best_cell: A2_no_block3
best_hit_1cm: 0.6509
best_delta_1cm: 0.0189
---

# plan-026 — Block Ablation (ABANDONED — user intent mismatch)

> **본 spec abandoned**. main agent 가 plan-025 의 followed_by 후보 (block ablation) 를 사용자 의도 재확인 없이 carry. 사용자의 plan-026 의도 = **GRU-attention paradigm 위 plan-025 1080D input 검증** — paradigm 자체가 다름.

## 본 LGBM 실험 finding (참고 only)

- **A2 (no block ③, LGBM 1058D) hit_1cm=0.6509** (+0.0189 vs C1 baseline 0.6320, plan-022 winner 99.66% 회수)
- A1 (no block ②) = 0.6320 (no effect), A3 (no block ④) = 0.6320 (no effect)
- finding: LGBM row-expand 에서 block ③ 22D per-anchor 가 *trivial self-prediction shortcut* trigger → mode collapse 원인.

## paradigm-level 해석

본 finding 은 **LGBM 한정 lesson** — GRU-attention 에서 block ③ 22D 가 query 의 anchor identity 로 *정상 작동* 예상. LGBM 의 mode collapse 는 paradigm mismatch evidence 일 뿐, input design 결함 아님.

→ plan-029 에서 동일 1080D input 으로 GRU-attention paradigm 정상 검증 예정.

## Cross-refs

- 폐기 결정: 2026-05-22 user turn
- spec/code archive: git history (bdf4839, f361d46, 33c5220, 9883d9c, 04ba0bf, d9daaf8, 3b0b4e8)
- analysis 산출물: 본 commit 으로 git rm
