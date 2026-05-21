---
plan_id: 024
finished_at: null
status: draft
band: null
best_metric:
  hit_1cm: null
  hit_1.5cm: null
  delta_1cm: null
  delta_1.5cm: null
  gap_ranking: null
exp_ids_completed: []
exp_ids_skipped: []
lb_score: null
---

# plan-024.results pair (WORKFLOW.md §11)

핵심 결과는 `analysis/plan-024/results.md` 의 11 항목 (G_final 시점 박제) 에 박제 예정. 본 pair file 은 frontmatter 4-way 토큰 일치 (WORKFLOW.md §4 / §11) 의무 충족용 stub. 현재 status=draft (c1 spec 박제 단계).

## 진행 상태 (G_final 미도달 — placeholder)

- G0 (인프라): [TODO]
- G1 (F0 + plan-022 carry reproduce): [TODO]
- G2 (cross-attention 5-fold OOF 최소 동등성): [TODO]
- G3 (lift + gap_ranking): [TODO]
- G_final (LB 회수 + 3-file sync): [TODO]

## 핵심 spec 위치

- `plans/plan-024-cross-attention-anchor-vocab.md` — 본 plan 의 §0~§14 spec 본문
- `analysis/plan-024/results.md` — G_final 시점 박제 예정 (현재 미생성)
- `analysis/plan-024/per_anchor_dist.json` — plan-022 carry 비교 박제 예정

## carry reference

- plan-022 winner: A6_bcc14_τ001 (hit_1cm 0.6528 / hit_1.5cm 0.8104 / Δ_1cm +0.0208 / Δ_1.5cm +0.0071)
- plan-008 measured gap_ranking: 0.0516 (base 27 cand) — architecture-extractable headroom
- plan-009 ranking_loss G1 fail: oof_soft_hit 0.6482 / gap_ranking 0.108 — 본 plan 의 caveat anchor

## follow-up plan 후보 (G_final 시점 박제 예정)

- plan-025 (가칭): ideas.md priority 5 (A1 Multi-window stat / A6 WAP composite / B3 STA/LTA / Multi-Parse Input / B2 Pct-of-rolling-std)
- plan-026 (가칭): anchor radius 확장 + F0 baseline ML 화
- plan-027 (가칭): plan-022 LGBM + plan-024 cross-attn ensemble
