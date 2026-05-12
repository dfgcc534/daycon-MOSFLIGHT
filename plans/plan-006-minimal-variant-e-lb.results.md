---
plan_id: 006
exp_id: E001_minimal-variant-e
lb_exp_id: E001_minimal-variant-e
lb_score: TBD
lb_submitted_at: 2026-05-12T11:13:37+09:00
status: partial
date: 2026-05-12 (Asia/Seoul)
---

# plan-006 results — Minimal Variant E LB Validation (physics_bias + soft averaging only)

본 plan = plan-005 의 통찰 "PB framework 의 95% 가 장식이고 진짜 엔진은 *27 후보 + physics_bias + soft averaging* 3 ingredients 뿐" 을 *1 LB 제출* 로 직접 검증하는 cheap experiment.

**LB carry-over open**: dacon-submit 응답 `{isSubmitted: True, detail: 'Success'}` (2026-05-12T11:13:37+09:00). `lb_score` 는 비동기 (DACON dashboard 회수 대기). 점수 도착 시 c5.1 follow-up commit 에서 3 파일 frontmatter `TBD` → `<float>` 동시 갱신 + status `partial` → `all_complete`.

상세 분석: `analysis/plan-006/results.md` 참조.

## §1. Exp summary

| field | value |
|---|---|
| exp_id | `E001_minimal-variant-e` |
| plan_id | 006 |
| based_on | plan-004 (full framework) + plan-005 (component decomposition) |
| compute | local CPU + plan-005 `corrected_*.npz` 재사용 (재학습 0) |
| wall_time | < 1 min (analysis-only) |
| Variant 정의 | `score[i, c] = 0.65 × physics_bias[c]`, soft averaging temp=0.03, GRU 제거, regime 제거 |

## §2. 핵심 수치

| Metric | Value | 비교 |
|---|---|---|
| **E_corrected.soft (OOF)** | **0.6524** | plan-005 추정 0.6517 → 추정 정확도 7bp |
| E_corrected.argmax (OOF) | 0.6491 | score sample-invariant — informational |
| E_raw.soft (OOF) | 0.6250 | corrector 효과 +0.0274pp |
| F_corrected.soft (uniform, sanity) | 0.6520 | < E_corrected (strict) — physics_bias 가 uniform 보다 +0.0004pp |
| **LB lb_score** | **TBD** | dacon 비동기 회수 대기 |

## §3. plan-005 통찰 LB 입증 — 결론 보류 (carry-over)

LB 점수 회수 후 결정:
- **시나리오 A** (`lb_score ≥ 0.6606`): plan-005 통찰 입증 — 단순화 path 정당.
- **시나리오 B** (`lb_score < 0.6606`): GRU/regime 의 *out-of-sample* 기여가 OOF 측정 (noise floor) 보다 큰 것 → OOF↔LB gap 신뢰도 재검토 필요.

## §4. 변경 이력

- 2026-05-12 (KST 11:13): c5 — dacon 제출 성공 `{isSubmitted: True, detail: Success}`. `lb_score: TBD` (carry-over open).
- (대기) c5.1 — 점수 회수 후 3 파일 frontmatter 동시 갱신, status `all_complete`.
