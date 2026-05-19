---
plan_id: 022
finished_at: 2026-05-19 (Asia/Seoul)
status: all_complete
band: positive
best_sub_exp: A6_bcc14_tau001
best_hit_1cm: 0.6528
best_hit_1.5cm: 0.8104
best_delta_1cm: +0.0208
best_delta_1.5cm: +0.0071
exp_ids_completed:
  - Z022_A1_octa7
  - Z022_A2_ico13
  - Z022_A3_cubocta13
  - Z022_A4_2shell13
  - Z022_A5_cube8
  - Z022_A6_bcc14
  - Z022_A7_fib13
exp_ids_skipped: []
lb_score: null
---

# plan-022.results pair (WORKFLOW.md §11)

핵심 결과는 `analysis/plan-022/results.md` 의 11 항목에 박제. 본 pair file 은
frontmatter 4-way 토큰 일치 (WORKFLOW.md §4 / §11) 의무 충족용 stub.

## 핵심 결과 요약

- **best**: A6_bcc14_tau001 (BCC 14 anchor + τ_cls=0.001 sharp soft label)
- **paired Δ**: Δ_1cm = +0.0208, Δ_1.5cm = +0.0071 (둘 다 PASS criterion +0.005 통과)
- **pass_both cell 수**: 10/21 cell
- **band**: positive (G3 PASS, severe 0건, warn 0건)

## 상세 분석 위치

- `analysis/plan-022/results.md` — 11 항목 G_final 종합
- `analysis/plan-022/paradigm_analysis.{json,md}` — 21 cell grid + marginals
- `analysis/plan-022/results_A{1..7}.{json,md}` — 7 sub-exp 개별 결과

## Follow-up

- plan-023: A6_bcc14 + corrector reg head 재투입 ablation
- plan-024: A6_bcc14 + GRU sub-exp + ensemble
- plan-025: DACON LB 측정 (사용자 quota confirm 필수)

## Post-G_final 자체실험 (2026-05-19)

A8_bcc15 (= A6_bcc14 + center, K=15) controlled ablation — `analysis/plan-022/results.md` §12.

핵심 finding: paradigm finding §8.1 의 "center 제거 = mode collapse 완화 → OOF
향상" 가설 **refuted**. A8 vs A6 Δ(hit@1cm) 평균 -0.9bp / Δ(hit@1.5cm) 평균
+0.2bp = seed noise 수준. `max_class_ratio` 는 ground-truth 자연 분포 (`q_true.mean`)
의 mirror 일 뿐 — distribution-match (KL ≤ 0.005, top1_acc 모두 일치) 직접
측정으로 selector 가 두 layout 다 자연 분포 충실히 추종 확인. A6 winner 결론
불변 (sweep 박제 유지), 그러나 우위 원인은 ① BCC 14 anchor geometry ② sharp
τ=0.001 이지 *center 제거 자체* 가 아님. 후속 진단 metric 권고: `max_class_ratio`
→ `dist_match_KL` + `top1_acc` 교체.

artifacts: `analysis/plan-022/diag_center_bias_a6_a8.py`, `diag_a8.json`,
`diag_a8.log`, `anchors.py:ANCHORS_A8`.
