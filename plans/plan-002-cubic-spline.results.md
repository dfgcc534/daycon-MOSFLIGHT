---
plan_id: 002
finished_at: 2026-05-10T05:12+09:00
status: partial
exp_ids_completed:
  - S001_cspline-natural-full
  - S002_cspline-notaknot-full
  - S003_cspline-window-grid
  - S004_smoothing-spline-tuned
exp_ids_skipped: []
best_exp_id_cv: S003_cspline-window-grid
best_exp_id_lb: TBD
submission_paths:
  - runs/baseline/S001_cspline-natural-full/submission.csv
  - runs/baseline/S002_cspline-notaknot-full/submission.csv
  - runs/baseline/S003_cspline-window-grid/submission.csv
  - runs/baseline/S004_smoothing-spline-tuned/submission.csv
lb_scores:
  S001: TBD
  S002: TBD
  S003: TBD
  S004: TBD
lb_metric: hit_rate (반경 비공개; plan-001 LB 0.60 = B001)
lb_submitted_at_first: 2026-05-10T05:09+09:00
lb_submitted_at_last:  2026-05-10T05:12+09:00
carry_over_reason: |
  dacon_submit_api 가 post-only API (점수 fetch 미지원). 4 isSubmitted=True 확인됨.
  사용자가 dacon.io 대회 페이지 (236716) 에서 4 점수 확인 후 본 frontmatter
  lb_scores 4 키 + analysis/plan-002/lb_log.md `lb_score` 칸 + registry notes
  일괄 갱신 → 그 시점에 status: all_complete 로 마감.
---

# plan-002 results — Cubic spline interpolation baseline

본 frontmatter 는 WORKFLOW.md §6 의 plan results 의무 요소. 본문 분석은
`analysis/plan-002/results.md` 에 박제.

## per-experiment 요약 (status / cv / 핵심 metric)

| exp_id | status | started_at | duration_sec | cv_mean_eucl ± std | per-axis MAE | hit@0.10 | best run dir | baseline diff vs B001 (mean Δ) | sign 일관성 | lb_score | 특이사항 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S001_cspline-natural-full   | complete | 2026-05-10T04:42 | 5.4   | 0.01742 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | runs/baseline/S001_cspline-natural-full   | +0.00448 | 1.00 | TBD | full-window natural BC; H1 confirmed |
| S002_cspline-notaknot-full  | complete | 2026-05-10T04:42 | 5.3   | 0.05370 ± 0.00282 | [0.0277, 0.0288, 0.0235] | 0.8815 | runs/baseline/S002_cspline-notaknot-full  | +0.04076 | 1.00 | TBD | aggressive 외삽; 본 plan 의 worst CV |
| S003_cspline-window-grid    | complete | 2026-05-10T04:43 | 226.8 | 0.01740 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | runs/baseline/S003_cspline-window-grid    | +0.00446 | 1.00 | TBD | chosen=[(5,nat),(5,nat),(4,nat)]; clamped 0회 채택 |
| S004_smoothing-spline-tuned | complete | 2026-05-10T04:48 | 17.1  | 0.03322 ± 0.00270 | [0.0191, 0.0176, 0.0115] | 0.9506 | runs/baseline/S004_smoothing-spline-tuned | +0.02027 | 1.00 | TBD | s=[1e-4,1e-4,1e-4]; H3 refuted (CV) |

CV-best = **S003** (cv_mean_eucl 0.01740; B001 floor 0.01294 미달, +0.00446 worse, sign 일관성 100%).

## H1/H2/H3 verdict (CV 기준; LB 보강 carry-over)

- H1 (full-window 보간 ≥ B001): **CV confirmed**. natural 0.0174, not-a-knot 0.0537 — 둘 다 등속 외삽보다 worse.
- H2 (windowed grid ≈ B001): **CV partially refuted**. clamped 가 단 1회도 안 뽑힘 (chord-derivative 가 noise 데이터에서 over-extrapolate); small-window natural 이 dominate 하지만 B001 floor 미달.
- H3 (smoothing 이 B001 위협): **CV refuted**. fit 영역 axis MAE 의 이득이 외삽 mean_eucl 로 transfer 안 됨.

## best 선택 사유

- best_exp_id_cv = S003: cv_mean_eucl 0.01740 (S001=0.01742 보다 1e-5 작음, tie-break 작은-window).
- best_exp_id_lb: 4 LB 점수 도착 후 결정.

## CV-LB 상관 분석

`analysis/plan-002/results.md §7` 에 점수 도착 후 산점 + Spearman ρ 박제 예정. 현재 prior:
- 비례 시나리오: S003 LB 1위.
- 반전 시나리오: S004 가 LB hit_rate tail 에서 회복.
- 약-상관 시나리오: 4 LB 비슷 (~0.005 noise 영역).

## submission 결과

- 4 csv 모두 sample_submission 스키마 100 % 일치 (rows=10000, dtype float64, NaN/Inf 0).
- LB API 응답 4건 모두 `{isSubmitted: True, detail: Success}`.
- Budget 4/5 사용. 1 contingency 슬롯.
- carry-over: 4 LB 점수 회수만 보류.

## 다음 plan 후보 (enumeration only — 우선순위 X)

1. Kalman / Savitzky-Golay 입력 평활 + polyfit
2. velocity model — t=0 instantaneous derivative 등속 외삽 (B001 일반화)
3. ensemble (B001, S003, S004)
4. neural seq2seq (LSTM / Transformer)
5. per-axis combo of {polyfit, cspline, smoothing}
6. hit-radius probing (별도 plan)
