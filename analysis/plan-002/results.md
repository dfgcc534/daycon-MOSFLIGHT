# plan-002 results — Cubic spline interpolation baseline

**status**: partial (4 LB submissions sent, scores 회수 carry-over)
**finished_at (CV portion)**: 2026-05-10 KST
**4 LB submissions submitted_at**: 2026-05-10T05:09~05:12 KST

---

## §1. 종합 표 — B001 + S001~S004 (5 entries)

| exp_id | method | hyperparam | cv_mean_eucl ± std | per-axis MAE [x, y, z] | hit@0.10 | runtime (s) | lb_score |
|---|---|---|---|---|---|---|---|
| **B001_linear-2pt** (plan-001) | polyfit | w=2, d=1 | **0.01294 ± 0.00058** | [0.0070, 0.0071, 0.0050] | 0.9923 | 0.8 | 0.60 (plan-001) |
| S001_cspline-natural-full | cspline | w=11, BC=natural | 0.01742 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | 5.4 | TBD |
| S002_cspline-notaknot-full | cspline | w=11, BC=not-a-knot | 0.05370 ± 0.00282 | [0.0277, 0.0288, 0.0235] | 0.8815 | 5.3 | TBD |
| S003_cspline-window-grid | cspline | per-axis [(5,nat),(5,nat),(4,nat)] | 0.01740 ± 0.00071 | [0.0096, 0.0096, 0.0066] | 0.9842 | 226.8 | TBD |
| S004_smoothing-spline-tuned | smoothing | k=3, s=[1e-4, 1e-4, 1e-4] | 0.03322 ± 0.00270 | [0.0191, 0.0176, 0.0115] | 0.9506 | 17.1 | TBD |

CV-best of S001~S004 = **S003** (0.01740). 4 변형 모두 B001 floor 0.01294 미달.

---

## §2. per-experiment 분석

### S001 — natural BC, full 11-pt window

CV mean_eucl 0.01742 — B001 floor 대비 +0.0045 worse, 모든 fold 에서 강하게 worse (paired Δ 분포 [+0.0043 ~ +0.0048], sign=1.00). Natural BC ("끝 곡률 0") 가정이 외삽을 *flat* 으로 만들지만, 11점 전체 보간이 oldtimesteps 노이즈를 fit 끝까지 끌고 와 boundary 곡률을 왜곡. 결과: B001 등속 외삽 (oldtimesteps 무시) 보다 *systematically* 부정확.

### S002 — not-a-knot BC, full 11-pt window

CV mean_eucl 0.05370 — B001 대비 +0.041 (3× worse), per-axis MAE 도 3× 영역. Not-a-knot 은 boundary 의 cubic 을 그대로 외삽 영역으로 연장하므로 노이즈가 amplify. hit@0.10 도 0.88 까지 떨어짐. 본 plan 의 *의도된 worst case* — 어떤 정상 데이터도 이 BC 로 외삽 + 노이즈를 같이 쥐고 가면 flat 가정보다 훨씬 부정확함.

### S003 — per-axis (window × bc_type) grid

CV mean_eucl 0.01740 — S001 과 동급. 12-cell grid (window ∈ {4,5,7,11} × BC ∈ {natural,not-a-knot,clamped}) 의 5-fold inner CV 가 5 outer fold 에서 동일하게 [(5,nat),(5,nat),(4,nat)] 선택. 즉 **clamped 가 한 번도 안 뽑힘**. H2 의 핵심 prediction (clamped 의 chord-derivative ≈ B001 등속 외삽) 은 정량적으로 *노이즈 데이터에서는 clamped 가 chord-derivative noise 까지 같이 외삽해 natural 보다 worse* 라는 결과로 부분 refuted. 작은 window + natural 이 fit 영역의 노이즈를 적게 가져오는 한편 boundary 곡률 0 가정이 외삽 안정.

### S004 — smoothing spline tuned

CV mean_eucl 0.03322 — 본 plan 의 가장 유망 후보였으나 *worst-but-S002*. Inner CV 가 모든 axis 에서 s=1e-4 선택 (s_grid {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} 중 lower-end 인접). s=0 (interpolation) 보다 약간의 smoothing 이 inner CV val MAE 를 줄이지만, 이는 fit 영역 거리 metric 에서의 이득. 외삽 영역 (t=+80) 에서는 boundary 곡률 fit 정확도가 더 중요해 smoothing 이 손해. 즉 inner CV 의 axis MAE 가 t=+80 의 mean_eucl 의 좋은 proxy 가 아님 (selection bias).

---

## §3. paired comparison — same-fold Δ vs B001

`Δ_fold[i] = S00x.fold_means[i] - B001.fold_means[i]` (음수 = S00x 가 우수). 5 fold 동일 split (kfold_split deterministic, seed=42, plan-001 §3.1 와 100 % 동등).

| exp_id | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean Δ | sign 일관성 |
|---|---|---|---|---|---|---|---|
| B001 (ref) | 0.01371 | 0.01201 | 0.01259 | 0.01313 | 0.01326 | — | — |
| S001 | +0.00465 | +0.00429 | +0.00439 | +0.00475 | +0.00430 | **+0.00448** | 1.00 (전부 worse) |
| S002 | +0.04376 | +0.03834 | +0.03812 | +0.04290 | +0.04065 | **+0.04076** | 1.00 |
| S003 | +0.00462 | +0.00428 | +0.00437 | +0.00473 | +0.00427 | **+0.00446** | 1.00 |
| S004 | +0.02081 | +0.01773 | +0.01880 | +0.02445 | +0.01958 | **+0.02027** | 1.00 |

모든 4 변형 × 모든 5 fold 에서 B001 보다 strictly worse. mean Δ ≫ B001 fold-σ (0.00058). 즉 noise 영역 *훨씬 위* — 단일 plan 의 4 spline 변형 그 어떤 것도 등속 외삽 baseline 을 넘지 못함을 *결정적으로* 박제.

---

## §4. S003 의 axis 별 chosen 분해

5 outer fold 모두 동일:

| axis | chosen (window, bc_type) |
|---|---|
| x | (5, natural) |
| y | (5, natural) |
| z | (4, natural) |

12-cell grid 중 clamped 는 어떤 axis/fold 에서도 1위가 아님. natural BC 가 dominate. 짧은 window (4~5) 가 11pt 보다 우월 — oldtimesteps 노이즈 누적이 외삽 정확도를 해친다는 plan-001 §1.3 의 핵심 통찰을 cspline 분기에서 재확인.

---

## §5. S004 의 axis 별 chosen + s_grid 곡선

5 outer fold 의 chosen s_per_axis = [1e-4, 1e-4, 1e-4] 4 개, [1e-3, 1e-4, 1e-4] 1 개. 즉 axis-x 가 한 fold 에서만 1e-3 으로 약간 더 smoothing. Final full-train re-tune = [1e-4, 1e-4, 1e-4] 3 axes 동일.

s=0 (interpolation) 은 inner CV 에서 axis MAE 가 약간 더 큼 (smoothing 의 이득 있음). 하지만 그 이득이 *외삽 metric 까지 carry over 안 됨* — H3 refuted.

s_grid axis MAE 곡선은 `runs/baseline/S004_smoothing-spline-tuned/summary.json` 의 `full_train_grid_errors` 에 박제.

---

## §6. H1 / H2 / H3 verdict

- **H1**: 11pt natural / not-a-knot 보간 → polyfit-11pt 급 (0.03~0.05) — **CV 에서 confirmed**. natural 0.0174 (0.02 미만으로 약간 더 좋음), not-a-knot 0.0537 (예측 범위 안). LB verdict 는 점수 회수 후 보강.
- **H2**: window × BC grid 가 B001 근접 — **partially refuted**. CV 0.0174 영역 (S001 동급) 에서 멈춤. clamped 가 한 번도 안 뽑힘. small-window 의 이득은 있으나 등속 polyfit 의 floor 를 넘지 못함.
- **H3**: smoothing 이 노이즈 흡수로 B001 위협 — **CV 에서 refuted**. CV 0.0332 — fit-영역 axis MAE 의 이득이 외삽 영역 mean_eucl 로 transfer 안 됨.

---

## §7. CV ↔ LB 상관 분석 (TBD — 점수 회수 후)

5 점 (B001 + S001~S004) 의 (cv_mean_eucl, lb_score) 산점 + Spearman ρ 는 4 LB 점수 회수 후 본 섹션에 추가. 현재 회수: B001 LB=0.60 (plan-001 frontmatter). 4 점수 도착 시 본 섹션 + frontmatter `lb_scores` dict 동시 갱신.

가설 (점수 도착 전 prior):
- CV winner = S003 (0.01740). LB winner 후보:
  - 비례 시나리오: S003 도 LB 1위 (CV ≈ LB proxy).
  - 반전 시나리오: S004 (smoothing) 가 LB 에서 회복 — fit 영역 noise smoothing 이 LB hit_rate 의 tail behavior (큰 오차 sample 비율) 에 도움.
  - cv-LB 상관 약함: S001~S004 LB 모두 비슷 — 4 변형이 외삽 prior 만 다르고 LB 측정 metric (hit-rate 반경 비공개) 에는 유의 신호 없음.
- §N+3 #8 caveat: LB 차이 0.005 영역 미만은 noise.

---

## §8. submission 결과

4 LB 제출 모두 isSubmitted=True (api Success). lb_log @ `analysis/plan-002/lb_log.md`.

| order | exp_id | submitted_at (KST) | api response |
|---|---|---|---|
| 1 | S004 | 2026-05-10T05:09 | Success |
| 2 | S003 | 2026-05-10T05:11 | Success |
| 3 | S001 | 2026-05-10T05:11 | Success |
| 4 | S002 | 2026-05-10T05:12 | Success |

Budget: 4/5 일일. 1 슬롯 contingency. Carry-over: 점수 회수만 (dacon_submit_api 가 post-only, 점수 fetch 미지원 → user 가 dacon.io 에서 4 점수 확인 후 lb_log + frontmatter 갱신).

---

## §9. 다음 plan 후보 (enumeration only)

1. **Kalman / Savitzky-Golay 입력 평활 → polyfit**: smoothing spline 이 "post-hoc 평활" 인 점이 H3 refute 의 원인 추정. 입력 측 평활 + 작은-window polyfit 이 다른 inductive bias.
2. **Velocity model**: t=0 에서의 instantaneous velocity 추정 + 등속 외삽 (B001 의 일반화). 6/8/10-pt polyfit derivative 와 비교.
3. **Ensemble (B001, S003, S004)**: CV-mean_eucl 측 약간의 다양성 + LB hit_rate 측 다른 tail. 단순 평균 vs CV-weighted.
4. **Neural seq2seq (LSTM / 1D-Transformer)**: 11pt × 3-axis 입력으로 +80 ms 출력. small data — 강한 augmentation + light model 필요.
5. **Per-axis combination of B001 + smoothing axis-wise**: B004 의 generalization. axis 별 best of {polyfit, cspline, smoothing}.
6. **Hit-radius probing**: 1 LB 슬롯 사용해 hit_rate 반경/분모 추정. 별도 plan 권장.

우선순위 결정은 local 권한 — CV-LB 상관 박제 (§7) 후 신중히.
