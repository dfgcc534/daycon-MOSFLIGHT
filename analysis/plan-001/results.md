# plan-001 — Detailed Analysis

> Companion to `plans/plan-001-polyfit-baseline.results.md`. 본 문서는 plan-specific 분석 (전 commit 의 numerical 결과 + 그래프-수준 비교 + 다음 plan 의 가설 seed) 을 보관한다.

## 1. Scoreboard (정렬 by cv_mean_eucl)

| rank | exp_id | cv_mean_eucl | gap vs B001 | window | degree | per_axis | dur (s) |
|---|---|---|---|---|---|---|---|
| 1 (tie) | B001_linear-2pt | 0.012941 | 0.0 % | 2 | 1 | none | 0.80 |
| 1 (tie) | B004_per-axis-grid | 0.012941 | +0.0 % | 2 | 1 | tune→all (2,1) | 3.29 |
| 3 | B002_linear-3pt | 0.015759 | +21.8 % | 3 | 1 | none | 0.78 |
| 4 | B003_quad-3pt | 0.020174 | +55.9 % | 3 | 2 | none | 0.85 |

→ 단순 등속 외삽 (`pred = last + 2·(last − prev)`) 이 closed-form 영역의 명확한 winner. degree 와 window 를 늘리는 시도는 모두 악화로 귀결.

## 2. fold-by-fold 안정성

| fold | B001 mean_eucl | B002 | B003 | B004 |
|---|---|---|---|---|
| 0 | 0.01371 | 0.01659 | 0.02116 | 0.01371 |
| 1 | 0.01201 | 0.01474 | 0.01894 | 0.01201 |
| 2 | 0.01259 | 0.01540 | 0.01890 | 0.01259 |
| 3 | 0.01313 | 0.01613 | 0.02096 | 0.01313 |
| 4 | 0.01326 | 0.01596 | 0.02092 | 0.01326 |

- 모든 fold 에서 ranking (B001=B004) < B002 < B003 일관. exp 간 gap (≥ 0.003) 이 cv_std 의 5 배 이상 → 통계적으로 의미 있는 ordering.

## 3. B004 grid full-train MAE (per axis)

axis 0 (x):

| (w, d) | MAE |
|---|---|
| **(2, 1)** | **0.00705** ← chosen |
| (3, 1) | 0.00871 |
| (3, 2) | 0.01052 |
| (5, 2) | 0.01121 |
| ... 나머지 12 셀 | 모두 ≥ 0.00871 |

axis 1 (y):

| (w, d) | MAE |
|---|---|
| **(2, 1)** | **0.00710** ← chosen |
| (3, 1) | 0.00871 |
| (3, 2) | 0.01101 |
| (5, 2) | 0.01101 |

axis 2 (z):

| (w, d) | MAE |
|---|---|
| **(2, 1)** | **0.00504** ← chosen |
| (3, 1) | 0.00593 |
| (5, 2) | 0.00760 |
| (5, 1) | 0.00816 |

→ 세 axis 모두 (2,1) 압도적 1 위. 2 위와의 gap 22-25 %. 다른 grid 점은 일체 경쟁 안 됨.

## 4. 가설 검증 결과

| 가설 | 예측 | 결과 | 판정 |
|---|---|---|---|
| H1 | degree 2/3 polyfit 이 degree 1 대비 30 % 이상 개선 | 모두 악화 (B003 +56 %, B002 +22 %) | **기각** |
| H2 | 11 points 가 degree 3 fit 에 충분 (overfit 결정적이지 않음) | window=3, deg=2 만 봐도 이미 노이즈에 휘둘림 (B003) | **기각** |
| H3 | axis 별 (w, d) 차이가 추가 이득 | 모든 axis 가 동일 (w=2, d=1) 선택 | **기각** |

3 가설 모두 기각이지만 실험은 성공 — *실패한 가설 자체* 가 신호. 노이즈 floor 가 가속도 정보보다 dominant 하다는 사실을 양적으로 박제.

## 5. 노이즈 vs 신호 estimation (informal)

B001 의 per-axis MAE ≈ [0.007, 0.007, 0.005]. 등속 외삽이 정확히 맞다면 잔차 = 노이즈. 따라서 입력 노이즈 σ 의 합리적 추정치 ≈ 0.005-0.007 m (단축당). EDA 의 input/label std (각 ~ 0.6-1.2 m) 대비 SNR ≈ 100-200 → 신호는 강하지만 가속도 항 (≤ 0.001-0.01 m / step² 추정) 과 동일 수위 → 가속도 추정이 노이즈에 휘둘림.

## 6. submission 분포 (B001)

- rows: 10000
- columns: id, x, y, z (sample_submission 스키마와 100 % 일치)
- NaN: 0
- range:
  - x: [0.463, 6.405] (test input X[-1] 분포와 거의 일치 — 등속 외삽이라 자연스러움)
  - y: [-2.585, 2.531]
  - z: [-1.611, 2.475]

## 7. 다음 plan 후보 — *우선순위 추천 (정성적)*

local 측이 결정. 본 server 의 추천 ordering:

1. **§7.1 가중 polyfit / EWMA**: 최저 risk, 빠른 검증 가능. window=11, exp-decay weight 1 개 hyperparameter 만 추가. B001 과 비교 fair.
2. **§7.2 입력 smoothing**: Savitzky-Golay (3-point window=5, polyorder=2) 등을 입력에 적용 후 B001 동일 외삽. 노이즈 floor 를 직접 낮춤.
3. **§7.5 Hit-rate aware**: 반경 정보가 leaderboard probing 으로 들어오면 그 시점 합류.
4. **§7.3 Neural seq2one**: 가성비는 closed-form 보다 낮을 가능성 (notes/competition-overview §7.1 의 prior). 그러나 분포 tail 처리에는 유리할 수 있음.
5. **§7.7 ensemble**: 충분한 다양성이 확보된 다음에만.

closed-form (§7.1, §7.2) → neural (§7.3) → ensemble (§7.7) 순으로 진행하는 것이 자원 대비 정보 획득 효율이 높을 가능성.

## 8. Open questions for local

- **laser 반경 추정**: ✅ 1 회 probe 완료 (B001, LB 0.60). 노트북 `R_HIT=0.01` 과 정합. 추가 probe (예: 의도적 perturbation 으로 반경 분해능 측정) 는 효용 대비 cost 검토 필요.
- **외부 데이터 활용 의향**: notes §6.2 에 ✅ 허용 — small-target trajectory 공개 데이터 (드론, 곤충) 의 noise floor 가 비슷하면 transfer learning seed 가능.
- **domain (scene) 추정**: 시퀀스 통계로 mixture 분류 시도할지.

## 10. LB calibration (B001 제출 결과)

| 지표 | 값 |
|---|---|
| OOF cv_mean_eucl | 0.01294 m |
| OOF median_eucl | ~0.0086 m |
| OOF hit@5cm | 0.957 |
| **public LB** | **0.60** |
| 추정 metric | hit rate @ 1cm |
| 추정 vs 실측 | lognormal 추정 0.55-0.60 → 실측 0.60 |

→ OOF 통계 → LB 환산식 (informal): `LB ≈ Φ((ln(0.01) − ln(median_eucl)) / σ_log)`. 이후 plan 들은 OOF mean_eucl 만 알아도 LB 추정 가능.

→ B001 의 hit@5cm = 0.957 vs LB hit@1cm = 0.60. 즉 5cm 반경은 거의 saturate, 1cm 가 진짜 분해능 영역. 다음 plan 의 ROI 는 **분포 좌측 (median 이하 영역) 을 1cm 안으로 더 밀어넣기** 에 집중해야 함 — outlier (p95 > 4cm) 잡는 것은 LB 영향이 작음.

## 9. references

- plan: `plans/plan-001-polyfit-baseline.md`
- handoff results: `plans/plan-001-polyfit-baseline.results.md`
- runs: `runs/baseline/B00{1..4}_*/`
- registry: `registry.csv` rows 1-4
- competition spec: `notes/competition-overview.md`
- WORKFLOW: `WORKFLOW.md` §4-§12
