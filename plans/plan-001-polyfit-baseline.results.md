---
plan_id: 001
finished_at: 2026-05-10 (Asia/Seoul)
status: all_complete
exp_ids_completed:
  - B001_linear-2pt
  - B002_linear-3pt
  - B003_quad-3pt
  - B004_per-axis-grid
exp_ids_skipped: []
best_exp_id: B001_linear-2pt
submission_path: runs/baseline/B001_linear-2pt/submission.csv
lb_score_public: 0.60
lb_metric: hit_rate_at_1cm
lb_submitted_at: 2026-05-10 (Asia/Seoul)
---

# plan-001 results — polyfit baseline

## 종합 표

| exp_id | window | degree | cv_mean_eucl | cv_std | per_axis_mae [x, y, z] | hit@0.05 | hit@0.10 | hit@0.20 | duration_sec |
|---|---|---|---|---|---|---|---|---|---|
| B001_linear-2pt | 2 | 1 | **0.01294** | 0.00058 | [0.0070, 0.0071, 0.0050] | 0.957 | 0.992 | 1.000 | 0.80 |
| B002_linear-3pt | 3 | 1 | 0.01576 | 0.00064 | [0.0087, 0.0087, 0.0059] | 0.939 | 0.989 | 1.000 | 0.78 |
| B003_quad-3pt | 3 | 2 | 0.02017 | 0.00103 | [0.0105, 0.0110, 0.0085] | 0.918 | 0.976 | 0.998 | 0.85 |
| B004_per-axis-grid | (2,2,2) | (1,1,1) | 0.01294 | 0.00058 | [0.0070, 0.0071, 0.0050] | 0.957 | 0.992 | 1.000 | 3.29 |

## per-experiment

### B001_linear-2pt — `pred = last + 2·(last − prev)`

- 상태: complete
- 시작/완료: 2026-05-10T01:04:09 KST → +0.80 s
- 핵심 metric: cv_mean_eucl = 0.01294 ± 0.00058
- best artifact: `runs/baseline/B001_linear-2pt/`
- baseline diff: 자기 자신 (floor)
- 특이사항: 합격 기준 B (mean_eucl ∈ [0.010, 0.020]) 통과. EDA 의 n=2000 spot check (0.0129) 와 5-fold 전체 평균이 4 자리에서 일치 — loader/eval 정합 확인.

### B002_linear-3pt — w=3, d=1 (window 확대)

- 상태: complete
- 시작/완료: 2026-05-10T01:04:38 KST → +0.78 s
- 핵심 metric: cv_mean_eucl = 0.01576 ± 0.00064
- baseline diff vs B001: window 2 → 3 (degree 1 유지). cv_mean_eucl +21.8 %.
- 특이사항: H1 의 "window 확대 = 노이즈 평균 효과" 가설 부분 기각. 1 개의 추가 과거 점이 노이즈 averaging 보다 staleness penalty 가 커서 오히려 전체 오차가 증가. 5 fold 모두에서 B001 대비 일관되게 악화 (per-fold gap 0.0028~0.0029).

### B003_quad-3pt — w=3, d=2 (degree 확대)

- 상태: complete
- 시작/완료: 2026-05-10T01:04:39 KST → +0.85 s
- 핵심 metric: cv_mean_eucl = 0.02017 ± 0.00103
- baseline diff vs B002: degree 1 → 2 (window 3 유지). cv_mean_eucl +28.0 % vs B002, +55.9 % vs B001.
- 특이사항: H1 완전 기각 (적어도 3-point 윈도우에서). 3 점 정확 적합 quadratic 은 가속도를 추정하지만 입력 노이즈가 그 항을 dominant 하게 만들어 외삽 시 오차가 누적. cv_std 도 0.00103 으로 가장 큼 — 노이즈 민감도 증가의 직접 증거.

### B004_per-axis-grid — per-axis (w, d) tuning

- 상태: complete
- 시작/완료: 2026-05-10T01:06:30 KST → +3.29 s
- 핵심 metric: cv_mean_eucl = 0.01294 ± 0.00058
- baseline diff vs B001: per_axis = tune (12-cell grid × 3 axes). 5 fold 전부 + 전체 train tune 모두 (w=2, d=1) 선택 → B001 과 비트-단위 동일 결과.
- 특이사항: H3 (per-axis 동역학 차이) 기각. 모든 axis 의 (w, d) → MAE 정렬에서 (2,1) 이 1 위, (3,1) 이 2 위, gap 22-25 %. 노이즈 수준 + 40 ms 샘플링이 짧은 windowmin window 를 axis 무관하게 우세하게 만든다.

## metric 분석

### per-axis MAE 비교 (cv 평균)

| metric | x | y | z |
|---|---|---|---|
| B001 | 0.0070 | 0.0071 | 0.0050 |
| 라벨 std (전체 10000) | 1.185 | 0.746 | 0.587 |
| MAE / std | 0.0059 | 0.0095 | 0.0086 |

- z 축 MAE 가 절대치로는 가장 작지만 라벨 std 도 가장 작아 정규화 후엔 y 와 비슷.
- x 축이 라벨 std 대비 가장 잘 예측됨 (forward 방향 — 모기의 평균 비행 방향과 일치 가설).

### Hit Rate 분포 (B001)

- 0.05 m: 95.7 % — 거의 모든 sample 이 5 cm 안에 들어옴
- 0.10 m: 99.2 %
- 0.20 m: 100.0 %
- 0.50 m: 100.0 %

→ 실제 laser 유효 반경이 5 cm 이상이면 hit-rate 차이가 크지 않을 가능성. 반경이 1~3 cm 라면 tail 관리가 결정적.

## 합격 기준 결과

| 조건 | 정의 | 결과 |
|---|---|---|
| A. 인프라 정상 | pytest green, B001 실행 가능 | ✅ 20 tests passed |
| B. EDA 재현 | B001 mean_eucl ∈ [0.010, 0.020] | ✅ 0.01294 |
| C. floor 무회귀 | best ≤ 0.013 | ✅ 0.01294 |
| D. 제출 가능 | submission.csv 스키마 100 % 일치 | ✅ rows=10000, cols=[id,x,y,z], NaN=0 |

## 리더보드 (LB) 결과

| 항목 | 값 |
|---|---|
| submission | `runs/baseline/B001_linear-2pt/submission.csv` |
| 제출일 | 2026-05-10 (KST) |
| **public LB score** | **0.60** |
| LB metric | hit rate @ 1cm 반경 (PB_0.6822 코드공유 노트북의 `R_HIT=0.01` 와 일관) |
| 사전 추정 (lognormal 외삽) | 0.55 ~ 0.60 → 실측 상단 일치 |
| 참고: PB_0.6822 노트북 | 0.6822 (Frenet+Attn-GRU+잔차MLP). gap = +0.082 |

→ closed-form floor 만으로 공유 노트북 대비 88% 수준. 신경망/물리 기반 정교화의 marginal gain 영역이 좁다는 강한 신호 — 이후 plan 들은 hit@1cm 기준선 0.60 을 명시적으로 넘는 design 이 필요.

## 다음 plan 후보 (enumeration only — 우선순위 미정)

1. **가중 polyfit / EWMA**: window 를 늘리되 최근 점에 지수 감쇠 가중. 노이즈 평균과 staleness 의 trade-off 를 hyperparameter 로 해소.
2. **노이즈 필터링 전처리**: Savitzky-Golay 또는 Kalman 으로 입력 11 점을 smooth 후 작은 window polyfit. 노이즈 floor 자체를 낮춤.
3. **Neural seq2one**: 11 → 3 작은 LSTM/Transformer/MLP. 학습 데이터 충분 (n=10000), domain generalization 필요.
4. **Per-sample variance-aware**: 입력 시퀀스의 분산이 높은 sample 에 대해서만 작은 window 사용, 안정적 sample 에는 큰 window. 메타 모델 또는 rule-based.
5. **Hit-rate aware loss**: scoring 이 hit-rate 라면 mean_eucl 학습이 sub-optimal. tail (p95) 을 직접 줄이는 quantile loss 검토.
6. **Domain 추정**: 입력 시퀀스 통계 (분산, 평균 속도) 로 implicit scene clustering → mixture of polyfit/model.
7. **ensemble**: B001 (closed-form) + neural model 의 단순 평균 또는 stacking.

## 한계 / caveats

- 5-fold CV + 1 회 LB probe (B001 제출) — 추가 probing 없이 closed-form floor 확립.
- Hit-rate 반경: 노트북 공유작 기준 1cm 으로 추정 (`R_HIT=0.01`). 공식 비공개라 실제 반경과 일치 여부는 추가 probe 필요.
- B004 per-axis tuning 의 결과가 B001 과 동일한 것은 *현 grid 한정* — 가중 polyfit 같은 다른 hyperparameter 축은 미탐색.

## 참조

- plan: `plans/plan-001-polyfit-baseline.md`
- detailed analysis: `analysis/plan-001/results.md`
- run dirs: `runs/baseline/B00{1..4}_*/`
- registry: `registry.csv` (4 rows for plan-001)
- best submission: `runs/baseline/B001_linear-2pt/submission.csv`
