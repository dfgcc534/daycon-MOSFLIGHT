---
plan_id: 001
version: 1
date: 2026-05-10 (Asia/Seoul)
status: draft
based_on: []
scope: full-stack (인프라 + closed-form baseline + submission)
exp_ids:
  - B001_linear-2pt
  - B002_linear-3pt
  - B003_quad-3pt
  - B004_per-axis-grid
---

# plan-001 v1 — Polynomial Fit Baseline (closed-form, no learning)

## §0. 한 줄 목적

> **학습 없이 마지막 몇 시점만 polyfit으로 외삽해 +80ms 시점 (x,y,z)를 예측하는 closed-form baseline을 확립하고, 후속 plan(neural model 등)이 반드시 넘어야 할 floor를 박제한다.**

---

## §0.5 Quick Reference

### 합격 기준 (G-gate sequence)

- 모든 4 exp 의 **5-fold CV mean Euclidean distance (m) + per-axis MAE + hit rate @ {0.05, 0.10, 0.20, 0.50} m** 가 registry + run dir 에 기록.
- **B001 mean_eucl ∈ [0.010, 0.020]** (re-EDA 재현).
- **best of {B001..B004} mean_eucl ≤ 0.013** (현 floor 비초과 시 severe).
- 최우수 exp 로 test 10,000개 예측 → `runs/baseline/{best}/submission.csv` (sample_submission 스키마 동일).

### G-gates

- G0: STAGE 0 인프라 commit chain 완료                    [TODO]
- G1: STAGE 1 B001 (linear-2pt) 결과 확보 + EDA 재현        [TODO]
- G2: STAGE 2 B002, B003 결과 확보                          [TODO]
- G3: STAGE 3 B004 per-axis grid 결과 확보                  [TODO]
- G_final: submission.csv + plan-001 results.md 작성       [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | code | `src/io.py` — `load_sample(id, split)`, `load_labels()`, `kfold_split(seed=42, k=5)`. spec @ §4 | [TODO] |
| c2 | code | `src/eval.py` — `eucl_per_sample`, `mean_eucl`, `per_axis_mae`, `hit_rate(radii)`. spec @ §3.3 | [TODO] |
| c3 | code | `src/baselines/window_polyfit.py` — `predict(X, window, degree, t_target=80)` vectorized. spec @ §5 | [TODO] |
| c4 | test | `tests/test_io.py` + `tests/test_eval.py` + `tests/test_window_polyfit.py` (synthetic). spec @ §4, §3.3, §5 | [TODO] |
| G0 | gate | `pytest tests/` green, `src.eval.mean_eucl` 와 `src.baselines.window_polyfit.predict` import OK | [TODO] |
| c5 | exp B001 | `configs/baseline/B001_linear-2pt.yaml` + run (5-fold) + `runs/baseline/B001_linear-2pt/{summary,history,run.log,config.snapshot}` + registry. spec @ §6 | [TODO] |
| G1 | gate | B001 mean_eucl ∈ [0.010, 0.020]; 벗어나면 severe `eda_mismatch` | [TODO] |
| c6 | exp B002 | `configs/baseline/B002_linear-3pt.yaml` + run + registry. spec @ §6 | [TODO] |
| c7 | exp B003 | `configs/baseline/B003_quad-3pt.yaml` + run + registry. spec @ §6 | [TODO] |
| G2 | gate | B002/B003 결과 기록 완료 | [TODO] |
| c8 | code | `src/baselines/window_polyfit.py` 에 `tune_per_axis(X, y, grid, k=5)` 추가. spec @ §6.4 | [TODO] |
| c9 | exp B004 | `configs/baseline/B004_per-axis-grid.yaml` + run + registry. spec @ §6 | [TODO] |
| G3 | gate | B004 결과 기록 완료, best of {B001..B004} mean_eucl ≤ 0.013 | [TODO] |
| c10 | sub | best exp_id 자동 결정 (registry mean_eucl argmin) → 전체 train 사용 (closed-form: 추가 fit 불필요) → test 10k 예측 → `runs/baseline/{best}/submission.csv`. 스키마 검증. spec @ §7 | [TODO] |
| c11 | docs | `analysis/plan-001/results.md` + `plans/plan-001-polyfit-baseline.results.md`. spec @ §N+2 | [TODO] |
| G_final | gate | 위 모두 완료 + §0.5 [TODO]→[DONE] sync (이게 §12.6 blacklist 의 유일한 예외) | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `eda_mismatch`: B001 mean_eucl 이 [0.010, 0.020] 밖 → 데이터/loader/평가 함수 버그 의심.
- `floor_regress`: best of {B001..B004} mean_eucl > 0.013 → re-EDA 결과 회귀, 코드 버그 의심.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가)

- whitelist 추가: `analysis/plan-001/**`, `runs/baseline/B00*/**`, `configs/baseline/B00*.yaml`
- blacklist 추가: (없음)

### Decision-note 사용 예

- `decision-note: spec-default — 5-fold seed=42, fold split = label.id 순서 기반 stride 5`
- `decision-note: spec-default — hit-radius 후보 {0.05, 0.10, 0.20, 0.50} m (실제 반경 비공개라 분포 보고)`
- `decision-note: data-partial — n/a (모든 sample 11 row 보장됨, EDA 검증)`

---

## §1. 배경

### 1.1 대회 사양 (notes/competition-overview.md)

- 입력: 11 timestep, t ∈ {-400, -360, ..., 0} ms (40ms 균등), per-step (x, y, z) m.
- 출력: t = **+80 ms** 의 (x, y, z) 1점.
- metric: 1차 = 3D Euclidean 거리, 최종 = laser 유효 반경 내 **Hit Rate** (반경 비공개).

### 1.2 EDA 결과 (2026-05-09 conversation, n=10,000)

- 모든 sample 정확히 11 row, NaN 0, ID 정합 100%.
- train/test 분포 거의 동일 (각 axis mean/std 차이 ≤ 0.1) → covariate shift 무시.
- target 분포는 input 의 last(0ms) 분포와 거의 일치 (label - last_input 의 std per axis ≈ 0.025~0.035).

### 1.3 Re-EDA (2026-05-10, n=2000) — Plan 의 결정적 근거

target = +80ms 로 정정 후 측정한 naive baselines:

| 방법 | mean_eucl | per-axis MAE | 비고 |
|---|---|---|---|
| `pred = last(0ms)` | 0.0514 | [0.031, 0.026, 0.018] | constant pred |
| `last + 1·(last-prev)` (잘못된 horizon, 참고용) | 0.0283 | [0.017, 0.015, 0.010] | t=+40ms 외삽 |
| **`last + 2·(last-prev)`** | **0.0129** | **[0.007, 0.007, 0.005]** | t=+80ms 직선 외삽, **window=2, deg=1** |
| polyfit deg=1 on 11 pts @ +80 | 0.0434 | [0.025, 0.024, 0.015] | 글로벌 fit이 노이즈에 휘둘림 |
| polyfit deg=2 on 11 pts @ +80 | 0.0331 | [0.019, 0.018, 0.012] | 동상 |
| polyfit deg=3 on 11 pts @ +80 | 0.0357 | [0.020, 0.020, 0.013] | 동상 |

**핵심 통찰**: 11점 전체 polyfit은 2점 등속 외삽보다 모두 나쁨. 오래된 시점의 노이즈가 fit을 dominate 함. 따라서 최적은 **최근 소수 시점 + 낮은 degree** 영역에 있을 가능성이 매우 높다.

---

## §2. Scope

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 모델 | per-axis (window, degree) closed-form polyfit |
| 학습 | 없음 (CV 는 hyperparameter selection 용 only) |
| hyperparameter | window ∈ {2, 3, 5, 7, 11}, degree ∈ {1, 2, 3}, valid 조합만 (window > degree) |
| target time | +80 ms (스펙 고정) |
| primary dev metric | mean 3D Euclidean distance (m) |
| 보조 metric | per-axis MAE, hit rate @ {0.05, 0.10, 0.20, 0.50} m |
| CV | 5-fold, seed=42, fold = `id` 순서 기반 stride-5 deterministic split |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| neural model (LSTM/Transformer/MLP) | 별도 plan. 본 plan 은 closed-form floor 만 |
| 가중 polyfit (exponential decay 등) | 별도 plan. uniform weight 만 |
| Kalman/Savitzky-Golay 입력 필터 | 별도 plan |
| target horizon ≠ +80 ms | 스펙 위반 |
| ensemble | baseline 단일 모델만 |
| hit-rate 직접 학습 | 학습 자체 안 함 |
| hit-radius 추정 (leaderboard probing) | 5회/일 제출 한도 절약, 별도 plan |
| test 데이터를 fit 에 사용 | 대회 룰 위반 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- k=5, seed=42.
- 결정론적 stride: fold f 에는 `label.id` 알파벳-순 정렬 후 인덱스 i mod 5 == f 인 sample.
- 구현: `src/io.py:kfold_split(ids, k=5, seed=42)`.
- 재현: 동일 seed → 동일 split 보장 (assertion test 추가).

### §3.2 합격 기준

| 조건 | 정의 |
|---|---|
| A. 인프라 정상 | `pytest tests/` green, B001 실행 가능 |
| B. EDA 재현 | B001 mean_eucl ∈ [0.010, 0.020] (위반 시 `eda_mismatch` severe) |
| C. floor 무회귀 | best of {B001..B004} mean_eucl ≤ 0.013 (위반 시 `floor_regress` severe) |
| D. 제출 가능 | `runs/baseline/{best}/submission.csv` 가 sample_submission 스키마와 100% 일치 (id 순서, 컬럼명, NaN 0) |

### §3.3 평가 점수 / 집계

- per fold metric: mean_eucl over fold's val set, per-axis MAE, hit_rate(r) for r ∈ {0.05, 0.10, 0.20, 0.50}
- per exp metric: 5 fold mean ± std
- exp 비교: mean_eucl 의 5-fold mean (median 아님 — small CV)
- best exp 선택 = mean_eucl argmin

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 산출 파일

- `src/__init__.py` (빈 파일)
- `src/io.py`:
  - `load_sample(sample_id: str, split: str = "train") -> np.ndarray (11, 3)` — (x,y,z) only, t는 spec 고정값
  - `load_all_samples(split) -> (ids: list[str], X: np.ndarray (n, 11, 3))`
  - `load_labels() -> (ids, y: np.ndarray (n, 3))`
  - `kfold_split(ids, k=5, seed=42) -> list[(train_idx, val_idx)]`
- `src/eval.py`:
  - `eucl(pred, true) -> np.ndarray (n,)` — per sample 3D euclidean
  - `mean_eucl`, `per_axis_mae`, `hit_rate(pred, true, r) -> float`
  - `summarize(pred, true, radii=(0.05, 0.10, 0.20, 0.50)) -> dict` — registry summary 키와 1:1
- `tests/test_io.py`, `tests/test_eval.py`, `tests/test_window_polyfit.py`

### §4.2 산출물 schema (run dir summary.json — registry 와 1:1)

```json
{
  "exp_id": "B00X_...",
  "n_train": 10000,
  "n_val_per_fold": 2000,
  "k": 5,
  "fold_metrics": [{"fold": 0, "mean_eucl": ..., "per_axis_mae": [..,..,..], "hit_rate": {...}}, ...],
  "cv_mean_eucl": ...,
  "cv_std_eucl": ...,
  "cv_per_axis_mae": [...],
  "cv_hit_rate": {"0.05": ..., "0.10": ..., "0.20": ..., "0.50": ...},
  "config": {"window": ..., "degree": ..., "t_target": 80, "per_axis": {...} | null}
}
```

### §4.3 종료 조건 (G0)

- `pytest -q tests/` exit 0
- `python -c "from src.eval import mean_eucl; from src.baselines.window_polyfit import predict; print('ok')"` 성공

---

## §5. STAGE 1 — B001 floor (G1)

### §5.1 B001_linear-2pt

| 항목 | 값 |
|---|---|
| type | baseline |
| baseline_id | (자기 자신, floor) |
| 단일 변경 변수 | N/A — floor |
| method | per-axis: `pred = X[:,-1,:] + 2*(X[:,-1,:] - X[:,-2,:])` (= window=2, degree=1, t_target=80) |
| config | `configs/baseline/B001_linear-2pt.yaml` (`{window: 2, degree: 1, t_target: 80, per_axis: null}`) |
| 기대 runtime | < 30 s (전부 vectorized numpy) |
| 성공 기준 | mean_eucl ∈ [0.010, 0.020] (re-EDA 재현) |
| 실패 시 | `eda_mismatch` severe — code/eval/loader 버그 |

### §5.2 G1 종료 조건

- `runs/baseline/B001_linear-2pt/summary.json` 작성, registry 1행 추가.
- `cv_mean_eucl` ∈ [0.010, 0.020].

---

## §6. STAGE 2~3 — Polynomial 변형 (G2, G3)

### §6.1 B002_linear-3pt — window 확대

| 항목 | 값 |
|---|---|
| baseline_id | B001 |
| 단일 변경 변수 | window 2 → 3 (degree 1 유지) |
| method | window=3, degree=1 polyfit @ t=80 |
| config | `configs/baseline/B002_linear-3pt.yaml` |
| 기대 runtime | < 30 s |
| 성공 기준 | 결과 기록 완료 (개선 보장 X — 가설 검증 자체가 목적) |
| 가설 | 노이즈 평균 효과 vs 시간 staleness — 어느 쪽이 우세한지 |

### §6.2 B003_quad-3pt — degree 확대

| 항목 | 값 |
|---|---|
| baseline_id | B002 |
| 단일 변경 변수 | degree 1 → 2 (window 3 유지) |
| method | window=3, degree=2 polyfit @ t=80 (가속도 항 포함) |
| config | `configs/baseline/B003_quad-3pt.yaml` |
| 기대 runtime | < 30 s |
| 성공 기준 | 결과 기록 완료 |
| 가설 | 가속도 정보가 +80ms 외삽에서 의미 있는 이득을 주는가 |

### §6.3 G2 종료 조건

- B002, B003 의 summary.json + registry 행 모두 존재.

### §6.4 B004_per-axis-grid — per-axis grid tuning

| 항목 | 값 |
|---|---|
| baseline_id | best of {B001, B002, B003} |
| 단일 변경 변수 | global (w, d) → per-axis (w, d) |
| grid | (window, degree) ∈ {(2,1), (3,1), (3,2), (5,1), (5,2), (5,3), (7,1), (7,2), (7,3), (11,1), (11,2), (11,3)} (window > degree 보장) |
| selection | per-axis: 5-fold CV val mean_eucl_axis (= per-axis MAE 가 아니라 per-axis squared-error mean) argmin |
| config | `configs/baseline/B004_per-axis-grid.yaml` (`{grid: ..., per_axis: true}`) — chosen (w, d) per axis 는 run dir summary 에 기록 |
| 기대 runtime | < 5 min (12 cells × 3 axes × 5 fold ≈ 180 fit 묶음, 모두 vectorized) |
| 성공 기준 | mean_eucl ≤ best of {B001, B002, B003} |
| 가설 | axis 마다 동역학 차이 (z 의 std 가 가장 작음) 가 hyperparameter 선택을 통해 reflect 됨 |

### §6.5 G3 종료 조건

- B004 summary + registry 행 존재, best of {B001..B004} mean_eucl ≤ 0.013.

---

## §7. STAGE 4 — Submission + Results (G_final)

### §7.1 Submission 절차 (c10)

1. registry 에서 baseline 4 행 중 cv_mean_eucl argmin → `best_exp_id`.
2. best 의 config 로 test 전체 10,000 sample 예측 (closed-form 이라 train fit 불필요; per-axis 케이스는 stored (w, d) 사용).
3. `runs/baseline/{best_exp_id}/submission.csv` 작성, 컬럼 = `id, x, y, z`, 행 순서는 `sample_submission.csv` 와 동일.
4. 스키마 assert: rows == 10000, columns 일치, NaN 0, dtype float64, id set 일치.

### §7.2 Results.md 산출 (c11)

- `analysis/plan-001/results.md`: exp별 metric 표, axis별 분해, B004 의 per-axis 선택 보고, hit_rate@radius 분포, baseline 들의 산점도(예측 vs 실제), 다음 plan 후보 나열.
- `plans/plan-001-polyfit-baseline.results.md`: WORKFLOW.md §6 frontmatter + 각 exp_id 의 (status, started_at, duration, 핵심 metric, best path, baseline diff, 특이사항).

### §7.3 G_final 종료 조건

- 4 exp summary + registry 4 baseline 행 + submission.csv 1개 + results.md 2개 모두 commit.
- §0.5 의 모든 [TODO] → [DONE] 마킹 (commit hash 포함, §12.6 blacklist 예외).

---

## §N+1. 작업량 회계

| 단위 | 수 |
|---|---|
| code commit (c1~c4) | 4 |
| exp commit (c5,c6,c7,c9) | 4 (각 fit + eval + registry append) |
| code 추가 commit (c8) | 1 |
| docs commit (c10,c11) | 2 |
| **총 commit** | **11** |
| G-gate | 5 (G0, G1, G2, G3, G_final) |
| 총 fit 호출 | B001~B003 = 5 fold × 3 = 15. B004 = 5 fold × 12 cells × 3 axes = 180. 모두 vectorized → 총 wall time < 10 min |

---

## §N+2. results.md 필수 항목

| 항목 | 내용 |
|---|---|
| frontmatter | `plan_id, finished_at (KST), status, exp_ids_completed, exp_ids_skipped` |
| 본문 per exp | 상태, started_at, duration, cv_mean_eucl±std, per-axis MAE, hit_rate@4 radii, best run dir path, baseline 대비 config diff, 특이사항 |
| 종합 표 | 4 exp × (mean_eucl, per_axis_mae, hit_rate@0.10, runtime) |
| best 선택 사유 | argmin mean_eucl + 동률 시 tie-breaker (smaller window 우선) |
| 다음 plan 후보 | enumeration only (가중 polyfit, Kalman 전처리, neural model 등). 우선순위 X |

---

## §N+3. 통계 함정 & caveats

1. **5-fold CV 신뢰구간 좁음** — n=10,000 / fold val ≈ 2,000 → mean_eucl SE ≈ std/√5. 두 exp 의 mean 차이가 fold-std 보다 작으면 noise. results.md 에 std 명시.
2. **Hit-rate 반경 비공개** — 4 후보 반경 분포만 보고, leaderboard 점수는 별도 plan 의 분석 영역.
3. **+80 ms horizon = 2 step** — single-step extrapolation 보다 noise 증폭. window=2 가 의외로 강한 이유: 가장 최근 변화율만 신뢰하고 노이즈 누적을 피함.
4. **per-axis tuning overfit 위험** — grid 12×3 = 36 candidates, 5-fold mean 으로 selection 하면 selection bias 가능. 별도 hold-out 으로는 검증하지 않음 (pre-reg). hold-out 검증은 leaderboard 가 사실상 대신 함.
5. **단위 m 가정** — competition-overview §3.3 근거. label, sample, sample_submission 모두 동일 단위로 가정.

---

## §N+4. 변경 이력

- v1 (2026-05-10): 초안. EDA + Re-EDA 결과를 바탕으로 4 exp closed-form baseline 설계.

---

## §N+5. 참조

- `notes/competition-overview.md` (대회 사양, +80ms target, hit-rate metric)
- `WORKFLOW.md` §4 (4-way token), §5 (plan obligations), §6 (results), §7 (run dir), §11 (handoff), §12 (autonomous protocol)
- `CLAUDE.md` (autonomous execution policy)
- 데이터: `data/{train,test}/*.csv` (각 11 row), `data/train_labels.csv`, `data/sample_submission.csv`
- 선행 EDA 산출 (in-conversation, 2026-05-09 ~ 2026-05-10): linear-2pt baseline 의 재계산 결과 mean_eucl=0.0129 (n=2000)
