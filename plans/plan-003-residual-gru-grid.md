---
plan_id: 003
version: 6
date: 2026-05-10 (Asia/Seoul)
status: draft
based_on:
  - 001
  - 002
  - B001_linear-2pt
scope: full-stack (lean residual-GRU baseline + 4 component ablation + winning-components combined train + 자동 LB submission)
exp_ids:
  - R001_baseline-residual-gru
  - R002_physics-features
  - R003_ema-extrapolate
  - R004_wingbeat-oscillation
  - R005_loss-mse
  - R006_combined-winners
---

# plan-003 v6 — Residual GRU Lean Baseline + Component Ablation Grid + Winning-Components Combined

## §0. 한 줄 목적

> **`linear extrap (= B001 식: X[:,-1] + 2·(X[:,-1]-X[:,-2])) + GRU(잔차) + Huber + relative coords` 로 정의되는 *lean residual-GRU baseline* (R001) 을 plan-001 floor (B001 cv_mean_eucl=0.01294, LB 0.60) 와 plan-002 floor (S001 LB 0.4932) 에 대해 정량 박제하고, `notes/mosquito-trajectory-ideas.md` 의 supportive component 4종 (R002 physics features / R003 EMA 외삽 baseline / R004 wing-beat oscillation / R005 Huber→MSE) 을 *각각 R001 위에 단일 변수만 변경* 하는 ablation 으로 비교 — 그리고 R002~R005 중 R001 대비 *cv_mean_eucl paired Δ < 0* 인 winning component 들을 모두 합친 R006_combined-winners 를 새로 학습해 그 1개 (또는 fallback 시 R001 1개) 만 dacon public LB 에 *autonomous loop 자율 제출* (dacon-submit skill, 사용자 승인 없음) 로 1 LB 점수 회수해 다음 plan 들의 selection anchor 박제. winning component combined train + 1 LB 회수는 본 plan 의 *의무 산출* 이며, 미회수 시 G_final 종료 불가.**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- 모든 5 ablation exp (R001~R005) 의 **5-fold CV mean_eucl + per-axis MAE + hit_rate@{0.05,0.10,0.20,0.50} m** 가 registry + run dir 에 기록.
- 5 ablation exp + R006 모두 NaN/Inf 0건 + training divergence 0건. 위반 시 `nn_numerical` severe.
- R001 baseline 이 B001 floor 0.01294 와 paired Δ ≤ +0.005. 위반 시 `residual_no_convergence` severe.
- best of {R001..R005} cv_mean_eucl < 0.030 (sanity; closed-form floor 의 ×2.3 한도). 위반 시 `nn_no_signal` severe.
- **R006_combined-winners 학습 + cv 평가 완료** (winning component 0개 시 R001 직접 복제). R006 cv 가 R001 cv + 0.001 초과 시 → fallback 분기 진입 (LB 제출 csv = R001 의 것; `combined_no_improvement` warn 박제 — severe X).
- **best 1 LB 제출 (필수, skip 불가, 자율 실행)**: R006 또는 fallback 시 R001 의 `submission.csv` → **autonomous loop 가 사용자 승인 없이 `dacon-submit` skill 1회 호출** + 1 LB 점수 회수. 사용자 confirm prompt 발생 X (CLAUDE.md autonomous policy).
- **best 1 LB 점수 results frontmatter 박제 의무**: `plans/plan-003-residual-gru-grid.results.md` 의 `lb_exp_id` (R006 또는 R001) + `lb_score` 필드에 기록되어야 G_final 종료 가능. 미회수 시 `lb_unsubmitted` severe.

### G-gates

- G0: STAGE 0 인프라 commit chain 완료 (residual-GRU model + training loop + 4개 component module + run.py method dispatch + tests green) [DONE post-c6 a142b3d]
- G1: STAGE 1 R001 lean baseline 결과 기록 + B001 paired comparison [DONE 1caa8ac]
- G2: STAGE 2 R002 (physics features), R003 (EMA baseline) 결과 기록 [DONE post-c9 0f5b86b]
- G3: STAGE 3 R004 (wing-beat), R005 (loss MSE) 결과 기록 [DONE post-c11 91f124c]
- G3.5: STAGE 3.5 — R001~R005 결과로 winning components 식별 + R006_combined-winners config 자동 생성 + 학습 + cv 평가 [DONE 7b9cc47 — winning=0 → R006 = R001 cp, fallback=False]
- G_final: R006 (또는 fallback R001) submission.csv 생성 + dacon-submit skill 자율 호출 + 1 LB 점수 회수 + results.md 작성 [DONE-partial 96bb5d2 — submission 자율 제출 완료 (isSubmitted=True), lb_score carry-over pending]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | code | `src/models/residual_gru.py` — `ResidualGRU(input_dim, hidden=64, layers=2, dropout=0.1)`. spec @ §4.1 | [DONE 06f0b09] |
| c2 | code | `src/baselines/linear_extrapolate.py` — `linear_extrap(X, t_target=80)` (= B001 식), `ema_extrapolate(X, alpha)`. spec @ §4.2 | [DONE db596aa] |
| c3 | code | `src/features/physics.py` (velocity/acceleration/jerk/curvature) + `src/features/oscillation.py` (wingbeat_fft). spec @ §4.3 | [DONE fa27728] |
| c4 | code | `src/training/train_residual.py` — fold 학습 (Huber/MSE loss selectable) + `make_feature_fn(components: list[str])` factory. spec @ §4.4 | [DONE c5d4110] |
| c5 | code | `src/run.py` 확장 — `method="gru-residual"` 분기 추가. spec @ §4.5 | [DONE 963ca16] |
| c6 | test | `tests/test_residual_gru.py`, `tests/test_features.py`, `tests/test_ema_extrapolate.py` (3 신규). spec @ §4.6 | [DONE a142b3d — .gitignore 1줄 동봉] |
| G0 | gate | `pytest -q tests/` exit 0; B001~B004, S001~S004 backward-compat smoke (cv_mean diff < 1e-4); torch import + device probe 성공 | [DONE post-c6 — 51 pytest pass + 8 closed-form Δ=0 + cuda=True 확인] |
| c7 | exp R001 | `configs/baseline/R001_baseline-residual-gru.yaml` + run + ckpt/fold{0..4}.pt + registry. spec @ §5 | [DONE 1caa8ac — cv=0.013383] |
| G1 | gate | R001 summary.json + 5 fold ckpt + cv_mean_eucl finite + B001 paired Δ ≤ +0.005 | [DONE 1caa8ac — paired Δ=+0.000442] |
| c8 | exp R002 | `configs/baseline/R002_physics-features.yaml` + run + registry. spec @ §6.1 | [DONE 60b3639 — cv=0.015157] |
| c9 | exp R003 | `configs/baseline/R003_ema-extrapolate.yaml` + run + registry. spec @ §6.2 | [DONE 0f5b86b — cv=0.014038] |
| G2 | gate | R002, R003 summary 모두 기록; R001 paired Δ 표 산출 가능 | [DONE post-c9 0f5b86b] |
| c10 | exp R004 | `configs/baseline/R004_wingbeat-oscillation.yaml` + run + registry. spec @ §7.1 | [DONE 04fee5f — cv=0.013476] |
| c11 | exp R005 | `configs/baseline/R005_loss-mse.yaml` + run + registry. spec @ §7.2 | [DONE 91f124c — cv=0.013388] |
| G3 | gate | R004, R005 summary 기록 | [DONE post-c11 91f124c] |
| c12 | sub-combined-train | `src/combine.py` 신규 (winning components 식별 + R006 config 자동 생성). R001~R005 summary.json 읽음 → winning 식별 → `configs/baseline/R006_combined-winners.yaml` 자동 작성 → `src.run.main` 으로 R006 학습 (winning 0개 시 R001 직접 복제, 학습 skip) → registry append. spec @ §8.1 | [DONE 7b9cc47 — winning=0 → R006 = R001 cp] |
| G3.5 | gate | R006 summary.json + (winning>0 시) ckpt + registry 행 존재; cv_mean_eucl finite. R006 cv > R001 cv + 0.001 시 → `combined_no_improvement` warn 박제 + fallback 플래그 set | [DONE 7b9cc47 — R006.cv=R001.cv, fallback=False] |
| c13 | sub-gen | `src/submit.py` 확장 (gru-residual method 분기 + ckpt fold ensemble; 후방호환 보존). lb_exp_id 결정 = (fallback 플래그 false 면 R006, true 면 R001) → 해당 exp 의 `runs/baseline/{lb_exp_id}/submission.csv` 생성. 스키마 검증 fail 시 `submission_schema_fail` severe. spec @ §8.2 | [DONE ae31834 — lb_exp_id=R006, R001+R006 csv 모두 생성] |
| c14 | sub-lb | **`dacon-submit` skill 자율 호출 (사용자 승인 X)** — lb_exp_id 의 submission.csv 1회 제출 → LB 점수 회수 → `analysis/plan-003/lb_log.md` 1행 기록 + registry notes 갱신. skill 부재 시 `dacon_submit_skill_missing` severe. spec @ §8.3 | [DONE 1c4831e — `{isSubmitted: True, detail: Success}`, lb_score carry-over pending] |
| c15 | docs | `analysis/plan-003/results.md` + `plans/plan-003-residual-gru-grid.results.md` (frontmatter `lb_exp_id`, `lb_score`, `combined_winning_components`, `combined_fallback`). spec @ §N+2 | [DONE 96bb5d2] |
| G_final | gate | 위 모두 완료 + §0.5 [TODO]→[DONE] sync (§12.6 blacklist 의 유일한 예외) + lb_score 박제 | [DONE-partial 96bb5d2 — lb_score 회수만 carry-over 대기] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `nn_numerical`: 학습 중 loss NaN/Inf 또는 gradient NaN 발생 → torch dtype/lr/gradient clip 점검. 자동 복구 X.
- `residual_no_convergence`: R001 cv_mean_eucl 의 5fold paired Δ vs B001 가 +0.005 초과 → 학습 루프 / target 정의 / inverse transform 버그 의심.
- `nn_no_signal`: best of {R001..R005} cv_mean_eucl ≥ 0.030 → architecture/data pipeline/normalization 근본 버그 의심.
- `cuda_oom`: GPU OOM 발생 시 batch_size 64→32→16 자동 감소 후 재시도 (3 단계). 모두 fail 시 severe.
- `lb_unsubmitted`: G_final 진입 시점에 `lb_score` 미회수 → 본 plan 의 *의무 산출* 위반.
- `submission_schema_fail`: lb_exp_id 의 submission.csv 가 sample_submission 스키마 검증 fail.
- `dacon_submit_skill_missing`: c14 진입 시 `dacon-submit` skill 부재 → 사용자 escalate.
- `backward_compat_drift`: G0 의 B001~B004 / S001~S004 cv_mean_eucl 가 registry 기존 값과 4 자리 이상 어긋남.
- **`combined_no_improvement`** (warn only, severe X): R006 cv > R001 cv + 0.001 → fallback 분기 진입 (R001 csv 를 LB). G_final 진입 차단 X — *그 자체가 본 plan 의 정보* (= "winning component 조합이 individual best 보다 못함, interaction effect 발현" 박제).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/models/residual_gru.py`, `src/baselines/linear_extrapolate.py`, `src/features/physics.py`, `src/features/oscillation.py`, `src/training/train_residual.py`, **`src/combine.py` (신규 — c12 의 winning 식별 + R006 config 자동 생성)**
  - `src/features/__init__.py`, `src/models/__init__.py`, `src/training/__init__.py`
  - `tests/test_residual_gru.py`, `tests/test_features.py`, `tests/test_ema_extrapolate.py`, **`tests/test_combine.py` (신규 — winning 식별 + config 자동 생성 검증)**
  - `configs/baseline/R00*.yaml` (R001~R005 + **R006 자동 생성**), `runs/baseline/R00*/**` (ckpt 는 `.gitignore` 으로 제외)
  - `analysis/plan-003/**` (특히 `analysis/plan-003/lb_log.md`, `analysis/plan-003/winning_trace.md`)
  - `.gitignore` (ckpt 디렉토리 제외 규칙 추가용 — 1회 수정 한정; **정확한 패턴 = `runs/baseline/*/ckpt/` 한 줄 append**)
- whitelist 확장: `src/run.py` (method dispatch 만), `src/submit.py` (gru-residual method dispatch + lb_exp_id 분기 추가), `src/baselines/__init__.py` (re-export)
- blacklist 추가:
  - `runs/baseline/B00*/**`, `configs/baseline/B00*.yaml` (plan-001 산출 — 절대 수정 금지)
  - `runs/baseline/S00*/**`, `configs/baseline/S00*.yaml` (plan-002 산출 — 절대 수정 금지)
  - `analysis/plan-001/**`, `analysis/plan-002/**`

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — torch backend 채택 (PyTorch ≥ 2.0); pip install 자율 시행`
- `decision-note: spec-default — GRU hidden=64, layers=2, dropout=0.1, lr=1e-3 (AdamW), batch=64, epochs=100, early-stop patience=10`
- `decision-note: spec-default — random seed = 42 (torch + numpy + cuda); fold 별 seed = 42 + fold_idx; cudnn.deterministic=True`
- `decision-note: spec-default — 학습 device = CUDA 가용 시 "cuda:0" (0번 GPU 강제 사용), 아니면 CPU + epochs 50 으로 자동 감소. 다중 GPU 환경에서도 0번 만 사용해 결과 reproducibility 보장. CUDA_VISIBLE_DEVICES 환경변수 의존 X — 코드에서 device 문자열 명시`
- `decision-note: spec-default — fold ensemble mean (5 ckpt 평균) 으로 test 예측. full-train 재학습 X`
- `decision-note: spec-default — wing-beat FFT (R004): per-axis n_bins=3 magnitude (DC + 2 harmonics)`
- `decision-note: spec-default — EMA alpha=0.5 (R003); 지수가중 속도 = Σ α^k · v[-k] / Σ α^k`
- `decision-note: spec-default — physics features (R002) curvature = |v × a| / |v|^3, NaN 시 0 대체`
- `decision-note: spec-default — c14 LB 제출 = autonomous loop 가 사용자 승인 없이 dacon-submit skill 1회 호출 (CLAUDE.md autonomous policy 적용)`
- `decision-note: spec-default — HuberLoss δ=1.0 (PyTorch torch.nn.HuberLoss default; caveat #7 동기)`
- `decision-note: spec-default — DataLoader(batch_size=64, shuffle=True for train / shuffle=False for val, num_workers=0, drop_last=False, pin_memory=(device.startswith("cuda")))`
- `decision-note: spec-default — feature_fn 출력 raw (normalize 없음). physics features (R002) 의 jerk magnitude (~78 m/s³) 도 그대로 — lean baseline 원칙. 학습 NaN 발생 시 nn_numerical severe 에 위임 (별도 plan 에서 normalization 검토)`
- `decision-note: spec-default — .gitignore 패턴 = runs/baseline/*/ckpt/ 한 줄 (plan-001/002 의 runs 영역 보호 위해 wildcard 좁게; runs/**/ckpt/ 같은 광역 패턴 사용 X)`
- `decision-note: spec-default — closed-form 분기 (predict_for_config) 와 gru-residual 분기 (_train_and_predict_residual_fold) 는 run_baseline 의 fold loop 안에서 method 키로 분리 호출. predict_for_config 시그니처 변경 X (후방호환 보존)`
- **`decision-note: spec-default — winning 기준 = R001 대비 paired mean Δ (= R00x.cv_mean_eucl - R001.cv_mean_eucl) < 0. statistical significance (|Δ| ≥ fold-σ) 는 strict mode 별도 plan`**
- **`decision-note: spec-default — additive 가정: feature 충돌 (R002 physics + R004 wingbeat 동시 winning) 시 *둘 다 합치기* (rel + physics + wingbeat = input_dim 22). interaction effect caveat #14 박제`**
- **`decision-note: spec-default — R006 fallback 임계값: R006.cv > R001.cv + 0.001 → fallback 발동 (R001 csv 를 LB). +0.001 = noise margin (5 fold mean 의 SE 영역)`**
- **`decision-note: spec-default — winning 0개 시 R006 = R001 직접 복제 (별도 학습 X, R001 ckpt + R001 submission.csv 그대로 사용). 단 registry 에 별도 행 (R006_combined-winners) append 하여 4-way token 일치성 보존`**
- **`decision-note: spec-default — combined feature_fn = make_feature_fn(["relative", optional "physics", optional "wingbeat"]) — 각 component 별 axis-wise concat. component list 가 자동 생성된 R006 config 의 feature_components 키`**

---

## §1. 배경

### §1.1 plan-001 + plan-002 결과 인계 (registry 기반)

| exp_id | plan | method | cv_mean_eucl | LB hit@1cm |
|---|---|---|---|---|
| **B001_linear-2pt** | 001 | polyfit (w=2, d=1) | **0.01294** ± 0.00058 | **0.60** |
| B004_per-axis-grid | 001 | polyfit per-axis tune | 0.01294 ± 0.00058 | (미제출) |
| S001_cspline-natural-full | 002 | cspline natural 11pt | 0.01742 ± 0.00071 | 0.4932 |
| S002_cspline-notaknot-full | 002 | cspline not-a-knot 11pt | 0.05370 ± 0.00282 | 0.1204 |
| S003_cspline-window-grid | 002 | cspline per-axis grid | 0.01740 ± 0.00071 | 0.4926 |
| S004_smoothing-spline-tuned | 002 | smoothing spline (s tune) | 0.03322 ± 0.00270 | 0.2178 |

핵심 인계 사실:
- **B001 (closed-form linear-2pt) 가 모든 closed-form 변형 대비 압도적 floor.** plan-002 의 cspline 4 변형 모두 B001 에 patently 열등. CV 차이가 LB 차이로 *증폭* — LB metric (hit@1cm) 이 cv_mean_eucl 보다 tail 에 민감.
- **closed-form 영역의 marginal gain 영역이 좁다는 강한 신호.** 노트북 공유작 (PB_0.6822, Frenet+Attn-GRU+잔차MLP) 와의 gap 0.082 는 *학습 모델 조합* 으로만 메울 수 있음.
- *plan-002 의 가장 유망 후보였던 smoothing spline (S004) 이 LB 에서 0.22 로 가장 약함* — CV ↔ LB 비단조. neural model 에서도 동일 함정 가능.

### §1.2 본 plan 의 가설 출발점 — Residual GRU + Lean Baseline + Winning-Components Combined

사용자 제시 (notes/mosquito-trajectory-ideas.md §5 잔차 예측 + 본 plan 의도서):

> "최종 예측 = 직선 외삽 (베이스라인) + GRU 출력 (잔차 보정). GRU 는 직선이 얼마나 틀렸는지만 학습. 모델은 어려운 10~20% 에만 집중 — 출력 분산이 작아 학습 안정."

원리:
- closed-form B001 이 LB 0.60 → 궤적의 80~90% 가 등속 외삽으로 설명됨.
- residual = `y_true - linear_extrap` 의 분포 표준편차는 절대좌표 분포의 ×0.1 영역 (B001 mean_eucl=0.013 이 곧 잔차 RMS 의 상한).
- GRU 가 잔차의 *signed* 성분 (방향성) 을 학습하면 각 sample 별로 ±수 mm 보정 가능. hit@1cm 영역에서 그 보정의 LB 가치는 *비선형적으로 큼*.

비판적 prior:
- residual 학습은 baseline 의 quality 에 직접 의존 — baseline 이 unbiased 가 아니면 GRU 가 baseline bias 를 학습 → 결국 단순 학습과 차이 없음.
- 11 timestep 의 짧은 시퀀스 + 10k sample 의 작은 데이터셋 → GRU 가 over-parametrize 되기 쉬움. lean baseline (R001) 은 *최소 hyperparameter* 로 시작.
- 본 plan 의 *4 개 component ablation* 은 이 lean baseline 위에서 *단일 변수만* 변경. 즉 R002~R005 의 결과는 "각 component 가 GRU residual 학습에 *독립적으로* 기여하는지" 를 측정.
- **v5 추가 — combined model (R006)**: ablation 결과로 *winning components* 식별 후 모두 합친 final model 을 학습. ablation 의 *목적성* (개별 검증 → 조합 selection) 명확화. 단, additive 가정 (interaction effect 0) 이 깨질 수 있다는 caveat 박제.

### §1.3 본 plan 의 결정적 근거

ablation 로 검증할 4개 component 가설 + 1개 combined verification 가설:

- **H1 (physics features, R002)**: per-axis 속도/가속도/JERK/곡률 입력이 GRU 의 residual 학습 표현력을 보강해 cv_mean_eucl 가 R001 대비 ↓.
- **H2 (EMA baseline, R003)**: 지수가중 속도 외삽 (recent ↑) 이 직선 외삽보다 unbiased baseline 이라 residual 분포가 좁아져 GRU 학습이 안정 → cv_mean_eucl ↓.
- **H3 (wing-beat, R004)**: 11pt 좌표의 미세 떨림 (FFT magnitude) 이 모기의 *비행 상태* (직진/회전/지그재그) 를 implicit class 로 노출 → GRU 가 상태별 잔차를 분리 학습.
- **H4 (loss MSE, R005)**: Huber loss 가 outlier 에 강건하다는 통설이 *본 데이터/잔차 분포에 실제로 작용* 하는지 검증. MSE 변경 시 cv 가 악화되면 H4 채택, 동등/개선이면 Huber prior 기각.
- **H5 (combined-additive, R006)**: H1~H4 의 winning component 들이 *additive 조합* (interaction effect ≈ 0) 으로 합쳐지면 cv_mean_eucl 가 R001 대비 단순 합산 이상 ↓. 기각 시 (R006.cv > R001.cv + 0.001): interaction effect 발현 → fallback (R001 LB 제출) + 그 자체가 본 plan 의 정보 (다음 plan 들의 selection 시 *조합 검증 필수* anchor 박제).

각 가설의 *기록 자체* (CV verdict + R006 LB 신호) 가 다음 plan 들의 anchor.

---

## §2. Scope

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 모델 | ResidualGRU (input_dim 가변, hidden=64, layers=2, dropout=0.1, FC(3) 출력) — *모든 exp 통일* |
| Architecture 변경 | **금지** (단일 변수 원칙 위반 방지). hidden/layers/dropout/lr 등 hyperparameter 도 모든 exp 동일 (R001~R006 통일) |
| 학습 | PyTorch + AdamW (lr=1e-3, weight_decay=1e-4), Huber loss (R005 만 MSE; R006 은 winning 판정에 따라 huber/mse), batch=64, epochs=100, early-stop patience=10 |
| Loss | Huber (R001~R004), MSE (R005 만), R006 = (R005 winning 시 mse else huber) |
| Baseline 외삽 | linear-2pt (= B001 식; R001/R002/R004/R005), EMA-가중 속도 (R003 만, alpha=0.5), R006 = (R003 winning 시 ema(alpha=0.5) else linear) |
| Input feature | relative coords (R001/R003/R005), + physics (R002), + wing-beat FFT (R004), R006 = relative + (R002 winning 시 +physics) + (R004 winning 시 +wingbeat) — *additive 합산* |
| Inference | fold ensemble mean (5 ckpt 평균 예측) — *모든 exp 통일* |
| target time | +80 ms (스펙 고정) |
| primary dev metric | mean 3D Euclidean distance (m) |
| 보조 metric | per-axis MAE, hit_rate @ {0.05, 0.10, 0.20, 0.50} m |
| CV | 5-fold, seed=42, `src/io.py:kfold_split` 그대로 재사용 |
| Test 예측 (LB) | lb_exp_id (= R006 또는 fallback 시 R001) 의 fold ensemble mean → 1 submission.csv |
| LB 제출 | 1개만 dacon-submit skill 자율 호출 (사용자 승인 X). lb_exp_id ∈ {R006, R001} |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| GRU hidden/layers/dropout/lr 변경 | 단일 변수 위반. 별도 plan (hyperparameter sweep) |
| TCN/Transformer/MLP backbone | 본 plan 의 scope 가 *component ablation + combined*, architecture 비교 X |
| GMM / NLL / quantile loss | 본 plan 의 loss ablation 은 Huber↔MSE 만 |
| TTA (X-Y rotation, Y-flip) | 본 plan 에서 명시적으로 제외. 별도 plan 에서 후처리만으로 추가 검증 가능 |
| 2개 이상 LB 제출 | 본 plan 사용자 결정으로 best 1개만 LB 제출. 다른 exp 의 LB 신호는 별도 plan 에서 회수 가능 |
| **R006 의 hyperparameter sweep** | combined model 도 R001 동일 hyperparameter (hidden/layers/lr/batch 등). winning component 조합만 변수 |
| **2개 이상 combined 변형** (예: physics+wingbeat vs physics-only) | 단일 R006 만. interaction 분리 측정은 별도 plan |
| **R006 의 architecture 변경** | hidden/layers 동일. winning component 조합 외 모든 hyperparameter R001 와 비트 동일 |
| 11pt → 다른 길이 시퀀스 | 데이터 스펙 고정 |
| fold 별 ensemble weight tuning | 단순 mean (균등 가중) |
| Kalman / Savitzky-Golay 입력 평활 | 별도 plan |
| ensemble (R00x + B001 + S00x 평균) | 별도 plan |
| target horizon ≠ +80 ms | 스펙 위반 |
| hit-radius 추정 (LB probing) | 5/일 한도 절약 |
| LB 점수로 hyperparameter selection | overfit LB. CV mean_eucl argmin 만 |
| polyfit/cspline 분기 회귀/재실행 | plan-001/002 산출 영구. backward-compat smoke 만 통과시킴 |
| ckpt 의 VCS 추적 | 무거운 binary, `.gitignore` 로 제외 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- plan-001 §3.1 / plan-002 §3.1 과 **완전 동일** 함수 (`src/io.py:kfold_split(ids, k=5, seed=42)`).
- 같은 fold = 같은 val ids → R00x 와 B001/S00x 의 fold-level paired 비교 가능.

### §3.2 합격 기준

| 조건 | 정의 |
|---|---|
| A. 인프라 정상 | `pytest tests/` green; `from src.models.residual_gru import ResidualGRU` import OK; G0 backward-compat smoke (B001~B004, S001~S004 cv_mean_eucl 4자리 일치) |
| B. 산술 안정성 | 6 exp 모두 (R001~R006) fold-level 학습 + OOF prediction 에 NaN/Inf 0건. training loss 가 1 epoch 내 NaN 발생 시 `nn_numerical` severe |
| C. R001 수렴 | R001 cv_mean_eucl 가 B001 대비 5fold paired mean Δ ≤ +0.005 (위반 시 `residual_no_convergence` severe) |
| D. 결과 sanity | best of {R001..R005} cv_mean_eucl < 0.030 (위반 시 `nn_no_signal` severe) |
| E. 비교 박제 | results.md 에 5 ablation exp × {cv_mean_eucl±std, per-axis MAE, hit_rate@0.10, vs B001 paired Δ, vs R001 paired Δ} 표 + R006 의 (cv vs R001 + winning trace) |
| F. **R006 학습 + cv 평가 (의무)** | winning components 식별 → R006 config 자동 생성 → 학습 (winning 0개 시 R001 직접 복제, 학습 skip) → cv 평가. R006 cv > R001 cv + 0.001 시 → `combined_no_improvement` warn 박제 + fallback 플래그 set (severe X) |
| G. **lb_exp_id 1 LB 제출 (의무, autonomous loop 자율 실행)** | lb_exp_id = (fallback false → R006, true → R001) → `runs/baseline/{lb_exp_id}/submission.csv` 생성 (스키마 100 % 일치) → **autonomous loop 가 사용자 승인 없이 `dacon-submit` skill 1회 호출** → LB 점수 회수 → results frontmatter `lb_score` + `lb_exp_id` + `combined_fallback` 박제. 미회수 시 `lb_unsubmitted` severe |

### §3.3 평가 점수 / 집계 (plan-001/002 §3.3 와 동일)

- per fold metric: `mean_eucl`, per-axis MAE, hit_rate(r) for r ∈ {0.05, 0.10, 0.20, 0.50}
- per exp metric: 5 fold mean ± std (median 아님)
- exp 비교: cv_mean_eucl argmin (1차), 동률 시 작은 input feature_dim 우선 (2차), 동률 시 alphabetical exp_id 우선 (3차)
- B001/R001 vs R00x **same-fold paired Δ**: 5 fold 각각의 mean_eucl 차이 + 부호 일관성 (5 fold 중 동일 부호 비율)
  - 데이터 소스: B001 → `runs/baseline/B001_linear-2pt/history.json`; R001~R006 → `runs/baseline/R00x_*/history.json`. plan-002 §3.3 fallback 규칙 동일.
  - Δ 정의: `Δ_fold[i] = R00x.fold_metrics[i].mean_eucl - {B001 or R001}.fold_metrics[i].mean_eucl` (음수 = R00x 우월).

### §3.4 Winning component 식별 기준 (R006 자동 생성용)

각 ablation exp R002~R005 에 대해:

```
winning(R00x) = (mean(R00x.fold_mean_eucl) - mean(R001.fold_mean_eucl)) < 0
              = paired mean Δ < 0
```

- 단순 부등호 (1차원 기준). statistical significance 검정 (|Δ| ≥ fold-σ, 부호 일관성 ≥ 4/5 등) 은 strict mode 별도 plan.
- noise margin 적용 X — 정확히 < 0 이면 winning. 0 ≤ Δ ≤ 0.001 영역도 *non-winning* (보수적).
- winning 결과는 `analysis/plan-003/winning_trace.md` 에 표 형태 박제 (exp_id, R001.cv, R00x.cv, Δ, winning yes/no).

R006 config 자동 생성 규칙 (`src/combine.py`):

| component | winning 시 R006 config 변경 |
|---|---|
| R002 (physics) | `feature_components` list 에 `"physics"` 추가 |
| R003 (EMA) | `baseline_type: ema`, `ema_alpha: 0.5` |
| R004 (wingbeat) | `feature_components` list 에 `"wingbeat"` 추가 |
| R005 (MSE) | `loss_type: mse` |

기본 (winning 0개) 시 R006 config = R001 와 비트 동일 → R006 = R001 직접 복제 (학습 skip, R001 ckpt + submission 그대로 사용). 단 registry 에 별도 행 append (4-way token 일치성 보존).

R002 winning + R004 winning 동시 시 **둘 다 합치기** (`feature_components: ["relative", "physics", "wingbeat"]`, input_dim = 3 + 10 + 9 = 22). additive 가정. interaction 검증은 별도 plan (caveat #14).

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 `src/models/residual_gru.py` (신규)

```python
class ResidualGRU(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 3)

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, 11, input_dim) → out: (B, 3) Δ (residual w.r.t. baseline extrapolation)
        out, _ = self.gru(X)
        return self.fc(out[:, -1, :])
```

- input_dim 은 cfg `input_dim` 에서 dynamic. R001=3, R002=13, R003=3, R004=12, R005=3, R006=3+10*physics+9*wingbeat (winning 에 따라 3, 12, 13, 22 중 하나).
- 출력은 *항상* 잔차 (Δx, Δy, Δz). 절대 좌표 = baseline + Δ (inference 시).
- weight init: PyTorch default.

### §4.2 `src/baselines/linear_extrapolate.py` (신규)

**중요**: `linear_extrap` 함수는 **B001_linear-2pt 의 closed-form 식과 비트-단위 동일** (plan-001 floor cv=0.01294 의 baseline 함수). 즉 R001/R002/R004/R005 의 baseline 외삽 = B001 prediction. residual = y_true - linear_extrap(X).

```python
import numpy as np
from src.io import TIMESTEPS_MS  # (n_t,)=11 의 [0,40,80,...,400] ms 균등 grid (40ms uniform 가정)


def linear_extrap(X: np.ndarray, t_target_ms: int = 80,
                  timesteps_ms: np.ndarray = TIMESTEPS_MS) -> np.ndarray:
    """B001 식 그대로: pred = X[:, -1] + (t_target / 40) * (X[:, -1] - X[:, -2])
    plan-001 best (cv=0.01294, LB=0.60) 의 baseline. residual = y_true - linear_extrap(X).
    """
    dt_target = t_target_ms / (timesteps_ms[-1] - timesteps_ms[-2])  # = 80/40 = 2.0
    return X[:, -1] + dt_target * (X[:, -1] - X[:, -2])

def ema_extrapolate(X: np.ndarray, alpha: float = 0.5,
                    t_target_ms: int = 80,
                    timesteps_ms: np.ndarray = TIMESTEPS_MS) -> np.ndarray:
    """지수가중 속도 외삽 (R003 전용; R006 은 R003 winning 시 사용).
    v_k = (X[:, -k] - X[:, -k-1]) / 40 ms, k=1..10
    v_ema = Σ α^(k-1) · v_k / Σ α^(k-1)  (k=1 most recent)
    pred = X[:, -1] + t_target_ms · v_ema
    """
    n, T, _ = X.shape
    dt_ms = float(timesteps_ms[1] - timesteps_ms[0])  # 균등 grid 가정 = 40.0 ms
    velocities = (X[:, 1:, :] - X[:, :-1, :]) / dt_ms  # (n, 10, 3) — 단위 m/ms
    weights = np.array([alpha ** k for k in range(velocities.shape[1])])  # array index k=0 = oldest
    weights = weights[::-1] / weights.sum()  # reverse → 가장 최근 velocity 가 weight α^0=1, 가장 오래된 = α^9 (docstring 의 1-indexed k=1 most recent 와 동치)
    v_ema = np.einsum("ntd,t->nd", velocities, weights)
    return X[:, -1] + t_target_ms * v_ema
```

- alpha=0.5: v_1 가중치 ≈ 0.5, v_2 ≈ 0.25, v_3 ≈ 0.125, ... — *최근 3 시점이 87.5%* 결정.
- 양 함수 모두 closed-form, GPU 불필요. numpy 전용.
- **B001 동등성 검증 (G0 smoke + tests)**: random X 에 대해 `linear_extrap(X)` 와 `predict(X, window=2, degree=1, t_target=80, timesteps=TIMESTEPS_MS)` (= `src.baselines.window_polyfit.predict`; plan-001 산출 모듈, 호출 시그너처는 src/baselines/window_polyfit.py 의 정의 그대로) 의 절대오차 < 1e-9.

### §4.3 `src/features/physics.py` + `src/features/oscillation.py` (신규)

`physics.py` (단위 통일: 입력 X 는 m, dt_sec=0.04 (40 ms 균등) → velocity m/s, acceleration m/s², jerk m/s³. caveat #5 / decision-note 의 m/s 표기와 일치):
```python
def velocity(X: np.ndarray, dt_sec: float = 0.04) -> np.ndarray:
    v = np.zeros_like(X)
    v[:, 1:, :] = (X[:, 1:, :] - X[:, :-1, :]) / dt_sec
    v[:, 0, :] = v[:, 1, :]  # forward fill
    return v

def acceleration(X: np.ndarray, dt_sec: float = 0.04) -> np.ndarray:
    v = velocity(X, dt_sec)
    a = np.zeros_like(X)
    a[:, 1:, :] = (v[:, 1:, :] - v[:, :-1, :]) / dt_sec
    a[:, 0, :] = a[:, 1, :]
    return a

def jerk(X: np.ndarray, dt_sec: float = 0.04) -> np.ndarray:
    a = acceleration(X, dt_sec)
    j = np.zeros_like(X)
    j[:, 1:, :] = (a[:, 1:, :] - a[:, :-1, :]) / dt_sec
    j[:, 0, :] = j[:, 1, :]
    return j

def curvature(X: np.ndarray, dt_sec: float = 0.04, eps: float = 1e-9) -> np.ndarray:
    """κ = |v × a| / |v|^3 per timestep. shape (n, 11, 1). v, a 단위 m/s, m/s² → κ 단위 1/m"""
    v = velocity(X, dt_sec)
    a = acceleration(X, dt_sec)
    cross = np.cross(v, a)
    num = np.linalg.norm(cross, axis=-1)
    den = np.linalg.norm(v, axis=-1) ** 3 + eps
    kappa = (num / den)[:, :, None]
    return np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
```

`oscillation.py`:
```python
def wingbeat_fft(X_relative: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """X_relative: (n, 11, 3) → (n, 11, 3*n_bins).
    각 axis 의 11pt 시퀀스 rfft → magnitude 의 첫 n_bins 성분 (DC + harmonics).
    timestep 차원에 broadcast (sequence-level summary feature)."""
    n, T, _ = X_relative.shape
    coefs = np.fft.rfft(X_relative, axis=1)  # (n, 6, 3)
    mags = np.abs(coefs)[:, :n_bins, :]  # (n, n_bins, 3)
    feats = mags.transpose(0, 2, 1).reshape(n, 3 * n_bins)
    return np.broadcast_to(feats[:, None, :], (n, T, 3 * n_bins)).copy()
```

- n_bins=3: DC + 1st + 2nd harmonic. 11pt × 40ms = 440ms 윈도우 → bin 폭 ≈ 2.27 Hz.
- sequence-level summary 라 각 timestep 에 동일 값 broadcast.

### §4.4 `src/training/train_residual.py` (신규)

```python
def train_fold(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    baseline_train: np.ndarray, baseline_val: np.ndarray,
    feature_fn,                      # X (n, 11, 3) → input (n, 11, input_dim)
    loss_type: str = "huber",        # "huber" | "mse"
    lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 64, epochs: int = 100,
    early_stop_patience: int = 10,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",  # 0번 GPU 강제
    seed: int = 42,
) -> dict:
    """
    학습:
      - target = y_train - baseline_train (residual)
      - input = feature_fn(X_train)  # raw, normalize 없음 (lean baseline 원칙). epoch loop 진입 *전 1회* 사전계산 후 tensor 캐스팅 (per-batch 재계산 금지 — feature_fn 의 numpy 비용 회피)
      - loss = nn.HuberLoss(delta=1.0) if loss_type=="huber" else nn.MSELoss()  # δ=1.0 = PyTorch default (caveat #7)
      - optim = AdamW(lr=lr, weight_decay=weight_decay)  # cfg["training"]["weight_decay"] 가 있으면 helper 가 전달, 부재 시 default 1e-4
      - DataLoader: batch_size=batch, shuffle=True (train) / shuffle=False (val),
                    num_workers=0, drop_last=False, pin_memory=(device.startswith("cuda"))
      - tensor dtype: torch.float32 (numpy float64 → tensor 시 .float() 명시 캐스팅)
      - epoch loop: train (1 epoch — per-batch: optimizer.zero_grad() → pred = model(x_b) → loss = loss_fn(pred, target_b) → loss.backward() → optimizer.step(); gradient clipping 미적용) → val mean_eucl 평가 → best 갱신/early-stop 카운트
      - early-stop on val mean_eucl (= ||baseline_val + model(feature_fn(X_val)) - y_val||₂.mean())
        - patience=early_stop_patience epoch 동안 개선 X 면 종료, best_state_dict 회복
        - **best_state_dict 박제 시 반드시 deep-copy**: `best_state_dict = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}` — `model.state_dict()` 가 model parameter 와 tensor memory 를 공유하므로 다음 epoch weight 갱신 시 silent overwrite 위험. cpu().clone() 으로 분리 보존.
      - seed: torch.manual_seed(seed), torch.cuda.manual_seed_all(seed),
              torch.backends.cudnn.deterministic=True, torch.backends.cudnn.benchmark=False
    return: {"best_state_dict": ..., "best_val_mean_eucl": ..., "best_epoch": int,
             "history": [{"epoch": i, "train_loss": ..., "val_mean_eucl": ...}, ...]}
    """
```

**Feature normalization 정책** (모든 feature_fn 공통):
- 출력은 *raw, normalize 없음*. physics features (R002) 의 jerk magnitude (~78 m/s³) 도 그대로 GRU 에 입력. 이유: lean baseline 원칙 + H1 의 본질 = "GRU 가 노이즈 dominant feature 를 무시 학습할 수 있는가" 검증.
- 학습 중 loss NaN/Inf 발생 시 → `nn_numerical` severe trigger (자동 복구 X). normalization 검토는 별도 plan.
- 별도 normalization 도입 시 단일 변수 원칙 위반 (= 새 component) 이므로 본 plan 에선 *전부 raw* 통일.

**feature_fn 정의 (R001~R005 전용 정적 함수)**:

```python
def relative_coords_feature(X: np.ndarray) -> np.ndarray:
    return X - X[:, -1:, :]  # (n, 11, 3)

def physics_feature(X: np.ndarray) -> np.ndarray:
    rel = relative_coords_feature(X)
    v = velocity(X); a = acceleration(X); j = jerk(X); k = curvature(X)
    return np.concatenate([rel, v, a, j, k], axis=-1)  # (n, 11, 13)

def wingbeat_feature(X: np.ndarray, n_bins: int = 3) -> np.ndarray:
    rel = relative_coords_feature(X)
    fft_feat = wingbeat_fft(rel, n_bins=n_bins)
    return np.concatenate([rel, fft_feat], axis=-1)  # (n, 11, 12)
```

**factory `make_feature_fn(components: list[str], wingbeat_n_bins: int = 3) -> callable`** (R006 combined 전용):

```python
def make_feature_fn(components: list[str], wingbeat_n_bins: int = 3):
    """components 는 ["relative"] (always 포함) + 선택적으로 ["physics", "wingbeat"].
    출력 feature 차원 = 3 + (10 if "physics" in components else 0)
                       + (3*wingbeat_n_bins if "wingbeat" in components else 0)
    예시:
      ["relative"]                          → input_dim = 3  (R001/R003/R005 동등)
      ["relative", "physics"]               → input_dim = 13 (R002 동등)
      ["relative", "wingbeat"]              → input_dim = 12 (R004 동등)
      ["relative", "physics", "wingbeat"]   → input_dim = 22 (R006 둘 다 winning)
    """
    assert "relative" in components, "relative 는 always 포함"
    use_physics = "physics" in components
    use_wingbeat = "wingbeat" in components

    def fn(X: np.ndarray) -> np.ndarray:
        parts = [relative_coords_feature(X)]
        if use_physics:
            parts.extend([velocity(X), acceleration(X), jerk(X), curvature(X)])
        if use_wingbeat:
            parts.append(wingbeat_fft(relative_coords_feature(X), n_bins=wingbeat_n_bins))
        return np.concatenate(parts, axis=-1)
    return fn
```

R001~R005 의 `_train_and_predict_residual_fold` 도 내부적으로 `make_feature_fn` 호출 가능 (단순화) — 단, R001 = `make_feature_fn(["relative"])`, R002 = `make_feature_fn(["relative", "physics"])`, R004 = `make_feature_fn(["relative", "wingbeat"])`. R003/R005 는 R001 와 feature_fn 동일.

학습 device: CUDA 가용 시 GPU. CPU fallback 시 epochs 50 으로 자동 감소 (decision-note).

### §4.5 `src/run.py` 확장 (method dispatch)

**원칙**: closed-form (polyfit / cspline / smoothing_spline) 와 gru-residual 은 *서로 다른 fold 내 구조* 라서 `predict_for_config` 시그니처 통합 X. 기존 `predict_for_config` 는 closed-form 만 처리 (변경 없음, 후방호환). gru-residual 은 신규 helper `_train_and_predict_residual_fold` 가 fold 내에서 train → ckpt 저장 → predict 수행.

**신규 helper 시그니처** (src/run.py 안에 정의):

```python
def _train_and_predict_residual_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg: dict, fold_idx: int, run_dir: Path,
) -> tuple[Path, np.ndarray, dict]:
    """
    1. cfg["feature_components"] (list[str]) → feature_fn = make_feature_fn(...)
       (R001~R005 의 config 도 feature_components 사용 = 통일.
        R001/R003/R005: ["relative"], R002: ["relative", "physics"],
        R004: ["relative", "wingbeat"], R006: dynamic.)
    2. cfg["baseline_type"] → baseline 함수 선택:
         "linear" → linear_extrap(X, t_target_ms)
         "ema"    → ema_extrapolate(X, alpha=cfg["ema_alpha"], t_target_ms)
       baseline_train = baseline_fn(X_tr); baseline_val = baseline_fn(X_va) — numpy
    3. input_dim 결정:
         input_dim = feature_fn(X_tr).shape[-1]
       sanity assert: input_dim == cfg["model"]["input_dim"] (config 와 일치)
    4. model = ResidualGRU(input_dim=..., hidden=cfg["model"]["hidden"], layers=..., dropout=...)
    5. info = train_fold(model, X_tr, y_tr, X_va, y_va, baseline_train, baseline_val,
                        feature_fn, loss_type=cfg["loss_type"], ...전부 cfg 에서 읽음)
       seed = cfg["training"]["seed"] + fold_idx (fold 별 seed)
    6. ckpt 저장:
         ckpt_path = run_dir / "ckpt" / f"fold{fold_idx}.pt"
         ckpt_path.parent.mkdir(parents=True, exist_ok=True)
         torch.save(info["best_state_dict"], ckpt_path)
    7. val prediction (numpy 회복):
         model.load_state_dict(info["best_state_dict"]); model.eval()
         with torch.no_grad():
             delta = model(torch.from_numpy(feature_fn(X_va)).float().to(device)).cpu().numpy()
         pred = baseline_val + delta  # shape (n_va, 3)
    8. fold_info = {
         "best_val_mean_eucl": float, "best_epoch": int,
         "train_duration_sec": float, "n_epochs_run": int,
       }
    return (ckpt_path, pred, fold_info)
    """
```

**`run_baseline` 의 fold loop 분기 박제**:

```python
fold_train_infos: list[dict] = []  # gru-residual 전용

for fi, (tr, va) in enumerate(folds):
    method = cfg.get("method", "polyfit")
    if method == "gru-residual":
        ckpt_path, pred, fold_info = _train_and_predict_residual_fold(
            X[tr], y[tr], X[va], y[va], cfg, fi, run_dir,
        )
        fold_train_infos.append(fold_info)
        log(f"fold {fi}: gru-residual best_val_mean_eucl={fold_info['best_val_mean_eucl']:.5f} "
            f"@epoch {fold_info['best_epoch']}, ckpt={ckpt_path}")
    else:
        # 기존 closed-form 분기 (polyfit / cspline / smoothing_spline) — 변경 없음
        if is_tune:
            info_partial = {}
            chosen, pred = _do_tune_and_predict(X[tr], y[tr], X[va], grid, tune_kind,
                                                 t_target=t_target, k=k, seed=seed,
                                                 info_out=info_partial)
            fold_chosen.append(chosen)
            _accumulate_fb(fb_total, info_partial)
        else:
            info_partial = {}
            pred = predict_for_config(X[va], cfg, info_out=info_partial)
            _accumulate_fb(fb_total, info_partial)

    oof_preds[va] = pred
    s = summarize(pred, y[va])
    s["fold"] = fi
    if method != "gru-residual" and is_tune:
        s["chosen_per_axis"] = [list(c) if not isinstance(c, (int, float)) else c for c in chosen]
    fold_metrics.append(s)
```

- `predict_for_config`, `_is_tune`, `_tune_grid`, `_do_tune_and_predict`, `_accumulate_fb` 모두 변경 없음 (= polyfit / cspline / smoothing_spline 분기 전부 후방호환 보존).
- `cfg["method"] == "gru-residual"` 시 `is_tune` 평가 안 함 (gru-residual 은 tune 개념 X — fold 별 model 학습 자체가 tune 대신).

**summary.json 추가 키** (method == "gru-residual" 시에만):
- `model_config`: dict — `{hidden, layers, dropout, lr, weight_decay, batch, epochs, early_stop_patience, loss_type, input_dim}`
- `feature_components`: list[str] (예: ["relative"], ["relative", "physics", "wingbeat"])
- `baseline_type`: str ∈ {"linear", "ema"}
- `ema_alpha`: float (baseline_type == "ema" 시에만 채움; 그 외 null)
- `wingbeat_n_bins`: int (feature_components 에 "wingbeat" 포함 시; 그 외 null)
- `fold_best_val_mean_eucl`: list[5]
- `fold_best_epoch`: list[5]
- `fold_train_duration_sec`: list[5]
- `train_device`: str ∈ {"cuda:0", "cpu"} (CUDA 가용 시 항상 "cuda:0" — 0번 GPU 강제; CPU fallback 시 "cpu")
- `total_train_duration_sec`: float = sum(fold_train_duration_sec)

### §4.6 tests/ (G0 시점 신규 3 파일 — c6; `tests/test_combine.py` 1 파일은 c12 시점 (= src/combine.py 생성과 동일 commit) 추가)

`tests/test_residual_gru.py`:
- forward shape: input (8, 11, 3) → output (8, 3).
- input_dim 가변: input (8, 11, 13) → output (8, 3); input (8, 11, 22) → output (8, 3).
- gradient flow: 1 step backward → gradient finite.

`tests/test_features.py`:
- velocity / acceleration / jerk 합성 검증.
- curvature 합성 (등속 직선 κ=0; 원운동 κ ≈ 1).
- wingbeat_fft: 합성 sinusoid → 예상 magnitude.
- **`make_feature_fn`**: 4 조합 (`["relative"]`, `["relative", "physics"]`, `["relative", "wingbeat"]`, `["relative", "physics", "wingbeat"]`) 각각 input_dim 3/13/12/22 출력 검증 + 결과가 정적 함수 (relative_coords_feature, physics_feature, wingbeat_feature) 와 비트 동일.

`tests/test_ema_extrapolate.py`:
- linear motion → ema_extrapolate ≈ linear_extrap.
- alpha=0/1 edge cases.
- finite output 보장.
- **`linear_extrap` ↔ B001 비트 동등성**: random X (n=100, 11, 3) 에 대해 `linear_extrap(X)` 와 `predict(X, window=2, degree=1, t_target=80, timesteps=TIMESTEPS_MS)` 의 절대오차 < 1e-9 — *lean baseline = plan-001 best 와 동일 함수* 비트 단위 검증.

`tests/test_combine.py` (신규 v5):
- `identify_winning(R001_summary, R00x_summaries) → list[str]` (winning 식별 기준 §3.4): synthetic summary dict (cv_mean_eucl 인위 설정) 4 케이스 — (a) 0개 winning, (b) 1개 (R002) winning, (c) 2개 (R002+R003) winning, (d) 4개 모두 winning.
- `build_r006_config(winning_components: list[str], r001_config: dict) → dict`:
  - 0개 winning → R001 config 와 비트 동일 (단 exp_id 만 R006_combined-winners 으로 변경).
  - R002 winning → `feature_components` 에 "physics" 포함 + `model.input_dim = 13`.
  - R003 winning → `baseline_type: ema, ema_alpha: 0.5`.
  - R004 winning → `feature_components` 에 "wingbeat" 포함 + `wingbeat_n_bins: 3`.
  - R005 winning → `loss_type: mse`.
  - R002 + R004 winning → `feature_components: ["relative", "physics", "wingbeat"], input_dim: 22`.

### §4.7 종료 조건 (G0)

- `pytest -q tests/` exit 0 (기존 테스트 + c6 의 신규 3 파일 — `tests/test_combine.py` 는 c12 에서 추가되므로 G0 시점 미존재).
- `python -c "import torch; from src.models.residual_gru import ResidualGRU; from src.training.train_residual import train_fold, make_feature_fn; print('ok', torch.cuda.is_available())"` 성공. (`src.combine` import sanity 는 c12 / G3.5 에서 별도 검증 — G0 시점 src/combine.py 미존재)
- backward-compat smoke: B001~B004, S001~S004 의 8 config 재실행 → cv_mean_eucl 4자리 일치 (`abs(new - old) < 1e-4`). 위반 시 `backward_compat_drift` severe.

---

## §5. STAGE 1 — Lean residual-GRU baseline (G1)

### §5.1 R001_baseline-residual-gru

| 항목 | 값 |
|---|---|
| type | baseline |
| baseline_id | B001_linear-2pt |
| 단일 변경 변수 | closed-form polyfit(w=2, d=1) → linear-extrap (B001 식 동일) + ResidualGRU(input_dim=3, hidden=64, layers=2) Huber + relative coords |
| method | gru-residual |
| feature_components | ["relative"] (input_dim=3) |
| baseline_type | linear (= B001 식: pred_baseline = X[:,-1] + 2·(X[:,-1] - X[:,-2])) |
| loss_type | huber |
| 기대 runtime | GPU < 5 min, CPU < 30 min |
| 성공 기준 | summary 기록, 5 fold ckpt 저장, cv_mean_eucl finite, B001 paired Δ ≤ +0.005 |
| 가설 | residual 학습이 valid 한가? (R001 = 다른 모든 R00x 의 reference) |

config 예시:
```yaml
exp_id: R001_baseline-residual-gru
type: baseline
plan_id: 003
method: gru-residual
feature_components: [relative]
baseline_type: linear
loss_type: huber
model:
  hidden: 64
  layers: 2
  dropout: 0.1
  input_dim: 3
training:
  lr: 1.0e-3
  weight_decay: 1.0e-4
  batch: 64
  epochs: 100
  early_stop_patience: 10
  seed: 42
t_target: 80
k: 5
seed: 42
baseline_id: B001_linear-2pt
```

### §5.2 G1 종료 조건

- `runs/baseline/R001_baseline-residual-gru/{summary.json, history.json, run.log, config.snapshot.yaml, ckpt/fold{0..4}.pt}` 모두 존재.
- registry 1행 append. cv_mean_eucl finite.
- B001 paired Δ ≤ +0.005 (위반 시 `residual_no_convergence` severe).

---

## §6. STAGE 2 — Component ablation A (R002, R003) (G2)

### §6.1 R002_physics-features

| 항목 | 값 |
|---|---|
| baseline_id | R001_baseline-residual-gru |
| 단일 변경 변수 | feature_components ["relative"] (input_dim=3) → ["relative", "physics"] (input_dim=13: rel 3 + vel 3 + acc 3 + jerk 3 + curvature 1) |
| 다른 모든 항목 | R001 동일 |
| 가설 | H1 — 고차 미분 피처가 GRU 의 residual 학습 표현력 보강 → R001 cv ↓ |

### §6.2 R003_ema-extrapolate

| 항목 | 값 |
|---|---|
| baseline_id | R001_baseline-residual-gru |
| 단일 변경 변수 | baseline_type linear → ema (alpha=0.5) |
| 다른 모든 항목 | R001 동일 |
| 가설 | H2 — EMA 가중 속도가 linear 보다 unbiased → residual 분포 좁아짐 → cv ↓ |

### §6.3 G2 종료 조건

- R002, R003 의 `summary.json` + 5 fold ckpt + registry 2행 추가.
- 모든 cv_mean_eucl finite.
- R001 paired Δ 표 산출 가능.

---

## §7. STAGE 3 — Component ablation B (R004, R005) (G3)

### §7.1 R004_wingbeat-oscillation

| 항목 | 값 |
|---|---|
| baseline_id | R001_baseline-residual-gru |
| 단일 변경 변수 | feature_components ["relative"] → ["relative", "wingbeat"] (input_dim=12: rel 3 + per-axis FFT n_bins=3 → 9 features) |
| 다른 모든 항목 | R001 동일 |
| 가설 | H3 — 11pt 좌표 미세 떨림 (FFT magnitude) 이 비행 상태를 implicit class 로 노출 |

### §7.2 R005_loss-mse

| 항목 | 값 |
|---|---|
| baseline_id | R001_baseline-residual-gru |
| 단일 변경 변수 | loss_type huber → mse |
| 다른 모든 항목 | R001 동일 |
| 가설 | H4 — Huber outlier robustness 가 본 데이터에 *실제로 작용* 하는지 |

### §7.3 G3 종료 조건

- R004, R005 의 `summary.json` + ckpt + registry 2행 추가.
- 모든 cv_mean_eucl finite.

---

## §8. STAGE 4 — Winning-Components Combined Train + Best-1 자율 LB (G3.5 + G_final)

본 stage 의 설계 원리: ablation 결과를 *목적성 있게 활용* — winning component 들을 모두 합쳐 final model R006 학습. additive 가정이 깨질 경우 fallback (R001 LB 제출). LB 1회만 자율 회수.

### §8.1 Winning identification + R006 자동 학습 (c12)

`src/combine.py` 신규:

```python
def identify_winning(r001_summary: dict, r00x_summaries: dict[str, dict],
                     noise_margin: float = 0.0) -> dict[str, bool]:
    """R002~R005 각각에 대해 paired mean Δ < -noise_margin 이면 winning.
    return: {"R002": bool, "R003": bool, "R004": bool, "R005": bool}
    """
    r001_cv = r001_summary["cv_mean_eucl"]
    return {
        exp_short: (s["cv_mean_eucl"] - r001_cv) < -noise_margin
        for exp_short, s in r00x_summaries.items()
    }

def build_r006_config(winning: dict[str, bool], r001_config: dict) -> dict:
    """winning 결과로 R006 config 생성 (R001 base + 변경 axes).
    component-axis mapping:
      R002 winning → feature_components += ["physics"]
      R003 winning → baseline_type = "ema", ema_alpha = 0.5
      R004 winning → feature_components += ["wingbeat"], wingbeat_n_bins = 3
      R005 winning → loss_type = "mse"
    input_dim 재계산 = 3 + 10*("physics" in fc) + 9*("wingbeat" in fc)
    return: dict (= R006 config; exp_id 강제 = "R006_combined-winners",
                  baseline_id = "R001_baseline-residual-gru")
    """
```

c12 commit 워크플로:
1. `src/combine.py` 작성.
2. R001~R005 summary.json 5개 읽음.
3. `identify_winning(R001, {R002, R003, R004, R005})` → winning dict.
4. `analysis/plan-003/winning_trace.md` 작성 (5 행 표: exp_id, R001.cv, R00x.cv, Δ, winning yes/no).
5. `build_r006_config(winning, R001_config)` → `configs/baseline/R006_combined-winners.yaml` 자동 작성.
6. **분기 — winning 0개 시**:
   - 학습 SKIP. R001 의 ckpt + summary 그대로 R006 디렉토리에 *cp* 또는 symlink (기록 일관성 위해 cp 권장).
   - `runs/baseline/R006_combined-winners/{summary.json (R001 복제 + exp_id 만 R006_combined-winners 으로 변경; baseline_id 는 R001 의 값 = "B001_linear-2pt" 그대로 유지 — 실제 baseline 함수 동일성 박제), history.json, ckpt/fold{0..4}.pt (R001 복제)}` 생성.
   - registry append (R006 행, notes = "winning=0, copied from R001").
   - decision-note: spec-default — winning 0개, R006 = R001 직접 복제. summary.json 의 baseline_id 는 R001 와 동일 ("B001_linear-2pt") — 실제 baseline 함수가 같기 때문 (build_r006_config 의 baseline_id="R001_..." 강제는 winning ≥ 1 분기 전용).
7. **분기 — winning ≥ 1개 시**:
   - `python -m src.run configs/baseline/R006_combined-winners.yaml` 실행 (= 일반 학습 경로).
   - run_baseline 의 gru-residual 분기로 학습 → ckpt 5개 저장 + summary.json + registry append.
8. R006 cv_mean_eucl 와 R001 cv_mean_eucl 비교:
   - R006.cv > R001.cv + 0.001 → fallback 플래그 set + `combined_no_improvement` warn 박제 (analysis/plan-003/winning_trace.md 끝에 fallback 사유 추가).
   - 그 외 → fallback false.

c12 commit msg 예시:
```
plan-003 c12: R006_combined-winners 학습 (winning={R002, R003}, input_dim=13, baseline=ema)

- R001 cv: 0.01XXX, R002 cv: 0.01YYY (winning), R003 cv: 0.01ZZZ (winning),
  R004 cv: 0.01AAA (non-winning), R005 cv: 0.01BBB (non-winning)
- R006 config 자동 생성: feature_components=[relative, physics], baseline_type=ema, ema_alpha=0.5, loss=huber
- R006 cv: 0.01CCC (vs R001 Δ=-0.000DD, < 0 → fallback false)
- 5 fold ckpt 저장, registry append

decision-note: spec-default — winning 기준 paired mean Δ < 0 (noise_margin=0)
decision-note: spec-default — additive 가정으로 winning 모두 합치기 (interaction 검증은 별도 plan)
```

### §8.2 lb_exp_id Submission 생성 (c13)

1. **`src/submit.py` 확장**:
   - 기존 polyfit/cspline/smoothing_spline 분기 → 변경 없이 동작 보장 (B001~B004, S001~S004 후방호환).
   - `"gru-residual"` 분기 신규: 5 fold ckpt 로드 → 각 ckpt 별 test 10k 예측 → 5 fold mean (ensemble). baseline (linear or ema) 는 numpy 로 산출 후 ckpt 별 잔차 + baseline → 5 예측 → mean.
2. **lb_exp_id 결정** (autonomous, decision-note 박제):
   - fallback 플래그 false → lb_exp_id = "R006_combined-winners"
   - fallback 플래그 true → lb_exp_id = "R001_baseline-residual-gru" + 사유 박제
3. test 예측 + submission.csv 생성 (분기별 절차):
   - **분기 (a) winning ≥ 1 + fallback false (lb_exp_id = R006)**: R006 의 5 fold ckpt 로 test 예측 → `runs/baseline/R006_combined-winners/submission.csv` 생성. R001 등 다른 exp 는 test 예측 / csv 생성 *skip*.
   - **분기 (b) winning ≥ 1 + fallback true (lb_exp_id = R001)**: R001 의 5 fold ckpt 로 test 예측 → `runs/baseline/R001_baseline-residual-gru/submission.csv` 생성. R006 디렉토리에는 submission.csv 생성 *skip* (registry R006 행은 lb_score 미박제, R001 행에만 박제 — §8.3 와 일관).
   - **분기 (c) winning = 0 (R006 = R001 cp 분기, lb_exp_id = R006)**: 우선 R001 의 5 fold ckpt 로 test 예측 → `runs/baseline/R001_baseline-residual-gru/submission.csv` 생성 (먼저). 그 다음 `runs/baseline/R006_combined-winners/submission.csv = cp(R001/submission.csv)` (비트 동일). 두 파일 모두 존재해야 c13 종료.
   - 어느 분기든 `runs/baseline/{lb_exp_id}/submission.csv` 가 c14 의 dacon-submit skill 인자로 사용되는 *유일한* 파일.
4. **스키마 assert** (plan-002 §8.1 동일):
   - `rows == 10000`, `columns == ["id", "x", "y", "z"]`, NaN/Inf 0건, dtype float64, id-set match.
   - 위반 시 `submission_schema_fail` severe.
5. c13 commit 에 lb_exp_id 의 submission.csv + 결정 사유 (decision-note: lb_exp_id={...}, fallback={true/false}, 사유) 박제.

### §8.3 lb_exp_id 자율 LB 제출 + 회수 (c14) — *의무, 사용자 승인 X*

> **dacon-submit skill 호출은 autonomous loop 가 사용자 confirm 없이 자율 실행. CLAUDE.md 의 autonomous policy 적용.**

- **`dacon-submit` skill 1회 자율 호출**:
  ```
  Skill(skill="dacon-submit",
        args="runs/baseline/{lb_exp_id}/submission.csv {lb_exp_id} 'plan-003 {R006 combined or R001 fallback}'")
  ```
- skill 응답으로 LB 점수 회수 → `analysis/plan-003/lb_log.md` 1행 기록 + registry 의 lb_exp_id 행 `notes` 컬럼에 `lb_score=0.XX` + `plans/plan-003-residual-gru-grid.results.md` frontmatter `lb_exp_id`, `lb_score`, `combined_winning_components`, `combined_fallback` 박제.
- skill 부재 시: `dacon_submit_skill_missing` severe → telegram alert + 사용자 escalate.
- fallback (skill 호출 자체가 막힌 경우): plan-002 §8.2 fallback 동일 (사용자 수동 업로드).

**Budget 운영**: 본 plan 은 1 슬롯만 필요. 5/일 budget 잔여 1+ 이면 즉시 제출. 잔여 0 이면 다음 일자 carry-over (status `partial` → `all_complete`).

**기록 위치**:
- `analysis/plan-003/lb_log.md`: 1행 표 (`lb_exp_id | submitted_at (KST) | lb_score | submission_filename | combined_fallback (true/false) | winning_trace_summary`).
- registry 의 lb_exp_id 행 `notes` 컬럼: `lb_score=0.XX`.
- `plans/plan-003-residual-gru-grid.results.md` frontmatter: 아래 §8.4 schema 참조.

**G_final 진입 차단 조건**:
- `lb_score` 미회수 → `lb_unsubmitted` severe.

### §8.4 Results.md 산출 (c15)

- `analysis/plan-003/results.md` 본문:
  - 종합 표: 7 행 (B001 + R001~R006) × (method, key hp/component, cv_mean_eucl ± std, per-axis MAE, hit_rate@{0.05,0.10,0.20,0.50}, **lb_score (lb_exp_id 만)**, training time).
  - per-experiment 분석 (plan-001/002 results 형식 동일).
  - **B001 vs R001~R006** paired comparison: same-fold Δ.
  - **R001 vs R002~R005** paired comparison: 같은 형식 (= winning 식별 근거 표).
  - **R001 vs R006** paired comparison: combined 효과 측정.
  - **Winning trace**: §3.4 의 winning 기준 적용 결과 (R002/R003/R004/R005 각각 winning yes/no + Δ + R006 자동 생성 config 요약).
  - 학습 안정성: fold 별 best_val_mean_eucl 분포 + early-stop 발동 epoch 분포.
  - H1~H4 검증/기각 (CV 축).
  - **H5 verdict (combined-additive)**: R006 cv vs (R001 cv + Σ winning Δ). interaction effect 추정 = R006.cv - (R001.cv + Σ winning Δ). 부호와 크기 분석.
  - **lb_exp_id 위치**: lb_score 와 B001 LB=0.60, S001 LB=0.49 와의 비교. neural 모델이 closed-form floor 를 넘는지 박제.
  - 다음 plan 후보 *enumeration only*: GRU hyperparameter sweep, TCN/Transformer 비교, **R001~R005 ablation 의 LB 신호 회수 (= 5 LB 제출 plan)**, ensemble R00x + B001, Kalman 전처리, hit-rate aware loss, TTA inference, **interaction effect 분리 측정 (R002+R004 vs R002 only vs R004 only LB 비교 plan)**.

- `analysis/plan-003/lb_log.md`: 1행 표 + lb_exp_id 결정 trace.

- `analysis/plan-003/winning_trace.md`: §8.1 step 4 + step 8 의 winning 식별 + R006 cv 비교 결과 박제.

- `plans/plan-003-residual-gru-grid.results.md`: WORKFLOW.md §6 frontmatter:
  ```yaml
  plan_id: 003
  finished_at: ... (KST)
  status: all_complete  # (또는 partial)
  exp_ids_completed: [R001_..., R002_..., R003_..., R004_..., R005_..., R006_combined-winners]
  exp_ids_skipped: []
  best_exp_id_cv_ablation: {best of R001..R005}  # cv_mean_eucl argmin
  combined_winning_components: [<winning short names>]  # 예: ["R002", "R003"] or []
  combined_fallback: true | false  # R006.cv > R001.cv + 0.001 시 true
  lb_exp_id: R006_combined-winners | R001_baseline-residual-gru
  lb_score: 0.XX
  lb_submission_path: runs/baseline/{lb_exp_id}/submission.csv
  lb_metric: hit_rate_at_1cm
  lb_submitted_at: ... (KST)
  train_device: cuda:0  # enum {"cuda:0", "cpu"} — §4.5 summary.json 의 train_device 값과 동기화
  total_train_time_sec: ...  # R001~R006 합산
  ```
  본문: 각 exp 의 (status, started_at, duration, 핵심 metric, best path, baseline diff vs B001 + vs R001, 특이사항). **lb_score 는 lb_exp_id 행에만**.

### §8.5 G_final 종료 조건 (의무 list)

- 6 exp summary.json (R001~R006) + 6 fold ensemble ckpt set (R006 winning 0 시 R001 복제로 충당) + registry 6행 + lb_exp_id 1 submission.csv (스키마 검증 통과) + **lb_score** + analysis/plan-003/{results.md, lb_log.md, winning_trace.md} + plans/plan-003-residual-gru-grid.results.md 모두 commit.
- §0.5 의 모든 [TODO] → [DONE] 마킹 (commit hash 포함).
- lb_score 회수 완료. 미회수 시 `lb_unsubmitted` severe.

---

## §N+1. 작업량 회계

| 단위 | 수 |
|---|---|
| code commit (c1~c5) | 5 (model, baselines/extrap, features, training, run.py 확장) |
| test commit (c6) | 1 (4 신규 테스트 파일 묶음 — combine 테스트 포함) |
| exp commit (c7~c11) | 5 (R001~R005 ablation) |
| sub-combined-train commit (c12) | 1 (R006 winning 식별 + config 자동 생성 + 학습 또는 R001 복제) |
| sub-gen commit (c13) | 1 (lb_exp_id submission.csv) |
| sub-lb commit (c14) | 1 (LB 점수 + lb_log.md, dacon-submit skill 자율 호출) |
| docs commit (c15) | 1 |
| **총 commit** | **15** |
| G-gate | 6 (G0, G1, G2, G3, G3.5, G_final) |
| 학습 시간 (예상) | GPU 가용: 6 exp (R001~R006) × ~5 min = 30 min (R006 = R001 복제 시 단축). CPU: 6 × ~30 min = 3 hr (epochs 50 자동 감소 시 1.5 hr). 추론 (lb_exp_id fold ensemble) +수 분 |
| LB 제출 시간 | 1회 × dacon-submit skill latency (~수십 초) |

---

## §N+2. results.md 필수 항목

| 항목 | 내용 |
|---|---|
| frontmatter | `plan_id=003, finished_at, status, exp_ids_completed, exp_ids_skipped, best_exp_id_cv_ablation, combined_winning_components, combined_fallback, lb_exp_id, lb_score, lb_submission_path, lb_metric, lb_submitted_at, train_device, total_train_time_sec` |
| 본문 per exp | 상태, started_at, duration, cv_mean_eucl±std, per-axis MAE, hit_rate@4 radii, fold_best_val_mean_eucl 분포, fold_train_epochs 분포, best run dir path, baseline diff vs B001 + vs R001, 특이사항. **lb_score 는 lb_exp_id 한 행에만** |
| 종합 표 | 7 행 (B001 + R001~R006) × (method, component, cv_mean_eucl, per_axis_mae, hit@0.10, **lb_score (lb_exp_id 만)**, training time) |
| paired comparison (CV) | (a) B001 vs R001~R006 same-fold Δ, (b) R001 vs R002~R005 same-fold Δ (winning 근거), (c) R001 vs R006 (combined 효과) |
| H1~H5 verdict | H1~H4 (ablation) 각 가설별 *CV* 검증/기각/부분기각. H5 (combined-additive) 검증: R006 cv vs (R001 cv + Σ winning Δ) 분석 (interaction effect 추정 + 부호 + 크기) |
| Winning trace | §3.4 winning 기준 적용 결과 표 (R002/R003/R004/R005 각각 winning yes/no + Δ + R006 자동 생성 config 요약) |
| Combined fallback | R006.cv > R001.cv + 0.001 시 fallback 발동 사유 박제. fallback false 시 R006 의 ablation 합산 정량 효과 |
| lb_exp_id 위치 | lb_score 와 B001 LB=0.60, S001 LB=0.49 와의 비교. neural 모델이 closed-form floor 를 넘는지 박제 |
| 학습 안정성 | fold 별 train loss 곡선 요약, early-stop 발동 분포, OOM/NaN 사고 0건 확인 |
| submission 결과 | lb_exp_id 1개만 제출, LB 1 점수, dacon-submit skill 자율 호출 trace |
| 다음 plan 후보 | enumeration only — GRU hyperparameter sweep, TCN/Transformer 비교, R001~R005 ablation 의 LB 신호 회수, ensemble, Kalman 전처리, hit-rate aware loss, TTA inference, **interaction effect 분리 측정 (R002+R004 vs single)** 등 |

---

## §N+3. 통계 함정 & caveats

1. **paired Δ 가 fold-σ 보다 작으면 noise** — 5 fold 만으로는 ±0.0006 영역의 차이는 noise. results.md 에 fold-σ 와 함께 명시. plan-002 §N+3 #1 동일.
2. **GRU 의 random seed 의존성** — 모든 exp 가 seed=42 + fold_idx 로 결정론적. cuDNN non-determinism 으로 ±0.001 영역 변동 가능. decision-note: `torch.backends.cudnn.deterministic=True` 채택.
3. **fold ensemble vs full-train 재학습** — 본 plan default = fold ensemble. ablation 으로 검증 안 함 (별도 plan).
4. **wing-beat FFT (R004) 의 frequency resolution** — 11pt × 40ms = 440ms 윈도우 → bin 폭 ≈ 2.27 Hz, max freq = 12.5 Hz. 모기의 wing-beat (~수백 Hz) 는 *aliasing* 으로 entry 부재. 따라서 R004 의 "wing-beat" 라벨은 *물리적 wing-beat 가 아닌* 11pt 의 저주파 oscillation 패턴 (거시적 trajectory 떨림). 결과 해석 시 명시.
5. **physics feature (R002) 의 미분 노이즈 amplification** — 입력 좌표 노이즈 σ ≈ 0.005 m → velocity σ ≈ 0.125 m/s → acceleration σ ≈ 3.1 m/s² → jerk σ ≈ 78 m/s³. 고차 미분일수록 SNR 악화. GRU 가 노이즈 dominant feature 를 무시 학습할 수 있는지가 H1 의 본질.
6. **EMA alpha=0.5 의 자의성** — alpha 자체가 hyperparameter. 본 plan 은 0.5 fixed (별도 plan 에서 sweep).
7. **R005 의 Huber δ 자의성** — PyTorch HuberLoss default δ=1.0. 잔차 분포가 ~수 mm 라 δ=1.0 는 거의 모든 잔차를 quadratic 영역으로 처리 → 사실상 MSE 와 비슷할 가능성. R005 와 R001 차이가 작으면 그 자체가 H4 의 결론 (Huber prior 약함).
8. **LB metric 의 잡음** — plan-002 §N+3 #8 동일. LB 차이가 0.005 영역보다 작으면 noise.
9. **5/일 budget 운영** — 본 plan 외 제출 발생 시 슬롯 미확보 가능 (단 본 plan 은 1 슬롯만 필요).
10. **CV winner ≠ LB winner 측정 불가** — 본 plan 은 1 LB 만 회수 → CV 와 LB 의 ranking divergence *직접 측정 불가*. plan-001/002 의 5점 + 본 plan 의 1점 = 통합 6점 으로만 산점 분석 가능. 다음 plan (5 LB 제출 plan) 으로 회수.
11. **Architecture 통일의 tradeoff** — 모든 R00x 가 동일 GRU(64,2,0.1) → component ablation 의 *순수* effect 측정 가능. 그러나 실제 LB 최적은 다른 architecture 가능 (별도 plan).
12. **TTA 보류의 의미** — 본 plan 에서 TTA 는 *명시적으로 제외*. 별도 plan 에서 후처리만으로 추가 검증 가능.
13. **best 1 LB 의 정보량 한계** — best 1 LB 점수는 *그 exp 의 LB 위치만* 박제. 다른 5 exp 의 LB 는 미회수 → component 별 LB 효과 측정 불가. 본 plan 의사결정 anchor 로서의 가치는 (a) neural 모델이 closed-form floor 를 넘었는지, (b) winning component 조합이 다음 plan 의 reference 로 valid 한지 *2가지로 한정*.
14. **Additive 가정 (H5) 의 한계 — interaction effect** — R006 의 winning component 합산은 *additive* 가정 (각 component 효과가 독립적이며 합산 가능). 실제로는 interaction 발현 가능 (예: physics feature + EMA baseline 가 *함께* 적용 시 baseline 이 이미 unbiased 라 physics 의 추가 정보가 redundant → 합산 효과 < 단순 합). H5 검증 = R006.cv vs (R001.cv + Σ winning Δ) 비교. interaction 분리 측정 (각 조합별 ablation) 은 별도 plan — 4 winning 가능 시 2^4=16 조합 폭증.
15. **Winning 기준의 보수성** — paired mean Δ < 0 단순 부등호 (statistical significance 미검정). |Δ| 가 fold-σ 영역인 경우 false positive winning 가능 → R006 에 무의미한 component 가 포함되어 학습 표현력 분산. 본 plan default 는 보수적 (noise_margin=0). strict mode (|Δ| ≥ fold-σ + 부호 일관성 ≥ 4/5) 는 별도 plan.
16. **Combined fallback 의 정보 가치** — fallback 발동 (R006.cv > R001.cv + 0.001) 자체가 *interaction effect 발현* 의 강한 신호. fallback 발동 시 다음 plan 들에서 *조합 검증 필수* anchor. fallback false 라도 H5 의 정량 검증 (interaction estimate) 은 results.md 에 박제.

---

## §N+4. 변경 이력

- v1 (2026-05-10): 초안. plan-001/002 결과 인계 + 사용자 의도서 + `notes/mosquito-trajectory-ideas.md` 의 6 component 중 5 채택 → R001 lean baseline + 5 ablation = 총 6 exp. 6 LB 제출 의무화.
- v2 (2026-05-10): TTA (구 R004) 사용자 결정으로 삭제 → wing-beat → R004, loss-mse → R005. 총 5 exp. 5 LB 제출 의무화.
- v3 (2026-05-10): 사용자 결정으로 LB 제출 정책 *5 모두 제출* → **best 1개만 자율 제출** 로 변경. dacon-submit skill 사용자 승인 없이 자율 실행. frontmatter `lb_scores` dict (5 키) → `best_lb_score` + `best_exp_id_lb` 단일 필드.
- v4 (2026-05-10): "추론 없이 코드 작성 가능" 기준 충족 위해 5 ambiguity 박제 — `_train_and_predict_residual_fold` 시그니처, HuberLoss δ=1.0 + DataLoader spec, normalization 정책, .gitignore 패턴, predict_for_config 분리. summary.json 추가 키 명시.
- v5 (2026-05-10): 사용자 제안으로 LB 제출 정책 *best 1 ablation 직접 제출* → **winning components 들을 합쳐 R006 자동 학습 + R006 (또는 fallback R001) 1 LB 제출** 로 변경. 핵심 추가: (a) `src/combine.py` 신규 (winning 식별 + R006 config 자동 생성), (b) `src/training/train_residual.py` 에 `make_feature_fn(components)` factory 추가 (R006 의 dynamic feature 합산), (c) STAGE 4 를 G3.5 (R006 학습) + G_final (LB 회수) 로 분리, (d) c12 sub-combined-train + c13 sub-gen + c14 sub-lb commit 분리, (e) `combined_no_improvement` warn (severe X) + fallback 메커니즘 (R006.cv > R001.cv + 0.001 시 R001 csv 를 LB), (f) winning 0개 시 R006 = R001 직접 복제 (학습 skip), (g) H5 (combined-additive) 가설 추가 + interaction effect caveat #14, (h) winning 기준 보수성 caveat #15 + combined fallback 의 정보 가치 caveat #16. frontmatter `best_lb_score` → `lb_exp_id` + `lb_score` + `combined_winning_components` + `combined_fallback`. 본 v5 의 R006 자동 생성 + fallback 메커니즘은 ablation 의 *목적성* (개별 검증 → 조합 selection) 명확화 + interaction effect 검증 anchor 박제.
- v6 (2026-05-10): 사용자 요청으로 GPU device = **"cuda:0" 강제 박제** (다중 GPU 환경에서도 0번 GPU 만 사용해 결과 reproducibility 보장; CUDA_VISIBLE_DEVICES 환경변수 의존 X — 코드에서 device 문자열 명시). §0.5 decision-note (학습 device 라인) + §4.4 train_fold default device + §4.4 train_fold spec 의 DataLoader pin_memory 조건 (`device=="cuda"` → `device.startswith("cuda")`, "cuda:0" 도 매칭) + summary.json `train_device` 값 enum (`{"cuda", "cpu"}` → `{"cuda:0", "cpu"}`) 동기화. PyTorch `torch.cuda.is_available()` / `torch.cuda.manual_seed_all(seed)` 호출은 그대로 (cuda 모듈 함수, device 문자열 무관).

---

## §N+5. 참조

- `plans/plan-001-polyfit-baseline.md` + `.results.md` (선행 plan: closed-form polyfit floor B001=0.01294, LB=0.60. **본 plan 의 lean baseline 외삽 함수 = B001 식 그대로**)
- `plans/plan-002-cubic-spline.md` + `.results.md` (선행 plan: closed-form cspline 4 변형 + LB 4 점수)
- `notes/mosquito-trajectory-ideas.md` (component 선정 근거 — main §3 (Huber), supp §1 (physics), supp §3 (oscillation), supp §5 (residual prediction), supp §7 (EMA))
- `notes/competition-overview.md` (대회 사양, +80 ms target, hit-rate metric)
- `WORKFLOW.md` §4 (4-way token), §5 (plan obligations), §6 (results), §7 (run dir), §11 (handoff), §12 (autonomous protocol)
- `CLAUDE.md` (autonomous execution policy — c14 의 *사용자 승인 없는 dacon-submit 자율 호출* 의 권한 근거 + c12 의 *winning 자동 식별 + R006 자동 학습* 의 자율 결정 근거)
- 데이터: `data/{train,test}/*.csv` (각 11 row), `data/train_labels.csv`, `data/sample_submission.csv`
- 코드 인계: `src/io.py` (kfold_split deterministic, TIMESTEPS_MS, T_TARGET_MS), `src/eval.py` (summarize), `src/run.py` (method dispatch 확장 대상), `src/submit.py` (gru-residual 분기 + lb_exp_id 결정 추가 대상), `src/baselines/window_polyfit.py` (B001 식 — *linear_extrap 의 비트 동등성 검증 reference*), `src/baselines/cubic_spline.py` (cspline 분기 후방호환 보존 대상)
- 외부 라이브러리: PyTorch ≥ 2.0 (`torch`, `torch.nn`, `torch.optim.AdamW`, `torch.nn.HuberLoss`, `torch.nn.MSELoss`), `numpy.fft` (R004 FFT), 기존 `scipy.interpolate` (cspline 유지)
- skill: `dacon-submit` (c14 의 자율 호출 대상; plan-002 §8.2 에서 이미 사용된 skill — 명세 동일)
