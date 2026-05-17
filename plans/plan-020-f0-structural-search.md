---
plan_id: 020
version: 1
date: 2026-05-18 (Asia/Seoul)
status: draft
based_on:
  - 004 (fold split + 18-regime infrastructure + 27-pool reference)
  - 006 (F0 baseline 0.6320 / 0.8033 산식 — plan-006 `frenet_par120_perp_neg020`)
  - 007 (CMA-ES infrastructure + F002 per-sample MLP coef precedent)
  - 014~016 (corrector paradigm ceiling 측정 — F0 family 한계 박제)
  - 017 (GRU-attention coef regressor — N3 overlap, plan-020 N1/N2/N5 와 직교 선택)
followed_by: []
scope: F0 (단일 공식, plan-006 frenet_par120_perp_neg020) 의 단독 hit@1cm / hit@1.5cm 갱신 17 후보 ablation (14 deterministic + 3 NN). plan-004 pipeline 통합 / dacon-submit / BMA = out-of-scope (follow-up plan).
exp_ids:
  - Z020_C01_helix
  - Z020_C02_ctra
  - Z020_C03_ctrv
  - Z020_C04_imm
  - Z020_C05_per_regime_f0
  - Z020_C06_quintic_hermite
  - Z020_C07_jerk_quartic
  - Z020_C08_singer
  - Z020_C09_kalman_smoother
  - Z020_C10_bishop_frame
  - Z020_C11_se3_twist
  - Z020_C12_wingbeat_corrected
  - Z020_C13_levy_prior
  - Z020_C14_trajectory_knn
  - Z020_N01_mlp_coef
  - Z020_N02_tcn_coef
  - Z020_N05_moe_f0
lb_score: null
band: null
---

# plan-020 v1 — F0 Structural Search: 17 후보 (14 deterministic + 3 NN) 의 단독 hit@1cm / hit@1.5cm 갱신

## §0. 한 줄 목적

> **F0 baseline** (plan-006 `frenet_par120_perp_neg020`, d1=1.98 / par=1.20 / perp=-0.20, **단독 hit@1cm = 0.6320, 단독 hit@1.5cm = 0.8033**) 을 **structurally 다른 17 후보 — 14 deterministic 공식 + 3 NN-coefficient predictor** 의 *단독* 5-fold OOF 측정으로 갱신. plan-006/007 의 single-formula ceiling (OOF ~0.65, LB ~0.67) 안에 갇혀 있던 paradigm 의 *family-level lever* 를 ablation 으로 분리 박제.
>
> **pass criteria**: 적어도 1 후보가 paired Δ ≥ +0.005 on *둘 다* (hit@1cm + hit@1.5cm). 0 통과 시 → halt 안 함, negative finding 박제 후 G_final.
>
> **out-of-scope**: plan-004 27-pool 통합 / dacon-submit / LB 측정 / BMA / IMM fusion. 전부 follow-up plan-021 (가칭) 으로 carry.
>
> **NN axis precedent**: N1 (per-sample MLP coef) = plan-007 F002 의 *일관 protocol 재측정*. N2 (TCN coef), N5 (MoE F0) = 신규. **N3 (BiGRU coef) / N4 (Transformer coef) 는 plan-017 overlap → plan-020 out-of-scope**.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 18 모듈 (baseline F0 reproduce + 14 deterministic + 3 NN) import + smoke + tests green. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] **AND** hit@1.5cm ∈ [0.8028, 0.8038]. plan-006 hard evidence ± 0.0005 정확 재현. 위반 시 `f0_reproduce_drift` severe.
- **G2.D**: 14 deterministic 후보 모두 5-fold OOF 측정 완료. NaN/Inf 0건. 각 후보의 (hit@1cm, hit@1.5cm) finite. 위반 시 `formula_numerical` severe.
- **G2.N**: 3 NN 후보 모두 5-fold OOF 측정 완료. train loss NaN/Inf 0건, val_hit > 0.10 (random baseline 통과). 위반 시 `nn_no_signal` severe.
- **G3 (family-level)**: 7 family 별 winner 선정 + 17 × 2 metric × 5-fold 결과 표 박제. **≥ 1 후보가 paired Δ ≥ +0.005 *둘 다*** (hit@1cm AND hit@1.5cm) 통과 시 G3 pass. 0 통과 = `all_negative` warn (severe X, negative finding 박제 후 G_final 진입).
- **G_final**: results.md + best 박제 + plan-017 overlap 해소 표 + follow-up plan 후보 박제. **LB 제출 의무 없음** (§0 narrative).

### G-gates

- G0: STAGE 0 인프라 [TODO]
- G1: STAGE 1 F0 baseline reproduce [TODO]
- G2.D: STAGE 2 14 deterministic 측정 [TODO]
- G2.N: STAGE 3 3 NN 측정 [TODO]
- G3: STAGE 4 family-level 분석 [TODO]
- G_final: STAGE 5 best 박제 + results [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-020-f0-structural-search.md` 본문 v1 작성 | [TODO] |
| c2 | code | `analysis/plan-020/baseline_f0.py` (plan-006 산식 1:1 재구현, reproduce-only) | [TODO] |
| c3 | code | `analysis/plan-020/formula_deterministic.py` (14 후보 산식, no learning) | [TODO] |
| c4 | code | `analysis/plan-020/formula_nn.py` (3 NN 후보 module + 학습 loop) | [TODO] |
| c5 | code | `analysis/plan-020/run_oof.py` 5-fold OOF runner (CPU + GPU 분기) | [TODO] |
| c6 | code | `analysis/plan-020/cma_es_fit.py` (annealed hit-direct objective + multi-seed) | [TODO] |
| c7 | test | `tests/test_plan020_smoke.py` 18 모듈 import + F0 reproduce | [TODO] |
| G0 | gate | smoke + tests green | [TODO] |
| c8 | exp G1 | F0 baseline 5-fold OOF reproduce → `analysis/plan-020/baseline_oof.{json,md}` | [TODO] |
| G1 | gate | F0 hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] | [TODO] |
| c9 | exp G2.D | 14 deterministic 후보 측정 → `analysis/plan-020/results_deterministic.{json,md}` | [TODO] |
| G2.D | gate | 14 후보 metric finite | [TODO] |
| c10 | exp G2.N | 3 NN 후보 학습 + 측정 → `analysis/plan-020/results_nn.{json,md}` | [TODO] |
| G2.N | gate | 3 NN metric finite + val_hit > 0.10 | [TODO] |
| c11 | analysis | family-level winner + paired Δ table → `analysis/plan-020/family_analysis.{json,md}` | [TODO] |
| G3 | gate | 17 × 2 × 5-fold table + family winner 박제 | [TODO] |
| c12 | docs | `plans/plan-020-f0-structural-search.results.md` + `analysis/plan-020/results.md` + frontmatter sync | [TODO] |
| G_final | gate | results 3-file sync + §0.5 [TODO]→[DONE] sync + follow-up 후보 박제 | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 에서 F0 reproduce 가 hit@1cm 0.6320 ± 0.0005 또는 hit@1.5cm 0.8033 ± 0.0005 밖. 추출/fold split/regime bin 버그 의심 → 즉시 halt.
- `formula_numerical`: deterministic candidate 출력에 NaN/Inf (예: helix κ=0 division, CTRA |ω| 발산). 각 candidate 의 *edge case fallback* 으로 회피.
- `nn_no_signal`: NN candidate val_hit < 0.10 (random baseline floor). architecture/normalization/loss 버그 의심.
- `nn_overfit`: NN candidate train_hit − val_hit > 0.10 (5-fold mean). regularization 부족 → dropout/weight_decay 강화 필요.
- `per_regime_overfit`: C5 per-regime F0 의 fold variance > 0.05. min sample threshold 강화 or global fallback.
- `plan017_carry_conflict`: 만약 plan-017 의 N3/N4 산출이 인계되고 plan-020 의 N1/N2/N5 와 ±0.01 이상 차이 발생 시 protocol divergence 박제 (severe X, warn only).
- `n1_drift_vs_f002`: N1 (plan-007 F002 재측정) 이 F002 의 OOF 0.6482 와 ±0.01 초과 차이. protocol 점검.
- `all_negative`: 17/17 후보 모두 paired Δ < +0.005 → halt 안 함, **negative finding 박제 후 G_final 진입** (paradigm-level evidence).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-020/**` (모듈, OOF 결과, family analysis, results.md)
  - `tests/test_plan020_smoke.py`
  - `runs/baseline/Z020_*/` (NN ckpt — `.gitignore` 적용)
- blacklist 추가:
  - plan-001~019 산출 (`runs/baseline/{B,S,R,P,D,E,F,H}*/**`, `analysis/plan-{001..019}/**`)
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — F0 baseline = plan-006 frenet_par120_perp_neg020 (d1=1.98, par=1.20, perp=-0.20). 산식 1:1 재구현, plan-007/014 import X.`
- `decision-note: spec-default — pass criteria = paired Δ ≥ +0.005 *둘 다* (hit@1cm AND hit@1.5cm). 한 metric 만 pass 면 partial 박제 후 family-level 검토.`
- `decision-note: spec-default — fold split = stable_fold_id(sample_id, 5), plan-004 carry. fold-internal regime fit (bins 누수 차단).`
- `decision-note: spec-default — CMA-ES seed list [20260518..20260522], multi-seed best-on-train → val. annealed schedule (smooth τ=0.003 → 0.001 → hard hit).`
- `decision-note: spec-default — N1 = plan-007 F002 재측정 (plan-020 일관 protocol). F002 의 OOF 0.6482 와 ±0.01 안 들어오면 protocol divergence 박제.`
- `decision-note: spec-default — N3/N4 (BiGRU/Transformer coef) = plan-017 overlap → plan-020 out-of-scope. plan-017 G_final 후 carry.`
- `decision-note: spec-default — LB 제출 = out-of-scope. plan-021 (가칭) follow-up.`
- `decision-note: spec-default — NN device = cuda:1 (project convention, plan-004 carry). deterministic 후보는 CPU.`

---

## §1. 배경

### §1.1 plan-006/007 의 single-formula ceiling 측정

| exp | plan | 산출 | 단독 OOF hit@1cm | LB |
|---|---|---|---|---|
| E001 (F0 + physics_bias) | 006 | plan-004 framework 95% 제거 + 27 후보 + physics_bias + soft avg | 0.6491 (argmax-corrected) | **0.6692** |
| F001 (CMA-ES 6 vars) | 007 | 단일 공식 6-param tune | 0.6403 (5-fold OOF) | — |
| F001 (best basis 8 vars) | 007 | 8-param tune | 0.6387 (single fit) | 0.6598 |
| **F002 (per-sample MLP coef)** | 007 | NN → (d1, par, perp) | **0.6482 (5-fold OOF)** | — |

→ **선형 family + per-sample coefficient (deterministic 또는 MLP-NN) 안 ceiling ≈ OOF 0.65 / LB 0.67**. plan-007 의 CMA-ES + MLP coef 가 *F0 보다 살짝 더* 갈 뿐.

### §1.2 F0 의 단독 measured 값 (plan-006 hard evidence)

- F0 = `p0 + 1.98·v_last + 1.20·acc_par_vec − 0.20·acc_perp_vec`
- hit@1cm = **0.6320** (10000 train 위 single-fold, plan-014 G0 reproduce 일치)
- hit@1.5cm = **0.8033**
- → 84% sample 이 F0 근방 1.5cm 안, 그 중 ~21% 가 1cm 밖 = *corrector 회수 zone*

### §1.3 NN-as-F0 의 *시도된* / *미시도* 분리

| 시도 | Plan | 형태 | 결과 | plan-020 처리 |
|---|---|---|---|---|
| NN → 3D coord 직접 회귀 | 003 R001-R006 | regression | LB 0.5688 (실패) | 재시도 X (paradigm 함정) |
| Per-sample MLP F0 coef | 007 F002 | NN → (d1, par, perp) | OOF 0.6482 ≈ global F0 | **N1 으로 재측정** (일관 protocol) |
| Attn-GRU selector | 004 P001 | NN → 27 후보 분류 | LB 0.6806 | F0 아님 (out-of-scope) |
| BiGRU + codebook corrector | 014-016 | NN → 7 anchor + magnitude | OOF 0.6425, LB 0.6638 | F0 아님 (out-of-scope) |
| GRU-attention coef | 017 | Attn-NN → coef | IN PROGRESS | **plan-020 N3/N4 out-of-scope** (plan-017 carry) |
| TCN F0 coef | — | NN → coef (TCN encoder) | 미시도 | **N2 신규** |
| Transformer F0 coef | — | NN → coef (Transformer) | 미시도 | plan-017 overlap 가능성 → out-of-scope |
| Mixture-of-experts F0 | — | gating NN + K expert formulas | 미시도 | **N5 신규** |
| SE(3)-equivariant NN F0 | — | — | 미시도 | out-of-scope (별도 plan 필요) |
| Neural ODE F0 | — | — | 미시도 (notes/ ★★ 평가) | out-of-scope |
| Diffusion residual F0 | — | — | 미시도 | out-of-scope |

→ plan-020 의 NN axis = **N1 (재측정) + N2 (신규) + N5 (신규)** 3 후보.

### §1.4 F0 의 *진정한* 한계 — paradigm-level 진단 (plan-014/015/016 박제)

- Oracle ceiling (E0b Frenet-orthogonal anchor, hindsight) = 0.8248
- Measured best (plan-014 G5) = 0.6425
- 회수율 = **5.4%** (= 0.0105 / 0.1928)

→ corrector model class / hyperparam / multi-seed *어떤* 조합으로도 회수율 5.4% 못 넘김 = **input feature 와 정답 방향 사이의 mutual information 부족**.

→ plan-020 의 가설: **F0 산식 자체를 *structurally 변경* 하면 input 의 mutual information 활용을 늘릴 수 있는가?**

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 후보 개수 | 14 deterministic + 3 NN = **17** |
| Fold split | `stable_fold_id(sample_id, 5)` (plan-004 carry) |
| 평가 metric | hit@1cm (R_HIT=0.01) + hit@1.5cm (R_HIT=0.015), 둘 다 5-fold concat OOF |
| Pass criteria | paired Δ ≥ +0.005 *둘 다* (hit@1cm AND hit@1.5cm) |
| NN device | cuda:1 (project convention) |
| Deterministic 학습 | CMA-ES, popsize=20, maxiter=200, seed [20260518..20260522] |
| NN 학습 | Adam, lr=1e-3, batch=256, epochs=50, annealed hit-aware loss, same 5 seeds |
| 결과 박제 | 17 × 2 metric × 5-fold table + family winner |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| plan-004 27-pool 통합 | follow-up plan (단독 측정 우선) |
| LB 제출 (dacon-submit) | follow-up plan |
| BMA / IMM mixture | 단독 측정 완료 후 conditional Phase B |
| N3 (BiGRU coef) | plan-017 in-progress overlap |
| N4 (Transformer coef) | plan-017 in-progress overlap |
| SE(3)-equivariant / Neural ODE / Diffusion NN | scope 외 (별도 plan 필요) |
| NN 직접 3D coord regression | plan-003 paradigm 함정 (이미 검증) |
| 27-candidate pool 수정 | plan-020 의 baseline 으로 사용만 (modify X) |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

| 분할 | 값 |
|---|---|
| folds | 5 |
| fold 할당 | `stable_fold_id(sample_id, 5)` (plan-004 selector.py L147 carry) |
| seed | fold split deterministic (sample_id hash-based, seed 없음) |
| regime bins fit | **fold-internal** (각 fold k 의 train_(not k) 에서 `fit_regime_bins(train_x, end_idx)` 호출, val 누수 차단) |

### §3.2 합격 기준 (정량)

- **G0**: 18 모듈 import + smoke + tests green
- **G1**: F0 reproduce hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038]
- **G2.D**: 14 deterministic 후보 5-fold OOF hit metric finite (`np.isfinite + 0 ≤ x ≤ 1`)
- **G2.N**: 3 NN 후보 5-fold OOF metric finite + train loss converged + val_hit > 0.10
- **G3**: 17 × 2 metric table 박제 + ≥ 1 후보 paired Δ ≥ +0.005 *둘 다*
  - 0 통과 시 → `all_negative` warn 박제 후 G_final 직진 (severe X)

### §3.3 평가 점수

| metric | 식 | 비교 |
|---|---|---|
| hit@1cm | `mean(||pred − gt||_2 ≤ 0.01)` | F0 baseline 0.6320 |
| hit@1.5cm | `mean(||pred − gt||_2 ≤ 0.015)` | F0 baseline 0.8033 |
| paired Δ | per-fold 차이의 평균 | sample-level paired, F0 와 같은 fold split |
| fold variance | per-fold metric 의 std | < 0.05 (overfit guard) |

### §3.4 후보 표

| family | # | 후보 | 학습 param | precedent |
|---|---|---|---|---|
| F0 baseline | — | `frenet_par120_perp_neg020` | 0 | plan-006 |
| F1 회전 | C1 | Local helix (κ, τ, v) | 3 (α, β, γ) | 미시도 |
| F1 회전 | C2 | CTRA closed-form | 0 | 미시도 |
| F1 회전 | C3 | CTRV (CTRA-lite) | 0 | 미시도 |
| F1 회전 | C4 | IMM (CV/CA/CT 3-mode 평균) | 3 transition probs | 미시도 |
| F2 data-driven | C5 | Per-regime F0 (18 × 3) | 54 | 미시도 (plan-007 F001 6-var CMA 와 직교) |
| F3 고차 미분 | C6 | Quintic Hermite endpoint spline | 0 | 미시도 |
| F3 고차 미분 | C7 | Jerk-aware quartic polynomial | 0 | 미시도 |
| F4 noise-adaptive | C8 | Singer maneuver model | 2 (τ_a, σ_a) | 미시도 |
| F4 noise-adaptive | C9 | Adaptive Kalman smoother + extrapolation | 2 (Q, R) | 미시도 (notes A.1 IMM-KF 와 직교) |
| F5 기하학 | C10 | Bishop rotation-minimizing frame | 0 | 미시도 |
| F5 기하학 | C11 | SE(3) exponential twist | 0 | 미시도 |
| F6 도메인 정보 | C12 | Wingbeat-corrected F0 (FFT pre-filter) | 1 (cutoff freq) | 미시도 (plan-003 R004 와 직교 — R004 는 feature, C12 는 input 전처리) |
| F6 도메인 정보 | C13 | Lévy-flight prior | 2 (α, scale) | 미시도 |
| F7 비모수 | C14 | Trajectory KNN displacement | 1 (k) grid | 미시도 (notes B.1 user spec carry) |
| F2 NN | N1 | Per-sample MLP F0 coef | NN (small MLP) | **plan-007 F002 재측정** |
| F2 NN | N2 | TCN F0 coef regressor | NN (TCN) | 미시도 |
| F2 NN | N5 | Mixture-of-experts F0 | NN (gating) | 미시도 |

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-020/
├── baseline_f0.py              # plan-006 산식 1:1 재구현 (reproduce only)
├── formula_deterministic.py    # 14 deterministic 후보
├── formula_nn.py               # 3 NN 후보 (N1, N2, N5)
├── cma_es_fit.py               # CMA-ES + annealed hit-direct objective
├── run_oof.py                  # 5-fold OOF runner (deterministic + NN 분기)
├── results_deterministic.{json,md}
├── results_nn.{json,md}
├── family_analysis.{json,md}
└── results.md                  # G_final synthesis
```

### §4.2 module top-level export 보장 (smoke test lock-in)

| symbol | module | type |
|---|---|---|
| `f0_baseline` | baseline_f0 | `Callable[[np.ndarray], np.ndarray]` (X → pred shape (N, 3)) |
| `R_HIT` | baseline_f0 | `float` (= 0.01) |
| `R_HIT_LOOSE` | baseline_f0 | `float` (= 0.015) |
| `C01..C14` | formula_deterministic | `dict[str, Callable]` (14 후보 함수 + edge case fallback) |
| `N01_mlp`, `N02_tcn`, `N05_moe` | formula_nn | `nn.Module` |
| `cma_es_fit` | cma_es_fit | `Callable` (annealed objective) |
| `run_oof_deterministic`, `run_oof_nn` | run_oof | `Callable` |

→ 위 7 symbol 중 하나라도 AttributeError 시 G0 `infra_drift` severe escalate.

### §4.3 baseline_f0.py 산식 lock-in

```python
# baseline_f0.py — plan-006 frenet_par120_perp_neg020 1:1 재구현
# 산식: F0 = p0 + 1.98·v_last + 1.20·acc_par_vec - 0.20·acc_perp_vec
# Hard evidence: hit@1cm = 0.6320, hit@1.5cm = 0.8033

import numpy as np

R_HIT = 0.01
R_HIT_LOOSE = 0.015
D1 = 1.98
PAR = 1.20
PERP = -0.20

def f0_baseline(x: np.ndarray, end_idx: int) -> np.ndarray:
    """x shape (N, T, 3), end_idx = T-1. returns (N, 3)."""
    p0 = x[:, end_idx]
    v_last = x[:, end_idx] - x[:, end_idx - 1]
    v_prev = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = v_last - v_prev
    
    speed = np.linalg.norm(v_last, axis=1, keepdims=True)
    tangent = v_last / (speed + 1e-9)
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par_vec = acc_par_scalar * tangent
    acc_perp_vec = acc - acc_par_vec
    
    return p0 + D1 * v_last + PAR * acc_par_vec + PERP * acc_perp_vec
```

### §4.4 tests (c7)

- 18 모듈 import (AttributeError 0건)
- F0 reproduce: 10000 train 위 hit@1cm ∈ [0.6315, 0.6325] (G1 의 사전 smoke)
- 각 deterministic candidate: shape (N, 3), finite, edge case fallback 동작
- 각 NN candidate: forward pass shape OK, GPU device 동작 (cuda:1)

---

## §5. STAGE 1 — F0 baseline reproduce (c8, G1)

### §5.1 실행

```bash
python -m analysis.plan-020.run_oof --candidate f0_baseline --fold-all
```

### §5.2 산출

- `analysis/plan-020/baseline_oof.json`:

```json
{
  "candidate": "f0_baseline",
  "n_samples": 10000,
  "hit_1cm_5fold_concat": 0.6320,
  "hit_1.5cm_5fold_concat": 0.8033,
  "hit_1cm_per_fold": [0.6XX, 0.6XX, ...],
  "hit_1.5cm_per_fold": [0.8XX, 0.8XX, ...],
  "fold_variance_1cm": 0.0XX,
  "fold_variance_1.5cm": 0.0XX
}
```

### §5.3 G1 합격 기준 (자동)

- `hit_1cm_5fold_concat ∈ [0.6315, 0.6325]`
- `hit_1.5cm_5fold_concat ∈ [0.8028, 0.8038]`
- 위반 시 `f0_reproduce_drift` severe → halt

### §5.4 시간 예산

- CPU only, < 1 min (산식 결정적)

---

## §6. STAGE 2 — 14 Deterministic 후보 측정 (c9, G2.D)

### §6.1 각 후보 산식 spec

#### C1. Local helix (κ, τ, v) — 3 param CMA-ES

```
Input: 마지막 5 점 (end_idx-4 ~ end_idx)
v, a, j = finite diff (1/0.040)
tangent = v / |v|
acc_perp = a - (a·tangent) tangent
normal = acc_perp / |acc_perp|
binormal = tangent × normal
κ (곡률) = |acc_perp| / |v|²
τ (비틀림) = (j · binormal) / (|v|³ · κ)

s = |v| · 0.080

p(t+80ms) = p[-1]
          + α · (sin(κs)/κ) · tangent
          + β · ((1-cos(κs))/κ) · normal
          + γ · (τs) · binormal

학습: α, β, γ (init=1.0, CMA-ES)
Edge case: κ < 1e-6 → linear+accel fallback
           τ clip [-10, 10]
```

#### C2. CTRA closed-form — 0 param

```
State: (x, y, z, v_xy, θ, ω, a_xy) from last 3 points
if |ω| > 1e-3:
    x(t+h) = x + (v/ω)(sin(θ+ωh) - sin(θ)) + (a/ω²)(cos(θ+ωh) - cos(θ) + ωh sin(θ+ωh))
    y similar
else: linear+accel
z(t+h) = z + h·v_z + 0.5·h²·a_z

Edge case: |ω| > 30 rad/s → clip
```

#### C3. CTRV (CTRA-lite) — 0 param

CTRA 에서 a=0 가정. turn-rate 만 사용.

#### C4. IMM (CV/CA/CT 3-mode 평균) — 3 transition probs

```
3 mode 예측: CV (constant velocity), CA (constant accel), CT (constant turn)
mode likelihood = train OOF 에서 각 mode 의 hit-rate
fixed transition matrix [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]]
inference: 3 mode 예측의 weighted avg

학습: transition matrix diagonal weight (CMA-ES)
```

#### C5. Per-regime F0 (18 × 3 = 54 param) — CMA-ES per regime

```
for fold k:
    for regime r in 0..17:
        train_mask = (fold != k) & (regimes == r)
        if train_mask.sum() < 100:
            (d1_r, par_r, perp_r) = (1.98, 1.20, -0.20)  # global F0 fallback
            continue
        (d1_r, par_r, perp_r) = cma_es_fit(F0_form, X[train_mask], y[train_mask],
                                            init=(1.98, 1.20, -0.20))
    val_pred = F0_form(X_val, params_per_regime[regimes[val]])
```

#### C6. Quintic Hermite endpoint spline — 0 param

```
Input: 마지막 4 점 + 끝점 v, a
6 constraints → quintic uniquely determined:
  p(0) = x[-1], p(-40ms) = x[-2], p(-80ms) = x[-3], p(-120ms) = x[-4]
  p'(0) = (x[-1] - x[-2]) / 40ms
  p''(0) = (x[-1] - 2x[-2] + x[-3]) / (40ms)²

해 = 5×5 linear system, axis-별 독립
p(80ms) = ...

Edge case: 행렬 near-singular → linear extrap fallback
```

#### C7. Jerk-aware quartic — 0 param

```
p(t+h) = p[-1] + h·v + 0.5·h²·a + (1/6)·h³·j
v, a, j = finite diff from last 4 points

Edge case: j norm > 100 m/s³ → clip
```

#### C8. Singer maneuver model — 2 param (τ_a, σ_a) CMA-ES

```
State [p, v, a], a 는 exp-decay correlated noise (τ_a maneuver time, σ_a variance).
p(t+h) ≈ p + h·v + 0.5·h²·a · exp(-h/τ_a) + ... (Taylor)

학습: τ_a (init=0.100 s), σ_a (init=1.0) CMA-ES
```

#### C9. Adaptive Kalman smoother + extrapolation — 2 param (Q, R)

```
State-space: [p, v, a], constant-acceleration 모델 + Wiener jerk noise (process noise Q).
Measurement noise R.
Backward smoother on 11 점 → forward propagation 80ms.

학습: log(Q), log(R) (CMA-ES)
```

#### C10. Bishop rotation-minimizing frame — 0 param

```
Frenet 의 normal 대신 parallel transport 로 rotation-minimizing frame 계산.
F0 산식 그대로 적용 (par/perp 정의가 Bishop frame 에서 변경).
```

#### C11. SE(3) exponential twist — 0 param

```
마지막 4 점에서 SE(3) twist ξ (linear + angular velocity) 추정.
p(t+h) = exp_SE3(h·ξ) · p[-1]

Edge case: angular velocity > 10 rad/s → clip
```

#### C12. Wingbeat-corrected F0 (FFT pre-filter) — 1 param (cutoff freq)

```
Input: 11 점 trajectory
Step 1: low-pass filter (cutoff freq f_c, 학습 param) — wingbeat ~600Hz, body motion ~10-50Hz
Step 2: cleaned trajectory → F0 산식 적용

학습: f_c (init=50 Hz, range [10, 200])
Edge case: 11 점 sub-Nyquist → FFT 대신 moving average fallback
```

#### C13. Lévy-flight prior — 2 param (α, scale)

```
Lévy-stable distribution prior with stability α ∈ (0, 2], scale.
F0 + Lévy 분포의 mode (= 0 vector for symmetric Lévy) = F0 그대로.
→ Lévy 는 *분포* output 이라 point estimate 로는 F0 와 동일.
→ 대신 F0 의 *방향* 만 Lévy heavy-tailed sampling (deterministic mode).

학습: α, scale (CMA-ES, hit objective)
```

→ 이 후보는 deterministic mode-only 라 marginal 예상. 측정 위해 포함.

#### C14. Trajectory KNN displacement — 1 (k) grid

```
Step 1 (train, 한 번): 모든 train sample 의 11-step normalized trajectory (v_last frame) → 33D
       Faiss IndexFlatL2 색인 + 정답 displacement vec (3D, same frame)
Step 2 (inference): val sample → 33D query → k-NN → k 개 정답 vec 평균 → frame inversion → final pred
       k ∈ {1, 3, 5, 10, 20} grid

학습: k (best k on train fold via OOF)
Edge case: faiss 가용성 — 미가용 시 sklearn KNeighborsRegressor fallback
```

### §6.2 산출 (`results_deterministic.json`)

```json
{
  "candidates": {
    "C01_helix":           {"hit_1cm": 0.XX, "hit_1.5cm": 0.XX, "delta_1cm": +0.XX, "delta_1.5cm": +0.XX, "fold_variance_1cm": 0.0XX, "fold_variance_1.5cm": 0.0XX, "params": {...}},
    "C02_ctra":            {...},
    ...
    "C14_trajectory_knn":  {...}
  },
  "baseline": {"f0_hit_1cm": 0.6320, "f0_hit_1.5cm": 0.8033}
}
```

### §6.3 G2.D 합격 기준 (자동)

- 14 candidate 모두 metric finite
- NaN/Inf 0건
- 위반 시 `formula_numerical` severe

### §6.4 시간 예산

- 14 module 학습/측정 (CPU, deterministic) ≈ 30 min
- CMA-ES 학습이 있는 후보 (C1, C4, C5, C8, C9, C12, C13) ≈ 2 hours
- 총 ~2.5 hours

---

## §7. STAGE 3 — 3 NN 후보 학습 + 측정 (c10, G2.N)

### §7.1 NN 후보 spec

#### N1. Per-sample MLP F0 coefficient (plan-007 F002 재측정)

```python
class N01_MLPCoef(nn.Module):
    def __init__(self, seq_dim=9, hidden=64):
        super().__init__()
        # input: last 3 timesteps × 9D = 27D (plan-007 F002 carry)
        self.net = nn.Sequential(
            nn.Linear(27, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),  # (d1, par, perp) 잔차
        )
    def forward(self, seq_feats):
        # seq_feats: (B, 3, 9D)
        x = seq_feats.flatten(1)  # (B, 27)
        delta = self.net(x)
        # Output: d1=1.98+delta[0], par=1.20+delta[1], perp=-0.20+delta[2]
        return torch.stack([1.98 + delta[:, 0], 1.20 + delta[:, 1], -0.20 + delta[:, 2]], dim=1)

# Final prediction = F0_form(x; predicted_coef)
```

- 학습: Adam lr=1e-3, batch=256, epochs=50, hit-aware loss schedule
- Device: cuda:1
- Seed list: [20260518, 20260519, 20260520, 20260521, 20260522] → best-on-train

#### N2. TCN F0 coefficient regressor (신규)

```python
class N02_TCNCoef(nn.Module):
    def __init__(self, seq_dim=9, hidden=32):
        super().__init__()
        # input: (B, 11, 9D), causal TCN with dilations [1, 2, 4]
        self.tcn = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=2, dilation=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=4), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 3),
        )
    def forward(self, seq):
        # seq: (B, 11, 9D) → (B, 9, 11) for Conv1d
        h = self.tcn(seq.transpose(1, 2))
        delta = self.head(h)
        return torch.stack([1.98 + delta[:, 0], 1.20 + delta[:, 1], -0.20 + delta[:, 2]], dim=1)
```

#### N5. Mixture-of-experts F0 (신규)

```python
class N05_MoE(nn.Module):
    """Gating NN selects mixture weight over K=4 expert formulas."""
    def __init__(self, seq_dim=9, hidden=32):
        super().__init__()
        # K=4 experts: F0_baseline, C1_helix (default α=β=γ=1), C6_hermite, C2_ctra
        self.K = 4
        # gating: input → softmax over K
        self.gate = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, self.K),  # logits over K experts
        )
    def forward(self, seq, expert_preds):
        # seq: (B, 11, 9D), expert_preds: (B, K, 3) — pre-computed deterministic
        logits = self.gate(seq.transpose(1, 2))
        weights = torch.softmax(logits, dim=1)  # (B, K)
        return (weights[:, :, None] * expert_preds).sum(dim=1)  # (B, 3)
```

→ K expert predictions 는 *fold 마다 pre-compute* (전체 train + val). gating NN 만 학습.

### §7.2 학습 spec (3 NN 공통)

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 50 |
| Early stop | val hit plateau 10 epoch |
| Device | cuda:1 |
| Seed list | [20260518..20260522] |
| Seed aggregation | best-on-train (5-fold OOF mean) |
| Loss schedule | annealed (epoch 0-15 smooth τ=0.003, 16-30 smooth τ=0.001, 31-50 hard hit + boundary weighting) |
| Loss form | `-(hit@1cm + 0.5·hit@1.5cm)` (scalarized multi-objective) |
| Fold-internal training | 5-fold (train_(not k) on fold k 위 학습, val_k 위 OOF) |

### §7.3 산출 (`results_nn.json`)

```json
{
  "candidates": {
    "N01_mlp_coef":  {"hit_1cm": 0.XX, "hit_1.5cm": 0.XX, ...},
    "N02_tcn_coef":  {...},
    "N05_moe":       {...}
  },
  "n01_vs_f002_drift": +0.XX  // |plan-020 N1 - plan-007 F002 0.6482|
}
```

### §7.4 G2.N 합격 기준

- 3 NN 모두 metric finite
- val_hit > 0.10 (random baseline floor)
- train_hit − val_hit < 0.10 (overfit guard, 미달 시 `nn_overfit` warn)
- N1 의 OOF 가 plan-007 F002 (0.6482) 와 ±0.01 안 (`n1_drift_vs_f002` warn)

### §7.5 시간 예산 (cuda:1)

| NN | 5-fold × 5-seed |
|---|---|
| N1 MLP | 25 min |
| N2 TCN | 50 min |
| N5 MoE | 100 min |
| **총** | **~3 hours** |

---

## §8. STAGE 4 — Family-level 분석 (c11, G3)

### §8.1 17 × 2 metric × 5-fold table

`analysis/plan-020/family_analysis.md` 에 marker table:

```markdown
| # | candidate | family | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | pass | fold_var_1cm |
|---|---|---|---|---|---|---|---|---|
| F0 baseline | — | — | 0.6320 | — | 0.8033 | — | — | 0.0XX |
| C1 | helix | F1 | 0.6XX | +0.XX | 0.8XX | +0.XX | ✓/✗ | 0.0XX |
| ... | | | | | | | | |
```

### §8.2 Family-level winner 선정

각 family (F1~F7) 안에서 *가장 큰 paired Δ (hit@1cm + 0.5·hit@1.5cm)* 후보 = winner.

### §8.3 NN vs Deterministic 직접 비교

- N1 (MLP coef) vs C5 (per-regime F0) — *학습 방식 분리* 효과
- N2 (TCN coef) vs C5 — architecture 효과
- N5 (MoE) vs C1+C2+C6+F0 단순 평균 — gating 효과

### §8.4 G3 합격 기준

- 17 × 2 table 박제 + 7 family winner 박제
- ≥ 1 후보 paired Δ ≥ +0.005 *둘 다*
- 0 통과 시 → `all_negative` warn 박제 후 G_final 진입

---

## §9. STAGE 5 — Best 박제 + Results (c12, G_final)

### §9.1 3-file frontmatter sync

- `plans/plan-020-f0-structural-search.md` top-level frontmatter
- `plans/plan-020-f0-structural-search.results.md`
- `analysis/plan-020/results.md`

세 파일 모두 다음 필드 동시 갱신:
- `status: all_complete` (또는 `partial` if G2.D / G2.N 부분 fail)
- `band: positive / marginal / negative` (G3 winner 의 paired Δ 기준)
- `best_candidate: <후보 이름>` (G3 winner)
- `best_hit_1cm: <float>`, `best_hit_1.5cm: <float>`

### §9.2 results.md 필수 항목

- F0 baseline measured (G1)
- 17 후보 hit@1cm + hit@1.5cm + paired Δ (5-fold concat) full table
- 7 family winner 박제
- NN vs Deterministic 직접 비교
- N1 = plan-007 F002 재측정 결과 비교 (drift 박제)
- decision-note 박제 list
- follow-up plan 후보 (post-G_final 분석 기반)
- caveats

### §9.3 plan-017 overlap 해소

- plan-017 status check (in progress → completion 까지 plan-020 N3/N4 carry)
- plan-017 의 N3/N4 산출이 plan-020 G_final 이전 도착 시 → results.md 의 *부록* 으로 추가 (plan-020 본 분석은 N1/N2/N5 기준 그대로)
- plan-017 결과가 plan-020 와 ±0.01 이상 차이 → `plan017_carry_conflict` warn 박제

### §9.4 G_final 합격 기준

- 3-file sync 완료
- §0.5 commit chain c1~c12 모두 [DONE]
- results.md 필수 항목 모두 박제
- follow-up plan 후보 ≥ 2건 박제

---

## §N+1. results.md 필수 항목

(plan-014 / plan-006 format 참조)

- plan_id, version, date, status, band, best_candidate
- F0 baseline measured (G1)
- 17 후보 × 2 metric × 5-fold concat 표
- 7 family winner 박제
- NN vs Deterministic 직접 비교 (table)
- N1 vs plan-007 F002 drift 박제
- decision-note 박제 list
- follow-up plan 후보 (post-G_final)

---

## §N+2. 통계 함정 & caveats

1. **Fold-internal regime fit 의무**: C5 의 18-regime fit + C8/C9 의 noise model fit 은 *반드시* train_(not k) 위에서만. val 누수 시 OOF 가 train hit 으로 inflate → false positive.

2. **Multi-seed best-on-train**: NN + CMA-ES 후보의 seed 분산. best-on-train 으로 val 보다 train metric 최적화 시 *seed selection bias* 가능. mitigate: train metric 으로만 seed 선택, val metric 은 *최종 평가* 만.

3. **NN-overfit risk** (특히 N5 MoE): gating NN 이 train 위 expert 마다 hindsight 잘 맞는 sample 학습 시 overfit. dropout=0.1 + weight_decay=1e-4 + early stop 적용.

4. **N1 vs plan-007 F002 drift**: plan-007 F002 가 같은 architecture 인데 OOF 차이 ±0.01 초과 시 *protocol divergence* 박제. drift 원인:
   - fold split 차이 (plan-007 = plan-004 carry, plan-020 = 동일)
   - seed 차이 (plan-007 = 20260606, plan-020 = 20260518..)
   - loss schedule 차이 (plan-007 = MSE, plan-020 = annealed hit-aware)
   → 이 중 *loss schedule* 가 가장 큰 영향 예상.

5. **C12 wingbeat FFT sub-Nyquist 위험**: 11 점 × 40 ms = 12.5 Hz Nyquist. wingbeat 600 Hz → fully aliased. cutoff freq 학습이 *aliased noise* 만 학습할 가능성. mitigate: moving-average fallback + 학습 후 visualization 검증.

6. **C14 KNN faiss 의존성**: `faiss-cpu` package 가용성 확인. 미가용 시 sklearn KNeighborsRegressor fallback 자동 (성능 차이 marginal).

7. **C13 Lévy mode = F0**: deterministic mode-only 라 단독 hit 가 *F0 와 동일* 예상. plan-020 안에서 측정 가치는 *분포 형태가 corrector 학습 신호에 영향* 가능성 박제만 (post hoc).

8. **단독 hit ↔ pipeline 가치 비례 보장 X**: plan-020 의 단독 winner 가 27-pool 통합 후 LB 향상까지 보장하지는 않음. *직교성* 측정은 follow-up plan-021 (가칭) 으로 carry.

9. **N5 MoE expert 선택 의존성**: K=4 expert (F0, helix, Hermite, CTRA) 가 *임의 선택*. 다른 expert set (예: F0, per-regime, KNN, Bishop) 와의 ablation 미시도 → follow-up plan.

10. **plan-017 overlap 해소 책임**: plan-020 G_final 시점에 plan-017 status check + N3/N4 결과 carry. plan-017 이 G_final 미달성 시 → plan-020 results 의 plan-017 부록은 "carry pending" 박제 후 종료.

---

## §N+3. 변경 이력

- v1 (2026-05-18): 초안 — 17 후보 (14 deterministic + 3 NN: N1/N2/N5) plan body. plan-017 overlap 으로 N3/N4 out-of-scope 박제. Maximum tier 선택.
- v1.1 (2026-05-18): narrative ("단일 공식 결과 최대화") 정합 점검 — §9 STAGE 5 (27-pool oracle delta, §0 out-of-scope 와 충돌) + §N+1 작업량 회계 삭제. STAGE 6 → STAGE 5 / c13 → c12 renumber. caveat #8 의 G4 의존 표현 단순화.

---

## §N+4. 참조

- `plans/archive/plan-006-minimal-variant-e-lb.md` — F0 산식 baseline 정의 + 0.6320 hard evidence
- `plans/archive/plan-007-formula-tuning.md` — CMA-ES infrastructure + F002 NN coef precedent
- `plans/plan-004-pb-0-6822-fullrun.md` — fold split + 18-regime + 27-pool 기반
- `plans/archive/plan-014-plan012-failure-inversion.md` — corrector paradigm ceiling 측정 (회수율 5.4%)
- `plans/plan-017-gru-attention-coeff-regressor.md` — N3/N4 overlap source (in progress)
- `notes/new-ideas.md` — KNN, IMM, SE(3), Lévy, Neural ODE 후보 풀
- `notes/mosquito-trajectory-ideas.md` — 도메인 지식 (wingbeat, Lévy flight, jerk feature)
- `notes/코드공유-upgrade.md` — plan-005 진단 + 5가지 직관
- `notes/drone-insights.md` — sub-second prediction regime + 외부 paper 인용
- `CLAUDE.md` — autonomous execution policy
- `WORKFLOW.md` — plan/results/registry convention (§0.5, §11, §12)
- `analysis/plan-004/regime_distribution.json` — 18×27 regime 통계 anchor
- `src/pb_0_6822/selector.py` — 27 후보, fold split, regime fit 함수 carry source
