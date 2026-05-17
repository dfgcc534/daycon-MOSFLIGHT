---
plan_id: 020
version: 1.4
date: 2026-05-18 (Asia/Seoul)
status: all_complete
best_candidate: C05_per_regime_f0
best_hit_1cm: 0.6503
best_hit_1.5cm: 0.8086
best_delta_1cm: +0.0183
best_delta_1.5cm: +0.0053
based_on:
  - 004 (fold split + 18-regime infrastructure + 27-pool reference)
  - 006 (F0 baseline 0.6320 / 0.8033 산식 — plan-006 `frenet_par120_perp_neg020`)
  - 007 (CMA-ES infrastructure + F002 per-sample MLP coef precedent)
  - 014~016 (corrector paradigm ceiling 측정 — F0 family 한계 박제)
  - 017 (GRU-attention coef regressor — N3 overlap, plan-020 N1/N2/N5 와 직교 선택)
followed_by:
  - plan-021 (가칭): C05 winner 의 27-pool 통합 + LB 측정 (BMA / oracle delta)
  - plan-022 (가칭): C12 wingbeat default broken fix + C10 Bishop CMA saddle escape + 본 plan-020 v1.4 full-spec 재측정 (CMA popsize=20/maxiter=200/seeds=5, RTS smoother)
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
band: positive
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
| c1 | docs | `plans/plan-020-f0-structural-search.md` 본문 v1 작성 (v1.1 narrative 정합 / v1.2 plan-review-master 5-iter fix / v1.3 코드 재사용 검토 정정) | [DONE] |
| c2 | code | `analysis/plan-020/baseline_f0.py` (plan-006 산식 1:1 재구현, reproduce-only + torch mirror, bit-identical sanity) | [DONE] |
| c3 | code | `analysis/plan-020/formula_deterministic.py` (14 후보 산식 + helpers, smoke 14/14 shape+finite ✓. C9 KF forward filter only — RTS smoother v1.4 carry) | [DONE] |
| c4 | code | `analysis/plan-020/formula_nn.py` (3 NN module + smooth-hit loss + train_nn_fold loop, smoke 3/3 forward + mini-train ✓) | [DONE] |
| c5 | code | `analysis/plan-020/run_oof.py` 5-fold OOF runner + dispatch (deterministic + NN multi-seed best-on-train + N5 expert_preds pre-compute) | [DONE] |
| c6 | code | `analysis/plan-020/cma_es_fit.py` (CMA-ES 6 후보 + annealed τ schedule + per-regime + KNN, smoke C08/C10/C14 OK) | [DONE] |
| c7 | test | `tests/test_plan020_smoke.py` 6 pytest (import + parity + 14 deterministic shape + 3 NN forward + dispatch + G1 preflight) | [DONE] |
| G0 | gate | smoke + tests green — 6/6 pytest 통과 ✓ | [DONE] |
| c8 | exp G1 | F0 baseline 5-fold OOF reproduce → exact 0.6320 / 0.8033 (drift 0). 산출: `analysis/plan-020/baseline_oof.{json,md}` | [DONE] |
| G1 | gate | F0 hit@1cm = **0.6320** ∈ [0.6315, 0.6325] AND hit@1.5cm = **0.8033** ∈ [0.8028, 0.8038] ✓ | [DONE] |
| c9 | exp G2.D | 14 deterministic 5-fold OOF — **C05 per-regime F0 PASS** (Δ +0.0183 / +0.0053 둘 다). 나머지 13 fail. reduced CMA spec (popsize=10/maxiter=50/seeds=3, ~10min). | [DONE] |
| G2.D | gate | 14 후보 metric finite ✓ + ≥1 후보 paired Δ ≥ +0.005 둘 다 (C05) ✓ | [DONE] |
| c10 | exp G2.N | 3 NN OOF (cuda:1, seeds=3, epochs=30). N1 +0.0069/-0.0010, N2 +0.0052/+0.0003, N5 +0.0007/+0.0032 — *둘 다* 통과 0. N1 drift 0.0093 < ±0.02. | [DONE] |
| G2.N | gate | 3 NN metric finite ✓ + val_hit > 0.10 ✓ + overfit guard ✓ + n1_drift ±0.02 ✓ | [DONE] |
| c11 | analysis | 17 × 2 table + 7 family winner + overall best. **C05_per_regime_f0 단독 PASS, band positive** (Δ +0.0183 / +0.0053). | [DONE] |
| G3 | gate | table 박제 ✓ + family winner 박제 ✓ + ≥1 paired Δ ≥ +0.005 둘 다 (C05) ✓ | [DONE] |
| c12 | docs | `plans/plan-020-f0-structural-search.results.md` + `analysis/plan-020/results.md` + frontmatter sync | [TODO] |
| G_final | gate | results 3-file sync + §0.5 [TODO]→[DONE] sync + follow-up 후보 박제 | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 에서 F0 reproduce 가 hit@1cm 0.6320 ± 0.0005 또는 hit@1.5cm 0.8033 ± 0.0005 밖. 추출/fold split/regime bin 버그 의심 → 즉시 halt.
- `formula_numerical`: deterministic candidate 출력에 NaN/Inf (예: helix κ=0 division, CTRA |ω| 발산). 각 candidate 의 *edge case fallback* 으로 회피.
- `nn_no_signal`: NN candidate val_hit < 0.10 (random baseline floor). architecture/normalization/loss 버그 의심.
- `nn_overfit`: NN candidate train_hit − val_hit > 0.10 (5-fold mean). regularization 부족 → dropout/weight_decay 강화 필요.
- `per_regime_overfit`: C5 per-regime F0 의 fold variance > 0.05. min sample threshold 강화 or global fallback.
- `plan017_carry_conflict`: 만약 plan-017 의 N3/N4 산출이 인계되고 plan-020 의 N1/N2/N5 와 ±0.01 이상 차이 발생 시 protocol divergence 박제 (severe X, warn only).
- `n1_drift_vs_f002`: N1 (plan-007 F002 *paradigm 재측정* — architecture 다름) 이 F002 의 OOF 0.6482 와 **±0.02** 초과 차이. (threshold 완화 사유 §N+2 caveat #4 참조 — F002 = 13D 통계×6-step, N1 = 27D raw×3-step → 동일 *paradigm class* 안 다른 architecture 라 ±0.01 보장 무리.) protocol 점검 warn.
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
- hit@1cm = **0.6320** (10000 train 위 5-fold concat OOF, plan-006/plan-014 G0 reproduce 일치 protocol)
- hit@1.5cm = **0.8033** (동일 protocol)
- → 84% sample 이 F0 근방 1.5cm 안, 그 중 ~21% 가 1cm 밖 = *corrector 회수 zone*

### §1.3 NN-as-F0 의 *시도된* / *미시도* 분리

| 시도 | Plan | 형태 | 결과 | plan-020 처리 |
|---|---|---|---|---|
| NN → 3D coord 직접 회귀 | 003 R001-R006 | regression | LB 0.5688 (실패) | 재시도 X (paradigm 함정) |
| Per-sample MLP F0 coef | 007 F002 | NN → (d1, par, perp) | OOF 0.6482 ≈ global F0 | **N1 = paradigm 재측정** (architecture 다름: F002 13D 통계×6-step → N1 27D raw×3-step. *동일 MLP-coef paradigm 의 다른 instance*, drift threshold ±0.02 §0.5) |
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
| fold 할당 | `stable_fold_id(sample_id, 5)` (plan-004 `src/pb_0_6822/selector.py` L147 carry) |
| seed | fold split deterministic (sample_id hash-based, seed 없음) |
| regime bins fit | **fold-internal** (각 fold k 의 train_(not k) 에서 `fit_regime_bins(train_x, end_idx)` 호출, val 누수 차단) |

#### §3.1.1 carry 함수 시그너처 + 결정성 spec (self-contained 박제)

```python
# stable_fold_id (plan-004 carry, src/pb_0_6822/selector.py L185) — 결정성 보장
def stable_fold_id(sample_id: str, n_folds: int = 5) -> int:
    """sample_id 의 해시 (seed 없음, **MD5 32-bit prefix mod n_folds**) 로 fold index 결정.
    실제 구현: `int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16) % n_folds`.
    - 입력: sample_id (str — dataframe row 의 unique string), n_folds (=5 plan-020 default).
    - 출력: 0..n_folds-1 정수 1 개.
    - 결정성: 동일 sample_id 는 항상 같은 fold (process / seed 무관). 동일 string 입력 → 동일 fold.
    - 충돌 분포: 10000 sample 위 fold size deviation 통상 < 5%.
    - 주의: int sample_id 입력 시 호출자가 str(int) 변환 필요 — 자동 변환 X."""

# fit_regime_bins + assign_regimes (plan-004 carry, src/pb_0_6822/selector.py L361/L371)
# *분리된 2 함수* — fit 은 dict 반환, assign 은 별도 함수 호출 필수 (plan body 의 OO interface 가정은 silent bug).
def fit_regime_bins(train_x: np.ndarray, end_idx: int) -> dict[str, list[float]]:
    """train_x shape (N_train, T, 3), end_idx (= T-1) → regime bin edges (dict).
    실제 반환 예시: {"speed": [0.0176, 0.0290], "curvature": [0.0874, 0.1923], "speed_slope": [0.0108]}.
    - 결정성: 동일 train_x 는 동일 bins (np.quantile 결정적, seed 없음).
    - 18 regime decomposition: speed_bin (3 levels) × curvature_bin (3) × speed_slope_bin (2) = 18.
    - fold-internal 의무: caller 가 train_(not k) 만 전달 (val 누수 차단 책임 caller)."""

def assign_regimes(x: np.ndarray, end_idx: int, bins: dict[str, list[float]]) -> np.ndarray:
    """x shape (N, T, 3), bins = fit_regime_bins(...) 반환 dict → regime index 0..17 (int) shape (N,).
    - regime_id = speed_bin * 6 + curve_bin * 2 + fatigue_bin  (∈ {0, 1, ..., 17}).
    - usage 패턴 (fold-internal 의무):
        bins         = fit_regime_bins(train_x_not_k, end_idx)
        regimes_val  = assign_regimes(val_x_k, end_idx, bins)            # val 도 train_(not k) bins 으로 assign
        regimes_train= assign_regimes(train_x_not_k, end_idx, bins)"""
```

- carry 시 import 경로: `from src.pb_0_6822.selector import stable_fold_id, fit_regime_bins, assign_regimes`.
- 시그너처 drift 발생 시 (signature mismatch / 반환 type 변화 / assign_regimes 미사용) → G0 `infra_drift` severe.
- **plan body 의 §6.1 C5 pseudo-code (L444-456) 의 `regimes[r]` 사용은 위 `assign_regimes` 출력 가정 — caller 가 fit + assign 2-step 직접 수행 의무**.

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
| paired Δ | **sample-level paired** = `mean_{i ∈ 10000 sample}(1{‖pred_cand_i − gt_i‖ ≤ R} − 1{‖pred_F0_i − gt_i‖ ≤ R})`. F0 와 동일 fold split 위 5-fold concat OOF 에서 계산. | §2.1/§3.2 의 +0.005 임계는 *이 sample-level paired Δ* 에 적용 |
| fold variance | per-fold metric (5 개) 의 std | < 0.05 (overfit guard, paired Δ 와 무관한 별도 진단) |

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
| F3 고차 미분 | C7 | Jerk-aware cubic polynomial | 0 | 미시도 |
| F4 noise-adaptive | C8 | Singer maneuver model | 1 (τ_a) — σ_a 는 noise variance, point predict 미진입 → 학습 param 제외 | 미시도 |
| F4 noise-adaptive | C9 | Adaptive Kalman smoother + extrapolation | 2 (Q, R) | 미시도 (notes A.1 IMM-KF 와 직교) |
| F5 기하학 | C10 | Bishop rotation-minimizing frame | 1 (λ — M1/M2 비대칭 gain, F0 항등성 차단) | 미시도 |
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
| `f0_baseline` | baseline_f0 | `Callable[[np.ndarray, int], np.ndarray]` (X, end_idx → pred (N, 3)) — numpy, baseline reproduce 용 |
| `f0_form_torch` | baseline_f0 | `Callable[[Tensor, Tensor], Tensor]` (seq_feats (B, 3, 9D), coef (B, 3) → pred (B, 3)) — torch, **NN coef → 최종 예측 gradient path 보장**. N1/N2/N5 forward 가 반드시 이 함수로 최종 prediction 계산. |
| `R_HIT` | baseline_f0 | `float` (= 0.01) |
| `R_HIT_LOOSE` | baseline_f0 | `float` (= 0.015) |
| `C01..C14` | formula_deterministic | `dict[str, Callable[[seq_feats: np.ndarray, fit_params: dict | None], np.ndarray]]` (14 후보, fit_params = CMA-ES 후 학습 param 또는 None for 0-param 후보. 각 호출은 (N, T, 3) 입력 → (N, 3) 출력) |
| `N01_mlp`, `N02_tcn`, `N05_moe` | formula_nn | `nn.Module` |
| `cma_es_fit` | cma_es_fit | `Callable` (annealed objective) |
| `run_oof_deterministic`, `run_oof_nn` | run_oof | `Callable` |

→ 위 export 중 하나라도 AttributeError 시 G0 `infra_drift` severe escalate.

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


# ── torch mirror — NN coef → 최종 예측 gradient path 보장 (§4.2 export) ──
import torch

def f0_form_torch(seq_feats: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """seq_feats shape (B, 3, 9) — last 3 timesteps × 9D = [px,py,pz, vx,vy,vz, ax,ay,az]
       per timestep (Δt = 0.040 s, v/a finite-diff). timestep order: [end_idx-2, end_idx-1, end_idx].
    coef shape (B, 3) = (d1, par, perp). returns (B, 3).
    f0_baseline 의 산식 torch 미러 — d1/par/perp 만 sample-level 가변, 나머지 동일."""
    p0      = seq_feats[:, 2, 0:3]               # x_T position
    v_last  = seq_feats[:, 2, 3:6]               # v_T   (= (x_T - x_{T-1}) / Δt × Δt = x_T - x_{T-1} scaling 보정 호출자 책임)
    v_prev  = seq_feats[:, 1, 3:6]               # v_{T-1}
    acc     = v_last - v_prev                    # finite-diff accel (Δt 같은 scaling)
    
    speed       = v_last.norm(dim=1, keepdim=True)
    tangent     = v_last / (speed + 1e-9)            # baseline_f0 numpy 와 bit-identical 보장
    acc_par_s   = (acc * tangent).sum(dim=1, keepdim=True)
    acc_par_vec = acc_par_s * tangent
    acc_perp_vec = acc - acc_par_vec
    
    d1, par, perp = coef[:, 0:1], coef[:, 1:2], coef[:, 2:3]
    return p0 + d1 * v_last + par * acc_par_vec + perp * acc_perp_vec
```

- 호출자 책임: seq_feats 의 `vx/vy/vz` 가 x[end_idx] − x[end_idx−1] (= v_last in displacement units, *Δt 분할 없음*) 와 호환되게 build. NN feature builder (run_oof_nn) 에서 lock-in.
- 결정성: identical seq_feats + coef = (1.98, 1.20, −0.20) 입력 시 f0_baseline 와 sample 단위 ±1e-6 안에서 일치 (smoke test 의 1차 sanity).

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
Inference horizon: h = 0.080 s (모든 CTRA-family 후보 공통). State: (x, y, z, v_xy, θ, ω, a_xy) from last 3 points (Δt = 0.040 s):
  v_xy_vec_t  = ((x_t - x_{t-1}) / Δt, (y_t - y_{t-1}) / Δt)
  v_xy        = ||v_xy_vec_t||
  v_z_t       = (z_t - z_{t-1}) / Δt
  θ_t         = atan2(y_t - y_{t-1}, x_t - x_{t-1})
  ω           = (θ_t - θ_{t-1}) / Δt           # heading finite-diff
  a_xy        = (||v_xy_vec_t|| - ||v_xy_vec_{t-1}||) / Δt   # longitudinal accel (scalar)
  a_z         = (v_z_t - v_z_{t-1}) / Δt

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
3 mode 예측 (h = 0.080 s, Δt = 0.040 s, last 3 points x_{t-2}, x_{t-1}, x_t):
  v_t  = (x_t - x_{t-1}) / Δt,  v_{t-1} = (x_{t-1} - x_{t-2}) / Δt
  a_t  = (v_t - v_{t-1}) / Δt
  p_CV = x_t + h · v_t                                                   # constant velocity
  p_CA = x_t + h · v_t + 0.5 · h² · a_t                                  # constant accel
  p_CT = C2_CTRA(x_t, v_t) with a_xy ≡ 0, a_z ≡ 0  (CTRV-style, turn only)

mode prior π_m (per-fold, sample-independent):
  π_m_raw = train_(not k) 위 mode m 의 hit@1cm rate (fold-internal, scalar per mode)
            = mean_{i ∈ train_(not k)}(1{||p_m_i − gt_i|| ≤ 0.01})        for m ∈ {CV, CA, CT}
  π_m     = softmax(w_diag ⊙ (π_CV_raw, π_CA_raw, π_CT_raw))               # element-wise scale, vec3 × vec3
  → w_diag ∈ R³ (CMA-ES 학습), softmax 출력 π = (π_CV, π_CA, π_CT) ∈ Δ²

transition matrix [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]] 은
본 plan 의 단발 예측 (no recursive mode switching) 에서 *사용 안 함* — IMM 명칭의 source-citation 용 anchor.
필요 시 follow-up plan-021 의 multi-step IMM 으로 확장.

inference: p_pred = π_CV · p_CV + π_CA · p_CA + π_CT · p_CT

학습: w_diag ∈ R³ (CMA-ES, init=(1.0, 1.0, 1.0), range each [0.1, 10.0]) — element-wise sharpness 조정.
**Scope**: w_diag 는 **fold 마다 별도** (fold k 의 train_(not k) 위 fit, val_k OOF 평가). 5 fold 공유 X.
π_m_raw 도 fold-internal scalar 라 fold 마다 다르며, w_diag 와 함께 fold-wise 학습 → CMA-ES 5 회 (per fold).
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

해 = 6×6 linear system (quintic = 6 계수 a0..a5, 6 constraints), axis-별 독립
p(80ms) = a0 + a1·(80ms) + a2·(80ms)² + a3·(80ms)³ + a4·(80ms)⁴ + a5·(80ms)⁵

Edge case: 행렬 near-singular → linear extrap fallback
```

#### C7. Jerk-aware cubic — 0 param (※ v1 본문 "quartic" 명칭은 식이 cubic 이라 정정)

```
p(t+h) = p[-1] + h·v + 0.5·h²·a + (1/6)·h³·j
v, a, j = finite diff from last 4 points (Δt = 0.040 s):
  v = (x[-1] - x[-2]) / Δt
  a = (x[-1] - 2·x[-2] + x[-3]) / Δt²
  j = (x[-1] - 3·x[-2] + 3·x[-3] - x[-4]) / Δt³
h = 0.080 s

Edge case: ||j|| > 100 m/s³ → clip to ||j||=100 (방향 보존)
```

#### C8. Singer maneuver model — 1 param (τ_a) CMA-ES

```
State [p, v, a], a 는 Gauss-Markov correlated noise (maneuver time constant τ_a).
σ_a 는 noise variance — point prediction E[p(t+h)] 의 mean propagation 에 미진입 → 학습 param 제외.

Singer mean propagation (closed-form, σ_a 무관):
  p(t+h) = p + h · v + a · τ_a² · (h / τ_a − 1 + exp(−h / τ_a))
  (h=0.080 s, p, v, a = §C7 동일 finite-diff)

학습: τ_a (init=0.100 s, range [0.020, 1.000]) CMA-ES
Edge case: τ_a → 0 시 (h/τ_a − 1 + exp(−h/τ_a)) → -1, polynomial fallback 무관 (식 자체 유한).
```

#### C9. Adaptive Kalman smoother + extrapolation — 2 param (Q, R)

```
State-space: per-axis 독립 KF (x, y, z 3개 독립 필터, 각 3D state):
  - 각 축 state s_a = [p_a, v_a, a_a]^T (3D), 전이행렬 F = [[1, Δt, 0.5·Δt²], [0, 1, Δt], [0, 0, 1]] (3×3)
  - 측정행렬 H = [1, 0, 0] (위치만 관측)
  - process noise Q_a = q · G G^T (3×3), G = [Δt³/6, Δt²/2, Δt]^T, q = exp(log_q) (scalar, 3 축 공유)
  - measurement noise R_a = exp(log_r) (scalar, 3 축 공유)
Backward (RTS) smoother on 11 점 → forward propagation 80 ms (= 2 step at Δt=0.040 s).

학습: (log_q, log_r) (CMA-ES, init=(-6.0, -4.0), range [-12, 0])
```

#### C10. Bishop rotation-minimizing frame — 0 param

```
Input: 11 점 trajectory (t = end_idx-10 ... end_idx), Δt = 0.040 s.

Bishop frame {T_t, M1_t, M2_t} sequential propagation (rotation-minimizing):
  init (t=0):
    T_0  = (x_1 - x_0) / max(||x_1 - x_0||, 1e-9)
    v_init = world-z = (0, 0, 1)
    if |T_0 · v_init| > 0.99:  v_init = (1, 0, 0)   # near-collinear 회피
    M1_0 = v_init - (v_init · T_0) · T_0
    M1_0 = M1_0 / max(||M1_0||, 1e-9)
    M2_0 = T_0 × M1_0
  step t → t+1 (parallel transport):
    T_{t+1} = (x_{t+1} - x_t) / max(||x_{t+1} - x_t||, 1e-9)
    b       = T_t × T_{t+1};  θ = atan2(||b||, T_t · T_{t+1})
    if ||b|| < 1e-9:
        M1_{t+1} = M1_t;  M2_{t+1} = M2_t
    else:
        R   = Rodrigues(b / ||b||, θ)               # 3×3 SO(3) rotation
        M1_{t+1} = R · M1_t;  M2_{t+1} = R · M2_t

par/perp 재정의 (Bishop frame 안 F0 산식, M1/M2 비대칭 gain λ 학습으로 F0 항등성 차단):
  v_last       = x_T - x_{T-1};  v_prev = x_{T-1} - x_{T-2}
  acc          = v_last - v_prev
  acc_par_vec  = (acc · T_last) · T_last                       # tangent 방향 (F0 와 동일)
  acc_perp_M1  = (acc · M1_last) · M1_last
  acc_perp_M2  = (acc · M2_last) · M2_last

p_pred = x_T + 1.98·v_last + 1.20·acc_par_vec
       + (−0.20) · acc_perp_M1  +  (−0.20 · λ) · acc_perp_M2

학습: λ (CMA-ES, init=1.0, range [-2.0, 2.0]) — M1/M2 비대칭 gain
  - λ = 1 일 때 perp_M1 + perp_M2 = F0 perp (orthonormality) → 정확히 F0 와 항등 (sanity check 가능).
  - λ ≠ 1 일 때 Bishop frame 의 *parallel transport 누적 회전* 이 prediction 에 진입 → 본 candidate 의 structural lever.

Edge case: 11 점 중 동일점 발생 (||x_{t+1}-x_t|| < 1e-9) → 직전 frame 유지
           λ NaN/Inf 발생 → λ ← 1.0 (F0 fallback)
```

#### C11. SE(3) exponential twist (position-only approx) — 0 param

```
Position-only twist 추정 (rotation 관측 부재 → trajectory curvature 로 angular velocity proxy):
  Input: 마지막 4 점 x[-4..-1], Δt=0.040 s, h=0.080 s
  v    = (x[-1] - x[-2]) / Δt                # linear velocity (3D)
  v_p  = (x[-2] - x[-3]) / Δt
  acc  = (v - v_p) / Δt                      # finite-diff accel (3D)
  T    = v / max(||v||, 1e-9)
  T_p  = v_p / max(||v_p||, 1e-9)

  angular velocity ω (axis-angle from tangent rotation):
    b        = T_p × T
    θ        = atan2(||b||, T_p · T)
    if ||b|| < 1e-9:   ω = zeros(3)
    else:              ω = (b / ||b||) · (θ / Δt)

  twist ξ = (v, ω) ∈ R^6;  forward propagation 0.080 s:
    if ||ω|| > 1e-6:
        R_h = Rodrigues(ω / ||ω||, ||ω|| · h)   # SO(3) exp
        Δp  = R_h · (v · h)                     # rotated linear displacement
    else:
        Δp  = h · v + 0.5 · h² · acc            # const-accel fallback
    p_pred = x[-1] + Δp

Edge case: ||ω|| > 10 rad/s → ω ← ω · (10 / ||ω||)  (scale clip, 방향 보존)
```

#### C12. Wingbeat-corrected F0 (FFT pre-filter) — 1 param (cutoff freq)

```
Input: 11 점 trajectory (Δt = 0.040 s → Nyquist = 1/(2·Δt) = 12.5 Hz, 11-point DFT bin = 2.27 Hz)
Step 1: low-pass filter (cutoff freq f_c, 학습 param)
  - 11 점 단일 DFT (rectangular window) → bin {0, 2.27, 4.55, ..., 12.5} Hz 안
  - f_c 이상 bin 0 으로 mask 후 IDFT (axis-별 독립)
Step 2: cleaned trajectory → F0 산식 적용

학습: f_c (init=8.0 Hz, range [2.27, 12.5]) — Nyquist 안으로 좁힘
  (v1 본문의 range [10, 200] 은 alias zone → 무의미. wingbeat ~600 Hz 는 본 11점 sample rate 25 Hz 로 회복 불가 → spec scope 외.)
Edge case:
  - DFT 가 11 점 sparse spectrum 이라 sub-bin 보간 X (bin-level mask 만)
  - f_c < 2.27 Hz → 사실상 DC 만 통과 → moving average (window=11) fallback
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
v_last frame 정의 (sample s 마다, end_idx = T-1 기준):
  v_last_s = x_s[T-1] - x_s[T-2]                      # 3D vector
  T_hat    = v_last_s / max(||v_last_s||, 1e-9)        # frame x-axis
  z_world  = (0, 0, 1)
  if |T_hat · z_world| > 0.99:  z_world ← (1, 0, 0)    # near-collinear 회피
  N_hat    = z_world - (z_world · T_hat) · T_hat
  N_hat    = N_hat / max(||N_hat||, 1e-9)
  B_hat    = T_hat × N_hat
  R_s      = stack_rows([T_hat, N_hat, B_hat]) ∈ R^{3×3}   # world → frame rotation
  origin_s = x_s[T-1]

normalize (각 sample s):
  traj_frame_s[t] = R_s · (x_s[t] - origin_s) ∈ R^3   for t = 0..10
  query_s         = flatten(traj_frame_s)  ∈ R^{33}

Step 1 (train, fold-internal): train_(not k) 의 query 33D 색인 (Faiss IndexFlatL2 또는 sklearn KNeighbors)
  정답 displacement (frame 안):
    disp_frame_s = R_s · (gt_s - origin_s) ∈ R^3
Step 2 (inference for val_k sample): query 33D → k-NN → disp_frame_avg (k 개 평균, 3D, frame)
  frame_inversion: pred_s = origin_s + R_s^T · disp_frame_avg

k 선정: k ∈ {1, 3, 5, 10, 20} grid. 각 fold k 의 train_(not k) 안 nested CV (split into k_fit / k_eval) 로
       best_k 선택, val_k 평가에 사용.
Edge case:
  - faiss 미가용 → sklearn KNeighborsRegressor (성능 차이 marginal)
  - ||v_last_s|| < 1e-9 → C14 skip, sample 별 F0 baseline 으로 fallback
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
        # 9D 구성 (per timestep): [px, py, pz, vx, vy, vz, ax, ay, az] — **displacement units, Δt 분할 없음**
        #   (baseline_f0 / f0_form_torch 와 단위 통일 — F0 계수 1.98/1.20/-0.20 이 displacement 기반 fit 값)
        #   p[t] = x[t]
        #   v[t] = x[t] - x[t-1]              # displacement (not divided by Δt)
        #   a[t] = v[t] - v[t-1]              # finite-diff of displacement
        #   timesteps: end_idx-2, end_idx-1, end_idx → 3 × 9 flatten = 27D
        # NOTE: NN normalization (BatchNorm / per-feature scale) 가 단위 absorbs → 학습 자체는 unit-agnostic.
        #       단위 통일 의무는 *f0_form_torch 의 1.98·v_last term 의 산식 일관성* 때문.
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

# Final prediction = f0_form_torch(seq_feats, predicted_coef) — §4.2 export, torch 버전 (NN gradient path 유지)
```

- 학습: Adam lr=1e-3, batch=256, epochs=50, hit-aware loss schedule
- Device: cuda:1
- Seed list: [20260518, 20260519, 20260520, 20260521, 20260522] → best-on-train

#### N2. TCN F0 coefficient regressor (신규)

```python
class N02_TCNCoef(nn.Module):
    def __init__(self, seq_dim=9, hidden=32):
        super().__init__()
        # input: (B, 11, 9D), dilated TCN with dilations [1, 2, 4]
        # 9D = N1 과 동일 [px,py,pz, vx,vy,vz, ax,ay,az] per timestep (**displacement units, §N1 ↔ f0_form_torch 와 통일**).
        # f0_form_torch 호출 시 seq_feats 는 (B, 3, 9D) — 본 11D 의 마지막 3 timestep slice [end_idx-2..end_idx].
        # NOTE: PyTorch Conv1d 의 padding 은 symmetric (양쪽). 11-step 입력 전부 t ≤ end_idx 의
        #       *과거* 이므로 future leak 자체가 없어 strict causal 불필요. symmetric padding
        #       (kernel=3, dilation=d → padding=d) 으로 출력 length = 입력 length = 11 유지.
        self.tcn = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=1, dilation=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4), nn.SiLU(),
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
        # K=4 experts (all deterministic, NN training X — gating 만 학습):
        #   [0] F0_baseline      (0-param, plan-006 산식)
        #   [1] C1_helix         (§6 STAGE 2 의 fold-fit α/β/γ freeze 사용. C1 학습 미완 시 default 1.0)
        #   [2] C6_hermite       (0-param)
        #   [3] C2_ctra          (0-param)
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

→ K expert predictions caching 책임 (gating NN 만 학습):
  - module: `run_oof_nn` 의 N5 학습 직전 pre-compute step (fold k loop 안에서 호출)
  - 저장: `runs/baseline/Z020_N05_moe/fold_k/expert_preds.npy`, shape `(N_total=10000, K=4, 3)`, dtype `float32`.
    N_total = 전체 10000 sample (train_(not k) + val_k union — fold 와 무관하게 *전체* 색인).
    indexing convention: `expert_preds[sample_id]` 에서 sample_id ∈ [0, 10000) 는 dataset 의 global row index (fold-agnostic).
  - expert 순서 (axis=1) 고정: [0]=F0_baseline, [1]=C1_helix, [2]=C6_hermite, [3]=C2_ctra
  - inference: `expert_preds_batch = torch.from_numpy(preds[batch_sample_ids]).to(cuda:1)` 로 N5.forward 의 expert_preds 인자 주입
  - cache 의 stale 방지: 각 fold 학습 시작 시 한 번 재빌드 — C1_helix expert 의 α/β/γ 는 fold k 의 train_(not k) 위 fit 한 값 사용 (fold-internal). 따라서 expert_preds 파일은 *fold 마다 다름*.

→ N5 forward signature 가 N1/N2 와 다름 (`forward(seq, expert_preds)` vs `forward(seq_feats)`). `run_oof_nn` 의 dispatch:
  - N1 (input shape (B, 3, 9D)) : `pred_coef = model(seq_feats_3)` → `pred = f0_form_torch(seq_feats_3, pred_coef)`
  - N2 (input shape (B, 11, 9D)) : `pred_coef = model(seq_feats_11)`
    → **dispatch 호출자가 마지막 3 timestep slice 수행**: `seq_feats_3 = seq_feats_11[:, -3:, :]`
    → `pred = f0_form_torch(seq_feats_3, pred_coef)` (f0_form_torch 시그너처 (B, 3, 9D) 강제 → slice 책임은 dispatch, N2 module 내부 X)
  - N5 (input shape (B, 11, 9D) + expert_preds (B, K=4, 3)) : `pred = model(seq_feats_11, expert_preds_batch)` (gating NN, expert mixture 결과 = pred 직접, f0_form_torch 미경유)
  → 같은 hit 평가 loop 진입.

### §7.2 학습 spec (3 NN 공통)

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 50 |
| Early stop | **train_(not k) hit plateau 10 epoch** (val 의존 금지 — multi-seed selection bias 규칙 §7.2 와 일관). val 은 최종 평가 전용. |
| Device | cuda:1 |
| Seed list | [20260518..20260522] |
| Seed aggregation | 각 fold k 마다 5 seed 학습 → 각 seed 의 *train_(not k) hit@1cm* 으로 best 1 seed 선택 → 그 seed 의 val_k OOF 만 보고. 5 fold val_k concat = OOF (val metric 으로 seed 선택 시 selection bias 발생 → 금지). |
| Smooth hit surrogate | `smooth_hit(pred, gt; R, τ) = sigmoid((R − ‖pred − gt‖_2) / τ)`. τ→0 일수록 hard hit 1-indicator 에 수렴 (gradient 작아지지만 0 아님). |
| Boundary weighting (epoch 31-50) | sample weight `w_i = 1 + 5·exp(−((R − d_i.detach())/0.001)²)`. d_i = ‖pred_i − gt_i‖ 를 `detach()` 하여 weight 자체가 gradient 통로가 되지 않게 함 (loss gradient 가 surrogate 만 통과). |
| Loss schedule | annealed step (warmup 없음): epoch 0–15 smooth τ=0.003, 16–30 smooth τ=0.001, 31–50 **smooth τ=0.0003 + boundary weighting** (τ→0 한계로 수렴, 학습 가능성 보존 — hard 1-indicator 는 gradient=0 이라 사용 X). early stop 시 진행 중 schedule 만 적용. |
| Loss form | `L = − mean_i [ w_i · ( smooth_hit_i(R=0.01) + 0.5 · smooth_hit_i(R=0.015) ) ]` (scalarized multi-objective, hit@1cm 기준 weight). epoch < 31 에서는 w_i ≡ 1. |
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
- N1 의 OOF 가 plan-007 F002 (0.6482) 와 **±0.02 안** (`n1_drift_vs_f002` warn — architecture 다름 완화)

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

각 family (F1~F7) 안에서 winner 선정 = 2 단계 (pass criterion 우선, scalarization 은 tie-break):
  1) §3.2 pass criterion (paired Δ ≥ +0.005 *둘 다* — hit@1cm AND hit@1.5cm) 통과 후보만 candidates.
  2) candidates 중 *가장 큰 Δ_combined = Δ_hit@1cm + 0.5·Δ_hit@1.5cm* 후보 1개 = winner.
  - pass criterion 통과 0건 시 winner = "없음" 박제 (family-level negative finding).
  - winner objective 의 가중합은 ranking tie-break 용이며 pass 자격은 둘 다 ≥ +0.005 가 필수.

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
- `best_candidate: <후보 이름>` (overall winner — 아래 규칙으로 단수 선정)
- `best_hit_1cm: <float>`, `best_hit_1.5cm: <float>`

#### §9.1.1 overall best_candidate 단수 선정 규칙

§8.2 의 family winner 7 개 (F1~F7 각 1개, "없음" 포함 가능) 중 다음 순서로 **1 후보** 선정:
  1) §3.2 pass criterion (paired Δ ≥ +0.005 *둘 다*) 통과 winner 들로 candidates 집합 구성.
  2) candidates 중 *가장 큰 Δ_combined = Δ_hit@1cm + 0.5·Δ_hit@1.5cm* 후보 = `best_candidate`.
  3) tie (Δ_combined 동률) 시 hit@1cm 우선, 그 다음 fold variance (작은 쪽).
  4) candidates 가 빈 경우 (G3 0 통과) → `best_candidate: "없음"`, `band: negative`, results.md 의 negative finding 박제.

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

4. **N1 vs plan-007 F002 drift** (threshold ±0.02, architecture 다름 — 코드 재사용 검토에서 확인된 drift 원인):
   - **input feature 구성 차이 (가장 큰 원인 예상)**: F002 = 13D *통계 aggregates* (pos_mean/std/range 9D + speed_mean/std/max/last 4D) 위 6 timestep window. N1 = 27D *raw sequence* (last 3 × [px,py,pz, vx,vy,vz, ax,ay,az] displacement). 다른 paradigm.
   - **train pool 차이**: F002 = 50K (10K original + 4× sliding views). N1 = 10K (original 만, sliding view 미사용).
   - fold split 동일 (plan-004 carry, MD5).
   - seed 차이 (F002 = 20260606 single, N1 = 20260518..20260522 multi-seed best-on-train).
   - loss schedule 차이 (F002 = MSE? stage3 baseline 0.63868 추정, N1 = annealed smooth-hit + boundary).
   - **결론**: ±0.01 의 *strict* drift threshold 는 architecture 동일 가정 — 본 plan 은 paradigm-class 동일 / instance 다름이라 ±0.02 로 완화. 그래도 초과 시 architecture / pool / loss 어느 lever 가 dominant 인지 분리 측정 권고 (follow-up plan-021).

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
- v1.2 (2026-05-18): plan-review-master 5-iter 자동 fix (BLOCKER 0 도달, 37 fix). 산식 박제 (C2/C4/C6/C7/C8/C9/C10/C11/C14) + NN spec (N1 9D feature / N2 dilated TCN / N5 expert_preds caching) + f0_form_torch torch mirror + annealed loss surrogate + best_candidate 단수 선정 규칙. C10 Bishop F0-degeneracy 차단 (λ 1 param). C12 cutoff Nyquist-aware [2.27, 12.5]. C13 Lévy v1 본문 정정 — wingbeat range alias zone [10, 200] → Nyquist 안. velocity 단위 displacement units 으로 통일.
- v1.3 (2026-05-18): 코드 재사용 검토 (feedback_code_reuse_correctness) — 6 carry 항목 cascade + signature 사전 검토. **DRIFT/VIOLATION fix 3건**: (a) §3.1.1 `stable_fold_id` hash blake2b → MD5 정정 (실제 `selector.py` L185), (b) §3.1.1 `fit_regime_bins` 가 dict 반환이고 별도 `assign_regimes` 호출 필수임을 박제 (OO `.assign()` 가정 silent bug 회피), (c) §0.5 / §1.3 / §7.4 / §N+2 #4 의 N1 drift threshold ±0.01 → ±0.02 완화 + drift 원인 보강 (F002 13D 통계×6-step ≠ N1 27D raw×3-step + train pool 50K vs 10K — 동일 paradigm-class 다른 instance).

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
