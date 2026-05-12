# plan-007 → plan-008 후보

## 시나리오 분기 (§9.2)

**Step 4 OOF gain vs Step 3 = +0.0095** (≤ +0.010 → **시나리오 B**)

> G3 (+0.005) ≠ scenario A (+0.010) — MLP 가 noise 위 미세 개선 (G3 통과) 했으나 *주력 무기* 수준 (scenario A) 미달. plan-007 의 단일 공식 + per-sample MLP framework 의 ceiling 측정 결과 = plan-006 baseline (0.6491) 과 거의 동급 — **단일 공식 framework 의 자체 한계 인정**.

---

## 후보 1: 단일 공식 framework 한계 인정 → 27 후보 풀 확장 (35+) **[추천]**

### 1.1 근거 metric

| 측정 | 값 | 함의 |
|---|---|---|
| plan-006 baseline (argmax + corrector) | 0.6491 | 현 framework ceiling |
| Step 4 MLP OOF (per-sample 단일 공식) | 0.6482 | per-sample 적응 후도 동급 |
| oracle (best of 27, raw) | 0.7188 (plan-005) | 후보 풀이 충분히 크면 도달 가능 ceiling |
| ceiling gap (plan-006 vs oracle) | 0.0697 (~7pp) | 후보 풀 확장의 *potential* room |
| plan-005 worst-100 분석 (regime 16/17) | hit 0.22~0.26 | 새 family 의 직접 타깃 후보 |

→ 단일 공식 + per-sample 적응이 +0.0095 만 회수했다는 사실은 단일 공식 framework 의 한계 신호. *후보 풀 자체를 확장* 해 selector arch 가 더 다양한 후보 중 best 를 고를 수 있게 하는 것이 plan-005 oracle 0.7188 와의 7pp gap 회수에 더 효율적.

### 1.2 예상 ROI

- **+1~3pp OOF** (보수적). plan-005 oracle gap 0.0697 의 일부 회수.
- **+0.5~1.5pp LB**. plan-006 LB 0.6692 → 0.68~0.69 추정.
- *high-leverage*: 5-10 개 후보 추가로 immediate measurable gain.

### 1.3 작업 범위

1. plan-005 worst-100 sample 의 ground-truth 식 회귀 (rotation-dominated, banking, 가속-감속 변화 etc.) 식별
2. 새 8 family 후보 추가:
   - banking_rotation (z 축 회전, plan-007 §N+3 #2 caveat 의 3D rotation)
   - acc_decay_extrapolation (가속 감쇄 + linear extrapolation 혼합)
   - jerk_amplifier (jerk 양수 sample 에서 3-step lookahead)
   - speed_dependent_par (speed 별 par coefficient 분기)
   - perp_correction_strong (regime 17 의 strong perp correction)
   - quadratic_jerk (jerk 의 quadratic term)
   - mean_velocity_3step (v_mean3 framework)
   - frenet_par100_perp_neg035 (plan-006 worst 분석의 직접 candidate)
3. selector 재학습 (plan-004 framework 그대로 — 27 → 35 후보) 후 LB 제출

### 1.4 선행 조건

- plan-005 의 worst-100 분석 산출 (이미 박제됨)
- plan-007 Step 2/3 의 LB 회수 (TBD → 실측 값)
- plan-004 src/pb_0_6822/{selector,boundary}.py import-only (수정 X)

---

## 후보 2: corrector 재설계 + Step 4 MLP OOF 결합

### 2.1 근거 metric

| 측정 | 값 | 함의 |
|---|---|---|
| Step 4 MLP OOF | 0.6482 | per-sample adaptive prediction available |
| plan-005 corrector_decomp +0.89pp | 0.6402 → 0.6491 (raw → corrected) | corrector 의 boundary 효과 박제 |
| MLP near-miss band hit 분포 (Step 4 oof_predictions.npz) | (분석 미실시) | corrector 적용 가능성 |
| plan-008 corrector 의 target | band-specific (1~1.5cm, 1.5~2cm) | known gain region |

→ plan-007 Step 4 MLP 의 OOF predictions (`oof_predictions.npz`) 가 *per-sample* 으로 박혀 있어, plan-008 의 band-specific corrector (plan-005 의 +0.89pp 효과) 와 결합하면 추가 회수 가능.

### 2.2 예상 ROI

- **+0.5~1.5pp OOF** (corrector 의 known gain ~0.89pp 가 single-formula 가 아닌 MLP 위에도 동작한다고 가정).
- **+0.3~1pp LB**. plan-006 LB 0.6692 → 0.675~0.685 추정.
- *medium-leverage*: 후보 1 대비 작은 gain 이지만 코드 변경 적음.

### 2.3 작업 범위

1. plan-007 Step 4 `runs/baseline/F002_formula-mlp/oof_predictions.npz` 의 err 분포 분석 (1cm, 1.5cm, 2cm band 별 hit/miss)
2. plan-005 의 corrector_decomp.{json,md} 재학습 — MLP per-sample prediction 위 (raw 위가 아님)
3. band-specific corrector 학습 (predicted err 가 1~1.5cm 인 sample 에 한해 *작은* correction 적용)
4. F003_mlp-corrector 산출 + LB 제출 (1 회)

### 2.4 선행 조건

- plan-007 Step 4 산출 (`oof_predictions.npz`) — *완료* (이미 박제)
- plan-005 의 corrector_decomp framework — *완료* (plan-005 §5)
- plan-007 Step 2/3 LB 회수 (TBD → 실측, 본 plan-008 의 baseline 박제 용)

---

## 후보 3: Step 4 LB 제출 단독 (carry-over)

### 3.1 근거 metric

| 측정 | 값 |
|---|---|
| Step 4 MLP OOF | 0.6482 |
| plan-006 LB | 0.6692 |
| 예상 LB | 0.66~0.67 (OOF vs LB gap 추정 +0.02pp) |

→ plan-007 Step 4 의 OOF (0.6482) 가 LB 에서 어떻게 나오는지 *측정만* 하는 carry-over. 비용 최소, 정보 max (단일 공식 framework 의 LB ceiling 박제).

### 3.2 예상 ROI

- **0 ~ -0.01pp LB** (likely no improvement vs plan-006 0.6692 LB).
- 그러나 *정보 가치*: 단일 공식 framework 의 LB ceiling 확정 → 후보 1 / 2 의 성공 여부 평가 기준.

### 3.3 작업 범위

- F002_formula-mlp/oof_predictions 의 *test set* 추론 (Step 4 model 5 fold ensemble 또는 best fold) + dacon-submit 1 회.

### 3.4 선행 조건

- plan-007 Step 4 산출 (완료)
- DACON 일일 5회 제한 (현재 2회 사용; 3회 남음)

---

## 권장 우선순위

1. **후보 1 (단일 공식 framework 한계 인정 → 27 후보 풀 확장)** — *추천*. plan-005 oracle gap 의 가장 큰 ROI.
2. **후보 3 (Step 4 LB 단독 제출)** — 빠른 측정, 후보 1 의 baseline 박제 가치.
3. **후보 2 (corrector 재설계)** — 후보 1 의 산출 (확장된 27→35 후보 풀) 위에 적용하면 시너지.

plan-008 의 §0 한 줄 목적 = "후보 1 + 후보 3 결합 — 새 8 family 후보 추가 후 plan-007 Step 4 + 새 풀의 LB 비교" 추천.
