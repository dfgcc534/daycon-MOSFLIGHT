---
plan_id: 019
date: 2026-05-15 (Asia/Seoul)
status: candidates_for_plan020
context: plan-019 G1/G2/G3 모두 WARN, single-stack 의 *실측 ceiling* ≈ A0 + 0.007 박제.
         plan-018 + plan-019 의 결합 결론: single-stack architecture / 단일 공식 framework 만으로
         plan-005 oracle 0.7188 / plan-004 LB 0.6822 도달 불가능. paradigm-shift 필수.
n_candidates: 4
recommended: A (corrector 결합)
---

# plan-020 후보 (≥ 2, paradigm-shift 필수)

## 우선순위 (cost × expected_gain)

| 후보 | cost | expected_gain | risk | priority |
|---|---|---|---|---|
| **A** corrector 결합 (plan-005/016 + S1) | low | +0.005~0.015 | low | **★ 권장** |
| B multi-stack (selector + boundary corrector) | high | +0.015~0.030 | medium | 2nd |
| C Learnable Basis + Sparse MoLE | medium | +0.005~0.015 | high (overfit) | 3rd |
| D DEQ | high | likely +0 (= S1) | high | not recommended |

---

## 후보 A: corrector 결합 (plan-005/016 + S1 EBIP base)

### Idea

- plan-019 S1 EBIP base (OOF=0.6552, ckpt @ `runs/baseline/F014_ebip-base/checkpoint_fold{0..4}.pt`) 의 5-fold OOF prediction 을, plan-005/016 corrector freeze 결과 와 *output ensemble* (좌표 mean) 으로 결합.

### Mechanism (plan-017 G1 carry)

- plan-017 G1 ensemble (LB 0.6640) 의 패턴: framework-disjoint 결합 = 좌표 mean 가 *서로 다른 error mode* 보강.
- candidates:
  - **C-1**: S1 EBIP base + plan-016 G1 BiGRU corrector (= plan-016 c1 c2 G1)
  - **C-2**: S1 EBIP base + plan-005 corrector freeze (= plan-005 D001 final)
  - **C-3**: S1 EBIP base + plan-016 G1 + plan-005 D001 (3-way mean)

### Spec

- preprocessing: 각 corrector 의 *test set inference* 결과 (좌표 (x, y, z) per sample_id).
- ensemble: 좌표 단위 weighted mean. weight = 좌표 별로 *OOF hit_rate inverse* (weighted by skill).
- 직접 학습 없음 — *inference-only*.

### Cost

- **low**: 30 분 inference + 5 분 ensemble + 1 LB submit.
- DACON quota 1 사용 (best ensemble combination).

### Expected gain

- plan-017 G1 ensemble (LB 0.6640) 가 plan-017 c1.4 best variant + plan-005 D001 의 결합 결과.
- 본 후보 = plan-019 S1 (paradigm-disjoint) 추가 → *complementarity 더 큼* 예상.
- 예상 LB: **0.66~0.68** (plan-017 G1 위 +0.005~0.015).

### Risk

- low: 좌표 mean 은 catastrophic interference 없음.
- 단점: plan-019 S1 의 OOF 가 0.6552 로 plan-005/016 보다 *낮음* — ensemble weight 자체가 작아질 가능성.

---

## 후보 B: multi-stack (selector + boundary corrector)

### Idea

- plan-004 LB 0.6822 의 full ensemble 의 *single-model 변형*. selector (anchor classify) + boundary corrector (high-error region focus). plan-019 S1 EBIP module 을 corrector 로 재사용.

### Mechanism

- plan-005 oracle 0.7188 = "best of 27 candidates per-sample" — sample-별 *optimal candidate* 가 다름.
- plan-004 LB 0.6822 = "27 candidates 각각의 inference + selector ensemble" — full ensemble.
- 본 후보 = "27 candidates → 8 basis 의 *learned selector*" — single-model 회수.
- Stage 1: selector (per-sample classify into "easy" / "medium" / "hard" region by predicted-error magnitude).
- Stage 2: easy region → S1 EBIP base 직접 inference. medium / hard region → S1 EBIP base + boundary corrector (residual MLP) 추가.

### Cost

- **high**: selector 학습 + boundary corrector 학습 + joint inference.
- 약 1-2 일 (구현 + 학습).

### Expected gain

- high-error region 회수 — plan-005 D001 의 *measured* high-error 분류 (≥ 0.04 m error 의 ~30%) 의 일부 회수.
- 예상 LB: **0.67~0.69** (plan-019 S1 위 +0.015~0.030).

### Risk

- medium: selector accuracy 가 OOF 영향 결정. plan-004 의 full ensemble selector (GRU + attn) 의 single-model 회수율 미확정.

---

## 후보 C: Learnable Basis + Sparse MoLE (brainstorm #4 carry)

### Idea

- 8 fixed basis 를 *learnable embedding* (Koopman lift / Fourier feature) 으로 확장.
- plan-018 A3 MoLE (head capacity) + 본 후보 (basis capacity) 결합.

### Mechanism

- basis 확장: 8 fixed → 8 + 16 learnable (= 24 total). learnable basis 는 *trajectory window 의 nonlinear lift*.
- Koopman lift: f(x) ≅ Σ φ_i(x) (φ_i learnable). 본 후보 의 변형: f(traj_window) = MLP → 16-d.
- MoLE: K=16 experts, top-2 routing — 다양한 expert 가 trajectory regime 별 basis 학습.

### Cost

- **medium**: basis 학습 + MoLE 학습.
- 약 4-6 시간 (구현 + 학습).

### Expected gain

- single-stack 의 *basis 자체 확장* 으로 ceiling push.
- plan-018 + plan-019 의 결합 결론으로 *single-stack ceiling 자체가 0.65~0.66* — basis 확장 효과 의문.
- 예상 LB: **0.66~0.67** (S1 위 +0.005~0.015).

### Risk

- **high (overfit on 10K)**: learnable basis 의 표현력 ↑ vs 데이터 ↓.
- plan-018 A3 MoLE 도 single-stack 의 head capacity 만 변경, OOF +0.0003 marginal — 본 후보의 basis 확장도 유사 ceiling 예상.

---

## 후보 D: DEQ (brainstorm #2)

### Idea

- 본 plan 의 unrolled GD T=5 (S1) 또는 T=3 (S2) 의 *infinite-depth limit*.
- Deep Equilibrium Model (Bai et al. 2019): fixed-point iteration p* = f(p*, traj) 까지 수렴.

### Mechanism

- forward: Anderson acceleration / Newton 으로 fixed point 찾음.
- backward: implicit differentiation (Jacobian inverse).
- 본 plan 의 unrolled GD 가 5-step 으로 underfit (energy minimum 미도달) 가능성 → DEQ 로 *진짜 implicit prediction* 달성.

### Cost

- **high**: DEQ 안정성 (Anderson 수렴), Jacobian implicit diff.
- 약 1-2 일 (구현 + 학습 + 안정화).

### Expected gain

- 본 plan 의 S1 (T=5) 결과 OOF +0.0070 — DEQ 가 수렴까지 가도 *동일 ceiling* 예상.
- 예상 LB: **0.65~0.66** (= S1 carry-over).

### Risk

- **high**: implicit diff 의 numerical stability + paradigm-similar 한계 confirmed.

### Verdict

- **권장 X** — single-stack 한정이므로 paradigm-shift 가 아닌 *unrolled GD 변형*. ceiling break 의 통로 아님.

---

## 종합 권장

**plan-020 = 후보 A (corrector 결합)** — cheapest, fastest, plan-017 G1 ensemble pattern 답습. 단일 plan-020 으로 최저 cost 진행.

- input: S1 EBIP base (F014) ckpt × plan-005 D001 + plan-016 G1 BiGRU corrector
- output: 좌표 weighted mean ensemble submission
- LB target: ≥ 0.67 (plan-017 G1 0.6640 위 +0.005~0.015)

후보 B (multi-stack) 는 plan-020 또는 plan-021 — 더 ambitious path, ROI 큼.

decision-note: brainstorm 6 candidates 의 single-stack 한정 결론 — paradigm-shift 가 ceiling break 의 유일 path. 본 plan-019 의 *실측 ceiling* ≈ A0 + 0.007 박제 후 plan-020 진행.
