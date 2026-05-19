---
plan_id: 022
finished_at: 2026-05-19 (Asia/Seoul)
status: all_complete
band: positive
best_sub_exp: A6_bcc14_tau001
best_hit_1cm: 0.6528
best_hit_1.5cm: 0.8104
best_delta_1cm: +0.0208
best_delta_1.5cm: +0.0071
exp_ids_completed:
  - Z022_A1_octa7
  - Z022_A2_ico13
  - Z022_A3_cubocta13
  - Z022_A4_2shell13
  - Z022_A5_cube8
  - Z022_A6_bcc14
  - Z022_A7_fib13
exp_ids_skipped: []
lb_score: null
---

# plan-022.results — Corrector-free Anchor Layout Sweep (selector-only LGBM)

## 1. plan-021 → plan-022 narrative bridge

plan-021 의 5 finding (corrector dead, mode collapse, etc.) 위 simplification + lever scan:
- corrector reg head 제거 → 모델 단순화 + plan-021 LGBM full Δ_1.5cm +0.0037 미달 한계 돌파
- 7 anchor layout × 3 τ_cls = 21 cell 전수 측정
- pass criterion = paired Δ_1cm ≥ +0.005 AND Δ_1.5cm ≥ +0.005

## 2. 21 cell grid (layout × τ_cls)

| layout | K | τ=0.001 | τ=0.003 | τ=0.005 |
|---|---|---|---|---|
| A1_octa7 | 7 | +0.0194/+0.0068 ✓ | +0.0112/+0.0048 ✗ | +0.0073/+0.0034 ✗ |
| A2_ico13 | 13 | +0.0199/+0.0070 ✓ | +0.0105/+0.0051 ✓ | +0.0068/+0.0036 ✗ |
| A3_cubocta13 | 13 | +0.0196/+0.0074 ✓ | +0.0119/+0.0049 ✗ | +0.0070/+0.0036 ✗ |
| A4_2shell13 | 13 | +0.0176/+0.0061 ✓ | +0.0081/+0.0037 ✗ | +0.0045/+0.0025 ✗ |
| A5_cube8 | 8 | +0.0180/+0.0076 ✓ | +0.0105/+0.0049 ✗ | +0.0075/+0.0035 ✗ |
| **A6_bcc14** | **14** | **+0.0208/+0.0071 ✓ 🏆** | +0.0129/+0.0058 ✓ | +0.0084/+0.0036 ✗ |
| A7_fib13 | 13 | +0.0201/+0.0073 ✓ | +0.0107/+0.0052 ✓ | +0.0056/+0.0031 ✗ |

10/21 cell pass_both=True. **0 dropped** (max_class_ratio 모두 < 0.95).

## 3. Best cell 🏆

**A6_bcc14_tau001** — BCC 14 anchor (6 axis + 8 corner, NO center), τ_cls=0.001:
- hit@1cm = **0.6528** (F0 0.6320 → +0.0208 paired Δ)
- hit@1.5cm = **0.8104** (F0 0.8033 → +0.0071 paired Δ)
- pass_both ✓, max_class_ratio = 0.105, fold_var_1cm = 0.0044

## 4. Layout-axis marginal (각 layout 의 best τ)

| rank | layout | best τ | Δ sum | Δ_1cm | Δ_1.5cm |
|---|---|---|---|---|---|
| 1 | A6_bcc14 | 0.001 | **+0.0279** | +0.0208 | +0.0071 |
| 2 | A7_fib13 | 0.001 | +0.0274 | +0.0201 | +0.0073 |
| 3 | A3_cubocta13 | 0.001 | +0.0270 | +0.0196 | +0.0074 |
| 4 | A2_ico13 | 0.001 | +0.0269 | +0.0199 | +0.0070 |
| 5 | A1_octa7 | 0.001 | +0.0262 | +0.0194 | +0.0068 |
| 6 | A5_cube8 | 0.001 | +0.0256 | +0.0180 | **+0.0076** |
| 7 | A4_2shell13 | 0.001 | +0.0237 | +0.0176 | +0.0061 |

## 5. τ_cls-axis marginal (모두 A6 winner)

| τ_cls | best layout | Δ_1cm | Δ_1.5cm | n PASS (across 7 layouts) |
|---|---|---|---|---|
| **0.001** | A6_bcc14 | **+0.0208** | +0.0071 | **7/7** |
| 0.003 | A6_bcc14 | +0.0129 | +0.0058 | 3/7 |
| 0.005 | A6_bcc14 | +0.0084 | +0.0036 | 0/7 |

## 6. Mode collapse 완화 finding

max_class_ratio (soft-mean 의 최대 anchor share) 분석:

| K | max_class @ τ=0.001 | uniform 1/K | ratio |
|---|---|---|---|
| 7 (A1) | 0.232 | 0.143 | 1.62× |
| 8 (A5) | 0.165 | 0.125 | 1.32× |
| 13 (A2/A3/A4/A7) | 0.116-0.151 | 0.077 | 1.51-1.96× |
| 14 (A6) | **0.105** | **0.071** | **1.48×** |

- **K 증가 → max_class_ratio 절대값 감소**, winner-take-all 자동 완화. plan-021 의 GRU "anchor 1/5/6 dead" mode collapse 가 LGBM selector-only 에선 발생 안 함 (모든 cell collapse < 0.25).
- A4_2shell13 의 max_class_ratio 0.116 (가장 낮음 K=13) — center + inner shell 의 prob mass spread 효과지만 **성능은 worst**. distribution 의 평탄성이 곧 좋은 prediction 아님.

## 7. plan-021 baseline 대비 향상

| metric | plan-021 A LGBM full (with reg head) | plan-022 A6_bcc14_tau001 (corrector-free) | Δ |
|---|---|---|---|
| hit@1cm | 0.6488 | **0.6528** | +0.0040 |
| hit@1.5cm | 0.8070 | **0.8104** | +0.0034 |
| Δ_1cm | +0.0168 | **+0.0208** | +0.0040 |
| Δ_1.5cm | +0.0037 ✗ | **+0.0071** ✓ | +0.0034 (PASS 전환) |
| pass_both | partial | **🎉 True** | — |

**plan-022 corrector-free 가 plan-021 corrector-full 보다 우월** — reg head 가 LGBM 에서 noise 였음 (plan-021 selector-only ablation finding 정합). + anchor layout 변경으로 1.5cm metric 미달 한계도 해결.

## 8. Paradigm-level finding

### 8.1 BCC 14 winner mechanism
- octahedron 6 axis: 단일축 오류 (1cm tight zone) sharp cover
- cube 8 corner: 3축 결합 오류 (1.5cm zone) coverage
- 두 paradigm 동시 활성화 → 1cm 최강 (Δ +0.0208) + 1.5cm 견조 (+0.0071) + 최저 mode collapse (max_class 0.105)
- **center 제거** = F0 over-pick 차단 (A5, A6 공통) → reg head 없이도 1.5cm 향상
- 사용자 narrative (초기 brainstorming 의 14-anchor 직관) 합치

### 8.2 H1/H2/H3/H4 검증 결과

| 가설 | 결과 | 증거 |
|---|---|---|
| H1 layout 효과 | supported | 13-14 layout > 7 layout (sum 0.027 vs 0.026) |
| H2 τ 효과 | refuted (반대) | τ=0.001 sharp 가 모든 layout 에서 best, 완화 시 monotonic Δ 감소 |
| H3 center 제거 | partial | A5 cube8 의 1.5cm 0.0076 = 최강 / A6 (no center) winner sum |
| H4 2-shell | **refuted** | A4_2shell13 sum 0.0237 = worst |

### 8.3 τ 효과 paradigm-level

`τ_cls = 0.001` (sharp soft label, q 거의 one-hot) 이 모든 layout 에서 최선. **anchor 추가 ↑ + τ 완화 ↑ 의 결합** 가설 (anchor 많아지면 sharp soft label 이 collapse 위험 → τ 완화 필요) **반박**. K=14 안에서도 sharp τ 가 best.

mechanism 추정: residual_true_frenet 가 anchor 격자 spacing 보다 훨씬 작은 분해능 (~0.0005m 이하) 으로 분포 → sharp soft label 이 정답 anchor 정확 지목 능력 가짐. τ 완화 = 정답 신호 약화 (noise 추가).

## 9. Follow-up plan 후보

- **plan-023 (가칭)**: best layout (A6_bcc14) + τ_cls=0.001 위 **corrector reg head 재투입** ablation. plan-022 가 reg head 제거로 향상됐지만, 다른 anchor layout 에선 reg head 가 의미 있을 가능성 (특히 1.5cm 미달 metric 보강). reg bound 범위 scan ({±0.005m default, ±0.0025m tight, ±0.01m loose}) + reg head sample_weight 조정.
- **plan-024 (가칭)**: A6_bcc14_tau001 위 **GRU sub-exp** + ensemble. plan-021 GRU 0.6408/0.8100 vs plan-022 A6 0.6528/0.8104. ensemble 잠재력 (GRU + LGBM marginal-disjoint sample 보완) 측정.
- **plan-025 (가칭)**: best layout + DACON LB 측정. plan-024 confirmed best 위 submit (사용자 5회 quota confirm 필수).

## 10. Severe / warn 박제

- **0 severe** 발동 (lgbm_numerical / f0_reproduce_drift / frenet_basis_degenerate 모두 미발동)
- **0 warn**: soft_label_collapse 도 미발동 (모든 max_class_ratio < 0.95, 실제 max = 0.232 in A1_tau0.001)
- all_negative 미발동 (10/21 cell pass_both)

## 11. Reproducibility 박제

- dataset_hash = `b91502db94fab67d` (10000 sample, plan-020/021 carry seed)
- F0 baseline: hit@1cm = 0.6320, hit@1.5cm = 0.8033 (plan-020 carry exact, ±0)
- fold split: stable_fold_id MD5 32-bit prefix mod 5, deterministic
- LGBM: n_estimators=500, lr=0.05, num_leaves=63, random_state=20260519
- 21 cell elapsed: total ≈ 4630s (~77 min)
- plan-021 module reuse: build_input.py (170D pipeline), dual_head_model.py:LgbmDualHead config carry

## artifacts

- `plans/plan-022-corrector-free-anchor-layout-sweep.md` (frontmatter sync)
- `analysis/plan-022/anchors.py`, `selector_only_model.py`, `run_oof.py`, `paradigm_analysis.py`, `baseline_carry.py`
- `analysis/plan-022/results_A1.{json,md}` ... `results_A7.{json,md}`
- `analysis/plan-022/paradigm_analysis.{json,md}`
- `analysis/plan-022/baseline_carry.json`
- `tests/test_plan022_smoke.py` (8 pytest pass)

## 12. Post-G_final 자체실험 — A8 center bias ablation (2026-05-19)

### 12.1 동기

본 sweep G3 §8.1 의 paradigm finding 은 "A6_bcc14 winner = **center 제거** 가
F0 over-pick 차단 → mode collapse 완화" 였다. 이 결론은 (a) 21-cell `max_class_ratio`
의 *aggregate* 비교 (center-있음 5종 vs 없음 2종) 와 (b) A1_octa7 의 0.232 max_class
시그니처에 근거. **두 약점**:

1. `max_class_ratio = probs_all.mean(axis=0).max()` 는 selector 출력의 최대 anchor
   share 만 봄. 진짜 봐야 할 건 *ground-truth 자연 분포* `q_true.mean(axis=0)` 와의
   일치도. 자연 분포가 한 anchor 에 mass 가 몰리는 layout 에선 selector 가 그를
   충실히 추종해도 max_class_ratio 가 자동 상승 — collapse 아닌 *correct tracking*.
2. center-있음 vs 없음 비교가 *controlled* 가 아님 (A1/A2/A3/A4/A7 vs A5/A6 는 geometry
   도 다름). 같은 anchor set 위 **center 만 toggle** 한 짝이 부재.

⇒ **A8_bcc15 = A6_bcc14 + center** (K=15) 설계. 본 sweep 의 21-cell invariant 보존
   위해 `LAYOUT_NAMES` 외부 export (`anchors.ANCHORS_A8`). A6 도 동일 코드로 재측정
   하여 진단 코드 검증 + 직접 비교.

### 12.2 진단량

표준 metric (hit, Δ, max_class_ratio) 외 다음 4 distribution-match 측정량 추가:

- **`q_true_max`** = `q_true.mean(axis=0).max()` — *자연* 분포의 최대 anchor share
- **`dist_match_KL`** = `KL(probs_avg || q_true_avg)` — selector aggregate ↔ ground-truth aggregate divergence (0 이면 완벽 매칭)
- **`top1_acc`** = `(probs.argmax(axis=1) == q_true.argmax(axis=1)).mean()` — sample-level 정답 anchor 적중률
- **`soft_CE`** = `-(q_true * log probs).sum(axis=1).mean()` — sample-level cross-entropy
- 부가: `natural_max_idx`, `pred_max_idx`, `natural_max_is_center`, `pred_max_is_center`

### 12.3 6 cell (A6 / A8 × 3 τ_cls) 결과

| layout | τ | K | hit@1cm | Δ_1cm | Δ_1.5cm | pass | max_class | q_true_max | KL | top1_acc | soft_CE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A6_bcc14 | 0.001 | 14 | 0.6528 | +0.0208 | +0.0069 | ✓ | 0.105 | 0.097 | 0.0022 | 0.1707 | 2.5346 |
| A6_bcc14 | 0.003 | 14 | 0.6449 | +0.0129 | +0.0058 | ✓ | 0.085 | 0.083 | 0.0001 | 0.1642 | 2.6056 |
| A6_bcc14 | 0.005 | 14 | 0.6404 | +0.0084 | +0.0036 | ✗ | 0.080 | 0.079 | 0.0000 | 0.1600 | 2.6256 |
| **A8_bcc15** | **0.001** | **15** | **0.6524** | **+0.0204** | **+0.0079** | **✓** | **0.135** | **0.111** | **0.0051** | **0.2415** | **2.5294** |
| A8_bcc15 | 0.003 | 15 | 0.6439 | +0.0119 | +0.0051 | ✓ | 0.095 | 0.085 | 0.0008 | 0.2348 | 2.6692 |
| A8_bcc15 | 0.005 | 15 | 0.6391 | +0.0071 | +0.0038 | ✗ | 0.084 | 0.080 | 0.0001 | 0.2350 | 2.6928 |

### 12.4 A8 vs A6 head-to-head 비교 (모든 τ)

| 지표 | τ=0.001 | τ=0.003 | τ=0.005 | 평균 |
|---|---|---|---|---|
| Δ(hit@1cm) | -0.0004 | -0.0010 | -0.0013 | **-0.0009 (seed noise 수준)** |
| Δ(hit@1.5cm) | +0.0010 | -0.0007 | +0.0002 | **+0.0002 (tied)** |
| Δ(top1_acc) | **+0.0708** | **+0.0706** | **+0.0750** | **+0.0721 (1.4× 일관 우위)** |
| Δ(soft_CE) | -0.0052 | +0.0636 | +0.0672 | +0.0419 |
| Δ(KL) | +0.0029 | +0.0007 | +0.0001 | +0.0012 (둘 다 ≤ 0.005) |

### 12.5 핵심 finding — H_center_collapse 가설 **refuted**

1. **OOF metric 영향 없음**: A8 가 1cm 에선 평균 -0.9bp (seed variance 사이즈),
   1.5cm 에선 평균 +0.2bp (tied). center 추가가 OOF hit metric 을 의미있게
   움직이지 않음 → "center 제거 = 1cm/1.5cm 향상" 인과 narrative **기각**.

2. **`max_class_ratio` 가 degenerate proxy 임 직접 입증**: A8 `max_class=0.135`
   > A6 `max_class=0.105`. 단순 비교라면 "A8 가 더 collapsed". 그러나 A8 의
   `q_true_max=0.111` > A6 `q_true_max=0.097` — *자연 분포 자체* 가 더 concentrated
   (center 가 11.1% 로 자연스럽게 가장 인기). KL=0.0051 (여전히 매우 작음)
   → selector 는 자연 분포를 충실히 추종 중. **max_class_ratio 의 차이는
   selector 의 행동이 아니라 layout geometry 가 만든 ground-truth 차이의 mirror**.

3. **Natural max anchor identity 직접 확인**: A8 의 `natural_max_idx=0` (center),
   `pred_max_idx=0` (center) — selector 가 자연 분포의 최대 anchor 를 정확히
   식별. A6 는 `natural_max_idx=1` (`+t̂` axis anchor), `pred_max_idx=1`. 두 layout
   다 selector 의 argmax 가 ground-truth 의 argmax 와 일치.

4. **top1_acc 향상의 함정**: A8 가 sample-level top-1 accuracy 는 모든 τ 에서
   +7pp (1.4×) 일관 우위. 그러나 OOF hit 으로 translation 안 됨. 이유 추정 — center
   anchor 픽 = "no-op correction" 이므로 *F0 가 이미 옳은 sample 에서 center 를
   고른다 = 옳지만 redundant*. A6 는 center 가 없어 "어느 axis anchor 든 작은
   weight 로 vote → soft-mean ≈ 작은 correction ≈ effectively no-op" 로 동일 결과
   달성. 즉 top1_acc 는 "정답 anchor 정확 식별" 을 측정하지만, hit metric 은 그것이
   "soft-mean prediction" 으로 변환되는 과정에서 center 와 거의 무 차이.

5. **plan-022 paradigm finding §8.1 의 부분 수정**: "center 제거 = OOF 향상 원인"
   은 controlled 검증에서 **기각**. A6 가 sweep winner 인 건 여전하지만, 그 우위는
   ① anchor geometry (BCC 14 의 axis + corner 조합) ② sharp τ=0.001 의 작용이지
   center 제거 자체가 아님. center 는 OOF 에 **neutral**.

### 12.6 함의

- **best layout 결론 불변**: A6_bcc14_tau001 이 여전히 sweep winner (Δ_1cm
  +0.0208 vs A8 +0.0204). 차이 4bp 는 seed noise 사이즈이므로 *practical tie* 로
  볼 수도 있으나, 제출 선택 시 A6 권장 (단순성 + sweep 박제).
- **mode collapse 진단법 교체 권고**: 후속 plan 의 진단 항목은 `max_class_ratio`
  대신 `dist_match_KL` (selector ↔ q_true aggregate) 와 `top1_acc` 를 사용. 자연
  분포 reference 가 있으므로 layout geometry 와 무관한 *true collapse* 측정 가능.
- **"center 가 nothing-anchor" 의 함의**: center 추가는 학습 부담 +K class 만
  발생시키고 metric 변화는 0 → 후속 layout 설계 시 center 포함/제거는 OOF 영향
  factor 가 아닌 *parsimony* (=K 작을수록 좋음) 만 따지면 충분. 다른 lever (
  reg head 재투입, ensemble paradigm, F0 자체 향상) 에 우선순위.

### 12.7 Artifacts

- `analysis/plan-022/diag_center_bias_a6_a8.py` — 진단 스크립트 (155 줄)
- `analysis/plan-022/diag_a8.json` — 6 cell 전체 결과 (12 진단량)
- `analysis/plan-022/diag_a8.log` — fold-by-fold console log
- `analysis/plan-022/anchors.py:ANCHORS_A8` — A8 정의 export (LAYOUT_NAMES 외부)

총 wall-clock = 1457s (~24 min, 6 cell × 5-fold OOF, 데이터 N=10000).
