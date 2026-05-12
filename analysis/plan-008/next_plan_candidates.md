# plan-008 → plan-009 후보 (Step 4 LB 0.65~0.67 시나리오 B/C)

본 plan 의 LB 예상 0.6503 + 0.022 gap ≈ **0.672** (carry-over to plan-008.1)
→ §10.2 시나리오 분류 = **C** (0.70~0.75 미만 — 시나리오 D 경계).

## 후보 1 — Selector arch 교체 (TCN / Transformer / MLP coeff regression) ★ main lever

**근거 metric**:
- gap_ranking 0.0516 (base 27) → 0.1119 (extended 25): selector 의 ranking 능력 한계
  가 후보 풀 확장과 무관하게 동작.
- top1_ranking_accuracy 0.126 (base) → 0.172 (extended): 17% 만 정확한 best 픽 →
  83% sample 에서 sub-optimal pick.
- main_bottleneck = "ranking" (diag c2 확정, gap_ranking ≫ gap_drift).
- caveat #13 (plan-007 framework 대체 시도 실패) 의 *plan-009 main task* 직접 trigger.

**예상 ROI**:
- top1_ranking 17.2% → 30%+ (2x 회수) 시 oof_soft_hit ≥ 0.70+ 도달 가능 (oracle 0.7562
  / 17.2% rank acc → 0.65 vs 30% rank acc → 0.72 추정).
- LB 회수 ~+0.05 (current 0.65 → 0.70 LB-OOF gap +0.022 = LB 0.72).

**작업 범위**:
- selector.py arch 후보: (a) TCNSelector (existing, plan-005 측정 약함), (b) Transformer
  4-layer (positional encoding + multi-head attn), (c) MLP per-sample coefficient
  regression (plan-007 framework 재사용 + extended pool로 회귀).
- ranking-specific loss: pairwise margin × 2.0 (current 0.25), fine_distill × 2.0
  (current 0.55), hardness-weighted softmax (top-K loss).
- 신규 ablation: existing 27 (base) vs extended 25 — selector arch 변화 효과 분리.

**선행 조건**:
- plan-008 의 extended pool (25 cands) + sanity baseline (27 cands) 활용.
- §0.5 의 G2 fallback path (hidden 64 + pairwise + distill + epoch_plus) 가 *arch
  교체 전 vs 후* 의 ablation 기준선 (총 5 측정).

## 후보 2 — boundary.py compute_corrector_loss hook 신설 + Step 4 band-specific 본격 적용 ★ plan-008 §7 carry-over

**근거 metric**:
- plan-005 corrector_oracle_gain = −0.0077 (corrector 가 oracle 0.7188 → 0.7111 깎음).
- plan-005 corrector_decomp: [0.5,1cm) hit 100→92.17% (−7.83pp), [1,1.5cm) hit
  0→9.77% (+9.77pp) — band-specific imbalance 명백.
- plan-008 c9 entry 시 boundary.py 의 LOSS_ATTR 부재 확정 → §7.3 spec 의 monkey-patch
  전략 *infeasible* in current codebase.

**예상 ROI**:
- plan-005 의 corrector_oracle_gain −0.0077 회복 + [1, 1.5cm) band 회수 강화 → 전체
  OOF +0.02~0.04 (plan-008 §1.4 H5 산술 derivation 그대로).

**작업 범위**:
- boundary.py 의 `train_net()` 의 reg=L2 loss 계산을 module-level callable
  `compute_corrector_loss(pred, target, raw=None, weight=None)` 로 추출.
- default 동작 (L2 loss) 보존 + monkey-patch 가능 hook 노출.
- plan-008 §7.2 의 `band_specific_corrector_loss` (이미 spec 완성) + cap fallback
  (§7.4 grid search 5λ × 2cap = 10 시도) 직접 적용.
- G3 합격 기준: per-band [0,0.5)≥0.99 / [0.5,1)≥0.95 / [1,1.5)≥0.30 + 전체 OOF
  ≥ Step3 OOF + 0.02.

**선행 조건**:
- boundary.py 본문 수정 (lock-in 해제 — plan-009 의 scope decision 필요).
- plan-008 c7 의 EXTENDED_CANDIDATES + selector OOF (post-arch-swap) 가 base.

## 후보 3 (선택) — Test-internal hyperparam re-tune + plan-007 MLP coeff 재시도

**근거 metric**:
- plan-007 Step 4 MLP coeff regression OOF 0.6482 (단일 공식 ceiling 입증) — extended
  pool *위에서* 재시도 가치.
- plan-008 §3.1 의 test_internal (50K subsamples) 활용 미흡 (Step 5 선택 미수행).

**예상 ROI**:
- LB +0.01~0.02 (보수적 추정, plan-007 MLP coeff 의 0.6482 → extended pool 적용 시
  0.66~0.67 OOF). 후보 1 의 selector arch swap 효과와 *추가* — 합산 시 LB 0.74~0.76.

**작업 범위**:
- 후보 1 의 새 selector arch 산출 + plan-007 의 mlp_coeff.py 의 extended pool 대응
  re-impl.
- Step 5 test_internal grid search (corrector λ / cap / temp) — plan-008 §9 spec 그대로.

**선행 조건**:
- 후보 1 (selector arch) 완료 후.
- plan-007 의 산출 (`analysis/plan-007/mlp_coeff.py`) 재사용 가능 verification.

## 후보 4 (대안 — 시나리오 D 회귀 옵션) — plan-006 framework 회귀 + regime 재도입 ablation

**근거 metric**:
- plan-008 의 Variant A baseline 결정 (regime LB marginal +0.001 = noise) 이 새 후보
  풀 위에서도 유효한지 미검증.
- plan-008 c7 의 family_effect +0.0037 = 후보 풀 확장 효과 marginal → regime infra
  부재 가 다른 ablation 의 *cofounder*.

**예상 ROI**:
- LB +0.005~0.015 (regime bias 부분 회수). plan-008 의 main hypothesis 부정 시 회귀
  안전판.

**작업 범위**:
- plan-006 의 단일 공식 framework 복원 + regime_prior_strength=0.45 (plan-004 default)
  로 selector 재학습.
- 비교 ablation: Variant A (plan-008 결정) vs full (regime 재도입) on extended pool.

**선행 조건**:
- plan-008.1 의 LB 측정 ≤ 0.66 (Variant A baseline 0.6796 미만) — 시나리오 D 진입.

---

**우선순위 권고 (plan-009 main scope)**: 후보 1 (selector arch) + 후보 2 (corrector
hook) **묶음 실행**. 둘 다 plan-008 의 직접 carry-over + 가설 정합. 후보 3/4 는
plan-009 scope 외 (plan-010 후속 또는 시나리오 분기).

---

## Ranking 개선 — 6 카테고리 ROI 표 (§10.2.1, v2.6 필수 박제)

**근거**: plan-008 의 ranking 능력 동결 (caveat #4, #13). top-1 ranking 12.6% 의
*직접 원인* = 현 loss 가 *binary hit/miss* 만 학습 (cross-entropy soft target =
"1cm 안 후보들 균등") — *진짜 best 픽* 학습 X. 모든 시나리오 (A~D) 에서 plan-009
main task 후보 가능.

### 카테고리 1: Loss 변경 (★ 최고 ROI, arch 보존)

| 후보 | mechanism | 예상 LB gain | 비용 |
|---|---|---|---|
| **1.3 NDCG@1 differentiable** | `loss = 1 − softmax(score)[oracle_best]` — top-1 ranking 의 differentiable proxy | **+0.03** | ★ (loss 함수만 교체) |
| 1.1 Pairwise margin | sorted pair 에 hinge — score 순서가 err 순서 일치 강제 | +0.02 | ★ |
| 1.2 Listwise (ListMLE) | `−log P(top-1 = oracle_best)` — top-1 log-likelihood 직접 | +0.02~0.03 | ★ (gradient 불안정 risk) |
| 1.4 Focal ranking | hard sample 가중 (1−score_best)^γ — 88% miss case 집중 | +0.01~0.02 | ★ (다른 loss 와 조합) |

### 카테고리 2: Selector arch 교체 (★★★ big change)

| 후보 | mechanism | 예상 LB gain | risk |
|---|---|---|---|
| 2.1 Set Transformer | candidate set 의 self-attention (`cand_i ↔ cand_j` 직접 비교) — 현 framework 의 "trajectory hidden 만 attend" 한계 해소 | +0.04~0.05 | mid (overfit) |
| 2.2 Twin pairwise | (i, j) binary classifier + round-robin — ranking 직접 학습 | +0.03~0.05 | mid (inference 666×) |
| 2.3 Transformer (full) | trajectory + 후보 통합 token sequence + bi-directional | +0.05 | **high (overfit, data 10K)** |
| 2.4 TCN | 1D causal conv — GRU 의 sequential bottleneck 제거 | +0.01~0.02 | low (marginal, seq 짧음) |

### 카테고리 3: Multi-stage selector (★★ 분해)

| 후보 | mechanism | 예상 LB gain | 비용 |
|---|---|---|---|
| 3.1 Coarse-to-fine 2-stage | Stage 1 cheap filter 37→top-5, Stage 2 expensive rerank — search space 5 로 축소 → ranking 정확도 ↑ | +0.03~0.05 | ★★★ (2 model train) |
| **3.2 Hard top-K filter** | test-time only: softmax 전 top-3 외 후보 −inf — centroid drift 직접 fix | **+0.02** | ★ (학습 X, 1 줄 추가) |
| 3.3 Per-trajectory family routing | family_pred → 해당 family 후보만 ranking — 시나리오 의존 | +0.02~0.03 | ★★ (hard routing gradient) |

### 카테고리 4: Score combination 재설계 (★★ 작은 변경)

| 후보 | mechanism | 예상 LB gain | risk |
|---|---|---|---|
| 4.1 Confidence-weighted | `final = c × gru + (1−c) × bias`, c=σ(MLP(hidden)) — GRU uncertainty 반영 | +0.01~0.02 | low |
| 4.2 Outlier penalty | `−λ × ∥cand − centroid∥` — soft 평균 안정성 | +0.01 | low |
| 4.3 Bias × GRU multiplicative | `bias × σ(gru)` 곱셈 — train hit_rate 낮은 후보 강제 down-weight | +0.01~0.02 | mid (physics_bias 틀린 sample 회수 불가) |

### 카테고리 5: Non-parametric class (caveat #20 박제)

| 후보 | mechanism | 예상 LB gain | 비용 |
|---|---|---|---|
| 5.1 KNN nearest-neighbor | K=5 유사 trajectory 의 t+1 displacement 평균 | +0.03~0.05 | ★★ |
| 5.2 GP residual | train residual GP fit → test posterior | +0.02~0.04 | ★★★ |
| 5.3 Per-sample MLP regression | direct (x,y,z) 회귀 (candidate 우회) | +0.02~0.05 | ★★★★ (overfit risk) |
| 5.4 Stacked residual | XGBoost on per-candidate errors | +0.02~0.04 | ★★ |

### 카테고리 6: Other (carry-over caveat 박제)

| 후보 | 출처 caveat | 비고 |
|---|---|---|
| 6.1 Regime-agnostic per-sample formula 회귀 | #12 | Family 4 drop 의 long-tail 회수 |
| 6.2 Greedy brute-force (template_pool 2^15 조합 전체 search) | #14 | local optimum 회피 |
| 6.3 Field 분리 (binormal_scale_fs / world_z_keep_trig) | #15 | semantic clean-up |

---

## plan-009 권장 sequence (§10.2.2, v2.6)

- **Phase 1 (cheap, no arch)**: 1.3 NDCG@1 + 1.1 Pairwise margin + 3.2 Hard top-K filter → 누적 **+0.05~0.07** (LB 0.73~0.75 가능)
- **Phase 2 (mid)**: 3.1 Coarse-to-fine 2-stage → +0.03~0.05 추가
- **Phase 3 (big, risky)**: 2.1 Set Transformer (Phase 1 결과 미흡 시) → +0.05
- **Phase 4 (carry-over 시나리오 D)**: 5.x non-parametric (KNN / per-sample MLP) → +0.03~0.05

→ Phase 1 만으로도 plan-009 의 minimal viable. plan-008.1 carry-over LB 측정 후 결정.

---

## 시나리오 분기 (§10.2 4-시나리오 본 plan 결과 매핑)

본 plan 결과: Step 3 OOF=0.6503 → LB 예상 ~0.672 → **시나리오 C 경계 (0.70~0.75 미만)**.

**시나리오 A** (Step 4 LB ≥ 0.80): 미달 — 본 plan G3 deferred, LB carry-over.
**시나리오 B** (0.75~0.80): 미달.
**시나리오 C** (0.70~0.75): 추정 도달 (LB ~0.67 carry-over 시).
**시나리오 D** (< 0.70): plan-006 framework 회귀 검토 (후보 4).

각 시나리오의 4 항목 (근거 metric / 예상 ROI / 작업 범위 / 선행 조건) 는 위
후보 1~4 표에 직접 박제됨. plan-008 의 결과 시나리오 C 기준 plan-009 priority
= 후보 1 (selector arch + ranking loss = Phase 1+2) + 후보 2 (corrector hook).
