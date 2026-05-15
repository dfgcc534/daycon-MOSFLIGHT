---
plan_id: 019
version: 1.1 (results synthesis)
date: 2026-05-15 (Asia/Seoul)
status: G_final_complete (G0 PASS / G1 WARN ebip_no_gain / G2 WARN icnn_no_gain / G3 WARN meta_adaptation_no_gain / G4 SKIP per user decision)
based_on:
  - 007 (Step 4 MLP OOF 0.6482, LB 0.6598 — A0 baseline reproduce)
  - 005 (oracle 0.7188 — paradigm ceiling reference, 본 plan 미달)
  - 004 (LB 0.6822 — single-model 돌파 target, 본 plan 미달)
  - brainstorm-iter-1~5 (6 candidates: meta-EBIP+ICNN ★ / EBIP / meta-EBIP / Learnable Basis+MoLE / DEQ / ICNN)
  - 018 (단일 stack architecture lever falsified — 본 plan 의 paradigm-shift 동기)
followed_by:
  - 020 후보 박제 (energy-based paradigm 의 *실측 ceiling* ≈ A0 + 0.007 → corrector 결합 또는 multi-stack 필수)
scope: brainstorm ranking #1 paradigm (meta-EBIP + ICNN hybrid) 의 progressive 3-stage ablation.
       3 stage (S1 EBIP base / S2 + ICNN convex / S3 + meta adaptation) × 5-fold OOF.
       G_final = LB > 0.70.
exp_ids:
  - F013_a0                  # A0 baseline reproduce (=plan-007 step 4 재현)
  - F014_ebip-base           # S1 EBIP base
  - F015_ebip-icnn           # S2 + ICNN convex
  - F016_meta-ebip-icnn      # S3 + meta (FOMAML)
lb_score: null    # G4 SKIP (사용자 결정, quota 보존)
g0_passed: true   # A0 OOF=0.6482 ∈ [0.6479, 0.6485]
g1_passed: false  # S1 OOF=0.6552, threshold 0.66
g2_passed: false  # S2 OOF=0.6520, threshold 0.68
g3_passed: false  # S3 OOF=0.6538, threshold 0.70 ⭐
best_stage: S1
best_stage_oof: 0.6552
delta_vs_a0: 0.0070
dacon_submits_used: 0   # 본 plan-019 단독
exception_policy: plan-007 §2.2 의 "End-to-end 학습 통합 out-of-scope" 의 **예외 plan**.
                  본 plan paradigm = energy-based implicit prediction — single-stack 의 일종.
                  multi-stack (≥ 3 stage) 은 여전히 out-of-scope.
---

# plan-019 v1.1 — Results (G1/G2/G3 모두 WARN: energy-based paradigm marginal-only)

## §1. G0~G4 결과 narrative

### G0 — A0 baseline reproduce (F013_a0, cec06e7)

- (a) Data shape (10000, 11, 3) ✓, axis_std ∈ (0.4, 1.5) ✓
- (b) const-velocity baseline MAE/axis_max = 0.0150 m ✓, hit_rate@1cm = 0.2463
- (c) **A0 5-fold OOF = 0.6482** ∈ [0.6479, 0.6485] target → **G0 PASS**
- per-fold: 0.6619 / 0.6453 / 0.6476 / 0.6510 / 0.6350 (concat 0.6482)
- plan-007 step 4 (OOF=0.6482) 와 *exact decimal-4 reproduce* — basis_terms 식 / soft_hit_loss / 5-fold split spec 모두 재현 성공.

### G1 — S1 EBIP base 5-fold OOF (F014_ebip-base, 4dd7c2a)

- model: CoeffMLP (77→32→8) + energy_mlp (3+77→32→32→1) + unrolled GD T=5, λ_init=1.0 (log_lambda=0)
- cond_dim = 13 handcrafted ⊕ 64 cnn_encoded = **77d** (§0.5 Critical amendment 적용)
- per-fold: 0.6713 / 0.6532 / 0.6517 / 0.6559 / 0.6436 → **OOF = 0.6552**
- target ≥ 0.66 → **G1 WARN** `ebip_no_gain` (under threshold by -0.0048)
- A0 대비 +0.0070 — explicit→implicit reformulation 의 marginal gain measured.

### G2 — S2 EBIP+ICNN 5-fold OOF (F015_ebip-icnn, aa0d1ed)

- 초기 buggy run: fold 0/2/3 catastrophic divergence (val_hit=0.0). 4 stability fix 적용 (decision-note: spec-amendment):
  1. SiLU → softplus activation (convex non-decreasing 보장)
  2. W_z_raw init randn*0.01 → randn*0.01 - 5.0 (softplus≈0.007, near-zero residual)
  3. inner_lr 0.1 → 0.02, log_lambda init 0 → -2, clamp [-3, 0]
  4. NaN guard (pred isfinite check + anchor fallback)
- per-fold: 0.6728 / 0.6463 / 0.6497 / 0.6495 / 0.6416 → **OOF = 0.6520**
- target ≥ 0.68 → **G2 WARN** `icnn_no_gain` (under threshold by -0.0280)
- S1 대비 -0.0032 — convex 제약이 capacity 감소시킴. icnn_min_softplus 5 fold 모두 ~0.0075 (init value preserved) → W_z 학습 거의 미발현.

### G3 — S3 Meta-EBIP+ICNN 5-fold OOF (F016_meta-ebip-icnn, 2bdf9e6)

- model: MetaEBIPICNN with FOMAML inner loop (c_inner_lr=0.01, c_inner_steps=1, basis_terms_prev horizon=1)
- per-fold: 0.6703 / 0.6507 / 0.6497 / 0.6545 / 0.6436 → **OOF = 0.6538**
- target ≥ 0.70 ⭐ → **G3 WARN** `meta_adaptation_no_gain` (under threshold by -0.0462)
- S2 대비 +0.0018, S1 대비 -0.0014 — FOMAML 의 추가 gain marginal-positive.

### G4 — best variant LB submit (SKIPPED per user decision)

- best variant = **S1 EBIP base** (OOF=0.6552), other stages (S2=0.6520, S3=0.6538) 보다 +0.0014~+0.0032 higher.
- plan-007 step 4 OOF-LB gap (+0.0116) carry 시 예상 S1 LB ≈ 0.6668 — G4 target 0.70 대비 -0.033 미달.
- DACON quota 보존 (오늘 quota plan-018 G2 SKIP 으로 미사용, 남은 5/5 carry to plan-020).
- 사용자 confirm 결과: "skip — G2 SKIP per plan-018 패턴".

## §2. 합격 기준 verdict — 4 가설 모두 falsified or marginal

본 plan §1.2 가설 (energy-based paradigm 의 ambition):

| 가설 | 검증 방법 | 측정 결과 | Verdict |
|---|---|---|---|
| **H1** explicit→implicit reformulation gain | S1 OOF ≥ 0.66 (= A0 + 0.012) | +0.0070 (= A0 + 0.007) | **partial** (+58% of target gain) |
| **H2** ICNN convex 가 stability 보강 + 천장 소량 하향 | S2 OOF ≥ 0.68 | -0.0032 vs S1, OOF 0.6520 | **directionally confirmed but ceiling miss** (가설의 천장 하향 측면은 부합, threshold 미달) |
| **H3** FOMAML per-sample adaptation 추가 gain | S3 OOF ≥ 0.70 ⭐ | +0.0018 vs S2 → 0.6538 | **marginal positive, target far miss** (+12% of target gain) |
| **H4** single-model LB > 0.70 | dacon-submit | LB 미측정 (G4 skip) | **inconclusive but OOF 추정 likely fail** (0.6668 estimated) |

**Summary**: energy-based implicit prediction paradigm 자체는 A0 위 +0.007 gain 으로 *실측 ceiling* 박제 — 그러나 ICNN convex / meta adaptation 의 추가 component 가 marginal/없음. plan-007 §9.2 "단일 공식 framework 한계 ≈ 0.6491" 및 plan-018 "single-stack architecture ceiling ≈ A0 + 0.0003" 와 정합 — **single-stack 의 implicit 변형도 동일 ceiling 근처**.

## §3. Premise verdict — energy-based paradigm 의 단일-stack 한계 falsified

**plan-019 premise (§1.2)**:
> plan-007 step 4 의 explicit form `pred = p0 + Σ c·B` 의 *implicit reformulation* (= argmin_p energy) 으로 universal correction 가능. ICNN convex 제약 + FOMAML inner adaptation 으로 ceiling 을 plan-005 oracle 0.7188 근처 (0.70+) 까지 push.

Verdict: **falsified**. measured 결과:
- implicit reformulation gain (+0.0070) = single-stack 의 *실측 ceiling* 도달 (plan-018 의 A3 +0.0003 보다는 23× higher, 그러나 target 0.66 미달).
- ICNN convex 제약 추가 = capacity 감소 (-0.003).
- FOMAML adaptation 추가 = marginal positive (+0.002 vs S2, 그러나 S1 보다 -0.001).
- 세 component 의 *progressive 가산성* 가설이 falsified — single-stack 의 ceiling 자체가 paradigm-invariant.

**Combined with plan-018 결론**:
- plan-018: "encoder 강화 / head capacity 만으로 single-stack ceiling break 불가" (4 arch 평균 A0+0.0003).
- plan-019: "implicit reformulation 으로 single-stack ceiling +0.007 push 가능, 그러나 target 0.66 미달" (3 stage 평균 A0+0.005).
- 종합: **single-stack paradigm 자체의 실측 ceiling ≈ A0 + 0.005~0.010**. plan-007 §9.2 의 *0.6491 한계* 추측 위 약간 push, 그러나 plan-004 LB 0.6822 와 oracle 0.7188 도달 불가능 — *multi-stack / corrector 결합 필수*.

## §4. plan-020 후보 (paradigm-shift 필수)

본 plan G1/G2/G3 모두 WARN, single-stack 의 *실측 ceiling* 박제 → **brainstorm 의 나머지 5 candidates 도 single-stack 한 부수, 모두 fail 예상**. plan-020 = **paradigm-shift 필요**.

후보 ≥ 2 박제 (analysis/plan-019/next_plan_candidates.md 참조):

### §4.1 후보 A: corrector 결합 (plan-005/016 paradigm + S1 best)

- **idea**: plan-019 의 S1 EBIP base (OOF=0.6552) 의 prediction 을 plan-005/016 의 corrector freeze 와 *output ensemble* (좌표 mean) 으로 결합. plan-017 G1 ensemble (LB 0.6640) 패턴 carry.
- **mechanism**: framework-disjoint 결합 — S1 (energy-based) + plan-016 G1 (BiGRU corrector). 둘이 *서로 다른 error mode* 보강 가능.
- **cost**: low (좌표 mean ensemble + freeze inference).
- **risk**: 둘 다 plan-007 fixed basis 위 동작 → framework-similar 일 수 있음.

### §4.2 후보 B: multi-stack (plan-004 paradigm 변형 — selector + boundary corrector)

- **idea**: plan-018 §4.2 carry. plan-004 LB 0.6822 의 *full ensemble* 아닌 *single-model* 변형 — selector (anchor classify) + boundary corrector (high-error region focus). plan-019 S1 의 EBIP energy module 을 corrector 로 재사용.
- **mechanism**: high-error region (~20% sample, plan-005 oracle 보다 멀리 있는 ~30% 의 0.04 m 이상) 에 model capacity 집중. plan-005 의 *27 후보 풀* 의 oracle 0.7188 를 *학습된 selector* 로 회수.
- **cost**: high (selector + corrector 분리 + joint training).
- **risk**: plan-004 LB 0.6822 가 full ensemble 결과, single-model 회수율 미확정.

### §4.3 후보 C: Learnable Basis + Sparse MoLE (brainstorm #4)

- **idea**: brainstorm #4 carry — 8 fixed basis 를 *learnable embedding* (Koopman lift / Fourier feature) 으로 확장. plan-018 의 A3 MoLE (head capacity) + 본 후보 (basis capacity) 결합.
- **mechanism**: plan-019 가 *single-stack ceiling* falsify 했으므로, single-stack 의 *basis 자체 확장* 으로 ceiling 추가 push 시도. plan-018 §4.4 carry.
- **cost**: medium (basis learning + overfit risk on 10K).
- **risk**: plan-018 / plan-019 의 combined 결과로 *single-stack ceiling 자체가 0.65~0.66* — basis 확장으로도 0.67 돌파 의문.

### §4.4 후보 D: DEQ (brainstorm #2, infinite-depth implicit layer)

- **idea**: brainstorm #2 carry — 본 plan 의 unrolled GD T=5 의 *infinite-depth limit* (Deep Equilibrium Model, Bai et al. 2019). fixed-point iteration p* = f(p*, traj) 까지 수렴.
- **mechanism**: 본 plan 의 S1/S2 가 T=3~5 step 으로 underfit (energy minimum 미도달) 가능성 — DEQ 로 수렴까지 가서 *진짜 implicit prediction* 달성.
- **cost**: high (DEQ 안정성, Jacobian implicit diff).
- **risk**: 본 plan 의 S2 ICNN unrolled GD 의 *실측 효과 미미* 관찰 → DEQ 도 동일 ceiling 예상. paradigm-shift 의도와 모순.

**권장**: **후보 A (corrector 결합)** — cheapest, plan-005 corrector 의 *measured complementarity* 와 plan-019 S1 의 implicit anchor 결합. 단일 plan-020 으로 최저 cost 진행.

## §5. measured 값 박제 (외부 reference)

| measure | value | source |
|---|---|---|
| F0 raw hit@1cm (plan-006 frozen) | 0.6320 | plan-014 G0 |
| const-velocity baseline hit@1cm | 0.2463 | plan-019 G0 (b) |
| plan-007 step 4 MLP OOF (A0 baseline) | **0.6482** | plan-007 §7 / plan-019 G0 reproduce |
| plan-007 step 4 LB | 0.6598 | plan-007 §8 |
| plan-018 G1 best A3 MoLE OOF | 0.6485 | plan-018 §1 (+0.0003 vs A0) |
| **plan-019 G1 S1 EBIP base OOF** | **0.6552** | 본 plan (+0.0070 vs A0) |
| **plan-019 G2 S2 EBIP+ICNN OOF** | **0.6520** | 본 plan (+0.0038 vs A0) |
| **plan-019 G3 S3 Meta-EBIP+ICNN OOF** | **0.6538** | 본 plan (+0.0056 vs A0) |
| plan-014/015 best LB | 0.6628 | plan-014 G5 |
| plan-016 G1 LB | 0.6638 | plan-016 G1 |
| plan-017 G1 ensemble LB | 0.6640 | plan-017 G1 |
| plan-004 ensemble LB | 0.6822 | reference |
| plan-005 oracle hit@1cm | 0.7188 | reference (single-model 도달 ceiling) |

→ plan-019 의 measured *single-stack implicit ceiling* ≈ A0 + 0.0070 = **0.6552** (3 stage 중 best).

## §6. LB carry-over

- plan-019 G4 LB skip (사용자 결정).
- DACON quota: **0/5 본 plan 사용**.
- best variant S1 의 submission.csv 미산출 (G4 SKIP). plan-020 후보 A (corrector 결합) 시 ensemble member 로 carry 가능 — `runs/baseline/F014_ebip-base/checkpoint_fold{0..4}.pt` 박제 유지.

## §7. 종료

- G_final 합격 (3 파일 sync):
  - results.md 신규 (본 파일) ✓
  - plan-019 frontmatter sync (status=G_final_complete, g{0..3}_passed, best_stage=S1, best_stage_oof=0.6552, delta_vs_a0=0.0070) — 본 commit
  - analysis/plan-019/results.md 신규 — 별도 작성
- plan-020 후보 ≥ 2 박제: ✓ (총 4 — corrector 결합 / multi-stack / learnable basis / DEQ)
- G4 LB measured: SKIP (사용자 결정)

decision-note: brainstorm 6 candidates 의 *single-stack 한정* 결론 — paradigm-shift (corrector 결합 또는 multi-stack) 가 ceiling break 의 유일 path. plan-019 의 *실측 ceiling* ≈ A0 + 0.007 박제 후 plan-020 진행.
