# plan-013 → plan-014 후보 (≥ 3)

plan-013 의 G3 OOF = 0.6381 (G1 동일, fallback path). LB 회수 = plan-013.1 carry-over.

**LB 결과로 plan-014 분기 결정** (§9.2 조건부 framework):

## (A) LB ≥ 0.68 path (가능성 낮음 — G3 OOF 0.6381 의 LB transfer +0.02 추정 시 0.66 부근)

`if LB ≥ 0.68`:

| # | candidate | rationale |
|---|---|---|
| A1 | **Step 4 27ext 의 separate MLP per candidate** (overfit 통제) | plan-013 의 27ext deferred 회수 — single shared MLP 가 overfit risk (216 params) 였으면 per-candidate MLP (separate 27 instance) 로 capacity 분산 |
| A2 | **5-fold mean + TTA rotation 4-way ensemble** | plan-013 의 5-fold mean 위에 TTA (trajectory yaw rotation 4-way) 추가 — LB *test set 의 yaw distribution* 다양성 활용 |
| A3 | **25 cand 의 추가 expansion (35+ candidates)** | plan-008 의 12 base_kept + 4 templates 위에 *4 templates more* greedy add — oracle ceiling 추가 +0.02 가능 (plan-008 §6.3 박제) |

## (B) 0.65 ≤ LB < 0.68 path (가장 가능성 높음 — plan-013 simplified pipeline 의 deflate baseline 위)

`elif 0.65 ≤ LB < 0.68`:

| # | candidate | rationale |
|---|---|---|
| B1 | **plan-004 BOUNDARY_MAIN 의 train_net 시그너처 확장 (`corrector_cls` arg)** | plan-013 의 simplified pipeline penalty 회수 — boundary.py 의 full pipeline (regime/env/pretrain/finetune) 를 그대로 reuse 하면서 corrector class 만 InICCorrectorWrapper 로 swap. 추정 회복량 +0.01~+0.02 (plan-004 boundary fold-0 0.6717 의 5-fold 확장) |
| B2 | **plan-007 basis_terms framework integration → Step 4 lever 실측** | plan-013 c5/c6 deferred 회수 — integrated_v3 의 dispatcher 에 plan-007 의 train_one_fold + compute_pred 통합, F0_only + 27ext 측정. 추정 회복량 +0.003~+0.010 |
| B3 | **plan-008 G1 candidate descriptor 별도 박제 → 25 cand swap 실측** | plan-013 c7 deferred 회수 — plan-008 selector training script 의 candidate definition 부분 복원해 cand_set.json 박제. 추정 회복량 +0.003~+0.010 (단, plan-013 §7.3 retain risk note: 25-way head capacity 가 27-way 보다 작음에 따른 retain 실패 가능성) |

## (C) LB < 0.65 path (paradigm 재발명 trigger — plan-012 lesson 반복 위험)

`else (LB < 0.65)`:

| # | candidate | rationale |
|---|---|---|
| C1 | **paradigm 완전 폐기 (KNN/GP/Diffusion)** — plan-012 carry-over 후보 A | plan-001~013 의 *residual+candidate* paradigm 자체 폐기. KNN over trajectory embeddings, Gaussian Process regression on (par, perp), Diffusion model conditional on trajectory. trade-off: zero-shot prior 부재, 데이터 효율 의문 |
| C2 | **Step 4 의 27ext 가 NEGATIVE 이면 plan-007 Step 3 (basis ablation) reuse + 결합** | Step 4 의 per-sample MLP 가 overfit 으로 -ΔOOF 보이면, plan-007 Step 3 의 global basis ablation (CMA-ES) 결과 직접 사용 + 결합 |
| C3 | **plan-008 candidate pool 50+ 으로 확장** | candidate set 자체를 50~80 으로 확장 후 selector head dim 확장 — oracle ceiling 자체를 끌어올리되 selector capacity 증가가 retain 비용 동반 |

## ★ plan-014 default 추천 (no LB 정보 기준)

**B1** (plan-004 boundary.py 의 `corrector_cls` arg 확장) 이 default.
- 이유: plan-013 의 가장 큰 deflate factor 는 *simplified pipeline penalty* (plan-004 의 full framework 미사용). 회수 자체가 가장 쉽고 (boundary.py train_net 시그너처 확장 1줄), 회복량 추정 +0.01~+0.02 로 가장 큼.
- B1 통과 후에야 B2 (Step 4) + B3 (25 cand) 의 실측이 *plan-004 full framework 위에서* 가능 — 본 plan 의 원래 3 lever stacking 의도가 plan-014 에서 realize.
