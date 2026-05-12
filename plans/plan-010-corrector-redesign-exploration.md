---
plan_id: 010
version: 1
date: 2026-05-13 (Asia/Seoul)
status: draft
based_on:
  - 004
  - 005
  - 006
  - 007
  - 008
  - 009
  - notes/PB_0.6822 코드공유.ipynb
followed_by:
  - 010.1 (LB carry-over; user manual dacon-submit)
  - 011 (TBD; candidates @ analysis/plan-010/next_plan_candidates.md)
scope: 단일공식 `frenet_par120_perp_neg020` (plan-006 picked, LB 0.6692 anchor) 위에 plan-004 corrector 의 7 결함 fix → 4 후보 폭넓은 탐색. (1) Z1+G2 = Minimum Viable Redesign + frozen GRU encoder reuse (★ cheap), (2) Z1+G1 = Z1 + CNN encoder learnable (encoder quality 검증), (3) Z3+G2 = Z1 + iterative refinement 3-step + frozen GRU (cap 한계 우회), (4) Z6 = end-to-end GRU+corrector 통합 (조건부, framework 재정의). plan-006 LB 0.6692 ceiling 회복 + 0.70+ 도달 + corrector framework 의 *본질 한계 vs 설계 결함* 분리 입증. LB 제출 0 회 (할당량 소진 인계, plan-009.1 까지 carry-over 묶음과 동일 정책) — plan-010.1 carry-over.
exp_ids:
  - H006_Z1G2-min-redesign        # ★ G1 main cheap — Z1 (결함 6 fix) + frozen plan-004 GRU encoder reuse
  - H007_Z1G1-cnn-encoder         # G2 — Z1 + CNN encoder learnable
  - H008_Z3G2-iterative           # ★ G3 main — Z1 + iterative refinement 3-step + frozen GRU
  - H009_Z6-end-to-end            # G4 조건부 — GRU + corrector e2e 통합
lb_score: null
---

# plan-010 v1 — Single-Formula Anchor + Corrector Redesign Exploration (Z1+G2 / Z1+G1 / Z3+G2 / Z6)

## §0. 한 줄 목적

> **plan-006 의 단일공식 `frenet_par120_perp_neg020` (LB 0.6692 anchor) 위에 plan-004 corrector 의 7 결함 (target cap-truncation, MSE-hit misalign, far/easy weight 비대칭, env head capacity 낭비, apply_scale hack, hard-coded band) 을 *체계적으로 fix* 한 4 후보 (Z1+G2 cheap → Z1+G1 encoder 학습 → Z3+G2 iterative cap 우회 → Z6 e2e 통합) 의 폭넓은 탐색.**
>
> **narrative**: plan-006/007 의 단일공식 + plan-004 corrector 결과 (LB 0.6692, OOF 0.6491) 가 *결함 corrector* 의 영향 분리 안 된 측정 → corrector 재설계로 *진정한 ceiling* 측정. plan-009 의 "corrector framework 본질 한계" 결론을 **②④ (Frenet frame, oracle ceiling) 본질** vs **①③⑤⑥⑦ 설계 결함** 으로 분리. 본질 한계는 plan-011 의 paradigm 교체 (KNN/GP/Diffusion) 진입 조건. 설계 결함은 본 plan 의 fix target.
>
> **Baseline 확정**: plan-006 LB 0.6692 (단일공식 frenet_par120_perp_neg020 + plan-004 corrector). OOF anchor = plan-006 argmax+corrector 의 0.6491.
>
> **Target LB (carry-over 회수 후 추정)**: **0.70~0.72** (Z1+G2 0.69~0.71, Z3+G2 0.70~0.72, Z6 0.70~0.74 high variance). plan-008 baseline LB 0.6812 ceiling 위 회수 시점에 framework 재정의 (plan-011 paradigm 교체) decision.
>
> **LB 제출 정책**: 본 plan 내 LB 제출 **0 회** (할당량 소진 상태 인계, plan-009.1 carry-over 패턴 답습). 모든 H006~H009 submission.csv 는 *생성·박제만*, LB 회수는 plan-010.1 carry-over.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0** (preflight + plan-006 baseline reproduce + plan-005 corrector_decomp 재측정): plan-004 corrector 7 결함 코드 verify (boundary.py L108~117 read) + plan-006 단일공식 `frenet_par120_perp_neg020` argmax baseline reproduce (`oof_hit_argmax ≈ 0.6320` ± 0.005) + plan-005 corrector_decomp band table 재측정 (`[0.5,1cm) hit_after`, `[1,1.5cm) hit_after`) + `analysis/plan-010/preflight.json` 생성. 위반 시 `preflight_artifact_missing` severe.
- **G1** (Phase 1 — ★ H006_Z1G2 Minimum Viable Redesign + frozen GRU): Z1 components (B1 uncapped target / A2 Huber loss / C1 easy=0 / C2 far=0.5 / D1 env head drop / E1 apply_scale=1.0) + G2 (plan-004 GRU encoder frozen forward → hidden 32-dim concat). 단일공식 frenet_par120_perp_neg020 만 사용 (K=1). (a) `oof_soft_hit ≥ 0.66` minimum, stretch **0.68**. (b) `[1, 1.5cm) hit_after ≥ 0.15` (plan-005 9.77% → 1.5x 회복). (c) `[0.5, 1cm) hit_after ≥ 0.95` (plan-005 92.17% regression 방어). (d) `corrector_oracle_gain ≥ 0` (plan-005 −0.77% → 양수). **LB 미제출** — submission.csv 생성만. 위반 시 `min_redesign_failure` severe (단, (a) 만 fail 시 warn-only — encoder 추가 검증 G2 진입).
- **G2** (Phase 2 — H007_Z1G1 CNN encoder learnable): Z1 + CNN encoder (3-layer 1D conv, hidden=64, 학습 가능). G1 의 frozen GRU 자리에 *corrector loss 로 직접 학습한 encoder*. (a) `oof_soft_hit ≥ G1 OOF + 0.005` marginal, stretch **G1 + 0.015**. (b) `encoder_param_count ≤ 50K` (over-fit risk control). **LB 미제출**. 위반 시 `cnn_encoder_marginal` warn-only.
- **G3** (Phase 3 — ★ H008_Z3G2 iterative refinement + frozen GRU): Z1 + iterative refinement (n_steps=3, per-step cap=3mm, parameter 공유 + step_idx embedding) + G2 frozen GRU. cap 한계 ① 의 *함의 우회*. (a) `oof_soft_hit ≥ G1 OOF + 0.01` minimum, stretch **G1 + 0.025**. (b) **`[1, 1.5cm) hit_after ≥ 0.30`** ★ (plan-005 9.77% → 3x 회복, plan-009 G2 미달 target 회수). (c) iterative 발산 방지: 모든 fold 의 OOF gap (val vs train) ≤ 0.05. **LB 미제출**. 위반 시 `iterative_divergence` severe (b 또는 c fail 시) / `iterative_marginal` warn-only (a 만 fail 시).
- **G4** (Phase 4 — H009_Z6 end-to-end, **조건부**): G1~G3 누적 best LB 추정 < **0.72** 일 때만 진입 (e2e cost 대비 ROI 분기). GRU + corrector 통합 학습 (trainable GRU + Z1 corrector + Huber loss). (a) `oof_soft_hit ≥ G3 OOF + 0.005`. (b) over-fit gap (train OOF − val OOF) ≤ 0.04 (작은 dataset over-fit 방어). **LB 미제출**. 위반 시 `e2e_overfit` warn-only.
- **G_final**: synthesis + plan-011 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 (`lb_score: TBD` — carry-over) + best Phase submission 박제 (path: `runs/baseline/<best_H_exp_id>/submission_*.csv`) + plan-010.1 carry-over instruction 박제.

### G-gates

- G0: preflight + plan-006 단일공식 baseline reproduce + plan-005 corrector_decomp 재측정 — `preflight.json` 생성 [TODO]
- G1: Phase 1 ★ H006_Z1G2 (Min Viable Redesign + frozen GRU) — OOF ≥ 0.66 + [1,1.5cm) hit ≥ 0.15 + [0.5,1cm) hit ≥ 0.95 + corrector_oracle_gain ≥ 0 [TODO]
- G2: Phase 2 H007_Z1G1 (CNN encoder learnable) — OOF ≥ G1 + 0.005 + encoder ≤ 50K params [TODO]
- G3: Phase 3 ★ H008_Z3G2 (iterative refinement + frozen GRU) — OOF ≥ G1 + 0.01 + [1,1.5cm) hit ≥ 0.30 + iter_gap ≤ 0.05 [TODO]
- G4: Phase 4 H009_Z6 (e2e, 조건부 LB 추정 < 0.72) — OOF ≥ G3 + 0.005 + overfit_gap ≤ 0.04 [TODO]
- G_final: synthesis + plan-011 후보 ≥ 2 + 3 파일 frontmatter sync (`lb_score: TBD` carry-over) + best Phase submission 경로 박제 + plan-010.1 instruction [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-010-corrector-redesign-exploration.md` v1 작성 | [TODO] |
| c2 | code | `analysis/plan-010/preflight.py` — plan-004 boundary.py 7 결함 verify + plan-006 단일공식 reproduce + plan-005 corrector_decomp 재측정. spec @ §4 | [TODO] |
| G0 | gate | `preflight.json` 생성 + 7 결함 verify ✓ + plan-006 reproduce ✓ + corrector_decomp band table 박제 | [TODO] |
| c3 | code | `src/pb_0_6822/corrector_redesign.py` — Z1 components 통합 모듈 (RedesignedCorrectionNet, huber_loss, no env head, apply_scale=1, weight schedule). spec @ §5.1 | [TODO] |
| c4 | code | `analysis/plan-010/h006_train.py` — H006_Z1G2 학습 wrapper (frozen plan-004 GRU forward + 5-fold OOF on 단일공식). spec @ §5.2 | [TODO] |
| c5 | exp | H006_Z1G2: 5-fold 학습 + submission 생성 (LB 미제출). spec @ §5 | [TODO] |
| G1 | gate | OOF ≥ 0.66 + per-band hit + corrector_oracle_gain ≥ 0 | [TODO] |
| c6 | code | `src/pb_0_6822/corrector_redesign.py` CNN encoder block 추가 + `analysis/plan-010/h007_train.py`. spec @ §6 | [TODO] |
| c7 | exp | H007_Z1G1: 5-fold 학습 + submission 생성. spec @ §6 | [TODO] |
| G2 | gate | OOF ≥ G1 + 0.005 + encoder ≤ 50K params | [TODO] |
| c8 | code | `src/pb_0_6822/corrector_redesign.py` IterativeRefinementCorrector + `analysis/plan-010/h008_train.py`. spec @ §7 | [TODO] |
| c9 | exp | H008_Z3G2: 5-fold 학습 + submission 생성. spec @ §7 | [TODO] |
| G3 | gate | OOF ≥ G1 + 0.01 + [1,1.5cm) hit ≥ 0.30 + iter_gap ≤ 0.05 | [TODO] |
| c10 | code | (조건부, G1~G3 LB 추정 < 0.72) `src/pb_0_6822/corrector_redesign.py` E2EGRUCorrector + `analysis/plan-010/h009_train.py`. spec @ §8 | [TODO] |
| c11 | exp | (조건부) H009_Z6: 5-fold e2e 학습 + submission 생성. spec @ §8 | [TODO] |
| G4 | gate | (조건부) OOF ≥ G3 + 0.005 + overfit_gap ≤ 0.04 | [TODO] |
| c12 | synthesis | `analysis/plan-010/results.md` + `next_plan_candidates.md` (≥ 2 후보) + best Phase submission path 박제 + plan-010.1 carry-over instruction + 4 결함 별 attribution table. spec @ §9 | [TODO] |
| G_final | gate | results.md + next plan 후보 ≥ 2 + 3 파일 frontmatter 동시 박제 + plan-010.1 instruction | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `preflight_artifact_missing` — G0 의 `preflight.json` 미생성 또는 plan-006 baseline reproduce 실패 (|measured − 0.6320| > 0.005)
- `min_redesign_failure` — G1 (b)/(c)/(d) 중 하나 fail (per-band hit 또는 corrector_oracle_gain). (a) 단독 fail 은 warn-only.
- `iterative_divergence` — G3 (b) [1,1.5cm) < 0.30 또는 (c) iter_gap > 0.05 fail
- `single_formula_residue` — selector 가 단일공식 외 다른 candidate 사용한 evidence (cand pool size > 1 또는 score variance > 1e-10)
- `frozen_gru_drift` — H006/H008 에서 plan-004 GRU encoder parameter 변경 detected (state_dict diff > 0)
- (v1.1 제거 유지) `lb_quota_exhausted` — LB 제출 0 회 정책으로 trigger 부재

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/corrector_redesign.py` (신규 모듈, 본 plan main code)
  - `analysis/plan-010/**` (preflight, h006~h009 wrapper, results, next_plan_candidates)
- whitelist 제외 (blacklist 추가):
  - `src/pb_0_6822/boundary.py` (touch X — Z1 의 모든 변경은 `corrector_redesign.py` 신규 모듈에서. `boundary.py` 의 `compute_corrector_loss` hook 은 *read-only reference*)
  - `src/pb_0_6822/selector.py` (touch X — frozen GRU 는 `selector.AttnGRUCandidateSelector` 의 forward only)
  - `src/pb_0_6822/candidates_extended.py` (plan-008 산출, 본 plan scope X — 단일공식 만 사용)
- 참조 (read-only):
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 산출, GRU checkpoint + corrector baseline reference)
  - `runs/baseline/F001_variant-e/**` (plan-006 산출, 단일공식 baseline)
  - `analysis/plan-005/corrector_decomp.{md,json}` (★ band table baseline)
  - `notes/PB_0.6822 코드공유.ipynb` (cell 6 boundary corrector 원본)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Huber beta=0.005 (5mm threshold) 채택 (cap=0.006 의 80% line)`
- `decision-note: spec-default — frozen GRU encoder hidden = 32 (plan-004 default), forward only`
- `decision-note: spec-default — iterative n_steps=3, per-step cap=3mm 채택 (누적 cap=9mm > 결함 단일 cap 6mm)`
- `decision-note: conditional-skip — G1~G3 LB 추정 ≥ 0.72 → G4 (Z6 e2e) skip, plan-011 carry-over`
- `decision-note: G0 evidence — plan-005 [1,1.5cm) hit_after 재측정 = 0.0X (vs 9.77% 박제) → drift 검증 후 G1 target band 조정`

---

## §1. 배경 / 이전 plan 인계

### §1.1 plan-004 corrector 7 결함 (사용자 challenge 후속, 2026-05-13)

| # | 결함 | 위치 | 영향 |
|---|---|---|---|
| ① | target = cap-truncated residual | boundary.py L108~110 | hard sample 진짜 방향 정보 *영원히* 못 봄 — [1,1.5cm) 회복률 9.77% 한계 직접 원인 |
| ② | MSE loss vs hit@1cm metric | boundary.py L259 | 학습 목표 ≠ 평가 목표 |
| ③ | far_weight 0.04 | boundary.py L114 | 1.7cm+ sample 학습신호 zero |
| ④ | easy_weight 0.20 | boundary.py L114 | [0.5,1cm) 100→92% regression 원인 (plan-005 corrector_decomp 측정) |
| ⑤ | env head (family CE) | boundary.py L185~190 | 작은 dataset (10K) 에 capacity 낭비 |
| ⑥ | apply_scale 0.75 hack | boundary.py L327 | 학습/추론 mismatch |
| ⑦ | hard-coded band [0.7, 1.7cm] | boundary.py L368~369 | 데이터 분포 무시 |

**plan-009 의 "corrector framework 본질 한계" 결론 재해석**:

| 한계 | 본질 / 결함 분리 |
|---|---|
| cap (6mm) | **결함 ①의 일부** (target 도 cap 으로 잘리는 게 문제. cap 자체보다 cap-truncated target 이 root) |
| Frenet frame (좌표계) | 본질 (좌표계 선택) |
| zero-sum (easy 손상 ↔ hard 회복) | **결함 ④** (easy weight 0.20 부작용) |
| oracle ceiling (12.6% unreachable) | 본질 |
| small dataset over-fit | **결함 ⑤+⑥** (capacity 낭비) 의 *artificial* over-fit |

→ **본질 한계 = 2 개 (Frenet frame, oracle ceiling) 뿐**. 나머지 3 개는 *결함의 결과*. 본 plan = 결함 fix 후 *진정한 본질 한계* 분리 검증.

### §1.2 plan-006/007 의 단일공식 + corrector 결과 재해석

| plan | 단일공식 | corrector | OOF | LB |
|---|---|---|---|---|
| plan-006 | frenet_par120_perp_neg020 (rank 1/27) | plan-004 (결함) | 0.6491 | **0.6692** |
| plan-007 Step 2 | CMA-ES tuned (6 vars) | plan-004 (결함) | 0.6403 | 0.6570 |
| plan-007 Step 3 | basis ablation best | plan-004 (결함) | 0.6403 | 0.6598 |
| plan-007 Step 4 | per-sample MLP coeff | plan-004 (결함) | 0.6482 | (carry-over 미회수) |

**재해석**: 4 측정 모두 *결함 corrector* 와 결합한 결과. 단일공식 framework 의 *진정한 ceiling* 은 미측정.

본 plan 의 가설: corrector 재설계 시 단일공식 + 재설계 corrector LB ≥ plan-006 의 0.6692 + Δ (Δ ≥ 0.01).

### §1.3 plan-009 carry-over (단일공식 path 미포함)

plan-009 는 *27 후보 + selector* 패러다임 위 ranking loss + corrector 강화. 결과:
- G1 ranking loss: OOF 0.6482 (regression −0.0021)
- G2 corrector 강화 (5 sub-exp additive): best b OOF 0.6653 (+0.0150 vs plan-008)
- ★ LB actual: H002 b = **0.6748** (plan-008 baseline 0.6812 ceiling 미달, -0.0064)

**plan-009 의 결론**: corrector framework "본질 한계" 재확인 — 단, *결함 corrector 위에서의 측정*. 본 plan 이 *결함 fix 후* 측정으로 재검증.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| selector | **단일공식 only** — `frenet_par120_perp_neg020` (plan-006 picked, CANDIDATES[17]) |
| selector arch | 사용 X (단일공식 = K=1 candidate, ranking 없음) |
| corrector arch | TinyCorrectionNet 기반 4 변형 (Z1+G2, Z1+G1, Z3+G2, Z6) |
| Loss redesign (Z1 B1+A2+C1+C2+D1+E1) | 6 결함 fix |
| Encoder input | snapshot (plan-004 cf 32-dim) + frozen GRU hidden 32-dim (G1/G3) / CNN encoder 64-dim (G2) / e2e GRU (G4 조건부) |
| Iterative refinement (Z3) | n_steps=3, per-step cap=3mm, parameter 공유 |
| LB 제출 | **0 회** (할당량 소진 인계, plan-010.1 carry-over) |
| 학습 데이터 | train 10K (plan-004 동일) |
| Validation | 5-fold OOF + 1-fold approx (조건부, c4/c5 빠른 iteration) |
| GPU | server cuda:1 (plan-004/005/008/009 동일) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 27 후보 selector + corrector | plan-008/009 의 path. 본 plan = 단일공식 path 분리 검증 |
| Set Transformer | plan-009 carry-over (조건부 SKIP). 본 plan scope X |
| KNN / GP / Diffusion (paradigm 교체) | plan-011 후보. 본 plan 의 4 후보 모두 0.70 미달 시 진입 조건 |
| boundary.py 본문 수정 | whitelist X. Z1 fix 는 신규 모듈 `corrector_redesign.py` 에서 |
| selector.py 본문 수정 | whitelist X. GRU 는 frozen forward only |
| candidates_extended.py 사용 | plan-008 산출. 본 plan = 단일공식 만 |
| LB 제출 | 할당량 소진 (plan-009.1 까지 사용). 본 plan = carry-over |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold OOF: `selector.stable_fold_id(sample_id, folds=5)` (plan-004 동일)
- val fold k → fold k 만 val, 나머지 4 fold train
- 5 fold OOF concat → `overall_oof_hit_soft` (= submission_step3.csv 의 hit@1cm)
- 1-fold approx (c5/c7 빠른 iteration): fold=0 만 측정, binomial std ≤0.005 (N_val ≈ 2020)

### §3.2 합격 기준

§0.5 G-gate sequence 참조.

### §3.3 평가 점수 / median 집계

- main metric: **5-fold concat OOF soft hit @ 1cm** (plan-004/005/008/009 동일)
- soft hit = `base.search_temperature(corrected, scores, true)["metrics"]["hit"]`
- per-band hit_after: `[0,0.5)`, `[0.5,1)`, `[1,1.5)`, `[1.5,2)`, `[2,∞)` (plan-005 corrector_decomp schema)
- corrector_oracle_gain = `corrected_oracle_hit − raw_oracle_hit` (plan-009 c8 schema)

---

## §4. STAGE 0 (G0) — Preflight + plan-006 baseline reproduce + corrector_decomp 재측정

### §4.1 산출물

- `analysis/plan-010/preflight.py` — 3 task 일괄 실행
- `analysis/plan-010/preflight.json` — schema:
```json
{
  "exp_id": "G0_preflight",
  "plan_004_corrector_flaws_verified": {
    "flaw_1_target_cap_truncation": "boundary.py:108-110 verified",
    "flaw_2_mse_loss": "boundary.py:259 verified",
    "flaw_3_far_weight_0.04": "boundary.py:114 verified",
    "flaw_4_easy_weight_0.20": "boundary.py:114 verified",
    "flaw_5_env_head": "boundary.py:185-190 verified",
    "flaw_6_apply_scale_0.75": "boundary.py:327, default 0.75 verified",
    "flaw_7_hardcoded_band": "boundary.py:368-369, low=0.007 high=0.017 verified"
  },
  "plan_006_baseline_reproduce": {
    "single_formula": "frenet_par120_perp_neg020",
    "candidate_idx": 17,
    "oof_argmax_hit_measured": <float>,
    "oof_argmax_hit_expected": 0.6320,
    "drift": <float>,
    "drift_threshold": 0.005,
    "reproduce_ok": <bool>
  },
  "corrector_decomp_remeasure": {
    "n_train": 10000,
    "band_table": {
      "[0,0.5cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[0.5,1cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[1,1.5cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[1.5,2cm)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>},
      "[2cm,inf)":   {"n_in_band": <int>, "hit_before": <float>, "hit_after": <float>, "delta": <float>}
    },
    "plan_005_baseline": {
      "[0.5,1cm)_hit_after": 0.9217,
      "[1,1.5cm)_hit_after": 0.0977
    },
    "drift_ok": <bool>
  }
}
```

### §4.2 실행

```bash
python -m analysis.plan-010.preflight \
  --root data \
  --plan-006-checkpoint runs/baseline/F001_variant-e/... \
  --plan-005-corrector-decomp analysis/plan-005/corrector_decomp.json \
  --out analysis/plan-010/preflight.json
```

### §4.3 G0 합격

- 7 결함 모두 verified (boundary.py 코드 라인 read + content match)
- plan-006 reproduce drift ≤ 0.005
- corrector_decomp drift ≤ 0.01 per band

---

## §5. STAGE 1 (G1) — H006_Z1G2 Minimum Viable Redesign + frozen GRU

### §5.1 corrector_redesign.py 신규 모듈 (Z1 components)

```python
# src/pb_0_6822/corrector_redesign.py

import torch
from torch import nn
import torch.nn.functional as F

class RedesignedCorrectionNet(nn.Module):
    """plan-010 §5.1 — Z1 components 통합.

    Z1 fix:
      - D1: env head 제거 (delta head only)
      - apply_scale = 1.0 (E1, args 에서 강제)
      - encoder_block: None | "frozen_gru" (G2) | "cnn" (G1) | "trainable_gru" (G4)

    Forward:
      cf [B, dim_cf] + (optional) encoder_emb [B, dim_enc] → delta [B, 3]
    """
    def __init__(self, dim_cf: int, hidden: int = 64, dim_encoder: int = 0):
        super().__init__()
        in_dim = dim_cf + dim_encoder
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.04),
        )
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden),
            ResidualMLPBlock(hidden),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        # D1: NO env head
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, cf: torch.Tensor, encoder_emb: torch.Tensor | None = None) -> torch.Tensor:
        if encoder_emb is not None:
            x = torch.cat([cf, encoder_emb], dim=-1)
        else:
            x = cf
        h = self.blocks(self.stem(x))
        return self.delta(h)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.005) -> torch.Tensor:
    """A2: Huber loss, beta=5mm threshold.

    pred, target: (B, 3). Returns (B,) per-sample.
    """
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none').sum(dim=1)


def weight_schedule(err: torch.Tensor, R_HIT: float = 0.01) -> torch.Tensor:
    """C1+C2: easy=0, boundary=1, far=0.5.

    err: (B,) raw err magnitude. Returns (B,) weight.
    """
    easy = err < R_HIT * 0.7      # < 0.7cm: weight 0 (C1, easy=0)
    boundary = (err >= R_HIT * 0.7) & (err < R_HIT * 1.7)  # [0.7, 1.7cm): 1.0
    far = err >= R_HIT * 1.7      # >= 1.7cm: 0.5 (C2)
    return torch.where(easy, 0.0, torch.where(boundary, 1.0, 0.5))


def uncapped_residual(target: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    """B1: uncapped raw residual (cap 은 inference 만)."""
    return target - cands  # NO cap_vectors
```

### §5.2 H006_Z1G2 학습 wrapper (`analysis/plan-010/h006_train.py`)

```python
# pseudo-code
def h006_z1g2_train(fold: int):
    # 1. Load plan-004 GRU checkpoint (frozen)
    gru = load_plan_004_gru_checkpoint()
    gru.eval()
    for p in gru.parameters():
        p.requires_grad = False

    # 2. Build 단일공식 candidates: frenet_par120_perp_neg020 only
    single_spec = selector.CANDIDATES[17]  # frenet_par120_perp_neg020
    cands = make_candidates_single(train_x, single_spec)  # [N, 1, 3] K=1

    # 3. Compute uncapped residual (B1)
    residual = uncapped_residual(target=train_y, cands=cands)  # [N, 1, 3]
    local_residual = vector_to_local(residual, basis, scale)  # NO cap

    # 4. Compute weight (C1+C2)
    err = torch.norm(target[:, None] - cands, dim=2)  # [N, 1]
    weights = weight_schedule(err)  # [N, 1]

    # 5. GRU forward (frozen) → hidden
    with torch.no_grad():
        gru_emb = gru.encode(train_x)  # [N, 32]
    gru_emb_per_cand = gru_emb.unsqueeze(1).expand(-1, 1, -1)  # [N, 1, 32]

    # 6. Build corrector input: cf [N, 1, 32] + gru_emb [N, 1, 32]
    cf = make_candidate_features_single(train_x, cands, single_spec)  # [N, 1, 32]

    # 7. Forward + huber loss
    model = RedesignedCorrectionNet(dim_cf=32, hidden=64, dim_encoder=32)
    delta = model(cf, encoder_emb=gru_emb_per_cand)  # [N, 1, 3]
    loss = (huber_loss(delta, local_residual) * weights).sum() / (weights.sum() + 1e-8)

    # 8. Inference: cap_vectors at cap=0.006 (Z1 E1: apply_scale=1.0)
    delta_vec = cap_vectors(local_to_vector(delta, basis, scale), cap=0.006)
    corrected = cands + 1.0 * delta_vec  # E1: apply_scale=1.0
```

### §5.3 H006 산출

- `runs/baseline/H006_Z1G2-min-redesign/`
  - `boundary_val_predictions.npz` (fold k val, K=1)
  - `test_predictions.npz`
  - `submission_h006.csv`
  - `report.json` (oof_soft_hit, per-band hit_after, corrector_oracle_gain, elapsed)
- `analysis/plan-010/h006_summary.json` (G1 metrics)

### §5.4 G1 합격

- (a) `oof_soft_hit ≥ 0.66` minimum, stretch 0.68
- (b) `[1, 1.5cm) hit_after ≥ 0.15`
- (c) `[0.5, 1cm) hit_after ≥ 0.95`
- (d) `corrector_oracle_gain ≥ 0`

### §5.5 G1 fail handling

- (a) 단독 fail (OOF < 0.66) → `min_redesign_marginal` warn-only, G2 진입 (encoder quality 추가 검증)
- (b) 또는 (c) 또는 (d) fail → `min_redesign_failure` severe, autonomous 옵션:
  - 옵션 a: G2 skip, G3 직접 진입 (iterative 가 cap 한계 우회 효과 가능)
  - 옵션 b: H006 의 per-component ablation 추가 sub-exp (B1/A2/C1/C2 단독 시도) — 결함 attribution
  - 옵션 c: G_final 직접 진입 (best Phase = H006 + plan-011 carry-over)

---

## §6. STAGE 2 (G2) — H007_Z1G1 CNN encoder learnable

### §6.1 CNN encoder block (`corrector_redesign.py` 추가)

```python
class TrajectoryCNNEncoder(nn.Module):
    """G1 (G2 단계의): 1-D CNN encoder over trajectory.

    Input:  x [N, T, 9] (SEQ_FEATURE_NAMES, T=last 10 step)
    Output: emb [N, 64]
    """
    def __init__(self, in_channels: int = 9, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [N, T, 9] → [N, 9, T] → CNN → [N, 64]
        return self.net(x.transpose(1, 2))
```

### §6.2 H007_Z1G1 학습 wrapper

H006 wrapper 와 동일, *변경*:
- `gru_emb = gru.encode(train_x)` → `cnn_emb = cnn_encoder(train_x_seq)`
- cnn_encoder 는 **trainable** (학습 가능)
- dim_encoder=64 (vs G1 32-dim GRU)
- parameter count assert ≤ 50K

### §6.3 G2 합격

- (a) `oof_soft_hit ≥ G1 OOF + 0.005` marginal, stretch G1 + 0.015
- (b) `encoder_param_count ≤ 50K`
- (c) over-fit gap (train OOF − val OOF) ≤ 0.05

### §6.4 G2 fail handling

- (a) fail → `cnn_encoder_marginal` warn-only, G3 진입 (iterative 가 더 큰 lever)
- (b) fail (≥ 50K params) → encoder hidden 축소 (32 → 16) retry

---

## §7. STAGE 3 (G3) — H008_Z3G2 Iterative refinement + frozen GRU

### §7.1 IterativeRefinementCorrector (`corrector_redesign.py` 추가)

```python
class IterativeRefinementCorrector(nn.Module):
    """Z3: n_steps × small_cap iterative refinement.

    각 step:
      cand_t → corrector(cand_t, encoder_emb, step_idx_emb) → delta_t (cap=per_step_cap)
      cand_{t+1} = cand_t + delta_t

    parameter 공유 (single corrector network), step_idx embedding 으로 step 구분.
    """
    def __init__(self, dim_cf: int, dim_encoder: int = 0, hidden: int = 64,
                 n_steps: int = 3, per_step_cap: float = 0.003):
        super().__init__()
        self.n_steps = n_steps
        self.per_step_cap = per_step_cap
        self.step_emb = nn.Embedding(n_steps, 8)  # step_idx embedding
        self.corrector = RedesignedCorrectionNet(
            dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder + 8
        )

    def forward(self, cf_at_step_0: torch.Tensor, encoder_emb: torch.Tensor | None,
                cands_at_step_0: torch.Tensor, basis, scale,
                # cf_recompute_fn: (cands) -> cf  (필요 시)
                ) -> torch.Tensor:
        """Returns: corrected_cands [N, K, 3] after n_steps."""
        cands = cands_at_step_0
        cf = cf_at_step_0
        for step in range(self.n_steps):
            step_emb = self.step_emb(torch.tensor([step], device=cands.device))
            step_emb = step_emb.expand(cands.shape[0], cands.shape[1], -1)
            enc = torch.cat([encoder_emb, step_emb], dim=-1) if encoder_emb is not None else step_emb
            delta_local = self.corrector(cf, encoder_emb=enc)  # [N, K, 3]
            delta_vec = cap_vectors(
                local_to_vector(delta_local, basis, scale),
                cap=self.per_step_cap,
            )
            cands = cands + delta_vec
            # NOTE: cf_recompute 는 *option* — 본 spec 은 step 별 cf 동일 (basis 동일).
            # 학습 안정성 위해 cf recompute X.
        return cands
```

### §7.2 H008_Z3G2 학습 wrapper

- frozen plan-004 GRU encoder (G1 동일)
- IterativeRefinementCorrector (n_steps=3, per_step_cap=3mm)
- 누적 cap = 9mm (단일 cap 6mm 보다 큼 + step 별 방향 재학습)
- loss = huber on *final* corrected position vs target (step 별 intermediate loss 옵션 — 본 spec X, 학습 안정성)

### §7.3 G3 합격

- (a) `oof_soft_hit ≥ G1 OOF + 0.01` minimum, stretch G1 + 0.025
- (b) ★ `[1, 1.5cm) hit_after ≥ 0.30` (plan-005 9.77% → 3x, plan-009 target 회수)
- (c) iter_gap (train OOF − val OOF) ≤ 0.05

### §7.4 G3 fail handling

- (b) fail → `iterative_divergence` severe, autonomous:
  - 옵션 a: per_step_cap 축소 (3mm → 2mm) + n_steps 증가 (3 → 5) retry
  - 옵션 b: G4 직접 진입 (e2e 가 cap 한계 더 잘 우회 가능)
  - 옵션 c: G_final 직접 진입 (corrector framework 본질 한계 인정 → plan-011 paradigm 교체 carry-over)
- (c) fail (iter_gap > 0.05) → over-fit. n_steps 축소 (3 → 2) retry

---

## §8. STAGE 4 (G4, 조건부) — H009_Z6 end-to-end GRU + corrector

### §8.1 진입 조건

- G1 ~ G3 누적 best LB 추정 < 0.72 시만 진입
- LB 추정 = best OOF + 0.022 (plan-005/008 trajectory gap)

### §8.2 E2EGRUCorrector (`corrector_redesign.py` 추가)

```python
class E2EGRUCorrector(nn.Module):
    """Z6: GRU encoder + corrector e2e (trainable GRU)."""
    def __init__(self, in_channels: int = 9, gru_hidden: int = 32,
                 dim_cf: int = 32, corrector_hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(in_channels, gru_hidden, batch_first=True)
        self.corrector = RedesignedCorrectionNet(
            dim_cf=dim_cf, hidden=corrector_hidden, dim_encoder=gru_hidden
        )

    def forward(self, x_seq: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        # x_seq [N, T, 9] → GRU → [N, T, 32] → last hidden [N, 32]
        _, h = self.gru(x_seq)
        gru_emb = h.squeeze(0)  # [N, 32]
        gru_emb_per_cand = gru_emb.unsqueeze(1).expand(-1, cf.shape[1], -1)
        return self.corrector(cf, encoder_emb=gru_emb_per_cand)
```

### §8.3 H009_Z6 학습 wrapper

- GRU + corrector 둘 다 trainable
- loss = huber on corrected position
- weight schedule + uncapped target (Z1 components)
- over-fit 방어: dropout 0.1, weight decay 1e-4, early stop patience 4

### §8.4 G4 합격

- (a) `oof_soft_hit ≥ G3 OOF + 0.005`
- (b) over-fit gap (train OOF − val OOF) ≤ 0.04
- (c) GRU encoder 가 selector task feature 와 *분기* 검증: GRU final hidden 의 PCA 1st component 가 plan-004 GRU 와 cosine similarity < 0.95 (feature drift evidence)

### §8.5 G4 fail handling

- (a) marginal — warn-only, best 채택
- (b) fail (overfit) → `e2e_overfit` warn-only, dropout ↑ + weight_decay ↑ retry
- (c) fail (cosine ≥ 0.95) → "trainable 의미 없음" — G3 best 유지

---

## §9. STAGE 5 (G_final) — synthesis + plan-011 후보 + carry-over

### §9.1 산출

- `analysis/plan-010/results.md` (10 section, OOF 표 + 4 결함 attribution + per-band table + caveat 검증)
- `analysis/plan-010/next_plan_candidates.md` (≥ 2 후보, decision-note transition)
- 3 파일 frontmatter sync:
  - `plans/plan-010-corrector-redesign-exploration.md` (status: partial/complete + LB carry-over + best_submission)
  - `plans/plan-010-corrector-redesign-exploration.results.md` (frontmatter only stub)
  - `analysis/plan-010/results.md` (자세한 finding)
- best Phase submission 경로 박제: `runs/baseline/<best_H_exp_id>/submission_<name>.csv`
- plan-010.1 carry-over instruction (다음 날 사용자 수동 dacon-submit, plan-008.1 + plan-009.1 묶음 패턴 답습)

### §9.2 results.md 필수 항목

1. §1 요약 (best phase, OOF, LB 추정 / TBD, plan-006 baseline 비교)
2. §2 OOF 표 (H006/H007/H008/H009 + plan-006 baseline)
3. §3 per-Phase contribution (ΔOOF)
4. §4 4 결함 attribution table:
   - Δ OOF for: B1 (uncapped target), A2 (Huber), C1 (easy=0), C2 (far=0.5), D1 (env drop), E1 (apply=1)
   - per-component ablation (G1 fail 시 §5.5 옵션 b 실행 결과)
5. §5 per-band Δ table (plan-005 corrector_decomp 패턴)
6. §6 caveat 검증 결과
7. §7 decision-note list
8. §8 plan-011 후보 (≥ 2)
9. §9 변경 이력
10. §10 plan-010.1 carry-over instruction

### §9.3 plan-011 후보 (≥ 2)

- 후보 1: **재설계 corrector + 27 후보 selector 결합** (plan-008 baseline 위 결함 fix 효과 측정)
- 후보 2: **재설계 corrector + plan-007 Step 4 per-sample MLP coeff 결합** (단일공식 framework 의 LB 미회수 carry-over 검증)
- 후보 3 (조건부, 4 후보 모두 0.70 미달 시): **paradigm 교체** (KNN over cand pool / GP posterior / plan-006 회귀 / Diffusion model)
- 후보 4 (조건부): **band-weight grid tuning on 재설계 corrector** (plan-009 G2 의 5-fold concat 재측정 + 재설계 corrector 위)

---

## §N+1. 작업량 총 회계

| commit | task | 예상 wall-time |
|---|---|---|
| c1 (docs) | plan-010 v1 spec 작성 | 0 (본 commit) |
| c2 (G0) | preflight.py + 3 task (verify + reproduce + decomp 재측정) | ~10 min |
| c3 (Z1 module) | corrector_redesign.py 신규 모듈 (RedesignedCorrectionNet + huber + weight_schedule + uncapped_residual) | ~15 min |
| c4 (H006 wrapper) | h006_train.py (frozen GRU forward + 5-fold OOF) | ~20 min |
| c5 (H006 학습) | 5-fold × 단일공식 × 결함 fix corrector + frozen GRU + Z1 components | ~30 min |
| c6 (CNN encoder) | corrector_redesign.py CNN block + h007_train.py | ~15 min |
| c7 (H007 학습) | 5-fold × CNN encoder learnable | ~50 min (CNN 학습) |
| c8 (Iterative) | corrector_redesign.py IterativeRefinementCorrector + h008_train.py | ~20 min |
| c9 (H008 학습) | 5-fold × iterative (3-step, parameter 공유) + frozen GRU | ~50 min |
| c10 (E2E, 조건부) | corrector_redesign.py E2EGRUCorrector + h009_train.py | ~25 min |
| c11 (H009 학습, 조건부) | 5-fold × trainable GRU + corrector e2e | ~80 min |
| c12 (synthesis) | results.md + next_plan_candidates.md + 3 파일 frontmatter sync + plan-010.1 instruction | ~30 min |
| **합계** | (조건부 G4 포함) | **~5.5 hr** (G4 skip 시 ~3.5 hr) |

---

## §N+2. results.md 필수 항목

§9.2 참조 (10 section).

---

## §N+3. 통계 함정 & caveats

1. **Huber beta=5mm 의 의미**: cap=6mm 의 ~80% 지점. err < 5mm 는 MSE-like (gradient ↑ near 0), err ≥ 5mm 는 L1-like (outlier robust). 단 huber 가 *완전 hit-aware* 는 아님 — plan-011 의 A1 (smooth hinge) 가 추가 검증 lever.

2. **frozen GRU 의 feature 부적합 risk**: plan-004 GRU 는 27 후보 ranking task 위 학습. 단일공식 + corrector 의 task feature 와 *다를* 가능성 — G2 (H007 CNN learnable) 가 비교 anchor.

3. **iterative refinement 의 발산 risk**: per_step_cap × n_steps = 9mm 누적. 작은 dataset (10K) + 매 step 방향 재학습 → noise 누적 발산 위험. step_idx embedding + parameter 공유 + huber loss 세 가지 안정장치.

4. **GRU forward (frozen) 의 input 차이**: plan-004 GRU 는 `x_seq` (full trajectory T step) 입력. 본 plan 의 frozen forward 는 *동일 입력* — drift 없음 보장 (state_dict diff = 0 검증).

5. **단일공식 K=1 의 oracle ceiling**: frenet_par120_perp_neg020 만 사용 시 oracle (best candidate 가 자기 자신) = 0.6320 (plan-007 per_candidate_hit.md 측정). corrector 가 *그 위* 로 끌어올림. 27 후보 + selector 의 oracle 0.7188 (plan-008 baseline) 보다 *낮은 천장* — 단일공식 + corrector 의 *진정한 ceiling* 검증.

6. **per-band hit_after target 0.30 의 근거**: plan-005 의 9.77% 가 결함 corrector 의 측정. 결함 ① (cap-truncated target) fix 시 *방향 학습 가능 영역* 확장 → 3x 회복 추정. 단 본질 한계 ② (Frenet frame) 와 ④ (oracle ceiling) 의 잔여 영향 미지.

7. **end-to-end (Z6) 의 over-fit dominance**: 10K small dataset + e2e GRU (parameter ~+100K). dropout/weight_decay 보수적 시작, G4 (c) cosine drift check 로 trainable 의미 검증.

8. **LB → OOF gap 가정**: plan-005/008 trajectory +0.022. plan-009 H002 b 의 +0.0095 (이상치 — fold 0 over-fit 의심) 와 차이. 본 plan = 5-fold concat strict → trajectory +0.022 채택. plan-010.1 회수 후 갱신.

9. **LB 제출 0 회**: 할당량 소진 인계. 모든 H006~H009 submission.csv 는 *생성·박제만*. plan-010.1 carry-over (사용자 수동 dacon-submit, plan-008.1 + plan-009.1 묶음과 동일 정책).

10. **frozen GRU 의 input format mismatch**: plan-004 GRU 의 forward signature 확인 필요 (c4 step). 단일공식 + frozen GRU encoder 는 *마지막 hidden state* 만 사용 — `gru.encode()` API 가 plan-004 selector module 에 존재 여부 검증 (G0 step).

11. **CNN encoder 의 input window**: 본 spec `T=last 10 step` 기본값. plan-004 SEQ_FEATURE_NAMES 9-dim × T step. T 가 trajectory 마다 다르면 padding 필요 — 학습 wrapper 에서 truncate/pad 처리.

12. **plan-006 baseline reproduce 의 OOF vs argmax**: plan-006 의 0.6491 = corrected argmax OOF, plan-007 per_candidate_hit 의 0.6320 = raw single-formula argmax (corrector 없이). G0 reproduce target = **0.6320 raw** (corrector 없이 단일공식 만). 본 plan 의 H006 OOF 와 비교 시 *후자 anchor*.

13. **caveat #13 (corrector framework 본질 한계 분리)**: 본 plan 의 모든 후보 0.70 미달 시 → 본질 한계 ②+④ (Frenet frame + oracle ceiling) 가 *진짜 bottleneck* 확정 → plan-011 paradigm 교체 (KNN/GP/Diffusion).

14. **Variant A residue 방지**: 본 plan 은 selector 학습 X (frozen GRU forward only). regime_prior_strength 등 selector hyperparam touch X — `single_formula_residue` severe trigger.

---

## §N+4. 변경 이력

- v1 (2026-05-13): 초안 — plan-009 carry-over + 사용자 challenge (plan-004 corrector 7 결함) + 4 후보 narrative (Z1+G2 → Z1+G1 → Z3+G2 → Z6). LB 제출 0 회 (plan-009.1 carry-over 패턴 답습). G0~G_final 6 gate, commit chain c1~c12 + G4 조건부.

---

## §N+5. 참조

- `plans/plan-004-pb-0-6822-fullrun.md` (corrector arch baseline)
- `plans/plan-005-pb-0-6822-diagnostic.md` (corrector_decomp band table)
- `plans/plan-006-minimal-variant-e-lb.md` (단일공식 baseline)
- `plans/plan-007-formula-tuning.md` (per_candidate_hit, Step 4 carry-over)
- `plans/plan-008-candidate-redefine-corrector-redesign.md` (oracle + ranking diagnostic)
- `plans/plan-009-selector-ranking-loss.md` (corrector 강화 5 sub-exp, framework 한계 finding)
- `notes/PB_0.6822 코드공유.ipynb` (cell 6 boundary corrector 원본)
- `WORKFLOW.md` (§12 Autonomous Execution Protocol)
- `CLAUDE.md` (Autonomous Execution Policy + Push 의무)
