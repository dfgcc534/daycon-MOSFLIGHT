---
plan_id: 012
version: 1
date: 2026-05-13 (Asia/Seoul)
status: written
based_on:
  - 004
  - 005
  - 006
  - 007
  - 010
  - 011
  - notes/PB_0.6822 코드공유.ipynb
  - notes/코드공유-upgrade.md
followed_by:
  - 012.1 (LB carry-over; user manual dacon-submit)
scope: 단일공식 + corrector path 의 *paradigm reframe*. plan-005~011 의 residual *regression* path 전체를 폐기하고 *Frenet local frame discrete-offset ring (9 후보) classification + soft-mean blending* 으로 직접 대체. 단일공식 (frenet_par120_perp_neg020, plan-006 LB 0.6692 baseline) 의 *형식적 hit potential* (~64%@1cm, ~84%@1.5cm) 을 anchor 로, *공식 예측점 주변 9 directional candidate* 위에서 plan-004 selector arch (CandidateAttentionGRUSelector) 의 head 만 27→9 swap. baseline lock (E0, G1) 후 4 axis core (E1 Frame × E2 Density × E3 Temperature × E4 Loss, G2) + 3 axis aux (E5 BoundaryWeight × E6 ScorerArch × E7 r=0 Prior, G3) ablation + best-stack 5-fold (G4) + plan-011 처럼 LB carry-over (0 제출).
exp_ids:
  - H019_phase0-preflight-oracle         # G0 — 64/84 hit measure + oracle ceiling (E8) + plan-006 reproduce
  - H020_phase1-baseline                 # G1 — E0 baseline 1-fold OOF lock-in
  - H021_phase2-frame                    # G2.E1 — Frenet vs World frame swap
  - H022_phase2-density                  # G2.E2 — ring density 9 / 17 / 5 swap
  - H023_phase2-temperature              # G2.E3 — τ scan (argmax + 5 points)
  - H024_phase2-loss                     # G2.E4 — L7 hinge vs distance regression
  - H025_phase3-boundary-weight          # G3.E5 — boundary sample weighting on/off
  - H026_phase3-scorer-arch              # G3.E6 — full Attn-GRU vs last-step MLP (★ 시계열 input 가치 측정)
  - H027_phase3-r0-prior                 # G3.E7 — r=0 logit prior 강도 (0 / +0.5 / +1.0)
  - H028_phase4-final-5fold              # G4 — best stack 5-fold + submission 박제
lb_score: null
---

# plan-012 v1 — Frenet Local-Frame Ring Classification (paradigm reframe)

## §0. 한 줄 목적

> **plan-005~011 의 residual *regression* path 전체 폐기 → *Frenet local frame discrete-offset ring (9 후보) classification + soft-mean blending* 으로 paradigm reframe**. 단일공식 (frenet_par120_perp_neg020, plan-006 LB 0.6692) 의 *형식적 hit potential* — 학습 없이 공식만으로 정답의 ~64% 가 1cm 이내, ~84% 가 1.5cm 이내 — 을 anchor 삼아, *공식 예측점 주변 9 directional candidate* (1 center + 4@0.5cm + 4@1.0cm) 위에서 plan-004 의 `CandidateAttentionGRUSelector` 를 head 만 27→9 swap. *residual 회귀가 직접적으로 실패한* (plan-005 destructive band [0.5, 1cm) -7.83pp + plan-010/011 의 corrector 4 후보·24 sub-exp 모두 NEGATIVE 또는 marginal) 가장 큰 이유 = *공식 자체가 이미 충분히 좋다; 남은 task 는 directional commit* 라는 통찰 위에 세움.
>
> **plan-010/011 과의 path 분리**:
> - plan-010 = corrector path 의 *depth* (7 결함 sequential fix). 4 후보 모두 marginal.
> - plan-011 = corrector path 의 *breadth* (4 axis × 24 sub-exp ablation). G1 결과 1/4 axis만 positive (In/ID +0.0050).
> - **plan-012 = paradigm 자체 교체** — corrector path 폐기 + selector-style classification path 신설.
>
> **재사용 / 비재사용**:
> - 재사용 = `selector.make_seq_features` (시계열 9-dim × 6-step encoder input, plan-004 그대로) + `selector.CandidateAttentionGRUSelector` encoder + `selector.search_temperature` (soft-mean 산출) + plan-006 의 single formula `frenet_par120_perp_neg020`.
> - 비재사용 = `boundary.py` (corrector path 전체), plan-010 `corrector_redesign.py`, plan-011 `corrector_redesign_v2.py`, plan-008 `candidates_extended.py` 의 27 물리후보.
>
> **Target**: 5-fold OOF ≥ 0.66, LB 추정 0.68~0.72 (plan-006 LB 0.6692 위 +0~+0.05). oracle ceiling (G0, E8) = 0.84 근방 예상 (= 학습 무관 이론치). best stack 5-fold OOF 와 oracle 의 gap = scorer 의 향후 lever.
>
> **LB 제출 정책**: 본 plan 내 LB 제출 **0 회** (plan-009.1 + plan-010.1 + plan-011.1 carry-over pattern 답습). 모든 sub-exp submission.csv 는 *생성·박제만*, LB 회수는 plan-012.1 carry-over.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0** (Phase 0 preflight + oracle ceiling): 3 measure 박제 — (a) F0 (frenet_par120_perp_neg020) 의 raw single-formula hit@1cm 과 hit@1.5cm 박제 (= "64/84 hypothesis 의 실측치"). (b) oracle ceiling (E8) = train_y 와 9 후보 중 *argmin distance* 후보의 hit@1cm 박제 (= "scorer 가 perfect 라면 도달 가능한 상한"). (c) plan-006 reproduce drift ≤ 0.005 (`oof_argmax_hit_corrected` baseline 0.6491 ±0.005). `analysis/plan-012/preflight.json` 생성. 위반 시 `preflight_artifact_missing` severe.
- **G1** (Phase 1 baseline lock-in): E0 (single ring 9 후보 + Frenet + Attn-GRU + τ=0.03 + L7) 의 fold-0 OOF soft hit ≥ **plan-011 In/ID anchor (0.6450)** — 즉 plan-012 paradigm 이 plan-011 best Phase 1 sub-exp 와 *최소 동등*. 미달 시 `baseline_below_anchor` warn (= paradigm path 자체가 무력 evidence; autonomous Phase 2 진행 + G_final 에서 path 폐기 판단). 위반 시 *halt 아님* — Phase 2 계속.
- **G2** (Phase 2 core ablation, 4 axis × ~10 sub-exp 총합): (a) 모든 4 axis 의 sub-exp informational 완료 (fail 없음 — attribution 목적). (b) 4 axis 중 *최소 1 axis* 에서 `max(ΔOOF) ≥ 0.005` (= "어느 한 axis 라도 positive lever 존재"). 위반 시 `phase2_no_positive_lever` severe — autonomous recovery (a) Phase 3 진행 후 G_final 에서 path-pivot 또는 (b) best Phase 1 baseline 으로 G4 직진.
- **G3** (Phase 3 aux ablation, 3 axis × ~6 sub-exp): (a) 모든 3 axis 의 sub-exp 완료. (b) 최소 1 axis positive 권장 (informational, hard fail 없음).
- **G4** (Phase 4 final 5-fold): best stack (Phase 2 best + Phase 3 best, additive 가정) 의 5-fold concat OOF soft hit ≥ G1 + 0.005. submission.csv 생성. 위반 시 `final_no_additive` warn — plan-012.1 carry-over 시 best Phase 2 sub-exp 단독 submission 도 후보.
- **G_final**: synthesis + plan-013 후보 ≥ 3 + 3 파일 frontmatter sync (`lb_score: null` carry-over) + best Phase submission 박제 + plan-012.1 carry-over instruction 박제.

### G-gates

- G0: preflight + oracle ceiling 박제 [TODO]
- G1: E0 baseline 1-fold OOF ≥ 0.6450 [TODO]
- G2: Phase 2 core (4 axis × ~10 sub-exp) — 최소 1 axis +0.005 [TODO]
- G3: Phase 3 aux (3 axis × ~6 sub-exp) — informational [TODO]
- G4: Phase 4 best stack 5-fold ≥ G1 + 0.005 [TODO]
- G_final: synthesis + plan-013 후보 + 3 파일 frontmatter sync + plan-012.1 instruction [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-012-frenet-ring-classification.md` v1 작성 | [TODO] |
| c2 | code | `src/pb_0_6822/ring_classifier.py` — 9-candidate ring generator + `CandidateAttentionGRUSelector` head swap (cand_dim 32→10) + L7 hinge loss + soft-mean inference. spec @ §4 | [TODO] |
| c3 | code+exp | `analysis/plan-012/preflight.py` — F0 raw hit@{1cm, 1.5cm} measure + oracle ceiling (E8) + plan-006 reproduce. spec @ §5 | [TODO] |
| G0 | gate | `preflight.json` 생성 + 3 measure 박제 (raw hit 64%/84% region + oracle + reproduce drift ≤0.005) | [TODO] |
| c4 | code+exp | `analysis/plan-012/phase1_baseline.py` — E0 1-fold OOF (★ baseline anchor 박제). spec @ §6 | [TODO] |
| G1 | gate | E0 OOF ≥ 0.6450 (plan-011 In/ID anchor) | [TODO] |
| c5 | code | `analysis/plan-012/phase2_core.py` — wrapper for E1~E4 (4 axis dispatcher). spec @ §7 | [TODO] |
| c6 | exp | Phase 2.E1 — Frame swap (Frenet vs World, 2 sub-exp) | [TODO] |
| c7 | exp | Phase 2.E2 — Density swap (5 / 9 / 17, 3 sub-exp) | [TODO] |
| c8 | exp | Phase 2.E3 — Temperature scan (argmax + {0.01, 0.03, 0.1, 0.3, 1.0}, 6 sub-exp) | [TODO] |
| c9 | exp | Phase 2.E4 — Loss swap (L7 hinge vs distance regression, 2 sub-exp) | [TODO] |
| G2 | gate | 4 axis informational 완료 + 최소 1 axis +0.005 ΔOOF | [TODO] |
| c10 | code | `analysis/plan-012/phase3_aux.py` — wrapper for E5~E7. spec @ §8 | [TODO] |
| c11 | exp | Phase 3.E5 — Boundary sample weighting on/off (2 sub-exp) | [TODO] |
| c12 | exp | Phase 3.E6 — Scorer arch (full Attn-GRU vs last-step MLP, 2 sub-exp). ★ 시계열 input 가치 측정 | [TODO] |
| c13 | exp | Phase 3.E7 — r=0 logit prior 강도 (0 / +0.5 / +1.0, 3 sub-exp) | [TODO] |
| G3 | gate | aux ablation 완료 (informational) | [TODO] |
| c14 | code+exp | `analysis/plan-012/phase4_final.py` — best Phase 2 + best Phase 3 stack 5-fold + submission 박제. spec @ §9 | [TODO] |
| G4 | gate | 5-fold OOF ≥ G1 + 0.005 + submission.csv 박제 | [TODO] |
| c15 | analysis | `analysis/plan-012/results.md` + `next_plan_candidates.md` (≥ 3 후보) + 3 파일 frontmatter sync + plan-012.1 carry-over instruction. spec @ §10 | [TODO] |
| G_final | gate | synthesis + plan-013 후보 + 3 파일 sync + plan-012.1 instruction | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `preflight_artifact_missing` — G0 의 `preflight.json` 미생성 또는 plan-006 reproduce 실패 (|measured − 0.6491| > 0.005). severity=**severe**.
- `phase2_no_positive_lever` — G2 의 4 axis 중 *어느 axis 도* +0.005 marginal 없음 (= plan-012 paradigm 자체 부정 신호). severity=**severe** 이지만 autonomous recovery (a) Phase 3 진행 후 G_final path-pivot 또는 (b) best Phase 1 baseline 으로 G4 직진 으로 *halt 아닌 path-pivot* — 사용자 escalate 불필요.
- `final_no_additive` — G4 의 best stack 5-fold OOF < G1 + 0.005 (= component 결합 super-additive 실패). severity=**warn**. autonomous = best Phase 2 single-axis sub-exp 의 5-fold 으로 fallback submission 박제.
- `frozen_gru_drift` — Phase 2.E6 full Attn-GRU sub-exp 에서 plan-004 GRU encoder weight drift detected (state_dict diff > 0). severity=**severe** (single-axis attribution 의 정합성 깨짐).
- `ring_geometry_drift` — Phase 1 E0 anchor 의 `compute_ring_offsets()` 출력이 spec 과 다름 (9 후보 좌표 ‖center‖ < 1e-6, ‖r=0.5 ring‖ ∈ [0.0048, 0.0052], ‖r=1.0 ring‖ ∈ [0.0098, 0.0102]). severity=**severe**.
- `dilution_collapse` — Phase 2.E3 의 τ ≥ 0.3 sub-exp 에서 *비-원점* 후보의 평균 |soft-mean delta| < 0.001m (= 1mm) — soft-mean 이 origin 으로 완전 회귀, directional commit 실패. severity=**warn** (E3 sub-exp 단독 무효, axis 전체는 진행).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 default 위 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/ring_classifier.py` (신규 모듈, 본 plan main code)
  - `analysis/plan-012/**` (preflight, phase1_*, phase2_*, phase3_*, phase4_*, results, next_plan_candidates)
  - `runs/baseline/H020_phase1-baseline/**` (= 본 plan baseline run)
  - `runs/baseline/H021_phase2-frame/**` ~ `runs/baseline/H028_phase4-final-5fold/**` (sub-exp runs)
- whitelist 제외 (blacklist 추가):
  - `src/pb_0_6822/boundary.py` (touch X — corrector path 전체 폐기, import 도 X)
  - `src/pb_0_6822/selector.py` (touch X — frozen forward only: `make_seq_features`, `CandidateAttentionGRUSelector` encoder, `search_temperature` reuse)
  - `src/pb_0_6822/corrector_redesign.py` / `corrector_redesign_v2.py` (plan-010/011 산출, 본 plan scope X)
  - `src/pb_0_6822/candidates_extended.py` (plan-008 산출, 본 plan scope X — single formula 만)
- 참조 (read-only):
  - `runs/baseline/F001_variant-e/**` (plan-006 산출, single formula baseline checkpoint)
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (plan-004 산출, GRU encoder pretrained weight)
  - `analysis/plan-005/corrector_decomp.{md,json}` (★ band table baseline — destructive band evidence)
  - `analysis/plan-007/per_candidate_hit.{md,json}` (★ raw single formula ranking)
  - `analysis/plan-011/results.md` (★ plan-011 In/ID +0.0050 anchor)
  - `notes/PB_0.6822 코드공유.ipynb` cell 4 (`CandidateAttentionGRUSelector` 원본)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Phase 1 baseline (E0) lock-in fold=0 only (~5min/sub-exp), Phase 4 5-fold 강제`
- `decision-note: spec-default — Phase 2/3 sub-exp 의 anchor combo = E0 (Frenet × density=9 × τ=0.03 × L7 × no-boundary-weight × full-Attn-GRU × r=0 prior=+0.0), 1-axis 만 변경`
- `decision-note: spec-default — ring r ∈ {0, 0.5, 1.0}cm × θ ∈ {0, π/2, π, 3π/2} = 9 후보 (1 center + 4×2 ring)`
- `decision-note: spec-default — Frenet anchor point = F0 (frenet_par120_perp_neg020) 의 *예측점*, |v| < 1e-6 시 world frame fallback`
- `decision-note: conditional-skip — G1 OOF < 0.6450 시 baseline_below_anchor warn + Phase 2 진행 (path 부정 evidence 박제만)`
- `decision-note: conditional-skip — G2 0/4 axis positive 시 phase2_no_positive_lever severe + autonomous recovery (Phase 3 진행 후 G_final path-pivot)`
- `decision-note: conditional-skip — G4 best stack 5-fold < G1 + 0.005 시 best Phase 2 single-axis 5-fold 으로 fallback submission`

---

## §1. 배경 / 이전 plan 인계

### §1.1 plan-005~011 의 residual regression path 한계 (★ paradigm 폐기 evidence)

| plan | path | best OOF (corrected) | best LB | evidence |
|---|---|---|---|---|
| plan-004 | 27-candidate selector + boundary corrector | 0.6491 (variant E reproduce) | **0.6822** (notebook) / 0.6692 (우리 data) | full-stack, *우리 data 의 best LB 까지의 anchor* |
| plan-005 | corrector_decomp 분해 | 0.6524 (5-fold soft) | — | ★ destructive band [0.5, 1cm) **-7.83pp** evidence |
| plan-006 | single formula + plan-004 corrector | 0.6491 (corrected) | **0.6692** | single formula path baseline |
| plan-007 | single formula 4-step 개선 (CMA-ES → per-sample MLP) | 0.6482 (Step 4) | carry-over 미회수 | regression path |
| plan-008 | candidate redefine + corrector redesign | — | marginal | 27→34 후보 확장 |
| plan-010 | corrector_redesign (Z1~Z6, 4 후보 depth) | 0.6320 (Z1 anchor) | — | ★ 4 후보 모두 marginal 또는 NEGATIVE |
| plan-011 | corrector_redesign_v2 (4 axis × 24 sub-exp breadth) | 0.6450 (★ In/ID best Phase 1) | — | ★ 1/4 axis positive only (In/ID +0.0050) |

**→ 결론**: 4 plan (006, 008, 010, 011) 의 corrector path 가 모두 *plan-006 LB 0.6692 base 위로 의미있게 못 올라옴*. plan-011 의 4-axis breadth 박제 결과 = 24 sub-exp 중 *단 1개* (In/ID) 만 +0.005 marginal. paradigm 자체의 한계 가능성 강함.

★ **plan-005 destructive band [0.5, 1cm)**: 2594 sample 중 -203 hits lost — **residual 회귀가 이미 hit 인 sample 을 *밀어내는* 측정 evidence**. plan-011 C008/L2 gate-asymmetric loss 도 이 band 완전히 못 회복 (gate output collapse 위험).

### §1.2 직관적 reframe — "공식이 이미 충분히 좋다"

**측정된 사실** (G0 preflight 에서 *재측정 + 박제 책임*):

| 사실 (가설) | 검증 method | 함의 |
|---|---|---|
| F0 (frenet_par120_perp_neg020) 의 raw hit@1cm ≈ 64% (= 학습 무관) | G0 preflight (a) | 공식 단독으로 *2/3 sample 정답 1cm 이내*. 학습 task = 나머지 36% 의 *방향성 commit* |
| F0 의 raw hit@1.5cm ≈ 84% | G0 preflight (a) | 공식 단독으로 *5/6 sample 정답 1.5cm 이내*. 36% 중 *20%* 는 0.5~1cm 만 적절히 shift 해도 hit 진입 |
| direct MLP residual prediction 매우 어려움 (= plan-010 Z1 G1 minimal NEGATIVE, plan-011 24 sub-exp 23 NEGATIVE) | plan-010/011 results | continuous residual regression 의 *smooth loss surface + label noise 1cm scale* 에서 학습 신호 약함 |

→ **세 사실의 곱셈** = `(공식이 이미 64% hit) × (1.5cm 까지 확장하면 84%) × (직접 residual 회귀는 실패)` → **"공식 예측점 주변 9 directional candidate 의 classification + soft-mean 으로 0.5~1cm directional commit 을 학습"** 으로 자연스럽게 reframe.

### §1.3 plan-004 selector arch (`CandidateAttentionGRUSelector`) 의 재사용 근거

| 사용 컴포넌트 | 위치 | 변경 |
|---|---|---|
| `selector.make_seq_features(x, end_idx, direction=1.0)` | `src/pb_0_6822/selector.py:406-449` | **1:1 동일 reuse**. (N, 6, 9) shape — 6 time step × 9 dim (speed/prev_speed/acc_norm/acc_par/perp_norm/jerk_norm/turn_cos/curvature + direction) |
| `selector.CandidateAttentionGRUSelector(seq_dim, cand_dim, hidden, cand_count)` | `src/pb_0_6822/selector.py:697-720` | **encoder 1:1 reuse, head 만 swap** — `cand_count` 27→9, `cand_dim` 32→**10** (ring-native: par, perp, dist, radius, angle_sin, angle_cos, ring_id one-hot 2 + ctx broadcast) |
| `selector.search_temperature(corrected_pos, scores, true_pos)["metrics"]["hit"]` | `src/pb_0_6822/selector.py` | **1:1 reuse** — soft-mean 산출 시 temperature scan 결과 metric 계산 |
| 단일공식 F0 = `frenet_par120_perp_neg020` (CANDIDATES[17]) | plan-006 §5.5 selector picked | plan-006 baseline 그대로 |
| GRU pretrained weight | `runs/baseline/P001_pb-0-6822-fullrun/**` | Phase 2.E6 sub-exp B (frozen GRU) 에서만 load + frozen forward |

★ **시계열 input 회복** (plan-011 §1.5 의 "corrector input snapshot 한계" 와의 차이):
- plan-011 Input axis 의 In-D/In-E/In-F = corrector 에 시계열 *재주입* 시도 — 모두 NEGATIVE 또는 marginal.
- plan-012 = corrector 가 아닌 *selector* head. plan-004 의 GRU 가 이미 시계열 처리 → 그대로 받음 (= plan-011 의 시계열 input *간접 회복* 시도의 본격 paradigm reframe).

### §1.4 plan-011 In/ID +0.0050 anchor 의 의미

plan-011 §13 results.md 박제:
- In/ID (frozen GRU encoder 으로 corrector 에 GRU embedding 시계열 주입) = ΔOOF +0.0050 (★ 4 axis 중 *유일* positive axis)
- best Phase 1 OOF = 0.6450 (= plan-006 corrected 0.6491 보다 -0.0041 낮음 — *axis-positive 인데도 baseline 보다 낮은* 이유 = anchor 자체가 plan-011 reproduce drift)

→ **plan-012 의 G1 합격선 = 0.6450** (= plan-011 의 best Phase 1 sub-exp 와 *최소 동등*). 미달 시 paradigm 자체 부정 evidence (warn 만, halt 아님).

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| selector head | **9-candidate ring classifier** (1 center + 4@0.5cm + 4@1.0cm in Frenet local frame) |
| selector encoder | plan-004 `CandidateAttentionGRUSelector` encoder (frozen 또는 trainable, E6 axis 에서 결정) |
| single formula F0 | `frenet_par120_perp_neg020` (plan-006 picked, CANDIDATES[17]) |
| LB 제출 | **0 회** (할당량 carry-over, plan-009.1+010.1+011.1 pattern 답습) |
| 학습 데이터 | train 10K (plan-004 동일) |
| Validation | Phase 1~3: 1-fold OOF (fold=0, N_val≈2020) approx. Phase 4: 5-fold concat |
| GPU | server cuda:1 (plan-004/005/008/009/010/011 동일) |
| Loss | L7 hit-aware smooth hinge (plan-011 §5.1 정의 reuse, baseline E0) |
| Inference | soft-mean blending (`Σ softmax(logits/τ) × candidate_pos`), τ=0.03 baseline |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| corrector path (residual regression) | plan-005~011 의 path. 본 plan = paradigm reframe 으로 분리. `boundary.py` / `corrector_redesign*.py` 일체 import X |
| 27 후보 physics candidate set | plan-008 산출. 본 plan = 9 ring candidate 만 |
| LB 제출 | 할당량 소진 (plan-011.1 까지 사용). 본 plan = carry-over |
| selector encoder 본문 수정 | `selector.py` blacklist. encoder reuse 만 (head 는 신규 모듈 `ring_classifier.py` 에서) |
| candidate 동적 생성 (학습 중 ring radius 자동 조정 등) | scope creep. Fixed ring geometry 만 (E2 에서 ring size 만 ablation) |
| TTA / multi-parse inference (plan-011 Phase 6) | 본 plan scope X. plan-013 후보로 carry-over |
| iterative refinement (plan-011 Phase 5 / Z3) | residual regression 의 iterative 변형 — paradigm 분리 |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold OOF: `selector.stable_fold_id(sample_id, folds=5)` (plan-004 동일)
- Phase 1~3 fold=0 only (N_val ≈ 2020, binomial std ≤ 0.005)
- Phase 4 5-fold concat 강제 (overall_oof_hit_soft)

### §3.2 합격 기준

§0.5 G-gate sequence 참조.

### §3.3 평가 점수 / metric

- main metric: **5-fold concat OOF soft hit @ 1cm** (Phase 4) 또는 **1-fold OOF soft hit** (Phase 1~3)
- soft hit = `selector.search_temperature(corrected_pos, scores, true_pos)["metrics"]["hit"]`
- ΔOOF (axis attribution) = `OOF_with_lever − OOF_anchor` per sub-exp
- **paradigm-specific metric**: `directional_commit_magnitude` = `mean(‖soft_mean_delta − origin‖₂ for sample where origin = F0 prediction)` — soft-mean 이 origin (= F0 예측점) 으로 얼마나 회귀했는지의 척도. < 1mm 시 `dilution_collapse` warn.

### §3.4 Anchor 정의 (Phase 2/3 의 모든 ablation 의 기준점)

**Anchor combo (= E0 baseline)**:
- F0 = `frenet_par120_perp_neg020`
- Frame = Frenet local frame (anchor point = F0 prediction)
- Ring = 9 candidates (1 center + 4@0.5cm + 4@1.0cm × {0, π/2, π, 3π/2} angles)
- Scorer encoder = full `CandidateAttentionGRUSelector` on `make_seq_features` (frozen plan-004 GRU pretrained weight 으로 init, full fine-tune; E6 sub-exp B 에서만 frozen)
- Scorer head = MLP (cand_dim=10 → hidden → 1 logit per candidate)
- Loss = L7 hit-aware smooth hinge (huber + softplus squared, plan-011 §5.1 정의)
- Inference τ = 0.03
- Boundary sample weight = uniform (off, E5 axis 에서 on swap)
- r=0 candidate logit prior = **+0.0** (no prior, E7 axis 에서 swap)

각 sub-exp = anchor combo 에서 *1 axis 만 변경*.

### §3.5 Plan-011 anchor 비교

| measure | plan-011 In/ID | plan-012 E0 (예상) | gap |
|---|---|---|---|
| 1-fold OOF (fold=0) | 0.6450 | ≥ 0.6450 (G1 lock) | 0~? |
| 5-fold OOF (concat) | TBD | TBD (G4) | TBD |
| oracle ceiling (= E8) | N/A | ~0.84 (G0 preflight) | N/A |

---

## §4. STAGE 0 (c2) — `src/pb_0_6822/ring_classifier.py` 신규 모듈

### §4.1 모듈 책임 (5 컴포넌트, self-contained)

```python
# src/pb_0_6822/ring_classifier.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.pb_0_6822 import selector as base  # encoder + make_seq_features 만 reuse


# ── 컴포넌트 1: ring geometry generator ──

def compute_ring_offsets(
    radii_cm: list[float] = (0.0, 0.5, 1.0),
    angles: list[float] = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2),
    drop_duplicate_center: bool = True,
) -> np.ndarray:
    """Generate 2D Frenet-frame offsets for ring candidate set.

    radii_cm=[0.0, 0.5, 1.0], angles=4 → 1 (center) + 4×0.5 + 4×1.0 = **9 candidates**.
    radii_cm=[0.0, 0.5, 1.0, 1.5], angles=4 → 1 + 4 + 4 + 4 = **13 candidates** (E2 dense variant).
    radii_cm=[0.0, 0.5], angles=4 → 1 + 4 = **5 candidates** (E2 sparse variant).

    Returns: ndarray shape (K, 2) in *Frenet local frame* (t, n) coords, meters.
        - center: (0, 0) — single entry regardless of `angles` length when r=0 (드롭 중복).
        - r>0 entries: (r * cos(θ), r * sin(θ)) for θ in angles.
    Output ordering: center first, then by ascending radius, within-radius by angle order.

    geometry invariants (G0 / `ring_geometry_drift` severe check):
        - K == 9 for default args
        - ‖offsets[0]‖ < 1e-6  (center)
        - ‖offsets[1:5]‖ ∈ [0.0048, 0.0052]  (r=0.5cm ring, 0.5cm ± 2%)
        - ‖offsets[5:9]‖ ∈ [0.0098, 0.0102]  (r=1.0cm ring)
    """


# ── 컴포넌트 2: Frenet basis @ formula prediction point ──

def build_frenet_basis_at_prediction(
    trajectory_x: np.ndarray,    # (N, T, 2) world coords
    end_idx: int,                # last observation index
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Frenet basis (t̂, n̂) per sample, anchored at F0 *formula prediction point*.

    Procedure (2D simplification of plan-011 §5.1 build_frenet_basis):
        v = x[:, end_idx] − x[:, end_idx − 1]                    # (N, 2)
        t̂ = v / ‖v‖
        n̂ = rot90(t̂)                                            # (N, 2) — 2D perpendicular
    Degenerate ‖v‖ < 1e-6 → fallback (t̂=(1,0), n̂=(0,1))         # world frame fallback
        + caller emit `ring_geometry_degenerate` info (not severe, just info).

    Returns: (t_hat, n_hat) — each (N, 2).
    """


def frenet_offsets_to_world(
    ring_offsets: np.ndarray,    # (K, 2) in Frenet (t, n)
    t_hat: np.ndarray,           # (N, 2)
    n_hat: np.ndarray,           # (N, 2)
    anchor_pos: np.ndarray,      # (N, 2) F0 predicted position
) -> np.ndarray:
    """Convert ring offsets from Frenet local frame to world frame candidate positions.

    cand_pos[i, k] = anchor_pos[i] + ring_offsets[k, 0] * t_hat[i] + ring_offsets[k, 1] * n_hat[i]

    Returns: (N, K, 2) world frame candidate positions.

    E1 (Frame swap) variant: caller passes `t_hat=(1,0), n_hat=(0,1)` for *world frame* fallback
        (= ring axes aligned to world x/y, not trajectory-aligned).
    """


# ── 컴포넌트 3: candidate feature builder (cand_dim = 10) ──

def make_ring_candidate_features(
    cand_pos_world: np.ndarray,   # (N, K, 2)
    anchor_pos: np.ndarray,       # (N, 2)
    t_hat: np.ndarray,            # (N, 2)
    n_hat: np.ndarray,            # (N, 2)
    ring_offsets: np.ndarray,     # (K, 2) Frenet
    horizon: int = 2,             # plan-004 default
) -> np.ndarray:
    """Build candidate feature tensor for ring classifier scorer head.

    Per-candidate features (10 dim):
        [0] par              — Frenet par offset (= ring_offsets[k, 0])
        [1] perp             — Frenet perp offset (= ring_offsets[k, 1])
        [2] dist             — ‖ring_offsets[k]‖ (radius)
        [3] radius_norm      — dist / max(0.015, 1e-6)  (normalized to [0, ~1])
        [4] angle_sin        — sin(atan2(perp, par))
        [5] angle_cos        — cos(atan2(perp, par))
        [6] ring_id_center   — 1 if dist < 1e-6 else 0
        [7] ring_id_inner    — 1 if 0.004 < dist < 0.007 (= 0.5cm ring) else 0
        [8] ring_id_outer    — 1 if 0.007 < dist < 0.012 (= 1.0cm ring) else 0
        [9] direction        — +1.0 (plan-004 single-direction convention)

    NOTE: ctx 9-dim (plan-004 `make_candidate_features`) 는 *broadcast 안 함*. scorer encoder 가
    이미 시계열 ctx 처리 → head 에서는 candidate-specific feature 만.

    Returns: (N, K, 10) float32.
    """


# ── 컴포넌트 4: classifier head (CandidateAttentionGRUSelector 의 head 만 swap) ──

class RingClassifierHead(nn.Module):
    """plan-004 CandidateAttentionGRUSelector 의 head 만 27→9 swap.

    encoder 부분 (GRU + cand_attn) 은 base.CandidateAttentionGRUSelector 그대로 사용
    (instantiate 시 cand_count=9, cand_dim=10 로 호출).
    """

    def __init__(self, hidden: int = 64, encoder_pretrained_path: str | None = None):
        super().__init__()
        self.scorer = base.CandidateAttentionGRUSelector(
            seq_dim=9,            # make_seq_features last dim
            cand_dim=10,           # make_ring_candidate_features last dim
            hidden=hidden,
            cand_count=9,          # 9 ring candidates
        )
        if encoder_pretrained_path is not None:
            # plan-004 P001 산출 의 GRU + cand_attn weight load
            # (head 부분은 random init — 27 logit vs 9 logit shape mismatch 으로 자동 skip)
            self._load_encoder_weights(encoder_pretrained_path)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        """
        seq:       (B, 6, 9)  make_seq_features
        cand_feat: (B, 9, 10) make_ring_candidate_features
        returns:   (B, 9) logits
        """
        return self.scorer(seq, cand_feat)

    def freeze_encoder(self):
        """E6 sub-exp B (frozen GRU) 용. encoder 의 GRU + cand_attn parameter 만 freeze,
        head MLP 는 trainable 유지."""
        ...


# ── 컴포넌트 5: L7 hit-aware hinge loss (plan-011 §5.1 reuse) ──

def hit_aware_hinge(
    corrected_pos: torch.Tensor,  # (B, 2) soft-mean predicted position
    target: torch.Tensor,         # (B, 2) true_y
    R_HIT: float = 0.01,
    smooth: float = 0.005,
) -> torch.Tensor:
    """L7 smooth hinge — squared smoothed-hinge (m² units, dimensional-compatible with huber).
    plan-011 §5.1 hit_aware_hinge 와 동일 수식, 2D 적용.

    Returns: (B,) per-sample loss.
    """
    excess = torch.norm(corrected_pos - target, dim=1) - R_HIT
    linear_hinge = F.softplus(excess / smooth) * smooth   # smooth max(0, x), units = m
    return linear_hinge ** 2                              # m²


def huber_loss(pred, target, beta=0.005):
    """plan-011 §5.1 huber_loss 와 동일. (B, 2) → (B,)."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none').sum(dim=1)


def l7_combined_loss(
    soft_mean_pos: torch.Tensor,   # (B, 2)
    target: torch.Tensor,          # (B, 2)
    R_HIT: float = 0.01,
    smooth: float = 0.005,
) -> torch.Tensor:
    """E0 baseline loss = `0.5 * huber + 0.5 * hit_aware_hinge` (plan-011 P1.L7 동일 weight).
    Returns: scalar (batch mean)."""
    huber = huber_loss(soft_mean_pos, target)
    hinge = hit_aware_hinge(soft_mean_pos, target, R_HIT=R_HIT, smooth=smooth)
    return (0.5 * huber + 0.5 * hinge).mean()


# ── 컴포넌트 6: soft-mean inference ──

def soft_mean_predict(
    logits: torch.Tensor,         # (B, 9) raw scorer logits
    cand_pos_world: torch.Tensor, # (B, 9, 2) candidate world positions
    temperature: float = 0.03,
    r0_logit_prior: float = 0.0,  # E7 axis
) -> torch.Tensor:
    """Soft-mean blending: prob = softmax(logits / τ + r0_prior_mask), pred_pos = Σ prob × cand_pos.

    r0_prior_mask: shape (9,), value = `r0_logit_prior` at index 0 (= center), 0 elsewhere.
    Special case temperature == 0 (E3 argmax sub-exp): hard argmax, no soft-mean.

    Returns: (B, 2) predicted world position.
    """
    if temperature <= 1e-8:
        # argmax (= τ → 0 limit)
        idx = logits.argmax(dim=-1)                              # (B,)
        return cand_pos_world.gather(1, idx[:, None, None].expand(-1, -1, 2)).squeeze(1)
    prior = torch.zeros(logits.shape[-1], device=logits.device)
    prior[0] = r0_logit_prior
    prob = F.softmax((logits + prior) / temperature, dim=-1)     # (B, 9)
    return (prob[:, :, None] * cand_pos_world).sum(dim=1)         # (B, 2)
```

### §4.2 smoke test (c2 직후 self-check)

```python
# tests/test_ring_classifier_smoke.py
def test_ring_geometry_invariant():
    offsets = compute_ring_offsets()
    assert offsets.shape == (9, 2)
    assert np.linalg.norm(offsets[0]) < 1e-6
    assert all(0.0048 <= np.linalg.norm(o) <= 0.0052 for o in offsets[1:5])
    assert all(0.0098 <= np.linalg.norm(o) <= 0.0102 for o in offsets[5:9])

def test_classifier_forward_shape():
    head = RingClassifierHead(hidden=64)
    seq = torch.randn(4, 6, 9)
    cand_feat = torch.randn(4, 9, 10)
    logits = head(seq, cand_feat)
    assert logits.shape == (4, 9)

def test_l7_loss_at_zero_residual():
    pred = torch.zeros(4, 2)
    target = torch.zeros(4, 2)
    loss = l7_combined_loss(pred, target)
    assert loss.item() < 1e-6  # at exact hit, both huber and hinge → 0

def test_soft_mean_argmax_limit():
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    cand = torch.randn(1, 9, 2)
    pred_soft = soft_mean_predict(logits, cand, temperature=1e-3)
    pred_argmax = soft_mean_predict(logits, cand, temperature=0.0)
    assert torch.allclose(pred_soft, pred_argmax, atol=1e-3)
```

---

## §5. STAGE 0 (c3, G0) — Phase 0 preflight + oracle ceiling

### §5.1 산출물

- `analysis/plan-012/preflight.py` — 3 task 일괄 실행
- `analysis/plan-012/preflight.json` — schema:

```json
{
  "exp_id": "H019_phase0-preflight-oracle",
  "f0_raw_hit_measure": {
    "description": "F0 (frenet_par120_perp_neg020) 단일공식 의 raw hit@1cm, hit@1.5cm — 학습 무관 측정",
    "single_formula": "frenet_par120_perp_neg020",
    "candidate_idx": 17,
    "n_train": 10000,
    "hit_at_1cm":   {"raw_count": <int>, "hit_rate": <float>, "expected_range_lower": 0.60, "expected_range_upper": 0.68},
    "hit_at_1_5cm": {"raw_count": <int>, "hit_rate": <float>, "expected_range_lower": 0.80, "expected_range_upper": 0.88},
    "captures_target_band": {"hit_at_1_5cm_minus_hit_at_1cm": <float>, "expected_range": [0.15, 0.25]}
  },
  "e8_oracle_ceiling": {
    "description": "oracle scorer (label-aware argmin) 으로 9 ring candidate 중 best 선택 시 hit@1cm",
    "ring_config": {"radii_cm": [0.0, 0.5, 1.0], "angles": [0.0, 1.5708, 3.1416, 4.7124], "K": 9},
    "frame": "Frenet (anchor = F0 prediction)",
    "n_train": 10000,
    "oracle_hit_at_1cm": <float>,
    "expected_range_lower": 0.80,
    "expected_range_upper": 0.88,
    "gap_vs_f0_raw_hit_1_5cm": <float>
  },
  "plan_006_reproduce": {
    "description": "plan-006 §5.5 Variant E baseline reproduce (drift threshold ±0.005)",
    "single_formula": "frenet_par120_perp_neg020",
    "oof_argmax_hit_corrected_measured": <float>,
    "oof_argmax_hit_corrected_expected": 0.6491,
    "drift": <float>,
    "drift_threshold": 0.005,
    "reproduce_ok": <bool>
  }
}
```

### §5.2 실행

```bash
python -m analysis.plan-012.preflight \
  --root data \
  --plan-006-checkpoint runs/baseline/F001_variant-e/checkpoint_best.pt \
  --out                 analysis/plan-012/preflight.json
```

> decision-note: spec-default — Frenet basis 산출 = `build_frenet_basis_at_prediction` (§4.1 컴포넌트 2). degenerate sample (‖v‖ < 1e-6) 은 world frame fallback + count 박제 (informational, severe X).

### §5.3 G0 합격

- `f0_raw_hit_measure.hit_at_1cm` ∈ [0.60, 0.68] (= 사용자 conversation 의 "~64%" anchor ± noise)
- `f0_raw_hit_measure.hit_at_1_5cm` ∈ [0.80, 0.88] (= "~84%" anchor)
- `e8_oracle_ceiling.oracle_hit_at_1cm` ∈ [0.80, 0.88] (= "공식 raw hit@1.5cm 의 *대부분* 을 ring candidate 가 cover" 검증)
- `plan_006_reproduce.drift ≤ 0.005`

위 4 조건 모두 통과 → G0 PASS. 위반 시 `preflight_artifact_missing` severe.

> 만약 `f0_raw_hit_measure.hit_at_1cm` 가 [0.60, 0.68] 밖이면 = "사용자 conversation 의 64% 추정이 plan-006 의 F0 와 *다른 공식* 을 가리킴" → decision-note `spec-update` 박제 후 진행 (anchor 는 *측정된 값* 으로 갱신, plan 본문 base 는 그대로).

---

## §6. STAGE 1 (c4, G1) — Phase 1 E0 Baseline Lock-in

### §6.1 sub-exp 정의 (1 sub-exp only)

| sub-exp | 변경 | OOF anchor | metric |
|---|---|---|---|
| **P1.E0** | anchor combo (= §3.4 의 8 component 그대로) — single training run | n/a (this is the anchor) | fold-0 OOF soft hit |

### §6.2 학습 spec

```python
# analysis/plan-012/phase1_baseline.py

from src.pb_0_6822 import ring_classifier as rc
from src.pb_0_6822 import selector as base
import torch

# 1. 데이터 + F0 prediction
train_x, train_y, sample_ids = base.load_data("data")           # (N, T, 2), (N, 2), (N,)
f0_idx = 17                                                      # frenet_par120_perp_neg020
all_cands = base.make_candidates(train_x, end_idx=base.END_IDX, horizon=2)  # (N, 27, 2)
f0_pred = all_cands[:, f0_idx, :]                                # (N, 2)

# 2. Frenet basis @ F0 prediction (E0 default: Frenet frame)
t_hat, n_hat = rc.build_frenet_basis_at_prediction(train_x, end_idx=base.END_IDX)

# 3. Ring offsets + world-frame candidates
ring = rc.compute_ring_offsets(radii_cm=[0.0, 0.5, 1.0],
                                angles=[0.0, np.pi/2, np.pi, 3*np.pi/2])
cand_world = rc.frenet_offsets_to_world(ring, t_hat, n_hat, anchor_pos=f0_pred)

# 4. Encoder input (seq) + candidate features
seq = base.make_seq_features(train_x, base.END_IDX)                              # (N, 6, 9)
cand_feat = rc.make_ring_candidate_features(cand_world, f0_pred, t_hat, n_hat, ring)  # (N, 9, 10)

# 5. Train (fold=0, ~5min on cuda:1)
model = rc.RingClassifierHead(hidden=64,
                              encoder_pretrained_path="runs/baseline/P001_pb-0-6822-fullrun/checkpoint_best.pt")
# fine-tune all (no freeze for baseline; E6 sub-exp B 에서 freeze)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
# loss = L7 combined (huber + hinge), inference = soft-mean τ=0.03
```

### §6.3 G1 합격

- fold=0 OOF soft hit ≥ **0.6450** (= plan-011 In/ID anchor)
- ★ `directional_commit_magnitude` ≥ 0.002 m (= 2mm, soft-mean 이 origin 으로 완전 회귀 안 함 sanity check)

위반 시 `baseline_below_anchor` warn (halt X), Phase 2 계속 진행. results.md 에 `paradigm_below_plan_011_evidence` 박제.

---

## §7. STAGE 2 (c5~c9, G2) — Phase 2 Core Ablation (4 axis)

### §7.1 anchor 위 1-axis swap matrix

각 sub-exp = anchor (E0) 에서 *지정 axis 1개만 변경*. fold=0 only.

#### E1 — Frame swap (c6, 2 sub-exp)

| sub-exp | 변경 | 직관 |
|---|---|---|
| P2.E1a | anchor (Frenet) | (E0 와 동일 — re-run for clean comparison) |
| **P2.E1b** | `t_hat=(1,0), n_hat=(0,1)` (world frame) | "Frenet 의 inductive bias 가 진짜인가, 아니면 회전만 더한 거인가" |

ΔOOF(E1) = `OOF(E1b) − OOF(E1a)` — *Frame 의 가치 분리 측정*.

#### E2 — Ring density swap (c7, 3 sub-exp)

| sub-exp | radii_cm | K | 직관 |
|---|---|---|---|
| P2.E2a | [0.0, 0.5] | 5 | sparse — directional resolution 4 방향 충분? |
| **P2.E2b** | [0.0, 0.5, 1.0] (anchor) | 9 | E0 |
| P2.E2c | [0.0, 0.5, 1.0, 1.5] | 13 | dense — soft-mean 평균 효과 향상 vs label noise 1cm scale 의 dilute trade-off |

ΔOOF(E2) = `max(OOF(E2a), OOF(E2c)) − OOF(E2b)`.

#### E3 — Temperature scan (c8, 6 sub-exp)

| sub-exp | τ | 의미 |
|---|---|---|
| **P2.E3a** | 0.0 (argmax) | hard commit, soft-mean 가치 직접 측정 |
| P2.E3b | 0.01 | sharper than baseline |
| P2.E3c | 0.03 (anchor) | E0 |
| P2.E3d | 0.1 | slightly dull |
| P2.E3e | 0.3 | dull, dilution risk |
| P2.E3f | 1.0 | almost uniform — sanity check |

ΔOOF(E3) = `max(OOF over all τ ≠ 0.03) − OOF(E3c)`.

**dilution_collapse check** (★ §0.5 severe trigger): E3d/E3e/E3f 에서 `directional_commit_magnitude < 0.001 m` 시 sub-exp 단독 warn (= soft-mean 이 origin 으로 완전 회귀).

#### E4 — Loss swap (c9, 2 sub-exp)

| sub-exp | loss | 직관 |
|---|---|---|
| **P2.E4a** | L7 hit-aware hinge (anchor) | E0 |
| P2.E4b | distance regression (`huber_loss(soft_mean_pos, target)` only, hinge 제거) | metric alignment 의 effect 분리 — "1cm 경계 cliff 의 가치" |

ΔOOF(E4) = `OOF(E4b) − OOF(E4a)` — *negative* 이면 L7 의 hit-aware 가 결정적.

### §7.2 G2 합격

- 4 axis 모두 informational 완료 (fail 없음)
- 4 axis 중 **최소 1 axis** 에서 `max(ΔOOF over sub-exp in axis) ≥ 0.005`

위반 시 `phase2_no_positive_lever` severe. autonomous recovery:
- option (a) Phase 3 진행 후 G_final 에서 path-pivot (= "plan-012 paradigm 부정, plan-013 후보 새 paradigm")
- option (b) best Phase 1 baseline (E0) 으로 G4 직진 (= "Phase 2 lever 없어도 E0 자체의 5-fold submission 박제")

> decision-note: spec-default — option (a) 우선 (informational 가치 높음). budget 부족 시 option (b).

---

## §8. STAGE 3 (c10~c13, G3) — Phase 3 Aux Ablation (3 axis)

### §8.1 sub-exp matrix

#### E5 — Boundary sample weighting (c11, 2 sub-exp)

| sub-exp | boundary weight | 직관 |
|---|---|---|
| **P3.E5a** | uniform (anchor) | E0 |
| P3.E5b | `1cm 근처 sample 에 ×3` (= plan-004 boundary corrector 의 idea 차용; weight = 3 if `0.005 < ‖F0_pred − true‖ < 0.015` else 1) | 20% 회수 대상 sample 집중 |

ΔOOF(E5) = `OOF(E5b) − OOF(E5a)`.

#### E6 — Scorer architecture (c12, 2 sub-exp, ★ 시계열 input 가치 측정)

| sub-exp | scorer | 직관 |
|---|---|---|
| **P3.E6a** | full `CandidateAttentionGRUSelector` (anchor — GRU over 6-step seq) | E0 |
| P3.E6b | last-step MLP (= `make_seq_features(train_x, end_idx)[:, -1, :]` 만 사용, GRU 우회) | "6-step kinematic context 가 directional commit 에 진짜 필요한가" |

ΔOOF(E6) = `OOF(E6b) − OOF(E6a)`.

★ **frozen_gru_drift check**: 두 sub-exp 모두 plan-004 pretrained GRU encoder 으로 init. P3.E6a 는 *full fine-tune*, P3.E6b 는 GRU 우회 (last-step only) — encoder weight 자체 diff 없음 (= severity check 불필요). 만약 *frozen GRU encoder + trainable head* 변형 추가 시 weight diff > 0 시 severe.

#### E7 — r=0 logit prior 강도 (c13, 3 sub-exp)

| sub-exp | prior on r=0 logit | 직관 |
|---|---|---|
| **P3.E7a** | +0.0 (anchor) | no prior |
| P3.E7b | +0.5 | mild — center commit 유리 |
| P3.E7c | +1.0 | strong — 64% 안전 sample 보호 강화, 20% 회수 손실 위험 |

ΔOOF(E7) = `max(OOF(E7b), OOF(E7c)) − OOF(E7a)`.

### §8.2 G3 합격

- 3 axis 모두 informational 완료
- positive lever 필수 아님 (informational only)

---

## §9. STAGE 4 (c14, G4) — Phase 4 Best Stack 5-fold + Submission

### §9.1 best stack 선정

- Phase 2 best axis = `argmax(ΔOOF over E1, E2, E3, E4)` (axis-level)
- Phase 3 best axis = `argmax(ΔOOF over E5, E6, E7)` (axis-level; ΔOOF < 0 인 axis 는 best 후보 X — 즉 anchor 유지)
- best stack = anchor (E0) + best Phase 2 lever + best Phase 3 lever (additive 가정; super-additive 검증은 G4 합격 기준에서)

### §9.2 5-fold + submission

```python
# analysis/plan-012/phase4_final.py

# 5-fold concat OOF + test inference + submission.csv
for fold in range(5):
    model = rc.RingClassifierHead(hidden=64, encoder_pretrained_path=...)
    # apply best stack levers
    train_subset = train_x[fold_id != fold]
    val_subset   = train_x[fold_id == fold]
    train(model, train_subset, ...)
    oof_preds[fold_id == fold] = predict(model, val_subset)

oof_soft_hit_5fold = compute_soft_hit(oof_preds, train_y, R_HIT=0.01)

# test inference (5-fold ensemble: mean of soft-mean predictions)
test_preds_ensemble = mean over folds of predict(model_fold, test_x)
write_submission_csv(test_preds_ensemble, sample_ids_test, "submission.csv")
```

### §9.3 G4 합격

- `5-fold concat OOF soft hit ≥ G1 + 0.005` (super-additive 검증)
- `submission.csv` shape == `data/sample_submission.csv` shape
- 모든 좌표 finite

위반 시 `final_no_additive` warn — autonomous fallback = best Phase 2 single-axis sub-exp 의 5-fold 으로 fallback submission 박제.

---

## §10. STAGE 5 (c15, G_final) — Synthesis + plan-013 후보

### §10.1 산출물

- `analysis/plan-012/results.md` — 본 plan 의 모든 G-gate 결과 요약
- `analysis/plan-012/next_plan_candidates.md` — plan-013 후보 ≥ 3개
- 3 파일 frontmatter sync:
  - `plans/plan-012-frenet-ring-classification.md` (`status: G_final_complete`, `lb_score: null`)
  - `plans/plan-012-frenet-ring-classification.results.md` (신규 작성)
  - `README.md` 또는 registry index (있을 경우, 본 plan 의 status row 추가)
- best Phase submission 박제: `runs/baseline/H028_phase4-final-5fold/submission.csv` (또는 fallback 시 best Phase 2 5-fold)

### §10.2 plan-013 후보 (최소 3개)

조건부 후보 framework:

| 조건 | plan-013 후보 |
|---|---|
| G4 best stack OOF ≥ 0.70 | (1) ring radius CMA-ES tuning (plan-007 paradigm 차용) (2) ring angle resolution 증가 (4→8 angles) (3) per-sample anchor (F0 외 다른 공식 sample-wise selection) |
| G4 0.65 ≤ OOF < 0.70 | (1) MoE over multiple formulas (F0 + F1 + ...) (2) test-time augmentation (rotation 4) (3) scorer architecture 강화 (Transformer) |
| G4 OOF < 0.65 | (1) paradigm 재폐기 — plan-013 = KNN / GP / Diffusion (plan-011 §2.2 의 carry-over 후보) (2) ring 후보 + corrector hybrid (3) 다른 single formula 기반 reframe |

★ **plan-012.1 carry-over instruction**: best Phase submission `.csv` 의 LB 수동 제출 (`dacon-submit` skill) + 결과를 plan-012 frontmatter `lb_score` 에 후속 박제 + plan-013 분기 결정.

---

## §11. 참조

- `WORKFLOW.md` §1~§12 (plan/results/registry 규약 + autonomous protocol)
- `CLAUDE.md` (Autonomous Execution Policy + Commit·Push 의무)
- `plans/plan-004-pb-0-6822-fullrun.md` (selector arch + make_seq_features anchor)
- `plans/plan-006-minimal-variant-e-lb.md` (single formula F0 = frenet_par120_perp_neg020 + LB 0.6692)
- `plans/plan-007-formula-tuning.md` (per_candidate_hit anchor)
- `plans/plan-011-single-formula-corrector-exploration.md` (★ residual regression path 의 4-axis 결과 + In/ID +0.0050 evidence + L7 hinge loss 정의)
- `analysis/plan-005/corrector_decomp.{md,json}` (★ destructive band [0.5, 1cm) -7.83pp evidence)
- `analysis/plan-007/per_candidate_hit.{md,json}` (★ raw single formula ranking)
- `notes/PB_0.6822 코드공유.ipynb` cell 4 (`CandidateAttentionGRUSelector` 원본 구현)

---

## §12. Plan 자기-완결 확인 (WORKFLOW.md §9 unbreakable #3)

본 plan 은 외부 채팅 로그 / 메모리 / 구두 합의 없이 단독으로 재구성 가능. 핵심 정의:
- F0 (single formula): plan-006 §5.5 + CANDIDATES[17] (= `frenet_par120_perp_neg020`)
- Ring geometry: §4.1 컴포넌트 1 (`compute_ring_offsets` 의 invariants)
- Frenet basis: §4.1 컴포넌트 2 (`build_frenet_basis_at_prediction`)
- L7 loss: §4.1 컴포넌트 5 (`l7_combined_loss`)
- Scorer encoder: plan-004 selector.py L697-720 (`CandidateAttentionGRUSelector`)
- make_seq_features: plan-004 selector.py L406-449

모든 G-gate 합격 기준은 §0.5 단독 참조로 판정 가능 (autonomous loop 매 turn 의 §0.5-only read 가능).

> *misc note*: plan-011 §1.5 의 "corrector input snapshot 한계" 가 본 plan 의 *간접 motivation* — corrector path 자체를 폐기하고 시계열 input 을 *원래 위치* (selector encoder) 에서 받는 것이 자연스러운 해결.
