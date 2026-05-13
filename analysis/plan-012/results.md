---
plan_id: 012
plan_title: Codebook Bake-off Classification + Regression Hybrid (paradigm reframe, 3D)
status: G_final_complete (warn-recovered)
date_completed: 2026-05-13 (Asia/Seoul)
exp_ids:
  - H019_phase0-preflight-codebook        # G0 PASS
  - H020_phase1-codebook-bakeoff          # G1 warn (winner_oof < anchor)
  - H022_phase2-codebook-K                # G2 severe-recovered
  - H023_phase2-temperature
  - H024_phase2-loss
  - H025_phase2-reg-head
  - H026_phase3-boundary-weight           # G3 informational
  - H027_phase3-scorer-arch
  - H028_phase3-r0-prior
  - H029_phase4-final-5fold               # G4 warn (final_no_additive, fallback)
final_submission: runs/baseline/H029_phase4-final-5fold/submission_anchor_fallback.csv
final_oof_5fold_hit_1cm: 0.6340 (best stack) / 0.6339 (anchor, fallback)
lb_score: null  # plan-012.1 carry-over manual submit
---

# plan-012 v2 — Codebook Bake-off Hybrid (paradigm reframe) — 결과 종합

## 한 줄 결론

> **paradigm reframe (3-way codebook bake-off + classifier+regression hybrid) 은 F0 raw hit 위 +0.002 만 추가 — paradigm 자체 limit.** 5-fold OOF 0.6340 < plan §0 target 0.66 (-0.026 미달). mean-regression trap 완전 회피 못함 (DCM ~0.4mm). plan-013 path-pivot 필요 (corrector + hybrid 합체 또는 paradigm 완전 폐기).

## G-gate sequence 요약

| G | spec 합격선 | 측정 | 상태 |
|---|---|---|---|
| G0 | preflight 6 essential checks | F0 hit@1cm=0.6320 / oracles 0.74~0.78 / kmeans min cluster 113 | PASS |
| G1 | winner_oof ≥ 0.6450 + DCM ≥ 0.002 | E0a winner OOF=0.6416 / DCM=0.00037 | warn (baseline_below_anchor + dilution G1 sanity) |
| G2 | 5 axis 중 1+ axis ΔOOF ≥ 0.005 | max ΔOOF = +0.0015 (E3 τ=0.01) | **severe-recovered** (phase2_no_positive_lever, option a) |
| G3 | informational only | max ΔOOF = +0.0020 (E8 r=0 +0.5) | PASS |
| G4 | best_stack ≥ anchor_5fold + 0.005 | Δ = +0.0001 | warn (final_no_additive, fallback) |
| G_final | synthesis + plan-013 후보 + 3 파일 sync | 본 commit | PASS (warn-recovered) |

## Phase 별 핵심 측정

### Phase 0 (G0) — preflight + codebook prep
- F0 (`frenet_par120_perp_neg020` = plan-006 CANDIDATES[17]) raw hit@1cm = **0.6320**, hit@1.5cm = 0.8033 (plan §1.3 "64% / 84%" hypothesis 일치)
- Oracle ceilings (codebook 위 *label-aware* argmin 가능 시):
  - Absolute-7Way: **0.7761**
  - Frenet-Orthogonal-7Way: **0.7797** ★ 최대
  - K-Means-7Way (fold-aware): 0.7436
- Per-axis marginal oracle (E2 dominant axis source):
  - Absolute family ranking: `x > y > z` (tie-break, gap 0.0019<0.003 → priority x)
  - Frenet family ranking: `n > b > t` (strict, gap(n,b)=0.0038)
  - ★ Frenet n-axis dominant — plan-005 §1.2.2 의 binormal evidence 와 다소 다른 양상

### Phase 1 (G1) — 3-Way bake-off
| sub-exp | val_hit@1cm | DCM (m) | 해석 |
|---|---|---|---|
| **E0a** (Absolute) | **0.6416** | 0.00037 ★ | winner (tie-break with E0c, priority E0a) |
| E0b (Frenet-ortho) | 0.6292 | 0.00236 | 가장 큰 DCM 이나 OOF 낮음 |
| E0c (K-Means) | 0.6416 | 0.00079 | tied with E0a |

★ **3 codebook 모두 hit@1cm 0.63 plateau** — paradigm 의 lever 가 weak. DCM 모두 < 1mm — soft blend (τ=0.03) + hard-label CE 가 classifier 를 mode 0 (center) 으로 끌어당기는 mean-regression trap 의 hybrid 변종.

### Phase 2 (G2, severe-recovered) — Core ablation on E0a (10 sub-exp, 5 axis)

| axis | max ΔOOF | best sub-exp | 해석 |
|---|---|---|---|
| E1 frame | — | SKIP (E0a → frame_axis_n/a) | — |
| E2 K density | +0.0010 | K=13 | K-density swap 의 marginal 효과 |
| **E3 τ scan** | **+0.0015** ★ | **τ=0.01** | sharper softmax 약간 도움 |
| E4 loss swap | +0.0005 | no_hinge | hinge 의 영향 미미 |
| E5 reg head | -0.0030 | (reg_off worse) | reg head informational 으로 약간 유용 |

★ 5 axis 모두 ΔOOF < 0.005 → `phase2_no_positive_lever` severe. **autonomous recovery option (a) 채택**: Phase 3 informational 진행 후 G_final path-pivot 결정.

### Phase 3 (G3, informational) — Aux ablation (4 sub-exp, 3 axis)

| axis | max ΔOOF | best sub-exp | 해석 |
|---|---|---|---|
| E6 boundary weight | +0.0000 | bweight_on | 1cm 근처 ×3 weight — no effect |
| E7 scorer arch | -0.0025 | (mlp worse) | GRU 약간 useful, ★ 시계열 input 가치 |
| **E8 r=0 prior** | **+0.0020** ★ | r=0 +0.5/+1.0 | center anchor 보호 → safe sample 회피, F0 raw direction 강화 |

overall best lever (Phase 2 + 3): **E8 r=0 +0.5 (+0.0020)** > E3 τ=0.01 (+0.0015) > all others.

### Phase 4 (G4, warn) — Best stack 5-fold

| config | 5-fold concat OOF hit@1cm |
|---|---|
| anchor (E0a only) | **0.6339** (= G1 winner_5fold_oof baseline) |
| best stack (E0a + τ=0.01 + r=0 +0.5) | **0.6340** |
| Δ | **+0.0001** (★ << +0.005 threshold) |

→ `final_no_additive` warn. fallback submission = `submission_anchor_fallback.csv`.

★ **paradigm-level evidence**: fold-0 (E0a 0.6416) vs 5-fold (0.6339) gap = 0.0077 — fold-별 variance 가 lever 효과 (+0.002) 매몰. additive 가정 5-fold scale 에서 무효화.

## 측정의 의미 — paradigm 의 limit

- **F0 raw 가 hit@1cm = 0.6320 으로 이미 plateau 근접**. classifier + regression hybrid 의 net 추가 = +0.002 hit rate (5-fold).
- DCM (directional commit magnitude) = 모든 sub-exp 에서 < 1mm (anchor radius 5mm 의 1/13~1/5). hybrid 가 substantive shift 못 함 — *soft* blend (τ=0.03) + hard-label CE 가 classifier 를 mode 0 (center) 으로 끌어당김.
- mean-regression trap 회피 의도 (§1.2.1) 가 *δ scale 회피* 로 해석되었으나, classifier 의 hard-label CE 자체가 mode 0 으로 수렴해 *direction commit 부재* 의 새 트랩 등장.
- plan-011 의 corrector path (best In/ID +0.0050) 가 정량적으로 본 plan 의 best stack lever (+0.0001) 보다 컸음 — **paradigm shift 가 plan-005~011 의 corrector path 보다 약했음**.
- plan §0 의 target 5-fold OOF ≥ 0.66 (= +0.011 above plan-006 corrected 0.6491) 와 비교 시 -0.026 미달.

## 자율 결정 박제 (decision-note 모음, audit 용)

| commit | decision-note |
|---|---|
| 9a89795 (c3) | spec-default — F0 산식 박제 (plan §1.4.1) 가 plan-006 CANDIDATES[17] 와 mismatch → 실제 식 (corrector_redesign_v2 parity) 으로 ring_classifier.py 수정 |
| 9a89795 (c3) | spec-default — F001_variant-e checkpoint 부재 → plan_006_reproduce skipped, informational |
| 9a89795 (c3) | spec-default — trajectory T=11 vs plan 박제 T=7 mismatch → end_idx=T-1=10 자율 진행 |
| fc74e58 (c4) | spec-default — GPU 부재 (CPU only), epoch 15 batch 512 patience 3 으로 spec default 축소. 3 sub-exp 동일 budget fair comparison |
| fc74e58 (c4) | spec-default — P001 .pt 부재 → GRU pretrained init skip, from-scratch init |
| fc74e58 (c4) | warn-handled — baseline_below_anchor + G1 sanity FAIL → §6.3 fallback Phase 2 informational |
| e6837df (c5~c10) | severe-recovered — phase2_no_positive_lever → option (a) Phase 3 informational + G_final path-pivot |
| 294148e (c11~c14) | spec-default — Phase 3 informational complete, E6 batch-level mean(weight) 근사 |
| d22e6a7 (c15) | warn-handled — G4 final_no_additive → fallback submission_anchor_fallback.csv |

## c2 hardening 박제 (plan-review-master + post-implementation spot-fix)

- 93db3fc: plan-review-master 5-iter spec hardening (12 BLOCKER + 7 AMBIGUITY)
- 9a89795: F0 산식 spot-fix in ring_classifier.py (plan-006 parity)
- 294148e: HybridScorerHead._extract_seq_hidden scorer arch dispatch (LastStepMLPScorer 호환)

추후 plan body spot-fix 필요 항목 (§12.6 blacklist 로 본 plan 안에서 미적용):
- plan §1.4.1 F0 산식 박제 → 실제 plan-006 식과 일치하도록 update
- plan §1.4.1 trajectory T=7 → T=11 update
- plan §4.1 base scorer attribute invariant 박제 (cand_proj/cand_attn/score_head → 실제 query/head)

## 산출물

- `analysis/plan-012/preflight.json` — G0 산출
- `analysis/plan-012/phase1_winner.json` — G1 winner 박제
- `analysis/plan-012/phase2_results.json` — G2 ablation
- `analysis/plan-012/phase3_results.json` — G3 ablation
- `analysis/plan-012/phase4_results.json` — G4 5-fold + submission
- `runs/baseline/H029_phase4-final-5fold/submission.csv` — best stack 5-fold ensemble
- `runs/baseline/H029_phase4-final-5fold/submission_anchor_fallback.csv` — ★ LB 제출 fallback (G4 warn 으로 선택됨)
- `analysis/plan-012/next_plan_candidates.md` — plan-013 후보 ≥ 3
- `src/pb_0_6822/ring_classifier.py` — 7 컴포넌트 모듈
- `src/pb_0_6822/ring_classifier_train.py` — run_sub_exp helper
- `tests/test_ring_classifier_smoke.py` — 19/19 pass

## plan-012.1 carry-over instruction

본 plan 내 LB 제출 = **0회** (할당량 carry-over pattern, plan-009.1+010.1+011.1 precedent).

- 사용자 manual 제출: `runs/baseline/H029_phase4-final-5fold/submission_anchor_fallback.csv` (G4 warn → fallback).
- 제출 후 `dacon-submit` skill 또는 manual upload, lb_score 후속 frontmatter sync.
- LB 결과에 따라 plan-013 분기 결정:
  - LB ≥ 0.65 (paradigm 가치 검증) → plan-013 = corrector+hybrid 합체 (Option C)
  - LB < 0.60 (paradigm regress) → plan-013 = paradigm 완전 폐기 (Option A)
  - 0.60 ≤ LB < 0.65 → plan-013 = F0 자체 교체 (Option B)
