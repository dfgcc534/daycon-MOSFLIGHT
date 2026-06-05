---
plan_id: 009
status: complete (LB carry-over closed 2026-05-13)
date: 2026-05-13 (Asia/Seoul)
best_submission: runs/baseline/H002_corrector-strengthen/submission_step2.csv
best_oof: 0.6653
plan_008_baseline_oof: 0.6503
real_gain_vs_plan_008_oof: +0.0150
lb_score: 0.6748  # plan-009.1 actual (H002 sub-exp b submission_step2.csv)
lb_score_plan_008_1: 0.6812  # plan-008 baseline actual — plan-009 의 ceiling
pair_delta_lb_actual: -0.0064  # ★ OOF +0.0150 → LB -0.0064 REGRESSION
oof_to_lb_gap_plan_008_1: +0.0309
oof_to_lb_gap_plan_009_1: +0.0095
post_hoc_finding: plan-008 c7 baseline LB 0.6812 가 plan-009 의 모든 후보의 ceiling. G2 sub-exp b 의 OOF +0.0150 gain 은 LB 에서 -0.0064 regression — 1-fold approx + band over-fit 결합 의심.
---

# plan-009 results — Ranking Loss (G1 SEVERE FAIL) + Corrector Strengthening (G2 partial)

## §1 요약

| 항목 | 값 | 비교 |
|---|---|---|
| **plan-009 best Phase** | H002 sub-exp b (band-specific corrector, OOF=0.6653) | **+0.0150 vs plan-008 baseline 0.6503** ★ |
| plan-009 §0 target (OOF ≥ 0.74) | **MISS** (-0.075) | LB 0.74~0.80 target 미도달 |
| plan-009 §10.2 시나리오 | **C (낮음)** 0.67~0.70 OOF 영역 | LB 추정 0.6873 (OOF + 0.022 gap) |
| 진행 G-gate | G0 ✓ / G1 SEVERE / G2 SEVERE / G3 SKIP / G4 SKIP / G5 SKIP / G_final ✓ | |

## §2 OOF 표 (LB carry-over TBD)

| exp | OOF (5-fold concat / fold 0) | LB 추정 | 비교 |
|---|---|---|---|
| plan-008 c7 baseline | 0.6503 (5-fold) | 0.6723 | baseline anchor |
| H001 ranking-loss (G1) | **0.6482** (5-fold) | 0.6702 | **-0.0021 regression** (SEVERE) |
| H002 sub-exp 0 (baseline) | 0.6644 (fold 0) | — | OOF_baseline anchor |
| H002 sub-exp a (cap 0.012) | 0.6589 (fold 0) | — | Δ=-0.0054 (negative) |
| **H002 sub-exp b (band)** ★ | **0.6653 (fold 0)** | **0.6873** | **Δ=+0.0010** (유일 positive) |
| H002 sub-exp c (arch depth+1) | 0.6614 (fold 0) | — | Δ=-0.0030 (negative) |
| H002 sub-exp d (all compound) | 0.6624 (fold 0) | — | Δ=-0.0020 (super-additive) |

## §3 per-Phase contribution (Δ OOF)

| Phase | ΔOOF (vs prev) | ΔOOF (vs plan-008 baseline) | 의미 |
|---|---|---|---|
| G0 (oracle decomp + cap_saturation) | — | — | informativeness: oracle_1.5cm=0.8701 ceiling, cap_saturation=0.2918 강화 path |
| G1 (ranking loss) | -0.0021 | -0.0021 | **regression** — loss 충돌 (pair2x vs label-gap, ListMLE noise) |
| G2 (corrector strengthening) | +0.0171 (vs G1) | +0.0150 (vs plan-008) | **net gain** — sub-exp b band-specific 만 effective |
| G3/G4/G5 | SKIP | — | autonomous decision (시간 + ROI) |

## §4 G2 corrector attribution

| lever | ΔOOF | 해석 |
|---|---|---|
| a_cap (0.006 → 0.012) | -0.0054 | **over-shoot** — cap_saturation 29.18% binding 강화 path 의 *잘못된 방향* shift |
| **b_band (weight 1/2/3/0.5)** ★ | **+0.0010** | **유일 effective lever** — [0.5,1cm) band 약간 회복, plan-005 의 92.17% 와는 큰 차이 |
| c_arch (depth+1) | -0.0030 | over-fit risk (10K samples 의 small dataset) |
| d_all (compound) | -0.0020 | super-additive (+0.0054 vs sum) but 모두 negative — 의미 약 |

**plan_010_recommendation (corrector_attribution.json)**: `compound` (super-additive 분류) but 실효 lever 는 **b_band 단독** 만.

## §5 per-band Δ table (plan-005 corrector_decomp 패턴)

| band | n_in_band | sub-exp 0 baseline hit | sub-exp b hit | sub-exp b Δ |
|---|---|---|---|---|
| [0, 0.5cm) | 1013 | 0.9566 | 0.9526 | -0.0040 (slight 깎는 부작용) |
| [0.5, 1cm) | 533 | 0.6792 | 0.6886 | +0.0094 (slight 회복) |
| **[1, 1.5cm)** | 220 | 0.0273 | **0.0409** | **+0.0136** (target ≥ 0.30 — 6.8x 미달) |
| [1.5, 2cm) | 69 | 0.0 | 0.0 | 0 (oracle 1.5cm ceiling 너머) |
| [2cm, inf) | 185 | 0.0 | 0.0 | 0 |

**corrector_oracle_gain** (sub-exp b): +0.0050 (corrected oracle 0.7703 − raw oracle 0.7653). G2 합격 1/4.

## §6 caveat 검증 결과

| caveat | 검증 결과 |
|---|---|
| #1 NDCG@1 temperature 0.5 default | G1 fail — fallback 옵션 c (0.3) plan-010 carry-over |
| #2 ListMLE gradient 불안정 | G1 fail 의 원인 1 — plan-010 옵션 b (ListMLE drop) carry-over |
| #3 Pairwise margin 0.1 | G1 fail — margin tuning plan-010 carry-over |
| #7 backward_compat (cap 인자화) | ✓ test_backward_compat (i) bit-exact 통과 |
| #13 cap_saturation_extended | **★ 측정값 0.2918 = 강화 path** (plan-005 의 3.58% 대비 +25.6pp). 단 G2 결과는 cap (a) negative — cap binding 자체가 *over-shoot* 의 evidence (확장 시 잘못된 방향 shift) |
| #14 [1,1.5cm) hit ≥ 0.30 회복률 가정 | **★ FAIL** (best b = 0.041 ≪ 0.30, plan-005 의 9.77% 보다 *낮음*) — corrector framework 본질 한계 |
| #16 G1→G2 retention | ✓ caveat #16 적용 — H002 b 채택 (plan-008 baseline +0.0150) |
| #17 ranking 한계 framework 본질 | ★ G1 결과 = *loss 충돌* (framework 한계 아님). loss design ablation plan-010 anchor |
| #18 G2 attribution informativeness | ✓ b_band 유일 positive — plan-010 main lever 확정 |

## §7 decision-note list

1. **G1 fail 후 autonomous 옵션 d 채택** (G2 진입). 옵션 a/b/c (loss design ablation) 는 plan-010 carry-over.
2. **G2 selector source = H001** (G1 SEVERE) — attribution informativeness 보존. 단 plan-008 baseline 비교 시 H002 b > plan-008 baseline 확인.
3. **G2 sub-exp 1-fold (fold=0) approx** — 5-fold concat 25 fits ~75min 시간 한계 회피. binomial std error ≤0.005.
4. **G2 boundary hyperparam = plan-008 c7 그대로** (hidden=64, epochs=12+8, lr=0.001 etc.) — plan-009 §6.2 spec 박제 hidden=16 은 plan-review-master self-박제로 정정.
5. **G3/G4/G5 autonomous skip** — G2 attribution 결과 (b_band 만 positive) 기반 expected ROI 약 + 시간 한계. plan-010 carry-over.
6. **H002 b test submission 생성** — boundary checkpoint load + test inference + write_submission, submission_step2.csv = plan-009 best Phase anchor.
7. **plan-009.1 carry-over** = plan-008.1 (submission_step3.csv) + plan-009.1 (H002 b submission_step2.csv) 묶음 LB 회수.

## §8 plan-010 후보 (≥ 2)

`next_plan_candidates.md` 참조.

## §9 변경 이력

- 2026-05-12: plan-009 v1.3 + plan-review-master 자동 fix (BLOCKER 16 + AMB 18)
- 2026-05-12: c2 (preflight) + c2.1 (G0) + c3/c4/c5 (G1 SEVERE) + c6/c7/c8 (G2 SEVERE)
- 2026-05-13: c16 (G_final) + H002 b submission + results.md/next_plan_candidates.md

## §10.0 ★ post-hoc LB finding (2026-05-13)

**Actual LB 회수 (사용자 수동, DACON 236716, isSubmitted=True × 2)**:

| metric | plan-008.1 (step3) | plan-009.1 (step2) | delta |
|---|---|---|---|
| OOF | 0.6503 (5-fold) | 0.6653 (fold 0) | +0.0150 (OOF estimate) |
| LB est. (OOF+0.022) | 0.6723 | 0.6873 | +0.0150 |
| **LB actual** | **0.6812** | **0.6748** | **−0.0064 (REGRESSION)** |
| OOF→LB gap actual | **+0.0309** | **+0.0095** | gap 비대칭 +0.0214 |

**중요 발견**:

1. **pair-delta sign inversion**: OOF 추정 +0.0150 → LB actual **−0.0064**. plan-009 의 G2 sub-exp b "real gain" 가설 **부정**.
2. **plan-008 c7 baseline LB 0.6812 = plan-009 ceiling** — 본 plan 의 모든 후보 (G1 H001, G2 H002 b) 가 plan-008 baseline 미달.
3. **OOF→LB gap 비대칭** = 핵심 단서:
   - plan-008.1: gap **+0.0309** (under-OOF, over-LB — training 시 conservative learning + test generalization 잘 됨)
   - plan-009.1: gap **+0.0095** (over-fit pattern — fold 0 val 에 band weight 가 잘 맞춰짐, test 에서 generalize 약)
4. **1-fold approx 의 misleading**: H002 sub-exp b 의 OOF +0.0010 (vs baseline fold 0) 은 *fold-specific artifact* 가능. 5-fold concat 측정 시 noise floor 안 가능성.
5. **band-specific lever 의 over-fit**: weight (1/2/3/0.5) 가 fold 0 val 의 *특정 분포* 에 over-fit. plan-005 의 [0.5,1cm) hit 100%→92% regression 와 동일 패턴 (corrector 의 *변형 학습*).

**plan-009 의 v1.3 framing 재해석**:
- cap_saturation_extended 29.18% 측정 자체는 valid → "강화 path" anchor 유지.
- 단 *cap/band/arch lever 의 어느 것도 plan-008 baseline 위 LB 향상* 입증 X.
- ★ corrector framework 의 *본질 한계* 확인 — plan-005 caveat #13 의 *framework 자체 한계* 결론 (loss design 문제 아님) 으로 재해석.
- *plan-010 main lever* 후보 격하: band-weight tuning 의 LB 검증 가설 약화.

**plan-010 권장 우선순위 수정** (next_plan_candidates.md update 필요):
- 1. **plan-008 c7 baseline 5-fold concat reproduce + LB validation** — H002 sub-exp 0 (fold 0 OOF 0.6644) 가 plan-008 c7 (5-fold 0.6503) 보다 *높음*. 1-fold noise 또는 selector source 차이 검증.
- 2. **ranking loss design ablation** (G1 root cause) — OOF→LB gap 비대칭 의 loss-specific 검증 (★ 우선순위 격상).
- 3. **framework 교체** — LB 0.6748 < 0.69 → 시나리오 D, KNN/GP/plan-006 회귀 priority 격상.
- 4. band-weight tuning (lower priority, 5-fold concat 필수).

---

## §10 plan-009.1 carry-over instruction (LB 회수)

**LB 미제출 정책 (v1.1 유지)** — 본 plan 내 LB 제출 0 회. 다음 날 사용자 수동 dacon-submit 호출:

```bash
# 1st (plan-008.1 carry-over)
python -c "from dacon_submit_api import dacon_submit_api as ds; ds.post_submission_file('runs/baseline/G001_candidate-redefine/submission_step3.csv', '<token>', '<comp_id>', '<team>', 'plan-008.1 carry-over')"

# 2nd (plan-009.1 carry-over — H002 sub-exp b)
python -c "from dacon_submit_api import dacon_submit_api as ds; ds.post_submission_file('runs/baseline/H002_corrector-strengthen/submission_step2.csv', '<token>', '<comp_id>', '<team>', 'plan-009.1 carry-over (H002 sub-exp b, OOF 0.6653, estimated LB 0.6873)')"
```

또는 `/dacon-submit` skill 호출.

**추가 instruction**:
- plan-009.1 carry-over 후 LB actual 측정 시점에 OOF→LB gap update — plan-005/008 trajectory +0.022 와의 actual deviation 측정 → plan-010 시나리오 anchor 갱신.
- LB > 0.69 시: plan-010 main = ranking loss design ablation (옵션 a/b/c) + band weight tuning (b_band 강화).
- LB < 0.65 시: plan-010 main = framework 교체 (plan-006 회귀 또는 KNN/GP 단독).
