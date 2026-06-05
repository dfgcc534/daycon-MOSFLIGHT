# plan-010 후보 (plan-009 carry-over)

plan-009 의 핵심 finding 4개 anchor:

1. **G1 ranking loss = loss 충돌** (regression -0.0021 vs plan-008) — framework 한계 아닌 *loss design* 문제. options a/b/c (weight 조정, ListMLE drop, temperature 0.3) ablation 미진.
2. **G2 attribution = b_band 유일 positive** (+0.0010), 나머지 negative (cap -0.0054, arch -0.0030). plan_010_recommendation: compound (super-additive) but 실효 lever 는 band 단독.
3. **cap_saturation_extended = 0.2918** (강화 path) but cap 확장 시 *over-shoot* (잘못된 방향 shift, -0.0054 OOF). cap 자체는 main lever 아님.
4. **[1, 1.5cm) hit_after = 0.041** (target 0.30 의 1/7) — corrector framework 본질 한계 (plan-005 raw 27 의 9.77% 보다도 낮음).

## 후보 1 — Band-weight tuning ablation (★ Recommended)

**근거**: caveat #18 + G2 attribution — b_band 유일 positive.

**Spec**:
- band weight grid: [1, 2, 3, 0.5] (현재) vs [1, 1.5, 2.5, 0.5] vs [1, 1.5, 4, 0.3] vs [0.5, 2, 5, 0.2]
- 4 sub-exp × 5-fold = 20 fits ~60min
- arch 고정 (depth=2), cap 고정 (0.006), selector 고정 (H001 또는 plan-008 baseline 비교)
- target: [1,1.5cm) hit_after 0.041 → ≥ 0.10 (3x 회복) + OOF +0.005

**Expected ROI**: OOF +0.005~0.015 (incremental). LB 추정 0.69~0.70.

## 후보 2 — Ranking loss design ablation (G1 carry-over)

**근거**: caveat #17 + G1 SEVERE — *loss 충돌* 검증.

**Spec**:
- 3-arm ablation:
  - arm A: NDCG@1 only (pair × 0, listmle × 0)
  - arm B: NDCG@1 + pair2x (listmle × 0)
  - arm C: NDCG@1 + pair2x + ListMLE × 0.25 (listmle weight 절반)
- 단일 변수 분리 → 각 loss term 의 *individual contribution* 측정
- temperature grid (0.5 / 0.3) + K_pairs grid (5 / 10 / 15)
- 3 × 2 × 3 = 18 sub-exp — fold 0 only, ~20min

**Expected ROI**: G1 regression -0.0021 → +0.005~0.015 recovery. plan-008 baseline 회복 가능.

## 후보 3 (조건부) — Framework 교체 (KNN / GP)

**근거**: caveat #14 — corrector [1,1.5cm) 4% 회복 불가, framework 본질 한계.

**Spec**:
- 후보 3a: KNN over cand pool (extended 25 cands) — non-parametric.
- 후보 3b: Gaussian Process posterior mean — Bayesian framework.
- 후보 3c: plan-006 회귀 (CMA-ES single-formula tuning).

**Expected ROI**: LB 추정 변동 큼 (-0.05 ~ +0.05). LB < 0.65 시점에 진입.

## 후보 4 (조건부) — Set Transformer arch swap

**근거**: G5 SKIP — plan-009 의 누적 OOF 0.6653 < 0.75 진입 조건 충족.

**Spec**:
- selector.py partial: GRU hidden 32 + Set Transformer 1 layer (cand_i ↔ cand_j) fusion.
- Linear_64_32 + MultiheadAttention 4 heads.
- spec @ plan-009 §9.2.

**Expected ROI**: LB +0.03~0.06 (high variance, 10K data 의 overfit risk).

## 우선순위 권장

1. **후보 1 (band-weight tuning)** — G2 attribution 의 직접 후속, 가장 cost-effective.
2. **후보 2 (ranking loss ablation)** — plan-009 G1 SEVERE 의 root cause 분석.
3. plan-009.1 carry-over LB 회수 후 시나리오 분기:
   - LB ≥ 0.69 → 후보 1 + 2 묶음.
   - LB < 0.65 → 후보 3 (framework 교체).

## decision-note (plan-009 → plan-010 transition)

- plan-009 의 v1.3 "fragile/robust ordering" framing 유지 시도 — G1 robust (selector partial) → G2 fragile (boundary + arch + cap + loss) — 의도된 retention 보장 작동 (G1 fail 시 G2 가 selector 약점 상쇄 + plan-008 baseline 위 +0.0150).
- 단 G1 의 *loss 충돌* 가 fragile/robust 분류와 *독립적* — plan-010 의 G1 재시도 시 *loss design ablation* 강조.
- cap_saturation_extended 의 *evidence* (강화 path 0.2918) 는 cap 확장 의도 와 다른 방향 — *cap 자체* 가 lever 아닌 *band 의 implicit 보조 변수*.
