---
plan_id: 027
version: 1.0
date: 2026-05-22 (Asia/Seoul)
status: draft
depends_on:
  - 025 ([DONE] 필수 — best variant 확정 후 본 plan 진입. plan-025 G2 band 가 negative 이면 본 plan §1.4 분기 decision A 적용 — combo lever 본질 무효, 본 plan 폐기 또는 plan-026 단독 위로 안내.)
  - 026 ([DONE] 필수 — best variant 확정 후 본 plan 진입. plan-026 G2 band 가 negative 이면 본 plan §1.4 분기 decision B 적용 — input enrichment 본질 무효, 본 plan 폐기 또는 plan-025 단독 위로 안내.)
based_on:
  - 024 (cross-attn anchor-vocab G2 FAIL hit_1cm 0.6370. 4축 root-cause: ① expansion gap → plan-025, ② static anchor → plan-026, ③ FE redundancy [carry caveat], ④ overfit [training schedule 별 lever]. 본 plan = ①과 ② 두 lever 의 orthogonal combo.)
  - 025 (Expansion mimic E0/E1/E2/E3 — root-cause #1 fix. best variant 의 loss formulation + grad signal unit = N×K=140k 가 본 plan 의 *loss/forward path* carry.)
  - 026 (Anchor-aligned input enrichment V0/V_γδ/V_αβ/V_all — root-cause #2 light workaround. best variant 의 cand_feat dim (174/177/193D) + Group α/β/γ/δ builder 가 본 plan 의 *cand_feat path* carry.)
  - 022 (14 BCC + τ=0.001 winner OOF 0.6528 — base layout carry, 변경 X)
  - 020 (5-fold stable_fold_id MD5)
inspired_by:
  - 사용자 명시 (turn 2026-05-22 session "plan-026"):
    "V5 = V4+ + pointwise expansion 이 가장 강한 single-shot 후보 — 이번 turn 의 입력 lever × plan-025 의 expansion lever 의 orthogonal combo"
  - plan-024 사후분석 §10 의 "plan-027 ensemble" 영역과 *별 정의* — 본 plan 의 plan-027 = *single-model combo* (one neural net with both levers), ensemble (LGBM × neural blend) 은 별 plan (가칭 plan-028 ensemble) 으로 분리.
code_reuse:
  - module: analysis/plan-025/model_pointwise.py | model_row_expansion.py | anchor_embed.py
    symbols: [PointwiseSelector | RowExpansionSelector | LearnableAnchorEmbed]
    reason: plan-025 best variant 의 *loss formulation + forward path* carry. W0 control + W2 combo 의 model backbone.
    가설 best variant: plan-025 결과 보고 c1.5 (post-026) 시점에 확정. 시점 placeholder = "E1 pointwise 또는 E2 row-expand" (E3 도 후보).
  - module: analysis/plan-026/cand_builder_v2.py | group_alpha.py | group_beta.py | group_gamma_delta.py
    symbols: [build_v2, build_group_alpha, build_group_beta, build_group_gd]
    reason: plan-026 best variant 의 *cand_feat builder* carry. W1 control + W2 combo 의 input path.
    가설 best variant: plan-026 결과 보고 c1.5 시점에 확정. 시점 placeholder = "V_all 193D 또는 V_αβ 177D".
  - module: analysis/plan-024/model.py
    symbols: [CandidateAttentionGRUSelector]
    reason: backbone carry (W1 control = plan-026 형식, plan-024 의 listwise CE loss). plan-025 best 가 pointwise/row-expand 면 별 backbone class.
  - module: analysis/plan-024/seq_builder.py
    symbols: [build_seq_feat]
    reason: seq_feat 95D plan-024 carry (단 사용자 catch — Group γ/δ 와 redundancy risk caveat 박제).
  - module: analysis/plan-022/anchors.py | selector_only_model.py
    symbols: [ANCHORS_A6, build_soft_label_with_tau]
    reason: 14 BCC + τ=0.001 carry.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline + paired Δ anchor.
  - module: src/io.py | src/pb_0_6822/selector.py
    symbols: [load_all_samples, load_labels, stable_fold_id]
    reason: data + fold carry.
followed_by:
  - (가칭 next plan): plan-028 (= 사후분석 §10 의 진짜 plan-027 영역) — ensemble (plan-022 LGBM + 본 plan W2 best + plan-025/026 best variant). 다른 inductive bias 의 blend.
  - (가칭): dynamic anchor (K-side fix) — 본 plan W2 가 < 0.6528 (band negative) 시 정당화. plan-022 paradigm 변경 (anchor 14 → 27+ sample-conditional).
scope: 단일 변수 = **두 직교 lever (loss/expansion + input/enrichment) 의 결합 곱**. 3 variant W0/W1/W2 head-to-head — W0=expansion only (plan-025 best), W1=input only (plan-026 best), W2=combo. anchor 14 BCC + τ=0.001 + hyperparam + fold split = 모두 carry. **plan-025/026 의 다른 variant (non-best) / hyperparam sweep / training schedule fix / ensemble (LGBM+neural blend) / corrector reg head / dynamic anchor / DACON LB submit / 16 lever FE 추가 / input aug σ=0.05 = out-of-scope**.
exp_ids:
  - Z027_W0_plan025_best_only
  - Z027_W1_plan026_best_only
  - Z027_W2_combo_orthogonal
# exp_id ↔ variant 매핑: frontmatter `Z027_W{n}_<short>` ↔ 본문 variant_key `W{n}_<short>`. best variant 확정 후 short 부분만 update.
lb_score: null
band: null
---

# plan-027 v1 — Expansion × Input Enrichment Orthogonal Combo

## §0. 한 줄 목적

> **plan-025 의 best variant (expansion lever, root-cause #1 fix) 와 plan-026 의 best variant (input enrichment lever, root-cause #2 light workaround) 의 직교 결합이 *additive* 또는 *super-additive* 한지 측정한다.** 3 variant head-to-head — W0=expansion only / W1=input only / W2=combo. **G3 PASS criterion = W2 hit_1cm ≥ max(W0, W1) + 0.003 (additive)** — 통과 시 plan-024 의 2가지 root-cause 동시 fix 의 single-model 정당화 + 가장 강한 single-shot 후보 확정. W2 가 max + 0.01 이상 = super-additive (synergy). W2 < max + 0.003 = redundancy / interference 박제 후 ensemble (다음 plan) 또는 dynamic anchor 분기. **session 지나도 까먹지 않기 위해 미리 박제** — plan-025/026 [DONE] 후 본 plan 진입.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 진입 조건 (필수)

- plan-025 status = `all_complete` AND best_variant 확정 (`E0/E1/E2/E3` 중 hit_1cm 최대) AND plan-025/.results.md frontmatter `best_variant` 박제됨.
- plan-026 status = `all_complete` AND best_variant 확정 (`V0/V_γδ/V_αβ/V_all` 중 hit_1cm 최대) AND plan-026/.results.md frontmatter `best_variant` 박제됨.
- 위반 시 `dependency_unmet` severe halt — 본 plan 의 lever 정의 자체 불가.

### §1.4 분기 decision rule (진입 시 자동 적용)

| plan-025 band | plan-026 band | 본 plan 진입? | 액션 |
|:-:|:-:|:-:|:--|
| positive / strong | positive / strong | **✅ 정상 진입** | W0/W1/W2 전부 실행 |
| positive / strong | negative | **conditional** | W0 single (= plan-025 best) 단독 측정 의의 X. W2 만 측정 후 plan-026 best 가 *실제* lever 인지 재검증. band 변경 시 따라감. |
| negative | positive / strong | **conditional** | W1 단독 의의 X. W2 만 측정 후 plan-025 best 가 lever 인지 재검증. |
| negative | negative | **❌ skip** | combo 의의 근본 부재 — `combo_invalid` warn 박제 + 본 plan 폐기 후 plan-028 (dynamic anchor) 분기. |

### 합격 기준 (G-gate sequence)

- **G0**: import + smoke + tests green (≥ 8/8). plan-025/026 의 best variant module 의 *import-only* dependency 확인. `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] 등 carry. plan-025 best variant reproduce + plan-026 best variant reproduce (W0 / W1 의 사전 sanity). 위반 시 `plan025_carry_drift` / `plan026_carry_drift` severe.
- **G2.W0**: plan-025 best variant 단독 reproduce — best_hit_1cm ± 0.0015 drift. 위반 시 `w0_drift` severe.
- **G2.W1**: plan-026 best variant 단독 reproduce — best_hit_1cm ± 0.0015 drift. 위반 시 `w1_drift` severe.
- **G2.W2**: combo — plan-025 best 의 loss/forward + plan-026 best 의 cand_feat. metric finite + max_class_ratio < 0.95 + soft_CE finite. 위반 시 `combo_numerical` severe.
- **G3 (additivity 판정)**: $\Delta_{combo} = W_2 - V_0$, $\Delta_{025} = W_0 - V_0$, $\Delta_{026} = W_1 - V_0$. additivity $A = \Delta_{combo} / (\Delta_{025} + \Delta_{026})$.
  - $A \geq 1.2$ → **band super_additive** (synergy 확정 — 가장 좋은 결과)
  - $1.0 \leq A < 1.2$ → **band additive** (두 lever 직교 ✓)
  - $0.7 \leq A < 1.0$ → **band partial_redundancy** (정보 일부 중복)
  - $A < 0.7$ → **band destructive** (두 lever 간섭)
  - $W_2 < \max(W_0, W_1)$ → **band interference** (combo 가 단독보다 못함 — alert + 원인 분석)
- **G_final**: results.md (10 항목) + best W# + additivity ratio + plan-024/022/025/026 head-to-head + per-fold variance + follow-up plan ≥ 2건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 + 8/8 pytest                                [TODO]
- G1: F0 + plan-025/026 best carry reproduce                     [TODO]
- G2.W0: plan-025 best 단독                                      [TODO]
- G2.W1: plan-026 best 단독                                      [TODO]
- G2.W2: combo (★ 핵심)                                          [TODO]
- G3: additivity 판정 (super / additive / partial / destructive / interference) [TODO]
- G_final: results + 3-file sync + follow-up                     [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-027-expansion-x-input-enrichment-combo.md` v1 작성 (본 commit) | [TODO] |
| c1.5 | docs | plan-025/026 결과 박제 후 *best variant 확정* — frontmatter `code_reuse` 의 module symbols 구체 update + §1.4 분기 결정 박제 | [TODO, 필수 pre-G0] |
| c1.7 | docs | plan-review-master 5-iter optional | [TODO, optional] |
| c2 | code | `analysis/plan-027/combo_model.py` — plan-025 best (loss/forward) × plan-026 best (cand_feat input) 통합 model class. spec @ §5.3 | [TODO] |
| c3 | code | `analysis/plan-027/combo_runner.py` — 5-fold OOF runner + 3 variant CLI (W0/W1/W2). spec @ §6 | [TODO] |
| c4 | test | `tests/test_plan027_smoke.py` — 8 pytest (combo_model forward shape + plan-025 best 단독 reproduce equality + plan-026 best 단독 reproduce equality + W2 의 두 path independence + F0 carry + mode collapse guard + 14 BCC invariant + samples/anchor floor) | [TODO] |
| G0 | gate | smoke + 8/8 pytest green | [TODO] |
| c5 | exp G1 | F0 + plan-025/026 best variant 동시 reproduce → `baseline_carry.json` 박제 | [TODO] |
| G1 | gate | F0 + plan-025/026 best 모두 tolerance 통과 | [TODO] |
| c6 | exp G2.W0 | plan-025 best 단독 reproduce control. `results_W0.json` 박제. | [TODO] |
| G2.W0 | gate | drift < 0.0015 ✓ | [TODO] |
| c7 | exp G2.W1 | plan-026 best 단독 reproduce control. `results_W1.json` 박제. | [TODO] |
| G2.W1 | gate | drift < 0.0015 ✓ | [TODO] |
| c8 | exp G2.W2 | combo — plan-025 best loss × plan-026 best cand_feat. `results_W2.json` 박제. | [TODO] |
| G2.W2 | gate | metric finite ✓ + max_class_ratio < 0.95 ✓ | [TODO] |
| c9 | analysis | 3 variant × 5-fold metric 표 + additivity ratio A 계산 + plan-024/022/025/026 head-to-head + band 판정 (super/additive/partial/destructive/interference) → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | band 판정 + super/additive 시 W2 ≥ max(W0, W1) + 0.003 추가 확인 | [TODO] |
| c10 | docs | `analysis/plan-027/results.md` 10 항목 + pair file + follow-up 2건 박제 (plan-028 ensemble / dynamic anchor) + 3-file frontmatter sync | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c10 [DONE] ✓ + follow-up 2+ 박제 ✓ | [TODO] |

### Plan-specific severe

- **`dependency_unmet`**: plan-025 또는 plan-026 의 `all_complete` 또는 `best_variant` frontmatter 부재. halt + telegram alert. 본 plan 진입 자체 불가.
- **`w0_drift` / `w1_drift`**: 단독 reproduce 가 plan-025/026 의 박제 best_hit_1cm 와 0.0015 이상 차이. carry drift = 모듈 변경 의심 → halt.
- **`combo_numerical`**: W2 forward/backward NaN/Inf 또는 mode collapse. halt.
- **`combo_invalid`** (warn only): §1.4 의 (negative, negative) cell. 본 plan 폐기 권고 박제 후 G_final 진입 — results.md 에 폐기 사유 박제.
- **`interference_detected`** (warn): G3 의 W2 < max(W0, W1). band=interference 박제 + 원인 분석 (gradient flow 측정 / overfit / 학습 instability 등) results.md §5 박제.

### Plan-specific paths

- whitelist 추가: `analysis/plan-027/**/*`
- whitelist 추가: `tests/test_plan027_smoke.py`
- blacklist 추가: `analysis/plan-024/**/*` `analysis/plan-025/**/*` `analysis/plan-026/**/*` (read-only import only)

### Decision-note 사용 예

- `decision-note: spec-default — best variant 확정 시점 = c1.5 (post-plan-026 [DONE]). plan-025/026 결과 보고 evolve 가능성 박제 — best 가 E2 row-expand × V_αβ 또는 E1 pointwise × V_all 의 *어느 조합인지* 사후 결정.`
- `decision-note: spec-default — combo_model 의 통합 방식 = "plan-025 best 의 model class 를 backbone 으로 + cand_feat 만 plan-026 best 의 build_v2 출력으로 swap". 두 module 의 *수직 합성* (loss path 위에 input path 차환).`
- `decision-note: spec-default — additivity threshold A=1.0/1.2/0.7 = 통계 noise band (per-fold std ~0.005, 5-fold OOF concat std ~0.0015) 고려한 보수 기준.`

---

## §1. 배경 / lever orthogonality 가설

### §1.1 plan-024 root-cause 의 정리 (사후분석 4축 carry)

| # | root-cause | fix plan | mechanism |
|:-:|:--|:--|:--|
| ① | expansion gap (LGBM 140k vs cross-attn 10k effective N) | plan-025 | loss/forward → grad unit (sample, anchor) 분리 → effective N ×14 |
| ② | static anchor × cross-attn mismatch | plan-026 (light) 또는 dynamic anchor (heavy) | query 측 anchor-conditional 비중 ↑ (light) 또는 key 자체 sample-conditional (heavy) |
| ③ | 16 lever FE 245D redundancy | (자체 fix 안함) | plan-024 v6 ablation 박제, plan-025/026 의 input 170D / 193D 가 *명시 anchor-aligned* 형식 — 다른 axis |
| ④ | 2M params + overfit | training schedule fix (별 lever, ep ↑ const lr) | post-G_final 영역 |

**본 plan = ① fix × ② fix 의 단일 모델 결합**.

### §1.2 가설: lever orthogonality

| 정보 차원 | plan-025 lever (loss/expansion) | plan-026 lever (input/enrichment) |
|:--|:--|:--|
| 학습 신호의 *양* (gradient signal 수) | 10k → 140k ↑ | 변경 X |
| forward 1회의 *질* (정보 밀도) | 변경 X | anchor-별 차이 비중 12% → 30% ↑ |
| Loss formulation | softmax CE (coupled) → pointwise MSE 또는 row-expand softmax | 그대로 (softmax CE) |
| cand_feat 차원 | 154D (plan-024 carry) | 174~193D (Group α/β/γ/δ 추가) |

**orthogonal 직관**: 두 lever 가 *다른 axis* (loss path vs input path) 를 건드림. *수학적으로 independent* 한 변경. 결합 시 가설:
- *additive*: $\Delta_{combo} = \Delta_{025} + \Delta_{026}$ (= A=1.0)
- *super-additive*: $\Delta_{combo} > \Delta_{025} + \Delta_{026}$ (= A>1.0, *synergy* — 한 lever 가 다른 lever 의 효과를 *증폭*. e.g. input 풍부화가 expansion 으로 늘어난 grad signal 의 *각 unit* 에 더 유의미한 학습 가능)
- *sub-additive*: $A < 1.0$ (정보 중복, 한 lever 가 이미 부분 fix 한 영역을 다른 lever 가 또 fix)
- *destructive*: $\Delta_{combo} < \max(\Delta_{025}, \Delta_{026})$ (간섭, 두 lever 의 학습 dynamics 가 충돌)

### §1.3 사용자 통찰 (turn 박제)

> "V5 = V4+ + pointwise expansion 이 가장 강한 single-shot 후보 — 이번 turn 의 입력 lever × plan-025 의 expansion lever 의 orthogonal combo"

본 plan = V5 의 정식 박제. 사용자가 *직관적으로* "V4+ × E1" 으로 명명한 조합을 정량 G-gate + additivity 측정 + branch 판정으로 확장.

### §1.4 분기 decision rule (재명시, §0.5 carry)

본 plan 의 진입 자체가 *조건부*. plan-025/026 의 G3 band 에 따라:

```
if plan025.band ∈ {positive, strong_positive} AND plan026.band ∈ {positive, strong_positive}:
    → 정상 진입. W0/W1/W2 3 variant 전부 측정.
elif plan025.band == negative AND plan026.band ∈ {positive, strong_positive}:
    → W2 만 측정. W0 의 lever 가 fail 박제 — combo 가 input 만의 효과 위 lift 있는지 측정.
elif plan026.band == negative AND plan025.band ∈ {positive, strong_positive}:
    → W2 만 측정. W1 의 lever 가 fail 박제 — combo 가 expansion 만의 효과 위 lift 있는지 측정.
elif plan025.band == negative AND plan026.band == negative:
    → skip 본 plan. `combo_invalid` warn 박제. 다음 plan = dynamic anchor (K-side fix).
```

이 decision 은 본 plan 의 c1.5 (post-026) 시점에서 *plan-025/026 의 frontmatter 읽고 자동* 분기. 사용자 confirm 불필요.

---

## §2. Scope

### §2.1 In-scope

| 항목 | 값 |
|:--|:--|
| anchor | 14 BCC (= plan-022/024/025/026 carry, 변경 X) |
| τ_cls | 0.001 (= 동일 carry) |
| seq_feat | 95D plan-024 carry (= 동일) |
| cand_feat | plan-026 best variant 의 dim (174 또는 177 또는 193D) — W1/W2 에서 사용 |
| backbone | plan-025 best variant 의 model class — W0/W2 에서 사용 |
| loss | plan-025 best 의 formulation (pointwise MSE / row-expand softmax / coupled CE+anchor embed) |
| fold | 5-fold stable_fold_id MD5 carry |
| variant | W0 (expansion only) / W1 (input only) / W2 (combo) = 3 variant |
| training budget | epoch + early stop + lr + batch — plan-025 best 또는 plan-026 best 중 *plan-025 best 우선* carry (decision-note 박제) |
| metric | OOF hit_1cm / hit_1.5cm / Δ_F0 / gap_ranking / top1_acc / max_class_ratio / soft_CE / dist_match_KL + **additivity ratio A** |
| compute | 3 variant × 5-fold CPU × ~250s ≈ **~13분** |

### §2.2 Out-of-scope

| 항목 | 이유 |
|:--|:--|
| plan-025/026 의 non-best variant 들 (E0/E2/E3 또는 V0/V_γδ 등) | best 만으로 lever 정의 명확. 다른 variant 는 plan-025/026 결과 박제로 충분. |
| Hyperparam sweep | single variable — 두 lever 의 결합 효과만 측정. |
| Training schedule fix (ep ↑ const lr) | 별 lever. |
| Ensemble (LGBM + neural blend) | next plan (plan-028 가칭) 영역 — 본 plan 은 *single-model combo*, ensemble 은 *blend*. 다른 axis. |
| Corrector reg head | 별 plan 영역. |
| Dynamic anchor (K-side) | follow-up B 영역 — 본 plan band negative 또는 interference 시 정당화. |
| DACON LB submit | quota 보호 + G3 PASS 후 별 plan. |
| Input aug σ=0.05 | plan-024 poss 3 진짜 lift +0.001 noise band 박제. |
| Hidden width / dropout scan | plan-024 v4/v7 효과 0 박제. |
| Anchor radius 확장 / F0 baseline ML 화 | 별 plan 영역. |

### §2.3 단일 변수 원칙

**변수 = 두 lever 의 결합 곱 (= "한 model 에 둘 다 적용 vs 둘 중 하나")**.

W0 vs V0 (plan-026 의 control, hit_1cm 0.6370) = plan-025 best 의 효과 측정 (= plan-025 자체에서 이미 박제됨, 본 plan W0 은 sanity reproduce).
W1 vs V0 = plan-026 best 의 효과 측정 (= plan-026 자체 박제, W1 은 sanity reproduce).
W2 vs V0 = combo 의 효과 측정. **본 plan 의 진짜 측정**.
$A = (W_2 - V_0) / ((W_0 - V_0) + (W_1 - V_0))$ = additivity.

---

## §3. 사전 등록

### §3.1 Fold split

5-fold stable_fold_id MD5 carry (변경 X).

### §3.2 G-gate 정량 (§0.5 carry, full 표 본문)

§0.5 carry. 추가:

- W0/W1 의 sanity drift tolerance = 0.0015 (= per-fold std × 0.5 보수)
- W2 의 numerical gate = max_class_ratio < 0.95 (= plan-022 §3.3 carry) + dist_match_KL < 0.01

### §3.3 평가 metric

primary: OOF hit_1cm + **additivity ratio A**.

secondary:
- per-fold variance
- gap_ranking, top1_acc, max_class_ratio, soft_CE, dist_match_KL
- W2 의 anchor-별 attention map analysis (informational, c9 에서 1개 sample fold 0 visualization)

### §3.4 band 판정 (§0.5 carry, 본문 표)

| A ratio | hit_1cm 조건 | band | 의미 |
|:--|:--|:--|:--|
| A ≥ 1.2 | W2 > max(W0, W1) | **super_additive** | 두 lever 가 synergy — 가장 좋은 결과 |
| 1.0 ≤ A < 1.2 | W2 ≥ max(W0, W1) | **additive** | 두 lever 직교 (가설 확정) |
| 0.7 ≤ A < 1.0 | W2 ≥ max(W0, W1) | **partial_redundancy** | 정보 일부 중복 단 lift 있음 |
| A < 0.7 | W2 ≥ max(W0, W1) | **mild_redundancy** | 큰 redundancy 단 winner 확정 |
| any | W2 < max(W0, W1) | **interference** | 간섭 — 원인 분석 + 다른 plan 분기 |

---

## §4. STAGE 0 — 인프라 (c2~c4 + G0)

### §4.1 module 작성

| module | symbol | 책임 |
|:--|:--|:--|
| `combo_model.py` | `ComboSelector` | plan-025 best 의 model class 를 backbone 으로 + cand_feat path 만 plan-026 best 의 build_v2 출력으로 swap. forward 의 cand_proj layer 의 in_features 만 차원 update. |
| `combo_runner.py` | `run_variant(W#, fold)` | 5-fold OOF runner. CLI: `--variant {W0|W1|W2} --fold {0..4|all}`. |

### §4.2 pytest (c4)

8 test 최소:
1. import 2 module + plan-025/026 best variant module dependency
2. ComboSelector forward shape: (B, seq, F=95) + (B, K=14, D_var) → (B, K=14) logits 또는 (B*K) scalar
3. W0 의 ComboSelector 가 plan-025 best 단독 reproduce — output element-wise *near-equal* (tol 1e-6)
4. W1 의 ComboSelector 가 plan-026 best 단독 reproduce — output element-wise *near-equal*
5. W2 의 forward: plan-025 loss path + plan-026 cand_feat — two path independence (gradient flow check)
6. F0 carry: f0_baseline 호출 → hit_1cm = 0.6320 ± 0.0005
7. mode collapse guard: random init forward 의 probs_all.mean(0).max() < 0.5
8. 14 BCC invariant: ANCHORS_A6 좌표 unchanged, ‖anchor_k‖ = 0.005m

### §4.3 G0 종료 조건

- 8/8 pytest pass
- import error 0
- 위반 시 `infra_drift` severe halt

---

## §5. STAGE 1~3 — Variant 사양

### §5.1 W0 — plan-025 best variant 단독 reproduce (control)

**구성**: plan-025 best variant 의 model + cand_feat 154D (= plan-024 carry, plan-026 input 안 적용).

**예상**: plan-025 의 best_hit_1cm ± 0.0015 (frontmatter 박제 값).

**산출**: `results_W0.json` (= plan-025 의 best variant results 와 매핑).

**의의**: W2 의 reference 1 — combo 가 W0 위에 *얼마나 추가* 인지 측정.

### §5.2 W1 — plan-026 best variant 단독 reproduce (control)

**구성**: plan-024 listwise CE backbone (= plan-026 의 V_all/V_αβ default) + cand_feat = plan-026 best 의 dim (174~193D).

**예상**: plan-026 의 best_hit_1cm ± 0.0015.

**산출**: `results_W1.json`.

**의의**: W2 의 reference 2.

### §5.3 W2 — combo (★ 본 plan 의 핵심)

**구성**:
- backbone = plan-025 best variant 의 model class (= PointwiseSelector / RowExpansionSelector / E1+anchor embed 중 best)
- loss = plan-025 best 의 formulation (pointwise MSE / row-expand softmax / etc.)
- cand_feat = plan-026 best 의 build_v2 출력 (= V_all 193D / V_αβ 177D / V_γδ 174D 중 best)
- seq_feat = plan-024 carry 95D (= 동일)
- hyperparam = plan-025 best 의 (epoch / lr / batch / weight_decay) carry — 단 plan-026 best 와 충돌 시 plan-025 우선 (decision-note 박제, plan-025 의 best 가 *loss formulation 의 stability* 결정 우선이라 더 sensitive)

**구현**: ComboSelector class 의 `__init__` 에서:
```
self.cand_proj = nn.Linear(in_features=plan026_best_cand_dim, out_features=hidden_dim)
self.seq_encoder = plan025_best_module.SeqEncoder(...)
self.loss_fn = plan025_best_module.loss_fn
# 나머지 forward = plan-025 best 그대로
```

**예상 lift**: $\Delta_{025} + \Delta_{026}$ 의 80~120% 추정. 즉 A ∈ [0.8, 1.2] 가 가장 가능성 높음 (additive 영역). super-additive 가능성 낮으나 0 아님 — input 풍부화가 expansion grad signal 의 *각 unit* 에 더 의미 학습 가능 시 가능.

**산출**: `results_W2.json`.

### §5.4 4-th variant 가능성 (out-of-scope, results.md 에 권고만)

W3 = W2 + plan-025 의 *2nd best variant* (e.g. anchor embed E3 lever 추가) — single variable 침범. 별 plan 영역 권고.

---

## §6. combo_runner.py 사양

### §6.1 CLI

```bash
python -m analysis.plan-027.combo_runner \
    --variant {W0|W1|W2} \
    --fold {0..4|all} \
    --output_dir analysis/plan-027/runs/{variant}/
```

### §6.2 산출 schema

```json
{
  "variant": "W2",
  "best_plan025_variant": "E1",            // c1.5 시점 확정
  "best_plan026_variant": "V_αβ",          // c1.5 시점 확정
  "cand_dim": 177,
  "loss_formulation": "pointwise_mse",
  "metrics": {
    "hit_1cm": 0.6xx,
    ...
  },
  "additivity": {
    "delta_025": 0.0xxx,
    "delta_026": 0.0xxx,
    "delta_combo": 0.0xxx,
    "A_ratio": 1.0xx,
    "band": "additive"  // super_additive / additive / partial / mild / interference
  },
  "per_fold": [...],
  "config": {...},
  "dataset_hash": "b91502db94fab67d"
}
```

---

## §7. paradigm_analysis (c9)

### §7.1 3 variant × 5-fold metric 표

| variant | cand_dim | loss | hit_1cm | Δ vs V0 | A ratio |
|:--|--:|:--|--:|--:|--:|
| V0 (plan-024 ref) | 154 | softmax CE | 0.6370 | — | — |
| W0 (plan-025 best) | 154 | (plan-025 best loss) | 0.6xx | +0.0xx | — |
| W1 (plan-026 best) | 174~193 | softmax CE | 0.6xx | +0.0xx | — |
| **W2 (combo)** | 174~193 | (plan-025 best) | **0.6xx** | **+0.0xx** | **1.0xx** |

### §7.2 head-to-head 표 (전체 plan 비교)

| plan | model | input dim | loss | hit_1cm | LB | band |
|:--|:--|:--|:--|--:|:--|:--|
| plan-022 winner | LGBM expansion | 170 | pointwise MSE (LGBM) | 0.6528 | — | positive |
| plan-024 v1 | cross-attn | 154 | softmax CE | 0.6370 | skip | negative |
| plan-025 best | cross-attn + expansion | 154 | plan-025 best | 0.6xx | — | (plan-025 band) |
| plan-026 best | cross-attn + enrichment | 174~193 | softmax CE | 0.6xx | — | (plan-026 band) |
| **plan-027 W2 (★)** | **combo** | **174~193** | **plan-025 best** | **0.6xx** | — | **(super/additive/partial/mild/interference)** |

### §7.3 band 별 결론 분기

- **super_additive (A ≥ 1.2)**: 두 lever synergy 확정. W2 가 단일 강력 winner. **next plan = plan-028 (ensemble — LGBM + W2 blend)** 가 plan-024 LB 0.6806 가까이 도달 후보.
- **additive (1.0 ≤ A < 1.2)**: 두 lever 직교 ✓ 가설 확정. W2 가 best single model. ensemble 또는 dynamic anchor 가 그 다음.
- **partial / mild redundancy (A < 1.0)**: 정보 중복 박제. winner 는 W2 단 lift 작음. dynamic anchor 또는 다른 paradigm shift 필요.
- **interference (W2 < max)**: 두 lever 학습 dynamics 충돌. 원인 분석 (gradient flow, loss landscape, instability) results.md §5 박제 후 fix 시도 또는 dynamic anchor 분기.

### §7.4 anchor-별 attention map analysis (informational)

W2 의 fold 0 prediction 의 attention map 시각화 1개 sample — combo 가 *어느 anchor 에 어떤 query dimension* 가 가장 기여 (gradient × activation) 측정. plan-024 의 mode collapse 패턴과 비교.

---

## §8. results.md 필수 항목 (c10, §N+2)

10 항목:

1. 한 줄 결론 + band 판정 (super/additive/partial/mild/interference)
2. §1 OOF metric table (3 variant × 8 metric + additivity)
3. §2 per-fold variance
4. §3 G-gate 결과
5. §4 plan-022/024/025/026 head-to-head
6. §5 band 별 원인 분석 (interference 시 gradient flow 박제)
7. §6 anchor-별 attention map (informational)
8. §7 comparison table (plan-022~027)
9. §8 follow-up plan 후보 (plan-028 ensemble / dynamic anchor)
10. §9 paths + commit chain final state

---

## §9. 작업량 회계

### §9.1 commit chain

c1 + c1.5 + (c1.7 optional) + c2~c4 + G0 + c5 + G1 + c6~c8 + G2 + c9 + G3 + c10 + G_final = **~12 commit** (= plan-025 보다 적음 — code 대부분 reuse)

### §9.2 compute

| variant | wall time | 근거 |
|:--|--:|:--|
| W0 (plan-025 best 단독) | ~plan-025 best time | carry |
| W1 (plan-026 best 단독) | ~plan-026 best time | carry |
| W2 (combo) | W0 + 20% overhead 추정 | input 확장 minor + loss 동일 |

→ 3 variant 총 **~700~900s ≈ 12~15분 CPU**.

---

## §N+3. 통계 함정 & caveats

### caveat #1: best variant 의 사후 확정 위험

본 plan spec 시점 (= plan-025/026 미실행 또는 미박제) 에 best variant 미정. c1.5 시점에 plan-025/026 의 frontmatter 읽고 *자동* 확정. 위험: spec 의 `code_reuse` symbols 가 placeholder ("E1 또는 E2 또는 E3" 등) — c1.5 에서 *정확한 module path* 박제 의무.

**완화**: c1.5 commit msg 에 `decision-note: best variant confirmed = plan-025:E?, plan-026:V_?` 명시. results.md 에 결정 사유 박제.

### caveat #2: plan-025/026 의 best 가 같은 fold variance 안

plan-025/026 의 G3 band 가 5-fold OOF std ~0.0015 보다 작은 lift 만 줄 경우, best variant 자체 noise. 본 plan 의 combo 가 *진짜 lever 효과* 측정 곤란.

**완화**: c1.5 시점에 plan-025/026 의 *best 와 2nd best 의 차이* 가 std × 1 (≈ 0.0015) 보다 작은지 측정. 작으면 `combo_invalid` warn 추가 (단 진입 자체는 가능, band 판정 conservative).

### caveat #3: training schedule 의 inheritance 충돌

plan-025 best 의 hyperparam (e.g. epoch 22 + lr 7e-4) 와 plan-026 best 의 hyperparam 이 다를 수 있음. 본 plan W2 에서 어느 것을 carry?

**완화**: decision-note 박제 = "plan-025 best 의 hyperparam 우선". 이유 — loss formulation 의 stability 가 더 sensitive (pointwise/row-expand 의 lr 변경 시 발산 risk). 단 plan-026 best 의 hyperparam 도 c1.5 results.md 박제. W2 의 sensitivity 평가 별 plan.

### caveat #4: gradient flow 의 instability 위험 (interference 가설)

plan-025 best 의 loss path (= pointwise MSE 또는 row-expand softmax) 가 *unrestricted scalar grad* 를 흘리는 동시에, plan-026 best 의 cand_feat 의 *anchor-별 align profile* 이 *correlated input distribution* 을 제공. gradient flow 의 *direction* 이 두 lever 의 학습 dynamics 와 충돌 가능 (e.g. anchor-별 align profile 이 pointwise MSE 에서는 *각 anchor 가 독립* 이라 잘 작동, softmax CE 에서는 *coupled* 라 충돌).

**완화**: combo_model 의 forward 의 첫 epoch 의 grad norm 박제. 첫 epoch grad norm > 10 (= init weight 의 100×) 시 `gradient_explosion` warn 박제 + 학습 진행 단 lr 0.5× 자동 감소 (decision-note 박제).

### caveat #5: super-additive 의 over-claim 위험

A > 1.2 박제 시 *통계적으로 유의미* 한지 검증 필요. 5-fold OOF std ~0.0015 환경에서 W2/W0/W1 의 동시 std 가 *correlated* (같은 fold = correlated) 라 A 의 std 더 정밀 측정 필요.

**완화**: c9 에서 *fold-bootstrap* 으로 A 의 95% CI 박제. CI 하한 > 1.0 이어야 super_additive 진짜 인정. CI 안 만족 시 band = additive 로 downgrade.

### caveat #6: plan-026 의 V_all (193D) 의 over-parameterization

cand_feat 193D 가 plan-024 v6 (245D) ablation 박제 (+0.0003) 의 paradigm risk. 본 plan W2 가 combo 환경에서도 동일 risk — expansion 의 effective N 14× 가 over-param 을 *완화* 단 가능성 X.

**완화**: per-group grad-norm (plan-026 carry) 의 W2 환경 재측정 — Group γ/δ 의 grad-norm 이 < 0.01 시 redundancy 박제. plan-026 의 redundancy_warning 의 W2 carry.

### caveat #7: ensemble (plan-028 가칭) 와의 axis 분리

본 plan = *single-model* combo. ensemble = *separate models 의 blend*. 두 axis 다름.
- single-model = 두 lever 의 *학습 단계 동시 적용* — 새로운 emergent property 가능
- ensemble = 두 lever 의 *별 모델 학습 후 inference blend* — independent + post-hoc

본 plan W2 의 결과가 ensemble 의 *upper bound* 아니다 — ensemble 의 diversity gain 이 더 클 가능성. 본 plan 의 결론은 *combo 단독* 의 결론, ensemble 은 plan-028 의 별 측정.

### caveat #8: K-side dynamic anchor 와의 정합 부재

본 plan = Q-side enrichment × loss expansion. K-side (anchor 자체 sample-conditional) 는 plan-026 §0.5 의 follow-up B 영역. 본 plan band negative / interference 시 K-side fix 가 정당화. 단 본 plan band super_additive 시도 K-side dynamic anchor 가 *추가* lift 가능성 — 별 plan 영역.

---

## §N+4. 변경 이력

- v1 (2026-05-22): 초안. 사용자 명시 ("V5 가 가장 강한 single-shot 후보" + "session 지나면 까먹겠다") 박제. plan-025/026 best variant 의 orthogonal combo + additivity ratio A 측정. 3 variant W0/W1/W2 + 5 band (super/additive/partial/mild/interference). dependency on plan-025 [DONE] + plan-026 [DONE] + best variant frontmatter 박제 필수. §1.4 분기 decision rule 자동 적용.

---

## §N+5. 참조

- `plans/plan-025-expansion-mimic-anchor-embed.md` (root-cause #1 fix, best variant TBD)
- `plans/plan-026-anchor-aligned-input-enrichment.md` (root-cause #2 light workaround, best variant TBD)
- `plans/plan-024-cross-attention-anchor-vocab.md` v1.1-rev2 + results.md §5 (사후분석 4축 anchor)
- `plans/plan-022-corrector-free-anchor-layout-sweep.md` (14 BCC + τ=0.001 winner)
- `plans/plan-020-f0-structural-search.md` (F0 baseline + 5-fold stable_fold_id)
- `WORKFLOW.md §1~§12` + `CLAUDE.md` (autonomous execution policy)
