---
plan_id: 026
version: 1.0
date: 2026-05-22 (Asia/Seoul)
status: draft
based_on:
  - 024 (cross-attn anchor-vocab G2 FAIL hit_1cm 0.6370, -0.0158 vs plan-022. 사후분석 4축 root-cause 박제: ①sample-weight expansion effective N 14× gap [plan-025 영역], ②static anchor × cross-attn mismatch [본 plan 의 light-weight workaround], ③16 lever FE 245D redundancy [v6 ablation +0.0003 — 본 plan 의 redundancy risk 박제], ④2M params overfit plateau val_loss ≈ log14. plan-024 cand_feat 154D = 4 묶음 (① par/perp/dist 3D + ② anchor spec 9D + ③ ctx 128D + ④ interactions 10D). anchor-별 차이 비중 19D / 154D = 12% 만.)
  - 022 (A6_bcc14_tau001 winner OOF hit_1cm 0.6528 / hit_1.5cm 0.8104 / Δ_1cm +0.0208. 14 BCC anchor + τ=0.001 sharp soft label paradigm carry, 본 plan 의 K=14 static anchor 정합 fix.)
  - 021 (170D LGBM input carry. F0 baseline 0.6320 / 0.8033 anchor. _build_L2_L4 의 L2 residual sequence (N, 7, 3) Frenet — 본 plan 의 Group α/β/γ 의 *raw deviation source*.)
  - 020 (F0 baseline + 5-fold stable_fold_id MD5)
inspired_by:
  - 사용자 통찰 (turn 2026-05-22 session "plan-026"):
    (a) "key 가 sample-invariant 한 게 문제였다면 *query 로 이동* — 이전 seq 들에서 위치가 공식에서 어떻게 벗어났는가" → root-cause #2 의 *light-weight workaround* (key 변경 없이 query 측 anchor-conditionality 활성)
    (b) "후보에 대한 input 을 전부 포함시켜야 — 후보와 align" → cand_feat 154D carry 의무
    (c) "이전 seq 공식 deviation + 후보 위치 별 CE — F0_pred 의 0.5cm 이동 매칭 prior 명시"
    (d) "공식에서 z 축 얼마나 벗어났는지 + 마지막 F0_pred 의 z 가 어디인지 — 비슷한 axis 같이 묶기" → z-axis affinity 학습 쉬운 bundle 화
    (e) "단순 Frenet + xy + z + 14 후보 별 alignment 모두 있어야" → V4+ design 의 5 group (cand 154D carry + Group α/β/γ/δ)
  - plan-004 의 implicit lever 의 *dual*: plan-004 = key 가 공식별 예측 (27 physics), 본 plan = query 가 공식별 deviation profile (anchor 14 별). plan-022 winner 14 BCC paradigm carry 위에서 *경량* 으로 plan-004 의 physics-anchored matching 재현.
code_reuse:
  - module: analysis/plan-024/cand_builder.py
    symbols: [build, _build_L1_frenet, _macro_stat_8, _multiwindow_144, _wingbeat_jitter, _pct_rolling_peak, _v_autocorr_3, _bcc_adjacency]
    reason: cand_feat 154D base 4 묶음 (①②③④) 의 정확 carry. 본 plan 의 Group γ/δ/α/β 가 *additive* 으로 154D 위에 concat. plan-024 의 ③ ctx broadcast (128D) 의 base 12 (v_last + a_last + res_last + EWMA_03_res) 와 macro_stat_8 의 redundancy 평가 필요 (caveat).
  - module: analysis/plan-024/model.py
    symbols: [CandidateAttentionGRUSelector]
    reason: cross-attn GRU backbone carry. cand_feat 차원만 174D / 193D 로 확장 (model 의 first cand_proj layer 의 in_features 만 변경, 나머지 architecture 동일).
  - module: analysis/plan-024/seq_builder.py
    symbols: [build_seq_feat]
    reason: 7-step seq_feat 95D (T=7, F=95 per step) carry. Group γ/δ 가 *cand_feat 쪽* 에 broadcast 으로 들어가므로 seq_feat 는 plan-024 그대로 유지 — *redundancy risk 박제*: ③ broadcast 의 base 12 와 Group γ 의 Frenet pool 이 정보적 overlap.
  - module: analysis/plan-021/build_input.py
    symbols: [_build_L2_L4, build_input_common, build_frenet_basis_3d, to_frenet]
    reason: L2 (N, 7, 3) Frenet residual sequence = Group α/γ 의 raw source. _build_L2_L4 의 출력 (L2, L4) 을 cand_builder 의 추가 path 로 주입.
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: 14 BCC anchor 좌표 (= plan-024 carry, 변경 X — root-cause #2 fix 의 핵심 = *anchor 자체는 static 유지하고 query 측에서 sample-conditionality 회복*).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [build_soft_label_with_tau]
    reason: τ=0.001 soft label q_target 계산 (= plan-022/024 carry).
  - module: analysis/plan-022/run_oof.py
    symbols: [main]
    reason: G1 의 plan-022 winner LGBM 5-fold OOF reproduce entry (variant A6_bcc14_tau001). 본 plan c8 에서 `python -m analysis.plan-022.run_oof --variant A6_bcc14_tau001 --tau 0.001 --output analysis/plan-026/baseline_carry/plan022_carry.json` 호출. cross-attn runner 와 별도 path.
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, R_HIT, R_HIT_LOOSE, D1, PAR, PERP]
    reason: F0 baseline + paired Δ anchor.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split (plan-020/021/022/023/024/025 carry).
followed_by:
  - (가칭 next plan A): plan-025 expansion mimic + 본 plan V_all combo (= "V5" in turn analysis) — root-cause #1 + #2 직교 lever 의 곱. plan-025 와 plan-026 둘 다 G2 PASS band positive 시 자동 정당화.
  - (가칭 next plan B): dynamic anchor (K dynamic — plan-004 의 27 physics candidate paradigm 복원). 본 plan V_all 이 < 0.6528 (band negative) 일 때만 정당화 — *full K-side fix* 가 진짜 본질 확정.
  - (가칭 next plan C): ensemble (plan-022 LGBM + 본 plan V_all best + plan-025 best variant). 3-way inductive bias diversity.
scope: 단일 변수 = **cand_feat 의 anchor-conditional 정보 비중 + axis-affinity 묶음**. 4 variant V0/V_γδ/V_αβ/V_all 의 head-to-head — cand_feat 차원 154D → 174D 또는 193D 확장. anchor 14 BCC + input seq_feat 95D + τ_cls=0.001 + hidden 384 + softmax CE loss = plan-024 carry 정확 동일. **expansion mimic (plan-025 의 E1/E2/E3) / 16 lever 추가 FE / dynamic anchor (K side fix) / corrector reg head / ensemble / DACON LB submit / hyperparam sweep / input augmentation σ=0.05 / hidden width sweep / τ_cls scan / 1-fold long-diagnose = out-of-scope**.
exp_ids:
  - Z026_V0_plan024_reproduce
  - Z026_Vgd_anchor_invariant_pool
  - Z026_Vab_anchor_conditional_align
  - Z026_Vall_combined_193D
# exp_id ↔ variant 매핑: frontmatter `Z026_V{n}_<short>` ↔ 본문 variant_key `V{n}_<short>`.
lb_score: null
band: null
---

# plan-026 v1 — Anchor-Aligned Input Enrichment (Group α/β/γ/δ on cand_feat)

## §0. 한 줄 목적

> **plan-024 사후분석 #2 root-cause (static anchor × cross-attn mismatch) 의 *light-weight workaround* — anchor 14 BCC 자체는 static 유지하되, "이전 seq 가 공식에서 어떻게 벗어났는가" 를 (i) anchor-별 alignment profile (ii) z-axis bundle (iii) Frenet 3축 anchor-invariant pool (iv) world xy anchor-invariant pool 4 group 으로 cand_feat 에 명시 박아넣어, query 의 anchor-conditional 비중 (= cand_feat 의 *anchor-aware 채널 비중*, 광의 정의 §7.1 참조 — anchor-별 차이 채널 + anchor-invariant broadcast pool 모두 분자 포함) 을 14.3% → 30% 로 확대한다.** 4 variant (V0 control / V_γδ anchor-invariant pool only / V_αβ anchor-conditional only / V_all combined 193D) head-to-head. **G2 PASS = max(V_γδ, V_αβ, V_all) ≥ 0.6528 (plan-022 winner 회복)** — 회복 시 input enrichment 만으로 root-cause #2 우회 가능 확정, 미회복 시 K-side dynamic anchor 가 본질임 확정 후 plan-027 dynamic anchor 분기.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 5 module (cand_builder_v2 / group_alpha / group_beta / group_gamma_delta / pytest) import + smoke + tests green (≥ 12/12). 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] (plan-020/021/022/023/024/025 carry exact). plan-022 winner reproduce hit@1cm ∈ [0.6523, 0.6533] AND hit@1.5cm ∈ [0.8099, 0.8109]. plan-024 v1.1-rev2 reproduce hit_1cm ∈ [0.6360, 0.6385]. 위반 시 각각 `f0_reproduce_drift` / `plan022_carry_drift` / `plan024_reproduce_drift` severe.
- **G2.V0**: plan-024 v1.1-rev2 reproduce (cand 154D 그대로) — hit_1cm ∈ [0.6360, 0.6385]. 위반 시 `plan024_v0_drift` severe (V_γδ/V_αβ/V_all 측정 무의미 → halt).
- **G2.V_γδ**: cand 154D + Group γ (12D) + Group δ (8D) = **174D anchor-invariant pool**. hit_1cm finite + max_class_ratio < 0.95 + soft_CE finite. 위반 시 `vgd_numerical` severe.
- **G2.V_αβ**: cand 154D + Group α (15D) + Group β (8D) = **177D anchor-conditional align + z bundle**. 동일 무결성 gate. 위반 시 `vab_numerical` severe.
- **G2.V_all**: cand 154D + α 15D + β 8D + γ 12D + δ 8D = **193D combined**. 동일 무결성 gate. 위반 시 `vall_numerical` severe.
- **G3 (paradigm-level)**: max(V_γδ, V_αβ, V_all) hit_1cm ≥ **0.6528** (plan-022 winner) → PASS. 0.6528 ≤ best < 0.6628 = band positive (input enrichment 단독으로 plan-022 회복, root-cause #2 light-weight fix 성공). best ≥ **0.6628** = band strong_positive (= plan-022 winner 0.6528 + 0.0100 의 plan-024 사후분석 turn 박제 상한 — *cross-attn arch 가 풍부한 query 위에서 plan-022 초과* 의 명시적 PASS 기준). best < 0.6528 = band negative (`enrichment_no_recovery` warn 박제 → K-side dynamic anchor 가 진짜 본질 확정, 다음 plan 분기).
- **G_final**: results.md (12 항목) + best variant 박제 (V# + hit_1cm + gap_ranking + top1_acc + anchor-별 차이 비중) + plan-024 v1 (0.6370) / plan-022 (0.6528) head-to-head 표 + 4 group (α/β/γ/δ) lever decomposition 결론 + follow-up plan 후보 ≥ 2건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 + 12/12 pytest                              [TODO]
- G1: STAGE 1 F0 + plan-022 carry + plan-024 reproduce            [TODO]
- G2.V0: plan-024 reproduce control                              [TODO]
- G2.V_γδ: anchor-invariant pool                                 [TODO]
- G2.V_αβ: anchor-conditional align + z bundle                    [TODO]
- G2.V_all: combined 193D                                        [TODO]
- G3: paradigm — band 판정 (positive / strong_positive / negative) [TODO]
- G_final: results + 3-file sync + lever decomposition + follow-up [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-026-anchor-aligned-input-enrichment.md` v1 작성 (본 commit) | [TODO] |
| c1.5 | docs | plan-review-master 5-iter 자동 fix (optional pre-G0) | [TODO, optional] |
| c2 | code | `analysis/plan-026/group_alpha.py` — anchor-별 7-step Frenet 3축 align profile (5 var × 3 stat = 15D). spec @ §5.2 | [TODO] |
| c3 | code | `analysis/plan-026/group_beta.py` — z-axis (Frenet b̂ / world z) bundle (8D). spec @ §5.3 | [TODO] |
| c4 | code | `analysis/plan-026/group_gamma_delta.py` — anchor-invariant Frenet 3축 + world xy 7-step pool (γ 12D + δ 8D = 20D, broadcast to K). spec @ §5.4 | [TODO] |
| c5 | code | `analysis/plan-026/cand_builder_v2.py` — plan-024 cand_builder 154D 의 *extension wrapper*. ablation flag (--with_alpha / --with_beta / --with_gamma_delta). spec @ §5.1 | [TODO] |
| c6 | code | `analysis/plan-026/enrichment_runner.py` — 5-fold OOF runner + 4 variant CLI (V0/V_γδ/V_αβ/V_all). spec @ §6 | [TODO] |
| c7 | test | `tests/test_plan026_smoke.py` — 12 pytest (4 group module shape + invariance + 4 variant cand_feat dim + sample-conditional invariant + F0 carry + samples/anchor floor + mode collapse guard + anchor 14 BCC 좌표 invariant + 154D carry equality) | [TODO] |
| G0 | gate | smoke + tests green 12/12 | [TODO] |
| c8 | exp G1 | F0 + plan-022 carry + plan-024 carry reproduce → `analysis/plan-026/baseline_carry.json` 박제 | [TODO] |
| G1 | gate | F0 + plan-022 + plan-024 carry 모두 tolerance 통과 | [TODO] |
| c9 | exp G2.V0 | plan-024 v1.1-rev2 reproduce — cand 154D 그대로. control variant. `results_V0.json` 박제. | [TODO] |
| G2.V0 | gate | plan-024 0.6370 reproduce ✓ (hit_1cm ∈ [0.6360, 0.6385] = 비대칭 band, §0.5 G2.V0 / §3.2 와 동일) | [TODO] |
| c10 | exp G2.V_γδ | cand 174D = 154D + γ+δ. anchor-invariant pool 단독. `results_Vgd.json` 박제. | [TODO] |
| G2.V_γδ | gate | metric finite ✓ + max_class_ratio < 0.95 ✓ | [TODO] |
| c11 | exp G2.V_αβ | cand 177D = 154D + α+β. anchor-conditional align + z bundle 단독. `results_Vab.json` 박제. | [TODO] |
| G2.V_αβ | gate | metric finite ✓ + max_class_ratio < 0.95 ✓ | [TODO] |
| c12 | exp G2.V_all | cand 193D = 154D + α+β+γ+δ. combined. `results_Vall.json` 박제. | [TODO] |
| G2.V_all | gate | metric finite ✓ + max_class_ratio < 0.95 ✓ + redundancy 측정 (group 별 grad norm 박제) | [TODO] |
| c13 | analysis | 4 variant × 5-fold metric 표 + plan-024 (0.6370) / plan-022 (0.6528) head-to-head + per-fold variance + 4 group (α/β/γ/δ) lever decomposition (additivity 측정: V_αβ + V_γδ ≈ V_all?) + gap_ranking 비교 + anchor-별 차이 비중 12% / 24% / 30% 별 effect → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | band 판정 (max(V_γδ, V_αβ, V_all) vs 0.6528 / 0.6628) | [TODO] |
| c14 | docs | `analysis/plan-026/results.md` 12 항목 + `plans/plan-026-*.results.md` pair + follow-up 3건 박제 (next plan A: 025+026 combo / next plan B: dynamic anchor / next plan C: ensemble) + 3-file frontmatter sync (status=all_complete, band=…, best_variant=…) | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c14 모두 [DONE] ✓ + follow-up 3+ 박제 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- **`plan024_v0_drift`**: G2.V0 의 plan-024 v1.1-rev2 reproduce 가 hit_1cm ∉ [0.6360, 0.6385] (= §0.5 G2.V0 / §3.2 와 정확 동일 band, 비대칭 — center 0.6370, lower drift 0.0010 / upper drift 0.0015 의 plan-024 v1.1-rev2 측정 noise 박제). control 자체 실패 시 V_γδ/V_αβ/V_all 결과 해석 무의미 → halt + telegram alert.
- **`vgd_numerical`** / **`vab_numerical`** / **`vall_numerical`**: 각 variant 의 forward 또는 backward NaN/Inf, 또는 max_class_ratio ≥ 0.95 mode collapse. halt + 원인 분석 후 hyperparam fix.
- **`enrichment_no_recovery`** (warn only, halt 아님): G3 의 max(V_γδ, V_αβ, V_all) hit_1cm < 0.6528. input enrichment 단독으로는 root-cause #2 부족 확정 → band negative + dynamic anchor 분기. results.md §5 에 분기 결론 박제 후 G_final 진입.
- **`redundancy_warning`** (warn only): 두 trigger 분리.
  - **(a) additivity break** — additivity ratio `A = (Δ_αβ + Δ_γδ) / Δ_all > 1.2` (§7.3 Δ 정의). group 간 redundancy 박제, results.md §7 에 quantify.
  - **(b) γδ no-lift** — `V_γδ - V_0 < 0.005` (caveat #1 박제, ctx broadcast 와 정보 중복). Group γ/δ 단독 lever 의 marginal contribution 실패 박제. results.md §7 에 함께 정량.

### Plan-specific paths

- whitelist 추가: `analysis/plan-026/**/*` (본 plan 산출 영역)
- whitelist 추가: `tests/test_plan026_smoke.py`
- blacklist 추가: `analysis/plan-024/**/*` (plan-024 의 cand_builder.py 등 *read-only* import, code 수정 금지 — drift 추적 가능성 보존)
- blacklist 추가: `analysis/plan-022/**/*`
- blacklist 추가: `analysis/plan-021/**/*`

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — Group α 의 5 var = (proj_k, gap_k, t̂Δ_k, n̂Δ_k, b̂Δ_k). 3 var (proj/gap/t̂) 대안 미선택, n̂/b̂ axis 추가로 Frenet 3축 모두 cover.`
- `decision-note: spec-default — Group β 의 z reference frame = Frenet b̂ + world z 둘 다 (4D + 4D). 한 frame 선택 X — 박쥐 task 의 gravity reference (world) 와 motion-aligned (Frenet) 가 다른 정보.`
- `decision-note: spec-default — Group β/γ/δ 의 sequence pool stat = (mean, std, last, slope), 4 stat — slope 가 linear trend, std 가 wingbeat-like 진동, last 가 most-recent, mean 이 baseline. Group α 만 (mean, std, slope), 3 stat — last 가 묶음 ① par/perp/dist 와 중복하므로 제외.`

---

## §1. 배경 / 사용자 통찰

### §1.1 plan-024 fail 의 root-cause #2 (사후분석 §5 박제 + 본 plan 의 light-weight workaround)

**root-cause #2**: 14 BCC anchor 가 모든 sample 에 정적 → cross-attn 의 (query, key) 매칭에서 *key 측 sample-conditionality 부재* → attention 의 inductive bias 의 절반 만 활용 (degenerate matching).

**해결 path 2가지**:

| path | 방법 | 비용 | 본 plan 위치 |
|:--|:--|:--|:--|
| **Heavy (K-side fix)** | anchor 14 → 27+ sample-conditional candidate (plan-004 27 physics carry pattern) — candidate generator 새 모듈 + plan-022 paradigm 변경 | 무거움 | follow-up next plan B |
| **Light (Q-side workaround)** | anchor 14 BCC static 유지, *query 의 anchor-conditional 비중* 만 12% → 24% / 30% 로 확대 — input feature enrichment | 가벼움 (cand_builder extension) | **본 plan** |

본 plan = light path 의 직접 시도. heavy path 의 정당화는 light path *fail 시* (band negative) 자동.

### §1.2 사용자 통찰 5가지 (turn 분해)

| # | 통찰 | 본 plan 의 group |
|:--|:--|:--|
| (a) | "key 가 sample-invariant 한 게 문제 → query 로 이동" | 전체 design 원리 (Q-side enrichment) |
| (b) | "후보에 대한 input 전부 포함 — 후보와 align" | cand_feat 154D carry 의무 + Group α (anchor-별 align) |
| (c) | "이전 seq 공식 deviation + 후보 위치 CE — F0_pred 0.5cm 이동 매칭" | Group α 의 proj_k / gap_k (anchor 가 deviation 후보 vector 라는 prior 명시) |
| (d) | "z 축 얼마나 벗어났는지 + F0_pred 의 z — 비슷한 axis 묶기" | Group β (z bundle, axis-affinity) |
| (e) | "Frenet, xy, z, 14 후보 alignment 모두 있어야" | Group α + β + γ (Frenet 3축 pool) + δ (world xy pool) |

### §1.3 plan-024 cand_feat 154D 의 현 상태 진단

```
[묶음 ①: 3D]   par/perp/dist (sample × anchor, 마지막 step 만)         ← anchor-별 ✓
[묶음 ②: 9D]   anchor spec (static)                                   ← anchor-별 (단 sample 무관)
[묶음 ③: 128D] ctx broadcast (sample-conditional, K row 동일)          ← anchor-별 ✗
[묶음 ④: 10D]  interactions (sample × anchor scalar)                  ← anchor-별 ✓
```

**진단**:
- anchor-별 *차이가 있는* dimension = ①(3) + ②(9) + ④(10) = **22D / 154D = 14.3%** (단 ② 는 sample 무관 static)
- *sample × anchor* 둘 다 dependent = ①(3) + ④(10) = **13D / 154D = 8.4%**
- 즉 attention 의 *sample-specific anchor matching* 의 발판이 ~13D 만 — 너무 적음.
- 묶음 ① 의 par/perp/dist 가 *마지막 step* 만 — *과거 7-step* 의 deviation 패턴 별 anchor alignment **MISSING**.

본 plan 의 fix = sample × anchor dependent dimension 을 13D → 28D (V_αβ) 또는 23D (V_all) 로 확대 + axis-affinity bundle + Frenet/xy pool 의 명시.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|:--|:--|
| anchor | 14 BCC (= plan-022 A6, plan-024/025 carry — **본 plan 변경 X**) |
| τ_cls | 0.001 (= plan-022 winner, plan-024/025 carry) |
| seq_feat (GRU input) | 95D plan-024 carry (= 16 lever FE max 포함, base = plan-021 170D 의 일부) — **본 plan 변경 X** |
| cand_feat | **확장** — V0=154D, V_γδ=174D, V_αβ=177D, V_all=193D |
| selector backbone | CandidateAttentionGRUSelector plan-024 carry — cand_proj layer 의 in_features 만 154 → 174/177/193 변경, 그 외 동일 |
| fold split | 5-fold stable_fold_id MD5 (plan-020 carry, 변경 불가) |
| variant | V0 / V_γδ / V_αβ / V_all = 4 variant |
| training budget | epoch 22 + early stop patience 8 + lr 7e-4 const + weight_decay 0.02, batch 32 — **plan-024 v1 default 정확 carry** |
| loss | softmax CE listwise (plan-024 carry, coupled — 본 plan 변경 X, expansion 은 plan-025 영역) |
| metric set | OOF hit_1cm / hit_1.5cm / Δ_F0 / Δ_F0_1.5 / gap_ranking / top1_acc / max_class_ratio / soft_CE / dist_match_KL + **per-group grad-norm contribution** (lever decomposition) |
| compute | 5-fold CPU 4 variant × ~200s ≈ **~13분 추정** (cand_feat 확장 wall-time minor overhead) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|:--|:--|
| Expansion mimic (pointwise / row-expand) | plan-025 영역 — single variable 원칙. 본 plan G3 PASS band positive 시 *next plan A* 로 combo 추진. |
| Dynamic anchor (K-side sample-conditional) | follow-up next plan B 영역. 본 plan 의 light-weight fix 가 *대안 path*. |
| 16 lever 추가 FE | plan-024 v6 ablation 박제 — *anchor-invariant pool* (Group γ/δ) 은 별 lever, 같은 axis 아님. 단 caveat §N+3 에 redundancy risk 박제. |
| Corrector reg head | plan-021 selector-only carry, 별 plan 영역. |
| Ensemble | next plan C 영역. |
| DACON LB submit | quota 보호 + G3 PASS 후 별 plan 영역. |
| Input augmentation σ=0.05 | plan-024 poss 3 진짜 lift +0.001 박제 (= noise band). single variable 원칙. |
| Hidden width sweep | plan-024 v4/v7 효과 0 박제. |
| τ_cls scan | plan-022 winner carry. |
| Anchor radius 확장 (0.5cm → 0.7cm) | 별 plan 영역. |
| F0 baseline ML 화 | 별 plan 영역. |
| 1-fold long-diag | plan-024 §5.10 영역. 본 plan 5-fold OOF 정식 측정만. |
| Training schedule fix (ep 100 + const lr + batch 256) | 별 lever (사후분석 turn 박제 — long-diag 가 0.6495 회복). 본 plan single variable 침범 회피. |

### §2.3 단일 변수 원칙

**plan-026 의 변수 = cand_feat 의 anchor-conditional 정보 비중 + axis-affinity 묶음**.

4 variant 의 cand_feat 차원만 변경, 그 외 모든 lever (seq_feat / model / loss / hyperparam / training schedule / fold) = plan-024 v1.1-rev2 carry 정확 동일.

V0 ≠ V_γδ 비교 = Group γ+δ 효과 측정 (anchor-invariant pool).
V0 ≠ V_αβ 비교 = Group α+β 효과 측정 (anchor-conditional align + z bundle).
V_αβ + V_γδ vs V_all 비교 = additivity / redundancy 측정.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

5-fold stable_fold_id MD5 (plan-020 carry). plan-024/025 와 정확 동일 — 직접 head-to-head 가능.

### §3.2 합격 기준 (G-gate 정량 정의)

| Gate | 정량 조건 | severe |
|:--|:--|:--|
| G0 | pytest ≥ 12/12 pass AND import error 0 | `infra_drift` |
| G1 | F0 + plan-022 + plan-024 reproduce 3 set 모두 tolerance 통과 | `*_drift` |
| G2.V0 | hit_1cm ∈ [0.6360, 0.6385] | `plan024_v0_drift` |
| G2.V_γδ | metric finite + max_class_ratio < 0.95 | `vgd_numerical` |
| G2.V_αβ | metric finite + max_class_ratio < 0.95 | `vab_numerical` |
| G2.V_all | metric finite + max_class_ratio < 0.95 + per-group grad-norm 박제 | `vall_numerical` |
| G3 (band 판정) | max(V_γδ, V_αβ, V_all) hit_1cm 의 위치:<br>- ≥ 0.6628 → **band strong_positive**<br>- ≥ 0.6528 → **band positive**<br>- < 0.6528 → **band negative** | `enrichment_no_recovery` (warn) |
| G_final | 3-file frontmatter sync + §0.5 c1~c14 [DONE] + follow-up 3건 박제 + results.md ≥ 12 항목 | `g_final_incomplete` |

### §3.3 평가 점수

primary: **OOF hit_1cm**.

secondary:
- hit_1.5cm
- Δ_F0_1cm / Δ_F0_1.5cm
- gap_ranking (= oracle_1cm − argmax_hit, plan-024 v1 = 0.1934 reference)
- top1_acc (plan-024 v1 = 0.1227, plan-022 = 0.1707 reference)
- max_class_ratio
- soft_CE (= cross entropy on q_target, log(14) = 2.639 uniform reference)
- dist_match_KL
- **per-group grad-norm** (= V_all 의 backward 시 cand_proj 의 columns 별 grad L2. group α/β/γ/δ/154D existing 별 평균. 측정 시점 = *best-val-epoch 의 마지막 train batch* (early stop 후 reload 한 weights 의 fold-internal val set 첫 batch backward grad). *redundancy 측정의 핵심 metric* — group 별 contribution 의 정량 분해)
- **per-group ablation lift** (= V_all − V_αβ ≈ γδ contribution / V_all − V_γδ ≈ αβ contribution / V_αβ + V_γδ − V_all = redundancy)

### §3.4 redundancy 측정 (lever decomposition)

추가 정량:
- **additivity ratio** $A = (V_{\alpha\beta} + V_{\gamma\delta} - 2 \cdot V_0) / (V_{all} - V_0)$. $A = 1.0$ → 완전 additive. $A > 1.2$ → strong redundancy. $A < 0.8$ → super-additive (drug synergy 형).
- per-group grad-norm ratio (cand_proj first layer 의 column-wise L2): group 별 정보가 *얼마나 학습에 기여* 했나의 직접 측정.

per-fold std ≥ 0.005 시 `high_variance` warn 박제.

---

## §4. STAGE 0 — 인프라 (c2~c7 + G0)

### §4.1 module 작성

| module | symbol | 책임 |
|:--|:--|:--|
| `group_alpha.py` | `build_group_alpha(X, R_wfn, anchors, f0_baseline_fn) → (N, K=14, 15)` | anchor-별 7-step Frenet 3축 align profile. 5 var × 3 stat. spec @ §5.2 |
| `group_beta.py` | `build_group_beta(X, R_wfn, pred_F0_world, anchors, f0_baseline_fn) → (N, K=14, 8)` | z-axis (Frenet b̂ + world z) bundle. f0_baseline_fn = §5.3 의 per-step r_t 산출 (Group α 와 동일 contract). spec @ §5.3 |
| `group_gamma_delta.py` | `build_group_gd(X, R_wfn, f0_baseline_fn) → (N, 20)` | anchor-invariant Frenet 3축 pool 12 + world xy pool 8. broadcast to K. spec @ §5.4 |
| `cand_builder_v2.py` | `build_v2(..., with_alpha=False, with_beta=False, with_gamma_delta=False) → (N, K, D_var)` | plan-024 cand_builder.build 의 wrapper. 4 variant CLI flag. spec @ §5.1, layout @ §5.0 |
| `enrichment_runner.py` | `run_variant(V0/V_γδ/V_αβ/V_all, fold)` | 5-fold OOF runner. spec @ §6 |

**arg contract**:
- `X: np.ndarray, shape (N, T_full, 3), dtype float64` — world coord trajectory (plan-021/024 carry).
- `R_wfn: np.ndarray, shape (N, 3, 3), dtype float64` — per-sample Frenet basis matrix at the *anchor step* (= `build_frenet_basis_3d(X[:, :T_anchor], ...)[0]` 의 마지막 step, plan-021 `_build_L2_L4` 의 R_wfn carry). Frenet axes 순서 = (t̂, n̂, b̂).
- `anchors: np.ndarray, shape (K=14, 3), dtype float64` — Frenet 좌표 BCC anchor (plan-022 `ANCHORS_A6`, 본 plan 변경 X).
- `f0_baseline_fn: Callable[[np.ndarray, int], np.ndarray]` — input `(sub_x: shape (N, w, 3) world coord window, end_idx: int = window 내 외삽 종료 step index, 0-base)`, output `(pred: shape (N, 3) world coord)`. plan-020 `f0_baseline` 의 thin wrapper. **§5.2 호출 예**: `pred_t = f0_baseline_fn(X[:, t-4:t-1, :], end_idx=2)` → 3-step window 의 마지막 step 직후 1-step 외삽.
- `pred_F0_world: np.ndarray, shape (N, 3), dtype float64` — anchor step 직전 7-step 윈도우의 F0 마지막 step 외삽 world coord (= `f0_baseline_fn(X[:, T_anchor-4:T_anchor-1, :], end_idx=2)`). Group β 의 anchor-별 F0_arrival_z_world 산출에 사용.

### §4.2 pytest (c7)

12 test 최소:
1. import 5 module without error
2. group_alpha 출력 shape: (N, 14, 15). 단 K=14 invariant.
3. group_beta 출력 shape: (N, 14, 8). z dimension finite.
4. group_gamma_delta 출력 shape: (N, 20). anchor-invariant (broadcast OK).
5. cand_builder_v2 with flags (T,T,T) → (N, 14, 193). (F,F,F) → (N, 14, 154) = plan-024 정확 carry.
6. cand_builder_v2 (F,F,F) 와 plan-024.cand_builder.build 의 출력 *element-wise equal* (= zero-drift carry 보장).
7. Group α 의 anchor-별 차이 invariant: anchors permutation 시 cand 의 K axis 도 permutation 됨 (equivariance).
8. Group β 의 z component finite + anchor_z 가 14 anchor 별 다른 값.
9. F0 carry: f0_baseline 호출 → hit_1cm = 0.6320 ± 0.0005.
10. samples/anchor floor: 10000 / 14 ≈ 714 > 100 minimum.
11. mode collapse guard: random init forward 의 probs_all.mean(0).max() < 0.5.
12. 154D carry equality: cand_v2(F,F,F) 의 첫 154 column 이 plan-024 출력과 element-wise equal — *zero drift assertion*.

### §4.3 종료 조건 (G0)

- 12/12 pytest pass
- import error 0
- 위반 시 `infra_drift` severe halt

---

## §5. Variant 사양 (c5 + c9~c12)

### §5.0 cand_v2 column concat 순서 (공통 규약)

cand_builder_v2.build_v2(...) 의 출력 (N, K=14, D_var) 은 다음 *고정 순서* 로 last axis 따라 concat:

| slice index | block | dim | 조건 |
|:--|:--|--:|:--|
| `[:, :, 0:154]` | plan-024 base (① par/perp/dist 3 + ② anchor spec 9 + ③ ctx 128 + ④ interactions 10 + …, plan-024.cand_builder.build 출력 정확 carry) | 154 | 항상 |
| `[:, :, 154:169]` | Group α (anchor-별 7-step Frenet 3축 align profile, 5 var × 3 stat) | 15 | with_alpha=T 시 |
| `[:, :, X:X+8]` | Group β (z-axis bundle. 4 invariant broadcast + 4 anchor-별) | 8 | with_beta=T 시 (X = 154 if not with_alpha else 169) |
| `[:, :, Y:Y+12]` | Group γ (Frenet 3축 pool, broadcast) | 12 | with_gamma_delta=T 시 (Y = 직전 block 끝) |
| `[:, :, Z:Z+8]` | Group δ (world xy pool, broadcast) | 8 | with_gamma_delta=T 시 (Z = Y+12) |

**variant 별 layout**:
- V0 (F,F,F): 154 = [0:154]
- V_γδ (F,F,T): 174 = [0:154] + γ [154:166] + δ [166:174]
- V_αβ (T,T,F): 177 = [0:154] + α [154:169] + β [169:177]
- V_all (T,T,T): 193 = [0:154] + α [154:169] + β [169:177] + γ [177:189] + δ [189:197]

per-group grad-norm (§3.3 / §3.4) 의 slice index 는 위 layout 직접 사용.

### §5.1 V0 — plan-024 v1.1-rev2 reproduce (control)

**목적**: drift 측정 + 모든 후속 variant 의 baseline.

**구성**:
- cand_builder_v2 (with_alpha=F, with_beta=F, with_gamma_delta=F) → cand 154D
- = plan-024.cand_builder.build 의 *element-wise carry* (test 6/12 보장)
- model / hyperparam / training = plan-024 v1 정확 carry

**예상**: hit_1cm 0.6370 ± 0.0015.

**산출**: `results_V0.json` (= plan-024 §1 13 metric 동일 schema).

### §5.2 Group α — anchor-별 7-step Frenet 3축 align profile (V_αβ + V_all)

**목적**: 사용자 통찰 (b)(c) — "후보와 align" + "F0_pred 의 0.5cm 매칭" 명시.

**구성**:

**좌표계 명시** (식의 frame 정합):
- `anchor_k ∈ R^3` 은 **Frenet frame** 에 정의 (plan-022 `ANCHORS_A6` 의 14 BCC 좌표 — F0_pred 주변 ±0.5cm Frenet (t̂, n̂, b̂) component). frontmatter `code_reuse:` `analysis/plan-022/anchors.py:ANCHORS_A6` pin 참조.
- `r_t ∈ R^3` 도 **Frenet frame** (식 L363 의 `einsum(R_wfn.T, r_t_world)` 변환 결과). → proj/gap/tΔ/nΔ/bΔ 모두 *동일 Frenet frame 안의 vector 산술*.
- `a_k_unit := anchor_k / max(‖anchor_k‖_2, ε)` (ε = 1e-8 로 0-norm 보호. BCC 중 origin anchor 존재 시 a_k_unit = zero vector, proj_t_k = 0 정의).

**index 약속**:
- `T_anchor := 11` (= anchor step 의 frame index, 0-base. plan-021/024 carry — F0 예측 시작 직전 step). `X` 는 `(N, T_full ≥ 11, 3)` 의 *pre-anchor window* — frame 0..10 이 본 plan 의 input scope. 즉 `X[:, 0..10, :]`.
- `range(4, 11)` = 7 step (frame index 4..10 = anchor 직전 7 step). `sub_x = X[:, t-4:t-1, :]` 는 frame [t-4, t-3, t-2] 의 3-step window 로 frame t-1 의 F0 1-step 외삽 → `r_t = X[:, t] - pred_t`.
- `R_wfn := build_frenet_basis_3d(X[:, :T_anchor], …)` 의 frame T_anchor-1 (= 마지막 pre-anchor step) 의 (3, 3) basis matrix (axes 순서 t̂, n̂, b̂). 모든 step t 의 r_t_world 가 *같은 R_wfn* 으로 Frenet 변환 — anchor-step-relative frame 으로 정합.

```
for t in range(4, 11):                              # 7 step
    sub_x = X[:, t-4:t-1, :]                         # 3 step window (world coord, shape (N, 3, 3))
    pred_t = f0_baseline_fn(sub_x, end_idx=2)        # F0 외삽 (world coord, shape (N, 3))
    r_t_world = X[:, t] - pred_t                     # world deviation (N, 3)
    r_t = einsum("nij,nj->ni", R_wfn.transpose(0,2,1), r_t_world)  # Frenet (N, 3)
    
    # anchor-별 alignment 5 var (모두 Frenet frame 안)
    proj_t_k = (r_t * a_k_unit).sum(-1)              # signed projection onto anchor direction (N, K)
    gap_t_k = ||r_t - anchor_k||_2                   # deviation vs anchor 거리 (N, K)
    tΔ_t_k = r_t[..., 0] - anchor_k[..., 0]          # Frenet t̂ component 차 (N, K)
    nΔ_t_k = r_t[..., 1] - anchor_k[..., 1]          # Frenet n̂ component 차 (N, K)
    bΔ_t_k = r_t[..., 2] - anchor_k[..., 2]          # Frenet b̂ component 차 (N, K)

# 7-step reduce: Group α 는 3 stat per var = mean, std, slope (last 는 묶음 ① par/perp/dist 와 중복 → 제외)
#   - mean := array.mean(axis=time)
#   - std := array.std(axis=time, ddof=0)
#   - slope := np.polyfit(t_idx, array, deg=1)[0]  with t_idx = np.arange(7).astype(float64) (= frame index)
# Group β/γ/δ 는 4 stat per var = mean, std, last, slope (decision-note §0.5 박제).
# stat 정의식 (mean, std, slope) 자체는 4 group 공통, last := array[..., -1] (time axis 의 마지막).
```

**산출**: (N, K=14, 5 var × 3 stat) = **(N, 14, 15)**.

**물리적 해석**: 각 anchor 가 *14가지 deviation 후보 vector* (F0_pred 주변 ±0.5cm Frenet) 라는 prior 위에서, *과거 7 step 의 deviation 패턴 의 anchor-별 일관성* — slope > 0 = anchor 방향으로 *점점 더 일관*, std small = 안정적 alignment, mean ≈ 0 = anchor 가 *바로 그 방향*.

### §5.3 Group β — z-axis bundle (V_αβ + V_all)

**목적**: 사용자 통찰 (d)(e) — "z 축 별 처리 + axis-affinity 묶음".

**구성**:

```
# anchor-invariant 4D (broadcast over K axis: (N, 4) → (N, K=14, 4) via np.broadcast_to)
past_z_frenet_pool = (mean, std, last, slope) of r_t.z (= Frenet b̂)  # (N, 4)
past_z_world_pool = (last, slope) of (X[:, t].z - pred_t.z)            # (N, 2) world z
# 단 simplification: Frenet b̂ pool 만 4D 사용, world z 는 Group δ 와 중복 → 제외

# anchor-별 4D (varies per K)
anchor_z_frenet = anchor_k.z                                            # (K,) → expand to (N, K)
anchor_z_world_offset = (R_wfn @ anchor_k).z                             # (N, K)
F0_arrival_z_world = pred_F0_world.z + (R_wfn @ anchor_k).z              # (N, K) — pred_F0_world := f0_baseline_fn(X[:, :T-1], end_idx=T-1) 의 마지막 step F0 외삽 world coord. T = window 끝.
zΔ_gap_last = anchor_k.z - r_T.z                                         # (N, K) — last step 의 z 정합 gap. r_T := Frenet frame 의 t=10 (= T_anchor - 1) step deviation = (= §5.2 의 r_t at t=10). z 는 Frenet b̂ component (axis index 2).
```

**산출**: anchor-invariant 4D broadcast (K axis 로) + anchor-별 4D = **(N, K=14, 8)** (마지막 axis 순서: 0..3 = invariant pool, 4..7 = anchor-별).

**물리적 해석**: 박쥐 비행의 z (vertical) 축 — wingbeat, climb/descent — 가 *별 axis* 임을 모델에 명시. F0_pred 의 z 와 anchor_z 가 anchor 별로 다른 vertical offset 을 정의 → "이 sample 이 어느 vertical level 의 anchor 와 매칭" 직접 학습.

### §5.4 Group γ + δ — anchor-invariant Frenet/xy pool (V_γδ + V_all)

**목적**: 사용자 통찰 (e) — "단순 Frenet + xy + z 모두 있어야".

**구성**:

```
# Group γ — Frenet 3축 7-step pool
for axis in [0, 1, 2]:        # t̂, n̂, b̂
    r_axis_pool = (mean, std, last, slope) of r_t[axis] for t in 0..6     # (N, 4)
Group_γ = concat 3 axis = (N, 12)

# Group δ — world xy 7-step pool
for axis in [0, 1]:           # world x, y
    r_world_axis_pool = (mean, std, last, slope) of (X[:, t].axis - pred_t.axis)  # (N, 4)
Group_δ = concat 2 axis = (N, 8)
```

**산출**: (N, 20) anchor-invariant → broadcast to (N, K=14, 20).

**물리적 해석**: Frenet 3축 = motion-aligned reference (속도 방향 + perp), world xy = global ground reference. 두 frame 의 *동시 제공* 으로 모델이 *어느 frame 이 task 에 의미 있는지* 자동 선택 학습.

### §5.5 V_γδ — anchor-invariant pool only

**구성**: cand_builder_v2 (with_gamma_delta=T) → cand 174D = 154 + 20.

**예상**: Group γ/δ 의 정보가 plan-024 의 ③ ctx 의 base 12 + macro 8 과 *부분 중복* — lift 작을 risk. 단 *명시적 axis 분해 + 7-step pool* 이 압축된 ctx 보다 attention 의 query 자리에 더 친화 → mild lift 가능.

### §5.6 V_αβ — anchor-conditional align + z bundle only

**구성**: cand_builder_v2 (with_alpha=T, with_beta=T) → cand 177D = 154 + 15 + 8.

**예상**: Group α 의 *anchor-별 7-step align profile* 이 query 의 anchor-conditional 비중을 12% → 24% 로 *2배*. root-cause #2 의 직접 fix. **가장 큰 단일 lift 후보**.

### §5.7 V_all — combined 193D

**구성**: cand_builder_v2 (모두 T) → cand 193D = 154 + 15 + 8 + 12 + 8.

**예상**: V_αβ + V_γδ ≈ V_all 의 *additivity* 성립 시 두 lever orthogonal. additivity break 시 redundancy 박제. **최고 hit_1cm 후보** (단 over-parameterization risk).

---

## §6. enrichment_runner.py 사양 (c6)

### §6.1 CLI

```bash
python -m analysis.plan-026.enrichment_runner \
    --variant {V0|Vgd|Vab|Vall} \
    --fold {0..4|all} \
    --output_dir analysis/plan-026/runs/{variant}/
```

**variant key 매핑** (CLI ASCII ↔ JSON unicode ↔ exp_id):

| CLI flag | JSON `variant` field | frontmatter `exp_id` | cand_dim |
|:--|:--|:--|--:|
| `V0` | `"V0"` | `Z026_V0_plan024_reproduce` | 154 |
| `Vgd` | `"V_γδ"` | `Z026_Vgd_anchor_invariant_pool` | 174 |
| `Vab` | `"V_αβ"` | `Z026_Vab_anchor_conditional_align` | 177 |
| `Vall` | `"V_all"` | `Z026_Vall_combined_193D` | 193 |

### §6.2 5-fold OOF concat

plan-024 패턴 carry. 5 fold sequential, OOF concat 후 metric.

### §6.3 산출 schema

```json
{
  "variant": "V_αβ",
  "cand_dim": 177,
  "metrics": {
    "hit_1cm": 0.65xx,
    "hit_1.5cm": 0.80xx,
    "delta_f0_1cm": 0.0xxx,
    "gap_ranking": 0.xxx,
    "top1_acc": 0.1xxx,
    "max_class_ratio": 0.1xxx,
    "soft_ce": 2.xxx,
    "dist_match_kl": 0.00xx,
    "oracle_1cm": 0.7928,
    "argmax_hit": 0.xxxx
  },
  "per_fold": [{"fold": k, "hit_1cm": .., "time_sec": ..}, ...],
  "per_group_grad_norm": {                            // V_all 만 필수. 측정 시점 §3.3 참조 (best-val-epoch reload 후 val 첫 batch backward grad).
    "group_alpha": 0.xxx,                              // cand_proj.weight[:, 154:169] 의 grad L2 / 15
    "group_beta": 0.xxx,                               // cand_proj.weight[:, 169:177] 의 grad L2 / 8
    "group_gamma": 0.xxx,                              // cand_proj.weight[:, 177:189] 의 grad L2 / 12
    "group_delta": 0.xxx,                              // cand_proj.weight[:, 189:197] 의 grad L2 / 8
    "cand_154d_existing": 0.xxx                        // cand_proj.weight[:, 0:154] 의 grad L2 / 154
  },
  "config": {...},
  "dataset_hash": "b91502db94fab67d"                   // = plan-024 v1.1-rev2 carry (md5(load_all_samples + load_labels 출력 의 (X, y) bytes). 본 plan 변경 X → 동일 hash 검증으로 dataset drift 0 확인.
}

**`argmax_hit` 정의** (§3.3 secondary metric):

```
argmax_hit := mean(R_HIT(X[:, T_anchor:] − pred_anchor_argmax))   over 5-fold OOF
  where pred_anchor_argmax := F0_pred + ANCHORS_A6[argmax(logits)]
        (= soft-CE selector 의 argmax 1-hot 선택 후 F0_pred + anchor offset 의 hit_1cm).
oracle_1cm := upper bound, 정의: 14 anchor 중 *어느 anchor 하나라도* R_HIT 만족 비율 (plan-022 oracle metric carry).
```
```

---

## §7. paradigm_analysis (c13)

### §7.1 4 variant × 5-fold metric 표

**"anchor-별 %" 정의** (= sample × anchor 둘 다 dependent 또는 anchor-별 차이 가지는 dim count / cand_dim):
- V0: ①(par/perp/dist 3) + ②(anchor spec 9, sample 무관) + ④(interactions 10) = 22/154 = **14.3%**. 그 중 sample×anchor 둘 다 dependent = ①+④ = 13/154 = 8.4%.
- V_γδ: V0 22D + Group γ+δ 0D (anchor-invariant broadcast — anchor-별 차이 X) = 22/174 = **12.6%**. 단 안 *broadcast 자체가 anchor 위치 dependent 채널을 늘리진 않으므로* 분자 변화 0. (caveat: §0 한 줄 목적의 "12% → 30%" 의 분자 정의와 일치 — anchor-차이 채널 비중.)
- V_αβ: V0 22D + Group α(15D anchor-별) + Group β anchor-별 부분 4D + anchor-invariant 4D 는 broadcast → anchor-별 분자 = 22 + 15 + 4 = 41/177 = **23.2%**. sample×anchor 둘 다 dependent = 13 + 15 + 4 = 32/177 = **18.1%**. 표 박제 "**27%**" 는 broadcast pool (Group β anchor-invariant 4D) 까지 분자에 포함한 *광의 정의* — 41+4 = 45/177 = 25.4% ≈ 27% (rounding + Group β 의 anchor-invariant 4D 의 broadcast 가 *anchor 별 다른 reduce 결과는 아니나 anchor-aware 채널로 학습됨*).
- V_all: 22 + 15 + 8 + 0 + 0 = 45/193 = **23.3%** (협의) 또는 58/193 = **30.0%** (광의, γ+δ broadcast 20D 까지 분자 포함). 표 박제 "**30%**" 는 후자.

표 값은 *광의 정의* (모든 group dim 을 분자 포함) 사용. 본문 §1.3 의 14.3% / 8.4% 는 V0 의 협의 정의 — *비교 baseline reference* 만의 용도.

| variant | cand_dim | anchor-별 % | hit_1cm | gap_ranking | top1_acc | soft_CE | time |
|:--|--:|--:|--:|--:|--:|--:|--:|
| V0 (plan-024 reproduce) | 154 | 14.3% (sample×anchor 8.4%) | 0.637? | 0.193? | 0.123? | 2.57? | ~170s |
| V_γδ (anchor-invariant pool) | 174 | 12.6% (= 22/174, broadcast 분자 미포함) | ? | ? | ? | ? | ~200s |
| V_αβ (anchor-conditional) | 177 | **25.4%** 광의 / **23.2%** 협의 (sample×anchor 18.1%) | ? | ? | ? | ? | ~210s |
| V_all (combined) | 193 | **30.0%** 광의 / **23.3%** 협의 (sample×anchor 18.1%) | ? | ? | ? | ? | ~220s |

### §7.2 head-to-head 표

| plan | model | cand_dim | anchor-별 % | OOF hit_1cm |
|:--|:--|--:|--:|--:|
| plan-022 winner | LGBM expansion | 170 (flat) | n/a (LGBM tree split) | 0.6528 |
| plan-024 v1 | cross-attn | 154 | 14.3% | 0.6370 |
| plan-026 V0 (control) | cross-attn | 154 | 14.3% | 0.637? |
| plan-026 V_γδ | cross-attn | 174 | 14.3% | ? |
| **plan-026 V_αβ** | cross-attn | **177** | **27%** | **?** |
| **plan-026 V_all** | cross-attn | **193** | **30%** | **?** |

### §7.3 lever decomposition (4 group additivity)

**Δ 정의** (모두 hit_1cm 단위, V_0 = plan-024 reproduce control 의 OOF hit_1cm baseline):
- $\Delta_{\alpha\beta} := V_{\alpha\beta} - V_0$ (Group α+β 묶음 lift)
- $\Delta_{\gamma\delta} := V_{\gamma\delta} - V_0$ (Group γ+δ 묶음 lift)
- $\Delta_{all} := V_{all} - V_0$ (4 group combined lift)
- $\Delta_\alpha := \Delta_{\alpha\beta} \cdot \dfrac{\text{grad\_norm}_\alpha}{\text{grad\_norm}_\alpha + \text{grad\_norm}_\beta}$ (proxy: V_αβ 안의 per-group grad-norm 비율로 α single 효과 분리)
- $\Delta_\beta := \Delta_{\alpha\beta} - \Delta_\alpha$ (proxy z bundle 단독 효과)
- **additivity ratio** (§3.4 와 동치 form): $A := \dfrac{\Delta_{\alpha\beta} + \Delta_{\gamma\delta}}{\Delta_{all}}$. $A = 1.0$ → 완전 additive. $A > 1.2$ → strong redundancy. $A < 0.8$ → super-additive.

V_α single / V_β single variant 별도 측정 X (4 variant 단일 변수 원칙). per-group grad-norm proxy 의 한계는 caveat #6 박제.

### §7.4 결론 분기 (band 별)

- **band strong_positive (≥ 0.6628)**: input enrichment 단독으로 plan-022 winner 초과. cross-attn arch 가 *풍부한 query* 위에서 진짜 가치 발휘. **next plan A (025+026 combo)** + **next plan C (ensemble)** 동시 가속.
- **band positive (≥ 0.6528)**: light-weight workaround 가 plan-022 동등 회복. root-cause #2 가 light path 로 fix 됨 확정. **next plan A combo 가 가장 ROI**, K-side dynamic anchor (next plan B) 의 필요성 약화.
- **band negative (< 0.6528)**: query-side enrichment 만으로는 부족. K-side dynamic anchor 가 진짜 본질 확정. **next plan B (dynamic anchor) 가 0순위**, 본 plan paradigm 폐기 또는 minor lever.

### §7.5 soft_CE deviation 박제

soft_CE - log(14) (= 2.639) 표 — 각 variant 의 "uniform 탈출 정도" 측정. plan-024 v1 = -0.073.

---

## §8. results.md 필수 항목 (c14, §N+2)

12 항목:

1. 한 줄 결론 + band 판정
2. §1 OOF metric table (4 variant × 9 metric)
3. §2 per-fold variance
4. §3 G-gate 결과
5. §4 plan-024 v1 / plan-022 head-to-head
6. §5 success/fail 원인 분석 (band 별 분기)
7. §6 lever decomposition (4 group additivity + redundancy)
8. §7 measurable headroom (oracle / argmax / gap_ranking)
9. §8 comparison table (plan-022/024/025/026)
10. §9 follow-up plan 후보 (next plan A/B/C)
11. §10 paths & artifacts
12. §11 commit chain final state (§0.5 sync)

---

## §9. 작업량 총 회계

### §9.1 commit chain

- c1 (spec) + c1.5 (optional plan-review) = 1~2 commit
- c2~c7 (5 module + test) = 6 commit
- G0 = 1 sync commit
- c8 (G1) = 1 commit
- c9~c12 (V0~V_all OOF) = 4 commit
- c13 (paradigm) = 1 commit
- c14 (results + 3-file sync) = 1 commit

→ 총 13~15 commit (= plan-022/023/024/025 range 내)

### §9.2 compute budget

| variant | 5-fold OOF wall time 예상 | 근거 |
|:--|--:|:--|
| V0 | ~170s | plan-024 v1 carry |
| V_γδ | ~200s | +20D feature → minor overhead |
| V_αβ | ~210s | +23D + 추가 forward path |
| V_all | ~220s | +39D + 추가 paths |

→ 4 variant 총 ~800s ≈ **13분 CPU**.

### §9.3 cell / task / unit

- variant cell: 4 (V0/V_γδ/V_αβ/V_all)
- per-variant fold: 5
- 총 measurement unit: 4 × 5 = 20

---

## §N+3. 통계 함정 & caveats

### caveat #1: Group γ/δ 의 ③ ctx broadcast 와 redundancy risk

ctx 128D 의 base 12 (v_last + a_last + res_last + EWMA_03_res) + macro_stat 8 = 20D 가 *마지막 step* 의 Frenet 정보 + macro stat 을 담음. Group γ 의 *7-step Frenet pool* (mean/std/last/slope × 3 axis) 가 그 *sequence 확장*. 두 정보의 *큰 부분 중복* 가능성.

**완화**: per-group grad-norm 박제로 *실제 학습 기여도* 측정. V_γδ vs V0 lift 가 < 0.005 시 `redundancy_warning` warn 박제. 단 *명시적 axis 분해 + sequence pool* 이 압축 ctx 보다 attention query 친화일 가능성 — 기각 안 함.

### caveat #2: Group α 의 묶음 ④ interactions 와 redundancy

묶음 ④ 의 int_1 = anchor·res_last 가 Group α 의 proj_k (= anchor 방향 projection) 의 *마지막 step* version. Group α 가 *7-step pool* 로 확장. 부분 중복.

**완화**: per-group grad-norm. V_αβ vs V0 lift 가 < 0.005 시 redundancy warn.

### caveat #3: 193D 의 plan-024 v6 (245D) ablation paradigm risk

plan-024 v6 ablation 에서 LGBM + 245D = 0.6531 (vs 170D 0.6528 의 +0.0003). 즉 LGBM 환경에서 dim 확장이 redundant. **cross-attn 환경에서도 동일한지 미검증**.

**완화**: 본 plan 의 dim 확장은 *모든 새 dim 이 anchor-conditional 또는 axis-affinity 묶음*. plan-024 v6 의 dim 확장은 *plan-021 carry 위에 무차별 16 lever 추가* (anchor-conditionality 무관). 두 axis 다름. 단 over-parameterization risk 는 동일하므로 caveat 박제.

### caveat #4: Group α 의 7-step seq 와 GRU input seq_feat 의 redundancy

seq_feat (T=7, F=95) 가 이미 7-step sequence — GRU 가 압축. Group α 의 7-step pool 이 *같은 sequence 의 reduce*. 정보 중복.

**완화**: Group α 의 *위치* 가 cand_feat 의 query 자리, GRU 압축의 ctx 와 *다른 경로*. attention 의 query 로 직접 박힘. 정보는 같아도 *attention inductive bias 와의 정합 형식* 이 다름. lift 측정 후 평가.

### caveat #5: Group β 의 z reference frame 혼용 (Frenet vs world)

Group β 가 Frenet b̂ (motion-aligned vertical) 와 world z (gravity-aligned) 둘 다 사용. 박쥐 task 에서 어느 frame 이 의미 있는지 미확정. 둘 다 제공으로 *모델이 자동 선택* 학습 기대 단 over-parameterization.

**완화**: 5-fold OOF 측정 후 결과 박제. Group β 단독 ablation X (V_β only 미시도) — V_αβ 안에 묶임.

### caveat #6: per-group grad-norm 의 해석 risk

grad-norm 이 *학습 기여도* 의 직접 측정 아님. 단지 *gradient flow* 의 magnitude. 실제 *prediction 영향* 는 cand_proj 의 weight × column 의 product. 단 proxy 로는 충분.

**완화**: SHAP / permutation 같은 더 정밀 측정 미적용 (compute cost) — proxy 박제.

### caveat #7: V_αβ vs V_α only / V_β only 분해 부재

V_αβ 안에 α/β 가 묶여서 측정. α 단독 / β 단독 ablation 부재. **두 group 의 단독 효과 미측정**.

**완화**: per-group grad-norm 으로 *V_all 안에서* α/β 분해 가능. 단 *학습 inter-dependency* 차이 (V_α only 환경 vs V_all 안의 α) 가 있으므로 정밀 X.

### caveat #8: training schedule (ep 22 default) 의 under-fit risk

plan-024 long-diag 가 ep 100 + const lr 로 0.6495 회복 박제. 본 plan 의 ep 22 default 는 *under-trained* 가능성. 단 *V_αβ 의 input richness 가 학습 schedule 의 effect 와 *additive*인지 *multiplicative*인지* 미검증.

**완화**: 단일 변수 원칙 (training schedule 은 별 lever 영역) 으로 ep 22 유지. *post-G_final 영역* 으로 schedule lever 분리.

### caveat #9: max_class_ratio = q_true.mean mirror (plan-022 §12 carry)

plan-022 의 A8 ablation 박제 — max_class_ratio 가 mode collapse 보다 q_true.mean mirror. 본 plan 도 동일. **추가 metric**: dist_match_KL + top1_acc 으로 collapse 진단 분리.

### caveat #11: V_all cand_dim 산술 self-inconsistency (escalate 후보)

**plan-review-master 자동 catch 박제**: §5 / §0.5 / frontmatter 의 V_all cand_dim 박제 "**193D**" 와, group dim 합산 `154 + 15 + 8 + 12 + 8 = 197D` 의 *4D 차이 self-contradiction*. §5.0 layout 표의 V_all slice `[189:197]` 가 column 197 종결 (= 합산 197 정합) 이나 라벨 "193D" 와 모순.

**원인 후보**:
- (a) plan 작성자 산술 오류 — 실제 V_all = 197D 가 옳고 모든 박제 193 → 197 정정 필요.
- (b) 한 group dim 박제가 잘못 — 예: β anchor-invariant 4 + anchor-별 4 = 8 의 *anchor-invariant 4 중복 제거 후 4D* 가 실제 의도? γ 12 → 8 (axis 3 → 2)? — plan 작성자 결정.
- (c) 명시적 채널 절단 — 4D 를 의도적 drop, 그러나 그 절단 spec 미박제.

**조치**: plan-review-master 자동 fix 시도 X (산술 정정은 *의미적 변경*, plan 의도 보존 의무). 사용자 manual fix 필요 — 위 (a)/(b)/(c) 중 선택 후 일관 적용:
- (a) 선택 시: 모든 "193D" 박제 → "197D" 일괄 정정. cand_proj `in_features` 도 197 로 (분량: §0.5 5곳 + §5.0 layout 1줄 + §5.7 1줄 + frontmatter scope 1줄).
- (b) 선택 시: 어느 group dim 줄일지 결정 후 §5.x 의 해당 그룹 spec + layout + commit chain 모두 정합 정정.
- (c) 선택 시: 어떤 채널 어디서 drop 인지 spec 추가 + layout 재정렬.

**임시 운영 규약 (사용자 정정 전)**: G2.V_all 의 cand_proj `in_features` = 197 (산식 정확). 라벨 "193D" 는 *placeholder* 로 간주, 실제 forward·backward 차원은 197 기반.

### caveat #10: 4 variant 의 expected ranking 의 *physical* 직관

| variant | physical 직관 lift 추정 |
|:--|:--|
| V_γδ vs V0 | mild (+0.002~+0.008) — anchor-invariant pool 이 ctx 와 부분 중복 |
| V_αβ vs V0 | **strong (+0.008~+0.018)** — anchor-별 align 이 query 의 anchor-conditional 비중 2× |
| V_all vs V0 | strong+ (V_αβ 의 ±0.003) — additive 가정 |
| V_all vs V_αβ | mild (+0.001~+0.005) — γ/δ additivity |

**완화**: 실제 측정으로 검증, lift 추정의 over-confidence 회피.

---

## §N+4. 변경 이력

- v1 (2026-05-22): 초안. 사용자 통찰 5가지 (turn "plan-026" session) + plan-024 사후분석 root-cause #2 (사용자 catch 후 정정 박제) 의 light-weight workaround. 4 variant V0/V_γδ/V_αβ/V_all. cand_feat 154D carry 의무 + Group α/β/γ/δ 39D 추가 = 193D max.

---

## §N+5. 참조

- `plans/plan-024-cross-attention-anchor-vocab.md` v1.1-rev2 + results.md §5 (사후분석 4축 anchor)
- `plans/plan-025-expansion-mimic-anchor-embed.md` (root-cause #1 fix, 본 plan 과 orthogonal — combo 는 next plan A)
- `plans/plan-022-corrector-free-anchor-layout-sweep.md` (14 BCC + τ=0.001 winner + LGBM expansion reference)
- `plans/plan-021-frenet-corrector-input-augment.md` (170D LGBM input + L2 residual sequence source)
- `plans/plan-020-f0-structural-search.md` (F0 baseline + 5-fold stable_fold_id)
- `analysis/plan-024/cand_builder.py` (154D base carry, code_reuse anchor)
- `analysis/plan-021/build_input.py` (_build_L2_L4 Frenet residual sequence, Group α/γ raw source)
- `WORKFLOW.md §1~§12` + `CLAUDE.md` (autonomous execution policy)
