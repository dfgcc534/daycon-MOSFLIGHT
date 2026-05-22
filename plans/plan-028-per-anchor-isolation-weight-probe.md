---
plan_id: 028
version: 1
date: 2026-05-22 (Asia/Seoul)
status: draft
inspired_by:
  - 025 (1080D LGBM C1+C2 모두 hit_1cm=0.6320=F0 mode collapse — best=C1 0.6320/0.8033, band=negative, max_class_ratio≈1/14, oracle 회수율 79.72%. paradigm_analysis §4 4가설 (a) τ_cls / (b) sample-weight expansion / (c) subclass self-consistency / (d) 1058D broadcast / 22D per-anchor 비율 50:1 dominance 박제. (d) most likely.)
  - 022 (best A6_bcc14_tau001 → hit_1cm 0.6531 / hit_1p5cm 0.8108 — K=14 BCC + τ_cls=0.001 + 170D LGBM selector paradigm. 본 plan **이겨야 할 목표**.)
  - 020 (F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id MD5. 본 plan 의 lower-bound reference.)
  - 024 (14-anchor oracle 0.7928 ceiling 박제 + feature engineering module 6개 source. plan-025 가 carry 한 동일 module 본 plan 도 carry.)
code_reuse:
  - module: analysis/plan-025/build_feat_1080.py
    symbols: [build_feat_1080, build_block1, build_block2, build_block3, build_block4]
    reason: 1080D feature pipeline 그대로 carry. 본 plan 의 block ablation 은 build_feat_1080 output 의 *index slice* 로 구현 (재계산 불필요). plan-025 worktree (worktree-plan-025-spec, commit @plan-025 G2.C2 시점) 에서 cherry-pick.
  - module: analysis/plan-025/run_oof.py
    symbols: [run_oof_cell_1080]
    reason: 5-fold OOF runner. 본 plan 의 cell config 는 이 runner 위에 (a) input dim subset + (b) sample-weight expansion on/off flag 만 inject. plan-025 worktree 에서 cherry-pick.
  - module: analysis/plan-024/cand_builder.py
    symbols: [build_cand_feat]
    reason: block ②③ (ctx 128D + per-anchor 22D) builder. plan-025 carry 그대로. cherry-pick 대상 (c2 단계, plan-025 와 동일 path).
  - module: analysis/plan-024/seq_builder.py
    symbols: [build_seq_feat]
    reason: block ④ (95×7 raw, 8-stat 압축으로 760D) source. cherry-pick 대상.
  - module: analysis/plan-024/torsion_calc.py
    symbols: [build]
    reason: Frenet torsion τ scalar. seq_builder internal call.
  - module: analysis/plan-024/quantile_carry.py
    symbols: [QuantileCarry, build_train_quantiles, apply_quantiles]
    reason: train fold quantile carry (omega_p90, jerk_p90). cand_builder threshold 주입.
  - module: analysis/plan-024/multiwindow_trim_build.py
    symbols: [load_trim]
    reason: 144→60 Multi-window trim index.
  - module: analysis/plan-024/anchor_vocab.py
    symbols: [build_anchor_vocab]
    reason: seq_builder internal call.
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: row-expand LGBM K-class softmax + soft label 산식. K=14 BCC + τ=0.001 carry. 본 plan model 그대로 — single 변수 = input dim subset + sample-weight flag.
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6, LAYOUT_NAMES]
    reason: K=14 BCC anchor codebook.
  - module: analysis/plan-022/run_oof.py
    symbols: [run_oof_cell]
    reason: per-cell 5-fold OOF runner. plan-022 winner (G1.b reproduce) 용.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra]
    reason: 170D plan-022 input pipeline = block ① (170D).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline injection + paired Δ anchor + hit metric.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split.
followed_by:
  - (gap 채움 사후 결정 — 본 plan G3 결과에 따라 input enrichment 정착 / paradigm 전환 / ensemble / F0 ML 중 어느 방향이 next 인지 박제 예정)
scope: plan-022 carry (K=14 BCC anchor + τ_cls=0.001 + soft label + F0 + 5-fold split) + plan-025 carry (1080D feature pipeline = block ① 170D + ② 128D + ③ 22D + ④ 760D) 위, **단일 변수 두 축** = (1) input dim subset (block ablation 4 cell, 가설 d 검증) + (2) sample-weight expansion on/off (1 cell, 가설 b 검증) = **G2.A 5 cell**. G2.A 결과 branch 에 따라 **G2.B = 1~2 추가 cell** (selector/hparam tweak). plan-022 winner 0.6531 lift 가 G3 합격 기준. corrector reg head / GRU / cross-attention / DACON submit / ensemble / anchor 변경 / τ_cls 변경 / fold 변경 = out-of-scope.
exp_ids:
  - Z028_B1_anchor22
  - Z028_B2_combo192
  - Z028_B3_no_anchor1058
  - Z028_B4_full1080_ref
  - Z028_W1_weight_off
  - Z028_Bx_branch (G2.B conditional, 1~2 exp_id 추가 — branch 확정 후 박제)
lb_score: null
---

# plan-028 v1 — Per-anchor Isolation × Sample-weight Probe (mode collapse 진단 + plan-022 lift)

## §0. 한 줄 목적

> **plan-025 mode collapse 의 두 가설 — (d) 1058D broadcast / 22D per-anchor 비율 50:1 LGBM split dominance + (b) sample-weight expansion (140k row × 14-class CE) 비효율 — 을 G2.A 5 cell 로 직접 검증한 뒤, winning configuration 위에서 G2.B 1~2 cell 로 plan-022 winner (hit_1cm 0.6531, hit_1p5cm 0.8108) 를 paired Δ > 0 로 lift.**
>
> **분석 축 (G2.A)**: plan-025 1080D feature pipeline 의 block 분해 ① 170D / ② 128D / ③ 22D / ④ 760D 위 4 cell — B1 (③ 22D per-anchor only) / B2 (①+③ = 192D, plan-022 base + per-anchor) / B3 (①+②+④ = 1058D, ③ 제외) / B4 (1080D full = plan-025 C1 carry, reference) — 가설 (d) 직접 검증. 추가 1 cell W1 (1080D full + sample-weight expansion OFF) 으로 가설 (b) 검증. 총 G2.A 5 cell.
>
> **승부 축 (G2.B)**: G2.A 결과의 4 branch (α / β / γ / δ) 중 1 branch 활성화 → 1~2 cell 추가 실험. branch 결정 함수는 §4.5 박제. branch 활성화 우선순위 α > β > γ > δ (복수 branch 조건 만족 시 우선순위 높은 것 1개만 실행).
>
> **합격 기준 (G3)**: G2.A + G2.B 통합 best cell 의 hit_1cm > 0.6531 (= plan-022 winner) AND paired Δ vs plan-022 winner > 0 → **PASS (band=positive)**. 0.6320 < best ≤ 0.6531 → partial band (F0 초과 but plan-022 미달). best ≤ 0.6320 → negative band (plan-025 와 동일 = mode collapse 잔존).
>
> **plan-025 와 차별점**: plan-025 는 input dim 확장 lever, 본 plan 은 *동일 1080D 안에서의 subset / weight flag* lever. input pipeline 자체는 plan-025 carry (재계산 없음). single 변수 = (dim subset, weight flag, branch hparam) 중 cell 마다 하나만.
>
> **out-of-scope**: anchor layout 변경 (K=14 BCC fix) / τ_cls 변경 (0.001 fix) / fold 변경 / soft-label 산식 / F0 baseline ML화 / cross-attention / GRU / corrector head / 새 feature engineering / 1080D 외 새 dim / DACON submit / ensemble / plan-026 carry (worktree-only block ablation spec 의 carry 금지 — 본 plan self-contained).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 본 plan path (analysis/plan-028/**, tests/test_plan028_smoke.py) + cherry-pick path (analysis/plan-025/{build_feat_1080.py, run_oof.py}, analysis/plan-024/{6 module + 1 data + __init__}, analysis/plan-022/*, analysis/plan-021/build_input.py, analysis/plan-020/baseline_f0.py) import + smoke + tests green. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038]. plan-022 winner reproduce — A6_bcc14 + τ=0.001 cell hit_1cm ∈ [0.6526, 0.6536] AND hit_1p5cm ∈ [0.8103, 0.8113]. plan-025 C1 carry — hit_1cm ∈ [0.6315, 0.6325] (= mode collapse reference). 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` / `plan025_C1_drift` severe.
- **G2.A (5 cell)**: B1/B2/B3/B4/W1 각 5-fold OOF metric finite + `max_class_ratio` 측정 + soft label sum=1 invariant. 위반 시 `lgbm_numerical` severe.
- **G2.B (conditional, 1~2 cell)**: §4.5 branch 함수로 α/β/γ/δ 중 1 개 활성화, 해당 branch 의 1~2 cell 실행. branch 미정 (조건 모두 false) → δ default (selector arch MLP per-sample softmax) 1 cell.
- **G3 (paradigm-level)**: best_cell = argmax(hit_1cm over G2.A + G2.B 통합). best_hit_1cm > 0.6531 → PASS (band=positive). 0.6320 < best_hit_1cm ≤ 0.6531 → partial band (warn `partial_lift`). best ≤ 0.6320 → negative band (warn `regression`). 0.6526 ≤ best_hit_1cm ≤ 0.6536 = `tight_band_around_p022` 경계 — paired Δ 부호로 결정.
- **G_final**: results.md (11 항목 = plan-025 form 일치) + best cell 박제 (cell_id + hparam + 모든 metric + max_class_ratio + top1_acc + paired Δ vs F0/plan-022/plan-025-C1) + 14-anchor oracle 회수율 (= best / 0.7928) + paradigm_analysis (가설 a/b/c/d 중 어느 것이 확정/기각됐는지 박제) + follow-up plan 후보 ≥ 2 건 + 3-file frontmatter sync.

### G-gates (commit 단위 milestone)

- G0: STAGE 0 인프라 + cherry-pick + tests [TODO]
- G1: STAGE 1 F0 + plan-022 winner + plan-025 C1 carry reproduce [TODO]
- G2.A: STAGE 2.A 5 cell (B1/B2/B3/B4/W1) [TODO]
- G2.B: STAGE 2.B conditional branch 1~2 cell [TODO]
- G3: STAGE 3 paradigm + best cell 박제 [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-028-per-anchor-isolation-weight-probe.md` v1 작성 (plan-review-master 자동 fix BLOCKER 0 도달) | [TODO] |
| c2 | chore | plan-025 worktree (`worktree-plan-025-spec`) 에서 `analysis/plan-025/{build_feat_1080.py, run_oof.py, __init__.py}` 3 file cherry-pick → 본 worktree `analysis/plan-025/` (post-cherry-pick read-only). plan-024 6 module + 1 data + __init__.py 는 main 에 머지된 상태가 아니므로 동일 worktree 에서 cherry-pick (= plan-025 c2 와 동일 source). | [TODO] |
| c3 | code | `analysis/plan-028/build_feat_subset.py` — `build_feat_1080` output 의 *index slice* 함수 4개: `slice_B1_anchor22(X)` / `slice_B2_combo192(X)` / `slice_B3_no_anchor1058(X)` / `slice_B4_full1080(X)` 그리고 `weight_flag(on=True/False)`. recomputation 금지 — slice only. | [TODO] |
| c4 | code | `analysis/plan-028/run_oof_subset.py` — plan-025 `run_oof_cell_1080` carry 위에 (a) input slice fn 주입 + (b) sample_weight expansion flag (on/off). CLI: `--cell {B1, B2, B3, B4, W1}` + 향후 branch cell 추가. | [TODO] |
| c5 | test | `tests/test_plan028_smoke.py` (≥ 8 pytest: import / 4 slice dim check (22, 192, 1058, 1080) / sample_weight on/off shape / LgbmSelectorOnly K=14 fit/predict smoke (subset dim) / F0 carry / soft label sum=1 / branch decision fn unit test) | [TODO] |
| G0 | gate | smoke + tests green | [TODO] |
| c6 | exp G1 | F0 baseline + plan-022 winner A6_bcc14_tau001 + plan-025 C1 (1080D full carry, sample-weight ON) reproduce. `analysis/plan-028/baseline_carry.json` 박제 (dataset_hash + 3 carry hash). | [TODO] |
| G1 | gate | F0 ∈ tight ✓ AND plan-022 winner ∈ tight ✓ AND plan-025 C1 ∈ tight ✓ | [TODO] |
| c7 | exp G2.A.B1 | Cell B1 (22D per-anchor only, sample-weight ON, hparam = plan-022 default) 5-fold OOF. `results_B1.json`. 예상 runtime ~3min (dim 작아서). | [TODO] |
| c8 | exp G2.A.B2 | Cell B2 (192D = block ①+③, sample-weight ON, hparam = plan-022 default) 5-fold OOF. `results_B2.json`. 예상 ~3min. | [TODO] |
| c9 | exp G2.A.B3 | Cell B3 (1058D = no block ③, sample-weight ON, hparam = plan-022 default) 5-fold OOF. `results_B3.json`. 예상 ~5min. | [TODO] |
| c10 | exp G2.A.B4 | Cell B4 (1080D full, sample-weight ON, hparam = plan-022 default = plan-025 C1) 5-fold OOF. **plan-025 C1 carry 와 동일 — c6 G1 결과 그대로 박제** (재실행 안 함). `results_B4.json`. | [TODO] |
| c11 | exp G2.A.W1 | Cell W1 (1080D full, sample-weight **OFF**, hparam = plan-022 default) 5-fold OOF. `results_W1.json`. 예상 ~5min. | [TODO] |
| G2.A | gate | B1/B2/B3/B4/W1 metric finite ✓ + max_class_ratio 박제 ✓ + paired Δ vs plan-025 C1 (= B4) per cell 박제 | [TODO] |
| c12 | analysis | G2.A 5 cell 표 + 가설 (d) verdict (B1 vs B3 / B2 vs B4 비교) + 가설 (b) verdict (W1 vs B4 비교) + branch 함수 (§4.5) 실행 결과 = α / β / γ / δ 중 어느 1 branch activate + 박제 → `paradigm_analysis_g2a.json` | [TODO] |
| c13 | exp G2.B.cell1 | activated branch 의 cell 1 (각 branch §4.5 정의 따라) 5-fold OOF. `results_Bx_1.json` | [TODO] |
| c14 | exp G2.B.cell2 | activated branch 의 cell 2 (있을 시) 5-fold OOF. `results_Bx_2.json`. branch δ 는 1 cell only — c14 skip 박제. | [TODO] |
| G2.B | gate | branch cell metric finite ✓ + max_class_ratio 박제 ✓ + paired Δ vs plan-022 winner per cell 박제 | [TODO] |
| c15 | analysis | G2.A + G2.B 통합 best_cell selection (tiebreaker: hit_1cm > paired Δ_p022 > runtime) + paired Δ vs F0/plan-022/plan-025-C1 + 14-anchor oracle 회수율 + 가설 a/b/c/d verdict 통합 → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | best_hit_1cm > 0.6531 → PASS / 0.6320 < best ≤ 0.6531 → partial_lift warn / best ≤ 0.6320 → regression warn | [TODO] |
| c16 | docs | 3-file frontmatter sync (status=all_complete, band=positive/partial/negative, best_cell) + `analysis/plan-028/results.md` (plan-025 form 11 항목) + `plans/plan-028-*.results.md` pair + follow-up ≥ 2 건 | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c16 모두 [DONE] ✓ + follow-up ≥ 2 건 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 F0 reproduce 가 plan-020/021/022/023/025 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt.
- `plan022_reproduce_drift`: G1 plan-022 winner reproduce (A6_bcc14 + τ=0.001) 가 hard evidence 0.6531 / 0.8108 ±0.0005 밖. → halt.
- `plan025_C1_drift`: G1 plan-025 C1 carry (1080D full + sample-weight ON + hparam default) reproduce 가 plan-025 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt. (= G2.A.B4 base reference)
- `lgbm_numerical`: G2.A/G2.B 어느 cell LGBM 출력 NaN/Inf. → halt.
- `soft_label_collapse`: cell 의 selector probs 가 단일 anchor 95% 이상 mass (`max_class_ratio > 0.95`). warn (severe 아님). 5+ cell drop = `soft_label_collapse_total` severe escalate.
- `slice_dim_mismatch`: c3 slice fn output dim ≠ {22, 192, 1058, 1080} per 함수. → halt.
- `branch_undefined`: c12 branch 함수가 α/β/γ/δ 어느 것도 activate 안 됨 — δ default 로 fallback (severe 아님, decision-note 박제).
- `weight_flag_silent`: W1 cell 의 sample_weight 가 실제로 expansion off 됐는지 (= 10000 sample × 1 weight, NOT 140000 row × per-row weight) self-check fail. → halt.
- `tight_band_around_p022`: G3 best_hit_1cm ∈ [0.6526, 0.6536] (= plan-022 winner ±0.0005). paired Δ 부호로 결정 — Δ > 0 → positive, Δ ≤ 0 → partial. warn 박제.
- `partial_lift`: G3 best ∈ (0.6320, 0.6531]. F0 초과 but plan-022 미달. warn 박제 후 G_final (band=partial).
- `regression`: G3 best ≤ 0.6320. plan-025 mode collapse 잔존. warn 박제 후 G_final (band=negative).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-028/**`
  - `tests/test_plan028_smoke.py`
  - `analysis/plan-025/{build_feat_1080.py, run_oof.py, __init__.py}` — **c2 cherry-pick 단계 유일한 plan-025 path 수정 허용** (file add only, post-c2 수정 금지)
  - `analysis/plan-024/{__init__.py, anchor_vocab.py, cand_builder.py, seq_builder.py, torsion_calc.py, quantile_carry.py, multiwindow_trim_build.py, multiwindow_trim.json}` — c2 cherry-pick 단계 유일한 plan-024 path 수정 허용 (plan-025 와 동일 패턴)
- blacklist:
  - `runs/baseline/**`
  - `analysis/plan-{001..027}/**` (단, c2 cherry-pick 으로 *추가* 된 plan-024/plan-025 path 는 read-only import 만 허용)
  - `plans/plan-{001..027}-*.md` (수정 금지)
- 참조 (read-only):
  - `analysis/plan-025/{build_feat_1080.py, run_oof.py}` (cherry-pick 후 read-only)
  - `analysis/plan-024/{cand_builder.py, seq_builder.py, torsion_calc.py, quantile_carry.py, multiwindow_trim_build.py, multiwindow_trim.json, anchor_vocab.py}` (동)
  - `analysis/plan-022/{selector_only_model.py, anchors.py, run_oof.py, baseline_carry.json}`
  - `analysis/plan-021/build_input.py`
  - `analysis/plan-020/{baseline_f0.py, baseline_oof.json}`
  - `src/{io.py, pb_0_6822/selector.py}`

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — c2 cherry-pick 은 worktree-plan-025-spec branch 의 latest commit (= G2.C2 시점) 에서 plan-025 3 file + plan-024 8 file 만. cherry-pick command: git checkout worktree-plan-025-spec -- analysis/plan-025/<file> analysis/plan-024/<file>`
- `decision-note: spec-default — block ablation 은 build_feat_1080 output 의 *index slice* 로 구현 (재계산 금지). slice index 는 §4.2 표 정의 따라 hard-coded.`
- `decision-note: spec-default — K=14 BCC anchor + τ_cls=0.001 + soft label + 5-fold + F0 = plan-022/025 carry fix. 변경 X.`
- `decision-note: spec-default — G2.A B4 (1080D full + sample-weight ON) 는 plan-025 C1 carry 와 *완전 동일* 입력/하이퍼파라미터 → c10 단계에서 G1 결과 그대로 박제, 재실행 금지 (runtime 절약).`
- `decision-note: spec-default — G2.A 5 cell hparam = plan-022 default (n_estimators=500, lr=0.05, num_leaves=63, random_state=20260522, sample_weight expansion 은 W1 만 OFF, 나머지 ON). single 변수 = (input slice OR weight flag).`
- `decision-note: spec-default — sample_weight expansion ON = plan-022/025 default (row 별 soft label 값 × N_anchor expansion = 140000 row, weight=1 per row). OFF = 10000 sample × 1 row, soft label 그대로 input 으로 1 row 당 14-class CE. row-expand 자체는 plan-022 LgbmSelectorOnly 의 fit signature 이므로 W1 cell 에서는 row-expand 끄거나 weight=soft_label 그대로 inject 두 가지 중 후자 채택 (LgbmSelectorOnly 의 fit_no_expand=True 옵션 추가).`
- `decision-note: spec-default — branch 함수 (§4.5) 가 복수 조건 만족 시 우선순위 α > β > γ > δ 로 1 branch only.`

---

## §1. 배경

### §1.1 plan-025 mode collapse 의 4 가설 (paradigm_analysis §4 박제)

| 가설 | 정의 | 본 plan 검증 cell | likelihood |
|:--|:--|:--|:--|
| (a) τ_cls=0.001 sharp soft label train/test gap | τ 너무 sharp 라 train fold 의 soft label 이 test fold 에서 generalize 안됨 | 본 plan 검증 X — τ_cls=0.001 fix (plan-022 winner carry) | 중 |
| (b) sample-weight expansion (140k row × 14-class CE) 비효율 | row-expand 후 row 별 weight=soft_label 로 fit. LightGBM 의 weight 처리가 14-class objective 와 충돌 가능. | **W1 (sample-weight OFF)** | 중 |
| (c) LgbmSelectorRowExpanded subclass self-consistency 약화 | row-expand subclass 의 fit/predict path 가 14-row 일관성 깨짐 | 본 plan 검증 X — model class fix (LgbmSelectorOnly carry) | 낮 |
| (d) 1058D broadcast / 22D per-anchor 50:1 dominance | broadcast feature 가 LGBM split gain 에서 per-anchor 22D 묻음 → row-discriminative 신호 못 잡음 | **B1 (22D only), B2 (192D = ①+③), B3 (1058D = no ③)** | **높 (most likely)** |

본 plan 의 분석 축 = (b) + (d) 동시 검증. (a) / (c) 는 spec 의도 보존 위해 별도 plan 으로 분리 (followed_by 후보).

### §1.2 plan-022 winner 가 본 plan 의 *경기 상대*

- plan-022 best A6_bcc14_tau001: hit_1cm = 0.6531, hit_1p5cm = 0.8108
- plan-022 winner 는 block ① 170D 만 사용 (broadcast feature 도 17D 정도 포함 — 즉 170D 안에서도 broadcast/per-anchor 비율이 plan-025 만큼 극단 아님)
- B2 (192D = block ① 170D + block ③ 22D) 가 가장 직접적인 plan-022 winner + ε 후보 cell
- B2 가 0.6531 + ε 이상이면 (d) 가설 확정 (= per-anchor 22D 가 보태질 때 lift), 0.6531 미달이면 broadcast feature 가 도리어 plan-022 winner 보다 noise 였다는 hint

### §1.3 14-anchor oracle ceiling 0.7928 의 의미

- plan-024 박제 (carry): test fold 각 sample 마다 14-anchor 중 *진짜* 최선의 anchor 를 oracle 이 골라줬을 때의 hit_1cm
- plan-022 winner 회수율 = 0.6531 / 0.7928 = 82.38%
- plan-025 C1 회수율 = 0.6320 / 0.7928 = 79.72% (= F0 동일 = mode collapse)
- 본 plan G3 PASS 시 회수율 ≥ 82.38% 진입. stretch (0.6700) = 84.5%. 100% = oracle 자체.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|:--|:--|
| 입력 feature pipeline | plan-025 1080D carry (build_feat_1080) — 재계산 없음, slice only |
| 입력 dim 변수 (G2.A 분석 축) | {22 (B1=③), 192 (B2=①+③), 1058 (B3=①+②+④), 1080 (B4=full)} = 4 cell |
| sample-weight 변수 (G2.A 분석 축) | {ON (default), OFF} — OFF cell = W1 (1080D + OFF) = 1 cell |
| Total G2.A | 5 cell |
| Anchor | K=14 BCC (A6_bcc14, ANCHORS_A6) fix |
| τ_cls | 0.001 fix |
| Soft label 산식 | plan-022 build_soft_label_with_tau carry |
| Model | plan-022 LgbmSelectorOnly carry (LGBM K-class softmax + row-expand 또는 no-expand flag) |
| LGBM hparam | plan-022 default (n_est=500, lr=0.05, num_leaves=63, rs=20260522) — G2.A 5 cell 동일. G2.B branch α/γ 만 hparam tweak. |
| Fold | plan-020/021/022 stable_fold_id 5-fold carry |
| F0 baseline | plan-020 carry, paired Δ anchor |
| G2.B branch | §4.5 정의 함수로 α/β/γ/δ 중 1 개 activate, 1~2 cell 추가 |
| Hit metric | hit_1cm (primary, R_HIT=0.01m), hit_1p5cm (secondary, R_HIT_LOOSE=0.015m) |
| 평가 | 5-fold OOF concat (plan-020 carry) |
| 합격 기준 | best_hit_1cm > 0.6531 AND paired Δ vs plan-022 winner > 0 |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|:--|:--|
| Anchor layout 변경 (K ≠ 14, BCC 외 codebook) | plan-022 winner carry — paradigm fix, 본 plan 변수 X |
| τ_cls 변경 | plan-022 winner carry, 본 plan 변수 X. 가설 (a) 는 별도 plan. |
| Fold 변경 | plan-020 stable_fold_id 5-fold carry |
| Soft label 산식 변경 | plan-022 build_soft_label_with_tau carry |
| F0 baseline 자체 변경 (ML화) | 본 plan 변수 X. 별도 followed_by 후보. |
| 새 dim 추가 (block ⑤ 등) | plan-025 1080D fix. subset 만 허용. |
| Cross-attention / GRU / corrector reg head | plan-024 검증 종료 / plan-021 dead lever |
| DACON submit | LB 측정 본 plan 변수 X (G3 metric = OOF only) |
| Ensemble (plan-022 + plan-028 등) | 별도 plan (ensemble = follow-up 후보) |
| Anchor radius ≠ 0.005m | hit_1cm 기준 fix |
| plan-026 (worktree-only) carry | spec isolation — 본 plan self-contained, plan-026 spec 도 별도 paradigm |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- 5-fold stable_fold_id (`src/pb_0_6822/selector.py:stable_fold_id`, MD5 기반)
- plan-020/021/022/023/025 carry — 변경 X
- inner-fold (early_stopping 등) = 본 plan 사용 안 함 (G2.A 5 cell hparam fix, G2.B branch δ 만 selector arch 변경 시 inner-fold 가능)

### §3.2 합격 기준

| 조건 | 정의 | 결과 band |
|:--|:--|:--|
| A (PASS) | best_hit_1cm > 0.6531 AND paired Δ vs plan-022 winner > 0 | positive |
| B (tight_band) | best_hit_1cm ∈ [0.6526, 0.6536] | tight (paired Δ 부호로 final 결정) |
| C (partial) | 0.6320 < best_hit_1cm ≤ 0.6531 (B 범위 밖) | partial (warn `partial_lift`) |
| D (negative) | best_hit_1cm ≤ 0.6320 | negative (warn `regression`) |

paired Δ 정의: per-sample 같은 fold split 위 hit_1cm 차이의 평균.

### §3.3 평가 점수 / median 집계

- per-fold hit_1cm/1p5cm 계산 후 5-fold OOF concat → 단일 hit_1cm
- 5-fold 분산 표시 (std, min, max) but median 아닌 concat 평균
- max_class_ratio = `probs_all.mean(axis=0).max()` (5-fold concat 위)
- top1_acc = `mean(argmax(probs) == hard_label)` (5-fold concat 위, hard label = K=14 BCC argmax of soft label)

---

## §4. STAGE 정의

### §4.1 STAGE 0 (G0) — 인프라

c1 (spec) → c2 (cherry-pick) → c3 (slice fn) → c4 (runner) → c5 (tests) → G0 gate.

산출물:
- `analysis/plan-028/__init__.py`
- `analysis/plan-028/build_feat_subset.py` (slice fn 4개 + weight_flag)
- `analysis/plan-028/run_oof_subset.py` (CLI: `--cell {B1,B2,B3,B4,W1,Bx_*}`)
- `tests/test_plan028_smoke.py` (≥ 8 pytest)
- cherry-pick (read-only): plan-025 3 file + plan-024 8 file

G0 종료 조건: pytest 12+ pass + import 정상.

### §4.2 STAGE 1 (G1) — Baseline carry

c6: F0 + plan-022 winner + plan-025 C1 3 carry reproduce.

baseline_carry.json:
```json
{
  "f0": {"hit_1cm": 0.6320, "hit_1p5cm": 0.8033},
  "plan022_winner": {"hit_1cm": 0.6531, "hit_1p5cm": 0.8108, "cell": "A6_bcc14_tau001"},
  "plan025_C1": {"hit_1cm": 0.6320, "hit_1p5cm": 0.8033, "max_class_ratio": 0.0714, "input_dim": 1080},
  "dataset_hash": "<sha256>",
  "fold_seed": "stable_fold_id MD5"
}
```

G1 tight band:
- F0: hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038]
- plan-022 winner: hit_1cm ∈ [0.6526, 0.6536] AND hit_1p5cm ∈ [0.8103, 0.8113]
- plan-025 C1: hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038] (= F0 동일)

### §4.3 STAGE 2.A (G2.A) — 5 cell (block ablation × sample-weight)

| Cell | Input | Sample-weight | Dim | 변경 변수 | 가설 매핑 |
|:--|:--|:--|--:|:--|:--|
| **B1** | block ③ only (per-anchor 22D) | ON | 22 | broadcast 완전 제거 | (d) most aggressive — broadcast 없이 22D 가 살아남는가 |
| **B2** | block ①+③ (plan-022 base 170D + per-anchor 22D) | ON | 192 | plan-022 winner + per-anchor 추가 | (d) lift 가장 likely cell — plan-022 winner 0.6531 + ε 후보 |
| **B3** | block ①+②+④ (no ③) | ON | 1058 | per-anchor 완전 제거 | (d) — per-anchor 없으면 mode collapse 회복? |
| **B4** | 1080D full | ON | 1080 | = plan-025 C1 carry (reference) | (d) baseline + (b) baseline. c6 G1 결과 그대로 박제, 재실행 금지. |
| **W1** | 1080D full | **OFF** | 1080 | sample-weight expansion off | (b) — weight expansion 이 mode collapse 원인인가 |

block ① / ② / ③ / ④ slice index (`build_feat_subset.py` 박제):
- block ① indices [0:170] (plan-022 build_input_lgbm_extra output)
- block ② indices [170:298] (cand_builder ctx broadcast 128D, 14 row 동일)
- block ③ indices [298:320] (cand_builder per-anchor 22D, 14 row 각 다름)
- block ④ indices [320:1080] (seq_builder 8-stat 760D, 14 row 동일)

slice fn output:
- `slice_B1_anchor22(X[N, 14, 1080]) → X[N, 14, 22]` = X[:, :, 298:320]
- `slice_B2_combo192(X) → X[:, :, np.r_[0:170, 298:320]]`
- `slice_B3_no_anchor1058(X) → X[:, :, np.r_[0:298, 320:1080]]`
- `slice_B4_full1080(X) → X[:, :, :]`

sample-weight ON / OFF 산식:
- ON (plan-022/025 default): `row-expand` → 각 sample 의 14 row 각자 weight=soft_label[anchor_idx] 로 fit. row 수 = N × 14 = 140,000.
- OFF (W1 only): `no-expand` → 각 sample 의 14 row 를 그대로 14-class objective 의 multi-output fit (LightGBM `multiclass` objective, label = hard argmax of soft label, weight=1). row 수 = N = 10,000.

W1 의 fit signature 차이는 `LgbmSelectorOnly.fit(..., expand=False)` flag 추가로 처리 (plan-025 carry runner 에 옵션 inject). predict signature 는 변경 X.

c7~c11 commit 단계마다 cell 1개씩 OOF 측정 후 `results_<cell>.json` 박제 (per-fold hit/1p5cm + concat hit + max_class_ratio + top1_acc + runtime + paired Δ vs B4/plan-022).

### §4.4 STAGE 2.B (G2.B) — Conditional branch (1~2 cell)

§4.5 branch 함수로 α/β/γ/δ 중 1 개 activate. c13~c14 단계에서 해당 branch 의 cell 실행.

| Branch | 활성 조건 | Cell 구성 (1~2 cell) | 비고 |
|:--|:--|:--|:--|
| **α** (input dim sweet spot) | B2 ≥ 0.6531 OR (B2 > plan-022 winner - 0.005 AND B2 > B4 + 0.005) | α1: B2 (192D) + LGBM hparam tweak (n_est=2000 + lr=0.02 + feature_fraction=0.7), α2: B2 (192D) + num_leaves=127 + min_data_in_leaf=10 | 2 cell |
| **β** (per-anchor 22D 단독 회복) | B1 > B3 + 0.005 AND B1 < 0.6531 (= 22D 가 broadcast 없을 때 살아남지만 plan-022 미달) | β1: B2 (192D) + selector arch = plan-022 LgbmSelectorOnly + 추가 22D feature normalization (z-score per fold), β2: B1 (22D) + LGBM hparam tweak (num_leaves=31, n_est=2000) | 2 cell |
| **γ** (sample-weight 가 진짜 원인) | W1 > B4 + 0.005 | γ1: W1 (1080D + weight OFF) + LGBM hparam tweak (lr=0.02 + n_est=2000), γ2: B2 (192D) + weight OFF | 2 cell |
| **δ** (default — 모든 cell ≤ B4 + 0.003) | 모든 G2.A cell이 B4 baseline 0.6320 + 0.003 = 0.6323 이하 (mode collapse 잔존) | δ1: B2 (192D) + selector arch 변경 — LGBM → 작은 MLP per-sample softmax (hidden=64, depth=2, lr=1e-3, epoch=50, CPU). only 1 cell. | 1 cell |

branch 우선순위: α > β > γ > δ. 복수 branch 조건 만족 시 우선순위 높은 것만 activate.

decision-note 박제 의무: c12 commit msg 에 activate 된 branch + 활성 조건 만족 cell 수치 명시.

### §4.5 Branch 결정 함수 (의사 코드)

```python
def decide_branch(B1, B2, B3, B4, W1) -> Literal["α", "β", "γ", "δ"]:
    """
    G2.A 5 cell hit_1cm 입력 → 1 branch activate.
    우선순위: α > β > γ > δ.
    """
    P022 = 0.6531  # plan-022 winner

    # α: input dim sweet spot
    if B2 >= P022 or (B2 > P022 - 0.005 and B2 > B4 + 0.005):
        return "α"

    # β: per-anchor 22D 단독 회복
    if B1 > B3 + 0.005 and B1 < P022:
        return "β"

    # γ: sample-weight 가 진짜 원인
    if W1 > B4 + 0.005:
        return "γ"

    # δ: default — mode collapse 잔존
    return "δ"
```

`tests/test_plan028_smoke.py` 에 4 case unit test 박제.

### §4.6 STAGE 3 (G3) — Paradigm + best_cell

c15: G2.A + G2.B 통합 best_cell selection.

tiebreaker:
1. hit_1cm 최대
2. tie 시 paired Δ vs plan-022 winner 최대
3. tie 시 runtime 최소

박제:
```json
{
  "best_cell": "<cell_id>",
  "best_hit_1cm": <float>,
  "best_hit_1p5cm": <float>,
  "paired_delta": {
    "vs_f0": {"hit_1cm": <>, "hit_1p5cm": <>},
    "vs_p022": {"hit_1cm": <>, "hit_1p5cm": <>},
    "vs_p025_C1": {"hit_1cm": <>, "hit_1p5cm": <>}
  },
  "oracle_recovery": <best_hit_1cm / 0.7928>,
  "hypothesis_verdict": {
    "(b) sample_weight": "<confirmed|rejected|inconclusive>",
    "(d) broadcast_dominance": "<confirmed|rejected|inconclusive>"
  },
  "band": "<positive|partial|negative|tight>"
}
```

가설 verdict 함수:
- (d) confirmed: B1 > B3 + 0.005 OR B2 > B4 + 0.005 (per-anchor 가 broadcast 보다 lift 줌)
- (d) rejected: B2 < B4 - 0.003 AND B1 < B3 - 0.003 (per-anchor 가 도리어 noise)
- (d) inconclusive: 위 외
- (b) confirmed: W1 > B4 + 0.005
- (b) rejected: W1 < B4 - 0.003
- (b) inconclusive: 위 외

### §4.7 STAGE 4 (G_final) — Results

c16:
- frontmatter sync: spec + results pair + analysis/plan-028/results.md → status=all_complete, band, best_cell, best_hit_1cm, best_hit_1p5cm, best_delta_1cm (vs F0), best_delta_1p5cm (vs F0), g{1,2,3,_final}_completed=true, exp_ids_completed/skipped
- `plans/plan-028-*.results.md` (plan-025 form 11 항목)
- `analysis/plan-028/results.md` (plan-025 form, G2.A 5 cell 표 + G2.B branch cell 표 + best_cell 박제 + paired Δ + oracle 회수율 + 가설 verdict + block 분해 + Runtime + max_class_ratio/top1_acc + follow-up + cross-refs)
- follow-up ≥ 2건 박제 (예: 가설 (a) τ_cls sweep / ensemble (plan-022 + plan-028 best) / F0 ML / oracle gap 추가 분석)

---

## §5. 작업량 총 회계

| STAGE | Commit 수 | Cell 수 | 예상 runtime (CPU) |
|:--|--:|--:|--:|
| G0 (c1~c5) | 5 | 0 | <10min (setup) |
| G1 (c6) | 1 | 3 carry reproduce | ~5min |
| G2.A (c7~c11) | 5 (B4 재실행 skip → c10 instant) | 4 신규 + B4 carry | ~15min total (B1 ~3min + B2 ~3min + B3 ~5min + B4 instant + W1 ~5min) |
| G2.A analysis (c12) | 1 | 0 (branch 결정) | <1min |
| G2.B (c13~c14) | 1~2 (branch δ = 1 cell, α/β/γ = 2 cell) | 1~2 | ~5~15min |
| G3 (c15) | 1 | 0 (best_cell selection) | <1min |
| G_final (c16) | 1 | 0 (results) | <5min |
| **Total** | **15~16** | **5~7 cell + 3 carry reproduce** | **~30~50min CPU** |

cell 수 / commit 수 / 작업량 plan-025 (2 cell, ~15min) 대비 약 3× 증가 — block ablation grid 가 5 cell 이라 자연스러움. spec 의 G2.A 5 cell 단일 변수 원칙 (WORKFLOW.md §9.2) 준수: 각 cell 은 baseline (= B4) 대비 한 변수만 변경.

---

## §6. results.md 필수 항목 (plan-025 form 일치)

1. plan_id / version / date / status / band / best_cell
2. G-gate 표 (G0/G1/G2.A/G2.B/G3/G_final 별 status + commit hash + 결과)
3. G2.A 5 cell 결과 표 (cell / hit_1cm / hit_1p5cm / Δ_1cm vs F0 / Δ_1cm vs p022 / Δ_1cm vs B4 / max_class_ratio / top1_acc / runtime)
4. G2.B branch + cell 결과 표 (활성 branch / cell / 위 동일 metric)
5. Best cell 박제 + paired Δ 3종 (F0, p022, p025-C1)
6. 가설 (b) + (d) verdict (rejected / confirmed / inconclusive + 수치 근거)
7. 14-anchor oracle 회수율 (best / 0.7928)
8. 1080D input block 분해 표 (plan-025 form 참고, 본 plan 의 G2.A 결과로 (d) 가설 update)
9. Runtime (G0~G_final per STAGE)
10. max_class_ratio + top1_acc + best_iteration (cell 별)
11. Follow-up plan 후보 ≥ 2건
12. Cross-refs (spec, results pair, baseline_carry, results_<cell>.json 5+ 개, paradigm_analysis, 참조 plan)

---

## §7. 통계 함정 & caveats

- **5-fold OOF concat 의 분산**: plan-022/025 carry — per-fold std 박제. paired Δ 검정은 같은 fold split 위 per-sample 차이의 평균 → fold 분산 영향 적음.
- **W1 cell 의 hard label vs soft label**: sample-weight OFF 시 LightGBM `multiclass` objective 가 hard label 필요. hard label = `argmax(soft_label[14])` (= K=14 BCC 의 single anchor). soft label 의 sharpness (τ=0.001) 가 hard argmax 와 동등하지 않은 sample 비율 측정 (≤ 5% 예상, plan-022 carry 박제) → W1 의 결과 해석 시 caveat 박제.
- **B1 (22D only) 의 LGBM split 최소 sample 수**: 22D 가 너무 작아 `min_data_in_leaf=20` 도 일부 leaf 에서 invalid 가능. fallback = leaf 부족 시 LGBM 의 default 처리 (= no split) 그대로.
- **G2.A.B4 의 재현성**: c10 단계에서 plan-025 C1 carry 의 hash 정확성 — plan-025 worktree 의 commit hash 박제 필수 (`decision-note: spec-default — B4 = plan-025 C1 carry hash <commit>`).
- **G2.B branch δ (MLP) 의 CPU 수렴**: plan-024 cross-attention 의 CPU under-converged 교훈 — 본 plan δ branch 의 MLP 는 작은 capacity (hidden=64, depth=2, epoch=50) 로 의도적으로 under-converged 위험 회피. 단 본 cell 이 plan-022 winner 못 이기면 paradigm-level conclusion = "LGBM 자체가 ceiling".
- **paired Δ 의 fold variance**: paired bootstrap 5000× 측정 박제 (plan-022/025 carry 산식, `analysis/plan-022/run_oof.py:bootstrap_paired_delta`). 95% CI 박제. CI 가 0 포함 시 partial band warn.
- **dataset_hash 일치**: G1 baseline_carry.json 의 dataset_hash 가 plan-022/025 baseline_carry.json 의 hash 와 일치 — 데이터 drift 차단.

---

## §8. 변경 이력

- v1 (2026-05-22): 초안. plan-025 mode collapse paradigm_analysis §4 의 가설 (b) + (d) 검증 + plan-022 winner lift 목표 spec.

---

## §9. 참조

- `plans/plan-025-candidate-concat-input-max.md` v1 (worktree-plan-025-spec) — mode collapse paradigm + 4 가설 source
- `plans/plan-025-candidate-concat-input-max.results.md` (worktree-plan-025-spec) — 0.6320 mode collapse hard evidence + paradigm_analysis.json
- `plans/plan-022-corrector-free-anchor-layout-sweep.md` + `.results.md` — plan-022 winner A6_bcc14_tau001 paradigm
- `plans/plan-024-…` (worktree-plan-024-combo) — 14-anchor oracle 0.7928 ceiling + feature engineering 6 module
- `plans/plan-020-polyfit-baseline.md` + `.results.md` — F0 baseline + 5-fold split + paired Δ 산식
- `WORKFLOW.md §1~§12` — plan/results/registry/Autonomous Execution Protocol 규약
- `CLAUDE.md` — Autonomous Execution Policy
- `memory/project_next_plan_direction.md` — 2026-05-22 박제: plan-025 ablation 시급 (= 본 plan 가 회수)
- `analysis/plan-024/oracle_ceiling.json` (carry) — 14-anchor oracle 0.7928

---

## §10. plan-028 self-contained 확인 (Spec 자기-완결 invariant, WORKFLOW.md §9.3)

본 plan 은 외부 컨텍스트 (채팅 로그, 메모리) 없이 단독 재구성 가능:

- §1.1 ~ §1.3 = 배경 (plan-025 mode collapse + plan-022 winner + oracle ceiling 수치 박제)
- §2.1 + §2.2 = Scope
- §3.1 ~ §3.3 = Pre-reg (fold, 합격 기준, 평가)
- §4.1 ~ §4.7 = STAGE 정의 (commit chain, cell config, branch 함수)
- §5 = 작업량
- §6 = results.md 필수 항목 (plan-025 form 일치)
- §7 = caveats
- §8 = 변경 이력
- §9 = 참조 (모든 plan / module / data 의 path)

§0.5 = autonomous loop 가 매 turn 읽을 self-updating log + commit chain 16-step + plan-specific severe 9 + paths whitelist/blacklist + decision-note 예시.

frontmatter `code_reuse` = 명시적 carry 모듈 18개 박제 (plan-025 3 + plan-024 7 + plan-022 3 + plan-021 1 + plan-020 1 + src 2 + 1 합산 17 + symbols 별 추가 박제 = 18).

본 plan G_final 도달 시 plans/plan-028-*.results.md + analysis/plan-028/results.md 2 file 생성, 3-file frontmatter sync 완료.
