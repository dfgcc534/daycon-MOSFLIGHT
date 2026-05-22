---
plan_id: 025
finished_at: 2026-05-22 (Asia/Seoul)
status: all_complete
best_cell: C1
best_hit_1cm: 0.6320
best_hit_1p5cm: 0.8033
best_delta_1cm: 0.0000
best_delta_1p5cm: 0.0000
band: negative
g1_completed: true
g2_completed: true
g3_completed: true
g_final_completed: true
exp_ids_completed:
  - Z025_C1_default
  - Z025_C2_adjusted
exp_ids_skipped: []
---

# plan-025.results — Candidate-concat Input Max (FINAL, band=negative)

## 핵심 결과 요약

- **best**: C1 (hit_1cm=0.6320, hit_1p5cm=0.8033 = F0 baseline 동일)
- **band**: **negative (regression)** — G3 PASS 기준 (> 0.6700) 실패 + plan-022 winner (0.6531) 도 못 도달
- **G3 verdict**: 1080D LGBM selector C1 + C2 모두 mode collapse → soft-mean ≈ F0
- **H1/H2/H3**: 모두 FAIL
- **Severe**: 0건 (단 `soft_label_collapse` 의 *반대 방향* mode collapse warn — probs uniform near 1/14)
- **14-anchor oracle 회수율**: 79.72% (= F0 baseline 와 동일, plan-022 winner 82.38% 대비 -2.66pp)

## 2 cell 결과

| Cell | hit_1cm | hit_1p5cm | Δ vs F0 | Δ vs p022 | max_class_ratio | top1_acc | runtime |
|:--|--:|--:|--:|--:|--:|--:|--:|
| C1 default | 0.6320 | 0.8033 | 0.0000 | -0.0211 | 0.0714 | 0.0879 | 334s |
| C2 adjusted | 0.6320 | 0.8033 | 0.0000 | -0.0211 | 0.0714 | 0.1006 | 316s |

C2 의 early_stopping fallback trigger (`decision-note: early_stop_fallback`) — LightGBM eval_set 가 multi-class 표준 1D y 만 받음, soft label 2D (N, K) 미지원 → C2 의 5 hparam adjust 중 4 개 (lr, n_est, feature_fraction, min_data_in_leaf) 만 effective + early_stopping=100 무효 → 사실상 C1 와 유사 학습.

## Paradigm finding

> **1080D row-expand LGBM = mode collapse**. probs 거의 uniform (max_class_ratio ≈ 1/14 = 0.0714) → soft-mean(ANCHORS_A6) ≈ Frenet origin → world prediction = F0 baseline 복사.

Root cause 가설 (paradigm_analysis.json §4):
- (a) τ_cls=0.001 sharp soft label train/test generalization 깨짐
- (b) sample-weight expansion (140k row) × 1080D feature 위 multi-class CE 학습 inefficiency
- (c) LgbmSelectorRowExpanded subclass fit 의 selector self-consistency 약화
- **(d) per-anchor block ③ 22D 만이 진정한 anchor discriminative, 1058D broadcast 가 LGBM split 시 noise 우위** ← 가장 likely

## Severe / warn 박제

- Severe 0건
- `soft_label_collapse` warn: 반대 방향 (uniform-near). max_class_ratio 0.0714 < 0.95 threshold 통과지만 의미상 selector decision power 없음.
- `regression` warn: best_hit_1cm 0.6320 < 0.6528 (plan-022 winner) → band=negative.
- `early_stop_fallback` decision-note (C2): LightGBM eval_set 미지원 → default fit 으로 fallback.

## Runtime

- 총 plan-025: ~15min CPU (G0~G3)
- spec 예상 (1.5~5h × 2 cell = 3.5~10h) 대비 *현저 단축* — early_stopping fallback 으로 n_est=500 (C1 default 와 동등) 학습으로 진행됨.

## Follow-up plan 후보

- **plan-026 (block ablation)** — G3=regression 에서도 valid. block ③ isolation 으로 root_cause (d) 검증.
- **plan-027 (ensemble)** — 2-way (p022 + p023) 만, plan-025 C1 ensemble 후보 부재 (band=regression).
- **plan-028 (F0 ML)** — F0 자체 개선이 가장 큰 lever (plan-025 = F0 ceiling 확인).

## Cross-refs

- spec + §0.5 commit chain: `plans/plan-025-candidate-concat-input-max.md`
- analysis dir: `analysis/plan-025/` (build_feat_1080.py, run_oof.py, baseline_carry.json, results_C1.json, results_C2.json, paradigm_analysis.json, results.md)
- carry: plan-020/021/022 module + plan-024 cherry-pick 8 file (commit 915dd26)
