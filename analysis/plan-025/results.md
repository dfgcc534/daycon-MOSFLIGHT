# plan-025 — Candidate-concat Input Max (FINAL)

> **G3 = regression band (negative)**. 1080D LGBM selector C1 + C2 모두 hit_1cm=0.6320 = F0 baseline 동일. selector mode collapse → soft-mean(ANCHORS) = origin → F0 prediction 복사. H1/H2/H3 모두 FAIL.

## 1. plan_id / version / date / status / band / best_cell

- plan_id: 025
- spec version: v1
- date: 2026-05-22 (Asia/Seoul)
- status: all_complete
- band: **negative** (regression)
- best_cell: C1 (tiebreaker by top1_acc, hit 동일)

## 2. G-gate 표

| Gate | Status | Commit | 결과 |
|:--|:--|:--|:--|
| G0 | ✅ | a3646dd | 12/12 pytest pass (3.33s) |
| G1.a | ✅ | e262299 | F0 0.6320/0.8033 ∈ tight band ✓ |
| G1.b | ✅ | e262299 | plan-022 winner reproduce 0.6531/0.8108 ∈ tight band ✓ |
| G2.C1 | ✅ | e31bf4e | hit_1cm=0.6320 (regression), 334s |
| G2.C2 | ✅ | 308411c | hit_1cm=0.6320 (C1 동일), 316s, early_stop fallback |
| G3 | ✅ | (본 commit) | regression band — best=C1=0.6320, paired Δ vs plan-022 = -0.0211 |
| G_final | ✅ | (본 commit) | 3-file frontmatter sync + results.md + follow-up 3건 |

## 3. 2 cell 결과 표

| Cell | hit_1cm | hit_1p5cm | Δ_1cm vs F0 | Δ_1cm vs p022 winner | max_class_ratio | top1_acc | oracle 회수율 | runtime |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| F0 baseline (G1.a) | 0.6320 | 0.8033 | 0.0000 | -0.0211 | — | — | 79.72% | — |
| p022 winner (G1.b) | 0.6531 | 0.8108 | +0.0211 | 0.0000 | 0.1054 | — | 82.38% | (plan-022 carry) |
| **C1** (default) | 0.6320 | 0.8033 | 0.0000 | -0.0211 | 0.0714 | 0.0879 | 79.72% | 334s |
| **C2** (adjusted) | 0.6320 | 0.8033 | 0.0000 | -0.0211 | 0.0714 | 0.1006 | 79.72% | 316s |

**best_cell = C1** (tiebreaker by top1_acc 약간 낮음 — 둘 다 무의미 수준).

## 4. Best cell 박제 + paired Δ

```json
{
  "best_cell": "C1",
  "best_hit_1cm": 0.6320,
  "best_hit_1p5cm": 0.8033,
  "delta_vs_F0": {"hit_1cm": 0.0000, "hit_1p5cm": 0.0000},
  "delta_vs_p022_winner": {"hit_1cm": -0.0211, "hit_1p5cm": -0.0075}
}
```

## 5. H1 / H2 / H3 검증 결과

| 가설 | 측정 | 결과 |
|:--|:--|:--|
| H1 (강): 1080D ≥ p022 170D + 0.005 lift | max(C1, C2) - 0.6531 = -0.0211 | **FAIL** (regression) |
| H2 (강): hit_1cm > 0.6700 stretch | max(C1, C2) = 0.6320 < 0.6700 | **FAIL** |
| H3 (약): C2 - C1 ≥ +0.003 | 0.6320 - 0.6320 = 0.0000 | **FAIL** (Δ=0) |

## 6. 14-anchor oracle 회수율

- oracle ceiling (plan-024 carry) = 0.7928
- best_cell 회수율 = 0.6320 / 0.7928 = **79.72%** (= F0 baseline 와 동일)
- plan-022 winner 회수율 = 0.6531 / 0.7928 = 82.38%
- **lift 잠재력 unused**: 14-anchor oracle 의 82.4 ~ 100% 사이 (4.5pp gap) 본 plan 에서 미발견

## 7. 1080D input block 분해 (잠재력 / 한계 분석)

| Block | Source | Dim | 본 plan 영향 평가 |
|:--|:--|--:|:--|
| ① plan-022 carry | build_input_common + lgbm_extra | 170 | (carry) — plan-022 winner 0.6528 의 input |
| ② cand_builder ctx | regime/Multi-window/STA-LTA/A10/etc | 128 | broadcast feature — LGBM split 시 14-row 동일 값 → discriminative power X |
| ③ cand_builder per-anchor | par/perp/dist + anchor spec + interactions | 22 | **유일한 per-anchor discriminative feature** — 단 mode collapse 로 선택 신호 약함 |
| ④ seq_builder 8-stat | per-channel last/first/mean/std/slope/max/min/range | 760 | broadcast feature — 시간 미세 패턴 압축, 단 ② 와 mostly redundant |
| Total | | 1080 | mode collapse 의 본질 = 1058D broadcast / 22D per-anchor 의 ratio (~50× noise) |

block ablation (plan-026) 에서 검증 필요: block ③ 만 isolate 했을 때 lift 회복 가능성.

## 8. Runtime

- G0 pytest: 3.33s
- G1: F0 ~수초 + plan-022 reproduce ~3min
- G2.C1: 334s (5.5min) CPU
- G2.C2: 316s (5.3min) CPU (early_stop fallback trigger — n_est=500 default 와 동등 학습량)
- 총 plan-025 G0~G3 약 ~15min CPU (예상 1.5-5h 대비 단축 — early_stop fallback 으로 짧음)

## 9. max_class_ratio + top1_acc + best_iteration_per_fold

| Cell | max_class_ratio | top1_acc | best_iteration_per_fold |
|:--|:--|:--|:--|
| C1 | 0.0714 (≈ 1/14 uniform) | 0.0879 | None (no early_stop) |
| C2 | 0.0714 (동일) | 0.1006 | None (early_stop fallback — LightGBM eval_set 가 multi-class 1D y 만 받음, soft label 2D 미지원) |

**Mode collapse 진단 (`soft_label_collapse` warn 보다 약하지만, 의미상 opposite)**:
- max_class_ratio < 0.95 임 (warn threshold 통과) — 그러나 *반대 방향* mode collapse: probs 거의 uniform → selector 결정 못함.
- top1_acc ≈ 1/14 + α → random 수준.
- 결론: selector 가 sample 별 어느 anchor 가 best 인지 못 가린다. soft-mean 으로 fall back → F0 prediction 복사.

## 10. Follow-up plan 후보

### plan-026 (block ablation, gated on plan-025 결과)

본 plan G3=regression 에서도 valid:
- baseline = plan-025 C1 (= F0 동일 0.6320)
- A1 (no block ②): 952D — 차이 미미 예상 (broadcast feature 제거, mode collapse 잔존)
- A2 (no block ③): 1058D — selector 자체가 sample-level 으로 회귀 (= mode collapse 의 본질이 이거인지 검증)
- A3 (no block ④): 320D — 차이 미미 예상

핵심 검증: **A2 가 baseline 과 동일 hit_1cm = 0.6320 이면** root_cause = "1058D broadcast feature dominance + 22D per-anchor signal 약함" (paradigm-level finding).

### plan-027 (ensemble)

band 조건 분기 →
- plan-025 C1 = regression band ⇒ **3-way ensemble 부재** → 2-way (p022 + p023) 만 측정
- E1 (equal 2-way), E3 (weight grid 2-way) 진행
- E2 (equal 3-way) skip 박제

### plan-028 (F0 baseline ML)

systematic forward bias 완화 + oracle 0.7928 → 0.85+ 확장. plan-025 결과 = F0 자체가 ceiling 인 셈 → F0 개선이 가장 큰 lever.

## 11. Cross-refs

- spec: `plans/plan-025-candidate-concat-input-max.md` (v1)
- results pair: `plans/plan-025-candidate-concat-input-max.results.md`
- baseline_carry: `analysis/plan-025/baseline_carry.json` (G1)
- 2 cell results: `analysis/plan-025/results_C1.json`, `results_C2.json`
- paradigm analysis: `analysis/plan-025/paradigm_analysis.json`
- plan-022 carry: `analysis/plan-022/` (A6_bcc14 anchor + LgbmSelectorOnly + run_oof_cell)
- plan-024 carry: `analysis/plan-024/` (8 file from commit 915dd26)
- memory: `project_next_plan_direction.md` (2026-05-22 user 한 줄 재정의)
