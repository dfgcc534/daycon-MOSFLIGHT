---
plan_id: 025
finished_at: null
status: partial
best_cell: null
best_hit_1cm: null
best_hit_1p5cm: null
best_delta_1cm: null
best_delta_1p5cm: null
band: null
g1_completed: true
g2_completed: false
exp_ids_completed: []
exp_ids_skipped:
  - Z025_C1_default (사유: G2 학습 미진행 — 본 plan 의 c7 [TODO])
  - Z025_C2_adjusted (사유: G2 학습 미진행 — 본 plan 의 c8 [TODO])
---

# plan-025.results — Candidate-concat Input Max (interim, G1 완료)

> **status = partial (G1 완료, G2 미진행)**. plan-025 의 *인프라 + G1 reproduce* 단계까지만 본 commit 시점에 완료. G2 (C1/C2 5-fold OOF 학습) 는 carry-forward 작업 (각 1.5~3h / 2~5h CPU 추정). G3 paradigm finding + G_final 도 동반 carry.

## 핵심 결과 요약 (G1 까지)

| Gate | Status | 결과 |
|:--|:--|:--|
| G0 | ✅ DONE (a3646dd) | 12/12 pytest pass (3.33s). build_feat_1080 / run_oof / tests + plan-024 module 8 file cherry-pick + plan-022/021/020 carry 모두 import 정상. |
| G1.a | ✅ DONE (e262299) | F0 baseline 5-fold concat OOF — hit@1cm = **0.6320** ∈ [0.6315, 0.6325] ✓ / hit@1p5cm = **0.8033** ∈ [0.8028, 0.8038] ✓ (plan-020/021/022/023 carry exact). |
| G1.b | ✅ DONE (e262299) | plan-022 winner A6_bcc14_tau001 reproduce — hit@1cm = **0.6531** ∈ [0.6523, 0.6533] ✓ / hit@1p5cm = **0.8108** ∈ [0.8099, 0.8109] ✓. paired Δ_1cm vs F0 = +0.0211, max_class_ratio = 0.1054, pass_both = True. |
| G2.C1 | ⏭ TODO | C1 (LGBM default carry, n_est=500 lr=0.05 num_leaves=63) 5-fold OOF, K=14 BCC + τ=0.001, 1080D input. **carry-forward** (예상 1.5~3h CPU). |
| G2.C2 | ⏭ TODO | C2 (n_est=2000 lr=0.03 feature_fraction=0.7 min_data_in_leaf=50 early_stopping=100) 5-fold OOF. **carry-forward** (예상 2~5h CPU). |
| G3 | ⏭ TODO | paradigm finding (max(C1, C2) hit@1cm > 0.6700 → PASS / ∈ [0.6528, 0.6700] → partial_lift / < 0.6528 → regression). |
| G_final | ⏭ TODO | results.md final + 3-file frontmatter sync + follow-up plan 후보 박제. |

## §1. 인프라 검증 (G0)

- `analysis/plan-025/build_feat_1080.py` (block ① 170D + ② 128D + ③ 22D + ④ 760D = 1080D row-expanded builder)
- `analysis/plan-025/run_oof.py` (5-fold OOF runner, C1/C2/G1 CLI + `LgbmSelectorRowExpanded` subclass = spec §4.4 선택 B fallback 적용)
- `tests/test_plan025_smoke.py` (12 pytest)
- `analysis/plan-024/` cherry-pick 8 file (commit 915dd26 from `worktree-plan-024-combo`): `__init__.py` + `anchor_vocab.py` + `cand_builder.py` + `seq_builder.py` + `torsion_calc.py` + `quantile_carry.py` + `multiwindow_trim_build.py` + `multiwindow_trim.json`

build_feat_1080 sanity (N=5 smoke): output shape (5×14, 1080) ✓ + NaN/Inf 부재 ✓ + dtype float32 ✓.

## §2. G1 reproduce 결과 (baseline_carry.json)

```json
{
  "F0": {"hit_1cm": 0.6320, "hit_1p5cm": 0.8033},
  "plan022_winner": {
    "hit_1cm": 0.6531, "hit_1p5cm": 0.8108,
    "raw": {
      "delta_1cm": 0.0211, "delta_1.5cm": 0.0075,
      "max_class_ratio": 0.1054, "fold_var_1cm": 0.0036, "fold_var_1.5cm": 0.0071,
      "pass_both": true
    }
  }
}
```

- F0 reproduce: plan-020/021/022/023 carry exact match.
- plan-022 winner reproduce: hard evidence (0.6528 / 0.8104) 와 tight band 통과 — fold split, soft label, anchor codebook, hparam 모두 정합.
- **`plan022_reproduce_drift` severe = 미발생** (band 통과).

## §3. G2 / G3 / G_final 미진행 사유

본 results.md commit 시점 = G1 완료 직후. G2 학습은 long-running (C1 1.5~3h + C2 2~5h, 총 ~3.5~8h CPU) — *carry-forward* 작업으로 박제.

향후 진행 시 본 results.md 의 §4~§9 추가 + frontmatter `status: partial → all_complete`, `band: null → positive/partial/negative`, `best_cell` 박제 + 3-file frontmatter sync.

## §10. Follow-up plan 후보 (spec §0.5 carry)

본 plan G_final 완료 후 (또는 사용자 결정 시 plan-025 G2 carry-forward 와 병렬):

- **plan-026 (block ablation, gated on plan-025 G2 결과)**: block ②/③/④ each-out ablation 으로 lift attribution. plan-025 의 1080D 가 PASS 라면 어느 block 이 dominant lever 인지 분리 검증.
- **plan-027 (ensemble)**: plan-025 winner + plan-022/023 winner soft-vote. plan-025 G2 미진행 시 plan-022/023 winner 만으로 시작 가능.
- **plan-028 (F0 baseline ML)**: F0 baseline ML 화 (현재 hand-crafted → systematic forward bias 완화). oracle 0.7928 → 0.85+ 추정 lever.

## §11. Cross-refs

- spec: `plans/plan-025-candidate-concat-input-max.md`
- baseline_carry: `analysis/plan-025/baseline_carry.json` (G1 박제)
- plan-022 carry: `analysis/plan-022/` (A6_bcc14 anchor + LgbmSelectorOnly + run_oof_cell)
- plan-024 module carry: `analysis/plan-024/` (commit 915dd26 from worktree-plan-024-combo, 8 file)
- memory: `project_next_plan_direction.md` (2026-05-22 user 한 줄 재정의 + input 1080D 박제)
