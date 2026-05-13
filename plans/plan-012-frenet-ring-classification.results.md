---
plan_id: 012
plan_title: Codebook Bake-off Classification + Regression Hybrid (paradigm reframe, 3D)
status: G_final_complete (warn-recovered; GPU spec-faithful re-run 동일 결론)
date_completed: 2026-05-13 (Asia/Seoul; CPU 초회) / 2026-05-14 (Asia/Seoul; GPU spec-faithful re-run)
date_started: 2026-05-13 (Asia/Seoul, plan-012 v2 spec replacement → c2~c16 + G_final 동일 일자 자율 진행)
exp_ids:
  - H019_phase0-preflight-codebook
  - H020_phase1-codebook-bakeoff
  - H022_phase2-codebook-K
  - H023_phase2-temperature
  - H024_phase2-loss
  - H025_phase2-reg-head
  - H026_phase3-boundary-weight
  - H027_phase3-scorer-arch
  - H028_phase3-r0-prior
  - H029_phase4-final-5fold
final_oof_5fold_hit_1cm: 0.6350  # GPU best stack (CPU: 0.6340)
final_oof_5fold_anchor_baseline: 0.6344  # GPU anchor (CPU: 0.6339)
final_submission: runs/baseline/H029_phase4-final-5fold/submission_anchor_fallback.csv  # GPU rerun anchor
lb_score: null  # plan-012.1 carry-over (사용자 manual submit)
followed_by:
  - 012.1 (LB carry-over)
  - 013 (Candidate C: corrector+hybrid 합체 — recommended default)
---

# plan-012 — Results

## 한 줄 결론

paradigm reframe (3-way codebook bake-off + classifier+regression hybrid) 은 F0 raw hit 위 +0.002~0.003 만 추가 — paradigm 자체의 limit 확인. 5-fold OOF 0.6350 (GPU best stack), target 0.66 와 -0.025 미달. plan-013 path-pivot 필요.

## CPU → GPU spec-faithful re-run (2026-05-14)

CPU 초회 (epochs=15/batch=512/patience=3) → GPU 재실행 (spec-default epochs=50/batch=256/patience=5). 결론 정성적 동일 = under-train hypothesis 기각, paradigm 자체 limit.

| metric | CPU (history) | GPU (current) |
|---|---|---|
| G1 winner OOF | 0.6416 | 0.6411 |
| G2 max ΔOOF | +0.0015 (τ=0.01) | +0.0015 (τ=0.0 또는 K=9) |
| G3 max ΔOOF | +0.0020 (E8 r=0 +0.5) | +0.0005 (E6 bweight_on) ★ E8 r0 prior 는 negative |
| G4 anchor 5-fold | 0.6339 | 0.6344 |
| G4 best stack 5-fold | 0.6340 | 0.6350 |
| G4 Δ | +0.0001 | +0.0006 |
| early stopping | epoch 4~7 | epoch 2~16 ← 50 epoch budget 도달 X |

CPU 결과는 `_cpu.json` / `_cpu.csv` 으로 history 박제.

## G-gate sequence

| G | spec | 측정 | 상태 |
|---|---|---|---|
| G0 | preflight 6 essential checks | F0 hit@1cm=0.6320 / oracles 0.74~0.78 / kmeans min cluster 113 | PASS |
| G1 | winner_oof ≥ 0.6450 + DCM ≥ 0.002 | E0a winner OOF=0.6416 / DCM=0.00037 | warn |
| G2 | 5 axis 중 1+ axis ΔOOF ≥ 0.005 | max ΔOOF = +0.0015 (E3 τ=0.01) | severe-recovered |
| G3 | informational only | max ΔOOF = +0.0020 (E8 r=0 +0.5) | PASS |
| G4 | best_stack ≥ anchor_5fold + 0.005 | Δ = +0.0001 | warn (final_no_additive, fallback) |
| G_final | synthesis + plan-013 후보 + 3 파일 sync | 본 commit | PASS |

## 주요 산출 (full detail = `analysis/plan-012/results.md`)

- Phase 1 winner: E0a (Absolute-7Way, tie-break with E0c per priority "Absolute > Frenet > K-Means") — CPU+GPU 동일
- Phase 2 best lever (GPU): E3 τ=0.0 또는 E2 K=9 (+0.0015 tied); CPU 는 E3 τ=0.01
- Phase 3 best lever (GPU): E6 bweight_on (+0.0005); CPU 의 E8 r=0 +0.5 (+0.0020) 은 under-train noise
- best stack 5-fold OOF: 0.6350 (GPU) / 0.6340 (CPU); anchor 5-fold OOF: 0.6344 (GPU) / 0.6339 (CPU); Δ=+0.0006 (GPU) / +0.0001 (CPU) → 둘 다 < +0.005
- LB submission: `submission_anchor_fallback.csv` (GPU rerun anchor, G4 fallback, manual submit pending)

## plan-013 후보 (full detail = `analysis/plan-012/next_plan_candidates.md`)

| 후보 | 핵심 | LB 분기 |
|---|---|---|
| A | paradigm 완전 폐기 (KNN/GP/Diffusion) | LB < 0.60 |
| B | F0 자체 교체 (per-sample formula selection) | 0.60 ≤ LB < 0.65 |
| **C** ★ | **corrector + hybrid 합체** (2-stage) | LB ≥ 0.65 (default recommended) |

## carry-over

plan-012.1 — 사용자 manual `dacon-submit` (skill) 후 lb_score 박제 + plan-013 분기 결정.
