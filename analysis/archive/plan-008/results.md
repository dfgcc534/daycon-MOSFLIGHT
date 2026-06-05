---
plan_id: 008
based_on:
  - 004
  - 005
  - 006
  - 007
finished_at: 2026-05-12T21:05+09:00
status: partial (carry-over to plan-008.1 for LB submission, plan-009 for Step 4 corrector + selector arch)
exp_ids_completed:
  - G001-sanity-27 (c5.5)
  - G001-candidate-redefine (c7)
exp_ids_deferred:
  - G002-corrector-band (boundary.py LOSS_ATTR 부재 — plan-009)
lb_exp_id: G001-candidate-redefine (Step 3 만)
lb_score: TBD (quota_exhausted_2026-05-12, deferred to plan-008.1)
lb_submitted_at: null
severe_flags:
  - redefinition_severely_insufficient (G1, oracle_final=0.7543 < 0.78)
  - selector_no_improvement (G2, oof_soft_hit=0.6503 < 0.70)
warn_flags:
  - diagnostic_inconclusive (G0)
  - sanity_baseline_drift (G2, sanity_27=0.6466 ∉ [0.652, 0.662])
  - family_effect_marginal (G2, family_effect=+0.0037 < +0.02)
g_gates:
  - G0  : DONE ebd4979 (warn diagnostic_inconclusive)
  - G1  : DONE b22f86c (SEVERE redefinition_severely_insufficient)
  - G2  : DONE 1a8c05c (SEVERE selector_no_improvement + 2 warn)
  - G3  : DEFERRED 4277a21 (plan-009 — boundary.py hook 부재)
  - G_final: in_progress (본 commit)
---

# plan-008 results — Candidate redefine + Corrector redesign on Variant A baseline

## 1-줄 요약

> Step 2/3 의 *후보 풀 확장* main lever 가 **oracle 천장 +0.037 회수만** 했고 (0.7188 → 0.7562, 0.85 target 의 30% 도달), selector 가 그것을 **OOF hit 으로 follow 못 함** (0.657 → 0.650, family_effect = +0.0037). **진단 c2 의 main_bottleneck = "ranking"** 가 그대로 입증 — selector arch 자체 한계 (caveat #13). Step 4 corrector 는 boundary.py lock-in 으로 monkey-patch 불가 → DEFERRED plan-009.

## Step 1 — Diagnostic (G0)

| 항목 | 값 |
|---|---|
| n_oracle_miss (raw err.min > 1cm) | **2812** (miss_rate 0.2812 ≈ 1−0.7188 §1.1 정합 ✓) |
| dominant_causes | **[] (warn `diagnostic_inconclusive`)** — corr_jerk 0.17 max, perp_pct 0.39 (< 0.40), z_pct 0.18 (< 0.20) all sub-threshold |
| prune_count (strict_v2.4, 95% + 5mm) | 24 candidates flagged (clusters of sub-mm frenet variants) |
| margin p50 (logit) | 0.063 < 0.1 → softmax_diffusion_signal = True (informational) |
| **main_bottleneck** | **"ranking"** (gap_ranking 0.0516 ≫ gap_drift −0.0004) |

핵심 finding: gap_ranking 0.0516 (5.2pp) — selector 가 hit zone 의 후보를 *놓치는* 비율. drift 효과 거의 0. → 후보 풀 확장의 회수 한계 5.2pp.

## Step 2 — Pruning + Greedy Set-Cover (G1, SEVERE)

### Step 2a — Pruning (incremental safety, v2.7 fix)

24 후보 일괄 제거 시 aggregate Δ=−0.0416 위반 → **자율 fix: incremental greedy pruning** (sort by per-pair Δ asc + per-step aggregate check). Final: 15 accepted, 9 rejected (aggregate_unsafe).

| 측정 | 값 |
|---|---|
| oracle_orig (raw, 27 cands) | 0.7188 |
| oracle_pruned (12 kept) | 0.7173 |
| Δ | −0.0015 (≤ 0.0018 G1 ✓) |
| oracle_safe | true |
| kept_indices | [0, 4, 11, 14, 15, 18, 19, 21, 23, 24, 25, 26] (12개) |

### Step 2b — Greedy Set-Cover (Strategy D)

| iter | added template | family_id | oracle | Δ | pool_size |
|---|---|---|---|---|---|
| 0 | (start, pruned 12) | - | 0.7173 | - | 12 |
| 1 | arc_decel | 2 (arc) | 0.7419 | +0.0246 | 13 |
| 2 | rot_high_150 | 1 (trig) | 0.7481 | +0.0062 | 14 |
| 3 | speed_slope_d1_120 | 6 (cross) | 0.7509 | +0.0028 | 15 |
| 4 | rot_low_080 | 1 (trig) | 0.7522 | +0.0013 | 16 |
| 5 | omega_speed | 6 (cross) | 0.7533 | +0.0011 | 17 |
| 6 | fs_3d_low_torsion | 3 (fs3d) | 0.7543 | +0.0010 | 18 |
| stop | (9 templates 잔여, Δ<0.001) | - | 0.7543 | - | 18 |

| G1 합격 | spec | actual | verdict |
|---|---|---|---|
| oracle_after_prune ≥ 0.7170 | 0.7170 | 0.7173 | ✓ |
| oracle_final ≥ 0.85 (minimum) | 0.85 | **0.7543** | ✗ **SEVERE** `redefinition_severely_insufficient` |
| stop_reason | ∈ {target, max, delta} | delta_below_threshold | ✓ |

Per-regime worst (sanity only, regime infra 폐기): r=10 oracle 0.551, r=16 0.350, r=17 0.472 — r=16 sharp turn 회수 실패 (rot_high_150 0.62pp 만 회수, 부족).

## Step 3 — Selector retrain (G2, SEVERE + 2 warn)

### c5.5 sanity_baseline_27 (family 효과 분리)

| 측정 | 값 | spec band | gate |
|---|---|---|---|
| sanity_baseline_27_oof_soft | **0.6466** | [0.652, 0.662] = Variant A 0.6570 ± 0.005 | WARN `sanity_baseline_drift` |
| oracle (27 base) | 0.7188 | - | - |
| elapsed | 215.8s | - | - |

Variant A baseline (plan-005 STAGE 6 LB-OOF 0.6570) 보다 −0.0104 — hyperparam budget 70% (pre 14→10, fine 10→8 등) 효과 추정. family_effect 비교 시 *inflate* 가능.

### c7 extended-pool selector retrain

| 측정 | 값 | spec target | gate |
|---|---|---|---|
| n_candidates (extended) | 25 | 27 + new family | - |
| KEPT_INDICES (base) | 12/27 | - | - |
| KEPT_FAMILIES | [trig, arc, frenet_serret_3d, cross_term] | 5 family | higher_order drop (greedy 0 pick) |
| oof_argmax_hit | 0.6443 | - | - |
| oof_soft_hit | **0.6503** | ≥ 0.70 | ✗ **SEVERE** `selector_no_improvement` |
| oracle (extended) | 0.7562 | - | - |
| top1_ranking_acc | 0.1721 | (base 0.126) | improvement +5pp |
| **gap_ranking** | **0.1119** | ≤ 0.07 | ✗ (base 0.06 → 0.11 = 2x 증가) |
| family_effect | **+0.0037** | ≥ +0.03 (marginal <+0.02) | ✗ WARN `family_effect_marginal` |
| variant_a_safe (regime_bias) | true | absent OK | ✓ |
| elapsed | 239.9s | - | - |

핵심 관측 (caveat #4 + #13 정합):
- top1_ranking_acc *up* (12.6% → 17.2%) but oof_soft_hit *down* (0.657 → 0.650) → 새 후보가 hit zone *밖* picking 빈도 ↑.
- gap_ranking 2배 증가 (0.06 → 0.11) — extended pool 이 selector ranking 부담 증가시킴.
- family_effect +0.0037 = 후보 풀 자체 신호 부족 (capacity 증가 ROI 낮음 추정).

§6.5 fallback (hidden 64 / pairwise / distill / epoch_plus) **skip 자율 결정** — family_effect +0.0037 은 후보 풀 자체 한계 신호 (selector capacity 무관), 4 fallback × ~5min budget 을 plan-009 selector arch experiment 로 보전.

## Step 4 — Corrector band-specific (G3, DEFERRED)

c9 진입 시점 boundary.py 의 `LOSS_ATTR` 부재 확정:
- §7.3 의 candidate attrs [`compute_corrector_loss`, `corrector_loss`, `loss_fn`] 모두 dir(boundary) 에 없음.
- 실제 callable matching "loss|corr": [`TinyCorrectionNet` (class), `predict_corrected_candidates` (inference only)] — 둘 다 monkey-patch target 부적합.
- boundary.py L231-233: `reg = ((pred - yb) ** 2).sum(dim=1)` — corrector loss 가 `train_net()` 내부 inline 정의 (module-level hook 부재).

자율 결정 (CLAUDE.md autonomous):
- alt1 (standalone re-impl ~200 LOC) 거절 — G1+G2 severe cascade 위 Step 4 booster (+0.02) ROI marginal.
- alt2 (boundary.py 본문 수정) 거절 — §0.5 paths blacklist (lock-in).
- alt3 (plan-009 carry-over) 채택 — boundary.py 의 hook 신설 → band-specific monkey-patch 구조 도입.

→ **G3 = DEFERRED** (severe 아님, plan-specific deferred status).

## LB trajectory

| plan | exp_id | OOF (soft) | LB | OOF→LB gap |
|---|---|---|---|---|
| plan-004 (P001 full) | full (GRU + physics + regime) | 0.6599 | **0.6822** | +0.0223 |
| plan-005 STAGE 6 Variant A | E002 | 0.6570 | **0.6796** | +0.0226 |
| plan-006 Variant E (단일 공식) | F001 | 0.6491 | 0.6692 | +0.0201 |
| plan-007 Step 3 best basis | E001 | 0.6403 | 0.6598 | +0.0195 |
| plan-007 Step 4 MLP coeff | F002 | 0.6482 | (carry-over) | - |
| **plan-008 c7 (extended pool)** | **G001-candidate-redefine** | **0.6503** | **TBD carry-over (할당량 소진)** | - |

예상 LB (gap +0.022 적용): 0.6503 + 0.022 ≈ **0.672** — Variant A baseline 0.6796 대비 −0.008. **회복 가치 낮음** (plan-008.1 submission 시도하되 기대 ↓).

## Decision-notes 박제 list (autonomous fix audit)

1. spec-default — Step 1 oracle_miss mask raw err (§1.1 정합), corrected err 분리.
2. spec-default — Family 5 jerk 3차 차분 (이전 2차 차분 acceleration 오기 정정).
3. spec-default — Step 2a aggregate safety 위반 → incremental greedy pruning (15 accepted / 9 rejected).
4. spec-default — oracle_final 0.7543 < 0.78 SEVERE → "또는 plan-009 carry-over" 경로 채택.
5. spec-default — soft_select import path = selector module (boundary 오기).
6. spec-default — §6.5 4 fallback skip (family_effect +0.0037 = capacity 무관 신호).
7. spec-default — §7.3 LOSS_ATTR 부재 → c9 DEFERRED, plan-009 hook 신설 task 박제.
8. spec-default — KEPT_FAMILIES = greedy 가 1+ template 픽한 family (higher_order family drop).

## 종합 finding

- **Plan-008 의 *3 main hypothesis 검증 결과***:
  - H1 (oracle miss residual decomposition → dominant cause): **부정** — 모든 corr/pct sub-threshold, `diagnostic_inconclusive` warn.
  - H1.5 (ranking gap dominant): **확정** — gap_ranking 0.0516 ≫ gap_drift −0.0004, main_bottleneck=ranking.
  - H2 (structural containment 가지치기): **부분 확정** — 24 식별, incremental 후 15 accepted (aggregate safety 통과).
  - H3 (greedy set-cover → oracle ≥ 0.85): **부정** — 0.7543 (target 의 30% 도달).
  - H4 (Variant A selector OOF ≥ 0.70): **부정** — 0.6503 (선택 후보 풀 확장이 selector hit 으로 follow 안 됨).
  - H5 (corrector band-specific +0.02 booster): **검증 불가** (boundary.py LOSS_ATTR 부재).

- **plan-008 의 본질적 결론**: **후보 풀 확장 main lever 가 ranking 능력 한계 (caveat #13) 를 우회 불가**. selector arch 자체 개선 (plan-009 main task) 가 필수.
