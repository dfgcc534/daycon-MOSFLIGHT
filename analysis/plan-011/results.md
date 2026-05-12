---
plan_id: 011
plan_title: Single-Formula + Corrector Path Exploration (4-axis breadth ablation)
status: partial (G1 complete, Phase 3+ autonomous skip)
date: 2026-05-13 (Asia/Seoul)
best_phase: Phase 1 In axis ID
best_oof_fold0: 0.6450
lb_score: TBD (plan-011.1 carry-over)
g1_pass: false (b condition — 0/4 axes strict positive)
autonomous_branch: option_a_phase3_skip
plan_012_carry_over: true
plan_011_1_carry_over: true
---

# plan-011 Results — Single-Formula + Corrector Path Exploration

## §1. 요약 (best Phase, 4 axis attribution, OOF, LB 추정 / TBD)

**Phase 1 24 sub-exp 완료. G1 (b) FAIL — 0/4 axes strict +0.005 threshold 통과.**

| metric | 값 | 비교 |
|--------|----|------|
| best Phase | Phase 1 In axis ID (CNN 64-dim encoder + cf 32-dim) | — |
| best fold-0 OOF | **0.6450** | vs P1.IA anchor 0.6401 = +0.00495 (strict 미달) |
| L axis best lever | L3 (asym, gate=1) | -0.0114 vs L0 (NEGATIVE) |
| In axis best lever | **ID (CNN encoder)** | +0.00495 vs IA (★ 가장 근접 positive) |
| M axis best lever | M1 (GateHead, tied with M0) | 0 vs M0 |
| F axis best lever | F0 (anchor, no swap) | 0 (formula_swap_marginal) |
| LB | TBD | submission @ `runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv` |
| LB 추정 (proxy via plan-006 OOF→LB gap +0.0201) | ~0.665 | plan-006 baseline 0.6692 보다 *낮은* 추정 |

**핵심 결론**: 단일공식 + corrector 4-axis breadth ablation 의 *ceiling 측정* 시도 결과 — fold-0 OOF 0.6545 (L0 anchor = plan-004 default) 가 가장 높음. 신규 4 axis lever 들이 plan-004 default 를 *능가하지 못함*. → corrector path 의 *구조적 제한* 노출 + plan-012 paradigm 교체 필요.

## §2. Phase 1 sub-exp 표 (4 axis 모두)

> `analysis/plan-011/phase1_attribution.md` 와 동일 — 본 §2 는 cross-reference 만.

전체 표 → `phase1_attribution.md` §2~§5.

## §3. ★ 발견 (informational, plan-012 carry-over 후보)

1. **L axis 의 negative direction**: Z1 minimum (no env_head, w=[0,1,0.5]) 가 plan-004 default 보다 fold-0 에서 *낮음* (-0.0124).
   - cause hypothesis: plan-004 의 *정교한 hyperparam tuning* (env head + apply_scale 0.75 + boundary band [0.7, 1.7cm]) 이 small data 환경에서 valuable.
   - plan-012 후보: Z1 minimum 의 *부분* (예: huber alone, weight schedule alone) 의 individual contribution 측정.

2. **In axis 의 marginal positive**: ID (CNN 64-dim) 가 +0.00495 — 단일 axis 중 *유일* positive 방향.
   - signal: cf 32-dim *외부* trajectory 시계열 정보 (CNN encoder) 가 marginal 이지만 valuable.
   - plan-012 후보: deeper CNN / attention / transformer encoder 로 확장.

3. **M axis 의 flat profile**: M1 = M0 (tied) — gate alone (no asymmetric loss) 효과 없음. M4 iterative -0.0693 (학습 spec bug — per-step partial target 정의 오류). M5 GMM NLL ≠ hit-rate optimization.
   - plan-012 후보: corrector arch 자체가 main lever 아닐 가능성 — paradigm 교체 (KNN, diffusion) 검토.

4. **F axis 의 implementation issue**: F3/F4 의 cand formula 가 selector.make_candidates 와 numerical 불일치 (0.6401 → 0.0980/0.0322).
   - cause: §8.1 의 self-contained cand formula 가 plan-006 F0 anchor 재현 검증 누락.
   - plan-011.1 carry-over 1순위: F3/F4 formula parity 보강 후 재실행.

## §4. Phase 3+ skip 사유 (autonomous)

§9.3 option a 채택 (G1 (b) FAIL):
- ❌ Phase 3 (4 pair, P3.1~P3.4): 0/4 axis positive → super-additive 측정 의미 없음.
- ❌ Phase 4 (triple stack P4.1~P4.2): triple 의 marginal 가 1+2+3 axis 합산이 *모두 negative or marginal* → 결합 효과 없음.
- ❌ Phase 5 (iterative): M4 iterative 자체가 -0.0693 — iterative path 의 baseline signal 부재.
- ❌ Phase 6 (augment): augment 의 base 가 없음 (Phase 3+ skip).

→ G_final 직접 진입 + plan-012 carry-over.

## §5. submission 박제

| sub-exp | submission path | OOF (fold-0) | LB |
|---------|-----------------|--------------|-----|
| best (In axis ID) | `runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv` | 0.6450 (fold-0 val, n=2020) | TBD (plan-011.1 carry-over) |

submission 생성 = `analysis/plan-011/generate_submission.py` — full train (10000) ID train + test (10000) inference.

## §6. LB 정책 (0 회 제출, plan-009/010 pattern 답습)

본 plan 내 LB 제출 0 회. submission.csv 생성·박제만. **LB 회수 = plan-011.1 carry-over**.

## §7. 4-axis attribution finding (informational)

`phase1_attribution.md` §6 cross-axis informational 박제. 핵심:
- L axis sub-exp 들이 *In-A anchor* (cf only) + *M0 anchor* (TinyCorrectionNet) 위에서 측정 → cf 32-dim *만* 의 정보로는 corrector 가 plan-004 default 능가 어려움.
- In axis 의 ID 만 positive 방향 — trajectory encoder + cf 의 *결합* 이 marginal 신호.
- M axis 의 M1 ≈ M0 — gate 의 효과는 *asymmetric loss 와 결합* 일 때만 발현 (= P1.L2, G0 의 D001 < 0.66 으로 skip).
- F axis 의 F3/F4 negative 는 *implementation bug* (formula parity).

## §8. caveat 검증

- caveat #17 (cross-axis bleed): L sub-exp 의 fixed anchor 가 *실효* 한계 — In̂=ID 위 L sub-exp 재실행 시 다른 결과 가능 (plan-011.1 carry-over).
- caveat #21 (small data over-fit): M6 (WiderShallow 306K) over-param — corrector path 의 *대형 모델* 진로 막힘.
- caveat #6 (multi-parse augment): IF (random 1 parse) +0.0015 < IB (full stats) +0.0035 → multi-parse augment 가 stats 학습 신호 희석 (informational).

## §9. plan-011 의 conclusion

**단일공식 + corrector path 의 ceiling 측정 시도는 plan-004 default 의 robustness 를 재확인**:
- Phase 1 4 axis 어느 곳도 plan-004 default 능가 못 함 (strict +0.005 threshold).
- In axis (시계열 encoder) 가 *유일* positive 방향 — corrector path 의 *진정한* 한계는 *input 정보* 가능성.

→ plan-012 paradigm 교체 후보 (≥ 3):
1. **CNN/transformer encoder 강화** — In axis ID +0.00495 signal 확장 (deeper architecture, attention).
2. **F3/F4 formula parity fix 후 재실행** — plan-011.1 carry-over (formula axis 의 진정한 측정).
3. **KNN / Diffusion paradigm** — corrector path 의 구조적 제한 인정 후 paradigm shift.

## §10. 변경 이력

- 2026-05-13: 초안 — Phase 1 G1 complete (autonomous Phase 3+ skip).
  - 24 sub-exp 실험 완료 (L=8, In=4 IC skip, M=7, F=5 F1/F2 fallback + F3/F4 broken).
  - autonomous option a → G_final 직접 진입, plan-012 carry-over.
