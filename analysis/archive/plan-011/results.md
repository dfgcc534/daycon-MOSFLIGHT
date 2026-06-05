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
| best Phase | Phase 1 In axis **IC** (R001 frozen GRU 64-dim + cf 32-dim) — *v1.2 post-fix update* | — |
| best fold-0 OOF | **0.6446** | vs P1.IA anchor 0.6431 = +0.0015 (v1.2 re-run; v1.0 ID=0.6450 tied within variance) |
| L axis best lever | L3 (asym, gate=1) | -0.0114 vs L0 (NEGATIVE) |
| In axis best lever | **IC (R001 frozen GRU 64-dim, v1.2)** | +0.0015 vs IA (v1.2 re-run); abs 0.6446 = v1.0 ID 0.6450 tied |
| M axis best lever | M1 (GateHead, tied with M0) | 0 vs M0 |
| F axis best lever | **F4 (LearnableSingleCandidate ★ v1.1 post-fix)** | **+0.0030** (positive direction, strict 0.005 미달) |
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
| best (In axis IC ★ v1.2) | `runs/baseline/H012_phase1-input-ablation/sub_IC/submission.csv` | 0.6446 (fold-0 val, n=2020) | TBD (plan-011.1 carry-over) |
| v1.0 (In axis ID) | `runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv` | 0.6450 (fold-0 val) | TBD |

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

- 2026-05-13 v1: 초안 — Phase 1 G1 complete (autonomous Phase 3+ skip).
  - 24 sub-exp 실험 완료 (L=8, In=4 IC skip, M=7, F=5 F1/F2 fallback + F3/F4 broken).
  - autonomous option a → G_final 직접 진입, plan-012 carry-over.
- 2026-05-13 v1.2 amendment (post-G_final spot-fix #2): IC (frozen GRU) 활성화.
  - **R001_baseline-residual-gru fold0.pt** (2-layer GRU(3, 64) same dataset) frozen reuse — plan-004 selector
    checkpoint *부재* 의 proxy 로 채택 (decision-note).
  - `FrozenGRUEncoder` 구현 갱신: torch.nn.GRU(3, 64, num_layers=2, dropout=0.08) + state_dict 부분 load + freeze.
  - spec 32-dim → 64-dim deviation 박제.
  - IC OOF 0.6446 (+0.0015 vs IA new anchor 0.6431; abs 0.6446 = v1.0 ID 0.6450 와 variance tied).
  - In̂ 결정 갱신: ID → **IC** (R001 frozen GRU). submission `runs/baseline/H012_phase1-input-ablation/sub_IC/submission.csv` 추가.

- 2026-05-13 v1.1 amendment (post-G_final spot-fix): F3/F4 cand formula parity fix 적용 후 재실행.
  - **bug fix**: `LearnableSingleCandidate.forward` 식이 selector.make_candidates 와 numerical 일치 (v_scale=h/2·time_scale, acc_scale=(h/2)²·time_scale², d2 multiplies v_prev not a_last, par/perp 는 acc_par_vec/acc_perp_vec 직접 곱).
  - init_coef = (1.98, 0.0, 1.20, -0.20, 0.0, 1.0) — CANDIDATES[17].d1=1.98 정정 (이전 1.94 → 0.04 magnitude offset 해소).
  - F4 init parity 검증: max(cand_init − F0 anchor) = 1.39e-05 m (numerical noise only) ✓.
  - 재실행 결과:
    - F0 = 0.6401 (anchor, unchanged)
    - F3 = 0.6361 (-0.0040 vs F0; per-sample MLP regression 가 단순 F0 fix 보다 낮음)
    - **F4 = 0.6431 (+0.0030 vs F0)** — *positive direction*, F axis best lever 재판정.
  - F̂ 결정 갱신: **F4** (learnable 6-coef, +0.0030) — 이전 F0 fix (no swap) 대체.
  - G1 status 재평가: F axis 가 +0.0030 도달 (strict 0.005 미달 but informational positive); In axis +0.00495 + F axis +0.0030 = **2 sub-thresh positive 신호** — 4 axis 중 2 axis 가 positive direction. G1 (b) strict 통과 여전 미달이지만 *조건부 P3.3 (L̂+F̂) 진행 의미* 가능.
  - 추가 carry-over: plan-011.1 추가 옵션 — P3.1 (L̂+In̂) + P3.3 (L̂+F̂) 5-fold 진행 후 super-additive 검증.
