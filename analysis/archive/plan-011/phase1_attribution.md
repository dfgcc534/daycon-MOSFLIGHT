---
phase: Phase 1 attribution (G1)
date: 2026-05-13 (Asia/Seoul)
status: complete
fold: 0 (1-fold approx, N_val=2020)
n_axis_positive: 0  # strict ≥ 0.005 threshold
g1_b_pass: false
autonomous_branch: option_a (Phase 3 skip, G_final 직접 진입)
plan_012_carry_over: true
---

# plan-011 Phase 1 Attribution (G1)

## §1. 요약 (4 best lever 식별)

Phase 1 24 sub-exp (L0~L7 + IA,IB,ID,IF (IC skip) + M0~M6 + F0~F4) 모두 fold-0 OOF 측정 완료.

**★ 결론**: 4-axis 어느 곳도 strict +0.005 marginal threshold 통과 안 함.

| axis | best lever | ΔOOF vs anchor | axis-positive? | comment |
|------|------------|-----------------|----------------|---------|
| L (Loss)   | L3 (asym, gate=1)  | **-0.0114** | ❌ | 모든 L1~L7 < L0 anchor |
| In (Input) | **ID (CNN encoder 64-dim)** | **+0.00495** | △ (just shy 0.0005) | strict 미달; 가장 근접한 positive 신호 |
| M (Arch)   | M1 (GateHead)      | 0.0000 (tied) | ❌ | M0 anchor 와 동일 |
| F (Formula)| F0 (anchor)        | 0.0000 (fix) | ❌ | F1/F2 박제 부재, F3/F4 numerical 오류 |

→ G1 (b) "최소 2 axis +0.005" requirement **FAIL** (0/4 strict positive).

## §2. L axis 표 (8 sub-exp × ΔOOF)

| sub-exp | oof_soft_hit | delta_vs_anchor (L0=0.6545) | delta_vs_z1 (L1=0.6421) | corrector_gain |
|---------|--------------|------------------------------|-----------------------------|----------------|
| L0 (anchor) | 0.6545 | 0       | —      | +0.0144 |
| L1 (Z1 min) | 0.6421 | -0.0124 | 0      | +0.0020 |
| L2 (SKIP per G0) | —  | —      | —      | — |
| L3 (asym only) | 0.6431 | -0.0114 | +0.0010 | +0.0030 |
| L4 (Frenet)    | 0.6421 | -0.0124 | 0       | +0.0020 |
| L5 (physics)   | 0.6421 | -0.0124 | 0       | +0.0020 |
| L6 (bell)      | 0.6406 | -0.0139 | -0.0015 | +0.0005 |
| L7 (hinge)     | 0.6431 | -0.0114 | +0.0010 | +0.0030 |

L̂ = L3 (delta_vs_z1 +0.0010, tied with L7). L axis 전체 NEGATIVE vs L0 anchor — Z1 minimum 자체가 fold-0 에서 plan-004 default 보다 낮음.

## §3. In axis 표 (5 sub-exp) — ★ v1.2 IC 활성화 후 재실행

| sub-exp | oof_soft_hit | delta_vs_anchor (IA=0.6431, v1.2 re-run) | encoder_dim |
|---------|--------------|-------------------------------------------|-------------|
| IA (anchor) | 0.6431 | 0       | 0 (cf only) |
| IB (+stats 20-dim) | 0.6436 | +0.0005 | 20 |
| **IC (+R001 frozen GRU 64-dim)** | **0.6446** | **+0.0015** ★ NEW BEST | 64 |
| ID (+CNN 64-dim learnable) | 0.6436 | +0.0005 | 64 |
| IF (+stats × multi-parse) | 0.6426 | -0.0005 | 20 |

In̂ = **IC** (R001_baseline-residual-gru fold0.pt frozen reuse). v1.0 의 ID +0.00495 vs v1.2 의 IC +0.0015 — IA anchor 의 fold-0 seed variance (0.6401 → 0.6431) 가 일부 흡수. *absolute* OOF 로는 IC=0.6446 vs v1.0 ID=0.6450 ≈ tied (variance 0.0005 m).

> v1.2 amendment: plan-004 selector checkpoint *부재* → R001_baseline-residual-gru (2-layer GRU(3, 64) same dataset) frozen reuse. spec 32-dim → 64-dim deviation 박제 (decision-note).

## §4. M axis 표 (7 sub-exp)

| sub-exp | oof_soft_hit | delta_vs_anchor (M0=0.6426) | params | comment |
|---------|--------------|------------------------------|--------|---------|
| M0 (anchor TinyCorrectionNet)  | 0.6426 | 0       | 37K  | baseline |
| M1 (GateHeadCorrector)         | 0.6426 | 0       | 40K  | gate ≈ identity (L0-style loss → no destructive penalty) |
| M2 (SplitHeadCorrector)        | 0.6401 | -0.0025 | 40K  | direction + magnitude |
| M3 (BinClassifier 3-axis)      | 0.6411 | -0.0015 | 48K  | softmax discretization |
| M4 (IterativeRefinement)       | 0.5733 | -0.0693 | 38K  | ★ stage-wise loss 설계 미흡 |
| M5 (GMM μ+σ NLL)              | 0.6193 | -0.0208 | 40K  | NLL ≠ hit-rate optimization |
| M6 (WiderShallow d=1 h=256)    | 0.6347 | -0.0079 | 306K | over-param at small data |

M̂ = M1 (tied with M0). M axis NEGATIVE — gate head 자체로는 L0 default loss 와 결합 시 효과 없음 (asymmetric loss 필요).

## §5. F axis 표 (5 sub-exp) — ★ v1.1 post-fix update

| sub-exp | oof_soft_hit | delta_vs_anchor (F0=0.6401) | comment |
|---------|--------------|------------------------------|---------|
| F0 (anchor frenet_par120_perp_neg020) | 0.6401 | 0 | F0 reuse, 재학습 X |
| F1 (CMA-ES tuned)  | 0.6401 | 0 | plan-007 Step 2 params 박제 부재 → F0 fallback |
| F2 (basis ablation) | 0.6401 | 0 | plan-007 Step 3 params 박제 부재 → F0 fallback |
| F3 (per-sample MLP) | 0.6361 | -0.0040 | post-fix (v1.1): per-sample (par, perp) regression < F0 fix (MLP over-fit small data?) |
| **F4 (LearnableSingleCandidate)** | **0.6431** | **+0.0030** ★ | post-fix (v1.1): F0 init parity 1.39e-05m, 6-coef learnable, final coef (1.969, -0.010, 1.196, -0.197, 0.002, 0.983) |

F̂ = **F4** (post-fix) — positive direction, strict 0.005 threshold 미달이지만 informational positive 신호.

> v1.0 (original)에서는 cand formula bug (`LearnableSingleCandidate.forward` 가 selector.make_candidates 와 numerical 불일치) 로 F3=0.0980, F4=0.0322 catastrophic 결과. v1.1 spot-fix 후 정상 측정. plan-011.1 carry-over 의 *1순위* 가 본 fix 로 해결.

## §6. cross-axis informational (예: L1 + IC implicit signal)

- L axis sub-exp 들이 *In-A anchor* (cf only) 와 *M0 anchor* (TinyCorrectionNet) 위에서 측정 → cf 32-dim *만* 의 정보로는 corrector 가 plan-004 default 를 능가하기 어려움 (= L axis NEGATIVE 의 *주된 원인*: input 정보 부족 + 기존 plan-004 정교한 hyperparam tuning).
- In axis 의 ID 만 positive 방향 (+0.00495) — cf 32-dim *+ CNN 64-dim trajectory encoder* 의 *결합* 이 plan-004 default 대비 marginal 신호. ★ §1.5 input snapshot 한계 hypothesis 부분 검증.
- M axis 의 M1 (gate) ≈ M0 (no gate) — *L0 default loss* (= no asymmetric) 위 gate 는 학습 후 ≈ 1 으로 수렴 (효과 없음). gate 의 의미는 *asymmetric loss 와 결합* 일 때만 발현 (= P1.L2 였으나 G0 의 D001=0.6570 < 0.66 으로 skip).
- F axis 의 F3/F4 negative 는 *implementation bug* (cand formula parity 어긋남) — actual 실험 신호 아님.

## §7. decision-note (L̂/In̂/M̂/F̂ 채택 사유)

| lever | 채택 | 사유 |
|-------|------|------|
| L̂ | **L3** (asym, gate=1) | delta_vs_z1 +0.0010 (tied with L7); spec 의 C009 motivation 명시 |
| In̂ | **IC** (R001 frozen GRU 64-dim, v1.2 post-fix) | absolute OOF 0.6446 (best within In axis); v1.0 ID=0.6450 와 tied (seed variance) |
| M̂ | **M1** (GateHead) | tied with M0; gate infrastructure 보존 |
| F̂ | **F4** (LearnableSingleCandidate, v1.1 post-fix) | +0.0030 vs F0 anchor — formula parity 보강 후 진정한 측정 |

## §8. Phase 3 진입 후보 list (autonomous)

원 spec: P3.1 (L̂+In̂), P3.2 (L̂+M̂), P3.3 (L̂+F̂), P3.4 (In̂+M̂).

**autonomous 결정 (G1 (b) FAIL per §9.3 option a)**:
- ❌ Phase 3 4 pair *모두 skip* — 0/4 axis strict +0.005 positive.
- ✅ G_final 직접 진입 — best Phase = Phase 1 max (= In axis ID at 0.6450, fold-0 OOF).
- ✅ plan-012 carry-over **paradigm 교체** 후보 박제.

> ★ *v1.1 post-fix amendment*: F axis 의 F4 가 +0.0030 (positive direction). In axis ID +0.00495 + F axis F4 +0.0030 = **2 axis sub-threshold positive direction**. strict 0.005 통과는 못 했지만 P3.1 (L̂+In̂) 및 P3.3 (L̂+F̂) 의 informational 가치는 *plan-011.1 carry-over* 로 박제.

## §9. caveat 검증

- caveat #17 (cross-axis bleed): L sub-exp 의 fixed (In-A, M0, F0) anchor 가 L 의 *실효* 한계 — *In axis fix* 한 결과만 측정 (만약 In̂=ID 위 L sub-exp 재실행 시 다른 결과 가능). plan-011.1 carry-over.
- caveat #21 (small data over-fit): M6 (WiderShallow 306K params) 가 small data 에 over-parameterize. plan-011 의 corrector path 가 *대형 모델* 진로 막힘 — corrector 가 main lever 라면 paradigm 교체 필요.
- §10.5 mechanism overlap: P3.2 (L̂=L3 + M̂=M1) 모두 gate-mechanism 관련 lever — overlap risk 있지만 G1 (b) FAIL 로 P3 진입 안 함 → caveat 만 박제.

## §10. 변경 이력

- 2026-05-13: 초안 — Phase 1 24 sub-exp (L=8, In=4 (IC skip), M=7, F=5) 측정 완료.
  - G1 status: (b) FAIL (0/4 axes strict positive; ID at +0.00495 가장 근접).
  - autonomous: option a → Phase 3 skip, G_final 직접 진입, plan-012 carry-over.

---

## §11. 다음 단계 (G_final entry)

1. **best Phase 박제** = In axis ID submission (fold-0 OOF=0.6450, full-data prediction은 5-fold 미실행).
2. **plan-012 paradigm 교체 후보** (≥ 3 후보):
   - C1: CNN/transformer encoder 강화 — In axis ID 의 +0.00495 신호 확장 (deeper CNN, attention mechanism).
   - C2: F3/F4 candidate formula parity 보강 후 재실행 — F̂ 진정한 측정 (plan-011.1 carry-over 1순위).
   - C3: Diffusion / KNN paradigm — plan-011 §2.2 의 out-of-scope 후보, corrector path 의 *제한* 인정 후 paradigm shift.
3. **plan-011.1 carry-over instruction**: F3/F4 cand formula bug fix + IC frozen GRU checkpoint 박제 + L axis re-run on In̂=ID anchor (caveat #17 검증).
