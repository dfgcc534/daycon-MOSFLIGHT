# plan-020 STAGE 3 G2.N — 3 NN 후보 5-fold OOF

## 결과 (cuda:1, seeds=[20260518,19,20], epochs=30, ~106 s total)

| candidate | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | pass (둘 다 ≥ +0.005) |
|---|---|---|---|---|---|
| F0 baseline | 0.6320 | — | 0.8033 | — | — |
| **N01_mlp_coef** | 0.6389 | +0.0069 | 0.8023 | −0.0010 | ✗ |
| **N02_tcn_coef** | 0.6372 | +0.0052 | 0.8036 | +0.0003 | ✗ |
| **N05_moe** | 0.6327 | +0.0007 | 0.8065 | +0.0032 | ✗ |

## G2.N 합격 기준 (§7.4) — **PASS** ✓

| 기준 | N01 | N02 | N05 |
|---|---|---|---|
| metric finite | ✓ | ✓ | ✓ |
| val_hit > 0.10 (random floor) | ✓ (0.639) | ✓ (0.637) | ✓ (0.633) |
| train − val hit < 0.10 (overfit guard) | ✓ (~0.005) | ✓ (~0.003) | ✓ (~0.000) |
| N1 vs plan-007 F002 (0.6482) ±0.02 | ✓ (drift = 0.0093) | — | — |

## Best-on-train seed 다양성 (multi-seed selection bias 진단)

| candidate | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| N01_mlp_coef | 20260518 | 20260518 | 20260519 | 20260519 | 20260518 |
| N02_tcn_coef | 20260518 | 20260518 | 20260520 | 20260519 | 20260518 |
| N05_moe | 20260520 | 20260520 | 20260520 | 20260520 | 20260520 |

- N5 모든 fold 가 동일 seed → expert mixture 의 안정된 minimum.
- N1/N2 fold 마다 different seed → multi-seed 의 진정한 다양성 확보.

## 진단 — NN coef paradigm 한계

- 3 NN 모두 hit@1cm 에서 marginal 향상 (+0.001 ~ +0.007) → input 의 mutual information 한계 검증.
- 그러나 hit@1.5cm 에서 동등 or marginal 향상 (-0.001 ~ +0.003) → *둘 다* criterion 통과 X.
- **paradigm conclusion**: NN coef regression (small MLP / TCN / MoE) 은 *single metric (hit@1cm) only* 향상의 한계. C05 deterministic per-regime (Δ_1cm +0.018, Δ_1.5cm +0.005) 와 **본질적 격차**: NN 이 *regime-conditional 분리* 의 정밀도를 따라가지 못함.
- 본 결과 = plan-014/plan-017 의 corrector paradigm (회수율 5.4% ceiling) 과 유사한 한계 — *input feature 와 정답 방향 사이 MI 부족* 가 paradigm-level bottleneck.

## C05 (deterministic) vs N1 (NN) 직접 비교 (§8.3)

| 비교 | C05 per-regime F0 | N01 MLP coef |
|---|---|---|
| hit@1cm Δ | **+0.0183** | +0.0069 |
| hit@1.5cm Δ | **+0.0053** | -0.0010 |
| pass criterion 둘 다 | ✓ | ✗ |
| 학습 방식 | CMA-ES per regime × 18 regime | Adam smooth-hit per sample |
| param 총수 | 54 | 27→64→64→3 (≈ 6.3K weight) |
| Wall time | ~10 min CPU | ~30 s GPU (cuda:1) |

→ **본 plan-020 narrative ("F0 단일 공식 결과 최대화")의 paradigm-level 결론**: 
  *NN-coef regression 보다 deterministic regime-conditional 분리* 가 더 효과적. 
  하지만 *learnable 효과 자체* 는 C05 의 CMA-ES per-regime fit 에서도 활용됨 — 
  paradigm 의 진정한 winner = *data-driven (F2) family 의 학습 방식 = CMA-ES per-regime*.
