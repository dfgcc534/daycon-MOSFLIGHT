# plan-023 paradigm_analysis — 12 cell large-N sweep

F0 baseline: hit@1cm = 0.6320, hit@1.5cm = 0.8033.

## 12 cell grid (N × τ_cls)

| layout | K | τ=0.001 Δ_1cm/Δ_1.5cm pass | τ=0.003 Δ_1cm/Δ_1.5cm pass | τ=0.005 Δ_1cm/Δ_1.5cm pass |
|---|---|---|---|---|
| B1_dodeca20 | 20 | +0.0193 / +0.0080 ✓ | +0.0124 / +0.0054 ✓ | +0.0081 / +0.0039 ✗ |
| B2_trunc_octa24 | 24 | +0.0200 / +0.0072 ✓ | +0.0122 / +0.0057 ✓ | +0.0083 / +0.0036 ✗ |
| B3_icosidodec30 | 30 | +0.0199 / +0.0077 ✓ | +0.0123 / +0.0057 ✓ | +0.0081 / +0.0040 ✗ |
| B4_fib50 | 50 | +0.0212 / +0.0075 ✓ | +0.0124 / +0.0054 ✓ | +0.0076 / +0.0041 ✗ |

## Best cell 🏆

- **B4_fib50_tau001** (K=50, τ_cls=0.001)
- hit@1cm = 0.6532, hit@1.5cm = 0.8108
- Δ_1cm = **+0.0212** (pass criterion +0.005)
- Δ_1.5cm = **+0.0075** (pass criterion +0.005)
- pass_both = **True**
- Δ sum = +0.0287 (plan-022 best 0.0279 대비 Δ +0.0008)
- max_class_ratio = 0.0316

## Layout-axis marginal (각 N 의 best τ — K trend)

| layout | K | best τ | Δ_1cm | Δ_1.5cm | Δ sum | pass_both | max_class |
|---|---|---|---|---|---|---|---|
| B1_dodeca20 | 20 | 0.001 | +0.0193 | +0.0080 | +0.0273 | True | 0.073 |
| B2_trunc_octa24 | 24 | 0.001 | +0.0200 | +0.0072 | +0.0272 | True | 0.066 |
| B3_icosidodec30 | 30 | 0.001 | +0.0199 | +0.0077 | +0.0276 | True | 0.050 |
| B4_fib50 | 50 | 0.001 | +0.0212 | +0.0075 | +0.0287 | True | 0.032 |

## τ_cls-axis marginal (각 τ 의 best layout)

| τ_cls | best layout | K | Δ_1cm | Δ_1.5cm | Δ sum | pass_both | max_class |
|---|---|---|---|---|---|---|---|
| 0.001 | B4_fib50 | 50 | +0.0212 | +0.0075 | +0.0287 | True | 0.032 |
| 0.003 | B3_icosidodec30 | 30 | +0.0123 | +0.0057 | +0.0180 | True | 0.040 |
| 0.005 | B3_icosidodec30 | 30 | +0.0081 | +0.0040 | +0.0121 | False | 0.037 |

## plan-022 best 대비 compare (A6_bcc14_tau001, Δ_sum=0.0279)

- cells beating Δ sum 0.0279: **1 / 12**
- cells beating Δ_1cm 0.0208: **1 / 12**
- cells beating Δ_1.5cm 0.0071: **4 / 12**
- best in plan-023: **B4_fib50_tau001**

### Cells beating plan-022 Δ sum:

| cell | K | Δ_1cm | Δ_1.5cm | sum | vs plan-022 sum |
|---|---|---|---|---|---|
| B4_fib50_tau001 | 50 | +0.0212 | +0.0075 | +0.0287 | +0.0008 |

## Mode collapse 완화 (max_class_ratio vs 1/K uniform)

| cell | K | max_class | uniform 1/K | ratio (mcr / 1/K) |
|---|---|---|---|---|
| B1_dodeca20_tau001 | 20 | 0.0727 | 0.0500 | 1.45 |
| B1_dodeca20_tau003 | 20 | 0.0587 | 0.0500 | 1.17 |
| B1_dodeca20_tau005 | 20 | 0.0553 | 0.0500 | 1.11 |
| B2_trunc_octa24_tau001 | 24 | 0.0658 | 0.0417 | 1.58 |
| B2_trunc_octa24_tau003 | 24 | 0.0505 | 0.0417 | 1.21 |
| B2_trunc_octa24_tau005 | 24 | 0.0468 | 0.0417 | 1.12 |
| B3_icosidodec30_tau001 | 30 | 0.0500 | 0.0333 | 1.50 |
| B3_icosidodec30_tau003 | 30 | 0.0399 | 0.0333 | 1.20 |
| B3_icosidodec30_tau005 | 30 | 0.0372 | 0.0333 | 1.11 |
| B4_fib50_tau001 | 50 | 0.0316 | 0.0200 | 1.58 |
| B4_fib50_tau003 | 50 | 0.0243 | 0.0200 | 1.22 |
| B4_fib50_tau005 | 50 | 0.0225 | 0.0200 | 1.12 |

### H3 check (effective temperature 변화): K=50 τ=0.001 max_class ≤ K=20 τ=0.001 × 0.5 ?

- K=50 (B4_fib50) τ=0.001 max_class = 0.0316
- K=20 (B1_dodeca20) τ=0.001 max_class = 0.0727
- ratio = 0.435
- H3 PASS = **True**

## G3 — 8/12 effective cell pass_both=True

- Dropped (soft_label_collapse): 0 cells (none)
- Effective denominator: 12 cells