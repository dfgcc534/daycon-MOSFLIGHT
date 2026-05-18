# plan-022 results_A2 — A2_ico13 (icosahedron + center, K=13)

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class |
|---|---|---|---|---|---|---|
| **0.001** | **0.6519** | **0.8103** | **+0.0199** ✓ | **+0.0070** ✓ | **🎉 True** | 0.150 |
| **0.003** | 0.6425 | 0.8084 | +0.0105 ✓ | +0.0051 ✓ | **🎉 True** | 0.109 |
| 0.005 | 0.6388 | 0.8069 | +0.0068 ✓ | +0.0036 ✗ | False | 0.097 |

## Finding

- **Best A2 cell = τ=0.001** (Δ_1cm +0.0199, Δ_1.5cm +0.0070) — A1_octa7_tau001 (Δ_1cm +0.0194, Δ_1.5cm +0.0068) 보다 marginal 향상 (+0.0005 / +0.0002).
- **2 cell PASS_BOTH** (τ=0.001 AND τ=0.003) — A1 (1 cell) 보다 τ_cls robustness 우월.
- max_class_ratio K=13 (~0.10-0.15) < K=7 (~0.18-0.23) — anchor 늘면서 winner-take-all 약화 (expected, soft-mean 의 max share = 1/K 근처).
- Platonic isotropic 효과 미세 (vs A1 marginal +0.0005 1cm).

G2.A2 PASS (3 cell metric finite + max_class_ratio < 0.95).

749s = ~12.5min (3 cell × ~250s, K=13 ≈ 2.17× K=7 train cost).
