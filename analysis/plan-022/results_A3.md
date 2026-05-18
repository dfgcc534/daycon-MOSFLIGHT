# plan-022 results_A3 — A3_cubocta13 (cuboctahedron + center, K=13, FCC neighbor uniform edge)

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class |
|---|---|---|---|---|---|---|
| **0.001** | **0.6516** | **0.8107** | **+0.0196** ✓ | **+0.0074** ✓ | **🎉 True** | 0.151 |
| 0.003 | 0.6439 | 0.8082 | +0.0119 ✓ | +0.0049 ✗ | False | 0.109 |
| 0.005 | 0.6390 | 0.8069 | +0.0070 ✓ | +0.0036 ✗ | False | 0.097 |

## Finding

- **Best A3 cell = τ=0.001** (Δ_1cm +0.0196, **Δ_1.5cm +0.0074 = current sweep 최고**).
- FCC cuboctahedron 우월점: 1.5cm metric 최강 — vertices 가 axis 결합 방향 ((±a, ±a, 0) 패턴) 이라 normal+tangent 결합 오류 sample 을 잘 cover.
- 1cm metric 은 A2 ico13 (+0.0199) 보다 -0.0003 약함 — Platonic isotropic 의 sharp peak 대비 cubocta 의 axis-aligned 결합이 1cm tight zone 에 미세 손실.
- Δ_1cm + Δ_1.5cm = 0.0270 (= A2 의 0.0269 보다 marginal +0.0001) — best by sum 기준 현재 1위.
- max_class_ratio K=13 우월 (~0.10-0.15, A1 의 ~0.18-0.23 대비).

G2.A3 PASS (3 cell metric finite + max_class_ratio < 0.95).

800s = 3 cell × ~265s.
