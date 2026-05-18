# plan-022 results_A5 — A5_cube8 (cube corners only, K=8, NO center)

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class |
|---|---|---|---|---|---|---|
| **0.001** | **0.6500** | **0.8109** | **+0.0180** ✓ | **+0.0076** ✓ | **🎉 True** | 0.165 |
| 0.003 | 0.6425 | 0.8082 | +0.0105 ✓ | +0.0049 ✗ | False | 0.142 |
| 0.005 | 0.6395 | 0.8068 | +0.0075 ✓ | +0.0035 ✗ | False | 0.135 |

## Finding — H3 partial supported

- Best A5_tau001: Δ_1cm +0.0180, **Δ_1.5cm +0.0076 = sweep NEW HIGH** (vs A3 0.0074).
- Δ sum 0.0256 < A3 0.0270 (1cm 손실로 sum 떨어짐).
- 가설 H3 (center 제거 → F0 over-pick 차단 → 1.5cm 향상) **partial supported**:
  - 1.5cm metric: A5 > A3 > A2 > A1 > A4 (best 0.0076 vs worst 0.0061)
  - 1cm metric: A5 < A1/A2/A3 (-0.0014 ~ -0.0019)
- 추정 mechanism: A5 의 8 corner 만 → final Frenet norm = 항상 0.005m (octant 방향). F0 가 정확한 sample (잔차 ≈ 0) 도 강제로 0.005m 떨어진 corner 로 assign → **1cm tight zone 손실** (0.005m corner ↔ 1cm 정답 거리 ≈ 0.0087m > 0.01m boundary 일부 fail). 잔차가 0.5cm 이상인 sample (1.5cm zone) 에선 corner 가 3축 결합 방향 cover 우월.
- max_class_ratio (0.135-0.165) — K=8 의 1/K=0.125 와 가까움. low-collapse.

G2.A5 PASS. 399s = 3 cell × ~133s (K=8 빠름).
