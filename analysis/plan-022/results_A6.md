# plan-022 results_A6 — A6_bcc14 (BCC 14 = octahedron 6 axis + cube 8 corner, NO center)

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class |
|---|---|---|---|---|---|---|
| **0.001** | **0.6528** | **0.8104** | **+0.0208** ✓ | **+0.0071** ✓ | **🎉 True (NEW BEST)** | 0.105 |
| **0.003** | 0.6449 | 0.8091 | +0.0129 ✓ | +0.0058 ✓ | **🎉 True** | 0.085 |
| 0.005 | 0.6404 | 0.8069 | +0.0084 ✓ | +0.0036 ✗ | False | 0.080 |

## Finding — Δ sum NEW HIGH

- **Best A6_tau001: Δ_1cm +0.0208 = sweep 1cm NEW HIGH** (vs A2 0.0199, A1 0.0194). Δ_1.5cm +0.0071 (3rd, A5/A3 보다 약간 낮음).
- **Δ sum 0.0279 = sweep NEW BEST overall** (vs A3 0.0270, A2 0.0269).
- BCC 14 = octahedron 6 axis (sharp 1-axis 오류 cover) + cube 8 corner (octant 결합 오류 cover) **두 paradigm 결합 효과**:
  - 1cm tight: axis anchor 가 단일축 오류 정확히 cover (A1 처럼)
  - 1.5cm zone: corner anchor 가 3축 결합 오류 cover (A5 처럼)
  - 두 type 동시 활성화 → 1cm 우월 + 1.5cm 견조 → sum 1위
- **2 cell PASS_BOTH** (τ=0.001, τ=0.003) — A2 와 동일 robustness.
- max_class_ratio 가장 낮음 (0.080-0.105) — K=14 의 1/K=0.071 와 거의 같음, **collapse 거의 없음** (가장 잘 distributed).

G2.A6 PASS. 817s = 3 cell × ~272s.

## 사용자 narrative 합치

사용자가 본 plan-022 brainstorming 시 처음 제안한 layout (14-anchor "정14면체") = A6 이며, **실제 sweep 의 winner**. 직관 정확 — octahedron + cube 결합이 1cm tight (Δ_1cm 최강) + 1.5cm 견조 + collapse 회피 (max_class 최저) 의 3-way 동시 달성.
