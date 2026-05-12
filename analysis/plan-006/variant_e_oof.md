# plan-006 Variant E OOF Hit (physics_bias × 0.65 + soft averaging, temp=0.03)

## 전체 hit

- **E_corrected.soft** = `0.6524` (main metric)
- E_corrected.argmax = `0.6491` (informational, score sample-invariant)
- E_raw.soft = `0.6250`
- E_raw.argmax = `0.6320`
- **F_corrected.soft** (uniform sanity) = `0.6520`

## 비교 (plan-005 측정/추정)

| Variant | OOF hit (soft) | 출처 |
|---|---|---|
| full (GRU + physics + regime) | 0.6599 | plan-005 측정 |
| Variant A (GRU + physics, no regime) | 0.6570 | plan-005 측정 |
| Variant B (physics + regime, no GRU) | 0.6547 | plan-005 측정 |
| **Variant E (physics 만, no GRU/regime)** | **0.6524** | **plan-006 측정** |
| Variant E (raw cands, no corrector) | 0.6250 | plan-006 informational |
| Variant F (uniform, no physics) | 0.6520 | plan-006 sanity |

## physics_bias 해석

- argmax 후보: **`frenet_par120_perp_neg020`** (idx=17)
- top-5: `frenet_par120_perp_neg020`, `frenet_best`, `frenet_par100_perp000`, `frenet_par120_perp020`, `frenet_par110_perp_neg020`

## Per-regime hit

| regime | n | E_corrected_soft | E_raw_soft |
|---|---|---|---|
| 0 | 661 | 0.8835 | 0.8805 |
| 1 | 629 | 0.9300 | 0.9205 |
| 2 | 663 | 0.8597 | 0.8643 |
| 3 | 458 | 0.8865 | 0.8777 |
| 4 | 615 | 0.5967 | 0.5431 |
| 5 | 274 | 0.7080 | 0.6606 |
| 6 | 544 | 0.7978 | 0.7665 |
| 7 | 701 | 0.7789 | 0.7618 |
| 8 | 592 | 0.7466 | 0.7230 |
| 9 | 562 | 0.7153 | 0.6797 |
| 10 | 546 | 0.4103 | 0.3608 |
| 11 | 355 | 0.4338 | 0.3746 |
| 12 | 549 | 0.6120 | 0.5829 |
| 13 | 916 | 0.5721 | 0.5426 |
| 14 | 476 | 0.5105 | 0.4643 |
| 15 | 749 | 0.4579 | 0.4299 |
| 16 | 354 | 0.2203 | 0.1893 |
| 17 | 356 | 0.2584 | 0.2275 |
