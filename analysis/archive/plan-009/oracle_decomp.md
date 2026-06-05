# plan-009 c2.1 — oracle_decomp.md

- source: `runs/baseline/G001_candidate-redefine/oof_selector_scores.npz`
- N samples: 10000, K cands: 25, D dim: 3

## Oracle ceilings (best-raw-cand err thresholds)

| ceiling | value |
|---|---|
| oracle_1cm   | 0.7562 |
| oracle_1.5cm | 0.8701 |
| oracle_2cm   | 0.9057 |

## 8-bin distribution (best-raw-cand err)

| bin | count | fraction |
|---|---|---|
| [0, 0.5cm) | 4896 | 0.4896 |
| [0.5cm, 1cm) | 2666 | 0.2666 |
| [1cm, 1.5cm) | 1139 | 0.1139 |
| [1.5cm, 2cm) | 356 | 0.0356 |
| [2cm, 3cm) | 364 | 0.0364 |
| [3cm, 5cm) | 316 | 0.0316 |
| [5cm, 10cm) | 227 | 0.0227 |
| [10cm, inf) | 36 | 0.0036 |

## Per-band n_in_band (hit_after 분모 anchor, Fix 22 §3.3 박제)

| band | n_in_band |
|---|---|
| [0, 0.5cm) | 4896 |
| [0.5cm, 1cm) | 2666 |
| [1cm, 1.5cm) | 1139 |
| [1.5cm, 2cm) | 356 |
| [2cm, inf) | 943 |

## plan-008 reference match
- oracle_1cm expected (plan-008 c7) = 0.7562
- oracle_1cm actual = 0.7562
- tol = 0.002 — match: **True**
