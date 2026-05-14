# IMM feasibility — oracle ceiling

- N = 10000, hit threshold = 1cm, target = t=+80ms forward prediction.

## Single-mode hit@1cm

| mode | hit@1cm | pred_err p50 (m) | p95 (m) |
|---|---|---|---|
| CV | 0.1273 | 0.0324 | 0.1185 |
| CA | 0.2123 | 0.0220 | 0.1033 |
| CT | 0.1124 | 0.0329 | 0.1259 |

**Best single-mode: CA = 0.2123**

## IMM oracle ceiling

- oracle_imm_hit_1cm (per-sample best-pred-mode) = **0.2851**
- ΔLB upper bound (vs best single-mode CA) = **+0.0728**

## Comparison to baselines

| baseline | LB | Δ vs oracle |
|---|---|---|
| plan-014/015 best_stack | 0.6628 | -0.3777 |
| plan-016 G1 multi-seed | 0.6638 | -0.3787 |

> 주의: oracle 은 *post-hoc* (Y label 사용한 best-mode picker) 라 *실제 IMM 으로 달성 가능한 LB 의 절대 upper bound* 만 의미. 실제 IMM 은 inferred posterior 사용하므로 oracle 아래.

## Best-pred-mode distribution

| mode | count | fraction |
|---|---|---|
| CV | 2777 | 27.77% |
| CA | 5319 | 53.19% |
| CT | 1904 | 19.04% |

## BIC-label vs best-pred-mode agreement: **53.17%**

Confusion matrix (rows = BIC label, cols = best-pred-mode):

|   | CV | CA | CT |
|---|---|---|---|
| **CV** | 246 | 263 | 122 |
| **CA** | 2366 | 4803 | 1514 |
| **CT** | 165 | 253 | 268 |

> Low agreement (< 50%) 시 BIC fit-residual 분류가 forward-prediction task 와 misaligned → IMM verdict 보수적 해석 필요.