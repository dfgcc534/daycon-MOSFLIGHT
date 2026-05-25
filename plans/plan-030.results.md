---
plan_id: plan-030
status: complete
based_on: plan-029
title: GRU-attention residual injection (results)
g3_oof_hit_1cm: 0.6294
g3_band: FAIL_regression
g1_oof_hit_1cm: 0.6436
primary_root_cause: "candidate 표현 의미 비일관성 — anchor 가 각 궤적 진행 맥락과 분리된 고정 거리·방향 기하 격자라 'anchor k' 의미가 샘플마다 흔들려 학습 불가. 구조적 천장 ≈0.63. 상세 §2.0"
followed_by: plan-031 (PB training procedure carry — *proximate* lever; §2.0 구조적 천장과는 별개 axis)
---

# plan-030 results

## §0. 한 줄 결론

**G3 FAIL_regression**: 5-fold OOF hit_1cm = **0.6294** < F0 baseline 0.6320 (Δ = **−0.0026**), plan-029 X1 0.6316 대비 **−0.0022 worse**.

**근본 실패 사유 (PRIMARY — §2.0)**:

> **plan-030 의 anchor 는 각 궤적의 진행 맥락과 분리된 고정 거리·방향의 기하학적 격자점이라, 같은 'anchor k' 가 샘플마다 다른 움직임을 가리키게 되고 — 모델이 '이 후보 = 이런 의미의 미래' 라는 일관된 대응을 학습할 수 없다.**

→ 이건 input feature axis fix (잔차 a/b) 나 training procedure 교체로는 못 넘는 **구조적 천장 (≈ 0.63)**. 부차 사유 (cross-fold variance, input axis net-negative, logit under-sharpening, training procedure gap) 는 §2.1 에 압축. 후속 방향 = §2.0 의 "후보를 안정 motion frame 에 묶기 (yaw) / 연속 회귀 전환". (plan-031 의 PB training procedure carry 는 *proximate* lever 로 별도 진행됨 — §3.)

## §0.5 Result Quick Reference

| 항목 | 값 |
|---|---|
| **G3 OOF hit_1cm** | **0.6294** |
| G3 OOF hit_1p5cm | 0.8033 |
| G3 max_class_ratio | 0.1227 (no mode collapse) |
| G3 top1_acc | 0.123 |
| G3 band | **FAIL_regression** (< F0 0.6320) |
| G1 (fold-0) hit_1cm | 0.6436 (PASS, F0 +0.0116) |
| G1 → G3 gap | **−0.0142** (cross-fold variance 큼) |
| F0 baseline | 0.6320 |
| plan-029 X1 (carry) | 0.6316 |
| plan-024 honest ceiling | 0.6387 |
| 5-fold elapsed | 681s ≈ 11분 24초 (CPU) |
| 1-fold elapsed | 140s |
| N_total | 10000 |
| K (anchors) | 14 |

## §1. Gate 진행

| gate | 결과 | 값 | PASS? | 사유 |
|---|---|---|---|---|
| G0 (data) | DONE | finite, max_class_ratio 0.1227 | ✓ | upstream cache 정합 |
| G1 (smoke) | DONE | 20/20 pytest green | ✓ | builder + model + train + finite |
| G2 (1-fold) | DONE | hit_1cm 0.6436 | ✓ (PASS > 0.6290) | fold-0 noise 가능성 — G3 와 큰 gap |
| G3 (OOF 5-fold) | DONE | hit_1cm 0.6294 | ✗ **FAIL_regression** | F0 baseline -0.0026, plan-029 X1 -0.0022 |
| G_final | N/A | — | ✗ | G3 fail → §5 fallback rule = plan-031 escalate |

## §2. 분석

### §2.0 [PRIMARY] candidate 표현의 의미 비일관성 — 구조적 천장 ≈ 0.63

**한 줄 진단**:
> plan-030 의 anchor 는 각 궤적의 진행 맥락과 분리된 고정 거리·방향의 기하학적 격자점이라, 같은 'anchor k' 가 샘플마다 다른 움직임을 가리키게 되고 — 모델이 '이 후보 = 이런 의미의 미래' 라는 일관된 대응을 학습할 수 없다.

**메커니즘 (왜 plan-030 에 정확히 꽂히나)**:

1. **의미 있는 후보 = 행동 모드**여야 함. "직진 지속 / 급좌회전 / 감속" 처럼 *모든 샘플에서 같은 뜻* 을 갖는 미래 종류라야 모델이 "feature 패턴 → 모드" 대응을 학습 가능.
2. **plan-030 의 `ANCHORS_A6` = 기하학적 격자점**: 잔차 공간을 균등 타일링한 BCC codebook 좌표일 뿐 행동 모드 아님. Frenet (`R_wfn`) 로 회전을 맞춰도 —
   - **tangent 축** (속도 방향) 은 그나마 일관 ("앞으로 더/덜").
   - **normal/binormal 축** 은 *수직 가속도 방향* 기준인데, 직진·저가속 궤적에선 **degenerate → world-z fallback** (`analysis/plan-021/build_input.py:build_frenet_basis_3d` 의 `~safe_perp` 분기). 즉 옆/위 anchor 의 기준축이 샘플마다 **임의로 바뀜**.
   - → "anchor 7" 이 어떤 샘플엔 "왼쪽 swerve", 다른 샘플엔 "위 상승" 을 뜻함. **의미 미고정**.
3. **soft-label (τ_cls=0.001) 타깃 자체가 의미적으로 noisy**: "이 샘플 ≈ anchor 7" 이라 가르쳐도 anchor 7 의 의미가 흔들리니 분류 신호가 일관성을 잃음 → 모델이 배울 안정 신호 부재.

**지지 증거** (기존 §2.x 관측을 이 진단으로 재해석):
- **max_class_ratio 0.1227 / top1_acc 0.123** (uniform 1.72×): anchor selection 이 *informative 하나 sharp 하지 않음* = "어느 후보인지 확신 못 함" = 의미 비일관성의 직접 징후 (mode collapse 아님).
- **logit-score std 0.78 → 2.74** (목표 1/τ_cls = 1000 대비 한참 미달): 후보 간 구분 신호가 약해 logit 을 못 키움.
- **`paradigm_root_cause.md` E2**: candidate paradigm 을 anchor residual ↔ F0 hypothesis 로 *바꿔도* 0.6309 ≈ X1 0.6316 (neutral). → **codebook 종류 무관하게 동일 ~0.63 천장** = 공통 결함 (격자에 motion-frame 의미 부재) 이 paradigm-wide ceiling 임을 시사.

**함의 (후속 방향)**:
- 후보를 **각 샘플의 안정적 motion frame 에 묶어** 전 샘플 동일 의미를 부여해야 함. **yaw heading frame** (모기가 움직이는 한 degenerate 안 됨 — Frenet normal 과 대비) 이 직접 해법 후보: "forward/lateral/vertical anchor" 가 전 샘플 동일 의미.
- 한 발 더: 후보 크기를 그 샘플 속도·변위에 비례시키면 "현 속도 유지" 같은 모드가 한 점으로 모임.
- **또는** 후보 표현 자체를 버리고 연속 회귀 (`notes/LB_0.6780 코드공유.ipynb` 식 = 칼만 잔차 직접 regression, LB 0.6780) — 격자 양자화 천장 자체가 없음.
- ⚠️ 본 진단은 *구조적 가설* (ablation 미검증). 검증 경로 = (i) 잔차 vs Frenet normal degeneracy flag 상관, (ii) yaw-frame anchor 교체 시 1-fold lift, (iii) 연속 회귀 baseline 비교.

### §2.1 [부차, 압축] 그 외 관측

- **cross-fold variance −0.0142** (G1 0.6436 → G3 0.6294): single-fold noise (≈0.021) 범위 내이나 systematic drift 가능. 추정 = 잔차 (a)/(b) zero-pad step (i=5,6, t_wall=−1,0) 의 fold 별 distribution shift, 또는 head MLP 382D over-param.
- **input axis fix net-negative** (X1 0.6316 → 0.6294, −0.0022): 잔차 (a)35D+(b)35D 추가 효과 ≈ 0~slight negative. single-head K/V tied attention 이 raw bottleneck-bypass 신호를 효과적으로 활용 못함 가능.
- **logit under-sharpening** (score_std 0.78→2.74 ≪ 목표 1000): learnable τ_model 또는 score-scale aux loss 가 lever 후보 (§2.0 의 징후이기도).
- **training procedure gap** (`paradigm_root_cause.md`): PB selector 0.6511 vs 단순 50ep 0.63 의 −0.02 = multi-phase + pairwise + prior + distill 누락 → plan-031 carry lever. **단 §2.0 천장과 별개 axis** — 학습을 고쳐도 후보 의미 비일관성 천장은 잔존.

## §3. Decision

**(A) PRIMARY — §2.0 구조적 천장 해소 (후속 plan 후보)**:
- 후보를 안정 motion frame (yaw heading) 에 묶어 전 샘플 동일 의미 부여, 또는 후보 표현 폐기 후 연속 회귀 (`notes/LB_0.6780` 식). 먼저 §2.0 검증 경로 (i)~(iii) 1-fold 실측으로 가설 확정 권장.

**(B) PROXIMATE — plan-031 진입 (실제 진행됨)**: plan-030 §5 fallback rule (G3 fail → PB training procedure carry). lever 압축 = multi-phase (pre/fine/freeze/epoch_plus) + pairwise margin (0.12) + regime/class prior (0.45/0.65) + fine-distill (0.55/0.07) + reverse-pretrain + batch4096/hidden48. 추정 lift +0.015~0.02. **단 (B) 는 PB-vs-X1 gap 의 lever 일 뿐 (A) 의 천장은 못 넘음** — input feature axis (residual block + Q/K) 는 plan-031 carry.

## §4. Ablation (§7 deferred, not run)

§5 fallback rule 의 c10~c13 ablation (잔차 a/b/slim7/head sample summary drop) 은 본 plan 에서 *미실행* — main lever (training procedure) 가 부재한 상태에서의 ablation 은 정보 가치 낮음. plan-031 의 PB carry 후 input axis ablation 재진행 권장.

## §5. Artifact

- `analysis/plan-030/results_g1.json` — G1 fold-0 metric + score_std_trajectory
- `analysis/plan-030/results_g3.json` — G3 5-fold OOF metric + hparams + fold_logs
- `analysis/plan-030/results_g3.npz` — oof_pred (10000, 3) + oof_probs (10000, 14)
- `tests/test_plan030_smoke.py` — 20 pytest green

## §6. Decision-note 박제

decision-note: plan-030 G3 FAIL_regression (hit_1cm 0.6294 < F0 0.6320). **PRIMARY root cause (§2.0) = candidate 표현 의미 비일관성** — anchor 가 진행 맥락과 분리된 고정 거리·방향 기하 격자라 'anchor k' 의미가 샘플마다 흔들려 (Frenet normal degeneracy + BCC 격자) 학습 불가, 구조적 천장 ≈0.63 (E2 codebook 무관 ~0.63 천장이 지지). 부차 사유 (cross-fold variance −0.0142, input axis net-negative, logit under-sharpening, training procedure gap) 는 §2.1 압축. 후속 = 후보를 yaw motion frame 에 묶기 / 연속 회귀 전환 (§2.0 검증 경로 i~iii). plan-031 PB training procedure carry 는 proximate lever 로 별도 진행 (천장과 별개 axis).
