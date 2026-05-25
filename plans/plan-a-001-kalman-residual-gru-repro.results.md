---
plan_id: plan-a-001
finished_at: 2026-05-26T03:10+09:00
status: all_complete
lane: a
exp_ids_completed:
  - KR001_notebook-repro
  - KR002_input-yaw-rot
best_exp_id: KR002_input-yaw-rot
g_repro_oof_hit_1cm: 0.6639
g_repro_band: EXCELLENT
g_yaw_delta: 0.0024
g_yaw_band: neutral
kalman_alone_hit_1cm: 0.5964
lb_score: null
band: EXCELLENT
---

# plan-a-001 results — Kalman-Residual GRU 노트북 재현 + 입력 yaw 회전 ablation

## §0. 한 줄 결론

**KR001 재현 EXCELLENT** — `notes/LB_0.6780 코드공유.ipynb` 파이프라인을 프로젝트 `stable_fold_id` 5-fold 위에 이식한 OOF hit_1cm = **0.6639**, F0 floor 0.6320 대비 **Δ+0.0319 (p≈0)**, plan-022 best 0.6528·plan-031 0.6397 를 모두 상회하는 **프로젝트 신기록**이며 노트북 자기-split OOF 0.6625 도 +0.0014 상회. paradigm(칼만 CV 잔차 + softhit + tanh×2cm clamp + F/W aux + yaw 타깃)이 프로젝트 split 으로 **robust transfer** 됨이 입증. **KR002(입력 seq yaw 회전) = neutral** — Δ=+0.0024 (p=0.32, 비유의) → H3 적중: 핵심 이득(출력 좌표계 정렬)은 *타깃* 회전이 이미 흡수, *입력* 회전의 순수 추가 기여는 통계적으로 0.

## §0.5 Result Quick Reference

| exp | OOF hit_1cm | hit_1p5cm | config A / B | vs 비교점 | band |
|---|---|---|---|---|---|
| **KR001** (repro) | **0.6639** | 0.8167 | 0.6612 / 0.6630 | F0 +0.0319 (p≈0), 노트북 +0.0014, plan-022 +0.0111 | **EXCELLENT** |
| **KR002** (+input-yaw) | **0.6663** | — | 0.6667 / 0.6671 | KR001 **+0.0024 (p=0.32)** | neutral (ns) |

- baseline: Kalman-alone OOF hit_1cm = **0.5964** (GRU 잔차 미적용, world 직접). F0 = 0.6320/0.8033.
- budget: 2 config(A lr5e-4·p0.3 / B lr1e-3·p0.1) × 5-fold stable_fold_id × 3 seed × 200ep, GPU L40S. KR001 631s / KR002 700s.
- paired permutation 10000 resample (sign-flip). N=10000.

## §1. 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1 재현** (≥F0, plausibly ≥plan-022) | ✅ **확증 (초과달성)** | OOF 0.6639 ≥ F0 0.6320 ✓, ≥ plan-022 0.6528 ✓, ≈ 노트북 0.6625 (+0.0014). paradigm transfer 성공. |
| **H2 부호 역전** (칼만+softhit+tanh → 잔차 GRU 양 기여) | ✅ **확증** | G1 fold0: hit 0.6733 vs Kalman-alone 0.6064 (**+0.067**). full: 0.6639 vs 0.5964 (**+0.0675**). plan-003 linear-extrap 잔차 GRU LB 0.5688 퇴보와 정반대 — 칼만 평활 baseline + metric-aware loss + 출력 clamp 조합이 잔차를 학습가능 구조신호로 전환. |
| **H3 입력 회전 lift 작음** (Δ≈+0.000~+0.005) | ✅ **적중 (neutral)** | Δ(KR002−KR001)=+0.0024 ∈ [0, +0.005] 예측 범위, 단 p=0.32 비유의. "타깃 회전이 이미 이득 흡수" 가설대로 입력 회전의 순수 기여는 0과 구분 불가. 음(−) 아님 → 절대 heading 신호 상실 손해도 미미. |

## §2. Gate 판정

| gate | 결과 | band/severity |
|---|---|---|
| G0 인프라+smoke | green (6 pass) | — |
| G1 1f1s hit > Kalman-alone | 0.6733 > 0.6064 (+0.067) | PASS |
| **G_repro** KR001 full | **0.6639** | **EXCELLENT** (≥0.6600) |
| **G_yaw** KR002 Δ vs KR001 | **+0.0024, p=0.32** | **neutral** (warn — 유의한 lift 없음) |
| G_final | results + sync + merge | 완료 |

## §3. exp 별 산출

### KR001_notebook-repro (G_repro EXCELLENT)
- ensemble OOF hit_1cm **0.6639** / hit_1p5cm 0.8167. config A 0.6612, B 0.6630 (ensemble > 양 config).
- vs F0 0.6320: Δ **+0.0319**, paired permutation **p≈0** (고도 유의).
- calibration add-on: 하드코드 α=(1,0.95,1) → 0.6638, OOF-fit α=(1.025,1,1) → 0.6641 (**+0.0002, 무변화**). overfit-risk flag=True (fit-α y축 1.0 이 노트북 0.95 와 0.05 deviation — 프로젝트 데이터는 y축 축소 비선호). **headline = uncalibrated 0.6639 확정**.
- 산출: `analysis/plan-a-001/results_kr001.json` + `.npz` (oof_pred/per_sample_hit).

### KR002_input-yaw-rot (G_yaw neutral)
- ensemble OOF hit_1cm **0.6663** (A 0.6667, B 0.6671). vs KR001 Δ=+0.0024, p=0.32 → **neutral**.
- **단일 변수 격리 audit**: scalar 40D 전부 회전불변(magnitude/cos, directional 0개) → 회전 비대상. seq rel/v/a (ch0-8) 만 rotate_xy(θ) 적용 (z 보존, last-step v_y≈0 yaw 정렬 확인). `rotation_class` 박제 in results_kr002.json.
- 산출: `analysis/plan-a-001/results_kr002.json` + `.npz`.

## §4. 해석 + 함의

1. **프로젝트 최고 OOF 갱신**: KR001 0.6639 가 기존 anchor/selector paradigm best(plan-022 0.6528) 를 +0.0111 상회. 칼만-잔차 paradigm 이 anchor paradigm 대비 우월한 OOF 를 (CPU under-convergence 의심 없이 GPU full-budget 으로) 달성. 단 이는 OOF 지표이며 LB 제출은 out-of-scope (DACON quota 정책).
2. **plan-003 부호 역전 메커니즘 규명**: linear extrap → 칼만 평활 baseline 교체가 잔차의 SNR 을 높여 GRU 가 jitter 가 아닌 구조신호를 학습 (G1 +0.067). softhit + tanh×2cm 가 metric 정렬 + 출력 clamp 로 경계점 손해를 차단.
3. **입력 좌표계 회전은 redundant**: 타깃을 yaw 좌표계로 두면 입력까지 회전해도 추가 이득이 통계적으로 0 (KR002 neutral). 노트북이 입력을 world 로 둔 설계 선택이 합리적임을 ablation 으로 확인 — 입력 속도 채널이 heading 을 이미 담아 GRU 가 self-rotate 가능.

## §5. Follow-up 후보 (번호 미할당)

- **칼만-잔차 paradigm 의 LB 검증**: 본 plan 은 OOF only. KR001 을 test 예측 + DACON 제출로 LB 회수 (사용자 quota confirm 필요) — OOF 0.6639 가 LB 0.68 대 노트북(0.6778) 과 어떻게 대응하는지.
- **anchor paradigm 과 칼만-잔차 ensemble**: 서로 다른 계보(0.6528 vs 0.6639) 의 OOF 상관 낮으면 blend 이득 가능성.
- **σ_obs mini-grid / aux head 기여 ablation**: 현 best σ(0.3mm) 고정 — {0.1,0.3,1.0}mm sweep + aux F/W drop 단일변수로 각 lever ROI 분해.
- **calibration overfit 정밀화**: fit-α y축 deviation 의 fold 별 안정성 점검 (현 headline uncalibrated 유지).

## §6. 재현

```
python analysis/plan-a-001/run_oof.py --gate g1                              # G1
python analysis/plan-a-001/run_oof.py --gate full --out results_kr001.json   # KR001
python analysis/plan-a-001/run_oof.py --gate full --input-yaw --out results_kr002.json  # KR002
python analysis/plan-a-001/compare.py                                        # paired permutation
```
- 5-fold = stable_fold_id (노트북 KFold 대신, 프로젝트 OOF 호환 — decision-note). noise_loo cache: `noise_cache.npz` (gitignore).
