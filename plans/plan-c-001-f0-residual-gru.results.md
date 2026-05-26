# plan-c-001 results — F0(perp=0.0) 잔차 GRU (FR001)

plan: `plans/plan-c-001-f0-residual-gru.md` (v2) · date: 2026-05-26 (Asia/Seoul) · lane c · status: **G_final**

## TL;DR

> **FR001 (F0(perp=0.0) 잔차 GRU + F0-자기진단 feature) OOF hit_1cm = 0.6622** (hit_1p5cm 0.8150).
> F0(perp=0.0) floor 0.6320 대비 **+0.0302 lift** → **G_f0 PASS** (bar floor+0.02=0.6520 여유 통과).
> Kalman KR002 OOF 0.6663 대비 **Δ −0.0041** (strong bar 직전, 사실상 동급). **잔차 GRU paradigm 이 baseline 을 Kalman→F0(perp=0.0) 로 바꿔도 lift 대부분 보존** (H2 사실상 확인).
> LB 미검증 (CV-LB 괴리 — §LB, 사용자 confirm gated).

## 1. 결과 (full OOF: 2cfg A/B × 5fold stable_fold_id × 3seed × 200ep, N=10000, GPU L40S, 557.6s)

| 항목 | hit_1cm | hit_1p5cm |
|---|---|---|
| **FR001 (ensemble A+B, uncalibrated)** | **0.6622** | 0.8150 |
| config A | 0.6615 | — |
| config B | 0.6619 | — |
| F0(perp=0.0) floor (baseline-alone) | 0.6320 | 0.8058 |
| **lift vs floor** | **+0.0302** | +0.0092 |

비교 기준 (carry):
- **F0(perp=0.0) floor** = 0.6320 (= plan-020 F0 perp=−0.20 와 1cm 동일; perp 제거가 1cm floor 불변). fold_mean 0.6319±0.0066.
- **KR002 (Kalman CV 잔차 GRU)** OOF 0.6663 / LB 0.6818 — FR001 Δ_OOF = **−0.0041**.
- KR001 0.6639 (Kalman, input-yaw off). KR001 의 floor 대비 lift = +0.0319 ≈ FR001 의 +0.0302 (잔차 GRU lift 크기 baseline 무관 보존).

**G_f0 band = PASS** (≥ floor+0.02; < KR002 0.6663 라 strong 미달, but 거의 동급).

## 2. Attribution (1-fold f0=0, 1-seed, 200ep — *약신호*, CV-LB 괴리로 OOF attribution 한계 명시)

| 변경 | fold0 OOF hit_1cm | Δ |
|---|---|---|
| F0(perp=0.0) floor (fold0 subset) | 0.6441 | — |
| (i) baseline-swap only (F0 잔차 GRU, **no** F0-feats) | 0.6733 | **+0.0292** vs floor |
| (ii) + F0-자기진단 feature (FR001) | 0.6738 | **+0.0005** vs (i) |

- **(i) baseline-swap = lift 의 거의 전부** — F0(perp=0.0) 위 잔차 GRU 학습이 floor 대비 +0.029 (KR002 paradigm 의 baseline-swap 이 F0 에서도 작동, H1 확인).
- **(ii) F0-자기진단 feature = OOF-neutral (+0.0005)** — KR002 의 입력 yaw 회전이 OOF-neutral 이었던 것과 동형. **OOF 만으로 폐기 금지** (CV-LB 괴리): plan-030:69 가 L2 step-별 F0 잔차를 PB selector 핵심 lift 신호로 박제 → test LB 에서 양 가능성. 1-fold·1-seed 라 신뢰도 낮음 (방향 힌트).

## 3. 가설 verdict

- **H1 (baseline swap 작동)**: ✅ PASS. F0(perp=0.0) 잔차 GRU OOF 0.6622 ≥ floor+0.02. attribution (i) +0.029 가 직접 근거.
- **H2 (Kalman 동급 가능)**: ✅ 사실상 확인. FR001 0.6622 vs KR002 0.6663 = Δ −0.0041 (Kalman 의 denoise 우위 일부만 미회수). 잔차 GRU lift 크기(+0.030)가 Kalman-baseline KR001 lift(+0.0319)와 동급 → "lift 는 baseline 품질보다 GRU 잔차 학습에서 온다" 지지.
- **H3 (CV-LB 괴리 재현)**: 미검증 (LB 미제출). OOF 에서 F0-feature neutral 이나 LB 검증 필요 (§LB).
- **메타 (Kalman 부산물 제외 정당성)**: ✅ F0(닫힌형)에 innovation/filtered-v/CV-CA 부재가 구조적 사실로 확인 (§1.2). innovation *개념* 은 F0 per-step 잔차(§1.3)로 회수 — feature 자체는 동작(NaN/발산 0), OOF-neutral.

## 4. §1.2 제외 (Kalman 부산물) ↔ §1.3 회수 (F0 자기진단) — 사후 audit

| Kalman 부산물 (제외) | 처리 | F0 버전 (§1.3 회수) |
|---|---|---|
| innovation `z(t)−state_pred(t)` | 제외 (재귀필터 산물) | → F0 per-step 잔차 `r_w(t)=f0_perp0(X[:,t-4:t-1])−X[:,t]` **회수** (L2 seq +3 / f0_conf +2 / EWMA +3 / STA-LTA +3) |
| filtered velocity | 제외 | 부재 (F0=raw v_last) |
| CV/CA 불일치 | 제외 | 부재 (단일 닫힌형) |
| gain K / covariance P | 제외 | 부재 (개념 없음) |
| filtered-yaw frame (KR004) | 제외 | 부재 → raw-yaw 유지 |
| W-aux (alt-Kalman target) | λ_W=0 비활성 (architecture byte-동일) | — |

입력: seq 9→**12채널** (L2 F0-잔차 slot=프레임 t, t<4 zero-pad), scalar 40→**48d**. 전부 f0_perp0 주입(baseline 일관)·yaw frame·관측창 leakage 무 (smoke 5 assert PASS).

## 5. Gate 통과

| gate | 결과 |
|---|---|
| G0 (smoke 5 assert) | ✅ import·finite·target⇄복원 정합(+음성통제 θ/base 뒤집기 fail)·F0-잔차 leakage(slot<4 zero)·W-aux grad0 |
| G1 (1-fold 1-seed) | ✅ 0.6738 ≥ floor(fold0)−0.005=0.6391 |
| G_f0 (full OOF) | ✅ **PASS** 0.6622 ≥ floor+0.02=0.6520 (paired permutation deferred — band 명확) |
| G_final | ✅ results 박제 + §0.5 sync + main merge |

NaN/Inf/divergence 0. seq(10000,11,12)·scal(10000,48). runtime 557.6s (GPU).

## LB (사용자 confirm gated)

- 본 plan headline = **OOF**. FR001 0.6622 는 KR002 OOF(0.6663)보다 약간 낮으나, **CV-LB 괴리** (입력 yaw·KR003 가 OOF-neutral·LB-positive 2연속) 때문에 OOF 열위가 LB 열위를 의미하지 않음.
- F0-자기진단 feature 가 OOF-neutral(+0.0005)인데 plan-030 이 L2 를 PB selector 핵심 신호로 박제 → **LB 에서 양 가능성**. 단 FR001 OOF 가 KR002·KR003(LB 0.6854) 보다 낮아 *현 LB 신기록 경신 기대는 낮음*.
- **DACON 제출 = 사용자 명시 confirm 필요** (quota, 특히 OOF Δ<0 라 더더욱 — [[feedback_dacon_submit_confirmation]]). 미제출.

## 6. 산출물

- `analysis/plan-c-001/f0_baseline.py` (f0_perp0 + floor), `f0_residual_feats.py`, `run_oof.py`, `tests/test_plan_c001_smoke.py`.
- `analysis/plan-c-001/f0_perp0_floor.json`, `results_fr001.json` (+`.npz`: oof_residual/oof_pred/per_sample_hit/y/base_pred/fold_ids), `results_fr001_g1.json`, `results_fr001_g1_nofeat.json` (attribution).

## 7. decision-note

- swap+feature 2변경의 attribution 은 1-fold ablation 으로 분리(필수 수행) — full 2-OOF 분리는 budget·CV-LB 괴리로 scope-out (§3 plan).
- F0(perp=0.0) floor = 0.6320 (perp 제거가 1cm 불변) → 본 plan 의 "perp 단일 계수 변경"은 baseline 품질 무손실, 잔차 GRU 가 lift 담당 확인.
- 후속 후보: F0-자기진단 LB 제출(사용자 confirm), perp on/off disagreement aux, L4 soft-hit, per-step F0 잔차 풍부화.
