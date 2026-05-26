---
plan_id: c-001
version: 2
date: 2026-05-26 (Asia/Seoul)
status: draft
lane: c
inspired_by:
  - a-002 (Kalman 부산물 feature plan. **본 plan = a-002 의 구조·실험 프로토콜을 그대로 carry 하되 baseline predictor 를 Kalman CV → F0(perp=0.0) 로 교체**. a-002 의 모든 입력 feature·frame·aux 가 Kalman 부산물이라 전부 제외 → §1.2 제외 보고)
  - a-001 (KR002 = Kalman CV 잔차 GRU + 입력 yaw 회전, 프로젝트 LB 신기록 0.6818. 핵심 발견 = **CV-LB 괴리**. 본 plan paradigm 의 모체 — baseline predictor 만 F0 로 swap)
  - 020 (F0 baseline = `p0 + 1.98·v_last + 1.20·acc_par + PERP·acc_perp`, hit_1cm 0.6320 floor. 본 plan = **PERP 계수만 -0.20 → 0.0** 으로 변경한 F0(perp=0.0) 를 잔차 GRU 의 baseline 으로 사용)
code_reuse:
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: 데이터 loader. X (N,11,3), y (N,3).
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: MD5 5-fold split (plan-a-001/a-002/OOF 비교 호환).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [R_HIT, R_HIT_LOOSE, D1, PAR, PERP, f0_baseline]
    reason: F0 floor + hit metric + F0 공식 본체. 본 plan 은 **PERP 만 0.0 으로 override** 한 변형(f0_perp0)을 신규 모듈에 별도 작성 (plan-020 상수 불변 보존).
  - module: analysis/plan-a-001/yaw.py
    symbols: [yaw_angle, rotate_xy, inverse_rotate_xy]
    reason: yaw 좌표계 회전 (KR002 lever, raw v_last 기반 — Kalman 무관). 본 plan 그대로 사용.
  - module: analysis/plan-a-001/features.py
    symbols: [build_seq_t3, build_scalar_40d, compute_noise, normalize_seq]
    reason: KR002 입력 파이프라인. **신규 채널 0 — Kalman 부산물 채널 전부 제외(§1.2)**, seq 9채널 + scalar 40d 그대로 carry.
  - module: analysis/plan-a-001/model.py
    symbols: [GRUModelMultiAux]
    reason: GRU+F/W multi-aux. 구조 **byte-동일 carry**. 단 W aux 의 target(alt-Kalman)이 Kalman-유도 → **λ_W=0 으로 비활성**(architecture 불변, gradient 0). F aux(naive last-pos)는 Kalman 무관 → 유지.
  - module: analysis/plan-a-001/losses.py
    symbols: [loss_combo, loss_aux_euclid]
    reason: combo(euclid+0.3 softhit) + aux. 변경 없음 (λ_W=0 만 cfg 로 전달).
  - module: analysis/plan-a-001/run_oof.py
    symbols: [main]
    reason: 5-fold OOF runner. baseline predictor 를 `--baseline f0-perp0` flag 로 swap (kalman_predict → f0_perp0). `--input-yaw`·`--f0-resid-feats` carry.
  - module: analysis/plan-021/build_input.py
    symbols: [_build_L2_L4]
    reason: **F0 고정공식 per-step 잔차(L2) 산출 로직 재사용** — f0_baseline_fn 자리에 f0_perp0 주입(잔차 = 본 plan baseline 과 일관). 본 plan 은 anchor-free → L2(7step×3=21) 만 추출(L4 soft-hit 제외, redundant), Frenet 대신 KR002 yaw frame(rotate_xy) 으로 회전. §1.3.
  - module: analysis/plan-024/cand_builder.py
    symbols: [build (ctx base/A1/A8 묶음 로직)]
    reason: **F0 잔차 sample-level 자기진단** (f0_conf 2 = 마지막 잔차 norm+step-speed spread, EWMA(α=0.3) F0-residual 3, A1 STA/LTA EWMA(0.5/0.1) 비 3) 산출 로직 재사용 — f0_perp0 주입, anchor 묶음 제외. axis 4/5 (anchor-invariant) → GRU scalar 직결. §1.3.
exp_ids:
  - FR001_f0perp0-residual-gru
---

# plan-c-001 — F0(perp=0.0) 잔차 GRU (Kalman → F0 baseline swap, Kalman 부산물 전제거)

## §0. 한 줄 목적

> **plan-a-001 KR002 paradigm (잔차 GRU + 입력 yaw 회전, LB 신기록 0.6818) 의 baseline predictor 를 Kalman CV 필터 → F0(perp=0.0) 닫힌형 외삽으로 교체**한다. F0(perp=0.0) = `p0 + 1.98·v_last + 1.20·acc_par_vec` (plan-020 F0 에서 **PERP 계수만 -0.20 → 0.0**, 즉 perpendicular 가속 보정항 제거). GRU 는 이 F0 외삽에 대한 *잔차*를 예측 (final = f0_pred + inverse_rotate(GRU 잔차)). **plan-a-002 의 실험 프로토콜(2cfg×5fold×3seed×200ep, OOF hit_1cm, paired permutation, CV-LB 괴리 caveat)을 그대로 carry** 하되, plan-a-002 의 모든 입력 feature·frame source·W-aux 가 *Kalman 부산물*이므로 **전부 제외**한다 (§1.2 제외 보고). **대신, Kalman(적응 재귀필터)에선 못 넣었지만 F0(고정 공식)이라서 정당해지는 자기진단 feature 를 추가**한다 (§1.3): F0 의 per-step fit 잔차 `raw(t+2)−f0_perp0(t)` 계열 — L2 잔차 시퀀스(21D) + f0_conf(2D) + EWMA(3D) + STA/LTA(3D). 이는 §1.2 에서 제외한 *Kalman innovation 의 F0 버전*이자, plan-030 이 PB selector 핵심 신호로 박제한 lift lever 다. 즉 본 plan 은 "잔차 GRU 가 Kalman 대신 F0(perp=0.0) baseline + F0-자기진단 feature 로도 LB lift 를 내는가"를 묻는 baseline-swap + F0-feature 실험이다. OOF 는 *sanity*, 진짜 verdict 는 LB (사용자 confirm gated).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | **F0(perp=0.0) 잔차 회귀** (KR002 paradigm 의 baseline predictor 만 swap — Kalman 아님, anchor/selector 아님) |
| baseline predictor | **F0(perp=0.0)** = `p0 + 1.98·v_last + 1.20·acc_par_vec` (plan-020 F0 에서 PERP −0.20→**0.0**, D1=1.98·PAR=1.20 불변) |
| 비교 기준 | (1) **F0 floor** perp=−0.20 = 0.6320 / perp=0.0 = **TBD(c1 산출)** (2) **KR002** OOF 0.6663·LB 0.6818 (Kalman 잔차 GRU — paradigm 모체) (3) KR001 0.6639 |
| data | `load_all_samples` X (N,11,3), y (N,3). horizon +80ms |
| 잔차 target | `rotate_xy(y − f0_perp0_pred, θ)`. final = `f0_perp0_pred + inverse_rotate_xy(GRU_out, θ)` |
| frame θ | `yaw_angle(raw v_last)` (KR002 default — **raw 속도, Kalman filtered 아님**) + `--input-yaw` 로 입력 rel/v/a 회전 |
| 입력 feature | seq **12채널** (build_seq_t3 9 + L2 F0(perp=0)-잔차 3, slot=프레임 t·t<4 zero-pad §4.2) + scalar **48d** (build_scalar_40d 40 + f0_conf 2 + EWMA(0.3) F0-잔차 3 + STA/LTA 3) |
| **F0-자기진단 feature (신규, §1.3)** | F0 의 per-step +2-예측오차 (식 §4.2) 계열. **고정 공식이라 sample 비교가능한 "F0-법칙 위배=maneuver 강도" 신호** (Kalman 이면 innovation = §1.2 제외 부산물). 전부 anchor-invariant(axis 4/5) + 관측창 산출(leakage 무) + f0_perp0 주입(baseline 일관) |
| **제외 (Kalman-유도)** | innovation·filtered-velocity·CV/CA 불일치·gain/covariance·filtered-yaw frame·W-aux(alt-Kalman target). 전부 *재귀필터 산물* → F0(닫힌형 1-shot)에 부재. **단 innovation 의 *개념*(baseline misfit)은 F0 고정공식 버전(§1.3 F0-자기진단)으로 회수**. §1.2 보고 |
| model/loss/ensemble | GRUModelMultiAux + combo(euclid+0.3 softhit) + **aux F(naive) λ0.3, aux W λ=0(비활성)**. 2cfg(A/B)×5fold stable_fold_id×3seed×200ep. architecture(layer 구성) KR002 동일, **입력 dim 만 9→12/40→48 자동확장** |
| 실험 | **FR001_f0perp0-residual-gru** 단일 (plan-a-002 KR003/KR004 는 둘 다 Kalman 부산물 실험 → 제외 후 baseline-swap 단일 실험으로 collapse, §3) |
| metric | OOF hit_1cm (world Euclid<0.01m) uncalibrated headline + hit_1p5cm. paired permutation 10k vs KR002 & vs F0(perp=0.0) floor |
| 합격 기준 | **G_f0 (FR001)**: OOF hit_1cm 보고 + band. **≥ F0(perp=0.0) floor + 0.02 = 잔차 GRU 가 F0 위에서 작동(PASS)** / ≥ KR002 0.6663 = Kalman 동급 이상(strong) / < floor = FAIL_no-lift(정보, halt X). LB 는 §6 사용자 gated |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [TODO] |
| c1 f0 baseline | §4.1 `analysis/plan-c-001/f0_baseline.py` — `f0_perp0(X, end_idx=10)` → (N,3) [PERP=0.0 override]. + F0(perp=0.0) floor OOF (hit_1cm/1p5cm) 산출 → `f0_perp0_floor.json` | [TODO] |
| c2 f0 잔차 feat | §4.2 `analysis/plan-c-001/f0_residual_feats.py` — `f0_resid_feats(X, theta)` → L2 잔차 seq(N,11,3 yaw-rot zero-pad) + f0_conf(N,2) + EWMA(N,3) + STA/LTA(N,3). `_build_L2_L4` 로직 재사용(f0_perp0 주입), Frenet→yaw frame. **leakage assert: step t 는 raw(t+2)≤t_obs 만 사용** | [TODO] |
| c3 runner flag | §4.3 `run_oof.py` 확장 — `--baseline f0-perp0` (kalman_predict→f0_perp0 swap, 잔차 target + final 복원), `--f0-resid-feats` (seq 9→12·scalar 40→48 concat), `--aux-w-weight 0` (W aux 비활성). `--input-yaw` carry | [TODO] |
| c4 smoke | §5 `tests/test_plan_c001_smoke.py` — import + 1f1s1e finite + **target⇄복원 정합 assert** (target=`rotate_xy(y−f0_perp0,θ)`·복원=`f0_perp0+inverse_rotate_xy(·,θ)` 가 *동일 θ·동일 f0_perp0 인스턴스* 로 y 재구성 → θ source/baseline mismatch·부호버그 검출; 자명 항등 회피용으로 target 과 복원이 분리 코드경로임을 전제, atol=1e-6) + **음성통제**: θ_복원=θ_target+0.1rad 또는 baseline 부호 뒤집기 주입 시 assert 가 *반드시 fail* (자명 항등 아님 입증) + 실제 GRU_out 경유 복원 finite·(N,3) shape 체크 + F0-잔차 leakage assert + W-aux gradient 0 assert | [TODO] |
| c5 G1 | §5 FR001 1-fold 1-seed full-ep — finite & ≥ F0(perp=0.0) floor(1-fold) | [TODO] |
| c6 FR001 full | §5 2cfg×5fold×3seed OOF → `results_fr001.json/.npz` | [TODO] |
| c7 results + merge | §5 `plan-c-001-...results.md` + §0.5 sync + lane-c worktree→main merge | [TODO] |

### G-gates

- G0: c1~c4 인프라 + smoke green + round-trip assert + **F0-잔차 leakage assert** (step t 가 raw(t+2)>t_obs 미참조, zero-pad 일관) + W-aux λ=0 gradient 0 assert
- G1: FR001 1-fold 1-seed hit_1cm finite & ≥ F0(perp=0.0) floor(1-fold) − 0.005 (잔차 GRU 가 baseline 위에서 학습 안정 sanity)
- G_f0 (G2): FR001 full OOF band 판정 (vs F0(perp=0.0) floor + vs KR002 0.6663, paired permutation)
- G_final: FR001 results 박제 + §0.5 sync + main merge

### Plan-specific 주의 (CV-LB 괴리 carry)

- plan-a-001 에서 입력 yaw 회전은 OOF neutral(+0.0024 ns)인데 LB +0.0060 였다. → **OOF Δ<threshold 라도 FAIL 아님**; G_f0 은 *F0-floor 대비 lift* 만 hard 요구. **OOF 만으로 paradigm 폐기 금지** (CV-LB 괴리 박제).
- 본 plan = baseline swap(Kalman→F0-perp0) + **저차원 F0-자기진단 feature**(seq+3·scalar+8, §1.3) — model 구조·loss·frame 은 KR002 동일. Kalman 보다 **F0 가 산술적으로 단순·고속** (행렬 필터링 없음) + "닫힌형 baseline + 그 baseline 의 per-step 잔차 자기진단 + GRU 보정"의 일반화 검증 의의. F0-feature 는 전부 저차원 물리기반 → CV-LB 괴리 환경 overfit 위험 최소.

---

## §1. 배경

### §1.1 paradigm 모체 (plan-a-001 KR002) + baseline 후보 (F0)

plan-a-001 결과:
- **KR002 (Kalman CV 잔차 GRU + 입력 yaw 회전)**: OOF 0.6663, **LB 0.6818 (프로젝트 신기록)**. paradigm = "baseline predictor 의 +80ms 외삽에 대해 GRU 가 yaw-frame 잔차를 학습 → final = baseline + 잔차".
- **CV-LB 괴리**: 입력 yaw 회전이 OOF 통계적 0 인데 LB 명확한 양 lift.

plan-020 **F0** = 닫힌형 운동학 외삽:
```
f0_pred = p0 + D1·v_last + PAR·acc_par_vec + PERP·acc_perp_vec
        = p0 + 1.98·v_last + 1.20·acc_par_vec − 0.20·acc_perp_vec   (plan-020 default)
```
- OOF hit_1cm 0.6320 (floor), hit_1p5cm 0.8033. 재귀필터(Kalman)와 달리 **per-sample 1-shot 닫힌형** — 행렬·gain·covariance·state 없음.

**본 plan 의 변경 = PERP 계수만 −0.20 → 0.0**:
```
f0_perp0_pred = p0 + 1.98·v_last + 1.20·acc_par_vec   (perpendicular 가속 보정항 제거)
```
D1=1.98, PAR=1.20 불변. 그 외 plan-020 F0 와 동일. F0(perp=0.0) floor 점수는 c1 에서 산출(plan-020 0.6320 와 소폭 차이 예상 — perp 항이 floor 에 주는 기여만큼).

**본 plan = baseline predictor swap (+ F0-자기진단 feature)**: KR002 의 `kalman_predict(X,'CV')` 를 `f0_perp0(X)` 로 교체 + §1.3 F0-잔차 feature 추가(seq+3·scalar+8). 잔차 target·final 복원·yaw frame·model 구조·loss·ensemble·OOF 프로토콜은 **KR002 carry** (입력 dim 만 확장). 즉 "잔차 GRU 가 Kalman CV 대신 F0(perp=0.0) baseline + 그 F0 의 per-step 잔차 자기진단 위에서도 lift 를 내는가"를 측정.

### §1.2 제외 — Kalman 부산물 전체 (plan-a-002 가 더하던 것을 본 plan 은 전부 뺀다)

plan-a-002 는 KR002 위에 Kalman 필터가 *산출하고 버리는* 부산물을 입력 feature 로 회수하는 plan 이었다. F0(perp=0.0)는 **닫힌형 1-shot 외삽 (재귀 state·필터링·모델쌍 없음)** 이라 그 부산물들이 **물리적으로 존재하지 않으므로** 전부 제외한다. 제외 목록 + 사유:

| 제외 항목 | plan-a-002 에서의 역할 | F0(perp=0.0)에 부재한 이유 |
|---|---|---|
| **(a) innovation** `z(t)−state_pred(t)` (seq +3채널) | per-step "등속예측 대비 측정 surprise" = maneuver 표지 | F0 는 마지막 프레임에서 +80ms 한 번 외삽 — *per-step 재귀 state 예측이 없음* → innovation 시퀀스 정의 불가 |
| **(b) filtered velocity** `state[:,1]` (seq +3채널) | Kalman denoise 속도 | F0 는 raw 유한차분 v_last 사용 — *재귀 필터링 state 없음* → filtered velocity 부재 |
| **(c) CV/CA 불일치** `kalman('CA')−kalman('CV')` (scalar +4) | 등속·등가속 두 *필터* 외삽 차 = 가속 모델링 중요도 | F0 는 단일 닫힌형 (CV/CA 모델쌍 없음). 가속은 acc_par 항으로 이미 baseline 에 흡수 — *disagreeing 모델쌍 부재* |
| **gain K / covariance P** | (plan-a-002 도 이미 제외: 선형 TI 필터 측정독립 = 정보 0) | F0 는 gain/covariance 개념 자체가 **없음** (vacuously 제외) |
| **filtered-yaw frame (KR004)** `θ=yaw_angle(filtered v_last)` | frame heading 안정화 lever | filtered velocity 부재 → frame 은 KR002 default `yaw_angle(raw v_last)` 유지. KR004 실험 제외 |
| **W-aux target** `y − kalman(σ=1.0mm)` | multi-aux 보조 supervision (alt-σ Kalman 잔차) | alt-Kalman 산물 → Kalman-유도. **W-aux head 는 architecture 보존 위해 유지하되 λ_W=0 (gradient 0)** 로 비활성 (model byte-동일 carry) |

**결과**: plan-a-002 가 더하던 입력 feature(seq+6채널, scalar+4)·frame source 변형·W-aux supervision 이 *전부* Kalman-유도라 제외. plan-a-002 의 두 실험 KR003(feature add)·KR004(frame)은 둘 다 Kalman 부산물 실험이므로 제외 후 **baseline-swap 단일 실험(FR001)로 collapse**. **단 제외된 innovation 의 *개념*(baseline 의 per-step misfit)은 F0(고정 공식)에선 정당한 입력 feature 로 회수 가능 → §1.3 에서 추가.**

### §1.3 추가 — F0 고정공식 자기진단 feature (Kalman innovation 의 F0 버전)

§1.2 에서 제외한 **innovation** (`z(t)−state_pred(t)`, Kalman 의 per-step "측정 surprise") 의 *개념* 은 "baseline 이 각 관측 step 을 얼마나 못 맞추나" 다. 이건 F0 에서도 계산 가능하고, **오히려 F0(고정 공식)에서 더 정당하다**:

- **Kalman 이면 못 넣는 이유**: Kalman 은 *적응 재귀필터* — state 가 매 샘플 관측에 맞춰 tracking → per-step 잔차(=innovation)는 (i) §1.2 에서 제외한 *필터 내부 부산물*이고, (ii) σ_obs/σ_proc 튜닝에 따라 값이 흔들리는 *필터 artifact* (순수 sample 물리 아님).
- **F0(고정 공식)이라 넣을 수 있는 이유**: F0 는 *per-sample 적응이 없는 보편 법칙*. per-step F0 +2-예측오차 (**authoritative 식·부호·valid step = §4.2**) 는 **"이 샘플이 F0 운동법칙을 시점별로 얼마나 위배하나" = sample 간 비교가능한 maneuver/non-ballistic 강도** 신호. 관측창에서만 산출(leakage 무).

**증거 (plan-030 §1.2)**: step 별 raw F0 잔차 (L2 21D = 7 step × 3축) 가 **PB selector 의 핵심 lift 신호**였으나 plan-024/29 anchor paradigm 으로 carry 안 됨 (`plans/plan-030.md:69`, expected lift +0.005~0.01). 본 plan 은 anchor-free F0-잔차 GRU 라 이 신호를 *직접* 입력으로 회수.

추가 feature (전부 `f0_perp0` 주입 = baseline 일관, anchor-invariant axis 4/5, KR002 yaw frame `rotate_xy(θ=yaw_angle(raw v_last))` 회전, 무관측 step zero-pad):

| feature | dim | 무엇 | source 로직 |
|---|---|---|---|
| **L2 F0-잔차 seq** | 7step×3 → (N,11,3) zero-pad(slot=프레임 t, §4.2), **GRU seq +3채널** | step별 F0 +2-예측오차 (식 §4.2) yaw 회전 3축 | `plan-021/_build_L2_L4` (f0_perp0 주입, Frenet→yaw) |
| **f0_conf** | scalar **+2** | f0_perp0 마지막 잔차 norm + step-speed std | `plan-024/cand_builder` A8 |
| **EWMA(α=0.3) F0-잔차** | scalar **+3** | 평활 F0 잔차 (yaw 3축) | cand_builder ctx base |
| **A1 STA/LTA** | scalar **+3** | EWMA(0.5)/EWMA(0.1) F0-잔차 비 = maneuver onset | cand_builder ctx A1 |

→ 입력: seq **9→12채널**, scalar **40→48d**. **L4 soft-hit(7step×2=14)는 제외** (L2 와 동일 잔차의 σ 변환 = redundant, 사용자 결정).

(여전히 out-of-scope: perp on/off disagreement(perp=0.0 vs −0.20), acc_perp 자체 feature 화 — F0-native 부산물의 *추가* 회수는 후속. 본 plan 은 innovation-개념의 F0 회수 + baseline swap 까지.)

## §2. 가설

- **H1 (baseline swap 작동)**: F0(perp=0.0) 외삽은 Kalman CV 와 동급의 "물리적으로 합당한 +80ms 초기추정" → GRU 잔차가 그 위에서 F0 floor 대비 명확히 lift. FR001 OOF hit_1cm ≥ F0(perp=0.0) floor + 0.02.
- **H2 (Kalman 동급 가능)**: 잔차 GRU 의 lift 대부분이 *baseline 품질*보다 *GRU 의 잔차 학습*에서 온다면, F0(perp=0.0) baseline 도 KR002(0.6663) 에 근접/동급 가능. (Kalman 의 denoise 우위가 GRU 잔차로 상당부분 흡수된다는 가설.)
- **H3 (CV-LB 괴리 재현 가능)**: OOF 에서 KR002 대비 neutral/소폭 음 이어도, F0 의 단순성이 test 분포에서 다르게 작동할 수 있음 (입력 yaw 처럼 OOF-neutral·LB-양 가능) → LB 사용자 confirm 후 검증.
- **메타 (Kalman 부산물 제외 정당성)**: innovation/filtered-v/CV-CA 를 빼는 것은 휴리스틱이 아니라 F0 가 *재귀필터가 아닌 닫힌형*이라는 구조적 사실에서 따라오는 *필연적* 제외 (§1.2).

## §3. 실험 목록

### FR001_f0perp0-residual-gru
- **type**: baseline predictor swap (Kalman CV → F0(perp=0.0)) + F0-자기진단 feature 추가 (§1.3)
- **baseline (비교군)**: F0(perp=0.0) floor (1차) + KR002 (Kalman 잔차 GRU, OOF 0.6663/LB 0.6818, 2차)
- **변경 변수**: (1) 잔차 GRU 의 baseline predictor `kalman_predict(X,'CV')` → `f0_perp0(X)`. 잔차 target = `rotate_xy(y − f0_perp0, θ)`, final = `f0_perp0 + inverse_rotate_xy(GRU_out, θ)`. (2) **F0-자기진단 feature 추가**(§1.3): seq +3채널(L2 F0-잔차) → 9→12, scalar +8(f0_conf 2+EWMA 3+STA/LTA 3) → 40→48. frame θ(raw v_last yaw)·model 구조·loss·ensemble·calibration 은 KR002 동일(입력 dim 만 자동 확장). W-aux λ=0(§1.2). **단일변수 아님**(swap+feature) → ablation 은 §3 attribution 으로 분리.
- **config/경로**: `run_oof.py --baseline f0-perp0 --f0-resid-feats --input-yaw --aux-w-weight 0`
- **기대 runtime**: KR002 ≈ 700s (GPU L40S). F0 는 Kalman 보다 산술 단순(행렬 필터 없음) → baseline 산출 더 빠름. CPU 시 seed 3→1 자동감소(decision-note carry).
- **성공 기준**: OOF hit_1cm 보고 + band. ≥ F0(perp=0.0) floor + 0.02 = 잔차 GRU 작동(PASS) / ≥ KR002 0.6663 = Kalman 동급 이상(strong) / < floor = FAIL_no-lift(정보). finite, NaN/Inf 0, round-trip assert green.
- **실패 분기**: < F0(perp=0.0) floor → 잔차 GRU 가 F0 baseline 을 오히려 깎음 (informative). KR002(Kalman) 가 lift 의 필요조건이었다는 결론 → 박제. severe 아님(정보).
- **attribution (FR001 결과 보고 시 필수)**: FR001 이 2변경(swap+feature)이라, FR001 full 직후 `--f0-resid-feats` on/off 1-fold ablation **필수 수행** 으로 (i) baseline-swap 효과 vs (ii) F0-자기진단 feature 효과 분리 → results 에 박제 (G_f0 가 2변수 합산이라 gate 혼입 해소). 추가로 floor 차(Kalman vs F0 perp0)도 분해. **CV-LB 괴리로 OOF attribution 은 약한 신호** 명시. **단일변수 full 분리(2 OOF 실험: swap-only / swap+feature)는 의도적 scope-out** — budget 2배 + CV-LB 괴리로 OOF attribution 자체가 약신호 + 사용자 의도='F0 단일 plan'. 따라서 G_f0 는 2변수 합산 PASS/FAIL 로 두고 1-fold ablation 으로 *방향 귀속만* 박제 (full 분리는 후속).

(plan-a-002 의 KR003·KR004 는 §1.2 대로 둘 다 Kalman 부산물 실험 → 제외 후 본 plan 에 존재하지 않음. 실험 프로토콜·budget·metric·gate 는 plan-a-002 와 동일하게 carry.)

## §4. 서버 작업 순서 (모듈 spec)

### §4.1 f0_baseline.py (c1)
- `f0_perp0(X, end_idx=10)` → `(N,3)`: F0 닫힌형 (PERP=0.0). **self-contained 식** (X (N,T,3), e=end_idx, ε=1e-12):
  ```
  p0      = X[:, e]                       # (N,3) e 프레임 위치
  v_last  = X[:, e]   − X[:, e-1]         # (N,3) 마지막 1-step 변위 (단위 = 위치Δ / 40ms 프레임)
  v_prev  = X[:, e-1] − X[:, e-2]         # (N,3)
  acc     = v_last − v_prev               # (N,3) 유한차분 가속 (변위Δ/프레임²)
  v_hat   = v_last / (‖v_last‖₂ + ε)      # (N,3) 단위 속도방향 (축=좌표 3, norm over xyz)
  a_par   = (acc · v_hat) · v_hat         # (N,3) 가속의 속도방향 평행성분 벡터 (= §0/§0.5 의 acc_par_vec, dot over xyz)
  # a_perp = acc − a_par                  # 본 plan 미사용 (PERP=0.0)
  f0_perp0 = p0 + 1.98·v_last + 1.20·a_par + 0.0·(acc − a_par)
           = p0 + 1.98·v_last + 1.20·a_par           # (N,3) world +80ms 예측
  ```
  - **단위/horizon 정합**: v_last 는 프레임당 변위(40ms step), D1=1.98 ≈ 2-프레임(80ms) horizon 의 *학습된* 무차원 계수(별도 dt 곱 없음). PAR=1.20 동일 무차원. horizon=2 step 은 계수에 흡수 (plan-020 fit 그대로 carry).
  - **sub-window 호출**: `f0_perp0(X[:, t-4:t-1], end_idx=2)` → 3-프레임 윈도(인덱스 t-4,t-3,t-2)에서 e=2 기준 위 식 적용 = 그 윈도 마지막(t-2)에서 +2 step(=t) 예측. (§4.2 L2 잔차용.) 최소 경계 t=4 → `X[:,0:3]`={0,1,2} (음수 인덱스 없음); `t<4` 는 호출 안 하고 §4.2 에서 zero-pad.
  - plan-020 `baseline_f0.py` 상수(PERP=−0.20)는 **수정 X** (plan-020 repro 보존). 본 모듈은 별도, 위 식의 PERP 만 0.0.
- F0(perp=0.0) floor: `f0_perp0` 예측으로 5-fold OOF hit_1cm/hit_1p5cm 산출 → `analysis/plan-c-001/f0_perp0_floor.json` 박제 (FR001 비교 기준 §0.5).
- **torch 경로 불요**: GRU 는 잔차만 학습, f0_perp0 는 비학습 numpy baseline (Kalman 과 동일 취급).

### §4.2 f0_residual_feats.py (c2)
- `f0_resid_feats(X, theta)` → dict: `{seq_resid (N,11,3), f0_conf (N,2), ewma (N,3), sta_lta (N,3)}`, **전 출력 dtype float32**. 전부 `f0_perp0` 주입 (baseline 일관, §1.3). theta = `yaw_angle(X[:,10]−X[:,9])` (= raw v_last, **sample-level 단일 θ (N,)** — 전 7 step `r_f(t)` 에 동일 θ 적용, step-가변 아님; KR002 frame carry), `rotate_xy` 가 샘플별 broadcast, ε=1e-6.
- **per-frame F0 잔차 (단일 정의, §1.3/§0.5 의 `raw(t+2)−f0(t)` 의 authoritative 식)** — valid frame `t ∈ {4,5,…,10}` (7개):
  ```
  r_w(t) = f0_perp0(X[:, t-4:t-1], end_idx=2) − X[:, t]    # (N,3) world, pred − actual (_build_L2_L4 부호)
  r_f(t) = rotate_xy(r_w(t), theta)                          # (N,3) KR002 yaw frame
  ```
  - `_build_L2_L4` 로직 재사용하되 (a) `f0_baseline_fn=f0_perp0`, (b) L4 미산출, (c) Frenet R_wfn → `rotate_xy(·,theta)` 교체.
  - **단일 정의·frame 일관**: 잔차의 단일 정의 = `r_w(t)`(pred−actual, world) — `raw(t+2)−f0(t)` 류 reindex 표기 미사용. **seq_resid·EWMA·STA-LTA 의 입력은 모두 yaw 회전된 `r_f(t)=rotate_xy(r_w(t),θ)`** 로 일관; f0_conf 의 norm 성분(‖r_w‖)만 frame-invariant 라 r_w 직접 사용. 부호는 학습 feature 라 무관하나 전 채널 `r_w` 부호(pred−actual)로 고정 (직관 "raw 가 F0 를 얼마나 벗어났나" = `−r_w`).
- **seq_resid (N,11,3)** — step-축 정렬: `seq_resid[:, t] = r_f(t)` for `t∈{4..10}`; `seq_resid[:, t] = 0` for `t∈{0,1,2,3}` (윈도 부족). **slot index = 프레임 t** → `build_seq_t3` (N,11,9) 의 frame-t row 와 동일 step 축 → concat (9→12채널) 정렬 보장.
- **f0_conf (N,2)** — `[‖r_w(10)‖₂, std_t(speed(t))]` (cand_builder A8): (i) 마지막 프레임 잔차 norm(xyz축), (ii) step-speed `speed(t)=‖X[:,t]−X[:,t-1]‖₂` (world raw 위치차, t=1..10) 의 std (**ddof=0**). 출력 dtype float32. (§0.5 "spread"=이 std 로 용어 통일. speed std 의 t=1..10 도메인은 잔차 valid-frame t=4..10 과 무관 — raw 위치차라 윈도 제약 없음, 의도된 차이.)
- **ewma (N,3)** — `r_f(t)` (t=4..10, 좌표 3축별) 의 EWMA(α=0.3): `s ← 0.3·r_f(t) + 0.7·s`, init `s=r_f(4)` (첫 valid). 출력 = 마지막 s (N,3).
- **sta_lta (N,3)** — `EWMA_{α=0.5}(r_f) / (|EWMA_{α=0.1}(r_f)| + ε)` per 축 (분모 0 방지 ε). 두 EWMA 모두 init=r_f(4).
- **leakage 안전**: r_w(t) 는 프레임 {t-4,t-3,t-2}(≤t-2) + actual X[:,t](t≤10) 만 사용 — 전부 관측 도메인 [0,10] 내. "+2 예측"은 외삽이지 미래 프레임 참조 아님. **smoke assert**: `seq_resid[:, :4]==0` & `t≥4 nonzero` & 어떤 산출도 X[:, >10] 미참조.
- backward-compat: plan-021/024 원본 **수정 X** (해당 plan repro 보존). 본 모듈은 별도 박제(f0_perp0 주입판).

### §4.3 run_oof.py 확장 (c3)
- 신규 flag `--baseline {kalman-cv, f0-perp0}` (default kalman-cv = KR002 호환). `f0-perp0` 시 baseline predictor = `f0_perp0(X)`, 그 외(kalman-cv) 시 기존 `kalman_predict`.
  - `baseline_pred = f0_perp0(X) if args.baseline=='f0-perp0' else kalman_predict(...)` 1곳 산출 후 **두 지점에 동일 주입**: (i) 잔차 target `rotate_xy(y−baseline_pred, θ)`, (ii) final 복원 `baseline_pred + inverse_rotate_xy(out, θ)`. 두 지점 모두 *같은* `baseline_pred`·`θ` 사용 (둘 중 한 곳만 swap 되는 누락은 c4 정합 assert 가 검출).
- 신규 flag `--f0-resid-feats`: `f0_resid_feats(X, theta)` 산출 → seq 에 seq_resid concat (9→12채널), scalar 에 f0_conf+ewma+sta_lta concat (40→48d). off 시 KR002 동일 입력.
- `--aux-w-weight FLOAT` (default 0.3 = KR002). FR001 은 **0** (W-aux 비활성, §1.2). F-aux λ 는 0.3 carry.
- `--input-yaw` carry (θ=yaw_angle(raw v_last)). `--filtered-yaw` 는 **미구현/금지** (Kalman filtered v 없음, §1.2).
- model n_channels(9→12)/scal_dim(40→48) 은 입력 dim 에서 자동 추론. GRUModelMultiAux **architecture(layer 구성) 불변**, 입력 projection dim 만 확장.

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke (1f1s1e finite) + **target⇄복원 정합 assert** + W-aux gradient 0 assert (λ_W=0) | green | severe halt if import/NaN/정합 실패 |
| G1 | FR001 1-fold 1-seed full-ep hit_1cm | ≥ F0(perp=0.0) floor(1-fold) − 0.005 (잔차 학습 안정 sanity) | warn |
| G_f0 | FR001 full OOF hit_1cm | **≥ floor+0.02 PASS** / ≥ KR002 0.6663 strong / < floor FAIL_no-lift(정보) | < floor = 정보, halt X |
| G_final | FR001 results 박제 + §0.5 sync + main merge | 완료 | — |

- **G_f0 PASS bar +0.02 근거**: KR001(Kalman 잔차 GRU)이 F0 floor 0.6320 → 0.6639 = **+0.0319** lift 였음(plan-a-001). F0 는 Kalman 보다 약한 baseline 이라 동일 잔차 GRU 의 lift 가 그보다 작을 수 있으므로, KR001 lift 의 보수적 ~2/3 수준 **+0.02** 를 "잔차 GRU 가 F0 위에서 *의미있게* 작동" 최소 margin 으로 설정 (그 미만이면 잔차 학습이 floor 를 거의 못 넘김).
- **band 산출 정의**: G_f0 의 "band" = 5-fold OOF hit_1cm 의 **fold 간 mean ± std** (3-seed 평균 후 fold spread). PASS/FAIL 분기는 *fold-평균 hit_1cm* 의 hard bar (floor+0.02 / KR002 0.6663) 로 결정 — band 는 안정성 보고용 (분기 기준 아님). **집계 단위 일치**: floor+0.02 비교 양변(F0(perp=0) floor, FR001 hit_1cm)·KR002 0.6663 전부 *fold-평균 hit_1cm* 동일 집계 (pooled 아님).
- statistic: paired permutation 10000 resample (FR001 vs F0(perp=0.0) floor sample-wise hit, FR001 vs KR002 OOF), p<0.05.
- artifact: `analysis/plan-c-001/f0_perp0_floor.json`, `results_fr001.json/.npz`, `plan-c-001-...results.md`.
- NaN/Inf/divergence 0 의무. cuda OOM 시 batch 256→128→64 자동 감소.
- **CV-LB 괴리 박제 의무**: results 에 OOF Δ(vs KR002, vs floor) 와 함께 "LB 검증 필요(사용자 gated)" 명시. OOF neutral 을 negative 결론으로 박제 금지.
- **제외/추가 보고 의무**: results 에 §1.2 제외 표(Kalman 부산물 6종) + §1.3 추가 표(F0-자기진단 4종) 재게시 — "Kalman→F0 swap 이 무엇을 버리고(innovation 등) 무엇을 F0 버전으로 회수했나(F0-잔차)" 사후 audit.

## §6. Out of scope

- **Kalman 부산물 feature 일체** (innovation·filtered-v·CV/CA·gain/covariance·filtered-yaw·W-aux target) — §1.2 대로 F0 닫힌형에 부재 → 제외. 제외 자체가 본 plan 의 보고 산출.
- **F0-자기진단 추가본** = §1.3 으로 *included* (per-step F0 잔차 L2 + f0_conf + EWMA + STA/LTA). **남은 out-of-scope**: L4 soft-hit(redundant), perp on/off disagreement(perp=0.0 vs −0.20), acc_perp 자체 feature 화 — F0-native 부산물의 *추가* 회수는 후속.
- **PERP 외 F0 계수 재튜닝** (D1·PAR sweep, perp 를 0 외 값으로) — 사용자 지시 = PERP 0.0 단일 변경. D1=1.98·PAR=1.20 carry.
- baseline 양다리 ensemble (Kalman+F0) — 단일 baseline swap 실험.
- anchor/selector paradigm.
- **autonomous DACON LB 제출** — quota 사용자 명시 confirm 필요 (특히 OOF Δ<threshold 시 더더욱). 본 plan headline = OOF. **단 CV-LB 괴리 때문에 FR001 의 LB 제출을 *사용자 confirm 후* 권장** (입력 yaw 처럼 OOF-neutral·LB-positive 가능성). 제출 = 별 turn, 사용자 승인 gated.

## §7. 참조

- `plans/plan-a-002-kalman-derived-features.md` — 본 plan 의 구조·프로토콜 모체 (Kalman 부산물 plan, 본 plan 이 그 부산물을 전부 제외).
- `plans/plan-a-001-kalman-residual-gru-repro.results.md` — KR001/KR002 OOF·LB, CV-LB 괴리, 잔차 GRU paradigm.
- `analysis/plan-020/baseline_f0.py` — F0 공식 본체 (D1/PAR/PERP, f0_baseline, hit metric). 본 plan 은 PERP 만 0.0.
- `analysis/plan-a-001/{yaw,features,model,losses,run_oof}.py` — KR002 파이프라인 (baseline predictor 자리만 swap).
- `plans/plan-030.md:69` — **F0-잔차 PB selector 핵심 신호 증거** (L2 21D, plan-024/29 carry 누락, expected lift +0.005~0.01). §1.3 근거.
- `analysis/plan-021/build_input.py:_build_L2_L4` — F0 per-step 잔차(L2)+soft-hit(L4) 산출 (f0_baseline_fn 주입형). 본 plan 은 f0_perp0 주입 + L2 만.
- `analysis/plan-024/cand_builder.py` — f0_conf(A8)·EWMA·STA/LTA F0-잔차 sample-level 로직.
- `notes/fe_axis_24_25_26_27_29.md` — FE 5-axis 분류 (F0-자기진단 = axis 4/5 anchor-invariant 확인).
- `WORKFLOW.md §4` — lane mutex + worktree→main merge (lane c 1번째 plan).

decision-note: spec-default — plan-c-001 = lane c 1번째 plan. plan-a-002 구조/프로토콜 carry + baseline predictor 만 Kalman CV→F0(perp=0.0)(사용자 지시: PERP −0.20→0.0 단일 계수 변경, D1/PAR carry). plan-a-002 의 모든 입력 feature·frame·W-aux 가 Kalman 부산물(재귀필터 산물)이라 F0(닫힌형)에 부재 → 전부 제외(§1.2 6종 보고). W-aux head 는 architecture 보존 위해 유지하되 λ_W=0 비활성(Kalman target 회피, model byte-동일). exp prefix FR(F0-Residual) 신규. LB 제출 = 사용자 confirm gated.
decision-note (v2, 사용자 지시): Kalman 이면 못 넣지만 F0(고정 공식)이면 넣을 수 있는 자기진단 feature 회수(§1.3) — Kalman innovation 의 F0 버전 = per-step `raw(t+2)−f0_perp0(t)` 잔차. 사용자 "Full 잔차 suite" 선택 → L2 잔차 seq(+3채널)+f0_conf(+2)+EWMA(+3)+STA/LTA(+3), seq 9→12·scalar 40→48. L4 soft-hit 는 redundant 로 제외(사용자 결정). 근거 = plan-030:69 (L2 가 PB selector 핵심 신호). f0_perp0 주입(baseline 일관), KR002 yaw frame 회전, anchor-free(axis 4/5), 관측창 leakage 무. FR001 = swap+feature 2변경 → on/off ablation 으로 attribution 분리.
