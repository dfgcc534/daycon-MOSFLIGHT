---
plan_id: c-001
version: 1
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
    reason: 5-fold OOF runner. baseline predictor 를 `--baseline f0-perp0` flag 로 swap (kalman_predict → f0_perp0). `--input-yaw` carry.
exp_ids:
  - FR001_f0perp0-residual-gru
---

# plan-c-001 — F0(perp=0.0) 잔차 GRU (Kalman → F0 baseline swap, Kalman 부산물 전제거)

## §0. 한 줄 목적

> **plan-a-001 KR002 paradigm (잔차 GRU + 입력 yaw 회전, LB 신기록 0.6818) 의 baseline predictor 를 Kalman CV 필터 → F0(perp=0.0) 닫힌형 외삽으로 교체**한다. F0(perp=0.0) = `p0 + 1.98·v_last + 1.20·acc_par_vec` (plan-020 F0 에서 **PERP 계수만 -0.20 → 0.0**, 즉 perpendicular 가속 보정항 제거). GRU 는 이 F0 외삽에 대한 *잔차*를 예측 (final = f0_pred + inverse_rotate(GRU 잔차)). **plan-a-002 의 실험 프로토콜(2cfg×5fold×3seed×200ep, OOF hit_1cm, paired permutation, CV-LB 괴리 caveat)을 그대로 carry** 하되, plan-a-002 의 모든 입력 feature·frame source·W-aux 가 *Kalman 부산물*이므로 **전부 제외**한다 (§1.2 제외 보고). 즉 본 plan 은 "잔차 GRU 가 Kalman 대신 F0(perp=0.0) 라는 더 싸고 단순한 baseline 위에서도 LB lift 를 내는가"를 묻는 baseline-swap 실험이다. OOF 는 *sanity*, 진짜 verdict 는 LB (사용자 confirm gated).

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
| 입력 feature | seq **9채널** (build_seq_t3) + scalar **40d** (build_scalar_40d) — **KR002 그대로, 신규 채널 0** |
| **제외 (Kalman-유도)** | innovation·filtered-velocity·CV/CA 불일치·gain/covariance·filtered-yaw frame·W-aux(alt-Kalman target). 전부 *재귀필터 산물* → F0(닫힌형 1-shot)에 부재. §1.2 보고 |
| model/loss/ensemble | GRUModelMultiAux + combo(euclid+0.3 softhit) + **aux F(naive) λ0.3, aux W λ=0(비활성)**. 2cfg(A/B)×5fold stable_fold_id×3seed×200ep. architecture KR002 byte-동일 |
| 실험 | **FR001_f0perp0-residual-gru** 단일 (plan-a-002 KR003/KR004 는 둘 다 Kalman 부산물 실험 → 제외 후 baseline-swap 단일 실험으로 collapse, §3) |
| metric | OOF hit_1cm (world Euclid<0.01m) uncalibrated headline + hit_1p5cm. paired permutation 10k vs KR002 & vs F0(perp=0.0) floor |
| 합격 기준 | **G_f0 (FR001)**: OOF hit_1cm 보고 + band. **≥ F0(perp=0.0) floor + 0.02 = 잔차 GRU 가 F0 위에서 작동(PASS)** / ≥ KR002 0.6663 = Kalman 동급 이상(strong) / < floor = FAIL_no-lift(정보, halt X). LB 는 §6 사용자 gated |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) | [TODO] |
| c1 f0 baseline | §4.1 `analysis/plan-c-001/f0_baseline.py` — `f0_perp0(X, end_idx=10)` → (N,3) [PERP=0.0 override]. + F0(perp=0.0) floor OOF (hit_1cm/1p5cm) 산출 → `f0_perp0_floor.json` | [TODO] |
| c2 runner flag | §4.2 `run_oof.py` 확장 — `--baseline f0-perp0` (kalman_predict→f0_perp0 swap, 잔차 target + final 복원), `--aux-w-weight 0` (W aux 비활성). `--input-yaw` carry | [TODO] |
| c3 smoke | §5 `tests/test_plan_c001_smoke.py` — import + 1f1s1e finite + 잔차/복원 round-trip assert (f0_pred + inverse_rotate(rotate(y−f0)) == y) + W-aux gradient 0 assert | [TODO] |
| c4 G1 | §5 FR001 1-fold 1-seed full-ep — finite & ≥ F0(perp=0.0) floor(1-fold) | [TODO] |
| c5 FR001 full | §5 2cfg×5fold×3seed OOF → `results_fr001.json/.npz` | [TODO] |
| c6 results + merge | §5 `plan-c-001-...results.md` + §0.5 sync + lane-c worktree→main merge | [TODO] |

### G-gates

- G0: c1~c3 인프라 + smoke green + round-trip assert + W-aux λ=0 gradient 0 assert
- G1: FR001 1-fold 1-seed hit_1cm finite & ≥ F0(perp=0.0) floor(1-fold) − 0.005 (잔차 GRU 가 baseline 위에서 학습 안정 sanity)
- G_f0 (G2): FR001 full OOF band 판정 (vs F0(perp=0.0) floor + vs KR002 0.6663, paired permutation)
- G_final: FR001 results 박제 + §0.5 sync + main merge

### Plan-specific 주의 (CV-LB 괴리 carry)

- plan-a-001 에서 입력 yaw 회전은 OOF neutral(+0.0024 ns)인데 LB +0.0060 였다. → **OOF Δ<threshold 라도 FAIL 아님**; G_f0 은 *F0-floor 대비 lift* 만 hard 요구. **OOF 만으로 paradigm 폐기 금지** (CV-LB 괴리 박제).
- 본 plan 은 baseline predictor 만 바뀐 *저위험 swap* (입력 feature dim·model·loss·frame 전부 KR002 동일) — Kalman 보다 **F0 가 산술적으로 단순·고속** (행렬 필터링 없음) 이라는 운영 이점 + "닫힌형 baseline + GRU 잔차"의 일반화 검증 의의.

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

**본 plan = baseline predictor swap**: KR002 의 `kalman_predict(X,'CV')` 를 `f0_perp0(X)` 로 교체. 잔차 target·final 복원·yaw frame·입력 feature·model·loss·ensemble·OOF 프로토콜은 **전부 KR002 carry**. 즉 "잔차 GRU 가 Kalman CV 대신 F0(perp=0.0) 라는 더 단순한 baseline 위에서도 lift 를 내는가"를 단일변수(baseline predictor)로 측정.

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

**결과**: plan-a-002 가 더하던 입력 feature(seq+6채널, scalar+4)·frame source 변형·W-aux supervision 이 *전부* Kalman-유도라 제외 → 본 plan 의 입력은 KR002 와 **완전 동일**(seq 9채널 + scalar 40d, raw-yaw frame), 단 baseline predictor 만 F0(perp=0.0). plan-a-002 의 두 실험 KR003(feature add)·KR004(frame)은 둘 다 Kalman 부산물 실험이므로 제외 후 **baseline-swap 단일 실험(FR001)로 collapse**.

(향후 후속 — out-of-scope: F0-native 부산물 회수 = per-step F0 형식 잔차, perp 항 disagreement(perp=0.0 vs −0.20), acc_perp 자체를 feature 로. 본 plan 은 *Kalman 제외*만 다루고 F0 부산물 추가는 하지 않음.)

## §2. 가설

- **H1 (baseline swap 작동)**: F0(perp=0.0) 외삽은 Kalman CV 와 동급의 "물리적으로 합당한 +80ms 초기추정" → GRU 잔차가 그 위에서 F0 floor 대비 명확히 lift. FR001 OOF hit_1cm ≥ F0(perp=0.0) floor + 0.02.
- **H2 (Kalman 동급 가능)**: 잔차 GRU 의 lift 대부분이 *baseline 품질*보다 *GRU 의 잔차 학습*에서 온다면, F0(perp=0.0) baseline 도 KR002(0.6663) 에 근접/동급 가능. (Kalman 의 denoise 우위가 GRU 잔차로 상당부분 흡수된다는 가설.)
- **H3 (CV-LB 괴리 재현 가능)**: OOF 에서 KR002 대비 neutral/소폭 음 이어도, F0 의 단순성이 test 분포에서 다르게 작동할 수 있음 (입력 yaw 처럼 OOF-neutral·LB-양 가능) → LB 사용자 confirm 후 검증.
- **메타 (Kalman 부산물 제외 정당성)**: innovation/filtered-v/CV-CA 를 빼는 것은 휴리스틱이 아니라 F0 가 *재귀필터가 아닌 닫힌형*이라는 구조적 사실에서 따라오는 *필연적* 제외 (§1.2).

## §3. 실험 목록

### FR001_f0perp0-residual-gru
- **type**: baseline predictor swap (단일변수: Kalman CV → F0(perp=0.0))
- **baseline (비교군)**: KR002 (Kalman 잔차 GRU, OOF 0.6663/LB 0.6818) + F0(perp=0.0) floor
- **변경 변수**: 잔차 GRU 의 baseline predictor 만 `kalman_predict(X,'CV')` → `f0_perp0(X)`. 잔차 target = `rotate_xy(y − f0_perp0, θ)`, final = `f0_perp0 + inverse_rotate_xy(GRU_out, θ)`. **입력 feature(seq 9채널+scalar 40d)·frame θ(raw v_last yaw)·model·loss·ensemble·calibration 전부 KR002 동일**. W-aux λ=0(§1.2).
- **config/경로**: `run_oof.py --baseline f0-perp0 --input-yaw --aux-w-weight 0`
- **기대 runtime**: KR002 ≈ 700s (GPU L40S). F0 는 Kalman 보다 산술 단순(행렬 필터 없음) → baseline 산출 더 빠름. CPU 시 seed 3→1 자동감소(decision-note carry).
- **성공 기준**: OOF hit_1cm 보고 + band. ≥ F0(perp=0.0) floor + 0.02 = 잔차 GRU 작동(PASS) / ≥ KR002 0.6663 = Kalman 동급 이상(strong) / < floor = FAIL_no-lift(정보). finite, NaN/Inf 0, round-trip assert green.
- **실패 분기**: < F0(perp=0.0) floor → 잔차 GRU 가 F0 baseline 을 오히려 깎음 (informative). KR002(Kalman) 가 lift 의 필요조건이었다는 결론 → 박제. severe 아님(정보).
- **attribution (deferred/optional)**: G_f0 이 KR002 대비 큰 gap 보이면 baseline 외삽 자체의 floor 차(Kalman vs F0 perp0)와 GRU 잔차 효과를 분해 1-fold 진단. **CV-LB 괴리로 OOF attribution 은 약한 신호** 명시.

(plan-a-002 의 KR003·KR004 는 §1.2 대로 둘 다 Kalman 부산물 실험 → 제외 후 본 plan 에 존재하지 않음. 실험 프로토콜·budget·metric·gate 는 plan-a-002 와 동일하게 carry.)

## §4. 서버 작업 순서 (모듈 spec)

### §4.1 f0_baseline.py (c1)
- `f0_perp0(X, end_idx=10)` → `(N,3)`: plan-020 `f0_baseline` 공식 재사용하되 **PERP=0.0** override (D1=1.98, PAR=1.20 유지). 구현은 `f0_baseline` 의 acc_par/acc_perp 분해 로직 그대로 + perp 항 계수만 0.
  - plan-020 `baseline_f0.py` 상수(PERP=−0.20)는 **수정 X** (plan-020 repro 보존). 본 모듈은 별도, PERP 만 인자/지역상수로 0.0.
- F0(perp=0.0) floor: `f0_perp0` 예측으로 5-fold OOF hit_1cm/hit_1p5cm 산출 → `analysis/plan-c-001/f0_perp0_floor.json` 박제 (FR001 비교 기준 §0.5).
- **torch 경로 불요**: GRU 는 잔차만 학습, f0_perp0 는 비학습 numpy baseline (Kalman 과 동일 취급).

### §4.2 run_oof.py 확장 (c2)
- 신규 flag `--baseline {kalman-cv, f0-perp0}` (default kalman-cv = KR002 호환). `f0-perp0` 시 baseline predictor = `f0_perp0(X)`, 그 외(kalman-cv) 시 기존 `kalman_predict`.
  - 잔차 target·final 복원의 `kalman_pred` 자리에 선택된 baseline 대입 (코드 1곳 추상화 — `baseline_pred = f0_perp0(X) if args.baseline=='f0-perp0' else kalman_predict(...)`).
- `--aux-w-weight FLOAT` (default 0.3 = KR002). FR001 은 **0** (W-aux 비활성, §1.2). F-aux λ 는 0.3 carry.
- `--input-yaw` carry (θ=yaw_angle(raw v_last)). `--filtered-yaw` 는 **미구현/금지** (Kalman filtered v 없음, §1.2).
- model n_channels(9)/scal_dim(40) 불변 (입력 feature KR002 동일). GRUModelMultiAux architecture byte-동일.

## §5. 합격 기준 + Gate

| gate | 검사 | PASS band | severity |
|---|---|---|---|
| G0 | import + smoke (1f1s1e finite) + **round-trip assert** (f0_pred+inverse_rotate(rotate(y−f0))==y, atol 1e-6) + W-aux gradient 0 assert (λ_W=0) | green | severe halt if import/NaN/round-trip 실패 |
| G1 | FR001 1-fold 1-seed full-ep hit_1cm | ≥ F0(perp=0.0) floor(1-fold) − 0.005 (잔차 학습 안정 sanity) | warn |
| G_f0 | FR001 full OOF hit_1cm | **≥ floor+0.02 PASS** / ≥ KR002 0.6663 strong / < floor FAIL_no-lift(정보) | < floor = 정보, halt X |
| G_final | FR001 results 박제 + §0.5 sync + main merge | 완료 | — |

- statistic: paired permutation 10000 resample (FR001 vs F0(perp=0.0) floor sample-wise hit, FR001 vs KR002 OOF), p<0.05.
- artifact: `analysis/plan-c-001/f0_perp0_floor.json`, `results_fr001.json/.npz`, `plan-c-001-...results.md`.
- NaN/Inf/divergence 0 의무. cuda OOM 시 batch 256→128→64 자동 감소.
- **CV-LB 괴리 박제 의무**: results 에 OOF Δ(vs KR002, vs floor) 와 함께 "LB 검증 필요(사용자 gated)" 명시. OOF neutral 을 negative 결론으로 박제 금지.
- **제외 보고 의무**: results 에 §1.2 제외 표(Kalman 부산물 6종) 재게시 — "Kalman→F0 swap 이 무엇을 버렸는가" 사후 audit.

## §6. Out of scope

- **Kalman 부산물 feature 일체** (innovation·filtered-v·CV/CA·gain/covariance·filtered-yaw·W-aux target) — §1.2 대로 F0 닫힌형에 부재 → 제외. 제외 자체가 본 plan 의 보고 산출.
- **F0-native 부산물 추가** (per-step F0 잔차, perp on/off disagreement, acc_perp feature 화) — 본 plan 은 *Kalman 제외*만; F0 부산물 회수는 후속.
- **PERP 외 F0 계수 재튜닝** (D1·PAR sweep, perp 를 0 외 값으로) — 사용자 지시 = PERP 0.0 단일 변경. D1=1.98·PAR=1.20 carry.
- baseline 양다리 ensemble (Kalman+F0) — 단일 baseline swap 실험.
- anchor/selector paradigm.
- **autonomous DACON LB 제출** — quota 사용자 명시 confirm 필요 (특히 OOF Δ<threshold 시 더더욱). 본 plan headline = OOF. **단 CV-LB 괴리 때문에 FR001 의 LB 제출을 *사용자 confirm 후* 권장** (입력 yaw 처럼 OOF-neutral·LB-positive 가능성). 제출 = 별 turn, 사용자 승인 gated.

## §7. 참조

- `plans/plan-a-002-kalman-derived-features.md` — 본 plan 의 구조·프로토콜 모체 (Kalman 부산물 plan, 본 plan 이 그 부산물을 전부 제외).
- `plans/plan-a-001-kalman-residual-gru-repro.results.md` — KR001/KR002 OOF·LB, CV-LB 괴리, 잔차 GRU paradigm.
- `analysis/plan-020/baseline_f0.py` — F0 공식 본체 (D1/PAR/PERP, f0_baseline, hit metric). 본 plan 은 PERP 만 0.0.
- `analysis/plan-a-001/{yaw,features,model,losses,run_oof}.py` — KR002 파이프라인 (baseline predictor 자리만 swap).
- `WORKFLOW.md §4` — lane mutex + worktree→main merge (lane c 1번째 plan).

decision-note: spec-default — plan-c-001 = lane c 1번째 plan. plan-a-002 구조/프로토콜 carry + baseline predictor 만 Kalman CV→F0(perp=0.0)(사용자 지시: PERP −0.20→0.0 단일 계수 변경, D1/PAR carry). plan-a-002 의 모든 입력 feature·frame·W-aux 가 Kalman 부산물(재귀필터 산물)이라 F0(닫힌형)에 부재 → 전부 제외(§1.2 6종 보고). KR003/KR004 → baseline-swap 단일 실험 FR001 로 collapse. W-aux head 는 architecture 보존 위해 유지하되 λ_W=0 비활성(Kalman target 회피, model byte-동일). exp prefix FR(F0-Residual) 신규. LB 제출 = 사용자 confirm gated.
