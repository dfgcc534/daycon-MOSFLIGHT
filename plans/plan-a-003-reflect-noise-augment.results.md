---
plan_id: a-003
finished_at: 2026-05-26T05:30+09:00
status: all_complete
lane: a
exp_ids_completed:
  - KR008_reflect-noise-aug
exp_ids_skipped:
  - KR009_reflect-only (deferred — KR008 LB +0.0008 noise floor 내, 분해 noise-limited)
best_exp_id: KR008_reflect-noise-aug
baseline_exp: KR003 (plan-a-002, OOF 0.6667 / LB 0.6854 record)
g_aug_oof_hit_1cm: 0.6671
g_aug_delta_vs_kr003: 0.0004
g_aug_p: 0.8668
g_aug_band: no_regression_PASS
kalman_alone_hit_1cm: 0.5964
lb_score: {KR008: 0.6862}
lb_best: 0.6862
lb_note: KR008 LB 0.6862 = 프로젝트 top (KR003 0.6854 +0.0008, KR002 0.6818 +0.0044). **단 +0.0008 은 LB 노이즈 floor(±0.002~0.003) 내 → inconclusive** — aug 의 진짜 lift 확정 불가 (top 숫자는 갱신하나 noise 와 구분 X). OOF +0.0004(ns)·LB +0.0008 둘 다 미미 → 수확 체감 thesis 확정 (yaw +0.0060 → 부산물 +0.0036 → aug +0.0008).
band: LB_top_but_within_noise
---

# plan-a-003 results — 반사 + 노이즈 augmentation (KR003 위 대칭/정칙화 lever)

## §0. 한 줄 결론

**반사(yaw frame y→−y, p=0.5) + 노이즈(σ=0.10) train-time augmentation 을 KR003 에 추가한 KR008 = OOF-neutral·무회귀** — OOF hit_1cm **0.6671**, KR003 0.6667 대비 Δ=**+0.0004 (p=0.87, ns)** → `no_regression_PASS`. **aug-off 는 KR003 과 bit-identical repro 확증** (RNG guard). 고차 물리 모델은 설계상 명시 배제 (GRU 가 raw seq 로 동역학 이미 학습 → 중복).

> **★ LB 결과 (DACON comp 236716)**: KR008 **0.6862 = 프로젝트 top** (KR003 0.6854 **+0.0008**, KR002 0.6818 +0.0044). **단 +0.0008 은 LB 노이즈 floor(±0.002~0.003, n=10000 paired) 내 → augmentation 의 진짜 lift 는 inconclusive** — top 숫자는 갱신했으나 noise 와 통계적으로 구분 불가. 강제 positive 주장 금지.

> **★ 메타 결론 — lever 수확 체감 확정**: LB lever 효과가 입력 yaw **+0.0060** → Kalman 부산물 **+0.0036** → augmentation **+0.0008(noise 내)** 로 단조 감소. 이 paradigm 은 record 부근에서 **노이즈 floor 에 수렴 완료** — 대칭(yaw)·자기진단(부산물)·정칙화(aug) 축을 차례로 소진. augmentation 은 무해(무회귀)하나 이득이 noise 와 구분 안 되는 지점. 추가 미세 lever 의 ROI 는 사실상 0, 다음은 구조적으로 다른 접근 필요.

## §0.5 Result Quick Reference

| exp | OOF hit_1cm | hit_1p5cm | config A / B | vs baseline OOF | **LB** |
|---|---|---|---|---|---|
| KR002 (plan-a-001) | 0.6663 | — | — | — | 0.6818 |
| KR003 (plan-a-002) | 0.6667 | 0.8205 | 0.6648 / 0.6676 | — | 0.6854 |
| **KR008** (+반사+노이즈 aug) | **0.6671** | 0.8228 | 0.6643 / 0.6677 | KR003 **+0.0004 (p=0.87)** | **0.6862** (top, +0.0008 = noise 내) |

- baseline: KR003 OOF 0.6667 / LB 0.6854. KR002 LB 0.6818. F0 0.6320. Kalman-alone 0.5964.
- aug: 반사 p=0.5 online (seq `_y` idx [1,4,7,10,13] + scalar cvca_y idx 41 + 타깃 y 부호 반전), 노이즈 σ=0.10 (표준화 seq 가산). **train-only**, val/test 원본.
- budget: 2cfg(A/B) × 5fold stable_fold_id × 3seed × 200ep, GPU L40S, 850s.
- **aug-off bit-identical**: `--reflect-aug`/`--noise-aug` 없으면 KR003 과 동일 (smoke 0.6637 일치, RNG 미소비 guard).
- paired permutation 10000 resample, N=10000. submission_kr008.csv 제출 → LB 0.6862. 사용자 confirm 후 제출(오늘 4번째).

## §1. 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1 (반사 → sample efficiency)** | ⚠️ **inconclusive (LB +0.0008 noise 내)** | OOF: KR008 0.6671 ≥ KR003 0.6667 (무회귀 ✓), Δ+0.0004 ns. LB: 0.6862 vs KR003 0.6854 = +0.0008 → **noise floor 내**, 반사 이득 확정도 반증도 불가. |
| **H2 (노이즈 → robust)** | ⚠️ **inconclusive (H1 과 묶음)** | KR008 = 반사+노이즈 묶음. 묶음 LB +0.0008 = noise 내. 노이즈 σ=0.10 가 학습 안정성 안 깨뜨림 (G1 0.6757). |
| **메타 (고차 물리 무의미)** | ✅ **설계 준수** | CTRV/고차 Kalman 배제. lever = 대칭/정칙화 축만. |
| **메타 (수확 체감)** | ✅ **확정** | LB Δ 가 +0.0060(yaw)→+0.0036(부산물)→+0.0008(aug, noise 내) 단조 감소. record 부근 노이즈 floor 수렴. |

## §2. Gate 판정

| gate | 결과 | band/severity |
|---|---|---|
| G0 인프라+smoke+반사항등성+aug-off repro | green (pytest 4 pass, aug-off 0.6637 bit-identical, aug-on finite) | — |
| G1 KR008 1f vs KR003 1f | 0.6757 vs 0.6762 (Δ−0.0005) | PASS |
| **G_aug** KR008 full vs KR003 | **0.6671**, Δ+0.0004 p=0.87 | **no_regression_PASS** (positive 미달) |
| **G_lb** KR008 LB vs KR003 0.6854 | **0.6862** (+0.0008, top) | **within noise floor → inconclusive** |
| G_final | results + §0.5 sync + main merge | 완료 |

## §3. exp 별 산출

### KR008_reflect-noise-aug (G_aug no_regression_PASS / G_lb inconclusive)
- ensemble OOF hit_1cm **0.6671** / hit_1p5cm 0.8228. config A 0.6643, B 0.6677.
- vs KR003 0.6667: Δ=+0.0004, paired permutation p=0.8668 → 무회귀 PASS, lift 비유의(neutral).
- **LB 0.6862** (DACON comp 236716, 사용자 confirm 후 제출): KR003 0.6854 +0.0008 = 프로젝트 top 이나 **noise floor(±0.002~0.003) 내 → inconclusive**.
- **aug-off repro 불변 확증**: off 시 train_one RNG 미소비 → KR003 bit-identical (smoke 0.6637 일치). aug-on 만 RNG 소비.
- 반사 대상 audit: seq `_y` (rel/v/a/innov/fv_y, idx 1,4,7,10,13) + scalar cvca_y(41) + 타깃 y. magnitude/cos/norm 불변 (반사 제외 정당, smoke 검증). `flags.reflect_idx_*` in results_kr008.json.
- 산출: `analysis/plan-a-002/results_kr008.json` + `.npz` + `submission_kr008.csv` (uncalibrated, LB 0.6862). registry `KR008_lb_submit`.

### KR009_reflect-only (deferred)
- 반사 단독 분해 — KR008 LB +0.0008 이 noise 내라 분해는 무의미(noise-limited) → **미실행**.

## §4. 해석 + 함의

1. **augmentation = 무해하나 이득 noise 수준**: 반사+노이즈가 학습 안정 유지(무회귀)하나 OOF 동률(+0.0004), LB +0.0008(floor 내). 반사 — yaw frame 이 회전을 이미 정규화 + GRU 가 좌우 maneuver 를 raw v/a 로 학습 가능해 *추가* sample efficiency 가 미미. 노이즈 정칙화도 이득 noise 수준.
2. **CV-LB 괴리는 이번엔 유의미한 lift 안 냄**: 입력 yaw(+0.0060)·부산물(+0.0036)은 OOF-neutral·LB-명확히-positive 였으나, KR008 은 OOF-neutral·LB-noise내. 즉 CV-LB 괴리가 *모든* OOF-neutral lever 를 LB-positive 로 만들지는 않음 — 이득이 실재할 때만 LB 가 잡고, aug 는 그 이득 자체가 noise 수준.
3. **paradigm 성숙 확정**: lever ROI +0.0060→+0.0036→+0.0008 단조 체감. Kalman-잔차 paradigm 은 record 0.6862 부근에서 대칭/정칙화 축까지 소진 — **추가 미세 lever ROI ≈ 0**. 다음 ROI 는 미세 튜닝이 아니라 별 paradigm·ensemble·데이터 재검토.
4. **인프라 자산**: `run_oof.py` 가 `--reflect-aug --noise-aug` 보유 (aug-off repro 불변) → 향후 aug 실험 재사용.

## §5. Follow-up 후보 (번호 미할당)

- **미세 lever 중단 권고**: yaw·부산물·aug 로 대칭/정칙화/자기진단 축 소진, lever ROI noise 수준 수렴. 추가 augmentation/feature grid 는 ROI ≈ 0.
- **구조적 전환 (다음 ROI)**: (a) 별 paradigm(anchor/selector PB 0.6806) 과의 *selector 기반* ensemble (단순 평균 X — plan-a-002 §5 noise-limited 박제), (b) calibration, (c) 데이터/metric 자체 재검토. 단 전부 record 0.6862 부근 noise floor 와 싸우는 것이라 신중.
- **KR009 반사 단독**: noise-limited → 미실행 유지.

## §6. 재현

```
# G1
python analysis/plan-a-002/run_oof.py --gate g1 --innov --filtered-v --cv-ca --input-yaw \
    --reflect-aug --noise-aug 0.10 --exp KR008 --out g1_kr008.json
# full OOF + submission
python analysis/plan-a-002/run_oof.py --gate full --innov --filtered-v --cv-ca --input-yaw \
    --reflect-aug --noise-aug 0.10 --predict-test --exp KR008 \
    --out results_kr008.json --compare-to results_kr003.npz
# aug-off repro (KR003 bit-identical 확인) + smoke
python analysis/plan-a-002/run_oof.py --gate smoke --innov --filtered-v --cv-ca --input-yaw --exp chk  # → 0.6637
python -m pytest tests/test_plan_a003_smoke.py -q
# 통계 분석 (read-only, 재학습 0)
python analysis/plan-a-002/kr008_stats.py   # → kr008_stats.json
```

## §7. 통계 분석 (KR008 — `kr008_stats.py`)

**방법**: OOF per_sample_hit (n=10000) 기반 — McNemar exact(`binomtest`) + paired permutation + bootstrap 95% CI. **LB 유의성은 test 라벨 부재로 OOF discordant rate 를 proxy SD(=√(b+c)/n)로 추론** (근사이나 verdict 가 2.6/2.5 vs 0.45 로 비-경계라 robust).

### Pairwise lever 유의성 (lever-decay 검정)

| pair (lever) | OOF Δ | disc b/c | McNemar p | bootstrap 95% CI | noise SD | **LB Δ** | LB Δ/SD | **verdict** |
|---|---|---|---|---|---|---|---|---|
| KR002 vs KR001 (입력 yaw) | +0.0024 | 276/252 | 0.317 | [−0.0021,+0.0069] | 0.0023 | **+0.0060** | +2.61 | **real** |
| KR003 vs KR002 (부산물) | +0.0004 | 108/104 | 0.837 | [−0.0025,+0.0033] | 0.0015 | **+0.0036** | +2.47 | **real** |
| **KR008 vs KR003 (aug)** | +0.0004 | 160/156 | **0.866** | **[−0.0031,+0.0039]** | 0.0018 | **+0.0008** | **+0.45** | **noise** |

→ yaw·부산물 LB lift 는 proxy SD 의 **>2.4배 = real** (OOF 로는 비유의였으나 LB 에서 실재 = CV-LB 괴리). **aug 는 0.45 SD = noise** — 방법론이 신호(yaw/부산물)와 noise(aug)를 깨끗이 구분하므로 verdict 신뢰 가능.

### 보강 4축 (aug = noise 확증)

1. **McNemar p=0.866** (discordant b=160/c=156, net=4 → 동전던지기).
2. **bootstrap Δ_OOF 95% CI [−0.0031, +0.0039]** — 0 포함.
3. **per-fold sign 불일치** (2/5 양, Δ −0.0064~+0.0049) → 일관 shift 아닌 fold-noise.
4. **config A/B spread 0.0034 > aug effect 0.0004** → aug 이득이 *config 선택 noise 보다 작음* (KR008 A 0.6643 / B 0.6677).
- **Power**: α=0.05 min detectable Δ = 1.96·0.0018 = **0.0035** > 실제 LB +0.0008 → **검출 불가**.

### 결론

KR008 LB 0.6862 = 프로젝트 top 이나 **+0.0008 은 4축(McNemar p=0.87 · CI 0 포함 · per-fold sign 불일치 · config spread > effect) + power(0.0008 < 0.0035 검출불가)으로 noise floor 내 — aug lift 통계적 미확정**. 동시에 yaw(+2.61 SD)·부산물(+2.47 SD)은 real 로 확정돼 **lever-decay(real→real→noise)** 가 검정으로 박제. artifact: `analysis/plan-a-002/kr008_stats.py` + `kr008_stats.json`.
