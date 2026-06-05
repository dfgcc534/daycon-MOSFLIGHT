---
plan_id: w-001
finished_at: TBD
status: in_progress
lane: w
exp_ids_completed: []
exp_ids_skipped: []
best_exp_id: W001_repro
baseline_exp: KR008 (plan-a-003, LB 0.6862 / OOF 0.6671)
notebook_self_lb: 0.7035 (private, euijin42)
g_repro_status: pending
g_lb_status: pending
band: TBD
---

# plan-w-001 results — rank1 euijin42 (private LB 0.7035) 재현

## §0. 한 줄 결론

**[PENDING G_repro + G_lb 완료]** — rank1 euijin42 노트북 (GRU + Neural ODE + HyperPhysics 30모델 등가중 블렌드) 우리 환경 (L40S CUDA) 재현. notebook self-LB 0.7035 ± 0.005 합격선 vs 실측 LB Δ TBD. 우리 best plan-a-003 KR008 LB 0.6862 대비 paradigm 격차 +0.0173 의 carrier 추정.

## §0.5 Result Quick Reference

| 항목 | 값 |
|---|---|
| G_smoke (c2) | PASS — N_EACH=1 2ep 3-arch finite + submission (10000,4) (L40S 3.4min) |
| G_repro (c3) | TBD — 30 model 학습 + submission_GOH30.csv |
| 학습 시간 | TBD min (예상 ~150min L40S, 노트북 134min MPS) |
| G_lb (c4) | TBD — DACON private LB vs notebook self-LB 0.7035 |
| band | TBD |

## §1. 가설 판정

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1 (재현 가능 ± 0.005)** | TBD | LB Δ from 0.7035 |
| **H2 (paradigm 격차 +0.0173 carrier = 3-arch blend vs single-arch)** | inferred TBD | plan-a-003 KR008 single-arch (Kalman+GRU) vs rank1 3-arch ensemble |
| **H3 (Soft R-Hit loss / cache pretrain / TTA / θ-oversample 중 어느 lever 핵심?)** | TBD (단독 ablation 별도 plan 필요) | reproduce 단독으론 lever 분리 불가 — plan-w-003/4/5 후보 |

## §2. Gate 판정

| gate | 결과 | band |
|---|---|---|
| G0 | PASS | meta.yaml + notebook_dump + GOH30_reproduce.py 박제 |
| G_smoke | PASS | 3 arch finite + submission shape OK |
| G_repro | TBD | 30 model + submission_GOH30.csv 생성 |
| G_lb | TBD (gated) | LB Δ from 0.7035 박제 |
| G_final | TBD | results.md + EXPERIMENTS row + main merge |

## §3. 실측 (학습 + LB)

### §3.1 학습 시간 (L40S CUDA, GPU 1 share)

| arch | seed | cumulative min |
|---|---|---|
| GRU | 0 | 2.2 |
| GRU | 1 | 3.4 |
| GRU | 2 | 4.6 |
| GRU | 3 | 5.9 |
| GRU | 4 | 7.1 |
| GRU | 5 | 8.3 |
| GRU | 6 | 10.6 |
| GRU | 7 | 12.8 |
| GRU | 8 | 15.6 |
| GRU | 9 | 18.3 |
| ODE | 0 | 23.4 |
| ODE | 1 | 27.8 |
| ODE | 2~9 | TBD |
| H | 0~9 | TBD |
| **TOTAL** | **TBD** | |

(노트북 MPS 기준 134min. L40S 우리 환경에서는 GPU 1 share 로 spike 변동 있음.)

### §3.2 submission_GOH30.csv

- shape: TBD (target (10000, 4))
- columns: TBD (target [id, x, y, z])
- finite check: TBD
- 분포 vs 우리 plan-a-003 KR008 submission: TBD (좌표 mean dist)

### §3.3 LB (gated)

- DACON 제출: TBD (사용자 confirm 후)
- public LB: TBD
- private LB: TBD
- Δ from notebook self-LB 0.7035: TBD
- Δ from our KR008 LB 0.6862: TBD

## §4. EXPERIMENTS.md row 박제

| plan_id | title | paradigm | OOF hit_1cm | LB | band | lesson |
|---|---|---|---|---|---|---|
| w-001 | rank1 euijin42 GOH30 재현 (GRU+ODE+HyperPhysics 30모델 블렌드) | multi-arch blend | TBD (full-fit, no OOF — follow-up wrapper 필요) | TBD (target 0.7035 ±0.005) | TBD | TBD |

## §5. Follow-up candidate plans

본 plan 은 단순 reproduce. lever 분리 검증은 별도 plan:

- **plan-w-002**: rank1 + KR008 ensemble (30모델 + KR008 = 31모델 등가 또는 가중 평균) — paradigm-diverse blend 효과 측정
- **plan-w-003**: Soft R-Hit loss 단독 ablation — KR008 위 R-Hit loss 만 적용해 OOF Δ 측정
- **plan-w-004**: cache pretrain (interior transfer e∈{5,6,7,8}) 단독 — KR008 학습 캐시 +5x → OOF Δ
- **plan-w-005**: Y-flip TTA 단독 — KR008 inference 에 TTA 만 추가 → LB Δ
- **plan-w-006**: θ-가중 oversampling 단독 — KR008 학습 sampler 만 θ-weight → OOF Δ (high-curvature tail 강조)
- **plan-w-007**: HyperPhysics 단독 paradigm — gray-box 만 학습해 paradigm 자체 가치 측정

## §6. Lesson 박제 (TBD)

[G_repro + G_lb 완료 후 채움]

- 재현 정확도 (Δ from notebook self-LB)
- 우리 KR008 vs rank1 paradigm 격차의 carrier 추정 (어느 lever 가 가장 큰가)
- CV-LB 괴리 여부 (full-fit 이라 OOF 없음 — caveat)
- 외부 가치 lever (우리에게 없는 것 중 plan-w-2~7 로 ablation 가치)

## §7. caveats

- **full-fit, no OOF**: plan-a-003 KR008 같은 paired-perm 불가. LB 단일 값 비교만.
- **30모델 blend = noise 흡수**: 단일 seed 변동 ±0.001~0.003 은 평균으로 사라짐. LB 차이가 +0.005 외라면 blend 가 아닌 실제 차이.
- **device 차이**: 노트북 MPS, 우리 L40S CUDA → ±0.001~0.003 변동 가능.
- **DACON quota**: 1일 5회. 본 plan G_lb 사용 1회. ±0.005 외이면 single-variable 격리 시도 (시드·EMA decay·epoch 수) 1회 더 사용 가능, 2회 초과 시 inconclusive.
- **license unknown**: meta.yaml license=unknown. reproduce + LB 검증은 학술 목적 fair-use 로 OK. 영구 공개 repo 화 / 우리 main merge 의 분산 시에는 사용자 결정.
