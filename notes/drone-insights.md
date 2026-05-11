# 드론 궤적 예측 도메인 pivot: 통찰 정리

> PB_0.6822 framework (모기 LiDAR trajectory prediction)을 드론 도메인 — 특히 비협조적·회피·손상 비행 — 으로 확장하는 방향성과 paper potential 정리. 가칭 **PhysCand**.
>
> ⚠️ **2026-05 prior art 검증 결과 반영** — 초기 novelty claim 다수가 prior art와 중복되었으나, **예측 horizon (sub-second)**이 결정적 differentiating axis로 부상.

---

## 🕒 결정적 framing: Prediction horizon이 *질적으로 다른 regime*

이 framework의 가장 강한 자리매김은 **time horizon**이다. Prior art 거의 전부가 multi-second 영역이고, 우리는 sub-second inner loop에 속한다.

### Horizon-stratified prior art landscape

| Horizon | Regime 특성 | Prior art 밀도 | 대표 방법 |
|---|---|---|---|
| **3~8 s** (long) | Intent + scene + social interaction dominate | 매우 dense | CoverNet (6s), MultiPath, TNT/DenseTNT, Trajectron++ |
| **0.5~2 s** (medium) | Maneuver mode + 동역학 dominate | dense | CPhy-ML (Nature 2024), Learning-IMM 2025, MTP, Counter-UAV outer loop |
| **50~200 ms** (sub-second) ← **우리** | **Kinematic continuity + sensor latency dominate** | **매우 sparse** | Kalman/EKF, classical IMM (3~5 mode), 일부 MHT |
| <50 ms | Pure state estimation | (다른 문제) | EKF/UKF, particle filter |

→ **우리 80ms는 prior art가 비어 있는 sub-second 영역**. CoverNet/CPhy-ML/Learning-IMM 모두 같은 *문제*를 푸는 게 아니라 **다른 horizon에서 다른 문제**를 푼다.

### 왜 sub-second regime은 질적으로 다른가

이 regime에서만 성립하는 특성:

1. **Intent/scene 신호가 거의 0** — 11 frame × 40ms = 440ms 관측은 intent를 추론할 시간이 안 됨. 오직 운동학적 연속성만 신호가 됨.
2. **Sensor latency가 horizon의 6~25%** — 5~20 ms LiDAR 지연이 80ms 예측에 결정적 영향. 1초 예측에선 0.5~2%로 무시 가능.
3. **Action commitment time scale** — 80ms는 *"방어 행동을 commit해야 할 시간"*. 결과는 *binary success/failure* (hit/miss). 평균 trajectory error는 무의미.
4. **Kinematic regime이 시간적으로 stable** — 80ms 안에 모기/드론의 행동 regime은 거의 안 바뀜. Empirical Bayes regime prior가 *정당화됨*. 6초에선 regime이 여러 번 transition → 같은 가정 깨짐.
5. **Heavy-tail outlier가 dominant** — Gaussian 가정이 가장 약한 영역. saccade-like 방향 전환은 sub-second에서 발생.

### 인용 가능한 외부 증거
- VR/AR 문헌: "60ms is the upper limit for acceptable latency" — sub-second에서 latency가 결정적
- 추적 문헌: "1ms of MEMS gyroscope data may be more informative than 100ms of optical tracking" — sub-second에서 *센서 latency가 dominant uncertainty*임을 직접 증언
- Counter-UAV 시스템: "perception loop closes in tens of milliseconds while airframe vibrates" — 100ms control loop과 0.5~2s 예측 horizon이 *별개로 운영*됨

### Framework의 각 design 결정이 sub-second에서만 *필수*인 이유

| Design | Sub-second (우리) | Long-horizon (CoverNet 등) |
|---|---|---|
| 27 physics candidate | 운동학적 reachable set 균등 sampling으로 충분 | 부족함. Intent 공간이 훨씬 큼. 학습된 anchor 필요 |
| Latency family | *필수*. 지연이 horizon의 6~25% | 불필요. 지연이 horizon의 <1% |
| 18-regime empirical Bayes | 정당. Regime이 80ms 안에 stable | 부적절. 6초 안에 regime 여러 번 변화 |
| Hit-boundary tiny correction | *필수*. Binary success/failure가 곧 평가 | 부적절. Mean trajectory error가 평가 |
| Frenet local frame | 강력. 운동학적 prior가 dominant | 부분적. Map/lane 좌표가 더 중요 |
| MHT/Kalman baseline | 자연스러운 비교 대상 | 비교 대상 아님 |

→ **모든 design 결정이 horizon 80ms에 *맞춰져* 있고, 다른 horizon에선 합리적이지 않다.** 이게 framework의 *진짜* 자리매김.

---

## 🎯 Reachability-Based Candidate Enumeration

Sub-second regime에서 framework의 한 줄 정의:

> 비행체 모델 M의 **forward reachable set R(+80ms)** 를 27개 후보로 sampling하고,
> selector가 *그중 어느 cell에 도달할지의 posterior*를 출력한다.

Sub-second이므로:
- R(+80ms)이 *비교적 작고 계산 가능*
- 27 candidate로 균등 sampling이 *충분히 dense*
- Online판단은 80ms 안에 다음 action commit → posterior가 *그대로 planner 입력*

```
출력:
  P("aggressive evade right") = 0.42
  P("hover decelerate")       = 0.31
  P("constant cruise")        = 0.20
  P("aggressive climb")       = 0.07

→ 80ms 안에 interceptor가 받아 즉시 commit.
  long-horizon 방법은 이 시간 안에 추론 자체가 안 끝남.
```

⚠️ CoverNet (CVPR 2020)이 bicycle-model dynamic anchor를 제안했지만 **6초 horizon에서**. 우리 sub-second 영역에는 bicycle model이 너무 거칠고, 27 candidate가 더 적합. *같은 trick이 horizon에 따라 다른 의미를 가짐*.

---

## 🔁 왜 드론 (sub-second 영역에서)

### Sub-domain별 매칭 강도

| Sub-domain | 매칭 | 비고 |
|---|---|---|
| **Counter-UAV interception inner loop** | ★★★ | 80ms = 요격 commit time, sensor latency 핵심 |
| **Adversarial / evasive drone tracking** | ★★★ | game-theoretic, prior art 약함 |
| **고속 회피 비행 (FPV freestyle)** | ★★ | mosquito-like kinematics, sub-second saccade |
| **손상/오작동 드론 추적** | ★★ | dynamics 불확실, sub-second 변동 |
| 강풍/난기류 속 소형 드론 | ★ | gust response = mosquito-like |
| 협조적 cruise 드론 | overkill | Kalman 충분 |
| **장거리 (>0.5s) 예측** | mismatch | **다른 regime — 우리 framework이 잘 작동하지 않음. 솔직히 인정** |

→ Framework은 **sub-second 영역의 비협조적 소형 비행체**에 specifically 맞춰져 있고, 다른 horizon으로 일반화를 *주장하지 않는 게* 정직.

### 회피 비행 + sub-second에서 Kalman/IMM이 망하는 이유

| 회피 비행 특성 | Kalman/IMM 문제 | 이 framework 대응 |
|---|---|---|
| 갑작스러운 saccade-like 방향 전환 | 선형 가정 깨짐, IMM mode-switch lag이 80ms와 비슷 | turn/jerk family가 *후보로 미리* 있음 — 전환 *예측* 불필요 |
| 고 jerk burst | Gaussian noise 깨짐 (heavy-tail) | 분류 기반이라 heavy-tail robust |
| 센서 지연이 horizon의 6~25% | noise filter로 흡수 시 발산 | **latency family로 enumerate** |
| Dynamics 불확실 (적 드론 사양 모름) | EKF는 정확한 dynamics 필요 | 27 후보로 dynamics 공간 sampling |
| 요격 = 작은 boundary 통과 | Kalman은 mean 최소화 (boundary 무관) | tiny correction이 boundary metric 직접 최적화 (→ §B 참조) |

---

## 🔧 Framework component 매핑 (모기 → 드론, sub-second 영역)

### 27 candidate family
```
base       → hovering / station-keeping
acc        → cruise with constant acceleration
frenet     → banked coordinated turn
turn       → aggressive evasive maneuver
jerk       → motor command transient (throttle step)
latency    → sensor-system delay
```

### Regime bin
- 모기: `(속도) × (곡률) × (속도변화 slope)` → 18
- 드론: `(속도) × (선회 g-force) × (고도변화)` → 18

### Reachable set 정의 (sub-second)
```
드론 모델 M (질량 m, 최대 추력 T_max, 모터 응답 시상수 τ)
  ↓
+80ms 내 reachable set R:
  - 최대 가속도        a_max = T_max/m
  - 최대 jerk          j_max = T_max/(m·τ)
  - 최대 선회 g-force  g_max = a_max·sin(θ_max)
  ↓
27 candidate = R의 sampling cell 중심
```

### 드론 클래스 모를 때 (Counter-UAV)
- **Conservative**: 가능한 드론 클래스의 합집합 R
- **Class-conditional**: 관측 11 frame에서 클래스 추정 → 해당 R만 사용

---

## 🌟 Novelty 정직 재평가 (horizon 인지 후)

| Element | 상태 | Prior art와의 관계 |
|---|---|---|
| **Sub-second horizon 자리매김 자체** | **★★★ 가장 강한 자리매김** | Prior art (CoverNet/CPhy-ML/Learning-IMM)는 모두 multi-second. 우리는 *별개 regime* |
| **27 × 18 = 486-entry empirical Bayes lookup-table-bias (§A)** | **★★★ 방법론 contribution의 핵심** | 정확한 recipe (sample-conditional 2D regime×candidate empirical prior + DNN residual) prior art에서 미발견. 표준 Logit Adjustment는 1D global. 상세는 §A 참조 |
| Latency family | **★★★ sub-second에서 필수** | Long-horizon에선 무의미. Sub-second에서 *처음으로 의미*가 생김 |
| Physics-prescriptive candidate enumeration | ⚠️ horizon-conditioned | CoverNet도 bicycle anchor 있지만 6초 horizon용. Sub-second 27-cell sampling은 미발견 |
| Capacity-aligned hit-boundary correction (§B) | ✓ 가장 살아남음 | Focal/Boundary loss와 차별. *Sub-second binary commit*에 특화. 상세는 §B 참조 |
| Reachability sampling grounding | ✓ narrow | Sub-second에서 R이 작아 sampling 정당화됨 |

### Master 차별점 (paper의 머니샷)

> **"기존 trajectory prediction은 multi-second horizon에서 intent/scene/social을 푼다.
> 우리는 sub-second horizon에서 kinematic continuity + sensor latency + binary commit을 푼다.
> 이건 *다른 문제*이고, 다른 design 원칙이 필요하다."**

이 한 문장이 모든 prior art reviewer 비판의 master 방어이자 paper의 자리매김.

---

## 📚 Prior art landscape (horizon 표시)

### Long-horizon (3~8 s) — 우리와 다른 regime

| 논문 | Horizon | 우리와의 관계 |
|---|---|---|
| **[CoverNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Phan-Minh_CoverNet_Multimodal_Behavior_Prediction_Using_Trajectory_Sets_CVPR_2020_paper.pdf)** (CVPR 2020) | 6 s | Bicycle anchor는 같은 컨셉이지만 *long-horizon에 최적화*. Sub-second 적용 미검증 |
| **MultiPath / TNT / DenseTNT** | 3~6 s | Anchor 기반이지만 scene/map context dominant |
| **Trajectron++** (ECCV 2020) | 1~8 s | Dynamics-integrated 연속 분포, scene-aware |

### Medium-horizon (0.5~2 s)

| 논문 | Horizon | 우리와의 관계 |
|---|---|---|
| **[CPhy-ML](https://www.nature.com/articles/s44172-024-00179-3)** (Nature Comm Eng 2024) | 보통 1~수 s (intent 위주) | Drone intent inference — sub-second에선 intent 신호가 없어 무용 |
| **[Learning-based IMM for drones](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25001196)** (Transp Res C 2025) | seconds-level | 3 Transformer + learned IMM, 우리와 가장 가까운 *시도*지만 horizon이 다름 |
| **[MTP](https://arxiv.org/abs/2110.09481)** (NVIDIA 2022) | nuScenes ~6 s | Multi-hypothesis 컨셉이 비슷하지만 long-horizon |
| **Anti-UAV outer loop** | 0.5~2 s | Counter-UAV planning에 쓰이지만 우리 inner loop과는 별개 |

### Sub-second (50~200 ms) — 우리 영역

| 방법 | 비고 |
|---|---|
| **Kalman / EKF / UKF** | 표준 baseline. dynamics가 정확해야 작동 |
| **Classical IMM** (3~5 model) | 가장 자연스러운 비교 baseline. mode 수 부족 |
| **MHT (Multiple Hypothesis Tracking)** | retrodiction-with-delay 변형이 latency 처리. Bayesian 형태 |
| **이 framework** | 27 candidate enumeration + DNN classifier + empirical Bayes prior + §B correction |

→ Sub-second non-cooperative target prediction with deep classifier는 **거의 비어 있는 영역**.

### 컨셉 조상 (인용 권장)
- IMM (Bar-Shalom 가족)
- MHT retrodiction (Blackman, Bar-Shalom)
- Hamilton-Jacobi reachability (Tomlin, Bansal)
- Set-based reachability (Althoff CORA)
- Funnel libraries / LQR-trees (Tedrake)
- Focal loss (Lin, ICCV 2017) — §B 참조
- Boundary loss segmentation (Kervadec, MIDL 2019) — §B 참조

### 응용 prior art
- [IMM-Informer for ADS-B flight](https://pmc.ncbi.nlm.nih.gov/articles/PMC12031469/) (2025, multi-second)
- [Deep IMM Filtering (NASA)](https://ntrs.nasa.gov/api/citations/20240003219/downloads/Deep%20IMM.pdf?attachment=true)
- [Anti-UAV Survey (CVPR Workshop 2025)](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/papers/Dong_Securing_the_Skies_A_Comprehensive_Survey_on_Anti-UAV_Methods_Benchmarking_CVPRW_2025_paper.pdf)
- [Boundary-Guided Trajectory Prediction](https://arxiv.org/abs/2505.06740) (2025, road boundary 위주)
- [Efficient and Robust Online Trajectory Prediction for Non-Cooperative UAVs](https://arc.aiaa.org/doi/10.2514/1.I010997) (AIAA JAIS) — horizon 확인 필요
- DMoE for drone trajectory
- PIRC (Physics-Informed Reservoir Computing)

---

## 📌 Element A: 27 × 18 = 486-Entry Empirical Bayes Lookup-Table-Bias

이 framework의 **방법론 contribution의 핵심**이자 sub-second regime에서 *uniquely* 효과적인 recipe.

### Recipe

```
[입력] 11 frame trajectory
       ↓
[Regime classifier] 운동학 통계 (속도/곡률/jerk slope)
                    → 18-bin 중 하나의 regime id
       ↓
[Lookup] 486-entry table[regime][candidate] = empirical hit rate
                                              (training data + EB shrinkage)
                    → 27차원 prior vector
       ↓
[DNN classifier] 11 frame → 27차원 residual logit
       ↓
[Combine] final_logit = α·log(prior_vec) + β·DNN_residual
       ↓
[Softmax] 27 candidate posterior
```

### 왜 *uniquely* 효과적인가 (10 reasons)

| # | 이유 | 비고 |
|---|---|---|
| 1 | **Data efficiency** | 486 통계값이 base-rate 분담 → DNN은 residual만 학습. 작은 drone 데이터셋에서 결정적 |
| 2 | **Sample-conditional (2D)** | 표준 Logit Adjustment는 1D global. 우리는 (regime × candidate) 2D 조건부. 정보 density 한 차원 높음 |
| 3 | **Compositional decomposition** | 표 = "kinematic 상태 → 어떤 물리 가설이 이김". DNN = "trajectory 모양 → 표에서 얼마나 벗어남". 깔끔 분업 |
| 4 | **Sub-second에서만 stable** | 80ms 안에 regime 안 바뀜 → 표값이 time-invariant. 6초면 regime 여러 번 전환 → 표 의미 잃음 |
| 5 | **EB shrinkage 자동 regularization** | `α = n/(n+18)` — sample 적은 cell은 global로 후퇴. hyperparameter 거의 없음 |
| 6 | **Interpretability** | `table[regime=5][frenet_par100] = 0.73` — 도메인 전문가가 직접 검증/디버그. Black-box DNN은 못 함 |
| 7 | **Distribution shift robust** | Regime 정의는 환경 무관. 환경 바뀌어도 표 안정. DNN residual만 흡수 |
| 8 | **CoverNet/MultiPath gap 메움** | CoverNet uniform softmax, MultiPath learned anchor prob (data-hungry). 우리는 empirical prior + DNN residual |
| 9 | **IMM 한계 우회** | IMM은 3~5 mode + Markov transition (state-evolution). 우리는 27 mode + current-state regime conditioning. 구조 자체가 다름 |
| 10 | **Sub-second + 작은 데이터에서 collectable** | Sliding window로 sub-second sample 다량 추출 → 486 cell 채울 수 있음. 6초 horizon에선 cell sparse |

### Paper 방법론 contribution 한 줄

> **"Sample-conditional 2D (regime × candidate) empirical-Bayes lookup-table-bias on DNN logits — Logit Adjustment의 2D 일반화 + 분류기 prior 주입 + IMM의 deep generalization을 동시에 묶은 새 recipe. Sub-second horizon에서 regime stability + cell collectability 두 조건이 동시 성립할 때만 정당화됨."**

### Prior art 정확한 비교

| 기법 | Prior 형태 | 조건부성 | 한계 |
|---|---|---|---|
| **Logit Adjustment** (NeurIPS 2023) | 1D global class frequency | None | uniform shift |
| **Neural Prior Estimation** | Learned latent prior | learned | data-hungry |
| **Empirical Bayes for trajectory** (2022) | Parameter-space prior | global | candidate logit과 무관 |
| **IMM Markov transition** | State-evolution matrix | previous state | candidate score 아닌 transition |
| **CoverNet softmax** | Uniform | None | base rate 학습 |
| **이 framework** | **2D (regime, candidate) empirical hit-rate** | **sample-conditional** | sub-second 영역에서만 유효 |

→ 정확한 등가물 없음. *narrow but real* novelty.

---

## 📌 Element B 참조: Capacity-Aligned Metric-Aware Correction

(b) **hit-boundary tiny correction module**은 우리 framework의 핵심 살아남는 novelty이고, **sub-second binary commit regime에 specifically 적합**.

상세 framing은 노트북 부록에 정리되어 있다:

→ **[PB_0.6822 코드공유.ipynb §부록 "설계자 관점 framing"](PB_0.6822%20코드공유.ipynb)** 참조.

### Sub-second drone interception에서 §B가 살아나는 이유
- **80ms = action commit time** — mean error 줄이는 게 아니라 *binary hit/miss flip이 중요*
- 1cm/0.6cm cap이 **요격 WEZ margin**과 직접 매핑
- "flip 가능한 sample에 집중" = "**아슬아슬한 요격 회복**"
- Long-horizon에선 평가가 ADE/FDE (평균 거리) → §B 효과 미미. Sub-second에선 hit rate → §B 효과 directly 입증

→ Counter-UAV sub-second 영역에서 §B의 design principle이 *논리적으로 가장 직결*되는 element.

---

## 📊 Data landscape

### Sub-second 회피 비행 실측 proxy
- **Blackbird Dataset** (MIT, Antonini 2018) — Mocap aggressive quadrotor, sub-second sampling
- **UZH-FPV Drone Racing** (UZH/ETH 2019) — FPV racing, high-frequency
- **EuRoC MAV** — cruise-y baseline
- **TUM VI** — visual-inertial

### Counter-UAV (detection 위주)
- **Anti-UAV Challenge** (CVPR Workshop, 다년간)
- **DUT Anti-UAV**, **Drone-vs-Bird Detection**, **MAVREC**

### Simulation (sub-second 통제 실험의 본업)
- **AirSim** (Microsoft) — sub-second sampling + 임의 LiDAR + adversarial RL 학습 가능
- **Flightmare** (UZH) — 빠른 quadrotor sim
- **gym-pybullet-drones** (UTIAS)
- **RotorS / Gazebo** (ETH)

### 권장 데이터 조합

```
[A] AirSim + 자체 adversarial drone policy (PPO로 예측기 회피 학습)
    ├─ Sub-second sampling (40ms frame interval)
    ├─ latency 주입 통제 실험 (5/10/20ms)
    ├─ 후보 수 ablation
    └─ end-to-end interception 성공률 측정

[B] Blackbird + UZH-FPV 실측 cross-validation
    └─ Sub-second window로 slicing → AirSim 학습 → zero-shot transfer

[C] (옵션) 모기 LiDAR 대회 데이터
    └─ "같은 sub-second kinematic class의 다른 testbed"

비교 baseline:
  - Kalman / EKF (sub-second 표준)
  - Classical IMM 3~5 mode
  - CoverNet 변형 (sub-second로 retrain) — *horizon mismatch 입증용*
  - Learning-IMM 2025 (sub-second로 retrain) — *horizon mismatch 입증용*
  - CPhy-ML — intent 신호 없음 입증
```

### 솔직한 gap
- Sub-second 회피 의도 + 외부 LiDAR + 3D trajectory 모두 만족하는 public 데이터 = **없음**
- AirSim 합성이 통제 실험의 본격 경로

---

## ⚖️ Paper claim 정직 framing (horizon-centric)

### Title 후보
- **"PhysCand: Sub-Second Trajectory Prediction for Non-Cooperative Aerial Targets via Reachability-Sampled Candidate Selection"**
- **"Latency-as-Hypothesis: Capacity-Aligned Trajectory Prediction in the Sub-Second Regime"**

### Abstract 골격 (horizon이 master positioning)

> Trajectory prediction has been extensively studied at multi-second horizons (3–8s) where intent, scene, and social interaction dominate (CoverNet, MultiPath, Trajectron++, CPhy-ML, learning-based IMM). We address a qualitatively different regime: **sub-second prediction (50–200ms)** for non-cooperative aerial targets, where kinematic continuity, sensor latency, and binary action commitment dominate. At this horizon, intent signals vanish, sensor delay becomes 6–25% of the prediction window, and the evaluation metric naturally becomes *hit-rate* rather than mean displacement error.
>
> We propose **PhysCand**, with three coupled design elements specifically suited to this regime:
>
> 1. **27 × 18 = 486-entry empirical-Bayes lookup-table-bias on DNN logits** (§A) — a 2D sample-conditional (regime × candidate) generalization of Logit Adjustment, combining classical empirical Bayes with deep classifier scoring. Stable at sub-second horizons (regime invariant within window) and collectable from sliding-window sub-second samples.
> 2. **Reachability-sampled 27-candidate enumeration** spanning behavior mode × coefficient × *time-scale variation* (the latter being the **latency family**, treating sensor delay as enumerable hypothesis rather than noise), replacing learned anchors that target multi-second intent space.
> 3. **Capacity-aligned hit-boundary correction module** (§B) — cap + bell-shape weighting + zero-init delta aligning learning signal with the binary action-commitment metric (see notebook appendix for design rationale).
>
> We validate on AirSim adversarial-drone simulation with **end-to-end interception success rate** comparison against Kalman/IMM/MHT (sub-second baselines) and CoverNet/CPhy-ML/Learning-IMM (multi-second methods retrained at our horizon, to expose the regime mismatch). Cross-validated on Blackbird and UZH-FPV agile flight data; insect LiDAR data included as a kinematic analogue testbed.

---

## 🛡️ Reviewer push-back & 방어 (horizon이 master 방어)

| 비판 | 방어 |
|---|---|
| **"CoverNet도 physics anchor 있다"** | **Horizon이 다르다.** CoverNet은 6초, 우리는 80ms. Bicycle model anchor가 sub-second에선 너무 거칠고 latency 차원 부재. *Sub-second로 retrain한 CoverNet baseline 비교*로 입증 |
| **"Learning-based IMM for drones (2025)도 있다"** | **Horizon이 다르다.** 그쪽은 seconds-level, 우리는 80ms. Transformer 3개 구조가 sub-second 11-frame 짧은 sequence에는 overparameterize. 직접 비교 baseline으로 |
| **"CPhy-ML (Nature 2024)이 있다"** | **Intent 추론은 sub-second에 불가능.** 80ms 안에 intent 신호 없음. CPhy-ML은 intent를 위해 multi-second 필요. Horizon mismatch |
| **"왜 이 horizon에 집중하는가?"** | Counter-UAV inner loop = 80ms commit time. *기존 multi-second 방법은 이 시간 안에 inference 자체가 안 끝남*. 응용 motivation 명확 |
| **"Multi-second horizon으로 일반화 못 하면 contribution 좁지 않나"** | **정직 인정. 일반화 주장 안 함.** 다른 horizon은 다른 문제이고, 이미 잘 풀려 있음 (CoverNet 등). 우리는 *sub-second specifically* |
| **"Hit-rate metric이 표준이 아니다"** | Sub-second action commit regime에서 ADE/FDE는 부적절. Binary success/failure가 자연. 별도 metric 자체가 contribution |
| **"18-regime empirical Bayes는 stratified prior 변형?"** | 표준 stratified prior는 *output 분포*에 적용. 우리는 *후보 routing logit*에 **2D sample-conditional**로 적용. Logit Adjustment는 1D global이지만 우리는 2D (regime × candidate). 정확한 등가물 없음 — §A 참조 |
| **"표준 Logit Adjustment / Neural Prior Estimation과 차이?"** | LA는 1D global class frequency. NPE는 learned latent prior. 우리는 **명시적 2D empirical hit-rate table** + sample-conditional 적용. 정보 density 한 차원 높고, EB shrinkage로 자동 regularize. 정확한 prior art 미발견 |
| **"왜 sub-second에서만 정당화되나"** | (a) 80ms 안에 regime stable → 표값 time-invariant (b) sliding window로 486 cell 채울 sample 추출 가능. 두 조건이 *동시에* 성립하는 horizon이 sub-second |
| **"27이라는 숫자가 임의적"** | Sub-second R(+80ms)이 비교적 작아 27 cell이 sufficient. 후보 수 ablation 그래프로 입증 |
| **"AirSim 합성에 의존 너무 큰가"** | Sub-second 회피 + 외부 LiDAR public 데이터 부재 → 합성이 *유일한 본격 경로*. 단 Blackbird/UZH-FPV 실측으로 cross-validation |
| **"방법론 본 학회로 가기엔 좁다"** | 정직 인정. 본 학회 대신 응용 (ICRA/IROS/AIAA) + Anti-UAV workshop이 적합 |

---

## 🧪 핵심 실험 우선순위 (시간 무관, 우선순위 순서)

1. **Framework 이식**: AirSim 80ms horizon에서 모기 코드 그대로 hit rate 측정
2. **Horizon-stratified baseline 비교**:
   - Sub-second native: Kalman / EKF / classical IMM (3~5 mode) / MHT
   - Multi-second 방법을 sub-second로 retrain: CoverNet / Learning-IMM 2025 / CPhy-ML — **horizon mismatch 입증용**
3. **Latency 주입 ablation** (0/5/10/20 ms): *latency family가 진짜 도움 되는지* 통제 실험. **paper 머니샷 1**
4. **§B 절제 ablation**: cap / boundary weighting / zero-init 각각 off — *capacity-aligned 세 요소가 진짜 한 묶음인지* 입증
5. **후보 수 ablation** (5/13/27/50): 27 정당화
6. **Adversarial policy 학습**: AirSim 드론이 예측기 회피하도록 PPO 학습 → 공진화 setup
7. **End-to-end interception**: 단순 planner 결합, 요격 성공률 측정. **paper 머니샷 2**
8. **Cross-domain**: Blackbird / UZH-FPV zero-shot transfer
9. **(옵션) 모기 ↔ 드론 transfer**: 같은 sub-second kinematic class 검증
10. **(옵션) Horizon scan**: 40/80/160/320 ms로 horizon 늘려가며 우리 framework 우위 감소 입증 — *우리 영역의 경계*

---

## ⚠️ 윤리 / 저작권 메모

- 원본 framework은 Dacon code share ([PB_0.6822 코드공유.ipynb](PB_0.6822%20코드공유.ipynb)) 작성자의 아이디어
- Paper 작성 시 **반드시 인용 + 사전 연락**
- 우리(연구자) 측 contribution은 다음으로 명확히 구분됨:
  - **Sub-second non-cooperative target prediction regime의 명시적 자리매김**
  - drone domain 이식 + AirSim 통제 실험
  - reachability framing & 학술 grounding
  - **End-to-end interception 성공률 평가 metric**
  - latency / 후보 수 / §B / horizon ablation
  - cross-dataset validation

---

## 🔗 관련 파일 / 외부 참조

- [PB_0.6822 코드공유.ipynb](PB_0.6822%20코드공유.ipynb) — 원본 framework + §부록 "설계자 관점 framing" (§B 상세)
- [mosquito-trajectory-ideas.md](mosquito-trajectory-ideas.md) — 모기 도메인 아이디어 카탈로그
