# Prior Paper → 대회 적용 가능 trick 정리

> 검색으로 확인된 prior art 6~7개에서 *PB_0.6822 framework 위에 즉시 얹을 수 있는* 구체적 전술만 추출.
> Paper novelty 논의가 아니라 **LB score를 어떻게 더 올리느냐**의 단서장.
>
> 관련 파일:
> - [PB_0.6822 코드공유.ipynb](PB_0.6822%20코드공유.ipynb) — 적용 대상 framework
> - [drone-insights.md](drone-insights.md) — paper용 전략 (별도)
> - [mosquito-trajectory-ideas.md](mosquito-trajectory-ideas.md) — 모기 도메인 아이디어 카탈로그

---

## 사용법

각 paper에 대해 다음 4축으로 정리:

- **핵심**: 원논문의 메인 idea 한 줄
- **대회 적용**: PB_0.6822 코드 위에 어떻게 얹는가 (concrete)
- **예상 이득**: 추정 LB 향상 (보수적)
- **작업량**: 코드/학습 비용 추정
- **주의**: 망할 수 있는 경로

ROI 순서대로 정렬 (가장 큰 신호부터).

---

## 1. ⭐⭐⭐ [Learning-based IMM filter for drones (Transp. Res. Part C, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25001196)

### 핵심
Position / velocity / acceleration **세 채널을 별도 Transformer로 병렬 예측** → LSTM이 transition 확률을 *적응적으로* 학습해서 융합.

### 대회 적용 — "3-channel selector ensemble"

PB_0.6822의 `attn_gru` selector 하나 대신, **입력 feature를 3가지로 분리한 selector 3개** 학습:

```python
# 현재: 하나의 SEQ_FEATURE 모두 입력
# 변경: 3종으로 분리
channel_pos = [좌표 변화 중심 feature]     # x, y, z deltas
channel_vel = [속도 중심 feature]          # speed, prev_speed_ratio, turn_cos
channel_acc = [가속도 중심 feature]        # acc_norm, jerk, curvature

selector_pos = AttnGRU(channel_pos) → logits_pos
selector_vel = AttnGRU(channel_vel) → logits_vel
selector_acc = AttnGRU(channel_acc) → logits_acc

# 융합 (단순 평균 또는 학습된 가중치)
logits = w_p · logits_pos + w_v · logits_vel + w_a · logits_acc
```

가벼운 버전: 같은 selector를 *input feature subset 3종*으로 재학습 → seed ensemble처럼 OOF에서 평균.

### 예상 이득
**+0.003 ~ +0.006**

### 작업량
- 학습 시간 ×3
- 코드 변경: feature 분리 함수 + 융합 로직 + 학습 스크립트 wrapping (1.5일)

### 주의
- 채널별 underfitting 우려 — channel별 hidden을 작게 (32~48) 시작
- 융합 가중치를 학습으로 풀려면 attention head 추가, 단순 평균이 더 안전한 첫 시도

---

## 2. ⭐⭐⭐ [CoverNet (Phan-Minh et al., CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Phan-Minh_CoverNet_Multimodal_Behavior_Prediction_Using_Trajectory_Sets_CVPR_2020_paper.pdf)

### 핵심
후보를 *고정 grid*가 아니라 **현재 state (속도/가속도/yaw rate) 의존**으로 bicycle-model에서 동적으로 생성 (input-conditional dynamic anchor).

### 대회 적용 — "State-conditional anchor 계수"

PB_0.6822의 27 후보 자체는 그대로 두되, **각 후보의 `(d1, par, perp)` 계수를 sample별로 미세 조정**:

```python
# 현재: CandidateSpec("frenet_par100_perp000", 1.98, 1.00, 0.00)
# 변경: 현재 state에 따라 계수 perturb

current_speed = compute_speed(x, end_idx)
current_curvature = compute_curvature(x, end_idx)

# 빠르고 직진인 모기 → par 증폭, perp 감쇠
# 느리고 휘는 모기 → par 감쇠, perp 증폭
par_scale = 1.0 + α · normalize(current_speed) - β · normalize(current_curvature)
perp_scale = 1.0 + γ · normalize(current_curvature)

# 후보 생성 시 적용
for spec in CANDIDATES:
    effective_par  = spec.par  * par_scale
    effective_perp = spec.perp * perp_scale
    candidate = p0 + spec.d1 * v_scale * d1 + ...
```

`α, β, γ`는 OOF에서 grid search (각 ∈ {0, 0.1, 0.2, 0.3}).

### 예상 이득
**+0.002 ~ +0.005**

### 작업량
- `make_candidates()` 수정 (1일)
- α, β, γ 3차원 grid search (학습 ×27)

### 주의
- oracle hit rate가 떨어지면 (정답이 후보 set 밖으로 나가면) α 축소
- 효과가 sample 균등하지 않으면 *fast* / *slow* mosquito 그룹별 분리 가능

---

## 3. ⭐⭐ [MTP — Multi-Hypothesis Tracking and Prediction (NVIDIA, 2022)](https://arxiv.org/abs/2110.09481)

### 핵심
단일 tracking 결과가 아니라 **과거 trajectory에 대한 *여러 parsing*** (서로 다른 smoothing/outlier 제거)을 prediction에 입력 → cascading error 감소, nuScenes 34% 향상.

### 대회 적용 — "Multi-parse input"

11 frame 좌표를 *3종으로 사전 가공*해서 각각 selector에 통과, logit 평균:

```python
# Parse 1: raw 좌표 (현재 PB_0.6822)
# Parse 2: Savitzky-Golay smoothing (window=5, order=2)
# Parse 3: EMA smoothing (window=3, alpha=0.6)

x_raw = trajectory
x_sg  = savgol_filter(trajectory, 5, 2, axis=time)
x_ema = ema_smooth(trajectory, alpha=0.6)

logits = (selector(x_raw) + selector(x_sg) + selector(x_ema)) / 3
```

### 예상 이득
**+0.002 ~ +0.004**

### 작업량
- 추론 wrapper 0.5일 (학습은 raw만 해도 됨, parse는 augmentation 효과)
- 만약 parse 종류별 학습 분리 시 ×3 학습 시간

### 주의
- LiDAR 노이즈 강도에 따라 SG/EMA 효과 다름. window/alpha를 OOF로 튜닝
- 학습 시 random parse 선택을 augmentation으로 쓰면 효과 ↑

---

## 4. ⭐⭐ [CPhy-ML (Perrusquia et al., Nature Comm Eng 2024)](https://www.nature.com/articles/s44172-024-00179-3)

### 핵심
Deep learning + **control physics 보존법칙** (운동량/에너지) 제약 → 48% 향상.

### 대회 적용 — "Physics conservation regularizer"

Tiny corrector의 출력 `Δx`가 *kinematically implausible* 한 경우 페널티:

```python
# 모기의 생물학적 jerk 한계 추정 (data에서 quantile 99 기준)
typical_jerk_step = 0.004  # 대략 4mm (40ms 사이 jerk delta 한도)

# corrector loss에 추가
delta = corrector_output
delta_jerk = compute_jerk_norm(delta, recent_acc, recent_jerk)

physics_penalty = max(0, delta_jerk - typical_jerk_step) ** 2
loss = boundary_weighted_loss + λ · physics_penalty
```

`λ ∈ {0.1, 0.3, 1.0, 3.0}` 로 OOF tuning.

### 예상 이득
**+0.001 ~ +0.003**

### 작업량
- corrector loss 함수 수정 (0.5일)
- λ grid search (학습 ×4)

### 주의
- λ가 너무 크면 corrector가 0에 collapse (zero-init이라 이미 보수적)
- typical_jerk_step을 OOF train fold의 99-quantile로 정확히 계산해야 데이터 일관성

---

## 5. ⭐⭐ [Boundary-Guided Trajectory Prediction (2025)](https://arxiv.org/abs/2505.06740)

### 핵심
MSE 대신 **Huber loss + 가장 가까운 예측 trajectory에 hard assignment**. 이상치 처리 + mode collapse 방지.

### 대회 적용 — "Corrector Huber loss"

PB_0.6822 corrector의 회귀 loss를 MSE → Huber로 교체:

```python
# 현재 (추정)
loss = (delta - target) ** 2 * boundary_weight

# 변경
loss = F.huber_loss(delta, target, delta=0.005, reduction='none') * boundary_weight
```

`delta` 파라미터 (transition point) = 0.005 (= R_hit/2) 가 자연스러운 시작점.

### 예상 이득
**+0.001 ~ +0.002**

### 작업량
- 한 줄 변경, 학습 ×1 (검증용)

### 주의
- huber `delta` 너무 크면 MSE 거의 동일, 너무 작으면 L1 거의 동일
- {0.002, 0.005, 0.010} 그리드면 충분

---

## 6. ⭐ [PTNet — Physically Feasible Vehicle Trajectory Prediction (Girase et al., 2021)](https://arxiv.org/abs/2104.14679)

### 핵심
하나의 경로 (pure pursuit) 위에 **여러 가속도 프로파일**을 후보로 enumerate. 후보 공간을 *경로 × 가속*으로 reparameterize.

### 대회 적용 — "Path × acceleration grid"

현재 27 candidate을 *7 path × 4 accel = 28*로 재구성:

```python
# 7개 path basis (frenet 방향 7개)
paths = [
    (par=1.0, perp=0.0),    # 직진
    (par=0.9, perp=0.2),    # 약좌선
    (par=0.9, perp=-0.2),   # 약우선
    (par=0.7, perp=0.4),    # 큰 좌선
    (par=0.7, perp=-0.4),   # 큰 우선
    (par=1.1, perp=0.0),    # 가속 직진
    (par=0.5, perp=0.0),    # 감속 직진
]

# 4개 accel profile
accels = [-0.3, 0.0, 0.3, 0.6]  # constant decel, hold, mild accel, hard accel

# 28 후보 생성
for path in paths:
    for a in accels:
        candidate = generate_with(path, a)
```

후보 의미가 *path-acceleration 평면에 정렬*되어 selector 학습이 쉬워질 수 있음.

### 예상 이득
**+0.001 ~ +0.003** (oracle 유지 시)

### 작업량
- `make_candidates()` 전면 재작성 (2일)
- oracle hit rate 확인 필수 (27 후보 대비 떨어지면 무의미)

### 주의
- 큰 작업이므로 **다른 trick들 다 끝낸 후 시도**
- oracle이 떨어지면 path/accel 수 증가, 그래도 안 되면 27 그대로 유지

---

## 7. ⭐ [Empirical Bayes for Trajectory Representation (2022)](https://arxiv.org/abs/2211.01696)

### 핵심
Empirical Bayes prior의 **shrinkage 강도를 데이터에서 학습**.

### 대회 적용 — "Shrinkage grid search"

PB_0.6822 `candidate_regime_bias()` 의 `shrink=18.0` 고정값 → grid search:

```python
# 현재 코드 (cell 4, line 334)
def candidate_regime_bias(candidates, target, regimes, regime_count, shrink=18.0):
    ...
    alpha = float(np.sum(mask) / (np.sum(mask) + shrink))

# 변경: shrink ∈ {6, 12, 18, 30, 50, 100} grid search
# OOF에서 hit rate 최대화 픽
```

### 예상 이득
**+0.001** (existing 메커니즘의 하이퍼튜닝)

### 작업량
- 거의 0. 학습 ×6 (regime bias만 다시 계산, selector 재학습 불필요할 수도)

### 주의
- regime_prior_strength를 함께 sweep 안 하면 shrink 효과 묻힘
- shrink와 regime_prior_strength의 *2D grid*가 더 정확

---

## 통합 plan v2 — ROI 순서

| 순서 | 작업 | 출처 | 예상 이득 | 작업량 | 누적 |
|---|---|---|---|---|---|
| 0 | (baseline 재현) | — | 0 | 1일 | 0.6822 |
| 1 | 3-channel selector ensemble | **Learning-IMM 2025** | +0.003~0.006 | 1.5일 + 학습×3 | 0.685~0.690 |
| 2 | 5-seed ensemble | competition heuristic | +0.003~0.008 | 학습×5 | 0.688~0.696 |
| 3 | Z-rotation TTA | ideas.md §Supp.6 | +0.002~0.005 | 0.5일 | 0.690~0.700 |
| 4 | State-conditional anchor (α/β/γ tune) | **CoverNet 2020** | +0.002~0.005 | 1일 + 학습×27 | 0.692~0.704 |
| 5 | Multi-parse input (raw/SG/EMA) | **MTP 2022** | +0.002~0.004 | 0.5일 | 0.694~0.707 |
| 6 | Cap + σ sharpen (corrector) | PB_0.6822 hp | +0.002~0.005 | 학습×5 | 0.696~0.711 |
| 7 | Corrector Huber loss | **Boundary-Guided 2025** | +0.001~0.002 | 한 줄 | 0.697~0.713 |
| 8 | Physics conservation reg | **CPhy-ML 2024** | +0.001~0.003 | 0.5일 + 학습×4 | 0.698~0.715 |
| 9 | Shrinkage grid search | **EB Trajectory 2022** | +0.001 | 학습×6 | 0.699~0.716 |
| 10 | Cascade 2-stage corrector | engineering | +0.001~0.004 | 1일 | 0.700~0.719 |
| 11 | Hard-sample fine-tune | OHEM 2016 정신 | +0.001~0.003 | 0.5일 | 0.701~0.721 |
| 12 | Path × accel reparameterization | **PTNet 2021** | +0.001~0.003 | 2일 (risk) | 0.702~0.723 |

→ **3-week sprint로 LB 0.6822 → 0.70~0.71 도달이 현실적 시나리오.** 상한 0.72는 모든 trick이 잘 맞을 때.

---

## 적용 시 안전망

1. **변경 1개씩만**. 두 trick 동시 적용 후 망하면 인과 추적 불가
2. **OOF hit rate를 매 변경마다 기록**. 매 commit msg에 `decision-note: OOF X.XXXX → Y.YYYY` 박제
3. **Oracle hit rate 동시 추적**. 후보 set 변경 시 oracle 떨어지면 그 변경 자체가 무의미
4. **각 ablation을 별도 commit**. git checkout으로 즉시 복원 가능하게
5. **Seed 고정 + fold 고정**. 재현 가능하게

---

## Prior paper별 출처 표시 의무

각 trick을 코드에 적용할 때 commit message에 출처 paper 1-line 인용:

```
[c-NN] state-conditional anchor coefficients
  ref: CoverNet (Phan-Minh, CVPR 2020) — input-conditional dynamic anchors
  decision-note: OOF 0.6824 → 0.6849 (+0.0025, α=0.15 β=0.10 γ=0.20)
```

→ 나중에 paper로 발전시킬 때 *어느 trick이 어느 paper에서 왔는지* 자동 추적 가능.

---

## 한 줄 결론

**Top 3 (Learning-IMM 3-channel, CoverNet state-conditional, MTP multi-parse) 가 paper-derived ROI 가장 큼.** 이거 3개만 잘 얹어도 prior art를 *실질적으로* 활용했다고 말할 수 있다. 나머지 4~5개는 정밀 튜닝 단계의 작은 손익.
