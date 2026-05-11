# PB_0.6822 Upgrade Idea: 이산 regime 표 → 연속 heatmap

## 현재 구조 (PB_0.6822 코드공유.ipynb)

```
1. (speed, curvature, speed_slope) feature 추출 (최근 4~6 프레임)
2. 분위수 컷으로 18 regime 분류 (3 × 3 × 2)
3. [18 regime × 27 후보] 표 계산 — empirical Bayes shrinkage 포함
4. 각 sample → 자기 regime의 행을 lookup → 27차원 bias
5. 최종 점수 = gru_logits + 0.65 × physics_bias + 0.45 × regime_bias[regime]
```

**단점**:
- 분위수 컷 경계에서 *discrete jump* (sample 두 개가 비슷한 feature인데 다른 regime이면 bias 점프)
- Sample 분포가 불균등할 때 작은 regime이 noisy → shrinkage로 보완하지만 *수동 hyperparameter* (`shrink=18`)
- Quantile cutoff, regime count, shrink coefficient 등 *여러 hyperparameter*
- Feature 수가 늘면 격자 셀 폭발 (현재 3×3×2=18, 4축이면 3×3×2×3=54, 5축이면 162…)

---

## 제안: 연속 heatmap (kernel-weighted bias)

이산 표 대신 **(speed, curvature, slope) 연속 공간 위의 부드러운 함수**로 bias를 표현.

### 핵심 수식

```
bias(x, c) = [Σ_i K(x - x_i) · hit_metric_i(c) + λ · global_metric(c)]
           / [Σ_i K(x - x_i) + λ]
```

- `x`: query sample의 feature (speed, curve, slope)
- `x_i`: train sample들의 feature
- `K`: Gaussian kernel — `K(d) = exp(-(d/bandwidth)²)`
- `λ`: pseudo-count (sparse 영역에서 global prior로 후퇴)
- `hit_metric`: `log(hit + 1e-4) - 18 × mean_distance`

마지막에 centering 적용해 framework 일관성 유지:
```
bias = bias - bias.mean(axis=candidates)
```

### 왜 sample 개수가 자동 처리되는가

| Feature 공간 영역 | Σ K (kernel sum) | 효과 |
|---|---|---|
| Sample 밀집 | Σ K >> λ | local 통계 dominant |
| Sample 희소 | Σ K ~ λ | local과 global blend |
| 데이터 없음 | Σ K ≈ 0 | global prior dominant |

→ **현재 코드의 empirical Bayes shrinkage가 kernel weighting 안에 자연 흡수**. `α = n/(n+18)` 식 별도로 안 필요.

---

## 장점

| 측면 | 이산 표 | 연속 heatmap |
|---|---|---|
| Quantization artifact | bin 경계 jump | 없음 (연속 부드러움) |
| Sample density 처리 | 명시적 shrinkage | **자동** (kernel weighting) |
| Hyperparameter | quantile cuts + shrink | **bandwidth 하나만** |
| 차원 확장성 | 격자 셀 폭발 | 자연 (feature 수 ↑ 시 부드럽게 대응) |
| Paper figure | 18-row 이산 grid | **3D 연속 surface** (paper 머니샷급) |
| 코드 복잡도 | regime fit/assign/shrink 3 함수 | kernel 1 함수 |

## 단점

| 측면 | 영향 |
|---|---|
| 추론 비용 | O(1) lookup → O(N_train) kernel sum |
| Bandwidth 선택 | 새 hyperparameter (CV로 1~2값 시도) |
| 차원의 저주 | feature ≥5D 가면 KDE 신중 (현재 3D는 안전) |
| 안정성 | bandwidth 너무 작으면 high variance |

---

## 구현 스케치

```python
def kernel_regime_bias(
    train_features,    # [N, 3]  in (speed, curve, slope), normalized
    train_cands,       # [N, 27, 3]
    train_y,           # [N, 3]
    query_features,    # [M, 3]
    bandwidth: float = 0.3,
    lam: float = 20.0,
):
    err = np.linalg.norm(train_cands - train_y[:, None, :], axis=2)  # [N, 27]
    hit_metric = np.log((err <= R_HIT).astype(float) + 1e-4) - 18.0 * err

    global_metric = hit_metric.mean(axis=0)  # [27]

    d = np.linalg.norm(query_features[:, None, :] - train_features[None, :, :], axis=2)  # [M, N]
    w = np.exp(-(d / bandwidth) ** 2)

    numer = w @ hit_metric + lam * global_metric[None, :]
    denom = w.sum(axis=1, keepdims=True) + lam

    bias = numer / denom
    bias = bias - bias.mean(axis=1, keepdims=True)
    return bias.astype(np.float32)
```

→ 기존 `fit_regime_bins` + `assign_regimes` + `candidate_regime_bias` 세 함수가 **이 한 함수로 통합**.

---

## 추론 비용 완화 (선택)

`O(N_train)` 부담을 줄이려면:

1. **Train sample을 대표점으로 압축**: k-means centroid K=200 정도로 축약 → O(K) per query
2. **KD-tree로 nearest neighbor만 kernel 계산**: 멀리 있는 sample은 자동 무시
3. **Precompute grid + 추론 시 보간**: query는 grid lookup + bilinear, train 시점 한 번만 grid 계산

가장 단순한 시작점은 *그대로* 돌려보고 속도 문제 있을 때 1~3번 적용.

---

## 시각화 (paper figure 후보)

3D feature 공간이라 한 장에 못 그리니 **2D marginal heatmap 27장** (후보별):

```
                후보 c의 P(hit) heatmap
       curvature
          ▲
       높음│ ░░░▒▒▒▓▓
          │ ░░▒▒▒▓▓▓     <- (speed, curve) 평면
          │ ░▒▒▒▒▓▓▓        slope는 marginal/평균
          │ ▒▒▒▒▓▓░░
       낮음│ ▒▒▓▓▓░░░
          └───────────────► speed
            느림        빠름
```

또는 **family별 통합** (base, acc, frenet, turn, jerk, latency 6장)으로 압축.

권장 3-set:
1. **Raw hit rate heatmap** (centering 전, 절대 비교)
2. **Centered bias heatmap** (framework이 실제 쓰는 값)
3. **Sample density heatmap** (Σ K of train samples)

---

## 기대 효과

| 항목 | 추정 |
|---|---|
| Hit rate (모기 단일) | +0.0 ~ +0.5 pp (marginal) |
| 코드 단순화 | regime fit/assign/shrinkage 제거 |
| Hyperparameter 수 | 줄어듦 (quantile + shrink → bandwidth) |
| **Paper figure** | **매우 강함** — 연속 surface가 핵심 contribution figure |
| **Drone 확장 시** | **결정적** — feature 수 늘면 격자보다 훨씬 깔끔 |

---

## 실행 체크리스트

- [ ] `kernel_regime_bias` 구현
- [ ] 기존 `regime_table` 호출부를 새 함수로 교체 (학습/추론 양쪽)
- [ ] Bandwidth ablation (0.2, 0.3, 0.4)
- [ ] λ ablation (10, 20, 50)
- [ ] Hit rate 1fold 비교: 기존 vs 제안
- [ ] 시각화 3-set 생성 (raw / centered / density)
- [ ] 추론 속도 측정 → 필요 시 KD-tree 또는 centroid 압축
- [ ] Drone domain 이식 시점에 재검증 (feature 수 늘었을 때 진가 발휘)

---

## 한 줄 정리

**이산 18-regime 표 + 명시적 shrinkage → 연속 (speed, curve, slope) 공간의 kernel-weighted heatmap.** Sample density가 *kernel weighting 안에 자연 흡수*되어 shrinkage 코드 제거, hyperparameter 하나(bandwidth)로 축소, *연속 surface heatmap*이라는 paper-grade visualization 공짜로 따라옴. Hit rate 이득은 모기 단일 도메인에선 marginal하지만 **drone 확장 시 feature 수 증가에 자연 대응**한다는 점에서 *구조적 가치*가 큼.

---

# PB_0.6822 Upgrade Idea 2: 사람이 손으로 만든 27 후보 → 데이터로 학습한 N 후보

## 현재 구조

```python
CANDIDATES = [
    CandidateSpec("p0_2d1",        d1=2.00, par=0.00, perp=0.00, d2=0.0, jerk=0.0, time_scale=1.00),
    CandidateSpec("acc_2d1_050",   d1=2.00, par=0.50, perp=0.50, ...),
    CandidateSpec("frenet_best",   d1=1.98, par=0.96, perp=-0.08, ...),
    ...                                                              # 27개
]                                                                    ↑
                                                              사람이 직관으로 결정
```

→ 6차원 계수 `(d1, d2, par, perp, jerk, time_scale)` 의 *27개 조합을 사람이 손으로 선택*. Candidate spec 변경은 코드 수정 + 재실행이 필요한 *manual loop*.

**단점**:
- 6차원 계수 공간을 인간 직관으로 탐색 불가 (효율적 조합을 빠뜨릴 가능성)
- 27이라는 숫자가 *부드러운 hyperparameter* (늘리기 어렵고 줄이기 모호)
- 새 도메인(drone)으로 옮기면 *처음부터 손으로 재설계* 필요
- Train 데이터 분포가 바뀌면 (예: 추가 데이터 들어오면) 손 후보가 *서서히 stale*

---

## 제안: 데이터 기반 candidate 학습

train 데이터로부터 **N개의 (d1, d2, par, perp, jerk, time_scale) 조합을 자동 학습**. 학습 후 N개 LEARNED_CANDIDATES가 *고정 상수*로 박혀, **추론 시에는 현재 27 손 후보와 완전히 동일하게 작동**.

```python
# 학습 단계 (train time만)
LEARNED_CANDIDATES = optimize_coefficients_from_data(train_data, n=N)

# 추론 단계 — 현재 코드와 같은 형태
candidates = make_candidates(x, end_idx, formulas=LEARNED_CANDIDATES)
```

→ **drop-in replacement**. `make_candidates` 호출부 외에는 코드 변경 0.

### Framework 철학과 정합

| 측면 | 영향 |
|---|---|
| 학습 시 | DL 사용 (계수 선택만) |
| **추론 시** | **고정 계수, DL 안 씀** |
| 후보 출력 | 결정적, 물리적 해석 가능 |
| 노이즈 외우기 risk | 매우 낮음 (학습 변수가 6×N개 뿐) |

→ "DL은 추론에 끼어들지 않는다" 라는 원 framework 철학 보존.

---

## 4가지 학습 방법

### A. K-means in coefficient space (가장 단순)

```python
# 1. 각 train sample에 대해 *오차 최소인 최적 계수* 분석적으로 풀어냄
optimal_coefs = solve_per_sample_best_coefs(train_x, train_y)   # [N_train, 6]

# 2. 6차원 공간에서 K-means
from sklearn.cluster import KMeans
km = KMeans(n_clusters=N).fit(optimal_coefs)

# 3. Centroid가 학습된 후보
LEARNED_CANDIDATES = [CandidateSpec(f"learned_{i}", *c) for i, c in enumerate(km.cluster_centers_)]
```

**장점**: 미분 불필요, 간단, 빠름
**단점**: per-sample 최적 계수 풀이가 closed-form이 아닐 수 있음

### B. Differentiable coefficient optimization (가장 강력)

```python
class CandidateLearner(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.coefs = nn.Parameter(torch.randn(n, 6) * 0.3)   # 학습됨

    def make_candidates(self, x, end_idx):
        # 현재 make_candidates와 동일한 수식, 단 coefs가 학습 가능
        ...

# 학습 loop
for batch in loader:
    cands = learner.make_candidates(batch.x, batch.end_idx)   # [B, N, 3]
    err = (cands - batch.y[:, None, :]).norm(dim=2)            # [B, N]
    # Soft min을 통해 미분 가능하게
    soft_min_err = -torch.logsumexp(-err / tau, dim=1).mean()
    soft_min_err.backward()
    opt.step()
```

**장점**: 직접 hit rate 최적화, end-to-end
**단점**: tau hyperparameter, mode collapse 위험

### C. Anchor learning (MultiPath/CoverNet 스타일)

```python
# 1. Train trajectory의 +80ms 좌표를 trajectory-relative 좌표계로 정규화
relative_targets = normalize_to_local_frame(train_y, train_x)

# 2. K-means → N anchor 좌표
anchors = KMeans(N).fit(relative_targets).cluster_centers_

# 3. Anchor → 6-coef 역변환 (선형 회귀)
LEARNED_CANDIDATES = inverse_solve_coefs_from_anchors(anchors)
```

**장점**: 자율주행에서 입증된 표준, 안정적
**단점**: 좌표 → 계수 역변환이 underdetermined일 수 있음

### D. Discrete combinatorial optimization

```python
# 1. 6-coef를 grid에 두기 (예: 각 축 10단계)
candidate_pool = generate_grid(d1=10, par=10, perp=10, ...)   # 10^6 후보

# 2. Greedy로 N개 선택 — oracle hit rate 최대화
selected = []
for _ in range(N):
    best = max(candidate_pool, key=lambda c: oracle_hit_with(selected + [c], train_y))
    selected.append(best)
```

**장점**: 명시적 oracle 최적화, 해석 가능
**단점**: 계산 비싸다, grid 해상도 trade-off

---

## 장점

| 측면 | 손 후보 (27) | 학습 후보 (N) |
|---|---|---|
| 6차원 공간 탐색 | 인간 직관 한계 | 데이터 기반 자동 |
| Oracle hit rate | baseline | 같은 N에서 ↑ 가능성 |
| N 조정 | 어려움 (코드 수정) | hyperparameter grid search |
| Domain transfer | 손으로 재설계 | 재학습만 |
| 새 데이터 들어오면 | 손 후보 stale | 재학습 |
| Paper contribution | 평범 | **"data-driven physics anchor learning"** |

## 단점

| 측면 | 영향 |
|---|---|
| **Mode collapse** | N개가 비슷한 곳에 모일 위험. Diversity regularization 필요 |
| **Train/val 누수** | 계수 학습이 train만 보도록 CV discipline 엄수 |
| **N 선택** | 새 hyperparameter (grid search) |
| **해석 가능성** | 학습된 계수는 "이름이 없음". 후처리로 family 클러스터링 후 라벨 부여 |
| **Oracle gap이 너무 작아질 위험** | 학습 후보가 너무 oracle에 가까우면 selector 학습이 단순화되어 generalization ↓. 적당한 diversity 유지 필요 |

---

## 구현 스케치 (Method B - Differentiable)

```python
import torch
import torch.nn as nn
import math

class LearnableCandidates(nn.Module):
    """N개의 (d1, d2, par, perp, jerk, time_scale)를 학습 가능 파라미터로 두는 모듈."""

    def __init__(self, n: int):
        super().__init__()
        # 초기화: 현재 27 손 후보 근처에서 약간의 jitter
        init = torch.tensor([
            [spec.d1, spec.d2, spec.par, spec.perp, spec.jerk, spec.time_scale]
            for spec in base.CANDIDATES[:n]
        ], dtype=torch.float32)
        if init.shape[0] < n:
            extra = init[:1].repeat(n - init.shape[0], 1) + torch.randn(n - init.shape[0], 6) * 0.1
            init = torch.cat([init, extra], dim=0)
        self.coefs = nn.Parameter(init)
        self.n = n

    def forward(self, p0, d1, d2, acc_par, acc_perp, jerk, horizon: int = 2):
        # current make_candidates와 동일 수식, 단 self.coefs가 학습됨
        v_scale = horizon / 2.0
        acc_scale = (horizon / 2.0) ** 2
        cands = []
        for i in range(self.n):
            c = self.coefs[i]   # [6]: d1, d2, par, perp, jerk, time_scale
            ts = c[5]
            cands.append(
                p0
                + c[0] * v_scale * ts * d1
                + c[1] * v_scale * ts * d2
                + c[2] * acc_scale * (ts ** 2) * acc_par
                + c[3] * acc_scale * (ts ** 2) * acc_perp
                + c[4] * acc_scale * (ts ** 2) * jerk
            )
        return torch.stack(cands, dim=1)   # [B, N, 3]


def train_candidate_learner(model, train_loader, opt, tau: float = 0.005,
                            diversity_weight: float = 0.1, epochs: int = 50):
    for epoch in range(epochs):
        for batch in train_loader:
            cands = model(batch.p0, batch.d1, batch.d2, batch.acc_par, batch.acc_perp, batch.jerk)
            err = (cands - batch.y.unsqueeze(1)).norm(dim=2)   # [B, N]

            # (1) Soft-min loss: 가장 가까운 후보의 오차 최소화
            soft_min = -torch.logsumexp(-err / tau, dim=1).mean()

            # (2) Diversity regularization: 계수 간 거리 최대화
            coef_dist = torch.cdist(model.coefs, model.coefs)
            diversity_loss = -coef_dist[~torch.eye(model.n, dtype=bool)].mean()

            loss = soft_min + diversity_weight * diversity_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
```

학습 끝나면 `model.coefs.detach().cpu().numpy()` 를 `CandidateSpec` 리스트로 변환해 박음.

---

## Idea 1과의 관계 — 직교성

두 upgrade가 *파이프라인의 서로 다른 단계*를 건드림:

| 질문 | 단계 | 현재 | 제안 |
|---|---|---|---|
| 어떤 후보를 만들 것인가 | Candidate generation | 손 27개 | **학습 N개** (Idea 2) |
| 각 후보를 regime별로 어떻게 평가할 것인가 | Scoring (regime 부분) | 18×27 이산 표 | **연속 heatmap** (Idea 1) |

→ **독립 적용 가능 → 4가지 ablation 조합**:

| 후보 | Regime | 명칭 |
|---|---|---|
| 손 27 | 이산 18 | baseline (현재) |
| 손 27 | 연속 heatmap | Idea 1 단독 |
| 학습 N | 이산 18 | Idea 2 단독 |
| **학습 N** | **연속 heatmap** | **둘 다 (최대 변경)** |

각각 단독 ablation으로 **두 contribution의 기여를 분리 측정** 가능 → paper structure에 깔끔.

---

## 잠재적 함정

| 함정 | 대응 |
|---|---|
| Mode collapse | Diversity regularization (`diversity_loss`), 또는 furthest-first init |
| Oracle gap 소실 | N을 적당히 (모기는 20~40 추천), oracle hit이 100% 가까이 가지 않도록 |
| Train 과적합 | CV로 N과 hyperparameter 결정, val hit rate 모니터링 |
| 학습 후보의 *물리적 해석 손실* | 학습 후 6-coef 공간에서 K-means → family 라벨 부여 (자동 family 분류) |
| Latency family 같은 *특수 구조* 누락 | 초기화에 손 후보 일부를 포함 + frozen subset 옵션 |

---

## 기대 효과

| 항목 | 추정 |
|---|---|
| Oracle hit rate | 같은 N에서 손 후보 대비 **+1~3 pp** 가능 |
| Selector hit rate | **+0.5~1.5 pp** (oracle 상승의 일부 흡수) |
| 코드 단순화 | 손 후보 수정 loop 제거 → 학습 한 번 |
| Hyperparameter | N, tau, diversity_weight 추가 (3개) |
| **Domain transfer 비용** | **결정적 ↓** — drone으로 옮길 때 *재학습만* 필요 |
| **Paper contribution** | "Data-driven physics candidate discovery" — MultiPath/CoverNet의 *coordinate anchor*를 *physics formula anchor*로 변환한 새 angle |

---

## 실행 체크리스트

- [ ] Method A (K-means) 먼저 — 가장 적은 구현 비용으로 *학습 후보가 손 후보보다 나은지* 빠른 검증
- [ ] Method A 결과가 promising 하면 Method B (differentiable) 구현
- [ ] N ablation (10, 20, 27, 40, 60)
- [ ] Diversity weight ablation (0.0, 0.05, 0.1, 0.3)
- [ ] Tau ablation (0.003, 0.005, 0.01)
- [ ] Initialization 비교: random vs 손 후보 seed
- [ ] Frozen subset 실험: latency family를 frozen으로 두고 나머지만 학습
- [ ] 학습된 계수의 6-coef 시각화 (PCA 2D plot, family 클러스터링)
- [ ] **Idea 1과 조합 ablation 4가지** (orthogonal contribution 입증)
- [ ] Drone domain 이식 시 *재학습 비용 측정*

---

## 한 줄 정리

**사람이 6차원 계수 공간을 직관으로 탐색해 27개를 손으로 고른 것을, *train 데이터로 N개를 자동 학습*으로 대체.** 학습 후 계수는 *고정 상수*로 박혀 추론 시 DL 추가 비용 0, framework 철학(DL이 추론에 안 끼어듦)도 보존. Idea 1(연속 heatmap)과 *파이프라인의 다른 단계*를 건드리므로 **독립적/직교적으로 적용 가능**, 4가지 ablation 조합으로 각 contribution을 분리 입증할 수 있어 paper structure에도 깔끔. 모기 단일에선 hit rate 이득이 중간 정도지만 **domain transfer (drone) 시 *재학습만으로 후보 재설계*가 끝난다는 점에서 구조적 가치가 큼**.

---

# 두 Idea의 통합 시각화

```
Trajectory (11 또는 5 프레임 + sliding window 옵션)
   │
   ▼
[STAGE 1: 후보 생성]                   ← Idea 2 (학습된 N 물리식)
   ┌──────────────────────────┐
   │ LEARNED_CANDIDATES[N]    │
   │  (train 시 학습됨)        │
   │  추론 시 고정             │
   └──────────────────────────┘
   → N개 후보 좌표 [B, N, 3]
   │
   ▼
[STAGE 2: 점수 매김]
   ├─ physics_bias × 0.65  (전체 평균, 고정)
   ├─ ★ 연속 heatmap regime bias × 0.45  ← Idea 1 (kernel-weighted)
   └─ gru_logits (attn_gru per-sample)
   → 최종 score [B, N]
   │
   ▼
[STAGE 3: Tiny correction]              (변경 없음)
   Boundary MLP delta 적용
   │
   ▼
[STAGE 4: 최종 선택]                    (변경 없음)
   margin gate → argmax 또는 soft
```

→ **두 변경 모두 *기존 단계의 한 함수 교체*** 로 끝나고, 다른 단계는 그대로 사용. 점진적 도입 + 분리 ablation에 매우 친화적인 구조.
