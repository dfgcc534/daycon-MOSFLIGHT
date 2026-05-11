# 모기 궤적 예측 대회: 아이디어 정리

> LiDAR로 관측한 모기의 11개 시점(40ms 간격) 3D 좌표 시퀀스를 기반으로 +80ms 후 위치를 예측하여 Hit Rate를 최대화하는 과제에 대한 모델링 아이디어 정리.

---

## 🎯 Main Model (핵심 예측 모델 후보)

궤적 자체를 직접 예측하는 메인 엔진 역할입니다.

### 1. 딥러닝 기반 시퀀스 예측 모델
- **Attention 기반 Sequence Model (Social-LSTM 계열)**
  - 11개라는 짧은 시퀀스 특성상 마지막 2~3개 시점(0ms에 가까운 시점)에 높은 가중치를 부여
  - 비선형성이 강한 궤적의 시간적 의존성 학습
  - +80ms라는 짧은 예측 구간에서 최근 관성이 가장 강한 신호

### 2. 확률 분포 기반 예측 (GMM Output)
- 단일 좌표 $(x, y, z)$가 아닌 **존재 확률 분포**로 예측 후 중심점 제출
- Hit Rate 지표(반경 내 명중) 특성상 분포 형태로 다루는 것이 자연스러움
- 레비 비행의 Heavy-tail 특성을 분포 출력으로 흡수 가능

### 3. 손실 함수 설계
- **Smooth L1 (Huber) Loss**: 이상치(급격한 방향 전환)에 덜 민감 → 안정적 학습
- 단순 MSE는 Heavy-tail 데이터에서 위험

### 4. 1D-CNN (Temporal Convolutional Network, TCN)
- **개념**: GRU/LSTM 대신 1차원 합성곱 필터로 시계열을 훑는 방식
- **추천 이유**:
  - 11개 시점은 RNN 계열을 쓰기엔 너무 짧음
  - 1D-CNN은 국소적 패턴(갑작스러운 덜컹거림, 미세한 방향 전환)을 필터로 빠르게 포착
  - 노이즈에 강하고 학습 속도가 압도적으로 빠름

### 5. KNN 궤적 매칭 (Non-parametric)
- **개념**: 딥러닝을 쓰지 않는 데이터 주도형 방식. Train 데이터에서 현재 Test 궤적 흐름과 가장 유사한 과거 비행 패턴 5~10개를 찾아 정답을 평균
- **추천 이유**:
  - 모기 비행이 몇 가지 전형적 패턴(직진, 급회전, 지그재그)으로 군집화될 가능성
  - 복잡한 수식 없이 직관적으로 Hit-Rate를 높일 수 있는 다크호스
- **⚠️ 주의**: 시퀀스 간 거리 계산 전, 마지막 시점을 (0,0,0)으로 **평행이동 정규화** 필수

### 6. 비행 행동 분류 기반 모델 분기 (Mixture of Experts) ⭐️
- **개념**: 단 하나의 완벽한 모델을 찾는 대신, **궤적 형태에 따라 전문 모델을 배정**하는 Routing 전략
- **논문 배경**: 모기 궤적(Track segments)에서 피처를 추출해 ANN으로 비행 패턴/클래스(암컷 vs 수컷, 휴식 vs 짝짓기 비행 등)를 **88% 이상 정확도**로 분류한 선행 연구 존재
- **대회 적용**:
  - 11개 시점을 보고 현재 모기가 '직진 중'인지, '급회전 중'인지, '제자리 비행(Hovering/Jittering) 중'인지 먼저 분류
  - K-Means 같은 클러스터링으로 궤적을 3~4개 패턴으로 분할 후, 각 패턴에 특화된 모델로 예측 분기(Routing)
  - 직진 패턴 → 단순 직선 외삽에 가중치 크게
  - 급회전 패턴 → GRU/1D-CNN 모델이 예측
- **효과**: 전체 Hit Rate를 안정적으로 상승. 단일 모델이 모든 패턴을 커버하려 할 때의 평균값 회귀 문제를 회피

### 7. 포인트 클라우드 접근법 (PointNet 계열) ⭐️
- **개념**: 데이터를 '시간 순서'가 아닌 **'공간적 형태'**로 바라보는 패러다임 전환
- **논문 배경**: 자율주행/LiDAR 3D 객체 탐지 분야에서 **PointPillars** 같은 아키텍처가 3D 환경 정보(거리, 속도)를 효율적으로 추출
- **대회 적용**:
  - 11개의 3D 좌표를 하나의 **작은 점군(Point Cloud)** 으로 모델에 입력
  - PointNet 구조는 시간적 순서에 얽매이지 않고, 11개 점이 3D 공간에서 이루는 **기하학적 곡선 형태** 전체를 파악하여 다음 점 추론
- **추천 이유**:
  - RNN/1D-CNN이 시계열이라는 고정관념에 묶여 있을 때의 대안
  - 노이즈가 심한 LiDAR 센서 데이터에서 강력한 방법론
  - 시퀀스 길이가 짧아도(11개) 기하 구조 학습에 유리

---

## 🛠️ Supportive (보조 기법 / 피처 / 분석 도구)

메인 모델의 입력을 풍부하게 하거나 일반화 성능을 끌어올리는 역할입니다.

### 1. 물리 기반 피처 엔지니어링 (Physics-Aware Features)
모기는 질량이 가벼워 공기 저항·가속도 변화가 즉각적이므로 고차 미분 피처가 의미 있음.

- **JERK (가속도 변화량)**: 방향 전환 직전의 전조 포착
- **Curvature (곡률)**: 직진 vs 회전 상태 구분
- **속도/가속도 벡터**: 기본이지만 필수

### 2. Lévy Flight 통계 모델링 (보조 가설)
- 참고: *"Characterizing the search flights of mosquitoes"* (2015, Scientific Reports)
- Aedes aegypti가 레비 비행 특성을 보임이 입증됨
- **활용**: 갑작스러운 방향 전환(Outlier) 가능성을 모델이 인지하도록 사전 분포 또는 분포 출력의 형태로 반영
- 평균값 회귀의 위험성을 보완하는 이론적 근거

### 3. LiDAR 도메인 특화 피처
- 참고: *"Laser radar for analysis of insect flight"* (2014, Applied Optics) — Lund University Brydegaard 팀
- **Wing-beat / Oscillation 추출**: 11개 좌표의 미세 떨림 패턴
- 모기의 종/상태를 암시 → 궤적 패턴의 잠재 클래스로 활용 가능

### 4. 도메인 일반화 (Domain Generalization)
실내/야외 등 환경이 섞인 데이터에 대응.

- **입력 정규화 (Standardization)**: 환경 간 스케일 차이 제거
- **Invariant Risk Minimization (IRM)** 또는 **Domain-Adversarial Training**
- **보조 태스크**: 11개 시점의 분산/속도 통계로 '환경' 클러스터링 후 멀티태스크 학습

### 5. 잔차 예측 (Residual Prediction) ⭐️⭐️⭐️
- **개념**: 모델이 미래 좌표를 맨땅에서 예측하지 않음. **"단순 직선 외삽 예측값"과 "실제 정답"의 오차($\Delta x, \Delta y, \Delta z$)** 만 학습
- **추천 이유**:
  - 직선 외삽 성능이 좋다는 것 = 궤적의 80~90%는 관성대로 움직인다는 뜻
  - 모델은 "직선 경로에서 얼마나 벗어날지(바람·날갯짓 미세 조정)"만 집중 학습
  - 출력 분포의 분산이 크게 줄어 성능이 비약적으로 상승

### 6. Test-Time Augmentation (TTA)
- **개념**: 추론 시 입력을 여러 각도로 변형해 모델에 통과시킨 뒤, 결과를 원래 좌표계로 되돌려 평균
- **추천 이유**:
  - Sensor-local 3D 좌표계의 물리적 대칭성 활용
  - **Z축(상향, 중력)**: 건드리면 안 됨
  - **X-Y 평면 회전 / Y축 Flip**: 물리적으로 동일한 비행 패턴 → 안전한 증강
- **효과**: 모델이 노이즈에 휘둘려 엉뚱한 방향으로 튀는 것을 강력히 억제

### 7. 시점별 가중치 부여 (EMA, Exponential Moving Average)
- **개념**: 11개 점을 동일하게 믿지 않음. -400ms의 오래된 데이터보다 -40ms, 0ms의 최근 데이터가 +80ms 예측에 압도적으로 중요
- **적용**: 속도·가속도 계산 시 최근 시점일수록 가중치를 **기하급수적으로 높게** 부여
- **시너지**: 잔차 예측의 베이스라인(직선 외삽)을 EMA로 구하면 베이스라인 자체의 정확도가 상승

### 8. 곤충 운동학(Kinematics) 기반 고차 미분 피처 ⭐️
- **개념**: 단순 속도/가속도를 넘어, **모기 특유의 급격한 방향 전환 시그널**을 잡아내는 피처 엔지니어링
- **논문 배경**: 3D 비디오그래피로 모기의 몸통/날개 운동학(Kinematics)을 고해상도 추적한 연구에서 **단순 위치 좌표를 넘어선 미세 움직임 변화 캡처**가 핵심으로 다뤄짐
- **대회 적용**:
  - **Jerk (3차 미분, 가속도 변화량)**: 방향 전환 직전 전조 시그널
  - **Curvature (곡률, 궤적 꼬임 정도)**: 직진/회전 상태 판별
  - 모기는 관성이 거의 없어 순식간에 방향을 바꿈 → 11개 시점 안에서 **Jerk 값이 갑자기 튀는 순간**을 모델이 인지하면 +80ms 잔차 예측 정밀도 상승
- **차별점**: §Supportive 1번(기본 JERK/Curvature) 대비, **3차 미분 + 곡률 + Jerk-spike 탐지**까지 확장한 정밀 피처 세트

---

## 📊 정리 표

| 구분 | 항목 | 역할 |
|------|------|------|
| Main | Attention 시퀀스 모델 | 궤적 직접 예측 |
| Main | GMM 분포 출력 | Hit Rate 지표 최적화 |
| Main | Huber Loss | 이상치 강건성 |
| Main | 1D-CNN (TCN) | 짧은 시퀀스 국소 패턴 추출 |
| Main | KNN 궤적 매칭 | Non-parametric 다크호스 |
| Main | **MoE 행동 분류 분기 ⭐** | 패턴별 전문 모델 Routing |
| Main | **PointNet 포인트 클라우드 ⭐** | 시계열→공간 형태 패러다임 전환 |
| Supportive | JERK / Curvature | 입력 피처 보강 |
| Supportive | Lévy Flight 가설 | 분포 끝단 관리 근거 |
| Supportive | Wing-beat Oscillation | LiDAR 특화 피처 |
| Supportive | IRM / 정규화 | 도메인 일반화 |
| Supportive | **잔차 예측 ⭐** | 베이스라인 + 오차만 학습 |
| Supportive | TTA (회전·Flip) | 추론 안정성 강화 |
| Supportive | EMA 시점 가중치 | 최근 시점 중요도 반영 |
| Supportive | **Kinematics 고차 미분 ⭐** | Jerk·Curvature·Spike 탐지 |

---

## 📚 참고 문헌

1. **Lévy Flight in Mosquitoes**
   *"Characterizing the search flights of mosquitoes"* (2015, Scientific Reports)
2. **LiDAR Insect Tracking**
   *"Laser radar for analysis of insect flight"* (2014, Applied Optics) — Lund University, Brydegaard et al.
3. **Trajectory Prediction (DL)**
   *"Social-LSTM: Toward Multi-agent Trajectory Prediction in Crowded Spaces"*
4. **Domain Generalization**
   Invariant Risk Minimization (IRM), Domain-Adversarial Training
5. **Mosquito Behavior Classification (ANN)**
   모기 궤적 segment 피처 → ANN 분류기로 비행 패턴/성별/행동 클래스 88%+ 정확도 분류
6. **Insect Kinematics (3D Videography)**
   모기 몸통/날개 운동학 고해상도 추적 — 위치 좌표를 넘어선 미세 움직임 캡처의 중요성
7. **3D Point Cloud / LiDAR DL**
   *"PointPillars: Fast Encoders for Object Detection from Point Clouds"* — 자율주행/LiDAR 3D 객체 탐지
