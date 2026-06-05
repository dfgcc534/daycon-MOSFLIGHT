---
plan_id: 006
generated: 2026-05-12T11:13:37+09:00
finalized: 2026-05-12T11:26:49+09:00
status: all_complete — 시나리오 A 채택 (LB=0.6692 ≥ 0.6606 cutoff)
scenario: A
selected_priority: A1 (후보 다양화) > A2 (corrector 재설계)
---

# plan-007 후보 (≥ 2)

**시나리오 A 채택** (LB=0.6692, plan-005 통찰 LB 단위 입증). 후속 plan-007 의 우선순위 = **A1 (후보 다양화) → A2 (corrector 재설계)**. 시나리오 B 후보 (B1/B2) 는 보존만 (재발동 가능성 낮음).

---

## 시나리오 A — Variant E LB ≥ 0.6606 (plan-005 통찰 *입증*)

### 후보 A1 — 후보 다양화 (27 → 35+ cands)

- **근거 metric**:
  - plan-006 §2.3 worst regime 16/17/10 의 hit < 0.42 → 현재 27 후보가 high-speed/turn regime 을 충분히 커버 못함.
  - physics_bias top-5 모두 frenet 계열 — *speed extrapolation* 류만 hot. 다른 family (radial / ema / ballistic) 추가 시 lift 가능.
- **예상 ROI**:
  - worst regime 3개 (n ≈ 1256) 의 hit 를 0.30 → 0.50 으로 끌어올리면 전체 OOF +0.025pp.
  - LB lift 추정 +0.02 ~ +0.04.
- **작업 범위**:
  - `selector.CANDIDATES` 에 8 후보 추가 (예: `radial_par150_perp030`, `ballistic_drag10`, `ema_alpha07` 등).
  - corrector 재학습 (~10min), Variant E 재측정 + LB 1회.
- **선행 조건**:
  - plan-005 의 corrector full-fit infra 재사용. plan-004 selector 는 *변경 없음* (regime 무관).
  - 추가 후보의 physics_bias contribution 측정 (uniform 대비 +0.001 이상이어야 의미).

### 후보 A2 — Boundary corrector 재설계 (residual 가설 검증)

- **근거 metric**:
  - plan-006 §2.2 corrector 의 lift = +0.0274 (전체 lift 의 *주력*).
  - plan-005 STAGE 5 의 corrector 손실 분해 (`failure_b001`) 에서 residual 학습 영역 식별됨.
- **예상 ROI**:
  - corrector 재설계 (현재 TinyCorrectionNet → deeper / multi-task) 로 lift +0.01 가능.
  - LB lift 추정 +0.01 ~ +0.025.
- **작업 범위**:
  - `src/pb_0_6822/boundary.py` 의 TinyCorrectionNet 수정 (residual block 추가 / multi-head loss).
  - 5-fold 재학습 (~30min), OOF + LB 측정.
- **선행 조건**:
  - 현재 corrector 의 *failure mode* 분석 완료 (plan-005 `failure_b001.{json,md}` 참조).
  - Variant E 의 *uniform centroid 대비 +0.0004pp* 의미 — corrector lift 가 main → corrector 가 새 main target.

---

## 시나리오 B — Variant E LB < 0.6606 (plan-005 통찰 *반증*)

### 후보 B1 — GRU/regime 의 OOF↔LB gap 분석

- **근거 metric**:
  - plan-006 OOF Variant E = 0.6524, OOF full = 0.6599 → OOF gap = 0.0075.
  - 만약 LB Variant E ≪ 0.66 (즉 LB gap ≫ OOF gap), GRU/regime 이 *out-of-sample* 에서 더 큰 lift.
  - plan-005 의 component_contribution 측정 (OOF 만) 이 LB 와 *체계적* 괴리.
- **예상 ROI**:
  - 측정 자체는 lift 없음 — 진단 plan.
  - 결과 활용: 차후 LB-targeted optimization 의 우선순위 재정렬.
- **작업 범위**:
  - Variant A (GRU + physics) 의 LB 측정 1회 (재학습 없음 — plan-005 산출 재사용).
  - Variant B (physics + regime) 의 LB 측정 1회.
  - 3-way LB 비교 표 + OOF↔LB gap 박제.
- **선행 조건**:
  - DACON 일일 quota 5/day — 2 LB 제출 필요.
  - plan-005 의 Variant A/B 산출 (`analysis/plan-005/variant_A_no_regime/`) 가 submission.csv 까지 박제됐는지 확인.

### 후보 B2 — Variant E + minimal GRU/regime 의 hybrid LB 측정 (binary search)

- **근거 metric**:
  - 시나리오 B 가정 하 GRU/regime 의 LB lift 가 OOF lift 보다 큼.
  - 어디서부터 정보가 살아나는지 binary search: E (0 gates) → E+regime → E+regime+GRU(no fine-tune) → full.
- **예상 ROI**:
  - LB lift 의 source 식별 → 후속 plan 의 *주력 target* 결정.
- **작업 범위**:
  - 4 variant 의 LB 측정 (각각 재학습 없음 — plan-004/005 산출 재사용).
  - 결과 표 + gap 분해 박제.
- **선행 조건**:
  - DACON 일일 quota 5/day — 3 LB 제출 (Variant E 는 본 plan 결과 재사용).
  - 각 variant 의 submission.csv 미리 생성 가능 (재학습 0).

---

## 우선순위 (LB=0.6692 회수 후 결정 — 시나리오 A)

- **A1 후보 다양화** (1순위, 채택) — worst regime 16/17/10 의 hit < 0.42 가 LB 손실의 주원인 추정. 27 → 35 후보로 high-speed/turn regime 커버. ROI 명확, 재학습 1회.
- **A2 corrector 재설계** (2순위) — corrector lift +0.0274 (전체 lift 의 주력) → corrector 가 새 main target. residual block 추가 / multi-task loss.
- (보존) B1/B2 — 시나리오 B 미발동, but `gap diff = -0.0039` 가 noise 인지 systemic 인지 검증하려면 미래 plan 에서 Variant A LB 1회 추가 측정 권장.

→ plan-007 = **A1 (후보 다양화)** 채택.
