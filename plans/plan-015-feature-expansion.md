---
plan_id: 015
version: 1
date: 2026-05-14 (Asia/Seoul)
status: draft (feature spec only — G-gate / experiment spec TBD)
based_on:
  - 014 (band=negative, best_stack 0.6425, oracle ceiling 0.8248, 회수율 5.4%)
scope: corrector input feature 확장 — 현 9D kinematic 의 표현력 부족이 plan-014 G3 5축 negative 의 root cause 신호. plan-015 STAGE 0 에서 feature 4개 (A/B 1순위 + C/D 2순위) 만 spec, 실험 G-gate 는 후속 v2 patch.
---

# plan-015 v1 — Feature Expansion (1, 2 순위 spec only)

## §0. 한 줄 목적

> **plan-014 corrector 의 input 표현력 부족 (oracle 0.82 vs measured 0.64, 회수율 5.4%) 을 직접 닫기 위해 9D → +4~7D feature 확장. 1순위 (A/B) 는 plan-014/005 의 measured evidence 가 직접 가리키는 누락 신호. 2순위 (C/D) 는 mechanically aligned, untested.**

---

## §0.5 Quick Reference

### G-gates

- G0: STAGE 0 feature spec (본 문서 §1)                  [DONE]
- G1+: 실험 G-gate                                        [TBD v2]

### commit chain (next-up)

- c1: plan-015 v1 draft — feature spec (A/B/C/D) 박제      [TODO]
- c2~: 실험 spec / pre-registration (v2 patch)            [TBD]

---

## §1. Feature 확장 후보 (1, 2 순위)

### 1순위 — 즉시 시도 (cheap, plan-014/005 evidence 직접 매핑)

#### A. F0 prior residual 직접 input ★

- **정의**: 11-step F0 prediction 좌표를 매 step encoder input 에 concat. per-step `(obs[t] − F0_pred[t])` 3D 추가 (혹은 F0_pred 자체 3D).
- **dim**: +3D (9D → 12D)
- **근거**: 현재 F0 는 corrector 의 *target basis* 로만 쓰이고 encoder 가 *자기가 보정해야 할 F0 prediction* 을 못 봄. plan-014 회수율 5.4% 의 가장 큰 누락 신호.
- **구현 위치 hint**: `src/pb_0_6822/plan014_paradigm.py` 의 feature 생성부 + F0 호출부 인접.

#### B. Frenet binormal axis 분리

- **정의**: 현 `perp_norm / speed` (normal + binormal 합쳐 1D) 을 *normal* / *binormal* magnitude 2D 로 분리.
- **dim**: +1D (9D → 10D, 단독) / A 와 합치면 13D
- **근거**: plan-005 진단 — binormal axis error 0.64cm vs parallel 4.51cm. plan-014 G2 oracle ceiling 도 Frenet-orthogonal codebook 이 최고 (0.8248). binormal direction = hidden lever. 코드공유-upgrade.md C010 (frenet-anisotropic-loss) 와 같은 진단축.

### 2순위 — 표현력 보강 (untested, mechanically aligned)

#### C. Multi-scale stride features

- **정의**: 9D feature 를 τ ∈ {1, 2, 3} stride 로 3번 계산해 concat (or BiGRU 3-stream).
- **dim**: 9D × 3 = +18D (or stream 형태)
- **근거**: mosquito wingbeat-aliased noise vs trajectory-level maneuver 의 시간 scale 분리. 11-frame 내 FFT 불가 (notes/new-ideas.md A.2) 이지만 multi-stride 는 가능.

#### D. Pairwise cross-step interaction

- **정의**: step t vs t-2 / t-4 의 cosine similarity, Δspeed, Δangle 등 explicit pairwise feature.
- **dim**: +2~6D
- **근거**: plan-014 G4 E7b (LastStep MLP) 가 BiGRU 대비 -0.005 → 시계열 가치 있지만 BiGRU 가 6-step 내 long-range pairwise pattern 다 못 잡을 가능성. explicit pairwise 가 보완.

---

## §2. Scope (명시적)

### §2.1 In-scope (본 v1)

| 항목 | 값 |
|---|---|
| 1순위 feature (A, B) spec | ✓ |
| 2순위 feature (C, D) spec | ✓ |

### §2.2 Out-of-scope (v1, v2 로 carry)

| 항목 | 이유 |
|---|---|
| G-gate / pre-registration | v2 spec patch 에서 박제 |
| corrector module 재구현 spec | v2 |
| 합격 기준 / band threshold | v2 |
| 3순위 이하 (snap, curvature rate, regime heatmap) | 회수율 marginal — plan-014 evidence 상 우선순위 낮음 |
| Negative evidence 후보 (FFT, path signatures, Neural ODE, Koopman) | notes/new-ideas.md A.2/B.3 에서 이미 negative |

---

## §3. 후속 (v2 spec patch 시 추가)

- A/B/C/D 4 feature 의 *순차 ablation* 또는 *bundled 비교* 결정
- baseline = plan-014 G5 best_stack (5-fold OOF 0.6425)
- 합격 기준 (예: Δ ≥ +0.005 over 0.6425)
- exp_id naming / registry append schema

---

## §N+4. 변경 이력

- v1 (2026-05-14): 1, 2 순위 feature (A/B/C/D) spec 박제. G-gate / 실험 spec 은 v2 carry.

---

## §N+5. 참조

- `plans/plan-014-plan012-failure-inversion.results.md` — band=negative, 회수율 5.4%, oracle 0.8248
- `plans/plan-013-plan004-framework-3lever-stacking.results.md` — LB 0.6381 join row 4
- `plans/plan-005-pb-0-6822-diagnostic.md` — binormal axis error 0.64cm evidence
- `notes/new-ideas.md` — A.2 (FFT N=11 fatal), B.3 (corrector 회수 한계 진단)
- `notes/코드공유-upgrade.md` — C010 frenet-anisotropic-loss / Idea 1 continuous regime
- `src/pb_0_6822/plan014_paradigm.py` — 현 9D feature 구현부
