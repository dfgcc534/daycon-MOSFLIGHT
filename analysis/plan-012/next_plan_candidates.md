# plan-013 후보 (plan-012 G_final 산출, 2026-05-13)

plan-012 5-fold OOF = **0.6340** (< 0.65) → plan §10.2 의 "OOF < 0.65" 분기 적용.

3 후보 모두 plan-013 candidate 으로 박제. LB carry-over 결과 (plan-012.1) 후 분기 결정.

---

## Candidate A — Paradigm 완전 폐기 (KNN / GP / Diffusion)

**전제**: plan-005~011 의 corrector path + plan-012 의 codebook hybrid 모두 marginal lever 만 — supervised parametric direction 자체가 limit.

**가설**: F0 raw 의 잔여 36% (= F0 fail sample) 가 *parametric* 신호 부재 (= noise-dominated 영역). 비모수 / 생성 모델 접근:
- **KNN** (residual_world 기반 nearest neighbor 보정)
- **GP** (Gaussian Process regression on residual; sparse subset 1k samples)
- **Diffusion** (3D position generative — 시계열 → posterior sampling)

**measure 후보**:
- KNN k=5, distance = ‖seq_feat[-1]‖ (last-step kinematic)
- GP RBF kernel, 1k subset for tractability, residual scale prior 0.005m
- Diffusion U-Net 1D backbone, 50 step DDPM

**plan-012 와의 연속성**:
- F0 raw + KNN residual 보정 = single-formula + non-parametric correction (corrector path 의 비모수 변종)
- 본 plan 의 `ring_classifier.f0_predict_*` reuse + neighbor lookup 추가
- 신규 모듈: `src/pb_0_6822/nonparametric.py` (KNN/GP wrappers)

**risk**:
- diffusion 학습 비용 (CPU 환경에서 부담)
- KNN 의 cold-start (test 의 nearest train 신뢰성)
- plan-005~011 의 supervised lever 가 모두 marginal 였으므로 비모수도 marginal 가능성

**측정 target**: 5-fold OOF ≥ 0.65 (paradigm switch 의 가치 증명)

---

## Candidate B — F0 자체 교체 (G0 oracle 박제 기반 best codebook re-selection)

**전제**: F0 (`frenet_par120_perp_neg020` = plan-006 CANDIDATES[17]) 자체가 ceiling 에 가깝다 — 다른 single formula 또는 *sample-wise F0 selection* 으로 baseline 끌어올림.

**가설**: plan-006 의 27 CANDIDATES 중 F0 (idx 17) 가 globally best 인지 *sample-wise* 적합도가 다른 candidate 다수가 있을 가능성 — *per-sample best formula* hit ceiling 측정.

**measure 후보**:
- 27 candidate 각각 raw hit 측정 + per-sample best (oracle) 산출
- per-sample best 의 hit ceiling = sample-wise oracle (= plan-007 ranking 참조)
- classifier 가 sample 별 F0 선택 (= 27-way classification, plan-012 의 hybrid arch 그대로 reuse)

**plan-012 와의 연속성**:
- 본 plan 의 HybridScorerHead K=27 으로 swap
- anchor 좌표 = 0 (formula 자체 = anchor 역할)
- reg head = (mode-별 small residual MLP, ±5mm)

**risk**:
- 27 candidate 중 F0 가 이미 *최대 일관성* 가질 가능성 (plan-007 evidence 일부)
- per-sample classification 의 학습 시그널 분리 어려움 (= mode 다양성 약함)

**측정 target**: per-sample oracle ≥ 0.70 (= F0 raw 0.6320 + 0.07 ceiling 확장)

---

## Candidate C — Corrector + Hybrid 합체 (2-stage: selector → hybrid)

**전제** (★ recommended): plan-011 의 corrector path 가 plan-012 의 hybrid 보다 약간 큰 lever (best In/ID +0.0050 vs best hybrid +0.0020). 두 path 직렬 결합 = corrector 가 1차 보정 + hybrid 가 2차 미세조정.

**가설**:
1. plan-011 의 single-formula corrector (4-axis 중 In/ID positive) 로 F0 → F0' 의 1차 보정 산출.
2. F0' (new corrected baseline) 위에 plan-012 의 hybrid codebook (E0a + τ=0.01 + r=0 +0.5) 적용.
3. corrector 와 hybrid 의 *complementary* lever (corrector = sample-wise scalar shift, hybrid = mode-discrete + small offset) — additive 가능성.

**measure 후보**:
- Stage 1: plan-011 의 best In/ID corrector reproduce + F0' = corrected_F0_pred
- Stage 2: F0' 를 anchor 로 본 plan-012 의 phase1_bakeoff.py 재실행
- 합격: 5-fold OOF ≥ 0.66 (plan-012 의 0.6340 + 0.026 = plan-006 corrected 0.6491 + 0.011)

**plan-012 와의 연속성**:
- 본 plan 의 `ring_classifier.py` 의 F0 산출만 corrected_F0 으로 교체
- 모든 phase 1~4 wrapper reuse
- 신규 모듈: `src/pb_0_6822/corrector_hybrid.py` (corrector inference + hybrid input wiring)

**risk**:
- corrector 가 1차 보정 시 residual scale 이 줄어 hybrid 의 lever 가 더 약해질 수 있음
- 두 stage 의 train/val split 동기화 (fold-aware corrector reuse + hybrid fold-별 학습 fairness)

**측정 target**: 5-fold OOF ≥ 0.66 (= plan-012 target 회복)

★ **recommended**: Candidate C 가 plan-012 의 hybrid path 와 plan-005~011 의 corrector path 를 *모두 살리는* 방향. plan-013 default 후보.

---

## 분기 결정 매트릭스 (plan-012.1 LB carry-over 후)

| plan-012.1 LB | 추천 plan-013 |
|---|---|
| LB ≥ 0.65 | **Candidate C** (corrector+hybrid 합체, hybrid 가치 검증됨) |
| 0.60 ≤ LB < 0.65 | **Candidate B** (F0 자체 교체, baseline 자체가 limit) |
| LB < 0.60 | **Candidate A** (paradigm 완전 폐기, non-parametric) |

---

## 신규 plan-013 작성 시 reuse 가능한 plan-012 자산

- `src/pb_0_6822/ring_classifier.py` — 7 컴포넌트 모듈 (codebook, hybrid, F0 산출식 spot-fixed)
- `src/pb_0_6822/ring_classifier_train.py` — run_sub_exp helper (3 phase reuse)
- `src/pb_0_6822/selector.py` — make_seq_features, stable_fold_id (변경 없음, plan-004 그대로)
- `analysis/plan-012/preflight.py` — G0 산출 식 (per_axis_marginal_oracle, oracle_hit_for_codebook 등)
- `runs/baseline/H019_phase0-preflight-codebook/preflight.json` — 데이터 분포 reference (재학습 불필요)
