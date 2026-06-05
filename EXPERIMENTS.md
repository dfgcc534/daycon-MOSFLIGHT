# EXPERIMENTS

대회: DACON 모기 비행궤적 예측 (comp 236716, 종료 2026-06-01). metric = hit@1cm.
Floors: F0 single-formula = 0.6320 · Kalman-alone = 0.5964 · B001 linear-2pt = 0.0129 mean_eucl.

이 문서는 모든 plan을 동등 level로 나열한다. 어떤 plan도 "best"로 강조하지 않는다 — 같은 컬럼·같은 row 구조. 수상자 노트북 재현이 합류하면 같은 표에 동일 컬럼으로 추가된다 (`WINNERS.md` 참조).

## Paradigm timeline

플랜은 paradigm 단위로 진행했고, 각 단계가 어떤 문(door)을 닫았다:

1. **001~003 — Closed-form polyfit / spline / lightweight Residual-GRU baselines.** Linear-2pt 가 가장 견고 (CV 0.0129, LB 0.60). spline 4 변형 모두 B001 LB 0.60 미달이나 CV/LB Spearman ρ=+0.90 일관성만 박제. R001 residual-GRU 가 CV 동급 + LB 0.5688 추가.
2. **004~009 — PB_0.6822 ipynb 의 *27-candidate selector + boundary corrector* framework 이식과 분해.** plan-004 LB 0.6806 즉시 도달 후 plan-005~009 가 selector / corrector / candidate-pool / ranking 4개 component 별 ablation. plan-008/009 가 27→25 candidate redefine + ranking loss 시도, LB 회귀 (0.6748) → ranking lever 한계 박제.
3. **011~013 — corrector redesign + plan-004 framework 3-lever stacking.** "plan-004 의 결함을 fix" 가설을 in/loss/arch/formula 4-axis ablation 으로 falsify (plan-011 best +0.00495 sub-threshold). plan-013 3-lever stacking 도 simplification penalty 로 OOF 0.6381 (sub-baseline).
4. **012, 014~019 — paradigm-shift attempts: Frenet-ring classification / codebook bake-off / feature expansion / multi-seed stabilization / Voxel CE 7³ / single-stack arch ablation / meta-EBIP+ICNN.** 모두 sub-threshold. plan-014 corrector ceiling 0.6425 박제, plan-018 single-stack arch lever 자체 falsify, plan-019 energy-based paradigm 도 marginal. "selector → corrector" framework 의 internal lever 가 거의 다 닫힘.
5. **020~023 — F0 structural search + Frenet corrector-free anchor sweep.** plan-020 C05 per-regime F0 +0.0183 OOF lift (단독 PASS). plan-021/022/023 가 anchor-codebook geometry 를 sweep — BCC14·Fib50 등 paradigm-positive 진입, plan-023 best OOF 0.6532.
6. **025~029 — LGBM row-expand + GRU-attention input-max.** plan-025 1080D LGBM mode collapse, plan-026/027 LGBM-based abandoned (user intent mismatch), plan-028 per-anchor isolation partial, plan-029 GRU-attention regression. anchor + LGBM/attention 결합 한계 박제.
7. **030~032 — PB training-procedure carry.** plan-030 residual axis fix 만으론 regression (-0.0026), plan-031 multi-phase + pairwise margin + prior + head slim 이 STRONG 회복 (OOF 0.6397, +0.0103 lift). plan-032 4-axis ablation 중 boundary corrector axis B 만 +0.0041 추가 lift. "main carrier = training procedure" 진단 *직접 검증*.
8. **a-001 ~ a-004 — Kalman-Residual GRU (lane A, 노트북 LB 0.6780 재현).** KR001 OOF 0.6639 (F0 +0.0319). KR002 입력-yaw 회전이 OOF neutral 인데 LB +0.0060 = CV-LB 괴리 첫 확증. KR003 Kalman 부산물 feature LB +0.0036 (2번째 괴리). KR008 aug LB +0.0008 (noise floor 내). KR010 multi-hypothesis = KILL (단일 GRU 가 conditional 최적). lever 수확 체감 + paradigm 종료 신호.
9. **b-001 — yaw-frame anchor-selector + attention restructure.** F0/Kalman 양 arm 모두 G3 FAIL_regression, "frame degeneracy 가 plan-030 실패의 carrier" 가설 기각. *training procedure carrier* 재확증.
10. **c-001 — F0(perp=0.0) 잔차 GRU.** 잔차-GRU paradigm 의 baseline-swap 성공 (F0 floor 0.6320 → 0.6622, +0.0302). KR002 0.6663 와 사실상 동급 — "lift 는 baseline 품질보다 GRU 잔차 학습에서 온다" 지지.
11. **d-001 — Neural ODE 노트북 재현.** OOF 0.6330 (F0 floor +0.0010 = 통계 동률). final-epoch overfitting 으로 fold0 peak 0.6663 → ep15 0.6307 퇴행. "학습 물리 ≈ F0 < 잔차" paradigm 서열 확장.

거대한 단일 lever 는 (a) **plan-004 framework 이식** (+0.082 over linear), (b) **잔차 GRU paradigm** (F0 또는 Kalman baseline 위 +0.030 lift, 두 baseline 무관), (c) **PB training procedure (multi-phase + pairwise + prior)** (+0.010 over single-phase) 의 3개. 나머지 paradigm-shift 시도 (codebook / feature expansion / arch / voxel CE / energy-based / multi-hypothesis / yaw-frame / neural ODE / 1080D LGBM) 는 모두 sub-threshold / neutral / KILL.

## Plan table

| plan_id | title | paradigm | OOF hit_1cm | LB | band | lesson |
|---|---|---|---|---|---|---|
| 001 | polyfit baseline (linear/quad, 2pt/3pt) | baseline | cv_eucl 0.01294 (B001) | 0.60 | positive | linear-2pt 가 noise-averaging 보다 staleness penalty 작아 win — window 확대·degree 확대 모두 악화 |
| 002 | Cubic spline baseline (natural/notaknot/window/smoothing) | baseline | cv_eucl 0.01740 (S003) | 0.4932 (S001) | negative | spline 4 변형 모두 B001 LB 0.60 미달; CV/LB Spearman ρ=+0.90 일관성만 확인 |
| 003 | Residual GRU lean baseline + 4-component ablation | baseline | cv_eucl 0.01338 (R001) | 0.5688 (R006) | inconclusive | physics/EMA/wingbeat/MSE 4 component 모두 winning=0 → R006 combined = R001 carry; residual-GRU floor 만 측정 |
| 004 | PB_0.6822 ipynb full-fit (selector + boundary corrector) | corrector framework | selector_soft 0.6511 / boundary 0.6718 | 0.6806 | positive | 노트북 framework 그대로 이식해 LB +0.08 lift; 27 후보 + soft averaging 의 가치 박제 |
| 005 | PB_0.6822 framework diagnostic (oracle/selector/corrector decomposition) | corrector | 0.660 (full) / 0.657 (no regime) / 0.655 (no GRU) | — (analysis-only) | inconclusive | "framework 95% 가 장식, 진짜 엔진 = 27 후보 + physics_bias + soft averaging" — corrector 가 oracle 을 깎는 corrector_hurts_oracle 발견 |
| 006 | Minimal Variant E LB validation (physics_bias + soft avg only) | corrector simplification | 0.6524 | 0.6692 | positive | GRU/regime 제거하고 27 cand + physics_bias 만으로 LB 0.6692 = plan-005 통찰 LB 검증 |
| 007 | Single-formula CMA-ES + basis ablation + per-sample MLP | single-formula | 0.6482 (Step 4 MLP) | 0.6598 | inconclusive | 단일 공식 framework 측정 ceiling 0.6482 ≈ plan-006 0.6491 — 새 ceiling 돌파 없음 |
| 008 | Candidate pool redefine 27→25 + corrector band loss | candidate-pool | 0.6503 | 0.6812 (carry 008.1) | inconclusive | greedy set-cover 로 oracle +0.0355 회복하나 selector 가 따라가지 못함 — main_bottleneck = "ranking" 박제 |
| 009 | Ranking-specific loss (NDCG/pairwise/ListMLE) + corrector strengthening | ranking-loss / corrector | 0.6653 (H002 best) | 0.6748 | negative | OOF +0.0150 이나 LB -0.0064 sign inversion; ranking lever LB 한계 + corrector strengthening robust 여지 |
| 010 | Single-formula anchor + corrector redesign | corrector-redesign | — (not executed) | — | superseded | plan-011 이 동일 narrative 4-axis ablation 으로 falsify — Z1/Z3 fix 가설 폐기 |
| 011 | Single-formula + corrector 4-axis breadth ablation (L/In/M/F) | corrector-redesign | 0.6450 (In axis ID, fold0) | (carry 011.1) | negative | 0/4 axes strict +0.005 통과 — In axis ID = +0.00495 단지 sub-threshold; plan-004 default 가 small-data 에 best tuned 임 확정 |
| 012 | Codebook bake-off classification + regression hybrid (3D) | codebook classification | 0.6350 (★ INVALID 재사용 환경) | — | inconclusive | plan-014 진단 "재사용 강박이 root cause" — 본 결과는 reference 사용 금지, historical only |
| 013 | plan-004 framework + 3-lever stacking (In/IC, Step4 MLP, 25-cand) | framework-stacking | 0.6381 | (carry 013.1) | negative | simplified pipeline penalty 로 baseline 미달; 3 sub-exp 다 deferred (framework gap × 2 + cand_set MISS × 1) |
| 014 | plan-012 paradigm 부활 from-scratch (corrector 5-phase) | corrector from-scratch | best_stack 0.6425 / anchor 0.6359 | 0.6628 | negative | corrector paradigm measured ceiling 0.6425 박제 — band 0.65 미달, deep path-pivot 필요 |
| 015 | Corrector input feature 확장 (A/B/C/D 순차 ablation) | feature-expansion | 0.6415 (E1, drop) | 0.6628 (carry) | negative | Feature A (F0 residual direct, 12D) ΔOOF=-0.001 → drop rule 발동, G2~G4 skip — "회수율 root cause = F0_pred missing" 가설 falsify |
| 016 | Corrector stabilization (multi-seed / val_loss / Feature B/C/D 단독) | multi-seed stabilization | 0.6452 (Path A) | 0.6638 | positive | multi-seed +0.0027 marginal under threshold; Path B/C 3/3 sub-threshold — paradigm-level 한계 measured |
| 017 | Low-cost stage 1 (3-plan ensemble + Voxel CE 7×7×7) | ensemble / Voxel-CE | 0.6452 (G1) / 0.6331 (G2) | 0.6640 (G1) | positive | G1 ensemble +0.0002 marginal positive; G2 Voxel CE -0.0121 negative_drop — paradigm-shift #1 권장 |
| 018 | Arch ablation single-model (A1 SetTrans / A2 PathSig / A3 MoLE / A6 GRU-attn) | arch-ablation | 0.6485 (A3 MoLE best) | — (G2 skip) | negative | 4/4 ablation arch ALL FAIL (best +0.0003 vs A0 0.6482); H1 encoder bottleneck falsify, H2 head capacity marginal — single-stack arch lever 한계 |
| 019 | Meta-EBIP + ICNN hybrid (energy-based 3-stage) | energy-based hybrid | 0.6552 (S1 EBIP best) | — (G4 skip) | negative | S1/S2/S3 모두 sub-threshold (best +0.0070 vs A0 < gates 0.66/0.68/0.70); energy-based paradigm marginal-only |
| 020 | F0 structural search (17 candidates: 14 deterministic + 3 NN) | F0 search | 0.6503 (C05 per-regime F0) | — | positive | C05 per-regime F0 만 paired Δ +0.0183 / +0.0053 통과; CTRA/CTRV/Singer/helix 고정 물리 14종 모두 F0 미달 |
| 021 | Frenet corrector with input augment (LGBM vs GRU dual head) | Frenet corrector | 0.6488 (A LGBM) / 0.6408 (B GRU) | — | positive | B GRU = pass_both gate keeper (Δ_1cm +0.0088 / Δ_1.5cm +0.0067) — 4 plan NN paradigm corrector ceiling 첫 양쪽 metric 통과 |
| 022 | Corrector-free anchor layout sweep (7 layouts × 3 τ_cls) | Frenet codebook | 0.6528 (A6 BCC14 τ=0.001) | — | positive | BCC14 + sharp soft label τ=0.001 best; 10/21 cell pass_both — anchor layout × τ_cls grid layout efficiency 측정 |
| 023 | Large-N anchor layout sweep (K=20/24/30/50) | Frenet codebook | 0.6532 (B4 Fib50 τ=0.001) | — | positive | K=50 Fibonacci spiral 이 plan-022 갱신; H2 (K=50 saturate) refuted — K=50 revival 박제, K=20~30 plateau |
| 025 | Candidate-concat input max (LGBM row-expand 1080D) | LGBM row-expand | 0.6320 (C1/C2 = F0) | — | negative | 1080D row-expand LGBM mode collapse (max_class_ratio ≈ 1/14 uniform) → soft-mean ≈ F0; per-anchor block ③ 22D 만 진정한 discriminative |
| 026 | Block ablation (no_block2 / no_block3 / no_block4) | LGBM block ablation | 0.6509 (A2 no_block3) | — | abandoned | user intent mismatch (LGBM 실행이나 GRU-attention 의도) — plan-029 재발행; block ③ 22D = LGBM trivial self-prediction trigger 박제 |
| 027 | 3-way ensemble (LGBM eq/weighted) | LGBM ensemble | 0.6529 (E3 weighted) | — | abandoned | user intent mismatch — paradigm-shared assumption 으로 prediction error correlated, anchor codebook 차이만으로는 ensemble diversity 부족 |
| 028 | Per-anchor isolation × sample-weight probe | anchor-isolation | 0.6509 (B3) | — | inconclusive | 5 가설 모두 inconclusive; 14-anchor oracle 회수율 82.10% — partial band, B4 baseline 동급 |
| 029 | GRU-attention input max (4-lever query/key/embedding/skip) | GRU-attention | 0.6316 (X1) | — | negative | G3 regression (-0.0004 vs F0); attention 학습 자체는 정상이나 paradigm framework 가 F0 floor 아래 — paradigm-distinct lever 전환 권장 |
| 030 | GRU-attention residual injection ((a)/(b) input feature axis fix) | residual injection | 0.6294 | — | negative | G3 FAIL_regression (-0.0026 vs F0, -0.0022 vs plan-029); 잔차 input feature 만으로 회복 불가 — "main carrier = training procedure" 진단 확정 |
| 031 | PB training procedure carry (multi-phase + pairwise + prior + head slim) | training-procedure | 0.6397 | — | positive (STRONG) | plan-030 0.6294 → +0.0103 lift; per-fold std 0.0044 안정; "main carrier = training procedure" 직접 검증; PB selector ensemble 0.6511 까지 -0.0114 잔여 |
| 032 | PB target 0.6511 도달 시도 (4 axis A/B/C/D ablation) | training-procedure ablation | 0.6438 (+axis B) | — | positive (STRONG) | axis B (boundary corrector 14a) 만 +0.0041 lift; A τ_cls / C label smoothing / D input axis 모두 zero/negative — axis D 결과 = plan-030 input axis 가 dead weight |
| a-001 | Kalman-Residual GRU 노트북 재현 + 입력 yaw 회전 | Kalman-residual | 0.6639 (KR001) / 0.6663 (KR002) | 0.6758 / 0.6818 | positive | F0 floor +0.0319; 입력 yaw 회전 OOF neutral 인데 LB +0.0060 = CV-LB 괴리 첫 확증 |
| a-002 | Kalman 부산물 입력 feature (innov + filtered_v + cv_ca) | Kalman-residual | 0.6667 (KR003) | 0.6854 (KR003) | positive | OOF neutral (+0.0004 ns) 인데 LB +0.0036 = CV-LB 괴리 2번째 사례; OOF-neutral 으로 폐기 안 한 plan 설계 보상 |
| a-003 | 반사 + 노이즈 augmentation (KR003 위) | Kalman-residual aug | 0.6671 (KR008) | 0.6862 (KR008) | inconclusive | OOF +0.0004 / LB +0.0008 noise floor 내; lever 수확 체감 확정 (yaw +0.0060 → 부산물 +0.0036 → aug +0.0008 단조 감소) — paradigm 종료 신호 |
| a-004 | Multi-hypothesis 후보 (2-head MCL) | multi-hypothesis | oracle@2 0.6787 (KR010) | not_submitted | KILL | G1_decisive KILL (oracle headroom +0.003 ≪ +0.020 gate); 단일 GRU 가 conditional 최적, multi-head 분화 부족 — fail-fast 96초 1-fold 로 breadth skip |
| b-001 | Yaw-frame anchor-selector + attention restructure | yaw-frame anchor | 0.6296 (B001 F0) / 0.6077 (B002 Kalman) | 0.6162 (kalman_only) | negative | 양 arm G3 FAIL_regression; "frame degeneracy 가 plan-030 실패 carrier" 가설 기각, training procedure carrier 재확증 |
| c-001 | F0(perp=0.0) 잔차 GRU (FR001) | F0-residual GRU | 0.6622 (FR001) | not_submitted | positive | F0 floor +0.0302 lift = KR001 Kalman 위 +0.0319 와 동급 → "lift 는 baseline 품질보다 GRU 잔차 학습에서 온다" 지지 |
| d-001 | Neural ODE 노트북 재현 (NODE001) | Neural ODE | 0.6330 (NODE001) | not_submitted | inconclusive | F0 +0.0010 통계 동률 (p=0.79); fold0 ep3 peak 0.6663 → ep15 0.6307 overfitting; "학습 물리 ≈ F0 < 잔차" paradigm 서열 확장 |
| w-001 | rank1 euijin42 GOH30 재현 (GRU+ODE+HyperPhysics 30모델 블렌드, private 1위) | multi-arch blend | n/a (full-fit) | 0.703 (notebook 0.7035 Δ-0.0005) | EXCELLENT | 노트북 출력 4자리 일치 재현; KR008 +0.0168 lift carrier = 3-arch inductive bias 다양성 + cv_1step yaw 잔차 공통 backbone + Soft R-Hit loss + cache pretrain interior + Y-flip TTA + θ-가중 oversample stack; 단일 lever ablation 후속 plan w-2~7 박제 |

## Notes

- `plan-005` = analysis-only diagnostic (no LB submission, no results.md — frontmatter inline in plan-005 본문).
- `plan-010` = superseded by plan-011, not executed.
- `plan-024` = number skipped (no plan).
- `plan-026/027` = abandoned (LGBM/GRU-attention paradigm mismatch).
- `plan-009` LB carry-over closed in plan-009.1; `plan-008` LB carry to plan-008.1.
- LB record trajectory (시계열, *historical only — not a ranking*): linear 0.60 → plan-004 0.6806 → a-001 KR002 0.6818 → a-002 KR003 0.6854 → a-003 KR008 0.6862 (noise floor) → w-001 GOH30 0.703 (rank1 reproduce).
- OOF record trajectory: F0 0.6320 → plan-022 0.6528 → plan-031 0.6397 (different paradigm) → a-001 KR001 0.6639 → a-002 KR003 0.6667 → a-003 KR008 0.6671 → c-001 FR001 0.6622 (F0-baseline).

## 어디서 더 보나

- 각 plan 본문: `plans/plan-{id}-*.md` (legacy `plan-{NNN}-*.md`) — 가설·실험설계·합격기준
- 각 plan 응답: `plans/plan-{id}-*.results.md` — 실측·band·lesson 본문
- archive 된 legacy: `plans/archive/plan-005~019-*.md` (paradigm 종료된 path)
- experiment registry: `registry.csv` — exp_id 단위 (한 plan 안 여러 exp)
- 산출물: `analysis/plan-{id}/` (분석 스크립트·figures) · `runs/baseline/{exp_id}/` (config snapshot·summary·history·log)
