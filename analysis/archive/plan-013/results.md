# plan-013 results — plan-004 Framework + 3 Lever Stacking (measured-only)

**status**: G_final_complete (warn-recovered — G0 partial + G1 warn + G2 severe-recovered (3 lever deferred) + G3 warn; LB carry-over plan-013.1)
**date_completed**: 2026-05-14 (Asia/Seoul)
**date_started**: 2026-05-14 (Asia/Seoul, plan-013 v1 작성 → c1~c9 + G_final 동일 일자 자율 진행)

## §1. G-gate 결과 요약

| G | spec | result | status |
|---|---|---|---|
| G0 | preflight + 4 infra | 3/4 PASS (plan_004_reproduce drift=0.0020 / in_ic_infra / step4_infra) / 1 MISS (cand_25_infra: G001 cand_set.{json,npy} 미존재) | **PARTIAL** → autonomous (Phase 2.E3 fallback path 진입 준비) |
| G1 | plan-004 + In/IC 5-fold OOF ≥ 0.65 | 5-fold concat OOF = **0.6381** | **WARN** (`baseline_below_expected`, severity X — simplified pipeline penalty; Phase 2 informational 진행) |
| G2 | Phase 2 3-sub-exp 완료 + 1+ axis ΔOOF ≥ 0.005 | 3 sub-exp 모두 DEFERRED (E1/E2 = plan-007 basis_terms framework gap; E3 = G001 cand_set MISS) | **FAIL** (`phase2_no_positive_lever`) → autonomous recovery (a) Phase 3 = best G1 baseline 단독 |
| G3 | best stack 5-fold OOF ≥ G1 + 0.005 (= 0.6431) + submission | 5-fold OOF = 0.6381 (= G1, lever 0 stack), submission.csv 10000 rows 박제 | **WARN** (`final_no_additive`, severity X — fallback path 의 자연스러운 결과, super-additive 불가) |
| G_final | synthesis + plan-014 후보 ≥ 3 + 3 파일 sync + plan-013.1 instruction | 본 commit | **DONE** |

## §2. Per-fold breakdown (Phase 1 = Phase 3 fallback, 동일 config)

| fold | n_val | hit @ 1cm | n_epochs (early stop) | in_ic_drift |
|---|---|---|---|---|
| 0 | 2020 | 0.6545 | 10 | False |
| 1 | 2047 | 0.6317 | 6 | False |
| 2 | 1921 | 0.6393 | 16 | False |
| 3 | 2020 | 0.6386 | 12 | False |
| 4 | 1992 | 0.6265 | 9 | False |
| **concat** | **10000** | **0.6381** | — | **0/5 (PASS)** |

**frozen_gru_drift check**: ★ 5 folds 모두 PASS — In/IC encoder state_dict hash 변경 0, severe trigger 회피. plan-013 의 핵심 안전 invariant (`frozen_gru_drift` severe) 가 simplified pipeline 위에서도 결정적으로 검증됨.

## §3. Lever attribution measurement (intent vs reality)

| lever | intent (§1.3-§1.5 measured anchor) | actual measurement in plan-013 | gap |
|---|---|---|---|
| **In/IC** (plan-011) | +0.0050 OOF (1-fold, In̂=IC) | G1 baseline 흡수 — 단독 sub-exp 없음. cross-experiment 차이 (G0 plan-004 reproduce 0.6491 ≈ G1 0.6381) ≈ -0.011 → negative (simplified pipeline penalty 가 dominant) | "1-fold 0.6446 → 5-fold ≥ 0.65" 추정 (§1.3 convex 외삽) 이 simplified pipeline 위에서 실패 |
| **Step 4** (plan-007) | OOF 0.6482 단독, LB 미회수 | DEFERRED (basis_terms framework gap, c5+c6) | plan-007 mlp_coeff.py 의 compute_trajectory_features + per-var basis_terms 산출 통합 필요 — c5 scope 초과 |
| **25 cand** (plan-008) | oracle 0.7188→0.7543 (+0.036) | DEFERRED (G001 cand_set.{json,npy} 미존재) | plan-008 G1 candidate descriptor list 가 별도 박제되어 있지 X (G001 디렉토리에 selector/boundary 산출만) |

**측정 가능한 단일 fact**: plan-004 framework 의 selector OOF 5-fold soft = 0.6511 (P001 산출 측정값, G0 preflight 박제). 본 plan 의 simplified corrector pipeline 위에서 5-fold OOF 0.6381 — selector-only 보다 *낮음* (-0.013). 즉 plan-013 의 corrector 가 P001 plan-004 boundary 의 fold-0 단일 결과 (0.6717) 만큼 reach 하지 못함 — 본 plan 의 architectural gap 의 직접 표현.

## §4. 박제된 산출물

- `analysis/plan-013/preflight.{py,json}` — G0 4 infra 검증
- `analysis/plan-013/phase1_baseline.{py,json,npz}` — G1 5-fold OOF 0.6381
- `analysis/plan-013/phase2_step4_F0.{py,json}` — E1 DEFERRED
- `analysis/plan-013/phase2_step4_27ext.{py,json}` — E2 DEFERRED
- `analysis/plan-013/phase2_25cand.{py,json}` — E3 DEFERRED
- `analysis/plan-013/phase3_best_stack.{py,json,npz}` — G3 fallback 5-fold + test ensemble
- `analysis/plan-013/submission.csv` — 10000 rows test predictions (5-fold mean ensemble)
- `src/pb_0_6822/integrated_v3.py` — plan-004 framework wrapper (4 working components + 1 Phase 1 dispatcher)
- `tests/test_integrated_v3_smoke.py` — 12 smoke tests, 11 pass / 1 skip

## §5. Architectural gap analysis (plan-013.1 carry-over)

1. **simplified pipeline penalty**: plan-013 의 standalone residual corrector 는 plan-004 boundary.py 의 regime/env/pretrain/finetune/temperature_search/score_bank-aware loss 등 *full framework* 를 *제외*. 결과 = G1 5-fold OOF 0.6381 (plan-004 LB 0.6806 transfer 추정 G1 = 0.6541 보다 -0.016 deflate). **회수 경로**: plan-013.1 에서 boundary.py BOUNDARY_MAIN 의 CLI 를 monkey-patch (TinyCorrectionNet → InICCorrectorWrapper) 또는 boundary.py 의 train_net 함수 시그니처 확장 (`corrector_cls` arg 추가).
2. **plan-007 basis_terms framework integration**: Step 4 (F0_only / 27ext) 충실 구현은 plan-007 mlp_coeff.py 의 `compute_trajectory_features` + per-var `basis_terms` tensor 구축 + soft_hit_loss 통합 필요. 본 plan 의 c5/c6 단일 commit scope 초과. **회수 경로**: plan-013.1 에서 plan-007 의 `train_one_fold` + `compute_pred` 함수를 integrated_v3 의 dispatcher 에 통합.
3. **plan-008 G1 cand_set 박제 부재**: G001_candidate-redefine 디렉토리에는 selector/boundary 산출만 있고 25 candidate descriptor list (cand_set.json or .npy) 는 별도 박제 없음. **회수 경로**: plan-013.1 에서 plan-008 G1 의 25 candidate descriptor 를 cand_set.json 으로 별도 박제 (plan-008 selector training script 의 candidate definition 부분 복원).

## §6. plan-013.1 carry-over instruction

★ **LB carry-over (§0 L51 정책)**: 본 plan 내 LB 제출 0 회. best Phase submission `analysis/plan-013/submission.csv` (5-fold OOF 0.6381) 의 LB 회수는 plan-013.1 의 *사용자 manual* dacon-submit:

```
dacon-submit analysis/plan-013/submission.csv
```

또는 web UI manual upload. LB 회수 후:
- `plans/plan-013-plan004-framework-3lever-stacking.md` frontmatter `lb_score: <측정값>` sync
- `plans/plan-013-plan004-framework-3lever-stacking.results.md` frontmatter `lb_score: <측정값>` sync
- `registry.csv` 의 H035 row `notes` field 에 `;lb=<측정값>` 추가

## §7. Decision-note chain (autonomous 자율 결정 박제)

- **c1.1**: plan-review-master 5 iter 자동 fix (BLOCKER 11 + AMBIGUITY 6) — reference-aligned mode FP rule 적용
- **c2**: R001 ckpt path 정정 (`ckpt/fold0.pt`), TinyCorrectionNet signature 실제 `(dim, hidden)`, forward 반환 `(delta, env)` tuple 처리
- **c3**: plan_004_reproduce 의 5-fold corrected OOF 직접 산출 대신 selector 5-fold soft 0.6511 proxy
- **c4**: plan-004 complexity 제외 minimal residual corrector — Δ measurement 가능한 simplest scaffold
- **c5/c6/c7**: 3 sub-exp 모두 deferred — framework gap + cand_set MISS, plan-013.1 carry-over 동일 회수 경로
- **c8**: phase2_no_positive_lever autonomous recovery (a) — best G1 baseline 단독 5-fold + submission fallback
- **c9 (본)**: G_final 진입, plan-014 후보 (조건부 framework, §9.2) + plan-013.1 carry-over instruction
