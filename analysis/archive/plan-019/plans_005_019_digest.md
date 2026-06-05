---
digest_scope: plan-005 to plan-019 (15 plans)
generated_at: 2026-05-15 (Asia/Seoul)
generated_by: plan-019 G_final carry — plan-020 (plan-004 upgrade direction) 입력자료
field_spec:
  검증목표: 1줄, 가설 형태 (재실험 의도 추적용)
  결과: PASS/PARTIAL/FAIL/SEVERE_FAIL/SUPERSEDED/INVALID + OOF/LB 측정값 inline
  실패사유: 1~2줄, root cause (재시도 방지용)
status_legend:
  PASS: G_final 합격 + LB 회수 + positive band
  PARTIAL: G_final 합격, 일부 G-gate WARN
  FAIL: G_final 도달했으나 핵심 가설 falsified
  SEVERE_FAIL: target 의 30% 미만 또는 catastrophic regression
  SUPERSEDED: 미실행, 후속 plan 으로 대체
  INVALID: paradigm 자체 결함, 결과 박제만 (재사용 금지)
note: |
  원본 plan-NNN-*.md / *.results.md 모두 그대로 보존 (WORKFLOW.md §12.6 blacklist 준수).
  본 digest 는 *탐색용 overlay* — 사실 출처는 항상 원본 file + git history.
---

# Plans 005–019 Digest (3-field compression)

15 plans 의 *검증목표 / 결과 / 실패사유* 박제. 원본 보존 (`plans/plan-NNN-*.md` + `*.results.md`).

> **탐색 방법**: `grep -A 4 "^## plan-NNN" analysis/plan-019/plans_005_019_digest.md`

---

## plan-005 — pb-0-6822-diagnostic

- **검증목표**: plan-004 framework 의 모든 약한 지점 정량 진단 (ceiling / selector / corrector 3축, 20 metric anchor 박제)
- **결과**: SKIP — diagnostic-only, 본문 STAGE 1~6 정의만, 실제 진단 코드 미실행. 20 metric anchor → plan-006~011 baseline reference
- **실패사유**: 별도 실패 없음 — diagnostic 의도가 plan-006 의 *직접 baseline* 으로 흡수돼 종결 (results.md 미생성)

## plan-006 — minimal-variant-e-lb

- **검증목표**: Variant E (frenet_par120_perp_neg020 단일공식) 의 LB 입증 — plan-005 통찰 "27후보+bias+soft averaging 만 필요" 의 LB 단위 검증
- **결과**: PASS — OOF 0.6524 / **LB 0.6692** (baseline anchor 등극, ≥ 0.6606 cutoff)
- **실패사유**: 부분 한계 — GRU+regime corrector 가 base 위 marginal (정량화 완료, plan-008 후속 carry)

## plan-007 — formula-tuning

- **검증목표**: CMA-ES 6변수 → basis ablation → Step 4 MLP per-sample coeff 가 단일공식 ceiling 위 +0.005+ 회수
- **결과**: PARTIAL — OOF 0.6482 / **LB 0.6598** (Step 3), MLP +0.0095 OOF gain 이 threshold 충족하나 LB 회수율 mismatch
- **실패사유**: single-stack ceiling 0.66 plateau 진입 — 단일공식 framework 의 표현력 자체가 LB 0.70 path 아님

## plan-008 — candidate-redefine-corrector-redesign

- **검증목표**: 27→25 candidate redefine + greedy set-cover (oracle 0.85 target) + corrector band-specific
- **결과**: FAIL — oracle 0.7543 (target 0.85 의 30% 미달), selector OOF 0.6503 (-0.007 vs plan-007), SEVERE flags 2개
- **실패사유**: ranking bottleneck — candidate pool 확장으로 oracle 은 올랐으나 selector 가 follow 못 함 (gap_ranking 0.0516 ≫ 0 drift)

## plan-009 — selector-ranking-loss

- **검증목표**: NDCG@1 + pairwise + ListMLE 3-component ranking loss 로 selector gap +0.02 회수, corrector 강화 G2 additive
- **결과**: SEVERE_FAIL — OOF 0.6482 (target 0.6703, -0.0221), top1_acc 0.0922 (-0.13 miss), G1 SEVERE; G2 H002 sub-exp b OOF 0.6653 carry
- **실패사유**: 3-component loss conflict — pairwise×2 의 err-rank vs label-gap 전략 충돌, ranking 자체가 fragile lever

## plan-010 — corrector-redesign-exploration

- **검증목표**: plan-004 corrector 7 결함 fix (cap-truncation / MSE-hit / far·easy weight / env head / apply_scale / hardcoded band / iterative refinement)
- **결과**: SUPERSEDED — 미실행. plan-011 이 동일 narrative 더 정밀 ablation 으로 대체
- **실패사유**: "결함 fix" 프레임 자체가 plan-004 default 가 small-data 에 최적화된 hyperparam 임을 간과 (fix 불가)

## plan-011 — single-formula-corrector-exploration

- **검증목표**: 4축 (Loss / Input / Arch / Formula) × 24 sub-exp 폭넓은 탐색, 1+ 축에서 strict +0.005 회수
- **결과**: FAIL — 0/4축 strict +0.005 미달 (In/ID 최근접 +0.00495), best OOF fold0 0.6446 (plan-006 대비 -0.0246)
- **실패사유**: 단일공식 + corrector 의 measured ceiling = 0.645 — 단일공식+corrector path 의 내재적 한계

## plan-012 — frenet-ring-classification

- **검증목표**: 3D Frenet codebook bake-off (Absolute / Orthogonal / K-Means 3경로 분류+회귀 hybrid) 로 selector arch 대체
- **결과**: INVALID — 5-fold OOF 0.6350 (plan-006 대비 -0.0342), codebook 3-way marginal (Δ<0.005), 8 ablation 누적 +0.005 미만
- **실패사유**: 코드 재사용 강박이 root cause — paradigm shift trap (mean-regression + commit magnitude underflow). 어떤 수치도 후속 reference 금지

## plan-013 — plan004-framework-3lever-stacking

- **검증목표**: plan-004 framework 위 plan-007/008/011 의 측정된 3 lever (In/IC frozen GRU + Step 4 MLP + 25 cand) additive stacking
- **결과**: PARTIAL (warn-recovered) — OOF 0.6381 / LB null (plan-013.1 carry), G0 partial (3/4 infra), G1 baseline fallback, G2/G3 DEFERRED
- **실패사유**: simplified pipeline (plan-004 boundary.py 의 regime/env/pretrain/finetune 제외) penalty — G1 baseline 자체 0.65 threshold 미달, lever stacking 진입 불가

## plan-014 — plan012-failure-inversion

- **검증목표**: F0 plan-006 frozen + corrector from-scratch BiGRU+HybridHead 로 plan-012 failure mode 회피, +0.005 OOF 회수
- **결과**: PARTIAL — OOF 0.6425 / **LB 0.6628** (band positive), G5 best K=9 + boundary_weight = +0.0066 OOF
- **실패사유**: corrector arch lever 자체가 F0 위 +0.005 회수 부재 (5축 ablation 모두 marginal). 회수율 5.4% premise falsified

## plan-015 — feature-expansion

- **검증목표**: feature A/B/C/D 순차 ablation, A (F0 residual direct) 가 plan-014 회수율 root cause 가설 검증
- **결과**: PARTIAL — OOF 0.6425 / LB 0.6628 (band negative despite LB positive), Feature A -0.001 → drop rule, G2~G4 skip
- **실패사유**: A 가설 falsified (9D base 에 F0 정보 implicit), root cause 미회수. LB-OOF gap +0.020 systematic underestimate 재해석만

## plan-016 — corrector-stabilization

- **검증목표**: 3 limitation 직접 closure (L1 feature redundancy / L2 fold variance / L3 early-stop jump)
- **결과**: PARTIAL — OOF 0.6452 / **LB 0.6638** (band positive), G1 multi-seed +0.0027 OOF / +0.0010 LB (sub-threshold), G2~G5 3/3 sub-threshold
- **실패사유**: paradigm ceiling 실측 확정 — 단일/ensemble 변형도 0.66 plateau. 단일-stack stabilization 만으로 break 불가

## plan-017 — low-cost-stage1

- **검증목표**: G1 plan-013/014_15/016 3-plan ensemble + G2 voxel CE 7×7×7 head 교체, 둘 중 1+ 가 OOF +0.003 회수
- **결과**: PARTIAL — OOF 0.6452 / **LB 0.6640** (band positive, G1 carry), G1 +0.0002 marginal pass / G2 OOF -0.0121 fail
- **실패사유**: G1 ensemble 효과 marginal (ε 가설 falsified), G2 voxel CE early stop (val_hit 학습 후 개선 없음). paradigm-shift 필요 measured

## plan-018 — arch-ablation-single-model

- **검증목표**: A0 baseline + A1~A6 (A4/A5 제외, 4 arch) architecture ablation, 1+ 가 plan-007 step 4 위 +0.005 회수
- **결과**: FAIL — OOF A3 MoLE 0.6485 best / LB skip (quota 보존), 4/4 sub-threshold
- **실패사유**: H1 encoder bottleneck 가설 falsified (A1/A2/A6 ≤ A0), H2 head capacity marginal +0.0003. single-stack arch lever 자체 한계

## plan-019 — meta-ebip-icnn-hybrid ★ ACTIVE (plan-020 baseline)

- **검증목표**: 3 stage progressive (S1 EBIP / S2 ICNN / S3 meta-FOMAML) energy-based implicit prediction 으로 LB > 0.70 회수 (plan-005 oracle 0.7188 의 97%)
- **결과**: FAIL — OOF S1 0.6552 best / LB skip (quota 보존), G1/G2/G3 모두 WARN. S1 EBIP +0.0070 / S2 ICNN -0.003 / S3 meta +0.0018
- **실패사유**: single-stack implicit 도 A0 위 +0.007 ceiling — energy-based 3 stage 의 component 추가가 marginal/없음. plan-007 §9.2 단일 공식 framework 한계 0.6491 위 약간 push 만 가능
