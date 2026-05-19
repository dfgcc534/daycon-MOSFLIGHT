---
plan_id: 023
version: 1.0
date: 2026-05-19 (Asia/Seoul)
status: in_progress
best_sub_exp: null
best_hit_1cm: null
best_hit_1.5cm: null
best_delta_1cm: null
best_delta_1.5cm: null
based_on:
  - 022 (best A6_bcc14_tau001 Δ_1cm +0.0208 / Δ_1.5cm +0.0071 둘 다 PASS. 10/21 cell pass_both. trend: K=7→14 anchor 에서 Δ sum 향상, sharp τ=0.001 winner. paradigm 박제: anchor + sharp τ 조합 lever 유효. Δ_1.5cm 가 +0.005 ceiling 근접 (+0.0071) — N>14 추가 lever 측정 필요)
  - 021 (selector-only ablation evidence carry: LGBM reg_offset = bound 의 0.5% (~0.01mm) dead → corrector-free spec carry. 170D input pipeline carry)
  - 020 (F0 baseline 0.6320 / 0.8033 + 5-fold stable_fold_id MD5)
inspired_by:
  - 022 (anchor layout sweep K∈{7,8,13,14} 측정 완료. K>14 (정 N면체 N=20/24/30/50) 미측정 — 사용자 narrative 2026-05-19 세션 "14N+" 박제)
code_reuse:
  - module: analysis/plan-022/anchors.py
    symbols: [LAYOUT_NAMES]
    reason: 기존 7 layout 의 dict 패턴 carry (B1~B4 동일 schema 로 확장).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [LgbmSelectorOnly, build_soft_label_with_tau]
    reason: model class + soft label 산식 그대로 carry. K 가변 parameter 만 변경 (7/8/13/14 → 20/24/30/50).
  - module: analysis/plan-022/run_oof.py
    symbols: [run_oof_cell]
    reason: per-cell 5-fold OOF runner 그대로 carry. anchor + τ 만 swap.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra]
    reason: 170D LGBM input pipeline 동일 (plan-022 §6.1 그대로 carry).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, PAR, PERP, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline injection + paired Δ anchor.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id]
    reason: 5-fold stable split (plan-020/021/022 carry).
followed_by:
  - plan-024 (가칭): best large-N layout (plan-023 결과) 위 corrector reg head 재투입 → 1.5cm metric ceiling 돌파 ablation (plan-022 followed_by 의 corrector 슬롯을 본 plan best 위로 carry)
  - plan-025 (가칭): N × radius shell 2D sweep — 본 plan 의 단일 shell 한계 박제 후 radius diversity lever 측정
scope: large-N (>14) vertex-transitive anchor layout sweep. 4 layout (N ∈ {20, 24, 30, 50}) × 3 τ_cls = 12 cell. baseline = F0 (plan-022 G3 criterion 동일). 추가 informational compare = plan-022 best (A6_bcc14_tau001, Δ_1cm +0.0208 / Δ_1.5cm +0.0071). 모든 input + LGBM hparam + fold split = plan-022 §6.1 / §6.2 그대로. 단일 변수 = anchor 좌표 (4 layout) + τ_cls (3 값). corrector reg head 재투입 / GRU / DACON LB / ensemble / N≥60 / radius ≠ 0.005m 단일 shell = out-of-scope.
exp_ids:
  - Z023_B1_dodeca20
  - Z023_B2_trunc_octa24
  - Z023_B3_icosidodec30
  - Z023_B4_fib50
# exp_id ↔ cell_key 매핑 룰: frontmatter `Z023_B{n}_<name>` (sub-exp 단위, 3 τ cell 묶음) ↔ 본문 cell_key `B{n}_<name>_tau{NNN}` (cell 단위, NNN ∈ {001, 003, 005}). `best_sub_exp` frontmatter 박제 시 cell_key 포맷 사용 (e.g. `B3_icosidodec30_tau001`) — Z023_ prefix 는 `exp_ids` 카탈로그 식별자, cell_key 는 measurement 단위 식별자.
lb_score: null
band: null
---

# plan-023 v1 — Large-N Anchor Layout Sweep (vertex-transitive, K>14, selector-only LGBM)

## §0. 한 줄 목적

> **plan-022 의 winner A6_bcc14_tau001 (K=14 + τ=0.001 sharp)** finding 위에서, **K 축을 N>14 large-N polyhedra (Platonic dodecahedron + Archimedean truncated octahedron / icosidodecahedron + Fibonacci quasi-uniform spiral) vertex 위로 확장** 하여 1.5cm hit-zone coverage 추가 향상 가능성 측정. 4 layout (N ∈ {20, 24, 30, 50}, B1/B2/B3 strict vertex-transitive + B4 jittered uniform) × 3 τ_cls = 12 cell 2D sweep. **plan-022 의 model / input / fold / soft-label 산식 / final composition / pass criterion 정확 carry — 단일 변수 = anchor 좌표 + τ_cls (= plan-022 §2 의 한 변수 원칙 동일 적용)**.
>
> **layout 후보 4 (모두 single shell ‖a‖ = 0.005m, vertex-transitive)**:
> 1. **B1** dodecahedron (20) — Platonic, K=14→20 자연 next step
> 2. **B2** truncated octahedron (24) — Archimedean, 가장 isotropic 24-point (∀ vertex 동일 dihedral)
> 3. **B3** icosidodecahedron (30) — Archimedean, 5-fold + 3-fold sym 균형, 30-point spherical t-design 후보
> 4. **B4** Fibonacci spiral N=50 — quasi-uniform (vertex-transitive 아니지만 jittered uniform), N upper-end risk zone probe
>
> **τ_cls scan**: {0.001, 0.003, 0.005}m (plan-022 §3.5 동일).
>
> **pass criterion**: 12 cell 중 ≥ 1 개가 paired Δ_1cm ≥ +0.005 AND Δ_1.5cm ≥ +0.005 → G3 PASS (plan-022 §3.2 G3 동일, baseline = F0). 추가 informational finding: plan-022 best Δ sum 0.0279 초과 cell 여부.
>
> **out-of-scope**: corrector reg head 재투입 / GRU sub-exp / LB 측정 / DACON submit / ensemble / anchor radius ≠ 0.005m / multi-shell layout / center anchor 추가 / N=60+ / chiral layout (snub cube/dodec) — handedness 결정 미박제로 본 plan 제외 (B2 truncated octahedron 으로 N=24 슬롯 충당).

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 2 module (anchors_largeN / run_oof_largeN) import + smoke + tests green. plan-022 selector_only_model.py import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline 5-fold concat OOF — hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038] (plan-020/021/022 carry). 위반 시 `f0_reproduce_drift` severe.
- **G2.B{n}** (n ∈ 1..4): sub-exp B{n} 의 3 τ_cls cell 모두 metric finite + `max_class_ratio < 0.95` (= soft-mean 의 최대 anchor share, `probs_all.mean(axis=0).max()` 식, plan-022 §3.3 / §6.2 일관). 위반 시 `lgbm_numerical` severe / `soft_label_collapse` warn.
- **G3 (paradigm-level)**: 12 cell (= 4 layout × 3 τ_cls) 중 ≥ 1 cell 이 paired Δ ≥ +0.005 *둘 다* 통과 → PASS. 0 통과 = `all_negative` warn 박제 후 G_final 진입.
- **G_final**: results.md + best cell 박제 (layout + τ_cls + Δ) + plan-022 best 대비 비교 박제 + follow-up plan 후보 ≥ 2 건 박제 + 3-file frontmatter sync.

### G-gates

- G0: STAGE 0 인프라 [TODO]
- G1: STAGE 1 F0 baseline reproduce [TODO]
- G2.B1: B1 dodeca20 3 τ_cls 측정 [TODO]
- G2.B2: B2 trunc_octa24 3 τ_cls 측정 [TODO]
- G2.B3: B3 icosidodec30 3 τ_cls 측정 [TODO]
- G2.B4: B4 fib50 3 τ_cls 측정 [TODO]
- G3: STAGE 3 best cell selection + paradigm finding [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-023-large-n-anchor-sweep.md` v1 작성 (plan-review-master 5-iter 자동 fix — iter 1 BLOCKER 1+AMB 5 fix / iter 2 AMB 3 fix / iter 3 BLOCKER 1+AMB 5 fix / iter 4 BLOCKER 0 sustained+AMB 2 fix / iter 5 BLOCKER 0 sustained+AMB 2 fix → 잔여 MINOR ~8) | [DONE — f87d902] |
| c2 | code | `analysis/plan-023/anchors_largeN.py` (4 layout numpy 상수 + smoke test: ‖a‖ = 0.005m exact, std ≤ 5e-10, np.unique == K — 모든 invariant 통과) | [DONE — 588e9d2] |
| c3 | code | `analysis/plan-023/run_oof_largeN.py` (5-fold OOF runner, plan-022 `run_oof_cell` importlib carry + 4 layout × 3 τ_cls = 12 cell sweep + CLI) | [DONE — 0b69ca7] |
| c4 | test | `tests/test_plan023_smoke.py` (7 pytest: import + 4 layout invariants + unique vertex + LgbmSelectorOnly K=50 fit/predict + soft label 4×3 sum=1 + F0 carry + samples/class floor) | [DONE — 2abd988] |
| G0 | gate | smoke + tests green — 7/7 pytest pass (112s, T5 K=50 LGBM ~110s) | [DONE — 2abd988] |
| c5 | exp G1 | F0 baseline 5-fold OOF reproduce → 0.6320 / 0.8033 (plan-020/021/022 carry exact). `analysis/plan-023/baseline_carry.json` 박제 (dataset_hash=b91502db94fab67d, n_samples=10000) | [DONE — a7198fb] |
| G1 | gate | F0 hit@1cm=0.6320 ∈ [0.6315, 0.6325] ✓ AND hit@1.5cm=0.8033 ∈ [0.8028, 0.8038] ✓ AND dataset_hash 일치 ✓ | [DONE — a7198fb] |
| c6 | exp G2.B1 | B1 dodeca20 — 3 τ_cls cell, **τ=0.001 PASS_BOTH** (Δ_1cm +0.0193, Δ_1.5cm +0.0080 = NEW BEST 1.5cm) + τ=0.003 PASS. sum 0.0273 < plan-022 0.0279. 1093s. | [DONE — 7c83eb1] |
| G2.B1 | gate | 3 cell metric finite ✓ + max_class_ratio < 0.95 ✓ (max 0.073) | [DONE — 7c83eb1] |
| c7 | exp G2.B2 | B2 trunc_octa24 — 3 τ_cls cell, τ=0.001 PASS_BOTH (Δ_1cm +0.0200, Δ_1.5cm +0.0072) + τ=0.003 PASS. sum 0.0272 < plan-022 0.0279. K=20→24 marginal 차이 미미. 1439s. | [DONE — b3bc1e3] |
| G2.B2 | gate | 3 cell metric finite ✓ + max_class_ratio < 0.95 ✓ (max 0.066) | [DONE — b3bc1e3] |
| c8 | exp G2.B3 | B3 icosidodec30 — 3 τ_cls cell, τ=0.001 PASS_BOTH (Δ_1cm +0.0199, Δ_1.5cm +0.0077) + τ=0.003 PASS. sum 0.0276 < plan-022 0.0279. plateau 확인 (B1/B2/B3 모두 0.027x). 2004s. | [DONE — 63469b5] |
| G2.B3 | gate | 3 cell metric finite ✓ + max_class_ratio < 0.95 ✓ (max 0.050) | [DONE — 63469b5] |
| c9 | exp G2.B4 | B4 fib50 — 3 τ_cls cell + `results_B4.{json,md}` 박제 | [TODO] |
| G2.B4 | gate | 3 cell metric finite ✓ + max_class_ratio < 0.95 ✓ | [TODO] |
| c10 | analysis | 12 cell paired Δ 표 + best cell 식별 + N-axis marginal + τ-axis marginal + plan-022 best 대비 compare + paradigm finding → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | ≥ 1 cell paired Δ ≥ +0.005 둘 다 ✓ — band positive (0 통과 시 negative + all_negative warn) | [TODO] |
| c11 | docs | 3-file frontmatter sync (status=all_complete, band=positive|negative, best_sub_exp=B{n}_<name>_tau<val>) + `analysis/plan-023/results.md` (11 항목) + `plans/plan-023-*.results.md` pair + follow-up plan-024/025 박제 | [TODO] |
| G_final | gate | 3-file sync ✓ + §0.5 c1~c11 모두 [DONE] ✓ + follow-up 2건 박제 ✓ | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `f0_reproduce_drift`: G1 reproduce 가 plan-020/021/022 hard evidence 0.6320 / 0.8033 ±0.0005 밖. → halt.
- `lgbm_numerical`: 12 cell 중 어느 LGBM classifier 출력 NaN/Inf. soft label CE / softmax 산출 버그 의심.
- `soft_label_collapse`: 12 cell 중 어느 layout 의 selector probs 가 단일 anchor 에 95% 이상 mass (= `max_class_ratio = probs_all.mean(axis=0).max() > 0.95`). τ_cls 너무 sharp 의심. warn (severe 아님). **G3 분모 영향**: drop 된 cell N_drop 개에 대해 G3 = "(12 − N_drop) cell 중 ≥ 1 PASS_BOTH". cumulative drop OK (각 cell warn 박제 + paradigm_analysis.json `dropped_cells` 리스트 누적). **halt threshold**: N_drop ≥ 7 (= 12 의 절반 초과) 시 severe `soft_label_collapse_majority` escalate (sweep paradigm 의미 상실).
- `frenet_basis_degenerate`: plan-021/022 carry — ‖v_last‖ < 1e-9 또는 ‖a_⊥‖ < 1e-9 sample 비율 > 5%. plan-021 fallback (R_wfn ← I_3) 그대로 적용.
- `samples_per_class_low`: 가장 큰 K=50 의 경우 10000/50=200 samples/class 평균 (uniform 가정). 단일 fold train (8000/50=160) 이 LGBM `num_leaves=63` 위 1 split/leaf ≥ 1 sample 가정 위반 (= class 1 개 이상 train fold 에 0 sample). soft → hard expansion 단계 (§4.3 LgbmSelectorOnly.fit) 에서 dummy injection (weight=1e-6) 이 이미 처리 — 따라서 `samples_per_class_low` 는 warn only (severe 아님). cell-level metric finite 만 검증.
- `all_negative`: 12 cell 모두 paired Δ < +0.005 *둘 다*. → warn 박제 후 G_final (paradigm-level evidence — N>14 lever 의 한계).

### Plan-specific paths (WORKFLOW.md §12.5/§12.6)

- whitelist 추가:
  - `analysis/plan-023/**`
  - `tests/test_plan023_smoke.py`
- blacklist (plan-001~022 산출 자동 변경 금지):
  - `runs/baseline/{B,S,R,P,D,E,F,H,Z020,Z021,Z022}*/**`
  - `analysis/plan-{001..022}/**` (단, **read-only import** 는 §4.3 의 plan-022 / plan-021 / plan-020 module reuse 만 예외)
- 참조 (read-only):
  - `analysis/plan-022/selector_only_model.py:{LgbmSelectorOnly, build_soft_label_with_tau}` — model carry
  - `analysis/plan-022/run_oof.py:run_oof_cell` — per-cell OOF runner carry
  - `analysis/plan-022/anchors.py:LAYOUT_NAMES` — naming schema carry
  - `analysis/plan-022/baseline_carry.json` — dataset hash carry
  - `analysis/plan-021/build_input.py` — 170D input pipeline carry
  - `analysis/plan-020/baseline_oof.json` — F0 0.6320 / 0.8033 hard evidence
  - `analysis/plan-020/baseline_f0.py` — F0 산식
  - `src/pb_0_6822/selector.py:stable_fold_id` — 5-fold split

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — B1 dodecahedron 좌표 = (±1,±1,±1) [8개 cube] ∪ (0,±1/φ,±φ) [4개] ∪ (±1/φ,±φ,0) [4개] ∪ (±φ,0,±1/φ) [4개] 총 20점, φ=(1+√5)/2, scale = 0.005/√3. 모든 vertex norm = 0.005m exact.`
- `decision-note: spec-default — B2 truncated octahedron 좌표 = permutations of (0,±1,±2) 24점, scale = 0.005/√5. 모든 vertex norm = 0.005m exact.`
- `decision-note: spec-default — B3 icosidodecahedron 좌표 = (±φ,0,0) cyclic-perm 6점 ∪ (±1/2,±φ/2,±φ²/2) cyclic-perm 24점 총 30점, scale = 0.005/φ. 모든 vertex norm = 0.005m exact.`
- `decision-note: spec-default — B4 Fibonacci spiral N=50 generator: plan-022 A7 의 fib_sphere 함수 정확 carry, N=12 → N=50 단일 인수 변경. r=0.005m × sphere unit vec.`
- `decision-note: spec-default — τ_cls scan = {0.001, 0.003, 0.005}m × 4 layout = 12 cell. 동시 측정 (selection 아님, 전수 박제). plan-022 §3.5 와 정합.`
- `decision-note: spec-default — LGBM hparam = plan-022 LgbmSelectorOnly 그대로 (n_estimators=500, lr=0.05, num_leaves=63, random_state=20260519). multi-output K-class softmax classifier (K ∈ {20, 24, 30, 50}).`
- `decision-note: spec-default — final_frenet = Σ_a probs[a] · ANCHORS[a] (reg_offset 항 제거, plan-022 carry).`
- `decision-note: spec-default — center anchor 추가 안 함. B1/B2/B3 모두 vertex-transitive — center 추가 시 vertex-transitivity 깨짐. plan-022 finding 중 best (A6_bcc14, A5_cube8) 가 모두 no-center 였다는 evidence 와 정합. B4 (fib50) 도 일관성 위해 no center.`
- `decision-note: spec-default — chiral layout (snub cube, snub dodecahedron) 제외 — handedness 결정 (right vs left) 별도 plan 으로 분리. N=24 슬롯은 truncated octahedron 으로 충당 (rational coord 단순).`

---

## §1. 배경

### §1.1 plan-022 의 finding 과 본 plan 의 응답

| plan-022 finding | 본 plan-023 응답 |
|---|---|
| **A6_bcc14_tau001 = best** (Δ_1cm +0.0208 / Δ_1.5cm +0.0071, sum 0.0279) | K=14 → K∈{20,24,30,50} 확장 시 추가 향상 가능성 측정 |
| **layout marginal**: A6 (14) > A7 (13) > A3 (13) > A5 (8) > A2 (13) > A4 (13) > A1 (7) | K=14 = 측정한 K 의 max. K>14 단조 향상 trend extrapolate 검증 |
| **τ_cls marginal**: τ=0.001 (sharp) 가 모든 layout 에서 best 또는 동률 | 본 plan 도 동일 3 τ 측정 — K↑ 시 sharp τ saturate point 측정 |
| **mode collapse 완화**: A6 의 `max_class_ratio=0.105` 가 가장 distributed | K=50 fib 의 max_class_ratio 예상 1/50=0.02 — collapse 완화 trend 확장 측정 |
| **Δ_1.5cm 만 보면 marginal (+0.0071)**: 1cm 의 +0.0208 의 약 1/3 | N>14 추가 lever 로 1.5cm hit-zone 추가 향상 시도 — 본 plan 의 핵심 motiv |

### §1.2 사용자 narrative — large-N (>14) sweep ("14N+" 세션)

plan-022 의 7 layout 은 K∈{7,8,13,14} 범위 — Platonic (octahedron 7, cube 8, icosahedron 13) + Archimedean (cuboctahedron 13) + 결합형 (BCC 14, 2-shell 13) + quasi-uniform (Fibonacci 13). **K>14 = 미측정 영역**. 사용자 narrative (2026-05-19 세션 "14N+") = "plan-022 winner 의 K=14 위 K↑ trend 측정". 본 plan-023 = **그 K↑ extrapolation 의 본격 실행**.

**왜 K∈{20, 24, 30, 50}?** plan-023 §0 의 catalog 분석 결과:
- **N=20 dodecahedron**: 14→20 자연 step-up, samples/class = 500 (안전), Platonic 5-fold sym
- **N=24 truncated octahedron**: 가장 isotropic 24-point Archimedean (vertex-transitive), samples/class = 417 (안전), rational coord (구현 단순)
- **N=30 icosidodecahedron**: 5-fold + 3-fold sym, 30-point spherical t-design 후보, samples/class = 333 (안전)
- **N=50 Fibonacci**: N upper-end risk zone probe, samples/class = 200 (warn 박제), quasi-uniform 비교

**N≥60 제외 이유**: samples/class < 200 → `samples_per_class_low` warn + LGBM K-class softmax 학습 통계량 약화 위험 (plan-022 의 `soft_label_collapse` warn trigger zone 진입 risk).

### §1.3 가설

**H1 (K↑ layout 효과)**: K∈{20,24,30} (vertex-transitive Archimedean/Platonic) 이 K=14 (plan-022 A6 BCC) 대비 paired Δ_1.5cm 더 큼 (1.5cm hit-zone coverage 더 finer grid).
**H2 (K↑ saturate point)**: K=50 Fibonacci 가 K=30 보다 못함 — samples/class 감소 (333→200) + mode collapse 완화 너무 강해 prob mass diffuse 가 1cm metric 손실 yield.
**H3 (sharp τ saturate point)**: K↑ 시 τ=0.001 의 effective temperature 가 K=1/K 의 평균 prob 와 어울려 K↑ → 같은 τ 가 더 sharp. 따라서 K=50 위 τ=0.001 cell 이 K=20 위 τ=0.001 보다 더 mode collapse 또는 max_class_ratio 변화 측정 가능.
**H4 (plan-022 best 갱신)**: 12 cell 중 ≥ 1 cell 의 Δ sum > 0.0279 (plan-022 A6_bcc14_tau001 sum).

### §1.4 baseline 두 layer

- **G3 합격용 baseline**: F0 (plan-022 동일, paired Δ ≥ +0.005 둘 다)
- **paradigm-level compare**: plan-022 best A6_bcc14_tau001 (Δ_1cm +0.0208 / Δ_1.5cm +0.0071, sum 0.0279) — 본 plan G_final results.md 의 §3 finding 박제 용

만약 12 cell 모두 plan-022 best 보다 Δ sum 못 미치면 → **N>14 lever 자체가 saturate** 라는 paradigm-level evidence (negative finding 도 valuable, follow-up plan-024/025 motiv).

---

## §2. 가설 검증 paradigm (한 변수 원칙)

| 변수 | 차원 | 값 |
|---|---|---|
| **anchor layout** | 4 | B1 dodeca20, B2 trunc_octa24, B3 icosidodec30, B4 fib50 |
| **τ_cls** | 3 | 0.001, 0.003, 0.005 (단위 m, plan-022 §3.5 동일) |
| 그 외 모두 (input 170D, LGBM hparam, fold split, soft label 산식 frame, final composition 식, F0 baseline) | 1 | plan-022 §3 / §6 default 그대로 |

→ **셀 (B{n}, τ)** 마다 plan-022 baseline 대비 **anchor 좌표 만** 변경 (τ_cls scan 은 plan-022 와 동일 범위 — 같은 τ 값 cross-plan 비교 가능). plan-022 §2 의 한 변수 원칙 동일 적용.

**정확 framing**: 본 plan 은 (anchor layout, τ_cls) **2-factor factorial sweep** (4 × 3 = 12 cell). plan-022 §2 의 "한 변수 원칙" framing 을 carry 하나, *cell 단위* 로 보면 2 변수 동시 변경. *hypothesis 단위* 로 보면 각 H{n} 검증 시 한 axis 를 marginalize:
- **H1 (K↑ layout 효과)** 검증: τ_cls axis marginalize (각 layout 의 best τ cell 선택) → 4 layout 의 paired Δ trend 비교 (§7.3 layout axis marginal).
- **H2 (K↑ saturate)** 검증: H1 의 K-trend 가 K=30 → K=50 사이 non-monotonic 인지 측정 (specifically `Δ_sum(B3) > Δ_sum(B4)` AND `Δ_1cm(B4) < Δ_1cm(B3)` 둘 다 만족 → saturate 박제).
- **H3 (sharp τ saturate point)** 검증: layout axis marginalize (각 τ_cls 의 best layout cell 선택) → 3 τ 의 paired Δ trend (§7.3 τ_cls axis marginal). 추가로 K=50 의 τ=0.001 cell `max_class_ratio` 가 K=20 의 τ=0.001 cell 대비 1/2 이하인지 검증 (effective temperature 변화 정량) — PASS/FAIL 결과는 §8.1 results.md 항목 #6 (mode collapse 완화 finding) 안에 박제.
- **H4 (plan-022 best 갱신)** 검증: 12 cell 중 ≥ 1 cell 의 `delta_1cm + delta_1.5cm > 0.0279` (§7.3 `compare_with_plan022_best` output 의 `n_cells_beating_sum >= 1`).

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021/022 carry)

| 분할 | 값 |
|---|---|
| folds | 5 |
| fold 할당 | `stable_fold_id(str(sample_id), 5)` (plan-020 carry, MD5 32-bit prefix mod 5) |
| seed | fold split deterministic (no seed) |
| F0 baseline | plan-020 `baseline_oof.json` 의 집계 metric (0.6320 / 0.8033) sanity carry. per-sample F0 pred 는 run-time 재계산 (`bf.f0_baseline(X, end_idx=10)`, < 1s). |

### §3.2 합격 기준 (정량)

- **G0**: 2 모듈 (anchors_largeN / run_oof_largeN) import + smoke + tests green
- **G1**: F0 reproduce hit@1cm ∈ [0.6315, 0.6325] AND hit@1.5cm ∈ [0.8028, 0.8038]
- **G2.B{n}** (n ∈ 1..4): sub-exp B{n} 의 3 τ_cls cell 모두 metric finite + max-class 비율 < 0.95 (soft_label_collapse 회피)
- **G3**: 12 cell 중 ≥ 1 개가 paired Δ_1cm ≥ +0.005 AND paired Δ_1.5cm ≥ +0.005
  - 0 통과 시 → `all_negative` warn 박제 후 G_final 직진

### §3.3 평가 점수

plan-022 §3.3 정확 carry:

| metric | 식 | 비교 |
|---|---|---|
| hit@1cm | `mean(‖final_world − gt‖₂ ≤ 0.01)` | F0 baseline 0.6320 |
| hit@1.5cm | `mean(‖final_world − gt‖₂ ≤ 0.015)` | F0 baseline 0.8033 |
| paired Δ | sample-level: `mean_i(1{‖pred_cell_i − gt_i‖ ≤ R} − 1{‖pred_F0_i − gt_i‖ ≤ R})`. 5-fold concat OOF. **`pred_cell_i` = 단일 (layout, τ_cls) cell** — 12 cell 각각 산출. `pred_F0_i` 산출 = `p022_run.run_oof_cell` 내부의 `common["pred_F0_world"]` (= `bf.f0_baseline(X, end_idx=10)` 의 frenet→world 변환, plan-022 §6.2 carry). 호출자 측 별도 계산 불필요. | +0.005 임계 적용 |
| max-class 비율 | `max_a (mean_i probs_i[a])` (= soft-mean 의 최대값) | < 0.95 (collapse 회피 임계) |
| fold variance | per-fold metric std | informational only (G3 binding 없음) |
| pass_both | `bool(delta_1cm >= 0.005 and delta_1.5cm >= 0.005)` (plan-022 `run_oof_cell` output schema carry, self-contained 박제) | G3 합격 기준 (≥ 1 cell PASS) |

### §3.4 Anchor layouts (4 codebook, 전수 측정)

각 layout 의 좌표 정의 (Frenet, 단위 m). **Anchor index 일관성**: 모든 layout 이 vertex-transitive 또는 quasi-uniform 으로 center 없음 → index 0 = generator 의 첫 vertex (deterministic order). plan-022 의 center-bearing layout (A1/A2/A3/A4/A7) 과 cross-plan index 매핑 불가능 — 본 plan 내부 일관성 만 보장.

#### B1: dodecahedron (20) — Platonic
20 dodecahedron vertices = (±1, ±1, ±1) 8개 cube vertex + (0, ±1/φ, ±φ), (±1/φ, ±φ, 0), (±φ, 0, ±1/φ) 12개 rectangle vertex. φ = (1+√5)/2. 모든 20점 `‖a‖ = √3` (unscaled). scale = 0.005 / √3 ≈ 0.002887.

```python
import numpy as np, itertools
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339...
INV_PHI = 1 / PHI            # 0.6180339... = φ - 1

# 8 cube vertices
cube8 = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float64)

# 12 rectangle vertices — 3 cyclic permutation groups × 4 sign combos each
rect12 = []
for sy, sz in itertools.product([-1, 1], repeat=2):
    rect12.append((0,        sy * INV_PHI, sz * PHI    ))     # (0, ±1/φ, ±φ)
    rect12.append((sy * INV_PHI, sz * PHI, 0           ))     # (±1/φ, ±φ, 0)
    rect12.append((sz * PHI, 0,            sy * INV_PHI))     # (±φ, 0, ±1/φ)
rect12 = np.array(rect12, dtype=np.float64)

DODECA20 = np.vstack([cube8, rect12])              # (20, 3) unscaled
ANCHORS_B1 = (DODECA20 * (0.005 / np.sqrt(3))).astype(np.float32)  # (20, 3), ‖a‖ = 0.005m exact
```

#### B2: truncated octahedron (24) — Archimedean
24 vertices = all signed permutations of (0, ±1, ±2). 모든 24점 `‖a‖ = √5` (unscaled). scale = 0.005 / √5 ≈ 0.002236.

```python
# 24 vertices = position-of-0 (3 choices) × choice of {±1, ±2} for other 2 positions (2) × signs (2 × 2)
verts = []
for zero_pos in range(3):                                      # which position is 0
    other = [i for i in range(3) if i != zero_pos]
    for swap in [(1, 2), (2, 1)]:                              # which of others is ±1 vs ±2
        for s1, s2 in itertools.product([-1, 1], repeat=2):
            v = [0.0, 0.0, 0.0]
            v[other[0]] = s1 * swap[0]
            v[other[1]] = s2 * swap[1]
            verts.append(tuple(v))
TRUNC_OCTA24 = np.array(verts, dtype=np.float64)               # (24, 3) unscaled
ANCHORS_B2 = (TRUNC_OCTA24 * (0.005 / np.sqrt(5))).astype(np.float32)  # ‖a‖ = 0.005m exact
```

#### B3: icosidodecahedron (30) — Archimedean
30 vertices = 6 axis ((±φ, 0, 0) cyclic perm) + 24 (±1/2, ±φ/2, ±φ²/2) cyclic perm. 모든 30점 `‖a‖ = φ` (unscaled). scale = 0.005 / φ ≈ 0.003090.

```python
PHI2 = PHI * PHI                                  # φ² = φ + 1

# 6 axis vertices: (±φ, 0, 0) and cyclic
axis6 = np.array([
    (+PHI, 0, 0), (-PHI, 0, 0),
    (0, +PHI, 0), (0, -PHI, 0),
    (0, 0, +PHI), (0, 0, -PHI),
], dtype=np.float64)

# 24 vertices: (±1/2, ±φ/2, ±φ²/2) and 2 cyclic permutations
cyclic24 = []
for sx, sy, sz in itertools.product([-1, 1], repeat=3):
    cyclic24.append((sx * 0.5,    sy * PHI/2, sz * PHI2/2))
    cyclic24.append((sz * PHI2/2, sx * 0.5,   sy * PHI/2))    # cyclic shift 1
    cyclic24.append((sy * PHI/2,  sz * PHI2/2, sx * 0.5))     # cyclic shift 2
cyclic24 = np.array(cyclic24, dtype=np.float64)                # (24, 3)

ICOSIDODEC30 = np.vstack([axis6, cyclic24])        # (30, 3) unscaled
ANCHORS_B3 = (ICOSIDODEC30 * (0.005 / PHI)).astype(np.float32)  # ‖a‖ = 0.005m exact
```

#### B4: Fibonacci spiral N=50 — quasi-uniform
plan-022 A7 의 `fib_sphere` generator 정확 carry, N=12 → N=50.

```python
def fib_sphere(N, r):
    phi_g = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * np.arange(N) / phi_g            # azimuth
    z = 1 - 2 * (np.arange(N) + 0.5) / N                # latitude (uniform in z)
    rho = np.sqrt(1 - z**2)
    return np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1) * r

ANCHORS_B4 = fib_sphere(50, 0.005).astype(np.float32)    # (50, 3) — vertex-transitive 아니지만 jittered uniform
```

각 layout invariant (`anchors_largeN.py` smoke test):
- `ANCHORS_B{n}.dtype == np.float32`
- `np.linalg.norm(ANCHORS_B{n}, axis=1).max() <= 0.005 + 1e-7`
- B1/B2/B3: `np.linalg.norm(...).std() <= 1e-6` (vertex-transitive — 모든 norm 동일)
- B4: `np.linalg.norm(...).max() <= 0.005 + 1e-7` (단위 sphere 위 — 모든 norm = r=0.005m exact)
- 좌표값 finite + shape `(N, 3)` with N ∈ {20, 24, 30, 50}
- **unique vertex 검증** (silent duplicate 차단): `np.unique(ANCHORS_B{n}, axis=0).shape[0] == N` for N ∈ {20, 24, 30, 50}. duplicate 생성 시 K_actual ≠ N silent bug → 즉시 fail. B1/B2/B3 의 cyclic permutation generator (각 § 좌표 정의 코드 참고) 가 의도된 N point 정확 산출 검증.

### §3.5 τ_cls scan

plan-022 §3.5 정확 carry:

| τ_cls (m) | anchor radius r=0.005m 대비 ratio (τ_cls / r) | 의미 |
|---|---|---|
| 0.001 | 1/5 = 0.2 | plan-021/022 default — sharp (q 거의 one-hot) |
| 0.003 | 3/5 = 0.6 | mid — q 가 인접 anchor 에 약간 spread |
| 0.005 | 1/1 = 1.0 | loose — q 가 다중 anchor 에 smooth |

각 (layout, τ) cell 에서 **soft label 계산만 τ_cls 영향** (model hparam / input / loss 동일). soft label 산식 = plan-022 `build_soft_label_with_tau` 직접 import — 본 plan 에서 새 산식 정의 없음.

**K↑ 시 effective temperature 변화 주의** (§1.3 H3): K=7 → K=50 으로 anchor 가 7배 늘면, 동일 τ_cls 위 softmax 의 effective sharpness 가 변함. 본 plan 은 plan-022 와 동일 τ 값 측정으로 cross-plan 비교 가능성 보장. K↑ 시 sharp 변화는 informational 분석 (§7.3).

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-023/
├── anchors_largeN.py            # 4 layout numpy 상수 + smoke (ANCHORS_B1 ... B4 + LAYOUT_NAMES_B)
├── run_oof_largeN.py            # 12-cell sweep runner (plan-022 run_oof_cell import + 4 layout × 3 τ loop)
├── paradigm_analysis.py         # c10 G3 entry — collect_cells / select_best / marginals / plan-022 compare
├── results_B{n}.{json,md}       # 4 sub-exp results (각 3 τ cell 포함)
├── paradigm_analysis.{json,md}  # c10 G3 — 12 cell 표 + marginals + plan-022 compare + finding
├── baseline_carry.json          # F0 0.6320 / 0.8033 + dataset hash (plan-022 carry)
└── results.md                   # G_final synthesis
```

### §4.2 module top-level export (smoke test lock-in)

| symbol | module | type |
|---|---|---|
| `ANCHORS_B1` .. `ANCHORS_B4` | anchors_largeN | `np.ndarray` (K, 3), K ∈ {20, 24, 30, 50}, float32 |
| `LAYOUT_NAMES_B` | anchors_largeN | `dict[str, np.ndarray]` — {"B1_dodeca20": ANCHORS_B1, "B2_trunc_octa24": ANCHORS_B2, "B3_icosidodec30": ANCHORS_B3, "B4_fib50": ANCHORS_B4} |
| `run_sweep_largeN` | run_oof_largeN | `Callable[[X, Y, folds], dict[str, dict]]` — 12 cell 전수 |
| `collect_cells` | paradigm_analysis | `Callable[[Path], dict[str, dict]]` — 4 `results_B{n}.json` load → flatten 12 cell dict |
| `select_best` | paradigm_analysis | `Callable[[dict[str, dict]], tuple[str, dict]]` — `(pass_both, Δ_1cm + Δ_1.5cm, -cell_key alphabetic)` 우선순위 |
| `marginals` | paradigm_analysis | `Callable[[dict[str, dict]], dict]` — layout-axis (4 row) + τ-axis (3 row) best |
| `compare_with_plan022_best` | paradigm_analysis | `Callable[[dict[str, dict]], dict]` — plan-022 A6_bcc14_tau001 (Δ_1cm +0.0208 / Δ_1.5cm +0.0071, sum 0.0279) 박제값 대비 본 plan 12 cell 비교. 반환 dict schema: `{"plan022_best": {"delta_1cm": 0.0208, "delta_1.5cm": 0.0071, "sum": 0.0279}, "cells_beating_sum": [{"cell_key": str, "delta_1cm": float, "delta_1.5cm": float, "sum": float, "delta_vs_plan022_sum": float}, ...], "cells_beating_1cm": [...same schema, filtered by Δ_1cm > 0.0208...], "cells_beating_1.5cm": [...filtered by Δ_1.5cm > 0.0071...], "n_cells_beating_sum": int, "best_cell_in_plan023": cell_key_str}` — §8.1 results.md 항목 #7 input |

→ AttributeError 시 G0 `infra_drift` severe.

### §4.3 plan-022 module reuse

**전제** (G0 smoke 검증): plan-022 `selector_only_model.py` / `run_oof.py` 는 plan-021 `build_input.py` / `dual_head_model.py` reuse 시 `importlib.util.spec_from_file_location` 패턴 사용 (plan-022 §4.3 carry) — package-relative import 없음. 본 plan-023 도 동일 패턴 적용.

```python
# analysis/plan-023/run_oof_largeN.py 상단
import importlib.util, sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent              # analysis/plan-023/
_REPO = _THIS.parent.parent                          # REPO root
_PLAN020 = _THIS.parent / "plan-020"                  # analysis/plan-020/
_PLAN021 = _THIS.parent / "plan-021"                  # analysis/plan-021/
_PLAN022 = _THIS.parent / "plan-022"                  # analysis/plan-022/
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_PLAN021))
sys.path.insert(0, str(_PLAN022))

# plan-020 baseline_f0 (F0 산식)
spec = importlib.util.spec_from_file_location("bf", _PLAN020 / "baseline_f0.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)

# plan-021 build_input
spec = importlib.util.spec_from_file_location("p021_build", _PLAN021 / "build_input.py")
p021_build = importlib.util.module_from_spec(spec); spec.loader.exec_module(p021_build)

# plan-022 selector_only_model + run_oof (= 본 plan 의 핵심 carry)
spec = importlib.util.spec_from_file_location("p022_sel", _PLAN022 / "selector_only_model.py")
p022_sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(p022_sel)
# p022_sel.LgbmSelectorOnly, p022_sel.build_soft_label_with_tau

spec = importlib.util.spec_from_file_location("p022_run", _PLAN022 / "run_oof.py")
p022_run = importlib.util.module_from_spec(spec); spec.loader.exec_module(p022_run)
# p022_run.run_oof_cell (per-cell 5-fold OOF runner)
```

→ `run_oof_largeN.py` 의 `run_sweep_largeN` 함수는 `p022_run.run_oof_cell(X, Y, folds, ANCHORS_B{n}, tau)` 를 12회 호출 (4 layout × 3 τ).

### §4.4 (LgbmSelectorOnly + build_soft_label_with_tau 는 plan-022 carry — 본 plan 변경 X)

plan-022 §4.4 spec 정확 carry. 본 plan 에서 신규 class 정의 없음. K ∈ {20, 24, 30, 50} 의 가변 parameter 만 `p022_sel.LgbmSelectorOnly(K=K)` constructor 인자 변경.

### §4.5 tests (c4)

`tests/test_plan023_smoke.py` 7 항목:
1. `sys.path.insert(0, "analysis/plan-023")` 후 `import anchors_largeN, run_oof_largeN` (plan-022 §4.5 test 1 의 sys.path + 직접 module name import 패턴 정확 carry — `analysis/plan-023/` 디렉토리의 hyphen 이 Python 모듈명 syntax 와 충돌 회피). ImportError / AttributeError 0건.
2. 4 layout 각각: `dtype == float32`, `‖a‖ ≤ 0.005m + 1e-7`, shape `(K, 3)`, K ∈ {20, 24, 30, 50} 정합
3. B1/B2/B3 vertex-transitive invariant: `np.linalg.norm(ANCHORS_B{n}, axis=1).std() <= 1e-6`. B4 (fib50) 도 단위 sphere 위 — `np.linalg.norm(ANCHORS_B4, axis=1).std() <= 1e-6` 검증 (모두 0.005m 단일 shell).
4. `LAYOUT_NAMES_B` dict 4 항목, key 정확 ("B1_dodeca20", "B2_trunc_octa24", "B3_icosidodec30", "B4_fib50")
5. plan-022 module reuse smoke: `from numpy.random import rand; from scipy.special import softmax as _softmax; _model = p022_sel.LgbmSelectorOnly(K=50); _model.fit(rand(200, 170), _softmax(rand(200, 50), axis=1))` — `fit` 는 `self` 반환 (plan-022 §4.4 LgbmSelectorOnly carry). 이어서 `probs = _model.predict(rand(100, 170))` — output shape `(100, 50)` + finite + row-sum=1 (K=50 max 검증).
6. `p022_sel.build_soft_label_with_tau` 출력 shape `(N, K)` + sum=1 + finite (3 τ 값 × 4 layout = 12 조합 검증)
7. F0 reproduce sanity (plan-020/021/022 baseline_oof.json carry, 0.6320 / 0.8033) + samples-per-class lower-bound 검증 (10000/50 = 200 floor warn-only).

→ 7/7 pass 시 G0 PASS.

---

## §5. STAGE 1 — F0 baseline reproduce (c5, G1)

### §5.1 실행

plan-022 §5.1 carry — `baseline_carry.py` 단순 변형 (plan-022 의 `baseline_carry.json` 의 `dataset_hash` 를 직접 carry).

```python
# analysis/plan-023/baseline_carry.py
import json, hashlib, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from src.io import load_all_samples

# (a) plan-020 metric carry
baseline = json.loads((REPO / "analysis/plan-020/baseline_oof.json").read_text())
f0 = baseline["f0_baseline"]
assert 0.6315 <= f0["hit_1cm_5fold_concat"] <= 0.6325, f"f0 1cm drift: {f0['hit_1cm_5fold_concat']}"
assert 0.8028 <= f0["hit_1.5cm_5fold_concat"] <= 0.8038, f"f0 1.5cm drift: {f0['hit_1.5cm_5fold_concat']}"

# (b) plan-022 baseline_carry.json 의 dataset_hash carry — 동일 dataset 인지 검증
p022_carry = json.loads((REPO / "analysis/plan-022/baseline_carry.json").read_text())
expected_hash = p022_carry["dataset_hash"]

ids, X = load_all_samples(split="train")
# Hash 산식 = plan-022 §5.1 carry (sha256 of "|".join(sorted-by-str id list), first 16 hex chars)
# plan-022 baseline_carry.json 의 dataset_hash 가 동일 산식으로 생성되어 있다고 박제 (cross-plan 정합 보장).
data_hash = hashlib.sha256(("|".join(sorted(map(str, ids)))).encode()).hexdigest()[:16]
assert data_hash == expected_hash, f"dataset shift: {data_hash} != {expected_hash}"

(REPO / "analysis/plan-023/baseline_carry.json").write_text(json.dumps({
    "f0_hit_1cm": f0["hit_1cm_5fold_concat"],
    "f0_hit_1.5cm": f0["hit_1.5cm_5fold_concat"],
    "dataset_hash": data_hash,
    "n_samples": len(ids),
    "source_baseline": "analysis/plan-020/baseline_oof.json",
    "carry_from": "analysis/plan-022/baseline_carry.json",
}, indent=2))
```

산출: `analysis/plan-023/baseline_carry.json`.

### §5.2 G1 합격 (자동)

- plan-020 baseline_oof.json metric ±0.0005 안 통과
- plan-022 dataset_hash 와 정확 일치
- 위반 시 `f0_reproduce_drift` severe

---

## §6. STAGE 2 — Sub-exp B1~B4 (c6~c9, G2.B1~G2.B4)

### §6.1 Input spec (= plan-022 §6.1 / plan-021 §6.1, 170D 그대로)

plan-022 §6.1 정확 carry. 본 plan code 변경 X.

| 채널 | dim | source |
|---|---|---|
| L1 Frenet trajectory | 99 | plan-021 §6.1.1 |
| L2 F0 residual sequence | 21 | plan-021 §6.1.2 |
| L4 F0 soft hit sequence | 14 | plan-021 §6.1.3 |
| L5 macro statistic | 9 | plan-021 §6.1.4 |
| L6 EWMA | 27 | plan-021 §6.1 ewma_last |
| **total** | **170** | |

→ `p021_build.build_input_common(X, bf.f0_baseline)` + `p021_build.build_input_lgbm_extra(X, L1)` (plan-022 §6.1 동일).

**X / Y / folds build site (caller 측 plan-023 진입점 spec — `run_oof_largeN.py:__main__` block 또는 G2.B{n} 실행 script)**:
```python
import numpy as np
from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import stable_fold_id

ids, X = load_all_samples(split="train")                     # ids: (N,), X: (N, 11, 3) float64
labels = load_labels()                                       # dict[sample_id, np.ndarray (3,)] — t=12 ground truth
Y = np.stack([labels[s] for s in ids], axis=0).astype(np.float64)  # (N, 3) world frame
folds = np.array([stable_fold_id(str(s), 5) for s in ids], dtype=np.int64)  # (N,) ∈ {0..4}

# 이후 모든 sub-exp 가 동일 X, Y, folds 재사용 (plan-022 §5.1 / §6.2 carry pattern).
cells = run_sweep_largeN(X, Y, folds)                        # 12-cell sweep
```
N = ids 개수 (= plan-022 carry, baseline_carry.json `n_samples` field 일치). plan-022 의 동일 patten 재사용.

### §6.2 Per-cell 5-fold OOF 식

plan-022 §6.2 `run_oof_cell` 정확 carry — 본 plan 의 `run_sweep_largeN` 은 12회 호출 wrapper:

```python
# analysis/plan-023/run_oof_largeN.py 본체
def run_sweep_largeN(X, Y, folds) -> dict[str, dict]:
    """4 layout × 3 τ_cls = 12 cell 전수 측정.

    Returns: dict, key = f"{layout_id}_tau{int(tau*1000):03d}"
             value = p022_run.run_oof_cell 반환 dict + {"layout": layout_id, "tau_cls": tau}

    `p022_run.run_oof_cell` signature carry (plan-022 §6.2):
        run_oof_cell(
            X: np.ndarray,        # (N, 11, 3) float64 — raw 11-step world-frame trajectory
            Y: np.ndarray,        # (N, 3)     float64 — ground-truth at t=12, world frame
            folds: np.ndarray,    # (N,)       int     — fold id ∈ {0..4}
            anchors: np.ndarray,  # (K, 3)     float32 — Frenet 좌표
            tau_cls: float,       #            float   — softmax temperature, m
        ) -> dict
    """
    cells = {}
    for layout_id, anchors in LAYOUT_NAMES_B.items():
        for tau in [0.001, 0.003, 0.005]:
            cell = p022_run.run_oof_cell(X, Y, folds, anchors, tau)
            cell_key = f"{layout_id}_tau{int(tau*1000):03d}"
            cells[cell_key] = {**cell, "layout": layout_id, "tau_cls": tau}
    return cells
```

### §6.3 Per-sub-exp 실행 (c6~c9)

각 c{i} (i=6..9) 는 하나의 anchor layout 의 3 τ_cls cell 측정:

```python
# c6 예시 (B1_dodeca20):
results_B1 = {}
for tau in [0.001, 0.003, 0.005]:
    cell = p022_run.run_oof_cell(X, Y, folds, ANCHORS_B1, tau)
    results_B1[f"tau_{tau:.3f}"] = cell

# JSON 저장: analysis/plan-023/results_B1.json
# MD 저장: analysis/plan-023/results_B1.md (3 cell 표)
```

**소요 시간 예상**:
- plan-022 K=14 (A6) — 단일 sub-exp (3 cell) ≈ 817s
- K↑ 시 LGBM K-class softmax 학습 시간 ∝ K. K=20/24/30/50 단일 sub-exp 예상:
  - B1 (K=20): ≈ 817s × (20/14) ≈ 1170s ≈ 20분
  - B2 (K=24): ≈ 1400s ≈ 23분
  - B3 (K=30): ≈ 1750s ≈ 29분
  - B4 (K=50): ≈ 2920s ≈ 49분
- **총 예상**: 4 sub-exp 합 ≈ 7240s ≈ 121분 ≈ 2시간 (sequential, parallel 안 함, 박제).

### §6.4 G2.B{n} 합격 (per sub-exp)

plan-022 §6.4 정확 carry:
- 3 cell 모두 metric finite (NaN/Inf 없음)
- 3 cell 모두 `max_class_ratio < 0.95` (soft_label_collapse 회피)
- 위반 시:
  - finite fail → `lgbm_numerical` severe
  - max_class_ratio ≥ 0.95 → `soft_label_collapse` warn (해당 cell drop, 나머지 진행)

**Drop schema (`results_B{n}.json` 표시 방법)**: drop 된 cell 의 entry 는 `results_B{n}["tau_{tau:.3f}"]` key 에 다음 sentinel dict 로 작성:
```python
{
  "K": K, "tau_cls": tau,
  "dropped": True,
  "drop_reason": "soft_label_collapse",        # 또는 "lgbm_numerical" (severe 시 entry 자체 없음 — halt 후 plan 중단)
  "max_class_ratio": <float ≥ 0.95>,
  "hit_1cm": None, "hit_1.5cm": None,
  "delta_1cm": None, "delta_1.5cm": None,
  "pass_both": False,
}
```
`collect_cells` (§7.1) 는 entry 의 `.get("dropped", False)` 검사로 drop cell 식별. `select_best` (§7.2) 는 `pass_both` False + delta `None` 이라 자동 후순위 (max 비교 시 `None < float` 비교 회피 위해 drop cell 은 `cells_pass_both = {k: v for k, v in cells.items() if not v.get("dropped", False)}` 로 필터 후 max). `paradigm_analysis.json` 에 `dropped_cells: [cell_key, ...]` 박제.

---

## §7. STAGE 3 — Paradigm analysis (c10, G3)

### §7.1 12 cell 표 산출

```python
# analysis/plan-023/paradigm_analysis.py
import json
from pathlib import Path

def collect_cells(plan_dir: Path) -> dict[str, dict]:
    """4 results_B{n}.json 을 flatten → 12 cell dict.

    cell_key 포맷: f"{layout_id}_tau{int(tau*1000):03d}"
                   예: "B1_dodeca20_tau001", "B2_trunc_octa24_tau003", "B4_fib50_tau005"
    """
    LAYOUT_IDS = ["B1_dodeca20", "B2_trunc_octa24", "B3_icosidodec30", "B4_fib50"]
    cells: dict[str, dict] = {}
    for layout_id in LAYOUT_IDS:
        sub_exp_key = layout_id.split("_")[0]                  # "B1", "B2", ...
        path = plan_dir / f"results_{sub_exp_key}.json"
        results_sub_exp = json.loads(path.read_text())
        for tau_key, cell in results_sub_exp.items():
            tau_val = float(tau_key.replace("tau_", ""))
            cell_key = f"{layout_id}_tau{int(tau_val*1000):03d}"
            cells[cell_key] = {**cell, "layout": layout_id, "tau_cls": tau_val}
    return cells
```

### §7.2 Best cell selection

plan-022 §7.2 정확 carry:

```python
def select_best(cells: dict[str, dict]) -> tuple[str, dict]:
    # drop cell (§6.4 sentinel) 은 비교 후보에서 제외 — delta_* = None 비교 회피.
    eligible = {k: v for k, v in cells.items() if not v.get("dropped", False)}
    best_key, best_cell = max(eligible.items(), key=lambda kv: (
        kv[1]["pass_both"],                                # priority 1: True > False
        kv[1]["delta_1cm"] + kv[1]["delta_1.5cm"],         # priority 2: sum 큰 cell 선호
        tuple(-ord(c) for c in kv[0]),                     # priority 3: cell_key alphabetic asc
        # — max(...) + -ord 조합으로 ASCII 작은 char (= 알파벳 앞쪽) 가 win
        # — 본 plan 의 cell_key prefix 가 B1/B2/B3/B4 + layout name 으로 unique → 길이-tuple 비교 fragility 무관
    ))
    return best_key, best_cell
```

### §7.3 Marginal 분석 + plan-022 compare

- **layout axis marginal**: 4 layout 의 3 τ_cls cell 중 best τ 선택 → 4 layout 비교 표 (K axis trend). "best τ" 선정 metric = §7.2 `select_best` 의 priority 동일 (`pass_both` desc, `delta_1cm + delta_1.5cm` desc, cell_key alphabetic asc tiebreak).
- **τ_cls axis marginal**: 3 τ_cls 의 4 layout cell 중 best layout 선택 → 3 τ 비교 표. "best layout" 선정 metric = 위와 동일 priority.
- **mode collapse 완화 측정**: 각 cell 의 `max_class_ratio` 와 K 의 관계 — uniform 가정 시 1/K, 실측 max_class_ratio 가 1/K 의 몇 배인지 박제. output schema: `paradigm_analysis.json["mode_collapse_table"]` = `[{"cell_key": str, "K": int, "max_class_ratio": float, "uniform_baseline": 1/K, "ratio_to_uniform": max_class_ratio / (1/K)}, ...]` (12 cell row). §1.3 H3 검증 결과 (K=50 τ=0.001 의 max_class_ratio ≤ K=20 τ=0.001 의 max_class_ratio × 0.5 PASS 여부) 도 같은 dict 의 `["h3_check"]` 필드에 `{"k50_tau001": float, "k20_tau001": float, "ratio": float, "pass": bool}` schema 로 박제.
- **plan-022 best 대비 compare**: `compare_with_plan022_best(cells)` 가 12 cell 중 plan-022 A6_bcc14_tau001 (Δ_1cm +0.0208 / Δ_1.5cm +0.0071, sum 0.0279) 박제값 초과 cell 박제

### §7.4 G3 합격

- 12 cell 중 ≥ 1 개가 paired Δ_1cm ≥ +0.005 AND Δ_1.5cm ≥ +0.005 → PASS, band positive
- 0 통과 → `all_negative` warn 박제, band negative, G_final 직진

---

## §8. STAGE 4 — G_final (c11)

### §8.1 산출

- `analysis/plan-023/results.md` (필수 11 항목):
  1. plan-022 → plan-023 narrative bridge (K=14 winner → K>14 extrapolation)
  2. 12 cell 표 (layout × τ_cls grid)
  3. best cell 박제 (layout + τ + metrics)
  4. layout axis marginal (K↑ trend)
  5. τ_cls axis marginal
  6. mode collapse 완화 finding (K vs max_class_ratio)
  7. plan-022 A6_bcc14_tau001 대비 향상 cell 박제 (또는 negative finding)
  8. paradigm-level finding (K>14 lever 의 효과 + saturate point 가능성)
  9. follow-up plan 후보 (plan-024 corrector reg head 재투입 / plan-025 N × radius shell 2D)
  10. severe/warn 박제 (`all_negative` 또는 `samples_per_class_low` 발동 여부 등)
  11. dataset hash + reproducibility 박제 (plan-022 dataset_hash carry 검증)

- frontmatter sync (plan + results + 본 plan 의 §0.5):
  - `status: all_complete | partial | failed`
  - `best_sub_exp: B{n}_<name>_tau<val>` 또는 `null`
  - `band: positive | negative | null`
  - `best_hit_1cm, best_hit_1.5cm, best_delta_1cm, best_delta_1.5cm` — `best_sub_exp != null` 일 때 best cell 의 4 metric 값 박제. **null 케이스**: `band: negative` (G3 0 통과) 또는 12 cell 모두 drop 시 4 metric field 모두 `null` (best_sub_exp 와 동시 null, frontmatter 일관). `band: positive` 인데 metric null = invalid 박제 (3-file sync 위반).
  - `lb_score: null` (out-of-scope)

### §8.2 G_final 합격

- 3-file frontmatter sync (= WORKFLOW.md §11 의 plan ↔ results ↔ registry 3-축 일치) 정확:
  - **file 1**: `plans/plan-023-large-n-anchor-sweep.md` 의 frontmatter
  - **file 2**: `plans/plan-023-large-n-anchor-sweep.results.md` 의 frontmatter
  - **file 3**: `analysis/plan-023/results.md` 의 frontmatter
  - 위 3 file 의 `best_sub_exp`, `band`, `status` 토큰이 정확히 같은 값
- §0.5 c1~c11 모두 [DONE] (hash 박제)
- `followed_by` 의 plan-024/025 항목 박제 유지

---

## §9. Out of scope (명시적으로 안 함)

- **corrector reg head 재투입** — plan-021 에서 dead 확인. plan-022 carry. follow-up plan-024 가칭.
- **GRU sub-exp** — plan-021/022 carry. follow-up plan 별도.
- **anchor radius ≠ 0.005m** — 단일 shell 박제. follow-up plan-025 (N × radius 2D sweep).
- **multi-shell layout** (예: dodecahedron 20 + icosahedron 12 의 2-shell 32 anchor) — 본 plan single shell vertex-transitive 만 측정. follow-up.
- **center anchor 추가** — vertex-transitive 깨짐 (§0 decision-note). 본 plan center-free 박제.
- **N ≥ 60** — samples/class < 200 위험. follow-up plan 별도 (예: N=60 truncated icosahedron + samples expansion).
- **chiral layout** (snub cube, snub dodecahedron) — handedness 결정 미박제. follow-up plan.
- **DACON LB 측정 / submit** — train OOF 만.
- **Ensemble**.
- **τ_cls < 0.001 또는 > 0.005** — plan-022 동일 3 값 carry.

---

## §10. 참조 (read-only — path blacklist 예외)

- `analysis/plan-022/anchors.py:LAYOUT_NAMES` — naming schema carry
- `analysis/plan-022/selector_only_model.py:{LgbmSelectorOnly, build_soft_label_with_tau}` — model + soft label carry
- `analysis/plan-022/run_oof.py:run_oof_cell` — per-cell OOF runner carry
- `analysis/plan-022/baseline_carry.json` — dataset hash carry
- `analysis/plan-022/paradigm_analysis.py` — analysis pattern carry
- `analysis/plan-022/results.md` — plan-022 best evidence (A6_bcc14_tau001, Δ_1cm +0.0208 / Δ_1.5cm +0.0071)
- `analysis/plan-021/build_input.py` — 170D input pipeline (build_input_common, build_input_lgbm_extra, to_frenet, build_frenet_basis_3d)
- `analysis/plan-021/dual_head_model.py` — LgbmDualHead.clf_head 구조 carry (plan-022 carry chain)
- `analysis/plan-020/baseline_oof.json` — F0 hard evidence 0.6320 / 0.8033
- `analysis/plan-020/baseline_f0.py` — F0 산식
- `src/io.py` — load_all_samples, load_labels
- `src/pb_0_6822/selector.py:stable_fold_id` — 5-fold split
- `WORKFLOW.md §5 / §12` — plan 의무 / autonomous protocol
- `CLAUDE.md` — autonomous execution policy (commit/push 의무)
