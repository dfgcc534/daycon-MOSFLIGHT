---
plan_id: 007
exp_ids:
  - F001_formula-ga
  - F002_formula-mlp
lb_exp_id: F001_formula-ga-step3
lb_score: TBD
lb_submitted_at: 2026-05-12T16:34:30+09:00
lb_recovered_at: null
status: partial
date: 2026-05-12 (Asia/Seoul)
---

# plan-007 results — Single-Formula CMA-ES + Basis Ablation + Per-Sample MLP

본 plan = plan-006 의 단일 공식 baseline (0.6491 argmax-corrected) 위에 4 단계 progression 으로 단일 공식 framework 의 데이터 ceiling 을 측정.

**status: partial** — Step 2 + Step 3 dacon-submit 모두 `{isSubmitted: True, detail: 'Success'}` (2026-05-12T16:31:35+09:00 / 16:34:30+09:00) 회수 — *DACON 응답에 lb_score 미포함*. plan §8.2 의 (True, None) 분기 → carry-over c5.1/c8.1 후속 회수 예약. 본 plan 의 frontmatter `lb_score = TBD` 유지 (§8.4 — Step 3 가 최종 LB).

## §1. Exp summary

| field | value |
|---|---|
| exp_ids | `F001_formula-ga` (Step 2/3), `F002_formula-mlp` (Step 4) |
| plan_id | 007 |
| based_on | plan-004 (selector lock-in) + plan-005 (oracle 0.7188) + plan-006 (단일 공식 0.6491) |
| compute | local CPU (CMA-ES) + cuda 2.8.0 (MLP) — ~5 분 wall-time (plan 예산 4~5 시간 대비 50x 빠름) |
| 단일 공식 출발점 | `frenet_par120_perp_neg020` (plan-006 1등, CANDIDATES[17]) |
| 데이터 증폭 | sliding 40K (end_idx ∈ [5,8], horizon=2) ∪ original 10K = **50K pool** (G0 PASS) |
| 산출 위치 | `analysis/plan-007/**`, `runs/baseline/F001_formula-ga/**`, `runs/baseline/F002_formula-mlp/**` |

## §2. G-gate 결과

| gate | result | metric | commit |
|---|---|---|---|
| G0 | PASS | aug_usable=True (quantile RMSE 0.001252 < 0.0015) | `117eeb4` |
| G1 | PASS | oof_hit_5fold=0.6403 ∈ [0.62, 0.78] | `b7a2a4a` |
| G2 | PASS | best_basis_hit=0.6387 ≥ 0.6342 (Step 2 single fit); 8 var basis | `963be03` |
| G3 | PASS | oof_hit=0.6482 ≥ 0.6437 (+0.005), gain +0.0095 vs Step 3 | `2c7eb3d` |
| G_final | partial | results.md + next_plan_candidates.md 박제, but lb_score=TBD (Step 3 회수 carry-over) | `c11_hash` |

## §3. 단일 공식 ceiling trajectory

| stage | metric | value |
|---|---|---|
| plan-006 argmax + corrector | OOF hit | 0.6491 (reference) |
| Step 2 CMA-ES 6 vars | oof_hit_5fold | 0.6403 |
| Step 3 best basis (8 vars) | best_basis_hit (50K single fit) | 0.6387 |
| Step 4 per-sample MLP | oof_hit (5-fold concat) | **0.6482** |

→ 단일 공식 + per-sample MLP framework 의 *측정된* ceiling = 0.6482 (plan-006 의 0.6491 와 거의 동급, -0.09pp). **새로운 ceiling 돌파 없음**.

## §4. 시나리오 분기 + plan-008 후보

scenario B (Step 4 OOF gain +0.0095 < +0.010) → 단일 공식 framework 한계 인정. plan-008 후보 ≥ 2:

1. **단일 공식 framework 한계 인정 → 27 후보 풀 확장 (35+)** *[추천]* — plan-005 worst-100 + oracle gap 0.0697 의 회수
2. **corrector 재설계 + Step 4 MLP OOF 결합** — plan-005 corrector +0.89pp 효과 가 MLP per-sample 위에서 동작 확인
3. **Step 4 LB 단독 제출 (carry-over)** — 단일 공식 framework 의 LB ceiling 박제

상세: `analysis/plan-007/next_plan_candidates.md`.

## §5. Carry-over open

- **c5.1**: Step 2 LB 회수 (DACON 웹페이지 조회) → lb_log + frontmatter 갱신
- **c8.1**: Step 3 LB 회수 → frontmatter `lb_score` (Step 3 값) + status `all_complete` 전환

본 plan 의 최종 endpoint = c8.1 close 후 plan-008 의 첫 task 로 carry-over.

상세 분석: `analysis/plan-007/results.md` 참조.
