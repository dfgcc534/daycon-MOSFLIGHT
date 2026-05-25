# plan-b-001 analysis results

상세 응답서: `plans/plan-b-001-yaw-attn-restructure.results.md`. 본 문서는 산출물 요약.

## G3 5-fold OOF (N=10000, K=14, 3-seed)

| arm | OOF hit_1cm | hit_1p5cm | baseline_hit_1cm (G0.5) | band | json |
|---|---|---|---|---|---|
| **B001 (F0)** ★best | **0.6296** | 0.8016 | 0.6320 | FAIL_regression | `results_g3_f0.json/.npz` |
| B002 (Kalman) | 0.6077 | 0.7968 | 0.5964 | FAIL_regression | `results_g3_kalman.json/.npz` |

G1 f0 (1-fold sanity): hit_1cm 0.6337 PASS, `results_g1_f0.json`.

## 결론
- 양 arm FAIL (≥0.6360 미달). best B001 0.6296 ≈ plan-030 0.6294 (+0.0002).
- **frame/attention/feature axis null** → carrier = training procedure (plan-031 A-track) 확정.
- **G0.5 모순 증거 확정**: Kalman 0.5964 < F0 0.6320 (hit_1cm). Kalman baseline swap net-negative.
- arm 비대칭: F0 arm 모델<baseline(−0.0024), Kalman arm 모델>baseline(+0.0113) — F0 floor 는 selector 로 못 넘음.

## 재현
```
python analysis/plan-b-001/run_oof.py --gate g3 --baseline f0 --quiet
python analysis/plan-b-001/run_oof.py --gate g3 --baseline kalman --quiet
```
데이터: repo-root `data/` (worktree 는 main 체크아웃 추출본 symlink).
