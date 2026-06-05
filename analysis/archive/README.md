# analysis/archive/

Paradigm-closed plans. 본문 plan 은 `plans/archive/` 와 짝.

| plan | paradigm | 닫힌 이유 |
|---|---|---|
| plan-005 | corrector decomposition diagnostic | "framework 95% 가 장식" 발견 — plan-006/007 로 simplification 이전 |
| plan-006 | minimal Variant E LB validation | LB 0.6692 → 다음 paradigm 으로 |
| plan-007 | single-formula CMA-ES + MLP | ceiling 0.6482 측정, 새 ceiling 없음 |
| plan-008 | candidate pool 27→25 redefine | greedy set-cover oracle +0.0355 / selector 못 따라감 |
| plan-009 | ranking-specific loss + corrector | OOF +0.0150 / LB -0.0064 sign inversion |
| plan-011 | corrector 4-axis breadth ablation | 0/4 strict +0.005 통과, plan-004 default best-tuned 확정 |
| plan-012 | codebook bake-off (★ INVALID 재사용 환경) | reference only — plan-014 로 from-scratch 재부활 |
| plan-013 | plan-004 framework 3-lever stacking | simplification penalty, baseline 미달 |
| plan-014 | corrector from-scratch 5-phase | ceiling 0.6425, band 0.65 미달 → paradigm-shift 권장 |
| plan-015 | corrector feature 확장 A/B/C/D | Feature A drop rule 발동, root cause 가설 falsify |
| plan-016 | multi-seed stabilization | Path A/B/C 3/3 sub-threshold |
| plan-017 | Voxel CE 7³ + ensemble | G1 ensemble +0.0002 / G2 Voxel CE -0.0121 |
| plan-018 | arch ablation (SetTrans/PathSig/MoLE/GRU-attn) | 4/4 ALL FAIL, encoder bottleneck 가설 falsify |
| plan-019 | meta-EBIP + ICNN energy-based | 모든 stage sub-threshold |

각 디렉토리의 본 결과는 `plans/archive/plan-{NNN}-*.results.md` 의 §0 한 줄 결론 + `EXPERIMENTS.md` 의 plan table row 에 압축됨.

신규 plan 이 닫힌 가설을 다시 열 때만 여기 참고. 안 그러면 무시.
