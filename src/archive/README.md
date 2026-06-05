# src/archive/

Paradigm-closed plan 의 코드. 본문 plan = `plans/archive/`, 분석 산출물 = `analysis/archive/`.

| 모듈 | plan | 닫힌 이유 |
|---|---|---|
| `plan018/` | plan-018 arch ablation | 4/4 arch ALL FAIL (Set Transformer / Path Signature / MoLE / GRU-attn 모두 +0.005 미달). encoder bottleneck 가설 falsify. |
| `plan019/` | plan-019 meta-EBIP + ICNN | S1/S2/S3 모두 sub-threshold. energy-based paradigm marginal-only. |

신규 plan 에서 직접 import 하지 말 것. winner 노트북이 같은 paradigm 을 다시 시도할 때만 historical reference.

`src/pb_0_6822/` 는 archive 아님 — lane a/b/c/d 모두 active 의존 (Kalman residual / Frenet anchor / corrector framework). 닫힌 path 가 아니라 *base 인프라*.

`src/plan019.common` 이 참조하던 `analysis/plan-007/*.json` 은 `analysis/archive/plan-007/*.json` 으로 이동됨 (path 갱신 완료).
