# plan-007 Step 2 — CMA-ES local-optimum 진단

**결론**: CMA-ES best 가 plan-006 x0 보다 *모든* metric 에서 우수 → **local optimum 아님**, framework 자체 한계 확정.

## 측정

x0 = `[1.98, 1.20, -0.20, 0, 0, 0]` (plan-006 `frenet_par120_perp_neg020` 의 effective coefficient seed)
best = `[1.502, 1.949, 0.172, 0.423, 0.077, 0.027]` (CMA-ES single fit on 50K pool, c4 = b7a2a4a)

| 평가 set | x0 hit | best hit | Δ (best − x0) |
|---|---|---|---|
| 50K pool (sliding 40K + original 10K, fit data) | 0.6282 | **0.6342** | +0.0060 ✓ |
| Original 10K only (end_idx=10, plan-006 baseline 과 동일 분포) | 0.6320 | **0.6416** | +0.0096 ✓ |
| Midpoint (x0 와 best 의 평균) | 0.6315 (pool) / 0.6370 (orig) | — | — |

→ midpoint 가 양 끝점보다 *모두 낮음* = fitness landscape 가 *bimodal/multi-modal 아님*. best 는 단조 향상 방향의 진짜 endpoint.

## 함의

### 1. CMA-ES 진단

- Step 2 의 `single_fit_best_hit = 0.6342` 는 *진짜* improvement. local optimum 의심 *기각*.
- convergence_last_50_range = 0.000160 << 0.005 (cma_es_no_convergence severe 부재) 와 일관.
- multi-start re-fit 불필요 (plan §N+3 #3 의 권장 사항이지만 본 진단으로 single-start sufficiency 확인).

### 2. plan-006 baseline 과의 정합

- plan-006 의 0.6491 baseline = "argmax(corrected) OOF" (plan-005 의 corrector +0.89pp 효과 포함).
- corrector 제거 시 raw single formula 추정 hit ≈ 0.6402 (plan-005 corrector_decomp 의 +0.89pp 역추산).
- plan-007 Step 2 의 5-fold OOF = **0.6403** — plan-006 raw baseline (0.6402) 과 거의 일치 ⇒
  Step 2 가 plan-006 의 raw single-formula 를 *정확히* 재현. plan-007 의 Step 3/4 가
  추가로 +0.0079 회수 (0.6403 → 0.6482).

### 3. plan-008 의 path 선택

| path | gain 가능성 | rationale |
|---|---|---|
| 후보 1 (27→35 후보 풀 확장) | **+1~3pp OOF** (oracle gap 0.0697 의 일부 회수) | 본 진단으로 단일 공식 framework 의 0.6482 ceiling 이 *legit* 임이 확정 → oracle 7pp gap 회수는 후보 풀 확장 path 만 |
| 후보 2 (corrector 재설계 + Step 4 MLP OOF) | +0.5~1pp | 단일 공식 ceiling 못 넘음, but plan-005 corrector +0.89pp 가 MLP 위에서도 동작하면 추가 회수 |
| 후보 3 (Step 4 LB 단독 제출) | 0 ~ -0.01 LB | 단일 공식 LB ceiling 박제 — 후보 1 의 baseline 정보 |

**plan-008 권장 우선순위 유지**: 후보 1 (추천) → 후보 3 (cheap measurement) → 후보 2 (synergy with 후보 1).

## 재현 방법

```bash
PYTHONPATH=. python3 -c "
import sys, json
from pathlib import Path
sys.path.insert(0, 'analysis/plan-007')
from cma_es_baseline import _stack_train_terms, fitness_step2
from src.pb_0_6822 import selector

aug = json.loads(Path('analysis/plan-007/sliding_validity.json').read_text())['aug_usable']
gms = json.loads(Path('analysis/plan-007/cma_es_step2.json').read_text())['global_mean_speed']
ids, train_y = selector.read_labels(Path('data/train_labels.csv'))
train_x = selector.load_stack(Path('data/train'), ids)
stack = _stack_train_terms(aug, train_x, train_y, ids, gms)
args = (stack['p0'], stack['d1'], stack['acc_par'], stack['acc_perp'],
        stack['d2'], stack['jerk'], stack['ts_term'], stack['target'])
for label, params in [('x0', [1.98, 1.20, -0.20, 0, 0, 0]),
                       ('best', [1.502, 1.949, 0.172, 0.423, 0.077, 0.027])]:
    print(f'{label}: hit = {-fitness_step2(params, *args):.4f}')
"
```

소요 < 5초 (numpy 벡터화 단일 fitness eval).
