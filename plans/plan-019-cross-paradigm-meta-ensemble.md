---
plan_id: 019
version: 1
date: 2026-05-15 (Asia/Seoul)
status: draft
based_on:
  - 018 (G1 fail — single-stack architecture lever falsified, 4 arch 모두 OOF ≤ 0.6485, +0.0003 marginal)
  - 017 (G1 ensemble LB 0.6640 — 3-plan submission mean 성공)
  - 016 (G1/G2 path A/B LB 0.6638 — corrector stabilization paradigm)
  - 007 (Step 4 MLP OOF 0.6482, LB 0.6598 — A0 baseline)
  - 004 (ensemble LB 0.6822 — target reference)
scope: plan-018 §4.1 (corrector 결합) 의 *learnable variant*.
       **frozen multi-paradigm prediction × learnable sample-adaptive meta-head** 로 LB > 0.67 회수.
       야심찬 paradigm (meta-EBIP / ICNN Convex Energy / DEQ) 은 본 plan 외 (plan-020 후보 carry).
       multi-stack (plan-018 §4.2) / learnable basis (plan-018 §4.4) / hit-aware loss (plan-018 §4.3) 도 본 plan 외.
exp_ids:
  - F011_predictions-extract        # plan-016/018/007 의 OOF + test prediction 박제
  - F012_mean-ensemble              # G1 simple mean
  - F013_meta-head                  # G2 learnable adaptive weight
lb_score: null
exception_policy: 본 plan = ensemble paradigm. 기존 plan 의 *frozen weight 또는 submission* import 만, retrain X (단 plan-018 A3 의 test inference 재현은 학습된 weight 부재로 *재학습* 1회 허용 — §4.1 박제).
---

# plan-019 v1 — Cross-Paradigm Meta-Ensemble (frozen + learnable head)

## §0. 한 줄 목적

> plan-018 G1 fail 의 *single-stack architecture lever falsified* 결론 위에서, **plan-016 corrector + plan-018 A3 MoLE + plan-007 step 4 A0** 의 *frozen* multi-paradigm prediction 을 **learnable sample-adaptive meta-head** (~500 params, simplex 가중치) 로 결합. LB ≥ 0.69 도달이 G_final (= plan-007 step 4 LB 0.6598 위 +0.030, plan-004 ensemble LB 0.6822 의 ~85% 수준). 본 plan 의 야심찬 paradigm (meta-EBIP / ICNN Convex Energy / DEQ — brainstorm iter 1~5 박제) 은 plan-020 후보로 carry.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 본 plan 의 task essence — "frozen multi-paradigm + learnable meta-head"

- plan-018 measured 결과: single-stack arch lever 의 OOF ceiling ≈ 0.6485. *encoder/head 강화* 무효.
- plan-017 G1 ensemble (3-plan submission mean) LB 0.6640 — *multi-paradigm mean ensemble* 의 효과 measured.
- 본 plan = mean ensemble 의 *learnable extension*. simplex 가중치 + small residual.
- target = LB ≥ 0.69 (plan-017 G1 LB 0.6640 위 +0.025, plan-004 ensemble LB 0.6822 의 ~85%).
- 단일 모델 rule 폐기 — *frozen ensemble of single models*. 새 trainable param 은 meta-head ~500 only.

### 3 frozen paradigm

| paradigm | source | output | OOF | LB |
|---|---|---|---|---|
| Stage 1: plan-016 corrector | runs/baseline/plan016_g2_path_b/ + H002 checkpoint | pred (B, 3) | 0.6452 | 0.6638 |
| Stage 2: plan-018 A3 MoLE | runs/baseline/F008_arch-ablation (재학습 1회) + analysis/plan-018/oof_pred_A3.npy | pred (B, 3) | 0.6485 | (미측정, 추정 0.6601) |
| Stage 3: plan-007 step 4 A0 | runs/baseline/F002_formula-mlp/checkpoint_fold*.pt | pred (B, 3) | 0.6482 | 0.6598 |

### 합격 기준 (G-gate sequence)

- **G0** (prediction extraction sanity): 3 paradigm 의 OOF (train 10K) + test (10K) prediction 박제. OOF hit 가 위 표 값과 ±0.0003 일치 확인. plan-018 A3 *재학습 필수* (checkpoint 부재 — §4.1). plan-016 OOF prediction *availability check* — 가용 시 G2 진행, 비가용 시 G2 SKIP (G1 만, plan-020 carry).
- **G1** (simple mean ensemble): 3 paradigm 의 test prediction 좌표 mean → submission. OOF mean concat ≥ max(individual OOFs) + 0.003 (= 0.6485 + 0.003 = 0.6515). LB ≥ 0.67 → PASS. LB 1회 submit.
- **G2** (learnable meta-head): plan-016 OOF 가용 시. meta-head (~500 params) 학습 + OOF + LB submit. OOF ≥ G1 OOF + 0.003. LB ≥ 0.69 → G_final PASS. LB 1회 submit.
- **G_final**: results.md + plan-020 후보 ≥ 2 + 3 파일 frontmatter sync.

LB 제출 = **총 ≤ 2회** (DACON daily 5 limit 안, plan-018 의 0/5 사용 + plan-017 의 1/5 사용 후 남은 2/5 carry).

### G-gates (commit 단위 milestone)

- G0: 3 paradigm OOF + test prediction 박제, A3 재학습, plan-016 OOF availability check  [TODO]
- G1: simple mean ensemble OOF ≥ 0.6515 + LB ≥ 0.67                                       [TODO]
- G2: meta-head OOF ≥ G1 + 0.003 + LB ≥ 0.69 (조건부, plan-016 OOF 가용 시)                 [TODO]
- G_final: results.md + plan-020 후보 ≥ 2 + frontmatter sync                                 [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-019-*.md` 본문 (본 파일) | [TODO] |
| c2 | code | `analysis/plan-019/predictions_extract.py` — 3 paradigm 의 OOF + test prediction 추출 + A3 재학습. spec @ §4 | [TODO] |
| c3 | exp | F011: prediction 박제 (predictions/{plan016,plan018_A3,plan007_step4}_{oof,test}.npy) | [TODO] |
| G0 | gate | 3 paradigm OOF/test 박제, OOF hit ±0.0003 일치 | [TODO] |
| c4 | code | `src/plan019/mean_ensemble.py` — 좌표 mean ensemble. spec @ §5.1 | [TODO] |
| c5 | exp | F012: mean ensemble OOF + test submission 산출. spec @ §5.1 | [TODO] |
| c6 | sub-lb | G1 dacon-submit + lb_log + frontmatter. spec @ §7 | [TODO] |
| G1 | gate | OOF ≥ 0.6515 + LB ≥ 0.67 | [TODO] |
| c7 | code | `src/plan019/meta_head.py` — Linear(20→16) + SiLU + Linear(16→3+2). spec @ §5.2 | [TODO] |
| c8 | exp | F013: meta-head 5-fold 학습 + OOF + test submission. spec @ §5.2 | [TODO] |
| c9 | sub-lb | G2 dacon-submit + lb_log + frontmatter. spec @ §7 | [TODO] |
| G2 | gate | OOF ≥ G1 + 0.003 + LB ≥ 0.69 | [TODO] |
| c10 | synthesis | `analysis/plan-019/results.md` + `next_plan_candidates.md` (≥ 2). spec @ §8 | [TODO] |
| G_final | gate | results.md + plan-020 후보 + 3 파일 frontmatter sync | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `prediction_mismatch`: G0 의 OOF hit 가 expected (0.6482 / 0.6485 / 0.6452) 와 |Δ| > 0.0003 — checkpoint 또는 data split 의 spec drift. halt + 사용자 escalate.
- `a3_retrain_unstable`: A3 재학습의 OOF 가 0.6485 ± 0.005 미달 (random seed / 데이터 split 불일치). retry 1회 후 escalate.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/plan019/**`, `analysis/plan-019/**`
  - `runs/baseline/F011_predictions-extract/**`, `runs/baseline/F012_mean-ensemble/**`, `runs/baseline/F013_meta-head/**`
  - **read-only**: `runs/baseline/F002_formula-mlp/checkpoint_fold*.pt` (plan-007 step 4 weight), `runs/baseline/plan016_g2_path_b/submission.csv` (plan-016 best test prediction), `runs/baseline/H002_corrector-strengthen/**/boundary_sub_*.pt` (plan-016 corrector weight, 위치 G0 에서 확인), `analysis/plan-018/oof_pred_A3.npy` (plan-018 A3 OOF), `analysis/plan-007/best_basis.json`
- blacklist 추가:
  - 위 read-only path 의 *수정* (frozen — 본 plan 의 모든 paradigm 은 *retrain X*, exception: plan-018 A3 의 test inference 재현이 위해 1회 재학습, 결과는 새 path `runs/baseline/F011_predictions-extract/A3_retrain/` 에 박제)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — meta-head arch = Linear(20→16) + SiLU + Linear(16→5) (out_dim = 2 simplex weight + 3 residual). param ~500.`
- `decision-note: data-fallback — plan-016 OOF prediction 부재 시 G2 SKIP, plan-020 후보 carry.`
- `decision-note: spec-default — A3 재학습 시 plan-018 v1.1 §5.0 framework 정확 carry (seed=42, 5-fold, batch=256, epoch=50, patience=8).`
- `decision-note: brainstorm-carry — meta-EBIP / ICNN Convex Energy / DEQ 후보는 plan-020 으로 carry (본 plan 외 — plan-018 falsification 반영하여 안전 path 우선).`

---

## §1. 배경

### §1.1 plan-018 G1 fail 의 implication (carry-over)

plan-018 §3 verdict:
- single-stack lever **falsified**: encoder 강화 (A1/A2/A6) 와 head capacity (A3) 모두 G1 threshold 미달.
- measured single-stack ceiling ≈ A0 + 0.0003 = **0.6485**.
- plan-007 §9.2 "단일 공식 framework 한계 ≈ 0.6491" 결론과 정합 confirmed.

plan-018 §4 후보 (plan-019 후보):
- A. corrector 결합 (low cost) — **본 plan**
- B. multi-stack (high cost, 야심차나 risky) — plan-020 후보
- C. hit-aware loss — plan-020 후보 (orthogonal lever)
- D. learnable basis — plan-020 후보

### §1.2 본 plan 의 가설

| 가설 | 검증 방법 | 합격 |
|---|---|---|
| H1: 3 paradigm 의 *prediction 분산* 이 sample 별로 다름 → mean ensemble 만으로 +0.003 ~ +0.020 gain | F012 OOF ≥ 0.6515 | G1 통과 |
| H2: sample 별 *paradigm 신뢰도* 가 trajectory features 로 예측 가능 → meta-head 가 mean 위 +0.003 추가 gain | F013 OOF ≥ G1 + 0.003 | G2 통과 |
| H3: 단일 ensemble 단계 LB ≥ 0.69 (plan-007 step 4 LB +0.030, plan-004 ensemble 85%) | G2 dacon-submit | LB ≥ 0.69 |

### §1.3 brainstorm iter 1~5 의 야심찬 후보 — plan-020 carry

본 plan-019 작성 직전 *5-iteration brainstorm loop* 박제 (chat history). LB > 0.7 ceiling 6 candidates:

| # | paradigm | 출처 | 예상 LB | step 4 spirit |
|---|---|---|---|---|
| 1 | meta-EBIP + ICNN hybrid | iter 4 | 0.72~0.74 + stability | ◎ |
| 2 | DEQ (Deep Equilibrium) | iter 5 | 0.71~0.73 | ◎ |
| 3 | meta-EBIP base | iter 4 | 0.72~0.74 | ◎ |
| 4 | Learnable Basis + Sparse MoLE | iter 1/2 | 0.71~0.72 | ◯ |
| 5 | EBIP base | iter 3 | 0.70~0.72 | ◎ |
| 6 | ICNN Convex Energy | iter 4 | 0.69~0.71 | ◎ |

**본 plan 외, plan-020 후보 풀로 carry**. plan-019 결과 (G1/G2 LB) 가 박혀야 plan-020 의 적정 ranking 결정 가능.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 3 frozen paradigm | plan-016 corrector / plan-018 A3 MoLE / plan-007 step 4 A0 |
| ensemble 방식 | (G1) 좌표 mean, (G2) learnable meta-head simplex weight + residual |
| meta-head input | [stage1_pred (3), stage2_pred (3), stage3_pred (3), stats_13d, pairwise_dist (3)] = 25-d |
| meta-head arch | Linear(25→16) + SiLU + Linear(16→5), output = 2 simplex weight + 3 residual |
| meta-head params | ~500 |
| meta-head training | 5-fold OOF on original 10K, seed=42, Adam lr=1e-3, batch=256, epoch=50, patience=8 |
| loss | soft_hit_loss (plan-007 §7.2 carry) |
| LB 제출 | ≤ 2회 (G1 + G2) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 3 paradigm 의 *retrain* | frozen. exception: plan-018 A3 의 *test inference 재현* 용 1회 재학습 (§4.1) |
| 새 paradigm arch (meta-EBIP / ICNN / DEQ / Learnable basis) | plan-020 후보 carry |
| 4번째 paradigm (e.g., plan-014/015 의 base / plan-017 G1 ensemble) | scope 폭증. plan-020 의 ensemble member 확장 후보 |
| multi-stack (selector + boundary corrector) | plan-018 §4.2 carry, plan-020 후보 |
| hit-aware loss 변경 | plan-018 §4.3 carry, plan-020 후보 |
| learnable basis | plan-018 §4.4 carry, plan-020 후보 |
| meta-head depth ≥ 2 hidden layer | overfitting risk. 본 plan single hidden layer |
| LB 제출 ≥ 3회 | DACON quota 보존, plan-020 carry |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 입력 데이터 + 분할

| 분할 | 출처 | 사용 |
|---|---|---|
| Train original (10K, end_idx=10, target=train_y) | `data/train/` | 3 paradigm 의 OOF (sample-aligned), meta-head 학습 |
| Test (10K, end_idx=10) | `data/test/` | 3 paradigm 의 test inference, meta-head inference, submission |

**Sample alignment** (모든 paradigm 의 OOF 가 *동일 sample_id 순서* 박제):
- 5-fold split: seed=42, sample_id-grouped, plan-007 §7.2 / plan-018 §3.1 carry.
- OOF prediction = (10000, 3) — fold concat 후 *원래 sample order* 로 재정렬.
- meta-head 학습 시 fold split 도 *동일 seed=42* 로 align.

### §3.2 합격 기준 (정량)

- **G0**: 3 paradigm 의 OOF hit (1cm) = (0.6482 / 0.6485 / 0.6452) ± 0.0003. test prediction shape (10000, 3).
- **G1**: mean ensemble OOF ≥ 0.6515 (= max(individual) + 0.003). LB ≥ 0.67 → PASS.
- **G2**: meta-head OOF ≥ G1 + 0.003 (= 0.6545+). LB ≥ 0.69 → G_final PASS.
- **G_final**: results.md + plan-020 후보 ≥ 2 + 3 파일 frontmatter sync.

### §3.3 평가

- OOF metric = original 10K 의 hit rate (threshold 0.01 m). plan-007 §3.3 carry.
- LB metric = DACON public LB hit rate.

---

## §4. STAGE 0 — Predictions Extraction (c2~c3)

### §4.1 `analysis/plan-019/predictions_extract.py` (c2, 신규 작성)

3 paradigm 의 OOF (10K) + test (10K) prediction 추출. 모두 (10000, 3) 형태 npy 박제.

#### Stage 1: plan-016 corrector

**1.1 test prediction**: `runs/baseline/plan016_g2_path_b/submission.csv` 의 (id, x, y, z) 컬럼 → (10000, 3) npy.

**1.2 OOF prediction (조건부)**:
- plan-016 의 checkpoint 위치 확인: `runs/baseline/H002_corrector-strengthen/fold_{0..4}/` 의 `boundary_sub_*.pt` 들.
- 가용 시 (`.pt` 모두 존재): 5-fold OOF inference 재현. plan-016 §6 의 inference spec carry. 결과 `predictions/plan016_oof.npy` (10000, 3).
- 비가용 시: G2 SKIP, plan-020 carry. G1 만 진행.

**1.3 결과 박제**:
- `analysis/plan-019/predictions/plan016_test.npy` (10000, 3)
- `analysis/plan-019/predictions/plan016_oof.npy` (10000, 3) — 가용 시
- `analysis/plan-019/predictions/plan016_availability.json` — `{"test": true, "oof": <bool>, "checkpoint_paths": [...], "expected_oof_hit": 0.6452, "measured_oof_hit": <float|null>}`

#### Stage 2: plan-018 A3 MoLE (재학습 1회)

**2.1 재학습 사유**: plan-018 의 ablation_runner.py 가 *5-fold OOF 만 산출, checkpoint 미박제*. 즉 test inference 재현 불가. 본 plan 의 G2 ensemble 위해 *동일 spec* 으로 1회 재학습.

**2.2 재학습 spec** (plan-018 v1.1 §5.0 framework 정확 carry):
- 5-fold (seed=42, sample_id-grouped, plan-007 §7.2 carry).
- A3 arch: `src/plan018/arch_modules.py` 의 MoLE head class (frozen import, retrain X — *weight* 만 재학습).
- training: Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8, grad_clip=2.0, soft_hit_loss.
- 출력: `runs/baseline/F011_predictions-extract/A3_retrain/checkpoint_fold{0..4}.pt`.

**2.3 검증**: 재학습 5-fold concat OOF ∈ [0.6480, 0.6490] (plan-018 결과 0.6485 ± 0.005, random seed variance). 위반 시 `a3_retrain_unstable` severe.

**2.4 결과 박제**:
- `analysis/plan-019/predictions/plan018_A3_oof.npy` (10000, 3) — 재학습 OOF
- `analysis/plan-019/predictions/plan018_A3_test.npy` (10000, 3) — 5-fold ensemble test inference (평균 또는 ensemble vote)
- `analysis/plan-018/oof_pred_A3.npy` (기존) 와 hit 차이 박제 (random seed effect).

#### Stage 3: plan-007 step 4 A0

**3.1 checkpoint**: `runs/baseline/F002_formula-mlp/checkpoint_fold{0..4}.pt` (이미 박제).

**3.2 inference**: 5-fold OOF + test 재현. arch = plan-007 §7.1 spec carry.

**3.3 결과 박제**:
- `analysis/plan-019/predictions/plan007_step4_oof.npy` (10000, 3)
- `analysis/plan-019/predictions/plan007_step4_test.npy` (10000, 3)

#### Stage 4: sanity check

- 3 paradigm 의 OOF hit 가 expected 값 ±0.0003 일치 확인.
- 위반 시 `prediction_mismatch` severe.

### §4.2 G0 합격 기준 (자동 판정)

- 6 npy 파일 (3 paradigm × {oof, test}, plan-016 OOF 조건부) 박제 완료.
- 3 paradigm 의 OOF hit 일치 (±0.0003).
- plan-016 OOF availability 박제 — `plan016_availability.json` 의 `"oof"` field.
- 위반 시 severe escalate.

### §4.3 시간 예산

- plan-016 OOF inference: ~5분 (checkpoint 로딩 + 5-fold inference)
- plan-018 A3 재학습 + inference: ~3분 (cuda, plan-018 measured 5-fold 80초 + test 30초)
- plan-007 step 4 inference: ~2분 (checkpoint 로딩)
- 박제: ~1분
- **총 ~10분**

---

## §5. STAGE 1 — Ensemble (c4~c8)

### §5.1 G1 Simple Mean Ensemble (c4~c5)

**`src/plan019/mean_ensemble.py`** (신규 작성):

```python
# coords mean over 3 paradigm
import numpy as np


def mean_ensemble(plan016: np.ndarray,
                  plan018_A3: np.ndarray,
                  plan007_step4: np.ndarray) -> np.ndarray:
    """
    Args:
        plan016, plan018_A3, plan007_step4: each (N, 3) prediction.
    Returns:
        (N, 3) — coords mean.
    """
    return (plan016 + plan018_A3 + plan007_step4) / 3.0
```

- OOF: 3 paradigm 의 OOF (사용 가능한 것만) mean → hit rate 측정.
- Test: 3 paradigm 의 test mean → submission.csv (id, x, y, z) 산출.

**G1 합격**: OOF ≥ 0.6515 + LB ≥ 0.67.

### §5.2 G2 Learnable Meta-Head (c7~c9)

**전제**: plan-016 OOF 가용 (G0 의 `plan016_availability.json["oof"] == true`). 비가용 시 G2 SKIP.

**`src/plan019/meta_head.py`** (신규 작성):

```python
import torch
import torch.nn as nn


class MetaHead(nn.Module):
    """sample 별 simplex 가중치 + small residual 출력.

    input (25-d):
        [stage1_pred (3), stage2_pred (3), stage3_pred (3),
         stats_13d, pairwise_dist (3)]

    pairwise_dist:
        [||s1 - s2||, ||s2 - s3||, ||s1 - s3||]   # 3 paradigm 간 disagreement

    output:
        weights (3,)  via softmax  — simplex (모든 paradigm 가 active sum=1)
        residual (3,) — small correction

    pred = w1·s1 + w2·s2 + w3·s3 + 0.1 · residual
    """
    def __init__(self, in_dim: int = 25, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),   # 3 weight logits + 3 residual
        )
        # residual scale = 0.1 (small relative to typical displacement ~1 m)
        self.res_scale = 0.1

    def forward(self,
                s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor,
                stats_13d: torch.Tensor) -> torch.Tensor:
        d12 = torch.norm(s1 - s2, dim=-1, keepdim=True)
        d23 = torch.norm(s2 - s3, dim=-1, keepdim=True)
        d13 = torch.norm(s1 - s3, dim=-1, keepdim=True)
        x = torch.cat([s1, s2, s3, stats_13d, d12, d23, d13], dim=-1)  # (B, 25)
        out = self.net(x)                                                # (B, 6)
        w_logits, residual = out[..., :3], out[..., 3:]                  # (B, 3), (B, 3)
        w = torch.softmax(w_logits, dim=-1)                              # (B, 3) simplex
        pred = (w[..., :1] * s1 + w[..., 1:2] * s2 + w[..., 2:3] * s3
                + self.res_scale * residual)
        return pred
```

- 5-fold OOF (seed=42, sample_id-grouped). fold 마다 *model 재학습*.
- training: Adam lr=1e-3, wd=1e-4, batch=256, epoch=50, patience=8.
- loss: soft_hit_loss(pred, target).
- early stop metric: val_hit (high-is-better, min_delta=1e-4).
- output: meta-head 의 5-fold OOF concat + test inference (5-fold ensemble mean).

**G2 합격**: OOF ≥ G1 + 0.003 + LB ≥ 0.69.

---

## §6. ablation runner + 결과 박제

### §6.1 `analysis/plan-019/results_table.json` (c10 의 일부)

```json
{
  "exp_id": "F012_F013_ensemble",
  "g0": {"plan016_oof_available": <bool>, ...},
  "g1_mean_ensemble": {
    "oof_hit_1cm": <float>,
    "fold_oofs": [...],
    "submission_path": "runs/baseline/F012_mean-ensemble/submission.csv",
    "lb_score": <float|null>
  },
  "g2_meta_head": {
    "skipped": <bool>,
    "oof_hit_1cm": <float|null>,
    "fold_oofs": [...|null],
    "n_params": 500,
    "submission_path": "runs/baseline/F013_meta-head/submission.csv",
    "lb_score": <float|null>
  }
}
```

---

## §7. LB 제출 정책

### §7.1 자율 호출

```python
# G1 끝 (c6):
Skill(skill="dacon-submit",
      args="runs/baseline/F012_mean-ensemble/submission.csv F012_mean-ensemble_plan-019")

# G2 끝 (c9, 조건부 — plan-016 OOF 가용 + G1 LB < 0.69):
Skill(skill="dacon-submit",
      args="runs/baseline/F013_meta-head/submission.csv F013_meta-head_plan-019")
```

**G2 conditional skip**: G1 LB ≥ 0.69 이미 도달 시 G2 LB submit skip (quota 보존). plan-020 carry.

### §7.2 응답 4-분기 처리 (plan-018 §8.2 동일 패턴)

| (isSubmitted, lb_score) | 처리 | frontmatter `lb_score` | status | severe |
|---|---|---|---|---|
| (True, float) | full success | `<float>` 소수 4자리 | `all_complete` | — |
| (True, None) | partial — carry-over commit | `TBD` | `partial` | — |
| (False, *) | retry 1회 (60초 sleep). 재실패 시 severe | `null` | `partial` | `lb_unsubmitted` |
| Skill exception | 즉시 escalate | `null` | `partial` | `dacon_submit_skill_missing` |

### §7.3 LB band 분류

- LB ≥ 0.69 → **G_final PASS** (plan-007 step 4 LB +0.030, plan-004 ensemble 85%)
- 0.67 ≤ LB < 0.69 → partial (G1 PASS target 만 달성)
- 0.66 ≤ LB < 0.67 → `partial — band carry from plan-017 G1 (0.6640)`
- LB < 0.66 → `lb_below_baseline` warn (ensemble lever 도 marginal)

---

## §8. STAGE 3 — Synthesis + plan-020 후보 (c10)

### §8.1 `analysis/plan-019/results.md`

frontmatter:
```yaml
---
plan_id: 019
based_on:
  - 018
  - 017
  - 016
  - 007
finished_at: <ISO8601 KST>
status: all_complete | partial
exp_ids_completed:
  - F011_predictions-extract
  - F012_mean-ensemble
  - F013_meta-head
lb_exp_id: F012_mean-ensemble | F013_meta-head
lb_score: <float|TBD|null>
lb_submitted_at: <ISO8601 KST>
g1_passed: <bool>
g2_passed: <bool|null>
g2_skipped: <bool>
---
```

본문:
- G0 prediction 박제 결과 (3 paradigm OOF hit, plan-016 availability)
- G1 mean ensemble OOF + LB
- G2 meta-head OOF + LB (조건부)
- plan-020 후보 ≥ 2

### §8.2 plan-020 후보 (시나리오 분기)

| G2 결과 | plan-020 후보 |
|---|---|
| **G2 PASS (LB ≥ 0.69)** | brainstorm iter 4 의 **#1 meta-EBIP + ICNN hybrid** (천장 0.72~0.74) — meta-head 의 학습 결과 위에 energy-based reformulation. plan-020 의 main paradigm. |
| **G2 partial (LB 0.67~0.69)** | (1) brainstorm iter 1 의 **Learnable Basis + Sparse MoLE** (plan-018 §4.4 + iter 1) — basis 자체 확장, (2) brainstorm iter 5 의 **DEQ** (천장 0.71~0.73) — adaptive depth. |
| **G2 SKIP / fail** | (1) plan-018 §4.2 **multi-stack** (selector + boundary corrector × A3), (2) plan-018 §4.3 **hit-aware loss** (orthogonal lever). brainstorm iter 4 의 **#6 ICNN Convex Energy** (천장 0.69~0.71, stability ⭐). |
| **G1 PASS but G2 SKIP (plan-016 OOF 부재)** | (1) plan-016 OOF 재산출 plan (작은 plan), (2) plan-017 G1 ensemble 의 sub-paradigm 확장 (4+ paradigm). |

### §8.3 frontmatter sync (3 파일)

- `plans/plan-019-cross-paradigm-meta-ensemble.md` top-level `lb_score`
- `plans/plan-019-cross-paradigm-meta-ensemble.results.md` frontmatter
- `analysis/plan-019/results.md` frontmatter

---

## §9. 코드 재사용 정책 (plan-018 §10 carry + 본 plan 특수)

### §9.1 핵심 원칙

> **확실하지 않으면 새 코드 생성**. spec ambiguity 발견 시 *해당 module 신규 작성*, 기존 코드 *import X*.

### §9.2 본 plan 의 신규 작성 / import 허용 / import 금지

| 영역 | 정책 |
|---|---|
| `src/plan019/mean_ensemble.py`, `meta_head.py` | **신규 작성**. 다른 plan source 의 trainable module import X. |
| `analysis/plan-019/predictions_extract.py`, `results_table.json` 생성 | 신규 작성. |
| plan-018 A3 의 arch class (재학습 위해) | `src/plan018/arch_modules.py` 의 MoLE head class **import 허용** (frozen, retrain — *weight* 만 새). |
| plan-016 corrector inference (조건부) | `src/plan016/` 또는 `src/plan_005/` 의 inference module import 허용 (frozen weight + frozen code). spec ambiguity 시 신규 작성. |
| plan-007 step 4 A0 inference | `runs/baseline/F002_formula-mlp/checkpoint_fold*.pt` 의 weight + `src/plan018/baseline_a0.py` (plan-018 의 A0 reproduce module) 사용. |
| basis_terms 식 (8 vars) | `analysis/plan-007/best_basis.json` 의 `best_basis_vars` order + plan-018 §5.0 의 basis_terms 사전 계산 식 carry. |
| 5-fold split | plan-007 §7.2 / plan-018 §3.1 carry. 코드 *재구현*. |
| soft_hit_loss | plan-007 §7.2 식 *재구현* (간단 함수). |
| brainstorm iter 1~5 의 paradigm (meta-EBIP / ICNN / DEQ 등) | **본 plan 외, plan-020 carry**. |

### §9.3 ambiguity 발견 시 처리

- 함수 시그니처 / 동작 의문 시 → 해당 함수 신규 작성. plan 본문 §9 에 "신규 작성 사유" 1 줄 박제.
- 특히 plan-016 의 corrector inference module 의 input/output contract 가 모호한 경우 *반드시* 신규 inference wrapper 작성.

---

## §10. References

### §10.1 본 plan 내 reference

- plan-018 §3 (G1 fail verdict), §4 (plan-019 후보 4 종)
- plan-018 §5.0 (ablation_runner framework — A3 재학습 spec)
- plan-018 §10 (코드 재사용 정책)
- plan-017 G1 ensemble (3-plan submission mean LB 0.6640)
- plan-016 G1/G2 (best stack LB 0.6638)
- plan-007 §7 (step 4 MLP A0 baseline)
- analysis/plan-018/ablation_results.json (A3 OOF 0.6485 + arch param)
- analysis/plan-007/best_basis.json (8 best basis vars)

### §10.2 brainstorm iter 1~5 carry (plan-020 후보)

- iter 1: Learnable Basis + Sparse MoLE
- iter 2: Set Transformer encoder + ISAB
- iter 3: EBIP base (Energy-Based Implicit Prediction)
- iter 4: meta-EBIP, ICNN Convex Energy (Amos 2017), meta-EBIP + ICNN hybrid
- iter 5: DEQ (Bai 2019), Cross-modal Contrastive (skip 평가)

---

## §11. 시간 예산 (전체)

| 단계 | 예상 소요 |
|---|---|
| c1 plan 작성 (본 파일) | (이미 완료) |
| c2~c3 predictions extract + A3 재학습 | ~10분 |
| c4~c5 mean ensemble OOF + submission | ~5분 |
| c6 G1 LB 제출 + 회수 | ~5분 |
| c7~c8 meta-head 학습 + OOF + submission (조건부) | ~15분 |
| c9 G2 LB 제출 + 회수 (조건부) | ~5분 |
| c10 synthesis | ~30분 |
| **총** | ~70분 wall-time (G2 진행 시) / ~25분 (G2 SKIP) |

---

## §12. End-of-Plan Checklist

- [ ] G0: 3 paradigm OOF/test prediction 박제, OOF hit ±0.0003 일치, plan-016 OOF availability 박제
- [ ] G1: mean ensemble OOF ≥ 0.6515 + LB ≥ 0.67
- [ ] G2 (조건부): meta-head OOF ≥ G1 + 0.003 + LB ≥ 0.69 (또는 SKIP 사유 박제)
- [ ] G_final: results.md + plan-020 후보 ≥ 2 + 3 파일 frontmatter sync
- [ ] 모든 commit + push 완료 (CLAUDE.md ⚠️ Commit · Push 의무)
- [ ] brainstorm iter 1~5 의 6 candidates 가 plan-020 후보 풀로 carry (§8.2 시나리오 분기 박제)
