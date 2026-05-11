---
plan_id: 004
version: 1
date: 2026-05-11 (Asia/Seoul)
status: draft
based_on:
  - 001
  - 002
  - 003
  - notes/PB_0.6822 코드공유.ipynb
scope: full-stack (notebook code extraction + project data full-fit + 18×27 regime distribution audit + autonomous LB submission)
exp_ids:
  - P001_pb-0-6822-fullrun
lb_score: null
---

# plan-004 v1 — PB_0.6822 Notebook Full-Fit + 18×27 Regime Distribution Audit

## §0. 한 줄 목적

> **Dacon 모기 궤적 예측 대회 공개 노트북 `notes/PB_0.6822 코드공유.ipynb` (Public LB 0.6822 달성, 27개 물리 후보 + Attn-GRU selector + 18×27 regime bias + Tiny boundary corrector 2-stage 구조) 의 cell 4/6 코드를 `src/pb_0_6822/{selector,boundary}.py` standalone 모듈로 추출 + project 데이터 (`data/`) 에 적용한 full 5-fold 학습 + corrector full-fit 으로 *우리 LB 점수 1개* 를 `dacon-submit` skill 자율 호출로 회수 (CLAUDE.md autonomous policy). 그 과정에서 `candidate_regime_bias()` 가 in-memory 로만 계산하는 18-regime × 27-candidate empirical Bayes bias 표의 train sample 분포를 `analysis/plan-004/regime_distribution.{json,md}` 로 박제 (notebook 에 *없는* 새 검증) — degenerate regime (sample < 50) 식별 및 (regime, candidate) hyper-specialized cell flagging 포함. LB 점수 회수 + regime 분포 박제 둘 다 본 plan 의 *의무 산출* 이며, 미달 시 G_final 종료 불가.**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- 노트북 cell 4 (~94KB SELECTOR_MAIN) + cell 6 (~22KB BOUNDARY_MAIN) 의 추출이 *완전* (CANDIDATES list 27개, fit_regime_bins/assign_regimes/candidate_regime_bias 등 핵심 함수 보존). 위반 시 `extraction_drift` severe.
- 1-fold smoke (`--fold-limit 1`, epoch=1) 의 cv hit-rate 가 finite + 추출 버그 catch (NaN/Inf/ImportError 없음). 위반 시 `selector_no_convergence` severe.
- Full 5-fold selector + boundary corrector full-fit 완료, 모든 fold metric finite. 위반 시 `nn_numerical` severe.
- `submission_boundary_tiny_soft.csv` schema 가 `data/sample_submission.csv` 와 100% 일치 (row count, column names, id 순서). 위반 시 `submission_shape_mismatch` severe.
- **18-regime × 27-candidate 표 + sample 분포 박제 의무**: `analysis/plan-004/regime_distribution.{json,md}` 에 (a) regime별 sample count histogram, (b) 18×27 hit-rate table, (c) degenerate regime list (sample < 50), (d) hyper-specialized cell list 모두 기록. 미박제 시 `regime_unaudited` severe.
- **best 1 LB 제출 (필수, skip 불가, 자율 실행)**: `runs/baseline/P001_pb-0-6822-fullrun/submission.csv` → autonomous loop 가 `dacon-submit` skill 1회 호출 (사용자 승인 X) + 1 LB 점수 회수.
- **lb_score frontmatter 박제 의무**: `plans/plan-004-pb-0-6822-fullrun.md` 의 `lb_score` 필드 (또는 results frontmatter) 에 회수된 LB 점수 기록되어야 G_final 종료 가능. 미회수 시 `lb_unsubmitted` severe.

### G-gates

- G0: STAGE 0 인프라 (모듈 추출 + smoke import + 기존 tests backward-compat) [TODO]
- G1: STAGE 1 1-fold smoke 통과 (cv hit-rate finite, no extraction drift) [TODO]
- G2: STAGE 2 full 5-fold selector 학습 완료 (`oof_selector_scores.npz` + `test_selector_scores.npz` finite) [TODO]
- G3: STAGE 3 full boundary corrector + `submission_boundary_tiny_{soft,argmax}.csv` 생성 [TODO]
- G3.5: STAGE 3.5 18×27 regime distribution 분석 박제 (`analysis/plan-004/regime_distribution.{json,md}`) [TODO]
- G_final: `submission.csv` schema 검증 + dacon-submit 자율 호출 + lb_score 박제 + results.md [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-004-pb-0-6822-fullrun.md` 작성 (본 파일) | [TODO] |
| c2 | code | `src/pb_0_6822/__init__.py` + `selector.py` extract (notebook cell 4 → standalone module). argparse signature 보존. spec @ §4.1 | [TODO] |
| c3 | code | `src/pb_0_6822/boundary.py` extract (cell 6, **유일 수정**: `import train_tcn_gru_candidate_selector as base` → `from src.pb_0_6822 import selector as base`). spec @ §4.2 | [TODO] |
| c4 | test | `tests/test_pb_0_6822_smoke.py` — 모듈 import + `CANDIDATES len==27` + `TinyCorrectionNet` 인스턴스화. spec @ §4.3 | [TODO] |
| G0 | gate | `pytest tests/test_pb_0_6822_smoke.py` + 기존 51 tests green (backward-compat) | [TODO] |
| c5 | code | `src/pb_0_6822/run_full.py` orchestrator + `configs/baseline/P001_pb-0-6822-fullrun.yaml` + `.gitignore` 1줄. spec @ §4.4 | [TODO] |
| c6 | exp smoke | 1-fold smoke (`run_full.py --smoke`) → `runs/baseline/P001_pb-0-6822-fullrun/smoke/`. spec @ §5 | [TODO] |
| G1 | gate | smoke summary finite, no extraction drift | [TODO] |
| c7 | exp selector | Full 5-fold selector (`--fold-limit 5`, no `--skip-full`, pre=10 fine=8 freeze=3 patience=4 epoch_plus=5). spec @ §6 | [TODO] |
| G2 | gate | `oof_selector_scores.npz` + `test_selector_scores.npz` finite + shape OK | [TODO] |
| c8 | exp corrector | Full boundary corrector (`--make-test`, `--test-score-bank`, epochs=12 fine=8 patience=4). spec @ §7 | [TODO] |
| G3 | gate | 2 csv 생성, finite, shape == sample_submission.csv | [TODO] |
| c9 | analysis | `analysis/plan-004/regime_distribution.py` → `regime_distribution.{json,md}`. spec @ §8 | [TODO] |
| G3.5 | gate | 18 regime histogram + 18×27 hit table + degenerate flag + hyper-specialized cell 모두 박제 | [TODO] |
| c10 | sub-gen | `runs/baseline/P001_pb-0-6822-fullrun/submission.csv` = soft csv 사본 + schema 100% 일치 검증. spec @ §9 | [TODO] |
| c11 | sub-lb | **`dacon-submit` skill 자율 호출** + `analysis/plan-004/lb_log.md` + `analysis/plan-004/results.md` + `plans/plan-004-pb-0-6822-fullrun.results.md` frontmatter `lb_score` 박제. spec @ §10 | [TODO] |
| G_final | gate | LB 점수 회수 + 모든 G-gate [DONE] + §0.5 sync | [TODO] |

### Plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `extraction_drift`: c6 smoke 의 cv hit-rate 가 NaN/Inf 또는 추출된 모듈에서 NameError/ImportError 발생 → cell-local 변수 (IPython display 등) 누락 의심.
- `selector_no_convergence`: selector train loss NaN/Inf 또는 OOF top-1 accuracy < 1/27 (= 0.037, 무작위 추측).
- `corrector_no_convergence`: boundary train loss NaN/Inf 또는 학습 후 hit-rate 가 *추가 보정 전* selector 점수보다 낮음.
- `nn_numerical`: 학습 중 NaN/Inf 또는 gradient NaN.
- `regime_degenerate`: 18 regime 중 sample <50 인 regime ≥ 1 (notebook 에 없는 새 검증, **warn only, severe X** — 정보 박제만).
- `regime_unaudited`: G3.5 진입 시점에 `analysis/plan-004/regime_distribution.{json,md}` 미존재.
- `submission_shape_mismatch`: submission.csv shape ≠ sample_submission.csv (row count, column names, id 순서).
- `lb_unsubmitted`: G_final 진입 시점에 `lb_score` 미회수 + carry-over 사유 미박제.
- `dacon_submit_skill_missing`: c11 진입 시 `dacon-submit` skill 부재 → 사용자 escalate.
- `backward_compat_drift`: G0 의 기존 51 tests cv_mean_eucl 가 registry 기존 값과 4 자리 이상 어긋남.

### Plan-specific paths (WORKFLOW.md §12.5/§12.6 추가/제외)

- whitelist 추가:
  - `src/pb_0_6822/__init__.py`, `src/pb_0_6822/selector.py`, `src/pb_0_6822/boundary.py`, `src/pb_0_6822/run_full.py`
  - `tests/test_pb_0_6822_smoke.py`
  - `configs/baseline/P001_pb-0-6822-fullrun.yaml`
  - `runs/baseline/P001_pb-0-6822-fullrun/**` (ckpt 는 `.gitignore` 제외 패턴 적용)
  - `analysis/plan-004/**` (특히 `regime_distribution.{py,json,md}`, `lb_log.md`, `results.md`)
  - `.gitignore` 1회 1줄 수정 (`runs/baseline/P001_*/ckpt/` 패턴 추가)
- blacklist 추가:
  - `notes/PB_0.6822 코드공유.ipynb` (원본 보존, 절대 수정 금지)
  - plan-001/002/003 산출 (`runs/baseline/{B,S,R}00*/**`, `configs/baseline/{B,S,R}00*.yaml`, `analysis/plan-{001,002,003}/**`)

### Decision-note 사용 예 (자율 결정 시 commit msg 박제)

- `decision-note: spec-default — extraction approach (notebook cell → .py module, papermill X) — src/* convention 정합성 + out_dir 제어 깔끔`
- `decision-note: spec-default — exp_id=P001_pb-0-6822-fullrun (P prefix = Public-baseline reimplementation)`
- `decision-note: spec-default — submission=soft csv 사본 (continuous probability-weighted blend, leaderboard 친화)`
- `decision-note: spec-default — epoch budget = notebook default 의 ~70% (Mac mps ~60min wall-time target)`
- `decision-note: spec-default — regime_degenerate threshold = 50 samples (empirical Bayes shrinkage 신뢰성 영역)`
- `decision-note: spec-default — hyper_specialized threshold = (regime, cand) hit-rate 가 해당 regime mean 의 ±50% 이탈`
- `decision-note: spec-default — 1-fold smoke 별도 dir (runs/.../P001*/smoke/) for full output 보호`
- `decision-note: spec-default — c11 dacon-submit 자율 호출 = 사용자 승인 X (CLAUDE.md autonomous policy)`
- `decision-note: spec-default — DATA_ROOT = repo/data/ (open.zip 자동 해제 if needed); notebook 의 LOCAL_DATA_ROOT 경로 무시`

---

## §1. 배경

### §1.1 plan-001/002/003 결과 인계 (registry 기반)

| exp_id | plan | method | cv_mean_eucl | LB |
|---|---|---|---|---|
| **B001_linear-2pt** | 001 | polyfit (w=2, d=1) | **0.01294** ± 0.00058 | **0.60** |
| S001_cspline-natural-full | 002 | cspline natural 11pt | 0.01742 ± 0.00071 | 0.4932 |
| R001_baseline-residual-gru | 003 | linear + GRU 잔차 + Huber | 0.01338 ± 0.00072 | (R006 으로 통합) |
| R006_combined-winners | 003 | R001 복제 (winning=0) | = R001 | **0.5688** |

핵심 인계:
- 가장 높은 LB = **B001 0.60**. R006 이 학습 모델임에도 R001 fallback (winning=0) 으로 LB 0.5688 → 오히려 closed-form 보다 약함.
- **노트북 PB_0.6822 (= 0.6822) 와 우리 best LB (0.60) 의 gap = 0.08** — 본 plan 의 잠재 회수 영역.
- 본 plan 의 목적은 *gap 회수* 자체보다 **노트북 framework 의 우리 데이터 적용 + 18×27 표 박제** (= 다음 plan 의 anchor 정보 확보).

### §1.2 노트북 framework 핵심 (대화 기반 정리)

(이미 사용자와의 사전 대화에서 정리 완료 — 자세한 내용은 `notes/PB_0.6822 코드공유.ipynb` cell 0 의 "Algorithm Notes: Physics Ladder" 참조)

1. 신경망에 *절대 좌표 책임 위임 X*. 27개 *물리 공식 후보 좌표* 가 절대 좌표 책임.
2. **Selector** (Attn-GRU): 27 후보 점수 logit + physics_bias (0.65) + regime_bias (0.45) 가산.
3. **18 regime × 27 candidate empirical Bayes 표**: (speed × curvature × speed_slope) quantile binning 으로 sample → regime, 각 (regime, candidate) cell 에 hit-rate 통계 + shrinkage.
4. **Corrector** (Tiny MLP): Frenet local frame 에서 ±0.6cm cap, zero-init delta, boundary weighting (1cm 근처 sample 에 loss 가중치 집중) 으로 selector 가 놓친 1~2cm zone sample 회수.
5. **2-stage sequential training**: selector → OOF score bank 저장 → corrector full-fit. OOF 로 누수 차단.

본 plan 의 검증 포인트: **18 regime 각각의 train sample 수가 empirical Bayes shrinkage 의 신뢰성 영역 (50+) 안에 있는가?**

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 모듈 추출 | notebook cell 4/6 → `src/pb_0_6822/{selector,boundary}.py` (코드 logic 1:1 보존) |
| 데이터 | project `data/` (open.zip 자동 해제) |
| 학습 | full 5-fold selector + boundary corrector full-fit |
| 분석 | 18-regime × 27-candidate 표 + sample 분포 박제 |
| 제출 | autonomous loop 가 `dacon-submit` skill 1회 호출 (사용자 승인 X) |
| 점수 | LB hit-rate@1cm 1개 회수 + plan frontmatter 박제 |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 노트북 0.6822 *재현* | 사용자 의도 = 우리 데이터 적용, 재현 X (data/seed 차이) |
| hyperparam 튜닝 | 1차 baseline 박제만 — tuning 은 후속 plan |
| 27 후보 수정 / regime 정의 변경 | extraction 단계 1:1 보존 — 변경은 후속 plan 의 ablation 변수 |
| End-to-end 학습 통합 | notebook 의 2-stage sequential 구조 그대로 보존 |
| GPU 환경 변경 | Mac mps 사용 — CUDA 환경은 후속 plan handoff |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

| 분할 | 값 |
|---|---|
| folds | 5 (notebook default, 본 plan 1차 baseline) |
| fold 할당 | `stable_fold_id(sample_id, 5)` (notebook cell 4 L147) — sample_id hash 기반 결정성 |
| seed | 20260606 (notebook cell 12 default), fold 별 변형 없음 |

### §3.2 합격 기준 (정량)

- **G0**: 모듈 import + `len(CANDIDATES) == 27` + `TinyCorrectionNet` 인스턴스화 + 기존 51 tests green
- **G1**: smoke cv hit-rate finite (no NaN/Inf, 0 ≤ hit ≤ 1)
- **G2**: `oof_selector_scores.npz["ens_scores"].shape == (N_train, 27)` + `test_selector_scores.npz["ens_scores"].shape == (N_test, 27)` + finite
- **G3**: `submission_boundary_tiny_soft.csv` shape == `data/sample_submission.csv` shape + 모든 좌표 finite
- **G3.5**: `regime_distribution.json` 에 18 regime histogram + 18×27 hit table + degenerate list 모두 존재
- **G_final**: `lb_score` 회수 (float, isSubmitted=True 응답) + `plans/plan-004-*.results.md` frontmatter 박제

### §3.3 평가 점수

- **CV**: 각 fold 별 selector soft/argmax/gate hit-rate + boundary 후 hit-rate (notebook cell 14 metric summary 형식)
- **LB**: dacon-submit 응답의 `lb_score` (carry-over 시 partial 처리)

---

## §4. STAGE 0 — 모듈 추출 인프라

### §4.1 `src/pb_0_6822/selector.py` (c2)

- 노트북 cell 4 (~94KB, ~2120 lines) 전체 추출
- `if __name__ == "__main__": main()` 보존 (단, 추출 모듈은 import 경로로도 호출 가능해야 함)
- argparse signature 보존: `--folds, --fold-limit, --out-dir, --models, --pre-epochs, --fine-epochs, --freeze-fine-epochs, --epoch-plus, --min-epochs, --patience, --hidden, --batch, --lr, --fine-lr-scale, --prior-strength, --regime-prior-strength, --pairwise-loss-weight, --pairwise-margin, --pairwise-min-label-gap, --fine-distill-weight, --fine-distill-temp, --reverse-pretrain, --norm-real-only, --skip-full, ...`
- `SELECTOR_MAIN = main` 노출 보존 (cell 4 마지막 라인)
- IPython-specific 호출 (display, %matplotlib) 제거 — 없을 것으로 예상

### §4.2 `src/pb_0_6822/boundary.py` (c3)

- 노트북 cell 6 (~22KB, ~517 lines) 전체 추출
- **유일 수정**: `import train_tcn_gru_candidate_selector as base` → `from src.pb_0_6822 import selector as base`
- argparse signature 보존: `--folds, --fold, --out-dir, --hidden, --epochs, --fine-epochs, --min-epochs, --patience, --batch, --lr, --fine-lr-scale, --cap, --apply-scale, --low, --high, --far-weight, --prior-strength, --regime-prior-strength, --score-bank, --score-key, --make-test, --test-score-bank, --test-score-key, --save-val-pred, --env-loss-weight, --seed, --device`
- `BOUNDARY_MAIN = main` 노출 보존

### §4.3 `tests/test_pb_0_6822_smoke.py` (c4)

```python
def test_selector_import():
    from src.pb_0_6822 import selector
    assert len(selector.CANDIDATES) == 27
    assert all(hasattr(c, "name") for c in selector.CANDIDATES)

def test_boundary_import():
    from src.pb_0_6822 import boundary
    assert boundary.TinyCorrectionNet is not None
    # 인스턴스화 (no forward)
    import torch
    model = boundary.TinyCorrectionNet(dim=20, hidden=64)
    assert model is not None

def test_regime_functions_signature():
    from src.pb_0_6822 import selector
    assert callable(selector.fit_regime_bins)
    assert callable(selector.assign_regimes)
    assert callable(selector.candidate_regime_bias)
```

### §4.4 `src/pb_0_6822/run_full.py` orchestrator (c5)

```python
# pseudo
from pathlib import Path
from src.pb_0_6822 import selector, boundary

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
OUT_DIR  = REPO / "runs/baseline/P001_pb-0-6822-fullrun"

def ensure_data_extracted():
    """data/open.zip → data/{train,test,train_labels.csv,sample_submission.csv} 자동 해제"""
    if not (DATA_ROOT / "train_labels.csv").exists():
        import zipfile
        with zipfile.ZipFile(DATA_ROOT / "open.zip") as z:
            z.extractall(DATA_ROOT)

def run_smoke():
    """1-fold smoke (mirror notebook cell 10/12 args)"""
    smoke_dir = OUT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    selector.main([
        '--root', str(DATA_ROOT), '--out-dir', str(smoke_dir),
        '--models', 'attn_gru',
        '--folds', '5', '--fold-limit', '1',
        '--pre-epochs', '1', '--fine-epochs', '1', '--freeze-fine-epochs', '1',
        '--epoch-plus', '0', '--min-epochs', '1', '--patience', '1',
        '--hidden', '48', '--batch', '4096',
        '--skip-full',  # smoke 는 full-fit X
        # 기타 default
    ])

def run_full():
    """Full 5-fold + corrector full-fit + submission"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # selector full
    selector.main([
        '--root', str(DATA_ROOT), '--out-dir', str(OUT_DIR),
        '--models', 'attn_gru',
        '--folds', '5', '--fold-limit', '5',
        '--pre-epochs', '10', '--fine-epochs', '8', '--freeze-fine-epochs', '3',
        '--epoch-plus', '5', '--min-epochs', '5', '--patience', '4',
        '--hidden', '48', '--batch', '4096',
        # no --skip-full → test_selector_scores.npz 자동 생성
    ])
    # boundary full
    boundary.main([
        '--root', str(DATA_ROOT), '--out-dir', str(OUT_DIR),
        '--folds', '5', '--fold', '0',
        '--score-bank', str(OUT_DIR / 'oof_selector_scores.npz'),
        '--test-score-bank', str(OUT_DIR / 'test_selector_scores.npz'),
        '--epochs', '12', '--fine-epochs', '8', '--min-epochs', '5', '--patience', '4',
        '--hidden', '64', '--batch', '8192',
        '--cap', '0.006', '--apply-scale', '1.0',
        '--make-test',
    ])

if __name__ == "__main__":
    import sys
    ensure_data_extracted()
    if '--smoke' in sys.argv:
        run_smoke()
    else:
        run_full()
```

---

## §5. STAGE 1 — 1-fold Smoke (c6)

- 실행: `python -m src.pb_0_6822.run_full --smoke`
- 산출: `runs/baseline/P001_pb-0-6822-fullrun/smoke/` (별도 dir, full 산출과 격리)
- 검증: `summary.json` 또는 stdout 의 cv hit-rate finite
- 시간 예산: ~5min (1 epoch × 1 fold)

---

## §6. STAGE 2 — Full Selector 5-fold (c7)

- 실행: `python -m src.pb_0_6822.run_full` (selector phase 만)
- 산출:
  - `runs/baseline/P001_pb-0-6822-fullrun/oof_selector_scores.npz` (OOF, shape `(N_train, 27)`)
  - `runs/baseline/P001_pb-0-6822-fullrun/test_selector_scores.npz` (full-fit test, shape `(N_test, 27)`)
- 검증: 두 npz 모두 finite, shape 정합
- 시간 예산: ~25~40min (mps)

---

## §7. STAGE 3 — Full Boundary Corrector (c8)

- 실행: 위 run_full 의 boundary phase
- 산출:
  - `runs/baseline/P001_pb-0-6822-fullrun/submission_boundary_tiny_soft.csv`
  - `runs/baseline/P001_pb-0-6822-fullrun/submission_boundary_tiny_argmax.csv`
  - `runs/baseline/P001_pb-0-6822-fullrun/boundary_tiny_correction_report.json`
- 검증: csv shape == sample_submission.csv shape, finite
- 시간 예산: ~15~25min (mps)

---

## §8. STAGE 3.5 — 18×27 Regime Distribution Analysis (c9)

### §8.1 분석 스크립트

```python
# analysis/plan-004/regime_distribution.py
import json
import numpy as np
from pathlib import Path
from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
OUT_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
ANALYSIS_DIR = REPO / "analysis/plan-004"

def main():
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1

    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)
    hist = np.bincount(regimes, minlength=18)

    cands = selector.make_candidates(train_x, end_idx, horizon=2)
    err = np.linalg.norm(cands - train_y[:, None, :], axis=2)
    R_HIT = selector.R_HIT  # 0.01

    # 18×27 hit-rate table
    hit_table = np.zeros((18, 27), dtype=np.float64)
    mean_dist_table = np.zeros((18, 27), dtype=np.float64)
    for r in range(18):
        mask = regimes == r
        if mask.sum() > 0:
            hit_table[r] = (err[mask] <= R_HIT).mean(axis=0)
            mean_dist_table[r] = err[mask].mean(axis=0)

    # Degenerate flags
    degenerate_regimes = [int(r) for r in range(18) if hist[r] < 50]
    # Hyper-specialized cells: (regime, candidate) hit-rate ≥ regime_mean × 1.5 또는 ≤ × 0.5
    regime_means = hit_table.mean(axis=1)
    hyper_cells = []
    for r in range(18):
        for c in range(27):
            if regime_means[r] > 0:
                ratio = hit_table[r, c] / regime_means[r]
                if ratio >= 1.5 or (ratio <= 0.5 and hit_table[r, c] >= 0.01):
                    hyper_cells.append({
                        "regime": int(r),
                        "candidate": int(c),
                        "candidate_name": selector.CANDIDATES[c].name,
                        "hit_rate": float(hit_table[r, c]),
                        "regime_mean": float(regime_means[r]),
                        "ratio": float(ratio),
                    })

    summary = {
        "n_total": int(len(train_y)),
        "regime_histogram": [int(h) for h in hist],
        "regime_bin_edges": {k: list(v) for k, v in bins.items()},
        "candidate_names": [c.name for c in selector.CANDIDATES],
        "hit_table": hit_table.tolist(),
        "mean_dist_table": mean_dist_table.tolist(),
        "regime_means": regime_means.tolist(),
        "degenerate_regimes": degenerate_regimes,
        "degenerate_count": len(degenerate_regimes),
        "hyper_specialized_cells": hyper_cells,
        "hyper_specialized_count": len(hyper_cells),
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "regime_distribution.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary)

def write_markdown(summary):
    """regime_distribution.md 작성 — 18×27 markdown table + flag list"""
    ...

if __name__ == "__main__":
    main()
```

### §8.2 산출

- `analysis/plan-004/regime_distribution.json` — 전체 분석 결과
- `analysis/plan-004/regime_distribution.md` — 사람 읽는 markdown 표 + 해석

### §8.3 검증

- regime histogram 의 18 entry 모두 존재
- 18×27 table 의 모든 cell 이 finite (`degenerate_regimes` 의 cell 은 0 가능)
- degenerate count + hyper-specialized count 모두 보고

---

## §9. STAGE 4 — Submission 준비 (c10)

- `runs/baseline/P001_pb-0-6822-fullrun/submission.csv` = `submission_boundary_tiny_soft.csv` 사본
- 검증: `pandas.read_csv` 로 `submission.csv` 와 `data/sample_submission.csv` shape + columns + id 순서 100% 일치

---

## §10. STAGE 5 — LB 제출 + 박제 (c11)

- `Skill(skill="dacon-submit", args="runs/baseline/P001_pb-0-6822-fullrun/submission.csv P001_pb-0-6822-fullrun")` 자율 호출
- 응답: `{isSubmitted: True/False, lb_score: float/None, detail: str}`
- 산출:
  - `analysis/plan-004/lb_log.md` 1행 박제 (KST 시각 + isSubmitted + lb_score)
  - `analysis/plan-004/results.md` 작성 (`plan-003/results.md` 형식 참고)
  - `plans/plan-004-pb-0-6822-fullrun.results.md` 작성 (frontmatter `lb_score`)
  - `plans/plan-004-pb-0-6822-fullrun.md` frontmatter `lb_score: <float>` 갱신 (§12.6 blacklist 의 §0.5 [TODO]→[DONE] 갱신과 동일 예외)

---

## §11. 작업량 총 회계

- 코드 추출: cell 4 (~94KB) + cell 6 (~22KB) = ~117KB 1회성 translation
- 학습: selector ~30min + corrector ~20min = ~50min (mps)
- 분석: ~2min
- 제출: 1 API call
- **총 wall-time 예산: ~60min**

---

## §N+2. results.md 필수 항목

(plan-003 format 참조)

- exp_id, plan_id, lb_exp_id, lb_score, lb_submitted_at
- per-fold CV metrics + soft/argmax/gate hit-rate
- 18×27 표 요약 (degenerate count, hyper-specialized count)
- 다음 plan 후보 (post-G_final 분석 기반)
- decision-note 박제 list

---

## §N+3. 통계 함정 & caveats

1. **노트북 0.6822 ≠ 우리 LB**: notebook 작성자의 환경/seed/데이터 분할이 다를 수 있어 우리 점수가 0.6822 보다 낮을 수 있음. *어떤 점수든 회수하면 G_final 충족* (재현이 목표 아님).
2. **Mac mps 정확도 차이**: 일부 op (특히 GRU 의 한국형 dtype) 가 CUDA 대비 hit-rate 1~2pp 손실 가능. 본 plan 1차 baseline 박제 후 후속 plan 에서 CUDA handoff.
3. **18 regime 의 도메인 적합성**: 노트북은 모기 비행 가정으로 (speed × curvature × speed_slope) 축 선택. 우리 데이터 (동일 dacon) 가 같은 분포라면 호환, 다르면 degenerate regime 증가. **G3.5 결과가 후속 plan 의 regime 재설계 anchor**.
4. **2-stage sequential vs end-to-end**: 본 plan 은 노트북 그대로 sequential. end-to-end 통합은 ablation 가치 있으나 후속 plan.
5. **dacon-submit 의 lb_score 비동기**: plan-002/003 패턴 — `lb_score: TBD` + `status: partial` 마감 후 점수 도착 시 follow-up commit 으로 `all_complete` 갱신.

---

## §N+4. 변경 이력

- v1 (2026-05-11): 초안 — plan-004 신규 작성, c1~c11 commit chain 박제, G0~G_final 7개 gate 정의.

---

## §N+5. 참조

- `notes/PB_0.6822 코드공유.ipynb` (소스, read-only)
- `WORKFLOW.md` §0.5, §11, §12 convention
- `plans/plan-003-residual-gru-grid.md` (format reference, c12/c13/c14 LB 자율 제출 패턴)
- `CLAUDE.md` (autonomous execution policy)
- `registry.csv` (exp_id append target)
- `src/submit.py` (dacon-submit infra)
