---
plan_id: w-001
version: 1
date: 2026-06-05 (Asia/Seoul)
status: in_progress
inspired_by:
  - a-003 (KR008 LB 0.6862, paradigm = Kalman residual GRU — 우리 best 비교군)
  - d-001 (NODE001 OOF 0.6330, paradigm = Neural ODE — 동일 paradigm 의 lower-tier 구현)
  - c-001 (FR001 OOF 0.6622, paradigm = F0 residual GRU — 동일 "base+residual GRU" 구조)
code_reuse:
  - module: notes/winners/rank1-euijin42/md_file.html
    symbols: [cell 1~25]
    reason: 노트북 원본 — 셀별 dump → notebook_dump.md → repo/GOH30_reproduce.py 로 단일 module 추출 (1:1 재현). 자율 수정 없음.
  - module: src/io.py
    symbols: [load_sample placeholder]
    reason: 우리 data path 인터페이스. 단 GOH30_reproduce.py 는 노트북 코드 보존 위해 자체 load_sample 가짐.
exp_ids:
  - W001_repro                  # 노트북 그대로 30모델 학습 + submission_GOH30.csv 생성
  - W001_lb_submit              # DACON 제출 → LB 비교 vs notebook self-LB 0.7035
winner_meta: notes/winners/rank1-euijin42/meta.yaml
---

# plan-w-001 — rank1 euijin42 (private LB 0.7035) 재현

## §0. 한 줄 목적

> **rank1 euijin42 노트북 (GRU + Neural ODE + HyperPhysics 30모델 등가중 블렌드, private LB 0.7035) 을 우리 환경에서 1:1 재현하고 DACON 제출로 LB 검증. 우리 best plan-a-003 KR008 LB 0.6862 대비 Δ +0.0173 차이의 paradigm/lever 격차를 measured 박제. 정확한 OOF 비교는 노트북이 full-fit 구조라 별도 fold-split wrapper 필요 (W001_oof는 follow-up).**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | GRU + Neural ODE + HyperPhysics blend (30 model, equal-weight) |
| baseline (우리 비교군) | plan-a-003 KR008 (Kalman residual GRU, LB 0.6862, OOF 0.6671) |
| 노트북 URL | https://dacon.io/competitions/official/236716/codeshare/14013 |
| 노트북 자기-LB | private 0.7035 (public ~0.69 추정) |
| 외부 데이터 사용 | no (DACON train/test/labels 만) |
| 사전학습 모델 사용 | no (from scratch 30 model) |
| metric | hit_1cm (R-Hit@1cm) — submission 단일 좌표 평가 |
| 합격 기준 | G_repro: notebook self-LB private 0.7035 ±0.005 (= 0.6985~0.7085) |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) + meta.yaml + notebook_dump.md | [DONE] |
| c1 코드 이식 | notes/winners/rank1-euijin42/repo/GOH30_reproduce.py — 단일 module | [DONE] |
| c2 smoke | analysis/plan-w-001/smoke.py — N_EACH=1 GRU_ODE_EPOCHS=2 H_EPOCHS=2 1 model finite check | [TODO] |
| c3 G_repro full | N_EACH=10 GRU_ODE_EPOCHS=55 H_EPOCHS=12 full-fit 30 model → submission_GOH30.csv | [TODO] |
| c4 G_lb (gated) | dacon-submit submission_GOH30.csv → LB Δ vs 0.7035 | [TODO] |
| c_final | plan-w-001-rank1-euijin42.results.md + §0.5 sync + EXPERIMENTS.md row + main merge | [TODO] |

### G-gates

- G0 (c0~c1): meta.yaml + notebook_dump + extracted .py 존재. **[DONE]**
- G_smoke (c2): N_EACH=1 1-seed 1-arch finite + submission shape (10000, 4). **[TODO]**
- G_repro (c3): 30 model 학습 완료 (~30~45min L40S 추정), submission_GOH30.csv (10000 row). **[TODO]**
- G_lb (c4, gated): notebook self-LB 0.7035 ±0.005 안. 사용자 confirm 후 DACON 제출 1회. **[TODO]**
- G_final (c_final): results 박제 + EXPERIMENTS row + main merge. **[TODO]**

### plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `winner_license_unknown` — meta.yaml license=unknown. **G_lb 진입 전 사용자 confirm 필수** (단순 LB 검증은 허용, repo 영구화/공개는 사용자 결정).
- `lb_repro_miss_0p01` — G_lb LB |Δ| > 0.01 면 single-variable 격리 1회 후 그래도 안 들면 `band: failed` 박제.

### plan-specific paths

- whitelist 추가: `notes/winners/rank1-euijin42/**` · `analysis/plan-w-001/**` · `runs/baseline/W001_*/**` · `models_goh30/**` · `submission_GOH30.csv`
- blacklist 추가: 우리 plan-a/b/c/d analysis/ 본문 (직접 수정 금지 — 비교 carry 만)

---

## §1. 배경

### §1.1 어느 문을 다시 여나

- **Neural ODE**: plan-d-001 (NODE001 OOF 0.6330, F0 floor +0.0010 통계 동률, overfitting 으로 inconclusive). rank1 의 ODE 도 RK4 nsteps=4 + damping vector 거의 동일 구조이나 **cv_1step residual 학습** + **cache 50K (interior transfer)** + **EMA** + **Cosine LR 55ep** + **soft R-Hit loss** 가 추가. → "Neural ODE paradigm 자체가 약함" 보다 "학습 procedure 의 차이가 carrier" 가설.
- **arch ensemble**: plan-018 (4/4 arch ablation ALL FAIL). 단 plan-018 은 single-stack 비교, rank1 은 **3개 서로 다른 inductive bias arch (GRU / ODE / HyperPhysics) 의 blend**. "single arch 한계 ↔ multi-arch blend 가치" 분리 검증.
- **HyperPhysics gray-box**: 우리에게 *없는* lever. Rodrigues 회전 기반 뱅킹 턴 + θ/speed 게이팅. plan-020 의 fixed-physics F0 14종 모두 falsify 했지만 **학습 가능 물리 + θ-가중 oversampling** 은 우리가 시도 안 함.

### §1.2 inspire 받을 lever (없으면 reproduce only)

- **Soft R-Hit loss** (1cm sigmoid 근사) — plan-031/032 의 training procedure carry 와 같은 계열, 우리 KR* 는 아직 안 씀.
- **Cache pretrain (interior transfer e∈{5,6,7,8})** — internal trajectory points 도 학습 신호로 사용. data augmentation × 5.
- **Y-flip TTA** — 좌우대칭 활용. 우리 plan-a-001 yaw 회전이 학습 시 augment 만 했고 inference TTA 안 함.
- **θ-가중 oversampling** — 급선회 샘플 최대 5x 가중. tail (high-curvature) 직접 강조.

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 재현 | 노트북 1:1 (코드 변경 없음, 환경변수 토글만) |
| 비교 | notebook self-LB 0.7035 ±0.005 (G_repro 합격선) |
| 제출 | 1회 (gated, 사용자 confirm) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 노트북 + 우리 plan ensemble | 별도 plan-w-002 로 분리 (자기-완결 §9-3) |
| 노트북 lever 단독 ablation | 별도 plan 으로 분리 (e.g., plan-w-003 = Soft R-Hit loss 만 KR008 에 적용) |
| OOF 비교 (paired-perm vs KR008) | 노트북 full-fit 구조 — fold split wrapper 별도 작업 필요, follow-up plan 으로 |
| HyperPhysics hyper 수정 | 노트북 명시값 (theta_thr=1.087618 등) 그대로 |
| 외부 데이터 / 사전학습 weight | 노트북에 없음 |

---

## §3. 사전 등록

### §3.1 Fold split

- **W001_repro**: full-fit (10K train all-in, no fold). 노트북 원본 그대로.
- (W001_oof = follow-up plan-w-002 에서 5-fold wrapper)

### §3.2 합격 기준

| gate | 조건 |
|---|---|
| G_smoke | N_EACH=1 1-seed 1-arch finite (no NaN), submission shape (10000, 4) |
| G_repro | 30 model 학습 완료, submission_GOH30.csv (10000 row, finite, schema match) |
| G_lb | private LB Δ from 0.7035 ∈ [-0.005, +0.005] = PASS / [-0.01, -0.005] = caveat / outside = failed |

### §3.3 평가

- metric: hit_1cm (DACON 자동 평가)
- 통계: DACON LB private 값 단일 (paired-perm 불가, full-fit 이라 OOF 없음)
- caveat: public LB 는 ±0.005 tolerance 외일 수 있음 (private 0.7035 는 final 기준)

---

## §4. 서버 작업 순서

| 단계 | 산출물 |
|---|---|
| 1. notebook_dump | `notes/winners/rank1-euijin42/notebook_dump.md` (셀 25개 추출) **[DONE]** |
| 2. meta.yaml | `notes/winners/rank1-euijin42/meta.yaml` (handle, score, dependencies, hyperparams) **[DONE]** |
| 3. 이식 | `notes/winners/rank1-euijin42/repo/GOH30_reproduce.py` (단일 module, 환경변수 토글) **[DONE]** |
| 4. smoke | `analysis/plan-w-001/smoke.py` — N_EACH=1, GRU_ODE_EPOCHS=2, H_EPOCHS=2, finite + shape check |
| 5. G_repro | `DATA_DIR=./data MODELS_DIR=./models_goh30 N_EACH=10 python -m notes.winners.rank1-euijin42.repo.GOH30_reproduce` → submission_GOH30.csv |
| 6. (gated) G_lb | `dacon-submit submission_GOH30.csv` → LB Δ 박제 |
| 7. G_final | results.md + EXPERIMENTS row + main merge |

---

## §5. results.md 필수 항목

- §0 한 줄 결론 (재현 PASS/FAIL · LB Δ from 0.7035 · 우리 KR008 0.6862 대비 격차)
- §1 가설 판정 (재현 가능성, paradigm 격차 carrier 추정)
- §2 Gate 판정 (G_smoke / G_repro / G_lb)
- §3 lesson 박제 (Soft R-Hit / cache pretrain / TTA / θ-oversample 중 어느 lever 가 핵심인지 추정)
- §4 EXPERIMENTS.md row 박제 (한 줄로 합류)
- §5 follow-up candidate (lever 단독 ablation plan 후보)

---

## §6. 통계 함정 & caveats

- **Device 차이**: 노트북 MPS (Apple Silicon), 우리 L40S CUDA. EMA dtype 캐스팅·fp 정밀도 차이로 ±0.001~0.003 LB 변동 가능. ±0.005 tolerance 안에 들어야 PASS.
- **시드 영향**: 노트북 set_seed(1000+k) 고정. 같은 시드여도 CUDA 비결정성 가능 → 같은 seed 라도 100% 동일 X.
- **30 model blend** = noise floor 줄이는 효과. ±0.001 변동은 blend 후 평균에서 흡수.
- **DACON LB**: private 는 final 기준, public 은 일일 변동. submit 시 private 값 확인 필요.
- **Full-fit**: held-out 없음 — overfit 검증 불가. notebook self-LB 가 신뢰 단일 metric.

---

## §7. 참조

- `notes/winners/rank1-euijin42/md_file.html` (DACON S3 presigned, 2026-06-05 fetch)
- `notes/winners/rank1-euijin42/notebook_dump.md` (cell 25개 추출)
- `notes/winners/rank1-euijin42/meta.yaml`
- `notes/winners/rank1-euijin42/repo/GOH30_reproduce.py` (단일 module)
- `WINNERS.md` (재현 protocol 전체)
- `EXPERIMENTS.md` (우리 비교군)
- `plans/plan-a-003-reflect-noise-augment.md` + `.results.md` (KR008 = 우리 best)
- `plans/plan-d-001-neural-ode-repro.md` + `.results.md` (NODE001 = 동일 paradigm 비교군)
- WORKFLOW.md §4 (lane mutex), §12 (autonomous protocol)
