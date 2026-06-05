---
plan_id: w-{NNN}         # winners lane. WORKFLOW.md §4 참조. NNN = lane w 내부 발행 번호.
version: 1
date: YYYY-MM-DD (Asia/Seoul)
status: draft
inspired_by:
  - (우리 plan 중 동일 paradigm — 예: a-001 if Kalman-residual / 018 if arch ablation)
  - (다른 winner plan 이 있으면 그 plan_id)
code_reuse:
  - module: notes/winners/{handle}.ipynb
    symbols: [모델 정의 셀, 학습 셀, predict 셀]
    reason: 노트북 원본 — 재현 1:1, 자율 수정 시 decision-note 박제
  - module: analysis/plan-a-001/run_oof.py
    symbols: [main, fold_split, hit_rate]
    reason: 우리 fold split + 평가 함수. winner 의 split 와 비교용
exp_ids:
  - W{NNN}_repro                 # 노트북 자체 재현
  - W{NNN}_oof_our_split         # 우리 stable fold split 위 hit@1cm
  - W{NNN}_lb_submit             # gated, DACON 제출
winner_meta: notes/winners/{handle}.meta.yaml
---

# plan-w-{NNN} — {handle} ({rank}위, {paradigm}) 재현

## §0. 한 줄 목적

> **rank{N} 노트북({paradigm}, 노트북 LB {0.XXXX})을 우리 환경에서 재현하고, 우리 plan-{X} 와 OOF hit@1cm + LB 동등 비교. paradigm 일치/차이·lever 단독 영향·CV-LB 괴리 여부를 박제. winner 라서 강조 X — 우리 plan 들과 같은 row 로 EXPERIMENTS.md 합류.**

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

| 항목 | 값 |
|---|---|
| paradigm | {Kalman-residual / Transformer / corrector / TabNet / ...} |
| baseline (우리 비교군) | plan-{X} ({exp_id}, OOF {0.YYYY} / LB {0.YYYY}) |
| 노트북 URL | https://dacon.io/competitions/official/236716/codeshare/XXXX |
| 노트북 자기-LB | {public 0.XXXX / private 0.YYYY} |
| 외부 데이터 사용 | yes/no (yes 면 라이선스 확인) |
| 사전학습 모델 사용 | yes/no (yes 면 라이선스 확인 — 대회 규칙 §6.1) |
| metric | hit_1cm (우리 paired-perm vs plan-{X}); 추가 hit_1.5cm |
| 합격 기준 | G_repro: 노트북 자기-LB ±0.005 / G_oof: paired-perm vs plan-{X} / G_lb: LB 제출 (gated) |

### Commit chain (예정)

| commit | spec | status |
|---|---|---|
| c0 spec | §0~§7 (본 파일) + plan-review-master | [TODO] |
| c1 노트북 dump | `analysis/plan-w-{NNN}/notebook_dump.md` 셀별 코드 요약 + `notes/winners/{handle}.meta.yaml` 작성 | [TODO] |
| c2 코드 이식 | 노트북 셀 → `analysis/plan-w-{NNN}/{model,train,predict}.py` 분리 | [TODO] |
| c3 smoke | `tests/test_plan_w{NNN}_smoke.py` — model fwd / 1f1s1e finite / shape OK | [TODO] |
| c4 G_repro (W{NNN}_repro) | 노트북 자체 OOF 재현 → `runs/baseline/W{NNN}_repro/summary.json` | [TODO] |
| c5 G_oof (W{NNN}_oof_our_split) | 우리 stable fold split 위 hit_1cm + paired-perm vs plan-{X} | [TODO] |
| c6 (gated) G_lb (W{NNN}_lb_submit) | DACON 제출 + LB 박제 | [TODO] |
| c_final | `plan-w-{NNN}-{handle}.results.md` + §0.5 sync + EXPERIMENTS.md row 추가 + main merge | [TODO] |

### G-gates

- G0: c1~c3 인프라 (notebook_dump + meta.yaml + 이식 + smoke green) [TODO]
- G_repro (c4): 노트북 자기-OOF ±0.005 안. 밖이면 single-variable 격리 1회 → 그래도 안 들면 `band: failed` 박제. [TODO]
- G_oof (c5): paired-perm p < 0.05 (positive/negative 둘 다 OK — sign + p 박제). [TODO]
- G_lb (c6, gated): 사용자 confirm 후 dacon-submit 1회. LB Δ 박제. [TODO]
- G_final (c_final): results 박제 + EXPERIMENTS.md 합류 + main merge. [TODO]

### plan-specific severe (WORKFLOW.md §12.3 default 위 추가분)

- `winner_license_unknown` — `notes/winners/{handle}.meta.yaml` 의 `license` 가 noted MIT/Apache 2.0/CC BY 가 아니거나 unknown 이면 G_repro 진입 전 멈춤 + escalate.
- `winner_externaldata_undeclared` — 노트북이 train_labels 외 데이터 사용하는데 meta 에 `data_used` 미선언 시 멈춤.

### plan-specific paths

- whitelist 추가: `analysis/plan-w-{NNN}/**` · `notes/winners/{handle}.*` · `runs/baseline/W{NNN}_*/**`
- blacklist 추가: 우리 plan-a/b/c/d 의 analysis/ 본문 (직접 수정 금지 — 비교는 carry 만)

---

## §1. 배경

### §1.1 어느 문을 다시 여나

(이 노트북이 우리 어느 plan 의 닫힌 가설을 다시 여는지. 예시:
- "rank1 이 Transformer 기반 → 우리 plan-018 arch ablation 이 falsify 한 H1 encoder bottleneck 을 재검"
- "rank1 도 Kalman 잔차 → plan-a-001~003 와 같은 paradigm, lever 차이만 비교")

### §1.2 inspire 받을 lever

- (노트북에서 우리에게 *없는* lever 1~3개. 예: hit-aware loss, TTA 회전, 외부 사전학습 backbone)

---

## §2. Scope (명시적)

### §2.1 In-scope

| 항목 | 값 |
|---|---|
| 재현 | 노트북 1:1 (random seed / fold split 차이만 수정) |
| 비교 | 우리 plan-{X} 와 동일 split 위 hit_1cm + paired-perm |
| 제출 | 1 회 (gated, 사용자 confirm) |

### §2.2 Out-of-scope (절대 안 함)

| 항목 | 이유 |
|---|---|
| 노트북 + 우리 plan ensemble | 별도 plan-w-{NNN+1} 로 분리 (자기-완결 §9-3) |
| 노트북 lever 단독 ablation | 별도 plan 으로 분리 |
| 외부 추가 데이터 도입 | 노트북에 명시된 것만 사용 |
| 사전학습 weight 비-라이선스 모델 | 대회 규칙 §6.1 위반 |

---

## §3. 사전 등록

### §3.1 Fold split

- repro: 노트북 명시 split (또는 노트북의 자체 OOF 정의)
- our_split: `analysis/plan-a-001/run_oof.py` 의 stable fold split (5-fold seed 20260514)

### §3.2 합격 기준

| gate | 조건 |
|---|---|
| G_repro | `\|repro_oof_hit_1cm - notebook_self_oof\| < 0.005` |
| G_oof | paired-perm vs plan-{X} (best-comparable) p < 0.05, sign + Δ 박제 |
| G_lb | submit 1회, LB Δ 박제 (positive/negative 둘 다 OK) |

### §3.3 평가

- metric primary: hit_1cm
- metric secondary: hit_1.5cm (선택)
- 통계: paired sign-flip permutation 10k

---

## §4. 서버 작업 순서

| 단계 | 산출물 |
|---|---|
| 1. notebook_dump | `analysis/plan-w-{NNN}/notebook_dump.md` (셀별 코드/주석 요약) |
| 2. meta.yaml | `notes/winners/{handle}.meta.yaml` (§2 WINNERS.md schema) |
| 3. 이식 | `model.py` / `train.py` / `predict.py` (셀 분리) |
| 4. smoke | `tests/test_plan_w{NNN}_smoke.py` (1f1s1e finite) |
| 5. G_repro | `run_oof.py` repro mode → `runs/baseline/W{NNN}_repro/` |
| 6. G_oof | `run_oof.py our_split mode + compare.py` → paired-perm |
| 7. (gated) G_lb | `dacon-submit` skill |
| 8. G_final | results.md + EXPERIMENTS.md row + main merge |

---

## §5. results.md 필수 항목

- §0 한 줄 결론 (재현 PASS/FAIL · 우리 best 와 paired Δ · LB Δ)
- §1 가설 판정 (H1 재현가능 / H2 우리보다 위 / H3 우리에게 없는 lever 단독 가치)
- §2 Gate 판정 (G_repro / G_oof / G_lb)
- §3 lesson 박제 (paradigm 일치·차이 / CV-LB 괴리 / 외부 lever 가치)
- §4 EXPERIMENTS.md row 박제 (한 줄로 합류)

---

## §6. 통계 함정 & caveats

- 노트북 LB 와 우리 LB 가 다른 split (public vs private) 이면 ±0.005 tolerance 보수적.
- 노트북에 명시 안 된 hyperparameter (예: dropout) 는 우리 default 채택 + `decision-note: spec-default`.
- TTA / multi-seed 가 노트북에 있으면 그대로 carry, 없으면 추가하지 않음 (단일 변수 원칙).
- paired-perm 의 paired 단위 = sample (같은 fold/seed 내).

---

## §7. 참조

- `notes/winners/{handle}.ipynb` (노트북 원본)
- `notes/winners/{handle}.meta.yaml` (메타데이터)
- `WINNERS.md` (재현 protocol 전체)
- `EXPERIMENTS.md` (비교 대상 plan 들)
- 우리 비교군 plan: `plans/plan-{X}-*.md`
- WORKFLOW.md §4 (lane mutex), §12 (autonomous protocol)
