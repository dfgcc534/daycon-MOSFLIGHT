# WINNERS — 수상자 코드 재현 protocol

대회 종료 후 수상자가 공개한 노트북(또는 repo)을 우리 코드베이스 위에서 재현하고, 우리 plan 들과 *동등한 row* 로 `EXPERIMENTS.md` 표에 합류시킨다.

수상자 결과를 "정답" 으로 두지 않는다. plan-a/b/c/d 와 동일한 row 구조로 비교 — band(positive/negative/inconclusive)·OOF·LB·lesson 4 컬럼이 채워질 때까지가 재현 1건.

---

## §1. 수상자 1건 = lane `w` plan 1개

수상자 노트북 N개 = `plan-w-001`, `plan-w-002`, ... (lane `w` = winners). `WORKFLOW.md §4` 의 lane mutex 그대로 적용. 각 plan 은 `plans/plan-template-winner.md` 골격 사용.

```
plans/
  plan-w-001-{winner-handle-or-rank}.md           ← 재현 plan (요청)
  plan-w-001-{winner-handle-or-rank}.results.md   ← 재현 결과 (응답)
notes/winners/
  {winner-handle-or-rank}.ipynb                   ← 원본 노트북 (다운로드)
  {winner-handle-or-rank}.meta.yaml               ← 메타데이터 (URL, LB score, 발견일 등)
analysis/plan-w-001/
  preflight.py                                    ← G0 환경/의존성 점검
  run_oof.py                                      ← OOF (5-fold) 재현 → hit@1cm
  run_test.py                                     ← test predict + submission.csv
  compare.py                                      ← 우리 plan vs 수상자 paired-perm
runs/baseline/W001_{winner-handle}/
  config.snapshot.yaml
  summary.json   history.json   {stage}.log
```

`W001` = exp_id (registry.csv). `winner-handle` = DACON 닉네임 또는 "rank{N}" (예: `rank1`, `rank2`).

---

## §2. notes/winners/{handle}.meta.yaml 스키마

원본 노트북 옆에 메타데이터 1 파일. 모든 필드 필수.

```yaml
handle: rank1                  # 또는 DACON 닉네임
rank: 1                        # private LB 순위
score:
  public: 0.6XXX               # 공개 가능 시
  private: 0.6XXX              # 최종 순위 기준
source_url: https://dacon.io/competitions/official/236716/codeshare/XXXX
fetched_at: 2026-06-XX
license: (노트북에 명시된 라이선스 / CC BY-NC 등 / unknown)
files:
  - notebook: notes/winners/rank1.ipynb
  - data_used: [train, train_labels, test, ...]      # train_labels 외 추가 데이터 사용 여부
paradigm_tag: (Kalman-residual / Transformer / TabNet / ...)  # 우리 EXPERIMENTS paradigm 컬럼과 align
reproduction_status: pending|in_progress|complete|failed
notes: |
  자유 메모 (외부 데이터 사용 / 사전학습 모델 / 특이 trick 등)
```

`reproduction_status` 는 plan 진행에 따라 update.

---

## §3. Plan-w 골격 (plan-template-winner.md)

`plans/plan-template-winner.md` 가 신규 winner plan 의 골격. 기존 `plan-template.md` 와 동일한 frontmatter + §0/§0.5/§1~§7 구조에 winner-specific 섹션 추가:

| §  | winner plan 에 채울 것 |
|---|---|
| §0 한 줄 목적 | "rank{N} 노트북 ({paradigm}) 을 우리 환경에서 재현, 우리 plan-{best-comparable} 와 OOF hit@1cm + LB 비교" |
| §0.5 Quick Reference | paradigm / 노트북 URL / 원 LB / 재현 target / G-gates |
| §1 배경 | 어느 plan 의 "닫힌 문" 을 다시 여는지 (예: plan-018 arch ablation falsify 했는데 rank1 이 Transformer 면 →  arch 가설 재검) |
| §2 가설 | "rank{N} 의 핵심 lever 가 우리 plan-{X} 에 더해지면 LB +Y" 식. 단순 재현이면 "노트북 LB ±0.005 재현" |
| §3 실험 | W{NNN} 1~3 exp_id — 단일 재현 / 우리 best 와 ensemble / lever 단독 ablation |
| §4 서버 작업 순서 | preflight → run_oof → run_test → compare → submit (gated) |
| §5 합격 기준 | G_repro: 노트북 자기-LB ±0.005 / G_oof: hit@1cm vs plan-a-003 KR008 paired-perm / G_lb: LB 제출 |
| §6 Out-of-scope | (예: 외부 데이터 추가 ✗ / 노트북 미공개 부분 추측 ✗) |
| §7 참조 | notes/winners/{handle}.{ipynb, meta.yaml} |

---

## §4. 재현 workflow (단계별)

### Stage 0 — 노트북 수령

1. DACON 코드공유 페이지에서 노트북 download → `notes/winners/{handle}.ipynb`.
2. `notes/winners/{handle}.meta.yaml` 작성 (§2 schema).
3. `analysis/plan-w-{NNN}/notebook_dump.md` 에 노트북 셀별 코드 요약 (재현 추적용).

### Stage 1 — plan 작성 (lane `w`)

1. 미사용 `w-{NNN}` 번호 발행 (`ls plans/plan-w-*` grep).
2. `plans/plan-template-winner.md` copy → `plans/plan-w-{NNN}-{handle}.md`.
3. §0~§7 채움. plan-review-master 자동 보정 (BLOCKER 0 수렴) 옵션.

### Stage 2 — 코드 이식

- 노트북 셀 → `analysis/plan-w-{NNN}/{module}.py` 분리 (carry 필요한 외부 의존만 frontmatter `code_reuse` 박제).
- 우리 `src/io.py` / data loader 인터페이스 위로 mount — 노트북 자체 데이터 로딩 코드는 분리 (path 차이만 흡수).

### Stage 3 — G_repro

- 노트북에 명시된 fold split (또는 그 노트북의 자체 OOF metric) 재현 → `runs/baseline/W{NNN}_{handle}/summary.json` 에 hit@1cm 박제.
- 노트북 자기-OOF ±0.005 안이면 PASS. 밖이면 random seed / fold split / data preprocessing 차이를 추적해 단일 변수 격리.

### Stage 4 — G_oof (우리 best 와 동일 split 비교)

- 우리 `plan-a-003` KR008 / `plan-c-001` FR001 등과 **stable fold split (`analysis/plan-a-001/run_oof.py` 의 split 함수)** 위에서 hit@1cm 비교 → paired-perm p.
- Δ + p 박제. 우리보다 위면 LB 제출 후보, 아래면 lesson 박제 후 종료.

### Stage 5 — G_lb (gated)

- DACON quota (5/day) 안에서 사용자 confirm 후 `dacon-submit` skill 로 제출.
- `registry.csv` 에 `W{NNN}_lb_submit` row 추가.

### Stage 6 — EXPERIMENTS.md 합류

- `EXPERIMENTS.md` 의 plan table 에 신규 row 추가 — paradigm, OOF, LB, band, lesson.
- "수상자라서" 강조 X. 우리 plan-{a~d} 와 같은 컬럼.

---

## §5. 자주 부딪힐 재현 마찰

| 마찰 | 우리 대처 |
|---|---|
| 노트북이 외부 사전학습 모델 download | 라이선스 확인 (대회 규칙 §6.1: MIT/Apache 2.0/CC BY 만 ✅). 라이선스 명시 없으면 메타데이터 `license: unknown` 박제하고 재현 보류. |
| 노트북이 임의 dropout / random init | seed 고정. ±0.005 안이면 PASS. |
| 노트북 fold split 비공개 | 노트북 자기-OOF 재현은 best-effort, 우리 split 위 OOF 가 정식 비교 metric. |
| 노트북이 우리에게 없는 dependency (예: torchcde, mmaction 등) | `notes/winners/{handle}.meta.yaml` 의 `dependencies:` 박제 + `pip install` 자율. |
| 노트북에 평가 코드만 있고 학습 코드 빠짐 | "재현 불가" 박제 + lesson "최종 가중치 reproducibility 부재" — `band: failed` 로 row 합류. |

---

## §6. 우리 plan 과의 차이 = lesson

각 winner plan 의 `*.results.md` §1 가설 판정 + §_lesson 에 다음을 박제 (EXPERIMENTS.md 에 단축 반영):

1. **paradigm 일치/차이** — 우리 어느 plan 과 같은/다른 paradigm 인가
2. **lever 단독 영향** — 노트북 핵심 lever 1~3개 단독으로 우리 best 에 더하면 ΔOOF
3. **재현 정확도** — 노트북 자기-LB ±몇 % 안에 들어왔나
4. **CV-LB 괴리 여부** — 우리 `plan-a-001` KR002 / `plan-a-002` KR003 사례와 동일한 OOF-neutral → LB-positive 패턴인지

---

## §7. 외부 데이터 / TTA / 앙상블

- 노트북이 외부 데이터 사용 시 대회 규칙 §6.2 (test 데이터 학습 사용 ✗) 만 위반 안 하면 재현 OK.
- 우리 plan 과의 **TTA 앙상블** 은 별도 plan-w-{NNN+1} 로 분리 (`plan_id`: `w-{NNN+1}`, `inspired_by: [w-{NNN}, a-003]`). 단일 plan 에 "재현 + 앙상블" 혼재 금지 (§9-3 자기-완결 위반).
