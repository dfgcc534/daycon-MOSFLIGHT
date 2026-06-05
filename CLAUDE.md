# CLAUDE.md

## 현 단계 (2026-06-05 갱신)

대회 종료 (2026-06-01). 본 디렉토리의 1순위 목표는 **수상자 노트북 재현 + 우리 plan 들과 동등 비교**.

- 신규 plan = lane `w` (winners) — `plans/plan-w-{NNN}-{handle}.md`. 골격 `plans/plan-template-winner.md`. protocol `WINNERS.md`.
- 기존 38 plans (legacy 001~032 + lane a/b/c/d) 는 **닫힌 path** — 새 lever 추가 X. 비교 baseline 으로만 carry.
- 모든 plan 동등 level. KR008 LB 0.6862 도 winner plan 도 같은 row (`EXPERIMENTS.md`).
- 닫힌 가설 재오픈은 winner 노트북의 새 evidence 가 있을 때만.

상세 문서:
- `WINNERS.md` — 수상자 재현 protocol (lane w, meta.yaml, G-gates)
- `EXPERIMENTS.md` — 38 plan 종합표 (paradigm timeline + row 단위 verdict)
- `WORKFLOW.md` — plan/results/registry 규약 본체 (§1~§12)

---

## Plan / 실행 규약 (요약)

자세한 건 `WORKFLOW.md`. 변경 없음. 핵심만:

- **Plan 명명**: `plan-{lane}-{NNN}-{slug}.md` (lane = mutex 단위, 1 worktree = 1 lane). legacy `plan-{NNN}-*` 그대로 유효.
- **Autonomous Execution**: `plan-{lane}-{NNN}` 은 G0 → G_final 자동 실행. 모호 시 권장 default + `decision-note:` 박제.
- **G_final → main 자율 merge**: `WORKFLOW.md §12.10` (sync-then-ff, worktree 안에서 직접). conflict 시에만 멈춤+escalate.
- **Commit/Push 의무**: 매 commit 직후 *즉시* push. batch 금지. `git log @{u}..` 가 매 turn 종료 직전 비어 있어야 함.

## Severe Issue (오직 멈춤)

`WORKFLOW.md §12.3` 의 9개 trigger 만 telegram alert + 멈춤. 그 외 = 자율 진행.

## 금지 어구

- "Should I proceed?" · "Is this OK?" · "Want me to continue?" · "Do you want me to..."

이 정책은 Claude 의 default conservative confirmation pattern 을 완전히 override 한다. 그냥 진행.

## 매 turn 시작 시 Read 시퀀스

1. `WORKFLOW.md §12` (Autonomous Execution Protocol)
2. 현재 plan (lane `w` 면 winner plan) 의 `§0.5 Quick Reference`
3. `git log -20 --oneline` 으로 현 commit 위치
4. §0.5 commit chain 의 다음 [TODO] commit → 그 commit 의 spec section 만 부분 read
