# CLAUDE.md

## Autonomous Execution Policy

이 프로젝트의 모든 `plans/plan-NNN-*.md` 는 **완전 자동 실행** 권한 부여:

- plan 전체를 G_final 까지 *자동 끝까지* 진행. 중간 사용자 permission 불필요.
- 디테일 결정 모호 시 → *권장 default* 채택 후 진행 (ask 금지).
- Plan 의도 = `§0 한 줄 목적` + `§0.5 Quick Reference` + `합격 기준`. 의도 위배 아닌 모든 결정은 자율.
- 자율 결정 시 commit msg 마지막에 `decision-note:` 1줄 박제 (사후 audit 용 — `WORKFLOW.md §12.9`).

## ⚠️ Commit · Push 정책 (MANDATORY)

**commit 은 의미 단위 (§11) — 현행 cadence 그대로 유지** (turn 마다 commit O). **push 는 plan 종료 (G_final) 시 *1회 일괄***. plan 진행 중에는 commit 만 local 에 쌓아두고 **turn 마다 push 하지 않는다**.

- commit 단위 = plan 1 / results 1 / code change 1 분리. binary 혼재 금지. `[TODO] → [DONE]` §0.5 sync commit 도 별도 commit (push 없이 누적).
- **push 시점은 다음 3개뿐**:
  1. **G_final 정상 종료** — 그때까지 누적 commit 전체 일괄 push.
  2. **severe halt** (`WORKFLOW.md §12.3`) — 작업 보존 위해 그때까지 commit 일괄 push **후** 멈춤.
  3. **사용자 명시 요청** (`push it`, `밀어` 등) — 즉시 push.
- push 실패 (rebase conflict / network): `git_rebase_conflict` 또는 `network` 사유 박제 + 1 회 retry, 그래도 실패 시 사용자 escalate.

**checkpoint**: plan 진행 중 `git log @{u}..` ahead > 0 은 **정상** (의도된 누적). G_final push **후에만** ahead 0.

## ❌ Push 관련 금지 어구 (新 정책 enforcement)

"commit 마다 push" · "turn 마다 즉시 push" · "2 개 쌓이면 push" (← G_final·severe·사용자 요청 *외* push 금지). 역으로 **G_final 도달 후 일괄 push 누락** 도 위반.

## 매 turn 시작 시 Read 시퀀스

(`WORKFLOW.md §12.2 step 1~6` 의 압축형 — plan 본문 전체 re-read 금지, token 낭비)

1. `WORKFLOW.md §12` (Autonomous Execution Protocol)
2. 현재 `plans/plan-NNN-*.md` 의 `§0.5 Quick Reference`
3. `git log -20 --oneline` 으로 현 commit 위치 파악
4. §0.5 commit chain 의 다음 [TODO] commit 식별 → 그 commit 의 spec section 만 offset/limit 부분 read

## Severe Issue

`WORKFLOW.md §12.3` 의 **9개 trigger** 발생 시에만 telegram alert + 멈춤. 그 외 모든 결정 = 자율 진행.

## 금지 어구

- "Should I proceed?" · "Is this OK?" · "Want me to continue?" · "Do you want me to..."

이 정책은 Claude 의 default conservative confirmation pattern 을 **완전히 override** 한다. 그냥 진행.

## 기존 정책 (불변)

- `WORKFLOW.md §1~§11` 의 plan/results/registry 규약 그대로 유효.
- §11 의 *server agent 의미 단위 commit* 은 자율 권한이 아니라 **의무** (push 는 G_final 일괄 — 위 §⚠️).
- 본 §Autonomous Execution Policy 는 위 정책의 *실행 확장* 이지 대체 아님.
