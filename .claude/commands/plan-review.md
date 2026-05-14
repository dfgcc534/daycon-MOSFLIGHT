Plan 파일 한 개를 두 축으로 점검하고 터미널에 마크다운 보고서만 출력한다 (파일 저장 없음).

두 축:
1. **코드 작성 가능성** — 이 plan 한 파일만으로 구현자가 추가 추론·검색 없이 식·시그너처·dtype·예외·단일 변수 경계까지 결정해 코드를 작성·실행 단계 직전까지 진입할 수 있는가.
2. **의도 분리/연속성** — 개발자의 의도가 plan 전체 흐름에 잘 드러나고 stage·phase 별로 분리되어 있는가. self-label 한 의도와 본문의 재사용 선언이 정합한가.

점검은 **plan 파일 한 개의 본문만** 본다. 외부 규약 문서·프로젝트 컨벤션·다른 파일은 일절 읽지 않는다.

## 인자 해석

- 인자 없음 → 현재 작업 디렉토리의 `plans/` 안에서 가장 최신 `plan-NNN-*.md` 중 같은 이름의 `.results.md` pair 가 **없는** 것을 자동 선택.
- 숫자 (예: `002`) → `plans/plan-{NNN-zero-pad-3}-*.md` 로 resolve. 매칭이 0개거나 2개 이상이면 사유 출력 후 종료.
- 그 외 문자열 → 임의 파일 경로로 간주. 파일 부재 시 사유 출력 후 종료.

## Steps

### 1. plan 로드
- 인자 해석으로 plan 절대경로 결정. 결정 실패 시 즉시 종료.
- `Read` 로 plan 본문 전체를 읽는다. 다른 파일은 읽지 말 것.

### 2. 섹션 버킷팅
- plan 본문에서 `^## ` 로 시작하는 헤딩을 모두 추출 → 각 섹션의 시작·끝 라인 산출.
- 섹션을 라인 합계 기준으로 **최대 4개 버킷에 균등 분배** (greedy: 가장 누적이 적은 버킷에 다음 섹션 할당). 큰 섹션 하나가 1 버킷을 단독 점유해도 무방. 헤딩이 4개 미만이면 그 수만큼 버킷 사용.
- 각 버킷의 섹션 라벨과 line range 를 메모해 둔다 (예: 버킷 1 = "§1~§3, L29-L140").

### 3. 5 sub-agent 병렬 호출
한 메시지 안에서 다음을 모두 spawn (foreground, 결과를 기다린다).

#### 3a. 코드 작성 가능성 축 — 버킷 수만큼 (최대 4개)
각 버킷마다 `Agent(subagent_type="general-purpose")` 호출. prompt 템플릿:

```
당신의 임무: 다음 plan 파일의 지정된 섹션만 보고 "구현자가 추가 추론·검색 없이
바로 코드를 작성·실행 직전까지 진입할 수 있는가" 점검.

plan 절대경로: {ABS_PATH}
담당 섹션: {SECTION_LABELS}, line range {LINE_RANGE}

규약:
- 위 plan 파일만 Read 한다. 다른 어떤 파일도 읽지 말 것.
- 담당 섹션 외의 라인은 cross-reference 용으로만 가볍게 참조 가능. 점검 대상은 담당 섹션.

위반 후보 (모두 BLOCKER/AMBIGUITY/MINOR 중 하나로 분류):
- 식·계산 정의 부재 (평균식, 경계조건 i=0, ZeroDivision/0-width 처리 등)
- 시그너처·반환 타입·dtype·예외 클래스 미명시
- timezone, inclusive/exclusive, list vs Enum 등 데이터 경계 모호
- 외부 저장소·외부 노트·이전 채팅 의존 (commit hash 미지정 포함)
- 한 실험·단계의 변경 변수가 단일하지 않거나 단일성이 implicit
- public API 와 내부 export 이름의 충돌, 누락된 re-export
- 자동 통과/실패 분기가 정성 표현으로 결정 불가능

분류 등급:
- BLOCKER : 코드 작성 자체가 불가능한 결함
- AMBIGUITY : 구현자가 해석 분기를 만나는 모호함
- MINOR : 개선 권고

보고 형식:
- 등급별로 묶어 한국어 bullet list. 각 finding 에 plan 내 line number 인용 필수.
- 600~700 단어 이내. 추측 금지 — 본문에 적힌 내용만으로 판정.
```

#### 3b. 의도 분리 축 — 1 agent
동일 메시지에서 병렬 spawn. `Agent(subagent_type="general-purpose")`. prompt:

```
당신의 임무: 다음 plan 파일 전체를 읽고 두 verdict 를 산출.

plan 절대경로: {ABS_PATH}

규약: 위 파일만 Read. 다른 파일 금지.

(A) 전체 narrative 연결성
  - 배경 → 가설 → 실험 → 실행의 인과 chain 이 끊김 없이 흐르는가?
  - "왜 이 가설들인가"·"왜 이 실험이 그 가설을 검증하는가" 매핑이 본문에 명시적인가?

(B) stage / 묶음 분리
  - 한 실험이 정확히 한 가설만 담는가?
  - 묶음(stage·phase) 간 의존성·skip 룰이 일관되게 적용되었는가?
  - 한 실험에서 여러 변수를 동시 변경해 의도 경계가 흐려지지 않는가?

(C) self-label 의도 vs 재사용 선언의 정합성 (★ 코드 재사용에 의한 의도 이탈 감지)
  - plan 의 scope / §0 한 줄 목적의 self-label (paradigm shift / new module
    / refactor / migration / minimal patch 등) 이 본문 "재사용 / 비재사용"
    명세와 일치하는가?
  - 핵심 module 을 *원래 다른 task 용* 으로 설계된 코드에서 reuse 한다면,
    그 사실과 task fit 재검토가 본문에 명시되어 있는가?
  - reuse 비중이 self-label 의 ambition 을 무력화하지 않는가? (예: "paradigm
    shift" 인데 핵심 encoder/baseline 대부분 reuse → 실질은 minimal patch
    — self-label 강등 필요)
  - frozen / pretrained / numpy 함수 재사용처가 plan 가설의 gradient 경로를
    끊지 않는가? (예: F0 를 numpy 로 호출해 gradient X → "F0 학습 가능 가설"
    self-label 과 모순)

각 verdict: "잘 드러남 / 부분적으로 흐림 / 흐림" 3단계 중 하나.
근거 line 인용 3개 이상 필수. 흐림 지점은 구체 라인 지목.

700 단어 이내, 한국어.
```

### 4. 합산 보고
5 agent 결과 수신 후, 메인 컨텍스트가 직접 다음 구조로 터미널에 마크다운 출력.

```markdown
## 결론 (한 문장)
{plan 식별자}: BLOCKER {N}건, AMBIGUITY {N}건+, MINOR {N}건. (A) {verdict} / (B) {verdict} / (C) {verdict}.

## (1) 코드 작성 가능성 — 버킷별 BLOCKER
### 버킷 1 ({라벨})
- **{항목명}** [{plan-basename}:{LINE}]({relative/path}#L{LINE}) — 한 줄 사유
### 버킷 2 ({라벨}) …
### 버킷 3 ({라벨}) …
### 버킷 4 ({라벨}) …

(BLOCKER 0건인 버킷은 "BLOCKER 없음" 한 줄. AMBIGUITY/MINOR 는 별도 섹션
없이 권장 수정 순서에서 선별 인용.)

## (2) 의도 분리/연속성
### (A) narrative — {verdict}
- 근거 line 인용 (3개 이상)
### (B) stage 분리 — {verdict}
- 흐림 지점 구체 지목 (line 인용)
### (C) self-label vs reuse 정합성 — {verdict}
- self-label 위치 (scope / §0) 와 reuse 명세 위치 양쪽 line 인용
- 흐림 시 권장 조치 한 줄 (self-label 강등 / reuse 명세 보강 / task fit 재검토)

## 권장 수정 순서
1. 가장 critical 한 BLOCKER 부터 N개 (각 항목 한 줄)
```

### 출력 형식 규약
- 모든 line 인용은 `[plan-basename:LINE](relative/path#LLINE)` 마크다운 링크 (VSCode 확장 환경에서 클릭 가능).
- relative path 는 cwd 기준. 절대경로 금지.
- "결론" 한 문장이 가장 위 — 사용자가 한눈에 critical 여부 판단.
- 본 보고서 외에 어떤 파일도 쓰지 않는다.

### 5. 종료
보고 출력 후 종료. plan 파일도 results 파일도 수정하지 않는다.
