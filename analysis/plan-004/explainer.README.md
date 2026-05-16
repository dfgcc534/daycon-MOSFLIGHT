# plan-004 입문자용 해설 — PDF 변환 가이드

이 디렉터리에는 plan-004 (PB_0.6822 fullrun, LB 0.6806) 의 입문자용 해설 문서가 있습니다.

## 파일 구성

```
analysis/plan-004/
├── explainer.md             ← 메인 본문 (Markdown, 한국어 + 영어 용어 병기)
├── explainer.README.md      ← 본 파일 (PDF 변환 가이드)
└── figures/
    ├── build_figures.py     ← 그림 생성 스크립트 (matplotlib + numpy + csv)
    ├── fig01_task_timeline.png
    ├── fig02_sample_trajectories.png
    ├── fig03_architecture.png
    ├── fig04_27_candidates.png
    ├── fig05_regime_heatmap.png
    ├── fig06_training_pipeline.png
    ├── fig07_metric_vs_l2.png
    └── fig08_corrector_lift.png
```

## 그림 재생성

```bash
python3 analysis/plan-004/figures/build_figures.py
```

- 의존성: `matplotlib`, `numpy` (stdlib `csv`, `json` 만 추가). pandas / torch 불필요.
- `data/train/` 와 `data/train_labels.csv` 가 필요 (fig02, fig04 의 실제 trajectory 시각화용).
- 출력: 8 PNG, ~150 DPI, total ~760 KB.

## PDF 변환 3가지 옵션

### Option A. pandoc + wkhtmltopdf (가장 추천 — 한국어 폰트 호환 좋음)

```bash
# macOS
brew install pandoc wkhtmltopdf

# Markdown → PDF
cd analysis/plan-004
pandoc explainer.md \
  --pdf-engine=wkhtmltopdf \
  --pdf-engine-opt=--enable-local-file-access \
  -V mainfont="Apple SD Gothic Neo" \
  -V geometry:margin=2cm \
  -o explainer.pdf
```

특징: HTML 경유 → 한국어 폰트 자동 fallback. 수식은 MathJax 렌더링 (인터넷 필요 시) 또는 raw LaTeX.

### Option B. pandoc + tectonic (LaTeX 엔진, 최고 품질)

```bash
brew install pandoc tectonic

cd analysis/plan-004
pandoc explainer.md \
  --pdf-engine=tectonic \
  -V CJKmainfont="Apple SD Gothic Neo" \
  -V geometry:margin=2cm \
  --highlight-style=tango \
  -o explainer.pdf
```

특징: 수식 품질 최상 (LaTeX 네이티브). 한국어 폰트 명시 필요. 설치량 ~150 MB.

### Option C. VS Code "Markdown PDF" 확장 (GUI, 가장 간단)

1. VS Code 에서 `Markdown PDF` 확장 설치 (저자: yzane)
2. `analysis/plan-004/explainer.md` 열기
3. 우클릭 → "Markdown PDF: Export (pdf)"

특징: 설치 없음 (확장 1개만). 한국어 자동 처리. 단, 페이지 break 조절은 본문에 `<div style="page-break-after: always"></div>` 직접 삽입 필요.

### Option D. marp-cli (슬라이드 스타일, 발표용)

발표용 슬라이드 PDF 가 필요하면:

```bash
npm install -g @marp-team/marp-cli
marp explainer.md --pdf --allow-local-files -o explainer-slides.pdf
```

(단, 본 explainer.md 는 *문서* 포맷 — 슬라이드용으로는 `---` 페이지 break 추가가 필요할 수 있음)

## 한국어 폰트 trouble-shooting

PDF 가 한국어 글자가 □ 로 깨질 경우:

1. **macOS**: `Apple SD Gothic Neo` 또는 `AppleGothic` 사용 (시스템 기본).
2. **Linux**: `sudo apt install fonts-nanum` 후 `NanumGothic` 지정.
3. **Windows**: `Malgun Gothic` 또는 `맑은 고딕`.

pandoc 명령에서 `-V mainfont` 또는 `-V CJKmainfont` 옵션으로 지정.

## 분량 가이드

- pandoc + A4 + margin 2cm 기준: 약 **12 ~ 14 페이지**
- 본문 그림 8개 (각 0.5 ~ 1 페이지 차지)
- 텍스트 본문 약 6000 단어

## 라이선스 / 공유

- 본 문서는 dacon-MOSFLIGHT 프로젝트 내부 협업자/외부 공유용 자료입니다.
- 노트북 작성자 (`notes/PB_0.6822 코드공유.ipynb`) 의 *Physics Ladder* 개념과 narrative 를 인용·확장.
- 외부 공유 시 LB 점수 0.6806 와 dacon 대회 ID 236716 명시 권장.
