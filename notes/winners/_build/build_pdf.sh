#!/usr/bin/env bash
# winner-gap-analysis.md → winner-gap-analysis.pdf (한글 PDF, tectonic+xeCJK)
# 의존: pandoc, tectonic, 시스템 폰트 AppleSDGothicNeo.ttc
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(cd "$HERE/.." && pwd)"
SRC="${1:-$DIR/winner-gap-analysis.md}"
OUT="${2:-$DIR/winner-gap-analysis.pdf}"
FONT="AppleSDGothicNeo.ttc"
FONTDIR="/System/Library/Fonts/"

pandoc "$SRC" -o "$OUT" --pdf-engine=tectonic \
  -H "$HERE/header.tex" \
  --toc --toc-depth=2 \
  -V documentclass=article \
  -V geometry:margin=2.2cm \
  -V fontsize=11pt \
  -V linkcolor=blue -V urlcolor=blue \
  -V CJKmainfont="$FONT"  -V CJKoptions="Path=$FONTDIR" \
  -V mainfont="$FONT"     -V mainfontoptions="Path=$FONTDIR" \
  -V monofont="Menlo.ttc" -V monofontoptions="Path=$FONTDIR"

echo "built: $OUT ($(stat -f%z "$OUT") bytes)"
