# notes/winners/

수상자 노트북 + 메타데이터.

각 노트북은 짝 `{handle}.ipynb` + `{handle}.meta.yaml` 로 둔다 (메타데이터 스키마 `WINNERS.md §2`).

대응 재현 plan = `plans/plan-w-{NNN}-{handle}.md` (`plans/plan-template-winner.md` 골격).

각 노트북은 DACON 렌더 HTML `md_file.html` + `meta.yaml` (`WINNERS.md §2`) + `notebook_dump.md` (셀별 추출) 로 둔다.

## 현재 보유 — Private ≥ 0.70 우승자 (comp 236716 codeshare, 2026-06-06 수집)

| folder | handle | rank/score | cs_id | paradigm |
|---|---|---|---|---|
| `rank1-euijin42/` | euijin42 | 1위 · Private 0.7035 | 14013 | GRU + Neural ODE + HyperPhysics 30모델 등가중 블렌드 |
| `rank2-munjwc25/` | munjwc25 | 2위 · Private 0.7031 | 14007 | 직교 multi-pool 앙상블 (Kalman-GRU + ODE + physics + tree) |
| `rank3-wlstn52/` | wlstn52 | 3위 · Private 0.7019 | 14009 | HyperPhysics + Kalman-GRU 2모델 OOF grid-search 가중평균 |
| `pb07011-14011/` | 남뻐글이 | Private 0.7011 | 14011 | physics-core + GRU 잔차 (3seed×5fold=15 K-fold) |
| `pb0700-14012/` | 노우딘 | Private 0.700 | 14012 | GRU+Transformer 물리 디코더 + LGBM 잔차 후보정 (10-fold) |

## 참고 (< 0.70 또는 내 plan 의 원본 — full 추출 X, 학습자료에서만 인용)

| cs_id | handle | 제목/score | 비고 |
|---|---|---|---|
| 14010 | CREE | [Private 9등] HyperPhysics_Calib_xy2 (~0.693) | GOH30·rank3 가 쓴 HyperPhysics 의 계보(ancestor) |
| 14002 | CREE | [LB 0.6+] Neural ODE 기반 | 내 plan-d-001 (NODE001) 원본 |
| 13997 | Trojan_Horse | [LB 0.6780] Kalman 잔차 + GRU + Calibration | 내 plan-a-001 (KR001) 원본 |
| 13980 | 대회못해요이제.. | PB 0.6822 | 내 plan-004 framework 원본 |
| 14008 | 곰고래 | PB 0.6996 | <0.70 |
| 14014 | 정조로408 | [Private 12등] HR-Aware Rotation Gate (~0.693) | <0.70 |
| 14015 | 캐빈 | Public 0.6872 | <0.70 |

## 산출물

- `winner-gap-analysis.md` / `.pdf` — 내 최고(LB 0.6862, KR008) vs 0.70+ 우승자 격차 학습 자료.

## 추가 절차 (신규 우승자)

1. codeshare 글 → `md_file.html` (presigned `code_url`, 60초 만료 → 즉시 fetch) 배치
2. `meta.yaml` 작성 (`WINNERS.md §2`)
3. `notebook_dump.md` 셀별 추출 (Phase B)
