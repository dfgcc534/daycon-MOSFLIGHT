---
plan_id: 011
followed_by: 012 (TBD)
date: 2026-05-13 (Asia/Seoul)
---

# plan-012 후보 (≥ 3) — plan-011 의 carry-over

plan-011 Phase 1 결과 (G1 (b) FAIL — 0/4 axes strict +0.005) 기반 paradigm-shift 후보:

## 후보 1: CNN/Transformer Encoder 강화 (★ 가장 신호 강함)

**Motivation**:
- plan-011 In axis ID (1D CNN 3-layer, 64-dim) 가 4 axis 중 *유일* positive 방향 (+0.00495).
- §1.5 input snapshot 한계 hypothesis 부분 검증.
- ID 의 marginal 신호를 *확장* 하여 +0.005 strict threshold 돌파 가능성.

**구체적 spec**:
- Deeper CNN: 5-6 layer, hidden 128-256, dilated conv (receptive field 확장).
- Self-attention over trajectory: 11-step trajectory 의 step-pair attention.
- Multi-scale CNN: kernel=[3, 5, 7] 병렬 + concat.

**Risk**:
- over-fit on small data (10K samples) — caveat #21 적용.
- depth 증가 시 plan-011 M6 (WiderShallow 306K) 의 -0.0079 와 동일 risk.

**Expected gain**: +0.005~0.015 vs L0 anchor (OOF 0.6595~0.6695).

## 후보 2: F3/F4 Formula Parity Fix 후 재실행 (★ plan-011.1 carry-over 1순위)

**Motivation**:
- plan-011 F axis 의 F3 (per-sample MLP) + F4 (LearnableSingleCandidate) 가 fold-0 OOF 0.0980/0.0322 — *catastrophic*.
- Cause: §8.1 cand formula (`p0 + 1.94*v_last*horizon^1 + 1.20*t_hat*(a·t̂) + ...`) 가 selector.make_candidates 의 정식 frenet 계산과 numerical 불일치.
- plan-007 Step 4 의 LB 미회수 가 *살아있는 카드* (§1.1) — formula axis 의 진정한 측정 미실현.

**구체적 spec**:
- F3 의 cand 생성을 `selector.make_candidates` 직접 호출로 wrap (per-sample par/perp override).
- F4 의 cand 생성을 동일 방식 (learnable 6 coef → selector spec 의 (d1, par, perp, ...) 으로 변환).
- 학습 loss = hit-aware (huber + smooth hinge) on `selector.make_candidates(...)` output.

**Risk**:
- formula axis 자체가 *flat ceiling* (plan-007 측정 시 모두 0.640~0.649 범위) — 큰 signal 못 잡을 가능성.

**Expected gain**: +0.002~0.010 vs F0 anchor (OOF 0.6421~0.6501).

## 후보 3: KNN / Diffusion / Hybrid Paradigm 교체

**Motivation**:
- plan-011 4 axis 어느 곳도 strict +0.005 통과 못 함 → corrector path 의 *구조적 제한*.
- candidates 27-pool selector (plan-004) 또는 단일공식 (plan-006) 모두 *Frenet 모델 family* 의 ceiling 부근.
- paradigm shift 필요: nearest-neighbor retrieval (training data 의 *유사 trajectory* 의 final position) 또는 diffusion-based prediction.

**구체적 spec (3 sub-paradigm)**:
- 3A. KNN: trajectory embedding (CNN) → cosine similarity → top-k (k=5~10) training samples 의 final position 평균.
- 3B. Diffusion: noise-to-position diffusion model (DDPM) — trajectory 를 condition 으로 final_pos 생성.
- 3C. Hybrid: Frenet candidate + KNN refinement (Frenet 으로 coarse, KNN 으로 fine-tune).

**Risk**:
- paradigm shift 의 implementation cost ↑↑.
- 처음 시도하는 paradigm 이라 baseline 박제 필요 (plan-012 c1 ~ c5).
- LB 제출 quota 소진 상태 (plan-011.1 carry-over) — paradigm shift 의 *test* 가 plan-012.1 또는 그 이후.

**Expected gain**: +0.01~0.05 if paradigm valid (OOF 0.66~0.70).

## 후보 우선순위 (자율 결정)

1. **plan-011.1 carry-over 1순위 = 후보 2** (F3/F4 fix) — implementation 비용 작음, 의미 있는 측정.
2. **plan-012 main path = 후보 1** (CNN/transformer encoder) — 가장 신호 강한 axis 의 확장.
3. **plan-012 backup path = 후보 3** (paradigm shift) — main path 도 failure 시 진입 조건.

진행 plan-012 v1 작성 시 위 우선순위로 §0.5 commit chain 설계.

## 추가 plan-011.1 carry-over (실험적 quick fixes)

- **IC frozen GRU checkpoint 박제**: plan-004 selector.AttnGRUCandidateSelector 의 state_dict load 자동화 (현재는 stub) — 그 후 In̂ vs IC 비교.
- **L axis re-run on In̂=ID anchor**: caveat #17 cross-axis bleed 검증 — In axis fixed (cf+CNN) 위 L sub-exp 재측정.
- **5-fold OOF reproduce for ID**: 본 plan 의 fold-0 1-fold approx 의 *5-fold concat* 측정 (binomial std ↓ 0.005 → 0.002).
- **submission.csv LB 회수**: `dacon-submit` 으로 ID submission LB 실측 (할당량 회복 후).
