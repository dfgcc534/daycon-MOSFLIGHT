---
plan_id: 029
version: 1.0
date: 2026-05-22 (Asia/Seoul)
status: written
best_cell: null
best_hit_1cm: null
best_hit_1p5cm: null
based_on:
  - 022 (winner A6_bcc14_tau001 OOF 0.6531 / 0.8108. K=14 BCC + τ=0.001. selector-only LGBM 170D baseline)
  - 024 (cross-attention GRU selector — paradigm 자체 미검증 fail. 본 plan = 동일 paradigm 위 hyperparameter 재설계 + plan-025 1080D input 정상 검증)
  - 025 (LGBM + 후보 concat + seq 압축 → mode collapse → F0 복사. paradigm mismatch evidence. 본 plan input = plan-025 의 1080D 그대로)
  - 020 (F0 baseline 0.6320/0.8033 + stable_fold_id MD5)
inspired_by:
  - 사용자 (2026-05-22): "plan-026, 027 은 gru-attention" — abandoned LGBM 026/027 의 통합 재발행. paradigm-level 검증 1회 plan.
  - plan-025 paradigm mismatch finding: block ③ 22D per-anchor 가 LGBM 에서 self-prediction trigger 였지만 GRU-attention 에서는 query 의 anchor identity 로 정상 작동 예상.
code_reuse:
  - module: analysis/plan-024/model.py (worktree-plan-024-combo branch 의 c2 cherry-pick 필요)
    symbols: [CandidateAttentionGRUSelectorCarry, CrossAttentionAnchorSelector]
    reason: 본 plan 의 backbone. hidden=384 → 196 변경, anchor_embed_dim=0 default 유지. training schedule 만 재설계 (epoch / lr / dropout / patience).
  - module: analysis/plan-024/cand_builder.py
    symbols: [build]
    reason: cand_feat 150D 산출. plan-025 build_feat_1080 의 source.
  - module: analysis/plan-024/seq_builder.py
    symbols: [build]
    reason: seq 95D × 7 step. GRU encoder input source (= plan-025 block ④ 의 raw 형태, 8-stat 압축 전).
  - module: analysis/plan-024/anchor_vocab.py
    symbols: [build]
    reason: seq_builder internal.
  - module: analysis/plan-024/torsion_calc.py
    symbols: [build]
    reason: seq_builder internal.
  - module: analysis/plan-024/quantile_carry.py
    symbols: [build, QuantileCarry]
    reason: fold-leakage 차단 quantile carry (omega_p90, jerk_p90).
  - module: analysis/plan-024/multiwindow_trim_build.py
    symbols: [load_trim]
    reason: 144→60 trim index.
  - module: analysis/plan-024/feature_weighted_dropout.py (worktree-plan-024-combo cherry-pick 필요)
    symbols: [FeatureWeightedDropout]
    reason: plan-024 의 input dropout lever. 본 plan training schedule 재설계의 일부.
  - module: analysis/plan-025/build_feat_1080.py
    symbols: [build_feat_1080, BLOCK_DIMS]
    reason: 1080D input builder. 본 plan baseline + head skip 의 source.
  - module: analysis/plan-022/anchors.py
    symbols: [ANCHORS_A6]
    reason: K=14 BCC anchor codebook (plan-022 winner carry).
  - module: analysis/plan-022/selector_only_model.py
    symbols: [build_soft_label_with_tau]
    reason: soft label 산식.
  - module: analysis/plan-021/build_input.py
    symbols: [build_frenet_basis_3d, build_input_common, build_input_lgbm_extra]
    reason: 170D plan-022 carry input pipeline (block ①).
  - module: analysis/plan-020/baseline_f0.py
    symbols: [f0_baseline, D1, R_HIT, R_HIT_LOOSE]
    reason: F0 baseline + hit metric.
  - module: src/io.py
    symbols: [load_all_samples, load_labels]
    reason: data loader.
  - module: src/pb_0_6822/selector.py
    symbols: [stable_fold_id, fit_regime_bins, assign_regimes]
    reason: 5-fold split + regime assignment.
supersedes_abandoned:
  - 026 (LGBM block ablation, user intent mismatch)
  - 027 (LGBM 3-way ensemble, user intent mismatch)
followed_by:
  - plan-030 (가칭, GRU-attention 결과 후속 — F0 ML 또는 corrector 부활)
scope: plan-024 cross-attention GRU selector paradigm 위 **hyperparameter 재설계** + **plan-025 1080D input 정상 검증**. hidden=196 (plan-024 384 의 0.51×), anchor_embed_dim=0 default, GRU encoder input = raw seq (B, 7, 95), cross-attention query = cand_feat (B, 14, 150), head skip = plan-025 block ①+④ (170+760=930D). training schedule = 50 epoch fixed (early stop disabled), lr=7e-4 cosine + warmup 5 epoch, AdamW (wd=1e-4), dropout=0.10, gradient_clip=1.0, batch=64, KL divergence soft label loss. K=14 BCC + τ=0.001 fix (plan-022 carry). 5-fold stable_fold_id. ensemble / DACON LB / corrector / F0 ML = out-of-scope.
exp_ids:
  - Z029_X1_gru_h196
lb_score: null
band: null
---

# plan-029 v1 — GRU-attention Input Max (hidden=196, 1080D + raw seq)

## §0. 한 줄 목적

> **plan-024 cross-attention paradigm 의 hyperparameter 재설계** + **plan-025 1080D input 정상 검증**. plan-025 LGBM 의 mode collapse (paradigm mismatch evidence) 가 GRU-attention 위에서는 *paradigm 자연스러운 작동* 으로 회복되는지 검증. **abandoned plan-026 + plan-027 의 통합 재발행** (사용자 plan-026/027 GRU-attention 의도 합의).
>
> **paradigm rationale**:
> 1. plan-024 fail (OOF 0.6370 vs plan-022 0.6528) = CPU under-converged 의심 + 다중 lever 동시 추가 confound. training schedule (epoch / patience / lr) 재설계로 paradigm 자체 정상 검증.
> 2. plan-025 (1080D LGBM) fail = GRU-attention 용 input 을 LGBM 에 잘못 사용. block ③ 22D per-anchor 가 LGBM 에서 self-prediction trigger. GRU-attention 에서는 query 의 anchor identity 로 정상 작동 예상.
> 3. plan-024 attention paradigm 의 implicit assumption (anchor identity at query, sequence at key, sample-level ctx broadcast) 이 input 1080D 안에 정확히 박혀 있음.
>
> **단일 cell (paradigm-level 검증 1회 plan)**:
> - **X1** = GRU(hidden=196) + cross-attention + head_mlp(skip=plan-025 block ①+④)
> - training schedule: epoch=50 fixed (no early stop), lr=7e-4 cosine, AdamW (wd=1e-4), dropout=0.10, gradient_clip=1.0, batch=64
>
> **pass criterion (G3)**:
> - **PASS**: hit_1cm > 0.6528 (= plan-022 winner) → paradigm 정상 검증, GRU-attention 의 mode collapse 미발생
> - **partial_drift**: 0.6320 ≤ hit_1cm ≤ 0.6528 → paradigm 작동 but plan-022 baseline 미회복
> - **regression**: hit_1cm < 0.6320 → paradigm mismatch 본질
>
> **out-of-scope**: ensemble (plan-030 후보) / DACON LB submit / boundary corrector / F0 ML / anchor layout 변경 / τ_cls 변경 / hidden ≠ 196 sweep / batch ≠ 64.

---

## §0.5 Quick Reference (autonomous loop 매 turn 읽는 section)

### 합격 기준 (G-gate sequence)

- **G0**: 5 module (model / run_oof / train / tests + plan-024 cherry-pick 9 file) import + smoke + tests green. plan-022 / plan-024 / plan-025 carry import 정상. 위반 시 `infra_drift` severe.
- **G1**: F0 baseline + plan-022 winner reproduce (plan-025 baseline_carry.json carry 또는 재산출). hit_1cm F0 ∈ [0.6315, 0.6325] AND plan-022 ∈ [0.6523, 0.6533]. 위반 시 `f0_reproduce_drift` / `plan022_reproduce_drift` severe.
- **G2.X1**: X1 cell 5-fold OOF metric finite + `max_class_ratio < 0.95`. mode collapse 표시 = `max_class_ratio ∈ [0.05, 0.1]` (near 1/K=0.071) — paradigm mismatch evidence. 위반 시 `numerical` severe / `mode_collapse` warn.
- **G3 (paradigm)**: PASS / partial_drift / regression 판정 (§0 criterion).
- **G_final**: results.md + 3-file frontmatter sync + follow-up plan-030 (가칭) 박제.

### G-gates

- G0: STAGE 0 인프라 + plan-024 cherry-pick (model.py + feature_weighted_dropout.py) [TODO]
- G1: STAGE 1 F0 + plan-022 winner reproduce [TODO]
- G2.X1: X1 5-fold OOF [TODO]
- G3: STAGE 3 paradigm 판정 [TODO]
- G_final: STAGE 4 results + 3-file sync [TODO]

### Commit chain (next-up)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-029-grunet-input-max.md` v1 작성 | [TODO] |
| c2 | chore | plan-024 추가 cherry-pick from `worktree-plan-024-combo` (commit 915dd26): `model.py` + `feature_weighted_dropout.py`. 기존 plan-025 cherry-pick (anchor_vocab/cand_builder/seq_builder/torsion_calc/quantile_carry/multiwindow_trim_build + json + __init__) 외 추가 2 file. | [TODO] |
| c3 | code | `analysis/plan-029/model.py` — plan-024 backbone wrapper (hidden=196, anchor_embed_dim=0, dropout=0.10). | [TODO] |
| c4 | code | `analysis/plan-029/train.py` — PyTorch 5-fold OOF training loop (epoch=50 fixed, lr=7e-4 cosine + warmup 5, AdamW wd=1e-4, gradient_clip=1.0, batch=64, KL loss). | [TODO] |
| c5 | code | `analysis/plan-029/run_oof.py` — orchestrator + G1 reproduce + 5-fold concat OOF + final metric. CLI `--cell X1` 또는 `--g1`. | [TODO] |
| c6 | test | `tests/test_plan029_smoke.py` — 8+ pytest (import / model forward shape / training step / soft label sum=1 / GRU input shape / cross-attention shape / Frenet→world 식 / plan-025 build_feat_1080 carry). | [TODO] |
| G0 | gate | smoke + tests green (예상 < 300s) | [TODO] |
| c7 | exp G1 | F0 + plan-022 winner reproduce (plan-025 baseline_carry.json 재사용 또는 재산출) → `baseline_carry.json` | [TODO] |
| G1 | gate | F0 hit ∈ tight band ✓ AND plan-022 winner hit ∈ tight band ✓ | [TODO] |
| c8 | exp G2.X1 | X1 5-fold OOF GRU-attention 학습. 예상 runtime: CPU 3-6h (50 epoch × 5 fold × N=8000 train, hidden=196 plan-024 384 의 27% 계산량). `results_X1.json` + `train_X1.log` 박제. | [TODO] |
| G2.X1 | gate | metric finite + max_class_ratio < 0.95. 또한 epoch 50 fully trained 검증 (early stop disabled) | [TODO] |
| c9 | analysis | X1 결과 + paired Δ vs F0 + paired Δ vs plan-022 winner + 14-anchor oracle 회수율 + mode collapse 진단 → `paradigm_analysis.{json,md}` | [TODO] |
| G3 | gate | paradigm 판정 (PASS / partial_drift / regression) | [TODO] |
| c10 | docs | 3-file frontmatter sync + `analysis/plan-029/results.md` + `plans/plan-029-*.results.md` pair + follow-up plan-030 (가칭) 박제 | [TODO] |
| G_final | gate | 3-file sync + §0.5 c1~c10 [DONE] | [TODO] |

### Plan-specific severe

- `infra_drift`: plan-024 cherry-pick 또는 plan-025 carry module import 실패.
- `f0_reproduce_drift` / `plan022_reproduce_drift`: G1 reproduce tight band 위반.
- `numerical`: PyTorch forward / backward NaN/Inf.
- `mode_collapse` (warn): max_class_ratio ∈ [0.05, 0.08] (near 1/K=0.071). paradigm mismatch finding 으로 박제, G2 계속 진행.
- `model_capacity_overflow`: GPU/CPU OOM 또는 학습 시간 > 12h.
- `plan024_cherry_pick_missing`: c2 cherry-pick 후 model.py / feature_weighted_dropout.py importlib 실패 → halt.

### Plan-specific paths

- whitelist:
  - `analysis/plan-029/**`
  - `tests/test_plan029_smoke.py`
  - `analysis/plan-024/{model.py, feature_weighted_dropout.py}` — **c2 cherry-pick 단계 유일 plan-024 path 수정 허용** (add only)
- blacklist: `analysis/plan-{001..028}/**` (read-only import 예외)

### Decision-note 사용 예

- `decision-note: spec-default — GRU encoder input = raw seq (B, T=7, C=95) from seq_builder.build(). plan-025 block ④ 760D (8-stat 압축) 는 head skip 으로 사용. raw seq 가 GRU 학습 source.`
- `decision-note: spec-default — cross-attention query = cand_feat (B, K=14, 150) from cand_builder.build(). plan-025 block ②③ 의 source. query MLP 입력.`
- `decision-note: spec-default — head skip = concat(h_final_bc 384, event_ctx 196, cand_feat 150, block① 170, block④ stat 760) → MLP → score. block ② ctx 128D 는 cand_feat 안에 포함 (묶음③ slice [12:140]).`
- `decision-note: spec-default — anchor_embed_dim=0 (plan-024 v5 default OFF carry). 사용자 명시.`
- `decision-note: spec-default — hidden=196 (사용자 명시). plan-024 384 의 51%. capacity 축소 + 학습 안정.`
- `decision-note: spec-default — training schedule = epoch=50 fixed (no early stop, plan-024 167s CPU under-converged 회피), lr=7e-4 cosine + warmup 5 epoch, AdamW (weight_decay=1e-4), dropout=0.10, gradient_clip=1.0, batch=64, KL divergence soft label loss (Σ q · log(q/p)). 사용자 "task 에 맞춰 재설계" 위임.`
- `decision-note: spec-default — random_state=20260522 (본 plan layer). plan-024 reproduce 와 별개.`
- `decision-note: spec-default — input feature 의 NaN/Inf 처리 = torch.nan_to_num(input, nan=0.0, posinf=1e3, neginf=-1e3) before forward. plan-021/024 의 sigmoid overflow warning 잔재 대응.`

---

## §1. 배경

### §1.1 plan-024/025 finding 과 본 plan 의 응답

| Plan | Best | hit_1cm | hit_1p5cm | Finding |
|:--|:--|--:|--:|:--|
| plan-022 | A6_bcc14_tau001 | 0.6528 | 0.8104 | LGBM 170D selector floor (winner) |
| plan-023 | B4_fib50_tau001 | 0.6532 | 0.8108 | anchor large-N marginal (+0.0004) |
| plan-024 | cross-attention | 0.6370 | 0.8092 | paradigm 자체 미검증 fail (CPU under-converged + 다중 lever) |
| plan-025 | C1 LGBM 1080D | 0.6320 | 0.8033 | mode collapse → F0 복사 (paradigm mismatch evidence) |
| plan-026 (abandoned) | A2 no-block③ | 0.6509 | 0.8118 | LGBM block ablation finding (사용자 의도 mismatch) |
| plan-027 (abandoned) | E3 weighted | 0.6529 | 0.8118 | LGBM ensemble negative (사용자 의도 mismatch) |

본 plan 의 응답:
- **plan-024 paradigm 정상 검증**: hidden 384→196, epoch 50 fixed, lr 재설계. CPU under-converged 의심 회피.
- **plan-025 1080D input 정상 활용**: block ① ② ③ ④ 의 attention-paradigm 분해 (§0 paradigm rationale 3).
- **abandoned plan-026 + plan-027 통합 재발행** (사용자 plan-026/027 GRU-attention 의도 합의).

### §1.2 paradigm 가설

- **H1 (강)**: GRU-attention 위 1080D input → hit_1cm ≥ 0.6528 (plan-022 winner 회복). plan-024 fail 의 원인 = CPU under-converged + hyperparameter 였음을 검증.
- **H2 (약)**: hit_1cm > 0.6531 (plan-022/023 winner 초과). paradigm-distinct lever 가 anchor selector ceiling 위 추가 lift.
- **H3 (강)**: max_class_ratio > 0.10 (mode collapse 미발생). plan-025 LGBM 의 1/K uniform 와 *질적으로 다름*.

H1 FAIL = paradigm 자체 ceiling 한계 (plan-022 LGBM floor 못 회복) → plan-030 후속 무의미, 다른 lever (F0 ML / corrector) 직행.
H1 PASS + H2 FAIL = paradigm 회복 but anchor selector ceiling 한계 (~0.6530).
H3 FAIL (mode collapse) = paradigm 본질 fail. 매우 unlikely (plan-024 0.6370 가 F0 0.6320 보다 +0.005 lift = mode collapse 아닌 증거).

### §1.3 baseline anchor

- **G1.a F0** (plan-020 carry): 0.6320 / 0.8033. 모든 paired Δ anchor.
- **G1.b plan-022 winner** (carry from plan-025 baseline_carry.json 또는 재산출): 0.6531 / 0.8108. paradigm ceiling reference.

---

## §2. 가설 검증 paradigm (한 변수 원칙)

| 축 | 변경 | 단일 변수 |
|:--|:--|:--|
| Anchor codebook | K=14 BCC fix | ✗ (carry) |
| τ_cls | 0.001 fix | ✗ (carry) |
| Soft label 산식 | `build_soft_label_with_tau` | ✗ (carry) |
| 5-fold split | `stable_fold_id` | ✗ (carry) |
| F0 baseline | `f0_baseline` | ✗ (carry) |
| Input 1080D | plan-025 동일 (+ raw seq 7×95 GRU input) | ✗ (carry) |
| **Model paradigm** | **LGBM → GRU-attention** | **✓ 본 plan 변수** |
| GRU hidden | 196 (plan-024 384 의 51%) | (단일 cell 의 sub-decision) |
| anchor_embed_dim | 0 (plan-024 v5 default OFF) | (단일 cell) |
| Training schedule | 50 epoch fixed + AdamW + cosine | (단일 cell, plan-024 재설계) |

본 plan 의 *paradigm 1축 단일 변수*. plan-024 와 비교 시 hyperparameter 도 변경 — 단일 cell 의 sub-decision, 사용자 명시 + decision-note 박제.

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split (plan-020/021/022/025 carry)

- 5-fold rotating, `stable_fold_id(sample_id_str, n_folds=5)`. MD5 deterministic.
- N=10000 samples → per-fold test ≈ 2000, train ≈ 8000.
- dataset_hash 일치 (plan-025 baseline_carry.json 의 hash carry).

### §3.2 합격 기준

| Gate | 합격 |
|:--|:--|
| G0 | 8+ pytest green + plan-024 cherry-pick 9 file import OK |
| G1.a | F0 hit_1cm ∈ [0.6315, 0.6325] AND hit_1p5cm ∈ [0.8028, 0.8038] |
| G1.b | plan-022 winner hit_1cm ∈ [0.6523, 0.6533] AND hit_1p5cm ∈ [0.8099, 0.8109] |
| G2.X1 | metric finite + max_class_ratio < 0.95 + epoch 50 fully trained ✓ |
| **G3** | PASS hit_1cm > 0.6528 / partial 0.6320 ≤ hit ≤ 0.6528 / regression < 0.6320 |
| G_final | 3-file sync + §0.5 c1~c10 [DONE] + follow-up 1+ 건 |

### §3.3 평가 점수

- **primary**: `hit_1cm` = mean(D1(pred, gt) ≤ 0.01). 5-fold concat OOF.
- **secondary**: `hit_1p5cm`, `top1_acc` (argmax probs vs gt anchor label), `max_class_ratio` (mode collapse 진단).
- **paired Δ**: vs F0 (G1.a) + vs plan-022 winner (G1.b).
- **14-oracle 회수율**: `best_hit_1cm / 0.7928`.

### §3.4 Model spec (X1 cell)

```python
# §3.4.1 forward path (단일 cell X1)
seq         = seq_builder.build(X, R_wfn, ANCHORS_A6, f0_baseline, quantile_carry)   # (B, 7, 95)
cand_feat   = cand_builder.build(X, R_wfn, pred_F0, ANCHORS_A6, f0_baseline, regimes, quantile_carry)  # (B, 14, 150)
feat_1080   = build_feat_1080(X, ANCHORS_A6, f0_baseline, quantile_carry)            # (B*14, 1080) — head skip source

# GRU encoder
out, h      = GRU(input_size=95, hidden=196, num_layers=2, dropout=0.10, batch_first=True)(seq)  # out (B, 7, 196), h (2, B, 196)
h_final     = h[-1]                                                                              # (B, 196)
h_final_bc  = h_final.unsqueeze(1).expand(-1, 14, -1)                                            # (B, 14, 196)

# Cross-attention (anchor query × seq key)
query       = query_mlp(cand_feat)                                                               # (B, 14, 196) — Linear(150→196)+GELU+Linear(196→196)
attn_logits = einsum("bth,bkh->bkt", out, query) / sqrt(196)                                     # (B, 14, 7)
attn        = softmax(attn_logits, dim=-1)                                                       # (B, 14, 7)
event_ctx   = einsum("bkt,bth->bkh", attn, out)                                                  # (B, 14, 196)

# Head MLP — skip connection 강화 (plan-025 block ①+④ 추가)
feat_1080_unflat = feat_1080.reshape(B, 14, 1080)                                                # (B, 14, 1080)
block1_skip      = feat_1080_unflat[:, 0, 0:170]                                                 # (B, 170) — anchor 무관 generic
block4_skip      = feat_1080_unflat[:, 0, 320:1080]                                              # (B, 760) — anchor 무관 seq stat (broadcast 동일)
block1_bc        = block1_skip.unsqueeze(1).expand(-1, 14, -1)                                   # (B, 14, 170)
block4_bc        = block4_skip.unsqueeze(1).expand(-1, 14, -1)                                   # (B, 14, 760)
head_in     = concat([h_final_bc, event_ctx, cand_feat, block1_bc, block4_bc], dim=-1)           # (B, 14, 196+196+150+170+760 = 1472)
head        = Linear(1472 → 384) → GELU → Dropout(0.15) → Linear(384 → 1)
score       = head(head_in).squeeze(-1)                                                          # (B, 14)
probs       = softmax(score, dim=-1)                                                             # (B, 14)
```

### §3.5 Training schedule (X1)

| Hparam | 값 | 사유 |
|:--|--:|:--|
| epochs | **50 fixed** | plan-024 167s under-converged 회피 |
| early_stopping | **disabled** | early stop noise 회피 |
| optimizer | AdamW | 표준 |
| lr | 7e-4 | attention 표준 |
| lr_schedule | cosine + warmup 5 epoch | attention 학습 안정 |
| weight_decay | 1e-4 | AdamW 표준 |
| dropout | 0.10 | plan-024 carry |
| gradient_clip | 1.0 | attention 안정 |
| batch_size | 64 | N=10000 / hidden=196 합리적 |
| random_state | 20260522 | 본 plan layer |
| loss | KL divergence | `Σ q · log(q / p)` over K=14 anchor. p = model softmax. q = soft label. plan-022 동일 paradigm. |

### §3.6 Loss 식 (KL divergence)

```python
log_probs = log_softmax(score, dim=-1)             # (B, 14)
soft_q    = build_soft_label_with_tau(gt, R_wfn, F0, ANCHORS_A6, tau_cls=0.001)  # (B, 14)
loss      = -(soft_q * log_probs).sum(dim=-1).mean()      # (= mean KL up to constant entropy of soft_q)
```

mean reduction (batch + K aggregation).

### §3.7 Prediction (eval mode)

```python
probs       = softmax(score, dim=-1)                                              # (B, 14)
residual_f  = einsum("bk,kj->bj", probs, ANCHORS_A6)                              # (B, 3) Frenet
residual_w  = einsum("bij,bj->bi", R_wfn_test, residual_f)                        # (B, 3) world
final_pred  = F0_test + residual_w                                                # (B, 3) world
hit_1cm     = (norm(final_pred - gt, dim=-1) <= 0.01).float().mean()
```

---

## §4. STAGE 0 — 인프라 (G0)

### §4.1 모듈 layout

```
analysis/plan-029/
├── __init__.py
├── model.py                 ← plan-024 model.py 의 hidden=196 wrapper (c3)
├── train.py                 ← PyTorch 5-fold OOF training (c4)
├── run_oof.py               ← orchestrator + G1 reproduce + CLI (c5)
├── baseline_carry.json      ← G1 박제 (c7)
├── results_X1.json          ← G2.X1 박제 (c8)
├── train_X1.log             ← 학습 진행 log (c8)
├── paradigm_analysis.{json,md}  ← c9
└── results.md               ← c10

analysis/plan-024/            ← c2 추가 cherry-pick from worktree-plan-024-combo
├── model.py                  ← 신규 cherry-pick
└── feature_weighted_dropout.py  ← 신규 cherry-pick
(기존 8 file: __init__.py + anchor_vocab.py + cand_builder.py + seq_builder.py + torsion_calc.py + quantile_carry.py + multiwindow_trim_build.py + multiwindow_trim.json — plan-025 c2 cherry-pick 으로 이미 main 존재)

tests/test_plan029_smoke.py   ← 8+ pytest (c6)
```

### §4.2 plan-024 cherry-pick (c2)

```bash
git checkout worktree-plan-024-combo -- analysis/plan-024/model.py analysis/plan-024/feature_weighted_dropout.py
```

commit hash carry: `worktree-plan-024-combo` 의 latest (commit 915dd26 또는 그 이후 minor patch).

### §4.3 tests (c6)

- `test_imports`: plan-024 model + feature_weighted_dropout + plan-025 build_feat_1080 + plan-022 anchors + plan-021 build_input + plan-020 baseline_f0 모두 import OK.
- `test_model_forward_shape`: dummy `seq (B=4, 7, 95)` + `cand_feat (B=4, 14, 150)` + `feat_1080 (B*14, 1080)` → score (B=4, 14).
- `test_gru_hidden_dim`: GRU encoder hidden = 196 (config).
- `test_anchor_embed_default_off`: model.anchor_embed_dim == 0 (default).
- `test_soft_label_sum_one`: build_soft_label_with_tau output row-sum = 1.
- `test_frenet_to_world_inverse`: round-trip Frenet → world → Frenet (identity within tolerance).
- `test_kl_loss_nonneg`: KL loss ≥ 0 (model prob ≠ uniform 시).
- `test_build_feat_1080_carry`: plan-025 build_feat_1080 output shape (B*14, 1080) + BLOCK_DIMS sum = 1080.

---

## §5. STAGE 1 — G1 reproduce (c7)

### §5.1 carry from plan-025 baseline_carry.json

```python
import json
prereq_path = "analysis/plan-025/baseline_carry.json"
with open(prereq_path) as f:
    p025_baseline = json.load(f)

F0_hit_1cm = p025_baseline["F0"]["hit_1cm"]      # 0.6320
F0_hit_1p5cm = p025_baseline["F0"]["hit_1p5cm"]  # 0.8033
p022_hit_1cm = p025_baseline["plan022_winner"]["hit_1cm"]    # 0.6531
p022_hit_1p5cm = p025_baseline["plan022_winner"]["hit_1p5cm"]  # 0.8108

assert 0.6315 <= F0_hit_1cm <= 0.6325, f"F0 drift: {F0_hit_1cm}"
assert 0.8028 <= F0_hit_1p5cm <= 0.8038
assert 0.6523 <= p022_hit_1cm <= 0.6533
assert 0.8099 <= p022_hit_1p5cm <= 0.8109
```

plan-025 의 baseline_carry.json 이 이미 reproduce 결과 박제 (main commit e262299). 본 plan 은 carry only (재산출 X).

### §5.2 G1 합격

- carry value 가 tight band ✓ → G1 PASS.
- 재산출 옵션 (decision-note): `--cell G1` CLI 로 본 plan 안에서 새로 reproduce 가능 (drift 발생 시 carry replace).

---

## §6. STAGE 2 — X1 cell 5-fold OOF (c8)

### §6.1 Per-fold loop

```python
for fold in range(5):
    train_idx = np.where(folds != fold)[0]
    test_idx = np.where(folds == fold)[0]
    X_tr, X_te = X[train_idx], X[test_idx]
    gt_tr, gt_te = gt[train_idx], gt[test_idx]

    # Frenet basis + F0
    R_wfn_tr = build_frenet_basis_3d(X_tr, end_idx=10)
    R_wfn_te = build_frenet_basis_3d(X_te, end_idx=10)
    F0_tr = f0_baseline(X_tr, end_idx=10).astype(np.float32)
    F0_te = f0_baseline(X_te, end_idx=10).astype(np.float32)

    # Fold-leakage 차단: train fold quantile
    qc = quantile_carry.build(X_tr, R_wfn_tr)

    # Input feature 산출 (train + test 동일 quantile)
    cand_tr = cand_builder.build(X_tr, R_wfn_tr, F0_tr, ANCHORS_A6, f0_baseline,
                                  regimes=assign_regimes(X_tr, end_idx=10, bins=fit_regime_bins(X_tr, end_idx=10)),
                                  quantile_carry=qc)
    seq_tr  = seq_builder.build(X_tr, R_wfn_tr, ANCHORS_A6, f0_baseline, quantile_carry=qc)
    feat_1080_tr = build_feat_1080(X_tr, ANCHORS_A6, f0_baseline, qc)
    # (test 동일 — 코드 생략)

    # Soft label
    q_tr = build_soft_label_with_tau(gt_tr, R_wfn_tr, F0_tr, ANCHORS_A6, tau_cls=0.001)

    # Model + Optimizer
    model = GRUNetX1(hidden=196, anchor_embed_dim=0, dropout=0.10)
    optimizer = AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, ...) + warmup 5 epoch
    
    # Training (epoch=50 fixed, no early stop)
    for epoch in range(50):
        for batch in batched(64):
            log_probs = log_softmax(model(seq_batch, cand_batch, feat_1080_batch), dim=-1)
            loss = -(q_batch * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    
    # Eval
    model.eval()
    with torch.no_grad():
        probs_te = softmax(model(seq_te, cand_te, feat_1080_te), dim=-1).cpu().numpy()  # (N_te, 14)
        residual_frenet = (probs_te[:, :, None] * ANCHORS_A6[None, :, :]).sum(axis=1)
        residual_world = np.einsum("nij,nj->ni", R_wfn_te, residual_frenet)
        final_pred = F0_te + residual_world
        oof_pred[test_idx] = final_pred
        oof_probs[test_idx] = probs_te

# Concat OOF metric
err = np.linalg.norm(oof_pred - gt, axis=1)
hit_1cm = (err <= 0.01).mean()
hit_1p5cm = (err <= 0.015).mean()
max_class_ratio = oof_probs.mean(axis=0).max()
top1_acc = (oof_probs.argmax(axis=1) == gt_anchor_label).mean()
```

### §6.2 Runtime 예상

- per-fold: 50 epoch × N_train=8000 / batch=64 = 6250 step/fold ≈ 30-40min CPU (hidden=196 plan-024 384 의 27% FLOPs)
- 5-fold total: 2.5-3.5h CPU

### §6.3 G2.X1 합격

- metric finite ✓ (NaN/Inf X)
- max_class_ratio < 0.95 ✓
- epoch 50 fully trained ✓ (no early stop)
- 위반 1 = severe halt

---

## §7. STAGE 3 — Paradigm finding (c9, G3)

### §7.1 X1 결과 표

| Metric | X1 | F0 (G1.a) | plan-022 (G1.b) | plan-024 | plan-025 C1 |
|:--|--:|--:|--:|--:|--:|
| hit_1cm | ?.???? | 0.6320 | 0.6531 | 0.6370 | 0.6320 |
| hit_1p5cm | ?.???? | 0.8033 | 0.8108 | 0.8092 | 0.8033 |
| max_class_ratio | ?.??? | — | 0.1054 | ? | 0.0714 |
| top1_acc | ?.???? | — | — | 0.1227 | 0.0879 |
| oracle 회수율 | ?.??% | — | 82.4% | 80.4% | 79.7% |
| runtime | ?h | — | (carry) | 167s (under-conv) | 334s |

### §7.2 G3 판정

- **PASS** (band=positive): hit_1cm > 0.6528 → paradigm 정상 검증, GRU-attention 의 mode collapse 미발생, plan-024 fail 의 원인 = CPU under-converged + hyperparameter 였음을 입증.
- **partial_drift** (band=partial): 0.6320 ≤ hit_1cm ≤ 0.6528 → paradigm 작동 but plan-022 baseline 미회복. lift 잠재력 박제 + plan-030 후속 (architecture 또는 input lever 변경).
- **regression** (band=negative): hit_1cm < 0.6320 → paradigm mismatch 본질. plan-024 paradigm 자체 폐기, plan-028 (F0 ML) 또는 plan-030 (corrector) 으로 전환.

### §7.3 Hypothesis 검증

- H1 (≥ 0.6528): PASS / partial / FAIL
- H2 (> 0.6531): PASS / FAIL
- H3 (max_class_ratio > 0.10): PASS / FAIL (mode collapse 미발생)

### §7.4 paradigm finding 박제

- plan-024 fail 의 본질 (hyperparameter vs paradigm 자체) 판정
- plan-025 mode collapse 의 GRU-attention 회복 여부 판정
- plan-030 후속 lever 우선순위 결정

---

## §8. STAGE 4 — G_final (c10)

### §8.1 산출

- `analysis/plan-029/results.md` (11 항목)
- `plans/plan-029-*.results.md` pair
- 3-file frontmatter sync (status=all_complete, band=positive/partial/negative, best_cell=X1, best_hit_1cm, best_delta_1cm)
- follow-up plan-030 (가칭) 후보 ≥ 1 건 박제

### §8.2 G_final 합격

- 3-file sync ✓
- §0.5 c1~c10 모두 [DONE] ✓
- follow-up 1+ 건 박제 ✓

---

## §9. Out of scope

- Ensemble (plan-030 후보)
- DACON LB submit (별개 결정)
- boundary corrector (plan-030 후보)
- F0 baseline ML (plan-028 후보)
- anchor layout 변경 (K=14 BCC fix)
- τ_cls 변경 (0.001 fix)
- hidden ≠ 196 sweep (단일 cell)
- batch ≠ 64, lr ≠ 7e-4 sweep
- corrector / 2-stage residual regression
- anchor_embed_dim ≠ 0 (사용자 명시 OFF)

---

## §10. 참조 (read-only)

- spec: plan-022 / plan-024 / plan-025 carry
- carry: plan-020/021/022/024/025 module (frontmatter `code_reuse` 참조)
- abandoned: plan-026 / plan-027 (supersedes_abandoned frontmatter 명시)
- memory: `project_next_plan_direction.md` (2026-05-22 user GRU-attention 의도 + plan-029/030 mapping)
