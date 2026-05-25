"""plan-b-001 c4 — query_builder (잔차b 제거, F1).

plan-030 query 64D 에서 **residual_b_flat 35D 제거** → 29D static anchor 정체성:
  anchor_spec   = cand_feat_150[:, :, 3:12]    (9)
  par_perp_dist = cand_feat_150[:, :, 0:3]     (3)
  interactions  = cand_feat_150[:, :, 140:150] (10)
  slim7         = extension_slim7              (7)
잔차(b) 는 model 의 attention bias 로 이동 (F1). cand_feat (par/perp/dist/interactions) 는
plan-030 cand_builder carry (Frenet 기반) 그대로 — query feature 라 frame 재계산 deviation
(decision-note: 후보 의미 일관성은 residual_b/anchor/decode/soft_label 의 yaw 화로 확보; query
projection 의 yaw 재계산은 cand_builder 전면 개작 비용 대비 효과 낮아 carry).
"""
from __future__ import annotations

import numpy as np


def build_query(cand_feat_150: np.ndarray, extension_slim7: np.ndarray) -> np.ndarray:
    """(N,K,150) + (N,K,7) → (N,K,29) float32 (잔차b 제거)."""
    cand = np.asarray(cand_feat_150)
    sl = np.asarray(extension_slim7)
    N, K, D = cand.shape
    assert D == 150, f"cand_feat_150 last dim must be 150, got {D}"
    assert sl.shape == (N, K, 7), f"slim7 shape mismatch: {sl.shape}"

    anchor_spec = cand[:, :, 3:12]
    par_perp_dist = cand[:, :, 0:3]
    interactions = cand[:, :, 140:150]
    query = np.concatenate([anchor_spec, par_perp_dist, interactions, sl], axis=-1).astype(np.float32)
    assert query.shape == (N, K, 29), f"query 29D mismatch: {query.shape}"
    return np.nan_to_num(query, nan=0.0, posinf=1e3, neginf=-1e3)


def extract_slim7_from_cand_ext_165(cand_ext_165: np.ndarray) -> np.ndarray:
    """plan-029 cand_ext_165 (N,K,165) → slim 7D (cols [159,158,160:165]). plan-030 carry."""
    arr = np.asarray(cand_ext_165)
    assert arr.shape[-1] == 165, f"cand_ext last dim must be 165, got {arr.shape[-1]}"
    cols = [159, 158, 160, 161, 162, 163, 164]
    return arr[:, :, cols].astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(20260526)
    N, K = 8, 14
    cand = rng.standard_normal((N, K, 150)).astype(np.float32)
    cand_ext = rng.standard_normal((N, K, 165)).astype(np.float32)
    sl = extract_slim7_from_cand_ext_165(cand_ext)
    q = build_query(cand, sl)
    assert q.shape == (N, K, 29) and not np.isnan(q).any()
    print(f"[smoke] query_builder OK — q={q.shape} (29D, 잔차b 제거)")
