"""plan-025 — Candidate-concat Input Max (1080D LGBM selector, K=14 BCC, τ=0.001).

cross-attention 버리고 LGBM + 후보 concat + seq 압축. plan-022 winner cell
(A6_bcc14, τ=0.001) 위 input 170D → 1080D 확장.

Block composition:
  ① plan-022 carry          170D  sample-level (broadcast K=14)
  ② cand_builder ctx        128D  sample-level (broadcast K=14)
  ③ cand_builder per-anchor  22D  sample × anchor (K=14 별 다름)
  ④ seq_builder 8-stat 압축 760D  sample-level (broadcast K=14)
  -----------------------------
  Total per LGBM row        1080D
"""
