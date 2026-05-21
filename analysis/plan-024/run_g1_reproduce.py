"""plan-024 c9 — G1 carry reproduce script.

F0 baseline 5-fold OOF + plan-022 winner (A6_bcc14_τ001) reproduce.
박제: analysis/plan-024/baseline_carry.json

Carry tolerance (§3.2):
  - F0 hit@1cm   ∈ [0.6315, 0.6325]  (carry 0.6320 ± 0.0005)
  - F0 hit@1.5cm ∈ [0.8028, 0.8038]  (carry 0.8033)
  - plan-022 winner hit_1cm   ∈ [0.6520, 0.6536]  (carry 0.6528 ± 0.0008)
  - plan-022 winner hit_1.5cm ∈ [0.8096, 0.8112]  (carry 0.8104)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import stable_fold_id

_spec = importlib.util.spec_from_file_location(
    "p020_bf", _REPO / "analysis" / "plan-020" / "baseline_f0.py"
)
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)

_spec = importlib.util.spec_from_file_location(
    "p022_anchors", _REPO / "analysis" / "plan-022" / "anchors.py"
)
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)

_spec = importlib.util.spec_from_file_location(
    "p022_oof", _REPO / "analysis" / "plan-022" / "run_oof.py"
)
p022_oof = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p022_oof)

N_FOLDS = 5
R_HIT = 0.01
R_HIT_LOOSE = 0.015
SEED = 20260521


def main():
    t0 = time.time()
    print("[G1] load data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    N = X.shape[0]
    folds = np.asarray([stable_fold_id(str(s), N_FOLDS) for s in ids], dtype=int)
    print(f"[G1] N={N} folds={np.bincount(folds).tolist()}", flush=True)

    # ── F0 baseline 5-fold concat OOF ────────────────────────────────
    print(f"[G1] F0 baseline reproduce ...", flush=True)
    pred_F0 = bf.f0_baseline(X, end_idx=10).astype(np.float64)
    d_F0 = np.linalg.norm(pred_F0 - Y, axis=1)
    f0_hit_1cm = float((d_F0 <= R_HIT).mean())
    f0_hit_15cm = float((d_F0 <= R_HIT_LOOSE).mean())
    print(f"  F0 hit@1cm   = {f0_hit_1cm:.4f}  (carry [0.6315, 0.6325])", flush=True)
    print(f"  F0 hit@1.5cm = {f0_hit_15cm:.4f}  (carry [0.8028, 0.8038])", flush=True)
    f0_carry_pass = (0.6315 <= f0_hit_1cm <= 0.6325) and (0.8028 <= f0_hit_15cm <= 0.8038)
    print(f"  F0 carry pass: {f0_carry_pass}", flush=True)

    # ── plan-022 A6_bcc14_τ001 reproduce ──────────────────────────────
    print(f"\n[G1] plan-022 A6_bcc14_tau001 reproduce ...", flush=True)
    anchors = anchors_mod.ANCHORS_A6
    t1 = time.time()
    cell = p022_oof.run_oof_cell(X, Y, folds, anchors, tau_cls=0.001, verbose=True)
    print(f"  hit@1cm   = {cell['hit_1cm']:.4f}  (carry [0.6520, 0.6536])", flush=True)
    print(f"  hit@1.5cm = {cell['hit_1.5cm']:.4f}  (carry [0.8096, 0.8112])", flush=True)
    p022_carry_pass = (
        0.6520 <= cell["hit_1cm"] <= 0.6536
        and 0.8096 <= cell["hit_1.5cm"] <= 0.8112
    )
    print(f"  plan-022 carry pass: {p022_carry_pass}  ({time.time()-t1:.0f}s)", flush=True)

    # ── baseline_carry.json 박제 ─────────────────────────────────────
    out = {
        "n_samples": int(N),
        "fold_dist": np.bincount(folds).tolist(),
        "f0_oof_hit_1cm": f0_hit_1cm,
        "f0_oof_hit_1.5cm": f0_hit_15cm,
        "f0_carry_pass": bool(f0_carry_pass),
        "plan022_winner_oof_hit_1cm": float(cell["hit_1cm"]),
        "plan022_winner_oof_hit_1.5cm": float(cell["hit_1.5cm"]),
        "plan022_winner_delta_1cm": float(cell["delta_1cm"]),
        "plan022_winner_delta_1.5cm": float(cell["delta_1.5cm"]),
        "plan022_carry_pass": bool(p022_carry_pass),
        "g1_pass": bool(f0_carry_pass and p022_carry_pass),
        "elapsed_sec": float(time.time() - t0),
    }
    out_path = _THIS / "baseline_carry.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[G1] G1_pass = {out['g1_pass']} → {out_path}", flush=True)
    print(f"[G1] total {out['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
