"""plan-b-001 — G1 (1-fold) + G3 (5-fold OOF) orchestrator, 2-arm baseline.

Usage:
  python analysis/plan-b-001/run_oof.py --gate g1 --baseline f0
  python analysis/plan-b-001/run_oof.py --gate g3 --baseline kalman --quiet
G0.5 (baseline standalone hit) 는 g3 결과의 baseline_hit_1cm 필드로 동봉 박제.
"""
from __future__ import annotations

import argparse
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


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train = _load(_THIS / "train.py", "pb_runoof_train")


def _band(h: float) -> str:
    if h >= 0.65:
        return "EXCELLENT"
    if h >= 0.6387:
        return "STRONG"
    if h >= 0.6360:
        return "PASS"
    if h >= 0.6320:
        return "BORDERLINE"
    return "FAIL_regression"


def run_g1(baseline_name, verbose=True):
    from src.io import load_all_samples, load_labels
    from src.pb_0_6822.selector import stable_fold_id
    t0 = time.perf_counter()
    _ids, X = load_all_samples()
    _ids2, gt = load_labels()
    assert _ids == _ids2
    X = X.astype(np.float32); gt = gt.astype(np.float32)
    folds = np.asarray([stable_fold_id(str(sid), _train.N_FOLDS) for sid in _ids], dtype=int)
    tr = np.where(folds != 0)[0]; te = np.where(folds == 0)[0]
    log = _train.train_one_fold(0, X[tr], X[te], gt[tr], baseline_name, verbose=verbose)
    err = np.linalg.norm(log["world_pred_te"] - gt[te], axis=1)
    err_b = np.linalg.norm(log["baseline_final_te"] - gt[te], axis=1)
    h = float((err <= 0.01).mean())
    return {
        "gate": "G1", "baseline": baseline_name, "fold": 0, "N_te": int(len(te)),
        "hit_1cm": h, "hit_1p5cm": float((err <= 0.015).mean()),
        "baseline_hit_1cm": float((err_b <= 0.01).mean()),
        "elapsed_s": time.perf_counter() - t0,
        "PASS_threshold": 0.6290, "PASS": h > 0.6290,
    }


def run_g3(baseline_name, verbose=True):
    res = _train.run_5fold_oof(baseline_name, verbose=verbose)
    res["gate"] = "G3"
    res["band"] = _band(res["hit_1cm"])
    res["PASS"] = res["hit_1cm"] >= 0.6360
    return res


def _dump(result, path):
    arrays = {k: result.pop(k) for k in ("oof_pred",) if k in result}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    if arrays:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["g1", "g3"], required=True)
    ap.add_argument("--baseline", choices=["f0", "kalman"], required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    fn = run_g1 if args.gate == "g1" else run_g3
    result = fn(args.baseline, verbose=not args.quiet)
    out = _THIS / f"results_{args.gate}_{args.baseline}.json"
    _dump(result, out)
    summary = {k: v for k, v in result.items() if k not in ("oof_pred", "fold_logs")}
    print(json.dumps(summary, indent=2))
    print(f"[done] {args.gate}/{args.baseline} → {out}")


if __name__ == "__main__":
    main()
