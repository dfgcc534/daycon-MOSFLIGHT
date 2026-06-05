"""plan-017 c3 (STAGE 1, G1) — 3-plan submission 좌표 mean ensemble.

§5.2 spec: 3 submission (plan-013, plan-014/015, plan-016 G1) 의 좌표 mean.
sample_submission.csv 순서 invariant + id-merge graceful fall-back.

Usage:
    python analysis/plan-017/g1_ensemble.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SOURCES = [
    ("plan_013", REPO_ROOT / "analysis/plan-013/submission.csv"),
    ("plan_014_15", REPO_ROOT / "runs/baseline/plan014_g5_phase4/submission_best.csv"),
    ("plan_016_g1", REPO_ROOT / "runs/baseline/plan016_g1_path_a/submission.csv"),
]


def align_to(s_ref: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """sample_submission.csv 순서 invariant 가 깨진 경우 id-merge 로 graceful 정렬."""
    if (s["id"].values == s_ref["id"].values).all():
        return s
    merged = s_ref[["id"]].merge(s, on="id", how="left", validate="one_to_one")
    if merged[["x", "y", "z"]].isnull().any().any():
        raise ValueError("id set mismatch (not just order)")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-017/g1_ensemble.json"))
    ap.add_argument("--run-dir", type=Path, default=Path("runs/baseline/plan017_g1_ensemble"))
    args = ap.parse_args()

    t0 = time.time()
    subs = {}
    for k, p in SOURCES:
        if not p.exists():
            print(f"ERROR: {p} missing", file=sys.stderr)
            return 1
        subs[k] = pd.read_csv(p)
        print(f"  loaded {k}: shape={subs[k].shape}", flush=True)

    ref = subs["plan_013"]
    aligned = {"plan_013": ref}
    aligned["plan_014_15"] = align_to(ref, subs["plan_014_15"])
    aligned["plan_016_g1"] = align_to(ref, subs["plan_016_g1"])

    xyz_stack = np.stack([
        aligned["plan_013"][["x", "y", "z"]].values,
        aligned["plan_014_15"][["x", "y", "z"]].values,
        aligned["plan_016_g1"][["x", "y", "z"]].values,
    ], axis=0)   # (3, 10000, 3)
    mean_xyz = xyz_stack.mean(axis=0)   # (10000, 3)

    # Per-source L2 distance to mean (variance proxy)
    dist_to_mean = np.linalg.norm(xyz_stack - mean_xyz[None, :, :], axis=2)   # (3, 10000)
    per_source_l2 = {
        SOURCES[i][0]: {
            "mean_dist_m": float(dist_to_mean[i].mean()),
            "p95_dist_m": float(np.quantile(dist_to_mean[i], 0.95)),
        } for i in range(3)
    }

    ids = ref["id"].tolist()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.run_dir / "submission.csv"
    out = pd.DataFrame({
        "id": ids,
        "x": [f"{v:.6f}" for v in mean_xyz[:, 0]],
        "y": [f"{v:.6f}" for v in mean_xyz[:, 1]],
        "z": [f"{v:.6f}" for v in mean_xyz[:, 2]],
    })
    out.to_csv(submission_path, index=False)
    print(f"\n  ensemble submission -> {submission_path}", flush=True)

    elapsed = time.time() - t0
    summary = {
        "exp_id": "H058_g1_ensemble",
        "plan_version": "v1.4",
        "sources": [str(p) for _, p in SOURCES],
        "n_samples": int(mean_xyz.shape[0]),
        "per_source_l2_to_mean": per_source_l2,
        "baseline_lb_plan016_g1": 0.6638,
        "submission_path": str(submission_path),
        "lb_score": None,   # post dacon-submit
        "lb_pass": None,
        "elapsed_seconds": elapsed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  per-source dist to mean: {per_source_l2}", flush=True)
    print(f"[plan-017 G1] artifact -> {args.out_json} ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
