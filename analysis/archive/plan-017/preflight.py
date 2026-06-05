"""plan-017 c2 (STAGE 0, G0) — preflight.

4 task per plan-017 v1.3 §4:
  (a) 3 submission file 존재 + 10000 row + header
  (b) plan-016 G1 artifact 값 정확 일치 (OOF=0.6452, LB=0.6638 tolerance 0)
  (c) Plan017VoxelCEHead module smoke (B=16, seq_len=6, feature_dim=9 → logits (16, 125))
  (d) Voxel grid 2cm window coverage measure on 10K train

Usage:
    python analysis/plan-017/preflight.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822 import plan014_paradigm as pp  # noqa: E402
from src.pb_0_6822 import plan017_voxel_ce as v17  # noqa: E402


# plan-016 G1 baseline anchor (tolerance 0)
BASELINE_OOF = 0.6452
BASELINE_LB = 0.6638

SUBMISSION_PATHS = {
    "plan_013": "analysis/plan-013/submission.csv",
    "plan_014": "runs/baseline/plan014_g5_phase4/submission_best.csv",
    "plan_016_g1": "runs/baseline/plan016_g1_path_a/submission.csv",
}


def task_a_files() -> dict:
    res = {}
    all_ok = True
    for k, p in SUBMISSION_PATHS.items():
        path = REPO_ROOT / p
        exists = path.exists()
        if not exists:
            res[k] = {"path": str(p), "exists": False}
            all_ok = False
            continue
        df = pd.read_csv(path)
        ok = (df.shape == (10000, 4)) and list(df.columns) == ["id", "x", "y", "z"]
        res[k] = {
            "path": str(p),
            "exists": True,
            "shape": list(df.shape),
            "schema_ok": bool(ok),
        }
        if not ok:
            all_ok = False
    res["all_ok"] = bool(all_ok)
    return res


def task_b_baseline() -> dict:
    g1_path = REPO_ROOT / "analysis/plan-016/g1_path_a.json"
    if not g1_path.exists():
        return {"path": str(g1_path), "exists": False, "ok": False}
    g1 = json.loads(g1_path.read_text())
    oof = g1.get("overall_oof_hit_1cm")
    lb = g1.get("lb_score")
    oof_match = (oof == BASELINE_OOF)
    lb_match = (lb == BASELINE_LB)
    return {
        "path": str(g1_path),
        "oof": oof,
        "lb": lb,
        "target_oof": BASELINE_OOF,
        "target_lb": BASELINE_LB,
        "oof_match": bool(oof_match),
        "lb_match": bool(lb_match),
        "ok": bool(oof_match and lb_match),
    }


def task_c_smoke() -> dict:
    """Voxel CE head smoke: forward + loss + backward."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260514)

    B, seq_len, feature_dim = 16, 6, 9
    encoder = pp.Plan014BiGRUEncoder(input_dim=feature_dim, hidden=128, num_layers=2, dropout=0.1).to(device)
    voxel_head = v17.Plan017VoxelCEHead(encoder_out_dim=encoder.out_dim).to(device)

    seq = torch.randn(B, seq_len, feature_dim, device=device)
    y = torch.randn(B, 3, device=device) * 0.01   # 1cm scale
    f0_pred = torch.zeros(B, 3, device=device)

    h = encoder(seq)
    logits = voxel_head(h)
    assert logits.shape == (B, v17.VOXEL_TOTAL), f"logits shape {logits.shape}"

    loss = v17.voxel_ce_loss(logits, y, f0_pred)
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
    loss.backward()

    # Forward predict smoke
    encoder.eval(); voxel_head.eval()
    with torch.no_grad():
        pred = v17.hybrid_predict_voxel(seq, encoder, voxel_head, f0_pred)
    assert pred.shape == (B, 3), f"pred shape {pred.shape}"

    return {
        "ok": True,
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "pred_shape": list(pred.shape),
        "device": str(device),
    }


def task_d_coverage() -> dict:
    """10K train sample 위 ||y - F0||₂ 분포 + coverage at ±3cm (VOXEL_DEPTH=7)."""
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float64); Y = Y.astype(np.float64)

    f0_fn = pp.Plan014F0Function()
    F0 = f0_fn(X)   # (N, 3)
    dist = np.linalg.norm(Y - F0, axis=1)   # (N,)

    voxel_half_m = v17.HALF * v17.VOXEL_WIDTH   # 0.03 (=3cm) for 7×7×7
    coverage_frac = float((dist <= voxel_half_m).mean())
    p50 = float(np.quantile(dist, 0.50))
    p95 = float(np.quantile(dist, 0.95))
    p99 = float(np.quantile(dist, 0.99))

    # plan-017 §4.1 (d) 3 단계 rule
    if coverage_frac >= 0.95:
        verdict = "OK_no_caveat"
        ok = True
    elif coverage_frac >= 0.90:
        verdict = "caveat"
        ok = True
    else:
        verdict = "FAIL"
        ok = False

    return {
        "ok": ok,
        "verdict": verdict,
        "voxel_half_m": voxel_half_m,
        "voxel_depth": v17.VOXEL_DEPTH,
        "voxel_total": v17.VOXEL_TOTAL,
        "coverage_frac": coverage_frac,
        "p50_dist_m": p50,
        "p95_dist_m": p95,
        "p99_dist_m": p99,
        "n_samples": int(len(dist)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("analysis/plan-017/preflight.json"))
    args = ap.parse_args()

    t0 = time.time()
    print("[plan-017 G0] (a) submission files ...", flush=True)
    a = task_a_files()
    print(f"  {a}", flush=True)

    print("[plan-017 G0] (b) plan-016 G1 baseline reproduce ...", flush=True)
    b = task_b_baseline()
    print(f"  {b}", flush=True)

    print("[plan-017 G0] (c) Voxel CE module smoke ...", flush=True)
    c = task_c_smoke()
    print(f"  {c}", flush=True)

    print("[plan-017 G0] (d) Voxel grid coverage (2cm) ...", flush=True)
    d = task_d_coverage()
    print(f"  {d}", flush=True)

    g0_passed = a["all_ok"] and b["ok"] and c["ok"] and d["ok"]

    elapsed = time.time() - t0
    summary = {
        "exp_id": "H057_g0_preflight",
        "plan_version": "v1.3",
        "task_a_files": a,
        "task_b_baseline": b,
        "task_c_smoke": c,
        "task_d_coverage": d,
        "g0_passed": bool(g0_passed),
        "elapsed_seconds": elapsed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[plan-017 G0] g0_passed = {g0_passed}, artifact -> {args.out_json} ({elapsed:.1f}s)", flush=True)
    return 0 if g0_passed else 1


if __name__ == "__main__":
    sys.exit(main())
