"""Generate submission CSV from a registry exp.

Picks the registry exp with min cv_mean_eucl (tie-breaker: smallest
window, then earliest exp_id), reloads its config snapshot, predicts on
test data, writes runs/{type}/{exp_id}/submission.csv.

CLI: python -m src.submit            # auto-pick best
     python -m src.submit B001       # use specific exp_id
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baselines.window_polyfit import predict, predict_per_axis, tune_per_axis
from src.io import TIMESTEPS_MS, load_all_samples, load_labels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
REGISTRY = PROJECT_ROOT / "registry.csv"
SAMPLE_SUB = PROJECT_ROOT / "data" / "sample_submission.csv"


def _summary_metric(run_dir: Path) -> tuple[float, int, str]:
    s = json.loads((run_dir / "summary.json").read_text())
    cfg = s["config"]
    if cfg.get("per_axis") and cfg["per_axis"] != "tune":
        windows = [int(w) for (w, _d) in cfg["per_axis"]]
        w_key = max(windows)
    elif cfg.get("per_axis") == "tune":
        chosen = s.get("final_chosen_per_axis") or [[2, 1]]
        w_key = max(c[0] for c in chosen)
    else:
        w_key = int(cfg.get("window", 11))
    return float(s["cv_mean_eucl"]), w_key, s["exp_id"]


def pick_best_exp() -> str:
    df = pd.read_csv(REGISTRY)
    df = df[df["status"] == "complete"]
    rows = []
    for _, r in df.iterrows():
        run_dir = PROJECT_ROOT / r["run_dir"]
        rows.append((*_summary_metric(run_dir), r["run_dir"]))
    rows.sort(key=lambda t: (t[0], t[1], t[2]))
    return rows[0][2]


def predict_with_exp(exp_id: str) -> tuple[Path, np.ndarray, list[str]]:
    df = pd.read_csv(REGISTRY)
    row = df[df["id"] == exp_id]
    if row.empty:
        raise SystemExit(f"exp_id {exp_id} not in registry")
    run_dir = PROJECT_ROOT / row.iloc[0]["run_dir"]
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg = summary["config"]

    test_ids, X_test = load_all_samples("test")

    if cfg.get("per_axis") == "tune":
        chosen = [tuple(c) for c in summary["final_chosen_per_axis"]]
        pred = predict_per_axis(X_test, chosen,
                                t_target=int(cfg.get("t_target", 80)),
                                timesteps=TIMESTEPS_MS)
    elif cfg.get("per_axis"):
        chosen = [tuple(c) for c in cfg["per_axis"]]
        pred = predict_per_axis(X_test, chosen,
                                t_target=int(cfg.get("t_target", 80)),
                                timesteps=TIMESTEPS_MS)
    else:
        pred = predict(X_test, int(cfg["window"]), int(cfg["degree"]),
                       t_target=int(cfg.get("t_target", 80)),
                       timesteps=TIMESTEPS_MS)
    return run_dir, pred, test_ids


def write_submission(run_dir: Path, pred: np.ndarray, test_ids: list[str]) -> Path:
    sample = pd.read_csv(SAMPLE_SUB)
    expected_ids = sample["id"].tolist()
    if set(test_ids) != set(expected_ids):
        raise SystemExit("test_ids do not match sample_submission ids")

    pred_by_id = {tid: pred[i] for i, tid in enumerate(test_ids)}
    rows = [(tid, *pred_by_id[tid]) for tid in expected_ids]
    out = run_dir / "submission.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "z"])
        for tid, x, y, z in rows:
            w.writerow([tid, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    df_out = pd.read_csv(out)
    assert list(df_out.columns) == ["id", "x", "y", "z"]
    assert len(df_out) == len(expected_ids)
    assert df_out["id"].tolist() == expected_ids
    assert df_out[["x", "y", "z"]].notna().all().all()
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp_id", nargs="?", default=None)
    args = p.parse_args(argv)
    exp_id = args.exp_id or pick_best_exp()
    print(f"[submit] using exp_id={exp_id}", flush=True)
    run_dir, pred, test_ids = predict_with_exp(exp_id)
    out = write_submission(run_dir, pred, test_ids)
    print(f"[submit] wrote {out} (n={len(test_ids)})", flush=True)


if __name__ == "__main__":
    main()
