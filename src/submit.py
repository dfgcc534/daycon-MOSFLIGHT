"""Generate submission CSV from a registry exp.

Picks the registry exp with min cv_mean_eucl (tie-breaker: smallest
window, then earliest exp_id), reloads its config snapshot, predicts on
test data, writes runs/{type}/{exp_id}/submission.csv.

Per plan-002 §8.1, dispatches on cfg["method"] (default "polyfit"):
  - polyfit: existing predict / predict_per_axis path (B001~B004 backward-compat).
  - cspline: predict_cspline / predict_cspline_per_axis; tune path uses
    summary["final_chosen_per_axis"] (full-train re-tune produced by run_baseline).
  - smoothing_spline: predict_smoothing_spline; tune path uses
    summary["final_chosen_s_per_axis"].

Per plan-003 §8.2, additional dispatch:
  - gru-residual: load 5 fold ckpts → per-ckpt predict (baseline + delta) → ensemble mean.

CLI: python -m src.submit            # auto-pick best
     python -m src.submit B001       # use specific exp_id
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.cubic_spline import (
    predict_cspline,
    predict_cspline_per_axis,
    predict_smoothing_spline,
)
from src.baselines.linear_extrapolate import ema_extrapolate, linear_extrap
from src.baselines.window_polyfit import predict, predict_per_axis
from src.io import TIMESTEPS_MS, load_all_samples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
REGISTRY = PROJECT_ROOT / "registry.csv"
SAMPLE_SUB = PROJECT_ROOT / "data" / "sample_submission.csv"


def _summary_metric(run_dir: Path) -> tuple[float, int, str]:
    s = json.loads((run_dir / "summary.json").read_text())
    cfg = s["config"]
    method = cfg.get("method", "polyfit")
    if method == "polyfit":
        if cfg.get("per_axis") and cfg["per_axis"] != "tune":
            windows = [int(w) for (w, _d) in cfg["per_axis"]]
            w_key = max(windows)
        elif cfg.get("per_axis") == "tune":
            chosen = s.get("final_chosen_per_axis") or [[2, 1]]
            w_key = max(c[0] for c in chosen)
        else:
            w_key = int(cfg.get("window", 11))
    elif method == "cspline":
        if cfg.get("per_axis") == "tune":
            chosen = s.get("final_chosen_per_axis") or [[11, "natural"]]
            w_key = max(int(c[0]) for c in chosen)
        elif cfg.get("per_axis"):
            w_key = max(int(c[0]) for c in cfg["per_axis"])
        else:
            w_key = int(cfg.get("window", 11))
    elif method == "gru-residual":
        w_key = 11  # gru-residual: full-window 가정 (key 정렬 무관)
    else:  # smoothing_spline
        w_key = 11  # full-window
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
    run_dir = PROJECT_ROOT / row.iloc[-1]["run_dir"]
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg = summary["config"]
    method = cfg.get("method", "polyfit")
    t_target = int(cfg.get("t_target", 80))

    test_ids, X_test = load_all_samples("test")

    if method == "polyfit":
        if cfg.get("per_axis") == "tune":
            chosen = [tuple(c) for c in summary["final_chosen_per_axis"]]
            pred = predict_per_axis(X_test, chosen, t_target=t_target, timesteps=TIMESTEPS_MS)
        elif cfg.get("per_axis"):
            chosen = [tuple(c) for c in cfg["per_axis"]]
            pred = predict_per_axis(X_test, chosen, t_target=t_target, timesteps=TIMESTEPS_MS)
        else:
            pred = predict(X_test, int(cfg["window"]), int(cfg["degree"]),
                           t_target=t_target, timesteps=TIMESTEPS_MS)

    elif method == "cspline":
        if cfg.get("per_axis") == "tune":
            chosen = [(int(c[0]), str(c[1])) for c in summary["final_chosen_per_axis"]]
            pred = predict_cspline_per_axis(X_test, chosen, t_target=t_target, timesteps=TIMESTEPS_MS)
        elif cfg.get("per_axis"):
            chosen = [(int(c[0]), str(c[1])) for c in cfg["per_axis"]]
            pred = predict_cspline_per_axis(X_test, chosen, t_target=t_target, timesteps=TIMESTEPS_MS)
        else:
            pred = predict_cspline(X_test, int(cfg["window"]), str(cfg["bc_type"]),
                                   t_target=t_target, timesteps=TIMESTEPS_MS)

    elif method == "smoothing_spline":
        if cfg.get("s_per_axis") == "tune":
            chosen_s = [float(s) for s in summary["final_chosen_s_per_axis"]]
            pred = predict_smoothing_spline(
                X_test, chosen_s, t_target=t_target, timesteps=TIMESTEPS_MS,
                s_grid=cfg.get("s_grid"),
            )
        else:
            pred = predict_smoothing_spline(
                X_test, [float(s) for s in cfg["s_per_axis"]],
                t_target=t_target, timesteps=TIMESTEPS_MS,
                s_grid=cfg.get("s_grid"),
            )

    elif method == "gru-residual":
        pred = _predict_gru_residual(X_test, cfg, run_dir, t_target)

    else:
        raise SystemExit(f"unknown method {method!r}")

    return run_dir, pred, test_ids


def _predict_gru_residual(
    X_test: np.ndarray, cfg: dict, run_dir: Path, t_target: int
) -> np.ndarray:
    """5 fold ckpt ensemble: baseline + mean(model_f(feature(X))).

    Per plan-003 §8.2.
    """
    # local imports to avoid torch dependency on closed-form-only paths
    import torch

    from src.models.residual_gru import ResidualGRU
    from src.training.train_residual import make_feature_fn

    feature_components = list(cfg.get("feature_components", ["relative"]))
    wingbeat_n_bins = int(cfg.get("wingbeat_n_bins", 3))
    feature_fn = make_feature_fn(feature_components, wingbeat_n_bins=wingbeat_n_bins)

    bt = cfg.get("baseline_type", "linear")
    if bt == "linear":
        baseline_test = linear_extrap(
            X_test, t_target_ms=t_target, timesteps_ms=TIMESTEPS_MS
        )
    elif bt == "ema":
        alpha = float(cfg.get("ema_alpha", 0.5))
        baseline_test = ema_extrapolate(
            X_test, alpha=alpha, t_target_ms=t_target, timesteps_ms=TIMESTEPS_MS
        )
    else:
        raise SystemExit(f"unknown baseline_type {bt!r}")

    feat_test = feature_fn(X_test).astype(np.float32, copy=False)
    feat_dim = feat_test.shape[-1]
    expected_dim = int(cfg["model"]["input_dim"])
    if feat_dim != expected_dim:
        raise SystemExit(
            f"feature_fn dim {feat_dim} != cfg model.input_dim {expected_dim} for "
            f"{cfg.get('exp_id')}"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_paths = sorted((run_dir / "ckpt").glob("fold*.pt"))
    if not ckpt_paths:
        raise SystemExit(f"no fold ckpts in {run_dir / 'ckpt'}")

    fold_deltas: list[np.ndarray] = []
    for ckpt_path in ckpt_paths:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = ResidualGRU(
            input_dim=feat_dim,
            hidden=int(cfg["model"]["hidden"]),
            layers=int(cfg["model"]["layers"]),
            dropout=float(cfg["model"]["dropout"]),
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        with torch.no_grad():
            delta = model(torch.from_numpy(feat_test).to(device)).cpu().numpy()
        fold_deltas.append(delta)
    delta_ensemble = np.mean(np.stack(fold_deltas, axis=0), axis=0)
    return baseline_test + delta_ensemble


def write_submission(run_dir: Path, pred: np.ndarray, test_ids: list[str]) -> Path:
    sample = pd.read_csv(SAMPLE_SUB)
    expected_ids = sample["id"].tolist()
    if set(test_ids) != set(expected_ids):
        raise SystemExit("test_ids do not match sample_submission ids")

    if not np.isfinite(pred).all():
        raise SystemExit(f"submission contains non-finite values for run {run_dir}")

    pred_by_id = {tid: pred[i] for i, tid in enumerate(test_ids)}
    rows = [(tid, *pred_by_id[tid]) for tid in expected_ids]
    out = run_dir / "submission.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "z"])
        for tid, x, y, z in rows:
            w.writerow([tid, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    # plan-002 §8.1 schema asserts: rows, columns, NaN/Inf, dtype, id-set match.
    df_out = pd.read_csv(out)
    assert list(df_out.columns) == ["id", "x", "y", "z"], list(df_out.columns)
    assert len(df_out) == len(expected_ids), (len(df_out), len(expected_ids))
    assert df_out["id"].tolist() == expected_ids, "id order mismatch with sample_submission"
    assert df_out[["x", "y", "z"]].notna().all().all(), "NaN found in submission"
    assert np.isfinite(df_out[["x", "y", "z"]].to_numpy()).all(), "Inf found in submission"
    for col in ("x", "y", "z"):
        assert df_out[col].dtype == np.float64, (col, df_out[col].dtype)
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
