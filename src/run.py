"""Baseline experiment runner.

Reads a YAML config, runs 5-fold CV (or per-config k), writes:
  runs/{type}/{exp_id}/summary.json
  runs/{type}/{exp_id}/history.json (per-fold metrics)
  runs/{type}/{exp_id}/run.log
  runs/{type}/{exp_id}/config.snapshot.yaml
and appends one row to registry.csv.

CLI: python -m src.run configs/baseline/B001_linear-2pt.yaml

Per plan-002 §4.4: cfg["method"] ∈ {"polyfit"(default), "cspline", "smoothing_spline"}
dispatches to the corresponding baseline implementation. polyfit branch is
backward-compatible (B001~B004 configs lack the `method` key and default to polyfit).

Per plan-003 §4.5: cfg["method"] == "gru-residual" → 신규 helper
`_train_and_predict_residual_fold` 가 fold 내부에서 train → ckpt 저장 → predict.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.baselines.cubic_spline import (
    predict_cspline,
    predict_cspline_per_axis,
    predict_smoothing_spline,
    tune_per_axis_cspline,
    tune_per_axis_smoothing,
)
from src.baselines.linear_extrapolate import ema_extrapolate, linear_extrap
from src.baselines.window_polyfit import predict, predict_per_axis, tune_per_axis
from src.eval import summarize
from src.io import TIMESTEPS_MS, kfold_split, load_all_samples, load_labels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
REGISTRY = PROJECT_ROOT / "registry.csv"

REGISTRY_COLS = [
    "id", "plan_id", "type", "status", "started_at", "finished_at",
    "duration_sec", "run_dir", "config_path", "baseline_id", "corrects", "notes",
]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")


def predict_for_config(
    X: np.ndarray, cfg: dict, info_out: dict | None = None
) -> np.ndarray:
    method = cfg.get("method", "polyfit")
    t_target = int(cfg.get("t_target", 80))

    if method == "polyfit":
        pa = cfg.get("per_axis")
        if pa and pa != "tune":
            configs = [tuple(c) for c in pa]
            return predict_per_axis(X, configs, t_target=t_target, timesteps=TIMESTEPS_MS)
        return predict(
            X, int(cfg["window"]), int(cfg["degree"]),
            t_target=t_target, timesteps=TIMESTEPS_MS,
        )

    if method == "cspline":
        pa = cfg.get("per_axis")
        if pa and pa != "tune":
            configs = [(int(c[0]), str(c[1])) for c in pa]
            return predict_cspline_per_axis(
                X, configs, t_target=t_target, timesteps=TIMESTEPS_MS
            )
        return predict_cspline(
            X, int(cfg["window"]), str(cfg["bc_type"]),
            t_target=t_target, timesteps=TIMESTEPS_MS,
        )

    if method == "smoothing_spline":
        sp = cfg.get("s_per_axis")
        if sp and sp != "tune":
            return predict_smoothing_spline(
                X, [float(s) for s in sp], t_target=t_target,
                timesteps=TIMESTEPS_MS, s_grid=cfg.get("s_grid"),
                info_out=info_out,
            )
        raise ValueError(
            "smoothing_spline non-tune mode requires s_per_axis: list[float]"
        )

    raise ValueError(f"unknown method {method!r}")


def append_registry(row: dict) -> None:
    file_exists = REGISTRY.exists() and REGISTRY.stat().st_size > 0
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_COLS})


def _is_tune(cfg: dict) -> tuple[bool, str]:
    """Return (is_tune, kind) where kind ∈ {"polyfit", "cspline", "smoothing_spline", ""}."""
    method = cfg.get("method", "polyfit")
    if method in ("polyfit", "cspline") and cfg.get("per_axis") == "tune":
        return True, method
    if method == "smoothing_spline" and cfg.get("s_per_axis") == "tune":
        return True, method
    return False, ""


def _tune_grid(cfg: dict, kind: str) -> list:
    if kind == "polyfit":
        return [tuple(c) for c in cfg.get("grid", [])]
    if kind == "cspline":
        return [(int(c[0]), str(c[1])) for c in cfg.get("grid", [])]
    if kind == "smoothing_spline":
        return [float(s) for s in cfg.get("s_grid", [])]
    raise ValueError(f"_tune_grid: unknown kind {kind!r}")


def _do_tune_and_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray,
    grid: list, kind: str, t_target: int, k: int, seed: int,
    info_out: dict | None = None,
):
    """One outer-fold tune-then-predict. Returns (chosen, pred)."""
    if kind == "polyfit":
        chosen, _err = tune_per_axis(X_tr, y_tr, grid, t_target=t_target,
                                     timesteps=TIMESTEPS_MS)
        pred = predict_per_axis(X_va, chosen, t_target=t_target, timesteps=TIMESTEPS_MS)
        return chosen, pred

    if kind == "cspline":
        chosen, _err = tune_per_axis_cspline(
            X_tr, y_tr, grid, t_target=t_target, k=k, seed=seed, timesteps=TIMESTEPS_MS
        )
        pred = predict_cspline_per_axis(X_va, chosen, t_target=t_target, timesteps=TIMESTEPS_MS)
        return chosen, pred

    if kind == "smoothing_spline":
        chosen, _err = tune_per_axis_smoothing(
            X_tr, y_tr, grid, t_target=t_target, n_folds=k, seed=seed, timesteps=TIMESTEPS_MS
        )
        pred = predict_smoothing_spline(
            X_va, chosen, t_target=t_target, timesteps=TIMESTEPS_MS,
            s_grid=grid, info_out=info_out,
        )
        return chosen, pred

    raise ValueError(f"_do_tune_and_predict: unknown kind {kind!r}")


def _accumulate_fb(total: dict, partial: dict | None) -> None:
    if not partial:
        return
    sub = partial.get("smoothing_fallback_count")
    if not sub:
        return
    for key, val in sub.items():
        total[key] = total.get(key, 0) + int(val)


def _baseline_fn_from_cfg(cfg: dict):
    """Return numpy callable: X (n,T,3) → pred (n,3) for residual baseline."""
    bt = cfg.get("baseline_type", "linear")
    t_target = int(cfg.get("t_target", 80))
    if bt == "linear":
        def fn(X: np.ndarray) -> np.ndarray:
            return linear_extrap(X, t_target_ms=t_target, timesteps_ms=TIMESTEPS_MS)
        return fn
    if bt == "ema":
        alpha = float(cfg.get("ema_alpha", 0.5))
        def fn(X: np.ndarray) -> np.ndarray:
            return ema_extrapolate(X, alpha=alpha, t_target_ms=t_target, timesteps_ms=TIMESTEPS_MS)
        return fn
    raise ValueError(f"unknown baseline_type {bt!r}")


def _train_and_predict_residual_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: dict,
    fold_idx: int,
    run_dir: Path,
) -> tuple[Path, np.ndarray, dict]:
    """Train one fold of residual-GRU and emit (ckpt_path, val_pred, fold_info).

    Per plan-003 §4.5.
    """
    # local import to avoid torch dependency on closed-form-only paths
    import torch

    from src.models.residual_gru import ResidualGRU
    from src.training.train_residual import make_feature_fn, train_fold

    feature_components = list(cfg.get("feature_components", ["relative"]))
    wingbeat_n_bins = int(cfg.get("wingbeat_n_bins", 3))
    feature_fn = make_feature_fn(feature_components, wingbeat_n_bins=wingbeat_n_bins)

    baseline_fn = _baseline_fn_from_cfg(cfg)
    baseline_train = baseline_fn(X_tr)
    baseline_val = baseline_fn(X_va)

    feat_dim = feature_fn(X_tr[:1]).shape[-1]
    expected_dim = int(cfg["model"]["input_dim"])
    if feat_dim != expected_dim:
        raise ValueError(
            f"feature_fn output dim {feat_dim} != cfg model.input_dim {expected_dim} "
            f"for components {feature_components}"
        )

    model = ResidualGRU(
        input_dim=feat_dim,
        hidden=int(cfg["model"]["hidden"]),
        layers=int(cfg["model"]["layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    base_seed = int(cfg.get("training", {}).get("seed", cfg.get("seed", 42)))
    fold_seed = base_seed + fold_idx
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs_cfg = int(cfg["training"].get("epochs", 100))
    if device == "cpu" and epochs_cfg > 50:
        epochs_cfg = 50  # decision-note: CPU fallback → epochs 50

    info = train_fold(
        model,
        X_tr, y_tr, X_va, y_va,
        baseline_train, baseline_val,
        feature_fn,
        loss_type=str(cfg.get("loss_type", "huber")),
        lr=float(cfg["training"].get("lr", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
        batch=int(cfg["training"].get("batch", 64)),
        epochs=epochs_cfg,
        early_stop_patience=int(cfg["training"].get("early_stop_patience", 10)),
        device=device,
        seed=fold_seed,
    )

    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold_idx}.pt"
    torch.save(info["best_state_dict"], ckpt_path)

    # predict on val (best weights already loaded inside train_fold)
    model.eval()
    with torch.no_grad():
        feat_va = feature_fn(X_va).astype(np.float32, copy=False)
        delta = model(torch.from_numpy(feat_va).to(device)).cpu().numpy()
    pred = baseline_val + delta

    fold_info = {
        "best_val_mean_eucl": float(info["best_val_mean_eucl"]),
        "best_epoch": int(info["best_epoch"]),
        "n_epochs_run": int(info["n_epochs_run"]),
        "train_duration_sec": float(info["train_duration_sec"]),
        "ckpt_path": str(ckpt_path.relative_to(run_dir.parent.parent.parent))
            if ckpt_path.is_absolute() else str(ckpt_path),
        "device": device,
    }
    return ckpt_path, pred, fold_info


def run_baseline(config_path: Path, X=None, y=None, ids=None) -> dict:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    exp_id = cfg["exp_id"]
    run_dir = RUNS_ROOT / cfg["type"] / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{now_kst()}] {msg}"
        log_lines.append(line)
        print(line, flush=True)

    started = now_kst()
    t0 = time.monotonic()
    log(f"start exp_id={exp_id}, config={config_path}")

    if X is None or y is None or ids is None:
        log("load train + labels")
        ids_X, X = load_all_samples("train")
        ids_y, y = load_labels()
        assert ids_X == ids_y, "id order mismatch between samples and labels"
        ids = ids_X
    log(f"n_train={len(ids)}")

    k = int(cfg.get("k", 5))
    seed = int(cfg.get("seed", 42))
    folds = kfold_split(ids, k=k, seed=seed)
    is_tune, tune_kind = _is_tune(cfg)
    grid = _tune_grid(cfg, tune_kind) if is_tune else None
    t_target = int(cfg.get("t_target", 80))
    method = cfg.get("method", "polyfit")

    fold_metrics: list[dict] = []
    fold_chosen: list = []
    fold_train_infos: list[dict] = []  # gru-residual 전용
    oof_preds = np.empty_like(y)
    fb_total: dict = {"step1_s_retry": 0, "step2_cubicspline": 0, "step3_last_input": 0}

    for fi, (tr, va) in enumerate(folds):
        if method == "gru-residual":
            ckpt_path, pred, fold_info = _train_and_predict_residual_fold(
                X[tr], y[tr], X[va], y[va], cfg, fi, run_dir,
            )
            fold_train_infos.append(fold_info)
            log(
                f"fold {fi}: gru-residual best_val_mean_eucl={fold_info['best_val_mean_eucl']:.5f} "
                f"@epoch {fold_info['best_epoch']}, ckpt={ckpt_path.name}"
            )
        elif is_tune:
            info_partial: dict = {}
            chosen, pred = _do_tune_and_predict(
                X[tr], y[tr], X[va], grid, tune_kind,
                t_target=t_target, k=k, seed=seed, info_out=info_partial,
            )
            fold_chosen.append(chosen)
            _accumulate_fb(fb_total, info_partial)
            log(f"fold {fi}: chosen per-axis={chosen}")
        else:
            info_partial = {}
            pred = predict_for_config(X[va], cfg, info_out=info_partial)
            _accumulate_fb(fb_total, info_partial)
        oof_preds[va] = pred
        s = summarize(pred, y[va])
        s["fold"] = fi
        if method != "gru-residual" and is_tune:
            s["chosen_per_axis"] = (
                [list(c) if not isinstance(c, (int, float)) else c for c in chosen]
            )
        fold_metrics.append(s)
        log(f"fold {fi}: mean_eucl={s['mean_eucl']:.5f} "
            f"per_axis_mae={[round(v, 4) for v in s['per_axis_mae']]}")

    arr_mean = np.array([f["mean_eucl"] for f in fold_metrics])
    cv_mean = float(arr_mean.mean())
    cv_std = float(arr_mean.std(ddof=0))
    cv_per_axis = np.mean([f["per_axis_mae"] for f in fold_metrics], axis=0).tolist()
    radii_keys = list(fold_metrics[0]["hit_rate"].keys())
    cv_hit = {kr: float(np.mean([f["hit_rate"][kr] for f in fold_metrics]))
              for kr in radii_keys}
    oof_summary = summarize(oof_preds, y)
    if method == "smoothing_spline":
        oof_summary["smoothing_fallback_count"] = fb_total
        oof_summary["n_oof_samples"] = int(y.shape[0])
    log(f"CV mean_eucl={cv_mean:.5f} ± {cv_std:.5f} | "
        f"OOF mean_eucl={oof_summary['mean_eucl']:.5f}")

    duration = round(time.monotonic() - t0, 3)
    finished = now_kst()

    summary = {
        "exp_id": exp_id,
        "type": cfg["type"],
        "method": method,
        "plan_id": str(cfg.get("plan_id", "001")),
        "started_at": started,
        "finished_at": finished,
        "duration_sec": duration,
        "n_train": len(ids),
        "k": k,
        "cv_mean_eucl": cv_mean,
        "cv_std_eucl": cv_std,
        "cv_per_axis_mae": cv_per_axis,
        "cv_hit_rate": cv_hit,
        "oof_summary": oof_summary,
        "fold_metrics": fold_metrics,
        "config": cfg,
    }

    if method == "gru-residual":
        # gru-residual 전용 summary 보강
        feature_components = list(cfg.get("feature_components", ["relative"]))
        baseline_type = str(cfg.get("baseline_type", "linear"))
        train_devices = sorted({fi["device"] for fi in fold_train_infos})
        train_device = train_devices[0] if len(train_devices) == 1 else "mixed"
        summary["model_config"] = {
            "hidden": int(cfg["model"]["hidden"]),
            "layers": int(cfg["model"]["layers"]),
            "dropout": float(cfg["model"]["dropout"]),
            "input_dim": int(cfg["model"]["input_dim"]),
            "lr": float(cfg["training"].get("lr", 1e-3)),
            "weight_decay": float(cfg["training"].get("weight_decay", 1e-4)),
            "batch": int(cfg["training"].get("batch", 64)),
            "epochs": int(cfg["training"].get("epochs", 100)),
            "early_stop_patience": int(cfg["training"].get("early_stop_patience", 10)),
            "loss_type": str(cfg.get("loss_type", "huber")),
        }
        summary["feature_components"] = feature_components
        summary["baseline_type"] = baseline_type
        summary["ema_alpha"] = float(cfg.get("ema_alpha", 0.5)) if baseline_type == "ema" else None
        summary["wingbeat_n_bins"] = (
            int(cfg.get("wingbeat_n_bins", 3)) if "wingbeat" in feature_components else None
        )
        summary["fold_best_val_mean_eucl"] = [fi["best_val_mean_eucl"] for fi in fold_train_infos]
        summary["fold_best_epoch"] = [fi["best_epoch"] for fi in fold_train_infos]
        summary["fold_train_duration_sec"] = [fi["train_duration_sec"] for fi in fold_train_infos]
        summary["train_device"] = train_device
        summary["total_train_duration_sec"] = float(
            sum(fi["train_duration_sec"] for fi in fold_train_infos)
        )

    if is_tune:
        if tune_kind == "polyfit":
            final_chosen, final_errors = tune_per_axis(
                X, y, grid, t_target=t_target, timesteps=TIMESTEPS_MS
            )
            summary["final_chosen_per_axis"] = [list(c) for c in final_chosen]
            summary["fold_chosen_per_axis"] = [
                [list(c) for c in cfgs] for cfgs in fold_chosen
            ]
            summary["full_train_grid_errors"] = {
                str(axis): {f"w{w}d{d}": e for (w, d), e in errs.items()}
                for axis, errs in final_errors.items()
            }
            log(f"final tune: chosen per-axis (w,d)={final_chosen}")
        elif tune_kind == "cspline":
            final_chosen, final_errors = tune_per_axis_cspline(
                X, y, grid, t_target=t_target, k=k, seed=seed, timesteps=TIMESTEPS_MS
            )
            summary["final_chosen_per_axis"] = [
                [int(c[0]), str(c[1])] for c in final_chosen
            ]
            summary["fold_chosen_per_axis"] = [
                [[int(c[0]), str(c[1])] for c in cfgs] for cfgs in fold_chosen
            ]
            summary["full_train_grid_errors"] = {
                f"w{w}_{bc}": err.tolist() for (w, bc), err in final_errors.items()
            }
            log(f"final tune: chosen per-axis (w,bc)={final_chosen}")
        elif tune_kind == "smoothing_spline":
            final_chosen, final_errors = tune_per_axis_smoothing(
                X, y, grid, t_target=t_target, n_folds=k, seed=seed,
                timesteps=TIMESTEPS_MS,
            )
            summary["final_chosen_s_per_axis"] = [float(s) for s in final_chosen]
            summary["fold_chosen_s_per_axis"] = [
                [float(s) for s in cfgs] for cfgs in fold_chosen
            ]
            summary["full_train_grid_errors"] = {
                f"s={s}": err.tolist() for s, err in final_errors.items()
            }
            log(f"final tune: chosen s per-axis={final_chosen}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "history.json").write_text(json.dumps(fold_metrics, indent=2))
    (run_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n")

    append_registry({
        "id": exp_id,
        "plan_id": str(cfg.get("plan_id", "001")),
        "type": cfg["type"],
        "status": "complete",
        "started_at": started,
        "finished_at": finished,
        "duration_sec": duration,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "config_path": str(Path(config_path).relative_to(PROJECT_ROOT)),
        "baseline_id": cfg.get("baseline_id", "") or "",
        "corrects": "",
        "notes": f"cv_mean_eucl={cv_mean:.5f}±{cv_std:.5f}",
    })
    log(f"DONE in {duration}s, run_dir={run_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    args = p.parse_args(argv)
    run_baseline(args.config)


if __name__ == "__main__":
    main()
