"""plan-007 STAGE 2 — CMA-ES baseline fit (6 base motion terms).

Spec: plans/plan-007-formula-tuning.md §5.

Single fit on full train pool (sliding 40K + original 10K = 50K, since G0=PASS) → best_params
for test inference. Separate 5-fold OOF re-fit for G1 metric (oof_hit_5fold).

Outputs:
  - analysis/plan-007/cma_es_step2.json
  - runs/baseline/F001_formula-ga/submission_step2.csv
  - runs/baseline/F001_formula-ga/submission.csv (= submission_step2.csv copy)
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cma
import numpy as np

from src.pb_0_6822 import selector

DATA_ROOT = Path("data")
OUT_DIR = Path("analysis/plan-007")
RUN_DIR = Path("runs/baseline/F001_formula-ga")
EPS = 1e-9
R_HIT = 0.01


# ---------------------------------------------------------------------------
# Motion terms (§5.1)
# ---------------------------------------------------------------------------

def _compute_base_terms(x: np.ndarray, end_idx: int, global_mean_speed: float):
    """Compute (p0, d1, acc_par, acc_perp, d2, jerk, ts_term) for x at given end_idx.

    x: (N, T, 3) — assumes T >= end_idx + 1 and end_idx >= 3.
    Returns 7 arrays each (N, 3) float32.
    """
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = d1 - d2
    d1_norm = np.linalg.norm(d1, axis=1, keepdims=True) + EPS  # (N, 1)
    tangent = d1 / d1_norm
    acc_par = (acc * tangent).sum(axis=1, keepdims=True) * tangent  # (N, 3)
    acc_perp = acc - acc_par
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    jerk = acc - prev_acc
    time_scale_factor = d1_norm / (global_mean_speed + EPS)  # (N, 1)
    ts_term = time_scale_factor * d1  # (N, 3) = ||d1||·d1/global_mean_speed
    return p0, d1, acc_par, acc_perp, d2, jerk, ts_term


def _stack_train_terms(aug_usable: bool, train_x: np.ndarray, train_y: np.ndarray,
                        ids: list[str], global_mean_speed: float):
    """Build the train sample stack per plan §5.2 spec.

    Returns dict with keys: p0, d1, acc_par, acc_perp, d2, jerk, ts_term, target, fold_id.
    All arrays length M = 50K (aug=True) or 10K (False). Same sample_id sliding views
    inherit parent fold (§3.1).
    """
    n_orig = len(ids)
    fold_ids_orig = np.asarray([selector.stable_fold_id(s, 5) for s in ids], dtype=np.int8)

    # original sample (end_idx=10, target = train_y)
    p0_o, d1_o, ap_o, ape_o, d2_o, jk_o, ts_o = _compute_base_terms(train_x, 10, global_mean_speed)
    blocks = [(p0_o, d1_o, ap_o, ape_o, d2_o, jk_o, ts_o, train_y.astype(np.float32), fold_ids_orig)]

    if aug_usable:
        for end_idx in range(5, 9):  # [5,8]
            p0_s, d1_s, ap_s, ape_s, d2_s, jk_s, ts_s = _compute_base_terms(train_x, end_idx, global_mean_speed)
            target_s = train_x[:, end_idx + 2].astype(np.float32)
            blocks.append((p0_s, d1_s, ap_s, ape_s, d2_s, jk_s, ts_s, target_s, fold_ids_orig))

    p0 = np.concatenate([b[0] for b in blocks], axis=0)
    d1 = np.concatenate([b[1] for b in blocks], axis=0)
    acc_par = np.concatenate([b[2] for b in blocks], axis=0)
    acc_perp = np.concatenate([b[3] for b in blocks], axis=0)
    d2 = np.concatenate([b[4] for b in blocks], axis=0)
    jerk = np.concatenate([b[5] for b in blocks], axis=0)
    ts_term = np.concatenate([b[6] for b in blocks], axis=0)
    target = np.concatenate([b[7] for b in blocks], axis=0)
    fold_id = np.concatenate([b[8] for b in blocks], axis=0)
    return dict(p0=p0, d1=d1, acc_par=acc_par, acc_perp=acc_perp, d2=d2, jerk=jerk,
                ts_term=ts_term, target=target, fold_id=fold_id)


# ---------------------------------------------------------------------------
# Fitness + CMA-ES (§5.2)
# ---------------------------------------------------------------------------

def fitness_step2(params, p0, d1, acc_par, acc_perp, d2, jerk, ts_term, target):
    a, b, c, d, e, f = params
    pred = p0 + a * d1 + b * acc_par + c * acc_perp + d * d2 + e * jerk + f * ts_term
    err = np.linalg.norm(pred - target, axis=1)
    return -float((err <= R_HIT).mean())  # CMA-ES minimizes


def _fit_single(stack: dict, seed: int = 20260606, popsize: int = 30, maxiter: int = 200,
                verbose: bool = True):
    x0 = [1.98, 1.20, -0.20, 0.0, 0.0, 0.0]
    sigma0 = 0.3
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "popsize": popsize, "maxiter": maxiter, "tolfun": 1e-5, "seed": seed, "verbose": -9,
    })
    args = (stack["p0"], stack["d1"], stack["acc_par"], stack["acc_perp"],
            stack["d2"], stack["jerk"], stack["ts_term"], stack["target"])
    history = []
    while not es.stop():
        solutions = es.ask()
        fitnesses = [fitness_step2(s, *args) for s in solutions]
        es.tell(solutions, fitnesses)
        gen_best = min(fitnesses)
        history.append(float(gen_best))
        if verbose and len(history) % 25 == 0:
            print(f"  gen {len(history)}: best fitness = {gen_best:.6f} (hit={-gen_best:.4f})")
    best_params = np.asarray(es.result.xbest, dtype=np.float32)
    best_hit = -float(es.result.fbest)
    return best_params, best_hit, history


def _predict(params, stack: dict, idx_slice=None) -> np.ndarray:
    a, b, c, d, e, f = params
    if idx_slice is None:
        idx_slice = slice(None)
    return (stack["p0"][idx_slice]
            + a * stack["d1"][idx_slice]
            + b * stack["acc_par"][idx_slice]
            + c * stack["acc_perp"][idx_slice]
            + d * stack["d2"][idx_slice]
            + e * stack["jerk"][idx_slice]
            + f * stack["ts_term"][idx_slice])


# ---------------------------------------------------------------------------
# 5-fold OOF re-fit (§5.4 G1 metric)
# ---------------------------------------------------------------------------

def run_5fold_oof_cma_es(aug_usable: bool, train_x: np.ndarray, train_y: np.ndarray,
                          ids: list[str], global_mean_speed: float) -> float:
    """For each fold k: train CMA-ES on 4 folds' all samples (aug + original);
    eval on 1 fold's *original end_idx=10* only. Concat 5 fold preds → oof_hit_5fold.
    """
    full_stack = _stack_train_terms(aug_usable, train_x, train_y, ids, global_mean_speed)
    n_orig = len(ids)
    # original-only stack indices (first n_orig rows of full_stack)
    fold_ids_orig = full_stack["fold_id"][:n_orig]
    target_orig = train_y.astype(np.float32)

    oof_pred = np.zeros_like(target_orig)
    for k in range(5):
        is_train = full_stack["fold_id"] != k
        train_substack = {key: arr[is_train] for key, arr in full_stack.items() if key != "fold_id"}
        train_substack["fold_id"] = full_stack["fold_id"][is_train]
        print(f"  fold {k}: train N={len(train_substack['p0']):,}")
        params, hit, _ = _fit_single(train_substack, seed=20260606, popsize=30, maxiter=200, verbose=False)
        print(f"  fold {k}: in-sample hit={hit:.4f}, params={params.tolist()}")

        # eval on this fold's original-end_idx=10 only
        val_mask = fold_ids_orig == k
        val_stack = {
            "p0": full_stack["p0"][:n_orig][val_mask],
            "d1": full_stack["d1"][:n_orig][val_mask],
            "acc_par": full_stack["acc_par"][:n_orig][val_mask],
            "acc_perp": full_stack["acc_perp"][:n_orig][val_mask],
            "d2": full_stack["d2"][:n_orig][val_mask],
            "jerk": full_stack["jerk"][:n_orig][val_mask],
            "ts_term": full_stack["ts_term"][:n_orig][val_mask],
        }
        val_pred = _predict(params, val_stack)
        oof_pred[val_mask] = val_pred
    err = np.linalg.norm(oof_pred - target_orig, axis=1)
    return float((err <= R_HIT).mean())


# ---------------------------------------------------------------------------
# Test inference + submission
# ---------------------------------------------------------------------------

def make_test_submission(best_params: np.ndarray, global_mean_speed: float) -> Path:
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)
    assert test_x.shape[1] == 11, f"expected test T=11, got {test_x.shape}"
    p0, d1, acc_par, acc_perp, d2, jerk, ts_term = _compute_base_terms(test_x, 10, global_mean_speed)
    a, b, c, d, e, f = best_params.tolist()
    pred = p0 + a * d1 + b * acc_par + c * acc_perp + d * d2 + e * jerk + f * ts_term
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sub_step2 = RUN_DIR / "submission_step2.csv"
    with sub_step2.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["id", "x", "y", "z"])
        for tid, (px, py, pz) in zip(test_ids, pred):
            writer.writerow([tid, f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"])
    sub_main = RUN_DIR / "submission.csv"
    sub_main.write_bytes(sub_step2.read_bytes())
    # schema 4-line assert
    with sub_main.open("r") as f_check:
        header = f_check.readline().strip()
        assert header == "id,x,y,z", f"submission header mismatch: {header!r}"
        rows = sum(1 for _ in f_check)
        assert rows == 10000, f"submission row count mismatch: {rows}"
    return sub_main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Read aug_usable from STAGE 1 result
    sliding = json.loads((OUT_DIR / "sliding_validity.json").read_text())
    aug_usable = sliding["aug_usable"]
    print(f"aug_usable = {aug_usable}")

    # Load train
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)

    # global_mean_speed (original 10K only, end_idx=10)
    d1_orig = train_x[:, 10] - train_x[:, 9]
    global_mean_speed = float(np.linalg.norm(d1_orig, axis=1).mean())
    print(f"global_mean_speed = {global_mean_speed:.6f}")

    # Single fit on full pool
    t0 = time.time()
    stack = _stack_train_terms(aug_usable, train_x, train_y, ids, global_mean_speed)
    print(f"train pool size M = {len(stack['p0']):,}")
    print("Single fit CMA-ES (popsize=30, maxiter=200, seed=20260606)...")
    best_params, single_fit_best_hit, history = _fit_single(stack, verbose=True)
    elapsed_single = time.time() - t0
    print(f"single_fit_best_hit = {single_fit_best_hit:.4f}")
    print(f"best_params = {best_params.tolist()}")

    # Convergence diagnostic: last 50 generations fbest variation
    last_50 = history[-50:] if len(history) >= 50 else history
    convergence_range = float(max(last_50) - min(last_50))
    print(f"convergence (last 50 gen fbest range) = {convergence_range:.6f}")

    # 5-fold OOF
    print("Running 5-fold OOF CMA-ES...")
    t_oof = time.time()
    oof_hit_5fold = run_5fold_oof_cma_es(aug_usable, train_x, train_y, ids, global_mean_speed)
    elapsed_oof = time.time() - t_oof
    print(f"oof_hit_5fold = {oof_hit_5fold:.4f}")

    # Save JSON
    result = {
        "best_params": [float(p) for p in best_params],
        "single_fit_best_hit": single_fit_best_hit,
        "oof_hit_5fold": oof_hit_5fold,
        "convergence_history": history,
        "convergence_last_50_range": convergence_range,
        "aug_usable": aug_usable,
        "n_train_pool": int(len(stack["p0"])),
        "global_mean_speed": global_mean_speed,
        "elapsed_single_sec": elapsed_single,
        "elapsed_oof_sec": elapsed_oof,
    }
    (OUT_DIR / "cma_es_step2.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Test submission
    sub_path = make_test_submission(best_params, global_mean_speed)
    print(f"submission written: {sub_path}")
    print(f"\n=== G1 check ===")
    print(f"oof_hit_5fold = {oof_hit_5fold:.4f} (target: 0.62 ≤ x ≤ 0.78)")
    print(f"convergence_range = {convergence_range:.6f} (target: < 0.005)")
    g1_pass = (0.62 <= oof_hit_5fold <= 0.78) and (convergence_range < 0.005)
    print(f"G1 pass = {g1_pass}")


if __name__ == "__main__":
    main()
