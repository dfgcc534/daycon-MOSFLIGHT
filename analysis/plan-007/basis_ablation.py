"""plan-007 STAGE 3 — new variable ablation (4 variables × cumulative add).

Spec: plans/plan-007-formula-tuning.md §6.

Cumulative addition of 4 new variables (speed_slope_d1, rotation_term, speed_norm_acc_par,
v_mean3_minus_d1) on top of Step 2's 6 base vars. Each step: CMA-ES fit, measure marginal_gain.
kept = marginal_gain >= 0.001 (inclusive). Final best basis = kept new vars + base.

Outputs:
  - analysis/plan-007/basis_ablation.{json,md}
  - runs/baseline/F001_formula-ga/submission_step3.csv (+ submission.csv replaced)
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

BASE_VARS = ["d1", "acc_par", "acc_perp", "d2", "jerk", "ts_term"]
NEW_VARS = ["speed_slope_d1", "rotation_term", "speed_norm_acc_par", "v_mean3_minus_d1"]


# ---------------------------------------------------------------------------
# Basis terms (full set — 10 vars: 6 base + 4 new)
# ---------------------------------------------------------------------------

def compute_all_terms(x: np.ndarray, end_idx: int, horizon: int,
                       global_mean_speed: float) -> dict[str, np.ndarray]:
    """Compute (p0, target-agnostic 10 basis terms) for x at given end_idx.

    Returns dict with keys:
      base: d1, acc_par, acc_perp, d2, jerk, ts_term  (each (N, 3))
      new:  speed_slope_d1, rotation_term, speed_norm_acc_par, v_mean3_minus_d1
      meta: p0
    """
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = d1 - d2
    d1_norm = np.linalg.norm(d1, axis=1, keepdims=True) + EPS
    tangent = d1 / d1_norm
    acc_par = (acc * tangent).sum(axis=1, keepdims=True) * tangent
    acc_perp = acc - acc_par
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    jerk = acc - prev_acc
    time_scale_factor = d1_norm / (global_mean_speed + EPS)
    ts_term = time_scale_factor * d1

    # New variables (§6.1)
    # mean_speed = mean of ||x[t+1]-x[t]|| for t ∈ [end-4, end-1] = 4 step deltas
    steps_for_mean = np.stack([
        x[:, end_idx - 3] - x[:, end_idx - 4],
        x[:, end_idx - 2] - x[:, end_idx - 3],
        x[:, end_idx - 1] - x[:, end_idx - 2],
        x[:, end_idx]     - x[:, end_idx - 1],
    ], axis=1)  # (N, 4, 3)
    step_norms = np.linalg.norm(steps_for_mean, axis=2)  # (N, 4)
    mean_speed = step_norms.mean(axis=1, keepdims=True) + EPS  # (N, 1)
    # speed_slope = (||x[end]-x[end-1]|| - ||x[end-4]-x[end-5]||) / mean_speed
    older_step = x[:, end_idx - 4] - x[:, end_idx - 5]
    older_step_norm = np.linalg.norm(older_step, axis=1, keepdims=True)
    speed_slope_scalar = (d1_norm - older_step_norm) / mean_speed  # (N, 1)
    speed_slope_d1 = speed_slope_scalar * d1  # (N, 3)

    # rotation_term: omega = atan2(np.cross(d2, d1)[2], np.dot(d2, d1)); apply 2D rotation
    cross_z = d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0]   # (N,)
    dot_xy = (d2 * d1).sum(axis=1)                          # (N,)
    omega = np.arctan2(cross_z, dot_xy)                     # (N,)
    theta = omega * horizon                                  # rotate by omega·horizon (horizon=2)
    cos_t, sin_t = np.cos(theta), np.sin(theta)              # (N,)
    rot_x = cos_t * d1[:, 0] - sin_t * d1[:, 1]
    rot_y = sin_t * d1[:, 0] + cos_t * d1[:, 1]
    rot_z = d1[:, 2]
    rot_d1 = np.stack([rot_x, rot_y, rot_z], axis=1)         # (N, 3)
    rotation_term = rot_d1 - d1

    # speed_norm·acc_par
    speed_norm_scalar = d1_norm / mean_speed                 # (N, 1)
    speed_norm_acc_par = speed_norm_scalar * acc_par         # (N, 3)

    # v_mean3 - d1
    v_mean3 = (x[:, end_idx] - x[:, end_idx - 3]) / 3.0
    v_mean3_minus_d1 = v_mean3 - d1

    return {
        "p0": p0,
        "d1": d1, "acc_par": acc_par, "acc_perp": acc_perp,
        "d2": d2, "jerk": jerk, "ts_term": ts_term,
        "speed_slope_d1": speed_slope_d1,
        "rotation_term": rotation_term,
        "speed_norm_acc_par": speed_norm_acc_par,
        "v_mean3_minus_d1": v_mean3_minus_d1,
    }


def stack_train_full(aug_usable: bool, train_x: np.ndarray, train_y: np.ndarray,
                      ids: list[str], global_mean_speed: float) -> dict:
    """Build 50K (aug) or 10K (no-aug) stack with all 10 basis terms + target + fold_id."""
    fold_orig = np.asarray([selector.stable_fold_id(s, 5) for s in ids], dtype=np.int8)

    blocks = []
    # original (end_idx=10, target = train_y)
    terms_o = compute_all_terms(train_x, 10, horizon=2, global_mean_speed=global_mean_speed)
    blocks.append((terms_o, train_y.astype(np.float32), fold_orig))
    if aug_usable:
        for end_idx in range(5, 9):
            terms_s = compute_all_terms(train_x, end_idx, horizon=2, global_mean_speed=global_mean_speed)
            target_s = train_x[:, end_idx + 2].astype(np.float32)
            blocks.append((terms_s, target_s, fold_orig))

    keys = ["p0"] + BASE_VARS + NEW_VARS
    stack = {}
    for k in keys:
        stack[k] = np.concatenate([b[0][k] for b in blocks], axis=0)
    stack["target"] = np.concatenate([b[1] for b in blocks], axis=0)
    stack["fold_id"] = np.concatenate([b[2] for b in blocks], axis=0)
    return stack


# ---------------------------------------------------------------------------
# CMA-ES fit helper (variable basis)
# ---------------------------------------------------------------------------

def _fitness_subset(params: np.ndarray, stack: dict, var_names: list[str], idx=None) -> float:
    if idx is None:
        idx = slice(None)
    pred = stack["p0"][idx].copy()
    for coeff, var in zip(params, var_names):
        pred = pred + coeff * stack[var][idx]
    err = np.linalg.norm(pred - stack["target"][idx], axis=1)
    return -float((err <= R_HIT).mean())


def cma_es_fit(stack: dict, var_names: list[str], seed: int = 20260606,
               popsize: int = 30, maxiter: int = 200, verbose: bool = False) -> tuple[np.ndarray, float, list]:
    """Single fit on full stack. x0 seed (plan §6.2): d1/acc_par/acc_perp 의 첫 3 entry 는
    plan-006 best init (1.98, 1.20, -0.20) — base_vars ordering 기준. 그 외 0.0.

    var_names 가 base_vars 순으로 시작한다고 가정 (실제 호출에서 항상 base 6 + new vars append).
    """
    x0 = []
    init_map = {"d1": 1.98, "acc_par": 1.20, "acc_perp": -0.20}
    for v in var_names:
        x0.append(init_map.get(v, 0.0))
    sigma0 = 0.3
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "popsize": popsize, "maxiter": maxiter, "tolfun": 1e-5, "seed": seed, "verbose": -9,
    })
    history = []
    while not es.stop():
        sols = es.ask()
        fits = [_fitness_subset(np.asarray(s, dtype=np.float32), stack, var_names) for s in sols]
        es.tell(sols, fits)
        history.append(float(min(fits)))
        if verbose and len(history) % 25 == 0:
            print(f"    gen {len(history)}: best fitness = {history[-1]:.6f}")
    return np.asarray(es.result.xbest, dtype=np.float32), -float(es.result.fbest), history


# ---------------------------------------------------------------------------
# Ablation loop (§6.2)
# ---------------------------------------------------------------------------

def stage3_ablation(stack: dict, stage2_best_hit: float) -> dict:
    results = []
    current_vars = list(BASE_VARS)
    prev_hit = stage2_best_hit
    t0 = time.time()
    for new_var in NEW_VARS:
        current_vars.append(new_var)
        print(f"  ablation: + {new_var}  (current size = {len(current_vars)})")
        best_params, best_hit, _ = cma_es_fit(stack, current_vars)
        marginal_gain = best_hit - prev_hit
        kept = marginal_gain >= 0.001
        print(f"    hit = {best_hit:.4f}  marginal_gain = {marginal_gain:+.4f}  kept = {kept}")
        results.append({
            "added_var": new_var,
            "current_vars": list(current_vars),
            "best_params": [float(p) for p in best_params],
            "best_hit": float(best_hit),
            "marginal_gain": float(marginal_gain),
            "kept": kept,
        })
        if not kept:
            current_vars.pop()
        else:
            prev_hit = best_hit
    print("  → final best basis fit (sanity recompute):")
    best_basis_params, best_basis_hit, _ = cma_es_fit(stack, current_vars)
    elapsed = time.time() - t0
    return {
        "ablation_steps": results,
        "best_basis_vars": current_vars,
        "best_basis_params": [float(p) for p in best_basis_params],
        "best_basis_hit": float(best_basis_hit),
        "stage2_best_hit_baseline": float(stage2_best_hit),
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Test submission with best basis
# ---------------------------------------------------------------------------

def make_test_submission(best_basis_vars: list[str], best_basis_params: list[float],
                          global_mean_speed: float) -> Path:
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)
    terms = compute_all_terms(test_x, end_idx=10, horizon=2, global_mean_speed=global_mean_speed)
    pred = terms["p0"].copy()
    for coeff, var in zip(best_basis_params, best_basis_vars):
        pred = pred + coeff * terms[var]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sub_step3 = RUN_DIR / "submission_step3.csv"
    with sub_step3.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["id", "x", "y", "z"])
        for tid, (px, py, pz) in zip(test_ids, pred):
            writer.writerow([tid, f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"])
    sub_main = RUN_DIR / "submission.csv"
    sub_main.write_bytes(sub_step3.read_bytes())
    with sub_main.open("r") as f_check:
        header = f_check.readline().strip()
        assert header == "id,x,y,z", f"submission header mismatch: {header!r}"
        rows = sum(1 for _ in f_check)
        assert rows == 10000, f"submission row count mismatch: {rows}"
    return sub_main


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_markdown(res: dict) -> str:
    lines = [
        "# plan-007 STAGE 3 — basis ablation",
        "",
        f"- baseline (Step 2 single fit) hit = **{res['stage2_best_hit_baseline']:.4f}**",
        f"- final best basis hit = **{res['best_basis_hit']:.4f}** (Δ = {res['best_basis_hit'] - res['stage2_best_hit_baseline']:+.4f})",
        f"- best basis variables = `{res['best_basis_vars']}` (size = {len(res['best_basis_vars'])})",
        f"- elapsed = {res['elapsed_sec']:.1f}s",
        "",
        "## Ablation steps",
        "",
        "| step | added | best_hit | marginal_gain | kept? |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(res["ablation_steps"], start=1):
        keep = "✓ kept" if s["kept"] else "✗ dropped"
        lines.append(
            f"| {i} | `{s['added_var']}` | {s['best_hit']:.4f} | {s['marginal_gain']:+.4f} | {keep} |"
        )
    lines.append("")
    lines.append("## Best basis coefficients")
    lines.append("")
    lines.append("| var | coeff |")
    lines.append("|---|---|")
    for v, c in zip(res["best_basis_vars"], res["best_basis_params"]):
        lines.append(f"| `{v}` | {c:+.4f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sliding = json.loads((OUT_DIR / "sliding_validity.json").read_text())
    aug_usable = sliding["aug_usable"]
    stage2 = json.loads((OUT_DIR / "cma_es_step2.json").read_text())
    stage2_best_hit = float(stage2["single_fit_best_hit"])
    global_mean_speed = float(stage2["global_mean_speed"])
    print(f"aug_usable = {aug_usable}, stage2 single_fit_best_hit = {stage2_best_hit:.4f}, gms = {global_mean_speed:.6f}")

    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)

    print("Building 50K stack with 10 terms (6 base + 4 new)...")
    stack = stack_train_full(aug_usable, train_x, train_y, ids, global_mean_speed)
    print(f"pool M = {len(stack['p0']):,}")

    print("Running ablation (4 vars × cumulative)...")
    res = stage3_ablation(stack, stage2_best_hit)
    print(f"final best_basis_hit = {res['best_basis_hit']:.4f}, vars = {res['best_basis_vars']}")

    (OUT_DIR / "basis_ablation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (OUT_DIR / "basis_ablation.md").write_text(render_markdown(res))

    # Test submission with final best basis
    sub = make_test_submission(res["best_basis_vars"], res["best_basis_params"], global_mean_speed)
    print(f"submission written: {sub}")

    g2_pass = res["best_basis_hit"] >= stage2_best_hit
    print(f"\n=== G2 check ===")
    print(f"best_basis_hit ({res['best_basis_hit']:.4f}) ≥ stage2_best_hit ({stage2_best_hit:.4f})? {g2_pass}")


if __name__ == "__main__":
    main()
