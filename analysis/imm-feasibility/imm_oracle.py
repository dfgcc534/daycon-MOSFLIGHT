"""IMM feasibility — c4 (Stage 4): IMM oracle ceiling = per-sample best-mode prediction hit@1cm.

per-mode forward predict t=+80ms 와 ground truth Y 비교:
  - pred_err_per_mode[i, m] = ‖pred_t80_m − Y[i]‖₂
  - single_mode_hit_1cm[m] = (10K sample 모두 m 모드 사용 시 hit rate)
  - oracle_imm_hit_1cm = per-sample best-pred-mode 선택 시 hit rate
  - ΔLB upper bound = oracle − max(single_mode)

산식: `/home/ahn/.claude/plans/glimmering-popping-river.md` §Approach IMM oracle ceiling.

Outputs (analysis/imm-feasibility/):
  - pred_err_per_mode.npy (10000, 3) float64
  - best_pred_mode.npy (10000,) int8
  - imm_oracle.json (single-mode hits, oracle hit, ΔLB upper bound, comparisons)
  - imm_oracle.md (human-readable)

Usage:
    python analysis/imm-feasibility/imm_oracle.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402


MODE_NAMES = ["CV", "CA", "CT"]
HIT_THRESHOLD = 0.01   # 1cm

BASELINE_LB_PLAN014_015 = 0.6628
BASELINE_LB_PLAN016_G1 = 0.6638


def quantiles(arr: np.ndarray, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict[str, float]:
    return {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("analysis/imm-feasibility"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/imm-feasibility"))
    args = ap.parse_args()

    t_start = time.time()

    pred_t80_path = args.in_dir / "pred_t80_per_mode.npy"
    if not pred_t80_path.exists():
        print(f"ERROR: {pred_t80_path} missing — run mode_distribution.py first.", file=sys.stderr)
        return 1

    print("[imm-oracle c4] loading artifacts + labels ...", flush=True)
    pred_t80 = np.load(pred_t80_path)   # (N, 3, 3)
    labels = np.load(args.in_dir / "mode_labels.npy")   # (N,)
    ids_X, X = load_all_samples("train")
    ids_Y, Y = load_labels()
    assert ids_X == ids_Y, "id alignment mismatch between X and Y"
    Y = Y.astype(np.float64)

    N = Y.shape[0]
    assert pred_t80.shape == (N, 3, 3), f"pred_t80 shape mismatch: {pred_t80.shape}"
    print(f"[imm-oracle c4] N={N}", flush=True)

    # ── per-mode forward prediction error at t=+80ms ────────────────────
    pred_err = np.linalg.norm(pred_t80 - Y[:, None, :], axis=-1)   # (N, 3)
    hit_per_mode = (pred_err <= HIT_THRESHOLD).astype(np.float64)   # (N, 3)

    single_mode_hit_1cm = {
        MODE_NAMES[m]: float(hit_per_mode[:, m].mean())
        for m in range(3)
    }

    pred_err_quantiles = {
        MODE_NAMES[m]: quantiles(pred_err[:, m]) for m in range(3)
    }

    # ── IMM oracle: per-sample best-pred mode ───────────────────────────
    best_pred_mode = np.argmin(pred_err, axis=1).astype(np.int8)
    oracle_err = pred_err[np.arange(N), best_pred_mode]
    oracle_hit_1cm = float((oracle_err <= HIT_THRESHOLD).mean())

    # ── BIC label vs best-pred-mode agreement ───────────────────────────
    bic_vs_pred_agreement = float((labels == best_pred_mode).mean())
    bic_vs_pred_confusion = np.zeros((3, 3), dtype=np.int64)
    for i in range(N):
        bic_vs_pred_confusion[labels[i], best_pred_mode[i]] += 1
    # rows = BIC label, cols = best-pred-mode

    # ── ΔLB upper bound ─────────────────────────────────────────────────
    max_single = max(single_mode_hit_1cm.values())
    max_single_mode = max(single_mode_hit_1cm, key=single_mode_hit_1cm.get)
    delta_lb_upper = oracle_hit_1cm - max_single
    delta_lb_vs_plan014 = oracle_hit_1cm - BASELINE_LB_PLAN014_015
    delta_lb_vs_plan016 = oracle_hit_1cm - BASELINE_LB_PLAN016_G1

    # ── Best-pred-mode distribution (counts) ────────────────────────────
    best_pred_counts = {
        MODE_NAMES[m]: int((best_pred_mode == m).sum()) for m in range(3)
    }
    best_pred_fracs = {m: best_pred_counts[m] / N for m in MODE_NAMES}

    # ── Artifact write ──────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "pred_err_per_mode.npy", pred_err)
    np.save(args.out_dir / "best_pred_mode.npy", best_pred_mode)

    summary = {
        "exp_id": "M002_imm-oracle",
        "n_samples": N,
        "hit_threshold_m": HIT_THRESHOLD,
        "single_mode_hit_1cm": single_mode_hit_1cm,
        "max_single_mode": max_single_mode,
        "max_single_mode_hit_1cm": max_single,
        "oracle_imm_hit_1cm": oracle_hit_1cm,
        "delta_lb_upper_bound_oracle_minus_best_single": delta_lb_upper,
        "delta_lb_vs_baseline_plan014_015": delta_lb_vs_plan014,
        "delta_lb_vs_baseline_plan016_g1": delta_lb_vs_plan016,
        "best_pred_mode_distribution": best_pred_fracs,
        "best_pred_mode_counts": best_pred_counts,
        "bic_vs_pred_agreement": bic_vs_pred_agreement,
        "bic_vs_pred_confusion_matrix_rows_bic_cols_pred": bic_vs_pred_confusion.tolist(),
        "pred_err_quantiles_per_mode_m": pred_err_quantiles,
        "comparison_to_baselines": {
            "plan014_015_LB": BASELINE_LB_PLAN014_015,
            "plan016_G1_LB": BASELINE_LB_PLAN016_G1,
        },
        "elapsed_seconds": time.time() - t_start,
    }
    (args.out_dir / "imm_oracle.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = []
    md.append("# IMM feasibility — oracle ceiling")
    md.append("")
    md.append(f"- N = {N}, hit threshold = {HIT_THRESHOLD*100:.0f}cm, target = t=+80ms forward prediction.")
    md.append("")
    md.append("## Single-mode hit@1cm")
    md.append("")
    md.append("| mode | hit@1cm | pred_err p50 (m) | p95 (m) |")
    md.append("|---|---|---|---|")
    for m in MODE_NAMES:
        q = pred_err_quantiles[m]
        md.append(f"| {m} | {single_mode_hit_1cm[m]:.4f} | {q['p50']:.4f} | {q['p95']:.4f} |")
    md.append("")
    md.append(f"**Best single-mode: {max_single_mode} = {max_single:.4f}**")
    md.append("")
    md.append("## IMM oracle ceiling")
    md.append("")
    md.append(f"- oracle_imm_hit_1cm (per-sample best-pred-mode) = **{oracle_hit_1cm:.4f}**")
    md.append(f"- ΔLB upper bound (vs best single-mode {max_single_mode}) = **{delta_lb_upper:+.4f}**")
    md.append("")
    md.append("## Comparison to baselines")
    md.append("")
    md.append("| baseline | LB | Δ vs oracle |")
    md.append("|---|---|---|")
    md.append(f"| plan-014/015 best_stack | {BASELINE_LB_PLAN014_015:.4f} | {delta_lb_vs_plan014:+.4f} |")
    md.append(f"| plan-016 G1 multi-seed | {BASELINE_LB_PLAN016_G1:.4f} | {delta_lb_vs_plan016:+.4f} |")
    md.append("")
    md.append("> 주의: oracle 은 *post-hoc* (Y label 사용한 best-mode picker) 라 *실제 IMM 으로 달성 가능한 LB 의 절대 upper bound* 만 의미. 실제 IMM 은 inferred posterior 사용하므로 oracle 아래.")
    md.append("")
    md.append("## Best-pred-mode distribution")
    md.append("")
    md.append("| mode | count | fraction |")
    md.append("|---|---|---|")
    for m in MODE_NAMES:
        md.append(f"| {m} | {best_pred_counts[m]} | {best_pred_fracs[m]*100:.2f}% |")
    md.append("")
    md.append(f"## BIC-label vs best-pred-mode agreement: **{bic_vs_pred_agreement*100:.2f}%**")
    md.append("")
    md.append("Confusion matrix (rows = BIC label, cols = best-pred-mode):")
    md.append("")
    md.append("|   | CV | CA | CT |")
    md.append("|---|---|---|---|")
    for r, mr in enumerate(MODE_NAMES):
        row = " | ".join(str(bic_vs_pred_confusion[r, c]) for c in range(3))
        md.append(f"| **{mr}** | {row} |")
    md.append("")
    md.append("> Low agreement (< 50%) 시 BIC fit-residual 분류가 forward-prediction task 와 misaligned → IMM verdict 보수적 해석 필요.")
    (args.out_dir / "imm_oracle.md").write_text("\n".join(md))

    # ── Console summary ─────────────────────────────────────────────────
    print(f"\n[imm-oracle c4] === summary ===", flush=True)
    print(f"  single_mode hit@1cm: {single_mode_hit_1cm}", flush=True)
    print(f"  best single: {max_single_mode} = {max_single:.4f}", flush=True)
    print(f"  oracle hit@1cm: {oracle_hit_1cm:.4f}", flush=True)
    print(f"  ΔLB upper bound (vs best single): {delta_lb_upper:+.4f}", flush=True)
    print(f"  ΔLB vs plan-014/015 baseline (0.6628): {delta_lb_vs_plan014:+.4f}", flush=True)
    print(f"  ΔLB vs plan-016 G1 (0.6638): {delta_lb_vs_plan016:+.4f}", flush=True)
    print(f"  BIC vs best-pred agreement: {bic_vs_pred_agreement*100:.2f}%", flush=True)
    print(f"  artifacts -> {args.out_dir}", flush=True)

    # Verification: oracle ≥ any single-mode (mathematical invariant)
    assert oracle_hit_1cm >= max_single - 1e-9, (
        f"oracle hit ({oracle_hit_1cm}) < max single ({max_single}) — invariant violated"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
