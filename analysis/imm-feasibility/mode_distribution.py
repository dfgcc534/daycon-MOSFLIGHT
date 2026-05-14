"""IMM feasibility — c3 (Stage 3): 10K train sample 위 CV/CA/CT 분류 + 분포 측정.

산식 / verdict rule: `/home/ahn/.claude/plans/glimmering-popping-river.md` §Approach.

Outputs (analysis/imm-feasibility/):
  - mode_labels.npy (10000,) int8 ∈ {0:CV, 1:CA, 2:CT}
  - mode_posteriors.npy (10000, 3) float32 (Schwarz posterior)
  - rss_per_mode.npy (10000, 3) float64
  - bic_per_mode.npy (10000, 3) float64
  - delta_bic.npy (10000,) float64
  - fallback_flags.npy (10000,) int8
  - pred_t80_per_mode.npy (10000, 3, 3) float64 (carry to imm_oracle.py)
  - mode_distribution.json (summary)
  - mode_distribution.md (human-readable)

Usage:
    python analysis/imm-feasibility/mode_distribution.py
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

from src.io import load_all_samples  # noqa: E402
from src.pb_0_6822 import motion_models as mm  # noqa: E402


MODE_NAMES = ["CV", "CA", "CT"]


def classify_verdict(fracs: dict[str, float]) -> str:
    """plan §Approach verdict rule:
        strong_collapse: top ≥ 0.75 AND second ≤ 0.15
        mild_collapse:   top ∈ [0.60, 0.75)
        split:           top < 0.50 AND all 3 ≥ 0.20
        intermediate:    else
    """
    sorted_f = sorted(fracs.values(), reverse=True)
    top, second, third = sorted_f
    if top >= 0.75 and second <= 0.15:
        return "strong_collapse"
    if 0.60 <= top < 0.75:
        return "mild_collapse"
    if top < 0.50 and third >= 0.20:
        return "split"
    return "intermediate"


def quantiles(arr: np.ndarray, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict[str, float]:
    return {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}


def posterior_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Shannon entropy per row of (N, 3). Returns (N,) in nats."""
    p = posteriors.astype(np.float64)
    # avoid log(0)
    safe = np.where(p > 1e-30, p, 1e-30)
    return -(p * np.log(safe)).sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/imm-feasibility"))
    args = ap.parse_args()

    t_start = time.time()
    print("[imm-feasibility c3] loading train data ...", flush=True)
    ids, X = load_all_samples("train")
    X = X.astype(np.float64)
    print(f"[imm-feasibility c3] N={X.shape[0]}, shape={X.shape}", flush=True)

    print(f"[imm-feasibility c3] fitting 3 motion models per sample ...", flush=True)
    t0 = time.time()
    result = mm.fit_all_samples(X)
    fit_elapsed = time.time() - t0
    print(f"[imm-feasibility c3] fit complete in {fit_elapsed:.1f}s", flush=True)

    labels = result["labels"]
    posteriors = result["posterior"]
    rss = result["rss"]
    bic = result["bic"]
    delta_bic = result["delta_bic"]
    fallback = result["fallback_flags"]
    pred_t80 = result["pred_t80"]

    # ── 분포 산출 ────────────────────────────────────────────────────────
    counts = {MODE_NAMES[m]: int((labels == m).sum()) for m in range(3)}
    N = int(labels.size)
    fracs = {m: counts[m] / N for m in MODE_NAMES}

    fallback_counts = {
        "stationary": int((fallback & mm.FALLBACK_STATIONARY).astype(bool).sum()),
        "degen_plane": int((fallback & mm.FALLBACK_DEGEN_PLANE).astype(bool).sum()),
        "non_monotone_angle": int((fallback & mm.FALLBACK_NON_MONOTONE_ANGLE).astype(bool).sum()),
    }

    rss_q_per_mode = {
        MODE_NAMES[m]: quantiles(rss[:, m]) for m in range(3)
    }

    delta_bic_stats = {
        **quantiles(delta_bic),
        "mean": float(delta_bic.mean()),
        "frac_gt_10": float((delta_bic > 10).mean()),
        "frac_gt_2": float((delta_bic > 2).mean()),   # Kass-Raftery weak evidence threshold
    }

    post_entropy = posterior_entropy(posteriors)
    entropy_stats = quantiles(post_entropy) | {"mean": float(post_entropy.mean())}

    verdict = classify_verdict(fracs)

    imm_implication = {
        "strong_collapse": "IMM 무가치 — single KF (top mode) 충분.",
        "mild_collapse": "IMM weak — ensemble member 일부 가치, 단독 paradigm 부족.",
        "split": "IMM strong 가치 — mode-specific KF 의 weighted ensemble 이 single KF 보다 우월 가능.",
        "intermediate": "verdict 모호 — ΔBIC 분포 + posterior entropy 추가 분석 필요.",
    }[verdict]

    # ── Artifact write ──────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.out_dir / "mode_labels.npy", labels)
    np.save(args.out_dir / "mode_posteriors.npy", posteriors)
    np.save(args.out_dir / "rss_per_mode.npy", rss)
    np.save(args.out_dir / "bic_per_mode.npy", bic)
    np.save(args.out_dir / "delta_bic.npy", delta_bic)
    np.save(args.out_dir / "fallback_flags.npy", fallback)
    np.save(args.out_dir / "pred_t80_per_mode.npy", pred_t80)

    summary = {
        "exp_id": "M001_mode-distribution",
        "n_samples": N,
        "mode_counts": counts,
        "mode_fractions": fracs,
        "verdict": verdict,
        "implications_for_imm": imm_implication,
        "delta_bic_stats": delta_bic_stats,
        "posterior_entropy_stats_nats": entropy_stats,
        "rss_quantiles_per_mode": rss_q_per_mode,
        "fallback_counts": fallback_counts,
        "k_params": {"CV": mm.K_CV, "CA": mm.K_CA, "CT": mm.K_CT},
        "n_residuals_per_sample": mm.N_RES,
        "elapsed_seconds": time.time() - t_start,
        "fit_elapsed_seconds": fit_elapsed,
    }
    (args.out_dir / "mode_distribution.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    md = []
    md.append(f"# IMM feasibility — mode distribution ({verdict})")
    md.append("")
    md.append(f"- N = {N} train samples")
    md.append(f"- Fit time: {fit_elapsed:.1f}s")
    md.append("")
    md.append("## Distribution")
    md.append("")
    md.append("| mode | count | fraction |")
    md.append("|---|---|---|")
    for m in MODE_NAMES:
        md.append(f"| {m} | {counts[m]} | {fracs[m]*100:.2f}% |")
    md.append("")
    md.append(f"**Verdict: `{verdict}`**")
    md.append("")
    md.append(f"> {imm_implication}")
    md.append("")
    md.append("## ΔBIC (decisiveness)")
    md.append("")
    md.append("| stat | value |")
    md.append("|---|---|")
    for k, v in delta_bic_stats.items():
        if "frac" in k:
            md.append(f"| {k} | {v*100:.2f}% |")
        else:
            md.append(f"| {k} | {v:.3f} |")
    md.append("")
    md.append("> Kass-Raftery: ΔBIC > 2 = weak evidence, > 6 = positive, > 10 = strong.")
    md.append("")
    md.append("## Posterior entropy (nats; max = log(3) ≈ 1.099)")
    md.append("")
    md.append("| stat | value |")
    md.append("|---|---|")
    for k, v in entropy_stats.items():
        md.append(f"| {k} | {v:.4f} |")
    md.append("")
    md.append("> 0 = one-hot (decisive collapse), log(3)≈1.099 = uniform (mode ambiguous).")
    md.append("")
    md.append("## RSS quantiles per mode (m²)")
    md.append("")
    md.append("| mode | p5 | p25 | p50 | p75 | p95 |")
    md.append("|---|---|---|---|---|---|")
    for m in MODE_NAMES:
        q = rss_q_per_mode[m]
        md.append(f"| {m} | {q['p5']:.2e} | {q['p25']:.2e} | {q['p50']:.2e} | {q['p75']:.2e} | {q['p95']:.2e} |")
    md.append("")
    md.append("## Fallback flag counts")
    md.append("")
    md.append("| flag | count |")
    md.append("|---|---|")
    for k, v in fallback_counts.items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append(f"- stationary: 모든 timestep 위치 동일 (std < {mm.EPS_STATIONARY}).")
    md.append(f"- degen_plane: PCA singular[2]/singular[0] > {mm.EPS_DEGEN_PLANE} (3D scatter, plane fit unreliable).")
    md.append(f"- non_monotone_angle: angular θ unwrap 후 |Δθ| > π (CT 부적합 fallback).")
    (args.out_dir / "mode_distribution.md").write_text("\n".join(md))

    # ── Console summary ─────────────────────────────────────────────────
    print(f"\n[imm-feasibility c3] === summary ===", flush=True)
    print(f"  mode counts: {counts}", flush=True)
    print(f"  mode fractions: " + ", ".join(f"{m}={fracs[m]*100:.2f}%" for m in MODE_NAMES), flush=True)
    print(f"  verdict: **{verdict}**", flush=True)
    print(f"  delta_bic p50={delta_bic_stats['p50']:.2f} p95={delta_bic_stats['p95']:.2f} frac>10={delta_bic_stats['frac_gt_10']*100:.1f}%", flush=True)
    print(f"  posterior entropy mean={entropy_stats['mean']:.3f} (max log(3)={np.log(3):.3f})", flush=True)
    print(f"  fallback: {fallback_counts}", flush=True)
    print(f"  artifacts -> {args.out_dir}", flush=True)
    print(f"  total elapsed: {time.time() - t_start:.1f}s", flush=True)

    # Verification: fractions sum to 1
    assert abs(sum(fracs.values()) - 1.0) < 1e-9, f"fractions sum mismatch: {sum(fracs.values())}"

    return 0


if __name__ == "__main__":
    sys.exit(main())
