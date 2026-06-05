"""plan-009 c2.1 (part 1): Oracle decomposition on extended 25 cands.

Spec @ §4.2 (plans/plan-009-selector-ranking-loss.md):
  best-raw-cand error 8-bin 분포 + 1cm/1.5cm/2cm ceiling 측정.

Per Fix 22 (§3.3) — per-band hit_after 식 박제 시 n_in_band 박제 anchor 도 산출.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = REPO_ROOT / "runs/baseline/G001_candidate-redefine/oof_selector_scores.npz"
OUT_DIR = REPO_ROOT / "analysis/plan-009"

BIN_EDGES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, np.inf]
BIN_LABELS = [
    "[0, 0.5cm)",
    "[0.5cm, 1cm)",
    "[1cm, 1.5cm)",
    "[1.5cm, 2cm)",
    "[2cm, 3cm)",
    "[3cm, 5cm)",
    "[5cm, 10cm)",
    "[10cm, inf)",
]

PERBAND_EDGES = [0.0, 0.005, 0.01, 0.015, 0.02, np.inf]
PERBAND_LABELS = [
    "[0, 0.5cm)",
    "[0.5cm, 1cm)",
    "[1cm, 1.5cm)",
    "[1.5cm, 2cm)",
    "[2cm, inf)",
]


def main() -> int:
    print("[plan-009 c2.1 oracle_decomp] start")
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"plan_008_artifact_missing — {NPZ_PATH} 부재 (G001 산출)"
        )
    d = np.load(NPZ_PATH, allow_pickle=True)
    cands = d["cands"].astype(np.float64)  # (N, 25, 3)
    y = d["y"].astype(np.float64)  # (N, 3)
    n_samples, n_cands, dim = cands.shape
    print(f"[oracle_decomp] cands shape={cands.shape}, y shape={y.shape}")

    err_per_cand = np.linalg.norm(cands - y[:, None, :], axis=-1)  # (N, 25)
    best_err = err_per_cand.min(axis=1)  # (N,)
    best_cand_idx = err_per_cand.argmin(axis=1)  # (N,)

    hist_counts, _ = np.histogram(best_err, bins=BIN_EDGES)
    bin_distribution = {
        label: {
            "count": int(count),
            "fraction": float(count / n_samples),
        }
        for label, count in zip(BIN_LABELS, hist_counts)
    }

    oracle_1cm = float((best_err <= 0.01).mean())
    oracle_1_5cm = float((best_err <= 0.015).mean())
    oracle_2cm = float((best_err <= 0.02).mean())

    n_in_band = {}
    for i, (lo, hi, lab) in enumerate(zip(PERBAND_EDGES[:-1], PERBAND_EDGES[1:], PERBAND_LABELS)):
        mask = (best_err >= lo) & (best_err < hi)
        n_in_band[lab] = int(mask.sum())

    summary = {
        "exp_id": "plan-009/oracle_decomp",
        "source_npz": str(NPZ_PATH.relative_to(REPO_ROOT)),
        "n_samples": int(n_samples),
        "n_candidates": int(n_cands),
        "dim": int(dim),
        "bin_distribution_8bin": bin_distribution,
        "oracle_1cm": oracle_1cm,
        "oracle_1_5cm": oracle_1_5cm,
        "oracle_2cm": oracle_2cm,
        "n_in_band_per_band": n_in_band,
        "n_oracle_miss_1cm": int((best_err > 0.01).sum()),
        "n_oracle_miss_1_5cm": int((best_err > 0.015).sum()),
        "best_err_stats": {
            "mean": float(best_err.mean()),
            "median": float(np.median(best_err)),
            "p25": float(np.percentile(best_err, 25)),
            "p75": float(np.percentile(best_err, 75)),
            "p95": float(np.percentile(best_err, 95)),
            "max": float(best_err.max()),
        },
        "best_cand_idx_distribution": {
            int(i): int((best_cand_idx == i).sum()) for i in range(n_cands)
        },
        "plan_008_reference_match": {
            "oracle_1cm_expected": 0.7562,
            "oracle_1cm_actual": oracle_1cm,
            "tolerance": 0.002,
            "match": abs(oracle_1cm - 0.7562) <= 0.002,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "oracle_decomp.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] oracle_decomp.json: {out_json.relative_to(REPO_ROOT)}")

    md_lines = [
        "# plan-009 c2.1 — oracle_decomp.md\n",
        f"- source: `{NPZ_PATH.relative_to(REPO_ROOT)}`",
        f"- N samples: {n_samples}, K cands: {n_cands}, D dim: {dim}",
        "",
        "## Oracle ceilings (best-raw-cand err thresholds)\n",
        "| ceiling | value |",
        "|---|---|",
        f"| oracle_1cm   | {oracle_1cm:.4f} |",
        f"| oracle_1.5cm | {oracle_1_5cm:.4f} |",
        f"| oracle_2cm   | {oracle_2cm:.4f} |",
        "",
        "## 8-bin distribution (best-raw-cand err)\n",
        "| bin | count | fraction |",
        "|---|---|---|",
    ]
    for label, info in bin_distribution.items():
        md_lines.append(f"| {label} | {info['count']} | {info['fraction']:.4f} |")
    md_lines += [
        "",
        "## Per-band n_in_band (hit_after 분모 anchor, Fix 22 §3.3 박제)\n",
        "| band | n_in_band |",
        "|---|---|",
    ]
    for label, n in n_in_band.items():
        md_lines.append(f"| {label} | {n} |")
    md_lines += [
        "",
        "## plan-008 reference match",
        f"- oracle_1cm expected (plan-008 c7) = 0.7562",
        f"- oracle_1cm actual = {oracle_1cm:.4f}",
        f"- tol = 0.002 — match: **{summary['plan_008_reference_match']['match']}**",
    ]
    out_md = OUT_DIR / "oracle_decomp.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[OK] oracle_decomp.md: {out_md.relative_to(REPO_ROOT)}")

    print("[plan-009 c2.1 oracle_decomp] complete")
    print(
        f"  oracle_1cm={oracle_1cm:.4f}  oracle_1.5cm={oracle_1_5cm:.4f}  "
        f"oracle_2cm={oracle_2cm:.4f}"
    )
    print(f"  n_in_band={n_in_band}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
