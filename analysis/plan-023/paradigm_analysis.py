"""plan-023 §7 — paradigm analysis: collect_cells / select_best / marginals
                                    + compare_with_plan022_best (신규).

12 cell (4 layout × 3 τ_cls) 결과 집계 + best cell + N/τ marginal
+ plan-022 best (A6_bcc14_tau001, Δ_sum=0.0279) 대비 비교 + mode collapse 측정.

CLI:
  python analysis/plan-023/paradigm_analysis.py \
         --out-json analysis/plan-023/paradigm_analysis.json \
         --out-md analysis/plan-023/paradigm_analysis.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve().parent

LAYOUT_IDS = [
    "B1_dodeca20",
    "B2_trunc_octa24",
    "B3_icosidodec30",
    "B4_fib50",
]
TAU_SCAN = [0.001, 0.003, 0.005]

# plan-022 best (carry hard evidence — A6_bcc14_tau001)
PLAN022_BEST = {
    "cell_key": "A6_bcc14_tau001",
    "K": 14,
    "delta_1cm": 0.0208,
    "delta_1.5cm": 0.0071,
    "sum": 0.0279,
}


def collect_cells(plan_dir: Path) -> dict[str, dict]:
    """4 results_B{n}.json → flatten 12 cell dict."""
    cells: dict[str, dict] = {}
    for layout_id in LAYOUT_IDS:
        sub_exp_key = layout_id.split("_")[0]               # "B1", ...
        path = plan_dir / f"results_{sub_exp_key}.json"
        results = json.loads(path.read_text())              # {"tau_0.001": {...}, ...}
        for tau_key, cell in results.items():
            tau_val = float(tau_key.replace("tau_", ""))
            cell_key = f"{layout_id}_tau{int(tau_val * 1000):03d}"
            cells[cell_key] = {
                **cell,
                "layout": layout_id,
                "tau_cls": tau_val,
            }
    return cells


def select_best(cells: dict[str, dict]) -> tuple[str, dict]:
    """(pass_both, Δ_1cm + Δ_1.5cm, alphabetic asc) lexicographic max — deterministic.

    drop cell (sentinel `dropped=True`) 은 후보에서 제외.
    """
    eligible = {k: v for k, v in cells.items() if not v.get("dropped", False)}
    best_key, best_cell = max(
        eligible.items(),
        key=lambda kv: (
            kv[1]["pass_both"],
            kv[1]["delta_1cm"] + kv[1]["delta_1.5cm"],
            tuple(-ord(c) for c in kv[0]),
        ),
    )
    return best_key, best_cell


def marginals(cells: dict[str, dict]) -> dict:
    """layout-axis (per layout best τ) + τ-axis (per τ best layout) marginals."""
    by_layout: dict[str, dict] = {}
    for layout_id in LAYOUT_IDS:
        layout_cells = {k: v for k, v in cells.items() if v["layout"] == layout_id}
        if not layout_cells:
            continue
        best_key, best_cell = select_best(layout_cells)
        by_layout[layout_id] = {**best_cell, "best_cell_key": best_key}

    by_tau: dict[str, dict] = {}
    for tau in TAU_SCAN:
        tau_cells = {k: v for k, v in cells.items() if abs(v["tau_cls"] - tau) < 1e-9}
        if not tau_cells:
            continue
        best_key, best_cell = select_best(tau_cells)
        by_tau[f"tau_{tau:.3f}"] = {**best_cell, "best_cell_key": best_key}

    return {"by_layout": by_layout, "by_tau": by_tau}


def compare_with_plan022_best(cells: dict[str, dict]) -> dict:
    """plan-022 best (A6_bcc14_tau001, Δ_sum=0.0279) 대비 본 plan 12 cell 비교.

    §7.3 schema: cells_beating_sum / cells_beating_1cm / cells_beating_1.5cm
                 (모두 strict > 비교, inclusive 안 함).
    """
    p022 = PLAN022_BEST
    out: dict[str, Any] = {"plan022_best": p022}

    def _cell_row(k: str, c: dict) -> dict:
        s = c["delta_1cm"] + c["delta_1.5cm"]
        return {
            "cell_key": k,
            "K": c.get("K"),
            "delta_1cm": c["delta_1cm"],
            "delta_1.5cm": c["delta_1.5cm"],
            "sum": s,
            "delta_vs_plan022_sum": s - p022["sum"],
        }

    eligible = {k: v for k, v in cells.items() if not v.get("dropped", False)}
    out["cells_beating_sum"] = sorted(
        [_cell_row(k, c) for k, c in eligible.items() if (c["delta_1cm"] + c["delta_1.5cm"]) > p022["sum"]],
        key=lambda r: -r["sum"],
    )
    out["cells_beating_1cm"] = sorted(
        [_cell_row(k, c) for k, c in eligible.items() if c["delta_1cm"] > p022["delta_1cm"]],
        key=lambda r: -r["delta_1cm"],
    )
    out["cells_beating_1.5cm"] = sorted(
        [_cell_row(k, c) for k, c in eligible.items() if c["delta_1.5cm"] > p022["delta_1.5cm"]],
        key=lambda r: -r["delta_1.5cm"],
    )
    out["n_cells_beating_sum"] = len(out["cells_beating_sum"])
    out["n_cells_beating_1cm"] = len(out["cells_beating_1cm"])
    out["n_cells_beating_1.5cm"] = len(out["cells_beating_1.5cm"])
    if eligible:
        best_key, _ = select_best(eligible)
        out["best_cell_in_plan023"] = best_key
    else:
        out["best_cell_in_plan023"] = None
    return out


def mode_collapse_table(cells: dict[str, dict]) -> dict:
    """§7.3 mode collapse 완화 측정: max_class_ratio / (1/K) ratio.

    H3 check: K=50 의 τ=0.001 cell `max_class_ratio` ≤ K=20 의 τ=0.001 cell × 0.5 ?
    """
    table = []
    for k, c in cells.items():
        if c.get("dropped", False):
            continue
        K = c["K"]
        mcr = c["max_class_ratio"]
        uniform = 1.0 / K
        table.append({
            "cell_key": k,
            "K": K,
            "max_class_ratio": mcr,
            "uniform_baseline": uniform,
            "ratio_to_uniform": mcr / uniform,
        })

    # H3 check
    k50 = next(
        (c for c in cells.values()
         if c.get("layout") == "B4_fib50" and abs(c.get("tau_cls", 0) - 0.001) < 1e-9),
        None,
    )
    k20 = next(
        (c for c in cells.values()
         if c.get("layout") == "B1_dodeca20" and abs(c.get("tau_cls", 0) - 0.001) < 1e-9),
        None,
    )
    h3 = None
    if k50 is not None and k20 is not None and not k50.get("dropped") and not k20.get("dropped"):
        mcr50 = k50["max_class_ratio"]
        mcr20 = k20["max_class_ratio"]
        h3 = {
            "k50_tau001": mcr50,
            "k20_tau001": mcr20,
            "ratio": mcr50 / mcr20 if mcr20 > 0 else None,
            "pass": (mcr50 <= 0.5 * mcr20),
        }

    return {"mode_collapse_table": table, "h3_check": h3}


def render_md(cells: dict[str, dict], best: tuple[str, dict],
              margins: dict, compare: dict, mc: dict, dropped: list[str]) -> str:
    """plan-023 §7 markdown report."""
    best_key, best_cell = best
    lines = [
        "# plan-023 paradigm_analysis — 12 cell large-N sweep",
        "",
        "F0 baseline: hit@1cm = 0.6320, hit@1.5cm = 0.8033.",
        "",
        "## 12 cell grid (N × τ_cls)",
        "",
        "| layout | K | τ=0.001 Δ_1cm/Δ_1.5cm pass | τ=0.003 Δ_1cm/Δ_1.5cm pass | τ=0.005 Δ_1cm/Δ_1.5cm pass |",
        "|---|---|---|---|---|",
    ]
    for layout_id in LAYOUT_IDS:
        row = [layout_id, ""]
        K_val = None
        for tau in TAU_SCAN:
            cell_key = f"{layout_id}_tau{int(tau * 1000):03d}"
            c = cells[cell_key]
            K_val = c["K"]
            pass_mark = "✓" if c["pass_both"] else "✗"
            if c.get("dropped"):
                row.append(f"DROPPED ({c.get('drop_reason')})")
            else:
                row.append(f"{c['delta_1cm']:+.4f} / {c['delta_1.5cm']:+.4f} {pass_mark}")
        row[1] = str(K_val)
        lines.append("| " + " | ".join(row) + " |")

    n_pass = sum(1 for c in cells.values() if not c.get("dropped") and c["pass_both"])
    lines += [
        "",
        "## Best cell 🏆",
        "",
        f"- **{best_key}** (K={best_cell['K']}, τ_cls={best_cell['tau_cls']})",
        f"- hit@1cm = {best_cell['hit_1cm']:.4f}, hit@1.5cm = {best_cell['hit_1.5cm']:.4f}",
        f"- Δ_1cm = **{best_cell['delta_1cm']:+.4f}** (pass criterion +0.005)",
        f"- Δ_1.5cm = **{best_cell['delta_1.5cm']:+.4f}** (pass criterion +0.005)",
        f"- pass_both = **{best_cell['pass_both']}**",
        f"- Δ sum = {best_cell['delta_1cm'] + best_cell['delta_1.5cm']:+.4f} (plan-022 best 0.0279 대비 Δ {(best_cell['delta_1cm'] + best_cell['delta_1.5cm']) - 0.0279:+.4f})",
        f"- max_class_ratio = {best_cell['max_class_ratio']:.4f}",
        "",
        "## Layout-axis marginal (각 N 의 best τ — K trend)",
        "",
        "| layout | K | best τ | Δ_1cm | Δ_1.5cm | Δ sum | pass_both | max_class |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for layout_id, m in margins["by_layout"].items():
        s = m["delta_1cm"] + m["delta_1.5cm"]
        lines.append(
            f"| {layout_id} | {m['K']} | {m['tau_cls']:.3f} | {m['delta_1cm']:+.4f} | "
            f"{m['delta_1.5cm']:+.4f} | {s:+.4f} | {m['pass_both']} | "
            f"{m['max_class_ratio']:.3f} |"
        )

    lines += [
        "",
        "## τ_cls-axis marginal (각 τ 의 best layout)",
        "",
        "| τ_cls | best layout | K | Δ_1cm | Δ_1.5cm | Δ sum | pass_both | max_class |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tau_key, m in margins["by_tau"].items():
        s = m["delta_1cm"] + m["delta_1.5cm"]
        lines.append(
            f"| {tau_key.replace('tau_', '')} | {m['layout']} | {m['K']} | "
            f"{m['delta_1cm']:+.4f} | {m['delta_1.5cm']:+.4f} | {s:+.4f} | "
            f"{m['pass_both']} | {m['max_class_ratio']:.3f} |"
        )

    lines += [
        "",
        "## plan-022 best 대비 compare (A6_bcc14_tau001, Δ_sum=0.0279)",
        "",
        f"- cells beating Δ sum 0.0279: **{compare['n_cells_beating_sum']} / 12**",
        f"- cells beating Δ_1cm 0.0208: **{compare['n_cells_beating_1cm']} / 12**",
        f"- cells beating Δ_1.5cm 0.0071: **{compare['n_cells_beating_1.5cm']} / 12**",
        f"- best in plan-023: **{compare['best_cell_in_plan023']}**",
        "",
    ]
    if compare["cells_beating_sum"]:
        lines.append("### Cells beating plan-022 Δ sum:")
        lines.append("")
        lines.append("| cell | K | Δ_1cm | Δ_1.5cm | sum | vs plan-022 sum |")
        lines.append("|---|---|---|---|---|---|")
        for r in compare["cells_beating_sum"]:
            lines.append(
                f"| {r['cell_key']} | {r['K']} | {r['delta_1cm']:+.4f} | "
                f"{r['delta_1.5cm']:+.4f} | {r['sum']:+.4f} | "
                f"{r['delta_vs_plan022_sum']:+.4f} |"
            )

    lines += [
        "",
        "## Mode collapse 완화 (max_class_ratio vs 1/K uniform)",
        "",
        "| cell | K | max_class | uniform 1/K | ratio (mcr / 1/K) |",
        "|---|---|---|---|---|",
    ]
    for r in mc["mode_collapse_table"]:
        lines.append(
            f"| {r['cell_key']} | {r['K']} | {r['max_class_ratio']:.4f} | "
            f"{r['uniform_baseline']:.4f} | {r['ratio_to_uniform']:.2f} |"
        )

    if mc["h3_check"] is not None:
        h3 = mc["h3_check"]
        lines += [
            "",
            "### H3 check (effective temperature 변화): K=50 τ=0.001 max_class ≤ K=20 τ=0.001 × 0.5 ?",
            "",
            f"- K=50 (B4_fib50) τ=0.001 max_class = {h3['k50_tau001']:.4f}",
            f"- K=20 (B1_dodeca20) τ=0.001 max_class = {h3['k20_tau001']:.4f}",
            f"- ratio = {h3['ratio']:.3f}" if h3["ratio"] is not None else "- ratio = N/A",
            f"- H3 PASS = **{h3['pass']}**",
        ]

    lines += [
        "",
        f"## G3 — {n_pass}/{12 - len(dropped)} effective cell pass_both=True"
        f"{f' (drop {len(dropped)})' if dropped else ''}",
        "",
        f"- Dropped (soft_label_collapse): {len(dropped)} cells "
        f"({dropped if dropped else 'none'})",
        f"- Effective denominator: {12 - len(dropped)} cells",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=_THIS / "paradigm_analysis.json")
    ap.add_argument("--out-md", type=Path, default=_THIS / "paradigm_analysis.md")
    args = ap.parse_args()

    cells = collect_cells(_THIS)
    print(f"[paradigm] collected {len(cells)} cells")

    # soft_label_collapse drop (sentinel `dropped=True` 또는 max_class_ratio > 0.95)
    dropped = [
        k for k, c in cells.items()
        if c.get("dropped", False) or c.get("max_class_ratio", 0) > 0.95
    ]
    if dropped:
        print(f"[paradigm] dropped {len(dropped)} cells: {dropped}")

    effective_cells = {k: v for k, v in cells.items() if k not in dropped}
    best_key, best_cell = select_best(effective_cells)
    margins = marginals(effective_cells)
    compare = compare_with_plan022_best(effective_cells)
    mc = mode_collapse_table(effective_cells)

    out: dict[str, Any] = {
        "cells": cells,
        "dropped_cells": dropped,
        "best_cell_key": best_key,
        "best_cell": best_cell,
        "marginals": margins,
        "compare_with_plan022_best": compare,
        "mode_collapse_table": mc["mode_collapse_table"],
        "h3_check": mc["h3_check"],
        "n_total": len(cells),
        "n_pass_both": sum(
            1 for c in cells.values()
            if not c.get("dropped", False) and c.get("pass_both", False)
        ),
        "n_effective": len(effective_cells),
    }
    args.out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"[paradigm] wrote {args.out_json}")

    md = render_md(cells, (best_key, best_cell), margins, compare, mc, dropped)
    args.out_md.write_text(md)
    print(f"[paradigm] wrote {args.out_md}")

    print(f"\n[paradigm] best = {best_key}")
    print(f"    Δ_1cm = {best_cell['delta_1cm']:+.4f}, "
          f"Δ_1.5cm = {best_cell['delta_1.5cm']:+.4f}")
    print(f"    pass_both = {best_cell['pass_both']}, "
          f"sum = {best_cell['delta_1cm'] + best_cell['delta_1.5cm']:.4f}")
    print(f"    cells beating plan-022 sum: {compare['n_cells_beating_sum']}")


if __name__ == "__main__":
    main()
