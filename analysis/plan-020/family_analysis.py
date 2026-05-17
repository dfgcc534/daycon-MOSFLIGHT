"""plan-020 c11 G3 — Family-level analysis: 17 후보 × 2 metric × 5-fold + family winner + overall best."""
from __future__ import annotations

import json
from pathlib import Path

THIS = Path(__file__).parent

PASS_THR = 0.005

FAMILY_MAP = {
    "C01_helix": "F1_회전",
    "C02_ctra": "F1_회전",
    "C03_ctrv": "F1_회전",
    "C04_imm": "F1_회전",
    "C05_per_regime_f0": "F2_data_driven",
    "C06_quintic_hermite": "F3_고차_미분",
    "C07_jerk_quartic": "F3_고차_미분",
    "C08_singer": "F4_noise_adaptive",
    "C09_kalman_smoother": "F4_noise_adaptive",
    "C10_bishop_frame": "F5_기하학",
    "C11_se3_twist": "F5_기하학",
    "C12_wingbeat_corrected": "F6_도메인_정보",
    "C13_levy_prior": "F6_도메인_정보",
    "C14_trajectory_knn": "F7_비모수",
    "N01_mlp_coef": "F2_data_driven",
    "N02_tcn_coef": "F2_data_driven",
    "N05_moe": "F2_data_driven",
}


def main():
    det = json.loads((THIS / "results_deterministic.json").read_text())
    nnr = json.loads((THIS / "results_nn.json").read_text())

    f0 = det["f0_baseline"]
    f0_hit_1cm = float(f0["hit_1cm_5fold_concat"])
    f0_hit_15cm = float(f0["hit_1.5cm_5fold_concat"])

    # 17 후보 표
    table = []
    for src in (det, nnr):
        for name, r in src.items():
            if name == "f0_baseline":
                continue
            d1 = float(r.get("delta_1cm", 0.0))
            d15 = float(r.get("delta_1.5cm", 0.0))
            table.append({
                "candidate": name,
                "family": FAMILY_MAP[name],
                "hit_1cm": float(r["hit_1cm"]),
                "delta_1cm": d1,
                "hit_1.5cm": float(r["hit_1.5cm"]),
                "delta_1.5cm": d15,
                "fold_variance_1cm": float(r.get("fold_variance_1cm", 0.0)),
                "fold_variance_1.5cm": float(r.get("fold_variance_1.5cm", 0.0)),
                "pass_both": (d1 >= PASS_THR and d15 >= PASS_THR),
                "delta_combined": d1 + 0.5 * d15,  # tie-break
            })
    table.sort(key=lambda x: (-x["pass_both"], -x["delta_combined"]))

    # family winners
    fam_winners = {}
    for fam in sorted(set(FAMILY_MAP.values())):
        cands = [r for r in table if r["family"] == fam]
        passers = [r for r in cands if r["pass_both"]]
        if passers:
            winner = max(passers, key=lambda x: x["delta_combined"])
            fam_winners[fam] = {
                "winner": winner["candidate"],
                "delta_1cm": winner["delta_1cm"],
                "delta_1.5cm": winner["delta_1.5cm"],
                "delta_combined": winner["delta_combined"],
            }
        else:
            fam_winners[fam] = {"winner": None, "reason": "no candidate passed +0.005 둘 다"}

    # overall best (§9.1.1 단수 선정)
    pass_pool = [r for r in table if r["pass_both"]]
    if pass_pool:
        # tie-break: Δ_combined → hit_1cm → fold_variance (smaller)
        pass_pool.sort(key=lambda x: (-x["delta_combined"], -x["hit_1cm"], x["fold_variance_1cm"]))
        best = pass_pool[0]
        overall = {
            "best_candidate": best["candidate"],
            "best_hit_1cm": best["hit_1cm"],
            "best_hit_1.5cm": best["hit_1.5cm"],
            "best_delta_1cm": best["delta_1cm"],
            "best_delta_1.5cm": best["delta_1.5cm"],
            "best_family": best["family"],
            "band": "positive" if best["delta_1cm"] >= 0.01 else "marginal",
        }
    else:
        overall = {
            "best_candidate": None,
            "band": "negative",
            "note": "all_negative — no candidate passed +0.005 둘 다",
        }

    out = {
        "f0_baseline": {"hit_1cm": f0_hit_1cm, "hit_1.5cm": f0_hit_15cm},
        "pass_threshold": PASS_THR,
        "n_candidates": len(table),
        "n_pass": int(sum(r["pass_both"] for r in table)),
        "table": table,
        "family_winners": fam_winners,
        "overall": overall,
    }
    out_path = THIS / "family_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    print(f"wrote {out_path}")

    # markdown
    md_lines = [
        "# plan-020 STAGE 4 G3 — Family-level analysis",
        "",
        f"## F0 baseline: hit@1cm = {f0_hit_1cm:.4f}, hit@1.5cm = {f0_hit_15cm:.4f}",
        "",
        "## 17 후보 × 2 metric × 5-fold concat OOF",
        "",
        "| # | candidate | family | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | fold_var_1cm | pass 둘 다 ≥ +0.005 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        md_lines.append(
            f"| {r['candidate']} | {r['family']} | {r['hit_1cm']:.4f} | "
            f"{r['delta_1cm']:+.4f} | {r['hit_1.5cm']:.4f} | {r['delta_1.5cm']:+.4f} | "
            f"{r['fold_variance_1cm']:.4f} | {'✓' if r['pass_both'] else '✗'} |"
        )
    md_lines += [
        "",
        "## Family-level winner (§8.2 2-단계: pass 우선, Δ_combined tie-break)",
        "",
        "| family | winner | Δ_1cm | Δ_1.5cm | Δ_combined |",
        "|---|---|---|---|---|",
    ]
    for fam, fw in fam_winners.items():
        if fw["winner"]:
            md_lines.append(
                f"| {fam} | {fw['winner']} | {fw['delta_1cm']:+.4f} | "
                f"{fw['delta_1.5cm']:+.4f} | {fw['delta_combined']:+.4f} |"
            )
        else:
            md_lines.append(f"| {fam} | 없음 | — | — | — |")
    md_lines += [
        "",
        "## Overall best_candidate (§9.1.1 단수 선정)",
        "",
    ]
    if overall.get("best_candidate"):
        md_lines += [
            f"- **best_candidate**: `{overall['best_candidate']}`",
            f"- **best_family**: `{overall['best_family']}`",
            f"- **best_hit_1cm**: {overall['best_hit_1cm']:.4f} (Δ {overall['best_delta_1cm']:+.4f})",
            f"- **best_hit_1.5cm**: {overall['best_hit_1.5cm']:.4f} (Δ {overall['best_delta_1.5cm']:+.4f})",
            f"- **band**: `{overall['band']}` (positive if Δ_1cm ≥ +0.01)",
        ]
    else:
        md_lines += [
            "- **best_candidate**: 없음 (all_negative)",
            f"- **band**: `{overall['band']}`",
        ]
    md_lines += [
        "",
        "## G3 합격 기준",
        "",
        f"- 17 × 2 metric table 박제 ✓",
        f"- 7 family winner 박제 ✓",
        f"- ≥ 1 후보 paired Δ ≥ +0.005 *둘 다*: {'✓' if out['n_pass'] >= 1 else '✗'} (n_pass = {out['n_pass']})",
        f"- G3 **{'PASS' if out['n_pass'] >= 1 else 'FAIL → all_negative warn'}**.",
    ]
    md_path = THIS / "family_analysis.md"
    md_path.write_text("\n".join(md_lines))
    print(f"wrote {md_path}")
    print()
    print(f"Summary: {out['n_pass']} / {out['n_candidates']} pass, "
          f"overall best = {overall.get('best_candidate') or '없음'}")


if __name__ == "__main__":
    main()
