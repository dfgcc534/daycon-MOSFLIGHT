"""plan-008 STAGE 1 진단 (c2).

spec @ plans/plan-008-candidate-redefine-corrector-redesign.md §4 (v2.5~v2.7).

Outputs:
  - analysis/plan-008/diagnostic.json — summary dict
  - analysis/plan-008/diagnostic.md   — human-readable report

Inputs:
  - data/train_labels.csv, data/train/*
  - analysis/plan-005/corrected_oof.npz  (key: 'corrected')
  - runs/baseline/P001_pb-0-6822-fullrun/oof_selector_scores.npz  (key: 'ens_scores')
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN005_DIR = REPO / "analysis/plan-005"
ANALYSIS_DIR = REPO / "analysis/plan-008"

R_HIT = 0.01


def stage1_diagnostic() -> dict:
    # ── 1. 입력 로드 ──
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1
    assert end_idx >= 4, (
        f"trajectory 길이 부족 — end_idx={end_idx}, jerk/curvature indexing 불가"
    )

    cands = selector.make_candidates(train_x, end_idx, horizon=2)
    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)

    z_oof = np.load(PLAN005_DIR / "corrected_oof.npz")
    corrected_cands = z_oof["corrected"]

    z_scores = np.load(REPO / "runs/baseline/P001_pb-0-6822-fullrun" / "oof_selector_scores.npz")
    oof_scores = z_scores["ens_scores"]

    # ── 2. Residual decomposition (oracle miss sample, v2.6: raw err) ──
    err_raw = np.linalg.norm(cands - train_y[:, None, :], axis=2)
    err = np.linalg.norm(corrected_cands - train_y[:, None, :], axis=2)
    best_idx = err_raw.argmin(axis=1)
    best_pred = cands[np.arange(len(train_y)), best_idx]
    err_vec = best_pred - train_y

    oracle_miss_mask = err_raw.min(axis=1) > R_HIT
    n_oracle_miss = int(oracle_miss_mask.sum())

    p0, d1, acc = selector.motion_terms(train_x, end_idx)
    eps = 1e-8
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + eps)
    err_par = (err_vec * tangent).sum(axis=1)
    err_perp_vec = err_vec - err_par[:, None] * tangent
    err_perp_xy = np.linalg.norm(err_perp_vec[:, :2], axis=1)
    err_z = err_vec[:, 2]

    d2 = train_x[:, end_idx - 1] - train_x[:, end_idx - 2]
    omega_z = np.arctan2(
        d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0],
        d2[:, 0] * d1[:, 0] + d2[:, 1] * d1[:, 1],
    )
    d1_norm = np.linalg.norm(d1, axis=1) + eps
    d2_par_scalar = (d2 * tangent).sum(axis=1)
    d2_perp_vec = d2 - d2_par_scalar[:, None] * tangent
    d2_perp_norm = np.linalg.norm(d2_perp_vec, axis=1)
    curvature = d2_perp_norm / (d1_norm ** 2 + eps)
    prev_acc = d2 - (train_x[:, end_idx - 2] - train_x[:, end_idx - 3])
    jerk_norm = np.linalg.norm(acc - prev_acc, axis=1)

    w = oracle_miss_mask
    err_norm_w = np.linalg.norm(err_vec[w], axis=1)
    corr_rotation = float(np.corrcoef(np.abs(omega_z[w]), err_norm_w)[0, 1])
    corr_curvature = float(np.corrcoef(curvature[w], err_norm_w)[0, 1])
    corr_jerk = float(np.corrcoef(jerk_norm[w], err_norm_w)[0, 1])

    err_par_var = float((err_par[w] ** 2).sum())
    err_perp_var = float((err_perp_xy[w] ** 2).sum())
    err_z_var = float((err_z[w] ** 2).sum())
    total_var = err_par_var + err_perp_var + err_z_var + eps
    par_pct = err_par_var / total_var
    perp_pct = err_perp_var / total_var
    z_pct = err_z_var / total_var

    # informational sanity only — regime sub-breakdown of oracle miss
    oracle_miss_regime_dist: dict[int, dict] = {}
    for r in range(18):
        n_in_miss = int(((regimes == r) & oracle_miss_mask).sum())
        n_total_r = int((regimes == r).sum())
        if n_in_miss > 0:
            oracle_miss_regime_dist[int(r)] = {
                "n_in_miss": n_in_miss,
                "miss_rate": float(n_in_miss / max(n_total_r, 1)),
            }

    dominant_causes: list[dict] = []
    if corr_rotation > 0.3 or perp_pct > 0.4:
        dominant_causes.append({
            "cause": "rotation",
            "evidence": {"corr_rotation": corr_rotation, "perp_pct": perp_pct},
            "recommended_family": "trig",
        })
    if corr_curvature > 0.3:
        dominant_causes.append({
            "cause": "curvature",
            "evidence": {"corr_curvature": corr_curvature},
            "recommended_family": "circular_arc",
        })
    if z_pct > 0.20:
        dominant_causes.append({
            "cause": "z_axis",
            "evidence": {"z_pct": z_pct},
            "recommended_family": "frenet_serret_3d",
        })
    if corr_jerk > 0.3:
        dominant_causes.append({
            "cause": "jerk",
            "evidence": {"corr_jerk": corr_jerk},
            "recommended_family": "higher_order_jerk",
        })

    # ── 3. 가지치기 후보 도출 (structural containment, v2.4 + v2.7 auto-relax) ──
    K_orig = 27
    hit_matrix = (err <= R_HIT)
    hit_rate = hit_matrix.mean(axis=0)
    coord_dist_matrix = np.zeros((K_orig, K_orig))
    containment_soft = np.zeros((K_orig, K_orig))
    containment_strict = np.zeros((K_orig, K_orig), dtype=bool)

    for i in range(K_orig):
        for j in range(K_orig):
            if i == j:
                continue
            coord_dist_matrix[i, j] = float(
                np.linalg.norm(corrected_cands[:, i] - corrected_cands[:, j], axis=1).mean()
            )
            n_i = int(hit_matrix[:, i].sum())
            if n_i == 0:
                containment_soft[i, j] = 1.0
                containment_strict[i, j] = True
            else:
                both = int((hit_matrix[:, i] & hit_matrix[:, j]).sum())
                containment_soft[i, j] = float(both / n_i)
                if both == n_i and int(hit_matrix[:, j].sum()) >= n_i:
                    containment_strict[i, j] = True

    def _identify_prune(soft_thr: float, dist_thr: float) -> list:
        out: list[dict] = []
        for i in range(K_orig):
            for j in range(K_orig):
                if i == j:
                    continue
                strict_ok = bool(containment_strict[i, j] and hit_rate[j] > hit_rate[i])
                soft_ok = bool(
                    containment_soft[i, j] >= soft_thr
                    and coord_dist_matrix[i, j] < dist_thr
                    and hit_rate[j] > hit_rate[i]
                )
                if strict_ok or soft_ok:
                    kept_mask = np.ones(K_orig, dtype=bool)
                    kept_mask[i] = False
                    oracle_after_raw = float(
                        (err_raw[:, kept_mask].min(axis=1) <= R_HIT).mean()
                    )
                    oracle_before_raw = float(
                        (err_raw.min(axis=1) <= R_HIT).mean()
                    )
                    delta = oracle_before_raw - oracle_after_raw
                    if delta < 0.001:
                        out.append({
                            "idx": i,
                            "name": selector.CANDIDATES[i].name,
                            "dominator_idx": j,
                            "dominator_name": selector.CANDIDATES[j].name,
                            "rule": "strict" if strict_ok else "soft",
                            "containment_soft": float(containment_soft[i, j]),
                            "coord_dist": float(coord_dist_matrix[i, j]),
                            "hit_rate_i": float(hit_rate[i]),
                            "hit_rate_j": float(hit_rate[j]),
                            "oracle_delta_if_removed": delta,
                        })
                    break
        return out

    prune_candidates = _identify_prune(soft_thr=0.95, dist_thr=0.005)
    prune_threshold_tier = "strict_v2.4"
    prune_threshold_used = {"soft": 0.95, "dist": 0.005}
    if len(prune_candidates) < 3:
        prune_candidates = _identify_prune(soft_thr=0.90, dist_thr=0.010)
        prune_threshold_tier = "relaxed_v2.7"
        prune_threshold_used = {"soft": 0.90, "dist": 0.010}

    # ── 4. Selector hit gap decomposition ──
    argmax_idx = oof_scores.argmax(axis=1)
    argmax_pred = corrected_cands[np.arange(len(train_y)), argmax_idx]
    argmax_err = np.linalg.norm(argmax_pred - train_y, axis=1)
    selector_argmax_hit = float((argmax_err <= R_HIT).mean())

    # spec 은 boundary.soft_select 를 가리키나 실제 location 은 selector.soft_select.
    # decision-note (spec-default): soft_select import path → selector module.
    soft_pred = selector.soft_select(corrected_cands, oof_scores, temperature=0.03)
    soft_err = np.linalg.norm(soft_pred - train_y, axis=1)
    selector_soft_hit = float((soft_err <= R_HIT).mean())

    oracle_hit = float((err.min(axis=1) <= R_HIT).mean())          # corrected oracle (downstream metric)
    oracle_hit_raw = float((err_raw.min(axis=1) <= R_HIT).mean())   # raw oracle (§1.1 canonical)
    top1_ranking_acc = float((argmax_idx == best_idx).mean())

    gap_ranking = oracle_hit - selector_argmax_hit
    gap_drift = selector_argmax_hit - selector_soft_hit

    selector_gap_decomposition = {
        "oracle_hit_raw": oracle_hit_raw,
        "oracle_hit_corrected": oracle_hit,
        "selector_argmax_hit": selector_argmax_hit,
        "selector_soft_hit": selector_soft_hit,
        "top1_ranking_accuracy": top1_ranking_acc,
        "gap_ranking": gap_ranking,
        "gap_drift": gap_drift,
        "main_bottleneck": "ranking" if gap_ranking > abs(gap_drift) else "drift",
    }

    # ── 5. Softmax diffusion ──
    sorted_scores = np.sort(oof_scores, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    margin_hist = {
        "p10": float(np.percentile(margin, 10)),
        "p25": float(np.percentile(margin, 25)),
        "p50": float(np.percentile(margin, 50)),
        "p75": float(np.percentile(margin, 75)),
        "p90": float(np.percentile(margin, 90)),
    }
    softmax_diffusion_signal = bool(margin_hist["p50"] < 0.1)

    # ── 6. Per-regime oracle (sanity only, decision 무관) ──
    per_regime_oracle: dict[int, dict] = {}
    for r in range(18):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        per_regime_oracle[int(r)] = {
            "n": int(mask.sum()),
            "current_oracle": float((err[mask].min(axis=1) <= R_HIT).mean()),
            "gap_to_target": float(0.85 - (err[mask].min(axis=1) <= R_HIT).mean()),
        }

    # ── 7. 박제 ──
    summary = {
        "mask_strategy": "oracle_miss_v2.5",
        "n_oracle_miss": n_oracle_miss,
        "oracle_miss_rate": float(oracle_miss_mask.mean()),
        "residual_breakdown_oracle_miss": {
            "par_pct": par_pct,
            "perp_pct": perp_pct,
            "z_pct": z_pct,
            "corr_rotation": corr_rotation,
            "corr_curvature": corr_curvature,
            "corr_jerk": corr_jerk,
        },
        "dominant_causes": dominant_causes,
        "prune_candidates": prune_candidates,
        "prune_count": len(prune_candidates),
        "prune_threshold_tier": prune_threshold_tier,
        "prune_threshold_used": prune_threshold_used,
        "selector_gap_decomposition": selector_gap_decomposition,
        "margin_top1_top2": margin_hist,
        "softmax_diffusion_signal": softmax_diffusion_signal,
        "per_regime_oracle_sanity": per_regime_oracle,
        "oracle_miss_regime_dist_sanity": oracle_miss_regime_dist,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "diagnostic.json").write_text(json.dumps(summary, indent=2))
    return summary


def render_markdown(s: dict) -> str:
    g = s["selector_gap_decomposition"]
    rb = s["residual_breakdown_oracle_miss"]
    causes = ", ".join(c["cause"] for c in s["dominant_causes"]) or "(none)"
    lines = [
        "# plan-008 STAGE 1 — Diagnostic (c2)",
        "",
        f"**결론 (1 줄)**: dominant_causes=[{causes}], prune_count={s['prune_count']} "
        f"(tier={s['prune_threshold_tier']}), main_bottleneck={g['main_bottleneck']} "
        f"(gap_ranking={g['gap_ranking']:.4f}, gap_drift={g['gap_drift']:.4f}).",
        "",
        "## Selector gap decomposition",
        "",
        "| metric | value |",
        "|---|---|",
        f"| oracle_hit_raw (best of 27, §1.1) | {g['oracle_hit_raw']:.4f} |",
        f"| oracle_hit_corrected | {g['oracle_hit_corrected']:.4f} |",
        f"| selector_argmax_hit | {g['selector_argmax_hit']:.4f} |",
        f"| selector_soft_hit (temp=0.03) | {g['selector_soft_hit']:.4f} |",
        f"| top1_ranking_accuracy (= argmax == best) | {g['top1_ranking_accuracy']:.4f} |",
        f"| **gap_ranking** (oracle − argmax) | **{g['gap_ranking']:.4f}** |",
        f"| **gap_drift** (argmax − soft) | **{g['gap_drift']:.4f}** |",
        f"| main_bottleneck | **{g['main_bottleneck']}** |",
        "",
        "*top-1 ranking 의 정확한 의미*: 27 후보 중 *raw best (oracle best)* 정확 픽 비율. "
        "1cm 안 들어가는 비율 (= argmax_hit) 과는 다름. 진짜 main metric = `gap_ranking` "
        "(selector 가 hit zone 의 후보를 *놓치는* 비율).",
        "",
        "## Oracle miss residual breakdown (v2.5)",
        "",
        f"- mask: `err_raw.min(axis=1) > 0.01`  →  n_oracle_miss = {s['n_oracle_miss']} "
        f"(miss_rate = {s['oracle_miss_rate']:.4f})",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| par_pct (tangent 분산) | {rb['par_pct']:.3f} |",
        f"| perp_pct (xy 수직 분산) | {rb['perp_pct']:.3f} |",
        f"| z_pct | {rb['z_pct']:.3f} |",
        f"| corr_rotation (\\|omega_z\\| vs err) | {rb['corr_rotation']:.3f} |",
        f"| corr_curvature (kinematic K vs err) | {rb['corr_curvature']:.3f} |",
        f"| corr_jerk (\\|jerk\\| vs err) | {rb['corr_jerk']:.3f} |",
        "",
        f"**dominant_causes**: {json.dumps(s['dominant_causes'], indent=2, ensure_ascii=False)}",
        "",
        "## 가지치기 후보 (structural containment, v2.4 + v2.7 auto-relax)",
        "",
        f"tier = `{s['prune_threshold_tier']}`, "
        f"soft_thr = {s['prune_threshold_used']['soft']}, "
        f"dist_thr = {s['prune_threshold_used']['dist']*1000:.0f}mm, "
        f"count = {s['prune_count']}",
        "",
        "| i | name | dom_idx | dom_name | rule | cont_soft | coord_dist (m) | hr_i | hr_j | oracle_δ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in s["prune_candidates"]:
        lines.append(
            f"| {p['idx']} | {p['name']} | {p['dominator_idx']} | {p['dominator_name']} | "
            f"{p['rule']} | {p['containment_soft']:.3f} | {p['coord_dist']:.4f} | "
            f"{p['hit_rate_i']:.3f} | {p['hit_rate_j']:.3f} | {p['oracle_delta_if_removed']:.5f} |"
        )
    lines += [
        "",
        "## margin (top1 − top2) 분포 (logit 단위)",
        "",
        "| pct | value |",
        "|---|---|",
    ]
    for k in ("p10", "p25", "p50", "p75", "p90"):
        lines.append(f"| {k} | {s['margin_top1_top2'][k]:.4f} |")
    lines += [
        "",
        f"softmax_diffusion_signal (p50 < 0.1) = `{s['softmax_diffusion_signal']}`",
        "",
        "## Per-regime oracle (sanity only, decision 무관)",
        "",
        "| regime | n | current_oracle | gap_to_0.85 |",
        "|---|---|---|---|",
    ]
    for r, v in sorted(s["per_regime_oracle_sanity"].items()):
        lines.append(
            f"| {r} | {v['n']} | {v['current_oracle']:.3f} | {v['gap_to_target']:+.3f} |"
        )
    lines += [
        "",
        "## Oracle miss regime distribution (sanity only)",
        "",
        "| regime | n_in_miss | miss_rate |",
        "|---|---|---|",
    ]
    for r, v in sorted(s["oracle_miss_regime_dist_sanity"].items()):
        lines.append(f"| {r} | {v['n_in_miss']} | {v['miss_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    s = stage1_diagnostic()
    md = render_markdown(s)
    (ANALYSIS_DIR / "diagnostic.md").write_text(md)
    g = s["selector_gap_decomposition"]
    print(
        f"[c2] n_oracle_miss={s['n_oracle_miss']} "
        f"miss_rate={s['oracle_miss_rate']:.4f} "
        f"prune_count={s['prune_count']}({s['prune_threshold_tier']}) "
        f"main_bottleneck={g['main_bottleneck']} "
        f"gap_ranking={g['gap_ranking']:.4f} gap_drift={g['gap_drift']:.4f}"
    )


if __name__ == "__main__":
    main()
