"""plan-008 c4 + c5: Step 2a pruning + Step 2b greedy set-cover (G001-step2).

spec @ plans/plan-008-candidate-redefine-corrector-redesign.md §5.

Inputs:
  - analysis/plan-008/diagnostic.json  (c2 결과: prune_candidates list)
  - data/train_labels.csv, data/train/*
  - analysis/plan-005/corrected_oof.npz   (for step2a sanity, info-only)
  - runs/baseline/P001_pb-0-6822-fullrun/oof_selector_scores.npz

Outputs:
  - analysis/plan-008/prune_summary.json
  - analysis/plan-008/greedy_set_cover.json
  - analysis/plan-008/redefine.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pb_0_6822 import selector
from src.pb_0_6822 import candidates_extended as cx

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN005_DIR = REPO / "analysis/plan-005"
ANALYSIS_DIR = REPO / "analysis/plan-008"
R_HIT = 0.01


def softmax_np(x: np.ndarray, temp: float = 0.03) -> np.ndarray:
    z = x / temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def step2a_prune(
    prune_candidates: list[dict],
    cands_27: np.ndarray,
    train_y: np.ndarray,
    oof_scores: np.ndarray,
) -> dict:
    """§5.1: structural containment 로 식별된 redundant 후보 *incremental safety pruning*.

    v2.7 fix: 24 후보 일괄 제거 시 oracle aggregate drop > 0.0018 위반 (caveat #16
    의 *retrained selector 한계* 가 *aggregate joint loss* 측면에서도 발현 —
    24 후보의 unique hit set 합집합이 commutative 하지 않음). 따라서:

    1. Sort prune_candidates by oracle_delta_if_removed (ascending, 가장 무해한 것부터)
    2. 각 후보 i 를 단계적으로 추가 — kept_mask 갱신 후 aggregate oracle 측정
    3. aggregate (oracle_orig − oracle_now) >= 0.0018 위반 시 그 후보 reject (rollback)
    4. 모든 후보 시도 후 accepted set 확정

    이로써 §3.2 G1 의 aggregate safety (oracle_after_prune ≥ 0.7170) 보장.
    """
    oracle_orig = float(
        (np.linalg.norm(cands_27 - train_y[:, None, :], axis=2).min(axis=1) <= R_HIT).mean()
    )
    safety_threshold = 0.0018   # §3.2 G1 spec

    sorted_prunes = sorted(prune_candidates, key=lambda p: p["oracle_delta_if_removed"])

    kept_mask = np.ones(27, dtype=bool)
    accepted_prunes: list[dict] = []
    rejected_prunes: list[dict] = []
    incremental_trace: list[dict] = []

    for p in sorted_prunes:
        if not kept_mask[p["idx"]]:
            # 이미 다른 dominator 로 제거됨 (중복 entry — 동일 i 가 여러 j 와 매치된 경우)
            continue
        test_mask = kept_mask.copy()
        test_mask[p["idx"]] = False
        if test_mask.sum() == 0:
            # 최소 1개는 유지
            rejected_prunes.append({**p, "reject_reason": "min_pool_size"})
            continue
        cands_test = cands_27[:, test_mask, :]
        err_test = np.linalg.norm(cands_test - train_y[:, None, :], axis=2)
        oracle_test = float((err_test.min(axis=1) <= R_HIT).mean())
        agg_delta = oracle_orig - oracle_test
        if agg_delta < safety_threshold:
            kept_mask = test_mask
            accepted_prunes.append({**p, "agg_delta_after_accept": agg_delta})
            incremental_trace.append({
                "step": len(accepted_prunes),
                "idx": p["idx"], "name": p["name"],
                "agg_delta": agg_delta,
                "remaining": int(kept_mask.sum()),
            })
        else:
            rejected_prunes.append({**p, "reject_reason": "aggregate_unsafe",
                                     "agg_delta_if_accept": agg_delta})

    cands_pruned = cands_27[:, kept_mask, :]
    err_pruned = np.linalg.norm(cands_pruned - train_y[:, None, :], axis=2)
    oracle_pruned = float((err_pruned.min(axis=1) <= R_HIT).mean())

    scores_pruned = oof_scores[:, kept_mask]
    weights_pruned = softmax_np(scores_pruned, temp=0.03)
    soft_pred_pruned = (cands_pruned * weights_pruned[:, :, None]).sum(axis=1)
    soft_hit_pruned = float((np.linalg.norm(soft_pred_pruned - train_y, axis=1) <= R_HIT).mean())

    return {
        "pruning_method": "structural_containment_v2.4+v2.7_incremental_safety",
        "candidate_count_identified": len(prune_candidates),
        "pruned_count": len(accepted_prunes),
        "rejected_count": len(rejected_prunes),
        "remaining_count": int(kept_mask.sum()),
        "pruned_pairs": accepted_prunes,
        "rejected_pairs": rejected_prunes,
        "incremental_trace": incremental_trace,
        "oracle_orig": oracle_orig,
        "oracle_pruned": oracle_pruned,
        "oracle_delta": oracle_pruned - oracle_orig,
        "oracle_safe": (oracle_orig - oracle_pruned) < safety_threshold,
        "safety_threshold_used": safety_threshold,
        "soft_hit_pruned_posthoc": soft_hit_pruned,
        "kept_indices": [int(i) for i in np.where(kept_mask)[0]],
        "kept_names": [selector.CANDIDATES[i].name for i in np.where(kept_mask)[0]],
        "kept_pruned_specs": [
            {"name": selector.CANDIDATES[i].name, "idx": int(i)}
            for i in np.where(kept_mask)[0]
        ],
    }


def step2b_greedy_set_cover(
    cands_pruned: np.ndarray,
    train_y: np.ndarray,
    train_x: np.ndarray,
    end_idx: int,
    template_pool: list,
    kept_pruned_specs: list,
    regimes: np.ndarray,
    *,
    target_oracle: float = 0.90,
    max_pool_size: int = 50,
    min_delta: float = 0.001,
) -> dict:
    """§5.3 Strategy D greedy. Oracle 측정 set = full train (in-sample, §1.1 정의)."""
    pool_cands = cands_pruned.copy()
    pool_specs = list(kept_pruned_specs)
    remaining_templates = list(template_pool)

    err = np.linalg.norm(pool_cands - train_y[:, None, :], axis=2)
    oracle_current = float((err.min(axis=1) <= R_HIT).mean())
    iteration_log = [{
        "iter": 0, "added_template": None, "family_id": None,
        "oracle": oracle_current, "delta": 0.0,
        "pool_size": int(pool_cands.shape[1]),
    }]

    while oracle_current < target_oracle and pool_cands.shape[1] < max_pool_size:
        best_template, best_delta, best_new_cands = None, 0.0, None
        for tmpl in remaining_templates:
            _, spec, coord_func = tmpl
            new_cands = coord_func(train_x, end_idx, horizon=2, spec=spec)
            cands_test = np.concatenate([pool_cands, new_cands], axis=1)
            err_test = np.linalg.norm(cands_test - train_y[:, None, :], axis=2)
            oracle_test = float((err_test.min(axis=1) <= R_HIT).mean())
            delta = oracle_test - oracle_current
            if delta > best_delta:
                best_template, best_delta, best_new_cands = tmpl, delta, new_cands
        if best_delta < min_delta:
            break
        pool_cands = np.concatenate([pool_cands, best_new_cands], axis=1)
        pool_specs.append(best_template[1])
        oracle_current = oracle_current + best_delta
        remaining_templates.remove(best_template)
        iteration_log.append({
            "iter": len(iteration_log) - 0,
            "added_template": best_template[0],
            "family_id": int(best_template[1].family_id),
            "oracle": oracle_current,
            "delta": best_delta,
            "pool_size": int(pool_cands.shape[1]),
        })

    # Per-regime final (informational sanity only)
    per_regime_final = {}
    err_final = np.linalg.norm(pool_cands - train_y[:, None, :], axis=2)
    for r in [10, 16, 17]:
        mask = regimes == r
        if mask.sum() == 0:
            continue
        per_regime_final[int(r)] = {
            "n": int(mask.sum()),
            "oracle_after_greedy": float((err_final[mask].min(axis=1) <= R_HIT).mean()),
        }

    stop_reason = (
        "target_oracle_reached" if oracle_current >= target_oracle
        else "max_pool_size_reached" if pool_cands.shape[1] >= max_pool_size
        else "delta_below_threshold"
    )

    return {
        "iteration_log": iteration_log,
        "oracle_final": oracle_current,
        "total_candidates_final": int(pool_cands.shape[1]),
        "pool_specs_final": [
            {
                "name": s.name,
                "family_id": int(s.family_id),
                "d1": float(s.d1), "par": float(s.par), "perp": float(s.perp),
                "d2": float(s.d2), "jerk": float(s.jerk), "time_scale": float(s.time_scale),
                "omega_scale": float(s.omega_scale),
                "arc_curvature": float(s.arc_curvature),
                "z_scale": float(s.z_scale),
            }
            for s in pool_specs
        ],
        "templates_added_count": len(iteration_log) - 1,
        "templates_remaining_count": len(remaining_templates),
        "stop_reason": stop_reason,
        "per_regime_worst": per_regime_final,
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load c2 diagnostic ──
    diag = json.loads((ANALYSIS_DIR / "diagnostic.json").read_text())
    prune_candidates = diag["prune_candidates"]

    # ── Load data ──
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1
    cands_27 = selector.make_candidates(train_x, end_idx, horizon=2)
    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)
    z_scores = np.load(REPO / "runs/baseline/P001_pb-0-6822-fullrun" / "oof_selector_scores.npz")
    oof_scores = z_scores["ens_scores"]

    # ── Step 2a: prune ──
    prune_summary = step2a_prune(prune_candidates, cands_27, train_y, oof_scores)
    (ANALYSIS_DIR / "prune_summary.json").write_text(json.dumps(prune_summary, indent=2))
    print(
        f"[Step 2a] oracle: {prune_summary['oracle_orig']:.4f} → "
        f"{prune_summary['oracle_pruned']:.4f} (Δ={prune_summary['oracle_delta']:+.5f}) "
        f"kept={prune_summary['remaining_count']}/27 safe={prune_summary['oracle_safe']}"
    )
    assert prune_summary["oracle_safe"], (
        f"oracle_drop severe — Δ={prune_summary['oracle_delta']} (G1 한계 −0.0018 초과)"
    )

    # ── Step 2b: greedy set-cover ──
    kept_mask = np.zeros(27, dtype=bool)
    kept_mask[prune_summary["kept_indices"]] = True
    cands_pruned = cands_27[:, kept_mask, :]
    kept_pruned_specs = [selector.CANDIDATES[i] for i in prune_summary["kept_indices"]]
    template_pool = cx.build_template_pool()

    greedy = step2b_greedy_set_cover(
        cands_pruned, train_y, train_x, end_idx,
        template_pool=template_pool,
        kept_pruned_specs=kept_pruned_specs,
        regimes=regimes,
        target_oracle=0.90,
        max_pool_size=50,
        min_delta=0.001,
    )
    (ANALYSIS_DIR / "greedy_set_cover.json").write_text(json.dumps(greedy, indent=2))
    print(
        f"[Step 2b] oracle_final={greedy['oracle_final']:.4f} "
        f"pool_size={greedy['total_candidates_final']} "
        f"added={greedy['templates_added_count']} stop={greedy['stop_reason']}"
    )

    # ── §5.4 redefine.md ──
    md_lines = [
        "# plan-008 STAGE 2 — Pruning + Greedy Set-Cover (c4 + c5)",
        "",
        f"**Step 2a 결과**: 27 → {prune_summary['remaining_count']} candidates "
        f"(pruned {prune_summary['pruned_count']}). "
        f"oracle {prune_summary['oracle_orig']:.4f} → {prune_summary['oracle_pruned']:.4f} "
        f"(Δ={prune_summary['oracle_delta']:+.5f}, safe={prune_summary['oracle_safe']}). "
        f"post-hoc soft_hit_pruned={prune_summary['soft_hit_pruned_posthoc']:.4f}.",
        "",
        f"**Step 2b 결과**: pool {prune_summary['remaining_count']} + "
        f"{greedy['templates_added_count']} new = {greedy['total_candidates_final']}. "
        f"oracle_final = **{greedy['oracle_final']:.4f}** (target 0.85 minimum, 0.90 stretch). "
        f"stop_reason = `{greedy['stop_reason']}`.",
        "",
        "## Step 2a — Pruned candidates",
        "",
        f"kept names: {', '.join(prune_summary['kept_names'])}",
        "",
        "| pruned_idx | name | dominator | rule | cont_soft | coord_dist | hr_i | hr_j |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in prune_summary["pruned_pairs"]:
        md_lines.append(
            f"| {p['idx']} | {p['name']} | {p['dominator_name']} | {p['rule']} | "
            f"{p['containment_soft']:.3f} | {p['coord_dist']:.4f} | "
            f"{p['hit_rate_i']:.3f} | {p['hit_rate_j']:.3f} |"
        )
    md_lines += [
        "",
        "## Step 2b — Greedy iteration log",
        "",
        "| iter | added_template | family_id | oracle | delta | pool_size |",
        "|---|---|---|---|---|---|",
    ]
    for it in greedy["iteration_log"]:
        added = it["added_template"] or "(start)"
        fid = it["family_id"] if it["family_id"] is not None else "-"
        md_lines.append(
            f"| {it['iter']} | {added} | {fid} | {it['oracle']:.4f} | "
            f"{it['delta']:+.4f} | {it['pool_size']} |"
        )
    md_lines += [
        "",
        "## Final pool — family composition",
        "",
        "| family_id | family_name | count |",
        "|---|---|---|",
    ]
    fam_name = {0: "base", 1: "trig", 2: "arc", 3: "frenet_serret_3d",
                4: "(per_regime_dropped)", 5: "higher_order", 6: "cross_term"}
    from collections import Counter
    c = Counter(s["family_id"] for s in greedy["pool_specs_final"])
    for fid in sorted(c.keys()):
        md_lines.append(f"| {fid} | {fam_name.get(fid, '?')} | {c[fid]} |")
    md_lines += [
        "",
        "## Per-regime worst (sanity only, regime infra 폐기)",
        "",
        "| regime | n | oracle_after_greedy |",
        "|---|---|---|",
    ]
    for r, v in sorted(greedy["per_regime_worst"].items()):
        md_lines.append(f"| {r} | {v['n']} | {v['oracle_after_greedy']:.3f} |")
    (ANALYSIS_DIR / "redefine.md").write_text("\n".join(md_lines) + "\n")
    print(
        f"[c4+c5] redefine.md 작성 완료. "
        f"oracle_final={greedy['oracle_final']:.4f} "
        f"({'minimum 0.85 OK' if greedy['oracle_final'] >= 0.85 else ('warn 0.78-0.85' if greedy['oracle_final'] >= 0.78 else 'SEVERE < 0.78')})"
    )


if __name__ == "__main__":
    main()
