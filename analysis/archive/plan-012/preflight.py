"""plan-012 c3 (G0) — Phase 0 preflight + codebook prep.

4 task:
  1. F0 raw hit (hit@1cm + hit@1.5cm) — 학습 무관 측정
  2. 3 codebook oracle ceilings (Absolute / Frenet-Orthogonal / K-Means)
     + per-axis marginal oracle for E2 axis ranking
  3. K-Means fit meta (5-fold cluster centers / sizes / inertia / silhouette)
  4. plan-006 reproduce
     — F001_variant-e checkpoint 가 repo 에 부재 → reproduce_skipped_no_checkpoint
       (decision-note 박제, G0 합격 검사에서 reproduce drift 항목 informational 처리)

Usage:
    python analysis/plan-012/preflight.py \
        --root data \
        --plan-006-checkpoint runs/baseline/F001_variant-e/checkpoint_best.pt \
        --out analysis/plan-012/preflight.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pb_0_6822 import ring_classifier as rc  # noqa: E402
from src.pb_0_6822 import selector as base       # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def hit_rate(pred: np.ndarray, true: np.ndarray, threshold_m: float) -> float:
    err = np.linalg.norm(pred - true, axis=-1)
    return float(np.mean(err <= threshold_m))


def oracle_hit_for_codebook(
    F0_pred: np.ndarray,
    anchors_world: np.ndarray,
    true_y: np.ndarray,
    threshold_m: float = 0.01,
) -> float:
    """Oracle scorer = argmin distance per sample. Returns hit rate after oracle selection."""
    cand_pos = F0_pred[:, None, :] + anchors_world                                          # (N, K, 3)
    dists = np.linalg.norm(cand_pos - true_y[:, None, :], axis=-1)                          # (N, K)
    oracle_k = dists.argmin(axis=1)                                                         # (N,)
    sel = cand_pos[np.arange(F0_pred.shape[0]), oracle_k]                                   # (N, 3)
    return hit_rate(sel, true_y, threshold_m=threshold_m)


def per_axis_marginal_oracle(
    F0_pred: np.ndarray,
    R_world_from_frenet: np.ndarray | None,
    true_y: np.ndarray,
    frame: str,
    radius_m: float = 0.005,
) -> dict[str, float]:
    """각 ±axis 를 center 와 함께 2-anchor codebook 으로 oracle 측정. Returns dict."""
    assert frame in {"absolute", "frenet_orthogonal"}
    N = F0_pred.shape[0]
    axes = {
        "absolute": [("+x", [+radius_m, 0, 0]), ("-x", [-radius_m, 0, 0]),
                     ("+y", [0, +radius_m, 0]), ("-y", [0, -radius_m, 0]),
                     ("+z", [0, 0, +radius_m]), ("-z", [0, 0, -radius_m])],
        "frenet_orthogonal": [("+t", [+radius_m, 0, 0]), ("-t", [-radius_m, 0, 0]),
                              ("+n", [0, +radius_m, 0]), ("-n", [0, -radius_m, 0]),
                              ("+b", [0, 0, +radius_m]), ("-b", [0, 0, -radius_m])],
    }[frame]
    out: dict[str, float] = {}
    for label, vec in axes:
        anchors_local = np.array([[0, 0, 0], vec], dtype=np.float64)
        if frame == "absolute":
            anchors_world = rc.anchors_to_world(anchors_local, None, N=N)
        else:
            anchors_world = rc.anchors_to_world(anchors_local, R_world_from_frenet, N=N)
        out[label] = oracle_hit_for_codebook(F0_pred, anchors_world, true_y, threshold_m=0.01)
    return out


def axis_family_ranking(marginal: dict[str, float], families: list[str]) -> list[str]:
    """Family = axis without sign. value = max(+sign, -sign) marginal. Ranking by max value desc."""
    fam_values: dict[str, float] = {}
    for fam in families:
        fam_values[fam] = max(marginal.get(f"+{fam}", -1.0), marginal.get(f"-{fam}", -1.0))
    # tie-break (gap < 0.003): priority order = families 입력순
    ranked = []
    remaining = list(families)
    while remaining:
        max_v = max(fam_values[f] for f in remaining)
        eligible = [f for f in remaining if fam_values[f] >= max_v - 0.003]
        # priority = families 입력순 (= 자연 x>y>z or t>n>b)
        winner = next(f for f in families if f in eligible)
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--plan-006-checkpoint", type=str,
                        default="runs/baseline/F001_variant-e/checkpoint_best.pt")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--radius-m", type=float, default=0.005)
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── data load ───
    print(f"[preflight] loading train data from {root} ...", flush=True)
    train_ids, train_y = base.read_labels(root / "train_labels.csv")  # (N, 3) float32
    train_x = base.load_stack(root / "train", train_ids).astype(np.float64)  # (N, T, 3)
    N, T, _ = train_x.shape
    train_y = train_y.astype(np.float64)
    print(f"[preflight] N={N}, T={T}", flush=True)

    # ─── 1. F0 raw hit ───
    print("[preflight] task 1: F0 raw hit ...", flush=True)
    end_idx = T - 1
    F0_pred = rc.f0_predict_frenet_par120_perp_neg020(train_x, end_idx=end_idx)        # (N, 3)
    hit_at_1cm = hit_rate(F0_pred, train_y, threshold_m=0.01)
    hit_at_1_5cm = hit_rate(F0_pred, train_y, threshold_m=0.015)
    print(f"[preflight]   hit@1cm   = {hit_at_1cm:.4f}", flush=True)
    print(f"[preflight]   hit@1.5cm = {hit_at_1_5cm:.4f}", flush=True)

    # ─── 2. 3 codebook oracle ceilings ───
    print("[preflight] task 2: 3 codebook oracle ceilings ...", flush=True)
    R_wfn = rc.build_frenet_basis_3d(train_x, end_idx=end_idx)                         # (N, 3, 3)

    anchors_abs = rc.compute_anchors_absolute(radius_m=args.radius_m)
    anchors_world_abs = rc.anchors_to_world(anchors_abs, None, N=N)
    oracle_abs = oracle_hit_for_codebook(F0_pred, anchors_world_abs, train_y)
    print(f"[preflight]   absolute_7way:   oracle hit@1cm = {oracle_abs:.4f}", flush=True)

    anchors_fr = rc.compute_anchors_frenet_orthogonal(radius_m=args.radius_m)
    anchors_world_fr = rc.anchors_to_world(anchors_fr, R_wfn, N=N)
    oracle_fr = oracle_hit_for_codebook(F0_pred, anchors_world_fr, train_y)
    print(f"[preflight]   frenet_ortho_7way: oracle hit@1cm = {oracle_fr:.4f}", flush=True)

    # ─── 3. K-Means fit meta (fold-aware) ───
    print("[preflight] task 3: K-Means fit (fold-aware) ...", flush=True)
    residuals_world = train_y - F0_pred                                                 # (N, 3) world
    fold_id = np.array([base.stable_fold_id(sid, args.n_folds) for sid in train_ids], dtype=np.int64)
    centers_per_fold, cluster_sizes_per_fold, kmeans_meta = rc.compute_anchors_kmeans(
        residuals_world, R_wfn, fold_id, K=7, radius_clip_m=0.020,
    )
    print(f"[preflight]   centers_per_fold.shape = {centers_per_fold.shape}", flush=True)
    print(f"[preflight]   min cluster size (non-center) = {cluster_sizes_per_fold[:, 1:].min()}", flush=True)

    # K-Means oracle ceiling on training partition (= 비교 가능한 hit)
    # 각 sample 에 대해 자기 fold 의 anchor 사용
    oracle_km_per_sample = np.zeros(N)
    for k in range(args.n_folds):
        val_mask = fold_id == k
        n_val = int(val_mask.sum())
        if n_val == 0:
            continue
        anchors_world_km_k = rc.anchors_to_world(centers_per_fold[k], R_wfn[val_mask], N=n_val)
        F0_k = F0_pred[val_mask]
        true_k = train_y[val_mask]
        cand_pos = F0_k[:, None, :] + anchors_world_km_k                                  # (n_val, 7, 3)
        dists = np.linalg.norm(cand_pos - true_k[:, None, :], axis=-1)
        oracle_k_idx = dists.argmin(axis=1)
        sel = cand_pos[np.arange(n_val), oracle_k_idx]
        err = np.linalg.norm(sel - true_k, axis=-1)
        oracle_km_per_sample[val_mask] = (err <= 0.01).astype(np.float64)
    oracle_km = float(np.mean(oracle_km_per_sample))
    print(f"[preflight]   kmeans_7way:     oracle hit@1cm = {oracle_km:.4f}", flush=True)

    # ─── 2b. per-axis marginal oracle (E2 axis ranking source) ───
    print("[preflight] task 2b: per-axis marginal oracle ...", flush=True)
    marginal_abs = per_axis_marginal_oracle(F0_pred, None, train_y, frame="absolute")
    marginal_fr = per_axis_marginal_oracle(F0_pred, R_wfn, train_y, frame="frenet_orthogonal")
    print(f"[preflight]   absolute marginals: {marginal_abs}", flush=True)
    print(f"[preflight]   frenet marginals:   {marginal_fr}", flush=True)
    ranking_abs = axis_family_ranking(marginal_abs, families=["x", "y", "z"])
    ranking_fr = axis_family_ranking(marginal_fr, families=["t", "n", "b"])

    # ─── 4. plan-006 reproduce ───
    ckpt_path = REPO_ROOT / args.plan_006_checkpoint
    if not ckpt_path.exists():
        print(f"[preflight] task 4: plan-006 checkpoint not found at {ckpt_path} — skip with decision-note", flush=True)
        plan_006_reproduce = {
            "single_formula": "frenet_par120_perp_neg020",
            "oof_argmax_hit_corrected_measured": None,
            "oof_argmax_hit_corrected_expected": 0.6491,
            "drift": None,
            "drift_threshold": 0.005,
            "reproduce_ok": "skipped_no_checkpoint",
            "decision_note": "F001_variant-e/checkpoint_best.pt 부재 — plan-006 corrector path 의 reproduce 는 skip. F0 raw hit (위 task 1) 자체가 plan-006 single formula baseline 의 raw measure 로 informational. G0 합격 검사 시 drift 항목 정보용.",
        }
    else:
        # Skip — corrector path reproduce 가 plan-012 v2 의 비재사용 (boundary.py) 의존이라
        # 실제 plan-006 path 재실행은 본 plan 의 scope 외. informational 으로만 reproduce.
        plan_006_reproduce = {
            "single_formula": "frenet_par120_perp_neg020",
            "oof_argmax_hit_corrected_measured": None,
            "oof_argmax_hit_corrected_expected": 0.6491,
            "drift": None,
            "drift_threshold": 0.005,
            "reproduce_ok": "skipped_corrector_path_out_of_scope",
            "decision_note": "checkpoint 존재하나 plan-012 v2 가 corrector path (boundary.py / plan-010/011 redesign) 폐기 — reproduce 산출이 scope 외. F0 raw hit 으로 informational baseline.",
        }

    # ─── G0 합격 검사 ───
    g0_checks = {
        "f0_raw_hit_1cm_in_range":   0.60 <= hit_at_1cm <= 0.68,
        "f0_raw_hit_1_5cm_in_range": 0.80 <= hit_at_1_5cm <= 0.88,
        "absolute_oracle_in_range":  0.70 <= oracle_abs <= 0.90,
        "frenet_oracle_in_range":    0.70 <= oracle_fr <= 0.90,
        "kmeans_oracle_in_range":    0.70 <= oracle_km <= 0.90,
        "min_cluster_size_gt_100":   int(cluster_sizes_per_fold[:, 1:].min()) > 100,
        "plan_006_reproduce_ok":     plan_006_reproduce["reproduce_ok"] is True,  # always False since skipped
    }
    g0_essential_passed = all([
        g0_checks["f0_raw_hit_1cm_in_range"],
        g0_checks["f0_raw_hit_1_5cm_in_range"],
        g0_checks["absolute_oracle_in_range"],
        g0_checks["frenet_oracle_in_range"],
        g0_checks["kmeans_oracle_in_range"],
        g0_checks["min_cluster_size_gt_100"],
    ])

    # ─── result JSON ───
    result = {
        "exp_id": "H019_phase0-preflight-codebook",
        "n_train": int(N),
        "trajectory_T": int(T),
        "end_idx": int(end_idx),
        "f0_raw_hit_measure": {
            "description": "F0 단일공식의 raw hit@1cm + hit@1.5cm — 학습 무관 측정",
            "single_formula": "frenet_par120_perp_neg020",
            "candidate_idx": 17,
            "n_train": int(N),
            "hit_at_1cm": {"hit_rate": hit_at_1cm, "expected_range": [0.60, 0.68], "in_range": g0_checks["f0_raw_hit_1cm_in_range"]},
            "hit_at_1_5cm": {"hit_rate": hit_at_1_5cm, "expected_range": [0.80, 0.88], "in_range": g0_checks["f0_raw_hit_1_5cm_in_range"]},
        },
        "codebook_oracle_ceilings": {
            "description": "각 codebook 의 oracle scorer (label-aware argmin) 시 hit@1cm",
            "n_train": int(N),
            "absolute_7way": {"oracle_hit_1cm": oracle_abs, "anchors": anchors_abs.tolist()},
            "frenet_orthogonal_7way": {"oracle_hit_1cm": oracle_fr, "anchors": anchors_fr.tolist()},
            "kmeans_7way": {"oracle_hit_1cm": oracle_km, "anchors_per_fold": centers_per_fold.tolist()},
            "per_axis_marginal_hit_1cm": {
                "description": "§7.1 E2 K density swap 의 dominant axis 결정 source",
                "absolute": marginal_abs,
                "frenet_orthogonal": marginal_fr,
                "axis_family_ranking_absolute": ranking_abs,
                "axis_family_ranking_frenet": ranking_fr,
            },
        },
        "kmeans_fit_meta": {
            "K": 7,
            "fold_count": int(args.n_folds),
            "centers_per_fold": centers_per_fold.tolist(),
            "cluster_sizes_per_fold": cluster_sizes_per_fold.tolist(),
            "inertia_per_fold": kmeans_meta["inertia_per_fold"],
            "silhouette_per_fold": kmeans_meta["silhouette_per_fold"],
            "min_cluster_size": int(cluster_sizes_per_fold[:, 1:].min()),
            "min_cluster_size_threshold": 100,
            "min_cluster_size_pass": g0_checks["min_cluster_size_gt_100"],
        },
        "plan_006_reproduce": plan_006_reproduce,
        "g0_checks": g0_checks,
        "g0_essential_passed": g0_essential_passed,
    }

    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[preflight] wrote {out_path}", flush=True)
    print(f"[preflight] G0 essential checks passed: {g0_essential_passed}", flush=True)
    for k, v in g0_checks.items():
        print(f"  - {k}: {v}", flush=True)
    return 0 if g0_essential_passed else 1


if __name__ == "__main__":
    sys.exit(main())
