"""plan-014 c4 (STAGE 0, G0) — preflight artifact (v4 spec).

5 task 일괄 실행 (§4 spec, v4 narrative):
  (a) F0 frozen reproduce — plan-006 frenet_par120_perp_neg020 constants
      (d1=1.98, par=1.20, perp=−0.20) 으로 train sample hit@1cm 측정 →
      plan-006 reference 0.6320 ± 0.005 일치
  (b) 3 codebook oracle ceiling (E0a Absolute / E0b Frenet-Orthogonal /
      E0c K-Means at radius=0.01m, hindsight label-aware oracle)
  (c) Gaussian σ=0.01m soft label entropy 평균 (target w_k 분포, 학습 전
      분석적 산출)
  (d) plan-012 results.md INVALID_REFERENCE disclaimer grep
  (e) per-axis marginal oracle ordering (§7.1 E2 K=5/9/13 anchor source)

재사용 끊김 (§0.5 v4):
  - `src.pb_0_6822.{selector,ring_classifier,boundary}` + plan-006 numpy F0
    함수 import 0.
  - F0 산식 / Frenet basis / anchor / K-Means 본 스크립트 안 재구현.
  - `src.io` (plan-001 utility) 만 import OK.
  - **F0 = frozen** (plain numpy function, requires_grad 개념 없음).

Usage:
    python analysis/plan-014/preflight.py \
        --root         data \
        --out          analysis/plan-014/preflight.json \
        --plan-012-ref plans/plan-012-frenet-ring-classification.results.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402

# F0 frozen constants (plan-006 frenet_par120_perp_neg020 — hard evidence carry)
F0_D1 = 1.98
F0_PAR = 1.20
F0_PERP = -0.20

# anchor scale (§2.1.A C3)
ANCHOR_RADIUS = 0.01

# soft label σ (§2.1.A C4)
SOFT_SIGMA = 0.01

# numeric stability
EPS = 1e-12
EPS_BASIS = 1e-6

# K-Means hyperparam (§2.1.B.1, plan-012 carry)
N_FOLDS = 5
KMEANS_K = 7
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 20260606
KMEANS_RADIUS_CLIP = 0.020
KMEANS_MIN_CLUSTER_SIZE_THRESHOLD = 100

# G0 합격 spec (§4.3)
F0_HIT_LO = 0.6270
F0_HIT_HI = 0.6370
SOFT_ENTROPY_MIN = 0.5


def stable_hash_fold(sample_id: str, n_folds: int = N_FOLDS, salt: str = "plan-014-v1") -> int:
    """§3.1: SHA256(f'{salt}::{sample_id}') → int.from_bytes([:8]) % n_folds."""
    digest = hashlib.sha256(f"{salt}::{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


# ──────────────────────────────────────────────────────────────────────────────
# Frenet finite-diff + basis (§2.1.A.1)
# ──────────────────────────────────────────────────────────────────────────────


def finite_diff_at(X: np.ndarray, end_idx: int) -> dict[str, np.ndarray]:
    v_last = X[:, end_idx] - X[:, end_idx - 1]
    v_prev = X[:, end_idx - 1] - X[:, end_idx - 2]
    v_prev2 = X[:, end_idx - 2] - X[:, end_idx - 3]
    acc = v_last - v_prev
    prev_acc = v_prev - v_prev2
    jerk = acc - prev_acc
    return {"v_last": v_last, "v_prev": v_prev, "acc": acc, "prev_acc": prev_acc, "jerk": jerk}


def build_frenet_basis_3d(X: np.ndarray, end_idx: int = 10) -> np.ndarray:
    """X: (N, T, 3). Returns R_world_from_frenet: (N, 3, 3), columns [t̂ | n̂ | b̂]."""
    fd = finite_diff_at(X, end_idx)
    v_last, acc = fd["v_last"], fd["acc"]
    N = X.shape[0]

    t_hat = v_last / (np.linalg.norm(v_last, axis=1, keepdims=True) + EPS)
    acc_par_scalar = np.sum(acc * t_hat, axis=1, keepdims=True)
    acc_perp_vec = acc - acc_par_scalar * t_hat
    perp_norm = np.linalg.norm(acc_perp_vec, axis=1, keepdims=True)

    degenerate = (perp_norm < EPS_BASIS).squeeze(-1)
    n_hat = np.where(
        perp_norm < EPS_BASIS,
        np.tile(np.array([[0.0, 0.0, 1.0]]), (N, 1)),
        acc_perp_vec / (perp_norm + EPS),
    )
    if degenerate.any():
        proj = np.sum(n_hat[degenerate] * t_hat[degenerate], axis=1, keepdims=True)
        n_hat[degenerate] = n_hat[degenerate] - proj * t_hat[degenerate]
        n_norm = np.linalg.norm(n_hat[degenerate], axis=1, keepdims=True)
        n_hat[degenerate] = n_hat[degenerate] / (n_norm + EPS)

    b_hat = np.cross(t_hat, n_hat)
    return np.stack([t_hat, n_hat, b_hat], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# F0 frozen prediction (§2.1.A C2 — plan-006 산식 본 module 재구현)
# ──────────────────────────────────────────────────────────────────────────────


def f0_predict_frozen(X: np.ndarray, end_idx: int = 10,
                      d1: float = F0_D1, par: float = F0_PAR, perp: float = F0_PERP) -> np.ndarray:
    """F0 = p0 + 1.98·v_last + 1.20·acc_par_vec + (−0.20)·acc_perp_vec.

    constants (d1, par, perp) = plan-006 frenet_par120_perp_neg020 carry (frozen).
    """
    fd = finite_diff_at(X, end_idx)
    v_last, acc = fd["v_last"], fd["acc"]
    t_hat = v_last / (np.linalg.norm(v_last, axis=1, keepdims=True) + EPS)
    acc_par_scalar = np.sum(acc * t_hat, axis=1, keepdims=True)
    acc_par_vec = acc_par_scalar * t_hat
    acc_perp_vec = acc - acc_par_vec
    p0 = X[:, end_idx]
    return p0 + d1 * v_last + par * acc_par_vec + perp * acc_perp_vec


# ──────────────────────────────────────────────────────────────────────────────
# Anchor functions (§2.1.B.1)
# ──────────────────────────────────────────────────────────────────────────────


def compute_anchors_absolute(radius_m: float = ANCHOR_RADIUS) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [+radius_m, 0.0, 0.0], [-radius_m, 0.0, 0.0],
            [0.0, +radius_m, 0.0], [0.0, -radius_m, 0.0],
            [0.0, 0.0, +radius_m], [0.0, 0.0, -radius_m],
        ],
        dtype=np.float64,
    )


def compute_anchors_frenet_orthogonal(radius_m: float = ANCHOR_RADIUS) -> np.ndarray:
    return compute_anchors_absolute(radius_m=radius_m)


def anchors_to_world(anchors_local: np.ndarray, R: np.ndarray | None, N: int) -> np.ndarray:
    K = anchors_local.shape[0]
    if R is None:
        return np.broadcast_to(anchors_local[None, :, :], (N, K, 3)).copy()
    return np.einsum("nij,kj->nki", R, anchors_local)


def compute_anchors_kmeans_fold(train_residuals_world: np.ndarray, R_train: np.ndarray,
                                 K: int = KMEANS_K, radius_clip_m: float = KMEANS_RADIUS_CLIP,
                                 n_init: int = KMEANS_N_INIT,
                                 random_state: int = KMEANS_RANDOM_STATE) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (anchors_local (K, 3), cluster_sizes (K-1,), inertia, silhouette).
    anchor[0] = center origin (prepend), anchor[1..K-1] = K-1 cluster centroids (sklearn output)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    residuals_local = np.einsum("nji,nj->ni", R_train, train_residuals_world)
    residuals_local = np.clip(residuals_local, -radius_clip_m, radius_clip_m)

    km = KMeans(n_clusters=K - 1, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(residuals_local)
    cluster_centers = km.cluster_centers_
    cluster_sizes = np.array([(labels == k).sum() for k in range(K - 1)], dtype=np.int64)
    inertia = float(km.inertia_)

    if len(residuals_local) > 5000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(residuals_local), 5000, replace=False)
        sil = float(silhouette_score(residuals_local[idx], labels[idx]))
    else:
        sil = float(silhouette_score(residuals_local, labels))

    anchors_local = np.concatenate([np.zeros((1, 3)), cluster_centers], axis=0)
    return anchors_local, cluster_sizes, inertia, sil


# ──────────────────────────────────────────────────────────────────────────────
# hit + oracle
# ──────────────────────────────────────────────────────────────────────────────


def hit_rate(pred: np.ndarray, true: np.ndarray, threshold_m: float = 0.01) -> float:
    err = np.linalg.norm(pred - true, axis=-1)
    return float(np.mean(err <= threshold_m))


def oracle_hit(F0_pred: np.ndarray, anchors_world: np.ndarray, true_y: np.ndarray,
                threshold_m: float = 0.01) -> tuple[float, np.ndarray]:
    """hindsight label-aware oracle: argmin_k ‖F0+anchor[k]−y‖, then hit rate."""
    cand_pos = F0_pred[:, None, :] + anchors_world
    dists = np.linalg.norm(cand_pos - true_y[:, None, :], axis=-1)
    oracle_k = dists.argmin(axis=1)
    sel = cand_pos[np.arange(F0_pred.shape[0]), oracle_k]
    return hit_rate(sel, true_y, threshold_m=threshold_m), oracle_k


# ──────────────────────────────────────────────────────────────────────────────
# per-axis marginal oracle (§4.1 (e))
# ──────────────────────────────────────────────────────────────────────────────


def per_axis_marginal_oracle(F0_pred: np.ndarray, R: np.ndarray | None, true_y: np.ndarray,
                              frame: str, radius_m: float = ANCHOR_RADIUS) -> dict[str, float]:
    assert frame in {"absolute", "frenet_orthogonal"}
    N = F0_pred.shape[0]
    axes = [
        ("+x" if frame == "absolute" else "+t", [+radius_m, 0, 0]),
        ("-x" if frame == "absolute" else "-t", [-radius_m, 0, 0]),
        ("+y" if frame == "absolute" else "+n", [0, +radius_m, 0]),
        ("-y" if frame == "absolute" else "-n", [0, -radius_m, 0]),
        ("+z" if frame == "absolute" else "+b", [0, 0, +radius_m]),
        ("-z" if frame == "absolute" else "-b", [0, 0, -radius_m]),
    ]
    out: dict[str, float] = {}
    for label, vec in axes:
        anchors_local = np.array([[0, 0, 0], vec], dtype=np.float64)
        anchors_world = anchors_to_world(anchors_local, R, N=N)
        h, _ = oracle_hit(F0_pred, anchors_world, true_y, threshold_m=0.01)
        out[label] = h
    return out


def axis_family_ranking(marginal: dict[str, float], families: list[str], tie_gap: float = 0.003) -> list[str]:
    """family value = max(+sign, −sign). desc ordering. priority tie-break: families 입력순."""
    fam_values = {fam: max(marginal[f"+{fam}"], marginal[f"-{fam}"]) for fam in families}
    ranked: list[str] = []
    remaining = list(families)
    while remaining:
        max_v = max(fam_values[f] for f in remaining)
        eligible = [f for f in remaining if fam_values[f] >= max_v - tie_gap]
        winner = next(f for f in families if f in eligible)
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


# ──────────────────────────────────────────────────────────────────────────────
# soft label entropy (§4.1 (c), target w_k 분포 entropy)
# ──────────────────────────────────────────────────────────────────────────────


def soft_label_entropy_mean(F0_pred: np.ndarray, anchors_world: np.ndarray, true_y: np.ndarray,
                             sigma: float = SOFT_SIGMA) -> float:
    cand_pos = F0_pred[:, None, :] + anchors_world
    d = np.linalg.norm(cand_pos - true_y[:, None, :], axis=-1)
    logits = -(d ** 2) / (2 * sigma ** 2)
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w = w / (w.sum(axis=1, keepdims=True) + EPS)
    H = -np.sum(w * np.log(w + EPS), axis=1)
    return float(np.mean(H))


# ──────────────────────────────────────────────────────────────────────────────
# plan-012 disclaimer grep (§4.1 (d))
# ──────────────────────────────────────────────────────────────────────────────


def plan_012_disclaimer_grep(plan_012_ref: Path) -> tuple[bool, dict]:
    import re
    text = plan_012_ref.read_text(encoding="utf-8") if plan_012_ref.exists() else ""
    has_invalid = "INVALID_REFERENCE" in text
    has_disclaimer = bool(re.search(r"^\s*disclaimer\s*:", text, re.MULTILINE))
    return (has_invalid and has_disclaimer), {
        "has_invalid_reference_token": has_invalid,
        "has_disclaimer_field": has_disclaimer,
        "ref_path_existed": plan_012_ref.exists(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("analysis/plan-014/preflight.json"))
    ap.add_argument("--plan-012-ref", type=Path,
                    default=Path("plans/plan-012-frenet-ring-classification.results.md"))
    args = ap.parse_args()

    t0 = time.time()
    print(f"[plan-014 G0 v4] loading data from {args.root} ...", flush=True)
    train_ids, X = load_all_samples(split="train", data_root=args.root)
    label_ids, Y = load_labels(data_root=args.root)
    assert train_ids == label_ids, "id order mismatch"
    N = X.shape[0]
    print(f"[plan-014 G0] N_train={N}, T={X.shape[1]}, end_idx=10. F0=plan-006 frozen "
          f"(d1={F0_D1}, par={F0_PAR}, perp={F0_PERP})", flush=True)

    fold_of = np.array([stable_hash_fold(sid) for sid in train_ids])

    # ── task (a) F0 frozen reproduce ─────────────────────────────────────
    print("[plan-014 G0] (a) F0 frozen reproduce ...", flush=True)
    F0_pred = f0_predict_frozen(X, end_idx=10)
    h1cm = hit_rate(F0_pred, Y, threshold_m=0.01)
    h15cm = hit_rate(F0_pred, Y, threshold_m=0.015)
    f0_in_range = (F0_HIT_LO <= h1cm <= F0_HIT_HI)
    print(f"  hit@1cm={h1cm:.4f} (range [{F0_HIT_LO}, {F0_HIT_HI}], in_range={f0_in_range}) "
          f"hit@1.5cm={h15cm:.4f}", flush=True)

    # ── task (b) 3 codebook oracle ceiling ───────────────────────────────
    print("[plan-014 G0] (b) 3 codebook oracle ceiling ...", flush=True)
    R = build_frenet_basis_3d(X, end_idx=10)

    anchors_abs = compute_anchors_absolute(radius_m=ANCHOR_RADIUS)
    anchors_abs_world = anchors_to_world(anchors_abs, None, N=N)
    oracle_abs, _ = oracle_hit(F0_pred, anchors_abs_world, Y)
    print(f"  E0a Absolute oracle_hit_1cm={oracle_abs:.4f}", flush=True)

    anchors_fro = compute_anchors_frenet_orthogonal(radius_m=ANCHOR_RADIUS)
    anchors_fro_world = anchors_to_world(anchors_fro, R, N=N)
    oracle_fro, _ = oracle_hit(F0_pred, anchors_fro_world, Y)
    print(f"  E0b Frenet-ortho oracle_hit_1cm={oracle_fro:.4f}", flush=True)

    residuals_world = Y - F0_pred
    fold_kmeans: list[dict] = []
    val_oracle_hits: list[float] = []
    kmeans_oracle_total = 0.0
    kmeans_n_total = 0
    for f in range(N_FOLDS):
        train_mask = (fold_of != f)
        val_mask = (fold_of == f)
        anchors_local, cluster_sizes, inertia, sil = compute_anchors_kmeans_fold(
            residuals_world[train_mask], R[train_mask]
        )
        anchors_val_world = anchors_to_world(anchors_local, R[val_mask], N=val_mask.sum())
        h, _ = oracle_hit(F0_pred[val_mask], anchors_val_world, Y[val_mask])
        val_oracle_hits.append(h)
        kmeans_oracle_total += h * val_mask.sum()
        kmeans_n_total += val_mask.sum()
        fold_kmeans.append({
            "fold": f,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "anchors_local": anchors_local.tolist(),
            "cluster_sizes": cluster_sizes.tolist(),
            "inertia": inertia,
            "silhouette": sil,
        })
    oracle_km = kmeans_oracle_total / kmeans_n_total
    print(f"  E0c K-Means oracle_hit_1cm={oracle_km:.4f} (per-fold {val_oracle_hits})", flush=True)

    min_cluster_size = int(min(min(fk["cluster_sizes"]) for fk in fold_kmeans))
    min_cluster_pass = min_cluster_size >= KMEANS_MIN_CLUSTER_SIZE_THRESHOLD
    print(f"  K-Means min_cluster_size={min_cluster_size} (threshold={KMEANS_MIN_CLUSTER_SIZE_THRESHOLD}, "
          f"pass={min_cluster_pass})", flush=True)

    abs_norms = np.linalg.norm(anchors_abs[1:], axis=-1)
    fro_norms = np.linalg.norm(anchors_fro[1:], axis=-1)
    anchor_scale_ok = (np.allclose(abs_norms, ANCHOR_RADIUS, atol=1e-6) and
                       np.allclose(fro_norms, ANCHOR_RADIUS, atol=1e-6))
    print(f"  anchor scale (E0a/E0b non-center) = {ANCHOR_RADIUS}m ± 1e-6 → {anchor_scale_ok}", flush=True)

    # ── task (c) soft label entropy ──────────────────────────────────────
    print("[plan-014 G0] (c) soft label entropy ...", flush=True)
    entropy_abs = soft_label_entropy_mean(F0_pred, anchors_abs_world, Y, sigma=SOFT_SIGMA)
    entropy_fro = soft_label_entropy_mean(F0_pred, anchors_fro_world, Y, sigma=SOFT_SIGMA)
    entropy_mean = float((entropy_abs + entropy_fro) / 2.0)
    entropy_ok = entropy_mean >= SOFT_ENTROPY_MIN
    print(f"  entropy (abs, fro) = ({entropy_abs:.3f}, {entropy_fro:.3f}) nat, mean={entropy_mean:.3f} "
          f"(≥{SOFT_ENTROPY_MIN}={entropy_ok})", flush=True)

    # ── task (d) plan-012 disclaimer grep ────────────────────────────────
    print("[plan-014 G0] (d) plan-012 disclaimer grep ...", flush=True)
    disclaimer_ok, disclaimer_detail = plan_012_disclaimer_grep(args.plan_012_ref)
    print(f"  disclaimer_ok={disclaimer_ok} {disclaimer_detail}", flush=True)

    # ── task (e) per-axis marginal oracle ────────────────────────────────
    print("[plan-014 G0] (e) per-axis marginal oracle ...", flush=True)
    marginal_abs = per_axis_marginal_oracle(F0_pred, None, Y, frame="absolute")
    marginal_fro = per_axis_marginal_oracle(F0_pred, R, Y, frame="frenet_orthogonal")
    ranking_abs = axis_family_ranking(marginal_abs, ["x", "y", "z"])
    ranking_fro = axis_family_ranking(marginal_fro, ["t", "n", "b"])
    print(f"  abs marginal={marginal_abs} → ranking={ranking_abs}", flush=True)
    print(f"  fro marginal={marginal_fro} → ranking={ranking_fro}", flush=True)

    # ── G0 합격 ─────────────────────────────────────────────────────────
    g0_checks = {
        "f0_init_in_range": f0_in_range,
        "anchor_scale_ok": anchor_scale_ok,
        "soft_entropy_ge_0_5": entropy_ok,
        "plan_012_disclaimer_ok": disclaimer_ok,
    }
    g0_passed = all(g0_checks.values())
    print(f"[plan-014 G0] g0_checks={g0_checks} → g0_essential_passed={g0_passed}", flush=True)

    elapsed = time.time() - t0
    artifact = {
        "exp_id": "H036_g0_preflight",
        "n_train": int(N),
        "trajectory_T": int(X.shape[1]),
        "end_idx": 10,
        "f0_raw_hit_measure": {
            "single_formula": "plan-006_frenet_par120_perp_neg020 (frozen, d1=1.98, par=1.20, perp=-0.20)",
            "hit_at_1cm": {
                "hit_rate": h1cm,
                "expected_range": [F0_HIT_LO, F0_HIT_HI],
                "in_range": f0_in_range,
            },
            "hit_at_1_5cm": {"hit_rate": h15cm},
        },
        "codebook_oracle_ceilings": {
            "E0a": {"oracle_hit_1cm": oracle_abs, "anchors": anchors_abs.tolist()},
            "E0b": {"oracle_hit_1cm": oracle_fro, "anchors": anchors_fro.tolist()},
            "E0c": {
                "oracle_hit_1cm": oracle_km,
                "per_fold_oracle_hit_1cm": val_oracle_hits,
                "anchors_per_fold": [fk["anchors_local"] for fk in fold_kmeans],
            },
        },
        "per_axis_marginal_oracle": {
            "absolute": marginal_abs,
            "frenet_orthogonal": marginal_fro,
            "axis_family_ranking_absolute": ranking_abs,
            "axis_family_ranking_frenet": ranking_fro,
        },
        "kmeans_fit_meta": {
            "K": KMEANS_K,
            "fold_count": N_FOLDS,
            "centers_per_fold": [fk["anchors_local"] for fk in fold_kmeans],
            "cluster_sizes_per_fold": [fk["cluster_sizes"] for fk in fold_kmeans],
            "inertia_per_fold": [fk["inertia"] for fk in fold_kmeans],
            "silhouette_per_fold": [fk["silhouette"] for fk in fold_kmeans],
            "min_cluster_size": min_cluster_size,
            "min_cluster_size_threshold": KMEANS_MIN_CLUSTER_SIZE_THRESHOLD,
            "min_cluster_size_pass": min_cluster_pass,
            "random_state": KMEANS_RANDOM_STATE,
            "radius_clip_m": KMEANS_RADIUS_CLIP,
        },
        "g0_checks": g0_checks,
        "g0_essential_passed": g0_passed,
        "plan_012_disclaimer_detail": disclaimer_detail,
        "soft_label_entropy_detail": {
            "absolute": entropy_abs,
            "frenet_orthogonal": entropy_fro,
            "mean": entropy_mean,
            "threshold_min": SOFT_ENTROPY_MIN,
        },
        "elapsed_seconds": elapsed,
        "plan_version": "v4.5",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"[plan-014 G0] artifact -> {args.out} ({elapsed:.2f}s)", flush=True)

    return 0 if g0_passed else 1


if __name__ == "__main__":
    sys.exit(main())
