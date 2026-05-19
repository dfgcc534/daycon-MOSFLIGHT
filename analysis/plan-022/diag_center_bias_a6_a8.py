"""plan-022 post-G_final 자체실험 — A6 (no center, K=14) vs A8 (+ center, K=15).

배경 (사용자 지적):
  - plan-022 본 sweep 의 max_class_ratio 는 'selector 출력의 anchor 별 평균
    share 의 max' 만 봄. 진짜 봐야 할 건 'selector 분포 ↔ ground-truth soft
    label 분포' 의 일치 정도. max_class_ratio 는 그 일치성의 side-effect 일
    뿐. ground-truth q_true 자체가 자연스럽게 한 anchor 에 mass 가 몰리는
    layout 에서는, 높은 max_class_ratio 가 collapse 가 아닌 충실한 매칭임.

본 실험:
  - A6_bcc14 (sweep winner, no center) 재측정 + A8_bcc15 (= A6 + center) 신규.
  - 각 cell 의 표준 metric (hit_*, Δ_*, max_class_ratio) + 사용자 지적의
    distribution-match 직접 측정량:
      * q_true_max         : full-dataset q_true.mean(axis=0).max() — 자연 분포 max
      * dist_match_KL      : KL(probs_avg || q_true_avg)
      * top1_acc           : (probs.argmax(axis=1) == q_true.argmax(axis=1)).mean()
      * soft_CE            : -(q_true * log(probs+1e-12)).sum(axis=1).mean()

CLI:
  python analysis/plan-022/diag_center_bias_a6_a8.py \\
       --out-json analysis/plan-022/diag_a8.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels                        # noqa: E402
from src.pb_0_6822.selector import stable_fold_id                        # noqa: E402

_spec = importlib.util.spec_from_file_location("anchors_022", _THIS / "anchors.py")
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)

_spec = importlib.util.spec_from_file_location("som_022", _THIS / "selector_only_model.py")
som = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(som)

N_FOLDS = 5
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_SCAN = [0.001, 0.003, 0.005]


def assign_folds(ids: list[str]) -> np.ndarray:
    return np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)


def _build_lgbm_features(X: np.ndarray):
    """run_oof.run_oof_cell 의 X_lgbm + common 부분 동일 carry."""
    common = som.p021_build.build_input_common(X, som.bf.f0_baseline)
    extra = som.p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    N = X.shape[0]
    X_lgbm = np.concatenate(
        [
            common["L1"].reshape(N, 99),
            common["L2"].reshape(N, 21),
            common["L4"].reshape(N, 14),
            extra,
        ],
        axis=1,
    ).astype(np.float32)
    return X_lgbm, common["R_wfn"], common["pred_F0_world"]


def run_cell_with_diag(
    X_lgbm: np.ndarray,
    Y: np.ndarray,
    R_wfn: np.ndarray,
    pred_F0: np.ndarray,
    folds: np.ndarray,
    anchors: np.ndarray,
    tau_cls: float,
    verbose: bool = True,
) -> dict:
    """5-fold OOF + standard metric + distribution-match diagnostics."""
    N = X_lgbm.shape[0]
    K = anchors.shape[0]

    pred_world = np.zeros((N, 3), dtype=np.float32)
    probs_all = np.zeros((N, K), dtype=np.float32)

    # full-dataset q_true (deterministic from Y/R_wfn/pred_F0/anchors/τ — no leakage)
    q_true_all = som.build_soft_label_with_tau(Y, R_wfn, pred_F0, anchors, tau_cls)

    for k in range(N_FOLDS):
        tr = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        q_train = som.build_soft_label_with_tau(
            Y[tr], R_wfn[tr], pred_F0[tr], anchors, tau_cls
        )
        model = som.LgbmSelectorOnly(K=K).fit(X_lgbm[tr], q_train)
        probs = model.predict(X_lgbm[va])
        probs_all[va] = probs

        final_frenet = probs @ anchors
        pred_world[va] = (
            np.einsum("nij,nj->ni", R_wfn[va], final_frenet) + pred_F0[va]
        )
        if verbose:
            d = np.linalg.norm(pred_world[va] - Y[va], axis=1)
            print(
                f"    fold {k}: hit@1cm={float((d <= R_HIT).mean()):.4f}",
                flush=True,
            )

    # standard metrics (= run_oof_cell 와 동일 정의)
    d_cell = np.linalg.norm(pred_world - Y, axis=1)
    d_f0 = np.linalg.norm(pred_F0 - Y, axis=1)
    hit_cell_1 = float((d_cell <= R_HIT).mean())
    hit_cell_15 = float((d_cell <= R_HIT_LOOSE).mean())
    hit_f0_1 = float((d_f0 <= R_HIT).mean())
    hit_f0_15 = float((d_f0 <= R_HIT_LOOSE).mean())

    per_fold_1, per_fold_15 = [], []
    for k in range(N_FOLDS):
        m = folds == k
        per_fold_1.append(float((d_cell[m] <= R_HIT).mean()) if m.any() else 0.0)
        per_fold_15.append(
            float((d_cell[m] <= R_HIT_LOOSE).mean()) if m.any() else 0.0
        )

    delta_1 = hit_cell_1 - hit_f0_1
    delta_15 = hit_cell_15 - hit_f0_15
    max_class_ratio = float(probs_all.mean(axis=0).max())

    # ── distribution-match diagnostics (사용자 지적) ──────────────────
    probs_avg = probs_all.mean(axis=0).astype(np.float64)         # (K,)
    q_true_avg = q_true_all.mean(axis=0).astype(np.float64)       # (K,)
    q_true_max = float(q_true_avg.max())

    # KL(probs_avg || q_true_avg) — 0 이면 selector 평균이 ground-truth 평균과 일치
    eps = 1e-12
    dist_match_KL = float(
        (probs_avg * (np.log(probs_avg + eps) - np.log(q_true_avg + eps))).sum()
    )

    # per-sample top-1 accuracy
    pred_top1 = probs_all.argmax(axis=1)
    true_top1 = q_true_all.argmax(axis=1)
    top1_acc = float((pred_top1 == true_top1).mean())

    # per-sample soft cross-entropy
    soft_CE = float(
        -(q_true_all.astype(np.float64)
          * np.log(probs_all.astype(np.float64) + eps)).sum(axis=1).mean()
    )

    # natural max class (ground-truth side) - max_class_ratio 비교용
    natural_max_idx = int(q_true_avg.argmax())
    pred_max_idx = int(probs_avg.argmax())

    return {
        "K": int(K),
        "tau_cls": float(tau_cls),
        "hit_1cm": hit_cell_1,
        "hit_1.5cm": hit_cell_15,
        "delta_1cm": delta_1,
        "delta_1.5cm": delta_15,
        "max_class_ratio": max_class_ratio,
        "fold_var_1cm": float(np.std(per_fold_1)),
        "fold_var_1.5cm": float(np.std(per_fold_15)),
        "pass_both": bool(delta_1 >= 0.005 and delta_15 >= 0.005),
        # distribution-match diagnostics
        "q_true_max": q_true_max,
        "dist_match_KL": dist_match_KL,
        "top1_acc": top1_acc,
        "soft_CE": soft_CE,
        "natural_max_idx": natural_max_idx,
        "pred_max_idx": pred_max_idx,
        "natural_max_is_center": (anchors.shape[0] == 15 and natural_max_idx == 0),
        "pred_max_is_center": (anchors.shape[0] == 15 and pred_max_idx == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=_THIS / "diag_a8.json")
    ap.add_argument("--tau-only", type=float, default=None,
                    help="단일 τ만 테스트 (debug)")
    args = ap.parse_args()

    t0 = time.time()
    print("[diag_a8] loading data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids, "ids mismatch"
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    folds = assign_folds(ids)
    print(
        f"[diag_a8] N={X.shape[0]} folds={np.bincount(folds).tolist()}",
        flush=True,
    )

    print("[diag_a8] building LGBM features ...", flush=True)
    X_lgbm, R_wfn, pred_F0 = _build_lgbm_features(X)
    print(f"[diag_a8] X_lgbm shape={X_lgbm.shape}", flush=True)

    layouts = {
        "A6_bcc14": anchors_mod.ANCHORS_A6,   # K=14, no center
        "A8_bcc15": anchors_mod.ANCHORS_A8,   # K=15, + center (post-G_final)
    }
    taus = [args.tau_only] if args.tau_only is not None else TAU_SCAN

    results: dict = {}
    for layout_name, anchors in layouts.items():
        print(f"\n=== {layout_name} (K={anchors.shape[0]}) ===", flush=True)
        results[layout_name] = {}
        for tau in taus:
            t1 = time.time()
            print(f"  [{layout_name}] τ_cls={tau:.3f} ...", flush=True)
            cell = run_cell_with_diag(
                X_lgbm, Y, R_wfn, pred_F0, folds, anchors, tau
            )
            cell["elapsed_sec"] = float(time.time() - t1)
            results[layout_name][f"tau_{tau:.3f}"] = cell
            print(
                f"  [{layout_name}] τ_cls={tau:.3f} "
                f"hit@1cm={cell['hit_1cm']:.4f} Δ_1cm={cell['delta_1cm']:+.4f} "
                f"Δ_1.5cm={cell['delta_1.5cm']:+.4f} pass={cell['pass_both']} | "
                f"max_class={cell['max_class_ratio']:.3f} q_true_max={cell['q_true_max']:.3f} "
                f"KL={cell['dist_match_KL']:.4f} top1_acc={cell['top1_acc']:.4f} "
                f"CE={cell['soft_CE']:.4f} | {cell['elapsed_sec']:.0f}s",
                flush=True,
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[diag_a8] wrote {args.out_json}", flush=True)
    print(f"[diag_a8] total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
