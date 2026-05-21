"""plan-024 v6 — plan-022 LGBM + plan-024 의 signature input 추가 (사용자 통찰 follow).

사용자 통찰: "변경 = 후보 + input. regime/corrector 영향 작음 입증."
→ arch (PB cross-attn) carry vs LGBM (plan-022) 비교 시, *LGBM 이 14 BCC 환경에서 더 잘함*.
→ 그러면 plan-024 framework 안 마지막 lever = **plan-022 LGBM input 에 plan-024 의 signature feature 추가**.

plan-022 base: L1(99) + L2(21) + L4(14) + lgbm_extra(36) = 170D LGBM.
v6 추가: Multi-window 60D (plan-024 §4.4.1 trim 후 60D, signature lever).
→ 총 170 + 60 = 230D LGBM.

가설: plan-022 0.6528 → 0.66+ (+0.01 추정, Multi-window 가 trajectory 의 sub-window
별 stat 으로 새로운 information). plan-024 framework 의 input axis 가 *cross-attn
없이도* 부분 작동하는지 확인.
"""
from __future__ import annotations

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

from src.io import load_all_samples, load_labels
from src.pb_0_6822.selector import stable_fold_id

_spec = importlib.util.spec_from_file_location("p020_bf", _REPO / "analysis" / "plan-020" / "baseline_f0.py")
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)
_spec = importlib.util.spec_from_file_location("p022_anchors", _REPO / "analysis" / "plan-022" / "anchors.py")
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)
_spec = importlib.util.spec_from_file_location("p022_som", _REPO / "analysis" / "plan-022" / "selector_only_model.py")
som = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(som)
_spec = importlib.util.spec_from_file_location("p021_build", _REPO / "analysis" / "plan-021" / "build_input.py")
p021_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p021_build)
_spec = importlib.util.spec_from_file_location("p024_cand", _THIS / "cand_builder.py")
cand_builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cand_builder)

N_FOLDS = 5
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_CLS = 0.001


def build_lgbm_input(X, R_wfn, multiwindow_60: np.ndarray) -> np.ndarray:
    """plan-022 LGBM 170D + plan-024 Multi-window 60D = 230D."""
    common = p021_build.build_input_common(X, bf.f0_baseline)
    extra = p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    N = X.shape[0]
    X_base = np.concatenate(
        [common["L1"].reshape(N, 99), common["L2"].reshape(N, 21),
         common["L4"].reshape(N, 14), extra],
        axis=1,
    ).astype(np.float32)  # 170D
    X_aug = np.concatenate([X_base, multiwindow_60.astype(np.float32)], axis=1)  # 230D
    return X_aug, common["R_wfn"], common["pred_F0_world"]


def main():
    t0 = time.time()
    print("[v6 LGBM+p024input] load data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    N = X.shape[0]
    folds = np.asarray([stable_fold_id(str(s), N_FOLDS) for s in ids], dtype=int)
    anchors = anchors_mod.ANCHORS_A6
    K = anchors.shape[0]

    # build plan-022 base + multi-window 60D
    common = p021_build.build_input_common(X, bf.f0_baseline)
    L1_frenet = common["L1"]
    R_wfn_all = common["R_wfn"]
    pred_F0_world_all = common["pred_F0_world"]

    # Multi-window 60D (plan-024 trim 사용)
    mw_path = _THIS / "multiwindow_trim.json"
    if not mw_path.exists():
        raise FileNotFoundError(f"{mw_path} not found — run G2 first")
    stat_144 = cand_builder._multiwindow_144(L1_frenet)
    with open(mw_path) as f:
        trim = json.load(f)
    kept_idx = np.asarray(trim["kept_indices"], dtype=np.int64)
    multiwindow_60 = stat_144[:, kept_idx]
    print(f"[v6] Multi-window 60D shape={multiwindow_60.shape}", flush=True)

    # build aug input
    extra = p021_build.build_input_lgbm_extra(X, L1=L1_frenet)
    X_base = np.concatenate(
        [L1_frenet.reshape(N, 99), common["L2"].reshape(N, 21),
         common["L4"].reshape(N, 14), extra],
        axis=1,
    ).astype(np.float32)
    X_lgbm = np.concatenate([X_base, multiwindow_60.astype(np.float32)], axis=1)
    print(f"[v6] X_lgbm shape={X_lgbm.shape} (plan-022 base 170 + Multi-window 60 = 230)", flush=True)

    pred_world = np.zeros((N, 3), dtype=np.float32)
    probs_all = np.zeros((N, K), dtype=np.float32)

    for k_fold in range(N_FOLDS):
        t_f = time.time()
        tr = np.where(folds != k_fold)[0]
        va = np.where(folds == k_fold)[0]
        q_train = som.build_soft_label_with_tau(
            Y[tr], R_wfn_all[tr], pred_F0_world_all[tr], anchors, TAU_CLS
        )
        model = som.LgbmSelectorOnly(K=K).fit(X_lgbm[tr], q_train)
        probs = model.predict(X_lgbm[va])
        probs_all[va] = probs
        final_frenet = probs @ anchors
        pred_world[va] = np.einsum("nij,nj->ni", R_wfn_all[va], final_frenet) + pred_F0_world_all[va]
        d = np.linalg.norm(pred_world[va] - Y[va], axis=1)
        print(f"  fold {k_fold}: hit@1cm={(d<=R_HIT).mean():.4f} time={time.time()-t_f:.0f}s", flush=True)

    # OOF metric
    d_cell = np.linalg.norm(pred_world - Y, axis=1)
    d_F0 = np.linalg.norm(pred_F0_world_all - Y, axis=1)
    hit_1cm = float((d_cell <= R_HIT).mean())
    hit_15cm = float((d_cell <= R_HIT_LOOSE).mean())
    f0_hit_1cm = float((d_F0 <= R_HIT).mean())
    f0_hit_15cm = float((d_F0 <= R_HIT_LOOSE).mean())

    # gap_ranking
    anchors_world = (
        np.einsum("nij,kj->nki", R_wfn_all, anchors.astype(np.float32))
        + pred_F0_world_all[:, None, :]
    )
    oracle_dist = np.linalg.norm(anchors_world - Y[:, None, :], axis=2).min(axis=1)
    oracle_1cm = float((oracle_dist <= R_HIT).mean())
    argmax_idx = probs_all.argmax(axis=1)
    argmax_pos = anchors_world[np.arange(N), argmax_idx]
    argmax_hit = float((np.linalg.norm(argmax_pos - Y, axis=1) <= R_HIT).mean())

    q_true_all = som.build_soft_label_with_tau(Y, R_wfn_all, pred_F0_world_all, anchors, TAU_CLS)
    top1_acc = float((probs_all.argmax(1) == q_true_all.argmax(1)).mean())
    eps = 1e-12
    soft_CE = float(-(q_true_all.astype(np.float64) * np.log(probs_all + eps)).sum(1).mean())

    out = {
        "N": N,
        "hit_1cm": hit_1cm, "hit_1.5cm": hit_15cm,
        "delta_1cm": hit_1cm - f0_hit_1cm, "delta_1.5cm": hit_15cm - f0_hit_15cm,
        "f0_hit_1cm": f0_hit_1cm, "f0_hit_1.5cm": f0_hit_15cm,
        "max_class_ratio": float(probs_all.mean(0).max()),
        "oracle_1cm": oracle_1cm, "argmax_hit": argmax_hit,
        "gap_ranking": oracle_1cm - argmax_hit,
        "top1_acc": top1_acc, "soft_CE": soft_CE,
        "elapsed_sec": float(time.time() - t0),
        "input_dim": X_lgbm.shape[1],
        "config": "plan-022 LGBM 170D base + plan-024 Multi-window 60D = 230D",
    }
    out_path = _THIS / "results_lgbm_p024input.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[v6] hit_1cm={hit_1cm:.4f} hit_1.5cm={hit_15cm:.4f} Δ_1cm={out['delta_1cm']:+.4f} "
          f"gap_ranking={out['gap_ranking']:.4f}", flush=True)
    print(f"[v6] vs plan-022 winner 0.6528: Δ = {hit_1cm - 0.6528:+.4f}", flush=True)
    print(f"[v6] total {out['elapsed_sec']:.1f}s → {out_path}", flush=True)


if __name__ == "__main__":
    main()
