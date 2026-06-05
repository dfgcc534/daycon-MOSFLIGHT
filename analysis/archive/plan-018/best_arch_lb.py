"""plan-018 c14 (STAGE 2, G2) — best arch LB submission.

G1 의 ablation_results.json 에서 best ablation arch 선정 후 5-fold OOF 의 test prediction 산출 → submission.csv.

Spec §8. dacon-submit 1회 (사용자 confirm 후).

Usage:
    python analysis/plan-018/best_arch_lb.py [--arch A1] [--epochs 50]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/plan-007"))
sys.path.insert(0, str(REPO_ROOT / "analysis/plan-018"))

from src.plan018.arch_modules import ARCH_REGISTRY  # noqa: E402
from src.pb_0_6822 import selector  # noqa: E402
from mlp_coeff import soft_hit_loss, hit_rate  # noqa: E402
from basis_ablation import stack_train_full  # noqa: E402
from ablation_runner import build_samples_with_window  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "analysis/plan-018"
RUN_DIR = REPO_ROOT / "runs/baseline/plan018_g2_best_arch"


def compute_test_basis_and_window(test_x: np.ndarray, best_basis_vars: list[str],
                                    global_mean_speed: float):
    """Compute basis_terms + window for test set (10K).

    Returns: (basis_terms (10000, 8, 3), window (10000, 6, 3), p0 (10000, 3))
    """
    from basis_ablation import compute_all_terms
    test_terms_dict = compute_all_terms(test_x, end_idx=10, horizon=2, global_mean_speed=global_mean_speed)
    basis_terms = np.stack([test_terms_dict[v] for v in best_basis_vars], axis=1).astype(np.float32)
    window = test_x[:, 5:11, :].astype(np.float32)
    p0 = test_x[:, 10, :].astype(np.float32)
    return basis_terms, window, p0


def train_one_fold(samples: dict, fold_k: int, arch_id: str,
                    best_basis_vars: list[str], stage3_best_params: np.ndarray,
                    test_data: dict,
                    n_epochs: int = 50, patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 1024, seed: int = 20260606):
    spec = ARCH_REGISTRY[arch_id]
    arch_class = spec["class"]
    input_type = spec["input_type"]

    torch.manual_seed(seed + fold_k)
    np.random.seed(seed + fold_k)

    is_train = samples["fold_id"] != fold_k
    is_val = (samples["fold_id"] == fold_k) & samples["is_orig_end10"]

    if input_type == "stats_13d":
        enc_t = torch.from_numpy(samples["traj_features"][is_train]).to(DEVICE)
        enc_v = torch.from_numpy(samples["traj_features"][is_val]).to(DEVICE)
        enc_test = torch.from_numpy(test_data["traj_features"]).to(DEVICE)
    else:
        enc_t = torch.from_numpy(samples["window"][is_train]).to(DEVICE)
        enc_v = torch.from_numpy(samples["window"][is_val]).to(DEVICE)
        enc_test = torch.from_numpy(test_data["window"]).to(DEVICE)

    terms_t = torch.from_numpy(samples["basis_terms"][is_train]).to(DEVICE)
    p0_t = torch.from_numpy(samples["p0"][is_train]).to(DEVICE)
    tgt_t = torch.from_numpy(samples["target"][is_train]).to(DEVICE)
    terms_v = torch.from_numpy(samples["basis_terms"][is_val]).to(DEVICE)
    p0_v = torch.from_numpy(samples["p0"][is_val]).to(DEVICE)
    tgt_v = torch.from_numpy(samples["target"][is_val]).to(DEVICE)
    terms_test = torch.from_numpy(test_data["basis_terms"]).to(DEVICE)
    p0_test = torch.from_numpy(test_data["p0"]).to(DEVICE)

    n_coeffs = len(best_basis_vars)
    model = arch_class(n_coeffs=n_coeffs, global_init=stage3_best_params).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = -1.0
    best_state = None
    no_improve = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(enc_t.shape[0], device=DEVICE)
        for i in range(0, enc_t.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            coeffs = model(enc_t[idx])
            pred = p0_t[idx] + (coeffs.unsqueeze(-1) * terms_t[idx]).sum(dim=1)
            loss = soft_hit_loss(pred, tgt_t[idx])
            if getattr(model, "aux_loss", None) is not None:
                loss = loss + model.aux_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            coeffs_v = model(enc_v)
            pred_v = p0_v + (coeffs_v.unsqueeze(-1) * terms_v).sum(dim=1)
            val_hit = hit_rate(pred_v, tgt_v)

        if val_hit > best_val + min_delta:
            best_val = val_hit
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        coeffs_test = model(enc_test)
        test_pred = (p0_test + (coeffs_test.unsqueeze(-1) * terms_test).sum(dim=1)).cpu().numpy()
    return test_pred, best_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None, help="override best arch (default from ablation_results.json)")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "best_arch_lb.json")
    args = ap.parse_args()

    # Load G1 ablation results
    g1_path = OUT_DIR / "ablation_results.json"
    if not g1_path.exists():
        print(f"ERROR: {g1_path} missing — run ablation_runner.py first", file=sys.stderr)
        return 1
    g1 = json.loads(g1_path.read_text())
    best_arch = args.arch or g1["best_arch"]
    if best_arch is None:
        print("ERROR: no best arch determined", file=sys.stderr)
        return 1

    best_oof = g1["arch_results"][best_arch]["oof_hit_1cm"]
    print(f"[plan-018 G2] best arch = {best_arch}, OOF = {best_oof:.4f}", flush=True)

    # Load plan-007 dependencies
    sliding = json.loads((REPO_ROOT / "analysis/plan-007/sliding_validity.json").read_text())
    aug_usable = sliding["aug_usable"]
    stage2 = json.loads((REPO_ROOT / "analysis/plan-007/cma_es_step2.json").read_text())
    stage3 = json.loads((REPO_ROOT / "analysis/plan-007/basis_ablation.json").read_text())
    global_mean_speed = float(stage2["global_mean_speed"])
    best_basis_vars = stage3["best_basis_vars"]
    stage3_best_params = np.asarray(stage3["best_basis_params"], dtype=np.float32)

    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    test_ids = sorted([p.stem for p in (DATA_ROOT / "test").glob("*.csv")])
    test_x = selector.load_stack(DATA_ROOT / "test", test_ids)
    print(f"  N_train={len(ids)}, N_test={len(test_ids)}", flush=True)

    # Build 50K pool + test basis/window
    print(f"  Building 50K pool + test data ...", flush=True)
    stack = stack_train_full(aug_usable, train_x, train_y, ids, global_mean_speed)
    samples = build_samples_with_window(stack, train_x, aug_usable, best_basis_vars)
    test_basis, test_window, test_p0 = compute_test_basis_and_window(test_x, best_basis_vars, global_mean_speed)
    from mlp_coeff import compute_trajectory_features
    test_features = compute_trajectory_features(test_x[:, 5:11, :])
    test_data = {
        "traj_features": test_features.astype(np.float32),
        "window": test_window,
        "basis_terms": test_basis,
        "p0": test_p0,
    }

    # 5-fold train + test pred → mean ensemble
    t0 = time.time()
    test_preds_per_fold = []
    for fold_k in range(5):
        t_fold = time.time()
        test_pred, best_val = train_one_fold(
            samples, fold_k, best_arch, best_basis_vars, stage3_best_params, test_data,
        )
        elapsed_fold = time.time() - t_fold
        test_preds_per_fold.append(test_pred)
        print(f"  fold {fold_k}: best_val_hit={best_val:.4f} elapsed={elapsed_fold:.1f}s", flush=True)

    test_pred_mean = np.stack(test_preds_per_fold, axis=0).mean(axis=0)   # (10000, 3)
    elapsed = time.time() - t0

    # Submission write
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    sample_ids = sample_sub["id"].tolist()
    id_to_idx = {sid: i for i, sid in enumerate(test_ids)}
    ordered = np.array([test_pred_mean[id_to_idx[sid]] for sid in sample_ids], dtype=np.float64)
    submission_path = RUN_DIR / "submission.csv"
    df = pd.DataFrame({
        "id": sample_ids,
        "x": [f"{v:.6f}" for v in ordered[:, 0]],
        "y": [f"{v:.6f}" for v in ordered[:, 1]],
        "z": [f"{v:.6f}" for v in ordered[:, 2]],
    })
    df.to_csv(submission_path, index=False)
    print(f"\n  submission -> {submission_path}", flush=True)

    summary = {
        "exp_id": "F009_best-arch-lb",
        "plan_version": "v1.2",
        "best_arch": best_arch,
        "best_arch_oof": best_oof,
        "best_basis_vars": best_basis_vars,
        "test_pred_mean_shape": list(test_pred_mean.shape),
        "submission_path": str(submission_path),
        "lb_score": None,   # post dacon-submit
        "elapsed_seconds": elapsed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[plan-018 G2] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
