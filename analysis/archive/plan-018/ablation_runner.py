"""plan-018 c11 (STAGE 1, G1) — 4 ablation arch (A1/A2/A3/A6) × 5-fold OOF.

A0 = plan-007 mlp_coeff.json import (no retrain).
A4/A5 제외 (v1.2 executor patch).

§5.0 common framework:
- training loop: Adam lr=1e-3, wd=1e-4, batch=1024, epoch=50, patience=8, grad_clip=2.0
- loss: soft_hit_loss (sigmoid, threshold=0.01, sharpness=200)
- 5-fold split: plan-007 §7.2 carry (seed=42 grouped by sample_id, sliding ∪ original)
- val = original end_idx=10 only

Uses plan-007 mlp_coeff.py infrastructure:
- build_all_samples (50K pool: 10K original + 40K sliding aug)
- soft_hit_loss, compute_pred, hit_rate

Usage:
    python analysis/plan-018/ablation_runner.py [--arch A1,A2,A3,A6]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis/plan-007"))

from src.plan018.arch_modules import ARCH_REGISTRY  # noqa: E402
from src.pb_0_6822 import selector  # noqa: E402
# plan-007 imports
from mlp_coeff import (  # noqa: E402
    compute_trajectory_features, soft_hit_loss, hit_rate, compute_pred, build_all_samples
)
from basis_ablation import stack_train_full  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_HIT = 0.01
DATA_ROOT = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "analysis/plan-018"


def build_samples_with_window(stack: dict, train_x: np.ndarray, aug_usable: bool,
                                best_basis_vars: list[str]) -> dict:
    """Extend plan-007 build_all_samples with trajectory window per sample (B, 6, 3).

    For original block (end_idx=10): train_x[:, 5:11, :]
    For sliding (end_idx=5..8): train_x[:, end_idx-5:end_idx+1, :]
    """
    samples = build_all_samples(stack, train_x, aug_usable, best_basis_vars)

    # Add window field
    windows_list = []
    windows_list.append(train_x[:, 5:11, :])   # original (end_idx=10)
    if aug_usable:
        for end_idx in range(5, 9):
            windows_list.append(train_x[:, end_idx - 5: end_idx + 1, :])
    windows = np.concatenate(windows_list, axis=0).astype(np.float32)
    samples["window"] = windows
    return samples


def train_one_fold(samples: dict, fold_k: int, arch_id: str,
                    best_basis_vars: list[str], stage3_best_params: np.ndarray,
                    n_epochs: int = 50, patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 1024, seed: int = 20260606):
    spec = ARCH_REGISTRY[arch_id]
    arch_class = spec["class"]
    input_type = spec["input_type"]

    torch.manual_seed(seed + fold_k)
    np.random.seed(seed + fold_k)

    is_train = samples["fold_id"] != fold_k
    is_val = (samples["fold_id"] == fold_k) & samples["is_orig_end10"]

    # Encoder input 분기
    if input_type == "stats_13d":
        enc_t = torch.from_numpy(samples["traj_features"][is_train]).to(DEVICE)
        enc_v = torch.from_numpy(samples["traj_features"][is_val]).to(DEVICE)
    elif input_type == "traj_6x3":
        enc_t = torch.from_numpy(samples["window"][is_train]).to(DEVICE)
        enc_v = torch.from_numpy(samples["window"][is_val]).to(DEVICE)
    else:
        raise ValueError(f"unknown input_type: {input_type}")

    terms_t = torch.from_numpy(samples["basis_terms"][is_train]).to(DEVICE)
    p0_t = torch.from_numpy(samples["p0"][is_train]).to(DEVICE)
    tgt_t = torch.from_numpy(samples["target"][is_train]).to(DEVICE)
    terms_v = torch.from_numpy(samples["basis_terms"][is_val]).to(DEVICE)
    p0_v = torch.from_numpy(samples["p0"][is_val]).to(DEVICE)
    tgt_v = torch.from_numpy(samples["target"][is_val]).to(DEVICE)

    n_coeffs = len(best_basis_vars)
    model = arch_class(n_coeffs=n_coeffs, global_init=stage3_best_params).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_idx = torch.arange(enc_t.shape[0], device=DEVICE)
    best_val = -1.0
    best_state = None
    no_improve = 0
    epoch_history = []
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(enc_t.shape[0], device=DEVICE)
        ep_loss_sum = 0.0
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
            ep_loss_sum += float(loss.item()) * idx.shape[0]
        ep_loss = ep_loss_sum / enc_t.shape[0]

        model.eval()
        with torch.no_grad():
            coeffs_v = model(enc_v)
            pred_v = p0_v + (coeffs_v.unsqueeze(-1) * terms_v).sum(dim=1)
            val_hit = hit_rate(pred_v, tgt_v)
        epoch_history.append({"epoch": epoch, "train_loss": ep_loss, "val_hit": val_hit})

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
        coeffs_v = model(enc_v)
        pred_v = p0_v + (coeffs_v.unsqueeze(-1) * terms_v).sum(dim=1)
        final_val_hit = hit_rate(pred_v, tgt_v)
    return {
        "fold": fold_k,
        "best_val_hit": best_val,
        "final_val_hit": final_val_hit,
        "n_params": n_params,
        "n_epochs_run": len(epoch_history),
        "val_pred": pred_v.cpu().numpy(),
        "val_indices": np.flatnonzero(is_val),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="A1,A2,A3,A6",
                    help="comma-separated arch ids (default: 4 ablation, A0 import-only)")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "ablation_results.json")
    args = ap.parse_args()

    arch_ids = [a.strip() for a in args.arch.split(",") if a.strip()]
    print(f"[plan-018 G1] DEVICE={DEVICE}, archs={arch_ids}", flush=True)

    # Load plan-007 dependencies
    sliding = json.loads((REPO_ROOT / "analysis/plan-007/sliding_validity.json").read_text())
    aug_usable = sliding["aug_usable"]
    stage2 = json.loads((REPO_ROOT / "analysis/plan-007/cma_es_step2.json").read_text())
    stage3 = json.loads((REPO_ROOT / "analysis/plan-007/basis_ablation.json").read_text())
    global_mean_speed = float(stage2["global_mean_speed"])
    best_basis_vars = stage3["best_basis_vars"]
    stage3_best_params = np.asarray(stage3["best_basis_params"], dtype=np.float32)

    print(f"  aug_usable={aug_usable}, best_basis_vars={best_basis_vars}", flush=True)
    print(f"  stage3_best_params={stage3_best_params}", flush=True)

    # Load data
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    n_orig = len(ids)
    print(f"  N_orig={n_orig}, train_x.shape={train_x.shape}", flush=True)

    # Build 50K pool
    print(f"  Building 50K sample pool ...", flush=True)
    stack = stack_train_full(aug_usable, train_x, train_y, ids, global_mean_speed)
    samples = build_samples_with_window(stack, train_x, aug_usable, best_basis_vars)
    print(f"  pool M = {len(samples['p0']):,}, window shape = {samples['window'].shape}", flush=True)

    # A0 import from plan-007 mlp_coeff.json
    mlp_coeff = json.loads((REPO_ROOT / "analysis/plan-007/mlp_coeff.json").read_text())
    a0_result = {
        "arch": "A0", "input_type": "stats_13d",
        "oof_hit_1cm": float(mlp_coeff["oof_hit"]),
        "fold_oofs": [f["best_val_hit"] for f in mlp_coeff.get("folds", [])],
        "n_params": 297,
        "imported_from": "analysis/plan-007/mlp_coeff.json",
    }
    print(f"\n  [A0 baseline] OOF={a0_result['oof_hit_1cm']:.4f} (imported)", flush=True)

    # 4 ablation arch
    all_results = {"A0": a0_result}
    orig_fold_ids = samples["fold_id"][:n_orig]

    for arch_id in arch_ids:
        if arch_id not in ARCH_REGISTRY:
            print(f"\n  WARN: skipping unknown arch {arch_id}", flush=True)
            continue
        print(f"\n[plan-018 G1] === arch {arch_id} ({ARCH_REGISTRY[arch_id]['input_type']}) ===", flush=True)
        t_arch = time.time()
        oof_pred = np.zeros((n_orig, 3), dtype=np.float32)
        fold_results = []
        for fold_k in range(5):
            t_fold = time.time()
            res = train_one_fold(samples, fold_k, arch_id, best_basis_vars, stage3_best_params)
            elapsed_fold = time.time() - t_fold
            val_mask = orig_fold_ids == fold_k
            oof_pred[val_mask] = res["val_pred"]
            fold_results.append({
                "fold": res["fold"], "best_val_hit": res["best_val_hit"],
                "n_epochs": res["n_epochs_run"], "n_params": res["n_params"],
                "elapsed_sec": elapsed_fold,
            })
            print(f"  fold {fold_k}: val_hit={res['best_val_hit']:.4f} epoch={res['n_epochs_run']} "
                  f"params={res['n_params']:,} elapsed={elapsed_fold:.1f}s", flush=True)

        err_oof = np.linalg.norm(oof_pred - train_y, axis=1)
        arch_oof = float((err_oof <= R_HIT).mean())
        elapsed_arch = time.time() - t_arch
        all_results[arch_id] = {
            "arch": arch_id,
            "input_type": ARCH_REGISTRY[arch_id]["input_type"],
            "oof_hit_1cm": arch_oof,
            "fold_oofs": [f["best_val_hit"] for f in fold_results],
            "fold_results": fold_results,
            "n_params": fold_results[0]["n_params"],
            "elapsed_sec": elapsed_arch,
        }
        np.save(OUT_DIR / f"oof_pred_{arch_id}.npy", oof_pred)
        print(f"  ★ {arch_id} 5-fold concat OOF = {arch_oof:.4f}, elapsed={elapsed_arch:.1f}s", flush=True)

    # G1 합격 판정
    g1_threshold = 0.6532  # = step 4 + 0.005 per spec §3.2
    ablation_oofs = {k: v["oof_hit_1cm"] for k, v in all_results.items() if k != "A0"}
    n_pass = sum(1 for v in ablation_oofs.values() if v >= g1_threshold)
    g1_passed = n_pass >= 1
    best_arch = max(ablation_oofs, key=ablation_oofs.get) if ablation_oofs else None
    best_arch_oof = ablation_oofs.get(best_arch) if best_arch else None

    print(f"\n[plan-018 G1] === G1 summary ===", flush=True)
    for k, v in all_results.items():
        marker = "★" if k == best_arch else ("  " if k == "A0" else "  ")
        print(f"  {marker} {k}: OOF={v['oof_hit_1cm']:.4f}, params={v.get('n_params', '?')}", flush=True)
    print(f"  G1 threshold (step4 + 0.005) = {g1_threshold:.4f}", flush=True)
    print(f"  ablation arch passing G1: {n_pass}/{len(ablation_oofs)}", flush=True)
    print(f"  best ablation arch: {best_arch} OOF={best_arch_oof:.4f}" if best_arch else "  best: None", flush=True)
    print(f"  G1_passed = {g1_passed}", flush=True)

    summary = {
        "exp_id": "F008_arch-ablation",
        "plan_version": "v1.2",
        "arch_results": all_results,
        "g1_threshold": g1_threshold,
        "g1_passed": g1_passed,
        "best_arch": best_arch,
        "best_arch_oof": best_arch_oof,
        "n_arch_passing": n_pass,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\n[plan-018 G1] artifact -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
