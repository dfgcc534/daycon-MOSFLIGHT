"""plan-019 c3 (STAGE 0) — A0 baseline reproduce.

plan-007 step 4 MLP (13-d stats → 8 coefficient) 의 직접 재구현 (§4.2 spec carry).
import X (§10 정책). soft_hit_loss / hit_rate / compute_basis_terms 는 모두
`src.archive.plan019.common` 재구현 사용.

Usage:
    python -m src.archive.plan019.baseline_a0 --out-json analysis/archive/plan-019/a0_baseline.json

Outputs:
    analysis/archive/plan-019/a0_baseline.json
    runs/baseline/F013_a0/oof_predictions.npz
    runs/baseline/F013_a0/checkpoint_fold{k}.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.archive.plan019.common import (  # noqa: E402
    R_HIT,
    BEST_BASIS_VARS,
    build_pool,
    compute_pred_from_anchor,
    hit_rate,
    load_artifacts,
    load_data,
    soft_hit_loss,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = REPO_ROOT / "runs/baseline/F013_a0"
OUT_DIR = REPO_ROOT / "analysis/plan-019"

# §3.2 G0 합격 기준
A0_OOF_MIN = 0.6479
A0_OOF_MAX = 0.6485


class CoefficientMLP(nn.Module):
    """A0 baseline — plan-007 §7.1 spec 재구현. params ~ 300.

    feat_dim=13 (handcrafted stats), n_coeffs=8 (plan-007 best basis).
    global_init: stage3_best_params (8-vec) → bias init, weight=0
    (학습 0 step 시점에서 plan-007 step 3 baseline 동작).
    """
    def __init__(self, *, feat_dim: int = 13, n_coeffs: int = 8,
                 global_init: np.ndarray | None = None, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
                self.mlp[-1].weight.zero_()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


def train_one_fold(samples: dict, fold_k: int, stage3_best_params: np.ndarray, *,
                    n_epochs: int = 50, patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 1024, seed: int = 20260606) -> dict:
    torch.manual_seed(seed + fold_k)
    np.random.seed(seed + fold_k)

    is_train = samples["fold_id"] != fold_k
    is_val = (samples["fold_id"] == fold_k) & samples["is_orig_end10"]

    feats_t = torch.from_numpy(samples["traj_features"][is_train]).to(DEVICE)
    terms_t = torch.from_numpy(samples["basis_terms"][is_train]).to(DEVICE)
    p0_t = torch.from_numpy(samples["p0"][is_train]).to(DEVICE)
    tgt_t = torch.from_numpy(samples["target"][is_train]).to(DEVICE)
    feats_v = torch.from_numpy(samples["traj_features"][is_val]).to(DEVICE)
    terms_v = torch.from_numpy(samples["basis_terms"][is_val]).to(DEVICE)
    p0_v = torch.from_numpy(samples["p0"][is_val]).to(DEVICE)
    tgt_v = torch.from_numpy(samples["target"][is_val]).to(DEVICE)

    model = CoefficientMLP(global_init=stage3_best_params).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_train = feats_t.shape[0]
    best_val = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        ep_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            coeffs = model(feats_t[idx])
            pred = compute_pred_from_anchor(terms_t[idx], coeffs, p0_t[idx])
            loss = soft_hit_loss(pred, tgt_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item()) * idx.shape[0]
        ep_loss /= n_train

        model.eval()
        with torch.no_grad():
            coeffs_v = model(feats_v)
            pred_v = compute_pred_from_anchor(terms_v, coeffs_v, p0_v)
            val_hit = hit_rate(pred_v, tgt_v)
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_hit": val_hit})

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
        coeffs_v = model(feats_v)
        pred_v = compute_pred_from_anchor(terms_v, coeffs_v, p0_v)
    return {
        "fold": fold_k,
        "train_n": int(is_train.sum()),
        "val_n": int(is_val.sum()),
        "best_val_hit": float(best_val),
        "n_epochs_run": len(history),
        "val_pred": pred_v.detach().cpu().numpy(),
        "val_indices": np.flatnonzero(is_val),
        "state_dict": best_state,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "a0_baseline.json")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plan-019 c4 / G0] DEVICE={DEVICE}", flush=True)

    artifacts = load_artifacts()
    print(f"  aug_usable={artifacts['aug_usable']}, "
          f"best_basis_vars={BEST_BASIS_VARS}", flush=True)
    print(f"  global_mean_speed={artifacts['global_mean_speed']:.6f}", flush=True)
    stage3_best_params = artifacts["stage3_best_params"]
    print(f"  stage3_best_params={stage3_best_params.tolist()}", flush=True)

    ids, train_x, train_y = load_data()
    n_orig = len(ids)
    print(f"  N_orig={n_orig}, train_x.shape={train_x.shape}", flush=True)

    print("  Building 50K pool ...", flush=True)
    t0 = time.time()
    samples = build_pool(
        train_x, train_y, ids,
        aug_usable=artifacts["aug_usable"],
        global_mean_speed=artifacts["global_mean_speed"],
        include_window=False,
        horizon_for_basis=2,
        include_prev_basis_h1=False,
    )
    print(f"    pool M = {len(samples['p0']):,}, "
          f"traj_features.shape={samples['traj_features'].shape}, "
          f"basis_terms.shape={samples['basis_terms'].shape} "
          f"({time.time() - t0:.1f}s)", flush=True)

    oof_pred = np.zeros((n_orig, 3), dtype=np.float32)
    orig_fold_ids = samples["fold_id"][samples["is_orig_end10"]]
    folds_info = []
    t1 = time.time()
    for fold_k in range(5):
        print(f"\n  === fold {fold_k} ===", flush=True)
        info = train_one_fold(samples, fold_k, stage3_best_params,
                               n_epochs=args.n_epochs, patience=args.patience,
                               batch_size=args.batch_size)
        # Save fold checkpoint
        ckpt_path = RUN_DIR / f"checkpoint_fold{fold_k}.pt"
        torch.save({"state_dict": info["state_dict"],
                    "best_val_hit": info["best_val_hit"],
                    "fold": fold_k}, ckpt_path)
        del info["state_dict"]
        # Assemble OOF (val_indices 는 50K pool 의 indices, 그 중 original-only 추출)
        val_mask_orig = orig_fold_ids == fold_k
        oof_pred[val_mask_orig] = info["val_pred"]
        folds_info.append({k: v for k, v in info.items() if k not in ("val_pred", "val_indices")})
        print(f"  fold {fold_k} done: best_val_hit={info['best_val_hit']:.4f}, "
              f"epochs={info['n_epochs_run']}", flush=True)

    elapsed = time.time() - t1
    err = np.linalg.norm(oof_pred - train_y, axis=1)
    oof_hit = float((err <= R_HIT).mean())
    in_range = A0_OOF_MIN <= oof_hit <= A0_OOF_MAX
    print(f"\n[plan-019 G0] A0 OOF = {oof_hit:.4f}, "
          f"target ∈ [{A0_OOF_MIN}, {A0_OOF_MAX}], in_range={in_range}", flush=True)
    print(f"  elapsed (training) = {elapsed:.1f}s", flush=True)

    np.savez(RUN_DIR / "oof_predictions.npz",
              oof_pred=oof_pred, train_y=train_y, ids=np.asarray(ids))

    result = {
        "exp_id": "F013_a0",
        "plan_version": "v1.1",
        "oof_hit": oof_hit,
        "a0_oof_target_min": A0_OOF_MIN,
        "a0_oof_target_max": A0_OOF_MAX,
        "g0_passed": in_range,
        "best_basis_vars": BEST_BASIS_VARS,
        "n_coeffs": 8,
        "feat_dim": 13,
        "stage3_best_params": stage3_best_params.tolist(),
        "elapsed_sec": elapsed,
        "device": DEVICE,
        "folds": folds_info,
    }
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  -> {args.out_json}", flush=True)

    return 0 if in_range else 1


if __name__ == "__main__":
    sys.exit(main())
