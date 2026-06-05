"""plan-019 c5 STAGE 1 — EBIP base (Energy-Based Implicit Prediction).

§5 spec + §0.5 spec-amendment:
  - feat_dim 본 spec 13 → amendment 77 (= 13 handcrafted ⊕ 64 cnn_encoded).
  - 본 plan default unroll_T=5, inner_lr=0.1, λ_init=exp(0)=1.0.
  - energy = ||p - anchor||² + λ · g_θ(p, cond_77d), g_θ small MLP (3+77 → 32 → 32 → 1).
  - forward: unrolled gradient descent on p, T steps, create_graph=self.training (FOMAML-friendly).

Usage:
    python -m src.archive.plan019.ebip_base --out-json analysis/archive/plan-019/s1_ebip_base.json

Outputs:
    analysis/archive/plan-019/s1_ebip_base.json
    runs/baseline/F014_ebip-base/oof_predictions.npz
    runs/baseline/F014_ebip-base/checkpoint_fold{k}.pt
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

from src.archive.plan019.cnn_encoder import TrajectoryCNNEncoder  # noqa: E402
from src.archive.plan019.common import (  # noqa: E402
    R_HIT,
    BEST_BASIS_VARS,
    build_pool,
    hit_rate,
    load_artifacts,
    load_data,
    soft_hit_loss,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = REPO_ROOT / "runs/baseline/F014_ebip-base"
OUT_DIR = REPO_ROOT / "analysis/plan-019"


class EBIPBase(nn.Module):
    """Energy-Based Implicit Prediction — bilinear anchor + small MLP correction.

    cond_dim = 13 (handcrafted) + 64 (cnn) = 77.
    energy_mlp: 3 + 77 → 32 → 32 → 1.
    coeff_mlp:  77 → 32 → 8, last-layer bias init = stage3_best_params, weight=0
    (untrained 시 anchor = plan-007 step 3 baseline).
    """
    def __init__(self, *, handcraft_dim: int = 13, cnn_dim: int = 64,
                 n_coeffs: int = 8, global_init: np.ndarray | None = None,
                 hidden: int = 32, unroll_T: int = 5, inner_lr: float = 0.1):
        super().__init__()
        self.cnn = TrajectoryCNNEncoder(in_channels=3, hidden=cnn_dim)
        cond_dim = handcraft_dim + cnn_dim
        self.coeff_mlp = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        if global_init is not None:
            with torch.no_grad():
                self.coeff_mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
                self.coeff_mlp[-1].weight.zero_()

        # energy MLP: input = (p ∈ R^3) ⊕ cond — convexity 미요구 (S2 ICNN 의 비교 baseline)
        self.energy_mlp = nn.Sequential(
            nn.Linear(3 + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.log_lambda = nn.Parameter(torch.tensor(0.0))     # λ_init = exp(0) = 1.0
        self.unroll_T = unroll_T
        self.inner_lr = inner_lr

    def encode(self, traj_features: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(window)                                                   # (B, 64)
        return torch.cat([traj_features, cnn_feat], dim=1)                            # (B, 77)

    def forward(self, traj_features: torch.Tensor, window: torch.Tensor,
                p0: torch.Tensor, basis_terms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_features:  (B, 13)
            window:         (B, 6, 3)
            p0:             (B, 3)
            basis_terms:    (B, 8, 3)
        Returns:
            pred:           (B, 3)
        """
        cond = self.encode(traj_features, window)                                     # (B, 77)
        coeffs = self.coeff_mlp(cond)                                                 # (B, 8)
        anchor = p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)                 # (B, 3)

        lam = torch.exp(self.log_lambda)
        p = anchor + 0.0   # differentiable starting point (preserve graph for outer-loop)
        for _t in range(self.unroll_T):
            p_in = p.detach().requires_grad_(True) if not self.training else p
            # NOTE: 학습 시 create_graph=True 로 outer backprop, 평가 시 first-order grad 만.
            if self.training:
                e_corr = self.energy_mlp(torch.cat([p_in, cond], dim=1)).squeeze(-1)
                # batch 단위 grad: 각 sample 의 energy 미분이 independent — sum 으로 합산 후 grad
                energy = ((p_in - anchor) ** 2).sum(dim=1).sum() + lam * e_corr.sum()
                grad_p = torch.autograd.grad(energy, p_in, create_graph=True)[0]
                p = p_in - self.inner_lr * grad_p
            else:
                e_corr = self.energy_mlp(torch.cat([p_in, cond], dim=1)).squeeze(-1)
                energy = ((p_in - anchor) ** 2).sum(dim=1).sum() + lam * e_corr.sum()
                grad_p = torch.autograd.grad(energy, p_in, create_graph=False)[0]
                p = (p_in - self.inner_lr * grad_p).detach()
        return p


def train_one_fold(samples: dict, fold_k: int, stage3_best_params: np.ndarray, *,
                    n_epochs: int = 50, patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 256, unroll_T: int = 5,
                    seed: int = 20260606) -> dict:
    torch.manual_seed(seed + fold_k)
    np.random.seed(seed + fold_k)

    is_train = samples["fold_id"] != fold_k
    is_val = (samples["fold_id"] == fold_k) & samples["is_orig_end10"]

    feats_t = torch.from_numpy(samples["traj_features"][is_train]).to(DEVICE)
    win_t = torch.from_numpy(samples["window"][is_train]).to(DEVICE)
    terms_t = torch.from_numpy(samples["basis_terms"][is_train]).to(DEVICE)
    p0_t = torch.from_numpy(samples["p0"][is_train]).to(DEVICE)
    tgt_t = torch.from_numpy(samples["target"][is_train]).to(DEVICE)
    feats_v = torch.from_numpy(samples["traj_features"][is_val]).to(DEVICE)
    win_v = torch.from_numpy(samples["window"][is_val]).to(DEVICE)
    terms_v = torch.from_numpy(samples["basis_terms"][is_val]).to(DEVICE)
    p0_v = torch.from_numpy(samples["p0"][is_val]).to(DEVICE)
    tgt_v = torch.from_numpy(samples["target"][is_val]).to(DEVICE)

    model = EBIPBase(global_init=stage3_best_params, unroll_T=unroll_T).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
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
            pred = model(feats_t[idx], win_t[idx], p0_t[idx], terms_t[idx])
            loss = soft_hit_loss(pred, tgt_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item()) * idx.shape[0]
        ep_loss /= n_train

        model.eval()
        with torch.enable_grad():
            # eval 도 unrolled GD 필요 — torch.no_grad() 사용 X.
            # 단 outer-graph 차단을 위해 forward 내부에서 detach.
            pred_v_list = []
            chunk = 1024
            for j in range(0, feats_v.shape[0], chunk):
                pred_v_list.append(
                    model(feats_v[j:j + chunk], win_v[j:j + chunk],
                          p0_v[j:j + chunk], terms_v[j:j + chunk]).detach()
                )
            pred_v = torch.cat(pred_v_list, dim=0)
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

    # final eval w/ best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.enable_grad():
        pred_v_list = []
        chunk = 1024
        for j in range(0, feats_v.shape[0], chunk):
            pred_v_list.append(
                model(feats_v[j:j + chunk], win_v[j:j + chunk],
                      p0_v[j:j + chunk], terms_v[j:j + chunk]).detach()
            )
        pred_v = torch.cat(pred_v_list, dim=0)
    return {
        "fold": fold_k,
        "train_n": int(is_train.sum()),
        "val_n": int(is_val.sum()),
        "best_val_hit": float(best_val),
        "n_epochs_run": len(history),
        "n_params": n_params,
        "val_pred": pred_v.detach().cpu().numpy(),
        "state_dict": best_state,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "s1_ebip_base.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--unroll-T", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plan-019 c6 / G1] DEVICE={DEVICE}, unroll_T={args.unroll_T}", flush=True)

    artifacts = load_artifacts()
    stage3_best_params = artifacts["stage3_best_params"]
    print(f"  best_basis_vars={BEST_BASIS_VARS}, stage3_best_params={stage3_best_params.tolist()}",
          flush=True)

    ids, train_x, train_y = load_data()
    n_orig = len(ids)
    print(f"  N_orig={n_orig}", flush=True)

    print("  Building 50K pool ...", flush=True)
    t0 = time.time()
    samples = build_pool(
        train_x, train_y, ids,
        aug_usable=artifacts["aug_usable"],
        global_mean_speed=artifacts["global_mean_speed"],
        include_window=True,
        horizon_for_basis=2,
        include_prev_basis_h1=False,
    )
    print(f"    pool M = {len(samples['p0']):,}, "
          f"window.shape={samples['window'].shape} ({time.time() - t0:.1f}s)", flush=True)

    oof_pred = np.zeros((n_orig, 3), dtype=np.float32)
    orig_fold_ids = samples["fold_id"][samples["is_orig_end10"]]
    folds_info = []
    t1 = time.time()
    for fold_k in range(5):
        print(f"\n  === fold {fold_k} ===", flush=True)
        info = train_one_fold(samples, fold_k, stage3_best_params,
                               n_epochs=args.n_epochs, patience=args.patience,
                               batch_size=args.batch_size, unroll_T=args.unroll_T)
        ckpt_path = RUN_DIR / f"checkpoint_fold{fold_k}.pt"
        torch.save({"state_dict": info["state_dict"],
                    "best_val_hit": info["best_val_hit"],
                    "fold": fold_k}, ckpt_path)
        del info["state_dict"]
        val_mask = orig_fold_ids == fold_k
        oof_pred[val_mask] = info["val_pred"]
        folds_info.append({k: v for k, v in info.items() if k != "val_pred"})
        print(f"  fold {fold_k} done: best_val_hit={info['best_val_hit']:.4f}, "
              f"epochs={info['n_epochs_run']}, params={info['n_params']}", flush=True)

    elapsed = time.time() - t1
    err = np.linalg.norm(oof_pred - train_y, axis=1)
    oof_hit = float((err <= R_HIT).mean())
    g1_threshold = 0.66
    g1_passed = oof_hit >= g1_threshold
    print(f"\n[plan-019 G1] S1 EBIP base OOF = {oof_hit:.4f}, "
          f"threshold {g1_threshold}, passed={g1_passed}", flush=True)
    print(f"  elapsed = {elapsed:.1f}s", flush=True)

    np.savez(RUN_DIR / "oof_predictions.npz",
              oof_pred=oof_pred, train_y=train_y, ids=np.asarray(ids))

    result = {
        "exp_id": "F014_ebip-base",
        "plan_version": "v1.1",
        "stage": "S1_EBIP_base",
        "oof_hit": oof_hit,
        "g1_threshold": g1_threshold,
        "g1_passed": g1_passed,
        "best_basis_vars": BEST_BASIS_VARS,
        "n_coeffs": 8,
        "handcraft_dim": 13,
        "cnn_dim": 64,
        "cond_dim": 77,
        "unroll_T": args.unroll_T,
        "inner_lr": 0.1,
        "batch_size": args.batch_size,
        "n_epochs_max": args.n_epochs,
        "patience": args.patience,
        "elapsed_sec": elapsed,
        "device": DEVICE,
        "folds": folds_info,
    }
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  -> {args.out_json}", flush=True)
    return 0 if g1_passed else 0   # G1 warn-only — non-zero exit X (severe 아님)


if __name__ == "__main__":
    sys.exit(main())
