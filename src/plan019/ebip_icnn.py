"""plan-019 c7 STAGE 2 — EBIP + ICNN Convex Energy.

§6 spec + §0.5 spec-amendment 적용:
  - cond_dim 13 → 77d (handcrafted ⊕ cnn). ICNN convexity 는 *p (3-d) 에만*, c 는 non-convex 허용.
  - §6.1 의 1-step Newton with diagonal Hessian → **unrolled GD T=3** 로 변경 (§0.5 High amendment).
    convex 보장이라 발산 X. 1-step diagonal Newton 은 사실상 constant-LR GD — ICNN advantage 실종.

ICNNEnergy (Amos 2017 fully-input-convex variant):
  - z_0   = σ(W_p^0 · p + W_c^0 · c + b^0)
  - z_l   = σ(W_z^l z_{l-1} + W_p^l p + W_c^l c + b^l),  W_z^l ≥ 0 via softplus
  - out   = (z_L · w_out_z).sum + w_out_p · p + w_out_c · c, w_out_z ≥ 0 via softplus

Usage:
    python -m src.plan019.ebip_icnn --out-json analysis/plan-019/s2_ebip_icnn.json

Outputs:
    analysis/plan-019/s2_ebip_icnn.json
    runs/baseline/F015_ebip-icnn/oof_predictions.npz
    runs/baseline/F015_ebip-icnn/checkpoint_fold{k}.pt
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.plan019.cnn_encoder import TrajectoryCNNEncoder  # noqa: E402
from src.plan019.common import (  # noqa: E402
    R_HIT,
    BEST_BASIS_VARS,
    build_pool,
    hit_rate,
    load_artifacts,
    load_data,
    soft_hit_loss,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = REPO_ROOT / "runs/baseline/F015_ebip-icnn"
OUT_DIR = REPO_ROOT / "analysis/plan-019"


class ICNNEnergy(nn.Module):
    """Input Convex NN — g_θ(p, c) is convex in p (Amos 2017 §3.2 fully-convex).

    p (3-d) 가 convex 변수, c (77-d cond) 는 non-convex 허용. weight 는 softplus 로
    non-negative reparametrize (W_z, w_out_z).
    """
    def __init__(self, *, p_dim: int = 3, c_dim: int = 77,
                 hidden: int = 32, n_layers: int = 2):
        super().__init__()
        self.W_p = nn.ModuleList([nn.Linear(p_dim, hidden, bias=False) for _ in range(n_layers + 1)])
        self._W_z_raw = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, hidden) * 0.01) for _ in range(n_layers)
        ])
        self.W_c = nn.ModuleList([nn.Linear(c_dim, hidden) for _ in range(n_layers + 1)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(n_layers + 1)])
        self.W_out_p = nn.Linear(p_dim, 1, bias=False)
        self._W_out_z_raw = nn.Parameter(torch.randn(hidden) * 0.01)
        self.W_out_c = nn.Linear(c_dim, 1)

    def _W_z(self, idx: int) -> torch.Tensor:
        return F.softplus(self._W_z_raw[idx])

    def _W_out_z(self) -> torch.Tensor:
        return F.softplus(self._W_out_z_raw)

    def forward(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """p: (B, 3), c: (B, 77). Returns (B,) scalar energy."""
        z = F.silu(self.W_p[0](p) + self.W_c[0](c) + self.b[0])
        for l in range(len(self._W_z_raw)):
            z = F.silu(
                z @ self._W_z(l).T + self.W_p[l + 1](p) + self.W_c[l + 1](c) + self.b[l + 1]
            )
        out = (z * self._W_out_z()).sum(dim=1, keepdim=True) \
              + self.W_out_p(p) + self.W_out_c(c)
        return out.squeeze(-1)

    def softplus_min(self) -> float:
        """ICNN constraint check helper — min softplus(W_z_raw) ≥ 0 expected."""
        vals = [float(F.softplus(w).min()) for w in self._W_z_raw]
        vals.append(float(F.softplus(self._W_out_z_raw).min()))
        return min(vals)


class EBIPICNN(nn.Module):
    """EBIP base 의 energy_mlp 를 ICNNEnergy 로 교체. argmin = unrolled GD T=3 (§0.5 amendment).

    EBIPBase 와 동일 conditioning (77d) — 코드 중복 최소화 위해 TrajectoryCNNEncoder 신규 instance.
    """
    def __init__(self, *, handcraft_dim: int = 13, cnn_dim: int = 64,
                 n_coeffs: int = 8, global_init: np.ndarray | None = None,
                 icnn_hidden: int = 32, icnn_layers: int = 2,
                 unroll_T: int = 3, inner_lr: float = 0.1):
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
        self.icnn = ICNNEnergy(p_dim=3, c_dim=cond_dim,
                                hidden=icnn_hidden, n_layers=icnn_layers)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.unroll_T = unroll_T
        self.inner_lr = inner_lr
        self.handcraft_dim = handcraft_dim
        self.cnn_dim = cnn_dim

    def encode(self, traj_features: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        return torch.cat([traj_features, self.cnn(window)], dim=1)

    def forward(self, traj_features: torch.Tensor, window: torch.Tensor,
                p0: torch.Tensor, basis_terms: torch.Tensor) -> torch.Tensor:
        cond = self.encode(traj_features, window)                                  # (B, 77)
        coeffs = self.coeff_mlp(cond)                                              # (B, 8)
        anchor = p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)              # (B, 3)
        lam = torch.exp(self.log_lambda)

        # E(p; c) = ||p - anchor||² + λ · g_icnn(p, c)   — convex in p (ICNN 보장)
        p = anchor + 0.0
        for _t in range(self.unroll_T):
            p_in = p if self.training else p.detach().requires_grad_(True)
            e_icnn = self.icnn(p_in, cond)
            energy = ((p_in - anchor) ** 2).sum(dim=1).sum() + lam * e_icnn.sum()
            grad_p = torch.autograd.grad(energy, p_in,
                                          create_graph=self.training)[0]
            if self.training:
                p = p_in - self.inner_lr * grad_p
            else:
                p = (p_in - self.inner_lr * grad_p).detach()
        return p


def train_one_fold(samples: dict, fold_k: int, stage3_best_params: np.ndarray, *,
                    n_epochs: int = 50, patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 256, unroll_T: int = 3,
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

    model = EBIPICNN(global_init=stage3_best_params, unroll_T=unroll_T).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_train = feats_t.shape[0]
    best_val = -1.0
    best_state = None
    no_improve = 0
    history = []
    icnn_min_softplus_per_epoch = []
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        ep_loss = 0.0
        nan_seen = False
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            pred = model(feats_t[idx], win_t[idx], p0_t[idx], terms_t[idx])
            loss = soft_hit_loss(pred, tgt_t[idx])
            if not torch.isfinite(loss):
                nan_seen = True
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item()) * idx.shape[0]
        if nan_seen:
            print(f"  ⚠️ fold {fold_k} epoch {epoch}: NaN loss — stop early", flush=True)
            break
        ep_loss /= n_train

        # ICNN constraint check (§6.2)
        min_sp = model.icnn.softplus_min()
        icnn_min_softplus_per_epoch.append(min_sp)

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
        val_hit = hit_rate(pred_v, tgt_v)
        history.append({"epoch": epoch, "train_loss": ep_loss,
                        "val_hit": val_hit, "icnn_min_softplus": min_sp})

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
    with torch.enable_grad():
        pred_v_list = []
        chunk = 1024
        for j in range(0, feats_v.shape[0], chunk):
            pred_v_list.append(
                model(feats_v[j:j + chunk], win_v[j:j + chunk],
                      p0_v[j:j + chunk], terms_v[j:j + chunk]).detach()
            )
        pred_v = torch.cat(pred_v_list, dim=0)
    icnn_min_softplus = float(min(icnn_min_softplus_per_epoch)) if icnn_min_softplus_per_epoch else None
    return {
        "fold": fold_k,
        "train_n": int(is_train.sum()),
        "val_n": int(is_val.sum()),
        "best_val_hit": float(best_val),
        "n_epochs_run": len(history),
        "n_params": n_params,
        "icnn_min_softplus_observed": icnn_min_softplus,
        "val_pred": pred_v.detach().cpu().numpy(),
        "state_dict": best_state,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "s2_ebip_icnn.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--unroll-T", type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plan-019 c8 / G2] DEVICE={DEVICE}, unroll_T={args.unroll_T}", flush=True)

    artifacts = load_artifacts()
    stage3_best_params = artifacts["stage3_best_params"]

    ids, train_x, train_y = load_data()
    n_orig = len(ids)
    print(f"  N_orig={n_orig}, best_basis_vars={BEST_BASIS_VARS}", flush=True)

    print("  Building 50K pool ...", flush=True)
    t0 = time.time()
    samples = build_pool(
        train_x, train_y, ids,
        aug_usable=artifacts["aug_usable"],
        global_mean_speed=artifacts["global_mean_speed"],
        include_window=True,
        horizon_for_basis=2,
    )
    print(f"    pool M = {len(samples['p0']):,} ({time.time() - t0:.1f}s)", flush=True)

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
              f"epochs={info['n_epochs_run']}, icnn_min_softplus={info['icnn_min_softplus_observed']}",
              flush=True)

    elapsed = time.time() - t1
    err = np.linalg.norm(oof_pred - train_y, axis=1)
    oof_hit = float((err <= R_HIT).mean())
    g2_threshold = 0.68
    g2_passed = oof_hit >= g2_threshold
    print(f"\n[plan-019 G2] S2 EBIP+ICNN OOF = {oof_hit:.4f}, "
          f"threshold {g2_threshold}, passed={g2_passed}", flush=True)
    print(f"  elapsed = {elapsed:.1f}s", flush=True)

    np.savez(RUN_DIR / "oof_predictions.npz",
              oof_pred=oof_pred, train_y=train_y, ids=np.asarray(ids))

    result = {
        "exp_id": "F015_ebip-icnn",
        "plan_version": "v1.1",
        "stage": "S2_EBIP_ICNN",
        "oof_hit": oof_hit,
        "g2_threshold": g2_threshold,
        "g2_passed": g2_passed,
        "best_basis_vars": BEST_BASIS_VARS,
        "n_coeffs": 8,
        "handcraft_dim": 13,
        "cnn_dim": 64,
        "cond_dim": 77,
        "icnn_hidden": 32,
        "icnn_n_layers": 2,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
