"""plan-019 c9 STAGE 3 — Meta-EBIP + ICNN (FOMAML inner loop).

§7 spec + §0.5 spec-amendments 적용:
  - cond_dim 13 → 77d (handcrafted ⊕ cnn). ICNN convexity on p, c non-convex.
  - §7.2 의 1-step Newton → S2 와 동일하게 **unrolled GD T=3** (§0.5 High amendment).
  - basis_terms_prev = compute_basis_terms(x, end_idx-1, horizon=1) — +40ms 예측용
  - basis_terms      = compute_basis_terms(x, end_idx,   horizon=2) — +80ms 예측용
  - 두 basis 는 build_pool 에서 분리 저장 (include_prev_basis_h1=True).

FOMAML inner loop (Finn 2017, first-order MAML):
  c_meta = coeff_mlp(cond)                                  # (B, 8)
  L_unsup(c) = ||window[-1] - (window[-2] + c · B_prev)||²   # self-supervised reconstruction
  c_τ = c_meta - η · ∇_c L_unsup(c_meta)                    # 1-step SGD on c
  pred = argmin_p [||p - p0 - c_τ · B||² + λ · g_icnn(p, c)] # unrolled GD T=3

Usage:
    python -m src.archive.plan019.meta_ebip_icnn --out-json analysis/archive/plan-019/s3_meta_ebip_icnn.json

Outputs:
    analysis/archive/plan-019/s3_meta_ebip_icnn.json
    runs/baseline/F016_meta-ebip-icnn/oof_predictions.npz
    runs/baseline/F016_meta-ebip-icnn/checkpoint_fold{k}.pt
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
from src.archive.plan019.ebip_icnn import ICNNEnergy  # noqa: E402  (S2 의 ICNN 재사용 OK — 같은 plan 내)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = REPO_ROOT / "runs/baseline/F016_meta-ebip-icnn"
OUT_DIR = REPO_ROOT / "analysis/plan-019"


class MetaEBIPICNN(nn.Module):
    """meta-EBIP + ICNN — FOMAML inner loop adaptation + ICNN convex energy.

    inner_steps=0 이면 S2 EBIPICNN 와 동일 (validation comparison check 가능).
    """
    def __init__(self, *, handcraft_dim: int = 13, cnn_dim: int = 64,
                 n_coeffs: int = 8, global_init: np.ndarray | None = None,
                 icnn_hidden: int = 32, icnn_layers: int = 2,
                 unroll_T: int = 3, p_inner_lr: float = 0.1,
                 c_inner_lr: float = 0.01, c_inner_steps: int = 1):
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
        self.p_inner_lr = p_inner_lr
        self.c_inner_lr = c_inner_lr
        self.c_inner_steps = c_inner_steps

    def encode(self, traj_features: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        return torch.cat([traj_features, self.cnn(window)], dim=1)

    @staticmethod
    def _self_supervised_loss(c: torch.Tensor, window: torch.Tensor,
                                basis_terms_prev: torch.Tensor) -> torch.Tensor:
        """Inner-loop pretext: predict window[-1] from window[-2] using c · basis_prev.

        Sample-wise sum (independent inner-loop per sample).
        """
        pred_recon = window[:, -2] + (c.unsqueeze(-1) * basis_terms_prev).sum(dim=1)
        # per-sample squared error sum — outer FOMAML 가 sum 으로 batch independent grad.
        return ((pred_recon - window[:, -1]) ** 2).sum()

    def forward(self, traj_features: torch.Tensor, window: torch.Tensor,
                p0: torch.Tensor, basis_terms: torch.Tensor,
                basis_terms_prev: torch.Tensor) -> torch.Tensor:
        cond = self.encode(traj_features, window)                                       # (B, 77)
        c_meta = self.coeff_mlp(cond)                                                   # (B, 8)

        # FOMAML inner adaptation — first-order (create_graph=False ⇒ outer-backprop X)
        c_tau = c_meta
        for _ in range(self.c_inner_steps):
            c_in = c_tau.detach().requires_grad_(True)
            L_unsup = self._self_supervised_loss(c_in, window, basis_terms_prev)
            grad_c = torch.autograd.grad(L_unsup, c_in, create_graph=False)[0]
            # FOMAML: c_meta - η·grad. outer backprop 는 c_meta 위에서만 (Finn 2017 §5.2).
            # 즉 outer gradient 가 c_tau 로 propagate 하지 않음 — 다음 line 의 (c_meta - ...).
            c_tau = c_meta - self.c_inner_lr * grad_c.detach()

        # S2 의 EBIPICNN forward — c 를 c_tau 로 대체. unrolled GD T=3 on p.
        anchor = p0 + (c_tau.unsqueeze(-1) * basis_terms).sum(dim=1)
        lam = torch.exp(self.log_lambda)

        p = anchor + 0.0
        for _t in range(self.unroll_T):
            p_in = p if self.training else p.detach().requires_grad_(True)
            e_icnn = self.icnn(p_in, cond)
            energy = ((p_in - anchor) ** 2).sum(dim=1).sum() + lam * e_icnn.sum()
            grad_p = torch.autograd.grad(energy, p_in,
                                          create_graph=self.training)[0]
            if self.training:
                p = p_in - self.p_inner_lr * grad_p
            else:
                p = (p_in - self.p_inner_lr * grad_p).detach()
        return p


def train_one_fold(samples: dict, fold_k: int, stage3_best_params: np.ndarray, *,
                    n_epochs: int = 50, patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 256, unroll_T: int = 3,
                    c_inner_lr: float = 0.01, c_inner_steps: int = 1,
                    seed: int = 20260606) -> dict:
    torch.manual_seed(seed + fold_k)
    np.random.seed(seed + fold_k)

    is_train = samples["fold_id"] != fold_k
    is_val = (samples["fold_id"] == fold_k) & samples["is_orig_end10"]

    def to_d(key):
        return torch.from_numpy(samples[key][is_train]).to(DEVICE)

    def to_v(key):
        return torch.from_numpy(samples[key][is_val]).to(DEVICE)

    feats_t, win_t, terms_t, terms_prev_t, p0_t, tgt_t = [to_d(k) for k in
        ("traj_features", "window", "basis_terms", "basis_terms_prev", "p0", "target")]
    feats_v, win_v, terms_v, terms_prev_v, p0_v, tgt_v = [to_v(k) for k in
        ("traj_features", "window", "basis_terms", "basis_terms_prev", "p0", "target")]

    model = MetaEBIPICNN(global_init=stage3_best_params, unroll_T=unroll_T,
                          c_inner_lr=c_inner_lr, c_inner_steps=c_inner_steps).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_train = feats_t.shape[0]
    best_val = -1.0
    best_state = None
    no_improve = 0
    history = []
    inner_divergent_per_epoch = []   # FOMAML inner loop divergence monitor
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        ep_loss = 0.0
        n_divergent = 0
        n_seen = 0
        nan_seen = False
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            # Monitor c_τ divergence: ||c_τ - c_meta||² > 10 → plan-specific severe
            # (단 epoch summary, batch-level halt X — early-stop 으로 자율 처리)
            pred = model(feats_t[idx], win_t[idx], p0_t[idx],
                          terms_t[idx], terms_prev_t[idx])
            loss = soft_hit_loss(pred, tgt_t[idx])
            if not torch.isfinite(loss):
                nan_seen = True
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item()) * idx.shape[0]
            n_seen += idx.shape[0]
        if nan_seen:
            print(f"  ⚠️ fold {fold_k} epoch {epoch}: NaN loss — stop early", flush=True)
            break
        ep_loss /= n_train
        inner_divergent_per_epoch.append(n_divergent / max(n_seen, 1))

        model.eval()
        with torch.enable_grad():
            pred_v_list = []
            chunk = 1024
            for j in range(0, feats_v.shape[0], chunk):
                pred_v_list.append(
                    model(feats_v[j:j+chunk], win_v[j:j+chunk], p0_v[j:j+chunk],
                          terms_v[j:j+chunk], terms_prev_v[j:j+chunk]).detach()
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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.enable_grad():
        pred_v_list = []
        chunk = 1024
        for j in range(0, feats_v.shape[0], chunk):
            pred_v_list.append(
                model(feats_v[j:j+chunk], win_v[j:j+chunk], p0_v[j:j+chunk],
                      terms_v[j:j+chunk], terms_prev_v[j:j+chunk]).detach()
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
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "s3_meta_ebip_icnn.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--unroll-T", type=int, default=3)
    ap.add_argument("--c-inner-lr", type=float, default=0.01)
    ap.add_argument("--c-inner-steps", type=int, default=1)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plan-019 c10 / G3] DEVICE={DEVICE}, unroll_T={args.unroll_T}, "
          f"c_inner_lr={args.c_inner_lr}, c_inner_steps={args.c_inner_steps}", flush=True)

    artifacts = load_artifacts()
    stage3_best_params = artifacts["stage3_best_params"]

    ids, train_x, train_y = load_data()
    n_orig = len(ids)
    print(f"  N_orig={n_orig}", flush=True)

    print("  Building 50K pool (with basis_terms_prev) ...", flush=True)
    t0 = time.time()
    samples = build_pool(
        train_x, train_y, ids,
        aug_usable=artifacts["aug_usable"],
        global_mean_speed=artifacts["global_mean_speed"],
        include_window=True,
        horizon_for_basis=2,
        include_prev_basis_h1=True,
    )
    print(f"    pool M = {len(samples['p0']):,}, "
          f"basis_terms_prev.shape={samples['basis_terms_prev'].shape} "
          f"({time.time() - t0:.1f}s)", flush=True)

    oof_pred = np.zeros((n_orig, 3), dtype=np.float32)
    orig_fold_ids = samples["fold_id"][samples["is_orig_end10"]]
    folds_info = []
    t1 = time.time()
    for fold_k in range(5):
        print(f"\n  === fold {fold_k} ===", flush=True)
        info = train_one_fold(samples, fold_k, stage3_best_params,
                               n_epochs=args.n_epochs, patience=args.patience,
                               batch_size=args.batch_size, unroll_T=args.unroll_T,
                               c_inner_lr=args.c_inner_lr, c_inner_steps=args.c_inner_steps)
        ckpt_path = RUN_DIR / f"checkpoint_fold{fold_k}.pt"
        torch.save({"state_dict": info["state_dict"],
                    "best_val_hit": info["best_val_hit"],
                    "fold": fold_k}, ckpt_path)
        del info["state_dict"]
        val_mask = orig_fold_ids == fold_k
        oof_pred[val_mask] = info["val_pred"]
        folds_info.append({k: v for k, v in info.items() if k != "val_pred"})
        print(f"  fold {fold_k} done: best_val_hit={info['best_val_hit']:.4f}, "
              f"epochs={info['n_epochs_run']}", flush=True)

    elapsed = time.time() - t1
    err = np.linalg.norm(oof_pred - train_y, axis=1)
    oof_hit = float((err <= R_HIT).mean())
    g3_threshold = 0.70
    g3_passed = oof_hit >= g3_threshold
    print(f"\n[plan-019 G3] S3 meta-EBIP+ICNN OOF = {oof_hit:.4f}, "
          f"threshold {g3_threshold} ⭐, passed={g3_passed}", flush=True)
    print(f"  elapsed = {elapsed:.1f}s", flush=True)

    np.savez(RUN_DIR / "oof_predictions.npz",
              oof_pred=oof_pred, train_y=train_y, ids=np.asarray(ids))

    result = {
        "exp_id": "F016_meta-ebip-icnn",
        "plan_version": "v1.1",
        "stage": "S3_Meta_EBIP_ICNN",
        "oof_hit": oof_hit,
        "g3_threshold": g3_threshold,
        "g3_passed": g3_passed,
        "best_basis_vars": BEST_BASIS_VARS,
        "n_coeffs": 8,
        "handcraft_dim": 13,
        "cnn_dim": 64,
        "cond_dim": 77,
        "icnn_hidden": 32,
        "icnn_n_layers": 2,
        "unroll_T": args.unroll_T,
        "p_inner_lr": 0.1,
        "c_inner_lr": args.c_inner_lr,
        "c_inner_steps": args.c_inner_steps,
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
