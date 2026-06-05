"""plan-007 STAGE 4 — per-sample MLP coefficient regression.

Spec: plans/plan-007-formula-tuning.md §7.

CoefficientMLP outputs per-sample basis coefficients (1 hidden × 32). Bias init from Step 3
best_basis_params; weight init 0 (= untrained = global Step 3 baseline). Soft-hit loss
(sigmoid·sharpness=200). 5-fold OOF: fold-wise model+optimizer reinit. Val = remaining fold's
*original end_idx=10 only*. 50 epoch + early stop (patience=8, min_delta=1e-4).

Outputs:
  - analysis/plan-007/mlp_coeff.json
  - runs/baseline/F002_formula-mlp/checkpoint_fold{k}.pt (5 ckpts)
  - runs/baseline/F002_formula-mlp/oof_predictions.npz
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pb_0_6822 import selector

# Reuse helpers from sibling script (basis_ablation.py in same dir).
import sys
sys.path.insert(0, str(Path(__file__).parent))
from basis_ablation import compute_all_terms, stack_train_full, BASE_VARS  # noqa: E402

DATA_ROOT = Path("data")
OUT_DIR = Path("analysis/plan-007")
RUN_DIR = Path("runs/baseline/F002_formula-mlp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_HIT = 0.01


# ---------------------------------------------------------------------------
# Trajectory features (§7.2 — feat_dim=13)
# ---------------------------------------------------------------------------

def compute_trajectory_features(x_window: np.ndarray) -> np.ndarray:
    """x_window: (N, 6, 3) — last 6 steps before end_idx (inclusive).
    Returns (N, 13) float32.
    """
    pos_mean = x_window.mean(axis=1)                                      # (N, 3)
    pos_std = x_window.std(axis=1)                                        # (N, 3)
    pos_range = x_window.max(axis=1) - x_window.min(axis=1)               # (N, 3)
    deltas = np.diff(x_window, axis=1)                                    # (N, 5, 3)
    speed_norms = np.linalg.norm(deltas, axis=2)                          # (N, 5)
    speed_mean = speed_norms.mean(axis=1, keepdims=True)                  # (N, 1)
    speed_std = speed_norms.std(axis=1, keepdims=True)                    # (N, 1)
    speed_max = speed_norms.max(axis=1, keepdims=True)                    # (N, 1)
    speed_last = speed_norms[:, -1:]                                       # (N, 1)
    feats = np.concatenate(
        [pos_mean, pos_std, pos_range, speed_mean, speed_std, speed_max, speed_last],
        axis=1,
    )
    assert feats.shape[1] == 13, feats.shape
    return feats.astype(np.float32)


def _window_for_end_idx(x: np.ndarray, end_idx: int) -> np.ndarray:
    """Last 6 steps ending at end_idx (inclusive). For end_idx=10 → x[:, 5:11]."""
    return x[:, end_idx - 5 : end_idx + 1]


# ---------------------------------------------------------------------------
# Model (§7.1)
# ---------------------------------------------------------------------------

class CoefficientMLP(nn.Module):
    def __init__(self, feat_dim: int, n_coeffs: int, global_init: np.ndarray):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, n_coeffs),
        )
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor(global_init, dtype=torch.float32))
            self.mlp[-1].weight.zero_()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


def soft_hit_loss(pred: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.01, sharpness: float = 200.0) -> torch.Tensor:
    err = torch.norm(pred - target, dim=1)
    return torch.sigmoid(sharpness * (err - threshold)).mean()


def hit_rate(pred: torch.Tensor, target: torch.Tensor, r: float = R_HIT) -> float:
    err = torch.norm(pred - target, dim=1)
    return float((err <= r).float().mean())


def compute_pred(basis_terms: torch.Tensor, coeffs: torch.Tensor,
                  p0: torch.Tensor) -> torch.Tensor:
    """basis_terms: (B, n_coeffs, 3); coeffs: (B, n_coeffs); p0: (B, 3) → (B, 3)."""
    return p0 + (coeffs.unsqueeze(-1) * basis_terms).sum(dim=1)


# ---------------------------------------------------------------------------
# Build all samples (train pool 50K) with traj features + basis_terms + target + fold_id
# ---------------------------------------------------------------------------

def build_all_samples(stack: dict, train_x: np.ndarray, aug_usable: bool,
                       best_basis_vars: list[str]) -> dict:
    """Reuse the 50K stack from basis_ablation, but also compute trajectory features.

    The stack was built block-wise (original 10K + 4 sliding views). We need traj_features
    per block (different end_idx). Rebuild here for clarity.
    """
    feats_list = []
    end_idx_list = []
    block_sizes = []
    # original block (end_idx=10)
    feats_list.append(compute_trajectory_features(_window_for_end_idx(train_x, 10)))
    end_idx_list.append(10)
    block_sizes.append(train_x.shape[0])
    if aug_usable:
        for end_idx in range(5, 9):
            feats_list.append(compute_trajectory_features(_window_for_end_idx(train_x, end_idx)))
            end_idx_list.append(end_idx)
            block_sizes.append(train_x.shape[0])
    traj_features = np.concatenate(feats_list, axis=0)
    # basis_terms order = best_basis_vars; shape (M, n_coeffs, 3)
    basis_terms = np.stack([stack[v] for v in best_basis_vars], axis=1)
    p0 = stack["p0"]
    target = stack["target"]
    fold_id = stack["fold_id"]
    # is_original_end10 mask: True only for the original block (first block_sizes[0] rows)
    is_orig_end10 = np.zeros(len(p0), dtype=bool)
    is_orig_end10[: block_sizes[0]] = True
    return dict(
        traj_features=traj_features.astype(np.float32),
        basis_terms=basis_terms.astype(np.float32),
        p0=p0.astype(np.float32),
        target=target.astype(np.float32),
        fold_id=fold_id,
        is_orig_end10=is_orig_end10,
    )


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def train_one_fold(samples: dict, fold_k: int, best_basis_vars: list[str],
                    stage3_best_params: np.ndarray, n_epochs: int = 50,
                    patience: int = 8, min_delta: float = 1e-4,
                    batch_size: int = 1024, seed: int = 20260606) -> tuple[CoefficientMLP, dict]:
    torch.manual_seed(seed + fold_k)
    np.random.seed(seed + fold_k)

    is_train = samples["fold_id"] != fold_k
    is_val = (samples["fold_id"] == fold_k) & samples["is_orig_end10"]
    train_n, val_n = int(is_train.sum()), int(is_val.sum())

    feats_t = torch.from_numpy(samples["traj_features"][is_train]).to(DEVICE)
    terms_t = torch.from_numpy(samples["basis_terms"][is_train]).to(DEVICE)
    p0_t = torch.from_numpy(samples["p0"][is_train]).to(DEVICE)
    tgt_t = torch.from_numpy(samples["target"][is_train]).to(DEVICE)
    feats_v = torch.from_numpy(samples["traj_features"][is_val]).to(DEVICE)
    terms_v = torch.from_numpy(samples["basis_terms"][is_val]).to(DEVICE)
    p0_v = torch.from_numpy(samples["p0"][is_val]).to(DEVICE)
    tgt_v = torch.from_numpy(samples["target"][is_val]).to(DEVICE)

    n_coeffs = len(best_basis_vars)
    feat_dim = samples["traj_features"].shape[1]
    model = CoefficientMLP(feat_dim, n_coeffs=n_coeffs, global_init=stage3_best_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_ds = TensorDataset(feats_t, terms_t, p0_t, tgt_t)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_hit = -1.0
    best_state = None
    no_improve = 0
    epoch_history = []
    for epoch in range(n_epochs):
        model.train()
        ep_loss_sum = 0.0
        for batch in loader:
            feats_b, terms_b, p0_b, tgt_b = batch
            coeffs = model(feats_b)
            pred = compute_pred(terms_b, coeffs, p0_b)
            loss = soft_hit_loss(pred, tgt_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_loss_sum += float(loss.item()) * len(feats_b)
        ep_loss = ep_loss_sum / train_n

        model.eval()
        with torch.no_grad():
            coeffs_v = model(feats_v)
            pred_v = compute_pred(terms_v, coeffs_v, p0_v)
            val_hit = hit_rate(pred_v, tgt_v)
        epoch_history.append({"epoch": epoch, "train_loss": ep_loss, "val_hit": val_hit})

        improved = val_hit > best_val_hit + min_delta
        if improved:
            best_val_hit = val_hit
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # final val predictions (for OOF assembly)
    with torch.no_grad():
        coeffs_v = model(feats_v)
        pred_v = compute_pred(terms_v, coeffs_v, p0_v)
    info = dict(
        fold=fold_k,
        train_n=train_n,
        val_n=val_n,
        best_val_hit=best_val_hit,
        n_epochs_run=len(epoch_history),
        epoch_history=epoch_history,
        val_pred=pred_v.cpu().numpy(),
        val_target=tgt_v.cpu().numpy(),
        val_indices=np.flatnonzero(is_val),
    )
    return model, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DEVICE = {DEVICE}")

    sliding = json.loads((OUT_DIR / "sliding_validity.json").read_text())
    aug_usable = sliding["aug_usable"]
    stage2 = json.loads((OUT_DIR / "cma_es_step2.json").read_text())
    stage3 = json.loads((OUT_DIR / "basis_ablation.json").read_text())
    global_mean_speed = float(stage2["global_mean_speed"])
    best_basis_vars = stage3["best_basis_vars"]
    stage3_best_params = np.asarray(stage3["best_basis_params"], dtype=np.float32)
    stage3_best_hit = float(stage3["best_basis_hit"])
    print(f"aug_usable={aug_usable}, best_basis_vars={best_basis_vars}, stage3_best_hit={stage3_best_hit:.4f}")

    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)

    print("Building stack + traj features...")
    stack = stack_train_full(aug_usable, train_x, train_y, ids, global_mean_speed)
    samples = build_all_samples(stack, train_x, aug_usable, best_basis_vars)
    print(f"pool M = {len(samples['p0']):,}, traj_features shape = {samples['traj_features'].shape}")
    print(f"is_orig_end10 count = {int(samples['is_orig_end10'].sum())}")

    # 5-fold OOF training
    n_orig = len(ids)
    oof_pred = np.zeros((n_orig, 3), dtype=np.float32)
    orig_fold_ids = samples["fold_id"][:n_orig]
    folds_info = []
    t0 = time.time()
    for fold_k in range(5):
        print(f"\n  === fold {fold_k} ===")
        model, info = train_one_fold(samples, fold_k, best_basis_vars, stage3_best_params)
        # Save checkpoint
        ckpt_path = RUN_DIR / f"checkpoint_fold{fold_k}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "best_basis_vars": best_basis_vars,
                    "best_val_hit": info["best_val_hit"]}, ckpt_path)
        # Assemble OOF pred via original-fold-id mask
        val_mask_orig = orig_fold_ids == fold_k
        oof_pred[val_mask_orig] = info["val_pred"]
        folds_info.append({
            "fold": info["fold"],
            "train_n": info["train_n"],
            "val_n": info["val_n"],
            "best_val_hit": info["best_val_hit"],
            "n_epochs_run": info["n_epochs_run"],
        })
        print(f"  fold {fold_k} done: best_val_hit={info['best_val_hit']:.4f}, "
              f"epochs={info['n_epochs_run']}")

    elapsed = time.time() - t0
    err_oof = np.linalg.norm(oof_pred - train_y, axis=1)
    oof_hit = float((err_oof <= R_HIT).mean())
    print(f"\noof_hit (5-fold concat, original 10K) = {oof_hit:.4f}")
    print(f"stage3 baseline = {stage3_best_hit:.4f}, gain = {oof_hit - stage3_best_hit:+.4f}")
    print(f"G3 target: oof_hit ≥ {stage3_best_hit + 0.005:.4f} → {'PASS' if oof_hit >= stage3_best_hit + 0.005 else 'FAIL'}")
    print(f"total elapsed: {elapsed:.1f}s")

    np.savez(RUN_DIR / "oof_predictions.npz",
             oof_pred=oof_pred, train_y=train_y, ids=np.asarray(ids))
    result = {
        "oof_hit": oof_hit,
        "stage3_best_hit": stage3_best_hit,
        "oof_gain_vs_stage3": oof_hit - stage3_best_hit,
        "g3_threshold": stage3_best_hit + 0.005,
        "g3_pass": oof_hit >= stage3_best_hit + 0.005,
        "best_basis_vars": best_basis_vars,
        "n_coeffs": len(best_basis_vars),
        "feat_dim": int(samples["traj_features"].shape[1]),
        "elapsed_sec": elapsed,
        "folds": folds_info,
        "device": DEVICE,
    }
    (OUT_DIR / "mlp_coeff.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
