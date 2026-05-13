"""plan-011 v1.2 — IC submission (R001 frozen GRU encoder).

Full-train (10000 samples) with R001_baseline-residual-gru fold0 frozen GRU + corrector → test prediction.

산출: runs/baseline/H012_phase1-input-ablation/sub_IC/submission.csv
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_loss_train as base  # type: ignore

from src.pb_0_6822 import selector as sel
from src.pb_0_6822 import corrector_redesign_v2 as v2

REPO = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SINGLE_IDX = 17
HORIZON = 2
END_IDX = 10
R_HIT = 0.01
SEED = 20260513 + 100 + ord("C")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "runs/baseline/H012_phase1-input-ablation/sub_IC")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r001-ckpt", type=Path,
                        default=REPO / "runs/baseline/R001_baseline-residual-gru/ckpt/fold0.pt")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Train data
    print("[data] loading full train (10000)...")
    df = pd.read_csv(args.data_root / "train_labels.csv")
    sample_ids = df["id"].astype(str).tolist()
    truth = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    x_seq = sel.load_stack(args.data_root / "train", sample_ids)
    cands_all = sel.make_candidates(x_seq, end_idx=END_IDX, horizon=HORIZON)
    cand = cands_all[:, SINGLE_IDX:SINGLE_IDX + 1, :][:, 0, :]
    cf = sel.make_candidate_features(x_seq, END_IDX, cands_all[:, SINGLE_IDX:SINGLE_IDX + 1, :],
                                      HORIZON, candidates_list=[sel.CANDIDATES[SINGLE_IDX]])[:, 0, :]
    target = (truth - cand).astype(np.float32)
    err = np.linalg.norm(truth - cand, axis=1).astype(np.float32)

    # Test data
    df_t = pd.read_csv(args.data_root / "sample_submission.csv")
    test_ids = df_t["id"].astype(str).tolist()
    print(f"[data] loading test ({len(test_ids)})...")
    x_seq_test = sel.load_stack(args.data_root / "test", test_ids)
    cands_test_all = sel.make_candidates(x_seq_test, end_idx=END_IDX, horizon=HORIZON)
    cand_test = cands_test_all[:, SINGLE_IDX:SINGLE_IDX + 1, :][:, 0, :]
    cf_test = sel.make_candidate_features(x_seq_test, END_IDX,
                                            cands_test_all[:, SINGLE_IDX:SINGLE_IDX + 1, :],
                                            HORIZON, candidates_list=[sel.CANDIDATES[SINGLE_IDX]])[:, 0, :]

    # R001 frozen GRU encoder
    print(f"[model] loading R001 frozen GRU from {args.r001_ckpt.relative_to(REPO)}...")
    gru_enc = v2.FrozenGRUEncoder(str(args.r001_ckpt), input_dim=3, hidden=64,
                                    num_layers=2, dropout=0.08).to(DEVICE)
    # corrector with dim_encoder=64
    corrector = v2.RedesignedCorrectionNet(dim_cf=32, hidden=64, dim_encoder=64).to(DEVICE)

    # Precompute encoder embeddings (frozen)
    with torch.no_grad():
        enc_train = gru_enc(torch.from_numpy(x_seq).to(DEVICE)).cpu().numpy()
        enc_test = gru_enc(torch.from_numpy(x_seq_test).to(DEVICE)).cpu().numpy()
    print(f"[encoder] train enc shape={enc_train.shape}, test enc shape={enc_test.shape}")

    # tensors
    cf_t = torch.from_numpy(cf).to(DEVICE)
    target_t = torch.from_numpy(target).to(DEVICE)
    err_t = torch.from_numpy(err).to(DEVICE)
    enc_t = torch.from_numpy(enc_train).to(DEVICE)
    cand_t = torch.from_numpy(cand).to(DEVICE)
    truth_t = torch.from_numpy(truth).to(DEVICE)

    opt = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=1e-4)
    N = cf_t.shape[0]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        corrector.train()
        perm = torch.randperm(N, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, N, args.batch):
            sel_ = perm[start:start + args.batch]
            cf_b, tg_b, er_b = cf_t[sel_], target_t[sel_], err_t[sel_]
            enc_b = enc_t[sel_]
            opt.zero_grad(set_to_none=True)
            delta, _ = corrector(cf_b, enc_b)
            per = v2.huber_loss(delta, tg_b)
            w = base.weight_schedule(er_b)
            loss = (per * w).sum() / (w.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(cf_b)
            n += len(cf_b)
        if epoch % 10 == 0:
            corrector.eval()
            with torch.no_grad():
                delta_all, _ = corrector(cf_t, enc_t)
                corrected = cand_t + v2.cap_6mm(delta_all)
                err_after = torch.norm(corrected - truth_t, dim=1)
                train_hit = float((err_after <= R_HIT).float().mean())
            print(f"  [IC full-train] ep{epoch:3d} loss={total/max(n,1):.4f} train_hit={train_hit:.4f}")
    print(f"[IC] elapsed {time.time()-t0:.1f}s")

    # Inference
    corrector.eval()
    cf_test_t = torch.from_numpy(cf_test).to(DEVICE)
    enc_test_t = torch.from_numpy(enc_test).to(DEVICE)
    cand_test_t = torch.from_numpy(cand_test).to(DEVICE)
    with torch.no_grad():
        delta_test, _ = corrector(cf_test_t, enc_test_t)
        corrected_test = cand_test_t + v2.cap_6mm(delta_test)
    corrected_np = corrected_test.cpu().numpy().astype(np.float64)

    df_sub = pd.DataFrame({
        "id": test_ids,
        "x": corrected_np[:, 0],
        "y": corrected_np[:, 1],
        "z": corrected_np[:, 2],
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = args.out_dir / "submission.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"✓ submission → {sub_path.relative_to(REPO)}")
    print(f"  n_test={len(df_sub)}, x [{corrected_np[:,0].min():.3f}, {corrected_np[:,0].max():.3f}]")


if __name__ == "__main__":
    main()
