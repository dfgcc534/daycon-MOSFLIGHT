"""plan-011 c13 — best Phase submission 박제 (In axis ID: cf 32 + CNN 64-dim encoder).

autonomous decision (G1 (b) FAIL): Phase 3+ 모두 skip. best Phase = Phase 1 In axis ID.

전략:
  - Train ID corrector on **all 10000 train** (no fold split — submission generation purpose).
  - Predict on 5000 test samples → corrected_pos = cand + cap6mm(delta).
  - cand = CANDIDATES[17] (frenet_par120_perp_neg020).
  - Save submission_id_best.csv → runs/baseline/H012_phase1-input-ablation/sub_ID/.

산출:
  - runs/baseline/H012_phase1-input-ablation/sub_ID/submission.csv (id, x, y, z)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

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
SEED = 20260513 + 100 + ord("D")  # In̂=ID seed


def load_train_full(data_root: Path) -> dict:
    print("[data] loading full train (10000)...")
    df = pd.read_csv(data_root / "train_labels.csv")
    sample_ids = df["id"].astype(str).tolist()
    truth = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    x_seq = sel.load_stack(data_root / "train", sample_ids)
    cands_all = sel.make_candidates(x_seq, end_idx=END_IDX, horizon=HORIZON)
    cand = cands_all[:, SINGLE_IDX:SINGLE_IDX + 1, :][:, 0, :]
    cf = sel.make_candidate_features(x_seq, END_IDX, cands_all[:, SINGLE_IDX:SINGLE_IDX + 1, :],
                                      HORIZON, candidates_list=[sel.CANDIDATES[SINGLE_IDX]])[:, 0, :]
    target = (truth - cand).astype(np.float32)
    err = np.linalg.norm(truth - cand, axis=1).astype(np.float32)
    return {
        "sample_ids": sample_ids, "x_seq": x_seq, "cf": cf, "cand": cand,
        "target": target, "truth": truth, "err": err,
    }


def load_test(data_root: Path, sample_submission_path: Path) -> dict:
    df = pd.read_csv(sample_submission_path)
    sample_ids = df["id"].astype(str).tolist()
    print(f"[data] loading test ({len(sample_ids)})...")
    x_seq = sel.load_stack(data_root / "test", sample_ids)
    cands_all = sel.make_candidates(x_seq, end_idx=END_IDX, horizon=HORIZON)
    cand = cands_all[:, SINGLE_IDX:SINGLE_IDX + 1, :][:, 0, :]
    cf = sel.make_candidate_features(x_seq, END_IDX, cands_all[:, SINGLE_IDX:SINGLE_IDX + 1, :],
                                      HORIZON, candidates_list=[sel.CANDIDATES[SINGLE_IDX]])[:, 0, :]
    return {"sample_ids": sample_ids, "x_seq": x_seq, "cf": cf, "cand": cand}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "runs/baseline/H012_phase1-input-ablation/sub_ID")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_data = load_train_full(args.data_root)
    test_data = load_test(args.data_root, args.data_root / "sample_submission.csv")

    # Build ID model + CNN encoder
    cnn = v2.TrajectoryCNNEncoder(in_channels=3, hidden=64).to(DEVICE)
    corrector = v2.RedesignedCorrectionNet(dim_cf=32, hidden=64, dim_encoder=64).to(DEVICE)
    params = list(cnn.parameters()) + list(corrector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # tensors
    cf_t = torch.from_numpy(train_data["cf"]).to(DEVICE)
    target_t = torch.from_numpy(train_data["target"]).to(DEVICE)
    err_t = torch.from_numpy(train_data["err"]).to(DEVICE)
    x_seq_t = torch.from_numpy(train_data["x_seq"]).to(DEVICE)
    cand_t = torch.from_numpy(train_data["cand"]).to(DEVICE)
    truth_t = torch.from_numpy(train_data["truth"]).to(DEVICE)

    N = cf_t.shape[0]
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        cnn.train(); corrector.train()
        perm = torch.randperm(N, device=DEVICE)
        total, n = 0.0, 0
        for start in range(0, N, args.batch):
            sel_ = perm[start:start + args.batch]
            cf_b = cf_t[sel_]; tg_b = target_t[sel_]; er_b = err_t[sel_]
            x_b = x_seq_t[sel_]
            enc_b = cnn(x_b)
            opt.zero_grad(set_to_none=True)
            delta, _ = corrector(cf_b, enc_b)
            per = v2.huber_loss(delta, tg_b)
            w = base.weight_schedule(er_b)
            loss = (per * w).sum() / (w.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            total += float(loss.detach()) * len(cf_b)
            n += len(cf_b)
        if epoch % 10 == 0:
            # eval on train (sanity — overfit OK for submission generation)
            cnn.eval(); corrector.eval()
            with torch.no_grad():
                enc = cnn(x_seq_t)
                delta, _ = corrector(cf_t, enc)
                corrected_pos = cand_t + v2.cap_6mm(delta)
                err_after = torch.norm(corrected_pos - truth_t, dim=1)
                train_hit = float((err_after <= R_HIT).float().mean())
            print(f"  [ID full-train] ep{epoch:3d} loss={total/max(n,1):.4f} train_hit={train_hit:.4f}")

    print(f"[ID] full-train elapsed {time.time()-t0:.1f}s")

    # inference on test
    cnn.eval(); corrector.eval()
    cf_test_t = torch.from_numpy(test_data["cf"]).to(DEVICE)
    x_seq_test_t = torch.from_numpy(test_data["x_seq"]).to(DEVICE)
    cand_test_t = torch.from_numpy(test_data["cand"]).to(DEVICE)
    with torch.no_grad():
        enc_test = cnn(x_seq_test_t)
        delta_test, _ = corrector(cf_test_t, enc_test)
        corrected_pos_test = cand_test_t + v2.cap_6mm(delta_test)

    corrected_np = corrected_pos_test.cpu().numpy().astype(np.float64)
    df_sub = pd.DataFrame({
        "id": test_data["sample_ids"],
        "x": corrected_np[:, 0],
        "y": corrected_np[:, 1],
        "z": corrected_np[:, 2],
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = args.out_dir / "submission.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"✓ submission → {sub_path.relative_to(REPO)}")
    print(f"  n_test={len(df_sub)}, x range [{corrected_np[:,0].min():.3f}, {corrected_np[:,0].max():.3f}]")
    print(f"  y range [{corrected_np[:,1].min():.3f}, {corrected_np[:,1].max():.3f}]")
    print(f"  z range [{corrected_np[:,2].min():.3f}, {corrected_np[:,2].max():.3f}]")


if __name__ == "__main__":
    main()
