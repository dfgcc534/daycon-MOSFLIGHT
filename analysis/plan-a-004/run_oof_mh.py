"""plan-a-004 §4.2 — Multi-Hypothesis OOF runner (run_oof_mh.py).

KR008 파이프라인(feature/aug/fold/kalman) 재사용 + multi-head(GRUMultiHead) train loop.
산출: per-sample K 후보(world) + selector → oracle@K, realized-hit(selector argmax), per-head 사용률.
n_heads=1 → KR008(run_oof.py) 과 동일 결과(bit-identical 검증). KILL gate(G1_decisive)용 g1 게이트.

Usage:
  python analysis/plan-a-004/run_oof_mh.py --gate g1 --innov --filtered-v --cv-ca --input-yaw \
    --reflect-aug --noise-aug 0.10 --n-heads 2 --gen mcl --route joint --exp KR010 --out g1_kr010.json
"""
from __future__ import annotations

import argparse
import importlib.util as _u
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_THIS = Path(__file__).resolve().parent
_A002 = _THIS.parent / "plan-a-002"
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402


def _L(base, name):
    s = _u.spec_from_file_location(f"m_{name}", base / f"{name}.py")
    m = _u.module_from_spec(s); s.loader.exec_module(m); return m


_kalman = _L(_THIS.parent / "plan-a-001", "kalman")
_yaw = _L(_THIS.parent / "plan-a-001", "yaw")
_kf = _L(_A002, "kalman_features")
_fe = _L(_A002, "features_ext")
_mm = _L(_THIS, "model_mh")
_lc = _L(_THIS, "losses_mcl")

R_HIT, R_HIT_LOOSE = 0.01, 0.015
CONFIGS = {"A": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=5e-4, p=0.3, wd=1e-4),
           "B": dict(hidden_size=64, num_layers=1, fc_hidden=128, lr=1e-3, p=0.1, wd=1e-4)}


def hit(p, y, t=R_HIT):
    return np.linalg.norm(p - y, axis=-1) <= t


def paired_perm(hb, ha, n=10000, seed=0):
    d = hb.astype(np.float64) - ha.astype(np.float64); obs = d.mean()
    rng = np.random.default_rng(seed); s = rng.choice([1.0, -1.0], size=(n, len(d)))
    return float(obs), float((np.abs((s * d).mean(1)) >= abs(obs)).mean())


def _cands_world(out, theta, kal):
    """model out(rotated residual) → world 후보 (N,K,3) + selector prob (N,K)."""
    if out["mode"] == "mdn":
        mu = out["mu"].detach().cpu().numpy(); w = torch.softmax(out["logit_w"], 1).detach().cpu().numpy()
        rot = mu
    else:
        rot = out["preds"].detach().cpu().numpy()
        sel = out["selector"]
        w = (torch.softmax(sel, 1).detach().cpu().numpy() if sel is not None
             else np.ones((rot.shape[0], rot.shape[1])) / rot.shape[1])
    K = rot.shape[1]
    cw = np.stack([kal + _yaw.inverse_rotate_xy(rot[:, k], theta) for k in range(K)], axis=1)
    return cw, w


def train_one_mh(cfg, seq_tr, scal_tr, tgt_m, tgt_F, tgt_W, seq_va, scal_va, theta_va, kal_va, y_va,
                 seed, device, epochs, patience, n_heads, gen, route, soft_top, batch=256,
                 reflect_aug=False, noise_aug=0.0, ridx_seq=None, ridx_scal=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = _mm.GRUMultiHead(n_channels=seq_tr.shape[2], scal_dim=scal_tr.shape[1],
                           hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                           fc_hidden=cfg["fc_hidden"], p=cfg["p"], n_heads=n_heads,
                           gen=gen, selector=(route == "joint")).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sq = torch.as_tensor(seq_tr, device=device); sc = torch.as_tensor(scal_tr, device=device)
    tm = torch.as_tensor(tgt_m, dtype=torch.float32, device=device)
    tF = torch.as_tensor(tgt_F, dtype=torch.float32, device=device)
    tW = torch.as_tensor(tgt_W, dtype=torch.float32, device=device)
    sq_va = torch.as_tensor(seq_va, device=device); sc_va = torch.as_tensor(scal_va, device=device)
    n = sq.shape[0]
    best_real, best_cw, best_w = -1.0, None, None
    no_imp = 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            bsq, bsc, btm, btF, btW = sq[idx], sc[idx], tm[idx], tF[idx], tW[idx]
            if noise_aug > 0:
                bsq = bsq + noise_aug * torch.randn_like(bsq)
            if reflect_aug:
                sign = torch.ones(bsq.shape[0], device=device)
                sign[torch.rand(bsq.shape[0], device=device) < 0.5] = -1.0
                bsq = bsq.clone()
                if ridx_seq:
                    bsq[:, :, ridx_seq] = bsq[:, :, ridx_seq] * sign[:, None, None]
                bsc = bsc.clone()
                if ridx_scal:
                    bsc[:, ridx_scal] = bsc[:, ridx_scal] * sign[:, None]
                btm = btm.clone(); btm[:, 1] = btm[:, 1] * sign
                btF = btF.clone(); btF[:, 1] = btF[:, 1] * sign
                btW = btW.clone(); btW[:, 1] = btW[:, 1] * sign
            opt.zero_grad()
            out = net(bsq, bsc)
            loss, _ = _lc.loss_mcl(out, btm, [btF, btW], soft_top=soft_top)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            out_va = net(sq_va, sc_va)
        cw, w = _cands_world(out_va, theta_va, kal_va)            # (Nv,K,3),(Nv,K)
        realized = cw[np.arange(len(w)), w.argmax(1)]
        rh = hit(realized, y_va).mean()
        if rh > best_real:
            best_real, best_cw, best_w, no_imp = rh, cw.copy(), w.copy(), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    return best_cw, best_w, best_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["smoke", "g1", "full"], default="g1")
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--gen", default="mcl"); ap.add_argument("--route", default="joint")
    ap.add_argument("--soft-top", type=int, default=1)
    ap.add_argument("--input-yaw", action="store_true"); ap.add_argument("--innov", action="store_true")
    ap.add_argument("--filtered-v", action="store_true"); ap.add_argument("--cv-ca", action="store_true")
    ap.add_argument("--reflect-aug", action="store_true"); ap.add_argument("--noise-aug", type=float, default=0.0)
    ap.add_argument("--exp", default="KR010"); ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--compare-to", default=None); ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.gate == "smoke":
        folds, seeds, epochs, configs, max_n = [0], [0], 1, ["A"], 512
    elif args.gate == "g1":
        folds, seeds, epochs, configs, max_n = [0], [0], args.epochs, ["A"], None
    else:
        folds, seeds, epochs, configs, max_n = list(range(5)), [0, 1, 2], args.epochs, ["A", "B"], None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"[plan-a-004 MH] exp={args.exp} gate={args.gate} dev={device} n_heads={args.n_heads} "
          f"gen={args.gen} route={args.route} soft_top={args.soft_top} ep={epochs}", flush=True)

    ids, X = load_all_samples("train"); lab, y = load_labels(); assert ids == lab
    if max_n:
        X, y, ids = X[:max_n], y[:max_n], ids[:max_n]
    N = X.shape[0]; fid = np.array([stable_fold_id(i, 5) for i in ids]); pm = np.isin(fid, folds)

    need = args.innov or args.filtered_v
    innov = fv = None
    if need:
        _, innov, fv = _kf.kalman_with_internals(X)
    theta = _yaw.yaw_from_last_step(X)
    kal = _kalman.kalman_predict(X, sigma_obs=_kalman.SIGMA_OBS_MAIN, sigma_proc=_kalman.SIGMA_PROC_MAIN)
    kal_alt = _kalman.kalman_predict(X, sigma_obs=_kalman.SIGMA_OBS_ALT, sigma_proc=_kalman.SIGMA_PROC_ALT)
    tgt_m = _yaw.rotate_xy(y - kal, theta).astype(np.float32)
    tgt_F = _yaw.rotate_xy(y - X[:, -1], theta).astype(np.float32)
    tgt_W = _yaw.rotate_xy(y - kal_alt, theta).astype(np.float32)
    cache = _A002 / "noise_cache.npz"
    noise = _fe.compute_noise(X, cache_path=(None if max_n else cache), key="train", with_loo=True)
    seq, seq_names = _fe.build_seq_ext(X, innov_arr=(innov if args.innov else None),
                                       filtered_v_arr=(fv if args.filtered_v else None),
                                       theta=theta, input_yaw=args.input_yaw)
    cv_ca = _kf.cv_ca_disagreement(X) if args.cv_ca else None
    scal, scal_names, _ = _fe.build_scalar_ext(X, noise["poly2"], noise["savgol"], noise["loo"],
                                               cv_ca_arr=cv_ca, theta=theta, input_yaw=args.input_yaw)
    ridx_seq = [i for i, nm in enumerate(seq_names) if nm.endswith("_y")]
    ridx_scal = [i for i, nm in enumerate(scal_names) if nm == "cvca_y"]

    # config 평균 (g1=A only). per-config OOF 후보/selector → ensemble (config 평균은 realized 만)
    N_cw = np.zeros((N, args.n_heads, 3)); N_w = np.zeros((N, args.n_heads))
    C = seq.shape[2]
    for cf in configs:
        for f in folds:
            tr = np.where(fid != f)[0]; va = np.where(fid == f)[0]
            ssq = StandardScaler().fit(seq[tr].reshape(-1, C)); ssc = StandardScaler().fit(scal[tr])
            cw, w, _ = train_one_mh(
                CONFIGS[cf], _fe.normalize_seq(seq[tr], ssq), ssc.transform(scal[tr]).astype(np.float32),
                tgt_m[tr], tgt_F[tr], tgt_W[tr],
                _fe.normalize_seq(seq[va], ssq), ssc.transform(scal[va]).astype(np.float32),
                theta[va], kal[va], y[va], seeds[0], device, epochs, args.patience,
                args.n_heads, args.gen, args.route, args.soft_top,
                reflect_aug=args.reflect_aug, noise_aug=args.noise_aug, ridx_seq=ridx_seq, ridx_scal=ridx_scal)
            N_cw[va] += cw / len(configs); N_w[va] += w / len(configs)

    # metric (predicted_mask subset)
    oracle = float((np.linalg.norm(N_cw[pm] - y[pm][:, None, :], axis=-1).min(1) <= R_HIT).mean())
    realized_pred = N_cw[np.arange(N), N_w.argmax(1)]
    realized = float(hit(realized_pred[pm], y[pm]).mean())
    per_head = np.bincount(N_w[pm].argmax(1), minlength=args.n_heads).tolist()
    kalman_alone = float(hit(kal[pm], y[pm]).mean())
    print(f"[done] exp={args.exp} oracle@{args.n_heads}={oracle:.4f} realized-hit={realized:.4f} "
          f"per_head={per_head} (kalman_alone={kalman_alone:.4f}) rt={time.time()-t0:.0f}s", flush=True)

    res = dict(exp=args.exp, gate=args.gate, n_heads=args.n_heads, gen=args.gen, route=args.route,
               soft_top=args.soft_top, N=int(N), n_pred=int(pm.sum()), folds=[int(f) for f in folds],
               seeds=[int(s) for s in seeds], epochs=epochs, configs=configs,
               oracle_at_k=oracle, realized_hit=realized, per_head_usage=per_head,
               kalman_alone=kalman_alone, runtime_sec=round(time.time() - t0, 1))
    if args.compare_to:
        cp = Path(args.compare_to)
        cp = cp if cp.is_absolute() else _A002 / cp
        zb = np.load(cp); hb = zb["per_sample_hit"]
        delta, p = paired_perm(hit(realized_pred[pm], y[pm]), hb[pm])
        res["compare"] = dict(base=str(cp.name), base_hit=float(hb[pm].mean()), delta=delta, p=p)
        print(f"  [compare vs {cp.name}] Δrealized={delta:+.4f} p={p:.4g}", flush=True)
    out_name = args.out or f"{args.exp.lower()}_{args.gate}.json"
    (_THIS / out_name).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    if args.gate == "full":
        np.savez(_THIS / out_name.replace(".json", ".npz"), cw=N_cw, w=N_w, y=y, fold_ids=fid,
                 realized_pred=realized_pred, per_sample_hit=hit(realized_pred, y))
    print(f"[saved] {out_name}", flush=True)
    return res


if __name__ == "__main__":
    main()
