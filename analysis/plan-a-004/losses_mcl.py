"""plan-a-004 §4.1 — Multiple-Choice-Learning (WTA) + selector CE + MDN NLL loss.

plan-a-001 losses.loss_combo (euclid + 0.3 softhit) 재사용. WTA wrapper + selector CE 신규.
- WTA: k* = argmin_k euclid(pred_k, tgt) (index detach, tie→최소 index). 승자 head 만 main grad.
       soft_top=m>1 시 상위 m head 1/m 가중(dead-head 방지) — main loss 만, selector target 무관.
- selector CE: target = 단일 argmin k* (grad-stop). selector 만 학습.
- MDN: mixture NLL.
"""
from __future__ import annotations

import importlib.util as _u
from pathlib import Path

import torch
import torch.nn.functional as F

_A001 = Path(__file__).resolve().parent.parent / "plan-a-001"


def _load(name):
    s = _u.spec_from_file_location(f"pa1_{name}", _A001 / f"{name}.py")
    m = _u.module_from_spec(s); s.loader.exec_module(m); return m


_losses = _load("losses")
loss_combo = _losses.loss_combo
LAMBDA_AUX = _losses.LAMBDA_AUX


def _euclid_per_sample(pred, tgt):  # (B,K,3),(B,3) → (B,K)
    return torch.sqrt(((pred - tgt[:, None, :]) ** 2).sum(-1) + 1e-12)


def loss_mcl(out, tgt, aux_tgts=None, *, lam_sel=0.1, soft_top=1):
    """multi-head WTA+CE (mode='mcl') 또는 MDN NLL (mode='mdn'). (total, dict 로그)."""
    aux_loss = 0.0
    if aux_tgts is not None and out.get("aux"):
        aux_loss = sum(_losses.loss_aux_euclid(a, t) for a, t in zip(out["aux"], aux_tgts))

    if out["mode"] == "mdn":
        mu, logvar, logit_w = out["mu"], out["logvar"], out["logit_w"]  # (B,K,3),(B,K,3),(B,K)
        logvar = logvar.clamp(-10, 2)
        logw = F.log_softmax(logit_w, dim=1)                            # (B,K)
        # per-component log N(tgt|μ,diag) = −0.5 Σ[(y−μ)²/σ² + logvar + log2π]
        comp = -0.5 * (((tgt[:, None, :] - mu) ** 2) / logvar.exp() + logvar + 1.8378770664).sum(-1)  # (B,K)
        nll = -torch.logsumexp(logw + comp, dim=1).mean()
        total = nll + LAMBDA_AUX * aux_loss
        return total, {"nll": float(nll.detach())}

    preds, sel = out["preds"], out["selector"]                          # (B,K,3),(B,K)|None
    d = _euclid_per_sample(preds, tgt)                                  # (B,K)
    kstar = d.argmin(dim=1)                                             # (B,) tie→최소 index (argmin 기본)
    if soft_top <= 1:
        main = loss_combo(preds[torch.arange(len(tgt)), kstar], tgt)    # 승자 head 만
    else:
        topk = d.topk(soft_top, dim=1, largest=False).indices          # (B,m)
        main = torch.stack([loss_combo(preds[torch.arange(len(tgt)), topk[:, j]], tgt)
                            for j in range(soft_top)]).mean()
    ce = (lam_sel * F.cross_entropy(sel, kstar.detach()) if sel is not None
          else torch.zeros((), device=preds.device))
    total = main + ce + LAMBDA_AUX * aux_loss
    return total, {"main": float(main.detach()), "ce": float(ce.detach()),
                   "head_usage": torch.bincount(kstar, minlength=preds.shape[1]).tolist()}
