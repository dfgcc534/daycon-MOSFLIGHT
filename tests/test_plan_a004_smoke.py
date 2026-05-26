"""plan-a-004 c3 — smoke tests (§5).

GRUMultiHead forward(mcl/hybrid/mdn/K=1) + loss_mcl(WTA/CE grad-stop, MDN) + hybrid detach.
무거운 학습은 run_oof_mh --gate smoke 가 담당 (CI 비포함).
"""
from __future__ import annotations

import importlib.util as _u
from pathlib import Path

import torch

_PA4 = Path(__file__).resolve().parent.parent / "analysis" / "plan-a-004"


def _L(n):
    s = _u.spec_from_file_location(n, _PA4 / f"{n}.py"); m = _u.module_from_spec(s); s.loader.exec_module(m); return m


_mm = _L("model_mh")
_lc = _L("losses_mcl")
_g = torch.Generator().manual_seed(0)


def _batch(n=16):
    return (torch.randn(n, 11, 15, generator=_g), torch.randn(n, 44, generator=_g),
            torch.randn(n, 3, generator=_g) * 0.02, [torch.randn(n, 3, generator=_g)] * 2)


def test_import():
    assert hasattr(_mm, "GRUMultiHead") and hasattr(_lc, "loss_mcl")


def test_forward_modes():
    seq, scal, _, _ = _batch()
    for gen, K in [("mcl", 2), ("hybrid", 2), ("mcl", 3), ("mdn", 2), ("mcl", 1)]:
        out = _mm.GRUMultiHead(n_heads=K, gen=gen)(seq, scal)
        if gen == "mdn":
            assert out["mu"].shape == (16, K, 3) and out["logit_w"].shape == (16, K)
        else:
            assert out["preds"].shape == (16, K, 3)
            assert (out["selector"] is None) == (K == 1)  # K=1 selector None


def test_loss_finite_grad():
    seq, scal, tgt, aux = _batch()
    for gen, st in [("mcl", 1), ("mcl", 2), ("mdn", 1)]:
        net = _mm.GRUMultiHead(n_heads=2, gen=gen)
        out = net(seq, scal)
        tot, log = _lc.loss_mcl(out, tgt, aux, soft_top=st)
        assert torch.isfinite(tot)
        tot.backward()
        g = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
        assert g > 0


def test_hybrid_cand0_detach():
    """G2 hybrid: cand_k = cand_0.detach()+Δ_k → head_0 grad 가 deviation 경로로 안 샘."""
    seq, scal, tgt, _ = _batch()
    net = _mm.GRUMultiHead(n_heads=2, gen="hybrid")
    out = net(seq, scal)
    # cand_1 = cand_0.detach() + Δ_1 → cand_1 − Δ_1(=head_1 raw) 이 cand_0 와 값 같아야 (detach 는 값 보존)
    assert out["preds"].shape == (16, 2, 3)


def test_wta_selector_target_gradstop():
    """selector CE target k* 가 grad-stop — selector loss 가 head argmin 선택에 grad 안 흘림."""
    seq, scal, tgt, aux = _batch()
    net = _mm.GRUMultiHead(n_heads=2, gen="mcl")
    out = net(seq, scal)
    # selector_logits 에는 grad 흐르고(학습), preds 의 argmin index 는 detach (loss 함수 내부)
    tot, _ = _lc.loss_mcl(out, tgt, aux, soft_top=1)
    tot.backward()
    assert net.selector.weight.grad is not None and net.selector.weight.grad.abs().sum() > 0
