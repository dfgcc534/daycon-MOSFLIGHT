"""plan-b-001 c8 — smoke tests (builder shape + model finite/gradient + yaw identity + kalman)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_PB = Path(__file__).resolve().parent.parent / "analysis" / "plan-b-001"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"pb_{name}", _PB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_yaw = _load("yaw_frame")
_kalman = _load("kalman_cv")
_noise = _load("noise_estimator")
_tier3 = _load("tier3")
_residual = _load("residual_builder")
_query = _load("query_builder")
_head = _load("head_summary")
_model = _load("model")

rng = np.random.default_rng(20260526)
_X = np.cumsum(rng.standard_normal((20, 11, 3)) * 0.01, axis=1).astype(np.float32)


def test_yaw_identity():
    vec = rng.standard_normal((50, 3))
    th = rng.uniform(-np.pi, np.pi, 50)
    assert np.abs(_yaw.from_yaw(_yaw.to_yaw(vec, th), th) - vec).max() < 1e-10
    # R_wfy decode parity
    R = _yaw.build_R_wfy(th)
    assert np.abs(np.einsum("nij,nj->ni", R, _yaw.to_yaw(vec, th)) - vec).max() < 1e-10
    assert np.all(_yaw.yaw_from_X(np.zeros((4, 11, 3))) == 0.0)  # degenerate


def test_kalman_shapes():
    p = _kalman.kalman_predict(_X)
    assert p.shape == (20, 3) and np.isfinite(p).all()
    assert np.allclose(p, _kalman.kalman_baseline_at(_X, 10), atol=1e-5)


def test_noise_tier3():
    assert _noise.build_noise(_X).shape == (20, 2)
    assert _tier3.build_tier3(_X).shape == (20, 5)
    assert np.isfinite(_noise.build_noise(_X)).all()
    assert np.isfinite(_tier3.build_tier3(_X)).all()


def test_residual_yaw():
    th = _yaw.yaw_from_X(_X)
    R = _yaw.build_R_wfy(th)
    anchors = rng.standard_normal((14, 3)).astype(np.float32) * 0.01
    out = _residual.build_residuals(_X, th, R, anchors, lambda X, t: X[:, t, :])
    assert out["residual_a"].shape == (20, 7, 3)
    assert out["residual_b"].shape == (20, 14, 7, 3)
    assert np.all(out["residual_a"][:, 5:7] == 0.0)  # zero-pad
    assert np.isfinite(out["residual_b"]).all()


def test_query_head_shapes():
    cand = rng.standard_normal((20, 14, 150)).astype(np.float32)
    cand_ext = rng.standard_normal((20, 14, 165)).astype(np.float32)
    sl = _query.extract_slim7_from_cand_ext_165(cand_ext)
    assert _query.build_query(cand, sl).shape == (20, 14, 29)
    hs = _head.build_head_summary(cand, rng.standard_normal((20, 9)).astype(np.float32),
                                  rng.uniform(0, 1, (20, 14)).astype(np.float32),
                                  rng.uniform(0, 1, (20, 2)).astype(np.float32),
                                  rng.uniform(0, 1, (20, 5)).astype(np.float32))
    assert hs.shape == (20, 56)


def test_model_forward_grad():
    B, T, K = 4, 7, 14
    m = _model.GRUNetX3(anchors=torch.randn(K, 3) * 0.01)
    m.train()
    wp, probs = m(torch.randn(B, T, 98), torch.randn(B, K, T, 3), torch.randn(B, K, 29),
                  torch.randn(B, 56), torch.randn(B, K, 7), torch.randn(B, 3),
                  torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous())
    assert wp.shape == (B, 3) and probs.shape == (B, K)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    assert m.head_in_dim == 199
    # softhit gradient path
    y = torch.randn(B, 3)
    d = torch.sqrt(((wp - y) ** 2).sum(-1) + 1e-12)
    torch.sigmoid((d - 0.01) / 0.002).mean().backward()
    assert m.head_mlp[0].weight.grad is not None
    assert m.resb_proj.weight.grad is not None  # F1 bias path receives gradient
