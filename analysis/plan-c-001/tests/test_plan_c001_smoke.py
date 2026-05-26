"""plan-c-001 §5 c4 smoke — import + finite + target⇄복원 정합(+음성통제) + leakage + W-aux grad 0."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent.parent           # analysis/plan-c-001
_PA1 = _THIS.parent / "plan-a-001"
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_f0 = _load("t_f0", _THIS / "f0_baseline.py")
_frf = _load("t_frf", _THIS / "f0_residual_feats.py")
_yaw = _load("t_yaw", _PA1 / "yaw.py")
_f0_020 = _load("t_f0_020", _THIS.parent / "plan-020" / "baseline_f0.py")
_model = _load("t_model", _PA1 / "model.py")
_losses = _load("t_losses", _PA1 / "losses.py")


def _toy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(0, 0.01, size=(n, 11, 3)), axis=1).astype(np.float64)
    y = X[:, -1] + rng.normal(0, 0.005, size=(n, 3))
    return X, y


def test_f0_perp0_formula_parity():
    """f0_perp0 = f0_baseline 에서 perp 항만 제거 — perp 보정 더하면 baseline 과 일치."""
    X, _ = _toy()
    e = X.shape[1] - 1
    v_last = X[:, e] - X[:, e - 1]
    acc = v_last - (X[:, e - 1] - X[:, e - 2])
    tan = v_last / (np.linalg.norm(v_last, axis=1, keepdims=True) + 1e-9)
    a_par = np.sum(acc * tan, axis=1, keepdims=True) * tan
    a_perp = acc - a_par
    recon = _f0.f0_perp0(X) + _f0_020.PERP * a_perp        # + (-0.20)·a_perp
    assert np.allclose(recon, _f0_020.f0_baseline(X, e), atol=1e-9)
    assert np.isfinite(_f0.f0_perp0(X)).all()


def test_target_reconstruction_identity():
    """target=rotate(y−base,θ), 복원=base+inverse_rotate(out,θ) 가 동일 θ·base 로 y 재구성."""
    X, y = _toy()
    base = _f0.f0_perp0(X)
    theta = _yaw.yaw_from_last_step(X)
    tgt = _yaw.rotate_xy(y - base, theta)
    recon = base + _yaw.inverse_rotate_xy(tgt, theta)
    assert np.allclose(recon, y, atol=1e-6), "target⇄복원 정합 실패"


def test_target_reconstruction_negative_control():
    """음성통제: θ_복원=θ+0.1 또는 base 부호 뒤집기 시 정합이 *반드시* 깨져야 함 (자명항등 아님)."""
    X, y = _toy()
    base = _f0.f0_perp0(X)
    theta = _yaw.yaw_from_last_step(X)
    tgt = _yaw.rotate_xy(y - base, theta)
    recon_bad_theta = base + _yaw.inverse_rotate_xy(tgt, theta + 0.1)
    assert not np.allclose(recon_bad_theta, y, atol=1e-6), "θ mismatch 인데 정합 — 자명항등"
    recon_bad_base = (-base) + _yaw.inverse_rotate_xy(tgt, theta)
    assert not np.allclose(recon_bad_base, y, atol=1e-6), "base 부호 뒤집었는데 정합"


def test_f0_resid_leakage():
    X, _ = _toy()
    theta = _yaw.yaw_from_last_step(X)
    _frf.assert_no_leakage(X, theta)


def test_w_aux_gradient_zero():
    """λ_W=0 → W-aux head(aux_heads[1]) gradient 0 (또는 None), main/F head 는 nonzero."""
    torch.manual_seed(0)
    net = _model.GRUModelMultiAux(n_channels=12, scal_dim=48, aux_dims=[3, 3], aux_clips=[None, None])
    seq = torch.randn(16, 11, 12)
    scal = torch.randn(16, 48)
    tm = torch.randn(16, 3) * 0.01
    tF = torch.randn(16, 3) * 0.01
    tW = torch.zeros(16, 3)
    om, ax = net(seq, scal)
    loss = (_losses.loss_combo(om, tm)
            + 0.3 * _losses.loss_aux_euclid(ax[0], tF)
            + 0.0 * _losses.loss_aux_euclid(ax[1], tW))
    loss.backward()
    gW = net.aux_heads[1].weight.grad
    assert gW is None or torch.allclose(gW, torch.zeros_like(gW)), "W-aux grad 비-0 (λ_W=0 위반)"
    gF = net.aux_heads[0].weight.grad
    gM = net.head_main.weight.grad
    assert gF is not None and gF.abs().sum() > 0, "F-aux grad 0 (학습 안 됨)"
    assert gM is not None and gM.abs().sum() > 0, "main grad 0"


if __name__ == "__main__":
    test_f0_perp0_formula_parity()
    test_target_reconstruction_identity()
    test_target_reconstruction_negative_control()
    test_f0_resid_leakage()
    test_w_aux_gradient_zero()
    print("[smoke] all 5 asserts PASS")
