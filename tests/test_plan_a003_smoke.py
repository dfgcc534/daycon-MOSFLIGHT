"""plan-a-003 c2 — smoke tests (§5).

반사 augment 의 핵심 불변식: (1) 반사 대상 index 가 `_y`/cvca_y 만(magnitude 미포함),
(2) 반사 2회 = 항등(involution), (3) y 반사가 벡터 L2 norm 보존(magnitude feature 불변).
aug-off bit-identical repro + aug-on finite 는 run_oof --gate smoke (Bash, CI 비포함) 가 담당.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_PA1 = Path(__file__).resolve().parent.parent / "analysis" / "plan-a-001"
_PA2 = Path(__file__).resolve().parent.parent / "analysis" / "plan-a-002"


def _load(base, name):
    spec = importlib.util.spec_from_file_location(f"m_{name}", base / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kf = _load(_PA2, "kalman_features")
_fe = _load(_PA2, "features_ext")
_yaw = _load(_PA1, "yaw")
_rng = np.random.default_rng(20260526)


def _kr003_names():
    X = np.cumsum(_rng.standard_normal((64, 11, 3)) * 0.01, axis=1)
    _, innov, fv = _kf.kalman_with_internals(X)
    cc = _kf.cv_ca_disagreement(X)
    theta = _yaw.yaw_from_last_step(X)
    _, seq_names = _fe.build_seq_ext(X, innov_arr=innov, filtered_v_arr=fv, theta=theta, input_yaw=True)
    noise = _fe._feat.compute_noise(X, with_loo=False)
    _, scal_names, _ = _fe.build_scalar_ext(
        X, noise["poly2"], noise["savgol"], noise["loo"], cv_ca_arr=cc, theta=theta, input_yaw=True)
    return seq_names, scal_names


def test_reflect_index_identification():
    """KR003 15ch/44 에서 반사 대상 = `_y` seq + cvca_y scalar (run_oof main 과 동일 로직)."""
    seq_names, scal_names = _kr003_names()
    reflect_idx_seq = [i for i, n in enumerate(seq_names) if n.endswith("_y")]
    reflect_idx_scal = [i for i, n in enumerate(scal_names) if n == "cvca_y"]
    assert reflect_idx_seq == [1, 4, 7, 10, 13], reflect_idx_seq
    assert reflect_idx_scal == [41], reflect_idx_scal
    # magnitude/cos/norm 채널은 반사 대상 아님 (cvca_norm·tier3·speed 등)
    assert "cvca_norm" in scal_names and scal_names.index("cvca_norm") not in reflect_idx_scal
    assert len(seq_names) == 15 and len(scal_names) == 44


def test_reflection_is_involution():
    """반사 2회 = 원본 (부호 반전 → 항등)."""
    x = torch.randn(8, 11, 15)
    idx = [1, 4, 7, 10, 13]
    sign = torch.tensor([1.0, -1, 1, -1, 1, -1, 1, -1])  # 임의 부분집합 반사
    x2 = x.clone()
    x2[:, :, idx] = x2[:, :, idx] * sign[:, None, None]
    x2[:, :, idx] = x2[:, :, idx] * sign[:, None, None]
    assert torch.allclose(x, x2)


def test_reflection_preserves_magnitude():
    """y→−y 반사는 벡터 L2 norm·cos 보존 → magnitude/cos feature 불변 (반사 대상 제외 정당)."""
    v = torch.randn(16, 3)
    vr = v.clone(); vr[:, 1] *= -1
    assert torch.allclose(v.norm(dim=-1), vr.norm(dim=-1))
    # 두 벡터 동시 반사 시 사이각(cos) 보존
    a = torch.randn(16, 3); b = torch.randn(16, 3)
    ar = a.clone(); ar[:, 1] *= -1
    br = b.clone(); br[:, 1] *= -1
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    cosr = (ar * br).sum(-1) / (ar.norm(dim=-1) * br.norm(dim=-1))
    assert torch.allclose(cos, cosr)


def test_run_oof_aug_flags_importable():
    """run_oof 가 aug flag/인자를 노출하는지 (train_one signature)."""
    ro = _load(_PA2, "run_oof")
    import inspect
    sig = inspect.signature(ro.train_one)
    for p in ["reflect_aug", "noise_aug", "reflect_idx_seq", "reflect_idx_scal"]:
        assert p in sig.parameters, p
