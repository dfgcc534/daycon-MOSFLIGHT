"""plan-c-001 §4.2 — F0 고정공식 per-step 잔차 자기진단 feature (Kalman innovation 의 F0 버전).

per-frame 잔차 단일 정의: r_w(t) = f0_perp0(X[:, t-4:t-1], end_idx=2) − X[:, t]  (pred − actual),
valid frame t ∈ {4..10} (7개). r_f(t) = rotate_xy(r_w(t), theta) (KR002 yaw frame).

산출 (전 출력 float32):
  seq_resid (N,11,3) — slot=프레임 t: seq_resid[:,t]=r_f(t) (t=4..10), t<4 zero-pad.
                       build_seq_t3 (N,11,9) 의 frame-t step 축과 정렬 → concat (9→12채널).
  f0_conf   (N,2)    — [‖r_w(10)‖₂, std_t(speed(t)) ddof=0]. speed=raw 위치차 t=1..10.
  ewma      (N,3)    — r_f(t) (t=4..10) 의 EWMA(α=0.3), init=r_f(4).
  sta_lta   (N,3)    — EWMA_{0.5}(r_f)/(|EWMA_{0.1}(r_f)|+ε).

theta = sample-level 단일 yaw_angle(X[:,10]-X[:,9]) (전 step 동일 적용). leakage 무
(r_w(t) 는 frame {t-4..t-2,t} 만 사용, +2 예측은 외삽).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_f0 = _load("pc_f0_baseline", _THIS / "f0_baseline.py")
_yaw = _load("pc_yaw", _THIS.parent / "plan-a-001" / "yaw.py")

VALID_FRAMES = list(range(4, 11))  # t ∈ {4..10}, 7개
EPS = 1e-6


def _ewma(seq: np.ndarray, alpha: float) -> np.ndarray:
    """seq (N, K, 3) → EWMA over K, init=seq[:,0]. returns (N,3)."""
    s = seq[:, 0].copy()
    for i in range(1, seq.shape[1]):
        s = alpha * seq[:, i] + (1.0 - alpha) * s
    return s


def f0_resid_feats(X: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
    """X (N,11,3), theta (N,) sample-level yaw. returns dict, 전부 float32."""
    N = X.shape[0]
    seq_resid = np.zeros((N, 11, 3), dtype=np.float32)
    r_f_seq = np.zeros((N, len(VALID_FRAMES), 3), dtype=np.float64)
    for i, t in enumerate(VALID_FRAMES):
        pred = _f0.f0_perp0(X[:, t - 4:t - 1], end_idx=2)  # (N,3) +2 예측 → frame t
        r_w = pred - X[:, t]                                 # (N,3) pred − actual
        r_f = _yaw.rotate_xy(r_w, theta)                     # (N,3) yaw frame
        seq_resid[:, t] = r_f.astype(np.float32)
        r_f_seq[:, i] = r_f

    # f0_conf: 마지막(t=10) 잔차 norm + step-speed std
    r_w_10 = _f0.f0_perp0(X[:, 6:9], end_idx=2) - X[:, 10]   # = r_w(10)
    conf_norm = np.linalg.norm(r_w_10, axis=1)               # (N,) frame-invariant
    speed = np.linalg.norm(X[:, 1:11] - X[:, 0:10], axis=2)  # (N,10) t=1..10 raw
    conf_spread = speed.std(axis=1, ddof=0)                  # (N,)
    f0_conf = np.stack([conf_norm, conf_spread], axis=1).astype(np.float32)

    ewma = _ewma(r_f_seq, 0.3).astype(np.float32)            # (N,3)
    e05 = _ewma(r_f_seq, 0.5)
    e01 = _ewma(r_f_seq, 0.1)
    sta_lta = (e05 / (np.abs(e01) + EPS)).astype(np.float32)  # (N,3)

    return dict(seq_resid=seq_resid, f0_conf=f0_conf, ewma=ewma, sta_lta=sta_lta)


def assert_no_leakage(X: np.ndarray, theta: np.ndarray) -> None:
    """smoke: seq_resid[:, :4]==0, t≥4 nonzero (일반), 모든 산출 관측창 내."""
    feats = f0_resid_feats(X, theta)
    sr = feats["seq_resid"]
    assert np.all(sr[:, :4] == 0.0), "seq_resid slot t<4 must be zero (윈도 부족)"
    assert np.isfinite(sr).all() and np.isfinite(feats["f0_conf"]).all(), "non-finite"
    assert np.isfinite(feats["ewma"]).all() and np.isfinite(feats["sta_lta"]).all()
    # t≥4 는 일반적으로 nonzero (잔차=0 인 degenerate 샘플 제외 전체합 nonzero)
    assert np.abs(sr[:, 4:]).sum() > 0, "valid frames all-zero (의심)"
