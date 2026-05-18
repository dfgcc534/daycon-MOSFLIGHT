"""plan-022 §4.3 + §4.4 + §3.5 — LgbmSelectorOnly + build_soft_label_with_tau.

plan-021 LgbmDualHead.clf_head 만 추출 (reg head 제거). soft label CE 근사 =
sample-weight expansion 위 LightGBM multiclass softmax classifier.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# ── plan-021 / plan-020 module reuse (§4.3) ───────────────────────────

_THIS = Path(__file__).resolve().parent             # analysis/plan-022/
_REPO = _THIS.parent.parent                          # REPO root
_PLAN020 = _THIS.parent / "plan-020"
_PLAN021 = _THIS.parent / "plan-021"

# transitive import 보조: plan-021 build_input.py 내부의 `from src.io ...` 등 대비
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_PLAN021) not in sys.path:
    sys.path.insert(0, str(_PLAN021))

# plan-020 baseline_f0 (F0 산식 carry)
_spec = importlib.util.spec_from_file_location("bf_022", _PLAN020 / "baseline_f0.py")
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)
# bf.f0_baseline, bf.D1, bf.PAR, bf.PERP, bf.R_HIT, bf.R_HIT_LOOSE

# plan-021 build_input + dual_head_model
_spec = importlib.util.spec_from_file_location("p021_build_022", _PLAN021 / "build_input.py")
p021_build = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p021_build)
# p021_build.build_input_common, build_input_lgbm_extra, to_frenet, build_frenet_basis_3d

_spec = importlib.util.spec_from_file_location("p021_dh_022", _PLAN021 / "dual_head_model.py")
p021_dh = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(p021_dh)
# p021_dh.LgbmDualHead — clf_head 부분만 carry


# ── §3.5 build_soft_label_with_tau ────────────────────────────────────


def build_soft_label_with_tau(
    gt: np.ndarray,                   # (N, 3) world frame, float64, meters
    R_wfn: np.ndarray,                # (N, 3, 3) Frenet basis, columns=[t̂, n̂, b̂]
    pred_F0_world: np.ndarray,        # (N, 3) F0 80ms 미래 예측, world frame, meters
    anchors: np.ndarray,              # (K, 3) Frenet 좌표, meters
    tau_cls: float,                   # softmax temperature, meters
) -> np.ndarray:
    """plan-021 build_soft_label 의 τ_cls 가변 버전.

    Returns: q (N, K) float32, row-sum=1.
    """
    gt = np.asarray(gt, dtype=np.float64)
    pred_F0_world = np.asarray(pred_F0_world, dtype=np.float64)
    R_wfn = np.asarray(R_wfn, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)

    residual = np.einsum("nij,nj->ni", R_wfn.transpose(0, 2, 1), gt - pred_F0_world)
    dist = np.linalg.norm(anchors[None, :, :] - residual[:, None, :], axis=2)  # (N, K)
    z = -dist / tau_cls
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q.astype(np.float32)


# ── §4.4 LgbmSelectorOnly class ───────────────────────────────────────


class LgbmSelectorOnly:
    """plan-021 LgbmDualHead.clf_head 만 추출. reg head 제거.

    Args:
        K (int): anchor count (= classifier output dim)
    """

    def __init__(self, K: int):
        from lightgbm import LGBMClassifier

        self.K = K
        # plan-021 LgbmDualHead 의 clf head config 정확 carry
        self.clf = LGBMClassifier(
            objective="multiclass",
            num_class=K,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            verbose=-1,
            random_state=20260519,
        )

    def fit(self, X: np.ndarray, q: np.ndarray) -> "LgbmSelectorOnly":
        """X (N, 170), q (N, K) soft label.

        sample-weight expansion 으로 soft label CE 근사.
        row order = C-order of q (j-major then i):
        at row j*K + i: X = X[j], y = i, weight = q[j, i] = q.flatten()[j*K + i]
        """
        N = X.shape[0]
        assert q.shape == (N, self.K), f"q shape {q.shape} != ({N}, {self.K})"

        X_expanded = np.repeat(X, self.K, axis=0)            # (N*K, 170)
        y_expanded = np.tile(np.arange(self.K), N)           # (N*K,)
        sample_weight = q.flatten()                           # (N*K,) C-order
        mask = sample_weight > 1e-6

        # multi-class safety — 누락 class 별로 dummy sample 1 개 inject
        present_classes = set(y_expanded[mask].tolist())
        missing = [k for k in range(self.K) if k not in present_classes]
        if missing:
            X_dummy = np.zeros((len(missing), X.shape[1]), dtype=X.dtype)
            y_dummy = np.array(missing, dtype=np.int64)
            w_dummy = np.full(len(missing), 1e-6, dtype=sample_weight.dtype)
            X_fit = np.concatenate([X_expanded[mask], X_dummy], axis=0)
            y_fit = np.concatenate([y_expanded[mask], y_dummy], axis=0)
            w_fit = np.concatenate([sample_weight[mask], w_dummy], axis=0)
        else:
            X_fit, y_fit, w_fit = X_expanded[mask], y_expanded[mask], sample_weight[mask]

        self.clf.fit(X_fit, y_fit, sample_weight=w_fit)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """returns probs (N, K) float32."""
        probs = self.clf.predict_proba(X)
        return probs.astype(np.float32)


# ── smoke (__main__) ──────────────────────────────────────────────────

def _smoke() -> None:
    """§4.5 T4/T5 smoke — build_soft_label_with_tau + LgbmSelectorOnly basic."""
    rng = np.random.default_rng(20260519)
    N, K = 100, 7

    # build_soft_label_with_tau
    gt = rng.standard_normal((N, 3)) * 0.01
    pred_F0 = rng.standard_normal((N, 3)) * 0.01
    R_wfn = np.tile(np.eye(3, dtype=np.float64)[None], (N, 1, 1))
    anchors = np.eye(3, dtype=np.float32)[:K] * 0.005 if K <= 3 else \
              np.vstack([[(0, 0, 0)], np.eye(3) * 0.005, -np.eye(3) * 0.005]).astype(np.float32)
    for tau in [0.001, 0.003, 0.005]:
        q = build_soft_label_with_tau(gt, R_wfn, pred_F0, anchors, tau)
        assert q.shape == (N, K), f"q shape {q.shape} != ({N}, {K})"
        assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5), f"q sum != 1 at τ={tau}"
        assert np.isfinite(q).all()

    # LgbmSelectorOnly fit/predict
    X = rng.standard_normal((N, 170)).astype(np.float32)
    q = rng.dirichlet(alpha=np.ones(K), size=N).astype(np.float32)
    model = LgbmSelectorOnly(K=K).fit(X, q)
    probs = model.predict(X[:50])
    assert probs.shape == (50, K), f"probs shape {probs.shape} != (50, {K})"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(probs).all()

    print(f"[smoke] build_soft_label_with_tau ✓ (3 τ)")
    print(f"[smoke] LgbmSelectorOnly(K={K}) fit ({N} sample) + predict ✓")


if __name__ == "__main__":
    _smoke()
