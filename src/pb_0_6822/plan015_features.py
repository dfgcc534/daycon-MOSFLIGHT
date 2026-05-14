"""plan-015 feature expansion — A/B/C/D feature 함수 정의 (v2.4 spec).

plan-014 `make_seq_features` 의 확장 wrapper. plan-014 module import (monkey-patch 아님).

signature (§5.1 v2.4 박제):
    make_seq_features_v2(
        X: np.ndarray,
        end_idx: int = 10,
        direction: float = 1.0,
        *,
        feature_flags: dict[str, bool],
    ) -> np.ndarray
        feature_flags = {"A": bool, "B": bool, "C": bool, "D": bool}
        return shape (N, 6, target_dim)

분기 로직 (§5.1 v2.4):
  내부 적용 순서 = A → B → C → D 순차.
  1. base = plan-014 make_seq_features(X, end_idx, direction)  # (N, 6, 9)
  2. A=True: + displacement_F0 3D  # +3D
  3. B=True: perp_norm/speed 1D → (acc_normal, acc_binormal)/speed 2D split  # +1D
  4. C=True: τ=1, τ=2 2 stream concat  # dim ×2
  5. D=True: + pairwise 6D  # +6D

단독 dim (다른 flag False):
  A 단독: 9+3 = 12D
  B 단독: 9-1+2 = 10D
  C 단독: 9×2 = 18D
  D 단독: 9+6 = 15D

cumulative E_n dim:
  A: 12
  A+B: 13
  A+B+C: 13×2 = 26
  A+B+C+D: 26+6 = 32
"""

from __future__ import annotations

import numpy as np

from src.pb_0_6822.plan014_paradigm import (
    Plan014F0Function,
    finite_diff_at,
    make_seq_features,  # plan-014 base 9D
    EPS,
    EPS_BASIS,
)


def _compute_displacement_F0(X: np.ndarray, end_idx: int, indices: list[int]) -> np.ndarray:
    """Feature A: per-step displacement_F0[s] = F0_pred[s] − X[:, s].

    Returns (N, len(indices), 3). F0_pred[s] = step-local horizon=2 prior
    (plan-006 frozen constants d1=1.98 / par=1.20 / perp=-0.20).
    """
    f0 = Plan014F0Function()  # plan-006 carry
    N = X.shape[0]
    out = np.zeros((N, len(indices), 3), dtype=np.float32)
    for i, s in enumerate(indices):
        if s < 2:
            # edge case: zero-fill (baseline end_idx=10 환경에서 미발생)
            continue
        # step-local horizon=2 prior (plan-014 §A.1 carry)
        v_last_s = X[:, s] - X[:, s - 1]
        v_prev_s = X[:, s - 1] - X[:, s - 2]
        acc_s = v_last_s - v_prev_s
        t_hat_s = v_last_s / (np.linalg.norm(v_last_s, axis=1, keepdims=True) + EPS)
        acc_par_scalar = np.sum(acc_s * t_hat_s, axis=1, keepdims=True)
        acc_par_vec = acc_par_scalar * t_hat_s
        acc_perp_vec = acc_s - acc_par_vec
        F0_pred_s = (X[:, s]
                     + f0.d1 * v_last_s
                     + f0.par * acc_par_vec
                     + f0.perp * acc_perp_vec)
        displacement = F0_pred_s - X[:, s]  # (N, 3), F0 방향 vector
        out[:, i, :] = displacement.astype(np.float32)
    return out


def _build_step_local_frenet_basis_dim_B(X: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Feature B: step-local n̂_s, b̂_s. Returns (n̂_s, b̂_s) each (N, 3).

    edge case (degenerate motion ‖v_s‖ < EPS_BASIS or ‖acc_perp‖ < EPS_BASIS):
      n̂_s = world ẑ post-orthogonalize (plan-014 §A.1 carry).
    """
    N = X.shape[0]
    v_s = X[:, s] - X[:, s - 1]
    acc_s = X[:, s] - 2 * X[:, s - 1] + X[:, s - 2]
    speed = np.linalg.norm(v_s, axis=1, keepdims=True)
    t_hat_s = v_s / (speed + EPS)
    acc_par_scalar = np.sum(acc_s * t_hat_s, axis=1, keepdims=True)
    acc_perp_vec = acc_s - acc_par_scalar * t_hat_s
    perp_norm = np.linalg.norm(acc_perp_vec, axis=1, keepdims=True)

    n_hat_s = np.where(
        perp_norm < EPS_BASIS,
        np.tile(np.array([[0.0, 0.0, 1.0]]), (N, 1)),
        acc_perp_vec / (perp_norm + EPS),
    )
    # post-orthogonalize degenerate cases
    degenerate = (perp_norm < EPS_BASIS).squeeze(-1)
    if degenerate.any():
        proj = np.sum(n_hat_s[degenerate] * t_hat_s[degenerate], axis=1, keepdims=True)
        n_hat_s[degenerate] = n_hat_s[degenerate] - proj * t_hat_s[degenerate]
        n_norm = np.linalg.norm(n_hat_s[degenerate], axis=1, keepdims=True)
        n_hat_s[degenerate] = n_hat_s[degenerate] / (n_norm + EPS)

    b_hat_s = np.cross(t_hat_s, n_hat_s)
    return n_hat_s.astype(np.float32), b_hat_s.astype(np.float32)


def _apply_B_split(base: np.ndarray, X: np.ndarray, indices: list[int]) -> np.ndarray:
    """Feature B: base 의 (5) perp_norm/speed 1D 자리에 (acc_normal, acc_binormal)/speed 2D swap.

    base: (N, T, D) where T = len(indices). Returns (N, T, D+1).
    """
    N, T, D = base.shape
    out = np.zeros((N, T, D + 1), dtype=np.float32)

    # (5) 자리 index = 4 (0-indexed), plan-014 feature order:
    # (0) speed, (1) prev_speed/speed, (2) acc_norm/speed, (3) acc_par/speed,
    # (4) perp_norm/speed, (5) jerk_norm/speed, (6) turn_cos, (7) curvature, (8) direction
    perp_idx = 4

    # copy base[..., :perp_idx] → out[..., :perp_idx]
    out[:, :, :perp_idx] = base[:, :, :perp_idx]

    # compute (acc_normal, acc_binormal)/speed per step
    for i, s in enumerate(indices):
        if s < 2:
            # edge: zero
            continue
        v_s = X[:, s] - X[:, s - 1]
        acc_s = X[:, s] - 2 * X[:, s - 1] + X[:, s - 2]
        speed = np.linalg.norm(v_s, axis=1) + EPS  # (N,)
        n_hat_s, b_hat_s = _build_step_local_frenet_basis_dim_B(X, s)
        acc_normal = np.sum(acc_s * n_hat_s, axis=1) / speed  # (N,) sign-aware
        acc_binormal = np.sum(acc_s * b_hat_s, axis=1) / speed  # (N,) sign-aware
        out[:, i, perp_idx] = acc_normal.astype(np.float32)
        out[:, i, perp_idx + 1] = acc_binormal.astype(np.float32)

    # copy base[..., perp_idx+1:] → out[..., perp_idx+2:]
    out[:, :, perp_idx + 2:] = base[:, :, perp_idx + 1:]
    return out


def _apply_C_multi_stride(X: np.ndarray, end_idx: int, direction: float, base_per_stride_fn) -> np.ndarray:
    """Feature C: τ=1, τ=2 2 stream concat. base_per_stride_fn(X, end_idx, direction, stride) → (N, 6, D).

    Returns (N, 6, D × 2).
    """
    base_t1 = base_per_stride_fn(X, end_idx, direction, stride=1)
    base_t2 = base_per_stride_fn(X, end_idx, direction, stride=2)
    return np.concatenate([base_t1, base_t2], axis=-1)


def _make_seq_features_stride(X: np.ndarray, end_idx: int, direction: float, stride: int) -> np.ndarray:
    """plan-014 make_seq_features 의 stride 변형. stride=1 = plan-014 original.

    indices 산출:
      stride=1: range(max(3, end_idx-5), end_idx+1, 1) → 6 step
      stride=2: range(max(3, end_idx-10), end_idx+1, 2) → 4 step + pad → 6 step

    Returns (N, 6, 9) (plan-014 base feature dim).
    """
    if stride == 1:
        return make_seq_features(X, end_idx=end_idx, direction=direction)
    # stride=2: re-implement with custom indices
    indices = list(range(max(3, end_idx - 10), end_idx + 1, stride))
    if len(indices) < 6:
        indices = [indices[0]] * (6 - len(indices)) + indices
    # plan-014 _turn_features_per_step 와 동일 산식, 단 step indices 만 다름
    from src.pb_0_6822.plan014_paradigm import _turn_features_per_step
    feats = []
    N = X.shape[0]
    for s in indices:
        f8 = _turn_features_per_step(X, s)
        dir_col = np.full((N, 1), direction, dtype=np.float32)
        feats.append(np.concatenate([f8, dir_col], axis=1))
    return np.stack(feats, axis=1)  # (N, 6, 9)


def _apply_D_pairwise(X: np.ndarray, indices: list[int]) -> np.ndarray:
    """Feature D: per-step 에 cross-step pairwise 6D 추가.

    3 pair × 2 stat = 6D: (s, s-2), (s, s-4), (s-2, s-4) × (cosine, Δspeed).
    Returns (N, len(indices), 6). edge case (s-4 < 0): zero fill.
    """
    N = X.shape[0]
    T = len(indices)
    out = np.zeros((N, T, 6), dtype=np.float32)

    for i, s in enumerate(indices):
        for j, (sa, sb) in enumerate([(s, s - 2), (s, s - 4), (s - 2, s - 4)]):
            if sb < 1 or sa < 1:
                # edge: zero (cosine=0, Δspeed=0)
                continue
            va = X[:, sa] - X[:, sa - 1]
            vb = X[:, sb] - X[:, sb - 1]
            speed_a = np.linalg.norm(va, axis=1)  # (N,)
            speed_b = np.linalg.norm(vb, axis=1)
            cos_sim = np.sum(va * vb, axis=1) / ((speed_a * speed_b) + EPS)
            d_speed = speed_a - speed_b
            out[:, i, 2 * j] = cos_sim.astype(np.float32)
            out[:, i, 2 * j + 1] = d_speed.astype(np.float32)
    return out


def make_seq_features_v2(
    X: np.ndarray,
    end_idx: int = 10,
    direction: float = 1.0,
    *,
    feature_flags: dict[str, bool],
) -> np.ndarray:
    """plan-015 feature expansion main entry. 분기 로직 = §5.1 v2.4 박제.

    Returns (N, 6, target_dim) where target_dim 은 feature_flags 별:
      단독: A=12, B=10, C=18, D=15
      cumulative: A=12, A+B=13, A+B+C=26, A+B+C+D=32
    """
    a = feature_flags.get("A", False)
    b = feature_flags.get("B", False)
    c = feature_flags.get("C", False)
    d = feature_flags.get("D", False)

    # base + A + B 산출 함수 (C 의 stream 별 호출 위해 closure)
    def _base_with_a_b(X_, end_idx_, direction_, stride: int) -> np.ndarray:
        # stride 별 base 9D
        base_s = _make_seq_features_stride(X_, end_idx_, direction_, stride)
        # A: + displacement_F0 (stride 의 indices 에 맞춰 산출)
        if stride == 1:
            indices = list(range(max(3, end_idx_ - 5), end_idx_ + 1, 1))
            if len(indices) < 6:
                indices = [indices[0]] * (6 - len(indices)) + indices
        else:
            indices = list(range(max(3, end_idx_ - 10), end_idx_ + 1, stride))
            if len(indices) < 6:
                indices = [indices[0]] * (6 - len(indices)) + indices
        if a:
            disp = _compute_displacement_F0(X_, end_idx_, indices)
            base_s = np.concatenate([base_s, disp.astype(np.float32)], axis=-1)
        # B: perp_norm/speed 1D → 2D split
        if b:
            base_s = _apply_B_split(base_s, X_, indices)
        return base_s

    # C 분기: τ=1, τ=2 stream concat (각 stream 에 A/B 이미 적용)
    if c:
        feat = _apply_C_multi_stride(X, end_idx, direction, _base_with_a_b)
    else:
        feat = _base_with_a_b(X, end_idx, direction, stride=1)

    # D: pairwise 6D 추가 (stride=1 의 indices 기준, single stride feature)
    if d:
        # use stride=1 indices
        indices = list(range(max(3, end_idx - 5), end_idx + 1, 1))
        if len(indices) < 6:
            indices = [indices[0]] * (6 - len(indices)) + indices
        pairwise = _apply_D_pairwise(X, indices)
        feat = np.concatenate([feat, pairwise], axis=-1)

    return feat.astype(np.float32)
