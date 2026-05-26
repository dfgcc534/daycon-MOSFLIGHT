"""plan-a-002 §4.2 — KR002 입력 파이프라인 확장 (innov·filtered_v seq, cv_ca scalar).

채널 순서 고정 규약 (§4.2):
  seq    = [KR002 9채널(rel/v/a) | innov 3 | filtered_v 3]   (flag off 시 블록 생략)
  scalar = [KR002 40 | cv_ca 회전3 + norm1]
신규 벡터 채널(innov/filtered_v/cv_ca-3D)은 KR002 와 동일 θ 로 rotate_xy (frame 일관, z 보존).
cv_ca norm = ‖CA−CV‖₂ (rotate_xy L2 불변 → 회전 전/후 동일).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_A001 = _THIS.parent / "plan-a-001"


def _load_a001(name: str):
    spec = importlib.util.spec_from_file_location(f"pa001_{name}", _A001 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_feat = _load_a001("features")
_yaw = _load_a001("yaw")
build_seq_t3 = _feat.build_seq_t3
build_scalar_40d = _feat.build_scalar_40d
compute_noise = _feat.compute_noise
normalize_seq = _feat.normalize_seq
SEQ_CHANNELS = _feat.SEQ_CHANNELS  # 9 (rel/v/a)


def rotate_all_triplets(seq: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """seq (N,T,C) 의 모든 3-채널 triplet 을 rotate_xy(θ) (z 보존). C 는 3 의 배수."""
    N, T, C = seq.shape
    assert C % 3 == 0, f"seq C={C} 는 3 의 배수여야 (전부 벡터 triplet)"
    out = seq.copy()
    th = np.repeat(theta, T)  # (N*T,)
    for s in range(0, C, 3):
        flat = seq[:, :, s:s + 3].reshape(-1, 3)
        out[:, :, s:s + 3] = _yaw.rotate_xy(flat, th).reshape(N, T, 3)
    return out.astype(np.float32)


def build_seq_ext(
    X: np.ndarray,
    *,
    innov_arr: np.ndarray | None = None,
    filtered_v_arr: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    input_yaw: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """KR002 build_seq_t3(9채널) + optional innov(3)/filtered_v(3) → (N,11,C) + names.

    input_yaw 시 모든 triplet(rel/v/a + 신규)을 동일 θ 로 rotate_xy.
    """
    blocks = [build_seq_t3(X)]
    names = list(SEQ_CHANNELS)
    if innov_arr is not None:
        blocks.append(innov_arr.astype(np.float32))
        names += ["innov_x", "innov_y", "innov_z"]
    if filtered_v_arr is not None:
        blocks.append(filtered_v_arr.astype(np.float32))
        names += ["fv_x", "fv_y", "fv_z"]
    seq = np.concatenate(blocks, axis=-1).astype(np.float32)
    if input_yaw:
        assert theta is not None, "input_yaw=True 인데 theta 미지정"
        seq = rotate_all_triplets(seq, theta)
    return seq, names


def build_scalar_ext(
    X: np.ndarray,
    noise_p: np.ndarray,
    noise_s: np.ndarray,
    noise_loo: np.ndarray,
    *,
    cv_ca_arr: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    input_yaw: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """KR002 scalar_40d(40, 회전불변) + optional cv_ca(회전3 + norm1) → (N,S) + names + rot_class.

    cv_ca norm = ‖CA−CV‖₂ (raw 에서 산출; rotate_xy L2 불변이라 회전 전/후 동일).
    cv_ca 3D 는 input_yaw 시 rotate_xy(θ), else raw. norm 은 항상 invariant.
    """
    base, base_names = build_scalar_40d(X, noise_p, noise_s, noise_loo)
    blocks = [base]
    names = list(base_names)
    rot_class = {n: "invariant" for n in base_names}  # 40D 전부 magnitude/cos
    if cv_ca_arr is not None:
        norm = np.linalg.norm(cv_ca_arr, axis=-1, keepdims=True)  # (N,1) 회전불변
        if input_yaw:
            assert theta is not None, "input_yaw=True 인데 theta 미지정"
            vec = _yaw.rotate_xy(cv_ca_arr, theta)
        else:
            vec = cv_ca_arr
        blocks.append(np.concatenate([vec, norm], axis=-1).astype(np.float32))
        cc_names = ["cvca_x", "cvca_y", "cvca_z", "cvca_norm"]
        names += cc_names
        rot_class.update({
            "cvca_x": "rotate", "cvca_y": "rotate", "cvca_z": "rotate",
            "cvca_norm": "invariant",
        })
    scal = np.concatenate(blocks, axis=-1).astype(np.float32)
    return scal, names, rot_class
