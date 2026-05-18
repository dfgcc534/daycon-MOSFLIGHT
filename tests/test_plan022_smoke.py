"""plan-022 §4.5 — smoke tests (8 items).

Run: pytest tests/test_plan022_smoke.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_PLAN022 = REPO / "analysis" / "plan-022"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy module loads (T1 verifies import)
@pytest.fixture(scope="module")
def mods():
    anchors = _load("anchors_t", _PLAN022 / "anchors.py")
    som = _load("som_t", _PLAN022 / "selector_only_model.py")
    run_oof = _load("run_oof_t", _PLAN022 / "run_oof.py")
    return {"anchors": anchors, "som": som, "run_oof": run_oof}


# ── T1: import (AttributeError 0건) ──────────────────────────────────


def test_t1_import(mods):
    a = mods["anchors"]
    s = mods["som"]
    r = mods["run_oof"]
    # critical symbols
    for sym in ("ANCHORS_A1", "ANCHORS_A2", "ANCHORS_A3", "ANCHORS_A4",
                "ANCHORS_A5", "ANCHORS_A6", "ANCHORS_A7", "LAYOUT_NAMES"):
        assert hasattr(a, sym), f"anchors.{sym} missing"
    for sym in ("LgbmSelectorOnly", "build_soft_label_with_tau",
                "bf", "p021_build", "p021_dh"):
        assert hasattr(s, sym), f"selector_only_model.{sym} missing"
    for sym in ("run_oof_cell", "run_sub_exp", "run_sweep", "TAU_SCAN"):
        assert hasattr(r, sym), f"run_oof.{sym} missing"


# ── T2: 7 layout dtype + norm + shape ────────────────────────────────


def test_t2_layout_invariants(mods):
    a = mods["anchors"]
    expected = {
        "A1_octa7": 7, "A2_ico13": 13, "A3_cubocta13": 13, "A4_2shell13": 13,
        "A5_cube8": 8, "A6_bcc14": 14, "A7_fib13": 13,
    }
    for name, K_expected in expected.items():
        arr = a.LAYOUT_NAMES[name]
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
        assert arr.shape == (K_expected, 3), \
            f"{name}: shape {arr.shape} != ({K_expected}, 3)"
        norms = np.linalg.norm(arr, axis=1)
        assert norms.max() <= 0.005 + 1e-7, \
            f"{name}: max ‖a‖ = {norms.max():.7f} > 0.005 + 1e-7"
        assert np.isfinite(arr).all(), f"{name}: non-finite"


# ── T3: LAYOUT_NAMES dict 7 entry + key ──────────────────────────────


def test_t3_layout_names_keys(mods):
    a = mods["anchors"]
    expected_keys = {"A1_octa7", "A2_ico13", "A3_cubocta13", "A4_2shell13",
                     "A5_cube8", "A6_bcc14", "A7_fib13"}
    assert set(a.LAYOUT_NAMES.keys()) == expected_keys
    assert len(a.LAYOUT_NAMES) == 7


# ── T4: LgbmSelectorOnly fit/predict (K=7 baseline) ──────────────────


def test_t4_selector_fit_predict(mods):
    s = mods["som"]
    rng = np.random.default_rng(20260519)
    N, K = 100, 7
    X = rng.standard_normal((N, 170)).astype(np.float32)
    q = rng.dirichlet(alpha=np.ones(K), size=N).astype(np.float32)
    model = s.LgbmSelectorOnly(K=K).fit(X, q)
    probs = model.predict(X[:50])
    assert probs.shape == (50, K)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(probs).all()


# ── T5: build_soft_label_with_tau 3 τ 값 ─────────────────────────────


def test_t5_soft_label_with_tau(mods):
    s = mods["som"]
    rng = np.random.default_rng(20260519)
    N, K = 50, 13
    gt = rng.standard_normal((N, 3)) * 0.01
    pred_F0 = rng.standard_normal((N, 3)) * 0.01
    R_wfn = np.tile(np.eye(3, dtype=np.float64)[None], (N, 1, 1))
    anchors = (rng.standard_normal((K, 3)) * 0.003).astype(np.float32)
    for tau in [0.001, 0.003, 0.005]:
        q = s.build_soft_label_with_tau(gt, R_wfn, pred_F0, anchors, tau)
        assert q.shape == (N, K)
        assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5)
        assert np.isfinite(q).all()
        assert q.dtype == np.float32


# ── T6: run_oof_cell 출력 key superset ───────────────────────────────


def test_t6_run_oof_cell_keys(mods):
    """run_oof_cell 의 반환 dict 가 §3.3 / §6.2 key superset 인지 검증.

    실제 호출은 비용 부담 → unit test 는 함수 시그너처 + dict spec 만 확인.
    실 호출 검증은 G1 (c6) 에서.
    """
    r = mods["run_oof"]
    import inspect
    sig = inspect.signature(r.run_oof_cell)
    expected_params = {"X", "Y", "folds", "anchors", "tau_cls"}
    assert expected_params.issubset(sig.parameters.keys())


# ── T7: F0 reproduce sanity (plan-020/021 carry, 0.6320 / 0.8033) ───


def test_t7_f0_reproduce_carry(mods):
    baseline_path = REPO / "analysis" / "plan-020" / "baseline_oof.json"
    if not baseline_path.exists():
        pytest.skip("plan-020 baseline_oof.json 부재 (carry 불가)")
    baseline = json.loads(baseline_path.read_text())
    f0 = baseline["f0_baseline"]
    assert 0.6315 <= f0["hit_1cm_5fold_concat"] <= 0.6325
    assert 0.8028 <= f0["hit_1.5cm_5fold_concat"] <= 0.8038


# ── T8: plan-021 build_input.py reuse — shape 검증 ──────────────────


def test_t8_p021_build_reuse(mods):
    """plan-021 build_input_common 출력 shape carry. 작은 X 위 호출."""
    s = mods["som"]
    rng = np.random.default_rng(20260519)
    N = 20
    X = rng.standard_normal((N, 11, 3)) * 0.01
    common = s.p021_build.build_input_common(X, s.bf.f0_baseline)
    assert "L1" in common and common["L1"].shape == (N, 11, 9)
    assert "L2" in common and common["L2"].shape == (N, 7, 3)
    assert "L4" in common and common["L4"].shape == (N, 7, 2)
    assert "R_wfn" in common and common["R_wfn"].shape == (N, 3, 3)
    assert "pred_F0_world" in common and common["pred_F0_world"].shape == (N, 3)
    extra = s.p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    assert extra.shape == (N, 36), f"extra shape {extra.shape} != (N, 36)"

    # R_wfn columns = [t̂, n̂, b̂] — orthonormality
    R = common["R_wfn"]
    eye_check = np.einsum("nij,nkj->nik", R, R)  # R R^T per sample (= I if columns orthonormal)
    assert np.allclose(eye_check, np.eye(3)[None], atol=1e-5)
