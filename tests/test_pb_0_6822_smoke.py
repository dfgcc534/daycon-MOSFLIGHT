"""tests/test_pb_0_6822_smoke.py — plan-004 c4 G0 gate.

Module extraction smoke test: import paths + 27 candidates + key functions.
NO training. NO data. Pure import + instance check.
"""
from __future__ import annotations


def test_selector_module_imports():
    from src.pb_0_6822 import selector

    assert callable(selector.SELECTOR_MAIN)
    assert callable(selector.fit_regime_bins)
    assert callable(selector.assign_regimes)
    assert callable(selector.candidate_regime_bias)
    assert callable(selector.candidate_physics_bias)
    assert callable(selector.make_candidates)
    assert callable(selector.make_candidate_features)
    assert callable(selector.motion_terms)
    assert callable(selector.read_labels)
    assert callable(selector.load_stack)
    assert callable(selector.stable_fold_id)
    assert callable(selector.metrics)


def test_selector_candidates_27():
    """27 physics candidates per notebook design — frenet/turn/jerk/latency families."""
    from src.pb_0_6822 import selector

    assert len(selector.CANDIDATES) == 27
    names = [c.name for c in selector.CANDIDATES]
    # Must include representative names from each family
    assert "p0_2d1" in names  # base family
    assert "frenet_best" in names  # frenet family
    assert "jerk_small_pos" in names  # jerk family
    assert any("latency" in n for n in names)  # latency family

    # All candidates have required fields
    for c in selector.CANDIDATES:
        assert hasattr(c, "name")
        assert hasattr(c, "d1")
        assert hasattr(c, "par")
        assert hasattr(c, "perp")
        assert hasattr(c, "d2")
        assert hasattr(c, "jerk")
        assert hasattr(c, "time_scale")


def test_selector_constants():
    from src.pb_0_6822 import selector

    assert selector.R_HIT == 0.01  # 1cm hit boundary
    assert len(selector.FAMILY_NAMES) == 6
    assert "base" in selector.FAMILY_NAMES
    assert "frenet" in selector.FAMILY_NAMES
    assert "latency" in selector.FAMILY_NAMES
    # Candidate family map exists
    assert hasattr(selector, "CANDIDATE_FAMILY")
    assert len(selector.CANDIDATE_FAMILY) == 27


def test_boundary_module_imports():
    from src.pb_0_6822 import boundary

    assert callable(boundary.BOUNDARY_MAIN)
    assert callable(boundary.local_frame)
    assert callable(boundary.vector_to_local)
    assert callable(boundary.local_to_vector)
    assert callable(boundary.cap_vectors)
    assert callable(boundary.make_rows)
    assert callable(boundary.build_pretrain)
    assert callable(boundary.train_net)
    assert callable(boundary.predict_delta)
    assert callable(boundary.evaluate)
    assert callable(boundary.predict_corrected_candidates)
    assert boundary.TinyCorrectionNet is not None
    assert boundary.ResidualMLPBlock is not None


def test_boundary_tiny_correction_net_instantiates():
    """TinyCorrectionNet must instantiate with reasonable dims + have delta zero-init."""
    import torch

    from src.pb_0_6822 import boundary

    model = boundary.TinyCorrectionNet(dim=20, hidden=64)
    assert model is not None

    # delta head's last linear must be zero-init (per notebook design — see L171-172)
    delta_last = model.delta[-1]
    assert torch.allclose(delta_last.weight, torch.zeros_like(delta_last.weight))
    assert torch.allclose(delta_last.bias, torch.zeros_like(delta_last.bias))


def test_boundary_uses_selector_as_base():
    """boundary.py must import selector as base — sanity check on import alias."""
    from src.pb_0_6822 import boundary, selector

    # base is the selector module
    assert boundary.base is selector
    # base re-exports key symbols boundary depends on
    assert boundary.base.R_HIT == 0.01
    assert len(boundary.base.CANDIDATES) == 27
