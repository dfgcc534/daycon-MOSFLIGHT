"""plan-012 c2 smoke test for ring_classifier.py."""

import numpy as np
import torch

from src.pb_0_6822 import ring_classifier as rc


# ── 컴포넌트 1: codebook geometry ──


def test_codebook_geometry_absolute():
    a = rc.compute_anchors_absolute()
    assert a.shape == (7, 3)
    assert np.linalg.norm(a[0]) < 1e-6
    norms = np.linalg.norm(a[1:7], axis=1)
    assert np.all(np.abs(norms - 0.005) < 1e-9)
    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    ) * 0.005
    assert np.allclose(a, expected, atol=1e-12)


def test_codebook_geometry_frenet():
    a = rc.compute_anchors_frenet_orthogonal()
    assert a.shape == (7, 3)
    assert np.linalg.norm(a[0]) < 1e-6
    assert np.allclose(np.linalg.norm(a[1:7], axis=1), 0.005)


def test_clip_norm_passthrough():
    v = np.array([[0.001, 0.0, 0.0], [0.010, 0.0, 0.0]], dtype=np.float64)
    out = rc.clip_norm(v, max_norm=0.005)
    assert np.allclose(out[0], [0.001, 0.0, 0.0])
    assert np.allclose(np.linalg.norm(out[1]), 0.005)


def test_compute_anchors_kmeans_fold_aware():
    rng = np.random.default_rng(42)
    N = 1000
    residuals = rng.normal(scale=0.005, size=(N, 3))
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    fold_id = np.arange(N) % 5
    centers, sizes, meta = rc.compute_anchors_kmeans(residuals, R, fold_id, K=7)
    assert centers.shape == (5, 7, 3)
    for k in range(5):
        assert np.linalg.norm(centers[k, 0]) < 1e-6
    assert sizes[:, 1:].min() > 10  # synthetic small data
    assert meta["K"] == 7


# ── 컴포넌트 2: Frenet basis 3D ──


def test_build_frenet_basis_3d_identity():
    # 직선 운동 + curvature 추가 → t 정상, n 정상, b cross
    x = np.zeros((4, 5, 3), dtype=np.float64)
    for i in range(5):
        x[:, i, 0] = i * 0.01           # +x velocity
        x[:, i, 1] = 0.001 * i * i      # parabolic +y → curvature
    R = rc.build_frenet_basis_3d(x, end_idx=4)
    assert R.shape == (4, 3, 3)
    # t̂ ≈ +x (predominantly; with curvature t̂_x ~ 0.82)
    assert (R[:, 0, 0] > 0.8).all()
    # determinant = +1 (right-handed) approx
    dets = np.linalg.det(R)
    assert np.all(np.abs(dets - 1.0) < 1e-6)


# ── 컴포넌트 3: anchors_to_world ──


def test_anchors_to_world_absolute():
    a = rc.compute_anchors_absolute()
    out = rc.anchors_to_world(a, None, N=4)
    assert out.shape == (4, 7, 3)
    assert np.allclose(out[0], a)
    assert np.allclose(out[3], a)


def test_anchors_to_world_frenet():
    a = rc.compute_anchors_frenet_orthogonal()
    R = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    out = rc.anchors_to_world(a, R, N=4)
    assert out.shape == (4, 7, 3)
    assert np.allclose(out[0], a)


# ── 컴포넌트 4: candidate features ──


def test_make_codebook_candidate_features():
    a = rc.compute_anchors_absolute()
    F0 = np.zeros((4, 3), dtype=np.float64)
    feats = rc.make_codebook_candidate_features(None, a, "absolute", None, F0)
    assert feats.shape == (4, 7, 11)
    assert feats.dtype == np.float32
    # codebook_id one-hot: only [8]=1
    assert (feats[..., 8] == 1.0).all()
    assert (feats[..., 9] == 0.0).all()
    assert (feats[..., 10] == 0.0).all()
    # is_origin: only k=0
    assert feats[0, 0, 4] == 1.0
    assert feats[0, 1, 4] == 0.0


# ── 컴포넌트 5: HybridScorerHead ──


def test_hybrid_forward_shape():
    head = rc.HybridScorerHead(K=7, hidden=64, cand_dim=11)
    seq = torch.randn(4, 6, 9)
    cand = torch.randn(4, 7, 11)
    logits, reg = head(seq, cand)
    assert logits.shape == (4, 7)
    assert reg.shape == (4, 7, 3)
    # tanh × 0.005 → |reg| ≤ 0.005 + tiny float epsilon
    assert reg.abs().max().item() < 0.0051


def test_hybrid_extract_seq_hidden_shape():
    head = rc.HybridScorerHead(K=7, hidden=64, cand_dim=11)
    assert hasattr(head.scorer, "gru")
    assert hasattr(head.scorer, "ctx_norm")
    seq = torch.randn(4, 6, 9)
    out = head._extract_seq_hidden(seq)
    assert out.shape == (4, 64)


def test_hybrid_freeze_encoder():
    head = rc.HybridScorerHead(K=7, hidden=64, cand_dim=11)
    head.freeze_encoder()
    for p in head.scorer.gru.parameters():
        assert not p.requires_grad
    for p in head.reg_head.parameters():
        assert p.requires_grad  # reg_head 는 trainable 유지


def test_last_step_mlp_scorer():
    s = rc.LastStepMLPScorer(seq_dim=9, cand_dim=11, hidden=64, cand_count=7)
    seq = torch.randn(4, 6, 9)
    cand = torch.randn(4, 7, 11)
    logits = s(seq, cand)
    assert logits.shape == (4, 7)


# ── 컴포넌트 6: losses ──


def test_hit_aware_hinge_zero_at_hit():
    pos = torch.zeros(4, 3)
    target = torch.zeros(4, 3)
    loss = rc.hit_aware_hinge(pos, target, R_HIT=0.01)
    # within R_HIT → softplus(-2.0)·smooth ≈ 0.0006 → squared 약 4e-7. 작은 값.
    assert loss.max().item() < 1e-3


def test_classifier_ce_runs():
    logits = torch.randn(4, 7, requires_grad=True)
    F0 = torch.zeros(4, 3)
    a = torch.tensor(rc.compute_anchors_absolute(), dtype=torch.float32)
    anchors_world = a[None, :, :].expand(4, -1, -1).contiguous()
    target = torch.zeros(4, 3)
    target[:, 0] = 0.005  # +x anchor 가 정답
    ce = rc.classifier_ce_loss(logits, F0, anchors_world, target)
    ce.backward()
    assert torch.isfinite(ce)


def test_hybrid_combined_loss_softblend():
    logits = torch.randn(4, 7)
    reg = torch.zeros(4, 7, 3)
    a = torch.tensor(rc.compute_anchors_absolute(), dtype=torch.float32)
    anchors_world = a[None, :, :].expand(4, -1, -1).contiguous()
    F0 = torch.zeros(4, 3)
    target = torch.zeros(4, 3)
    target[:, 0] = 0.005
    loss = rc.hybrid_combined_loss(logits, reg, anchors_world, F0, target, temperature=0.03)
    assert torch.isfinite(loss)


def test_hybrid_combined_loss_argmax():
    logits = torch.randn(4, 7)
    reg = torch.zeros(4, 7, 3)
    a = torch.tensor(rc.compute_anchors_absolute(), dtype=torch.float32)
    anchors_world = a[None, :, :].expand(4, -1, -1).contiguous()
    F0 = torch.zeros(4, 3)
    target = torch.zeros(4, 3)
    loss = rc.hybrid_combined_loss(logits, reg, anchors_world, F0, target, temperature=0.0)
    assert torch.isfinite(loss)


# ── 컴포넌트 7: hybrid_predict ──


def test_hybrid_predict_shape_argmax():
    logits = torch.randn(4, 7)
    reg = torch.zeros(4, 7, 3)
    a = torch.tensor(rc.compute_anchors_absolute(), dtype=torch.float32)
    anchors_world = a[None, :, :].expand(4, -1, -1).contiguous()
    F0 = torch.zeros(4, 3)
    out = rc.hybrid_predict(logits, reg, anchors_world, F0, temperature=0.0)
    assert out.shape == (4, 3)


def test_hybrid_predict_r0_prior():
    logits = torch.zeros(4, 7)
    reg = torch.zeros(4, 7, 3)
    a = torch.tensor(rc.compute_anchors_absolute(), dtype=torch.float32)
    anchors_world = a[None, :, :].expand(4, -1, -1).contiguous()
    F0 = torch.zeros(4, 3)
    out_neutral = rc.hybrid_predict(logits, reg, anchors_world, F0, temperature=0.0, r0_logit_prior=0.0)
    out_prior = rc.hybrid_predict(logits, reg, anchors_world, F0, temperature=0.0, r0_logit_prior=10.0)
    # r=0 anchor = origin → strong prior 시 prediction = F0 (no shift)
    assert torch.allclose(out_prior, F0)


# ── F0 산출식 ──


def test_f0_predict_frenet_par120_perp_neg020():
    # 직선 운동: par_coef=1.20 → 다음 step 위치 = x[end] + 1.20 * v_par * t̂
    x = np.zeros((3, 5, 3), dtype=np.float64)
    for i in range(5):
        x[:, i, 0] = i * 0.01  # +x 직선 운동, speed=0.01
    F0 = rc.f0_predict_frenet_par120_perp_neg020(x)
    # v_par = 0.01, par=1.20, perp=0 → delta_x = 0.012
    expected_x = 4 * 0.01 + 1.20 * 0.01
    assert np.allclose(F0[:, 0], expected_x, atol=1e-9)
    assert np.allclose(F0[:, 1], 0.0, atol=1e-9)
    assert np.allclose(F0[:, 2], 0.0, atol=1e-9)
