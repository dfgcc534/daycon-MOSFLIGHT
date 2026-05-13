"""plan-011 c3 — corrector_redesign_v2.py

self-contained module: plan-010 의 RedesignedCorrectionNet 을 inline 정의 (plan-010 c3 미실행 상태)
+ plan-011 의 4 axis (Loss × Input × Arch × Formula) 신규 components.

★ 본 모듈은 plan-011 §5.1 / §6.1 / §7.1 / §8.1 의 spec 을 그대로 구현.
★ boundary.py 의 ResidualMLPBlock 만 read-only reference (whitelist OK).

decision-note: spec-default — plan-010 c3 (corrector_redesign.py) 미실행 → RedesignedCorrectionNet
                를 본 v2 모듈에 inline 정의 (= plan-011 자족화). plan-010 진행 시 부모 클래스 swap.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

from src.pb_0_6822.boundary import ResidualMLPBlock


R_HIT = 0.01
EPS = 1e-6
CAP_6MM = 0.006


# ──────────────────────────────────────────────────────────────────────────────
# Z1 base (plan-010 §5.1) — RedesignedCorrectionNet
# ──────────────────────────────────────────────────────────────────────────────


class RedesignedCorrectionNet(nn.Module):
    """plan-010 §5.1 Z1 minimum: D1 (env head 제거) + E1 (apply_scale=1.0) + B1 (uncapped target).

    Forward contract (plan-011 §7.1 unified):
      forward(cf, encoder_emb=None) -> (delta: (B, 3), aux: dict)
    """

    def __init__(self, dim_cf: int, hidden: int = 64, dim_encoder: int = 0):
        super().__init__()
        in_dim = dim_cf + dim_encoder
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.04),
        )
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden),
            ResidualMLPBlock(hidden),
        ])
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        # D1: NO env head
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def _trunk(self, cf: torch.Tensor, encoder_emb: Optional[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([cf, encoder_emb], dim=-1) if encoder_emb is not None else cf
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(
        self, cf: torch.Tensor, encoder_emb: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        h = self._trunk(cf, encoder_emb)
        delta = self.delta(h)
        return delta, {}


# ──────────────────────────────────────────────────────────────────────────────
# Loss components (plan-011 §5.1)
# ──────────────────────────────────────────────────────────────────────────────


def huber_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.005) -> torch.Tensor:
    """L1: Huber loss, beta=5mm. (B,) per-sample."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="none").sum(dim=1)


def asymmetric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    raw_hit_mask: torch.Tensor,
    corrected_pos: torch.Tensor,
    lambda_destructive: float = 8.0,
) -> torch.Tensor:
    """L2/L3: replacement multiplier (destructive 시 base × λ). per plan-011 §5.1.

    pred: raw delta (BEFORE gate, BEFORE cap). caller 가 GateHead aux["raw_delta"] 주입.
    """
    base = huber_loss(pred, target)
    err_after = torch.norm(corrected_pos - target, dim=1)
    destructive = raw_hit_mask & (err_after > R_HIT)
    return torch.where(destructive, base * lambda_destructive, base)


def build_frenet_basis(
    trajectory_x: torch.Tensor, end_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """plan-011 §5.1 self-contained Frenet basis @ end_idx.

    trajectory_x: (B, T, 3) world. end_idx: (B,) int (last observation index).
    Returns (R_world_to_local: (B, 3, 3), valid_mask: (B,) bool).
      R rows = (t̂, n̂, b̂).
    Degenerate (‖v‖<EPS or ‖n‖<EPS) → identity basis + valid_mask=False.
    """
    B = trajectory_x.shape[0]
    device = trajectory_x.device

    # gather x_T, x_{T-1}, x_{T-2}
    idx0 = end_idx
    idx1 = (end_idx - 1).clamp(min=0)
    idx2 = (end_idx - 2).clamp(min=0)
    arange_b = torch.arange(B, device=device)
    x_T = trajectory_x[arange_b, idx0]      # (B, 3)
    x_T1 = trajectory_x[arange_b, idx1]
    x_T2 = trajectory_x[arange_b, idx2]

    v = x_T - x_T1                          # (B, 3)
    a = v - (x_T1 - x_T2)                   # (B, 3)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    t_hat = v / (v_norm + EPS)              # (B, 3)
    a_par = (a * t_hat).sum(-1, keepdim=True) * t_hat
    n = a - a_par                           # perpendicular component
    n_norm = torch.norm(n, dim=-1, keepdim=True)
    n_hat = n / (n_norm + EPS)
    b_hat = torch.cross(t_hat, n_hat, dim=-1)

    valid = (v_norm.squeeze(-1) > EPS) & (n_norm.squeeze(-1) > EPS)
    R = torch.stack([t_hat, n_hat, b_hat], dim=1)   # (B, 3, 3) rows = basis vectors
    eye = torch.eye(3, device=device).expand(B, -1, -1)
    R = torch.where(valid.view(-1, 1, 1), R, eye)
    return R, valid


def world_to_local(vec_world: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """R @ vec_world. R: (B, 3, 3), vec_world: (B, 3). Returns (B, 3)."""
    return torch.einsum("bij,bj->bi", R, vec_world)


def frenet_anisotropic_loss(
    pred_local: torch.Tensor,
    target_local: torch.Tensor,
    w_par: float = 1.0,
    w_perp: float = 1.0,
    w_bi: float = 0.1,
) -> torch.Tensor:
    """L4: Frenet local-frame anisotropic. (B,) per-sample.

    Additive auxiliary loss; caller combines with huber: total = huber + λ_aniso * this.
    """
    diff = pred_local - target_local
    return w_par * diff[:, 0] ** 2 + w_perp * diff[:, 1] ** 2 + w_bi * diff[:, 2] ** 2


def physics_conservation_loss(
    delta_step: torch.Tensor,
    recent_acc: torch.Tensor,
    typical_jerk_step: float = 0.004,
) -> torch.Tensor:
    """L5: jerk-penalty. delta_step = corrector delta / horizon (caller scales).

    delta_step, recent_acc: (B, 3) — all step-domain (m/step). Returns (B,) penalty.
    """
    jerk = torch.norm(delta_step - recent_acc, dim=1)
    excess = torch.clamp(jerk - typical_jerk_step, min=0.0)
    return excess ** 2


def bell_shape_weight(err: torch.Tensor, R_HIT_v: float = R_HIT, sigma: float = 0.005) -> torch.Tensor:
    """L6: Gaussian weight at R_HIT. err: (B,). Returns (B,) ∈ (0, 1]."""
    return torch.exp(-((err - R_HIT_v) / sigma) ** 2)


def hit_aware_hinge(
    corrected_pos: torch.Tensor, target: torch.Tensor, R_HIT_v: float = R_HIT, smooth: float = 0.005
) -> torch.Tensor:
    """L7: smooth squared hinge. Returns (B,) m² units (huber 와 동차)."""
    err = torch.norm(corrected_pos - target, dim=1)
    excess = err - R_HIT_v
    linear_hinge = F.softplus(excess / smooth) * smooth
    return linear_hinge ** 2


def gmm_nll_loss(mu: torch.Tensor, logsigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """diagonal Gaussian NLL, batch-mean. mu/logsigma/target: (B, 3)."""
    inv_var = torch.exp(-2.0 * logsigma)
    sq_err = (target - mu) ** 2
    nll = 0.5 * (sq_err * inv_var + 2.0 * logsigma).sum(dim=1)
    return nll.mean()


def cap_6mm(delta: torch.Tensor, cap: float = CAP_6MM) -> torch.Tensor:
    """Z1 cap @ 6mm. delta: (..., 3). Returns clipped (..., 3)."""
    norm = torch.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(cap / (norm + EPS), max=1.0)
    return delta * scale


# ──────────────────────────────────────────────────────────────────────────────
# Arch variants (plan-011 §7.1) — unified contract `forward(cf, encoder_emb) -> (delta, aux)`
# ──────────────────────────────────────────────────────────────────────────────


class GateHeadCorrector(RedesignedCorrectionNet):
    """M1 / L2 base: TinyCorrectionNet + sigmoid gate head."""

    def __init__(self, dim_cf: int, hidden: int = 64, dim_encoder: int = 0, gate_bias_init: float = 2.0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.constant_(self.gate_head[-1].bias, gate_bias_init)

    def forward(self, cf, encoder_emb=None):
        h = self._trunk(cf, encoder_emb)
        raw_delta = self.delta(h)
        gate = torch.sigmoid(self.gate_head(h))   # (B, 1)
        delta = gate * raw_delta
        return delta, {"gate": gate, "raw_delta": raw_delta}


class SplitHeadCorrector(RedesignedCorrectionNet):
    """M2: direction (unit) + magnitude (softplus) split heads."""

    def __init__(self, dim_cf, hidden=64, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.magnitude_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1), nn.Softplus(),
        )

    def forward(self, cf, encoder_emb=None):
        h = self._trunk(cf, encoder_emb)
        direction = F.normalize(self.direction_head(h), dim=-1)
        magnitude = self.magnitude_head(h)
        delta = direction * magnitude
        return delta, {"direction": direction, "magnitude": magnitude}


class BinClassifierCorrector(RedesignedCorrectionNet):
    """M3: 3-axis factorized bin classification (bin_dim=60, bin_size=1mm)."""

    def __init__(self, dim_cf, hidden=64, dim_encoder=0, bin_dim=60, bin_size=0.001):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None
        self.bin_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, bin_dim),
            ) for _ in range(3)
        ])
        self.bin_size = bin_size
        self.bin_dim = bin_dim
        bin_centers = torch.linspace(
            -bin_dim / 2 * bin_size, bin_dim / 2 * bin_size, bin_dim, dtype=torch.float32
        )
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, cf, encoder_emb=None):
        h = self._trunk(cf, encoder_emb)
        delta_per_axis = []
        bin_probs = []
        for head in self.bin_heads:
            logits = head(h)
            prob = F.softmax(logits, dim=-1)
            expected = (prob * self.bin_centers).sum(dim=-1)
            delta_per_axis.append(expected)
            bin_probs.append(prob)
        delta = torch.stack(delta_per_axis, dim=-1)
        return delta, {"bin_probs": bin_probs}


class GMMCorrector(RedesignedCorrectionNet):
    """M5: diagonal Gaussian (μ, σ). Loss = gmm_nll_loss(mu, logsigma, target)."""

    def __init__(self, dim_cf, hidden=64, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.delta = None
        self.mu_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.logsigma_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, cf, encoder_emb=None):
        h = self._trunk(cf, encoder_emb)
        mu = self.mu_head(h)
        logsigma = self.logsigma_head(h).clamp(min=-6.0, max=0.0)
        return mu, {"logsigma": logsigma}


class WiderShallowCorrector(RedesignedCorrectionNet):
    """M6: depth=1, hidden=256."""

    def __init__(self, dim_cf, hidden=256, dim_encoder=0):
        super().__init__(dim_cf=dim_cf, hidden=hidden, dim_encoder=dim_encoder)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden)])


class IterativeRefinementCorrector(nn.Module):
    """M4 / Phase 5 Z3: n_steps iterative refinement with parameter sharing.

    plan-011 §7.1 self-contained spec. base_corrector dim_cf = dim_cf + 8 (step_idx_emb concat).
    """

    def __init__(self, dim_cf, hidden=64, dim_encoder=0, n_steps=3, per_step_cap=0.003, dim_step_emb=8):
        super().__init__()
        self.n_steps = n_steps
        self.per_step_cap = per_step_cap
        self.dim_step_emb = dim_step_emb
        self.step_idx_emb = nn.Embedding(n_steps, dim_step_emb)
        nn.init.normal_(self.step_idx_emb.weight, mean=0.0, std=0.02)
        self.base_corrector = RedesignedCorrectionNet(
            dim_cf=dim_cf + dim_step_emb, hidden=hidden, dim_encoder=dim_encoder
        )

    def forward(self, cf, encoder_emb=None):
        """Returns (accumulated_delta: (B, 3), aux: dict with per_step_deltas)."""
        B = cf.shape[0]
        device = cf.device
        per_step_deltas = []
        accumulated = torch.zeros(B, 3, device=device, dtype=cf.dtype)
        for t in range(self.n_steps):
            step_idx = torch.full((B,), t, device=device, dtype=torch.long)
            emb = self.step_idx_emb(step_idx)              # (B, dim_step_emb)
            cf_t = torch.cat([cf, emb], dim=-1)
            delta_t, _ = self.base_corrector(cf_t, encoder_emb)
            delta_t = cap_6mm(delta_t, cap=self.per_step_cap)
            per_step_deltas.append(delta_t)
            accumulated = accumulated + delta_t
        return accumulated, {"per_step_deltas": per_step_deltas}


# ──────────────────────────────────────────────────────────────────────────────
# Input encoders (plan-011 §6.1)
# ──────────────────────────────────────────────────────────────────────────────


class TrajectoryStatsFeature(nn.Module):
    """In-B: hand-crafted 20-dim trajectory statistics (no learnable param)."""

    def forward(self, trajectory_x: torch.Tensor) -> torch.Tensor:
        return self.compute(trajectory_x)

    def compute(self, trajectory_x: torch.Tensor) -> torch.Tensor:
        # trajectory_x: (B, T, 3). All inter-frame quantities are step-domain.
        v = trajectory_x[:, 1:] - trajectory_x[:, :-1]          # (B, T-1, 3)
        a = v[:, 1:] - v[:, :-1]                                # (B, T-2, 3)
        s = torch.norm(v, dim=-1)                                # (B, T-1) speed per step
        a_norm = torch.norm(a, dim=-1)                          # (B, T-2)

        # Frenet projections of accel per t (using v_t basis)
        v_unit = v[:, 1:] / (s[:, 1:, None] + EPS)              # (B, T-2, 3) t̂ at t (aligns with a)
        a_par_scalar = (a * v_unit).sum(-1)                      # (B, T-2)
        a_par_vec = a_par_scalar[:, :, None] * v_unit
        a_perp_vec = a - a_par_vec
        a_perp_scalar = torch.norm(a_perp_vec, dim=-1)          # (B, T-2)

        cos_tt = (v[:, :-1] * v[:, 1:]).sum(-1) / (
            (torch.norm(v[:, :-1], dim=-1) * torch.norm(v[:, 1:], dim=-1)) + EPS
        )                                                        # (B, T-2)

        jerk = a[:, 1:] - a[:, :-1]                              # (B, T-3, 3)
        jerk_norm = torch.norm(jerk, dim=-1)                    # (B, T-3)

        # curvature κ = ‖a_perp‖ / ‖v‖²
        curvature = a_perp_scalar / (s[:, 1:] ** 2 + EPS)        # (B, T-2)

        s_safe = s[:, 1:] + EPS                                  # speeds aligned with a (T-2)

        feat = torch.stack([
            s.mean(-1),                              # 0: speed mean
            s.std(-1),                                # 1: speed std
            s[:, -1],                                 # 2: speed last
            s.max(-1).values,                         # 3: speed max
            (a_norm / s_safe).mean(-1),               # 4: acc/v mean
            (a_norm / s_safe).std(-1),                # 5: acc/v std
            (a_norm / s_safe).max(-1).values,         # 6: acc/v max
            (a_par_scalar / s_safe).mean(-1),         # 7: a_par/v mean
            (a_par_scalar / s_safe).std(-1),          # 8: a_par/v std
            (a_perp_scalar / s_safe).mean(-1),        # 9: a_perp/v mean
            (a_perp_scalar / s_safe).std(-1),         # 10: a_perp/v std
            (a_perp_scalar / s_safe).max(-1).values,  # 11: a_perp/v max
            jerk_norm.mean(-1),                       # 12: jerk mean
            jerk_norm.std(-1),                        # 13: jerk std
            jerk_norm.max(-1).values,                 # 14: jerk max
            cos_tt.mean(-1),                          # 15: turn_cos mean
            cos_tt.std(-1),                           # 16: turn_cos std
            cos_tt[:, -1],                            # 17: turn_cos last
            curvature.mean(-1),                       # 18: curvature mean
            curvature.max(-1).values,                 # 19: curvature max
        ], dim=-1)
        return feat                                   # (B, 20)


class FrozenGRUEncoder(nn.Module):
    """In-C: frozen pretrained GRU encoder (R001 baseline-residual-gru reuse).

    decision-note: spec-default — plan-004 selector.AttnGRUCandidateSelector checkpoint *부재* →
      R001_baseline-residual-gru (2-layer GRU(3, 64), 같은 dataset xyz 학습) 를 frozen encoder proxy 로 사용.
      hidden 64-dim (spec 32-dim 과 deviation; decision-note 박제).
      fold-aware ckpt load: val_fold=k 시 fold_k.pt checkpoint 사용 (해당 fold 의 train data 만 본 model).
    """

    def __init__(
        self, ckpt_path: str, input_dim: int = 3, hidden: int = 64, num_layers: int = 2,
        dropout: float = 0.08,
    ):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(
            input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        # load state_dict (gru.* keys; fc.* 는 무시)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gru_sd = {k.replace("gru.", ""): v for k, v in sd.items() if k.startswith("gru.")}
        self.gru.load_state_dict(gru_sd)
        # freeze
        for p in self.gru.parameters():
            p.requires_grad_(False)
        self.gru.eval()

    @torch.no_grad()
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, T, 3) trajectory xyz. Returns (B, hidden) last-timestep last-layer hidden."""
        _, h = self.gru(x_seq)   # h: (num_layers, B, hidden)
        return h[-1]              # last layer hidden


class TrajectoryCNNEncoder(nn.Module):
    """In-D: 1-D CNN over SEQ feature. plan-010 §6.1 reuse — self-contained inline minimal.

    decision-note: spec-default — plan-010 spec 미박제 → 1D CNN 3-layer, kernel=3, channels=[32,64,64].
    """

    def __init__(self, in_channels: int = 9, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, hidden, kernel_size=3, padding=1), nn.GELU(),
        )
        self.hidden = hidden

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, T, C). Returns (B, hidden) — global max-pool over T."""
        # (B, T, C) → (B, C, T) for Conv1d
        x = x_seq.transpose(1, 2)
        h = self.net(x)             # (B, hidden, T)
        return h.max(dim=-1).values  # (B, hidden)


class MultiParseInput(nn.Module):
    """In-F: raw + Savitzky-Golay + EMA — 3 parse single source-of-truth.

    Train: random 1 parse augment per batch. Inference: 3 parse cf 평균.
    Window=5, order=2 (SG), alpha=0.6 (EMA).
    """

    def __init__(self, sg_window: int = 5, sg_order: int = 2, ema_alpha: float = 0.6):
        super().__init__()
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.ema_alpha = ema_alpha

    def parse(self, trajectory_x: torch.Tensor, end_idx: torch.Tensor, mode: str = "train"):
        """Returns either single parse (train) or list of 3 (inference).

        - train mode: torch.randint over {0:raw, 1:SG, 2:EMA} → single (B, T, 3) parse.
        - inference: tuple (raw, SG, EMA) — caller computes cf for each, then averages.
        """
        raw = trajectory_x
        sg = self._savgol(trajectory_x)
        ema = self._ema(trajectory_x)
        if mode == "train":
            k = int(torch.randint(0, 3, (1,)).item())
            return (raw, sg, ema)[k]
        elif mode == "inference":
            return raw, sg, ema
        else:
            raise ValueError(f"mode must be 'train' or 'inference', got {mode}")

    def _savgol(self, x: torch.Tensor) -> torch.Tensor:
        """Simple SG poly-2, window=5 smoothing approximation via Conv1d-like kernel.

        decision-note: spec-default — scipy 의존 회피 위해 5-point poly-2 SG coef inline 박제.
        coef = [-3, 12, 17, 12, -3] / 35 (well-known SG window=5, order=2, smoothing).
        """
        kernel = torch.tensor([-3.0, 12.0, 17.0, 12.0, -3.0], device=x.device, dtype=x.dtype) / 35.0
        # x: (B, T, 3). Conv over T per dim independently.
        pad = 2
        x_padded = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")  # (B, 3, T+4)
        out = F.conv1d(x_padded, kernel.view(1, 1, 5).expand(3, 1, 5), groups=3)
        return out.transpose(1, 2)  # (B, T, 3)

    def _ema(self, x: torch.Tensor) -> torch.Tensor:
        """EMA smoothing with alpha=0.6 (recursive in time dim)."""
        # x: (B, T, 3). EMA forward pass.
        out = torch.empty_like(x)
        out[:, 0] = x[:, 0]
        a = self.ema_alpha
        for t in range(1, x.shape[1]):
            out[:, t] = a * x[:, t] + (1.0 - a) * out[:, t - 1]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Formula variants (plan-011 §8.1)
# ──────────────────────────────────────────────────────────────────────────────


class PerSampleMLPFormula(nn.Module):
    """F3: per-sample (par, perp) regression. in_dim=12 (plan-011 §8.1 박제)."""

    def __init__(self, in_dim: int = 12, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        with torch.no_grad():
            self.net[-1].bias[0] = 1.20
            self.net[-1].bias[1] = -0.20

    def forward(self, ctx_features: torch.Tensor) -> torch.Tensor:
        return self.net(ctx_features)  # (B, 2) → par, perp per sample


class LearnableSingleCandidate(nn.Module):
    """F4: 6-coef learnable candidate — selector.make_candidates canonical frenet 식과 parity.

    init_coef = (k_d1, k_d2, k_par, k_perp, k_jerk, k_time).
    F0 anchor (CANDIDATES[17] = frenet_par120_perp_neg020) numerically exact:
      (d1=1.98, d2=0.0, par=1.20, perp=-0.20, jerk=0.0, time_scale=1.0)

    Canonical formula (selector.make_candidates 그대로):
      v_scale   = (horizon / 2) * time_scale
      acc_scale = (horizon / 2)² * time_scale²
      cand = p0
             + d1   * v_scale   * v_last        (last velocity vector)
             + d2   * v_scale   * v_prev        (prev velocity vector)
             + par  * acc_scale * acc_par_vec   (acc · t̂) t̂
             + perp * acc_scale * acc_perp_vec  (acc - acc_par_vec)
             + jerk * acc_scale * jerk_vec      (acc - prev_acc)
    """

    def __init__(self, init_coef=(1.98, 0.0, 1.20, -0.20, 0.0, 1.0)):
        super().__init__()
        self.coef = nn.Parameter(torch.tensor(init_coef, dtype=torch.float32))

    def forward(
        self,
        p0: torch.Tensor,
        v_last: torch.Tensor,
        v_prev: torch.Tensor,
        acc_par_vec: torch.Tensor,
        acc_perp_vec: torch.Tensor,
        jerk_vec: torch.Tensor,
        horizon: float = 2.0,
        coef: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate single candidate world position (selector parity).

        Args:
          p0:           (B, 3) last observed position.
          v_last:       (B, 3) = x_T − x_{T−1}.
          v_prev:       (B, 3) = x_{T−1} − x_{T−2}.
          acc_par_vec:  (B, 3) = (acc · t̂) t̂ where acc=v_last−v_prev, t̂=v_last/‖v_last‖.
          acc_perp_vec: (B, 3) = acc − acc_par_vec.
          jerk_vec:     (B, 3) = acc − prev_acc where prev_acc = v_prev − (x_{T−2} − x_{T−3}).
          horizon:      int (default 2).
          coef:         (B, 6) per-sample override (F3). None → broadcast self.coef.
        """
        if coef is None:
            c = self.coef.unsqueeze(0).expand(v_last.shape[0], -1)
        else:
            c = coef
        half_h = horizon / 2.0
        v_scale = half_h * c[..., 5:6]                    # (B, 1)
        acc_scale = (half_h ** 2) * (c[..., 5:6] ** 2)    # (B, 1)
        return (
            p0
            + c[..., 0:1] * v_scale * v_last
            + c[..., 1:2] * v_scale * v_prev
            + c[..., 2:3] * acc_scale * acc_par_vec
            + c[..., 3:4] * acc_scale * acc_perp_vec
            + c[..., 4:5] * acc_scale * jerk_vec
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test (run as module)
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    B, T = 4, 10
    cf = torch.randn(B, 32)
    enc = torch.randn(B, 32)
    traj = torch.randn(B, T, 3)
    end_idx = torch.tensor([T - 1] * B)

    print("[smoke] Z1 base:")
    m0 = RedesignedCorrectionNet(dim_cf=32, hidden=64, dim_encoder=32)
    d, aux = m0(cf, enc)
    print(f"  delta {tuple(d.shape)}, aux={list(aux.keys())}")

    for name, cls in [
        ("M1 GateHead", GateHeadCorrector),
        ("M2 SplitHead", SplitHeadCorrector),
        ("M3 BinClassifier", BinClassifierCorrector),
        ("M5 GMM", GMMCorrector),
        ("M6 WiderShallow", WiderShallowCorrector),
    ]:
        m = cls(dim_cf=32, dim_encoder=32) if cls is not WiderShallowCorrector else cls(dim_cf=32, dim_encoder=32, hidden=256)
        d, aux = m(cf, enc)
        print(f"  {name}: delta {tuple(d.shape)}, aux keys={list(aux.keys())}")

    print("[smoke] M4 Iterative:")
    m4 = IterativeRefinementCorrector(dim_cf=32, dim_encoder=32, n_steps=3)
    d, aux = m4(cf, enc)
    print(f"  delta {tuple(d.shape)}, n_steps={len(aux['per_step_deltas'])}")

    print("[smoke] Loss:")
    pred = torch.randn(B, 3, requires_grad=True)
    target = torch.randn(B, 3)
    print(f"  huber: {huber_loss(pred, target).mean():.4f}")
    print(f"  hit_aware_hinge: {hit_aware_hinge(pred, target).mean():.6f}")
    R, valid = build_frenet_basis(traj, end_idx)
    print(f"  frenet basis R={tuple(R.shape)}, valid={valid.tolist()}")

    print("[smoke] Input:")
    stats = TrajectoryStatsFeature()(traj)
    print(f"  TrajectoryStats: {tuple(stats.shape)}")
    cnn = TrajectoryCNNEncoder()(torch.randn(B, T, 9))
    print(f"  CNN: {tuple(cnn.shape)}")
    mpi = MultiParseInput()
    parsed = mpi.parse(traj, end_idx, mode="train")
    print(f"  MultiParse(train) one parse: {tuple(parsed.shape)}")
    p_raw, p_sg, p_ema = mpi.parse(traj, end_idx, mode="inference")
    print(f"  MultiParse(inference) 3 parse: each {tuple(p_raw.shape)}")

    print("[smoke] Formula:")
    f3 = PerSampleMLPFormula(in_dim=12)
    par_perp = f3(torch.randn(B, 12))
    print(f"  F3 (par, perp): {tuple(par_perp.shape)}")
    f4 = LearnableSingleCandidate()
    v_last_t = torch.randn(B, 3); v_prev_t = torch.randn(B, 3)
    acc_t = v_last_t - v_prev_t
    tangent_t = v_last_t / (torch.norm(v_last_t, dim=-1, keepdim=True) + 1e-6)
    acc_par_t = (acc_t * tangent_t).sum(-1, keepdim=True) * tangent_t
    acc_perp_t = acc_t - acc_par_t
    jerk_t = torch.randn(B, 3)
    cand = f4(
        p0=torch.randn(B, 3),
        v_last=v_last_t, v_prev=v_prev_t,
        acc_par_vec=acc_par_t, acc_perp_vec=acc_perp_t, jerk_vec=jerk_t,
    )
    print(f"  F4 cand_pos: {tuple(cand.shape)}")

    print("\n✓ smoke test passed")
