"""plan-013 c2 — integrated v3 wrapper: plan-004 framework + 3 lever stacking.

5 컴포넌트 (plans/plan-013-plan004-framework-3lever-stacking.md §4.1):
1. InICEmbedder — frozen R001 GRU layer-0 hidden encoder (plan-011 v1.2 reuse)
2. InICCorrectorWrapper — TinyCorrectionNet 의 cf input 에 In/IC embedding broadcast concat
3. Step4CoeffMLP — per-sample 8-vars coeff MLP (plan-007 reuse, F0_only / 27ext modes)
4. load_25_cand_set — plan-008 G1 25-candidate descriptor loader
5. run_integrated_v3 — Phase 1/2/3 dispatcher (P001 selector scores reuse + 자체 corrector 학습)

decision-note: plan-004 framework 의 selector 는 P001 산출 `oof_selector_scores.npz` 로 frozen reuse,
corrector 만 본 wrapper 에서 재학습. 3 lever 는 corrector training 단계에서 hook 주입.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from src.pb_0_6822 import boundary as base_bnd
from src.pb_0_6822 import selector as base_sel


# ── 컴포넌트 1: In/IC frozen GRU encoder embedding (plan-011 v1.2 reuse) ──

class InICEmbedder(nn.Module):
    """plan-011 v1.2 In/IC: frozen R001 2-layer GRU(3, 64) encoder 의 layer 0 (= plan-011 명명 'layer 1') 추출.

    Input: (B, T, 3) world coords sequence
    Output: (B, T, 64) hidden states from GRU layer 0
    state_dict loaded from runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt.
    All parameters frozen (requires_grad=False).
    Invariant: state_dict diff > 0 시 frozen_gru_drift severe.
    """

    # plan spec 박제 path: runs/baseline/R001_baseline-residual-gru/checkpoint_fold0.pt
    # 실제 산출 path: runs/baseline/R001_baseline-residual-gru/ckpt/fold0.pt
    # decision-note: 자율 결정 — 실제 산출 path 채택. plan-013 §10 참조 path 박제 정정은 결과 results.md 에 박제.
    DEFAULT_CKPT = "runs/baseline/R001_baseline-residual-gru/ckpt/fold0.pt"

    def __init__(self, ckpt_path: str = DEFAULT_CKPT, strict_load: bool = True):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=64, num_layers=1, batch_first=True)
        ckpt_path_p = Path(ckpt_path)
        if not ckpt_path_p.exists():
            if strict_load:
                raise FileNotFoundError(f"R001 checkpoint not found: {ckpt_path_p}")
            # smoke test / preflight 환경에서는 weight 미로드 path 허용 (frozen 상태만 보장)
            self._loaded = False
        else:
            state = torch.load(ckpt_path_p, map_location="cpu", weights_only=True)
            # R001 actual structure: flat dict with "gru.weight_ih_l0" style keys (no nesting).
            # plan-spec assumed nested state["gru"][...] — adapt.
            if isinstance(state, dict) and "gru" in state and isinstance(state["gru"], dict):
                gru_state = state["gru"]
                key_prefix = ""
            else:
                gru_state = state
                key_prefix = "gru."
            layer0_state = {
                "weight_ih_l0": gru_state[f"{key_prefix}weight_ih_l0"],
                "weight_hh_l0": gru_state[f"{key_prefix}weight_hh_l0"],
                "bias_ih_l0":   gru_state[f"{key_prefix}bias_ih_l0"],
                "bias_hh_l0":   gru_state[f"{key_prefix}bias_hh_l0"],
            }
            self.gru.load_state_dict(layer0_state)
            self._loaded = True
        for p in self.parameters():
            p.requires_grad = False
        # frozen 박제: 초기 state_dict hash (frozen_gru_drift severe check 시 비교용)
        self._init_state_hash = self._compute_state_hash()

    def _compute_state_hash(self) -> int:
        return hash(tuple(p.detach().cpu().numpy().tobytes() for p in self.parameters()))

    def forward(self, trajectory_x: torch.Tensor) -> torch.Tensor:
        """trajectory_x: (B, T, 3) → returns (B, T, 64)."""
        output, _h_n = self.gru(trajectory_x)
        return output


# ── 컴포넌트 2: corrector 에 In/IC embedding 시계열 input 주입 ──

class InICCorrectorWrapper(nn.Module):
    """plan-004 TinyCorrectionNet 의 cf input 에 In/IC GRU embedding last-step concat.

    Original cf: (B, K=27, 32). Augmented cf: (B, K, 96) = (B, K, 32+64).
    base_corrector signature 는 plan-004 의 (dim, hidden) — wrapper 는 dim=dim_cf 로 forward.

    train loop 매 batch 직전 `wrapper._cached_trajectory = batch_traj` 를 set 해야 함
    (= dataloader 가 (cf, traj, y) 함께 반환).
    """

    def __init__(
        self,
        ckpt_path: str = InICEmbedder.DEFAULT_CKPT,
        strict_load: bool = True,
        dim_cf: int = 96,
        hidden: int = 64,
    ):
        super().__init__()
        self.embedder = InICEmbedder(ckpt_path=ckpt_path, strict_load=strict_load)
        # plan-004 TinyCorrectionNet signature = (dim, hidden), return = (delta, env)
        self.base_corrector = base_bnd.TinyCorrectionNet(dim=dim_cf, hidden=hidden)
        self._cached_trajectory: torch.Tensor | None = None

    def _impl_forward(
        self, cf_base: torch.Tensor, trajectory_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cf_base: (B, K, 32), trajectory_x: (B, T, 3) → returns (delta (B, K, 3), env (B, K, 4))."""
        emb = self.embedder(trajectory_x)                          # (B, T, 64)
        emb_last = emb[:, -1, :]                                   # (B, 64)
        emb_broadcast = emb_last[:, None, :].expand(-1, cf_base.shape[1], -1)  # (B, K, 64)
        cf_aug = torch.cat([cf_base, emb_broadcast], dim=-1)       # (B, K, 96)
        return self.base_corrector(cf_aug)

    def forward(self, cf_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """1-arg adapter, plan-004 train loop 호환. trajectory 는 _cached_trajectory 에서 read."""
        if self._cached_trajectory is None:
            raise RuntimeError(
                "InICCorrectorWrapper.forward: _cached_trajectory 미설정. "
                "train loop 가 매 batch 직전 `wrapper._cached_trajectory = traj` 를 set 해야 함."
            )
        return self._impl_forward(cf_base, self._cached_trajectory)


# ── 컴포넌트 3: Step 4 per-sample MLP coeff (plan-007 reuse) ──

# plan-007 박제 8 vars best basis (CMA-ES + greedy ablation)
STEP4_BEST_BASIS = (
    "d1", "acc_par", "acc_perp", "d2", "jerk",
    "ts_term", "speed_slope_d1", "rotation_term",
)


class Step4CoeffMLP(nn.Module):
    """plan-007 Step 4 per-sample 8 vars MLP coeff.

    Modes:
    - "F0_only" (Phase 2.E1): F0 (frenet_par120_perp_neg020) coeff 만 sample-wise 조정
    - "27ext" (Phase 2.E2): 27 후보 공통 MLP + candidate_idx one-hot input
    """

    def __init__(
        self,
        mode: Literal["F0_only", "27ext"],
        n_vars: int = 8,
        feat_dim: int = 13,
        hidden: int = 32,
        n_candidates: int = 27,
    ):
        super().__init__()
        self.mode = mode
        self.n_vars = n_vars
        self.feat_dim = feat_dim
        self.n_candidates = n_candidates
        self.basis_names = list(STEP4_BEST_BASIS)
        if mode == "F0_only":
            in_dim = feat_dim
        elif mode == "27ext":
            in_dim = feat_dim + n_candidates
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_vars),
        )

    def forward(
        self, last_step_feat: torch.Tensor, candidate_idx: int | None = None
    ) -> torch.Tensor:
        """last_step_feat: (B, feat_dim).

        candidate_idx: Phase 2.E2 27ext 모드에서 단일 int (batch 내 sample 모두 같은 candidate).
            train loop 의 caller 가 27 후보 각각에 대해 27 회 forward 호출 (vectorized broadcast 미사용).
        returns: (B, n_vars=8)
        """
        if self.mode == "F0_only":
            return self.mlp(last_step_feat)
        if candidate_idx is None:
            raise ValueError("27ext mode requires candidate_idx")
        cand_onehot = torch.zeros(
            last_step_feat.shape[0], self.n_candidates, device=last_step_feat.device,
            dtype=last_step_feat.dtype,
        )
        cand_onehot[:, candidate_idx] = 1.0
        return self.mlp(torch.cat([last_step_feat, cand_onehot], dim=-1))


# ── 컴포넌트 4: 25 cand redesign loader (plan-008 G1 reuse) ──

def load_25_cand_set(
    plan_008_g1_dir: str = "runs/baseline/G001_candidate-redefine",
) -> list[dict]:
    """plan-008 G1 산출의 25 candidate *formula descriptor* list 로드.

    Returns: len=25 list of dict (plan-004 selector.make_candidate_features 호환 spec).
    Invariant: 25 후보 좌표가 plan-008 G1 박제와 drift > 1e-6 시 cand_set_25_drift severe.
    """
    g1_dir = Path(plan_008_g1_dir)
    path_json = g1_dir / "cand_set.json"
    path_npy = g1_dir / "cand_set.npy"
    if path_json.exists():
        with open(path_json) as f:
            cand_list = json.load(f)
    elif path_npy.exists():
        cand_list = list(np.load(path_npy, allow_pickle=True))
    else:
        raise FileNotFoundError(
            f"plan-008 G1 cand_set 미존재: tried {path_json} and {path_npy}"
        )
    if len(cand_list) != 25:
        raise ValueError(f"expected 25 candidates, got {len(cand_list)}")
    return cand_list


# ── 컴포넌트 5: integrated entry (Phase 1/2/3 dispatcher) ──

def run_integrated_v3(
    config: dict,
    fold: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_ids: np.ndarray,
    test_x: np.ndarray | None = None,
    test_sample_ids: np.ndarray | None = None,
) -> dict:
    """Integrated training + inference entry for Phase 1/2/3 sub-exp.

    spec: §4.1 docstring. 본 구현은 plan-004 framework 의 selector 부분을 P001 산출
    `oof_selector_scores.npz` 로 frozen reuse, corrector 만 본 wrapper 에서 재학습.

    config keys:
        - "use_in_ic": bool
        - "use_step4": Literal["off", "F0_only", "27ext"]
        - "use_25_cand": bool
        - "epochs": int (default 50)
        - "patience": int (default 5)
        - "batch_size": int (default 256)
        - "lr": float (default 3e-4)
        - "seed": int (default 42)
        - "p001_dir": str (default "runs/baseline/P001_pb-0-6822-fullrun")
        - "device": str (default auto-detect)

    Returns: §4.1 spec 박제 dict.

    NOTE: c2 본 commit 은 *구조 스켈레톤* — 실제 5-fold training 의 full wiring 은
    c4 (phase1_baseline) 에서 본 함수를 호출하며 검증. 본 함수 자체는 다음 단계
    에서 incrementally 확장.
    """
    # c4 implementation: 간소화 standalone residual corrector. P001 frozen selector scores 위에
    # InICCorrectorWrapper (use_in_ic=True) 또는 base TinyCorrectionNet (use_in_ic=False) 학습.
    # plan-004 의 regime/env/pretrain/finetune complexity 제외 — Δ measurement 위한 minimal pipeline.
    return _run_phase1_simplified(
        config=config,
        fold=fold,
        train_x=train_x,
        train_y=train_y,
        sample_ids=sample_ids,
        test_x=test_x,
        test_sample_ids=test_sample_ids,
    )


def _run_phase1_simplified(
    config: dict,
    fold: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_ids: np.ndarray,
    test_x: np.ndarray | None = None,
    test_sample_ids: np.ndarray | None = None,
) -> dict:
    """Phase 1 baseline simplified training (decision-note: c4 minimal pipeline).

    Pipeline:
        1. fold split: stable_fold_id(sid, 5) == fold → val_idx; else train_idx
        2. base predictions: P001 OOF selector scores 의 argmax-soft-weighted candidate average
        3. residual target: true_y - base_pred (per sample)
        4. corrector (InICCorrectorWrapper or TinyCorrectionNet) 학습:
           input = (cf_base, trajectory_x); output = residual delta
        5. corrected_pred = base_pred + corrector_delta (clipped)
        6. val_scores = P001 OOF selector scores 의 val_idx subset (frozen reuse)
    """
    if config.get("use_25_cand", False):
        raise NotImplementedError("Phase 2.E3 25 cand swap: c4 scope 외 (cand_25_infra MISS)")
    if config.get("use_step4", "off") != "off":
        raise NotImplementedError(f"Phase 2 Step 4 ({config['use_step4']}): c4 scope 외")

    device = torch.device(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    seed = int(config.get("seed", 42))
    epochs = int(config.get("epochs", 50))
    patience = int(config.get("patience", 5))
    batch_size = int(config.get("batch_size", 256))
    lr = float(config.get("lr", 3e-4))
    cap = float(config.get("cap", 0.006))
    p001_dir = config.get("p001_dir", "runs/baseline/P001_pb-0-6822-fullrun")

    base_sel.set_torch_seed(seed)

    # 1. fold split (stable_fold_id, §3.1 / §4.1 step 1)
    fold_ids = np.array([base_sel.stable_fold_id(sid, 5) for sid in sample_ids])
    val_mask = fold_ids == fold
    train_mask = ~val_mask
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(train_mask)[0]

    # 2. base predictions per sample (P001 frozen reuse)
    p001 = load_p001_selector_scores(p001_dir, which="oof")
    # oof_selector_scores.npz keys: typically "ens_scores" (N, 27) or model-specific
    score_key = next((k for k in ("ens_scores", "scores", "attn_gru_scores") if k in p001), None)
    if score_key is None:
        raise KeyError(f"P001 score bank keys: {list(p001.keys())} — no recognizable score key")
    selector_scores = np.asarray(p001[score_key], dtype=np.float32)  # (N, 27)
    if selector_scores.shape[0] != len(train_x):
        raise ValueError(
            f"P001 score shape {selector_scores.shape} mismatch train_x {train_x.shape[0]}"
        )

    end_idx = train_x.shape[1] - 1
    horizon = 2
    cands_all = base_sel.make_candidates(train_x, end_idx, horizon=horizon)  # (N, 27, 3)
    cf_base_all = base_sel.make_candidate_features(train_x, end_idx, cands_all, horizon=horizon)  # (N, 27, 32)

    # soft-weighted base prediction
    soft = torch.softmax(torch.from_numpy(selector_scores), dim=-1).numpy()
    base_pred_all = np.sum(soft[:, :, None] * cands_all, axis=1)  # (N, 3)
    residual_target = train_y - base_pred_all  # (N, 3)

    # 3. corrector instantiate
    if config.get("use_in_ic", True):
        wrapper = InICCorrectorWrapper(dim_cf=96, hidden=64).to(device)
    else:
        # Phase 1 default = use_in_ic True, fallback base corrector path
        wrapper = base_bnd.TinyCorrectionNet(dim=32, hidden=64).to(device)

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr)

    # 4. train loop
    cf_tensor = torch.from_numpy(cf_base_all).to(device)
    traj_tensor = torch.from_numpy(train_x).to(device)
    target_tensor = torch.from_numpy(residual_target).to(device)
    cands_tensor = torch.from_numpy(cands_all).to(device)
    soft_tensor = torch.softmax(torch.from_numpy(selector_scores), dim=-1).to(device)

    n_train = len(train_idx)
    best_val_hit = -1.0
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        wrapper.train()
        # shuffle train indices
        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_idx_t = torch.from_numpy(batch_idx).to(device)
            cf_b = cf_tensor[batch_idx_t]                 # (B, K, 32)
            traj_b = traj_tensor[batch_idx_t]             # (B, T, 3)
            target_b = target_tensor[batch_idx_t]         # (B, 3)
            soft_b = soft_tensor[batch_idx_t]             # (B, K)

            if isinstance(wrapper, InICCorrectorWrapper):
                wrapper._cached_trajectory = traj_b
                delta_per_cand, _env = wrapper(cf_b)       # (B, K, 3), (B, K, 4)
            else:
                # base TinyCorrectionNet expects (B*K, 32) flat
                cf_flat = cf_b.reshape(-1, cf_b.shape[-1])
                delta_flat, _env = wrapper(cf_flat)
                delta_per_cand = delta_flat.reshape(cf_b.shape[0], cf_b.shape[1], 3)
            # apply cap
            delta_per_cand = torch.tanh(delta_per_cand / cap) * cap
            # aggregate: residual = soft-weighted delta over candidates
            delta_agg = (soft_b[:, :, None] * delta_per_cand).sum(dim=1)  # (B, 3)
            loss = torch.nn.functional.huber_loss(delta_agg, target_b, delta=0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # 5. eval on val
        wrapper.eval()
        with torch.no_grad():
            val_idx_t = torch.from_numpy(val_idx).to(device)
            cf_v = cf_tensor[val_idx_t]
            traj_v = traj_tensor[val_idx_t]
            soft_v = soft_tensor[val_idx_t]
            if isinstance(wrapper, InICCorrectorWrapper):
                wrapper._cached_trajectory = traj_v
                delta_v, _env = wrapper(cf_v)
            else:
                cf_flat = cf_v.reshape(-1, cf_v.shape[-1])
                delta_flat, _env = wrapper(cf_flat)
                delta_v = delta_flat.reshape(cf_v.shape[0], cf_v.shape[1], 3)
            delta_v = torch.tanh(delta_v / cap) * cap
            delta_v_agg = (soft_v[:, :, None] * delta_v).sum(dim=1).cpu().numpy()  # (N_val, 3)
            corrected_v = base_pred_all[val_idx] + delta_v_agg
            val_y = train_y[val_idx]
            val_hit = float(np.mean(np.linalg.norm(corrected_v - val_y, axis=1) <= 0.01))

        if val_hit > best_val_hit:
            best_val_hit = val_hit
            best_state = {k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    # restore best state
    if best_state is not None:
        wrapper.load_state_dict(best_state)

    # 6. final inference
    wrapper.eval()
    with torch.no_grad():
        val_idx_t = torch.from_numpy(val_idx).to(device)
        cf_v = cf_tensor[val_idx_t]
        traj_v = traj_tensor[val_idx_t]
        soft_v = soft_tensor[val_idx_t]
        if isinstance(wrapper, InICCorrectorWrapper):
            wrapper._cached_trajectory = traj_v
            delta_v, _env = wrapper(cf_v)
        else:
            cf_flat = cf_v.reshape(-1, cf_v.shape[-1])
            delta_flat, _env = wrapper(cf_flat)
            delta_v = delta_flat.reshape(cf_v.shape[0], cf_v.shape[1], 3)
        delta_v = torch.tanh(delta_v / cap) * cap
        delta_v_agg = (soft_v[:, :, None] * delta_v).sum(dim=1).cpu().numpy()
        val_preds = base_pred_all[val_idx] + delta_v_agg

    # in_ic frozen check
    if isinstance(wrapper, InICCorrectorWrapper):
        in_ic_init_hash = wrapper.embedder._init_state_hash
        in_ic_final_hash = wrapper.embedder._compute_state_hash()
    else:
        in_ic_init_hash = 0
        in_ic_final_hash = 0

    # 7. test inference (optional)
    test_preds = None
    test_scores_out = None
    if test_x is not None:
        p001_test = load_p001_selector_scores(p001_dir, which="test")
        test_score_key = next(
            (k for k in ("ens_scores", "scores", "attn_gru_scores") if k in p001_test), None
        )
        if test_score_key is None:
            raise KeyError(f"P001 test score bank keys: {list(p001_test.keys())}")
        test_selector_scores = np.asarray(p001_test[test_score_key], dtype=np.float32)
        test_scores_out = test_selector_scores
        if test_selector_scores.shape[0] != len(test_x):
            raise ValueError(
                f"P001 test score {test_selector_scores.shape} vs test_x {test_x.shape[0]}"
            )
        test_end_idx = test_x.shape[1] - 1
        test_cands = base_sel.make_candidates(test_x, test_end_idx, horizon=horizon)
        test_cf_base = base_sel.make_candidate_features(test_x, test_end_idx, test_cands, horizon=horizon)
        test_soft = torch.softmax(torch.from_numpy(test_selector_scores), dim=-1).numpy()
        test_base_pred = np.sum(test_soft[:, :, None] * test_cands, axis=1)

        with torch.no_grad():
            cf_t = torch.from_numpy(test_cf_base).to(device)
            traj_t = torch.from_numpy(test_x).to(device)
            soft_t = torch.from_numpy(test_soft).to(device)
            if isinstance(wrapper, InICCorrectorWrapper):
                wrapper._cached_trajectory = traj_t
                delta_t, _env = wrapper(cf_t)
            else:
                cf_flat = cf_t.reshape(-1, cf_t.shape[-1])
                delta_flat, _env = wrapper(cf_flat)
                delta_t = delta_flat.reshape(cf_t.shape[0], cf_t.shape[1], 3)
            delta_t = torch.tanh(delta_t / cap) * cap
            delta_t_agg = (soft_t[:, :, None] * delta_t).sum(dim=1).cpu().numpy()
            test_preds = (test_base_pred + delta_t_agg).astype(np.float32)

    return {
        "val_preds": val_preds.astype(np.float32),
        "val_scores": selector_scores[val_idx],
        "test_preds": test_preds,
        "test_scores": test_scores_out,
        "oof_metric": {"hit": best_val_hit},
        "corrector_state": best_state,
        "in_ic_init_hash": in_ic_init_hash,
        "in_ic_final_hash": in_ic_final_hash,
        "val_idx": val_idx,
        "n_epochs_trained": epoch + 1,
    }


# ── Helpers (frozen invariant check, score bank IO) ──

def check_in_ic_frozen(wrapper: InICCorrectorWrapper) -> bool:
    """frozen_gru_drift severe check: 초기 hash 와 현재 hash 비교."""
    current = wrapper.embedder._compute_state_hash()
    return current == wrapper.embedder._init_state_hash


def load_p001_selector_scores(
    p001_dir: str = "runs/baseline/P001_pb-0-6822-fullrun",
    which: Literal["oof", "test"] = "oof",
) -> dict:
    """P001 산출 selector scores (frozen reuse) 로드.

    Returns: dict with keys "scores" (np.ndarray (N, 27)) + 기타 meta.
    """
    p = Path(p001_dir)
    fname = "oof_selector_scores.npz" if which == "oof" else "test_selector_scores.npz"
    bank = p / fname
    if not bank.exists():
        raise FileNotFoundError(f"P001 selector score bank 미존재: {bank}")
    data = dict(np.load(bank, allow_pickle=True))
    return data
