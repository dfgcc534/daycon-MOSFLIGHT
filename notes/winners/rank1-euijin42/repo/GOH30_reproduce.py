"""GOH30_reproduce — rank1 euijin42 노트북 1:1 재현 (private LB 0.7035).

원본: DACON 코드공유 14013 (notes/winners/rank1-euijin42/notebook_dump.md cell 1~25 inline).
변경: 노트북 셀 → 단일 .py module, 환경변수 토글 (DATA_DIR, MODELS_DIR, N_EACH).
구조:
  1. setup / constants
  2. feature engineering (yaw rotation, seq+scalar)
  3. preprocessing (norm_stats + cache 50K)
  4. model defs (AttnGRU, ODEModel, HyperPhysics_xy2)
  5. losses + training (combined_loss, train_cache_seed, train_h_seed)
  6. training execution (FROM_SCRATCH=True → 30 models)
  7. predict + 30-model blend → submission CSV
"""
# ══════════════════════════════════════════════════════════════════════════════
# cell 1 · setup
# ══════════════════════════════════════════════════════════════════════════════
import os
import glob
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


DATA_DIR     = os.environ.get('DATA_DIR', './data')   # 우리 repo: ./data
FROM_SCRATCH = os.environ.get('FROM_SCRATCH', 'True').lower() == 'true'
MODELS_DIR   = os.environ.get('MODELS_DIR', './models_goh30')
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

# constants — 0.7035 재현 위해 고정
DT = 0.04
PRED_DT = 0.08
CLIP_THR = 1.33
SPEED_BINS = [0.0, 0.3, 0.6, 0.9, 1.2, np.inf]
SIGMA = 0.02
RHIT_TAU = 0.0015
RHIT_W = 2.0
HW = 0.5
GW = 0.5
FLIP_PROB = 0.5
NOISE_STD = 0.02
Y_FLIP = [1, 4, 7, 10]
INTERIOR_E = [5, 6, 7, 8]
N_EACH = int(os.environ.get('N_EACH', '10'))   # 시드 수 (smoke 시 override)
GRU_ODE_EPOCHS = int(os.environ.get('GRU_ODE_EPOCHS', '55'))
H_EPOCHS = int(os.environ.get('H_EPOCHS', '12'))
EMA_DECAY = 0.9


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ══════════════════════════════════════════════════════════════════════════════
# cell 2 · feature engineering
# ══════════════════════════════════════════════════════════════════════════════
def load_sample(path):
    df = pd.read_csv(path)
    return df[['x', 'y', 'z']].to_numpy(dtype=np.float32)


def yaw_rotation_matrix(velocity):
    vx, vy = float(velocity[0]), float(velocity[1])
    speed_xy = np.sqrt(vx ** 2 + vy ** 2)
    if speed_xy < 1e-6:
        return np.eye(3, dtype=np.float32)
    cos_yaw, sin_yaw = vx / speed_xy, vy / speed_xy
    return np.array([[cos_yaw, sin_yaw, 0.0], [-sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def extract_seq_features(smoothed_pos, smoothed_vel, rot):
    last_pos = smoothed_pos[-1]
    rel_pos = (smoothed_pos - last_pos) @ rot.T
    vel_rot = smoothed_vel @ rot.T
    accel = np.zeros_like(vel_rot)
    accel[1:-1] = (vel_rot[2:] - vel_rot[:-2]) / (2 * DT); accel[0] = accel[1]; accel[-1] = accel[-2]
    jerk = np.zeros_like(accel)
    jerk[1:-1] = (accel[2:] - accel[:-2]) / (2 * DT); jerk[0] = jerk[1]; jerk[-1] = jerk[-2]
    speed = np.linalg.norm(vel_rot, axis=1, keepdims=True)
    v_norm = vel_rot / (speed + 1e-12)
    cos_sim = (v_norm[:-1] * v_norm[1:]).sum(axis=1)
    angular_vel = np.concatenate([[cos_sim[0]], cos_sim])
    features = np.concatenate([rel_pos, vel_rot, accel, jerk, angular_vel[:, None]], axis=1)
    return features.astype(np.float32)


def extract_scalar_features(smoothed_pos, smoothed_vel):
    speeds = np.linalg.norm(smoothed_vel, axis=1); last_speed = float(speeds[-1])
    vel_diff = np.diff(smoothed_vel, axis=0) / DT; accel_mag = np.linalg.norm(vel_diff, axis=1)
    last_accel = float(accel_mag[-1]); mean_accel = float(accel_mag.mean())
    t = np.arange(len(smoothed_pos), dtype=np.float32); r2_list = []
    for dim in range(3):
        y = smoothed_pos[:, dim]; coeffs = np.polyfit(t, y, 1); y_pred = np.polyval(coeffs, t)
        ss_res = float(np.sum((y - y_pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_list.append(1.0 - ss_res / (ss_tot + 1e-10))
    linearity = float(np.mean(r2_list)); clip_flag = float(last_speed > CLIP_THR)
    v_norm = smoothed_vel / (np.linalg.norm(smoothed_vel, axis=1, keepdims=True) + 1e-12)
    cos_sim_all = (v_norm[:-1] * v_norm[1:]).sum(axis=1)
    dir_consistency = float(cos_sim_all.mean()); delta_speed = float(speeds[-1] - speeds[-2])
    last_dir_change = float(cos_sim_all[-1])
    last_vel_norm = v_norm[-1]; last_accel_vec = vel_diff[-1]
    tangential = np.dot(last_accel_vec, last_vel_norm) * last_vel_norm
    last_normal_accel = float(np.linalg.norm(last_accel_vec - tangential))
    speed_bin = np.zeros(5, dtype=np.float32)
    for k in range(5):
        if SPEED_BINS[k] <= last_speed < SPEED_BINS[k + 1]: speed_bin[k] = 1.0; break
    scalar = np.array([last_speed, last_accel, mean_accel, linearity, clip_flag,
                       dir_consistency, delta_speed, last_dir_change, last_normal_accel], dtype=np.float32)
    return np.concatenate([scalar, speed_bin])


def window_features(W):
    W = W.astype(np.float64); vel = np.gradient(W, DT, axis=0); rot = yaw_rotation_matrix(vel[-1])
    seq = extract_seq_features(W, vel, rot); b14 = extract_scalar_features(W, vel)
    sp = np.linalg.norm(vel, axis=1); steps = np.linalg.norm(np.diff(W, axis=0), axis=1); L = len(W)
    path = float(steps.sum()); net = float(np.linalg.norm(W[-1] - W[0])); straight = net / (path + 1e-8)
    t = np.arange(float(L))
    noise = float(np.mean([(W[:, d] - np.polyval(np.polyfit(t, W[:, d], 2), t)).std() for d in range(3)]))
    k = min(4, L); acc_trend = float(np.polyfit(np.arange(float(k)), sp[-k:], 1)[0])
    sc = np.concatenate([b14, [float(sp.max()), float(sp.std()), float(sp[-3:].mean()), float(sp[-5:].mean()),
                               path, straight, noise, acc_trend]]).astype(np.float32)
    base = (W[-1] + 2.0 * (W[-1] - W[-2])).astype(np.float32)
    return seq.astype(np.float32), sc, rot.astype(np.float32), base, W[-1].astype(np.float32)


def build_features_clean(X):
    return window_features(X)


def normalize(seq, scalar, stats):
    seq_n = ((seq - stats['seq_mean']) / stats['seq_std']).astype(np.float32)
    scal_n = ((scalar - stats['scalar_mean']) / stats['scalar_std']).astype(np.float32)
    return seq_n, scal_n


# ══════════════════════════════════════════════════════════════════════════════
# cell 3 · preprocessing — invoked from main(), not at import
# ══════════════════════════════════════════════════════════════════════════════
def build_cache(data_dir):
    """Build STATS + CACHE from raw train/. Returns (CACHE, STATS, train_paths, labels)."""
    train_dir = f'{data_dir}/train'
    train_paths = sorted(glob.glob(f'{train_dir}/*.csv'))
    labels = pd.read_csv(f'{data_dir}/train_labels.csv').sort_values('id').reset_index(drop=True)[['x', 'y', 'z']].to_numpy(np.float32)
    assert len(train_paths) == len(labels), (len(train_paths), len(labels))
    print(f'train 궤적 {len(train_paths):,}개')

    _SEQ, _SC = [], []
    for p in train_paths:
        X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy(); s, sc, *_ = build_features_clean(X); _SEQ.append(s); _SC.append(sc)
    _SEQ = np.stack(_SEQ); _SC = np.stack(_SC)
    STATS = {'seq_mean': _SEQ.reshape(-1, 13).mean(0), 'seq_std': _SEQ.reshape(-1, 13).std(0),
             'scalar_mean': _SC.mean(0), 'scalar_std': _SC.std(0)}
    print('norm_stats 생성 (seq 13 + scalar 22)')

    SEQ, SCAL, MASK, TGT, ROT, SPD, TRAJ, REAL = [], [], [], [], [], [], [], []
    for i, p in enumerate(train_paths):
        X = load_sample(p).astype(np.float64)
        ex = [(10, labels[i])] + [(e, X[e + 2]) for e in INTERIOR_E]
        for e, tgt in ex:
            W = X[:e + 1]; L = len(W)
            seq, sc, rot, base, lp = window_features(W)
            seq_n = ((seq - STATS['seq_mean']) / STATS['seq_std']).astype(np.float32)
            sc_n = ((sc - STATS['scalar_mean']) / STATS['scalar_std']).astype(np.float32)
            pad = 11 - L
            seq11 = np.zeros((11, 13), np.float32); seq11[pad:] = seq_n
            mask = np.zeros(11, np.float32); mask[pad:] = 1.0
            tgt_rot = (rot @ (tgt - base)).astype(np.float32)
            SEQ.append(seq11); SCAL.append(sc_n); MASK.append(mask); TGT.append(tgt_rot); ROT.append(rot)
            SPD.append(float(np.linalg.norm(np.gradient(W, DT, axis=0)[-1]))); TRAJ.append(i); REAL.append(int(e == 10))
    CACHE = dict(seq=np.stack(SEQ), scal=np.stack(SCAL), mask=np.stack(MASK), tgt=np.stack(TGT),
                 rot=np.stack(ROT), spd=np.array(SPD, np.float32), traj=np.array(TRAJ), real=np.array(REAL), labels=labels)
    print(f'캐시 생성: {len(SEQ):,} examples (real {sum(REAL):,})')
    return CACHE, STATS, train_paths, labels


# ══════════════════════════════════════════════════════════════════════════════
# cell 4a · AttnGRU (phaseG)
# ══════════════════════════════════════════════════════════════════════════════
class AttnGRU(nn.Module):
    def __init__(self, seq_dim=13, scal_dim=22, h=128, nl=3, dr=0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.attn = nn.Linear(h*2, 1)
        self.head = nn.Sequential(nn.Linear(h*6+scal_dim, 256), nn.GELU(), nn.Dropout(dr),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, seq, scal, mask):
        x = self.proj(seq); out, _ = self.gru(x); last = out[:, -1, :]; m = mask.unsqueeze(-1)
        mean = (out*m).sum(1)/m.sum(1).clamp(min=1)
        score = self.attn(out).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = (torch.softmax(score, dim=1).unsqueeze(-1)*out).sum(1)
        return self.head(torch.cat([last, mean, att, scal], -1))


# ══════════════════════════════════════════════════════════════════════════════
# cell 4b · Neural ODE (phaseODE)
# ══════════════════════════════════════════════════════════════════════════════
class ODEModel(nn.Module):
    def __init__(self, seq_dim=13, scal_dim=22, h=128, nl=2, dr=0.15, latent=96, nsteps=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.to_latent = nn.Sequential(nn.Linear(h*4+scal_dim, latent), nn.LayerNorm(latent), nn.GELU())
        self.accel = nn.Sequential(nn.Linear(3+3+latent, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dr),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.damping = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.bias = nn.Parameter(torch.zeros(3))
        self.nsteps = nsteps; self.dt = 0.08/nsteps

    def _deriv(self, rpos, rvel, lat):
        a = self.accel(torch.cat([rpos, rvel, lat], -1))
        return rvel, -self.damping*rvel+a

    def forward(self, seq, scal, mask):
        x = self.proj(seq); out, _ = self.gru(x); m = mask.unsqueeze(-1)
        mean = (out*m).sum(1)/m.sum(1).clamp(min=1)
        lat = self.to_latent(torch.cat([out[:, -1, :], mean, scal], -1))
        rpos = torch.zeros(seq.size(0), 3, device=seq.device); rvel = torch.zeros_like(rpos)
        for _ in range(self.nsteps):
            dt = self.dt
            dp1, dv1 = self._deriv(rpos, rvel, lat)
            dp2, dv2 = self._deriv(rpos+0.5*dt*dp1, rvel+0.5*dt*dv1, lat)
            dp3, dv3 = self._deriv(rpos+0.5*dt*dp2, rvel+0.5*dt*dv2, lat)
            dp4, dv4 = self._deriv(rpos+dt*dp3, rvel+dt*dv3, lat)
            rpos = rpos+(dt/6)*(dp1+2*dp2+2*dp3+dp4)
            rvel = rvel+(dt/6)*(dv1+2*dv2+2*dv3+dv4)
        return rpos+self.bias


# ══════════════════════════════════════════════════════════════════════════════
# cell 4c · HyperPhysics (phaseH)
# ══════════════════════════════════════════════════════════════════════════════
class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, min_win=3, mode="extended", device="cpu"):
        X_tensor = torch.tensor(X, dtype=torch.float32); y_tensor = torch.tensor(y, dtype=torch.float32)
        windows = []
        for i in range(len(X)):
            targets = [4, 5, 6, 7, 8, 9, 10, 12] if mode == "extended" else [12, 10]
            for target_idx in targets:
                end_idx = target_idx - 2
                max_w = end_idx + 2 if mode == "extended" else (12 if target_idx == 12 else 10)
                for w in range(min_win, max_w):
                    windows.append((i, w, target_idx))
        X_list = []; y_list = []
        for i, w, target_idx in windows:
            X_orig = X_tensor[i]; end_idx = target_idx - 2
            pts = X_orig[end_idx - w + 1: end_idx + 1]
            target = y_tensor[i] if target_idx == 12 else X_orig[target_idx]
            if w < 11:
                v0 = pts[1] - pts[0]; n_pad = 11 - w
                js = torch.arange(n_pad, 0, -1, dtype=torch.float32)
                pad = pts[0:1] - js.unsqueeze(1) * v0.unsqueeze(0)
                X_padded = torch.cat([pad, pts], dim=0)
            else:
                X_padded = pts.clone()
            X_list.append(X_padded); y_list.append(target)
        self.X_all = torch.stack(X_list).to(device); self.y_all = torch.stack(y_list).to(device)
        diffs = self.X_all[:, 1:] - self.X_all[:, :-1]
        n1 = diffs[:, 1:].norm(dim=2).clamp(min=1e-8); n2 = diffs[:, :-1].norm(dim=2).clamp(min=1e-8)
        cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2) / (n1 * n2)).clamp(-1, 1)
        theta_last = torch.acos(cos_t[:, -1])
        self.theta_weights = (1.0 + 4.0 * (theta_last / 1.0).clamp(0, 1)).cpu().numpy()

    def __len__(self): return len(self.X_all)
    def __getitem__(self, idx): return self.X_all[idx], self.y_all[idx]


def _ema_va_local(diffs_local, alpha, beta):
    B, T, _ = diffs_local.shape
    one_m_a = 1.0 - alpha; one_m_b = 1.0 - beta
    vs = diffs_local.new_empty(B, T, 3); v = diffs_local[:, 0]; vs[:, 0] = v
    for t in range(1, T):
        v = alpha * diffs_local[:, t] + one_m_a * v; vs[:, t] = v
    vl = vs[:, -1]
    ad = vs[:, 1:] - vs[:, :-1]; a = ad[:, 0]
    for t in range(1, T - 1):
        a = beta * ad[:, t] + one_m_b * a
    return vl, a


def _soft_hit_loss(pred, target, thr=0.013012, k=408.348):
    return (1 - torch.sigmoid(-(torch.norm(pred - target, dim=1) - thr) * k)).mean()


def extract_features(X, mean_stats=None, std_stats=None, dir_net=None, heading_mode="3step"):
    device = X.device
    p_last = X[:, 10]; diffs = X[:, 1:] - X[:, :-1]
    n1 = diffs[:, 1:].norm(dim=2, keepdim=True) + 1e-8; n2 = diffs[:, :-1].norm(dim=2, keepdim=True) + 1e-8
    cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2, keepdim=True) / (n1 * n2)).clamp(-1, 1)
    theta_seq = torch.acos(cos_t).squeeze(2)
    theta = theta_seq[:, -1:]; theta_mean = theta_seq.mean(1, keepdim=True); theta_std = theta_seq.std(1, keepdim=True)
    theta_vel = theta_seq[:, -1:] - theta_seq[:, -2:-1]
    theta_acc = theta_seq[:, -1:] - 2 * theta_seq[:, -2:-1] + theta_seq[:, -3:-2]
    theta_trend = theta_seq[:, -1:] - theta_seq[:, -3:].mean(1, keepdim=True)
    if dir_net is not None:
        speed_seq = diffs.norm(dim=2); state = torch.cat([speed_seq, theta_seq], dim=1)
        if dir_net[0].in_features == 29:
            z_speed_seq = diffs[:, :, 2].abs(); state = torch.cat([state, z_speed_seq], dim=1)
        weights = F.softmax(dir_net(state), dim=1); v_sm = (diffs * weights.unsqueeze(2)).sum(dim=1)
    else:
        v_sm = (3 * diffs[:, -1] + 2 * diffs[:, -2] + diffs[:, -3]) / 6.0 if heading_mode == "3step" else diffs[:, -1]
    fwd = v_sm / (v_sm.norm(dim=1, keepdim=True) + 1e-8)
    up_w = torch.zeros_like(fwd); up_w[:, 2] = 1.0
    up_w[fwd[:, 2].abs() > 0.99] = torch.tensor([0., 1., 0.], device=device)
    right = torch.cross(fwd, up_w, dim=1); right = right / (right.norm(dim=1, keepdim=True) + 1e-8)
    up = torch.cross(right, fwd, dim=1); up = up / (up.norm(dim=1, keepdim=True) + 1e-8)
    R = torch.stack([fwd, right, up], dim=2)
    v_last = diffs[:, -1]; v_prev1 = diffs[:, -2]; speed = v_last.norm(dim=1, keepdim=True)
    a_last = v_last - v_prev1; acc_mag = a_last.norm(dim=1, keepdim=True)
    v_local = torch.matmul(v_last.unsqueeze(1), R).squeeze(1)
    a_local = torch.matmul(a_last.unsqueeze(1), R).squeeze(1)
    X_local = torch.matmul(X - p_last.unsqueeze(1), R); p_std_local = X_local.std(1)
    v_local_abs = v_local.abs()
    jerk_g = diffs[:, -1] - 2 * diffs[:, -2] + diffs[:, -3]
    jerk_l = torch.matmul(jerk_g.unsqueeze(1), R).squeeze(1); jerk_mag = jerk_g.norm(dim=1, keepdim=True)
    features = torch.cat([v_local, a_local, speed, acc_mag, theta, theta_mean, theta_std, theta_trend,
                          theta_vel, theta_acc, p_std_local, v_local_abs, jerk_l, jerk_mag], dim=1)
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(0, keepdim=True); std_stats = features.std(0, keepdim=True) + 1e-8
    return (features - mean_stats) / std_stats, diffs, p_last, theta, theta_mean, theta_std, theta_seq, R, speed, mean_stats, std_stats


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
    def forward(self, x): return self.ln(x + self.net(x))


class PriorBiasedLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_bias):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('prior_bias', prior_bias.clone().detach())
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x) + self.prior_bias


def rodrigues_rotate(v, w):
    theta = w.norm(dim=1, keepdim=True); k = w / (theta + 1e-8)
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)
    dot = (v * k).sum(dim=1, keepdim=True); cross = torch.cross(k, v, dim=1)
    return v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)


class HyperPhysics_xy2(nn.Module):
    def __init__(self, input_dim=24, **kwargs):
        super().__init__()
        self.sh_thr = kwargs.pop('sh_thr', 0.013012); self.sh_k = kwargs.pop('sh_k', 408.348044)
        self.mse_w = kwargs.pop('mse_w', 129.172037); self.local_w = kwargs.pop('local_w', 0.050941)
        self.theta_thr = kwargs.pop('theta_thr', 1.087618); self.speed_thr = kwargs.pop('speed_thr', 0.034583)
        self.lr = 0.005400; self.wd = 0.005659
        self.register_buffer("mean_stats", torch.zeros(1, input_dim)); self.register_buffer("std_stats", torch.ones(1, input_dim))
        prior_dir = torch.tensor([-10., -10., -10., -10., -10., -10., -10., 0., 0.693, 1.098])
        self.dir_net = nn.Sequential(nn.Linear(29, 24), nn.LayerNorm(24), nn.GELU(), PriorBiasedLinear(24, 10, prior_dir))
        prior_ema = torch.zeros(6)
        self.temporal_net = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU(), PriorBiasedLinear(32, 6, prior_ema))
        prior_dyn = torch.tensor([0., 0., 0., 0., 0., 0.] + [-4.] * 24)
        self.dynamics_net = nn.Sequential(nn.Linear(input_dim, 96), nn.LayerNorm(96), nn.GELU(), ResBlock(96), PriorBiasedLinear(96, 30, prior_dyn))
        self.omega_w = nn.Parameter(torch.tensor([0.0, -0.5, -1.0]))
        self.omega_net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 48), nn.GELU(), nn.Linear(48, 3))
        with torch.no_grad():
            nn.init.normal_(self.omega_net[-1].weight, std=0.01); nn.init.zeros_(self.omega_net[-1].bias)
        self.diffusion_net = nn.Sequential(nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3))

    def get_features(self, X, mean_stats=None, std_stats=None):
        return extract_features(X, mean_stats, std_stats, self.dir_net, heading_mode="3step")

    @staticmethod
    def _rotation_vector(d_prev, d_curr):
        n_prev = d_prev.norm(dim=1, keepdim=True).clamp(min=1e-8); n_curr = d_curr.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d_hat_prev = d_prev / n_prev; d_hat_curr = d_curr / n_curr
        cross = torch.linalg.cross(d_hat_prev, d_hat_curr, dim=1); sin_t = cross.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_t = (d_hat_prev * d_hat_curr).sum(1, keepdim=True).clamp(-0.9999, 0.9999); theta = torch.atan2(sin_t, cos_t)
        speed_gate = torch.sigmoid((n_prev + n_curr) * 500 - 5)
        return cross / sin_t * theta * speed_gate

    def forward(self, features, diffs, p_last, theta, speed, R):
        B = diffs.shape[0]
        ema_raw = self.temporal_net(features[:, 8:17])
        alpha = torch.sigmoid(ema_raw[:, 0:3]) * 0.8 + 0.1; beta = torch.sigmoid(ema_raw[:, 3:6]) * 0.199 + 0.8
        dyn_raw = self.dynamics_net(features)
        w_v = 2.0 + dyn_raw[:, 0:3]; w_a = 1.0 + dyn_raw[:, 3:6]
        v_local_abs = features[:, 17:20]; v_local_abs2 = v_local_abs * v_local_abs; theta2 = theta * theta
        exp_v = (F.softplus(dyn_raw[:, 6:9]) * v_local_abs + F.softplus(dyn_raw[:, 9:12]) * v_local_abs2 +
                 F.softplus(dyn_raw[:, 12:15]) * theta + F.softplus(dyn_raw[:, 15:18]) * theta2)
        exp_a = (F.softplus(dyn_raw[:, 18:21]) * v_local_abs + F.softplus(dyn_raw[:, 21:24]) * v_local_abs2 +
                 F.softplus(dyn_raw[:, 24:27]) * theta + F.softplus(dyn_raw[:, 27:30]) * theta2)
        diffs_local = torch.matmul(diffs, R)
        vl, al = _ema_va_local(diffs_local, alpha, beta)
        diff_speed = diffs_local.norm(dim=2)
        def rv_masked(ka, kb):
            rv = self._rotation_vector(diffs_local[:, ka], diffs_local[:, kb])
            valid = ((diff_speed[:, ka] > 1e-5) & (diff_speed[:, kb] > 1e-5)).float()
            return rv * valid.unsqueeze(1), valid
        ov1, vm1 = rv_masked(-2, -1); ov2, vm2 = rv_masked(-3, -2); ov3, vm3 = rv_masked(-4, -3)
        w_logits = self.omega_w.view(1, 3).expand(B, -1)
        masks = torch.stack([vm1, vm2, vm3], dim=1)
        w_attn = F.softmax(w_logits.masked_fill(masks == 0, -1e9), dim=1)
        omega_hist = (w_attn[:, 0].unsqueeze(1) * ov1 + w_attn[:, 1].unsqueeze(1) * ov2 + w_attn[:, 2].unsqueeze(1) * ov3)
        current_speed = speed.view(B, 1) if speed is not None else diff_speed[:, -1].unsqueeze(1)
        omega_speed_gate = torch.sigmoid(current_speed * 500 - 5)
        omega_delta = self.omega_net(features) * omega_speed_gate
        theta_scalar = theta.view(B, 1)
        theta_gate = torch.sigmoid((theta_scalar - self.theta_thr) * 10)
        speed_gate_strong = torch.sigmoid((current_speed - self.speed_thr) * 200)
        rotation_gate = theta_gate * speed_gate_strong
        omega = (omega_hist + omega_delta) * rotation_gate
        v_rotated = rodrigues_rotate(vl, omega)
        pred_local = (w_v * torch.exp(-exp_v)) * v_rotated + (w_a * torch.exp(-exp_a)) * al
        log_var = self.diffusion_net(features).clamp(min=-5.0, max=5.0)
        pred_global = p_last + torch.einsum('nij,nj->ni', R, pred_local)
        return pred_global, pred_local, log_var

    def compute_loss(self, pp, yr, pred_local=None, yr_local=None, log_var=None, **kwargs):
        sh = _soft_hit_loss(pp, yr, thr=self.sh_thr, k=self.sh_k)
        loss = sh + self.mse_w * F.mse_loss(pp, yr)
        if pred_local is not None and yr_local is not None and log_var is not None:
            squared_error = (pred_local - yr_local) ** 2
            nll_loss = 0.5 * (torch.exp(-log_var) * squared_error + log_var)
            loss = loss + self.local_w * nll_loss.mean()
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# cell 5 · losses + training
# ══════════════════════════════════════════════════════════════════════════════
def combined_loss(pred, true):
    d = 0.01; hub = F.huber_loss(pred, true, delta=d) / (0.5*d*d)
    d2 = (pred-true).pow(2).sum(-1); soft = (1 - torch.exp(-d2 / (2*SIGMA**2))).mean()
    dd = torch.sqrt(d2 + 1e-12); sr = -torch.sigmoid((0.01 - dd) / RHIT_TAU).mean()
    return HW*hub + GW*soft + RHIT_W*sr


def r_hit(p, t, thr=0.01):
    return float(np.mean(np.linalg.norm(p - t, axis=1) <= thr))


def train_cache_seed(seed, factory, CACHE):
    dev = DEVICE
    seq = torch.tensor(CACHE['seq']); scal = torch.tensor(CACHE['scal'])
    msk = torch.tensor(CACHE['mask']); tgt = torch.tensor(CACHE['tgt'])
    N = len(seq); idx_all = np.arange(N)
    torch.manual_seed(1000 + seed); np.random.seed(1000 + seed)
    model = factory().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=GRU_ODE_EPOCHS)
    flip = torch.tensor(Y_FLIP, device=dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, GRU_ODE_EPOCHS + 1):
        model.train(); np.random.shuffle(idx_all)
        for i in range(0, N, 256):
            b = idx_all[i:i + 256]
            s = seq[b].to(dev); c = scal[b].to(dev); mk = msk[b].to(dev); tg = tgt[b].to(dev)
            if torch.rand(1).item() < FLIP_PROB:
                s = s.clone(); s[:, :, flip] *= -1; tg = tg.clone(); tg[:, 1] *= -1
            s = s + torch.randn_like(s) * NOISE_STD * mk.unsqueeze(-1)
            opt.zero_grad(); loss = combined_loss(model(s, c, mk), tg); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else: ema[k] = v.detach().clone()
    return ema


def train_h_seed(seed, X, Y):
    dev = DEVICE; set_seed(1000 + seed)
    ds = SlidingWindowDataset(X, Y, min_win=3, mode="extended", device=dev)
    loader = DataLoader(ds, batch_size=256, sampler=WeightedRandomSampler(ds.theta_weights, len(ds), replacement=True))
    model = HyperPhysics_xy2().to(dev)
    with torch.no_grad():
        *_, mn, st = model.get_features(torch.tensor(X, dtype=torch.float32, device=dev))
        model.mean_stats.copy_(mn); model.std_stats.copy_(st)
    opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.6)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, H_EPOCHS + 1):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            ft, df, pl, th, _, _, _, Rt, sp, _, _ = model.get_features(Xb, model.mean_stats, model.std_stats)
            pp, pred_local, log_var = model(ft, df, pl, th, sp, Rt)
            yr_local = torch.matmul((yb - pl).unsqueeze(1), Rt).squeeze(1)
            loss = model.compute_loss(pp, yb, pred_local, yr_local, log_var)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else: ema[k] = v.detach().clone()
    return ema


# ══════════════════════════════════════════════════════════════════════════════
# cell 7 · predict + blend
# ══════════════════════════════════════════════════════════════════════════════
def predict_all(data_dir, STATS, train_paths_unused=None):
    """Test set으로 30모델 블렌드 예측. submission df 반환."""
    test_dir = f'{data_dir}/test'
    test_paths = sorted(glob.glob(f'{test_dir}/*.csv'))
    ids = [os.path.basename(p)[:-4] for p in test_paths]
    seqs, scals, rots, bases = [], [], [], []
    for p in test_paths:
        X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy()
        seq, sc22, rot, base, _ = build_features_clean(X); seq_n, sc_n = normalize(seq, sc22, STATS)
        seqs.append(seq_n); scals.append(sc_n); rots.append(rot); bases.append(base)
    seqT = torch.tensor(np.stack(seqs)); scalT = torch.tensor(np.stack(scals))
    rotT = np.stack(rots); baseT = np.stack(bases); maskT = torch.ones(len(seqT), 11); flipT = torch.tensor(Y_FLIP)
    XtT = torch.tensor(np.stack([load_sample(p) for p in test_paths]))

    def predict_resid(fp, factory):
        m = factory().to(DEVICE); m.load_state_dict(torch.load(fp, map_location=DEVICE, weights_only=False)['model_state']); m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seqT), 256):
                s = seqT[i:i+256].to(DEVICE); c = scalT[i:i+256].to(DEVICE); mk = maskT[i:i+256].to(DEVICE)
                pr = m(s, c, mk).cpu().numpy(); sf = s.clone(); sf[:, :, flipT] *= -1
                pf = m(sf, c, mk).cpu().numpy(); pf[:, 1] *= -1; out.append((pr + pf) / 2)
        r = np.concatenate(out)
        return baseT + np.einsum('bij,bj->bi', rotT.transpose(0, 2, 1), r)

    def predict_h(fp):
        m = HyperPhysics_xy2().to(DEVICE); m.load_state_dict(torch.load(fp, map_location=DEVICE, weights_only=False)['model_state']); m.eval()
        def fwd(Z):
            o = []
            with torch.no_grad():
                for i in range(0, len(Z), 256):
                    b = Z[i:i+256].to(DEVICE)
                    ft, df, pl, th, _, _, _, Rt, sp, _, _ = m.get_features(b, m.mean_stats, m.std_stats)
                    pp, _, _ = m(ft, df, pl, th, sp, Rt); o.append(pp.cpu().numpy())
            return np.concatenate(o)
        pr = fwd(XtT); Xf = XtT.clone(); Xf[:, :, 1] *= -1; pf = fwd(Xf); pf[:, 1] *= -1
        return (pr + pf) / 2

    preds = []
    for k in range(N_EACH):
        fp = f'{MODELS_DIR}/phaseG_full_{k}.pt'
        if os.path.exists(fp): preds.append(predict_resid(fp, AttnGRU))
    for k in range(N_EACH):
        fp = f'{MODELS_DIR}/phaseODE_full_{k}.pt'
        if os.path.exists(fp): preds.append(predict_resid(fp, ODEModel))
    for k in range(N_EACH):
        fp = f'{MODELS_DIR}/phaseH_full_{k}.pt'
        if os.path.exists(fp): preds.append(predict_h(fp))
    print(f'블렌드 {len(preds)}모델 (GRU+ODE+H)')

    ens = np.mean(preds, 0)
    sub = pd.DataFrame({'id': ids, 'x': ens[:, 0], 'y': ens[:, 1], 'z': ens[:, 2]})
    return sub


# ══════════════════════════════════════════════════════════════════════════════
# cell 6 · main — 30모델 학습 + 제출
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f'device: {DEVICE} | FROM_SCRATCH: {FROM_SCRATCH} | DATA_DIR: {DATA_DIR} | N_EACH: {N_EACH} | GRU_ODE_EPOCHS: {GRU_ODE_EPOCHS} | H_EPOCHS: {H_EPOCHS}')
    CACHE, STATS, train_paths, labels = build_cache(DATA_DIR)
    X_train = np.stack([load_sample(p) for p in train_paths]).astype(np.float32)
    Y_train = labels

    if FROM_SCRATCH:
        t0 = time.time()
        for k in range(N_EACH):
            torch.save({'model_state': train_cache_seed(k, AttnGRU, CACHE)}, f'{MODELS_DIR}/phaseG_full_{k}.pt')
            print(f'  GRU seed{k} 완료 ({(time.time()-t0)/60:.1f}min)')
        for k in range(N_EACH):
            torch.save({'model_state': train_cache_seed(k, ODEModel, CACHE)}, f'{MODELS_DIR}/phaseODE_full_{k}.pt')
            print(f'  ODE seed{k} 완료 ({(time.time()-t0)/60:.1f}min)')
        for k in range(N_EACH):
            torch.save({'model_state': train_h_seed(k, X_train, Y_train)}, f'{MODELS_DIR}/phaseH_full_{k}.pt')
            print(f'  H   seed{k} 완료 ({(time.time()-t0)/60:.1f}min)')
        print(f'전체 학습 완료 ({(time.time()-t0)/60:.1f}min)')
    else:
        n = len(glob.glob(f'{MODELS_DIR}/phase*_full_*.pt'))
        assert n == 3 * N_EACH, f'{MODELS_DIR}에 {3*N_EACH}개 .pt 필요 (현재 {n}개). FROM_SCRATCH=True로 학습하세요.'
        print(f'기존 학습본 {n}개 로드 사용')

    sub = predict_all(DATA_DIR, STATS, train_paths)
    out_csv = os.environ.get('SUBMISSION_OUT', './submission_GOH30.csv')
    sub.to_csv(out_csv, index=False)
    print(f'저장: {out_csv} {sub.shape}')


if __name__ == '__main__':
    main()
