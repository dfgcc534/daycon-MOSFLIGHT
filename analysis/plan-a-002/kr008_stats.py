"""KR008 통계 분석 (plan-a-003 §7) — read-only, 재학습 0.

KR008(반사+노이즈 aug) LB +0.0008 vs KR003 가 noise 인지 엄밀 검정 +
4-exp lever-decay 유의성. 기존 OOF per_sample_hit npz 만 사용 (test 라벨 없으므로
LB 유의성은 OOF discordant rate 를 proxy SD 로 추론).

paired_perm/hit_mask 는 plan-a-002 run_oof.py 와 동일 정의 (torch 미import 위해 inline 복제).
McNemar exact = scipy.stats.binomtest. bootstrap CI 신규.

Usage: python analysis/plan-a-002/kr008_stats.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

_THIS = Path(__file__).resolve().parent
_A001 = _THIS.parent / "plan-a-001"
R_HIT = 0.01

# exp → (npz path, json path, LB score)
EXPS = {
    "KR001": (_A001 / "results_kr001.npz", _A001 / "results_kr001.json", 0.6758),
    "KR002": (_A001 / "results_kr002.npz", _A001 / "results_kr002.json", 0.6818),
    "KR003": (_THIS / "results_kr003.npz", _THIS / "results_kr003.json", 0.6854),
    "KR008": (_THIS / "results_kr008.npz", _THIS / "results_kr008.json", 0.6862),
}
# lever pair: (treatment, baseline, lever 이름)
PAIRS = [
    ("KR002", "KR001", "입력 yaw 회전"),
    ("KR003", "KR002", "Kalman 부산물 feature"),
    ("KR008", "KR003", "반사+노이즈 augmentation"),
]


def paired_perm(hit_b, hit_a, n_resample=10000, seed=0):
    """paired sign-flip permutation (run_oof.py 동일). (delta, p)."""
    d = hit_b.astype(np.float64) - hit_a.astype(np.float64)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_resample, d.shape[0]))
    null = (signs * d[None, :]).mean(axis=1)
    return float(obs), float((np.abs(null) >= abs(obs)).mean())


def bootstrap_ci(hit_b, hit_a, n_boot=10000, seed=0, alpha=0.05):
    """paired Δ 의 bootstrap CI (sample 재추출). (lo, hi)."""
    d = hit_b.astype(np.float64) - hit_a.astype(np.float64)
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def main():
    # load
    hits, ys, fids, cfg, lb = {}, {}, {}, {}, {}
    for e, (npz, js, lbs) in EXPS.items():
        z = np.load(npz)
        hits[e] = z["per_sample_hit"].astype(bool)
        ys[e] = z["y"]
        fids[e] = z["fold_ids"]
        cfg[e] = json.loads(Path(js).read_text())["config_hit_1cm"]
        lb[e] = lbs
    n = len(ys["KR008"])
    for e in EXPS:
        assert np.allclose(ys[e], ys["KR008"]), f"{e} y 정렬 불일치"

    report = {"n": n, "pairs": {}, "per_fold_kr008_vs_kr003": {}, "config_variance": {}, "lever_decay": []}
    lines = [f"# KR008 통계 분석 (n={n}, read-only)\n"]

    # 1. pairwise significance + 2. LB 매핑
    lines.append("## Pairwise significance (OOF McNemar/CI/permutation + LB proxy)\n")
    lines.append("| pair | lever | OOF Δ | disc b/c | McNemar p | perm p | bootstrap 95% CI | noise SD | LB Δ | LB Δ/SD | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, base, name in PAIRS:
        hb, ha = hits[t], hits[base]
        oof_d = hb.mean() - ha.mean()
        b = int((hb & ~ha).sum())   # treatment hit, baseline miss
        c = int((~hb & ha).sum())   # baseline hit, treatment miss
        mc_p = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        perm_d, perm_p = paired_perm(hb, ha)
        ci_lo, ci_hi = bootstrap_ci(hb, ha)
        noise_sd = math.sqrt(b + c) / n          # paired Δ SD (McNemar 근사)
        lb_d = lb[t] - lb[base]
        lb_sd_ratio = lb_d / noise_sd
        # verdict: LB Δ 가 2·SD 초과 + 같은 부호면 real, 아니면 noise (LB 라벨 부재 → proxy SD 기반)
        verdict = "real" if lb_d > 2 * noise_sd else ("noise" if abs(lb_d) <= 2 * noise_sd else "neg")
        report["pairs"][f"{t}_vs_{base}"] = dict(
            lever=name, oof_delta=round(oof_d, 4), b=b, c=c, mcnemar_p=round(mc_p, 4),
            perm_delta=round(perm_d, 4), perm_p=round(perm_p, 4),
            bootstrap_ci=[round(ci_lo, 4), round(ci_hi, 4)], noise_sd=round(noise_sd, 4),
            lb_delta=round(lb_d, 4), lb_delta_over_sd=round(lb_sd_ratio, 2), verdict=verdict)
        lines.append(f"| {t} vs {base} | {name} | {oof_d:+.4f} | {b}/{c} | {mc_p:.3f} | {perm_p:.3f} | "
                     f"[{ci_lo:+.4f},{ci_hi:+.4f}] | {noise_sd:.4f} | {lb_d:+.4f} | {lb_sd_ratio:+.2f} | **{verdict}** |")
        report["lever_decay"].append(dict(lever=name, lb_delta=round(lb_d, 4), verdict=verdict))

    # 3. per-fold (KR008 vs KR003)
    lines.append("\n## Per-fold breakdown (KR008 vs KR003) — sign-consistency\n")
    lines.append("| fold | n | KR003 | KR008 | Δ |")
    lines.append("|---|---|---|---|---|")
    fid = fids["KR003"]
    deltas = []
    for f in range(5):
        m = fid == f
        d = hits["KR008"][m].mean() - hits["KR003"][m].mean()
        deltas.append(d)
        report["per_fold_kr008_vs_kr003"][f"fold{f}"] = dict(
            n=int(m.sum()), kr003=round(float(hits["KR003"][m].mean()), 4),
            kr008=round(float(hits["KR008"][m].mean()), 4), delta=round(float(d), 4))
        lines.append(f"| {f} | {int(m.sum())} | {hits['KR003'][m].mean():.4f} | {hits['KR008'][m].mean():.4f} | {d:+.4f} |")
    n_pos = sum(1 for d in deltas if d > 0)
    sign_consistent = n_pos == 5 or n_pos == 0
    report["per_fold_kr008_vs_kr003"]["sign_consistent"] = sign_consistent
    report["per_fold_kr008_vs_kr003"]["n_positive"] = n_pos
    lines.append(f"\n→ 양 fold {n_pos}/5, sign-consistent={sign_consistent} "
                 f"(부호 섞임 → fold-noise, 일관 shift 아님)" if not sign_consistent
                 else f"\n→ sign-consistent ({n_pos}/5 동일 부호) = 일관 shift")

    # 4. A/B config variance vs effect
    lines.append("\n## A/B config variance vs effect size\n")
    lines.append("| exp | config A | config B | |A−B| |")
    lines.append("|---|---|---|---|")
    for e in EXPS:
        ab = abs(cfg[e]["A"] - cfg[e]["B"])
        report["config_variance"][e] = dict(A=cfg[e]["A"], B=cfg[e]["B"], spread=round(ab, 4))
        lines.append(f"| {e} | {cfg[e]['A']:.4f} | {cfg[e]['B']:.4f} | {ab:.4f} |")
    kr008_spread = abs(cfg["KR008"]["A"] - cfg["KR008"]["B"])
    aug_effect = abs(report["pairs"]["KR008_vs_KR003"]["oof_delta"])
    lines.append(f"\n→ KR008 config A/B spread={kr008_spread:.4f} vs aug OOF effect={aug_effect:.4f} "
                 f"→ **config noise {'>' if kr008_spread > aug_effect else '<='} aug effect** "
                 f"(effect 가 cfg 선택 noise 보다 {'작음' if kr008_spread > aug_effect else '큼'})")

    # 5. power
    lines.append("\n## Power (α=0.05)\n")
    sd008 = report["pairs"]["KR008_vs_KR003"]["noise_sd"]
    min_det = 1.96 * sd008
    report["power"] = dict(noise_sd=sd008, min_detectable_delta_2sided=round(min_det, 4),
                           lb_delta=0.0008, detectable=bool(0.0008 > min_det))
    lines.append(f"- KR008-vs-KR003 noise SD = {sd008:.4f} → min detectable Δ (1.96·SD) = {min_det:.4f}")
    lines.append(f"- 실제 LB Δ = +0.0008 → 검출 {'가능' if 0.0008 > min_det else '불가'} (Δ < min detectable)")

    out = "\n".join(lines)
    print(out)
    (_THIS / "kr008_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[saved] {(_THIS / 'kr008_stats.json').name}")
    return report


if __name__ == "__main__":
    main()
