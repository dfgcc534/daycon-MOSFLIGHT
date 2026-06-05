"""plan-006 Variant E (physics_bias + soft averaging) 검증 entry.

재학습 X, plan-004/005 산출 재사용. 부재 시 fallback 재생성.

Module-level import 정책:
  - selector  : STAGE 0~3 전반 (read_labels, load_stack, make_candidates,
                candidate_physics_bias, fit_regime_bins, assign_regimes,
                read_submission_ids, soft_select, CANDIDATES, R_HIT, EPS).
  - boundary  : 본 plan 에서 사용 안 함. plan spec L222 의 `boundary.soft_select`
                는 실제로 `selector.soft_select` 임. decision-note 박제 (commit msg).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
PLAN004_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"
PLAN005_DIR = REPO / "analysis/plan-005"
ANALYSIS_DIR = REPO / "analysis/plan-006"
RUN_DIR = REPO / "runs/baseline/E001_minimal-variant-e"
DEVICE = "cuda:1"

PHYSICS_WEIGHT = 0.65
SOFT_TEMP = 0.03


def verify_inputs() -> dict[str, Path]:
    """G0 — 필수 산출 검증. 부재 시 재생성 path 안내."""
    paths = {
        "corrected_oof":  PLAN005_DIR / "corrected_oof.npz",
        "corrected_test": PLAN005_DIR / "corrected_test.npz",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"plan-005 corrected_*.npz 부재: {missing}. "
            f"재생성: `python -c \"import sys; sys.path.insert(0, 'analysis/plan-005'); "
            f"import diagnostic; diagnostic.rerun_corrector_save_intermediates()\"`"
        )
    # shape + finite check
    for k, p in paths.items():
        z = np.load(p)
        for kk in ("cands", "corrected"):
            arr = z[kk]
            assert arr.ndim == 3 and arr.shape[1:] == (27, 3), \
                f"{k}.{kk} shape={arr.shape}"
            assert np.isfinite(arr).all(), f"{k}.{kk} has non-finite"
    return paths


def _measure(
    cands: np.ndarray,
    score_E: np.ndarray,
    train_y: np.ndarray,
    r_hit: float,
) -> dict:
    """cands: [N, 27, 3] float32 → {argmax, soft, _err_argmax, _err_soft}.

    score_E 가 sample 무관 상수 → argmax 도 sample 무관 (numpy first-index 규약).
    """
    pick = cands[np.arange(len(train_y)), score_E.argmax(axis=1)]
    err_arg = np.linalg.norm(pick - train_y, axis=1)
    soft = selector.soft_select(cands, score_E, temperature=SOFT_TEMP)
    err_soft = np.linalg.norm(soft - train_y, axis=1)
    return {
        "argmax": float((err_arg <= r_hit).mean()),
        "soft":   float((err_soft <= r_hit).mean()),
        "_err_argmax": err_arg, "_err_soft": err_soft,
    }


def stage1_variant_e_oof() -> dict:
    """Variant E (physics_bias + soft averaging) 의 OOF hit 측정.

    E_raw (27 raw cands) + E_corrected (27 corrected cands) 둘 다.
    Plan-004 모듈 계약:
      - selector.candidate_physics_bias(candidates, target) → np.ndarray[27] float32 centered
      - selector.soft_select(cands [N,27,3], scores [N,27], temperature float) → [N,3] float32
      - selector.R_HIT == 0.01 (m).
    """
    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    z = np.load(PLAN005_DIR / "corrected_oof.npz")
    raw_cands       = z["cands"].astype(np.float32)
    corrected_cands = z["corrected"].astype(np.float32)

    end_idx = train_x.shape[1] - 1
    train_cands_check = selector.make_candidates(train_x, end_idx, horizon=2).astype(np.float32)
    assert np.allclose(train_cands_check, raw_cands, atol=1e-5), \
        "raw_cands mismatch — plan-005 산출이 stale 일 수 있음"
    physics_bias = selector.candidate_physics_bias(raw_cands, train_y)
    assert physics_bias.shape == (27,), f"shape: {physics_bias.shape}"
    assert physics_bias.dtype == np.float32, f"dtype: {physics_bias.dtype}"
    assert abs(physics_bias.mean()) < 1e-5, f"not centered: mean={physics_bias.mean()}"

    score = (PHYSICS_WEIGHT * physics_bias[None, :]).astype(np.float32)
    score_E = np.broadcast_to(score, (len(train_y), 27))

    r_hit = selector.R_HIT
    assert abs(r_hit - 0.01) < 1e-9, f"R_HIT drift: {r_hit}"

    E_raw       = _measure(raw_cands,       score_E, train_y, r_hit)
    E_corrected = _measure(corrected_cands, score_E, train_y, r_hit)

    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)
    per_regime: dict[int, dict] = {}
    for r in range(18):
        mask = regimes == r
        n = int(mask.sum())
        if n == 0:
            continue
        per_regime[int(r)] = {
            "n": n,
            "E_corrected_soft": float((E_corrected["_err_soft"][mask] <= r_hit).mean()),
            "E_raw_soft":       float((E_raw["_err_soft"][mask] <= r_hit).mean()),
        }
    assert sum(v["n"] for v in per_regime.values()) == len(train_y), \
        f"per_regime n sum mismatch"

    score_F = np.zeros((len(train_y), 27), dtype=np.float32)
    soft_F = selector.soft_select(corrected_cands, score_F, temperature=SOFT_TEMP)
    err_F = np.linalg.norm(soft_F - train_y, axis=1)
    F_corrected_soft = float((err_F <= r_hit).mean())

    for d in (E_raw, E_corrected):
        d.pop("_err_argmax", None); d.pop("_err_soft", None)
    summary: dict = {
        "E_raw":       E_raw,
        "E_corrected": E_corrected,
        "per_regime":  per_regime,
        "F_corrected_soft": F_corrected_soft,
        "physics_bias_argmax_idx":  int(physics_bias.argmax()),
        "physics_bias_argmax_name": str(selector.CANDIDATES[int(physics_bias.argmax())].name),
        "physics_bias_top5_names": [
            str(selector.CANDIDATES[i].name)
            for i in np.argsort(-physics_bias)[:5].tolist()
        ],
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "variant_e_oof.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def stage2_submission_csv() -> Path:
    """Variant E test inference + submission.csv 생성."""
    test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
    z = np.load(PLAN005_DIR / "corrected_test.npz")
    corrected_test = z["corrected"]

    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1
    train_cands = selector.make_candidates(train_x, end_idx, horizon=2)
    physics_bias = selector.candidate_physics_bias(train_cands, train_y)

    score = np.broadcast_to(
        (PHYSICS_WEIGHT * physics_bias[None, :]).astype(np.float32),
        (len(test_ids), 27),
    )
    pred = selector.soft_select(corrected_test, score, temperature=SOFT_TEMP)

    ref = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    assert ref.shape[1] == 4, f"sample_submission columns != 4: {ref.shape[1]}"
    id_col_name = ref.columns[0]
    coord_col_names = list(ref.columns[1:])
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame({
        id_col_name:        test_ids,
        coord_col_names[0]: pred[:, 0],
        coord_col_names[1]: pred[:, 1],
        coord_col_names[2]: pred[:, 2],
    })
    out_path = RUN_DIR / "submission.csv"
    sub_df.to_csv(out_path, index=False)

    sub = pd.read_csv(out_path)
    assert sub.shape == ref.shape, f"shape: sub={sub.shape} ref={ref.shape}"
    assert list(sub.columns) == list(ref.columns), "columns mismatch"
    assert (sub.iloc[:, 0].astype(str).values == ref.iloc[:, 0].astype(str).values).all(), \
        "id order mismatch"
    coord_cols = sub.columns[1:]
    coord_numeric = sub[coord_cols].apply(pd.to_numeric, errors="coerce")
    assert coord_numeric.notna().all().all(), "non-numeric coord"
    assert np.isfinite(coord_numeric.to_numpy()).all(), "non-finite coord (NaN/±inf)"
    return out_path


def _write_md(summary: dict) -> None:
    """variant_e_oof.md 박제 (해석용)."""
    e_c = summary["E_corrected"]
    e_r = summary["E_raw"]
    lines: list[str] = []
    lines.append("# plan-006 Variant E OOF Hit (physics_bias × 0.65 + soft averaging, temp=0.03)\n")
    lines.append("## 전체 hit\n")
    lines.append(f"- **E_corrected.soft** = `{e_c['soft']:.4f}` (main metric)")
    lines.append(f"- E_corrected.argmax = `{e_c['argmax']:.4f}` (informational, score sample-invariant)")
    lines.append(f"- E_raw.soft = `{e_r['soft']:.4f}`")
    lines.append(f"- E_raw.argmax = `{e_r['argmax']:.4f}`")
    lines.append(f"- **F_corrected.soft** (uniform sanity) = `{summary['F_corrected_soft']:.4f}`")
    lines.append("")
    lines.append("## 비교 (plan-005 측정/추정)\n")
    lines.append("| Variant | OOF hit (soft) | 출처 |")
    lines.append("|---|---|---|")
    lines.append("| full (GRU + physics + regime) | 0.6599 | plan-005 측정 |")
    lines.append("| Variant A (GRU + physics, no regime) | 0.6570 | plan-005 측정 |")
    lines.append("| Variant B (physics + regime, no GRU) | 0.6547 | plan-005 측정 |")
    lines.append(f"| **Variant E (physics 만, no GRU/regime)** | **{e_c['soft']:.4f}** | **plan-006 측정** |")
    lines.append(f"| Variant E (raw cands, no corrector) | {e_r['soft']:.4f} | plan-006 informational |")
    lines.append(f"| Variant F (uniform, no physics) | {summary['F_corrected_soft']:.4f} | plan-006 sanity |")
    lines.append("")
    lines.append("## physics_bias 해석\n")
    lines.append(f"- argmax 후보: **`{summary['physics_bias_argmax_name']}`** (idx={summary['physics_bias_argmax_idx']})")
    lines.append("- top-5: " + ", ".join(f"`{n}`" for n in summary["physics_bias_top5_names"]))
    lines.append("")
    lines.append("## Per-regime hit\n")
    lines.append("| regime | n | E_corrected_soft | E_raw_soft |")
    lines.append("|---|---|---|---|")
    for r in sorted(summary["per_regime"].keys(), key=int):
        v = summary["per_regime"][r]
        lines.append(f"| {r} | {v['n']} | {v['E_corrected_soft']:.4f} | {v['E_raw_soft']:.4f} |")
    lines.append("")
    (ANALYSIS_DIR / "variant_e_oof.md").write_text("\n".join(lines))


def stage3_log_lb(
    *,
    is_submitted: bool,
    lb_score: float | None,
    detail: str = "OK",
    exp_id: str = "E001_minimal-variant-e",
) -> None:
    """analysis/plan-006/lb_log.md 박제. existing 시 row append."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ANALYSIS_DIR / "lb_log.md"
    kst = timezone(timedelta(hours=9))
    ts = datetime.now(kst).isoformat(timespec="seconds")
    if lb_score is None and is_submitted:
        lb_str = "TBD"
    elif lb_score is None:
        lb_str = "null"
    else:
        lb_str = f"{lb_score:.4f}"
    if len(detail) > 80:
        detail = detail[:77] + "..."
    new_row = f"| {ts} | {exp_id} | {str(is_submitted).lower()} | {lb_str} | {detail} |"
    if log_path.exists():
        content = log_path.read_text().rstrip() + "\n" + new_row + "\n"
    else:
        content = (
            "# plan-006 LB 제출 log\n\n"
            "| timestamp_kst | exp_id | isSubmitted | lb_score | detail |\n"
            "|---|---|---|---|---|\n"
            + new_row + "\n"
        )
    log_path.write_text(content)


if __name__ == "__main__":
    import sys
    verify_inputs()
    print("[G0] verify_inputs OK", flush=True)
    s = stage1_variant_e_oof()
    _write_md(s)
    print(f"[G1] stage1 OK: E_corrected.soft={s['E_corrected']['soft']:.4f} "
          f"E_raw.soft={s['E_raw']['soft']:.4f} F={s['F_corrected_soft']:.4f}", flush=True)
    p = stage2_submission_csv()
    print(f"[G2] stage2 OK: {p}", flush=True)
    sys.exit(0)
