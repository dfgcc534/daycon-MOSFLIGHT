"""plan-009 c2: Preflight verification + lb_baseline.json 신설.

Spec @ §4.1 (plan-009-selector-ranking-loss.md):
  1. plan-008 산출 5+1 submission variant 존재 확인.
  2. plan-008 metric 4 항목 verify (analysis/plan-008/selector_retrain.json 참조).
  3. analysis/plan-009/lb_baseline.json 신설 (v1.1 spec — LB 제출 0 회 정책).

위반 시 severe `plan_008_artifact_missing` (§4.3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_008_RUN = REPO_ROOT / "runs/baseline/G001_candidate-redefine"
PLAN_008_ANALYSIS = REPO_ROOT / "analysis/plan-008"
OUT_DIR = REPO_ROOT / "analysis/plan-009"

# §4.1 박제 6 submission paths (본문은 "5 submission variant" 표현, step3 포함 6 file)
EXPECTED_SUBMISSIONS = [
    "submission_step3.csv",
    "submission_attn_gru_selector_soft.csv",
    "submission_boundary_tiny_argmax.csv",
    "submission_boundary_tiny_soft.csv",
    "submission_selector_ensemble_argmax.csv",
    "submission_selector_ensemble_soft.csv",
]

# §1.1 표의 plan-008 핵심 metric 4 항목 (plan-008 selector_retrain.json 의 field)
EXPECTED_METRICS = {
    "oof_soft_hit": 0.6503,
    "oracle_extended_pool": 0.7562,
    "top1_ranking_accuracy": 0.1721,
    "gap_ranking": 0.1119,
}
TOL = 1e-4


def verify_submissions() -> None:
    missing = []
    for fname in EXPECTED_SUBMISSIONS:
        path = PLAN_008_RUN / fname
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
    if missing:
        raise FileNotFoundError(
            f"plan_008_artifact_missing — {len(missing)} submission 부재: {missing}"
        )
    print(
        f"[OK] plan-008 submission {len(EXPECTED_SUBMISSIONS)} 개 모두 존재 "
        f"(§4.1 박제 그대로)."
    )


def verify_metrics() -> dict:
    path = PLAN_008_ANALYSIS / "selector_retrain.json"
    if not path.exists():
        raise FileNotFoundError(f"plan_008_artifact_missing — {path} 부재")
    d = json.loads(path.read_text())
    actual = {k: d.get(k) for k in EXPECTED_METRICS}
    mismatched = []
    for k, expected in EXPECTED_METRICS.items():
        v = actual[k]
        if v is None or abs(v - expected) > TOL:
            mismatched.append({"key": k, "actual": v, "expected": expected})
    if mismatched:
        raise ValueError(
            f"plan_008_metric_mismatch — {len(mismatched)} 항목 불일치 (tol={TOL}): {mismatched}"
        )
    print(f"[OK] plan-008 metric 4 항목 verify (tol={TOL}): {actual}")
    return actual


def write_lb_baseline(metrics: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_id": "009",
        "version": "v1.1 spec (LB 제출 0 회 정책 — 할당량 소진 인계)",
        "based_on": ["008"],
        "baseline_submission_path": "runs/baseline/G001_candidate-redefine/submission_step3.csv",
        "baseline_metrics": {
            "oof_soft_hit": metrics["oof_soft_hit"],
            "oracle_extended_pool": metrics["oracle_extended_pool"],
            "top1_ranking_accuracy": metrics["top1_ranking_accuracy"],
            "gap_ranking": metrics["gap_ranking"],
            "n_train": 10000,
            "n_candidates_extended": 25,
            "n_oracle_miss_extended": 2812,
        },
        "carry_over": {
            "plan_008_1": {
                "submission_path": "runs/baseline/G001_candidate-redefine/submission_step3.csv",
                "expected_lb": None,
                "anchor": "OOF + 0.022 gap (plan-005/008 trajectory)",
                "estimated_lb": round(metrics["oof_soft_hit"] + 0.022, 4),
                "status": "pending dacon-submit (할당량 소진, 다음 날 사용자 수동 호출)",
            },
            "plan_009_1": {
                "submission_path": None,
                "anchor": "best Phase submission of plan-009 (TBD at G_final)",
                "estimated_lb": None,
                "status": "pending plan-009 completion",
            },
        },
        "lb_history_anchor": {
            "plan_005_stage6_variant_a_lb": 0.6796,
            "note": "plan-009 §N+5 박제. OOF→LB gap 추정 anchor.",
        },
        "decision_note": (
            "spec-default — lb_baseline.json schema 가 plan-009 본문에 명시 부재로 "
            "자율 결정 채택 (CLAUDE.md Autonomous Execution Policy)."
        ),
    }
    out_path = OUT_DIR / "lb_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] lb_baseline.json 신설: {out_path.relative_to(REPO_ROOT)}")
    return out_path


def main() -> int:
    print("[plan-009 c2 preflight] start")
    verify_submissions()
    metrics = verify_metrics()
    out_path = write_lb_baseline(metrics)
    print(f"[plan-009 c2 preflight] complete → {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
