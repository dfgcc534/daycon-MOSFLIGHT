"""plan-013 c5 — Phase 2.E1 (Step 4 F0_only) — DEFERRED.

spec @ plan-013 §7.1.

DEFER 사유:
- Step 4 F0_only 의 충실 구현은 plan-007 `mlp_coeff.py` 의 8 basis_terms 산출 (compute_trajectory_features
  + per-var (d1/acc_par/acc_perp/d2/jerk/ts_term/speed_slope_d1/rotation_term) tensor 구축) 의 *기존
  framework* 위에서 가능. plan-013 §2.1 의 'frozen reuse only' (= plan-007 mlp_coeff.py 본문 수정 X)
  와 simplified pipeline (G1 baseline = standalone residual corrector) 사이 architectural gap.
- 충실 구현하려면 plan-007 의 build_all_samples + basis_terms 산출 함수를 본 pipeline 에 통합 필요 —
  c5 단일 commit scope 초과.

자율 결정 (§0.5 L95 `phase2_no_positive_lever` autonomous recovery 의 부분 적용):
- ΔOOF(E1) = "deferred" 박제. Phase 2 의 1+ axis ≥ 0.005 조건은 deferred lever 가 informational
  contribute 0 으로 산입.
- Phase 3 (c8) 는 §0.5 L95 의 fallback path 진입: best Phase 1 baseline 단독 5-fold + submission.

산출: analysis/plan-013/phase2_step4_F0.json (deferred 박제).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="plan-013 c5 Phase 2.E1 — deferred")
    parser.add_argument("--out", default="analysis/plan-013/phase2_step4_F0.json")
    args = parser.parse_args()

    result = {
        "exp_id": "H032_phase2-step4-F0",
        "status": "deferred",
        "config_intended": {
            "use_in_ic": True,
            "use_step4": "F0_only",
            "use_25_cand": False,
        },
        "delta_oof": None,
        "delta_oof_threshold_positive_lever": 0.005,
        "positive_lever": False,
        "defer_reason": (
            "Step 4 F0_only 의 충실 구현은 plan-007 mlp_coeff.py 의 8 basis_terms 산출 framework 위에서 가능. "
            "plan-013 의 'frozen reuse only' 와 simplified pipeline 의 architectural gap — c5 scope 초과."
        ),
        "carry_over_to_plan_013_1": (
            "plan-007 basis_terms framework 를 integrated_v3 로 통합 후 Step 4 F0_only 5-fold 재측정."
        ),
        "fallback_path": "§0.5 L95 phase2_no_positive_lever autonomous recovery — Phase 3 = best G1 baseline 단독 submission.",
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[phase2.E1] saved: {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
