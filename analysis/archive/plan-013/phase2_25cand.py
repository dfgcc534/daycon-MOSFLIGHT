"""plan-013 c7 — Phase 2.E3 (25 cand redesign) — DEFERRED.

spec @ plan-013 §7.3.

DEFER 사유: G0 preflight (c3) 의 cand_25_infra MISS — plan-008 G1 `cand_set.{json,npy}` 미존재.
G001_candidate-redefine 디렉토리에는 selector/boundary 산출만 있고 candidate descriptor list 가
별도 박제 안 됨. 25 cand swap 자체가 불가능 → 측정 무의미.

자율 결정: §0.5 L95 phase2_no_positive_lever + cand_25_infra MISS → Phase 3 fallback path 진입.

산출: analysis/plan-013/phase2_25cand.json (deferred 박제).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="plan-013 c7 Phase 2.E3 — deferred")
    parser.add_argument("--out", default="analysis/plan-013/phase2_25cand.json")
    args = parser.parse_args()

    result = {
        "exp_id": "H034_phase2-25cand-redesign",
        "status": "deferred",
        "config_intended": {
            "use_in_ic": True,
            "use_step4": "off",
            "use_25_cand": True,
        },
        "delta_oof": None,
        "delta_oof_threshold_positive_lever": 0.005,
        "positive_lever": False,
        "defer_reason": (
            "G0 preflight cand_25_infra MISS — plan-008 G1 cand_set.{json,npy} 미존재. "
            "G001_candidate-redefine 디렉토리에 selector/boundary 산출만 있고 candidate descriptor list 별도 박제 X."
        ),
        "carry_over_to_plan_013_1": (
            "plan-008 G1 의 25 candidate descriptor 를 cand_set.json (or .npy) 으로 별도 박제 후 "
            "selector head dim 27→25 swap + 5-fold 재측정. retain risk note (§7.3) 도 함께 검증."
        ),
        "fallback_path": "§0.5 L95 phase2_no_positive_lever + cand_25_infra MISS → Phase 3 fallback.",
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[phase2.E3] saved: {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
