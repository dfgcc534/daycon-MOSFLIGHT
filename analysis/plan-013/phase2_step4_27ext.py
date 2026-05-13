"""plan-013 c6 — Phase 2.E2 (Step 4 27ext) — DEFERRED.

spec @ plan-013 §7.2.

DEFER 사유: c5 와 동일 — plan-007 basis_terms framework architectural gap. 27ext 는 추가로
candidate-conditional one-hot input (13+27=40 dim) 위 MLP 학습 필요, c5 의 basis_terms 통합
이전엔 측정 무의미.

자율 결정: §0.5 L95 phase2_no_positive_lever autonomous recovery 진입.
산출: analysis/plan-013/phase2_step4_27ext.json (deferred 박제).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="plan-013 c6 Phase 2.E2 — deferred")
    parser.add_argument("--out", default="analysis/plan-013/phase2_step4_27ext.json")
    args = parser.parse_args()

    result = {
        "exp_id": "H033_phase2-step4-27ext",
        "status": "deferred",
        "config_intended": {
            "use_in_ic": True,
            "use_step4": "27ext",
            "use_25_cand": False,
        },
        "delta_oof": None,
        "delta_oof_threshold_positive_lever": 0.005,
        "positive_lever": False,
        "defer_reason": (
            "c5 와 동일 — plan-007 basis_terms framework architectural gap. 27ext 는 추가로 "
            "candidate-conditional one-hot (13+27=40 dim) MLP 학습 필요."
        ),
        "carry_over_to_plan_013_1": (
            "plan-007 basis_terms framework + 27 candidate-conditional MLP 통합 후 재측정. "
            "step4_27ext_overfit warn (train-val gap > 0.05) check 도 함께."
        ),
        "fallback_path": "§0.5 L95 phase2_no_positive_lever autonomous recovery — Phase 3 fallback.",
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[phase2.E2] saved: {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
