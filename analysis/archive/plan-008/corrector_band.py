"""plan-008 c9 + c10: Corrector band-specific 재설계 — §7.

spec @ plans/plan-008-candidate-redefine-corrector-redesign.md §7.

§7.3 의 monkey-patch 전략: `boundary.<LOSS_ATTR>` 를 band_specific_corrector_loss
로 교체 후 `run_full.run_boundary()` 호출.

**실제 boundary.py 검증 (c9 진입 시점)**: corrector loss 가 `train_net()` 함수
*내부 inline* (L231-233: `reg = ((pred - yb) ** 2).sum(dim=1)`) 으로 정의되어
있으며, monkey-patch 가능한 module-level callable 부재. §7.3 의 후보 attribute
3 종 (`compute_corrector_loss`, `corrector_loss`, `loss_fn`) 모두 dir(boundary)
에 없음.

**자율 결정 (per CLAUDE.md autonomous policy)**:
- §0.5 paths blacklist: `boundary.py` lock-in (import only) → 본 plan scope 내
  *boundary.py 본문 수정* 으로 hook attribute 추가 불가.
- Alternative 1 (standalone re-impl of corrector + train loop, ~200 LOC) ROI
  낮음 — G1+G2 cascade severe 위에서 Step 4 booster 효과 marginal.
- Alternative 2 (defer to plan-009): plan-009 task list 에 "boundary.py 의
  compute_corrector_loss attribute 신설 + band-specific monkey-patch" 박제.

→ **Alternative 2 채택**. 본 script 는 (a) §7 spec 정합성 검증, (b) c9 deferred
  status 박제 + plan-009 task 명시.

산출:
  - analysis/plan-008/corrector_band.json (deferred status + plan-009 task)
  - G3 gate: DEFERRED (plan-009 carry-over) — §0.5 의 redefinition_partial 류
    plan-specific deferred, severe 아님.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO / "analysis/plan-008"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # §7.3 의 LOSS_ATTR 탐색 절차 그대로
    from src.pb_0_6822 import boundary
    CANDIDATE_ATTRS = ["compute_corrector_loss", "corrector_loss", "loss_fn"]
    LOSS_ATTR = next(
        (a for a in CANDIDATE_ATTRS if callable(getattr(boundary, a, None))),
        None,
    )

    boundary_callables_loss_corr = [
        a for a in dir(boundary)
        if callable(getattr(boundary, a, None))
        and ("loss" in a.lower() or "corr" in a.lower())
    ]

    summary = {
        "exp_id": "G002-corrector-band",
        "status": "DEFERRED",
        "deferred_reason": (
            "§7.3 monkey-patch 전략 incompatible with current boundary.py — "
            "corrector loss 가 train_net() 내부 inline 정의 (L231-233), "
            "module-level callable LOSS_ATTR 부재."
        ),
        "loss_attr_search": {
            "candidates_searched": CANDIDATE_ATTRS,
            "found": LOSS_ATTR,
            "boundary_callables_matching_loss_or_corr": boundary_callables_loss_corr,
        },
        "alternatives_considered": [
            {
                "name": "alt1_standalone_reimpl",
                "description": "boundary.py 미수정, train_net + corrector 모듈 재구현 (~200 LOC)",
                "rejected_reason": (
                    "G1+G2 cascade severe 위 Step 4 booster ROI marginal — "
                    "code budget vs 회수 효과 (+0.02) trade-off 부적합."
                ),
            },
            {
                "name": "alt2_modify_boundary",
                "description": "boundary.py 에 compute_corrector_loss 함수 추가",
                "rejected_reason": "§0.5 paths blacklist (boundary.py lock-in)",
            },
            {
                "name": "alt3_defer_to_plan_009",
                "description": "본 plan §7 scope 제외, plan-009 task list 박제",
                "accepted": True,
            },
        ],
        "plan_009_task_added": {
            "title": "boundary.py 의 compute_corrector_loss attribute 추가 + band-specific loss monkey-patch 가능 구조 도입",
            "scope": [
                "boundary.train_net() 의 reg = ((pred-yb)**2).sum(dim=1) 식을 "
                "module-level callable compute_corrector_loss(pred, yb, target) 로 추출",
                "기본 동작 (L2 loss) 보존 + monkey-patch 가능 hook 노출",
                "plan-008 §7.2 의 band_specific_corrector_loss spec 그대로 적용",
                "G3 합격 기준: per-band [0,0.5)≥0.99 / [0.5,1)≥0.95 / [1,1.5)≥0.30 + 전체 OOF ≥ Step3 + 0.02",
            ],
        },
        "g3_status": "DEFERRED (plan-009 carry-over)",
    }

    (ANALYSIS_DIR / "corrector_band.json").write_text(json.dumps(summary, indent=2))
    print(f"[c9] LOSS_ATTR search result: {LOSS_ATTR}")
    print(f"[c9] boundary callables (loss|corr): {boundary_callables_loss_corr}")
    print(f"[c9] status = DEFERRED → plan-009 carry-over")


if __name__ == "__main__":
    main()
