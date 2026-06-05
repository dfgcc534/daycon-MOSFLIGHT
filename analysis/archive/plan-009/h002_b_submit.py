"""plan-009 c16-prep: H002 sub-exp b test inference + submission CSV.

§10.1 best Phase 선정 anchor — H002 sub-exp b (band-specific, OOF 0.6653,
plan-008 baseline 대비 +0.0150 real gain).

작업:
- corrector_strengthen.py 의 setup_extended_pool / band_specific_corrector_loss
  / make_args reuse.
- boundary checkpoint sub_b/boundary_sub_b.pt load.
- test cands + cf 준비 (plan-008 c7 boundary.main 의 --make-test 패턴).
- predict_corrected_candidates 호출.
- soft + argmax submission CSV 2 variants 생성.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.pb_0_6822 import boundary  # noqa: E402
from src.pb_0_6822 import selector  # noqa: E402

# analysis/plan-009/ 디렉토리에 `-` → Python 식별자 invalid. importlib 동적 import.
import importlib.util  # noqa: E402
_cs_spec = importlib.util.spec_from_file_location(
    "corrector_strengthen", REPO / "analysis/plan-009/corrector_strengthen.py"
)
_cs = importlib.util.module_from_spec(_cs_spec)
_cs_spec.loader.exec_module(_cs)
setup_extended_pool = _cs.setup_extended_pool
band_specific_corrector_loss = _cs.band_specific_corrector_loss
make_args = _cs.make_args

DATA_ROOT = REPO / "data"
H001_RUN = REPO / "runs/baseline/H001_ranking-loss"
H002_RUN = REPO / "runs/baseline/H002_corrector-strengthen"
OUT_DIR = REPO / "analysis/plan-009"
SUB_EXP = "b"


def main() -> int:
    print("[plan-009 c16-prep: H002 sub-exp b test submission] start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    setup_extended_pool()
    args = make_args(SUB_EXP)

    # ★ band override (b)
    original_hook = boundary.compute_corrector_loss
    boundary.compute_corrector_loss = band_specific_corrector_loss
    print("  [override] boundary.compute_corrector_loss = band_specific")

    try:
        # ── train data prep (cm/cs 재현 필요) ──
        selector.set_torch_seed(args.seed)
        ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
        train_x = selector.load_stack(DATA_ROOT / "train", ids)
        fold_ids = np.asarray([selector.stable_fold_id(s, args.folds) for s in ids])
        tr = fold_ids != args.fold

        final_cf3, *_ = boundary.make_rows(
            train_x[tr], train_y[tr], train_x.shape[1] - 1, 2,
            cap=args.cap, low=args.low, high=args.high, far_weight=args.far_weight,
        )
        _, _, cm, cs = selector.normalize_fit(
            np.zeros((1, 6, len(selector.SEQ_FEATURE_NAMES)), dtype=np.float32),
            final_cf3,
        )
        print(f"[norm] cm.shape={cm.shape if hasattr(cm,'shape') else type(cm)}  cs.shape={cs.shape if hasattr(cs,'shape') else type(cs)}")

        # ── test data prep ──
        test_ids = selector.read_submission_ids(DATA_ROOT / "sample_submission.csv")
        test_x = selector.load_stack(DATA_ROOT / "test", test_ids)
        test_end_idx = test_x.shape[1] - 1
        test_cands = selector.make_candidates(test_x, test_end_idx, horizon=2)
        test_cf3 = selector.make_candidate_features(test_x, test_end_idx, test_cands, horizon=2)
        test_cf3 = ((test_cf3 - cm) / cs).astype(np.float32)
        tt, tn, tb, test_speed = boundary.local_frame(test_x, test_end_idx)
        test_scale = np.maximum(test_speed * 2.0, selector.EPS)
        n_test = len(test_ids)
        print(f"[test] n_test={n_test}  cands.shape={test_cands.shape}")

        # ── test selector scores (from H001) ──
        test_score_bank = np.load(H001_RUN / "test_selector_scores.npz", allow_pickle=True)
        bank_test_cands = test_score_bank["cands"]
        test_scores = test_score_bank["ens_scores"].astype(np.float32)
        max_delta = float(np.max(np.abs(bank_test_cands - test_cands)))
        assert max_delta < 1e-5, f"test cand mismatch: {max_delta}"
        assert test_scores.shape == (n_test, 25), f"test scores shape {test_scores.shape}"

        # ── Model load (TinyCorrectionNet default — sub-exp b uses default arch) ──
        in_dim = test_cf3.shape[-1]
        model = boundary.TinyCorrectionNet(in_dim, args.hidden).to(device)
        ckpt_path = H002_RUN / "fold_0/sub_b/boundary_sub_b.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"[load] {ckpt_path.relative_to(REPO)}")

        # ── Predict corrected test cands ──
        corrected_test = boundary.predict_corrected_candidates(
            model, test_cf3, test_cands, (tt, tn, tb), test_scale, args, device,
        )
        print(f"[predict] corrected_test.shape={corrected_test.shape}")

        # ── Submission write (soft + argmax) ──
        soft_pred = selector.soft_select(corrected_test, test_scores, temperature=0.03)
        argmax_idx = test_scores.argmax(axis=1)
        argmax_pred = corrected_test[np.arange(n_test), argmax_idx]

        soft_file = H002_RUN / "submission_boundary_tiny_soft.csv"
        arg_file = H002_RUN / "submission_boundary_tiny_argmax.csv"
        step_file = H002_RUN / "submission_step2.csv"  # plan-009 best Phase anchor

        selector.write_submission(soft_file, test_ids, soft_pred)
        selector.write_submission(arg_file, test_ids, argmax_pred)
        step_file.write_bytes(soft_file.read_bytes())
        print(f"[OK] submission_boundary_tiny_soft.csv: {soft_file.relative_to(REPO)}")
        print(f"[OK] submission_boundary_tiny_argmax.csv: {arg_file.relative_to(REPO)}")
        print(f"[OK] submission_step2.csv (= soft, plan-009 best Phase anchor): {step_file.relative_to(REPO)}")

        # Summary
        summary = {
            "exp_id": "H002_corrector-strengthen/sub_b/test",
            "sub_exp": SUB_EXP,
            "checkpoint": str(ckpt_path.relative_to(REPO)),
            "test_score_source": "H001_ranking-loss/test_selector_scores.npz (ens_scores)",
            "submissions": {
                "soft": str(soft_file.relative_to(REPO)),
                "argmax": str(arg_file.relative_to(REPO)),
                "step2_canonical": str(step_file.relative_to(REPO)),
            },
            "n_test": n_test,
            "decision_note": (
                "plan-009 best Phase submission anchor. soft = boundary_tiny_soft.csv "
                "기준 (plan-008 c7 의 submission_step3.csv pattern 답습). "
                "H002 b OOF 0.6653 → estimated LB ≈ 0.6873 (OOF + 0.022 gap, "
                "plan-009 §10.2 시나리오 C 영역 0.69~0.72 의 상단)."
            ),
        }
        out_summary = OUT_DIR / "h002_b_submit.json"
        out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"[OK] {out_summary.relative_to(REPO)}")
    finally:
        boundary.compute_corrector_loss = original_hook
        print("  [restore] boundary.compute_corrector_loss = original")

    print("[plan-009 c16-prep H002 sub-exp b test submission] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
