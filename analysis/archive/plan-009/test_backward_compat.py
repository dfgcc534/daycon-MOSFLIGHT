"""plan-009 c6 backward_compat verify — caveat #7 박제.

Spec @ §N+3 caveat #7 (plans/plan-009-selector-ranking-loss.md).

Test 내용:
  (i) compute_corrector_loss(pred, yb) 2 인자 호출 결과 ==
      ((pred - yb) ** 2).sum(dim=1)  with torch.allclose(atol=1e-7).
  (ii) plan-004/005 c-step reproduce — sanity_baseline_27.json 의
       oof_soft_hit ±1e-4 일치 검증. (시간 한계 회피로 c6 시점 skip,
       caveat #7 의 (ii) 는 c8 G2 학습 시점에 effective verify —
       boundary 학습 결과가 plan-008 c7 와 일치하면 hook 의 default
       path backward compat 검증.)

(i) unit test 만 실행. (ii) 는 c8 학습 결과 검증.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.pb_0_6822 import boundary  # noqa: E402


def test_default_l2_equivalence() -> None:
    """(i) compute_corrector_loss default == per-sample squared L2 sum."""
    torch.manual_seed(42)
    B, D = 64, 3
    pred = torch.randn(B, D, dtype=torch.float32)
    target = torch.randn(B, D, dtype=torch.float32)

    # plan-009 hook
    reg_hook = boundary.compute_corrector_loss(pred, target)

    # 기존 form
    reg_old = ((pred - target) ** 2).sum(dim=1)

    assert reg_hook.shape == (B,), f"hook reg shape {reg_hook.shape} != (B={B},)"
    assert reg_hook.shape == reg_old.shape, f"hook {reg_hook.shape} vs old {reg_old.shape}"
    assert torch.allclose(reg_hook, reg_old, atol=1e-7), (
        f"hook reg != old reg: max diff = {(reg_hook - reg_old).abs().max().item()}"
    )
    print(f"[OK] (i) compute_corrector_loss default == ((pred-target)**2).sum(dim=1) "
          f"(B={B}, D={D}, max_diff={(reg_hook - reg_old).abs().max().item():.2e})")


def test_weight_argument() -> None:
    """weight 인자 통과 시 = reg * weight."""
    torch.manual_seed(7)
    B, D = 32, 3
    pred = torch.randn(B, D, dtype=torch.float32)
    target = torch.randn(B, D, dtype=torch.float32)
    weight = torch.rand(B, dtype=torch.float32)

    reg_w = boundary.compute_corrector_loss(pred, target, weight=weight)
    reg_no_w = boundary.compute_corrector_loss(pred, target)
    expected = reg_no_w * weight
    assert torch.allclose(reg_w, expected, atol=1e-7), (
        f"weight 인자 위배: max diff = {(reg_w - expected).abs().max().item()}"
    )
    print(f"[OK] weight 인자: reg_w = reg * weight ✓")


def test_monkey_patch_lookup() -> None:
    """boundary.compute_corrector_loss = <new> 후 module-attribute lookup 동적 patch 검증."""
    original = boundary.compute_corrector_loss

    def _doubled(pred, target, raw=None, weight=None):
        return ((pred - target) ** 2).sum(dim=1) * 2.0

    boundary.compute_corrector_loss = _doubled
    try:
        torch.manual_seed(11)
        pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        reg_patched = boundary.compute_corrector_loss(pred, target)
        expected = ((pred - target) ** 2).sum(dim=1) * 2.0
        assert torch.allclose(reg_patched, expected, atol=1e-7), "patched hook 미작동"
        print("[OK] monkey-patch lookup: boundary.compute_corrector_loss override → reg × 2.0 적용 ✓")
    finally:
        boundary.compute_corrector_loss = original
    # restore 후 default 동작 보존 검증
    reg_post = boundary.compute_corrector_loss(torch.zeros(4, 3), torch.zeros(4, 3))
    assert torch.allclose(reg_post, torch.zeros(4)), "restore 후 default 동작 위배"
    print("[OK] restore: monkey-patch 해제 후 default 복귀 ✓")


def main() -> int:
    print("[plan-009 c6 backward_compat test] start (caveat #7 (i) 단위 unit test)")
    test_default_l2_equivalence()
    test_weight_argument()
    test_monkey_patch_lookup()
    print("[plan-009 c6 backward_compat test] all passed.")
    print("note: (ii) plan-004/005 oof_soft_hit ±1e-4 일치는 c8 G2 학습 결과로 effective verify "
          "(sub-exp 0 baseline 의 OOF 가 plan-008 c7 boundary 학습 결과와 일관 시 통과).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
