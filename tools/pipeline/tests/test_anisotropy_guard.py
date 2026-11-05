"""Log-space anisotropy stays finite and caps scales only when enabled."""

import numpy as np
import pytest

import prior_maths


def test_log_anisotropy_has_bounded_gradient_even_at_huge_ratio() -> None:
    x = np.array([[1000.0, 0.0, -1000.0]])
    assert prior_maths.aniso_penalty(x) == pytest.approx(2000 - np.log(10))
    h = 1e-4
    shifted = x.copy()
    shifted[0, 0] += h
    assert (prior_maths.aniso_penalty(shifted) - prior_maths.aniso_penalty(x)) / h == pytest.approx(
        1
    )
    assert prior_maths.aniso_penalty([[0, 1, 0]]) == 0


def test_scale_guard_uses_extent_and_leaves_small_axes_unchanged() -> None:
    values = np.array([[-20.0, 1.0, 1000.0]])
    guarded = prior_maths.guard_log_scales(values, extent=20, share=0.05)
    np.testing.assert_array_equal(guarded, [[-20, 0, 0]])
    np.testing.assert_array_equal(values, [[-20, 1, 1000]])


def test_disabled_scale_guard_never_imports_torch_or_changes_runner() -> None:
    from types import SimpleNamespace

    import quality_trainer

    # No splats or optimizer is needed when this feature is off.
    quality_trainer.guard_scales(SimpleNamespace(hex_settings={"aniso_weight": 0}))


def test_scale_guard_rejects_invalid_extent_or_share() -> None:
    for extent, share in ((0, 0.05), (1, 0), (1, 2), (float("nan"), 0.05)):
        with pytest.raises(ValueError):
            prior_maths.guard_log_scales([[0, 1, 2]], extent=extent, share=share)
