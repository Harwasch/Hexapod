"""`inpaint_models`: the inpainting models the GPU side may load."""

from __future__ import annotations

import inpaint_models


def test_every_inpainting_model_is_one_a_product_may_run() -> None:
    # Each listed model's licence allows commercial use (the module docstring's table);
    # a model whose weights forbid it is not listed at all, not even for comparison.
    assert set(inpaint_models.MODELS) == {"sdxl", "qwen"}
