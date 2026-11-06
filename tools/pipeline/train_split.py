"""Explicit no-holdout support for the pinned gsplat Dataset and Runner seam.

The upstream modulo split cannot express zero validation images. At test_every=0
we select every index for train and none for val, and skip val evaluation. A positive
value retains upstream behavior. Bench's external test set never enters this dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def install(module: dict[str, Any], runner: Any, cfg: Any) -> None:
    if cfg is None or getattr(cfg, "test_every", 8) != 0:
        return
    dataset = module.get("Dataset")
    if dataset is None or runner is None:
        raise ValueError("test_every=0 needs the pinned Dataset/Runner seam")

    class AllTraining(dataset):  # type: ignore[misc,valid-type]
        def __init__(self, parser: Any, split: str = "train", **kwargs: Any) -> None:
            original = parser.test_every
            parser.test_every = 1
            try:
                super().__init__(parser, split=split, **kwargs)
            finally:
                parser.test_every = original
            self.indices = (
                np.arange(len(parser.image_names)) if split == "train" else np.array([], dtype=int)
            )

    module["Dataset"] = AllTraining

    def no_evaluation(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    runner.eval = no_evaluation
