"""Not gsplat's `examples/datasets/colmap.py`. The same `Parser` / `Dataset` surface
`lod_optimise.py` reads, over a `cameras.npz` instead of a COLMAP model: `names`,
`camtoworlds` (N, 4, 4, COLMAP's frame), `K` (3, 3) and `images` (N, H, W, 3) uint8.

The split is gsplat v1.5.3's: indices `i % test_every == 0` are `val`, the rest `train`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class Parser:
    def __init__(
        self, data_dir: str, factor: int = 1, normalize: bool = False, test_every: int = 8
    ) -> None:
        assert factor == 1 and not normalize
        data = np.load(Path(data_dir) / "cameras.npz")
        self.image_names = [str(name) for name in data["names"]]
        self.camtoworlds = data["camtoworlds"].astype(np.float64)
        self.K = data["K"].astype(np.float64)
        self.images = data["images"]
        self.test_every = test_every


class Dataset:
    def __init__(self, parser: Parser, split: str = "train") -> None:
        self.parser = parser
        indices = np.arange(len(parser.image_names))
        if split == "train":
            self.indices = indices[indices % parser.test_every != 0]
        else:
            self.indices = indices[indices % parser.test_every == 0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "K": torch.from_numpy(self.parser.K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "image": torch.from_numpy(np.ascontiguousarray(self.parser.images[index])),
            "image_id": item,
        }
