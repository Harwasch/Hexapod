"""Lazy measured-depth supervision for gsplat 512d366b; Python 3.10 compatible.

Metric L1 is intentional: ARKit supplies scale, so scale-invariant loss would discard
its useful constraint. All reductions remain tensors; no per-step device sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def patch_source(source: str) -> str:
    if "    arkit_depth_loss: bool" in source:
        raise ValueError("ARKit depth trainer already patched")
    anchors = (
        "    depth_loss: bool = False",
        "        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale",
        'render_mode="RGB+ED" if cfg.depth_loss else "RGB",',
        "            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)",
    )
    for anchor in anchors:
        if source.count(anchor) != 1:
            raise ValueError(f"ARKit depth trainer seam moved: expected one {anchor!r}")
    source = source.replace(
        anchors[0],
        anchors[0]
        + "\n    arkit_depth_loss: bool = False"
        + "\n    arkit_depth_weight: float = 0.01"
        + "\n    arkit_depth_warmup: int = 1000",
    )
    source = source.replace(
        anchors[1],
        "        import depth_trainer as _hex_depth\n"
        "        if cfg.arkit_depth_loss:\n"
        "            _hex_depth.configure(self)\n" + anchors[1],
    )
    source = source.replace(
        anchors[2],
        'render_mode="RGB+ED" if (cfg.depth_loss or (cfg.arkit_depth_loss and '
        'step >= cfg.arkit_depth_warmup)) else "RGB",',
    )
    return source.replace(
        anchors[3],
        anchors[3] + "\n            if cfg.arkit_depth_loss and step >= cfg.arkit_depth_warmup:\n"
        "                import depth_trainer as _hex_depth\n"
        "                loss = loss + _hex_depth.depth_loss(\n"
        "                    depths, data['arkit_depth'].to(device, non_blocking=True),\n"
        "                    step, cfg.arkit_depth_warmup, cfg.arkit_depth_weight)\n",
    )


class DepthDataset:
    """Wrap the already-split training dataset. One low-res file per getitem, no cache."""

    def __init__(self, dataset: Any, directory: Path) -> None:
        self.dataset = dataset
        self.directory = directory

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch

        data: dict[str, Any] = self.dataset[item]
        name = self.parser.image_names[self.indices[item]]
        path = self.directory / (name + ".npy")
        if path.is_file():
            depth = np.load(path, allow_pickle=False)
        else:
            # Missing measurements contribute zero; fixed shape for batch_size=1.
            depth = np.zeros((1, 1), dtype=np.float32)
        data["arkit_depth"] = torch.from_numpy(depth)
        return data


def configure(runner: Any) -> None:
    cfg = runner.cfg
    if cfg.normalize_world_space or cfg.patch_size is not None or cfg.batch_size != 1:
        raise ValueError("ARKit depth requires metric world, full images and batch_size=1")
    directory = Path(cfg.data_dir) / "arkit-depth-priors"
    if not directory.is_dir():
        raise ValueError("ARKit depth loss requested without prepared train-only priors")
    runner.trainset = DepthDataset(runner.trainset, directory)


def depth_loss(rendered: Any, target: Any, step: int, warmup: int, weight: float) -> Any:
    import torch
    import torch.nn.functional as functional

    if step < warmup:
        return rendered.sum() * 0.0
    predicted = functional.interpolate(
        rendered.permute(0, 3, 1, 2), size=target.shape[-2:], mode="bilinear", align_corners=False
    )[:, 0]
    valid = torch.isfinite(target) & (target > 0)
    clean = torch.where(valid, target, torch.zeros_like(target))
    residual = torch.where(valid, (predicted - clean).abs(), torch.zeros_like(predicted))
    return weight * residual.sum() / valid.sum().clamp_min(1)
