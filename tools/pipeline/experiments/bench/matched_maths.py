"""Shared matched cameras, without torch or a renderer dependency."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def choose_frame(
    names: list[str], centres: npt.ArrayLike, up: npt.ArrayLike, override: str | None
) -> str:
    """Lower median altitude among frozen test cameras; filename breaks ties."""
    if not names or len(set(names)) != len(names):
        raise ValueError("need distinct frozen test names")
    if override is not None:
        if override not in names:
            raise ValueError("--frame must name a frozen test frame")
        return override
    points, vertical = np.asarray(centres, dtype=float), np.asarray(up, dtype=float)
    if (
        points.shape != (len(names), 3)
        or vertical.shape != (3,)
        or not np.isfinite(points).all()
        or not np.isfinite(vertical).all()
    ):
        raise ValueError("invalid test camera centres or up vector")
    ordered = sorted(zip(points @ vertical, names, strict=True))
    return ordered[(len(ordered) - 1) // 2][1]


def closeup_camera(camera: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    """Central quarter of the image area, rendered afresh at native crop resolution.

    Half the width/height gives a narrower FOV without changing the pixel pitch.
    Neither predictions nor cropped GT are enlarged or interpolated.
    """
    width, height = int(camera["width"]), int(camera["height"])
    if min(width, height) < 2:
        raise ValueError("close-up needs at least 2x2 pixels")
    crop_width, crop_height = width // 2, height // 2
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    intrinsics = np.array(camera["K"], dtype=float, copy=True)
    intrinsics[0, 2] -= left
    intrinsics[1, 2] -= top
    return (
        {**camera, "K": intrinsics.tolist(), "width": crop_width, "height": crop_height},
        (left, top, left + crop_width, top + crop_height),
    )
