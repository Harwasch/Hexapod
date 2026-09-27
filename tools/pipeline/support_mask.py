"""Where the data supports a high-quality reconstruction, as a voxel mask of any shape.

The quality stage decides per gaussian whether a splat has earned its place (views,
angular spread, pixel size). This module turns that verdict into a *region*: the voxels
that hold enough well-supported splats, dilated a little so a surface's own thickness and
a refine's densification have room. A Refine trains inside it and drops what falls outside
it -- so the good data decides the shape, whether that is one object, an L of two walls,
or three separate things on a lawn. No sphere, box or picked subject is involved.

It travels as JSON (a job parameter): the grid's origin, voxel size and dimensions, and
its occupancy as zlib-compressed packed bits in base64. A mostly-empty 80^3 grid is a few
kilobytes.
"""

from __future__ import annotations

import base64
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["MAX_CELLS", "SupportMask", "build"]

#: The longest side of the grid, in voxels. Finer costs bytes in a job parameter and
#: gains nothing a 2-voxel dilation would not blur away.
MAX_CELLS = 80
#: A voxel counts as supported when it holds at least this many keep splats: one stray
#: keep splat is not a surface, and two in one voxel usually are part of one.
MIN_PER_VOXEL = 2
#: Voxels of margin added round the supported ones.
DILATE = 2
#: The most encoded bytes accepted when parsing, so a job parameter cannot be made huge.
MAX_ENCODED_BYTES = 256 * 1024


@dataclass(frozen=True)
class SupportMask:
    origin: tuple[float, float, float]
    voxel: float
    dims: tuple[int, int, int]
    occupied: npt.NDArray[np.bool_]  # shape `dims`, already dilated

    @property
    def voxels(self) -> int:
        return int(self.occupied.sum())

    def contains(self, xyz: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Whether each point falls in a supported voxel. Non-finite points never do."""
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        ijk = np.floor((np.where(finite[:, None], points, 0.0) - self.origin) / self.voxel)
        dims = np.asarray(self.dims)
        inside = finite & (ijk >= 0).all(axis=1) & (ijk < dims).all(axis=1)
        result = np.zeros(points.shape[0], dtype=bool)
        index = ijk[inside].astype(np.int64)
        result[inside] = self.occupied[index[:, 0], index[:, 1], index[:, 2]]
        return result

    def to_dict(self) -> dict[str, object]:
        packed = np.packbits(self.occupied.reshape(-1))
        return {
            "kind": "voxels",
            "frame": "colmap",
            "origin": [round(v, 6) for v in self.origin],
            "voxel": round(self.voxel, 9),
            "dims": list(self.dims),
            "voxels": self.voxels,
            "bits": base64.b64encode(zlib.compress(packed.tobytes(), 9)).decode("ascii"),
        }

    @staticmethod
    def parse(value: object) -> SupportMask | None:
        """The dict `to_dict` wrote, checked; None for no mask. Anything malformed is
        refused by name, because a mask silently ignored is an uncropped run."""
        if value is None:
            return None
        if not isinstance(value, Mapping) or value.get("kind") != "voxels":
            raise ValueError("support_mask must be a voxel mask written by the quality stage")
        origin = value.get("origin")
        dims = value.get("dims")
        voxel: Any = value.get("voxel")
        bits = value.get("bits")
        if (
            not isinstance(origin, list)
            or len(origin) != 3
            or not all(_finite(v) for v in origin)
            or not isinstance(dims, list)
            or len(dims) != 3
            or not all(
                isinstance(d, int) and not isinstance(d, bool) and 0 < d <= 4 * MAX_CELLS
                for d in dims
            )
            or not _finite(voxel)
            or float(voxel) <= 0
            or not isinstance(bits, str)
            or len(bits) > MAX_ENCODED_BYTES
        ):
            raise ValueError("support_mask has a malformed origin, voxel, dims or bits")
        cells = int(np.prod(dims))
        # Bounded decompression: the packed grid is at most cells/8 bytes, so a parameter
        # cannot expand into gigabytes.
        inflater = zlib.decompressobj()
        packed = inflater.decompress(base64.b64decode(bits), (cells + 7) // 8 + 1)
        raw = np.frombuffer(packed, dtype=np.uint8)
        occupied = np.unpackbits(raw)[:cells]
        if occupied.size != cells:
            raise ValueError("support_mask's bits do not fill its dims")
        return SupportMask(
            origin=(float(origin[0]), float(origin[1]), float(origin[2])),
            voxel=float(voxel),
            dims=(int(dims[0]), int(dims[1]), int(dims[2])),
            occupied=occupied.astype(bool).reshape(dims),
        )


def build(
    xyz: npt.ArrayLike,
    *,
    max_cells: int = MAX_CELLS,
    min_per_voxel: int = MIN_PER_VOXEL,
    dilate: int = DILATE,
) -> SupportMask | None:
    """The supported region of `xyz` (the keep tier's centres), or None when too few.

    The grid spans the points' 0.5-99.5 percentile box (plus the dilation), so one far
    stray does not stretch the voxels coarse; strays outside it are simply not supported.
    """
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 32:
        return None
    low = np.percentile(points, 0.5, axis=0)
    high = np.percentile(points, 99.5, axis=0)
    span = float(np.max(high - low))
    if not math.isfinite(span) or span <= 0:
        return None
    voxel = span / max(1, max_cells - 2 * dilate)
    origin = low - dilate * voxel
    dims = np.minimum(np.ceil((high - low) / voxel).astype(int) + 1 + 2 * dilate, 4 * max_cells)
    ijk = np.floor((points - origin) / voxel).astype(np.int64)
    inside = (ijk >= 0).all(axis=1) & (ijk < dims).all(axis=1)
    counts = np.zeros(tuple(int(d) for d in dims), dtype=np.int32)
    np.add.at(counts, (ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]), 1)
    occupied = counts >= min_per_voxel
    for _ in range(max(0, dilate)):
        occupied = _dilate(occupied)
    return SupportMask(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        voxel=float(voxel),
        dims=(int(dims[0]), int(dims[1]), int(dims[2])),
        occupied=occupied,
    )


def _dilate(grid: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """One voxel of 6-neighbour dilation, without scipy."""
    out = grid.copy()
    out[1:, :, :] |= grid[:-1, :, :]
    out[:-1, :, :] |= grid[1:, :, :]
    out[:, 1:, :] |= grid[:, :-1, :]
    out[:, :-1, :] |= grid[:, 1:, :]
    out[:, :, 1:] |= grid[:, :, :-1]
    out[:, :, :-1] |= grid[:, :, 1:]
    return out


def _finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
