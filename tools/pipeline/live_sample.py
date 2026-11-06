"""Exact spatial thumbnail selection with memory bounded by the thumbnail budget.

Four windowed passes: valid bounds, voxel populations, per-voxel winners, then gather
only the retained SH-0 columns. SH rest is never copied. No arrays scale with PLY size.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from captures_bridge import PlyLayout, iter_ply_rows, pack_spz, ply_layout, sigmoid

SCORE_COLUMNS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")
PACK_COLUMNS = (*SCORE_COLUMNS, "f_dc_0", "f_dc_1", "f_dc_2", "rot_0", "rot_1", "rot_2", "rot_3")
WINDOW_ROWS = 65_536  # divisible by the original nearest-camera chunk size
Array = npt.NDArray[Any]


def _scored(
    layout: PlyLayout, centres: npt.NDArray[np.float64] | None, focal: float
) -> Iterator[tuple[Array, Array, Array]]:
    if centres is not None:
        centres = centres[np.isfinite(centres).all(axis=1)]
    for start, data in iter_ply_rows(layout, SCORE_COLUMNS, WINDOW_ROWS * layout.dtype.itemsize):
        xyz = np.stack([data[k] for k in SCORE_COLUMNS[:3]], axis=1)
        scales = np.stack([data[k] for k in SCORE_COLUMNS[4:]], axis=1)
        axes = np.sort(np.exp(np.clip(scales, -30.0, 30.0)), axis=1)
        area = axes[:, 1] * np.minimum(axes[:, 2], 10.0 * axes[:, 1])
        distance = np.ones(len(xyz))
        if centres is not None and len(centres):
            norms = np.sum(centres**2, axis=1)
            for offset in range(0, len(xyz), 4096):
                points = xyz[offset : offset + 4096].astype(np.float64)
                squared = np.sum(points**2, axis=1)[:, None] + norms
                squared -= 2 * (points @ centres.T)
                distance[offset : offset + len(points)] = np.sqrt(
                    np.maximum(squared.min(axis=1), 0)
                )
        score = (
            sigmoid(data["opacity"])
            * focal**2
            * np.minimum(area / np.maximum(distance**2, 1e-12), 1.0)
        )
        good = (
            np.isfinite(xyz).all(axis=1)
            & np.isfinite(score)
            & np.isfinite(scales).all(axis=1)
            & (np.abs(xyz) < 2_000.0).all(axis=1)
        )
        indices = np.flatnonzero(good)
        yield indices + start, xyz[indices].astype(np.float64), score[indices]


def _ranks(cells: Array) -> Array:
    starts = np.maximum.accumulate(
        np.where(np.r_[True, cells[1:] != cells[:-1]], np.arange(len(cells)), 0)
    )
    return np.arange(len(cells)) - starts


def snapshot_splat(
    ply: Path,
    target: Path,
    *,
    max_gaussians: int = 100_000,
    camera_centres: npt.NDArray[np.float64] | None = None,
    focal_px: float = 1.0,
) -> tuple[int, int, int]:
    """Pack the same rank/score/index selection as the original in-memory sampler.

    If R complete rounds of voxel winners fit, only the best R+1 rows per voxel
    can win. Their total is at most budget + occupied voxels (grid size O(budget)).
    Sorting/merging each window retains that bound even for a single dense voxel.
    Returns (kept, input rows, SPZ bytes); malformed/truncated PLYs still raise.
    """
    if max_gaussians < 1:
        raise ValueError("thumbnail budget must be positive")
    layout = ply_layout(ply)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    count = 0
    for indices, xyz, _ in _scored(layout, camera_centres, focal_px):
        count += len(indices)
        if len(indices):
            low = np.minimum(low, xyz.min(axis=0))
            high = np.maximum(high, xyz.max(axis=0))
    span = float(np.max(high - low)) if count else 0.0
    width = max(span / max(1.0, (max_gaussians / 4) ** (1 / 3)), 1e-9)
    edge = max(1, math.ceil(span / width))

    def voxel(xyz: Array) -> Array:
        cells = np.minimum(np.floor((xyz - low) / width), edge - 1).astype(np.int64)
        return np.asarray((cells[:, 0] * edge + cells[:, 1]) * edge + cells[:, 2], dtype=np.int64)

    populations = np.zeros(edge**3, dtype=np.int64)
    for _, xyz, _ in _scored(layout, camera_centres, focal_px):
        populations += np.bincount(voxel(xyz), minlength=len(populations))
    # First incomplete round: retain one more than every completely affordable round.
    lo, hi = 0, min(count, max_gaussians) + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if np.minimum(populations, mid).sum() <= max_gaussians:
            lo = mid
        else:
            hi = mid
    per_voxel = lo + 1
    kept = np.empty(0, dtype=np.int64)
    scores = np.empty(0, dtype=np.float64)
    cells = np.empty(0, dtype=np.int64)
    for indices, xyz, score in _scored(layout, camera_centres, focal_px):
        if not len(indices):
            continue
        kept = np.concatenate((kept, indices))
        scores = np.concatenate((scores, score))
        cells = np.concatenate((cells, voxel(xyz)))
        order = np.lexsort((kept, -scores, cells))
        order = order[_ranks(cells[order]) < per_voxel]
        kept, scores, cells = kept[order], scores[order], cells[order]
    if len(kept):
        take = np.lexsort((kept, -scores, _ranks(cells)))[:max_gaussians]
        kept = np.sort(kept[take])
    gathered = {name: np.empty(len(kept), dtype=np.float32) for name in PACK_COLUMNS}
    for start, data in iter_ply_rows(layout, PACK_COLUMNS, WINDOW_ROWS * layout.dtype.itemsize):
        begin, end = np.searchsorted(kept, [start, start + len(data["x"])])
        selected = kept[begin:end] - start
        for name in PACK_COLUMNS:
            gathered[name][begin:end] = data[name][selected]

    def stack(names: tuple[str, ...]) -> Array:
        return np.stack([gathered[name] for name in names], axis=1)

    blob = pack_spz(
        stack(("x", "y", "z")),
        stack(("f_dc_0", "f_dc_1", "f_dc_2")),
        gathered["opacity"],
        stack(("scale_0", "scale_1", "scale_2")),
        stack(("rot_1", "rot_2", "rot_3", "rot_0")),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(blob)
    partial.replace(target)
    return len(kept), layout.count, len(blob)
