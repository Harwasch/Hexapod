"""Frozen pre-windowing sampler: byte-for-byte selection oracle for regression tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

from captures_bridge import pack_spz, read_ply, sigmoid

MAX_GAUSSIANS = 100_000
_SPZ_LIMIT = 2_000.0


def snapshot_splat(
    ply: Path,
    target: Path,
    *,
    max_gaussians: int = MAX_GAUSSIANS,
    camera_centres: npt.NDArray[np.float64] | None = None,
    focal_px: float = 1.0,
) -> tuple[int, int, int]:
    """Pack a spatially stratified screen-contribution thumbnail of `ply` as SPZ.

    Returns (kept, of, bytes). Degree-0 colour only: a preview does not need the view-
    dependent terms, and they are most of an SPZ. Raises `SplatFormatError` on a PLY the
    trainer has not finished writing.
    """
    data = read_ply(ply)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    log_scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
    alpha = sigmoid(data["opacity"])
    # Upper bound on projected ellipse area: the two largest axes, with the long
    # axis capped at 10x the middle one. A needle cannot buy priority with length.
    axes = np.sort(np.exp(np.clip(log_scales, -30.0, 30.0)), axis=1)
    area = axes[:, 1] * np.minimum(axes[:, 2], 10.0 * axes[:, 1])
    distance = np.ones(len(xyz))
    if camera_centres is not None and len(camera_centres):
        centres = camera_centres[np.isfinite(camera_centres).all(axis=1)]
        if len(centres):
            # Bounded chunks: no full N-splats by N-cameras temporary and no RNG.
            camera_norms = np.sum(centres**2, axis=1)
            for start in range(0, len(xyz), 4096):
                points = xyz[start : start + 4096].astype(np.float64)
                squared = np.sum(points**2, axis=1)[:, None] + camera_norms
                squared -= 2 * (points @ centres.T)
                distance[start : start + len(points)] = np.sqrt(np.maximum(squared.min(axis=1), 0))
    # Cap angular area at one steradian, including splats at the camera centre.
    score = alpha * focal_px**2 * np.minimum(area / np.maximum(distance**2, 1e-12), 1.0)
    good = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(score)
        & np.isfinite(log_scales).all(axis=1)
        & (np.abs(xyz) < _SPZ_LIMIT).all(axis=1)
    )
    candidates = np.flatnonzero(good)
    if candidates.size > max_gaussians:
        positions = xyz[candidates].astype(np.float64)
        span = float(np.max(np.ptp(positions, axis=0)))
        width = max(span / max(1.0, (max_gaussians / 4) ** (1 / 3)), 1e-9)
        cells = np.minimum(
            np.floor((positions - positions.min(axis=0)) / width),
            max(0, math.ceil(span / width) - 1),
        ).astype(np.int64)
        _, cell = np.unique(cells, axis=0, return_inverse=True)
        # Within each voxel order by contribution, ties by original index. Take the
        # first from every voxel before its second, then fill the remaining budget.
        order = np.lexsort((candidates, -score[candidates], cell))
        ordered_cells = cell[order]
        starts = np.maximum.accumulate(
            np.where(np.r_[True, ordered_cells[1:] != ordered_cells[:-1]], np.arange(len(order)), 0)
        )
        rank = np.arange(len(order)) - starts
        take = np.lexsort((candidates[order], -score[candidates[order]], rank))[:max_gaussians]
        candidates = np.sort(candidates[order[take]])
    sh0 = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)[candidates]
    quat = np.stack([data["rot_1"], data["rot_2"], data["rot_3"], data["rot_0"]], axis=1)
    blob = pack_spz(
        xyz[candidates],
        sh0,
        data["opacity"][candidates],
        log_scales[candidates],
        quat[candidates],
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(blob)
    partial.replace(target)
    return int(candidates.size), int(xyz.shape[0]), len(blob)
