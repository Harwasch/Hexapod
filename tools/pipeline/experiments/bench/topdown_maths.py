"""Numpy-only top-down camera: central 90% of centres, narrow perspective, no world-up guess."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def frame_camera(
    means: npt.ArrayLike, centres: npt.ArrayLike, up: npt.ArrayLike | None, *, size: int = 512
) -> dict[str, Any]:
    """Look along -up with a 5-degree vertical FOV and a 5% framing margin.

    Keep the nearest ceil(0.9*N) centres to the coordinate median, then enclose their
    projection. The margin is for Gaussian footprints; this is an overview, not a crop
    of the exported PLY. Camera-plane fallback has no gravity sign: choose the side
    toward the cameras, or a deterministic axis sign when that dot product is zero.
    """
    points = np.asarray(means, dtype=np.float64)
    cameras = np.asarray(centres, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("need nonempty Nx3 Gaussian centres")
    if not np.isfinite(points).all() or size < 1:
        raise ValueError("nonfinite centres or invalid image size")
    middle = np.median(points, axis=0)
    count = int(np.ceil(0.9 * len(points)))
    distances = np.linalg.norm(points - middle, axis=1)
    core = points[np.argsort(distances, kind="stable")[:count]]
    method = "camera-up"
    if up is None:
        if cameras.ndim != 2 or cameras.shape[1] != 3 or len(cameras) < 3:
            raise ValueError("camera-plane fallback needs at least three cameras")
        if not np.isfinite(cameras).all():
            raise ValueError("nonfinite camera centres")
        _, singular, axes = np.linalg.svd(cameras - cameras.mean(axis=0), full_matrices=False)
        if singular[1] <= np.finfo(float).eps * max(cameras.shape) * singular[0]:
            raise ValueError("camera centres do not define a plane")
        vertical = axes[-1]
        side = float(vertical @ (cameras.mean(axis=0) - middle))
        if side < 0 or (side == 0 and vertical[np.argmax(np.abs(vertical))] < 0):
            vertical = -vertical
        method = "camera-plane"
    else:
        vertical = np.asarray(up, dtype=np.float64)
    if vertical.shape != (3,) or not np.isfinite(vertical).all() or np.linalg.norm(vertical) == 0:
        raise ValueError("invalid up vector")
    vertical = vertical / np.linalg.norm(vertical)
    forward = -vertical
    axis = np.eye(3)[np.argmin(np.abs(vertical))]
    right = np.cross(forward, axis)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    projected = core @ rotation.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    target = ((low + high) / 2) @ rotation
    half = float(np.max((high - low)[:2]) / 2)
    if half <= 0:
        raise ValueError("central Gaussian centres have no projected extent")
    focal = size / (2 * np.tan(np.deg2rad(5) / 2))
    # At the closest core depth, even the widest core corner fits with this margin.
    distance = 1.05 * half * focal / (size / 2) + float((high[2] - low[2]) / 2)
    eye = target + distance * vertical
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    depths = core @ forward + view[2, 3]
    return {
        "viewmat": view.tolist(),
        "K": [[float(focal), 0, size / 2], [0, float(focal), size / 2], [0, 0, 1]],
        "width": size,
        "height": size,
        "near": float(depths.min() / 2),
        "far": float(depths.max() * 2),
        "up": vertical.tolist(),
        "upMethod": method,
        "framedCentres": count,
        "totalCentres": len(points),
        "fovDegrees": 5,
    }
