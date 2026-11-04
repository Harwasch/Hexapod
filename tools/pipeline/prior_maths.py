"""Geometry-prior reference maths, NumPy only; losses are derived here, not copied."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sim3

F64 = npt.NDArray[np.float64]


def cache_key(frame: Path, model_revision: str, geometry: Any) -> str:
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    # Intrinsics/preprocessing affect the prediction too; frame bytes alone are insufficient.
    return hashlib.sha256(
        json.dumps([digest, model_revision, geometry], sort_keys=True).encode()
    ).hexdigest()


def scale_shift(
    prediction: npt.ArrayLike, target: npt.ArrayLike, mask: npt.ArrayLike
) -> tuple[float, float]:
    p, t = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(p) & np.isfinite(t) & (t > 0)
    x, y = p[valid], t[valid]
    if len(x) < 3 or np.ptp(x) <= np.finfo(float).eps:
        raise ValueError("depth alignment needs three finite, nonconstant samples")
    # Three IRLS fits: downweight large residuals using their median absolute scale.
    weights = np.ones_like(x)
    fit = np.array([1.0, 0.0])
    for _ in range(3):
        design = np.stack([x, np.ones_like(x)], 1)
        fit = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[
            0
        ]
        residual = np.abs(design @ fit - y)
        scale = max(float(np.median(residual)), np.finfo(float).eps)
        weights = np.minimum(1, 1.5 * scale / np.maximum(residual, np.finfo(float).eps))
    if fit[0] <= 0:
        raise ValueError("depth alignment has nonpositive scale")
    return float(fit[0]), float(fit[1])


def depth_normals(depth: npt.ArrayLike, intrinsic: npt.ArrayLike) -> F64:
    z, k = np.asarray(depth, dtype=float), np.asarray(intrinsic, dtype=float)
    y, x = np.indices(z.shape)
    rays = np.stack([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)], -1)
    points = rays * z[..., None]
    normals = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), np.finfo(float).eps)
    # Face the viewing camera; MoGe normals are OpenCV camera-frame surface normals.
    normals *= np.where((normals * points[1:-1, 1:-1]).sum(-1, keepdims=True) > 0, -1, 1)
    return normals


def aniso_penalty(log_scales: npt.ArrayLike, ratio: float = 10) -> float:
    scales = np.asarray(log_scales, dtype=float)
    return float(np.maximum(0, np.exp(np.ptp(scales, axis=-1)) - ratio).mean())


def aligned_chunk(
    predicted: F64, anchors: F64, sparse: F64, radius: float
) -> tuple[F64, dict[str, Any]]:
    """Robust correspondence fit, then an independent scale/residual refusal in SfM units.

    Anchor pairs are model point-map samples at projected sparse points, never held-out
    images. A 5%-of-scene-radius residual is an explicit uncalibrated acceptance rule,
    not a numerical tolerance. Pose-conditioned chunks should already have near-unit scale.
    """
    if len(anchors) < 8 or len(anchors) != len(sparse):
        raise ValueError("dense chunk needs at least eight SfM correspondences")
    fit = sim3.fit_sim3(
        {str(i): p for i, p in enumerate(anchors)},
        {str(i): p for i, p in enumerate(sparse)},
        max_error_m=0.05 * radius,
    )
    transform = fit.similarity
    if not 0.5 <= transform.scale <= 2 or len(fit.inliers) < 0.6 * len(anchors):
        raise ValueError("dense chunk failed scale/SfM consensus check")
    mapped = transform.scale * (predicted @ transform.rotation.T) + transform.translation
    return mapped, {"scale": transform.scale, "inliers": len(fit.inliers), "anchors": len(anchors)}


def voxel_select(points: F64, voxel: float, maximum: int) -> npt.NDArray[np.int64]:
    if voxel <= 0 or maximum < 0:
        raise ValueError("invalid dense voxel or cap")
    _, indices = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    indices.sort()
    if len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]
    return indices.astype(np.int64)


def cache_geometry(
    image: npt.ArrayLike, intrinsic: npt.ArrayLike, mask: npt.ArrayLike | None = None
) -> dict[str, Any]:
    """Include undistorted pixels: equal frame bytes/K need not imply equal distortion."""
    pixels = np.asarray(image)
    return {
        "K": np.asarray(intrinsic).tolist(),
        "shape": list(pixels.shape),
        "imageSha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "maskSha256": None
        if mask is None
        else hashlib.sha256(np.asarray(mask).tobytes()).hexdigest(),
        "preprocessing": "moge2-normal-level6-known-fov-v1",
    }
