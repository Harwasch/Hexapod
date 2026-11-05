"""Frozen background geometry and bounded export policy, independent of torch.

Far means outside five times the sparse scene's 90th-percentile radius. The sphere
is at ten radii. Both are in the original COLMAP frame, never a learned moving frame.
Placement deliberately strips marked background before computing terrestrial bounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import splat_io


def frame(points: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], float]:
    xyz = np.asarray(points, dtype=float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 3:
        raise ValueError("far field needs three finite sparse points")
    center = np.median(xyz, axis=0)
    radius = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 90))
    if radius <= 0:
        raise ValueError("far field needs a nonzero scene radius")
    return center, radius


def mask(points: npt.ArrayLike, center: npt.ArrayLike, radius: float) -> npt.NDArray[np.bool_]:
    return np.asarray(
        np.linalg.norm(np.asarray(points) - np.asarray(center), axis=1) > 5 * radius, dtype=np.bool_
    )


def sphere(points: npt.ArrayLike, rgb: npt.ArrayLike, count: int) -> dict[str, Any]:
    center, radius = frame(points)
    i = np.arange(count, dtype=float)
    y = 1 - 2 * (i + 0.5) / count
    theta = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - y * y)
    directions = np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1
    color = np.median(np.asarray(rgb, dtype=float), axis=0) / 255
    return {
        "means": (center + directions * radius * 10).astype(np.float32),
        "log_scales": np.full((count, 3), np.log(15 * radius / np.sqrt(count)), dtype=np.float32),
        "quats": quats,
        "logit_opacity": np.full(count, 3, dtype=np.float32),
        "f_dc": np.tile((color - 0.5) / 0.28209479177387814, (count, 1)).astype(np.float32),
    }


def export_rows(
    columns: dict[str, Any], center: npt.ArrayLike, radius: float, share: float
) -> tuple[dict[str, Any], npt.NDArray[np.bool_]]:
    """Bound *all* geometrically far rows, including learned outliers, by final count.

    Retain the highest-opacity far rows. The denominator is the surviving near count,
    so pruning/cropping foreground cannot silently exceed the promised fraction.
    """
    distant = mask(np.stack([columns[n] for n in ("x", "y", "z")], 1), center, radius)
    allowed = int((~distant).sum() * share / (1 - share))
    selected = ~distant
    indices = np.flatnonzero(distant)
    order = np.argsort(-np.asarray(columns["opacity"])[indices], kind="stable")
    selected[indices[order[:allowed]]] = True
    result = {k: v[selected] for k, v in columns.items()}
    result["far_field"] = distant[selected].astype(np.float32)
    return result, selected


def strip(source: Path, target: Path) -> Path:
    reader = splat_io.SplatReader(source)
    if "far_field" not in reader.properties:
        return source
    properties = tuple(p for p in reader.properties if p != "far_field")
    with splat_io.PlyWriter(target, properties) as writer:
        for _, columns in reader.chunks():
            keep = columns["far_field"] < 0.5
            writer.append({k: columns[k][keep] for k in properties})
    return target
