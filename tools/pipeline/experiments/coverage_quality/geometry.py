"""Small numpy geometry shared by the observational and held-out experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
CONES = (10, 20, 25, 30, 45, 60)
SHARPNESS = (0.5, 1, 2, 4)
KERNELS = tuple((cone, sharp) for cone in CONES for sharp in SHARPNESS)


def kernel_sum(row: dict[str, Any], cone: float, sharpness: float) -> float:
    """Exact grid sufficient statistic; no angular rounding in compact derived CSVs."""
    if "credit_sums" in row:
        return float(row["credit_sums"][KERNELS.index((cone, sharpness))])
    cosine = np.cos(np.radians(row["angles_deg"]))
    return float(credit(cosine, cone, sharpness).sum())


def unit(v: Array) -> Array:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def hemisphere(normal: Array, count: int = 14) -> Array:
    """Equal-area Fibonacci hemisphere, deterministically oriented about a PCA normal."""
    normal = unit(normal)
    seed = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = unit(np.cross(normal, seed))
    bitangent = np.cross(normal, tangent)
    z = (np.arange(count) + 0.5) / count
    phi = np.arange(count) * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.asarray(
        (radius * np.cos(phi))[:, None] * tangent
        + (radius * np.sin(phi))[:, None] * bitangent
        + z[:, None] * normal
    )


def credit(cosine: Array, cone: float = 25, sharpness: float = 1) -> Array:
    edge = np.cos(np.radians(cone))
    return np.asarray(np.clip((cosine - edge) / (1 - edge), 0, 1) ** sharpness)


def spread(directions: Array) -> float:
    """Exact pairwise angle, chunked, with bounded memory."""
    if len(directions) < 2:
        return 0.0
    lowest = 1.0
    for start in range(0, len(directions), 128):
        lowest = min(lowest, float((directions[start : start + 128] @ directions.T).min()))
    return float(np.degrees(np.arccos(np.clip(lowest, -1, 1))))


def features(
    point: Array,
    normal: Array,
    centres: Array,
    visible: Array,
    focal: float = 1000,
) -> dict[str, float]:
    """No learned constants: score names expose each hypothesis under comparison."""
    delta = centres[visible] - point
    if not len(delta):
        return dict.fromkeys(
            [
                "views",
                "spread",
                "distinct15",
                "fisher",
                "hairs",
                "hairs_resolution",
                "fisher_resolution",
                "min_distance",
                "median_distance",
                "incidence",
                "best_mm_px",
                "hemisphere_coverage",
            ],
            0.0,
        )
    distances = np.linalg.norm(delta, axis=1)
    directions = unit(delta)
    normal = normal * (1 if np.dot(normal, directions.mean(0)) >= 0 else -1)
    incidence = np.maximum(directions @ normal, 0)
    footprint = distances * 1000 / focal / np.maximum(incidence, 0.1)
    resolution = np.minimum(1, (2 / np.maximum(footprint, 1e-6)) ** 2)
    dots = directions @ hemisphere(normal).T
    hairs = credit(dots)
    distinct: list[Array] = []
    for direction in directions:
        if not distinct or np.max(np.asarray(distinct) @ direction) <= np.cos(np.radians(15)):
            distinct.append(direction)
    # Bearing-only point-position information, not a trained-parameter Fisher matrix.
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    information = (projectors / np.maximum(distances[:, None, None] ** 2, 1e-6)).sum(0)
    fisher = float(np.linalg.slogdet(np.eye(3) + information)[1])
    weighted = (projectors * resolution[:, None, None]).sum(0)
    return {
        "views": float(len(delta)),
        "spread": spread(directions),
        "distinct15": float(len(distinct)),
        "fisher": fisher,
        "hairs": float(np.minimum(hairs.sum(0) / 8, 1).mean()),
        "hairs_resolution": float(np.minimum((hairs * resolution[:, None]).sum(0) / 8, 1).mean()),
        "fisher_resolution": float(np.linalg.slogdet(np.eye(3) + weighted)[1]),
        "min_distance": float(distances.min()),
        "median_distance": float(np.median(distances)),
        "incidence": float(incidence.mean()),
        "best_mm_px": float(footprint.min()),
        "hemisphere_coverage": float((dots.max(0) >= np.cos(np.radians(25))).mean()),
    }
