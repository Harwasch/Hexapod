"""Depth pixels to metric ARKit world points, independent of inference and storage.

Depth is axial metres along camera -Z, not distance along a normalized ray.
Transforms are row-major camera-to-world. Image coordinates have Y down; world Y is up.
Pixel centres are integer coordinates; resizing preserves half-pixel boundaries.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Bounds:
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    valid_pixels: int
    selected_pixels: int


def lift_depth(
    depth: F64,
    mask: npt.NDArray[np.bool_],
    intrinsics: F64,
    camera_to_world: F64,
    image_size: tuple[int, int],
) -> F64:
    """Return Nx3 world points. Holes (<=0, NaN, inf) are omitted, never filled.

    The mask is sampled on the depth grid. Intrinsics belong to the original RGB
    image, whose (width, height) is image_size. No distortion model is assumed.
    Rigid-transform tolerances allow float32 pose serialization, as capture_bundle does.
    """
    if depth.ndim != 2 or 0 in depth.shape or mask.shape != depth.shape:
        raise ValueError("depth and mask must be matching nonempty 2D grids")
    if mask.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    k, t = intrinsics, camera_to_world
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("invalid intrinsics")
    if (
        k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.array_equal(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1])
    ):
        raise ValueError("intrinsics must be an unskewed pinhole matrix")
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("camera-to-world must be a proper rigid transform")
    rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[rows, columns]
    u = (columns + 0.5) * width / depth.shape[1] - 0.5
    v = (rows + 0.5) * height / depth.shape[0] - 0.5
    camera = np.column_stack(((u - k[0, 2]) * z / k[0, 0], -(v - k[1, 2]) * z / k[1, 1], -z))
    return np.asarray(camera @ t[:3, :3].T + t[:3, 3], dtype=np.float64)


def bounds(points: F64, selected_pixels: int) -> Bounds | None:
    """World-axis bounds of observed surface samples; not hidden object dimensions."""
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("expected finite Nx3 points")
    if selected_pixels < len(points):
        raise ValueError("selected pixel count is smaller than valid count")
    if not len(points):
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    center, extent = (low + high) / 2, high - low
    return Bounds(
        (float(center[0]), float(center[1]), float(center[2])),
        (float(extent[0]), float(extent[1]), float(extent[2])),
        len(points),
        selected_pixels,
    )
