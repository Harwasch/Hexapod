"""Shared rescore geometry, colour and morphology definitions; NumPy only."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .topdown_maths import frame_camera

MODES = ("raw", "aligned", "cc", "aligned+cc")
RESOLUTIONS = ("full", "half")


def affine_correct(prediction: npt.ArrayLike, target: npt.ArrayLike) -> tuple[Any, Any, Any]:
    """gsplat affine convention: fit prediction = a*GT+b, then invert and clip.

    Independent implementation from channel means/covariance. Like the pinned
    cc_psnr, this is per-channel gain/offset, not a cross-channel 3x4 matrix.
    """
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    p_mean, t_mean = p.mean(0), t.mean(0)
    slope = ((p * t).mean(0) - p_mean * t_mean) / np.maximum(
        (t * t).mean(0) - t_mean * t_mean, 1e-8
    )
    offset = p_mean - slope * t_mean
    slope = np.where(np.abs(slope) < 1e-8, 1, slope)
    return np.clip((p - offset) / slope, 0, 1), slope, offset


def half_image(image: npt.ArrayLike) -> Any:
    """Exact 2x2 area average; fail rather than silently round odd frozen dimensions."""
    pixels = np.asarray(image, dtype=np.float64)
    h, w = pixels.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("full/half scoring requires even frozen image dimensions")
    return pixels.reshape(h // 2, 2, w // 2, 2, *pixels.shape[2:]).mean(axis=(1, 3))


def half_camera(camera: dict[str, Any]) -> dict[str, Any]:
    if camera["height"] % 2 or camera["width"] % 2:
        raise ValueError("full/half scoring requires even frozen camera dimensions")
    intrinsic = np.array(camera["K"], dtype=float)
    intrinsic[:2] /= 2
    return {
        **camera,
        "K": intrinsic.tolist(),
        "height": camera["height"] // 2,
        "width": camera["width"] // 2,
    }


def se3(twist: npt.ArrayLike) -> Any:
    """SE(3) exponential, rotation vector first and translational tangent last."""
    x = np.asarray(twist, dtype=float)
    w, v = x[:3], x[3:]
    theta = np.linalg.norm(w)
    skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    a = np.sinc(theta / np.pi)
    b = 0.5 * np.sinc(theta / (2 * np.pi)) ** 2
    c = (1 - a) / theta**2 if theta > 1e-4 else 1 / 6 - theta**2 / 120
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + a * skew + b * (skew @ skew)
    out[:3, 3] = (np.eye(3) + b * skew + c * (skew @ skew)) @ v
    return out


def pose_motion(original: npt.ArrayLike, aligned: npt.ArrayLike, extent: float) -> dict[str, float]:
    a, b = np.asarray(original), np.asarray(aligned)
    ca, cb = -a[:3, :3].T @ a[:3, 3], -b[:3, :3].T @ b[:3, 3]
    cosine = np.clip((np.trace(b[:3, :3] @ a[:3, :3].T) - 1) / 2, -1, 1)
    return {
        "translationPct": float(100 * np.linalg.norm(cb - ca) / extent),
        "rotationDeg": float(np.rad2deg(np.arccos(cosine))),
    }


def hull(points: npt.ArrayLike) -> Any:
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3:
        raise ValueError("footprint needs at least three distinct support points")
    ordered = sorted(map(tuple, points.tolist()))

    def side(a: Any, b: Any, c: Any) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    halves = []
    for order in (ordered, ordered[::-1]):
        chain: list[Any] = []
        for point in order:
            while len(chain) > 1 and side(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        halves.append(chain[:-1])
    result = np.array(halves[0] + halves[1])
    if len(result) < 3:
        raise ValueError("degenerate support footprint")
    return result


def footprint(camera: dict[str, Any], polygon: npt.ArrayLike) -> Any:
    """Pixel-centre inclusion in the convex support polygon, no splat-derived mask."""
    world = np.asarray(polygon)
    view = np.asarray(camera["viewmat"])
    xyz = world @ view[:3, :3].T + view[:3, 3]
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("support footprint behind overhead camera")
    pixels = xyz @ np.asarray(camera["K"]).T
    outline = hull(pixels[:, :2] / pixels[:, 2:])
    yy, xx = np.mgrid[: camera["height"], : camera["width"]] + 0.5
    mask = np.ones(xx.shape, dtype=bool)
    for a, b in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        mask &= (b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0]) >= 0
    if not mask.any():
        raise ValueError("empty overhead footprint")
    return mask


def shared_cameras(
    points: npt.ArrayLike, centres: npt.ArrayLike, up: npt.ArrayLike | None
) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("invalid frozen sparse points")
    centre = np.median(points, axis=0)
    distances = np.linalg.norm(points - centre, axis=1)
    core = points[np.argsort(distances, kind="stable")[: int(np.ceil(0.9 * len(points)))]]
    extent = float(2 * np.max(np.linalg.norm(core - centre, axis=1)))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("degenerate frozen scene extent")
    top = frame_camera(points, centres, up, size=384)
    vertical = np.asarray(top["up"])
    axes = np.asarray(top["viewmat"])[:2, :3]
    planar = hull((core - centre) @ axes.T)
    polygon = centre + planar @ axes
    views = []
    for angle in (90, 75, 60):
        direction = np.sin(np.deg2rad(angle)) * vertical + np.cos(np.deg2rad(angle)) * axes[0]
        camera = frame_camera(core, centres, direction, size=384)
        # frame_camera trims 10% itself. Back up enough to include *all* of our
        # already selected support and plane corners, with the same 5% margin.
        support = np.concatenate([core, polygon])
        view = np.asarray(camera["viewmat"])
        projected = support @ view[:3, :3].T + view[:3, 3]
        focal = camera["K"][0][0]
        required = 1.05 * np.max(np.abs(projected[:, :2]), axis=1) * focal / 192
        view[2, 3] += max(0, float(np.max(required - projected[:, 2])))
        depths = support @ view[2, :3] + view[2, 3]
        camera.update(
            viewmat=view.tolist(),
            near=float(depths.min() / 2),
            far=float(depths.max() * 2),
            framedCentres=len(core),
            totalCentres=len(points),
        )
        camera.update(name=f"overhead-{angle}", downAngleDeg=angle)
        footprint(camera, polygon)
        views.append(camera)
    return {
        "extent": extent,
        "extentDefinition": (
            "diameter about coordinate median enclosing nearest 90% of frozen SfM points"
        ),
        "up": vertical.tolist(),
        "upMethod": top["upMethod"],
        "footprint": polygon.tolist(),
        "footprintDefinition": (
            "convex hull of central-90% SfM points projected onto the plane through their "
            "coordinate median perpendicular to up; pixel centres inside projected polygon"
        ),
        "cameras": views,
    }


def needle_scores(columns: dict[str, Any], extent: float) -> dict[str, Any]:
    far = np.asarray(columns.get("far_field", np.zeros(len(columns["x"])))) >= 0.5
    scales = np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1)[~far]
    opacity = np.asarray(columns["opacity"])[~far]
    if not np.isfinite(scales).all() or not np.isfinite(opacity).all() or extent <= 0:
        raise ValueError("nonfinite splat scales/opacity or invalid extent")
    weight = 1 / (1 + np.exp(-np.clip(opacity, -80, 80)))

    def counts(mask: Any) -> dict[str, Any]:
        return {
            "count": int(mask.sum()),
            "share": float(mask.mean()) if len(mask) else None,
            "opacityWeight": float(weight[mask].sum()),
            "opacityWeightedShare": float(weight[mask].sum() / weight.sum())
            if weight.sum()
            else None,
        }

    return {
        "foregroundCount": len(scales),
        "farFieldCount": int(far.sum()),
        "farFieldMarked": "far_field" in columns,
        "elongation": counts(np.ptp(scales, axis=1) > np.log(10)),
        "oversized": {
            str(percent): counts(scales.max(axis=1) > np.log(extent * percent / 100))
            for percent in (1, 2, 5)
        },
    }


def trained_at(source: dict[str, Any], full: list[int]) -> dict[str, Any]:
    from prior_maths import resize_geometry

    params = source["result"].get("trainParams", {})
    h, w = full
    h, w, _, _ = resize_geometry(h, w, params.get("train_max_side") or 0)
    factor = int(params.get("data_factor", 1))
    h, w = h // factor, w // factor
    h, w, _, _ = resize_geometry(h, w, params.get("step_max_side") or 0)
    return {
        "height": h,
        "width": w,
        "source": "frozen dimensions + recorded train_max_side/data_factor/step_max_side",
    }


def row_base(source: dict[str, Any], full: list[int]) -> dict[str, Any]:
    return {
        "runId": source["runId"],
        "ply": source["plyKind"],
        "plyBytes": source["plyBytes"],
        "variant": source["result"].get("variant", {}).get("name", "unknown"),
        "trainedAt": trained_at(source, full),
        "trainingLoss": "historical PLY; not modified or retrained by rescore",
    }
