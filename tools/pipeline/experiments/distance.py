"""CPU, bounded-memory splat distance to the capture polyline; no renderer or Modal.

Metres are an assumption-based yardstick, not georeferencing. A local RANSAC ground
plane and median camera height set one shared scale for all PLYs of a frozen scene.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sfm
from splat_io import SplatReader

EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
LABELS = ("0-2", "2-5", "5-10", "10-20", "20-50", "50+")
CHUNK = 32768


def distance_to_path(points: npt.ArrayLike, centres: npt.ArrayLike) -> Any:
    """Exact Euclidean distance to closed line segments, including repeated centres.

    O(points * segments) time, O(points) working memory. Call on bounded chunks.
    A one-camera path is a point. Inputs must share the original COLMAP frame.
    """
    xyz, path = np.asarray(points, dtype=float), np.asarray(centres, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("points and camera path must be Nx3")
    if not len(path) or not np.isfinite(xyz).all() or not np.isfinite(path).all():
        raise ValueError("distance needs finite points and a nonempty finite camera path")
    best = np.square(xyz - path[0]).sum(axis=1)
    for a, b in pairwise(path):
        direction = b - a
        length2 = float(direction @ direction)
        if length2 == 0:
            continue
        delta = xyz - a
        t = np.clip(np.einsum("ij,j->i", delta, direction) / length2, 0, 1)
        residual = delta - t[:, None] * direction
        best = np.minimum(best, np.einsum("ij,ij->i", residual, residual))
    return np.sqrt(best)


def ground_scale(
    points: npt.ArrayLike,
    centres: npt.ArrayLike,
    up: npt.ArrayLike,
    height_m: float = 1.4,
    *,
    tolerance_fraction: float = 0.05,
) -> dict[str, Any]:
    """Fit the dominant approximately horizontal plane near/below the camera path.

    Local = nearest half of sparse points by distance to a registered camera, below
    that camera along the camera-up estimate. Deterministic RANSAC rejects walls
    (>45 degrees from up), then SVD refines its inliers. Threshold = 5% of median
    local camera distance; this is a declared heuristic, not a surveyed accuracy.
    """
    xyz, cameras = np.asarray(points, float), np.asarray(centres, float)
    normal_up = np.array(up, dtype=float, copy=True)
    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("assumed camera height must be finite and positive")
    if not np.isfinite(normal_up).all() or np.linalg.norm(normal_up) == 0:
        raise ValueError("no usable camera-up direction")
    normal_up /= np.linalg.norm(normal_up)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 30 or not len(cameras) or not np.isfinite(cameras).all():
        raise ValueError("ground fit needs at least 30 finite sparse points and finite cameras")
    nearest, best = np.zeros(len(xyz), dtype=int), np.full(len(xyz), np.inf)
    for i, camera in enumerate(cameras):
        d2 = np.square(xyz - camera).sum(axis=1)
        use = d2 < best
        nearest[use], best[use] = i, d2[use]
    cutoff = float(np.median(np.sqrt(best)))
    below = np.einsum("ij,j->i", xyz - cameras[nearest], normal_up) < 0
    local = xyz[below & (best <= cutoff**2)]
    local_distances = np.sqrt(best[below & (best <= cutoff**2)])
    if len(local) < 30:
        raise ValueError("too few local sparse points below the cameras for a ground fit")
    tolerance = float(np.median(local_distances)) * tolerance_fraction
    if tolerance <= 0:
        raise ValueError("degenerate ground-fit scale")
    rng = np.random.default_rng(42)
    sample = local[rng.choice(len(local), min(10000, len(local)), replace=False)]
    best_count, plane = 0, None
    for _ in range(1536):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        normal /= length
        if normal @ normal_up < 0:
            normal = -normal
        offset = float(-a @ normal)
        if normal @ normal_up < np.cos(np.pi / 4):
            continue
        if np.percentile(cameras @ normal + offset, 10) <= 0:
            continue
        count = int((np.abs(sample @ normal + offset) <= tolerance).sum())
        if count > best_count:
            best_count, plane = count, (normal, offset)
    if plane is None or best_count < max(30, 0.15 * len(sample)):
        raise ValueError("no dominant local ground plane (requires 15% RANSAC support)")
    normal, offset = plane
    for _ in range(3):
        inliers = local[np.abs(local @ normal + offset) <= tolerance]
        if len(inliers) < 3:
            raise ValueError("refined ground plane lost its support")
        centre = inliers.mean(axis=0)
        _, singular, vectors = np.linalg.svd(inliers - centre, full_matrices=False)
        if singular[1] <= singular[0] * 1e-6:
            raise ValueError("ground support is collinear")
        normal = vectors[-1]
        if normal @ normal_up < 0:
            normal = -normal
        offset = float(-centre @ normal)
    heights = cameras @ normal + offset
    if normal @ normal_up < np.cos(np.pi / 4) or np.percentile(heights, 10) <= 0:
        raise ValueError("refined ground plane is inconsistent with camera height/up")
    residuals = np.abs(local @ normal + offset)
    inlier = residuals <= tolerance
    if inlier.mean() < 0.15:
        raise ValueError("refined ground plane lost its support")
    median_height = float(np.median(heights))
    return {
        "status": "estimated",
        "assumedCameraHeightM": height_m,
        "metresPerUnit": height_m / median_height,
        "planeNormal": normal.tolist(),
        "planeOffset": offset,
        "cameraHeightUnits": dict(
            zip(
                ("p10", "median", "p90"), np.percentile(heights, [10, 50, 90]).tolist(), strict=True
            )
        ),
        "localPoints": len(local),
        "inlierPoints": int(inlier.sum()),
        "inlierShare": float(inlier.mean()),
        "localRadiusUnits": cutoff,
        "ransacToleranceUnits": tolerance,
        "inlierResidualMedianUnits": float(np.median(residuals[inlier])),
        "normalToCameraUpDeg": float(np.degrees(np.arccos(np.clip(normal @ normal_up, -1, 1)))),
        "method": (
            "nearest half of SfM points, below nearest camera; 1536 RANSAC trials, "
            "45deg up constraint, 15% minimum support, three SVD refinements"
        ),
        "warning": (
            "Assumed 1.4m phone height unless overridden; camera-up is not gravity. "
            "Rocks, slopes, camera bob and weak ground texture can bias the plane. "
            "This is not surveyed metric scale."
        ),
    }


def scene_geometry(scene: Path, height_m: float = 1.4) -> dict[str, Any]:
    model = sfm.read_model(scene / "poses")
    images = list(model.images)
    if not images:
        raise ValueError("scene has no registered cameras")
    meta_path = scene / "frames_meta.json"
    frames = json.loads(meta_path.read_text()).get("frames", []) if meta_path.exists() else []
    metadata = {frame["name"]: frame for frame in frames if isinstance(frame, dict)}
    # candidateIndex/offsetS are capture order, never COLMAP image IDs or registration order.
    originals = {metadata[i.name].get("original") for i in images if i.name in metadata}
    order_key = next(
        (
            key
            for key in ("candidateIndex", "offsetS")
            if len(originals) == 1
            and all(
                isinstance(metadata.get(i.name, {}).get(key), (int, float))
                and np.isfinite(metadata[i.name][key])
                for i in images
            )
            and len({metadata[i.name][key] for i in images}) == len(images)
        ),
        None,
    )
    if order_key:
        images.sort(key=lambda i: metadata[i.name][order_key])
        ordering = f"frames_meta.{order_key} (single capture)"
    else:
        # Frozen video exports use numeric frame names. Refuse arbitrary alphabetical
        # camera ordering when metadata cannot establish a capture sequence.
        matches = [re.fullmatch(r"(.*?)(\d+)\.[^.]+", i.name) for i in images]
        if (
            not all(matches)
            or len({m[1] for m in matches if m}) != 1
            or len({int(m[2]) for m in matches if m}) != len(images)
        ):
            raise ValueError(
                "cannot establish capture order: need metadata or numbered frame names"
            )
        images.sort(key=lambda i: int(re.findall(r"\d+(?=\.[^.]+$)", i.name)[0]))
        ordering = "numeric frame filename (metadata unavailable/ambiguous)"
    centres = np.stack([i.centre for i in images])
    up = sfm.camera_up(model)
    scale: dict[str, Any]
    try:
        if up is None:
            raise ValueError("no camera-up estimate")
        points = sfm.read_points(scene / "poses")
        scale = ground_scale(points, centres, up.up, height_m)
        # Sensitivity to ground consensus width; not a confidence interval.
        sensitivity = []
        for fraction in (0.025, 0.1):
            try:
                fitted = ground_scale(points, centres, up.up, height_m, tolerance_fraction=fraction)
                sensitivity.append(
                    {"toleranceFraction": fraction, "metresPerUnit": fitted["metresPerUnit"]}
                )
            except ValueError as error:
                sensitivity.append({"toleranceFraction": fraction, "error": str(error)})
        scale["toleranceSensitivity"] = sensitivity
    except ValueError as error:
        scale = {"status": "unavailable", "reason": str(error), "assumedCameraHeightM": height_m}
    return {
        "scene": str(scene.resolve()),
        "path": {
            "ordering": ordering,
            "names": [i.name for i in images],
            "centres": centres.tolist(),
            "registeredCameras": len(images),
            "includesHeldout": True,
        },
        "scale": scale,
    }


def measure_ply(ply: Path, geometry: dict[str, Any]) -> dict[str, Any]:
    """Shares use all rows as denominator; marked far-field rows are also broken out."""
    reader = SplatReader(ply, chunk=CHUNK)
    scale = geometry["scale"]
    report: dict[str, Any] = {
        "protocol": "camera-path-distance-v1",
        "ply": str(ply.resolve()),
        "count": reader.count,
        "scale": scale,
        "farFieldMarkerPresent": "far_field" in reader.properties,
        "definitions": {
            "distance": "3D Euclidean distance to nearest point on capture-order camera polyline, "
            "including held-out cameras; original COLMAP coordinates",
            "bands": "lower inclusive, upper exclusive; 50+ includes exactly 50m",
            "shares": "countShare and opacityShare divide by ALL "
            "rows / summed sigmoid(logit opacity); "
            "farField rows are subsets, not extra budget",
            "farField": "PLY far_field >= 0.5 only; absent marker means unknown, "
            "never inferred from distance",
            "projectedPixelArea": "not computed: requires every splat projected over training "
            "cameras; opacity weight is not screen coverage",
        },
    }
    if scale["status"] != "estimated":
        return {**report, "status": "unavailable", "bands": None, "beyond": None}
    sums = np.zeros((2, len(EDGES), 2), dtype=float)
    beyond = np.zeros((2, 2, 2), dtype=float)
    names = ["x", "y", "z", "opacity"] + (["far_field"] if report["farFieldMarkerPresent"] else [])
    before = ply.stat()
    for _, columns in reader.chunks(names):
        xyz = np.stack([columns[k] for k in ("x", "y", "z")], axis=1)
        opacity = np.asarray(columns["opacity"], dtype=float)
        if not np.isfinite(opacity).all():
            raise ValueError(f"{ply}: nonfinite opacity")
        weights = np.exp(-np.logaddexp(0, -opacity))
        distances = distance_to_path(xyz, geometry["path"]["centres"]) * scale["metresPerUnit"]
        bands = np.searchsorted(EDGES[1:], distances, side="right")
        far = columns.get("far_field", np.zeros(len(xyz)))
        if not np.isfinite(far).all():
            raise ValueError(f"{ply}: nonfinite far_field marker")
        for j, selected in enumerate((np.ones(len(xyz), dtype=bool), far >= 0.5)):
            sums[j, :, 0] += np.bincount(bands[selected], minlength=len(EDGES))
            sums[j, :, 1] += np.bincount(
                bands[selected], weights=weights[selected], minlength=len(EDGES)
            )
            for k, threshold in enumerate((10, 20)):
                keep = selected & (distances > threshold)
                beyond[j, k] += [int(keep.sum()), weights[keep].sum()]
    after = ply.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{ply}: file changed during measurement")
    total_opacity = float(sums[0, :, 1].sum())

    def row(values: Any) -> dict[str, Any]:
        return {
            "count": int(values[0]),
            "countShare": float(values[0] / reader.count) if reader.count else None,
            "opacitySum": float(values[1]),
            "opacityShare": float(values[1] / total_opacity) if total_opacity else None,
        }

    report.update(
        status="complete",
        opacitySum=total_opacity,
        bands=[
            {
                "bandM": label,
                **row(sums[0, i]),
                "farField": row(sums[1, i]) if report["farFieldMarkerPresent"] else None,
            }
            for i, label in enumerate(LABELS)
        ],
        beyond={
            str(threshold): {
                **row(beyond[0, i]),
                "farField": row(beyond[1, i]) if report["farFieldMarkerPresent"] else None,
            }
            for i, threshold in enumerate((10, 20))
        },
        farField=row(sums[1].sum(axis=0)) if report["farFieldMarkerPresent"] else None,
    )
    return report


def markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Camera-path distance yardstick",
        "",
        "Approximate metres from a fitted local ground plane and assumed phone height; "
        "not surveyed distances.",
        "",
    ]
    scale = document["geometry"]["scale"]
    if scale["status"] == "estimated":
        lines += [
            f"Assumed height: {scale['assumedCameraHeightM']:.2f} m. "
            f"Scale: {scale['metresPerUnit']:.4f} m/COLMAP unit. "
            f"Ground support: {scale['inlierPoints']}/{scale['localPoints']} local SfM points.",
            "",
        ]
    else:
        lines += [f"Scale unavailable: {scale['reason']}", ""]
    for report in document["rows"]:
        lines += [f"## {report['ply']}", ""]
        if report["status"] != "complete":
            lines += ["No approximate-metre bands: scale unavailable.", ""]
            continue
        lines += [
            "| Approx. m | Count | Count share | Opacity share | Marked far count |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for band in report["bands"]:
            far = band["farField"]["count"] if band["farField"] is not None else "unknown"
            count = "n/a" if band["countShare"] is None else f"{band['countShare']:.2%}"
            opacity = "n/a" if band["opacityShare"] is None else f"{band['opacityShare']:.2%}"
            lines.append(f"| {band['bandM']} | {band['count']:,} | {count} | {opacity} | {far} |")
        lines.append("")
        for threshold, values in report["beyond"].items():
            if values["countShare"] is not None:
                lines.append(
                    f"Beyond {threshold} m: {values['countShare']:.2%} of the splat budget "
                    f"({values['count']:,} rows)."
                )
        lines += [
            "",
            "Far-field markings are reported only when present in the PLY. "
            "Opacity weighting is sigmoid(logit opacity), not visibility or projected area.",
            "",
        ]
    return "\n".join(lines)


def run(
    scene: Path, plys: list[Path], output: Path | None, height_m: float = 1.4
) -> dict[str, Any]:
    geometry = scene_geometry(scene, height_m)
    document = {
        "protocol": "camera-path-distance-v1",
        "geometry": geometry,
        "rows": [measure_ply(ply, geometry) for ply in plys],
    }
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "distance.json").write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n"
        )
        (output / "distance.md").write_text(markdown(document))
    return document
