"""Explicit affine image registration; fit residuals are not absolute positional accuracy."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from affine import Affine
from pyproj import CRS, Geod, Transformer
from shapely.geometry import MultiPoint

from app.schemas.geojson import Polygon
from app.schemas.land_image_registrations import (
    ControlPointFit,
    ImageRegistrationCreate,
    ImageRegistrationResult,
)


def fit(request: ImageRegistrationCreate, width: int, height: int) -> ImageRegistrationResult:
    pixels = np.array([[point.image_x, point.image_y] for point in request.points])
    locations = np.array([[point.longitude, point.latitude] for point in request.points])
    if np.any(pixels[:, 0] > width) or np.any(pixels[:, 1] > height):
        raise ValueError("A control point is outside the saved image.")
    if np.ptp(locations[:, 0]) > 10 or np.ptp(locations[:, 1]) > 10:
        raise ValueError(
            "Align a regional map within 10 degrees; date-line crossings need a separate projection workflow."
        )
    longitude, latitude = locations.mean(axis=0)
    geod = Geod(ellps="WGS84")
    if abs(latitude) > 85 or any(
        geod.inv(longitude, latitude, x, y)[2] > 125_000 for x, y in locations
    ):
        raise ValueError(
            "Choose control points within a 250 km regional extent, away from the poles."
        )
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude:.12f} +lon_0={longitude:.12f} +datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    reverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    targets = np.column_stack(forward.transform(locations[:, 0], locations[:, 1]))
    # Normalized coordinates make the rank/conditioning test independent of image dimensions.
    design = np.column_stack(
        (pixels[:, 0] / width - 0.5, pixels[:, 1] / height - 0.5, np.ones(len(pixels)))
    )
    coefficients, _residuals, rank, singular = np.linalg.lstsq(design, targets, rcond=None)
    if rank < 3 or singular[-1] / singular[0] < 1e-4:
        raise ValueError(
            "Control points are collinear or too narrowly spaced. Spread them across the image."
        )
    a, b = coefficients[0] / width, coefficients[1] / height
    c = coefficients[2] - coefficients[0] / 2 - coefficients[1] / 2
    affine = Affine(float(a[0]), float(b[0]), float(c[0]), float(a[1]), float(b[1]), float(c[1]))
    scale = float(np.sqrt(abs(affine.determinant)))
    if scale < 0.001 or not np.isfinite(scale):
        raise ValueError("The map positions collapse the image or imply an unsupported scale.")
    predicted = design @ coefficients
    longitude_fit, latitude_fit = reverse.transform(predicted[:, 0], predicted[:, 1])
    errors = np.array(
        [
            geod.inv(x, y, px, py)[2]
            for (x, y), px, py in zip(locations, longitude_fit, latitude_fit, strict=True)
        ]
    )
    fits = []
    loo_errors = []
    for index, point in enumerate(request.points):
        keep = np.arange(len(pixels)) != index
        leave_coefficients, _, leave_rank, leave_singular = np.linalg.lstsq(
            design[keep], targets[keep], rcond=None
        )
        leave_error = None
        if leave_rank == 3 and leave_singular[-1] / leave_singular[0] >= 1e-4:
            px, py = reverse.transform(*(design[index] @ leave_coefficients))
            leave_error = float(geod.inv(point.longitude, point.latitude, px, py)[2])
            loo_errors.append(leave_error)
        fits.append(
            ControlPointFit(
                label=point.label,
                predicted_longitude=float(longitude_fit[index]),
                predicted_latitude=float(latitude_fit[index]),
                error_m=float(errors[index]),
                leave_one_out_error_m=leave_error,
            )
        )
    # Densify edges because a straight local-projection edge is curved in longitude/latitude.
    corners = [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]
    ring = []
    for first, second in pairwise(corners):
        for t in np.linspace(0, 1, 17)[:-1]:
            pixel = (first[0] + t * (second[0] - first[0]), first[1] + t * (second[1] - first[1]))
            east, north = affine * pixel
            if np.hypot(east, north) > 250_000:
                raise ValueError(
                    "The image extent extrapolates beyond 250 km. Check the point matches and coverage."
                )
            x, y = reverse.transform(east, north)
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError("The fitted image extent cannot be projected.")
            ring.append([float(x), float(y)])
    ring.append(ring[0])
    if np.ptp([p[0] for p in ring]) > 10:
        raise ValueError(
            "The fitted image crosses the date line or exceeds the supported regional extent."
        )
    footprint = Polygon(coordinates=[ring])
    bounds = [
        min(p[0] for p in ring),
        min(p[1] for p in ring),
        max(p[0] for p in ring),
        max(p[1] for p in ring),
    ]
    coverage = float(MultiPoint(pixels).convex_hull.area / (width * height))
    rms = float(np.sqrt(np.mean(errors**2)))
    warnings = [
        "Fit error measures agreement with the selected control points; it does not establish surveyed accuracy.",
        "An affine fit cannot remove all distortions in historical maps. "
        "Alignment does not establish boundaries or rights.",
    ]
    if coverage < 0.25:
        warnings.append(
            "Control points cover less than a quarter of the image. Alignment outside them is extrapolated."
        )
    if rms > scale * 2:
        warnings.append(
            "Average fit error exceeds two source-image pixels. Check the point matches."
        )
    if min(width, height) < 512:
        warnings.append(
            f"Fine-scale alignment is limited by this {width} x {height}-pixel preview."
        )
    if affine.determinant > 0:
        warnings.append(
            "The fitted map is mirrored. Check that image and map points are paired correctly."
        )
    if len(loo_errors) != len(pixels):
        warnings.append(
            "Some leave-one-out checks are undefined because the remaining points are poorly distributed."
        )
    return ImageRegistrationResult(
        crs=crs.to_wkt(),
        transform=list(affine)[:6],
        image_width=width,
        image_height=height,
        footprint=footprint,
        bounds=bounds,
        rms_error_m=rms,
        maximum_error_m=float(errors.max()),
        leave_one_out_rms_m=float(np.sqrt(np.mean(np.square(loo_errors))))
        if len(loo_errors) == len(pixels)
        else None,
        control_point_coverage=coverage,
        approximate_m_per_pixel=scale,
        points=fits,
        warnings=warnings,
    )
