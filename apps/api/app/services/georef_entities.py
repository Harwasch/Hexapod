"""ARKit metres -> local ENU -> WGS84, without assuming an unmeasured heading.

No Apple/numpy dependencies. Placement input must be from the *same* capture and
ARKit-seeded training frame. Device accuracy and clock correlation need field validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from typing import Any

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
A = 6378137.0
E2 = 6.6943799901413165e-3


def finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("coordinates must be finite")


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    alt: float

    def __post_init__(self) -> None:
        finite(self.lat, self.lon, self.alt)
        if not -90 <= self.lat <= 90 or not -180 <= self.lon <= 180:
            raise ValueError("WGS84 coordinate out of range")


def geodetic_to_ecef(point: GeoPoint) -> Vec3:
    lat, lon = math.radians(point.lat), math.radians(point.lon)
    prime = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return (
        (prime + point.alt) * math.cos(lat) * math.cos(lon),
        (prime + point.alt) * math.cos(lat) * math.sin(lon),
        (prime * (1 - E2) + point.alt) * math.sin(lat),
    )


def ecef_to_geodetic(point: Vec3) -> GeoPoint:
    finite(*point)
    x, y, z = point
    radius = math.hypot(x, y)
    if math.hypot(radius, z) < 1:
        raise ValueError("geodetic coordinates undefined at Earth's centre")
    lat = math.atan2(z, radius * (1 - E2))
    for _ in range(12):
        prime = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        lat = math.atan2(z + E2 * prime * math.sin(lat), radius)
    # Dot onto the ellipsoid normal remains stable at both poles and equator.
    alt = radius * math.cos(lat) + z * math.sin(lat) - A * math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return GeoPoint(math.degrees(lat), math.degrees(math.atan2(y, x)), alt)


def _basis(origin: GeoPoint) -> Mat3:
    lat, lon = math.radians(origin.lat), math.radians(origin.lon)
    return (
        (-math.sin(lon), math.cos(lon), 0),
        (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)),
        (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)),
    )


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def ecef_to_enu(point: Vec3, origin: GeoPoint) -> Vec3:
    finite(*point)
    anchor = geodetic_to_ecef(origin)
    delta = (point[0] - anchor[0], point[1] - anchor[1], point[2] - anchor[2])
    e, n, u = _basis(origin)
    return dot(e, delta), dot(n, delta), dot(u, delta)


def enu_to_ecef(point: Vec3, origin: GeoPoint) -> Vec3:
    finite(*point)
    e, n, u = _basis(origin)
    x, y, z = geodetic_to_ecef(origin)
    return (
        x + e[0] * point[0] + n[0] * point[1] + u[0] * point[2],
        y + e[1] * point[0] + n[1] * point[1] + u[1] * point[2],
        z + e[2] * point[0] + n[2] * point[1] + u[2] * point[2],
    )


@dataclass(frozen=True)
class Similarity:
    rotation: Mat3
    translation: Vec3
    scale: float = 1

    def __post_init__(self) -> None:
        finite(self.scale, *self.translation, *(v for row in self.rotation for v in row))
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        # Float32 serialization bound, not a fitted accuracy tolerance.
        for i in range(3):
            for j in range(3):
                if abs(dot(self.rotation[i], self.rotation[j]) - (i == j)) > 1e-5:
                    raise ValueError("rotation must be orthonormal")
        a, b, c = self.rotation
        cross = (b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0])
        if dot(a, cross) < 0:
            raise ValueError("rotation must preserve handedness")

    def apply(self, point: Vec3) -> Vec3:
        finite(*point)
        r, t, s = self.rotation, self.translation, self.scale
        return s * dot(r[0], point) + t[0], s * dot(r[1], point) + t[1], s * dot(r[2], point) + t[2]

    def inverse(self, point: Vec3) -> Vec3:
        finite(*point)
        p = tuple((point[i] - self.translation[i]) / self.scale for i in range(3))
        r = self.rotation
        return (
            r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2],
            r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2],
            r[0][2] * p[0] + r[1][2] * p[1] + r[2][2] * p[2],
        )


def arkit_to_enu(heading_rad: float, translation: Vec3 = (0, 0, 0)) -> Similarity:
    """Heading clockwise from north of ARKit -Z; at zero: (x,y,z)->(x,-z,y)."""
    finite(heading_rad)
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return Similarity(((c, 0, -s), (-s, 0, -c), (0, 1, 0)), translation)


@dataclass(frozen=True)
class TrackPair:
    world: Vec3
    enu: Vec3
    accuracy_m: float
    vertical_valid: bool = True

    def __post_init__(self) -> None:
        finite(*self.world, *self.enu, self.accuracy_m)
        if self.accuracy_m <= 0:
            raise ValueError("accuracy must be positive")


@dataclass(frozen=True)
class HeadingFit:
    transform: Similarity
    heading_rad: float
    residual_m: float
    uncertainty_m: float
    heading_uncertainty_rad: float
    vertical_known: bool


def fit_heading(pairs: Sequence[TrackPair], *, min_baseline_m: float = 10) -> HeadingFit | None:
    """Weighted 2D rigid fit, retaining ARKit metric scale. Reject short/noisy walks.

    Policy: >=4 paired fixes, baseline >=10m and >=3x RMS GPS accuracy,
    residual <=2x accuracy and scale disagreement <=25%. These are acceptance
    policies, not claims of measured device accuracy. No heading from a stationary walk.
    """
    if len(pairs) < 4:
        return None
    weights = [1 / p.accuracy_m**2 for p in pairs]
    total = sum(weights)
    wx = sum(w * p.world[0] for w, p in zip(weights, pairs, strict=True)) / total
    wz = sum(w * -p.world[2] for w, p in zip(weights, pairs, strict=True)) / total
    east = sum(w * p.enu[0] for w, p in zip(weights, pairs, strict=True)) / total
    north = sum(w * p.enu[1] for w, p in zip(weights, pairs, strict=True)) / total
    spread = cross = along = target_spread = 0.0
    for w, p in zip(weights, pairs, strict=True):
        x, z, e, n = p.world[0] - wx, -p.world[2] - wz, p.enu[0] - east, p.enu[1] - north
        spread += w * (x * x + z * z)
        target_spread += w * (e * e + n * n)
        along += w * (x * e + z * n)
        cross += w * (z * e - x * n)
    accuracy = math.sqrt(len(pairs) / total)
    baseline = 2 * math.sqrt(spread / total)
    if baseline < max(min_baseline_m, 3 * accuracy) or spread == 0:
        return None
    if abs(math.sqrt(target_spread / spread) - 1) > 0.25:
        return None
    heading = math.atan2(cross, along)
    rotation = arkit_to_enu(heading)
    centroid = rotation.apply((wx, 0, -wz))
    # Vertical translation is only usable when callers provide a verified height datum.
    up = sum(w * (p.enu[2] - p.world[1]) for w, p in zip(weights, pairs, strict=True)) / total
    transform = arkit_to_enu(heading, (east - centroid[0], north - centroid[1], up))
    residual = math.sqrt(
        sum(
            w * sum((transform.apply(p.world)[i] - p.enu[i]) ** 2 for i in (0, 1))
            for w, p in zip(weights, pairs, strict=True)
        )
        / total
    )
    if residual > 2 * accuracy:
        return None
    uncertainty = max(accuracy, residual)
    return HeadingFit(
        transform,
        heading,
        residual,
        uncertainty,
        math.atan2(uncertainty, baseline / 2),
        all(p.vertical_valid for p in pairs),
    )


def pair_bundle_tracks(
    frames: Sequence[Mapping[str, Any]],
    fixes: Sequence[Mapping[str, Any]],
    *,
    frame_zero_unix: float,
    origin: GeoPoint,
    max_gap_s: float = 1,
) -> list[TrackPair]:
    """Interpolate normal-tracking poses at GPS timestamps, never extrapolate.

    `frame_zero_unix` is an explicit clock correlation supplied by capture ingestion;
    manifest.startedAt is NOT silently assumed to equal the first AR frame's time.
    Invalid vertical fixes use origin height for the horizontal-only fit.
    """
    finite(frame_zero_unix, max_gap_s)
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    times = [float(f["timestamp"]) for f in frames]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in pairwise(times)):
        raise ValueError("frame timestamps must strictly increase")
    result = []
    for fix in fixes:
        accuracy = float(fix["horizontalAccuracyM"])
        if not math.isfinite(accuracy) or accuracy <= 0:
            continue
        time = float(fix["timestamp"]) - frame_zero_unix
        for i in range(len(frames) - 1):
            left, right = frames[i], frames[i + 1]
            if not times[i] <= time <= times[i + 1] or times[i + 1] - times[i] > max_gap_s:
                continue
            if left["trackingState"] != "normal" or right["trackingState"] != "normal":
                break
            fraction = (time - times[i]) / (times[i + 1] - times[i])
            a, b = left["transform"], right["transform"]
            world: Vec3 = (
                a[12] + fraction * (b[12] - a[12]),
                a[13] + fraction * (b[13] - a[13]),
                a[14] + fraction * (b[14] - a[14]),
            )
            point = GeoPoint(float(fix["lat"]), float(fix["lon"]), origin.alt)
            result.append(
                TrackPair(
                    world,
                    ecef_to_enu(geodetic_to_ecef(point), origin),
                    accuracy,
                    vertical_valid=False,
                )
            )
            break
    return result


@dataclass(frozen=True)
class Alignment:
    origin: GeoPoint
    source: str
    transform: Similarity | None
    uncertainty_m: float | None
    height_datum: str
    heading_uncertainty_rad: float | None = None
    vertical_known: bool = True

    def __post_init__(self) -> None:
        if self.uncertainty_m is not None:
            finite(self.uncertainty_m)
            if self.uncertainty_m < 0:
                raise ValueError("uncertainty must be nonnegative")
        if self.heading_uncertainty_rad is not None:
            finite(self.heading_uncertainty_rad)
            if not 0 <= self.heading_uncertainty_rad <= math.pi:
                raise ValueError("heading uncertainty out of range")

    def locate(self, world: Vec3) -> GeoPoint | None:
        # EntitySchema.geo.alt specifically means ellipsoid height.
        if (
            self.transform is None
            or self.height_datum != "ellipsoid-wgs84"
            or not self.vertical_known
        ):
            return None
        return ecef_to_geodetic(enu_to_ecef(self.transform.apply(world), self.origin))

    def uncertainty_at(self, world: Vec3) -> float | None:
        if self.transform is None or self.uncertainty_m is None:
            return None
        # Keep shared GPS error: averaging observations must not shrink it away.
        angle = self.heading_uncertainty_rad
        return self.uncertainty_m + (0 if angle is None else math.hypot(world[0], world[2]) * angle)


def _vec(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("expected three coordinates")
    return float(value[0]), float(value[1]), float(value[2])


def resolve_alignment(
    *,
    georef: Mapping[str, Any] | None = None,
    placement: Mapping[str, Any] | None = None,
    arkit_training_frame_verified: bool = False,
    site_frame: Mapping[str, Any] | None = None,
    gps: Mapping[str, Any] | None = None,
    track_fit: HeadingFit | None = None,
    height_datum: str = "msl-unverified",
) -> Alignment | None:
    """Priority: capture placement+georef, chosen site frame, GPS.

    `track_fit` must be expressed about the chosen origin. A site origin is not
    a measured translation from the session. Without a fit, transform stays None.
    """
    origin_uncertainty: float | None = None
    if georef is not None:
        origin = GeoPoint(float(georef["lat"]), float(georef["lon"]), float(georef["height"]))
        frame = georef.get("frame", {})
        known_heading = frame.get("headingSource") in ("capture", "exif-gps")
        if placement is None and frame.get("recentre") is False and known_heading:
            placement = {
                "rotation": frame["rotation"],
                "translation": frame["translationM"],
                "scale": frame["scale"],
            }
        origin_uncertainty = georef.get("uncertaintyM")
        if placement is not None and arkit_training_frame_verified and known_heading:
            rows = placement["rotation"]
            if len(rows) != 3:
                raise ValueError("expected 3x3 rotation")
            transform = Similarity(
                (_vec(rows[0]), _vec(rows[1]), _vec(rows[2])),
                _vec(placement["translation"]),
                float(placement["scale"]),
            )
            uncertainty = georef.get("uncertaintyM")
            if uncertainty is not None and (not math.isfinite(uncertainty) or uncertainty < 0):
                raise ValueError("invalid placement uncertainty")
            return Alignment(origin, "pipeline", transform, uncertainty, height_datum)
        source = "pipeline"
    elif site_frame is not None:
        origin = GeoPoint(
            float(site_frame["lat"]), float(site_frame["lon"]), float(site_frame["height"])
        )
        height_datum = str(site_frame["heightDatum"])
        source = "site"
    elif gps is not None and float(gps["horizontalAccuracyM"]) > 0:
        origin_uncertainty = float(gps["horizontalAccuracyM"])
        finite(origin_uncertainty)
        altitude = (
            float(gps.get("altitudeM", 0)) if float(gps.get("verticalAccuracyM", -1)) >= 0 else 0
        )
        origin = GeoPoint(float(gps["lat"]), float(gps["lon"]), altitude)
        source = "gps"
        # CoreLocation altitude is not an ellipsoid measurement. A caller may supply
        # a geoid-corrected origin through site_frame; never put raw MSL in geo.alt.
        height_datum = "msl-unverified"
    else:
        return None
    return Alignment(
        origin,
        source,
        None if track_fit is None else track_fit.transform,
        origin_uncertainty if track_fit is None else track_fit.uncertainty_m,
        height_datum,
        None if track_fit is None else track_fit.heading_uncertainty_rad,
        track_fit is not None and track_fit.vertical_known,
    )


def align_bundle(
    *,
    frames: Sequence[Mapping[str, Any]],
    fixes: Sequence[Mapping[str, Any]],
    frame_zero_unix: float | None = None,
    georef: Mapping[str, Any] | None = None,
    placement: Mapping[str, Any] | None = None,
    site_frame: Mapping[str, Any] | None = None,
    arkit_training_frame_verified: bool = False,
    height_datum: str = "msl-unverified",
) -> Alignment | None:
    """Ingestion entry point for parsed frames.jsonl/location.jsonl and stored sidecars.

    Raw CoreLocation altitude is not ellipsoid altitude. A GPS-only horizontal fit
    remains useful provenance, but geo.alt stays absent until a vertical tie exists.
    """
    usable = [
        fix
        for fix in fixes
        if math.isfinite(float(fix["horizontalAccuracyM"]))
        and float(fix["horizontalAccuracyM"]) > 0
    ]
    gps = min(usable, key=lambda fix: float(fix["horizontalAccuracyM"])) if usable else None
    alignment = resolve_alignment(
        georef=georef,
        placement=placement,
        site_frame=site_frame,
        gps=gps,
        arkit_training_frame_verified=arkit_training_frame_verified,
        height_datum=height_datum,
    )
    if alignment is None or alignment.transform is not None or frame_zero_unix is None:
        return alignment
    fit = fit_heading(
        pair_bundle_tracks(frames, fixes, frame_zero_unix=frame_zero_unix, origin=alignment.origin)
    )
    if fit is None:
        return alignment
    return replace(
        alignment,
        transform=fit.transform,
        uncertainty_m=fit.uncertainty_m,
        heading_uncertainty_rad=fit.heading_uncertainty_rad,
        vertical_known=fit.vertical_known,
    )


def georeference_entity(entity: Mapping[str, Any], alignment: Alignment | None) -> dict[str, Any]:
    """B calls before persisting a validated Entity. Never mutate caller-owned evidence."""
    result = deepcopy(dict(entity))
    anchor = result["anchor"]
    world = anchor["world"]
    point = (float(world["x"]), float(world["y"]), float(world["z"]))
    finite(*point)
    geo = None if alignment is None else alignment.locate(point)
    # A previous run's anchor must not survive a newer unresolved alignment.
    anchor.pop("geo", None)
    if geo is not None:
        anchor["geo"] = asdict(geo)
    metadata = {
        "source": None if alignment is None else alignment.source,
        "transform": None
        if alignment is None or alignment.transform is None
        else {
            "rotation": [list(row) for row in alignment.transform.rotation],
            "translation": list(alignment.transform.translation),
            "scale": alignment.transform.scale,
        },
        "headingKnown": alignment is not None and alignment.transform is not None,
        "heightDatum": None if alignment is None else alignment.height_datum,
        "verticalKnown": alignment is not None and alignment.vertical_known,
        "uncertaintyM": None if alignment is None else alignment.uncertainty_at(point),
        "originUncertaintyM": None if alignment is None else alignment.uncertainty_m,
        "origin": None if alignment is None else asdict(alignment.origin),
    }
    result.setdefault("attrs", {})["georeference"] = metadata
    return result
