"""Where a pipeline-registered splat sits on the globe, and the footprint drawn round it.

A run's splat is placed east/north/up about the coordinate its georeference found: that is
the tileset's root transform (tools/pipeline `splat_tiles`), and `registration.json` keeps
it as `georef.lat`/`lon`/`height`. Two things draw on the globe about that origin, and both
go through here so that they agree on it and on how metres become degrees:

* `app/worker/registration.py`, which outlines a new site as the splat's own bounding box
  (`boundary`) when it registers a run;
* `app/services/assets.py`'s `set_scale`, which resizes the splat at runtime about the same
  origin and moves everything the catalog says about where it is with it (`rescale_site`,
  `rescale_footprint`, `rescale_ground_samples`): the site's boundary and centroid, the
  asset's footprint, its measured ground. A re-run undoes it the same way before the new
  tileset arrives at its own scale.

Rescaling is about the origin by a ratio -- the new scale over the old -- rather than a
fresh outline from the bounding box, so a boundary somebody redrew (`PATCH /sites/{id}`)
is resized as drawn, not replaced. `boundary`'s rectangle is linear in degrees about the
origin, so for one it never redrew the two are the same thing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from pydantic import ValidationError
from shapely import affinity

from app.models import Asset, Site
from app.schemas.asset import GroundSample
from app.schemas.geojson import Polygon
from app.services import geometry
from app.services.errors import InvalidInputError

#: Half-width of the fallback footprint, in metres, around the placed coordinate.
#:
#: A capture's real extent is a measurement, and A8's `manifest` stage supplies it:
#: `registration.json` now carries the packaged splat's own local bounding box, and the
#: boundary below is that box on the globe. This square is what a run with no manifest
#: still gets -- a recipe without the stage, or a run under the stub -- and it stays 60 m
#: rather than being invented smaller to look precise.
PLACEHOLDER_HALF_EXTENT_M = 30.0

#: No side of a site's boundary is allowed to be thinner than this, in metres.
#:
#: A capture with no extent along an axis (one gaussian, a flat wall scanned face-on) would
#: otherwise produce a degenerate polygon that PostGIS accepts and nothing can be clicked on.
MIN_HALF_EXTENT_M = 0.5

METRES_PER_DEGREE = 111_320.0


@dataclass(frozen=True)
class Origin:
    """The placed coordinate a splat's tileset is east/north/up about: the root transform."""

    lat: float
    lon: float
    #: Ellipsoid height of the origin, in metres: what `groundSamples`' heights are over.
    height: float


def rectangle(
    lat: float, lon: float, east: tuple[float, float], north: tuple[float, float]
) -> Polygon:
    """A boundary in degrees from metre offsets east and north of the placed coordinate."""
    scale = max(math.cos(math.radians(lat)), 1e-6)
    west_deg, east_deg = (v / (METRES_PER_DEGREE * scale) for v in east)
    south_deg, north_deg = (v / METRES_PER_DEGREE for v in north)
    ring = [
        [lon + west_deg, lat + south_deg],
        [lon + east_deg, lat + south_deg],
        [lon + east_deg, lat + north_deg],
        [lon + west_deg, lat + north_deg],
        [lon + west_deg, lat + south_deg],
    ]
    return Polygon(type="Polygon", coordinates=[ring])


def boundary(lat: float, lon: float, box: tuple[list[float], list[float]] | None) -> Polygon:
    """A site's footprint: the capture's measured extent, or the placeholder square.

    A7 drew a 60 m square around the placed coordinate and said in a comment that it was
    waiting for A8's measured extent. This is that extent -- the packaged splat's own
    bounding box (`box`, east/north/up metres about the origin), which is the thing the
    console will draw a site outline around, offset from the placed origin exactly as the
    gaussians are.
    """
    if box is None:
        half = PLACEHOLDER_HALF_EXTENT_M
        return rectangle(lat, lon, (-half, half), (-half, half))
    (min_e, min_n, _), (max_e, max_n, _) = box
    return rectangle(lat, lon, _widen(min_e, max_e), _widen(min_n, max_n))


def _widen(low: float, high: float) -> tuple[float, float]:
    if high - low >= 2 * MIN_HALF_EXTENT_M:
        return (low, high)
    middle = (low + high) / 2
    return (middle - MIN_HALF_EXTENT_M, middle + MIN_HALF_EXTENT_M)


def registered_origin(metadata: Mapping[str, Any] | None) -> Origin | None:
    """The origin a site's registered run placed its splat at, or None if it has none.

    `registration.georef` on the site's metadata: the worker writes the run's whole
    `registration.json` there, and refreshes it when a re-run repoints the site. A site
    made by hand, or seeded, has none -- nothing says where its tiles' origin is.
    """
    registration = (metadata or {}).get("registration")
    georef = registration.get("georef") if isinstance(registration, Mapping) else None
    if not isinstance(georef, Mapping):
        return None
    lat, lon = _number(georef.get("lat")), _number(georef.get("lon"))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return Origin(lat=lat, lon=lon, height=_number(georef.get("height")) or 0.0)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None


def rescale_site(site: Site, origin: Origin, factor: float) -> None:
    """Resize the site's boundary about `origin` by `factor`, and move its centroid with it."""
    site.boundary = _rescaled_footprint(site.boundary, origin, factor, "the site's boundary")
    centroid = affinity.scale(
        to_shape(site.centroid), xfact=factor, yfact=factor, origin=(origin.lon, origin.lat)
    )
    site.centroid = from_shape(centroid, srid=geometry.SRID)


def rescale_footprint(asset: Asset, origin: Origin, factor: float) -> None:
    """Resize the asset's own footprint about `origin` by `factor`, when it has one."""
    if asset.footprint is not None:
        asset.footprint = _rescaled_footprint(
            asset.footprint, origin, factor, "the asset's footprint"
        )


def rescale_ground_samples(samples: object, origin: Origin, factor: float) -> list[dict[str, Any]]:
    """The render config's `groundSamples`, moved as the splat's ground moves.

    A cell `(e, n, u)` metres from the origin is at `factor * (e, n, u)` once the splat is
    resized about it, so its height over the origin scales too: the clamp then compares the
    resized model's ground with the terrain at the place that ground now is.
    """
    out: list[dict[str, Any]] = []
    for row in samples if isinstance(samples, list) else []:
        try:
            sample = GroundSample.model_validate(row)
            moved = GroundSample(
                lon=origin.lon + (sample.lon - origin.lon) * factor,
                lat=origin.lat + (sample.lat - origin.lat) * factor,
                height=origin.height + (sample.height - origin.height) * factor,
            )
        except ValidationError as error:
            raise InvalidInputError(
                f"a scale of {factor:g} times would move a ground sample off the globe"
            ) from error
        out.append(moved.model_dump(mode="json", by_alias=True))
    return out


def _rescaled_footprint(value: WKBElement, origin: Origin, factor: float, what: str) -> WKBElement:
    scaled = affinity.scale(
        to_shape(value), xfact=factor, yfact=factor, origin=(origin.lon, origin.lat)
    )
    stored = from_shape(scaled, srid=geometry.SRID)
    # Read back through the same validation every footprint gets on the wire, so a resize
    # that would leave the globe is a 422 now rather than a site nobody can read later.
    try:
        geometry.wkb_to_footprint(stored)
    except (ValidationError, ValueError) as error:
        raise InvalidInputError(
            f"a scale of {factor:g} times would move {what} off the globe"
        ) from error
    return stored
