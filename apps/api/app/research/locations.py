"""Where a research step is looking, as a small point, box or saved-record reference.

Events carry these so the agent card can draw a line from the step it is on to that place
while it works, and drop what it found onto the map. Geometry itself never travels in an
event: a stream frame is bounded, so every location here is a point, a box or a reference.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel
from shapely.geometry import shape

from app.research.providers.archives import commons_radius
from app.research.providers.base import SourceContext
from app.schemas.research import EventLocation, EventLocationRef

Meaning = Literal[
    "sample-point",
    "query-bbox",
    "query-radius",
    "catalog-coordinate",
    "catalog-footprint",
    "analysis-extent",
    "array-zone",
    "focus",
    "result",
]

# Providers that ask about one representative point of the land.
POINT_PROVIDERS = frozenset({"usgs-elevation", "nasa-power", "usda-soils", "usda-ecological-sites"})
# Providers that query the land's bounding box.
BOX_PROVIDERS = frozenset({"gbif-occurrences", "fema-flood-zones", "usgs-historical-maps"})
# Providers that query the land polygon itself.
EXTENT_PROVIDERS = frozenset({"epa-ecoregions"})


def _round(value: float) -> float:
    return round(float(value), 7)


def point(
    lon: float,
    lat: float,
    meaning: Meaning,
    *,
    radius_m: float | None = None,
    ref: EventLocationRef | None = None,
) -> EventLocation:
    return EventLocation(
        kind="point",
        point=(_round(lon), _round(lat)),
        radius_m=radius_m,
        ref=ref,
        meaning=meaning,
    )


def box(
    bounds: Iterable[float], meaning: Meaning, *, ref: EventLocationRef | None = None
) -> EventLocation:
    west, south, east, north = (float(value) for value in bounds)
    return EventLocation(
        kind="bbox",
        bbox=(_round(west), _round(south), _round(east), _round(north)),
        ref=ref,
        meaning=meaning,
    )


def _geometry(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    return value.model_dump() if isinstance(value, BaseModel) else dict(value)


def geometry_location(
    geometry: Mapping[str, Any] | BaseModel | None,
    meaning: Meaning,
    *,
    ref: EventLocationRef | None = None,
) -> EventLocation | None:
    """A point stays a point; any other shape becomes its bounding box."""
    if geometry is None:
        return None
    data = _geometry(geometry)
    try:
        if data.get("type") == "Point":
            lon, lat = data["coordinates"][:2]
            return point(lon, lat, meaning, ref=ref)
        bounds = shape(data).bounds
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if len(bounds) != 4 or any(value != value for value in bounds):  # empty or NaN
        return None
    return box(bounds, meaning, ref=ref)


def union_location(
    geometries: Iterable[Mapping[str, Any] | BaseModel],
    meaning: Meaning,
    *,
    ref: EventLocationRef | None = None,
) -> EventLocation | None:
    """One box around several shapes (a map's features, a gallery's catalog locations)."""
    found = [
        location
        for location in (geometry_location(item, meaning) for item in geometries)
        if location is not None
    ]
    if not found:
        return None
    if len(found) == 1:
        only = found[0]
        return only.model_copy(update={"ref": ref}) if ref else only
    west = south = float("inf")
    east = north = float("-inf")
    for location in found:
        corners = location.bbox or (*(location.point or (0, 0)), *(location.point or (0, 0)))
        west, south = min(west, corners[0]), min(south, corners[1])
        east, north = max(east, corners[2]), max(north, corners[3])
    return box((west, south, east, north), meaning, ref=ref)


def land_box(context: SourceContext, meaning: Meaning = "query-bbox") -> EventLocation:
    return box(context.geometry.bounds, meaning)


def provider_location(provider: str, context: SourceContext) -> EventLocation | None:
    """What a registered source asks about: its sample point, query box or search circle."""
    if provider in POINT_PROVIDERS:
        lon, lat = context.point
        return point(lon, lat, "sample-point")
    if provider in BOX_PROVIDERS:
        return land_box(context, "query-bbox")
    if provider in EXTENT_PROVIDERS:
        return land_box(context, "analysis-extent")
    if provider == "commons-place-images":
        lon, lat = context.point
        return point(lon, lat, "query-radius", radius_m=float(commons_radius(context)))
    return None


def focus_location(snapshot: Mapping[str, Any] | None) -> EventLocation | None:
    """The map feature a question is about, pinned when the run was created."""
    if not snapshot:
        return None
    feature = snapshot.get("feature") or {}
    ref = EventLocationRef(
        artifact_id=uuid.UUID(str(snapshot["artifactId"])),
        feature_id=str(snapshot["featureIndex"]),
    )
    return geometry_location(feature.get("geometry"), "focus", ref=ref)


def media_location(
    media: Mapping[str, Any] | BaseModel | None, evidence_id: uuid.UUID | None = None
) -> EventLocation | None:
    """An archive item's catalog coordinate (a photo) or footprint (a map sheet)."""
    if media is None:
        return None
    data = _geometry(media)
    geometry = data.get("location")
    if not geometry:
        return None
    meaning: Meaning = (
        "catalog-footprint"
        if data.get("location_meaning", data.get("locationMeaning")) == "catalog-footprint"
        else "catalog-coordinate"
    )
    ref = EventLocationRef(evidence_id=evidence_id) if evidence_id else None
    return geometry_location(geometry, meaning, ref=ref)
