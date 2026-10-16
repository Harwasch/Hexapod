"""Point facts for the inspector's "At this point": what public data says about one spot.

Five lookups, each against a source the land research already registers (SourceSpec) and
through the same bounded fetches, so this adds no hosts:

* elevation -- USGS EPQS (3DEP), an interpolated model value, not a surveyed height;
* soil -- USDA Soil Data Access, the survey map unit containing the point;
* land-cover -- ESA WorldCover 2021, one 10 m cell read from its COG;
* flood-zone -- FEMA NFHL, a point query without geometry;
* ecoregion -- EPA Level IV ecoregions (2011), a regional framework.

Every answer says what kind of thing it is (`basis`), how old (`vintage`), where it came from
(the exact query URL) and what it does not mean (`caveat`). A failure is "unavailable (why)"
and never reads as "nothing here"; "empty" and "uncovered" say what they do not establish.
Nothing is persisted: answers are cached in process, keyed by the point rounded to about a
metre, for 6 h (2 min for failures). They are public data, so the cache is not per workspace.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from app.analysis import worldcover
from app.research.providers.base import SourceError, SourceSpec
from app.research.providers.ecology import ECOLOGY_SOURCES, REGION_LIMIT, region_at
from app.research.providers.open_data import SOURCES, epqs_at
from app.research.providers.us_land import (
    US_SOURCES,
    covers_us,
    flood_zone_at,
    soil_components_at,
)
from app.schemas.place import PlaceFact, PlaceFactKind, PlaceFactSource

#: Someone is waiting on each row, so point lookups give up sooner than research runs do.
TIMEOUT_S = 10
CACHE_SIZE = 1024
CACHE_TTL_S = 6 * 3600
#: A failure is retried soon: most are a slow or briefly unavailable agency service.
FAILURE_TTL_S = 120

Status = Literal["available", "empty", "uncovered", "unavailable"]
Basis = Literal[
    "modelled-surface",
    "mapped-survey",
    "classified-imagery",
    "regulatory-map",
    "regional-framework",
]

_cache: OrderedDict[tuple[str, float, float], tuple[float, PlaceFact]] = OrderedDict()
_lock = threading.Lock()

SPECS: dict[str, SourceSpec] = {
    "elevation": SOURCES["usgs-elevation"],
    "soil": US_SOURCES["usda-soils"],
    "land-cover": worldcover.SPEC,
    "flood-zone": US_SOURCES["fema-flood-zones"],
    "ecoregion": ECOLOGY_SOURCES["epa-ecoregions"],
}
BASIS: dict[str, Basis] = {
    "elevation": "modelled-surface",
    "soil": "mapped-survey",
    "land-cover": "classified-imagery",
    "flood-zone": "regulatory-map",
    "ecoregion": "regional-framework",
}
ELEVATION_LIMIT = (
    "Interpolated from the USGS 3D Elevation Program model at this point; not a surveyed height."
)
SOIL_LIMIT = (
    "Survey map-unit estimate for the unit containing this point; not a soil test or a "
    "parcel-wide soil map."
)
COVER_LIMIT = (
    "Broad class for the 10 m cell containing this point, classified from 2021 Sentinel "
    "imagery; not today's condition or species."
)
FLOOD_LIMIT = (
    "Effective regulatory mapping (FEMA NFHL); check the effective FIRM and any amendments "
    "before decisions."
)
FLOOD_EMPTY = (
    "No NFHL flood zone at this point. This does not establish low flood risk or complete map "
    "coverage."
)
ECOREGION_LIMIT = (
    REGION_LIMIT.split(". ")[0] + ". It names the region around this point, not the habitat at it."
)


def _source(kind: str, url: str) -> PlaceFactSource:
    spec = SPECS[kind]
    return PlaceFactSource.model_validate(
        {
            "provider": spec.id,
            "name": spec.name,
            "url": url,
            "documentation_url": spec.documentation_url,
            "license": spec.license,
            "attribution": spec.attribution,
            "resolution": spec.resolution,
        }
    )


@dataclass
class _Answer:
    """What one lookup found, before it becomes a PlaceFact. `url` is None only when no
    request was made (outside the source's coverage) or the request failed."""

    status: Status
    label: str
    caveat: str
    url: str | None = None
    detail: str | None = None
    vintage: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _elevation(client: httpx.Client, lon: float, lat: float) -> _Answer:
    if not covers_us(lon, lat):
        return _Answer(
            "uncovered", "USGS elevation covers the United States only.", ELEVATION_LIMIT
        )
    raw, url, value = epqs_at(client, lon, lat, timeout=TIMEOUT_S)
    if value is None:
        return _Answer(
            "uncovered", "USGS has no elevation value at this point.", ELEVATION_LIMIT, url=url
        )
    attributes = raw.get("attributes")
    acquired = str(
        (attributes.get("AcquisitionDate") if isinstance(attributes, dict) else None) or ""
    )
    year = re.search(r"(?:19|20)\d\d", acquired)
    resolution = raw.get("resolution")
    resolution_m = float(resolution) if isinstance(resolution, int | float) else None
    return _Answer(
        "available",
        f"{value:,.1f} m",
        ELEVATION_LIMIT,
        url=url,
        detail=f"From a {resolution_m:g} m elevation model" if resolution_m else None,
        vintage=f"DEM acquired {year.group(0)}" if year else None,
        data={
            "elevationM": value,
            "resolutionM": resolution_m,
            "acquisitionDate": acquired or None,
            "rasterId": raw.get("rasterId"),
        },
    )


def _soil(client: httpx.Client, lon: float, lat: float) -> _Answer:
    if not covers_us(lon, lat):
        return _Answer("uncovered", "USDA soil survey covers the United States only.", SOIL_LIMIT)
    records, url, _query = soil_components_at(client, lon, lat, timeout=TIMEOUT_S)
    if not records:
        return _Answer(
            "empty",
            "No mapped soil components at this point.",
            "The soil survey returned no map unit here. This does not describe the ground; "
            "the survey may not cover this spot.",
            url=url,
        )
    first = records[0]
    parts = [
        f"{row.get('compname') or 'Unnamed component'}"
        + (f" {row['comppct_r']:.0f}%" if isinstance(row.get("comppct_r"), float) else "")
        for row in records[:3]
    ]
    if first.get("drainagecl"):
        parts.append(str(first["drainagecl"]))
    if first.get("hydgrp"):
        parts.append(f"hydrologic group {first['hydgrp']}")
    return _Answer(
        "available",
        str(first.get("muname") or first.get("compname") or "Unnamed map unit"),
        SOIL_LIMIT,
        url=url,
        detail=" · ".join(parts),
        data={
            "mukey": first.get("mukey"),
            "mapUnit": first.get("muname"),
            "components": [
                {
                    "name": row.get("compname"),
                    "percent": row.get("comppct_r"),
                    "drainage": row.get("drainagecl"),
                    "hydrologicGroup": row.get("hydgrp"),
                    "slopePct": row.get("slope_r"),
                }
                for row in records[:10]
            ],
        },
    )


def _land_cover(client: httpx.Client, lon: float, lat: float) -> _Answer:
    try:
        code, url = worldcover.class_at(client, lon, lat)
    except worldcover.NoTileError as missing:
        return _Answer("uncovered", "WorldCover has no tile here.", COVER_LIMIT, url=str(missing))
    labels = {value: label for value, label, _ in worldcover.CLASSES}
    if code is None or code not in labels:
        return _Answer(
            "empty",
            "No land-cover class for this cell.",
            "WorldCover has no recognized class for this 10 m cell. This does not describe "
            "what is on the ground.",
            url=url,
            vintage="2021",
            data={"classCode": code, "tile": worldcover.tile_for(lon, lat)},
        )
    return _Answer(
        "available",
        f"{labels[code]} (class {code})",
        COVER_LIMIT,
        url=url,
        vintage="2021",
        data={"classCode": code, "classLabel": labels[code], "tile": worldcover.tile_for(lon, lat)},
    )


def _flood_zone(client: httpx.Client, lon: float, lat: float) -> _Answer:
    if not covers_us(lon, lat):
        return _Answer(
            "uncovered", "FEMA flood mapping covers the United States only.", FLOOD_LIMIT
        )
    records, url = flood_zone_at(client, lon, lat, timeout=TIMEOUT_S)
    if not records:
        return _Answer("empty", "No mapped zone", FLOOD_EMPTY, url=url)
    first = records[0]
    zone = str(first.get("FLD_ZONE") or "unknown")
    sfha = str(first.get("SFHA_TF") or "").upper() == "T"
    subtype = str(first.get("ZONE_SUBTY") or "").strip()
    description = (
        "Special Flood Hazard Area"
        if sfha
        else subtype.capitalize() or "Outside the Special Flood Hazard Area"
    )
    dfirm = str(first.get("DFIRM_ID") or "").strip()
    others = sorted(
        {str(row.get("FLD_ZONE")) for row in records[1:] if row.get("FLD_ZONE")} - {zone}
    )
    return _Answer(
        "available",
        f"Zone {zone}",
        FLOOD_LIMIT,
        url=url,
        detail=description
        + (f" · map {dfirm}" if dfirm else "")
        + (f" · also mapped here: zone {', '.join(others)}" if others else ""),
        data={
            "zone": zone,
            "sfha": sfha,
            "subtype": subtype or None,
            "dfirmId": dfirm or None,
            "recordId": first.get("FLD_AR_ID"),
            "zones": [zone, *others],
        },
    )


def _ecoregion(client: httpx.Client, lon: float, lat: float) -> _Answer:
    if not covers_us(lon, lat):
        return _Answer("uncovered", "EPA ecoregions cover the United States only.", ECOREGION_LIMIT)
    values, url = region_at(client, lon, lat, timeout=TIMEOUT_S)
    if not values or not values.get("US_L4NAME"):
        return _Answer(
            "empty",
            "No mapped ecoregion at this point.",
            "EPA's regional map returned no Level IV region here, which happens offshore and "
            "at the edges of its coverage.",
            url=url,
            vintage="2011 edition",
        )
    return _Answer(
        "available",
        str(values["US_L4NAME"]),
        ECOREGION_LIMIT,
        url=url,
        detail=f"Level III: {values['US_L3NAME']}" if values.get("US_L3NAME") else None,
        vintage="2011 edition",
        data={
            "level4Code": values.get("US_L4CODE"),
            "level4Name": values.get("US_L4NAME"),
            "level3Name": values.get("US_L3NAME"),
            "level2Name": values.get("NA_L2NAME"),
        },
    )


LOOKUPS: dict[str, Callable[[httpx.Client, float, float], _Answer]] = {
    "elevation": _elevation,
    "soil": _soil,
    "land-cover": _land_cover,
    "flood-zone": _flood_zone,
    "ecoregion": _ecoregion,
}


def lookup(
    fact: PlaceFactKind,
    lon: float,
    lat: float,
    client: httpx.Client | None = None,
    *,
    retry: bool = False,
) -> PlaceFact:
    """One fact at a point. `retry` (the inspector's Retry button) skips a cached failure;
    a cached answer is still returned, so retrying never spends a lookup on a known fact."""
    key = (fact, round(lon, 5), round(lat, 5))
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > time.monotonic() and not (retry and hit[1].status == "unavailable"):
            _cache.move_to_end(key)
            return hit[1].model_copy(update={"cached": True})
    owned = client is None
    http = client or httpx.Client(headers={"User-Agent": "LivingWorld-LandResearch/1.0"})
    try:
        answer = LOOKUPS[fact](http, lon, lat)
    except (httpx.HTTPError, SourceError, ValueError, KeyError, TypeError, OSError) as error:
        # OSError covers rasterio's RasterioIOError and the raster reader's own refusals.
        # Like retrieve(): no response bodies or server internals in what the user sees.
        status_code = getattr(error, "status_code", None)  # RasterStatusError
        if isinstance(error, httpx.HTTPStatusError):
            reason = f"HTTP {error.response.status_code}"
        elif isinstance(status_code, int):
            reason = f"HTTP {status_code}"
        else:
            reason = type(error).__name__
        answer = _Answer(
            "unavailable",
            f"{SPECS[fact].name} could not be retrieved ({reason})",
            "The lookup failed, so this says nothing about the place. Try again shortly.",
        )
    finally:
        if owned:
            http.close()
    result = PlaceFact(
        kind=fact,
        status=answer.status,
        label=answer.label,
        detail=answer.detail,
        basis=BASIS[fact],
        vintage=answer.vintage,
        caveat=answer.caveat,
        longitude=lon,
        latitude=lat,
        retrieved_at=datetime.now(UTC),
        cached=False,
        source=_source(fact, answer.url) if answer.url else None,
        data=answer.data,
    )
    ttl = FAILURE_TTL_S if answer.status == "unavailable" else CACHE_TTL_S
    with _lock:
        _cache[key] = (time.monotonic() + ttl, result)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def clear_cache() -> None:
    """Tests only: every lookup after this one goes to its source."""
    with _lock:
        _cache.clear()
