"""Streams, creeks and ditches the USGS National Hydrography Dataset maps in and near a Land.

Read live from the NHD map service (Flowline - Large Scale, 1:24,000), clipped to the boundary
grown by BUFFER_M so a creek just outside the line still counts, and never saved. The browser
cannot ask NHD itself (no CORS), so the API asks on its behalf. An answer is kept in memory
for an hour per land revision, within CACHE_BYTES. A failed lookup is not kept, so the next
request tries again; an answer too large to read is kept for ten minutes, because asking again
cannot help.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import ValidationError
from pyproj import CRS, Transformer
from shapely.geometry import LineString as ShapelyLine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from sqlalchemy.orm import Session

from app.research.providers.base import SourceError, SourceSpec, fetch_json
from app.research.providers.us_land import covers_us
from app.schemas.geojson import LineString
from app.schemas.land_hydrography import HydrographyFeature, LandHydrography
from app.schemas.place import PlaceFactSource
from app.services.land import get_land

LAYER = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6"
SPEC = SourceSpec(
    id="usgs-nhd",
    name="USGS National Hydrography Dataset (NHD) flowlines",
    domain="physical",
    coverage="United States",
    resolution="1:24,000 (Flowline - Large Scale); mapped channels, not surveyed banks",
    license="Public domain (U.S. Government work)",
    attribution="U.S. Geological Survey, National Hydrography Dataset (NHD)",
    documentation_url="https://www.usgs.gov/national-hydrography/national-hydrography-dataset",
    endpoint=LAYER + "/query",
)
BUFFER_M = 100.0
RECORD_LIMIT = 500
TIMEOUT_S = 20
CACHE_TTL_S = 3600
TOO_LARGE_TTL_S = 600
#: Answers kept in memory, by their estimated size (`_weight`). One at NHD's 2 MB answer limit
#: (about 80,000 vertices) holds about 11 MB of Python objects, so a count of answers is no
#: bound on the 1 GB API VM; a typical land's answer is tens of kilobytes.
CACHE_BYTES = 32 * 1024 * 1024
TOO_LARGE = (
    "This land has more mapped flowlines nearby than one USGS NHD answer can carry, so none "
    "are shown. Trying again will not help; a smaller outline would fit."
)
Kind = Literal["perennial", "intermittent", "ephemeral", "unclassified", "artificial-path", "other"]
#: NHD hydrographic categories of stream/river flowlines (FType 460). A stream/river without
#: one (FCode 46000, common in NHD) is "unclassified": a natural channel of unknown permanence.
KINDS: dict[int, Kind] = {
    46006: "perennial",
    46003: "intermittent",
    46007: "ephemeral",
}
FTYPES = {
    460: "Stream/river",
    336: "Canal/ditch",
    334: "Connector",
    420: "Underground conduit",
    428: "Pipeline",
    558: "Artificial path",
    566: "Coastline",
}

#: (land, revision) -> (expires at, answer, estimated bytes).
_cache: OrderedDict[tuple[uuid.UUID, int], tuple[float, LandHydrography, int]] = OrderedDict()
_lock = threading.Lock()


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": "LivingWorld-LandHydrography/1.0"})


def _source(url: str) -> PlaceFactSource:
    return PlaceFactSource.model_validate(
        {
            "provider": SPEC.id,
            "name": SPEC.name,
            "url": url,
            # The exact query answers JSON; the layer page is what a person can read.
            "link_url": LAYER,
            "documentation_url": SPEC.documentation_url,
            "license": SPEC.license,
            "attribution": SPEC.attribution,
            "resolution": SPEC.resolution,
        }
    )


def describe(fcode: int) -> tuple[Kind, str]:
    ftype = fcode // 100
    label = FTYPES.get(ftype, "Flowline")
    if ftype == 460:
        kind: Kind = KINDS.get(fcode, "unclassified")
        return kind, f"{label}, " + ("permanence not recorded" if kind == "unclassified" else kind)
    if ftype == 558:
        # NHD draws a stream's course through a pond, lake or wide river as an artificial
        # path; keeping it with the streams keeps a creek's line unbroken.
        return "artificial-path", f"{label} carrying a stream through a pond, lake or wide river"
    return "other", label


def _metric(geometry: BaseGeometry) -> tuple[Transformer, Transformer]:
    point = geometry.representative_point()
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={point.y} +lon_0={point.x} +datum=WGS84 +units=m +no_defs"
    )
    return (
        Transformer.from_crs(4326, local, always_xy=True),
        Transformer.from_crs(local, 4326, always_xy=True),
    )


def parse(raw: dict[str, Any], boundary: BaseGeometry) -> tuple[list[HydrographyFeature], bool]:
    """NHD GeoJSON flowlines clipped to the boundary plus BUFFER_M, longest first."""
    forward, back = _metric(boundary)
    area = transform_geometry(forward.transform, boundary).buffer(BUFFER_M)
    rows = raw.get("features")
    if not isinstance(rows, list):
        raise SourceError("The hydrography service returned an unexpected response.")
    features: list[HydrographyFeature] = []
    for row in rows:
        properties = row.get("properties") or {}
        identifier = properties.get("permanent_identifier")
        fcode = properties.get("fcode")
        geometry = row.get("geometry")
        if not isinstance(identifier, str) or not isinstance(fcode, int) or not geometry:
            continue
        try:
            line = transform_geometry(forward.transform, shape(geometry))
        except (ValueError, TypeError, AttributeError):
            continue
        clipped = line.intersection(area)
        pieces = [
            part
            for part in getattr(clipped, "geoms", [clipped])
            if isinstance(part, ShapelyLine) and part.length > 0
        ]
        name = properties.get("gnis_name")
        kind, description = describe(fcode)
        for index, piece in enumerate(pieces, 1):
            try:
                outline = LineString.model_validate(
                    {
                        "type": "LineString",
                        "coordinates": [
                            [round(x, 7), round(y, 7)]
                            for x, y in transform_geometry(back.transform, piece).coords
                        ],
                    }
                )
            except ValidationError:
                continue
            features.append(
                HydrographyFeature(
                    id=identifier if index == 1 else f"{identifier}/{index}",
                    permanent_identifier=identifier,
                    name=name if isinstance(name, str) and name.strip() else None,
                    fcode=fcode,
                    kind=kind,
                    description=description,
                    geometry=outline,
                    length_m=round(piece.length, 1),
                )
            )
    features.sort(key=lambda feature: (-feature.length_m, feature.id))
    exceeded = raw.get("exceededTransferLimit") or (raw.get("properties") or {}).get(
        "exceededTransferLimit"
    )
    return features, bool(exceeded) or len(rows) >= RECORD_LIMIT


def lookup(
    land_id: uuid.UUID, revision: int, boundary: BaseGeometry, client: httpx.Client
) -> LandHydrography:
    retrieved = datetime.now(UTC)
    common: dict[str, Any] = {
        "land_id": land_id,
        "boundary_revision": revision,
        "buffer_m": BUFFER_M,
        "retrieved_at": retrieved,
    }
    point = boundary.representative_point()
    if not covers_us(point.x, point.y):
        return LandHydrography(
            **common,
            status="uncovered",
            message="The USGS NHD maps streams in the United States only; this land is outside it.",
            features=[],
            truncated=False,
            source=_source(LAYER),
        )
    west, south, east, north = boundary.bounds
    # Grow the query box by the clip distance (latitude degrees are about 110.6 km).
    pad_lat = BUFFER_M / 110_574
    pad_lon = BUFFER_M / (111_320 * max(0.05, math.cos(math.radians(point.y))))
    params: dict[str, str | int | float | bool] = {
        "geometry": f"{west - pad_lon},{south - pad_lat},{east + pad_lon},{north + pad_lat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "permanent_identifier,gnis_name,fcode",
        "returnGeometry": "true",
        # 6 decimals (about 0.1 m) and 1e-5 degrees (about 1 m) of generalisation keep a dense
        # drainage network inside fetch_json's answer limit; at 1:24,000 NHD lines are good to
        # about 12 m, so neither moves a channel meaningfully.
        "geometryPrecision": 6,
        "maxAllowableOffset": 0.00001,
        "resultRecordCount": RECORD_LIMIT,
        "f": "geojson",
    }
    url = str(httpx.URL(SPEC.endpoint, params=params))
    try:
        raw, url = fetch_json(client, SPEC, params, timeout=TIMEOUT_S)
        if "error" in raw:
            raise SourceError("The hydrography service rejected the query.")
        features, truncated = parse(raw, boundary)
    except (httpx.HTTPError, SourceError, ValueError, json.JSONDecodeError) as error:
        return LandHydrography(
            **common,
            status="unavailable",
            message=TOO_LARGE
            if isinstance(error, SourceError) and error.too_large
            else "The USGS hydrography service did not answer. Try again in a minute.",
            features=[],
            truncated=False,
            source=_source(url),
        )
    named = sorted({feature.name for feature in features if feature.name})
    if features:
        flowlines = len({feature.permanent_identifier for feature in features})
        message = (
            f"{flowlines} mapped flowline{'s' if flowlines != 1 else ''} within "
            f"{BUFFER_M:.0f} m of this land"
            + (f", including {', '.join(named[:3])}" if named else "")
            + ". Mapped channels at 1:24,000, not surveyed banks."
        )
    else:
        message = f"The USGS NHD maps no streams or ditches within {BUFFER_M:.0f} m of this land."
    if truncated:
        message += f" The service stopped at {RECORD_LIMIT} flowlines; more may exist."
    return LandHydrography(
        **common,
        status="available" if features else "empty",
        message=message,
        features=features,
        truncated=truncated,
        source=_source(url),
    )


def hydrography(db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID) -> LandHydrography:
    land = get_land(db, workspace_id, land_id)
    key = (land.id, land.revision)
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] > now:
            _cache.move_to_end(key)
            return cached[1]
    db.commit()  # Release the auth transaction before waiting for a public provider.
    boundary = shape(land.boundary.model_dump())
    with _client() as client:
        result = lookup(land.id, land.revision, boundary, client)
    keep = (
        CACHE_TTL_S
        if result.status != "unavailable"
        else TOO_LARGE_TTL_S
        if result.message == TOO_LARGE
        else 0
    )
    if keep:
        weight = _weight(result)
        with _lock:
            # Expired answers go first: a revised boundary is a new key, so the old revision's
            # answer would otherwise stay until the budget pushed it out.
            fresh = time.monotonic()
            for stale in [k for k, (expires, _, _) in _cache.items() if expires <= fresh]:
                del _cache[stale]
            _cache[key] = (now + keep, result, weight)
            _cache.move_to_end(key)
            total = sum(entry[2] for entry in _cache.values())
            while total > CACHE_BYTES and len(_cache) > 1:
                total -= _cache.popitem(last=False)[1][2]
    return result


def _weight(result: LandHydrography) -> int:
    """About the bytes an answer holds in memory: measured at ~140 per vertex (a list of two
    floats) and ~2 KB per feature's other fields."""
    return 4096 + sum(2048 + 140 * len(feature.geometry.coordinates) for feature in result.features)
