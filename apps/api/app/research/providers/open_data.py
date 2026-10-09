from __future__ import annotations

import math
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import httpx
from shapely.geometry import Point

from app.research.providers.archives import ARCHIVE_SOURCES, commons, historical_maps
from app.research.providers.base import (
    SourceContext,
    SourceError,
    SourceResult,
    SourceSpec,
    evidence,
    fetch_json,
)
from app.research.providers.us_land import US_SOURCES, flood_zones, soils
from app.schemas.research import EvidenceContent

SOURCES = {
    "usgs-elevation": SourceSpec(
        id="usgs-elevation",
        name="USGS elevation",
        domain="physical",
        coverage="United States and covered territories",
        resolution="Interpolated elevation at one point; source DEM resolution varies",
        license="U.S. public domain",
        attribution="U.S. Geological Survey, 3D Elevation Program",
        documentation_url="https://epqs.nationalmap.gov/v1/docs",
        endpoint="https://epqs.nationalmap.gov/v1/json",
    ),
    "nasa-power": SourceSpec(
        id="nasa-power",
        name="NASA POWER climate and solar resource",
        domain="energy",
        coverage="Global",
        resolution="Regional model/satellite grids; not a roof or on-site weather measurement",
        license="NASA open data; acknowledge POWER and its contributing data sources",
        attribution="NASA POWER Project, NASA Langley Research Center",
        documentation_url="https://power.larc.nasa.gov/docs/services/api/temporal/climatology/",
        endpoint="https://power.larc.nasa.gov/api/temporal/climatology/point",
    ),
    "gbif-occurrences": SourceSpec(
        id="gbif-occurrences",
        name="GBIF biodiversity observations",
        domain="ecology",
        coverage="Global, uneven observation coverage",
        resolution="Individual records with varying coordinate uncertainty",
        license="Record-specific CC0 or CC BY; other licenses excluded",
        attribution="GBIF and the original occurrence data publishers",
        documentation_url="https://techdocs.gbif.org/en/openapi/v1/occurrence",
        endpoint="https://api.gbif.org/v1/occurrence/search",
    ),
}

SOURCES.update(US_SOURCES)
OVERVIEW_SOURCES = tuple(SOURCES)
SOURCES.update(ARCHIVE_SOURCES)


def elevation(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = SOURCES["usgs-elevation"]
    lon, lat = context.point
    raw, url = fetch_json(
        client, spec, {"x": lon, "y": lat, "wkid": 4326, "units": "Meters", "includeDate": True}
    )
    value = float(raw.get("value", -1_000_000))
    if not math.isfinite(value) or value <= -999_999:
        return SourceResult(
            spec.id, "uncovered", "USGS returned no elevation at the selected sample point."
        )
    data = {
        "elevationM": value,
        "samplePoint": [lon, lat],
        "source": raw.get("rasterId"),
        "acquisitionDate": raw.get("attributes", {}).get("AcquisitionDate"),
        "resolution": raw.get("resolution"),
    }
    item = evidence(
        spec,
        url,
        "USGS elevation sample",
        raw,
        "A representative point within the boundary was sampled. "
        "This is not the land's minimum, maximum, or mean elevation, and is not a surveyed height.",
    )
    return SourceResult(
        spec.id,
        "available",
        f"USGS estimates {value:,.1f} m elevation at a sample point inside the land.",
        [("sample", item)],
        data,
    )


def climate(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = SOURCES["nasa-power"]
    lon, lat = context.point
    raw, url = fetch_json(
        client,
        spec,
        {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON",
        },
    )
    parameters = raw.get("properties", {}).get("parameter", {})
    if not parameters:
        raise SourceError("NASA POWER returned no climate parameters.")
    fill = raw.get("header", {}).get("fill_value", -999)
    cleaned: dict[str, dict[str, float | None]] = {}
    for key, values in parameters.items():
        cleaned[key] = {
            month: float(value)
            if isinstance(value, int | float) and math.isfinite(value) and value != fill
            else None
            for month, value in values.items()
        }
    data = {
        "parameters": cleaned,
        "definitions": raw.get("parameters", {}),
        "period": {
            "start": raw.get("header", {}).get("start"),
            "end": raw.get("header", {}).get("end"),
            "description": raw.get("header", {}).get("range"),
        },
        "samplePoint": [lon, lat],
        "sources": raw.get("header", {}).get("sources", []),
    }
    item = evidence(
        spec,
        url,
        "Regional solar and climate climatology",
        raw,
        "Regional grid data sampled at a point within the boundary. "
        "Local shade, terrain, microclimate and individual roofs are not resolved.",
        regional=True,
    )
    annual = cleaned.get("ALLSKY_SFC_SW_DWN", {}).get("ANN")
    units = raw.get("parameters", {}).get("ALLSKY_SFC_SW_DWN", {}).get("units", "source units")
    summary = (
        f"Regional mean daily solar irradiation: {annual:,.2f} {units}."
        if annual is not None
        else "Monthly regional climate and solar-resource values are available."
    )
    return SourceResult(spec.id, "available", summary, [("climatology", item)], data)


def occurrences(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = SOURCES["gbif-occurrences"]
    west, south, east, north = context.geometry.bounds
    if east - west > 10 or north - south > 10:
        return SourceResult(
            spec.id,
            "uncovered",
            "Select a smaller area to inspect individual biodiversity observations.",
        )
    raw, url = fetch_json(
        client,
        spec,
        {
            "decimalLatitude": f"{south},{north}",
            "decimalLongitude": f"{west},{east}",
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "limit": 100,
        },
    )
    records: list[dict[str, Any]] = []
    items: list[tuple[str, EvidenceContent]] = []
    excluded_license = 0
    excluded_generalized = 0
    for record in raw.get("results", []):
        license_url = str(record.get("license", ""))
        if license_url not in {
            "http://creativecommons.org/publicdomain/zero/1.0/legalcode",
            "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
            "http://creativecommons.org/licenses/by/4.0/legalcode",
            "https://creativecommons.org/licenses/by/4.0/legalcode",
        }:
            excluded_license += 1
            continue
        if record.get("informationWithheld") or record.get("dataGeneralizations"):
            excluded_generalized += 1
            continue
        lon, lat = record.get("decimalLongitude"), record.get("decimalLatitude")
        if (
            not isinstance(lon, int | float)
            or not isinstance(lat, int | float)
            or not context.geometry.covers(Point(lon, lat))
        ):
            continue
        # Do not publish point coordinates. Uncertainty can extend beyond the property.
        data = {
            "id": str(record["key"]),
            "species": record.get("species") or record.get("scientificName", "Unidentified taxon"),
            "date": record.get("eventDate"),
            "basis": record.get("basisOfRecord"),
            "coordinateUncertaintyM": record.get("coordinateUncertaintyInMeters"),
            "publisher": record.get("institutionCode")
            or record.get("datasetName", "GBIF data publisher"),
            "datasetKey": record.get("datasetKey"),
            "license": license_url,
        }
        item = evidence(
            spec,
            f"https://www.gbif.org/occurrence/{record['key']}",
            str(data["species"]),
            data,
            "The published coordinate falls within the selected polygon. "
            "Unknown or nonzero coordinate uncertainty can extend outside it; "
            "this is not proof of current occupancy.",
        )
        item.license = license_url
        item.attribution = (
            f"{data['publisher']}; occurrence {record['key']}, "
            f"dataset {data['datasetKey']}; accessed via GBIF"
        )
        item.record_id = str(record["key"])
        if isinstance(data["date"], str):
            with suppress(ValueError):
                item.observed_at = datetime.fromisoformat(data["date"].replace("Z", "+00:00"))
                if item.observed_at.tzinfo is None:
                    item.observed_at = item.observed_at.replace(tzinfo=UTC)
        records.append(data)
        items.append((str(record["key"]), item))
    data = {
        "records": records,
        "boundingBoxCount": raw.get("count", 0),
        "sampleSize": len(raw.get("results", [])),
        "truncated": not raw.get("endOfRecords", True),
        "excludedLicense": excluded_license,
        "excludedGeneralized": excluded_generalized,
        "queryUrl": url,
    }
    summary = (
        f"{len(records)} openly licensed observation records in the retrieved sample "
        "have coordinates inside this land. Records do not establish current presence or species coverage."
    )
    return SourceResult(
        spec.id,
        "available" if records else "empty",
        summary
        if records
        else "No usable observations inside the polygon in this sample. This is not evidence that species are absent.",
        items,
        data,
    )


ADAPTERS = {
    "usgs-elevation": elevation,
    "nasa-power": climate,
    "gbif-occurrences": occurrences,
    "usda-soils": soils,
    "fema-flood-zones": flood_zones,
    "commons-place-images": commons,
    "usgs-historical-maps": historical_maps,
}


def retrieve(provider: str, context: SourceContext, client: httpx.Client) -> SourceResult:
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        raise SourceError("Unknown source provider.")
    try:
        return adapter(context, client)
    except (httpx.HTTPError, SourceError, ValueError, KeyError, TypeError) as error:
        # No credentials, response bodies or server internals in user-visible errors.
        reason = (
            f"HTTP {error.response.status_code}"
            if isinstance(error, httpx.HTTPStatusError)
            else type(error).__name__
        )
        return SourceResult(
            provider, "unavailable", f"{SOURCES[provider].name} could not be retrieved ({reason})."
        )
