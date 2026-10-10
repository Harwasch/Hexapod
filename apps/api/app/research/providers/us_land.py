from __future__ import annotations

import math
from typing import Any

import httpx
from pyproj import Geod
from shapely import get_num_coordinates, orient_polygons
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

from app.research.providers.base import (
    SourceContext,
    SourceError,
    SourceResult,
    SourceSpec,
    evidence,
    fetch_json,
)

US_SOURCES = {
    "usda-soils": SourceSpec(
        id="usda-soils",
        name="USDA mapped soil components",
        domain="physical",
        coverage="United States and surveyed territories; sampled map unit only",
        resolution="Survey map-unit component estimates; not a soil boring or parcel-wide survey",
        license="U.S. public domain",
        attribution="USDA Natural Resources Conservation Service, Soil Data Access",
        documentation_url="https://sdmdataaccess.sc.egov.usda.gov/WebServiceHelp.aspx",
        endpoint="https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
    ),
    "fema-flood-zones": SourceSpec(
        id="fema-flood-zones",
        name="FEMA mapped flood zones",
        domain="hazards",
        coverage="United States, where NFHL digital flood mapping is available",
        resolution="Regulatory flood-zone polygons; effective mapping and local changes must be checked",
        license="U.S. public domain",
        attribution="Federal Emergency Management Agency, National Flood Hazard Layer",
        documentation_url="https://www.fema.gov/flood-maps/national-flood-hazard-layer",
        endpoint="https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query",
    ),
}


def in_us(context: SourceContext) -> bool:
    lon, lat = context.point
    return (-180 <= lon <= -60 and 17 <= lat <= 73) or (140 <= lon <= 146 and 12 <= lat <= 22)


def soils(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = US_SOURCES["usda-soils"]
    if not in_us(context):
        return SourceResult(spec.id, "uncovered", "USDA soil mapping does not cover this location.")
    lon, lat = context.point
    # The only interpolated SQL values are finite geometry-derived numbers, never free text.
    query = (
        "SELECT TOP 50 mu.mukey, mu.muname, c.compname, c.comppct_r, "  # noqa: S608
        "c.drainagecl, c.hydgrp, c.slope_r "
        f"FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT ({lon:.10f} {lat:.10f})') AS a "
        "INNER JOIN mapunit AS mu ON mu.mukey=a.mukey "
        "LEFT JOIN component AS c ON c.mukey=mu.mukey "
        "WHERE c.majcompflag='Yes' ORDER BY c.comppct_r DESC"
    )
    raw, url = fetch_json(
        client, spec, {"query": query, "format": "JSON+COLUMNNAME"}, method="POST", encoding="json"
    )
    table = raw.get("Table", [])
    if not table:
        return SourceResult(
            spec.id,
            "empty",
            "No soil map-unit components were returned at the sample point. This is not a soil condition assessment.",
        )
    if not isinstance(table, list) or not isinstance(table[0], list):
        raise SourceError("Soil Data Access returned an unexpected table.")
    columns = [str(key).lower() for key in table[0]]
    records: list[dict[str, Any]] = []
    for values in table[1:51]:
        row = dict(zip(columns, values, strict=True))
        for key in ("comppct_r", "slope_r"):
            value = row.get(key)
            number = float(value) if value is not None and value != "" else None
            row[key] = number if number is not None and math.isfinite(number) else None
        records.append(row)
    if not records:
        return SourceResult(
            spec.id, "empty", "No usable mapped soil components were returned at this point."
        )
    data = {
        "records": records,
        "samplePoint": [lon, lat],
        "query": query,
        "truncated": len(records) >= 50,
    }
    item = evidence(
        spec,
        url,
        "Soil survey components at an interior sample point",
        data,
        "A representative interior point selects a survey map unit. Component percentages "
        "describe that map unit, not this land's species cover, parcel-wide soil fractions, or "
        "tested ground conditions.",
        regional=True,
    )
    return SourceResult(
        spec.id,
        "available",
        "Mapped soils at one interior point include "
        + ", ".join(str(row.get("compname") or "unnamed component") for row in records[:3])
        + ". Component proportions are map-unit estimates.",
        [("sample", item)],
        data,
    )


def flood_zones(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = US_SOURCES["fema-flood-zones"]
    if not in_us(context):
        return SourceResult(spec.id, "uncovered", "FEMA NFHL does not cover this location.")
    land = context.geometry
    west, south, east, north = land.bounds
    if east - west > 1 or north - south > 1:
        return SourceResult(
            spec.id, "uncovered", "Choose a smaller land area for bounded flood-zone analysis."
        )
    raw, url = fetch_json(
        client,
        spec,
        {
            "f": "geojson",
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FLD_AR_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STUDY_TYP,DFIRM_ID",
            "returnGeometry": "true",
            "resultRecordCount": 100,
        },
    )
    if "error" in raw or raw.get("type") != "FeatureCollection":
        raise SourceError("FEMA returned no valid feature collection.")
    records = []
    geod = Geod(ellps="WGS84")
    total_vertices = 0
    skipped = 0
    for feature in raw.get("features", [])[:100]:
        geometry = shape(feature["geometry"])
        if not geometry.is_valid:
            skipped += 1
            continue
        clipped = geometry.intersection(land)
        if clipped.is_empty or not isinstance(clipped, Polygon | MultiPolygon) or clipped.area == 0:
            continue
        total_vertices += int(get_num_coordinates(clipped))
        if total_vertices > 20_000:
            raise SourceError(
                "Flood-zone output exceeds the detailed geometry limit; narrow the selected land."
            )
        properties = feature.get("properties", {})
        area, _ = geod.geometry_area_perimeter(orient_polygons(clipped))
        records.append(
            {
                "zone": str(properties.get("FLD_ZONE") or "Unknown"),
                "subtype": str(properties.get("ZONE_SUBTY") or ""),
                "specialFloodHazard": str(properties.get("SFHA_TF") or "Unknown"),
                "studyType": str(properties.get("STUDY_TYP") or ""),
                "recordId": str(properties.get("FLD_AR_ID") or ""),
                "mapId": str(properties.get("DFIRM_ID") or ""),
                "intersectedAreaM2": abs(area),
                "geometry": mapping(clipped),
            }
        )
    truncated = bool(raw.get("exceededTransferLimit")) or len(raw.get("features", [])) >= 100
    if not records:
        return SourceResult(
            spec.id,
            "empty",
            "No flood-zone polygons intersected this land in the returned sample. This does not "
            "establish low flood risk or complete map coverage.",
            data={"truncated": truncated, "invalidFeaturesSkipped": skipped},
        )
    data = {"records": records, "truncated": truncated, "invalidFeaturesSkipped": skipped}
    item = evidence(
        spec,
        url,
        "Flood-zone intersections with the land boundary",
        data,
        "Returned flood polygons were clipped to the land. Coverage can be incomplete, and "
        "polygons can overlap. No returned polygon does not mean no flood risk. Verify the "
        "effective FIRM and any map amendments for decisions.",
    )
    item.spatial_relevance = "intersects"
    zones = sorted({str(row["zone"]) for row in records})
    return SourceResult(
        spec.id,
        "available",
        f"Returned FEMA mapping intersects this land in zone(s) {', '.join(zones)}. "
        "Verify effective mapping and local amendments.",
        [("zones", item)],
        data,
    )
