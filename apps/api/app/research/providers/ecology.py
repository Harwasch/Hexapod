from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import httpx
from shapely import orient_polygons
from shapely.geometry import MultiPolygon, Polygon, mapping

from app.research.providers.base import (
    SourceContext,
    SourceError,
    SourceResult,
    SourceSpec,
    evidence,
    fetch_json,
)
from app.research.providers.us_land import in_us

ECOLOGY_SOURCES = {
    "epa-ecoregions": SourceSpec(
        id="epa-ecoregions",
        name="EPA regional ecological context (2011)",
        domain="ecology",
        coverage="United States where EPA Level IV mapping is available",
        resolution="Regional framework compiled at 1:250,000; not a site habitat or restoration prescription",
        license="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="U.S. EPA ORD ecoregion data; EPA OEI map service; December 2011 edition",
        documentation_url="https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states",
        endpoint="https://geodata.epa.gov/arcgis/rest/services/ORD/USEPA_Ecoregions_Level_III_and_IV/MapServer/7/query",
    ),
    "usda-ecological-sites": SourceSpec(
        id="usda-ecological-sites",
        name="USDA soil-linked ecological site references",
        domain="ecology",
        coverage="United States and surveyed territories; ecological associations vary by survey",
        resolution="Ecological-class associations of major components in a sampled soil map unit",
        license="U.S. public domain",
        attribution="USDA Natural Resources Conservation Service, Soil Data Access",
        documentation_url="https://sdmdataaccess.sc.egov.usda.gov/WebServiceHelp.aspx",
        endpoint="https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
    ),
}
REGION_LIMIT = (
    "EPA's December 2011 regional framework is compiled at 1:250,000 and is intended for large "
    "geographic extents. These are regional context labels intersecting the boundary, not surveyed "
    "habitat boundaries, current vegetation, species occupancy, soil fractions or a restoration prescription. "
    "Consult current state maps, field evidence and local reference sites before defining targets."
)
SITE_LIMIT = (
    "A representative point selects one mapped soil unit; major component associations describe that map unit, "
    "not verified ecological sites or their proportions across the selected land. Descriptions may be absent "
    "or provisional. Verify soils, hydrology, disturbance, climate and a local reference community before "
    "using an association as a restoration target."
)


def bound_records(data: dict[str, Any]) -> None:
    # Keep complete parseable records in the source excerpt; never truncate JSON text.
    while len(json.dumps(data, ensure_ascii=False)) > 29_000:
        if not data["records"]:
            raise SourceError("The ecological source metadata exceeds the evidence limit.")
        data["records"].pop()
        data["truncated"] = True


def regions(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = ECOLOGY_SOURCES["epa-ecoregions"]
    if not in_us(context):
        return SourceResult(
            spec.id, "uncovered", "This EPA regional framework covers the United States."
        )
    geom = context.geometry
    west, south, east, north = geom.bounds
    if east - west > 5 or north - south > 5:
        return SourceResult(
            spec.id, "uncovered", "Use a smaller land area for bounded regional context lookup."
        )
    if not isinstance(geom, Polygon | MultiPolygon) or not geom.is_valid:
        raise SourceError("The land boundary must be a valid polygon.")
    # ArcGIS polygon outer rings are clockwise and holes counterclockwise.
    oriented = orient_polygons(geom, exterior_cw=True)
    polygons = list(oriented.geoms) if isinstance(oriented, MultiPolygon) else [oriented]
    rings = [ring for polygon in polygons for ring in mapping(polygon)["coordinates"]]
    query_geometry = json.dumps(
        {"rings": rings, "spatialReference": {"wkid": 4326}}, separators=(",", ":")
    )
    fields = "OBJECTID,US_L4CODE,US_L4NAME,US_L3CODE,US_L3NAME,NA_L2CODE,NA_L2NAME,NA_L1CODE,NA_L1NAME,STATE_NAME"
    raw, url = fetch_json(
        client,
        spec,
        {
            "f": "json",
            "geometry": query_geometry,
            "geometryType": "esriGeometryPolygon",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": fields,
            "returnGeometry": "false",
            "resultRecordCount": 100,
            "orderByFields": "OBJECTID",
        },
        method="POST",
    )
    if "error" in raw or not isinstance(raw.get("features"), list):
        raise SourceError("EPA returned no usable regional feature response.")
    records: list[dict[str, Any]] = []
    for feature in raw["features"][:100]:
        if not isinstance(feature, dict):
            raise SourceError("EPA returned an invalid region record.")
        values = feature.get("attributes", {})
        if not isinstance(values, dict):
            raise SourceError("EPA returned an invalid region record.")
        records.append(
            {
                key: (str(values[key])[:300] if values.get(key) is not None else None)
                for key in fields.split(",")
            }
        )
    truncated = bool(raw.get("exceededTransferLimit")) or len(raw["features"]) >= 100
    data = {
        "records": records,
        "truncated": truncated,
        "edition": "December 2011",
        "queryBounds": [west, south, east, north],
        "queryGeometrySha256": hashlib.sha256(query_geometry.encode()).hexdigest(),
        "relation": "Server-side polygon intersection, including exclusions; no area shares calculated.",
        "limitation": REGION_LIMIT,
    }
    bound_records(data)
    item = evidence(
        spec,
        url,
        "Regional ecoregion labels intersecting the land",
        data,
        REGION_LIMIT,
        regional=True,
    )
    names = sorted({str(row["US_L4NAME"]) for row in records if row.get("US_L4NAME")})
    summary = (
        (
            "The 2011 regional map intersects "
            + ", ".join(names[:5])
            + ". This is regional context, not a site habitat assessment."
        )
        if names
        else "No Level IV regional labels were returned. Map coverage may be incomplete; "
        "no ecological absence is inferred."
    )
    return SourceResult(
        spec.id, "available" if records else "empty", summary, [("regions", item)], data
    )


def ecological_sites(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = ECOLOGY_SOURCES["usda-ecological-sites"]
    if not in_us(context):
        return SourceResult(
            spec.id,
            "uncovered",
            "USDA soil-linked ecological site data does not cover this location.",
        )
    lon, lat = context.point
    query = (
        "SELECT TOP 50 mu.mukey, mu.muname, c.cokey, c.compname, c.comppct_r, "  # noqa: S608
        "ec.ecoclassid, ec.ecoclassname, ec.ecoclasstypename "
        f"FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT ({lon:.10f} {lat:.10f})') AS a "
        "INNER JOIN mapunit AS mu ON mu.mukey=a.mukey "
        "INNER JOIN component AS c ON c.mukey=mu.mukey "
        "INNER JOIN coecoclass AS ec ON ec.cokey=c.cokey "
        "WHERE c.majcompflag='Yes' AND ec.ecoclassid IS NOT NULL "
        "ORDER BY c.comppct_r DESC, c.cokey, ec.ecoclassid"
    )
    raw, url = fetch_json(
        client, spec, {"query": query, "format": "JSON+COLUMNNAME"}, method="POST", encoding="json"
    )
    table = raw.get("Table", [])
    if not isinstance(table, list) or (table and not isinstance(table[0], list)):
        raise SourceError("Soil Data Access returned an invalid ecological-class table.")
    records: list[dict[str, Any]] = []
    if table:
        columns = [str(k).lower() for k in table[0]]
        if set(columns) != {
            "mukey",
            "muname",
            "cokey",
            "compname",
            "comppct_r",
            "ecoclassid",
            "ecoclassname",
            "ecoclasstypename",
        }:
            raise SourceError("The ecological-class table has unexpected columns.")
        for values in table[1:51]:
            row: dict[str, Any] = {
                k: (None if v is None else str(v)[:300])
                for k, v in zip(columns, values, strict=True)
            }
            proportion = float(row["comppct_r"]) if row.get("comppct_r") not in (None, "") else None
            row["comppct_r"] = (
                proportion
                if proportion is not None and math.isfinite(proportion) and 0 <= proportion <= 100
                else None
            )
            code = str(row.get("ecoclassid") or "")
            row["descriptionUrl"] = (
                f"https://edit.sc.egov.usda.gov/catalogs/esd/{code[1:5]}/{code}"
                if re.fullmatch(r"[RF][0-9]{3}[A-Z]{2}[0-9]{3}[A-Z]{2}", code)
                else None
            )
            records.append(row)
    data = {
        "records": records,
        "samplePoint": [lon, lat],
        "query": query,
        "truncated": len(records) >= 50,
        "limitation": SITE_LIMIT,
    }
    bound_records(data)
    item = evidence(
        spec,
        url,
        "Soil-linked ecological class associations at a sample point",
        data,
        SITE_LIMIT,
        regional=True,
    )
    summary = (
        (
            f"{len(records)} ecological-class associations were returned for major soil components at one point. "
            "These are reference candidates, not verified ecological sites across the land."
        )
        if records
        else (
            "No ecological-class associations were returned at the soil sample point. "
            "A local reference ecosystem must be established from other evidence."
        )
    )
    return SourceResult(
        spec.id, "available" if records else "empty", summary, [("sites", item)], data
    )
