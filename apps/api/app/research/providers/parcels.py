"""County parcel services for California, chosen by where a search falls.

`data/ca_parcels.json` lists every county's public parcel layer, each verified with a live
query at its county seat (re-check them with `python -m app.scripts.verify_ca_parcels`), and
`data/ca_counties.geojson` holds the county outlines (US Census cartographic boundaries, public
domain) used to route a search to the counties it touches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from app.research.providers.base import SourceSpec

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class ParcelSource:
    county: str
    fips: str
    status: Literal["available", "none", "restricted"]
    endpoint: str | None
    id_field: str | None
    attribution: str
    license: str
    documentation_url: str
    notes: str

    @property
    def spec(self) -> SourceSpec:
        if self.endpoint is None:
            raise ValueError(f"{self.county} County has no parcel service.")
        return SourceSpec(
            id=f"ca-parcels/{self.fips}",
            name=f"{self.county} County parcels",
            domain="parcels",
            coverage=f"{self.county} County, California",
            resolution="County assessor parcel mapping; not a boundary survey",
            license=self.license,
            attribution=self.attribution,
            documentation_url=self.documentation_url,
            endpoint=self.endpoint.rstrip("/") + "/query",
        )


@dataclass(frozen=True)
class _Registry:
    sources: tuple[ParcelSource, ...]
    outlines: tuple[BaseGeometry, ...]
    index: STRtree


@cache
def _registry() -> _Registry:
    listed = json.loads((DATA / "ca_parcels.json").read_text())["counties"]
    by_fips = {}
    for entry in listed:
        # Only GeoJSON output is parsed; a layer that answers in Esri JSON alone is not used.
        usable = entry["status"] == "available" and entry.get("format") == "geojson"
        by_fips[entry["fips"]] = ParcelSource(
            county=entry["county"],
            fips=entry["fips"],
            status=entry["status"] if entry["status"] != "available" or usable else "none",
            endpoint=entry["endpoint"] if usable else None,
            id_field=entry["idField"] if usable else None,
            attribution=entry.get("attribution") or f"County of {entry['county']}",
            license=entry.get("license") or "No terms stated",
            documentation_url=entry.get("documentationUrl") or entry.get("endpoint") or "",
            notes=entry.get("notes") or "",
        )
    outlines = json.loads((DATA / "ca_counties.geojson").read_text())["features"]
    sources: list[ParcelSource] = []
    geometries: list[BaseGeometry] = []
    for feature in outlines:
        source = by_fips.get(feature["properties"]["fips"])
        if source is None:
            continue
        sources.append(source)
        geometries.append(shape(feature["geometry"]))
    return _Registry(tuple(sources), tuple(geometries), STRtree(geometries))


#: Counties whose parcel service is used in the app but not by the public parcel lookup
#: (`POST /public/parcels/near`), because their published terms restrict redistribution, with
#: the words that say so. Each available county's layer, service and ArcGIS item metadata was
#: read on 2026-10-20; the rest state no terms or only disclaimers (no warranty, not a survey),
#: which do not restrict showing the outlines with their attribution. Sierra County, whose data
#: is sold, is not `available` at all.
NO_PUBLIC_REUSE: dict[str, str] = {
    "06017": "El Dorado County's item licence: 'These results are not to be redistributed "
    "unless agreed upon with the originator.'",
}


def counties_touching(area: BaseGeometry) -> list[ParcelSource]:
    """Counties whose outline meets the search area, the most-covered first."""
    registry = _registry()
    hits = [
        (registry.outlines[int(index)].intersection(area).area, registry.sources[int(index)])
        for index in registry.index.query(area)
        if registry.outlines[int(index)].intersects(area)
    ]
    hits.sort(key=lambda hit: -hit[0])
    return [source for _, source in hits]
