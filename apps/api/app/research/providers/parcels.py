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
    #: Text the county's terms require with any copy of its data (Kern's), else None.
    disclaimer: str | None = None

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
            disclaimer=entry.get("disclaimer") or None,
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


# Which counties the public parcel lookup (`POST /public/parcels/near`, the landing page's site
# finder) asks: verified counties only, Harrison's decision of 2026-10-20. Every available
# county's layer, service and ArcGIS item metadata was read on 2026-10-20, and each available
# county is in exactly one of the three sets below (a test holds that, so a county added to the
# registry has to be placed). docs/PUBLIC_PARCELS.md lists the excluded counties and why. A
# county pending terms is enabled by moving it to PUBLIC_VERIFIED once its terms are confirmed.
# Sierra County, whose data is sold, is not `available` at all.

#: Asked. Terms read, and they allow showing the outlines with attribution: no terms stated, or
#: only disclaimers (no warranty, not a survey). Kern's terms also require its disclaimer
#: statement with any copy, which the lookup returns as a notice (`ParcelSource.disclaimer`).
PUBLIC_VERIFIED: frozenset[str] = frozenset(
    {
        "06001", "06003", "06005", "06007", "06009", "06011", "06013", "06015", "06019",
        "06021", "06025", "06027", "06029", "06031", "06033", "06035", "06043", "06045",
        "06047", "06057", "06061", "06065", "06069", "06071", "06075", "06077", "06081",
        "06083", "06085", "06089", "06093", "06095", "06099", "06101", "06103", "06105",
        "06107", "06109", "06113", "06115",
    }
)  # fmt: skip

#: Counties whose terms could not be read: no reachable ArcGIS item, and the layer and service
#: state none. Humboldt, Los Angeles, Madera, Marin, Mono, Monterey, Napa, Orange, Sacramento,
#: San Diego, San Luis Obispo, Santa Cruz, Ventura.
_TERMS_UNREAD = (
    "06023", "06037", "06039", "06041", "06051", "06053", "06055", "06059", "06067",
    "06073", "06079", "06087", "06111",
)  # fmt: skip

#: Not asked until their terms are confirmed, with why.
PENDING_TERMS: dict[str, str] = {
    **dict.fromkeys(
        _TERMS_UNREAD,
        "Terms could not be read: no reachable ArcGIS item, and the layer and service state none.",
    ),
    "06097": "Sonoma County's licence allows sharing with attribution but not redistributing "
    "modified data; whether showing chosen parcels together counts needs confirming.",
}

#: Never asked by the public lookup, with the words that say why.
NO_PUBLIC_REUSE: dict[str, str] = {
    "06017": "El Dorado County's item licence: 'These results are not to be redistributed "
    "unless agreed upon with the originator.'",
}


def public_counties() -> frozenset[str]:
    """The counties (FIPS) the public parcel lookup may ask."""
    return PUBLIC_VERIFIED


def available_sources() -> tuple[ParcelSource, ...]:
    """Every county with a usable parcel service."""
    return tuple(source for source in _registry().sources if source.status == "available")


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
