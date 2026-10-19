"""The demo ranch: a real place on California's Central Coast that the simulated fleet works.

**One file holds it**, `demo_ranch.json` beside this module, and both sides read it: this seed
(the `demo-ranch` catalog site) and the web app (`apps/web/src/missions/demoRanchData.ts`
imports the same JSON, for the built-in offline copy and the simulated project). The file has
two halves:

* **The source half, written by hand.** The demo outline, Figueroa Creek from the USGS NHD,
  the ESA WorldCover grassland and oak patches, the simulated fence, the two machine bases,
  the camera framing, and where each layer came from. Swapping the place means editing this
  half and nothing else.
* **`derived`, written by `build`, never by hand.** The three zone footprints, their geodesic
  areas and chip anchors, the creek and fence lengths, the site's area. Every acreage the app
  shows comes from here, so a zone's acres are its footprint's acres.

To change the place: edit the source half, then

    uv run python -m app.seed.demo_ranch --write   # rewrites `derived`, prints zone acres
    pnpm --filter @twin/web exec vitest run src/__tests__/demoRanch.test.ts
                                                   # prints a camera block if the old one
                                                   # no longer frames every zone and machine
    uv run python -m app.seed                      # _refresh brings an existing database along

`tests/test_demo_ranch.py` fails when `derived` is stale, so a hand edit cannot slip through.

The name is fictional (checked: no OSM or Nominatim match nearby), and the description says
what is simulated and that the outline is not a property line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

from pyproj import Geod, Transformer
from shapely.geometry import Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import polylabel, transform

from app.schemas.common import Attribution, GeoPosition, LicenseMetadata
from app.schemas.geojson import Polygon as PolygonSchema
from app.schemas.site import SiteCreate
from app.seed.data import overview_bookmark

PATH = Path(__file__).with_name("demo_ranch.json")
GENERATED_BY = "python -m app.seed.demo_ranch --write"

#: Zone outlines lose detail under this, in metres: WorldCover is a 10 m raster.
SIMPLIFY_M = 5.0
#: Holes smaller than this are filled. In the grassland they are single oaks and one-pixel
#: specks of the raster, which a mower steers around rather than plans around.
MIN_HOLE_M2 = 1_000.0
#: How far mowing stays back from the creek corridor's edge.
CREEK_SETBACK_M = 10.0
DIGITS = 7

_GEOD = Geod(ellps="WGS84")


def load() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return document


def _utm_epsg(lon: float, lat: float) -> int:
    zone = math.floor((lon + 180.0) / 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _largest(geometry: BaseGeometry) -> Polygon:
    parts = [g for g in getattr(geometry, "geoms", [geometry]) if isinstance(g, Polygon)]
    if not parts:
        raise ValueError("the derivation left no polygon; check the source geometry")
    return max(parts, key=lambda part: part.area)


def _round[G: BaseGeometry](geometry: G) -> G:
    def rounded(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        return (_round_coord(x), _round_coord(y))

    result: G = transform(rounded, geometry)
    return result


def _round_coord(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DIGITS)
    return [round(v, DIGITS) for v in value]


def _area_m2(polygon_lonlat: Polygon) -> float:
    """Geodesic area on WGS84: what PostGIS ST_Area(geography) reports for the same ring."""
    area, _ = _GEOD.geometry_area_perimeter(orient(polygon_lonlat, 1.0))
    return round(abs(area), 1)


def _length_m(line_lonlat: BaseGeometry) -> float:
    return round(_GEOD.geometry_length(line_lonlat), 1)


def _as_polygon(geometry: BaseGeometry) -> Polygon:
    if not isinstance(geometry, Polygon):
        raise TypeError(f"expected a Polygon, got {geometry.geom_type}")
    return geometry


def _fill_small_holes(polygon: Polygon) -> Polygon:
    keep = [ring for ring in polygon.interiors if Polygon(ring).area >= MIN_HOLE_M2]
    return Polygon(polygon.exterior, keep)


def _coords(geometry: BaseGeometry) -> Any:
    return json.loads(json.dumps(mapping(geometry)))["coordinates"]


def build(source: dict[str, Any]) -> dict[str, Any]:
    """The `derived` block for `source`: pure and deterministic, so the test can rebuild it
    and compare. Zones are single Polygons because the map draws a zone's first outer ring.

    * Z-08 Creek draw: the creek buffered to `corridorWidthM`, inside the outline, less the
      fence strip where the creek runs under the fence (a zone is one polygon, and the strip
      would otherwise be cut in two by the corridor).
    * Z-21 Fence line: the `stripWidthM` strip inside the fence.
    * Z-14 West grassland: the WorldCover grassland inside the outline, less the creek
      corridor (plus a setback), the fence strip and the oak woodland.
    """
    boundary_ll = shape(source["boundary"]["geometry"])
    features = source["features"]
    lon0, lat0 = boundary_ll.centroid.coords[0]
    epsg = _utm_epsg(lon0, lat0)
    to_m = Transformer.from_crs(4326, epsg, always_xy=True).transform
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True).transform

    boundary = transform(to_m, boundary_ll)
    creek_feature = features["creek"]
    creek_ll = shape(creek_feature["geometry"])
    creek_m = transform(to_m, creek_ll)
    if isinstance(creek_m, Polygon):
        corridor = creek_m.intersection(boundary)
    else:
        corridor = creek_m.buffer(float(creek_feature["corridorWidthM"]) / 2).intersection(boundary)
    fence_feature = features["fence"]
    fence_m = transform(to_m, shape(fence_feature["geometry"]))
    strip = _largest(fence_m.buffer(float(fence_feature["stripWidthM"])).intersection(boundary))
    creek_zone = _largest(corridor.difference(strip))
    grass = transform(to_m, shape(features["grassland"]["geometry"])).intersection(boundary)
    oak = transform(to_m, shape(features["oakWoodland"]["geometry"]))
    grass_zone = _largest(
        grass.difference(corridor.buffer(CREEK_SETBACK_M)).difference(strip).difference(oak)
    )

    width = fence_feature["stripWidthM"]
    corridor_width = creek_feature["corridorWidthM"]
    creek_name = creek_feature.get("name") or "the creek"
    plans = {
        "Z-14": (
            grass_zone,
            features["grassland"]["source"],
            "WorldCover grassland inside the demo outline, less the creek corridor, "
            "the fence strip and the oak woodland",
        ),
        "Z-21": (
            strip,
            fence_feature["source"],
            f"{width:g} m strip inside the simulated fence on the demo outline",
        ),
        "Z-08": (
            creek_zone,
            creek_feature["source"],
            f"{corridor_width:g} m corridor along {creek_name} (USGS NHD) inside the demo outline",
        ),
    }
    zones: dict[str, Any] = {}
    for zone_id, (footprint_m, source_id, basis) in plans.items():
        simple = _largest(footprint_m.simplify(SIMPLIFY_M, preserve_topology=True))
        simplified = _fill_small_holes(simple)
        footprint_ll = orient(_round(_as_polygon(transform(to_ll, simplified))), 1.0)
        anchor = _round(transform(to_ll, polylabel(simplified, tolerance=SIMPLIFY_M)))
        zones[zone_id] = {
            "footprint": {"type": "Polygon", "coordinates": _coords(footprint_ll)},
            "areaM2": _area_m2(footprint_ll),
            "anchor": list(anchor.coords[0]),
            "source": source_id,
            "basis": basis,
        }

    creek_inside = transform(to_ll, creek_m.intersection(boundary))
    centroid = _round(transform(to_ll, boundary.centroid))
    return {
        "generatedBy": GENERATED_BY,
        "siteAreaM2": _area_m2(_as_polygon(boundary_ll)),
        "centroid": list(centroid.coords[0]),
        "zones": zones,
        "creekLengthM": _length_m(creek_inside),
        "fenceLengthM": _length_m(shape(fence_feature["geometry"])),
    }


def dumps(document: dict[str, Any]) -> str:
    """JSON with one coordinate pair per line, so a moved vertex is a one-line diff."""
    text = json.dumps(document, indent=1, ensure_ascii=False)
    pair = re.compile(r"\[\s*(-?[\d.eE+-]+),\s*(-?[\d.eE+-]+)\s*\]")
    return pair.sub(r"[\1, \2]", text) + "\n"


def write() -> dict[str, Any]:
    document = load()
    document["derived"] = build(document)
    PATH.write_text(dumps(document), encoding="utf-8")
    return document


def _acres(m2: float) -> float:
    return m2 / 4046.8564224


def _site(document: dict[str, Any]) -> SiteCreate:
    derived = document["derived"]
    sources: dict[str, Any] = document["sources"]
    used = [document["boundary"]["source"]] + [
        feature["source"] for feature in document["features"].values()
    ]
    used_ids = [source_id for source_id in sources if source_id in used]
    camera = document["camera"]
    target_lon, target_lat = camera["target"]
    lon, lat = derived["centroid"]
    return SiteCreate(
        slug=document["slug"],
        name=document["name"],
        description=document["description"],
        boundary=PolygonSchema.model_validate(document["boundary"]["geometry"]),
        centroid=GeoPosition(longitude=lon, latitude=lat, height=None),
        metadata={
            "origin": "demo-ranch",
            "simulated": ["fleet", "zones", "plans", "work log"],
            "sources": {source_id: sources[source_id] for source_id in used_ids},
        },
        attribution=[
            Attribution(
                text=sources[source_id]["text"],
                organization=sources[source_id].get("organization"),
                url=sources[source_id].get("url"),
            )
            for source_id in used_ids
        ],
        # WorldCover's licence is the one that binds: its credit must stay visible.
        license=LicenseMetadata(
            name="CC BY 4.0",
            spdx_id="CC-BY-4.0",
            url="https://creativecommons.org/licenses/by/4.0/",
            requires_attribution=True,
            notes=(
                "Zone outlines derive from ESA WorldCover 2021 (CC BY 4.0) and the USGS "
                "National Hydrography Dataset (public domain). The outline is a demo area, "
                "and the fleet, zones, plans and work log are simulated."
            ),
        ),
        camera_bookmarks=[
            overview_bookmark(
                "Overview",
                target_lon,
                target_lat,
                float(camera["targetHeightM"]),
                float(camera["headingDeg"]),
                float(camera["pitchDeg"]),
                float(camera["rangeM"]),
            )
        ],
    )


@cache
def demo_ranch_site() -> SiteCreate:
    """The `demo-ranch` catalog site, read from the JSON once per process."""
    return _site(load())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--write", action="store_true", help="rewrite the derived block of demo_ranch.json"
    )
    args = parser.parse_args(argv)
    document = write() if args.write else load()
    derived = document["derived"]
    stale = not args.write and build(document) != derived
    for zone_id, zone in derived["zones"].items():
        sys.stdout.write(f"{zone_id}: {_acres(zone['areaM2']):.1f} ac\n")
    sys.stdout.write(
        f"site: {_acres(derived['siteAreaM2']):.1f} ac, creek {derived['creekLengthM']:.0f} m, "
        f"fence {derived['fenceLengthM']:.0f} m\n"
    )
    if stale:
        sys.stdout.write("derived is stale: run with --write\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
