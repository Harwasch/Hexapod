"""Coverage measurement: how far a machine travels to work an outline.

A port of `packages/geo/src/coverage.ts` (the console's preview passes) that the planner sizes
work with. Passes run along the outline's longer side, one spacing apart, and the length is
what lies inside the outline. It goes further than the preview: holes and every part of a
MultiPolygon are honoured (shapely clips the passes), and the projection is an azimuthal
equidistant one centred on the outline, so lengths are metres on the ground. The fixtures in
`packages/geo/fixtures/coverage_cases.json` hold both implementations to the same numbers.

It measures a planning estimate, not a field-ready route: no headland turns, obstacles or
slope, and the transit between passes is approximated by one spacing per span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pyproj import Geod, Transformer
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform

GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Coverage:
    kind: Literal["area", "line"]
    #: Geodesic area of an outline (holes excluded); None for a line.
    area_m2: float | None
    #: Working length: the passes inside an outline, or the line itself.
    length_m: float
    #: Travel between passes, one spacing per span after the first.
    transit_m: float
    passes: int
    #: Separate stretches of work: a pass that crosses a hole or a notch counts twice.
    spans: int
    spacing_m: float | None
    heading_deg: float | None
    #: True when `max_passes` widened the spacing past the one asked for.
    capped: bool


def geodesic_area_m2(geometry: BaseGeometry) -> float:
    """Area on the ellipsoid, holes excluded; orient() makes exteriors and holes sign-consistent."""
    polygons = (
        list(geometry.geoms)
        if isinstance(geometry, MultiPolygon)
        else [geometry]
        if isinstance(geometry, Polygon)
        else []
    )
    return sum(abs(GEOD.geometry_area_perimeter(orient(item))[0]) for item in polygons)


def _segments(geometry: BaseGeometry) -> list[LineString]:
    """The line pieces of an intersection result (points where a pass grazes a vertex drop out)."""
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [piece for part in geometry.geoms for piece in _segments(part)]
    return []


def measure(
    geometry: dict[str, Any],
    spacing_m: float,
    *,
    heading_deg: float | None = None,
    max_passes: int = 5000,
) -> Coverage | None:
    """Coverage of a GeoJSON geometry at `spacing_m` between passes; None for a point (or
    anything else that cannot be worked)."""
    shaped = shape(geometry)
    if shaped.is_empty:
        return None
    if isinstance(shaped, LineString):
        return Coverage(
            kind="line",
            area_m2=None,
            length_m=float(GEOD.geometry_length(shaped)),
            transit_m=0.0,
            passes=1,
            spans=1,
            spacing_m=None,
            heading_deg=None,
            capped=False,
        )
    if not isinstance(shaped, Polygon | MultiPolygon) or not spacing_m > 0:
        return None
    west, south, east, north = shaped.bounds
    projection = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={(south + north) / 2} +lon_0={(west + east) / 2} +datum=WGS84 +units=m",
        always_xy=True,
    )
    local = transform(projection.transform, shaped)
    minx, miny, maxx, maxy = local.bounds
    # Travel along the longer side, as the preview does (preferredHeadingDeg).
    heading = (
        float(heading_deg)
        if heading_deg is not None
        else (90.0 if maxx - minx >= maxy - miny else 0.0)
    )
    # Heading is clockwise from north (+y); rotating by (heading - 90) puts travel along +x.
    turned = rotate(local, heading - 90, origin=(0, 0))
    minx, miny, maxx, maxy = turned.bounds
    extent = maxy - miny
    if not extent > 0:
        return None
    spacing = spacing_m
    capped = False
    if math.ceil(extent / spacing) > max_passes:
        spacing = extent / max_passes
        capped = True
    count = max(1, math.ceil(extent / spacing))
    rows = [miny + spacing * (k + 0.5) for k in range(count)]
    rows = [y for y in rows if y <= maxy]
    sweep = MultiLineString([[(minx - 1, y), (maxx + 1, y)] for y in rows])
    pieces = _segments(sweep.intersection(turned))
    length = sum(piece.length for piece in pieces)
    passes = len({round(piece.coords[0][1], 6) for piece in pieces})
    spans = len(pieces)
    return Coverage(
        kind="area",
        area_m2=geodesic_area_m2(shaped),
        length_m=float(length),
        transit_m=float(max(0, spans - 1) * spacing),
        passes=passes,
        spans=spans,
        spacing_m=float(spacing),
        heading_deg=heading,
        capped=capped,
    )
