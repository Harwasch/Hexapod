"""California parcel services: re-query every "available" county at its test point.

Reads ``app/research/providers/data/ca_parcels.json`` and, for each county marked
``available``, sends an envelope query (about 200 m around ``testPoint``) to the layer's
``/query`` endpoint with the recorded ``format``. A county passes when the reply has at
least one polygon whose coordinates fall near the point and whose ``idField`` is filled.
Prints one row per county and exits 1 if any fail.

    uv run python -m app.scripts.verify_ca_parcels                 # every available county
    uv run python -m app.scripts.verify_ca_parcels Kern "San Diego"   # just these
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

DATA_PATH = (
    Path(__file__).resolve().parents[1] / "research" / "providers" / "data" / "ca_parcels.json"
)
TIMEOUT_S = 25.0
HALF_SIDE_DEG = 0.002  # envelope half-width around the test point
NEAR_DEG = 0.02  # how far the first feature's bbox may sit from the point
F_PARAM = {"geojson": "geojson", "esrijson": "json"}


def _coords(node: Any, out: list[tuple[float, float]]) -> None:
    if (
        isinstance(node, list)
        and len(node) >= 2
        and all(isinstance(v, int | float) for v in node[:2])
    ):
        out.append((float(node[0]), float(node[1])))
    elif isinstance(node, list):
        for child in node:
            _coords(child, out)


def check(county: dict[str, Any]) -> tuple[bool, int, str, str]:
    """(passed, feature count, sample id, detail) for one county."""
    lon, lat = county["testPoint"]
    fmt = county["format"]
    params = {
        "f": F_PARAM[fmt],
        "where": "1=1",
        "geometry": f"{lon - HALF_SIDE_DEG},{lat - HALF_SIDE_DEG},{lon + HALF_SIDE_DEG},{lat + HALF_SIDE_DEG}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultRecordCount": "5",
    }
    started = time.monotonic()
    try:
        response = httpx.get(
            f"{county['endpoint']}/query", params=params, timeout=TIMEOUT_S, follow_redirects=True
        )
    except httpx.HTTPError as exc:
        return False, 0, "", f"{type(exc).__name__}: {str(exc)[:80]}"
    try:
        body = response.json()
    except ValueError:
        return False, 0, "", f"HTTP {response.status_code}, not JSON: {response.text[:60]!r}"
    elapsed = f"{time.monotonic() - started:.1f}s"
    if response.status_code != 200 or "error" in body:
        return False, 0, "", f"HTTP {response.status_code} {json.dumps(body.get('error', ''))[:80]}"
    features = body.get("features") or []
    if not features:
        return False, 0, "", f"no features ({elapsed})"
    first = features[0]
    if fmt == "geojson":
        geometry, props = first.get("geometry") or {}, first.get("properties") or {}
        is_polygon = geometry.get("type") in ("Polygon", "MultiPolygon")
        rings = geometry.get("coordinates")
    else:
        geometry, props = first.get("geometry") or {}, first.get("attributes") or {}
        is_polygon = "rings" in geometry
        rings = geometry.get("rings")
    points: list[tuple[float, float]] = []
    _coords(rings, points)
    if not is_polygon or not points:
        return False, len(features), "", f"not a polygon ({elapsed})"
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    near = (
        min(xs) - NEAR_DEG <= lon <= max(xs) + NEAR_DEG
        and min(ys) - NEAR_DEG <= lat <= max(ys) + NEAR_DEG
    )
    if not near:
        return (
            False,
            len(features),
            "",
            f"far from point: x {min(xs):.3f}..{max(xs):.3f} y {min(ys):.3f}..{max(ys):.3f}",
        )
    parcel_id = str(props.get(county["idField"]) or "").strip()
    if not parcel_id:
        return False, len(features), "", f"{county['idField']} missing or empty ({elapsed})"
    return True, len(features), parcel_id, elapsed


def main(argv: list[str]) -> int:
    wanted = set(argv[1:])
    counties = [
        c
        for c in json.loads(DATA_PATH.read_text(encoding="utf-8"))["counties"]
        if c["status"] == "available" and (not wanted or c["county"] in wanted)
    ]
    unknown = wanted - {c["county"] for c in counties}
    if unknown:
        sys.stderr.write(f"not available or unknown: {', '.join(sorted(unknown))}\n")
        return 2
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check, counties))
    sys.stdout.write(f"{'county':<16} {'result':<6} {'n':>2}  {'id':<20} detail\n")
    failed = 0
    for county, (passed, count, parcel_id, detail) in zip(counties, results, strict=True):
        failed += not passed
        mark = "PASS" if passed else "FAIL"
        sys.stdout.write(
            f"{county['county']:<16} {mark:<6} {count:>2}  {parcel_id[:20]:<20} {detail}\n"
        )
    sys.stdout.write(f"\n{len(counties) - failed}/{len(counties)} passed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
