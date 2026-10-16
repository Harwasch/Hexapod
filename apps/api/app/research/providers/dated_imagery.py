"""Dated public imagery for an area: the acquisitions the Time Lantern steps through.

Three public sources, each as one list entry per real acquisition, never an interpolated year:

* **NAIP** (USDA Farm Service Agency aerial photographs, U.S. public domain), one source per
  year it was flown here, drawn from a Planetary Computer mosaic of that year's quarter quads.
  The parts are the quarter quads with their own flight dates. Their accuracy is the sentence the
  item's own FGDC record states ("+/- 6 meters to ground specification"); a year whose records
  carry no metadata says "not stated" rather than borrowing another year's.
* **Sentinel-2** (Copernicus, attribution required), the clearest scene of each dry season
  (June to September) since 2016, so that years compare like with like on Mediterranean
  rangeland, where spring green and summer gold are different pictures of the same ground.
* **USGS historical topographic maps** (U.S. public domain), one source per edition of each
  sheet covering the area, tiled by this API from the collection's GeoTIFFs
  (app/services/usgs_topo_tiles.py). Sheets from 1947 on were made to the National Map Accuracy
  Standards; earlier ones predate them, and say so.

Every fetch goes to a registered endpoint (SourceSpec, fetch_json: no redirects, bounded) or,
for the NAIP metadata text, to one allow-listed prefix. Catalogs are cached in process for 6 h
per area rounded to 0.01°, mosaic registrations for 24 h. The three SourceSpecs join the
research registry (open_data.SOURCES) so /research/sources lists them; the overview does not
run them.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from typing import Any, NamedTuple
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import httpx
from pydantic import HttpUrl, ValidationError
from shapely.geometry import shape
from shapely.ops import unary_union

from app.analysis.vegetation import LICENSE_URL as SENTINEL_LICENSE_URL
from app.research.providers.base import (
    SourceContext,
    SourceError,
    SourceResult,
    SourceSpec,
    fetch_json,
)
from app.schemas.dated_imagery import (
    DatedImageryCatalog,
    DatedImageryPart,
    DatedImageryProviderStatus,
    DatedImagerySource,
    DatedImageryTiles,
    StatedAccuracy,
)
from app.schemas.geojson import Polygon
from app.services import usgs_topo_tiles
from app.services.errors import InvalidInputError

PC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_MOSAIC_REGISTER = "https://planetarycomputer.microsoft.com/api/data/v1/mosaic/register"
PC_DATA = "https://planetarycomputer.microsoft.com/api/data/v1"
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
NAIP_METADATA_PREFIX = "https://naipeuwest.blob.core.windows.net/naip/"
USGS_COPYRIGHT_URL = (
    "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
)
NMAS_URL = "https://pubs.usgs.gov/fs/1999/0171/report.pdf"

DATED_IMAGERY_SOURCES = {
    "naip-aerial": SourceSpec(
        id="naip-aerial",
        name="NAIP aerial photographs by year",
        domain="history",
        coverage="Contiguous United States, flown state by state about every two years since 2003",
        resolution="0.6 m since about 2016 in most states, 1 m before",
        license="U.S. public domain (USDA Farm Service Agency)",
        attribution="USDA FSA National Agriculture Imagery Program; served by Microsoft Planetary Computer",
        documentation_url="https://planetarycomputer.microsoft.com/dataset/naip",
        endpoint=PC_SEARCH,
    ),
    "sentinel-2-true-colour": SourceSpec(
        id="sentinel-2-true-colour",
        name="Sentinel-2 true-colour scenes, one per dry season",
        domain="history",
        coverage="Global land from 2016; the clearest June to September scene of each year",
        resolution="10 m",
        license="Copernicus Sentinel Data legal notice (free, full and open use with attribution)",
        attribution="Contains modified Copernicus Sentinel data; served by Microsoft Planetary Computer",
        documentation_url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
        endpoint=PC_SEARCH,
    ),
    "usgs-historical-topo-sheets": SourceSpec(
        id="usgs-historical-topo-sheets",
        name="USGS historical topographic maps as map tiles",
        domain="history",
        coverage="United States and covered territories; historical quadrangle collection",
        resolution="Scanned sheets: about 2 m a pixel at 1:24,000, 5 m at 1:62,500",
        license="U.S. public domain",
        attribution="U.S. Geological Survey, Historical Topographic Map Collection",
        documentation_url="https://www.usgs.gov/programs/national-geospatial-program/historical-topographic-maps-preserving-past",
        endpoint=TNM_PRODUCTS,
    ),
}
#: Planetary Computer's mosaic registration: the NAIP source's second endpoint, registered so
#: that fetch_json's "registry endpoints only" holds. Not a source of its own.
MOSAIC_REGISTER = SourceSpec(
    id="naip-aerial-mosaic",
    name="NAIP yearly mosaics",
    domain="history",
    coverage=DATED_IMAGERY_SOURCES["naip-aerial"].coverage,
    resolution=DATED_IMAGERY_SOURCES["naip-aerial"].resolution,
    license=DATED_IMAGERY_SOURCES["naip-aerial"].license,
    attribution=DATED_IMAGERY_SOURCES["naip-aerial"].attribution,
    documentation_url="https://planetarycomputer.microsoft.com/docs/reference/data/",
    endpoint=PC_MOSAIC_REGISTER,
)

#: Each side of an area asked about, at most (degrees): a ranch, not a county.
MAX_SIDE_DEG = 0.25
CATALOG_TTL_S = 6 * 3600
#: A catalog with a provider that failed is asked again soon.
FAILURE_TTL_S = 120
MOSAIC_TTL_S = 24 * 3600
CACHE_SIZE = 256
#: NAIP search pages (100 items each); a 0.25° area holds about 200 items over ten years.
NAIP_PAGES = 4
NAIP_MAX_LEVEL = 18
SENTINEL_MAX_LEVEL = 15
SENTINEL_FIRST_YEAR = 2016
SENTINEL_MAX_CLOUD = 10
TOPO_MAX_LEVEL = 17
TOPO_MAX_RECORDS = 50
#: Sheets kept per catalog, newest first: enough for every edition of the sheets around a ranch.
TOPO_MAX_SHEETS = 16
METADATA_MAX_BYTES = 64 * 1024
TIMEOUT_S = 25

NAIP_DATE_MEANING = (
    "Flight dates of the photographs in this area; a year's quarter quads may have been flown "
    "on different days."
)
SENTINEL_DATE_MEANING = (
    "Acquisition date of the clearest scene (under 10% cloud) between June and September: the "
    "dry season, so years compare like with like on rangeland that is green in spring and gold "
    "in summer."
)
TOPO_DATE_MEANING = (
    "Catalog publication year; surveying, revision and scanning may have different dates."
)
NOT_STATED_NAIP = "This item's record states no positional accuracy."
PREDATES_NMAS = "Predates the National Map Accuracy Standards (1947); no accuracy is stated."


def round_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """An area rounded outward to 0.01°: the catalog's cache key and the area it searches."""
    west, south, east, north = bbox

    def down(value: float) -> float:
        return math.floor(round(value * 100, 6)) / 100

    def up(value: float) -> float:
        return math.ceil(round(value * 100, 6)) / 100

    return down(west), down(south), up(east), up(north)


def check_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    if not all(math.isfinite(value) for value in bbox):
        raise InvalidInputError("The area must be four finite numbers.")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise InvalidInputError("The area must run west to east and south to north.")
    if east - west > MAX_SIDE_DEG or north - south > MAX_SIDE_DEG:
        raise InvalidInputError(f"Choose a smaller area: each side at most {MAX_SIDE_DEG}°.")
    return bbox


def _box(west: float, south: float, east: float, north: float) -> Polygon:
    return Polygon(
        coordinates=[[[west, south], [east, south], [east, north], [west, north], [west, south]]]
    )


def _polygon(geometry: Any) -> Polygon | None:
    """A STAC item's footprint as a Polygon, or None for anything else."""
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        return None
    try:
        return Polygon.model_validate(geometry)
    except ValidationError:
        return None


def _day(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _self_link(item: dict[str, Any]) -> str | None:
    for link in item.get("links", []):
        href = link.get("href") if isinstance(link, dict) else None
        if (
            link.get("rel") == "self"
            and isinstance(href, str)
            and href.startswith("https://planetarycomputer.microsoft.com/api/stac/v1/")
        ):
            return href
    return None


def _union_bounds(parts: list[DatedImageryPart]) -> list[float]:
    west, south, east, north = unary_union(
        [shape(part.footprint.model_dump()) for part in parts]
    ).bounds
    return [west, south, east, north]


class _TTLCache:
    """A small thread-safe cache with an expiry per entry (seconds of `clock`)."""

    def __init__(self, size: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.size, self.clock = size, clock
        self.entries: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self.lock:
            hit = self.entries.get(key)
            if hit is None:
                return None
            if hit[0] <= self.clock():
                del self.entries[key]
                return None
            self.entries.move_to_end(key)
            return hit[1]

    def put(self, key: Any, value: Any, ttl: float) -> None:
        with self.lock:
            self.entries[key] = (self.clock() + ttl, value)
            self.entries.move_to_end(key)
            while len(self.entries) > self.size:
                self.entries.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()


_catalogs = _TTLCache(CACHE_SIZE)
_mosaics = _TTLCache(CACHE_SIZE)
_metadata = _TTLCache(CACHE_SIZE)


def clear_caches() -> None:
    """Forget every cached catalog, mosaic and metadata sentence (tests)."""
    for cache in (_catalogs, _mosaics, _metadata):
        cache.clear()


# ---- NAIP --------------------------------------------------------------------------------


def _naip_items(bbox: tuple[float, float, float, float], client: httpx.Client) -> list[Any]:
    spec = DATED_IMAGERY_SOURCES["naip-aerial"]
    params: dict[str, Any] = {
        "collections": "naip",
        "bbox": ",".join(str(value) for value in bbox),
        "limit": 100,
    }
    items: list[Any] = []
    for _ in range(NAIP_PAGES):
        raw, _url = fetch_json(client, spec, params, timeout=TIMEOUT_S)
        features = raw.get("features")
        if not isinstance(features, list):
            raise SourceError("The NAIP catalog returned an unexpected response.")
        items.extend(features)
        token = _next_token(raw)
        if token is None:
            break
        params = {**params, "token": token}
    return items


def _next_token(raw: dict[str, Any]) -> str | None:
    """The paging token of a search's `next` link, when it pages the same search endpoint."""
    for link in raw.get("links", []):
        href = link.get("href") if isinstance(link, dict) else None
        if link.get("rel") != "next" or not isinstance(href, str):
            continue
        if not href.startswith(PC_SEARCH + "?"):
            return None
        tokens = parse_qs(urlsplit(href).query).get("token")
        return tokens[0] if tokens else None
    return None


def _naip_mosaic(
    client: httpx.Client, year: int, states: list[str], bbox: tuple[float, float, float, float]
) -> str:
    """A registered Planetary Computer mosaic of one NAIP year over the area: its search id."""
    key = (year, tuple(states), bbox)
    cached = _mosaics.get(key)
    if isinstance(cached, str):
        return cached
    state_filter: dict[str, Any] = (
        {"op": "=", "args": [{"property": "naip:state"}, states[0]]}
        if len(states) == 1
        else {
            "op": "or",
            "args": [{"op": "=", "args": [{"property": "naip:state"}, s]} for s in states],
        }
    )
    body = {
        "filter-lang": "cql2-json",
        "filter": {
            "op": "and",
            "args": [
                {"op": "=", "args": [{"property": "collection"}, "naip"]},
                {"op": "=", "args": [{"property": "naip:year"}, str(year)]},
                state_filter,
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, _box(*bbox).model_dump()],
                },
            ],
        },
        "sortby": [{"field": "datetime", "direction": "desc"}],
    }
    raw, _url = fetch_json(
        client, MOSAIC_REGISTER, body, method="POST", encoding="json", timeout=TIMEOUT_S
    )
    search_id = raw.get("searchid") or raw.get("id")
    if not isinstance(search_id, str) or not re.fullmatch(r"[0-9a-f]{16,64}", search_id):
        raise SourceError("The NAIP mosaic registration returned an unexpected response.")
    _mosaics.put(key, search_id, MOSAIC_TTL_S)
    return search_id


def _naip_accuracy(client: httpx.Client, year: int, href: str | None) -> StatedAccuracy:
    """The item's own FGDC sentence on positional accuracy, fetched once per year, or "not
    stated" when the item has no metadata record (NAIP from 2020 on, here)."""
    if not href or not href.startswith(NAIP_METADATA_PREFIX) or not href.endswith(".txt"):
        return StatedAccuracy(kind="not-stated", basis=NOT_STATED_NAIP)
    cached = _metadata.get(("naip", year, href))
    if isinstance(cached, StatedAccuracy):
        return cached
    text = _fetch_text(client, href)
    accuracy = _stated_accuracy(text, href)
    _metadata.put(("naip", year, href), accuracy, CATALOG_TTL_S)
    return accuracy


def _fetch_text(client: httpx.Client, url: str) -> str:
    """A small text record from the allow-listed NAIP prefix: no redirects, bounded."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.query or parsed.fragment or "/../" in parsed.path:
        raise SourceError("The NAIP metadata record is not at an approved location.")
    with client.stream("GET", url, follow_redirects=False, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > METADATA_MAX_BYTES:
                raise SourceError("The NAIP metadata record exceeded its size limit.")
    return data.decode("utf-8", errors="replace")


def _stated_accuracy(text: str, url: str) -> StatedAccuracy:
    """The sentence of an FGDC record that states NAIP's ground specification, quoted."""
    flat = " ".join(text.split())
    for sentence in re.split(r"(?<=\.)\s+", flat):
        if "to ground specification" not in sentence:
            continue
        match = re.search(r"\+/-\s*(\d+(?:\.\d+)?)\s*met", sentence)
        return StatedAccuracy(
            kind="stated-by-source",
            value_m=float(match[1]) if match else None,
            basis=sentence[:1200],
            citation_url=HttpUrl(url),
        )
    return StatedAccuracy(kind="not-stated", basis=NOT_STATED_NAIP)


def _day_label(day: date) -> str:
    return f"{day.day} {day:%B %Y}"


def naip_years(
    bbox: tuple[float, float, float, float], client: httpx.Client
) -> list[DatedImagerySource]:
    """One source per year NAIP was flown over the area, newest first."""
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in _naip_items(bbox, client):
        if not isinstance(item, dict):
            continue
        properties = item.get("properties") or {}
        raw_year = properties.get("naip:year")
        if not isinstance(raw_year, (str, int)) or not str(raw_year).isdigit():
            continue
        by_year[int(raw_year)].append(item)
    years = sorted(by_year, reverse=True)
    # A mosaic registration and a metadata record per year: asked side by side.
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda year: _naip_year(bbox, client, year, by_year[year]), years))
    return [source for source in found if source is not None]


def _naip_year(
    bbox: tuple[float, float, float, float],
    client: httpx.Client,
    year: int,
    items: list[dict[str, Any]],
) -> DatedImagerySource | None:
    spec = DATED_IMAGERY_SOURCES["naip-aerial"]
    parts: list[DatedImageryPart] = []
    states: set[str] = set()
    resolutions: list[float] = []
    for item in items:
        properties = item.get("properties") or {}
        footprint = _polygon(item.get("geometry"))
        record = _self_link(item)
        if footprint is None or record is None:
            continue
        state = properties.get("naip:state")
        if isinstance(state, str) and re.fullmatch(r"[a-z]{2}", state):
            states.add(state)
        gsd = properties.get("gsd")
        if isinstance(gsd, (int, float)) and gsd > 0:
            resolutions.append(float(gsd))
        metadata = (item.get("assets") or {}).get("metadata") or {}
        href = metadata.get("href") if isinstance(metadata, dict) else None
        parts.append(
            DatedImageryPart(
                observed_on=_day(properties.get("datetime")),
                observed_year=year,
                footprint=footprint,
                record_url=HttpUrl(record),
                accuracy=_naip_accuracy(client, year, href if isinstance(href, str) else None),
            )
        )
    if not parts or not states:
        return None
    parts.sort(key=lambda part: part.observed_on or date(year, 1, 1), reverse=True)
    parts = parts[:40]
    search_id = _naip_mosaic(client, year, sorted(states), bbox)
    query = urlencode(
        {"collection": "naip", "assets": "image", "asset_bidx": "image|1,2,3", "format": "png"}
    )
    days = sorted({part.observed_on for part in parts if part.observed_on})
    if not days:
        label = f"Flown {year}"
    elif len(days) == 1:
        label = f"Flown {_day_label(days[0])}"
    elif days[0].month == days[-1].month:
        label = f"Flown {days[0].day} to {_day_label(days[-1])}"
    else:
        label = f"Flown {days[0].day} {days[0]:%b} to {days[-1].day} {days[-1]:%b %Y}"
    return DatedImagerySource(
        id=f"naip-{year}",
        provider=spec.id,
        kind="aerial-photo",
        title=f"NAIP {year} aerial photograph",
        date_label=label,
        date_meaning=NAIP_DATE_MEANING,
        sort_date=days[-1] if days else date(year, 12, 31),
        resolution_m=min(resolutions) if resolutions else 1.0,
        bounds=_union_bounds(parts),
        tiles=DatedImageryTiles(
            scheme="web-mercator",
            url_template=(
                f"{PC_DATA}/mosaic/{search_id}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x?{query}"
            ),
            maximum_level=NAIP_MAX_LEVEL,
        ),
        license=spec.license,
        license_url=HttpUrl("https://planetarycomputer.microsoft.com/dataset/naip"),
        attribution=spec.attribution,
        parts=parts,
    )


# ---- Sentinel-2 ---------------------------------------------------------------------------


def _sentinel_year(
    bbox: tuple[float, float, float, float], client: httpx.Client, year: int
) -> DatedImagerySource | None:
    spec = DATED_IMAGERY_SOURCES["sentinel-2-true-colour"]
    raw, _url = fetch_json(
        client,
        spec,
        {
            "collections": ["sentinel-2-l2a"],
            "bbox": list(bbox),
            "datetime": f"{year}-06-01T00:00:00Z/{year}-09-30T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": SENTINEL_MAX_CLOUD}},
            "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
            "limit": 1,
        },
        method="POST",
        encoding="json",
        timeout=TIMEOUT_S,
    )
    features = raw.get("features")
    if not isinstance(features, list):
        raise SourceError("The Sentinel-2 catalog returned an unexpected response.")
    if not features or not isinstance(features[0], dict):
        return None
    item = features[0]
    item_id = item.get("id")
    footprint = _polygon(item.get("geometry"))
    record = _self_link(item)
    observed = _day((item.get("properties") or {}).get("datetime"))
    if (
        not isinstance(item_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", item_id)
        or footprint is None
        or record is None
        or observed is None
    ):
        return None
    cloud = (item.get("properties") or {}).get("eo:cloud_cover")
    query = urlencode(
        {
            "collection": "sentinel-2-l2a",
            "item": item_id,
            "assets": "visual",
            "asset_bidx": "visual|1,2,3",
            "nodata": 0,
            "format": "png",
        }
    )
    part = DatedImageryPart(
        observed_on=observed,
        observed_year=observed.year,
        footprint=footprint,
        record_url=HttpUrl(record),
        accuracy=StatedAccuracy(
            kind="not-stated",
            basis="The scene's record states no positional accuracy for this area.",
        ),
    )
    return DatedImagerySource(
        id=f"sentinel-2-{item_id}",
        provider=spec.id,
        kind="satellite",
        title=f"Sentinel-2 {observed.year} true colour",
        date_label=f"Seen {_day_label(observed)}",
        date_meaning=SENTINEL_DATE_MEANING,
        sort_date=observed,
        resolution_m=10,
        bounds=_union_bounds([part]),
        tiles=DatedImageryTiles(
            scheme="web-mercator",
            url_template=f"{PC_DATA}/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x?{query}",
            maximum_level=SENTINEL_MAX_LEVEL,
        ),
        license=spec.license,
        license_url=HttpUrl(SENTINEL_LICENSE_URL),
        attribution=(
            f"Contains modified Copernicus Sentinel data {observed.year}; "
            "served by Microsoft Planetary Computer"
        ),
        parts=[part],
        warnings=[
            f"The catalog puts the scene's cloud cover at {float(cloud):.1f}%; "
            "a cloud may hide part of this area."
        ]
        if isinstance(cloud, (int, float)) and cloud >= 1
        else [],
    )


def sentinel2_years(
    bbox: tuple[float, float, float, float],
    client: httpx.Client,
    *,
    today: date | None = None,
) -> list[DatedImagerySource]:
    """The clearest dry-season scene of each year since 2016, newest first."""
    last = (today or datetime.now(UTC).date()).year
    years = list(range(last, SENTINEL_FIRST_YEAR - 1, -1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda year: _sentinel_year(bbox, client, year), years))
    return [source for source in found if source is not None]


# ---- USGS historical topographic maps ------------------------------------------------------


def _sheet_of(url: str) -> str | None:
    """The sheet name (STATE/STATE_name_..._geo) of an allow-listed GeoTIFF URL."""
    if not url.startswith(usgs_topo_tiles.PREFIX) or not url.endswith(".tif"):
        return None
    sheet = unquote(url[len(usgs_topo_tiles.PREFIX) : -len(".tif")])
    return sheet if usgs_topo_tiles.SHEET.fullmatch(sheet) and ".." not in sheet else None


def _topo_accuracy(year: int, scale: int) -> StatedAccuracy:
    if year >= 1947:
        return StatedAccuracy(
            kind="standard",
            value_m=round(scale * 0.0254 / 50, 2),
            basis=(
                "National Map Accuracy Standards (1947): 90% of well-defined points within "
                "1/50 inch at map scale, if the sheet states compliance"
            ),
            citation_url=HttpUrl(NMAS_URL),
        )
    return StatedAccuracy(kind="not-stated", basis=PREDATES_NMAS)


def _datum_warning(neatline: tuple[float, float, float, float]) -> str:
    name, accuracy, better_missing = usgs_topo_tiles.datum_shift(neatline)
    stated = f", stated accuracy about {accuracy:g} m" if accuracy else ", accuracy not stated"
    missing = (
        " A more accurate grid-based shift exists but is not installed on this server."
        if better_missing
        else ""
    )
    return f"Shifted from NAD27 to WGS 84 with “{name}”{stated}.{missing}"


class _Sheet(NamedTuple):
    year: int
    scale: int
    name: str
    sheet: str
    box: tuple[float, float, float, float]
    meta: str


def usgs_topo_sheets(
    bbox: tuple[float, float, float, float], client: httpx.Client
) -> list[DatedImagerySource]:
    """One source per edition of each topographic sheet covering the area, newest first.
    1:250,000 sheets only when nothing finer covers it; reprints of a sheet in the same year
    and scale count once."""
    spec = DATED_IMAGERY_SOURCES["usgs-historical-topo-sheets"]
    raw, _url = fetch_json(
        client,
        spec,
        {
            "datasets": "Historical Topographic Maps",
            "bbox": ",".join(str(value) for value in bbox),
            "max": TOPO_MAX_RECORDS,
            "outputFormat": "JSON",
        },
        timeout=TIMEOUT_S,
    )
    if raw.get("errors"):
        raise SourceError("The National Map catalog returned an error.")
    records = raw.get("items")
    if not isinstance(records, list):
        raise SourceError("The National Map catalog returned an unexpected response.")
    west, south, east, north = bbox
    candidates: list[_Sheet] = []
    seen: set[tuple[str, int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        url = str((record.get("urls") or {}).get("GeoTIFF", ""))
        sheet = _sheet_of(url)
        meta = str(record.get("metaUrl", ""))
        if sheet is None or not meta.startswith("https://www.sciencebase.gov/catalog/item/"):
            continue
        match = usgs_topo_tiles.SHEET.fullmatch(sheet)
        if match is None:
            continue
        year, scale, name = int(match["year"]), int(match["scale"]), match["name"]
        extent = record.get("boundingBox") or {}
        try:
            box = (
                float(extent["minX"]),
                float(extent["minY"]),
                float(extent["maxX"]),
                float(extent["maxY"]),
            )
            usgs_topo_tiles.check_neatline(box)
        except (KeyError, TypeError, ValueError):
            continue
        if not (box[0] < east and box[2] > west and box[1] < north and box[3] > south):
            continue
        if (name, year, scale) in seen:
            continue
        seen.add((name, year, scale))
        candidates.append(_Sheet(year, scale, name, sheet, box, meta))
    finer = [c for c in candidates if c.scale < 250_000]
    kept = sorted(finer or candidates, key=lambda c: (c.year, -c.scale), reverse=True)
    sources: list[DatedImagerySource] = []
    for year, scale, name, sheet, box, meta in kept[:TOPO_MAX_SHEETS]:
        neatline, snapped = usgs_topo_tiles.snap_neatline(box)
        ring = usgs_topo_tiles.neatline_wgs84(neatline)
        warnings = [_datum_warning(neatline)]
        if not snapped:
            warnings.append(
                "The sheet's edge could not be matched to the map grid, so the catalog footprint "
                "trims it: a strip of margin may show, or of map be cut."
            )
        part = DatedImageryPart(
            observed_on=None,
            observed_year=year,
            footprint=Polygon(coordinates=[[list(point) for point in ring]]),
            record_url=HttpUrl(meta),
            accuracy=_topo_accuracy(year, scale),
        )
        query = urlencode(
            {
                "sheet": sheet,
                "w": neatline[0],
                "s": neatline[1],
                "e": neatline[2],
                "n": neatline[3],
            },
            quote_via=quote,
        )
        sources.append(
            DatedImagerySource(
                id=f"usgs-topo-{quote(sheet.split('/')[-1], safe='')}",
                provider=spec.id,
                kind="historical-map",
                title=f"USGS 1:{scale:,} {name}",
                date_label=f"Map dated {year}",
                date_meaning=TOPO_DATE_MEANING,
                sort_date=date(year, 1, 1),
                resolution_m=round(scale * 0.0254 / 300, 2),
                map_scale=scale,
                bounds=list(usgs_topo_tiles.ring_bounds(ring)),
                tiles=DatedImageryTiles(
                    scheme="geographic",
                    url_template=f"/api/v1/imagery/usgs-topo/tiles/{{z}}/{{x}}/{{y}}.png?{query}",
                    maximum_level=TOPO_MAX_LEVEL,
                ),
                license=spec.license,
                license_url=HttpUrl(USGS_COPYRIGHT_URL),
                attribution=spec.attribution,
                parts=[part],
                warnings=warnings,
            )
        )
    return sources


# ---- The catalog ----------------------------------------------------------------------------

Lister = Callable[[tuple[float, float, float, float], httpx.Client], list[DatedImagerySource]]
PROVIDERS: dict[str, Lister] = {
    "naip-aerial": naip_years,
    "sentinel-2-true-colour": sentinel2_years,
    "usgs-historical-topo-sheets": usgs_topo_sheets,
}


def _status(provider: str, sources: list[DatedImagerySource]) -> DatedImageryProviderStatus:
    spec = DATED_IMAGERY_SOURCES[provider]
    if not sources:
        return DatedImageryProviderStatus(
            provider=provider,
            name=spec.name,
            status="empty",
            summary=f"{spec.name}: none found for this area.",
        )
    years = sorted({source.sort_date.year for source in sources})
    span = f"{years[0]}" if len(years) == 1 else f"{years[0]} to {years[-1]}"
    return DatedImageryProviderStatus(
        provider=provider,
        name=spec.name,
        status="available",
        summary=f"{len(sources)} found, {span}.",
    )


def _failure(provider: str, error: Exception) -> DatedImageryProviderStatus:
    spec = DATED_IMAGERY_SOURCES[provider]
    reason = (
        f"HTTP {error.response.status_code}"
        if isinstance(error, httpx.HTTPStatusError)
        else "timed out"
        if isinstance(error, httpx.TimeoutException)
        else type(error).__name__
    )
    return DatedImageryProviderStatus(
        provider=provider,
        name=spec.name,
        status="unavailable",
        summary=f"{spec.name} could not be reached ({reason}). Try again in a few minutes.",
    )


def cached(bbox: tuple[float, float, float, float]) -> DatedImageryCatalog | None:
    """The catalog for an area if this server has it (no outbound call), else None."""
    hit = _catalogs.get(round_bbox(bbox))
    return hit.model_copy(update={"cached": True}) if isinstance(hit, DatedImageryCatalog) else None


def discover(bbox: tuple[float, float, float, float], client: httpx.Client) -> DatedImageryCatalog:
    """Every dated source for an area, newest first, with each provider's status. A provider
    that fails is "unavailable" and the others still answer."""
    area = round_bbox(check_bbox(bbox))
    hit = cached(area)
    if hit is not None:
        return hit

    def ask(provider: str) -> list[DatedImagerySource] | DatedImageryProviderStatus:
        try:
            return PROVIDERS[provider](area, client)
        except (httpx.HTTPError, SourceError, ValueError, KeyError, TypeError) as error:
            return _failure(provider, error)

    # The three providers are different services: asked side by side, the slowest decides.
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        answers = list(pool.map(ask, PROVIDERS))
    sources: list[DatedImagerySource] = []
    statuses: list[DatedImageryProviderStatus] = []
    failed = False
    for provider, answer in zip(PROVIDERS, answers, strict=True):
        if isinstance(answer, DatedImageryProviderStatus):
            statuses.append(answer)
            failed = True
            continue
        sources.extend(answer)
        statuses.append(_status(provider, answer))
    sources.sort(key=lambda source: source.sort_date, reverse=True)
    catalog = DatedImageryCatalog(
        bounds=list(area),
        retrieved_at=datetime.now(UTC),
        sources=sources,
        providers=statuses,
    )
    _catalogs.put(area, catalog, FAILURE_TTL_S if failed else CATALOG_TTL_S)
    return catalog


# ---- As research sources ------------------------------------------------------------------


def _adapter(provider: str) -> Callable[[SourceContext, httpx.Client], SourceResult]:
    """A dated source as the research registry runs it: what exists over the land, listed,
    not stored as evidence (the Time Lantern shows it; nothing here is a finding)."""

    def run(context: SourceContext, client: httpx.Client) -> SourceResult:
        west, south, east, north = context.geometry.bounds
        if east - west > MAX_SIDE_DEG or north - south > MAX_SIDE_DEG:
            return SourceResult(
                provider, "uncovered", "Choose a smaller area to list dated imagery."
            )
        found = PROVIDERS[provider](round_bbox((west, south, east, north)), client)
        status = _status(provider, found)
        return SourceResult(
            provider,
            status.status,
            status.summary + " Listed for the Time Lantern; not stored as evidence.",
            [],
            {
                "sources": [
                    {
                        "title": source.title,
                        "dateLabel": source.date_label,
                        "dateMeaning": source.date_meaning,
                        "resolutionM": source.resolution_m,
                        "license": source.license,
                    }
                    for source in found
                ]
            },
        )

    return run


ADAPTERS = {provider: _adapter(provider) for provider in PROVIDERS}
