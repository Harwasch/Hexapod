"""Map tiles cut from USGS historical topographic sheets, for the Time Lantern.

The Historical Topographic Map Collection publishes every scanned sheet as a GeoTIFF on
prd-tnm.s3.amazonaws.com: tiled 512 px JPEG with seven overviews, on the sheet's own NAD27
polyconic projection, collar and all. A tile here is a 256 px PNG on the WGS1984Quad grid (the
grid the land raster and alignment tiles use, `land_rasters.TMS`), read through the one raster
allowlist (`raster_io.RasterOpener`, byte ranges only, bounded per tile) and warped with GDAL,
reading the overview that matches the tile's resolution.

The collar -- the white margin with the legend, the title and the neighbouring sheet names --
is removed by the neatline: the sheet's printed edge, which lies on the series graticule
(7.5 minutes for 1:24,000 sheets, 15 for 1:62,500, and so on) in NAD27. The catalog's bounding
box is that neatline when it sits on the 7.5-minute grid; it is snapped there when within
0.01°, and used as given (with a warning, in the catalog) otherwise. The neatline is shifted
from NAD27 to WGS 84 with the same operation PROJ picks for the warp, and only pixels inside it
are opaque.

Nothing is stored: sheets are read remotely per tile and the PNGs cached in process (and for a
week by browsers, as public data). The URL is built only from the fixed prefix and a validated
sheet name, so the route cannot be turned into a fetch of anything else.
"""

from __future__ import annotations

import io
import math
import re
import threading
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from urllib.parse import quote

import httpx
import morecantile
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from pyproj.transformer import AreaOfInterest, TransformerGroup
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.io import DatasetReader
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds as window_from_bounds

from app.analysis.raster_io import RasterOpener, ReadBudget
from app.services.errors import InvalidInputError, NotFoundError
from app.services.land_rasters import TMS, transparent_tile

PREFIX = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/GeoTIFF/"
#: STATE/STATE_Map name_scan id_year_scale_geo, as the collection names its GeoTIFFs.
SHEET = re.compile(
    r"^(?P<state>[A-Z]{2})/(?P=state)_(?P<name>[A-Za-z0-9 .'()-]{1,80})"
    r"_(?P<scan>\d{3,7})_(?P<year>\d{4})_(?P<scale>\d{4,7})_geo$"
)
#: A neatline wider or taller than this is not one sheet (1:250,000 sheets are 1° by 2°).
MAX_SPAN_DEG = 2.0
#: Every standard series sheet edge lies on a multiple of 7.5 minutes.
GRATICULE_DEG = 7.5 / 60
#: A catalog edge this close to the graticule is the graticule (degrees).
SNAP_DEG = 0.01
TILE_PX = 256
#: Rendered tiles kept in process: a pan over a sheet and back is free.
CACHE_TILES = 256
#: Sheets read at once. Each read holds a server thread for a few range requests.
MAX_IN_FLIGHT = 4
SLOT_WAIT_S = 15.0
USER_AGENT = "LivingWorld-DatedImagery/1.0"

Neatline = tuple[float, float, float, float]


class RenderBusy(Exception):  # noqa: N818 - a condition, answered 503
    """Every slot is reading a sheet; the tile is not drawn now, rather than queued."""


def parse_sheet(sheet: str) -> re.Match[str]:
    """The sheet name's parts, or InvalidInputError for anything that is not one."""
    match = SHEET.fullmatch(sheet)
    if match is None or ".." in sheet:
        raise InvalidInputError(
            "Name a USGS historical topographic sheet as STATE/STATE_name_..._geo."
        )
    return match


def sheet_url(sheet: str) -> str:
    """The sheet's GeoTIFF, built only from the fixed prefix and a validated name."""
    parse_sheet(sheet)
    return f"{PREFIX}{quote(sheet, safe='/')}.tif"


def check_neatline(neatline: Neatline) -> Neatline:
    west, south, east, north = neatline
    if not all(math.isfinite(value) for value in neatline):
        raise InvalidInputError("The neatline must be four finite numbers.")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise InvalidInputError("The neatline must run west to east and south to north.")
    if east - west > MAX_SPAN_DEG or north - south > MAX_SPAN_DEG:
        raise InvalidInputError("The neatline is larger than one map sheet.")
    return neatline


def snap_neatline(bounds: Neatline) -> tuple[Neatline, bool]:
    """The catalog box on the 7.5-minute graticule when every edge is within SNAP_DEG of it,
    and whether it was snapped. A sheet off the standard grid keeps the catalog's box."""
    snapped = tuple(round(value / GRATICULE_DEG) * GRATICULE_DEG for value in bounds)
    if all(abs(a - b) <= SNAP_DEG for a, b in zip(snapped, bounds, strict=True)):
        west, south, east, north = (round(value, 6) for value in snapped)
        return (west, south, east, north), True
    return bounds, False


def _transformer(neatline: Neatline) -> Transformer:
    west, south, east, north = neatline
    # PROJ warns that the NADCON grid is missing; the catalog says so in words (`datum_shift`).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Transformer.from_crs(
            "EPSG:4267",
            "EPSG:4326",
            always_xy=True,
            area_of_interest=AreaOfInterest(west, south, east, north),
        )


@lru_cache(maxsize=512)
def neatline_wgs84(neatline: Neatline) -> tuple[tuple[float, float], ...]:
    """The NAD27 neatline as a closed WGS 84 ring, densified so its curved edges stay curved."""
    west, south, east, north = neatline
    steps = 16
    ring: list[tuple[float, float]] = []
    for i in range(steps):
        ring.append((west + (east - west) * i / steps, south))
    for i in range(steps):
        ring.append((east, south + (north - south) * i / steps))
    for i in range(steps):
        ring.append((east - (east - west) * i / steps, north))
    for i in range(steps):
        ring.append((west, north - (north - south) * i / steps))
    xs, ys = _transformer(neatline).transform([p[0] for p in ring], [p[1] for p in ring])
    shifted = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
    return (*shifted, shifted[0])


def ring_bounds(ring: tuple[tuple[float, float], ...]) -> Neatline:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


@lru_cache(maxsize=256)
def datum_shift(neatline: Neatline) -> tuple[str, float | None, bool]:
    """The NAD27 to WGS 84 operation PROJ uses for a sheet here: its name, its stated accuracy
    (m, when it states one) and whether a better operation exists that this server lacks (the
    NADCON grids are not installed)."""
    west, south, east, north = neatline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        group = TransformerGroup(
            "EPSG:4267",
            "EPSG:4326",
            always_xy=True,
            area_of_interest=AreaOfInterest(west, south, east, north),
        )
    if not group.transformers:
        return "no operation available", None, not group.best_available
    best = group.transformers[0]
    name = best.description.replace(" (with axis order normalized for visualization)", "")
    accuracy = best.accuracy if best.accuracy and best.accuracy > 0 else None
    return name, accuracy, not group.best_available


_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(headers={"User-Agent": USER_AGENT})
        return _client


@contextmanager
def open_sheet(url: str) -> Iterator[DatasetReader]:
    """The sheet's GeoTIFF, read by byte ranges through the raster allowlist. One tile's
    read is bounded: a few overview blocks, never the sheet."""
    opener = RasterOpener(
        _http(), url, ReadBudget(max_bytes=24 * 1024 * 1024, max_requests=96, max_seconds=20)
    )
    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_CACHEMAX=32 * 1024 * 1024),
        rasterio.open(url, opener=opener, driver="GTiff") as source,
    ):
        yield source


_tiles: OrderedDict[tuple[str, Neatline, int, int, int], bytes] = OrderedDict()
_tiles_lock = threading.Lock()
_slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)


def clear_cache() -> None:
    """Forget every rendered tile (tests)."""
    with _tiles_lock:
        _tiles.clear()


def render(
    sheet: str,
    neatline: Neatline,
    z: int,
    x: int,
    y: int,
) -> bytes:
    """One 256 px PNG tile of a sheet, transparent outside its neatline (the collar) and off
    the sheet altogether."""
    url = sheet_url(sheet)
    neatline = check_neatline(neatline)
    if not 0 <= z <= 22 or not 0 <= x < 2 ** (z + 1) or not 0 <= y < 2**z:
        raise NotFoundError("historical map tile", f"{z}/{x}/{y}")
    bounds = TMS.bounds(morecantile.Tile(x, y, z))
    ring = neatline_wgs84(neatline)
    west, south, east, north = ring_bounds(ring)
    if bounds.right <= west or bounds.left >= east or bounds.top <= south or bounds.bottom >= north:
        return transparent_tile()
    key = (sheet, neatline, z, x, y)
    with _tiles_lock:
        hit = _tiles.get(key)
        if hit is not None:
            _tiles.move_to_end(key)
            return hit
    if not _slots.acquire(timeout=SLOT_WAIT_S):
        raise RenderBusy
    try:
        with open_sheet(url) as source:
            if source.count < 3 or source.crs is None:
                raise ValueError("The sheet is not a georeferenced RGB image.")
            rgba = _warp(source, (bounds.left, bounds.bottom, bounds.right, bounds.top))
    finally:
        _slots.release()
    transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, TILE_PX, TILE_PX)
    inside = geometry_mask(
        [{"type": "Polygon", "coordinates": [list(ring)]}],
        out_shape=(TILE_PX, TILE_PX),
        transform=transform,
        invert=True,
    )
    rgba[3] = np.where(inside & (rgba[3] > 0), 255, 0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(np.moveaxis(rgba, 0, 2), mode="RGBA").save(output, format="PNG")
    png = output.getvalue()
    with _tiles_lock:
        _tiles[key] = png
        _tiles.move_to_end(key)
        while len(_tiles) > CACHE_TILES:
            _tiles.popitem(last=False)
    return png


def _warp(source: DatasetReader, tile: Neatline) -> np.ndarray:
    """The tile's pixels from the sheet, RGB plus where the sheet has any (band 4).

    Warped to WGS 84 at the sheet's own resolution and read at 256 px, so GDAL reads the
    overview matching the tile (a warped VRT's implicit overviews) instead of the full scan.
    The part of the tile off the sheet stays transparent."""
    left, bottom, right, top = tile
    rgba = np.zeros((4, TILE_PX, TILE_PX), dtype=np.uint8)
    with WarpedVRT(source, crs="EPSG:4326", resampling=Resampling.bilinear, add_alpha=True) as vrt:
        vb = vrt.bounds
        west, east = max(left, vb.left), min(right, vb.right)
        south, north = max(bottom, vb.bottom), min(top, vb.top)
        if west >= east or south >= north:
            return rgba
        px = TILE_PX / (right - left)
        py = TILE_PX / (top - bottom)
        x0, x1 = round((west - left) * px), round((east - left) * px)
        y0, y1 = round((top - north) * py), round((top - south) * py)
        if x1 <= x0 or y1 <= y0:
            return rgba
        window = window_from_bounds(west, south, east, north, vrt.transform)
        data = vrt.read(
            [1, 2, 3, vrt.count],
            window=window,
            out_shape=(4, y1 - y0, x1 - x0),
            resampling=Resampling.bilinear,
        )
        rgba[:, y0:y1, x0:x1] = data
    return rgba
