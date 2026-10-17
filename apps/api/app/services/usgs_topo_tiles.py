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
week by browsers, as public data). What a sheet is read for is kept in process too, shared by
every tile (`_SheetFile`): its size and ETag, the header and IFD blocks it is opened with, and
the image blocks neighbouring tiles share, so a tile costs only the ranges no tile has read yet.
The URL is built only from the fixed prefix and a validated sheet name, so the route cannot be
turned into a fetch of anything else.

Reads run on threads of the tiler's own, MAX_IN_FLIGHT at once (`render_tile`), never on the
request threadpool every other route shares. A tile waiting for a reader waits on the event loop,
holding no thread; past MAX_WAITING or SLOT_WAIT_S it is answered busy (503, Retry-After), which
the Time Lantern's layers wait out and ask again (apps/web/src/cesium/busyTiles.ts).
"""

from __future__ import annotations

import io
import math
import re
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import NamedTuple
from urllib.parse import quote

import anyio
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

from app.analysis.raster_io import RasterHttpFile, RasterOpener, ReadBudget
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
#: Sheets read at once, each on a thread of the tiler's own (`render_tile`): a read is a few
#: range requests to S3 and a warp, about a second.
MAX_IN_FLIGHT = 4
#: Tiles that may wait for a reader; the next is answered busy at once. Waiting holds no thread,
#: but a long queue only draws tiles the lens has already left.
MAX_WAITING = 12
#: How long a tile waits for a reader before it is answered busy (s).
SLOT_WAIT_S = 4.0
#: The Retry-After a busy tile is answered with (s).
RETRY_AFTER_S = 2
#: Sheet bytes kept in process, in RasterHttpFile's 64 KiB blocks: 32 MiB.
CACHE_BLOCKS = 512
#: Sheets whose size and ETag are kept.
CACHE_SHEETS = 64
#: How long a sheet's size and ETag are trusted without a probe (s), and so how long bytes read
#: from one version may be drawn after S3 has another (the collection's scans are not revised in
#: place; a new scan is a new sheet name). A range not yet read still sends If-Match and checks
#: the answer's size and ETag, so a changed sheet fails that one tile and is probed afresh.
VERSION_TTL_S = 3600.0
USER_AGENT = "LivingWorld-DatedImagery/1.0"

Neatline = tuple[float, float, float, float]


class RenderBusy(Exception):  # noqa: N818 - a condition, answered 503
    """Every reader is busy and the queue is full, or the wait ran out: the tile is not drawn
    now, and the answer says when to ask again."""


class Tile(NamedTuple):
    """A validated tile request: the sheet's URL, the neatline, the tile's bounds (degrees)."""

    sheet: str
    url: str
    neatline: Neatline
    z: int
    x: int
    y: int
    bounds: Neatline


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


Version = tuple[int, str | None, str | None]

_bytes_lock = threading.Lock()
#: Sheet URL -> (when it was learnt, its size, ETag and Last-Modified).
_versions: OrderedDict[str, tuple[float, Version]] = OrderedDict()
#: (sheet URL, version, first byte, last byte) -> the bytes.
_blocks: OrderedDict[tuple[str, Version, int, int], bytes] = OrderedDict()


def _known_version(url: str) -> Version | None:
    with _bytes_lock:
        known = _versions.get(url)
        if known is None or time.monotonic() - known[0] > VERSION_TTL_S:
            return None
        return known[1]


def _remember_version(url: str, version: Version) -> None:
    with _bytes_lock:
        _versions[url] = (time.monotonic(), version)
        _versions.move_to_end(url)
        while len(_versions) > CACHE_SHEETS:
            _versions.popitem(last=False)


def _forget_version(url: str) -> None:
    with _bytes_lock:
        _versions.pop(url, None)


class _SheetFile(RasterHttpFile):
    """A sheet by byte ranges -- RasterHttpFile's allowlist, budget, If-Match and version check
    on every answer -- sharing what it reads with every other tile of the sheet: the opening
    probe is skipped while the sheet's version is known, and each block is fetched once."""

    def __init__(
        self, client: httpx.Client, url: str, budget: ReadBudget, failures: list[OSError]
    ) -> None:
        # Read by the probe RasterHttpFile.__init__ makes, so set first.
        self._known = _known_version(url)
        self._failures = failures
        super().__init__(client, url, budget)

    def read(self, size: int = -1) -> bytes:
        # rasterio's read callback (_vsiopener.pyx, pyopener_read) does not handle an exception:
        # one raised here stays pending, and GDAL's next callback fails with a SystemError. So
        # the error is kept for open_sheet to raise, and GDAL sees a short read, an error to it.
        try:
            return super().read(size)
        except OSError as error:
            self._failures.append(error)
            return b""

    def _range(self, start: int, end: int, *, probe: bool = False) -> bytes:
        if probe:
            if self._known is not None:
                self.size, self.etag, self.last_modified = self._known
                return b""
            data = super()._range(start, end, probe=True)
            _remember_version(self.url, (self.size, self.etag, self.last_modified))
            return data
        key = (self.url, (self.size, self.etag, self.last_modified), start, end)
        with _bytes_lock:
            hit = _blocks.get(key)
            if hit is not None:
                _blocks.move_to_end(key)
                return hit
        try:
            data = super()._range(start, end)
        except OSError:
            # Changed (a 412 to If-Match, another size or ETag) or unreachable: probed afresh.
            _forget_version(self.url)
            raise
        with _bytes_lock:
            _blocks[key] = data
            _blocks.move_to_end(key)
            while len(_blocks) > CACHE_BLOCKS:
                _blocks.popitem(last=False)
        return data


class _SheetOpener(RasterOpener):
    """RasterOpener, opening the sheet as a _SheetFile. Every file it opens shares one known
    version, so the cross-file check RasterOpener makes holds by construction."""

    def __init__(self, client: httpx.Client, url: str, budget: ReadBudget) -> None:
        super().__init__(client, url, budget)
        #: What a read failed with, in its own words (rasterio would replace it).
        self.failures: list[OSError] = []

    def __call__(self, path: str, mode: str = "r") -> RasterHttpFile:
        if path != self.url or mode not in ("r", "rb"):
            raise OSError("Only the registered raster can be opened, in read-only mode.")
        file = _SheetFile(self.client, self.url, self.budget, self.failures)
        self.files.append(file)
        return file


@contextmanager
def open_sheet(url: str) -> Iterator[DatasetReader]:
    """The sheet's GeoTIFF, read by byte ranges through the raster allowlist. One tile's
    read is bounded: a few overview blocks, never the sheet. A read that failed raises its own
    error (an OSError), whatever GDAL made of the short read."""
    opener = _SheetOpener(
        _http(), url, ReadBudget(max_bytes=24 * 1024 * 1024, max_requests=96, max_seconds=20)
    )
    try:
        with (
            rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_CACHEMAX=32 * 1024 * 1024),
            rasterio.open(url, opener=opener, driver="GTiff") as source,
        ):
            yield source
    except Exception as error:
        if opener.failures:
            raise opener.failures[0] from error
        raise
    if opener.failures:
        raise opener.failures[0]


_tiles: OrderedDict[tuple[str, Neatline, int, int, int], bytes] = OrderedDict()
_tiles_lock = threading.Lock()
#: Who may read a sheet now (`render_tile`): MAX_IN_FLIGHT at once, the rest wait here.
_readers = anyio.Semaphore(MAX_IN_FLIGHT)
#: The readers' threads, apart from the threadpool every other route's work runs on.
_reader_threads = anyio.CapacityLimiter(MAX_IN_FLIGHT)


def clear_cache() -> None:
    """Forget every rendered tile and every sheet byte read (tests)."""
    with _tiles_lock:
        _tiles.clear()
    with _bytes_lock:
        _versions.clear()
        _blocks.clear()


def plan(sheet: str, neatline: Neatline, z: int, x: int, y: int) -> Tile:
    """A tile request checked -- the sheet's name, the neatline, the tile on the grid -- with
    no read and no projection: cheap enough for the event loop."""
    url = sheet_url(sheet)
    neatline = check_neatline(neatline)
    if not 0 <= z <= 22 or not 0 <= x < 2 ** (z + 1) or not 0 <= y < 2**z:
        raise NotFoundError("historical map tile", f"{z}/{x}/{y}")
    b = TMS.bounds(morecantile.Tile(x, y, z))
    return Tile(sheet, url, neatline, z, x, y, (b.left, b.bottom, b.right, b.top))


def cached_png(tile: Tile) -> bytes | None:
    """The tile, if this process has drawn it."""
    key = (tile.sheet, tile.neatline, tile.z, tile.x, tile.y)
    with _tiles_lock:
        hit = _tiles.get(key)
        if hit is not None:
            _tiles.move_to_end(key)
        return hit


def draw(tile: Tile) -> bytes:
    """One 256 px PNG tile of a sheet, transparent outside its neatline (the collar) and off
    the sheet altogether (without reading it). Blocks: run it on a reader's thread."""
    ring = neatline_wgs84(tile.neatline)
    west, south, east, north = ring_bounds(ring)
    left, bottom, right, top = tile.bounds
    if right <= west or left >= east or top <= south or bottom >= north:
        return transparent_tile()
    with open_sheet(tile.url) as source:
        if source.count < 3 or source.crs is None:
            raise ValueError("The sheet is not a georeferenced RGB image.")
        rgba = _warp(source, tile.bounds)
    transform = from_bounds(left, bottom, right, top, TILE_PX, TILE_PX)
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
    key = (tile.sheet, tile.neatline, tile.z, tile.x, tile.y)
    with _tiles_lock:
        _tiles[key] = png
        _tiles.move_to_end(key)
        while len(_tiles) > CACHE_TILES:
            _tiles.popitem(last=False)
    return png


def render(sheet: str, neatline: Neatline, z: int, x: int, y: int) -> bytes:
    """A tile, drawn on the calling thread (tests and scripts; the route uses `render_tile`)."""
    tile = plan(sheet, neatline, z, x, y)
    return cached_png(tile) or draw(tile)


async def render_tile(
    tile: Tile, *, gone: Callable[[], Awaitable[bool]] | None = None
) -> bytes | None:
    """A tile for the route: from the cache at once, else drawn on a reader's thread once one
    is free. RenderBusy when MAX_WAITING tiles already wait, or after SLOT_WAIT_S. None when
    `gone` says the client left while the tile waited (the lens moved on): nothing is read."""
    hit = cached_png(tile)
    if hit is not None:
        return hit
    if _readers.value == 0 and _readers.statistics().tasks_waiting >= MAX_WAITING:
        raise RenderBusy
    with anyio.move_on_after(SLOT_WAIT_S) as waited:
        await _readers.acquire()
    if waited.cancelled_caught:
        raise RenderBusy
    try:
        if gone is not None and await gone():
            return None
        # Another request may have drawn it while this one waited.
        return cached_png(tile) or await anyio.to_thread.run_sync(
            draw, tile, limiter=_reader_threads
        )
    finally:
        _readers.release()


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
