"""Tiles cut from USGS historical topographic sheets (app/services/usgs_topo_tiles.py).

The sheet is a GeoTIFF written here, as the collection publishes them: NAD27 polyconic, RGB,
tiled, with a collar around the map. The reader is swapped for one that opens it locally, or
for a fake S3 that answers byte ranges; the rest -- the URL guard, the neatline, the warp, the
collar cut, the readers and their queue -- is the real thing."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import anyio
import httpx
import morecantile
import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from pyproj import Transformer
from rasterio.io import DatasetReader
from rasterio.transform import from_bounds

from app.services import usgs_topo_tiles as topo
from app.services.errors import InvalidInputError, NotFoundError
from app.services.land_rasters import TMS

SHEET = "CA/CA_Los Olivos_292579_1959_24000_geo"
NEATLINE = (-120.125, 34.625, -120.0, 34.75)
POLYCONIC = (
    'PROJCS["unnamed",GEOGCS["NAD27",DATUM["North_American_Datum_1927",'
    'SPHEROID["Clarke 1866",6378206.4,294.978698213898]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4267"]],PROJECTION["Polyconic"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-120.0625],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'
)
#: The paper's colour: the collar, the map under it, the same here; only the neatline differs.
INK = (196, 120, 64)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def write_sheet(path: Path) -> Path:
    """A sheet 0.02° larger than its neatline on every side, as a scan with its collar is."""
    west, south, east, north = NEATLINE
    margin = 0.02
    to_sheet = Transformer.from_crs("EPSG:4267", POLYCONIC, always_xy=True)
    xs, ys = to_sheet.transform(
        [west - margin, east + margin, east + margin, west - margin],
        [south - margin, south - margin, north + margin, north + margin],
    )
    width, height = 512, 640
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs=POLYCONIC,
        transform=from_bounds(min(xs), min(ys), max(xs), max(ys), width, height),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        for band, value in enumerate(INK, start=1):
            dataset.write(np.full((height, width), value, dtype=np.uint8), band)
        dataset.build_overviews([2, 4], rasterio.enums.Resampling.average)
    return path


@pytest.fixture
def sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The sheet, opened locally in place of its USGS URL."""
    path = write_sheet(tmp_path / "sheet.tif")

    @contextmanager
    def local(url: str) -> Iterator[DatasetReader]:
        assert url == f"{topo.PREFIX}CA/CA_Los%20Olivos_292579_1959_24000_geo.tif"
        with rasterio.open(path) as source:
            yield source

    monkeypatch.setattr(topo, "open_sheet", local)
    topo.clear_cache()
    return path


def rgba(png: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png))
    assert image.size == (256, 256)
    return np.asarray(image.convert("RGBA"))


@pytest.mark.parametrize(
    "name",
    [
        "CA/CA_Los Olivos_292579_1959_24000_geo.tif",  # a suffix
        "CA/../CA_Los Olivos_292579_1959_24000_geo",
        "CA/sub/CA_Los Olivos_292579_1959_24000_geo",
        "https://example.com/CA/CA_x_292579_1959_24000_geo",
        "CA/NV_Los Olivos_292579_1959_24000_geo",  # another state's sheet in this folder
        "CA/CA_Los Olivos_292579_1959_24000",
        "CA/CA_Los Olivos..x_292579_1959_24000_geo",
    ],
)
def test_only_collection_sheet_names_are_read(name: str) -> None:
    with pytest.raises(InvalidInputError):
        topo.sheet_url(name)


def test_the_url_is_built_from_the_fixed_prefix() -> None:
    assert topo.sheet_url(SHEET) == (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/GeoTIFF/"
        "CA/CA_Los%20Olivos_292579_1959_24000_geo.tif"
    )


def test_neatline_must_be_one_sheet() -> None:
    with pytest.raises(InvalidInputError):
        topo.check_neatline((-121.0, 34.0, -118.0, 34.5))
    with pytest.raises(InvalidInputError):
        topo.check_neatline((-120.0, 34.75, -120.125, 34.625))
    with pytest.raises(InvalidInputError):
        topo.check_neatline((float("nan"), 34.0, -120.0, 34.5))


def test_neatline_snaps_to_the_graticule_only_when_close() -> None:
    assert topo.snap_neatline((-120.1251, 34.6248, -120.0003, 34.7502)) == (NEATLINE, True)
    off = (-120.11, 34.63, -120.0, 34.75)
    assert topo.snap_neatline(off) == (off, False)


def test_a_tile_off_the_sheet_is_transparent_without_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(url: str) -> None:
        raise AssertionError("read a sheet for a tile off it")

    monkeypatch.setattr(topo, "open_sheet", never)
    tile = TMS.tile(-118.0, 34.7, 12)
    png = topo.render(SHEET, NEATLINE, 12, tile.x, tile.y)
    assert rgba(png)[..., 3].max() == 0


def test_a_tile_on_the_edge_keeps_the_map_and_cuts_the_collar(sheet: Path) -> None:
    tile = TMS.tile(-120.005, 34.70, 13)
    pixels = rgba(topo.render(SHEET, NEATLINE, 13, tile.x, tile.y))
    bounds = TMS.bounds(morecantile.Tile(tile.x, tile.y, 13))
    # The neatline's east edge, shifted from NAD27 to WGS 84 (some 90 m west here).
    edge = topo.ring_bounds(topo.neatline_wgs84(NEATLINE))[2]
    column = round((edge - bounds.left) / (bounds.right - bounds.left) * 256)
    assert 120 < column < 200
    inside, collar = pixels[:, column - 20], pixels[:, column + 20]
    assert (inside[:, 3] == 255).all()
    assert tuple(inside[128, :3]) == pytest.approx(INK, abs=2)
    # The scan goes on past the neatline (the collar), but none of it is drawn.
    assert (collar[:, 3] == 0).all()


def test_tile_coordinates_must_exist() -> None:
    with pytest.raises(NotFoundError):
        topo.render(SHEET, NEATLINE, 3, 999, 0)


def test_route_serves_a_public_cached_png(client: TestClient, sheet: Path) -> None:
    tile = TMS.tile(-120.05, 34.70, 14)
    response = client.get(
        f"/api/v1/imagery/usgs-topo/tiles/14/{tile.x}/{tile.y}.png",
        params={"sheet": SHEET, "w": -120.125, "s": 34.625, "e": -120.0, "n": 34.75},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "public, max-age=604800"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert rgba(response.content)[..., 3].min() == 255


class FakeS3:
    """The sheet's bucket: byte ranges with an ETag, honouring If-Match, every range noted."""

    def __init__(self, path: Path) -> None:
        self.data = path.read_bytes()
        self.etag = '"v1"'
        self.ranges: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == topo.sheet_url(SHEET)
        self.ranges.append(request.headers["Range"])
        if request.headers.get("If-Match", self.etag) != self.etag:
            return httpx.Response(412)
        start, end = (int(v) for v in request.headers["Range"].removeprefix("bytes=").split("-"))
        end = min(end, len(self.data) - 1)
        return httpx.Response(
            206,
            stream=httpx.ByteStream(self.data[start : end + 1]),
            headers={"Content-Range": f"bytes {start}-{end}/{len(self.data)}", "ETag": self.etag},
        )


@pytest.fixture
def s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeS3]:
    bucket = FakeS3(write_sheet(tmp_path / "sheet.tif"))
    monkeypatch.setattr(topo, "_http", lambda: httpx.Client(transport=httpx.MockTransport(bucket)))
    topo.clear_cache()
    yield bucket
    topo.clear_cache()


def test_tiles_of_a_sheet_share_its_version_and_bytes(s3: FakeS3) -> None:
    first = TMS.tile(-120.05, 34.70, 14)
    assert rgba(topo.render(SHEET, NEATLINE, 14, first.x, first.y))[..., 3].min() == 255
    opened = list(s3.ranges)
    assert opened[0] == "bytes=0-0"  # the probe: size and ETag
    s3.ranges.clear()
    # The tile beside it: no probe, no header read again, nothing fetched twice.
    second = TMS.tile(-120.05, 34.70, 14)
    second = morecantile.Tile(second.x + 1, second.y, 14)
    assert rgba(topo.render(SHEET, NEATLINE, 14, second.x, second.y))[..., 3].min() == 255
    assert "bytes=0-0" not in s3.ranges
    assert not set(s3.ranges) & set(opened)


def test_a_sheet_that_changed_fails_one_tile_and_is_probed_afresh(s3: FakeS3) -> None:
    tile = TMS.tile(-120.05, 34.70, 14)
    topo.render(SHEET, NEATLINE, 14, tile.x, tile.y)
    s3.etag = '"v2"'
    s3.ranges.clear()
    # Bytes already read stand until the version is asked again (VERSION_TTL_S); a tile that
    # needs a range not yet read asks with If-Match, and S3 refuses the old version.
    topo._blocks.clear()
    with pytest.raises(OSError):
        topo.render(SHEET, NEATLINE, 14, tile.x + 1, tile.y)
    # Probed afresh: the new version is known, and the tile draws from it.
    assert "bytes=0-0" in s3.ranges
    assert topo._known_version(topo.sheet_url(SHEET)) == (len(s3.data), '"v2"', None)
    assert rgba(topo.render(SHEET, NEATLINE, 14, tile.x + 1, tile.y))[..., 3].min() == 255


@pytest.mark.anyio
async def test_a_full_queue_is_answered_busy_at_once_and_holds_no_shared_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    def slow(tile: topo.Tile) -> bytes:
        release.wait(10)
        return b"png"

    monkeypatch.setattr(topo, "draw", slow)
    topo.clear_cache()
    tiles = [
        topo.plan(SHEET, NEATLINE, 14, x, 0)
        for x in range(topo.MAX_IN_FLIGHT + topo.MAX_WAITING + 1)
    ]
    drawn: list[bytes | None] = []

    async def ask(tile: topo.Tile) -> None:
        drawn.append(await topo.render_tile(tile))

    async with anyio.create_task_group() as group:
        for tile in tiles[:-1]:
            group.start_soon(ask, tile)
        with anyio.fail_after(5):
            while topo._readers.statistics().tasks_waiting < topo.MAX_WAITING:
                await anyio.sleep(0.01)
        # Every reader is reading and the queue is full, yet the threadpool every other route
        # runs on is untouched: the readers have threads of their own, the queue none.
        assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
        started = time.monotonic()
        with pytest.raises(topo.RenderBusy):
            await topo.render_tile(tiles[-1])
        assert time.monotonic() - started < 0.5
        release.set()
    assert drawn == [b"png"] * (len(tiles) - 1)


@pytest.mark.anyio
async def test_a_tile_that_waits_too_long_is_answered_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(topo, "SLOT_WAIT_S", 0.2)
    monkeypatch.setattr(topo, "draw", lambda tile: pytest.fail("drew while every reader was busy"))
    topo.clear_cache()
    for _ in range(topo.MAX_IN_FLIGHT):
        await topo._readers.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(topo.RenderBusy):
            await topo.render_tile(topo.plan(SHEET, NEATLINE, 14, 1, 1))
        assert 0.15 < time.monotonic() - started < 1.0
    finally:
        for _ in range(topo.MAX_IN_FLIGHT):
            topo._readers.release()


@pytest.mark.anyio
async def test_a_tile_whose_client_left_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(topo, "draw", lambda tile: pytest.fail("read for a client that left"))
    topo.clear_cache()

    async def gone() -> bool:
        return True

    assert await topo.render_tile(topo.plan(SHEET, NEATLINE, 14, 1, 1), gone=gone) is None
    assert topo._readers.value == topo.MAX_IN_FLIGHT


def test_route_answers_busy_at_once_when_every_reader_is_taken(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(topo, "MAX_WAITING", 0)
    topo.clear_cache()
    # The readers are taken on the app's own event loop, as requests in flight would hold them.
    portal = client.portal
    assert portal is not None
    for _ in range(topo.MAX_IN_FLIGHT):
        portal.call(topo._readers.acquire)
    try:
        started = time.monotonic()
        response = client.get(
            "/api/v1/imagery/usgs-topo/tiles/14/100/100.png",
            params={"sheet": SHEET, "w": -120.125, "s": 34.625, "e": -120.0, "n": 34.75},
        )
        elapsed = time.monotonic() - started
    finally:
        for _ in range(topo.MAX_IN_FLIGHT):
            portal.call(topo._readers.release)
    assert response.status_code == 503
    assert response.headers["retry-after"] == str(topo.RETRY_AFTER_S)
    assert response.headers["cache-control"] == "no-store"
    assert elapsed < 1.0


def test_route_refuses_another_host(client: TestClient) -> None:
    response = client.get(
        "/api/v1/imagery/usgs-topo/tiles/14/1/1.png",
        params={
            "sheet": "https://example.com/x",
            "w": -120.125,
            "s": 34.625,
            "e": -120.0,
            "n": 34.75,
        },
    )
    assert response.status_code == 422
