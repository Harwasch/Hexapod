"""Tiles cut from USGS historical topographic sheets (app/services/usgs_topo_tiles.py).

The sheet is a GeoTIFF written here, as the collection publishes them: NAD27 polyconic, RGB,
tiled, with a collar around the map. The reader is swapped for one that opens it locally; the
rest -- the URL guard, the neatline, the warp and the collar cut -- is the real thing."""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
def sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A sheet 0.02° larger than its neatline on every side, as a scan with its collar is."""
    west, south, east, north = NEATLINE
    margin = 0.02
    to_sheet = Transformer.from_crs("EPSG:4267", POLYCONIC, always_xy=True)
    xs, ys = to_sheet.transform(
        [west - margin, east + margin, east + margin, west - margin],
        [south - margin, south - margin, north + margin, north + margin],
    )
    width, height = 512, 640
    path = tmp_path / "sheet.tif"
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
