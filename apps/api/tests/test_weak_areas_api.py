"""`POST /sites/{id}/epochs/{id}/weak-areas`: an epoch's weak cells as re-scan zones (A4).

Registered through `register` as the worker registers a run, with its grid in the (moto)
bucket beside the tiles, then asked for over HTTP.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import shape
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.main import create_app
from app.models import Capture, Epoch
from app.models.enums import CaptureKind
from app.services import confidence_maps
from app.storage import S3Storage, get_public_storage
from tests.confidence_fixtures import blob, encode_grid, measured, register_run
from tests.test_asset_scale import LAT, LON
from tests.test_weak_areas import BITS


@pytest.fixture
def api(db: Session, storage: S3Storage) -> Iterator[TestClient]:
    app = create_app()

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_public_storage] = lambda: storage
    confidence_maps.GRIDS.clear()
    with TestClient(app) as client:
        yield client


def _capture(db: Session, slug: str = "yard") -> Capture:
    capture = Capture(slug=slug, name=slug.title(), kind=CaptureKind.VIDEO, metadata_={})
    db.add(capture)
    db.commit()
    return capture


def _epoch(db: Session, capture: Capture) -> Epoch:
    db.expire_all()
    epoch = db.query(Epoch).filter(Epoch.capture_id == capture.id).one()
    return epoch


def ask(api: TestClient, epoch: Epoch, **body: Any) -> Any:
    return api.post(f"/api/v1/sites/{epoch.site_id}/epochs/{epoch.id}/weak-areas", json=body)


def test_a_measured_epoch_s_weak_areas_are_zones_on_its_site(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The yard's first scan (fixture e1): weak in its north half, a few cells unseen."""
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"), grid=blob("e1"))
    epoch = _epoch(db, capture)

    response = ask(api, epoch)

    assert response.status_code == 200, response.text
    read = response.json()
    assert (read["policy"], read["cellM"], read["level"]) == ("weak-v1", 1.0, 2)
    assert (read["weakCells"], read["totalCells"]) == (656, 1312)
    assert read["truncatedByBudget"] is False
    assert "rule of thumb" in read["heuristic"]
    assert 1 <= len(read["areas"]) <= 3
    areas = [shape(area["footprint"]) for area in read["areas"]]
    assert [a["areaM2"] for a in read["areas"]] == sorted(
        (a["areaM2"] for a in read["areas"]), reverse=True
    )
    for area, polygon in zip(read["areas"], areas, strict=True):
        assert polygon.is_valid and area["cells"] > 0
        assert area["pattern"] in ("nadir", "oblique", "orbit")
        assert area["hint"] and "rule of thumb" in area["hint"]
        # Within a few hundred metres of the site's origin: placed in its frame.
        assert abs(polygon.centroid.x - LON) < 0.01 and abs(polygon.centroid.y - LAT) < 0.01
    # The one area is the north half (41 m x 16 m, every weak cell), weak for views or
    # spread, with columns up to 4.75 m tall but 41 m across: wider than an orbit circles
    # (ORBIT_MAX_DIAMETER_M, 40 m), so two crossing obliques, by the heuristic.
    [biggest] = read["areas"]
    assert (biggest["cells"], biggest["areaM2"]) == (656, 656.0)
    assert (biggest["reliefM"], biggest["diameterM"]) == (4.75, 41.0)
    assert biggest["pattern"] == "oblique" and biggest["gimbalPitchDeg"] == -45.0
    reasons = biggest["reasons"]
    assert reasons["views"] + reasons["spread"] + reasons["unseen"] == biggest["cells"]
    assert areas[0].centroid.y > LAT  # north of the origin

    # Asked again: the grid comes from the cache, and the answer is the same.
    hits = confidence_maps.GRIDS.hits
    assert ask(api, epoch).json() == read
    assert confidence_maps.GRIDS.hits == hits + 1


def test_a_budget_truncates_and_says_so(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"), grid=blob("e1"))
    epoch = _epoch(db, capture)

    read = ask(api, epoch, budgetM2=50, maxAreas=2, focus=[LON, LAT + 0.0002]).json()

    assert read["truncatedByBudget"] is True
    assert sum(area["areaM2"] for area in read["areas"]) <= 50 + 1.0
    assert ask(api, epoch, maxAreas=9).status_code == 422
    assert ask(api, epoch, budgetM2=-1).status_code == 422


def test_an_unmeasured_epoch_is_a_409_that_says_so(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured(None))
    epoch = _epoch(db, capture)

    response = ask(api, epoch)

    assert response.status_code == 409
    assert response.json()["code"] == "confidence-not-measured"
    assert "splat upload" in response.json()["detail"]


def test_an_unknown_epoch_or_site_is_a_404(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, second = _capture(db, "first"), _capture(db, "second")
    register_run(db, storage, tmp_path, first, measured("e1"), grid=blob("e1"))
    register_run(db, storage, tmp_path, second, measured("e1"), grid=blob("e1"))
    one, other = _epoch(db, first), _epoch(db, second)
    assert one.site_id != other.site_id
    unknown = "00000000-0000-4000-8000-000000000000"
    assert (
        api.post(f"/api/v1/sites/{one.site_id}/epochs/{unknown}/weak-areas", json={}).status_code
        == 404
    )
    assert (
        api.post(f"/api/v1/sites/{unknown}/epochs/{one.id}/weak-areas", json={}).status_code == 404
    )
    # Another site's epoch is not this site's.
    assert (
        api.post(f"/api/v1/sites/{one.site_id}/epochs/{other.id}/weak-areas", json={}).status_code
        == 404
    )


def test_an_unaligned_phone_epoch_is_weak_everywhere_for_its_alignment(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A phone epoch placed by GPS alone (+/-10 m) on 1 m cells: every score capped to 6,
    every scored cell carries the alignment bit. One zone, flown nadir to overlap the
    reference area, so the next capture can be aligned visually."""
    score = np.full((20, 30), 6, dtype=np.uint8)
    reasons = np.full((20, 30), BITS["alignment"], dtype=np.uint8)
    reasons[:, :10] |= BITS["views"]
    data, grid = encode_grid(
        2,
        -15,
        -10,
        score,
        reasons,
        cap={"uncertaintyM": 10.0, "priorEpochs": 1, "factor": 0.05, "source": "georef.json"},
    )
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured(None) | {"confidence": grid}, grid=data)
    epoch = _epoch(db, capture)

    read = ask(api, epoch).json()

    assert (read["weakCells"], read["totalCells"]) == (600, 600)
    [area] = read["areas"]
    assert area["cells"] == 600 and area["areaM2"] == pytest.approx(600.0)
    assert area["pattern"] == "nadir" and "reference area" in area["hint"]
    assert area["reasons"]["alignment"] == 600
    listed = api.get(f"/api/v1/sites/{epoch.site_id}/epochs").json()[0]["confidence"]
    assert listed["alignmentCap"] == {
        "uncertaintyM": 10.0,
        "factor": 0.05,
        "priorEpochs": 1,
        "source": "georef.json",
    }


def test_a_tall_weak_patch_with_a_metric_capture_says_how_to_fly_it(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    score = np.full((40, 40), 220, dtype=np.uint8)
    reasons = np.zeros((40, 40), dtype=np.uint8)
    relief = np.zeros((40, 40), dtype=np.uint8)
    score[5:13, 5:13] = 60
    reasons[5:13, 5:13] = BITS["spread"]
    relief[5:13, 5:13] = 20  # 5 m
    score[25:30, 20:36] = 90
    reasons[25:30, 20:36] = BITS["gsd"]
    data, grid = encode_grid(2, -20, -20, score, reasons, relief, median_gsd_mm=8.0)
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured(None) | {"confidence": grid}, grid=data)

    read = ask(api, _epoch(db, capture)).json()

    by_pattern = {area["pattern"]: area for area in read["areas"]}
    assert set(by_pattern) == {"orbit", "nadir"}
    assert by_pattern["orbit"]["reliefM"] == 5.0 and by_pattern["orbit"]["gimbalPitchDeg"] == -30
    assert by_pattern["nadir"]["targetGsdM"] == pytest.approx(0.008)
    assert by_pattern["nadir"]["areaM2"] == pytest.approx(80.0)


def test_a_grid_that_cannot_be_read_is_a_409_not_a_500(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"))  # no file beside the tiles
    epoch = _epoch(db, capture)
    response = ask(api, epoch)
    assert response.status_code == 409 and response.json()["code"] == "confidence-unreadable"

    raw = bytearray(gzip.decompress(blob("e1")))
    raw[100] ^= 1
    for name, data, words in (
        ("damaged", gzip.compress(bytes(raw)), "sha256"),
        ("foreign", blob("e2"), "larger than"),
    ):
        other = _capture(db, name)
        register_run(db, storage, tmp_path, other, measured("e1"), grid=data)
        response = ask(api, _epoch(db, other))
        assert response.status_code == 409 and response.json()["code"] == "confidence-unreadable"
        assert words in response.json()["detail"]


def test_a_grid_too_large_to_be_one_is_not_read(
    api: TestClient,
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(confidence_maps, "MAX_GRID_BYTES", 1000)
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"), grid=blob("e1"))
    response = ask(api, _epoch(db, capture))
    assert response.status_code == 409 and "over 1000" in response.json()["detail"]


def test_the_cache_is_keyed_by_url_and_planes() -> None:
    """A rewritten grid at the same URL (one bucket) is a new entry, not a stale hit; and
    the cache never holds more than its bytes."""
    cache = confidence_maps.GridCache(max_bytes=10_000)
    from app.analysis import cell_grid
    from tests.confidence_fixtures import descriptor

    e1 = cell_grid.decode(descriptor("e1"), blob("e1"))
    cache.put(("u", "a"), e1)
    assert cache.get(("u", "a")) is e1 and cache.get(("u", "b")) is None
    e2 = cell_grid.decode(descriptor("e2"), blob("e2"))
    cache.put(("v", "b"), e2)  # e2 is ~20 kB: over the bound, not kept
    assert cache.get(("v", "b")) is None
    cache.put(("w", "a"), e1)
    cache.put(("x", "a"), e1)
    assert cache.get(("u", "a")) is None  # the least recently used went


def test_a_rejected_epoch_shows_nothing_and_says_so(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"), grid=blob("e1"))
    epoch = _epoch(db, capture)
    rejected = Epoch(
        site_id=epoch.site_id,
        ordinal=2,
        observed_at=epoch.observed_at,
        observed_at_source="created",
        footprint=epoch.footprint,
        platform="unknown",
        status="rejected",
        join_method="spatial",
    )
    db.add(rejected)
    db.commit()
    response = ask(api, rejected)
    assert response.status_code == 409 and response.json()["code"] == "confidence-not-measured"
    assert "shows no tiles (rejected)" in response.json()["detail"]


def test_weak_areas_take_the_write_token(db: Session, storage: S3Storage, tmp_path: Path) -> None:
    """Every POST is gated (test_write_token)."""
    from app.config import Settings

    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"), grid=blob("e1"))
    epoch = _epoch(db, capture)
    app = create_app(Settings(api_write_token="a4-token"))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_public_storage] = lambda: storage
    with TestClient(app) as client:
        path = f"/api/v1/sites/{epoch.site_id}/epochs/{epoch.id}/weak-areas"
        assert client.post(path, json={}).status_code == 401
        allowed = client.post(path, json={}, headers={"Authorization": "Bearer a4-token"})
        assert allowed.status_code == 200


def test_a_grid_rewritten_at_the_same_url_is_read_again(storage: S3Storage) -> None:
    """One bucket rewrites a run's keys in place (a Refine of the same job): the same URL
    with another descriptor is another grid, never the cached one."""
    from app.models import EpochMap
    from tests.confidence_fixtures import descriptor

    confidence_maps.GRIDS.clear()
    key = "runs/refined/package/splat/confidence.bin"
    url = storage.public_url(key)

    def row(epoch: str) -> EpochMap:
        return EpochMap(url=url, descriptor=descriptor(epoch), summary={})

    storage.put_object(key, blob("e1"), "application/octet-stream")
    hits = confidence_maps.GRIDS.hits
    assert confidence_maps.load_grid(storage, row("e1")).finest.level == 2
    storage.put_object(key, blob("e2"), "application/octet-stream")
    assert confidence_maps.load_grid(storage, row("e2")).finest.level == 1
    assert confidence_maps.load_grid(storage, row("e2")).finest.level == 1
    assert confidence_maps.GRIDS.hits == hits + 1


def test_a_confidence_grid_without_its_channels_is_a_409_not_a_500(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Bytes and sha256 intact, but `score` renamed: a grid this API cannot read weak cells
    from, refused when it is read, not when the weak mask asks for the channel."""
    capture = _capture(db)
    document = measured("e1")
    document["confidence"]["channels"][0]["name"] = "quality"
    register_run(db, storage, tmp_path, capture, document, grid=blob("e1"))
    response = ask(api, _epoch(db, capture))
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "confidence-unreadable"
    assert "score" in response.json()["detail"]


def test_weak_areas_stay_inside_the_site_boundary(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The yard's weak north half, on a site whose boundary takes in only its western half
    and has a no-go hole: every zone is inside the boundary and outside the hole."""
    from geoalchemy2.shape import from_shape
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import box

    from app.analysis.weak_areas import site_projection
    from app.models import Site, SiteFrame

    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"), grid=blob("e1"))
    epoch = _epoch(db, capture)
    frame = db.get(SiteFrame, epoch.site_id)
    assert frame is not None
    projection = site_projection(frame.lat, frame.lon, frame.height)

    def globe(polygon: ShapelyPolygon) -> list[tuple[float, float]]:
        xs, ys = polygon.exterior.coords.xy
        lon, lat = projection.to_geographic(np.asarray(xs), np.asarray(ys))
        return list(zip(lon.tolist(), lat.tolist(), strict=True))

    hole = box(2, 20, 6, 24)
    boundary = ShapelyPolygon(globe(box(-10, -10, 15, 50)), [globe(hole)])
    site = db.get(Site, epoch.site_id)
    assert site is not None
    site.boundary = from_shape(boundary, srid=4326)
    db.commit()

    read = ask(api, epoch).json()
    assert read["areas"], read
    for area in read["areas"]:
        polygon = shape(area["footprint"])
        assert boundary.buffer(1e-9).covers(polygon)
        assert polygon.intersection(ShapelyPolygon(globe(hole))).area < 1e-14
        assert area["areaM2"] < 640.0
