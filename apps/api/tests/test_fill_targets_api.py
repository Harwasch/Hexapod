"""`GET /sites/{id}/fill-targets`: the cells a re-scan left weak (`rescan-gate-v1`, A4).

The yard's two fixture epochs registered through `register` on one site: e1 (1 m cells,
weak in its north half) and its re-scan e2 (0.5 m cells, weak only in a north-east patch).
The targets are checked cell by cell against a brute-force reading of both grids.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.analysis import cell_grid, rescan_gate
from app.api.deps import _db
from app.main import create_app
from app.models import Capture, Epoch
from app.models.enums import CaptureKind
from app.services import confidence_maps
from app.storage import S3Storage, get_public_storage
from tests.confidence_fixtures import blob, descriptor, encode_grid, measured, register_run
from tests.test_registration_epochs import _echoed

FIRST = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
SECOND = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)


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


def _capture(db: Session, slug: str, captured: datetime) -> Capture:
    capture = Capture(
        slug=slug, name=slug.title(), kind=CaptureKind.VIDEO, metadata_={}, captured_at=captured
    )
    db.add(capture)
    db.commit()
    return capture


def _epoch(db: Session, capture: Capture) -> Epoch:
    db.expire_all()
    return db.query(Epoch).filter(Epoch.capture_id == capture.id).one()


def two_epochs(db: Session, storage: S3Storage, tmp_path: Path) -> tuple[Epoch, Epoch]:
    first = _capture(db, "first", FIRST)
    register_run(db, storage, tmp_path, first, measured("e1"), grid=blob("e1"))
    site_id = _epoch(db, first).site_id
    second = _capture(db, "second", SECOND)
    register_run(db, storage, tmp_path, second, measured("e2", _echoed(site_id)), grid=blob("e2"))
    e1, e2 = _epoch(db, first), _epoch(db, second)
    assert e1.site_id == e2.site_id and (e1.ordinal, e2.ordinal) == (1, 2)
    return e1, e2


def brute_force(level: int) -> dict[tuple[int, int], bool]:
    """Every cell of e2's plane at `level`: is it weak there, and weak in e1 there too?"""
    planes = {
        name: cell_grid.decode(descriptor(name), blob(name)).at(level) for name in ("e1", "e2")
    }

    def weak(name: str, i: int, j: int) -> bool:
        plane = planes[name]
        di, dj = i - plane.i0, j - plane.j0
        if not (0 <= di < plane.ni and 0 <= dj < plane.nj):
            return False
        score = int(plane.channels["score"][dj, di])
        unseen = bool(int(plane.channels["reasons"][dj, di]) & 16)
        return (0 < score < 128) or unseen

    b = planes["e2"]
    return {
        (i, j): weak("e2", i, j) and weak("e1", i, j)
        for j in range(b.j0, b.j0 + b.nj)
        for i in range(b.i0, b.i0 + b.ni)
    }


def test_the_re_scan_s_still_weak_cells_are_the_targets(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    e1, e2 = two_epochs(db, storage, tmp_path)

    response = api.get(f"/api/v1/sites/{e2.site_id}/fill-targets")

    assert response.status_code == 200, response.text
    read = response.json()
    assert (read["policy"], read["frame"], read["bitOrder"]) == (
        "rescan-gate-v1",
        "tileset",
        "little",
    )
    assert read["epochId"] == str(e2.id)  # the latest published, by default
    assert read["weakIn"] == [str(e1.id)] and read["comparedWith"] == [str(e1.id)]
    # Compared at the coarser of 1 m (e1) and 0.5 m (e2).
    assert (read["level"], read["cellM"]) == (2, 1.0)
    mask = rescan_gate.unpack(read["mask"], read["ni"], read["nj"])
    assert read["rowBytes"] == (read["ni"] + 7) // 8
    expected = brute_force(2)
    found = {
        (read["i0"] + di, read["j0"] + dj): bool(mask[dj, di])
        for dj in range(read["nj"])
        for di in range(read["ni"])
    }
    assert found == expected
    assert read["cells"] == sum(expected.values()) > 0
    # Only the north-east patch e2 left weak, and only where e1 was weak too.
    assert read["cells"] < descriptor("e1")["summary"]["weak"]

    same = api.get(f"/api/v1/sites/{e2.site_id}/fill-targets", params={"epochId": str(e2.id)})
    assert same.json() == read


def test_before_a_re_scan_there_is_nothing_to_fill(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first = _capture(db, "first", FIRST)
    register_run(db, storage, tmp_path, first, measured("e1"), grid=blob("e1"))
    e1 = _epoch(db, first)
    response = api.get(f"/api/v1/sites/{e1.site_id}/fill-targets")
    assert response.status_code == 409
    assert response.json()["code"] == "no-rescan-yet"

    # The first epoch of a re-scanned site is no re-scan either.
    second = _capture(db, "second", SECOND)
    register_run(
        db, storage, tmp_path, second, measured("e2", _echoed(e1.site_id)), grid=blob("e2")
    )
    response = api.get(f"/api/v1/sites/{e1.site_id}/fill-targets", params={"epochId": str(e1.id)})
    assert response.status_code == 409 and response.json()["code"] == "no-rescan-yet"


def test_an_unmeasured_epoch_has_no_targets_and_is_no_earlier_evidence(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A splat upload before or after a measured epoch: as B it is a 409 saying so; as
    an earlier epoch it is not compared with."""
    upload = _capture(db, "upload", FIRST)
    register_run(db, storage, tmp_path, upload, measured(None))
    site_id = _epoch(db, upload).site_id
    second = _capture(db, "second", SECOND)
    register_run(db, storage, tmp_path, second, measured("e2", _echoed(site_id)), grid=blob("e2"))

    response = api.get(f"/api/v1/sites/{site_id}/fill-targets")
    assert response.status_code == 409 and response.json()["code"] == "no-rescan-yet"
    response = api.get(
        f"/api/v1/sites/{site_id}/fill-targets", params={"epochId": str(_epoch(db, upload).id)}
    )
    assert response.status_code == 409 and response.json()["code"] == "confidence-not-measured"


def test_unknown_sites_and_epochs_are_404s(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    e1, _ = two_epochs(db, storage, tmp_path)
    unknown = "00000000-0000-4000-8000-000000000000"
    assert api.get(f"/api/v1/sites/{unknown}/fill-targets").status_code == 404
    response = api.get(f"/api/v1/sites/{e1.site_id}/fill-targets", params={"epochId": unknown})
    assert response.status_code == 404


def test_a_cell_fixed_by_the_re_scan_or_weak_only_in_it_is_not_a_target() -> None:
    """The gate's three conditions, one cell each, on 1 m planes."""
    bits = {"views": 1, "unseen": 16}
    a_score = np.array([[60, 60, 200, 0]], dtype=np.uint8)  # weak, weak, strong, nothing
    a_reasons = np.array([[bits["views"], bits["views"], 0, 0]], dtype=np.uint8)
    b_score = np.array([[60, 200, 60, 0]], dtype=np.uint8)
    b_reasons = np.array([[bits["views"], 0, bits["views"], bits["unseen"]]], dtype=np.uint8)
    a_blob, a_desc = encode_grid(2, 0, 0, a_score, a_reasons)
    b_blob, b_desc = encode_grid(2, 0, 0, b_score, b_reasons)
    a, b = cell_grid.decode(a_desc, a_blob), cell_grid.decode(b_desc, b_blob)
    result = rescan_gate.still_weak(b, [a])
    # Cell 0: weak in both -> target. Cell 1: fixed by B. Cell 2: weak only in B.
    # Cell 3: unseen in B, but A had no evidence there (not weak in A) -> not a target.
    assert result.mask.tolist() == [[True, False, False, False]]
    assert result.contributed == [0]
    assert rescan_gate.unpack(result.packed(), 4, 1).tolist() == result.mask.tolist()


def test_grids_at_different_levels_compare_at_the_coarser() -> None:
    """A 0.5 m epoch over a 2 m one: read at 2 m through its min-of-children pyramid, so a
    coarse cell with one weak child is weak."""
    fine_score = np.full((4, 4), 200, dtype=np.uint8)
    fine_reasons = np.zeros((4, 4), dtype=np.uint8)
    fine_score[0, 0] = 40
    fine_reasons[0, 0] = 1
    coarse_blob, coarse_desc = encode_grid(3, 0, 0, [[50]], [[2]])
    fine_blob, fine_desc = encode_grid(1, 0, 0, fine_score, fine_reasons)
    fine = cell_grid.decode(fine_desc, fine_blob)
    coarse = cell_grid.decode(coarse_desc, coarse_blob)
    result = rescan_gate.still_weak(fine, [coarse])
    assert (result.level, result.i0, result.j0, result.ni, result.nj) == (3, 0, 0, 1, 1)
    assert result.mask.tolist() == [[True]]


def test_only_published_epochs_are_earlier_evidence(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """An earlier epoch that is not published (rejected, or a future `superseded`) is not
    what the site shows, so it is not compared with, map or no map."""
    from sqlalchemy import text

    e1, e2 = two_epochs(db, storage, tmp_path)
    db.execute(text("UPDATE epochs SET status = 'rejected' WHERE id = :id"), {"id": e1.id})
    db.commit()
    response = api.get(f"/api/v1/sites/{e2.site_id}/fill-targets")
    assert response.status_code == 409 and response.json()["code"] == "no-rescan-yet"

