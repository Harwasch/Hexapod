"""`GET /sites/{id}/fill-targets`: the cells a re-scan left weak (`rescan-gate-v1`, A4).

The yard's two fixture epochs registered through `register` on one site: e1 (1 m cells,
weak in its north half) and its re-scan e2 (0.5 m cells, weak only in a north-east patch).
The targets are e2's own finest cells, checked cell by cell against a brute-force reading of
both grids.
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


def brute_force() -> dict[tuple[int, int], bool]:
    """Every cell of e2's finest plane: is it weak there, and weak in the e1 cell that
    holds it? e1 is the coarser (1 m against 0.5 m), so each e2 cell sits in one e1 cell."""
    e1 = cell_grid.decode(descriptor("e1"), blob("e1")).finest
    e2 = cell_grid.decode(descriptor("e2"), blob("e2")).finest
    shift = e1.level - e2.level
    assert shift > 0

    def weak(plane: cell_grid.Plane, i: int, j: int) -> bool:
        di, dj = i - plane.i0, j - plane.j0
        if not (0 <= di < plane.ni and 0 <= dj < plane.nj):
            return False
        score = int(plane.channels["score"][dj, di])
        unseen = bool(int(plane.channels["reasons"][dj, di]) & 16)
        return (0 < score < 128) or unseen

    return {
        (i, j): weak(e2, i, j) and weak(e1, i >> shift, j >> shift)
        for j in range(e2.j0, e2.j0 + e2.nj)
        for i in range(e2.i0, e2.i0 + e2.ni)
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
    # At e2's own 0.5 m cells, not e1's 1 m: a fill touches only what the re-scan saw weak.
    assert (read["level"], read["cellM"]) == (1, 0.5)
    e2_finest = descriptor("e2")["levels"][0]
    assert (read["i0"], read["j0"], read["ni"], read["nj"]) == tuple(
        e2_finest[k] for k in ("i0", "j0", "ni", "nj")
    )
    mask = rescan_gate.unpack(read["mask"], read["ni"], read["nj"])
    assert read["rowBytes"] == (read["ni"] + 7) // 8
    expected = brute_force()
    found = {
        (read["i0"] + di, read["j0"] + dj): bool(mask[dj, di])
        for dj in range(read["nj"])
        for di in range(read["ni"])
    }
    assert found == expected
    assert read["cells"] == sum(expected.values()) > 0
    # e2's weak north-east patch lies wholly in e1's weak north half: every cell e2 left
    # weak is a target, and not one it saw well.
    assert read["cells"] == descriptor("e2")["summary"]["weak"]
    # Each target says which earlier epoch's weak cell it re-scanned: here always e1.
    assert rescan_gate.unpack_satisfied(read["satisfiedBy"], read["cells"]) == [0] * read["cells"]

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


def _grid(level: int, score: object, reasons: object) -> cell_grid.Grid:
    data, desc = encode_grid(level, 0, 0, score, reasons)
    return cell_grid.decode(desc, data)


def test_a_finer_re_scan_opens_only_its_own_weak_cells() -> None:
    """A 0.5 m re-scan of one weak 2 m cell: fifteen of its sixteen cells now strong. Only
    the one still weak is a target; the gate never widens a weak child to its parent."""
    fine_score = np.full((4, 4), 200, dtype=np.uint8)
    fine_reasons = np.zeros((4, 4), dtype=np.uint8)
    fine_score[1, 2] = 40
    fine_reasons[1, 2] = 1
    fine = _grid(1, fine_score, fine_reasons)
    coarse = _grid(3, [[50]], [[2]])
    result = rescan_gate.still_weak(fine, [coarse])
    assert (result.level, result.i0, result.j0, result.ni, result.nj) == (1, 0, 0, 4, 4)
    expected = np.zeros((4, 4), dtype=bool)
    expected[1, 2] = True
    assert result.mask.tolist() == expected.tolist()
    assert result.cells == 1 and result.contributed == [0]
    assert rescan_gate.unpack_satisfied(result.satisfied_packed(), 1) == [0]


def test_a_coarser_re_scan_needs_the_whole_cell_weak_before() -> None:
    """A 2 m re-scan over a 0.5 m epoch: its weak cell re-scanned weak cells only when no
    part the earlier epoch saw in it was strong (cells it had no evidence for do not count
    either way)."""
    coarse = _grid(3, [[60]], [[1]])
    partly = np.full((4, 4), 40, dtype=np.uint8)
    partly[3, 3] = 220  # one quarter-metre strip seen well
    assert rescan_gate.still_weak(coarse, [_grid(1, partly, np.ones((4, 4)))]).cells == 0
    wholly = np.full((4, 4), 40, dtype=np.uint8)
    wholly[0, :] = 0  # no evidence there: neither strong nor weak
    result = rescan_gate.still_weak(coarse, [_grid(1, wholly, np.ones((4, 4)))])
    assert result.mask.tolist() == [[True]] and result.level == 3


def test_the_latest_epoch_to_cover_a_cell_is_the_one_re_scanned() -> None:
    """An earlier epoch counts for a cell only when its footprint covers that cell, and the
    newest that does is what the re-scan is a re-scan of; each target records it."""
    views = 1
    oldest = _grid(2, [[60, 60, 60]], [[views, views, views]])
    # Newer: saw cell 0 well, never reached cell 1, saw cell 2 weak.
    newer = _grid(2, [[220, 0, 70]], [[0, 0, views]])
    b = _grid(2, [[50, 50, 50]], [[views, views, views]])
    result = rescan_gate.still_weak(b, [newer, oldest])  # newest first
    # Cell 0: the newer epoch fixed it, so B's weakness there is new, not a re-scan's.
    # Cell 1: the newer epoch's footprint misses it, so the oldest speaks for it.
    assert result.mask.tolist() == [[False, True, True]]
    assert rescan_gate.unpack_satisfied(result.satisfied_packed(), result.cells) == [1, 0]
    assert result.contributed == [0, 1]


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


def test_fill_targets_take_the_write_token(db: Session, storage: S3Storage, tmp_path: Path) -> None:
    """Each call may decode a grid per earlier epoch; its one consumer, the fill workflow,
    holds the token."""
    from app.config import Settings

    _, e2 = two_epochs(db, storage, tmp_path)
    app = create_app(Settings(api_write_token="a4-token"))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_public_storage] = lambda: storage
    with TestClient(app) as client:
        path = f"/api/v1/sites/{e2.site_id}/fill-targets"
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer a4-token"}).status_code == 200


def test_at_most_eight_earlier_epochs_are_compared_and_it_says_so(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    from datetime import timedelta

    first = _capture(db, "first", FIRST)
    register_run(db, storage, tmp_path, first, measured("e1"), grid=blob("e1"))
    site_id = _epoch(db, first).site_id
    ids = [_epoch(db, first).id]
    for index in range(1, 10):
        capture = _capture(db, f"again-{index}", FIRST + timedelta(days=index))
        register_run(
            db, storage, tmp_path, capture, measured("e1", _echoed(site_id)), grid=blob("e1")
        )
        ids.append(_epoch(db, capture).id)
    latest = ids[-1]
    read = api.get(f"/api/v1/sites/{site_id}/fill-targets").json()
    assert read["epochId"] == str(latest)
    assert read["capped"] is True
    # The eight most recent before it, newest first.
    assert read["comparedWith"] == [str(i) for i in reversed(ids[1:9])]
    fewer = api.get(f"/api/v1/sites/{site_id}/fill-targets", params={"epochId": str(ids[5])}).json()
    assert fewer["capped"] is False and len(fewer["comparedWith"]) == 5
