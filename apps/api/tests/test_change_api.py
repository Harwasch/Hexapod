"""`GET /sites/{id}/change?from=&to=`: the change between two dates (`change-v1`, A5).

The yard's two fixture epochs registered through `register` on one site, each with its
confidence grid (synthetic-yard-epochs) and height surface (synthetic-yard-surfaces): e2
has a shed and a stack the earlier e1 had not, e1 a spoil heap e2 has not. The answer is
held to P5's demo reference over the same grids (`surfaces.json`), and every refusal has
its code.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import shape
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.main import create_app
from app.models import Capture, Epoch
from app.models.enums import CaptureKind
from app.services import confidence_maps
from app.storage import S3Storage, get_public_storage
from tests.confidence_fixtures import blob, measured, register_run
from tests.surface_fixtures import DOCUMENT, surface_blob, surface_descriptor
from tests.test_asset_scale import HEIGHT
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


def surveyed(
    epoch: str,
    document: dict[str, Any] | None = None,
    *,
    surface: bool = True,
    confidence: bool = True,
    uncertainty_m: float | None = None,
    origin_height_m: float = HEIGHT,
) -> dict[str, Any]:
    out = measured(epoch if confidence else None, document)
    if confidence and uncertainty_m is not None:
        out["confidence"]["alignmentCap"] = {
            **out["confidence"]["alignmentCap"],
            "uncertaintyM": uncertainty_m,
        }
    if surface:
        out["surface"] = surface_descriptor(epoch, origin_height_m)
    return out


def two_epochs(
    db: Session, storage: S3Storage, tmp_path: Path, **second: Any
) -> tuple[Epoch, Epoch]:
    first = _capture(db, "first", FIRST)
    register_run(
        db, storage, tmp_path, first, surveyed("e1"), grid=blob("e1"), surface=surface_blob("e1")
    )
    site_id = _epoch(db, first).site_id
    later = _capture(db, "second", SECOND)
    register_run(
        db,
        storage,
        tmp_path,
        later,
        surveyed("e2", _echoed(site_id), **second),
        grid=blob("e2"),
        surface=surface_blob("e2"),
    )
    e1, e2 = _epoch(db, first), _epoch(db, later)
    assert e1.site_id == e2.site_id
    return e1, e2


def get(api: TestClient, a: Epoch, b: Epoch) -> Any:
    return api.get(f"/api/v1/sites/{a.site_id}/change", params={"from": str(a.id), "to": str(b.id)})


def test_the_change_from_e1_to_e2_is_p5_s_reference(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    e1, e2 = two_epochs(db, storage, tmp_path)

    response = get(api, e1, e2)

    assert response.status_code == 200, response.text
    read = response.json()
    assert (read["policy"], read["derived"], read["calibration"], read["frame"]) == (
        "change-v1",
        True,
        "uncalibrated",
        "tileset",
    )
    assert read["label"] == "Derived from height surfaces, not observed"
    assert (read["fromEpochId"], read["toEpochId"]) == (str(e1.id), str(e2.id))
    reference = DOCUMENT["change"]
    assert (read["level"], read["nativeLevel"], read["cellM"]) == (2, 2, 1.0)
    assert read["counts"] == reference["counts"]
    assert read["lowConfidence"]["raised"] == reference["weakChanged"] == 23
    assert read["uncertainty"] == {"fromM": 0.0, "toM": 0.062}
    assert read["thresholdSigma"] == 3
    assert read["appearance"]["status"] == "not-computed"
    assert "never splat pixels" in read["appearance"]["reason"]
    # The least: 3 cm of sigma each side at 1 m cells (e2's 0.5 m children by max), u 0.062 m.
    assert read["minDetectableM"] == pytest.approx(3 * (2 * 0.03**2 + 0.062**2) ** 0.5, abs=1e-3)
    assert read["areaLevel"] == 2 and read["areaCellM"] == 1.0

    areas = read["areas"]
    changed = [(a["class"], a["cells"], a["lowConfidence"]) for a in areas[:3]]
    # The shed (6 x 4 m, seen well), the stack where e1 was weak, the spoil heap gone.
    assert changed == [("raised", 24, False), ("raised", 23, True), ("lowered", 24, False)]
    shed, stack, spoil = areas[:3]
    assert shed["meanDeltaM"] == pytest.approx(2.5, abs=0.03) and shed["areaM2"] == 24.0
    assert stack["meanDeltaM"] == pytest.approx(1.0, abs=0.03)
    assert spoil["meanDeltaM"] == pytest.approx(-1.2, abs=0.03)
    # The ground did not move: e2's ground is its 0.5 m children's least, on the slope.
    assert shed["meanGroundDeltaM"] == pytest.approx(0.0, abs=0.03)
    for area in areas:
        geometry = shape(area["footprint"])
        assert area["footprint"]["type"] == "MultiPolygon" and geometry.is_valid
    insufficient = [a for a in areas if a["class"] == "insufficient"]
    assert sum(a["cells"] for a in insufficient) == reference["counts"]["insufficient"]
    assert all(a["meanDeltaM"] is None for a in insufficient)
    assert sum(a["cells"] for a in areas) == sum(reference["counts"].values())


def test_from_and_to_the_other_way_round_swap_raised_and_lowered(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    e1, e2 = two_epochs(db, storage, tmp_path)
    forward, back = get(api, e1, e2).json(), get(api, e2, e1).json()
    assert back["counts"]["raised"] == forward["counts"]["lowered"]
    assert back["counts"]["lowered"] == forward["counts"]["raised"]
    assert back["uncertainty"] == {"fromM": 0.062, "toM": 0.0}


def test_an_unaligned_re_scan_sees_no_change_and_says_how_much_it_could(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """e2 placed by GPS alone at +/-10 m: the threshold is ~30 m, so the 2.5 m shed is not
    a change it can see -- and nothing is called changed."""
    e1, e2 = two_epochs(db, storage, tmp_path, uncertainty_m=10.0)
    read = get(api, e1, e2).json()
    assert read["minDetectableM"] == pytest.approx(30.0, abs=0.01)
    assert read["counts"]["raised"] == read["counts"]["lowered"] == 0
    assert read["uncertainty"]["toM"] == 10.0


@pytest.mark.parametrize(
    ("second", "code"),
    [
        ({"surface": False}, "surface-not-measured"),
        ({"confidence": False}, "confidence-not-measured"),
        ({"origin_height_m": HEIGHT - 2.0}, "frames-differ"),
    ],
)
def test_what_cannot_be_compared_is_refused_with_its_code(
    api: TestClient,
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    second: dict[str, Any],
    code: str,
) -> None:
    e1, e2 = two_epochs(db, storage, tmp_path, **second)
    response = get(api, e1, e2)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == code


def test_only_published_epochs_of_the_site_are_compared(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    import uuid

    e1, e2 = two_epochs(db, storage, tmp_path)
    same = get(api, e1, e1)
    assert same.status_code == 422
    stranger = _capture(db, "stranger", SECOND)
    register_run(
        db, storage, tmp_path, stranger, surveyed("e1"), grid=blob("e1"), surface=surface_blob("e1")
    )
    other = _epoch(db, stranger)
    assert other.site_id != e1.site_id
    assert get(api, e1, other).status_code == 404
    unknown = api.get(
        f"/api/v1/sites/{uuid.uuid4()}/change", params={"from": str(e1.id), "to": str(e2.id)}
    )
    assert unknown.status_code == 404
    db.execute(text("UPDATE epochs SET status = 'rejected' WHERE id = :id"), {"id": e1.id})
    db.commit()
    db.expire_all()  # as a request's own session would read it
    response = get(api, e1, e2)
    assert response.status_code == 409 and response.json()["code"] == "epoch-not-published"
    assert api.get(f"/api/v1/sites/{e1.site_id}/change").status_code == 422  # from/to required


def test_a_surface_missing_from_the_bucket_is_unreadable(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    e1, e2 = two_epochs(db, storage, tmp_path)
    epoch = api.get(f"/api/v1/sites/{e2.site_id}/epochs").json()[1]
    key = confidence_maps.storage_key(storage, epoch["surface"]["url"])
    assert key is not None
    storage.delete_object(key)
    response = get(api, e1, e2)
    assert response.status_code == 409 and response.json()["code"] == "surface-unreadable"


def test_a_damaged_surface_is_unreadable_not_a_change(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    e1, e2 = two_epochs(db, storage, tmp_path)
    epoch = api.get(f"/api/v1/sites/{e2.site_id}/epochs").json()[1]
    key = confidence_maps.storage_key(storage, epoch["surface"]["url"])
    assert key is not None
    damaged = bytearray(copy.copy(surface_blob("e2")))
    damaged[-12] ^= 0xFF
    storage.put_object(key, bytes(damaged), "application/octet-stream")
    response = get(api, e1, e2)
    assert response.status_code == 409 and response.json()["code"] == "surface-unreadable"


def test_the_change_is_open_without_the_write_token(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A read of public data for the viewer's change drape: open, like every GET but the
    fill workflow's targets."""
    from app.config import Settings

    e1, e2 = two_epochs(db, storage, tmp_path)
    app = create_app(Settings(api_write_token="a5-token"))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_public_storage] = lambda: storage
    confidence_maps.GRIDS.clear()
    with TestClient(app) as client:
        assert get(client, e1, e2).status_code == 200


def test_the_change_spends_the_cell_map_bucket(api: TestClient) -> None:
    spec = api.app.openapi()  # type: ignore[attr-defined]
    operation = spec["paths"]["/api/v1/sites/{site_id}/change"]["get"]
    assert "security" not in operation
    from app.api.v1 import sites

    [route] = [
        r for r in sites.router.routes if getattr(r, "path", "") == "/sites/{site_id}/change"
    ]
    calls = {d.call.__name__ for d in route.dependant.dependencies}  # type: ignore[attr-defined]
    assert "limit_cell_map" in calls


def test_a_surface_finer_than_its_confidence_grid_is_refused_not_a_crash(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A run whose surface (0.5 m) is finer than its own confidence grid (1 m) -- no surface
    stage writes that, but a descriptor can say it: nothing vouches for its cells at 0.5 m,
    and the answer is a 409 saying the grids cannot be read together, never a 500."""
    first = _capture(db, "first", FIRST)
    document = surveyed("e2")
    document["confidence"] = measured("e1")["confidence"]
    register_run(
        db, storage, tmp_path, first, document, grid=blob("e1"), surface=surface_blob("e2")
    )
    site_id = _epoch(db, first).site_id
    later = _capture(db, "second", SECOND)
    register_run(
        db,
        storage,
        tmp_path,
        later,
        surveyed("e2", _echoed(site_id)),
        grid=blob("e2"),
        surface=surface_blob("e2"),
    )
    response = get(api, _epoch(db, first), _epoch(db, later))
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "confidence-unreadable"


def test_an_unknown_epoch_is_a_404_whatever_the_other_one_is(
    api: TestClient, db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Both epochs are resolved before either one's status is judged: a rejected `from`
    with an unknown `to` is a 404 for `to`, not a 409 for `from`."""
    import uuid

    e1, _ = two_epochs(db, storage, tmp_path)
    db.execute(text("UPDATE epochs SET status = 'rejected' WHERE id = :id"), {"id": e1.id})
    db.commit()
    db.expire_all()
    missing = uuid.uuid4()
    response = api.get(
        f"/api/v1/sites/{e1.site_id}/change", params={"from": str(e1.id), "to": str(missing)}
    )
    assert response.status_code == 404, response.text
    assert str(missing) in response.text
