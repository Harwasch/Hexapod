"""`GET`/`PUT /sites/{id}/check-targets`: the site's surveyed marks, replaced as a set."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models import SiteCheckTarget
from tests.conftest import site_payload

TOKEN = "check-targets-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    """The real write-token check, with a token configured (test_write_token's pattern)."""
    app = create_app(Settings(api_write_token=TOKEN))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


def _target(label: str, **overrides: Any) -> dict[str, Any]:
    target: dict[str, Any] = {
        "label": label,
        "role": "check",
        "lat": 47.6443,
        "lon": -122.138,
        "height": 10.8,
        "heightDatum": "ellipsoid-wgs84",
        "sigmaHM": 0.005,
        "sigmaVM": 0.01,
        "source": "rtk-rover",
        "surveyedAt": "2026-10-01T09:30:00Z",
    }
    target.update(overrides)
    return target


def _site(client: TestClient) -> str:
    response = client.post("/api/v1/sites", json=site_payload(assets=[]), headers=AUTH)
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _put(client: TestClient, site_id: str, *targets: dict[str, Any]) -> Any:
    response = client.put(
        f"/api/v1/sites/{site_id}/check-targets", json={"targets": list(targets)}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    return response.json()


def _rows(db: Session) -> list[SiteCheckTarget]:
    db.expire_all()
    return list(db.scalars(select(SiteCheckTarget).order_by(SiteCheckTarget.created_at)))


def test_put_then_get_round_trips_the_set(client: TestClient) -> None:
    site_id = _site(client)
    assert client.get(f"/api/v1/sites/{site_id}/check-targets").json() == []

    put = _put(client, site_id, _target("T2"), _target("T1", role="control"))
    got = client.get(f"/api/v1/sites/{site_id}/check-targets").json()

    assert got == put
    assert [t["label"] for t in got] == ["T1", "T2"]
    first = got[0]
    assert {k: first[k] for k in _target("T1")} == _target(
        "T1", role="control", surveyedAt="2026-10-01T09:30:00Z"
    )
    assert first["siteId"] == site_id and uuid.UUID(first["id"])


def test_an_unchanged_target_keeps_its_row(client: TestClient, db: Session) -> None:
    site_id = _site(client)
    [before] = _put(client, site_id, _target("T1"))
    [after] = _put(client, site_id, _target("T1"))
    assert after["id"] == before["id"]
    assert len(_rows(db)) == 1


def test_an_edit_retires_the_old_row_and_starts_a_new_one(client: TestClient, db: Session) -> None:
    site_id = _site(client)
    [before] = _put(client, site_id, _target("T1"))
    [after] = _put(client, site_id, _target("T1", lat=47.6444))
    assert after["id"] != before["id"] and after["lat"] == 47.6444
    old, new = _rows(db)
    assert str(old.id) == before["id"] and old.retired_at is not None
    assert str(new.id) == after["id"] and new.retired_at is None


def test_a_target_left_out_is_retired_not_deleted(client: TestClient, db: Session) -> None:
    site_id = _site(client)
    _put(client, site_id, _target("T1"), _target("T2"))
    kept = _put(client, site_id, _target("T2"))
    assert [t["label"] for t in kept] == ["T2"]
    rows = {row.label: row for row in _rows(db)}
    assert rows["T1"].retired_at is not None and rows["T2"].retired_at is None
    assert _put(client, site_id) == []
    assert all(row.retired_at is not None for row in _rows(db))


def test_writes_need_the_token_and_reads_do_not(client: TestClient) -> None:
    site_id = _site(client)
    url = f"/api/v1/sites/{site_id}/check-targets"
    assert client.put(url, json={"targets": [_target("T1")]}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.put(url, json={"targets": [_target("T1")]}, headers=wrong).status_code == 401
    assert client.get(url).status_code == 200


def test_an_unknown_site_is_404(client: TestClient) -> None:
    missing = uuid.uuid4()
    url = f"/api/v1/sites/{missing}/check-targets"
    assert client.get(url).status_code == 404
    assert client.put(url, json={"targets": [_target("T1")]}, headers=AUTH).status_code == 404


@pytest.mark.parametrize(
    "bad",
    [
        {"sigmaHM": 0},
        {"heightDatum": "egm96"},
        {"role": "gcp"},
        {"lat": 91},
        {"surveyedAt": "2026-10-01T09:30:00"},
        {"extra": 1},
    ],
    ids=["sigma", "datum", "role", "lat", "naive-time", "extra-field"],
)
def test_a_target_that_is_not_a_survey_is_refused(client: TestClient, bad: dict[str, Any]) -> None:
    site_id = _site(client)
    url = f"/api/v1/sites/{site_id}/check-targets"
    response = client.put(url, json={"targets": [_target("T1", **bad)]}, headers=AUTH)
    assert response.status_code == 422, response.text


def test_a_label_named_twice_is_refused(client: TestClient) -> None:
    site_id = _site(client)
    url = f"/api/v1/sites/{site_id}/check-targets"
    response = client.put(
        url, json={"targets": [_target("T1"), _target("T1", lat=47.6)]}, headers=AUTH
    )
    assert response.status_code == 422
    assert "T1" in response.text


def test_two_replacements_of_one_site_happen_one_after_the_other(
    client: TestClient, db: Session, engine: Any
) -> None:
    """`replace_active` locks the site row first. While another transaction holds that
    lock, a replacement waits rather than reading the active set under it; released, the
    two run one after the other, and the site ends with one active row per label -- not
    two replacements that both retired T1 and both inserted its successor (which the
    partial unique index would have refused, failing one of them)."""
    import threading
    import time

    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    from app.schemas.check_target import CheckTargetSet
    from app.services import check_targets as service

    site_id = uuid.UUID(_site(client))
    _put(client, str(site_id), _target("T1"))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    errors: list[BaseException] = []

    def replace(lat: float) -> None:
        session = factory()
        try:
            payload = CheckTargetSet.model_validate({"targets": [_target("T1", lat=lat)]})
            service.replace_active(session, site_id, payload)
        except BaseException as error:  # pragma: no cover - the failure being tested for
            errors.append(error)
        finally:
            session.close()

    holder = factory()
    holder.execute(text("SELECT 1 FROM sites WHERE id = :id FOR UPDATE"), {"id": site_id})
    threads = [threading.Thread(target=replace, args=(lat,)) for lat in (47.6401, 47.6402)]
    for thread in threads:
        thread.start()
    time.sleep(0.5)
    assert all(thread.is_alive() for thread in threads), "a replacement did not wait for the lock"
    holder.rollback()
    holder.close()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors, errors
    active = [row for row in _rows(db) if row.retired_at is None]
    assert len(active) == 1 and active[0].lat in (47.6401, 47.6402)
    assert len(_rows(db)) == 3
