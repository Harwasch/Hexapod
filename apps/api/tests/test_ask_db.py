"""PostGIS integration gate: scope, XYZ anchors, replay and atomic entity persistence."""

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select

from app.models import AskRun, ScanEntity
from app.schemas.ask import AskRequest, Entity
from app.schemas.capture import CaptureCreate
from app.services import ask_entities as ledger
from app.services.captures import create_capture
from app.services.errors import ConflictError, NotFoundError
from tests.test_ask import ENTITY, REQUEST


def capture(db: Any, name: Any = "scan") -> Any:
    return create_capture(db, CaptureCreate(name=name, kind="video"))


def test_entity_metric_geo_storage_and_capture_scope(db: Any) -> None:
    first, second = (capture(db), capture(db, "second"))
    entity = Entity.model_validate(
        {
            **ENTITY,
            "anchor": {
                "world": ENTITY["anchor"]["world"],
                "geo": {"lat": 35, "lon": -120, "alt": 15},
            },
        }
    )
    row = ledger.add(db, first, entity)
    db.commit()
    assert ledger.get_row(db, first, row.id).entity == entity.model_dump(
        mode="json", exclude_none=True
    )
    assert (
        db.scalar(select(func.ST_AsText(ScanEntity.world_anchor)).where(ScanEntity.id == row.id))
        == "POINT Z (1 2 -3)"
    )
    assert (
        db.scalar(select(func.ST_SRID(ScanEntity.geo_anchor)).where(ScanEntity.id == row.id))
        == 4326
    )
    assert db.scalar(select(func.ST_Z(ScanEntity.geo_anchor)).where(ScanEntity.id == row.id)) == 15
    with pytest.raises(NotFoundError):
        ledger.get_row(db, second, row.id)
    with pytest.raises(NotFoundError):
        ledger.check_site(first, uuid.uuid4())
    assert ledger.rows(db, second) == []
    ledger.assign(row, Entity.model_validate(ENTITY))
    db.commit()
    assert row.geo_anchor is None
    db.delete(row)
    db.commit()
    assert ledger.rows(db, first) == []


def test_request_claim_completion_and_replay_save_entities_once(db: Any) -> None:
    scan = capture(db)
    request = AskRequest.model_validate({**REQUEST, "scanId": str(scan.id)})
    run, replay = ledger.claim(db, request, scan)
    assert not replay
    with pytest.raises(ConflictError, match="running"):
        ledger.claim(db, request, scan)
    db.rollback()
    complete: list[dict[str, Any]] = [
        {"type": "entity", "entity": ENTITY},
        {"type": "speech", "text": "Plant."},
        {"type": "done"},
    ]
    ledger.finish(db, run, complete, scan)
    same, replay = ledger.claim(db, request, scan)
    assert replay and same.events == complete
    assert len(ledger.rows(db, scan)) == 1
    with pytest.raises(ConflictError, match="different"):
        ledger.claim(db, request.model_copy(update={"question": "Other"}), scan)
    db.rollback()
    assert db.get(AskRun, request.id).state == "complete"


def test_failed_request_can_restart_without_partial_entities(db: Any) -> None:
    scan = capture(db)
    request = AskRequest.model_validate({**REQUEST, "scanId": str(scan.id)})
    run, _ = ledger.claim(db, request, scan)
    run.state = "failed"
    db.commit()
    run, replay = ledger.claim(db, request, scan)
    assert not replay and run.state == "running"
    assert ledger.rows(db, scan) == []


def test_entity_endpoints_scope_and_crud(client: Any, db: Any) -> None:
    first, second = (capture(db), capture(db, "other"))
    root = f"/api/v1/captures/{first.id}/entities"
    response = client.post(root, json=ENTITY)
    assert response.status_code == 201
    entity_id = response.json()["id"]
    assert client.get(root).json()[0]["entity"] == ENTITY
    assert client.get(f"/api/v1/captures/{second.id}/entities/{entity_id}").status_code == 404
    assert (
        client.put(f"{root}/{entity_id}", json={**ENTITY, "label": "tree"}).json()["entity"][
            "label"
        ]
        == "tree"
    )
    assert client.delete(f"{root}/{entity_id}").status_code == 204
    assert client.get(root).json() == []
