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
    assert ledger.get_row(db, first, row.id).entity == ledger.prepare(first, entity).model_dump(
        mode="json", exclude_none=True
    )
    assert (
        db.scalar(select(func.ST_AsText(ScanEntity.world_anchor)).where(ScanEntity.id == row.id))
        == "POINT Z (1 2 -3)"
    )
    # Client-provided geography without placement is not trusted.
    assert row.geo_anchor is None
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
    assert client.get(root).json()[0]["entity"] == ledger.prepare(
        first, Entity.model_validate(ENTITY)
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    assert client.get(f"/api/v1/captures/{second.id}/entities/{entity_id}").status_code == 404
    assert (
        client.put(f"{root}/{entity_id}", json={**ENTITY, "label": "tree"}).json()["entity"][
            "label"
        ]
        == "tree"
    )
    assert client.delete(f"{root}/{entity_id}").status_code == 204
    assert client.get(root).json() == []


def test_recorded_route_commits_and_replays_entities_once(
    client: Any, db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = capture(db)
    monkeypatch.setattr(client.app.state.settings, "ask_mode", "replay")
    monkeypatch.setattr(client.app.state.settings, "anthropic_api_key", None)
    payload = {**REQUEST, "scanId": str(scan.id), "question": "How tall is the tree?"}
    from tests.test_ask import events

    first = client.post("/v1/ask", json=payload)
    assert first.status_code == 200
    assert events(first)[-1] == {"type": "done"}
    assert len(ledger.rows(db, scan)) == 1
    run = db.get(AskRun, payload["id"])
    assert run.state == "complete" and run.events == events(first)
    # Saved answers remain available after switching modes, with no configured key.
    client.app.state.settings.ask_mode = "live"
    second = client.post("/v1/ask", json=payload)
    assert events(second) == events(first)
    assert len(ledger.rows(db, scan)) == 1
    changed = client.post("/v1/ask", json={**payload, "question": "How many trees?"})
    assert changed.status_code == 409


def test_mock_route_without_key_and_invalid_replay_cleanup(
    client: Any, db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client.app.state.settings, "ask_reasoner", "mock")
    monkeypatch.setattr(client.app.state.settings, "ask_analysis_enabled", False)
    monkeypatch.setattr(client.app.state.settings, "anthropic_api_key", None)
    monkeypatch.setattr(client.app.state.settings, "ask_mode", "live")
    from tests.test_ask import events

    response = client.post("/v1/ask", json=REQUEST)
    assert response.status_code == 200 and events(response)[-1] == {"type": "done"}
    assert db.get(AskRun, REQUEST["id"]).state == "complete"
    client.app.state.settings.ask_mode = "replay"
    scan = capture(db)
    payload = {**REQUEST, "id": "unsupported-replay", "scanId": str(scan.id)}
    assert client.post("/v1/ask", json=payload).status_code == 422
    assert db.get(AskRun, payload["id"]).state == "failed"
    assert ledger.rows(db, scan) == []
