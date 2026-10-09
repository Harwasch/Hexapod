from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.land import LandArea
from app.models.research import ResearchRun
from app.research import inventory, queue
from app.research.model import (
    CompleteAction,
    DecisionResult,
    InventoryReadAction,
    MappedAssetProposeAction,
    MappedAssetSearchAction,
    ResearchDecision,
)
from app.research.providers.base import SourceContext
from app.research.worker import ResearchWorker
from app.schemas.geojson import Point, Polygon
from app.schemas.land_selection import CandidateRequest
from app.services.errors import InvalidInputError, NotFoundError
from tests.test_ecology_research import start
from tests.test_land import BODY
from tests.test_land_features import FEATURE


def point_response(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST" and b"node" in request.content
    assert b"timeout%3A20" in request.content
    return httpx.Response(
        200,
        json={
            "elements": [
                {
                    "type": "node",
                    "id": 123,
                    "lon": -77.05,
                    "lat": 38.888,
                    "tags": {"power": "pole", "ref": "Fixture pole"},
                }
            ]
        },
    )


def test_exact_point_candidate_source_hash_and_geometry(db: Session) -> None:
    request = CandidateRequest(kind="point", point=Point(coordinates=[-77.05, 38.888]))
    with httpx.Client(transport=httpx.MockTransport(point_response)) as http:
        result = inventory.mapped_search(
            db, SourceContext(Polygon.model_validate(BODY["boundary"])), request, http
        )
    assert result.status == "available" and len(result.evidence) == 1
    item = result.evidence[0][1]
    assert str(item.url) == "https://www.openstreetmap.org/node/123"
    feature = inventory.mapped_feature(
        item.model_dump(mode="json"), uuid.uuid4(), uuid.uuid4(), None, None, ""
    )
    assert feature.geometry.coordinates == [-77.05, 38.888]
    assert feature.category == "power" and feature.status == "candidate"
    assert feature.external_ref and feature.external_ref.record_id == "node/123"
    payload = json.loads(item.excerpt)
    payload["candidate"]["geometry"]["coordinates"][0] += 1
    item.excerpt = json.dumps(payload)
    with pytest.raises(InvalidInputError, match="no longer matches"):
        inventory.mapped_feature(
            item.model_dump(mode="json"), uuid.uuid4(), uuid.uuid4(), None, None, ""
        )


@pytest.mark.parametrize("status,expected", [(503, "unavailable"), (200, "empty")])
def test_mapped_outage_and_empty_are_distinct(db: Session, status: int, expected: str) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, json={"elements": []}))
    ) as http:
        result = inventory.mapped_search(
            db,
            SourceContext(Polygon.model_validate(BODY["boundary"])),
            CandidateRequest(kind="point", point=Point(coordinates=[-77.05, 38.888])),
            http,
        )
    assert result.status == expected and not result.evidence


def writable(feature: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "requestKey",
        "name",
        "category",
        "geometry",
        "source",
        "status",
        "description",
        "attributes",
        "evidenceIds",
        "externalRef",
    }
    return {key: value for key, value in feature.items() if key in fields}


def create_feature(
    client: TestClient, db: Session, **overrides: Any
) -> tuple[LandArea, dict[str, Any]]:
    land = client.post("/api/v1/land", json=BODY).json()
    response = client.post(f"/api/v1/land/{land['id']}/features", json={**FEATURE, **overrides})
    assert response.status_code == 201, response.text
    row = db.get(LandArea, uuid.UUID(land["id"]))
    assert row
    return row, response.json()


def test_revision_geometry_pages_preserve_holes_and_private_scope(
    client: TestClient, db: Session
) -> None:
    outer = [
        [-122.14, 47.64],
        [-122.13, 47.64],
        [-122.13, 47.65],
        [-122.14, 47.65],
        [-122.14, 47.64],
    ]
    hole = [
        [-122.138, 47.642],
        [-122.136, 47.642],
        [-122.136, 47.644],
        [-122.138, 47.644],
        [-122.138, 47.642],
    ]
    land, feature = create_feature(
        client, db, geometry={"type": "Polygon", "coordinates": [outer, hole]}
    )
    identifier = uuid.UUID(feature["id"])
    first = inventory.read(db, land.workspace_id, land.id, identifier, 1, "geometry", 0, 6, None)
    second = inventory.read(
        db,
        land.workspace_id,
        land.id,
        identifier,
        1,
        "geometry",
        first["nextOffset"],
        6,
        datetime.fromisoformat(first["asOf"]),
    )
    assert first["partRingSizes"] == [[5, 5]] and first["vertexCount"] == 10
    assert [item["coordinates"] for item in first["data"] + second["data"]] == outer + hole
    assert second["data"][0]["ring"] == 1 and second["nextOffset"] is None
    path = f"/api/v1/land/{land.id}/features/{identifier}"
    assert (
        client.put(
            path,
            json={
                **writable(feature),
                "requestKey": str(uuid.uuid4()),
                "expectedRevision": 1,
                "note": "User verified",
                "name": "Revised asset",
                "status": "confirmed",
            },
        ).status_code
        == 200
    )
    pinned = inventory.read(db, land.workspace_id, land.id, identifier, 1, "overview", 0, 20, None)
    assert pinned["name"] == FEATURE["name"] and pinned["currentRevision"] == 2
    assert pinned["data"]["status"] == "candidate"
    with pytest.raises(NotFoundError):
        inventory.read(db, uuid.uuid4(), land.id, identifier, 1, "overview", 0, 20, None)
    with pytest.raises(NotFoundError):
        inventory.read(db, land.workspace_id, land.id, identifier, 99, "overview", 0, 20, None)


def test_point_and_inspection_pages_pin_recording_cutoff(client: TestClient, db: Session) -> None:
    land, feature = create_feature(client, db)
    identifier = uuid.UUID(feature["id"])
    points = inventory.read(db, land.workspace_id, land.id, identifier, 1, "geometry", 0, 20, None)
    assert points["data"] == [{"part": 0, "ring": 0, "index": 0, "coordinates": [-122.135, 47.645]}]
    path = f"/api/v1/land/{land.id}/features/{identifier}/inspections"
    payload = {
        "observedAt": "2026-01-01T00:00:00Z",
        "condition": "fair",
        "notes": "Fixture",
        "measurements": {"height": 10},
        "measurementUnits": {"height": "m"},
    }
    assert client.post(path, json=payload).status_code == 201
    page = inventory.read(db, land.workspace_id, land.id, identifier, 1, "inspections", 0, 1, None)
    assert page["data"][0]["measurement_units"] == {"height": "m"}
    assert client.post(path, json=payload).status_code == 201
    pinned = inventory.read(
        db,
        land.workspace_id,
        land.id,
        identifier,
        1,
        "inspections",
        0,
        20,
        datetime.fromisoformat(page["asOf"]),
    )
    assert len(pinned["data"]) == 1
    current = inventory.read(
        db, land.workspace_id, land.id, identifier, 1, "inspections", 0, 1, None
    )
    assert current["nextOffset"] == 1
    remaining = inventory.read(
        db,
        land.workspace_id,
        land.id,
        identifier,
        1,
        "inspections",
        1,
        1,
        datetime.fromisoformat(current["asOf"]),
    )
    assert remaining["nextOffset"] is None and len(remaining["data"]) == 1


class AssetModel:
    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        data = json.loads(context)
        actions = data["previousActions"]
        action: (
            MappedAssetSearchAction
            | MappedAssetProposeAction
            | InventoryReadAction
            | CompleteAction
        )
        if not actions:
            action = MappedAssetSearchAction(kind="search_mapped_assets", source_kind="point")
        elif len(actions) == 1:
            source = next(iter(data["retrieved"].values()))
            action = MappedAssetProposeAction(
                kind="propose_mapped_asset", evidence_id=uuid.UUID(source["evidenceIds"][0])
            )
        elif len(actions) == 2:
            saved = json.loads(actions[-1]["result"])
            action = InventoryReadAction(
                kind="read_inventory", feature_id=uuid.UUID(saved["featureId"]), section="geometry"
            )
        else:
            source = next(
                value
                for value in data["retrieved"].values()
                if value["provider"] == "private-inventory"
            )
            assert source["data"]["data"][0]["coordinates"] == [-77.05, 38.888]
            action = CompleteAction(
                kind="complete",
                summary="Candidate mapped; review in Assets.",
                evidence_ids=source["evidenceIds"],
            )
        return DecisionResult(
            ResearchDecision(progress="Inspecting mapped assets", action=action), 100
        )


def test_agent_search_propose_read_and_preserve_confirmed_duplicate(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    inv, _run, _body = start(client, kind="investigation", analysis=None)
    path = f"/api/v1/research/investigations/{inv['id']}"
    with httpx.Client(transport=httpx.MockTransport(point_response)) as http:
        db.rollback()
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=AssetModel(), client=http
        ).run_once()
        db.expire_all()
        detail = client.get(path).json()
        assert detail["runs"][0]["status"] == "succeeded", detail["runs"]
        assert len(detail["artifacts"]) == 1 and len(detail["evidence"]) == 2
        source = next(
            item for item in detail["evidence"] if item["provider"] == "private-inventory"
        )
        feature_id = source["inventory"]["featureId"]
        feature_path = f"/api/v1/land/{inv['landId']}/features/{feature_id}"
        feature = client.get(feature_path).json()
        assert feature["status"] == "candidate" and feature["revision"] == 1
        assert (
            client.put(
                feature_path,
                json={
                    **writable(feature),
                    "expectedRevision": 1,
                    "note": "Confirmed by user",
                    "requestKey": str(uuid.uuid4()),
                    "status": "confirmed",
                },
            ).status_code
            == 200
        )
        assert (
            client.post(
                path + "/runs",
                json={"requestKey": str(uuid.uuid4()), "question": "Find the pole again"},
            ).status_code
            == 201
        )
        db.rollback()
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=AssetModel(), client=http
        ).run_once()
        db.expire_all()
        feature = client.get(feature_path).json()
        assert feature["status"] == "confirmed" and feature["revision"] == 2
        assert len(client.get(f"/api/v1/land/{inv['landId']}/features").json()) == 1


class FinishModel:
    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        data = json.loads(context)
        source = next(iter(data["retrieved"].values()))
        assert source["data"]["revision"] == 1
        return DecisionResult(
            ResearchDecision(
                progress="Recovered snapshot",
                action=CompleteAction(
                    kind="complete",
                    summary="Original asset snapshot retained.",
                    evidence_ids=source["evidenceIds"],
                ),
            ),
            100,
        )


def test_inventory_recovery_reuses_original_page_after_asset_edit(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    inv, _run, _body = start(client, kind="investigation", analysis=None)
    land = db.get(LandArea, uuid.UUID(inv["landId"]))
    assert land
    response = client.post(f"/api/v1/land/{land.id}/features", json=FEATURE)
    assert response.status_code == 201
    feature = response.json()
    action = InventoryReadAction(kind="read_inventory", feature_id=uuid.UUID(feature["id"]))
    key = "inventory/" + hashlib.sha256(action.model_dump_json().encode()).hexdigest()
    data = inventory.read(
        db, land.workspace_id, land.id, action.feature_id, None, "overview", 0, 20, None
    )
    claimed = queue.claim(db)
    assert claimed
    run_id, token = claimed
    decision = ResearchDecision(progress="Reading asset", action=action)
    queue.checkpoint(
        db,
        run_id,
        token,
        {
            "sources": {},
            "steps": 1,
            "actions": [],
            "output_tokens": 0,
            "pending_action": decision.model_dump(mode="json"),
            "pending_inventory": {"key": key, "data": data},
        },
    )
    assert (
        client.put(
            f"/api/v1/land/{land.id}/features/{feature['id']}",
            json={
                **writable(feature),
                "requestKey": str(uuid.uuid4()),
                "expectedRevision": 1,
                "note": "Concurrent update",
                "name": "New name",
            },
        ).status_code
        == 200
    )
    row = db.get(ResearchRun, run_id)
    assert row
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    assert ResearchWorker(sessions, Settings(_env_file=None), model=FinishModel()).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert len(detail["evidence"]) == 1
    page = detail["evidence"][0]
    assert (
        page["inventory"]["revision"] == 1
        and json.loads(page["excerpt"])["name"] == FEATURE["name"]
    )
    assert page["inventory"]["sha256"] == hashlib.sha256(page["excerpt"].encode()).hexdigest()


class ForeignModel:
    def __init__(self, feature_id: str, evidence_id: str) -> None:
        self.feature_id, self.evidence_id = feature_id, evidence_id

    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        data = json.loads(context)
        actions = data["previousActions"]
        action: InventoryReadAction | MappedAssetProposeAction | CompleteAction
        if not actions:
            action = InventoryReadAction(
                kind="read_inventory", feature_id=uuid.UUID(self.feature_id)
            )
        elif len(actions) == 1:
            assert "not found" in actions[-1]["result"]
            action = MappedAssetProposeAction(
                kind="propose_mapped_asset", evidence_id=uuid.UUID(self.evidence_id)
            )
        else:
            assert not data["retrieved"]
            assert "featureId" not in actions[-1]["result"]
            action = CompleteAction(
                kind="complete", summary="Other land records were not read or copied."
            )
        return DecisionResult(
            ResearchDecision(progress="Checking record scope", action=action), 100
        )


def test_foreign_land_asset_and_candidate_cannot_be_read_or_copied(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    inv, _run, _body = start(client, kind="investigation", analysis=None)
    db.rollback()
    with httpx.Client(transport=httpx.MockTransport(point_response)) as http:
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=AssetModel(), client=http
        ).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    asset = next(item for item in detail["evidence"] if item["provider"] == "private-inventory")
    mapped = next(item for item in detail["evidence"] if item["provider"] == "mapped-asset")
    other, _run, _body = start(client, kind="investigation", analysis=None)
    db.rollback()
    assert ResearchWorker(
        sessions,
        Settings(_env_file=None),
        model=ForeignModel(asset["inventory"]["featureId"], mapped["id"]),
    ).run_once()
    db.expire_all()
    result = client.get(f"/api/v1/research/investigations/{other['id']}").json()
    assert result["runs"][0]["status"] == "succeeded" and result["evidence"] == []
    assert client.get(f"/api/v1/land/{other['landId']}/features").json() == []


def test_mapped_source_recovery_keeps_evidence_without_refetch(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_solar import BOUNDARY

    inv, _run, _body = start(client, kind="investigation", analysis=None)
    claimed = queue.claim(db)
    assert claimed
    run_id, token = claimed
    action = MappedAssetSearchAction(kind="search_mapped_assets", source_kind="point")
    decision = ResearchDecision(progress="Finding mapped assets", action=action)
    state = {
        "sources": {},
        "steps": 1,
        "actions": [],
        "output_tokens": 0,
        "pending_action": decision.model_dump(mode="json"),
    }
    queue.checkpoint(db, run_id, token, state)
    context = SourceContext(BOUNDARY)
    request = CandidateRequest(
        kind="point", point=Point(coordinates=list(context.point)), radius_m=action.radius_m
    )

    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Simulated interruption after evidence commit")

    with (
        httpx.Client(transport=httpx.MockTransport(point_response)) as http,
        monkeypatch.context() as patch,
    ):
        patch.setattr(queue, "save_artifact", interrupted)
        worker = ResearchWorker(sessions, Settings(_env_file=None), model=AssetModel(), client=http)
        with pytest.raises(RuntimeError, match="Simulated interruption"):
            worker._mapped_assets(db, run_id, token, request, context, http, state)
    row = db.get(ResearchRun, run_id)
    assert row
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()

    def reject(request: httpx.Request) -> httpx.Response:
        pytest.fail("A saved mapped source snapshot must not be fetched again")

    with httpx.Client(transport=httpx.MockTransport(reject)) as http:
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=AssetModel(), client=http
        ).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded" and detail["runs"][0]["attempt"] == 2
    assert len(detail["artifacts"]) == 1 and len(detail["evidence"]) == 2
    assert len(client.get(f"/api/v1/land/{inv['landId']}/features").json()) == 1
