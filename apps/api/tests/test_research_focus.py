from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.research import Evidence, ResearchArtifact, ResearchRun
from app.research import queue
from app.research.focus import model_context, read_geometry
from app.research.model import (
    CompleteAction,
    DecisionResult,
    EvidenceReadAction,
    FocusGeometryAction,
    ResearchDecision,
)
from app.research.worker import ResearchWorker
from app.services.errors import InvalidInputError
from tests.test_research import evidence, start


def mapped_source(client: TestClient, db: Session) -> tuple[dict[str, Any], str, str, str]:
    land, inv, run, _ = start(client)
    client.post(f"/api/v1/research/runs/{run['id']}/cancel")
    source = Evidence(
        run_id=run["id"],
        source_key="fixture",
        content=evidence().model_dump(mode="json"),
    )
    db.add(source)
    db.flush()
    artifact = ResearchArtifact(
        run_id=run["id"],
        output_key="map",
        content={
            "title": "Synthetic survey",
            "method": "Software test fixture",
            "evidenceIds": [str(source.id)],
            "output": {
                "kind": "map",
                "unit": "m",
                "legend": "Synthetic values",
                "features": [
                    {
                        "label": "Survey point",
                        "value": 0,
                        "geometry": {"type": "Point", "coordinates": [-77.05, 38.89]},
                    }
                ],
            },
        },
    )
    db.add(artifact)
    db.commit()
    return land, inv["id"], str(artifact.id), str(source.id)


def test_focus_pins_same_land_evidence_and_is_idempotent(client: TestClient, db: Session) -> None:
    land, _, artifact_id, evidence_id = mapped_source(client, db)
    other_inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Follow-up", "boundaryRevision": 1},
    ).json()
    endpoint = f"/api/v1/research/investigations/{other_inv['id']}/runs"
    body = {
        "requestKey": str(uuid.uuid4()),
        "question": "What about this?",
        "focus": {"artifactId": artifact_id, "featureIndex": 0},
    }
    response = client.post(endpoint, json=body)
    assert response.status_code == 201, response.text
    assert response.json()["focusLabel"] == "Survey point"
    assert response.json()["focus"] == body["focus"]
    run = db.get(ResearchRun, uuid.UUID(response.json()["id"]))
    assert run is not None and run.focus_snapshot is not None
    snapshot = run.focus_snapshot
    assert snapshot["feature"]["value"] == 0 and snapshot["unit"] == "m"
    assert snapshot["boundaryRevision"] == 1
    copied = db.get(Evidence, uuid.UUID(snapshot["evidenceIds"][0]))
    original = db.get(Evidence, uuid.UUID(evidence_id))
    assert copied and original and copied.id != original.id and copied.content == original.content
    queue.validate_citations(db, run, [copied.id])
    assert client.post(endpoint, json=body).json()["id"] == str(run.id)
    assert client.post(endpoint, json={**body, "focus": None}).status_code == 409
    # A retry never recopies or re-resolves source content.
    artifact = db.get(ResearchArtifact, uuid.UUID(artifact_id))
    assert artifact
    artifact.content = {
        **artifact.content,
        "output": {"kind": "map", "features": [], "legend": "Changed fixture"},
    }
    db.commit()
    assert client.post(endpoint, json=body).json()["id"] == str(run.id)
    db.expire_all()
    assert run.focus_snapshot == snapshot
    assert len(list(db.scalars(select(Evidence).where(Evidence.run_id == run.id)))) == 1


def test_focus_rejects_other_land_and_invalid_features(client: TestClient, db: Session) -> None:
    _, inv, artifact_id, _ = mapped_source(client, db)
    _, other_inv, _, _ = start(client)
    body = {
        "requestKey": str(uuid.uuid4()),
        "question": "Inspect",
        "focus": {"artifactId": artifact_id, "featureIndex": 0},
    }
    assert (
        client.post(
            f"/api/v1/research/investigations/{other_inv['id']}/runs", json=body
        ).status_code
        == 404
    )
    endpoint = f"/api/v1/research/investigations/{inv}/runs"
    db.rollback()
    for index in (-1, 1, 2000):
        assert (
            client.post(
                endpoint, json={**body, "focus": {"artifactId": artifact_id, "featureIndex": index}}
            ).status_code
            == 422
        )
        db.rollback()
    artifact = db.get(ResearchArtifact, uuid.UUID(artifact_id))
    assert artifact
    artifact.content = {**artifact.content, "output": {"kind": "document", "markdown": "Not a map"}}
    db.commit()
    assert client.post(endpoint, json=body).status_code == 422


def test_focus_geometry_pages_preserve_ring_paths_and_boundaries() -> None:
    ring = [[index / 10000, 0] for index in range(2000)]
    snapshot: dict[str, Any] = {
        "boundaryRevision": 1,
        "feature": {
            "label": "Fixture",
            "geometry": {"type": "MultiPolygon", "coordinates": [[ring, [[0, 0], [1, 1], [0, 0]]]]},
        },
    }
    context = model_context(snapshot, 2)
    assert context and context["boundaryDiffersFromInvestigation"] and context["geometryOmitted"]
    assert "geometry" not in context["feature"] and "geometry" in snapshot["feature"]
    page = read_geometry(snapshot, 1999, 3)
    assert page["nextOffset"] == 2002
    assert [v["path"] for v in page["vertices"]] == [[0, 0, 1999], [0, 1, 0], [0, 1, 1]]
    assert read_geometry(snapshot, 2002, 3)["nextOffset"] is None
    with pytest.raises(InvalidInputError):
        read_geometry(snapshot, 2003, 3)
    with pytest.raises(InvalidInputError):
        read_geometry(None, 0, 3)


def test_worker_uses_pinned_feature_and_its_copied_citations(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    _, inv, artifact_id, _ = mapped_source(client, db)
    response = client.post(
        f"/api/v1/research/investigations/{inv}/runs",
        json={
            "requestKey": str(uuid.uuid4()),
            "question": "What about this?",
            "focus": {"artifactId": artifact_id, "featureIndex": 0},
        },
    )
    assert response.status_code == 201
    run_id = response.json()["id"]
    db.rollback()

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            state = json.loads(context)
            focus = state["focusedMapFeature"]
            assert focus["feature"]["value"] == 0
            assert focus["feature"]["geometry"]["coordinates"] == [-77.05, 38.89]
            assert focus["method"] == "Software test fixture"
            actions = state["previousActions"]
            action: FocusGeometryAction | EvidenceReadAction | CompleteAction
            if not actions:
                action = FocusGeometryAction(kind="read_focused_geometry")
            elif len(actions) == 1:
                assert json.loads(actions[-1]["result"])["vertices"][0]["path"] == []
                action = EvidenceReadAction(
                    kind="read_research_evidence", evidence_id=focus["evidenceIds"][0]
                )
            else:
                action = CompleteAction(
                    kind="complete",
                    summary="The selected survey point is a synthetic fixture.",
                    evidence_ids=focus["evidenceIds"],
                )
            return DecisionResult(
                ResearchDecision(progress="Inspecting selection", action=action), 100
            )

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as http:
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=Model(), client=http
        ).run_once()
    result = client.get(f"/api/v1/research/runs/{run_id}").json()
    assert result["status"] == "succeeded", result
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert "synthetic fixture" in detail["messages"][-1]["content"]
