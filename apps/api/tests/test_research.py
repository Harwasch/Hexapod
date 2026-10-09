from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.research import Evidence, Finding, ResearchRun
from app.research import queue
from app.schemas.research import (
    ArtifactContent,
    EvidenceContent,
    FindingContent,
    TimelineEntry,
    TimelineOutput,
)
from app.services.errors import InvalidInputError
from tests.test_land import BODY
from tests.test_workspaces import headers
from tests.test_workspaces import identity_client as _identity_client

oidc_client = _identity_client


def start(client, headers=None):
    land = client.post("/api/v1/land", json=BODY, headers=headers).json()
    inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Understand this land", "boundaryRevision": 1},
        headers=headers,
    )
    assert inv.status_code == 201, inv.text
    inv = inv.json()
    body = {"requestKey": str(uuid.uuid4()), "kind": "overview", "question": "What is here?"}
    run = client.post(
        f"/api/v1/research/investigations/{inv['id']}/runs", json=body, headers=headers
    )
    assert run.status_code == 201, run.text
    return land, inv, run.json(), body


def evidence():
    return EvidenceContent(
        provider="fixture",
        title="Reference survey",
        url="https://example.test/survey",
        license="CC0",
        attribution="Test survey",
        retrieved_at=datetime.now(UTC),
        excerpt="Survey records woodland.",
        spatial_relevance="within",
        relevance_note="Survey footprint lies inside the selected revision.",
    )


def finding(evidence_id):
    return FindingContent(
        title="Woodland survey",
        summary="A survey documents woodland.",
        category="ecology",
        evidence_ids=[evidence_id],
        confidence="supported",
        uncertainty="The survey is historical; current cover has not been measured.",
    )


def test_research_pins_revision_idempotency_cancel_and_stream(client: TestClient, db: Session):
    land, inv, run, body = start(client)
    endpoint = f"/api/v1/research/investigations/{inv['id']}"
    assert client.post(endpoint + "/runs", json=body).json()["id"] == run["id"]
    assert (
        client.post(
            endpoint + "/runs", json={**body, "question": "A different request"}
        ).status_code
        == 409
    )
    assert (
        client.put(f"/api/v1/land/{land['id']}", json={**BODY, "expectedRevision": 1}).status_code
        == 200
    )
    detail = client.get(endpoint).json()
    assert detail["investigation"]["stale"] is True
    assert detail["investigation"]["boundaryRevision"] == 1
    assert len(detail["messages"]) == 1
    assert detail["page"]["totals"]["runs"] == 1
    claimed = queue.claim(db)
    assert claimed is not None
    run_id, lease = claimed
    eid = queue.save_evidence(db, run_id, lease, "survey", evidence())
    assert queue.save_evidence(db, run_id, lease, "survey", evidence()) == eid
    fid = queue.save_finding(db, run_id, lease, "woodland", finding(eid))
    assert queue.save_finding(db, run_id, lease, "woodland", finding(eid)) == fid
    assert (
        client.put(f"/api/v1/research/findings/{fid}", json={"disposition": "pinned"}).status_code
        == 200
    )
    detail = client.get(endpoint).json()
    assert detail["findings"][0]["disposition"] == "pinned"
    assert detail["evidence"][0]["id"] == str(eid)
    cancelled = client.post(f"/api/v1/research/runs/{run_id}/cancel")
    assert cancelled.json()["status"] == "cancelled"
    with pytest.raises(queue.LeaseLostError):
        queue.save_finding(db, run_id, lease, "late", finding(eid))
    events = client.get(f"/api/v1/research/runs/{run_id}/events").json()
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    stream = client.get(f"/api/v1/research/runs/{run_id}/stream", headers={"Last-Event-ID": "2"})
    assert stream.status_code == 200
    assert "id: 1\n" not in stream.text and "id: 2\n" not in stream.text
    assert '"kind": "cancelled"' in stream.text
    assert len(db.scalars(select(Evidence)).all()) == 1
    assert len(db.scalars(select(Finding)).all()) == 1


def test_expired_worker_is_fenced_and_checkpoints_survive(client: TestClient, db: Session):
    start(client)
    claimed = queue.claim(db)
    assert claimed
    run_id, first = claimed
    queue.checkpoint(db, run_id, first, {"completedSources": ["survey"]})
    run = db.get(ResearchRun, run_id)
    run.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    reclaimed = queue.claim(db)
    assert reclaimed and reclaimed[0] == run_id and reclaimed[1] != first
    second = reclaimed[1]
    assert db.get(ResearchRun, run_id).checkpoint == {"completedSources": ["survey"]}
    with pytest.raises(queue.LeaseLostError):
        queue.heartbeat(db, run_id, first)
    assert queue.claim(db) is None
    queue.finish(db, run_id, second, "succeeded", "Research completed.")
    assert client.get(f"/api/v1/research/runs/{run_id}").json()["attempt"] == 2
    assert queue.claim(db) is None


def test_citations_must_belong_to_the_investigation(client: TestClient, db: Session):
    start(client)
    first_id, first_lease = queue.claim(db)
    eid = queue.save_evidence(db, first_id, first_lease, "survey", evidence())
    queue.finish(db, first_id, first_lease, "succeeded", "Done")
    start(client)
    other_id, other_lease = queue.claim(db)
    with pytest.raises(InvalidInputError):
        queue.save_finding(db, other_id, other_lease, "foreign", finding(eid))
    db.rollback()
    local = queue.save_evidence(db, other_id, other_lease, "local", evidence())
    artifact = ArtifactContent(
        title="History",
        method="Reference survey dates",
        evidence_ids=[local],
        output=TimelineOutput(
            kind="timeline",
            entries=[
                TimelineEntry(date="1900", title="Survey", description="Survey", evidence_ids=[eid])
            ],
        ),
    )
    with pytest.raises(InvalidInputError):
        queue.save_artifact(db, other_id, other_lease, "foreign-timeline", artifact)
    db.rollback()
    assert not db.scalars(select(Finding)).all()


def test_private_research_and_events_follow_workspace_membership(oidc_client, db: Session):
    client, token = oidc_client
    alice = headers(token())
    workspace = client.post("/api/v1/workspaces", headers=alice, json={"name": "Private"}).json()[
        "id"
    ]
    alice = headers(token(), workspace)
    land, inv, _run, _ = start(client, alice)
    run_id, lease = queue.claim(db)
    eid = queue.save_evidence(db, run_id, lease, "survey", evidence())
    fid = queue.save_finding(db, run_id, lease, "finding", finding(eid))
    other = client.post(
        "/api/v1/workspaces", headers=headers(token("bob")), json={"name": "Other"}
    ).json()["id"]
    bob = headers(token("bob"), other)
    for path in (
        f"/land/{land['id']}/investigations",
        f"/research/investigations/{inv['id']}",
        f"/research/runs/{run_id}",
        f"/research/runs/{run_id}/events",
        f"/research/runs/{run_id}/stream",
        f"/research/evidence/{eid}",
    ):
        assert client.get("/api/v1" + path, headers=bob).status_code == 404
    assert client.post(f"/api/v1/research/runs/{run_id}/cancel", headers=bob).status_code == 404
    assert (
        client.put(
            f"/api/v1/research/findings/{fid}", json={"disposition": "dismissed"}, headers=bob
        ).status_code
        == 404
    )


def test_overview_is_idempotent_per_boundary_revision(client: TestClient):
    area = client.post("/api/v1/land", json=BODY).json()
    url = f"/api/v1/land/{area['id']}/overview"
    first = client.post(url, json={"boundaryRevision": 1})
    assert first.status_code == 200, first.text
    assert client.post(url, json={"boundaryRevision": 1}).json() == first.json()
    assert (
        client.put(f"/api/v1/land/{area['id']}", json={**BODY, "expectedRevision": 1}).status_code
        == 200
    )
    assert client.post(url, json={"boundaryRevision": 1}).status_code == 409
    second = client.post(url, json={"boundaryRevision": 2}).json()
    assert second["investigation"]["id"] != first.json()["investigation"]["id"]
    assert second["run"]["id"] != first.json()["run"]["id"]
