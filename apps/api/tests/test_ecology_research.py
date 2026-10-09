from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.research import ResearchRun
from app.research import queue
from app.research.model import (
    CompleteAction,
    DecisionResult,
    EvidenceReadAction,
    ResearchDecision,
    TaxonAction,
)
from app.research.providers.taxonomy import CHECKLIST, match
from app.research.worker import ResearchWorker, pack
from app.schemas.land_ecology import TaxonQuery
from tests.test_ecological_sources import taxon_client
from tests.test_land import BODY
from tests.test_solar import BOUNDARY


def transport(request: httpx.Request) -> httpx.Response:
    if request.url.host == "geodata.epa.gov":
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "attributes": {
                            "OBJECTID": 674,
                            "US_L4CODE": "65n",
                            "US_L4NAME": "Chesapeake Rolling Coastal Plain",
                        }
                    }
                ]
            },
        )
    if request.url.host == "sdmdataaccess.sc.egov.usda.gov":
        return httpx.Response(200, json={"Table": []})
    if request.url.path.endswith("/metadata"):
        return httpx.Response(
            200, json={"mainIndex": {"datasetKey": CHECKLIST, "datasetAlias": "Fixture COL index"}}
        )
    if request.url.path.endswith("/match"):
        return httpx.Response(
            200,
            json={
                "usage": {"key": "4R47Z", "name": "Quercus alba L.", "rank": "SPECIES"},
                "diagnostics": {"matchType": "EXACT"},
            },
        )
    assert request.url.params["checklistKey"] == CHECKLIST
    return httpx.Response(200, json={"results": [], "count": 0, "endOfRecords": True})


def start(
    client: TestClient, **overrides: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    land = client.post(
        "/api/v1/land", json={**BODY, "boundary": BOUNDARY.model_dump(mode="json")}
    ).json()
    inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Ecological context", "boundaryRevision": 1},
    ).json()
    body = {
        "kind": "ecology",
        "question": "Research ecological context",
        "requestKey": str(uuid.uuid4()),
        "analysis": {
            "dataset": "ecological-context",
            "taxa": [{"scientificName": "Quercus alba", "kingdom": "Plantae"}],
        },
        **overrides,
    }
    result = client.post(f"/api/v1/research/investigations/{inv['id']}/runs", json=body)
    assert result.status_code == 201, result.text
    return inv, result.json(), body


def test_ecology_run_is_typed_idempotent_and_retains_empty_source_evidence(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
) -> None:
    inv, run, body = start(client)
    endpoint = f"/api/v1/research/investigations/{inv['id']}"
    assert client.post(endpoint + "/runs", json=body).json()["id"] == run["id"]
    assert (
        client.post(
            endpoint + "/runs",
            json={**body, "analysis": {"dataset": "ecological-context", "taxa": []}},
        ).status_code
        == 409
    )
    assert client.post(endpoint + "/runs", json={**body, "kind": "solar"}).status_code == 422
    db.rollback()
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    db.expire_all()
    detail = client.get(endpoint).json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert len(detail["evidence"]) == len(detail["findings"]) == len(detail["artifacts"]) == 3
    site = next(row for row in detail["findings"] if "soil-linked" in row["title"])
    assert site["confidence"] == "uncertain" and "No ecological-class" in site["summary"]
    taxon = next(row for row in detail["evidence"] if row["provider"] == "col-taxonomy")
    assert json.loads(taxon["excerpt"])["indexObserved"]["datasetAlias"] == "Fixture COL index"
    events = client.get(f"/api/v1/research/runs/{run['id']}/events").json()
    assert [e["payload"]["status"] for e in events if e["kind"] == "source"] == [
        "available",
        "empty",
        "empty",
        "available",
    ]


@pytest.mark.parametrize("outage", [False, True])
def test_ecology_step_budget_and_provider_failure_are_partial(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    outage: bool,
) -> None:
    inv, _run, _body = start(client, budget={"maxSteps": 30 if outage else 1})
    db.rollback()
    with httpx.Client(
        transport=httpx.MockTransport(
            (lambda request: httpx.Response(503)) if outage else transport
        )
    ) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "partial"
    assert len(detail["evidence"]) == (0 if outage else 1)


def test_taxonomy_recovery_reuses_checkpoint_without_network_or_second_step(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
) -> None:
    query = TaxonQuery(scientific_name="Quercus alba", kingdom="Plantae")
    inv, _run, _body = start(
        client,
        budget={"maxSteps": 1},
        analysis={
            "dataset": "ecological-context",
            "includeEcoregions": False,
            "includeEcologicalSites": False,
            "includeOccurrences": False,
            "taxa": [query.model_dump(mode="json")],
        },
    )
    claimed = queue.claim(db)
    assert claimed
    identifier, token = claimed
    key = ResearchWorker._taxon_key(query)
    with taxon_client(
        {"usage": {"key": "4R47Z", "name": "First snapshot"}, "diagnostics": {"matchType": "EXACT"}}
    ) as http:
        source = match(query, http)
    queue.checkpoint(
        db,
        identifier,
        token,
        {
            "sources": {},
            "steps": 1,
            "actions": [],
            "output_tokens": 0,
            "ecology_task": key,
            "pending_taxonomy": {"key": key, "result": pack(source)},
        },
    )
    row = db.get(ResearchRun, identifier)
    assert row
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()

    def reject(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Recovered source must not be fetched again")

    with httpx.Client(transport=httpx.MockTransport(reject)) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert detail["runs"][0]["attempt"] == 2
    assert len(detail["evidence"]) == 1 and "First snapshot" in detail["evidence"][0]["excerpt"]


class EcologyModel:
    def __init__(self, previous: bool = False, foreign: uuid.UUID | None = None) -> None:
        self.previous, self.foreign = previous, foreign

    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        data = json.loads(context)
        action: TaxonAction | EvidenceReadAction | CompleteAction
        if not data["previousActions"]:
            if self.foreign:
                action = EvidenceReadAction(kind="read_research_evidence", evidence_id=self.foreign)
            elif self.previous:
                source = next(
                    item
                    for item in data["savedResearchEvidence"]
                    if item["provider"] == "col-taxonomy"
                )
                action = EvidenceReadAction(
                    kind="read_research_evidence", evidence_id=uuid.UUID(source["id"])
                )
            else:
                action = TaxonAction(
                    kind="match_taxon", query=TaxonQuery(scientific_name="Quercus alba")
                )
        else:
            if self.foreign:
                assert not data["retrieved"]
                action = CompleteAction(
                    kind="complete", summary="Evidence from another investigation cannot be read."
                )
            else:
                item = next(iter(data["retrieved"].values()))
                assert "Quercus alba" in json.dumps(item["data"])
                action = CompleteAction(
                    kind="complete",
                    summary="A name match does not verify field identification.",
                    evidence_ids=item["evidenceIds"],
                )
        return DecisionResult(
            ResearchDecision(progress="Reviewing source evidence", action=action), 100
        )


def test_agent_matches_names_reads_prior_evidence_and_rejects_foreign_evidence(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
) -> None:
    inv, _run, _body = start(client, kind="investigation", analysis=None)
    endpoint = f"/api/v1/research/investigations/{inv['id']}"
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        db.rollback()
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=EcologyModel(), client=http
        ).run_once()
        db.expire_all()
        detail = client.get(endpoint).json()
        assert detail["runs"][0]["status"] == "succeeded"
        evidence_id = detail["evidence"][0]["id"]
        assert evidence_id in detail["messages"][-1]["content"]
        assert (
            client.post(
                endpoint + "/runs",
                json={"requestKey": str(uuid.uuid4()), "question": "Read that name match"},
            ).status_code
            == 201
        )
        db.rollback()
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=EcologyModel(previous=True), client=http
        ).run_once()
        db.expire_all()
        detail = client.get(endpoint).json()
        assert all(run["status"] == "succeeded" for run in detail["runs"])
        assert len(detail["evidence"]) == 1 and evidence_id in detail["messages"][-1]["content"]
        other, _run, _body = start(client, kind="investigation", analysis=None)
        db.rollback()
        assert ResearchWorker(
            sessions,
            Settings(_env_file=None),
            model=EcologyModel(foreign=uuid.UUID(evidence_id)),
            client=http,
        ).run_once()
        db.expire_all()
        foreign_detail = client.get(f"/api/v1/research/investigations/{other['id']}").json()
        assert foreign_detail["evidence"] == []
        assert foreign_detail["runs"][0]["status"] == "succeeded"
        assert "cannot be read" in foreign_detail["messages"][-1]["content"]
