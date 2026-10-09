from __future__ import annotations

import json

import httpx
from sqlalchemy import select

from app.config import Settings
from app.models.research import ResearchRun
from app.research.model import CompleteAction, DecisionResult, ResearchDecision, RetrieveAction
from app.research.providers.base import SourceContext
from app.research.providers.open_data import retrieve
from app.research.worker import ResearchWorker
from app.schemas.geojson import Polygon
from tests.test_land import BODY
from tests.test_research import start


def transport(request):
    if request.url.host == "epqs.nationalmap.gov":
        return httpx.Response(
            200,
            json={"value": "123.4", "resolution": 1, "attributes": {"AcquisitionDate": "6/5/2021"}},
        )
    if request.url.host == "power.larc.nasa.gov":
        return httpx.Response(
            200,
            json={
                "properties": {"parameter": {"ALLSKY_SFC_SW_DWN": {"JAN": 2.0, "ANN": 4.0}}},
                "header": {"range": "2001-2020", "fill_value": -999},
                "parameters": {
                    "ALLSKY_SFC_SW_DWN": {"units": "kW-hr/m^2/day", "longname": "Solar irradiation"}
                },
            },
        )
    return httpx.Response(200, json={"count": 0, "results": [], "endOfRecords": True})


def test_overview_runs_without_a_model_and_publishes_cited_outputs(client, db, sessions):
    _, inv, _run, _ = start(client)
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        worker = ResearchWorker(sessions, Settings(_env_file=None), client=http)
        assert worker.run_once()
        assert not worker.run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert len(detail["evidence"]) == 2
    assert len(detail["findings"]) == 2
    assert detail["artifacts"][0]["output"]["kind"] == "chart"
    assert detail["artifacts"][0]["output"]["series"][0]["values"][0] == 2.0
    assert detail["artifacts"][0]["output"]["series"][0]["values"][1] is None
    assert "123.4" in detail["findings"][0]["summary"]
    assert all(f["evidenceIds"] for f in detail["findings"])
    assert len(detail["messages"]) == 2


def test_provider_outage_is_partial_not_empty(client, sessions):
    _, inv, run, _ = start(client)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "partial"
    assert detail["findings"] == []
    events = client.get(f"/api/v1/research/runs/{run['id']}/events").json()
    assert [event["payload"]["status"] for event in events if event["kind"] == "source"] == [
        "unavailable"
    ] * 3


class TestModel:
    __test__ = False

    def decide(self, context, max_tokens):
        state = json.loads(context)
        if not state["retrieved"]:
            action = RetrieveAction(kind="retrieve_source", provider="usgs-elevation")
        else:
            item = state["retrieved"]["usgs-elevation"]
            action = CompleteAction(
                kind="complete",
                summary="The sampled elevation is 123.4 m; a point does not describe the full terrain.",
                evidence_ids=item["evidenceIds"],
            )
        return DecisionResult(
            ResearchDecision(progress="Checking the terrain evidence", action=action), 150
        )


def test_model_uses_tools_and_retains_citations(client, db, sessions):
    _, inv, run, _ = start(client)
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    row.kind = "investigation"
    db.commit()
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        assert ResearchWorker(
            sessions, Settings(_env_file=None), model=TestModel(), client=http
        ).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert detail["evidence"][0]["id"] in detail["messages"][-1]["content"]
    assert "123.4 m" in detail["messages"][-1]["content"]


def test_missing_model_is_explicit(client, db, sessions):
    _, _inv, run, _ = start(client)
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    row.kind = "investigation"
    db.commit()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    result = client.get(f"/api/v1/research/runs/{run['id']}").json()
    assert result["status"] == "failed"
    assert "not configured" in result["error"]


def test_gbif_preserves_record_license_and_excludes_generalized_and_outside():
    record = {
        "key": 1,
        "scientificName": "Example species",
        "decimalLongitude": -122.135,
        "decimalLatitude": 47.645,
        "license": "http://creativecommons.org/licenses/by/4.0/legalcode",
        "coordinateUncertaintyInMeters": 500,
        "datasetName": "Reference survey",
        "eventDate": "2000-01-01",
    }
    results = [
        record,
        {**record, "key": 2, "license": "http://creativecommons.org/licenses/by-nc/4.0/legalcode"},
        {**record, "key": 3, "informationWithheld": "Sensitive taxon"},
        {**record, "key": 4, "decimalLongitude": 0},
    ]
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"results": results, "count": 4, "endOfRecords": True}
            )
        )
    ) as http:
        result = retrieve(
            "gbif-occurrences", SourceContext(Polygon.model_validate(BODY["boundary"])), http
        )
    assert result.status == "available"
    assert len(result.evidence) == 1
    assert result.evidence[0][1].license == record["license"]
    assert result.evidence[0][1].observed_at.year == 2000
    assert "decimalLongitude" not in result.evidence[0][1].excerpt
    assert "not proof" in result.evidence[0][1].relevance_note
    assert result.data["excludedGeneralized"] == 1
    assert result.data["excludedLicense"] == 1


def test_redirects_are_not_followed():
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
        )
    ) as http:
        result = retrieve(
            "usgs-elevation", SourceContext(Polygon.model_validate(BODY["boundary"])), http
        )
    assert result.status == "unavailable"
