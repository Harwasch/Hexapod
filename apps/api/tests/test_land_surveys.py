from __future__ import annotations

import copy
import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.workspace import Workspace
from app.research.model import DecisionResult, ResearchDecision
from app.research.worker import ResearchWorker
from app.schemas.geojson import Polygon
from app.schemas.land_surveys import SurveyCreate
from app.services import land_surveys
from app.services.errors import InvalidInputError, NotFoundError
from tests.test_scenarios import RESTORATION


def rectangle(west: float, east: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, 38.887], [east, 38.887], [east, 38.889], [west, 38.889], [west, 38.887]]
        ],
    }


BOUNDARY = rectangle(-77.055, -77.045)
BODY: dict[str, Any] = {
    "name": "Public National Mall software fixture",
    "boundary": BOUNDARY,
    "source": {"method": "drawn", "label": "Synthetic boundary"},
}


def payload() -> dict[str, Any]:
    return {
        "requestKey": str(uuid.uuid4()),
        "name": "Synthetic species survey",
        "boundaryRevision": 1,
        "observedOn": "2025-07-01",
        "observer": "Software fixture",
        "method": "visual-cover",
        "design": "purposive",
        "assessedStrata": ["herb", "canopy"],
        "methodNotes": "Synthetic observations, not real National Mall ecology.",
        "plots": [
            {
                "label": "A",
                "boundary": rectangle(-77.055, -77.053),
                "observations": [
                    {
                        "taxon": "Species A",
                        "stratum": "herb",
                        "identification": "tentative",
                        "percentCover": 80,
                    },
                    {
                        "taxon": "Species B",
                        "stratum": "canopy",
                        "identification": "verified",
                        "percentCover": 90,
                    },
                ],
            },
            {
                "label": "B",
                "boundary": rectangle(-77.053, -77.049),
                "observations": [
                    {
                        "taxon": "Species A",
                        "stratum": "herb",
                        "identification": "tentative",
                        "percentCover": 20,
                    }
                ],
            },
        ],
    }


def test_area_weighting_overlapping_strata_and_missing_observations() -> None:
    body = payload()
    summary = land_surveys.summarize(
        Polygon.model_validate(BOUNDARY), SurveyCreate.model_validate(body)
    )
    a, b = summary.species
    assert summary.sampled_fraction == pytest.approx(0.6, rel=1e-6)
    assert a.mean_percent == pytest.approx(40, abs=1e-5)
    assert b.mean_percent == 90 and b.measured_plots == 1
    assert b.assessed_sample_fraction == pytest.approx(1 / 3, rel=1e-6)
    body["plots"][1]["completeInventory"] = True
    complete = land_surveys.summarize(
        Polygon.model_validate(BOUNDARY), SurveyCreate.model_validate(body)
    )
    assert complete.species[1].mean_percent == pytest.approx(30, abs=1e-5)
    assert complete.species[1].measured_plots == 2
    assert summary.species[0].mean_percent + summary.species[1].mean_percent > 100


def test_point_intercept_denominators_and_invalid_effort() -> None:
    body = payload()
    body["method"] = "point-intercept"
    for plot in body["plots"]:
        plot["samplePoints"] = 200
        for o in plot["observations"]:
            o["hits"] = o.pop("percentCover") * 2
    summary = land_surveys.summarize(
        Polygon.model_validate(BOUNDARY), SurveyCreate.model_validate(body)
    )
    assert summary.species[0].mean_percent == pytest.approx(40, abs=1e-5)
    body["plots"][0]["observations"][0]["hits"] = 201
    with pytest.raises(ValidationError, match="hit"):
        SurveyCreate.model_validate(body)
    body = payload()
    body["plots"][0]["observations"].append(copy.deepcopy(body["plots"][0]["observations"][0]))
    with pytest.raises(ValidationError, match="once"):
        SurveyCreate.model_validate(body)


@pytest.mark.parametrize("kind", ["overlap", "outside", "false-census", "hole"])
def test_rejects_unmeasured_or_double_counted_area(kind: str) -> None:
    body = payload()
    boundary = copy.deepcopy(BOUNDARY)
    if kind == "overlap":
        body["plots"][1]["boundary"] = body["plots"][0]["boundary"]
    elif kind == "outside":
        body["plots"][1]["boundary"] = rectangle(-77.03, -77.02)
    elif kind == "false-census":
        body["design"] = "census"
    else:
        boundary["coordinates"].append(
            [
                [-77.0548, 38.8875],
                [-77.0545, 38.8875],
                [-77.0545, 38.888],
                [-77.0548, 38.888],
                [-77.0548, 38.8875],
            ]
        )
    with pytest.raises(InvalidInputError):
        land_surveys.summarize(Polygon.model_validate(boundary), SurveyCreate.model_validate(body))


def test_immutable_private_surveys_corrections_staleness_and_scenario_links(
    client: TestClient, db: Session
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    endpoint = f"/api/v1/land/{land['id']}/surveys"
    body = payload()
    preview = client.post(endpoint + "/preview", json=body)
    assert preview.status_code == 200, preview.text
    saved = client.post(endpoint, json=body)
    assert saved.status_code == 201, saved.text
    survey = saved.json()
    assert survey["summary"] == preview.json()
    assert client.post(endpoint, json=body).json()["id"] == survey["id"]
    assert client.post(endpoint, json={**body, "name": "Changed"}).status_code == 409
    corrected = client.post(
        endpoint,
        json={
            **body,
            "requestKey": str(uuid.uuid4()),
            "supersedesId": survey["id"],
            "name": "Correction",
        },
    )
    assert corrected.status_code == 201
    assert len(client.get(endpoint).json()) == 2
    scenario = {
        "name": "Restoration",
        "boundaryRevision": 1,
        "inputs": RESTORATION,
        "fieldSurveyIds": [survey["id"]],
    }
    result = client.post(f"/api/v1/land/{land['id']}/scenarios", json=scenario)
    assert result.status_code == 201, result.text
    assert result.json()["fieldSurveyIds"] == [survey["id"]]
    foreign = Workspace(name="Foreign")
    db.add(foreign)
    db.commit()
    with pytest.raises(NotFoundError):
        land_surveys.scoped(db, foreign.id, uuid.UUID(land["id"]), uuid.UUID(survey["id"]))
    db.rollback()
    assert (
        client.put(f"/api/v1/land/{land['id']}", json={**BODY, "expectedRevision": 1}).status_code
        == 200
    )
    assert client.get(endpoint + "/" + survey["id"]).json()["stale"]
    assert (
        client.post(
            f"/api/v1/land/{land['id']}/scenarios", json={**scenario, "boundaryRevision": 2}
        ).status_code
        == 422
    )
    other = client.post("/api/v1/land", json=BODY).json()
    assert client.get(f"/api/v1/land/{other['id']}/surveys/{survey['id']}").status_code == 404
    assert (
        client.post(
            f"/api/v1/land/{other['id']}/surveys", json={**body, "supersedesId": survey["id"]}
        ).status_code
        == 404
    )


@pytest.mark.parametrize("long_page", [False, True])
def test_agent_reads_immutable_survey_evidence(
    client: TestClient, db: Session, sessions: sessionmaker[Session], long_page: bool
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    body = payload()
    if long_page:
        body["methodNotes"] = '"' * 5000
        body["plots"] = [body["plots"][0]]
        body["plots"][0]["observations"] = [
            {
                "taxon": f"{index:03d}" + "樹" * 197,
                "stratum": "herb",
                "identification": "tentative",
                "percentCover": 80,
            }
            for index in range(50)
        ]
    survey = client.post(f"/api/v1/land/{land['id']}/surveys", json=body).json()
    investigation = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Field ecology", "boundaryRevision": 1},
    ).json()
    run = client.post(
        f"/api/v1/research/investigations/{investigation['id']}/runs",
        json={
            "kind": "investigation",
            "question": "What was observed?",
            "requestKey": str(uuid.uuid4()),
        },
    )
    assert run.status_code == 201, run.text

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            data = json.loads(context)
            assert data["fieldSurveys"][0]["id"] == survey["id"]
            source = next(
                (v for v in data["retrieved"].values() if v["provider"] == "field-survey"), None
            )
            action = (
                {
                    "kind": "read_field_survey",
                    "surveyId": survey["id"],
                    "count": 50 if long_page else 1,
                }
                if source is None
                else {
                    "kind": "complete",
                    "summary": "User-recorded plot observations retrieved.",
                    "evidenceIds": source["evidenceIds"],
                }
            )
            return DecisionResult(
                ResearchDecision.model_validate(
                    {"progress": "Read observations", "action": action}
                ),
                100,
            )

    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    detail = client.get(f"/api/v1/research/investigations/{investigation['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    evidence = detail["evidence"][0]
    assert evidence["survey"]["sha256"] == survey["sha256"]
    page = json.loads(evidence["excerpt"])
    assert len(page["species"]) == 1 if not long_page else 0 < len(page["species"]) < 50
    assert page["nextOffset"] == len(page["species"])
    assert len(evidence["excerpt"]) <= 29000
    assert evidence["id"] in detail["messages"][-1]["content"]


def test_viewer_can_read_but_cannot_record_observations(client: TestClient) -> None:
    from fastapi import FastAPI

    from app.api.workspace_deps import WorkspaceAccess, workspace_access
    from app.models.workspace import PILOT_WORKSPACE_ID
    from app.services.identity import Principal

    land = client.post("/api/v1/land", json=BODY).json()
    endpoint = f"/api/v1/land/{land['id']}/surveys"
    saved = client.post(endpoint, json=payload()).json()
    assert isinstance(client.app, FastAPI)
    client.app.dependency_overrides[workspace_access] = lambda: WorkspaceAccess(
        PILOT_WORKSPACE_ID, Principal("test-viewer"), "viewer"
    )
    try:
        assert client.get(endpoint + "/" + saved["id"]).status_code == 200
        assert client.post(endpoint + "/preview", json=payload()).status_code == 200
        assert client.post(endpoint, json=payload()).status_code == 403
    finally:
        client.app.dependency_overrides.pop(workspace_access)
