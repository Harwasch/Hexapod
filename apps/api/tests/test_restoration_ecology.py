from __future__ import annotations

import copy
import math
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.restoration_ecology import evaluate, trajectory
from app.schemas.restoration_ecology import CoverResponse, RestorationEcology
from app.schemas.scenarios import RestorationInputs
from app.services.errors import InvalidInputError
from tests.test_land_surveys import BODY
from tests.test_land_surveys import payload as survey_payload
from tests.test_scenarios import RESTORATION


def target(**overrides: Any) -> dict[str, Any]:
    return {
        "taxon": "Species A",
        "stratum": "herb",
        "baselinePercent": 20,
        "baselineBasis": "Hypothetical software fixture",
        "targetLow": 60,
        "targetHigh": 80,
        "targetYear": 5,
        "targetBasis": "hypothetical",
        "rationale": "Illustrative target only",
        "monitoringMethod": "Repeat the mapped quadrats and identification review",
        "monitoringSeason": "Same summer sampling window",
        "responseIfOffTrack": "Check detection and disturbance before changing treatment",
        **overrides,
    }


def plan(*targets: dict[str, Any]) -> dict[str, Any]:
    return {
        "referenceBasis": "A local reference community still needs verification",
        "siteConstraints": "Check hydrology, soils and disturbance",
        "speciesTargets": list(targets or (target(),)),
    }


def response(**overrides: Any) -> CoverResponse:
    return CoverResponse.model_validate(
        {
            "startYear": 1,
            "asymptoteLow": 100,
            "asymptoteHigh": 100,
            "annualRateLow": math.log(2),
            "annualRateHigh": math.log(2),
            "basis": "Synthetic half-time, not an ecological measurement",
            **overrides,
        }
    )


def test_conditional_response_has_known_half_time_start_delay_and_zero_rate() -> None:
    rows = trajectory(20, response(), 4)
    assert [row.low for row in rows] == pytest.approx([20, 20, 60, 80, 90])
    assert all(row.low == row.high for row in rows)
    assert trajectory(70, response(annualRateLow=0, annualRateHigh=0), 50)[-1].high == 70
    decline = trajectory(80, response(startYear=0, asymptoteLow=0, asymptoteHigh=0), 2)
    assert [row.low for row in decline] == pytest.approx([80, 40, 20])


@pytest.mark.parametrize("bounds", [(0, 30), (70, 100), (10, 90)])
def test_response_envelope_contains_interior_parameter_combinations(
    bounds: tuple[int, int],
) -> None:
    p = response(
        startYear=0,
        asymptoteLow=bounds[0],
        asymptoteHigh=bounds[1],
        annualRateLow=0.1,
        annualRateHigh=0.9,
    )
    rows = trajectory(50, p, 10)
    for row in rows:
        for step in range(11):
            asymptote = bounds[0] + (bounds[1] - bounds[0]) * step / 10
            for rate in (0.1, 0.3, 0.5, 0.7, 0.9):
                value = asymptote + (50 - asymptote) * math.exp(-rate * row.year)
                assert row.low - 1e-10 <= value <= row.high + 1e-10


def test_unknown_baselines_independent_strata_and_invalid_links() -> None:
    p = RestorationEcology.model_validate(
        plan(
            target(baselinePercent=None),
            target(
                taxon="Species B",
                stratum="canopy",
                baselinePercent=90,
                targetLow=90,
                targetHigh=100,
            ),
        )
    )
    result = evaluate(p, {})
    assert result.species[0].baseline_percent is None
    assert result.species[0].projection == []
    assert result.species[1].target.target_low == 90
    assert result.species[0].target.target_high + result.species[1].target.target_high == 180
    with pytest.raises(InvalidInputError, match="baseline"):
        evaluate(
            RestorationEcology.model_validate(
                plan(target(baselinePercent=None, response=response().model_dump()))
            ),
            {},
        )
    with pytest.raises(ValidationError, match="citation"):
        RestorationEcology.model_validate(plan(target(targetBasis="reference-evidence")))
    with pytest.raises(ValidationError, match="existing treatment"):
        RestorationInputs.model_validate(
            {
                **RESTORATION,
                "monitoringYears": [5],
                "ecology": plan(target(treatmentNames=["Invented"])),
            }
        )
    with pytest.raises(ValidationError, match="monitoring visit"):
        RestorationInputs.model_validate({**RESTORATION, "ecology": plan()})


def test_species_scenario_pins_survey_scope_and_rejects_unknown_or_foreign_sources(
    client: TestClient, db: Session
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    survey = client.post(f"/api/v1/land/{land['id']}/surveys", json=survey_payload()).json()
    p: dict[str, Any] = {
        "name": "Restoration targets",
        "boundaryRevision": 1,
        "requestKey": str(uuid.uuid4()),
        "inputs": {
            **RESTORATION,
            "monitoringYears": [1, 2, 5],
            "ecology": plan(
                target(
                    baselineSurveyId=survey["id"],
                    baselinePercent=None,
                    response=response().model_dump(mode="json"),
                ),
                target(
                    taxon="Species B",
                    stratum="canopy",
                    baselineSurveyId=survey["id"],
                    baselinePercent=None,
                    targetLow=80,
                    targetHigh=95,
                ),
            ),
        },
    }
    path = f"/api/v1/land/{land['id']}/scenarios"
    saved = client.post(path, json=p)
    assert saved.status_code == 201, saved.text
    data = saved.json()
    a, b = data["result"]["ecology"]["species"]
    assert a["baselinePercent"] == pytest.approx(40, abs=1e-5)
    assert a["assessedLandFraction"] == pytest.approx(0.6, abs=1e-5)
    assert b["baselinePercent"] == 90 and b["assessedLandFraction"] == pytest.approx(0.2, abs=1e-5)
    assert a["surveySha256"] == survey["sha256"] and a["identificationStatus"] == ["tentative"]
    assert a["baselineScope"] == "surveyed-plots"
    assert client.post(path, json=p).json()["id"] == data["id"]
    from app.models.land import LandArea
    from app.research.scenarios import read as read_scenario
    from app.services.errors import NotFoundError

    land_row = db.get(LandArea, uuid.UUID(land["id"]))
    assert land_row
    overview = read_scenario(
        db, land_row.workspace_id, land_row.id, uuid.UUID(data["id"]), 1, "overview", 0, 3
    )
    assert overview["counts"]["species-targets"] == 2
    first_page = read_scenario(
        db, land_row.workspace_id, land_row.id, uuid.UUID(data["id"]), 1, "species-results", 0, 1
    )
    assert first_page["data"][0]["baseline_percent"] == pytest.approx(40, abs=1e-5)
    assert first_page["nextOffset"] == 1
    assert first_page["snapshotSha256"] == overview["snapshotSha256"]
    with pytest.raises(NotFoundError):
        read_scenario(db, uuid.uuid4(), land_row.id, uuid.UUID(data["id"]), 1, "overview", 0, 3)
    with pytest.raises(NotFoundError):
        read_scenario(
            db, land_row.workspace_id, land_row.id, uuid.UUID(data["id"]), 99, "overview", 0, 3
        )
    changed = copy.deepcopy(p)
    changed["inputs"]["ecology"]["speciesTargets"][0]["taxon"] = "Unrecorded taxon"
    assert client.post(path + "/preview", json=changed).status_code == 422
    changed = copy.deepcopy(p)
    changed["inputs"]["ecology"]["referenceEvidenceIds"] = [str(uuid.uuid4())]
    assert client.post(path + "/preview", json=changed).status_code == 422
    other = client.post("/api/v1/land", json={**BODY, "name": "Other land"}).json()
    assert client.post(f"/api/v1/land/{other['id']}/scenarios/preview", json=p).status_code == 404
    revision = {**BODY, "expectedRevision": 1}
    assert client.put(f"/api/v1/land/{land['id']}", json=revision).status_code == 200
    assert client.post(path + "/preview", json={**p, "boundaryRevision": 2}).status_code == 422
    old = client.get(path + "/" + data["id"]).json()
    assert old["stale"] and old["result"] == data["result"]


def test_evidence_picker_is_land_scoped_and_searches_literal_titles(
    client: TestClient, db: Session
) -> None:
    from app.research import queue
    from tests.test_research import evidence, start

    land, inv, _run, _body = start(client)
    claimed = queue.claim(db)
    assert claimed
    run_id, token = claimed
    source = evidence().model_copy(update={"title": "Reference cover: 50% plant"})
    eid = queue.save_evidence(db, run_id, token, "reference", source)
    queue.finish(db, run_id, token, "succeeded", "Source available")
    result = client.get(f"/api/v1/land/{land['id']}/evidence", params={"query": "%"})
    assert result.status_code == 200
    assert result.json()[0]["id"] == str(eid)
    assert result.json()[0]["investigationId"] == inv["id"]
    assert "excerpt" not in result.json()[0]
    assert client.get(f"/api/v1/land/{land['id']}/evidence", params={"query": "_"}).json() == []
    other = client.post("/api/v1/land", json=BODY).json()
    assert client.get(f"/api/v1/land/{other['id']}/evidence").json() == []
    assert client.get(f"/api/v1/land/{uuid.uuid4()}/evidence").status_code == 404


@pytest.mark.parametrize("foreign_citation", [False, True])
def test_agent_species_scenarios_validate_nested_citations(
    client: TestClient, db: Session, sessions: sessionmaker[Session], foreign_citation: bool
) -> None:
    import json

    from app.config import Settings
    from app.research import queue
    from app.research.model import DecisionResult, ResearchDecision
    from app.research.worker import ResearchWorker
    from tests.test_research import evidence, start

    land, inv, _run, _body = start(client)
    claimed = queue.claim(db)
    assert claimed
    run_id, token = claimed
    eid = queue.save_evidence(db, run_id, token, "reference", evidence())
    queue.finish(db, run_id, token, "succeeded", "Reference source saved")
    citation = str(uuid.uuid4() if foreign_citation else eid)
    inputs = {
        **RESTORATION,
        "monitoringYears": [1, 2, 5],
        "ecology": plan(
            target(
                targetBasis="reference-evidence",
                evidenceIds=[citation],
                treatmentNames=["Planting"],
            )
        ),
    }
    if not foreign_citation:
        inputs["ecology"]["speciesTargets"].extend(
            [
                target(
                    taxon=f"Species {i}",
                    baselineBasis="b" * 2000,
                    rationale="r" * 3000,
                    monitoringMethod="m" * 2000,
                    responseIfOffTrack="a" * 2000,
                )
                for i in range(3)
            ]
        )
    response = client.post(
        f"/api/v1/research/investigations/{inv['id']}/runs",
        json={
            "requestKey": str(uuid.uuid4()),
            "question": "Save the supplied species goal as a scenario",
        },
    )
    assert response.status_code == 201

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            state = json.loads(context)
            action: dict[str, Any]
            if not state["previousActions"]:
                action = {
                    "kind": "create_scenario",
                    "name": "Agent species goals",
                    "inputs": inputs,
                }
            elif not foreign_citation and len(state["previousActions"]) == 1:
                saved_action = json.loads(state["previousActions"][0]["result"])
                action = {
                    "kind": "read_scenario",
                    "scenarioId": saved_action["scenarioId"],
                    "revision": saved_action["revision"],
                    "section": "species-targets",
                    "count": 10,
                }
            else:
                if not foreign_citation:
                    saved_source = next(
                        v for v in state["retrieved"].values() if v["provider"] == "saved-scenario"
                    )
                    assert saved_source["data"]["data"][0]["taxon"] == "Species A"
                    assert saved_source["data"]["nextOffset"] is not None
                    assert len(json.dumps(saved_source["data"], ensure_ascii=False)) <= 29_000
                action = {"kind": "complete", "summary": "The scenario request was evaluated."}
            return DecisionResult(
                ResearchDecision.model_validate(
                    {"progress": "Reviewing supplied goals", "action": action}
                ),
                100,
            )

    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert all(item["status"] == "succeeded" for item in detail["runs"])
    saved = client.get(f"/api/v1/land/{land['id']}/scenarios").json()
    assert len(saved) == (0 if foreign_citation else 1)
    if saved:
        assert saved[0]["result"]["ecology"]["species"][0]["target"]["evidenceIds"] == [str(eid)]
