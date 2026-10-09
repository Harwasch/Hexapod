from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.scenarios import analyze
from app.models.workspace import Workspace
from app.schemas.scenarios import RestorationInputs, SolarInputs
from app.services import scenarios
from app.services.errors import InvalidInputError, NotFoundError
from tests.test_land import BODY

SOLAR: dict[str, Any] = {
    "kind": "solar",
    "usableRoofAreaM2": 100,
    "moduleEfficiency": 0.2,
    "annualPlaneIrradiationKwhM2": 1000,
    "irradiationBasis": "Reference fixture at module plane",
    "tiltDegrees": 20,
    "azimuthDegrees": 180,
    "shadeLoss": 0,
    "systemLoss": 0,
    "degradationPerYear": 0,
    "selfConsumptionFraction": 1,
    "purchaseRatePerKwh": 0.2,
    "exportRatePerKwh": 0.1,
    "tariffEscalation": 0,
    "installedCost": 10000,
    "upfrontIncentive": 0,
    "annualMaintenanceCost": 100,
    "maintenanceEscalation": 0,
    "discountRate": 0,
    "financedFraction": 0,
    "loanInterestRate": 0,
    "loanYears": 5,
    "years": 10,
    "currency": "USD",
    "assumptions": "Deterministic test assumptions",
}
RESTORATION: dict[str, Any] = {
    "kind": "restoration",
    "referenceEcosystem": "Native meadow",
    "surveyDate": "2026-01-01",
    "surveyMethod": "Field quadrats with mutually exclusive ground-cover classes",
    "confidence": "field-survey",
    "cover": [
        {
            "name": "Native vegetation",
            "baselinePercent": 20,
            "targetPercent": 90,
            "evidenceBasis": "Quadrats",
        },
        {"name": "Other", "baselinePercent": 80, "targetPercent": 10, "evidenceBasis": "Quadrats"},
    ],
    "treatments": [
        {
            "name": "Planting",
            "areaHa": 1,
            "costPerHa": 1000,
            "year": 0,
            "objective": "Establish native cover",
        }
    ],
    "monitoringYears": [1, 2],
    "monitoringCostPerVisit": 100,
    "contingencyFraction": 0.1,
    "discountRate": 0,
    "currency": "USD",
    "assumptions": "Targets are not predictions",
}


def test_solar_cash_flow_units_financing_and_sensitivity() -> None:
    result = analyze(SolarInputs.model_validate(SOLAR), 10000)
    assert result.summary["capacityKwDc"] == 20
    assert result.summary["firstYearGenerationKwh"] == 20000
    assert result.rows[1]["netCashFlow"] == 3900
    assert result.summary["netPresentValue"] == 29000
    assert result.summary["paybackYear"] == 3
    lower_npv = result.sensitivity[0]["netPresentValue"]
    base_npv = result.summary["netPresentValue"]
    assert isinstance(lower_npv, int | float) and isinstance(base_npv, int | float)
    assert lower_npv < base_npv
    financed = analyze(
        SolarInputs.model_validate({**SOLAR, "financedFraction": 0.5, "loanInterestRate": 0}), 10000
    )
    assert financed.summary["initialEquity"] == 5000
    assert financed.summary["annualLoanPayment"] == 1000
    assert financed.rows[5]["debtService"] == 1000
    assert financed.rows[6]["debtService"] == 0
    assert financed.summary["netPresentValue"] == result.summary["netPresentValue"]
    shaded = analyze(
        SolarInputs.model_validate({**SOLAR, "shadeLoss": 0.25, "systemLoss": 0.2}), 10000
    )
    assert shaded.summary["firstYearGenerationKwh"] == 12000
    tiny_rate = analyze(
        SolarInputs.model_validate({**SOLAR, "financedFraction": 1, "loanInterestRate": 1e-16}),
        10000,
    )
    assert tiny_rate.summary["annualLoanPayment"] == pytest.approx(2000)
    with pytest.raises(InvalidInputError, match="roof area"):
        analyze(SolarInputs.model_validate(SOLAR), 10)
    with pytest.raises(ValidationError):
        SolarInputs.model_validate({**SOLAR, "discountRate": float("nan")})


def test_restoration_cover_costs_and_invalid_surveys() -> None:
    result = analyze(RestorationInputs.model_validate(RESTORATION), 20000)
    assert result.summary["areaHa"] == 2
    assert result.rows[0]["baselineHa"] == 0.4
    assert result.rows[0]["targetHa"] == 1.8
    assert result.summary["totalCost"] == pytest.approx(1320)
    assert result.summary["presentValueCost"] == pytest.approx(1320)
    with pytest.raises(ValidationError, match="100"):
        RestorationInputs.model_validate(
            {**RESTORATION, "cover": [{**RESTORATION["cover"][0], "targetPercent": 40}]}
        )
    with pytest.raises(InvalidInputError, match="treatment"):
        analyze(RestorationInputs.model_validate(RESTORATION), 100)


def test_scenario_revisions_are_pinned_private_and_immutable(
    client: TestClient, db: Session
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/scenarios"
    payload = {
        "name": "Warehouse option",
        "boundaryRevision": 1,
        "inputs": SOLAR,
        "requestKey": str(uuid.uuid4()),
    }
    preview = client.post(f"{path}/preview", json=payload)
    assert preview.status_code == 200, preview.text
    response = client.post(path, json=payload)
    assert response.status_code == 201, response.text
    scenario = response.json()
    assert client.post(path, json=payload).json()["id"] == scenario["id"]
    assert client.post(path, json={**payload, "name": "Different inputs"}).status_code == 409
    assert scenario["result"] == preview.json()
    revised = client.put(
        f"{path}/{scenario['id']}",
        json={**payload, "expectedRevision": 1, "inputs": {**SOLAR, "installedCost": 12000}},
    )
    assert revised.status_code == 200, revised.text
    assert revised.json()["revision"] == 2
    assert (
        client.put(f"{path}/{scenario['id']}", json={**payload, "expectedRevision": 1}).status_code
        == 409
    )
    history = client.get(f"{path}/{scenario['id']}/revisions").json()
    assert history[1]["inputs"]["installedCost"] == 10000
    assert history[0]["inputs"]["installedCost"] == 12000
    client.put(
        f"/api/v1/land/{land['id']}",
        json={**BODY, "expectedRevision": 1, "note": "Boundary changed"},
    )
    assert client.get(f"{path}/{scenario['id']}").json()["stale"]
    assert (
        client.post(
            path,
            json={**payload, "requestKey": str(uuid.uuid4()), "evidenceIds": [str(uuid.uuid4())]},
        ).status_code
        == 422
    )
    private = Workspace(name="Another workspace")
    db.add(private)
    db.commit()
    with pytest.raises(NotFoundError):
        scenarios.scoped(db, private.id, uuid.UUID(land["id"]), uuid.UUID(scenario["id"]))


def test_research_agent_creates_a_reproducible_scenario(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    import json

    from sqlalchemy import select

    from app.config import Settings
    from app.models.research import ResearchRun
    from app.research.model import CompleteAction, DecisionResult, ResearchDecision, ScenarioAction
    from app.research.worker import ResearchWorker
    from tests.test_research import start

    land, _inv, run, _ = start(client)
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    assert row is not None
    row.kind = "investigation"
    db.commit()

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            action: CompleteAction | ScenarioAction
            state = json.loads(context)
            action = (
                ScenarioAction(
                    kind="create_scenario",
                    name="Agent solar option",
                    inputs=SolarInputs.model_validate(SOLAR),
                )
                if not state["previousActions"]
                else CompleteAction(
                    kind="complete", summary="A hypothetical solar option is saved for review."
                )
            )
            return DecisionResult(
                ResearchDecision(progress="Calculating the supplied assumptions", action=action),
                100,
            )

    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    saved = client.get(f"/api/v1/land/{land['id']}/scenarios").json()
    assert len(saved) == 1
    assert saved[0]["name"] == "Agent solar option"
    assert saved[0]["result"]["summary"]["firstYearGenerationKwh"] == 20000
    assert saved[0]["inputs"]["assumptions"] == SOLAR["assumptions"]
