from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError
from shapely.geometry import shape

from app.models.workspace import Workspace
from app.schemas.land_actions import LandActionCreate
from app.services import land_actions
from app.services.errors import NotFoundError
from tests.test_land import BODY

ACTION = {
    "title": "Restore the west meadow",
    "objective": "Establish native vegetation and monitor cover",
    "boundaryRevision": 1,
    "startDate": "2026-10-10",
    "currency": "USD",
    "steps": [
        {
            "id": "survey",
            "title": "Field baseline",
            "startDay": 0,
            "days": 2,
            "estimatedCost": 500,
            "costBasis": "Surveyor quote",
            "successMeasure": "Record permanent quadrats",
        },
        {
            "id": "plant",
            "title": "Plant native meadow",
            "startDay": 2,
            "days": 3,
            "dependsOn": ["survey"],
            "successMeasure": "Document planted species and density",
        },
    ],
    "constraints": [{"text": "Confirm access with the landholder", "resolved": False}],
}


def setup_action(client):
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/actions"
    body = {**ACTION, "requestKey": str(uuid.uuid4())}
    response = client.post(path, json=body)
    assert response.status_code == 201, response.text
    return land, path, body, response.json()


def test_action_review_immutable_history_and_private_mission_handoff(client, db, alembic_config):
    land, path, body, action = setup_action(client)
    assert action["totalKnownCost"] == 500 and action["uncostedSteps"] == 1
    assert client.post(path, json=body).json()["id"] == action["id"]
    identifier = f"{path}/{action['id']}"
    review = {"expectedRevision": 1, "note": "Reviewed work and evidence"}
    assert client.post(identifier + "/approve", json=review).status_code == 422
    assert (
        client.post(identifier + "/mission", json={**review, "projectId": "demo"}).status_code
        == 422
    )
    body["constraints"] = [
        {
            "text": "Confirm access with the landholder",
            "resolved": True,
            "resolution": "Written permission recorded by operator",
        }
    ]
    revised = client.put(
        identifier, json={**body, "expectedRevision": 1, "note": "Permission confirmed"}
    )
    assert revised.status_code == 200, revised.text
    review["expectedRevision"] = 2
    approved = client.post(identifier + "/approve", json=review)
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved" and approved.json()["approvedBy"]
    response = client.post(identifier + "/mission", json={**review, "projectId": "demo"})
    assert response.status_code == 201, response.text
    mission = response.json()
    assert mission["status"] == "scheduled" and mission["endDate"] == "2026-10-14"
    assert mission["areas"][0]["footprint"] == approved.json()["effectiveBoundary"]
    assert (
        client.post(identifier + "/mission", json={**review, "projectId": "demo"}).json()["id"]
        == mission["id"]
    )
    assert client.get(identifier + "/mission").json()["id"] == mission["id"]
    from alembic import command

    with pytest.raises(RuntimeError, match="private missions"):
        command.downgrade(alembic_config, "0015")

    # Public legacy routes cannot read, mutate, enumerate or delete this private mission.
    assert client.get("/api/v1/plans", params={"projectId": "demo"}).json() == []
    assert client.get(f"/api/v1/plans/{mission['id']}").status_code == 404
    assert (
        client.patch(
            f"/api/v1/plans/{mission['id']}/status", json={"status": "dispatched"}
        ).status_code
        == 404
    )
    assert client.delete(f"/api/v1/plans/{mission['id']}").status_code == 404
    changed = client.put(
        identifier,
        json={**body, "title": "Revised target", "expectedRevision": 2, "note": "New objective"},
    ).json()
    assert (
        changed["revision"] == 3 and changed["status"] == "draft" and changed["approvedAt"] is None
    )
    history = client.get(identifier + "/revisions").json()
    assert history[1]["revision"] == 2 and history[1]["status"] == "scheduled"
    assert history[2]["constraints"][0]["resolved"] is False
    assert client.get(identifier + "/mission", params={"revision": 2}).json()["id"] == mission["id"]
    private = Workspace(name="Other workspace")
    db.add(private)
    db.commit()
    with pytest.raises(NotFoundError):
        land_actions.scoped(db, private.id, uuid.UUID(land["id"]), uuid.UUID(action["id"]))


def test_actions_reject_stale_boundary_invalid_links_and_conflicting_revisions(client):
    land, path, body, action = setup_action(client)
    identifier = f"{path}/{action['id']}"
    changed = client.put(
        f"/api/v1/land/{land['id']}",
        json={**BODY, "expectedRevision": 1, "note": "New boundary review"},
    )
    assert changed.status_code == 200, changed.text
    assert client.get(identifier).json()["staleReasons"]
    result = client.post(
        identifier + "/approve", json={"expectedRevision": 1, "note": "Old boundary"}
    )
    assert result.status_code == 409
    assert (
        client.put(
            identifier, json={**body, "expectedRevision": 2, "note": "Wrong revision"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            path, json={**body, "requestKey": str(uuid.uuid4()), "evidenceIds": [str(uuid.uuid4())]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            path,
            json={
                **body,
                "requestKey": str(uuid.uuid4()),
                "features": [{"id": str(uuid.uuid4()), "revision": 1}],
            },
        ).status_code
        == 422
    )


def test_action_exclusions_and_step_footprints_survive_mission_handoff(client):
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/actions"
    exclusion = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.136, 47.644],
                [-122.134, 47.644],
                [-122.134, 47.646],
                [-122.136, 47.646],
                [-122.136, 47.644],
            ]
        ],
    }
    body = {**ACTION, "constraints": [], "exclusions": [exclusion]}
    result = client.post(path, json=body)
    assert result.status_code == 201, result.text
    assert shape(result.json()["effectiveBoundary"]).area < shape(land["boundary"]).area
    bad = {**body, "steps": [{**ACTION["steps"][0], "footprint": exclusion}]}
    assert client.post(path, json=bad).status_code == 422
    bad["exclusions"] = [land["boundary"]]
    assert client.post(path, json=bad).status_code == 422


def test_action_dependency_cycles_and_cost_basis_are_validated():
    body = {**ACTION, "steps": [dict(item) for item in ACTION["steps"]]}
    body["steps"][0]["dependsOn"] = ["plant"]
    with pytest.raises(ValidationError, match="prerequisites"):
        LandActionCreate.model_validate(body)
    body["steps"][0]["dependsOn"] = []
    body["steps"][0]["costBasis"] = ""
    with pytest.raises(ValidationError, match="basis"):
        LandActionCreate.model_validate(body)


def test_action_preserves_scenario_revision_and_detects_changed_assumptions(client):
    from tests.test_scenarios import SOLAR

    land = client.post("/api/v1/land", json=BODY).json()
    scenario_path = f"/api/v1/land/{land['id']}/scenarios"
    scenario_body = {"name": "Roof option", "boundaryRevision": 1, "inputs": SOLAR}
    scenario = client.post(scenario_path, json=scenario_body).json()
    action_path = f"/api/v1/land/{land['id']}/actions"
    action = client.post(
        action_path,
        json={**ACTION, "constraints": [], "scenario": {"id": scenario["id"], "revision": 1}},
    ).json()
    revised = client.put(
        f"{scenario_path}/{scenario['id']}",
        json={**scenario_body, "expectedRevision": 1, "inputs": {**SOLAR, "installedCost": 12000}},
    )
    assert revised.status_code == 200, revised.text
    prior = client.get(f"{scenario_path}/{scenario['id']}", params={"revision": 1})
    assert prior.status_code == 200 and prior.json()["inputs"]["installedCost"] == 10000
    assert (
        client.get(f"{scenario_path}/{scenario['id']}", params={"revision": 99}).status_code == 404
    )
    current = client.get(f"{action_path}/{action['id']}").json()
    assert current["scenario"]["revision"] == 1 and "scenario" in current["staleReasons"][0]
    assert (
        client.post(
            f"{action_path}/{action['id']}/approve",
            json={"expectedRevision": 1, "note": "Old assumptions"},
        ).status_code
        == 409
    )


def test_agent_can_save_an_action_draft_but_cannot_approve_or_schedule(client, db, sessions):
    import json

    from sqlalchemy import select

    from app.config import Settings
    from app.models.research import ResearchRun
    from app.research.model import (
        ActionDraftAction,
        CompleteAction,
        DecisionResult,
        ResearchDecision,
    )
    from app.research.worker import ResearchWorker
    from tests.test_research import start

    land, _investigation, run, _ = start(client)
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    row.kind = "investigation"
    db.commit()

    class Model:
        def decide(self, context, max_tokens):
            state = json.loads(context)
            assert "inventoryFeatures" in state
            action = (
                ActionDraftAction(
                    kind="create_action_draft", draft=LandActionCreate.model_validate(ACTION)
                )
                if not state["previousActions"]
                else CompleteAction(
                    kind="complete",
                    summary="A draft with an unresolved access constraint is ready for review.",
                )
            )
            return DecisionResult(
                ResearchDecision(progress="Drafting work for review", action=action), 100
            )

    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    actions = client.get(f"/api/v1/land/{land['id']}/actions").json()
    assert len(actions) == 1
    assert actions[0]["status"] == "draft"
    assert actions[0]["approvedAt"] is None and actions[0]["missionId"] is None
    assert not actions[0]["constraints"][0]["resolved"]
