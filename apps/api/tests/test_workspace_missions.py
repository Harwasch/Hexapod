from __future__ import annotations

import uuid
from typing import Any

from fastapi.testclient import TestClient

from tests.test_land import BODY
from tests.test_land_actions import setup_action
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client

MISSIONS = "/api/v1/workspaces/missions"


def plan_body(project_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "projectId": project_id,
        "title": "Weekly fence check",
        "objective": "Fly the north fence every week",
        "goal": "monitor the fence every week",
        "cadence": "weekly",
        "startDate": "2026-10-16",
        "zoneIds": ["L-0000aaaa"],
        "areas": [
            {
                "id": "L-0000aaaa",
                "name": "North fence",
                "footprint": BODY["boundary"],
                "attribution": "Inventory · confirmed · revision 2 · Drawn on map",
            }
        ],
        "estimates": {"acres": 0, "machineHours": 0.1, "calendarDays": 1, "pathM": 300},
    }
    body.update(overrides)
    return body


def schedule_action(client: TestClient) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A land with one approved action scheduled under the land's own project."""
    land, path, body, action = setup_action(client)
    identifier = f"{path}/{action['id']}"
    body["constraints"] = [
        {"text": "Confirm access", "resolved": True, "resolution": "Permission recorded"}
    ]
    revised = client.put(identifier, json={**body, "expectedRevision": 1, "note": "Resolved"})
    assert revised.status_code == 200, revised.text
    review = {"expectedRevision": 2, "note": "Reviewed"}
    assert client.post(identifier + "/approve", json=review).status_code == 200
    scheduled = client.post(
        identifier + "/mission", json={**review, "projectId": f"land:{land['id']}"}
    )
    assert scheduled.status_code == 201, scheduled.text
    return land, action, scheduled.json()


def test_land_scheduled_missions_are_listed_and_acted_on_privately(client: TestClient) -> None:
    land, action, mission = schedule_action(client)
    project = f"land:{land['id']}"
    listed = client.get(MISSIONS, params={"projectId": project})
    assert listed.status_code == 200, listed.text
    [found] = listed.json()
    assert found["id"] == mission["id"] and found["landId"] == land["id"]
    assert found["boundaryRevision"] == 1 and found["private"] is True
    assert found["landActionId"] == action["id"]
    assert [p["id"] for p in client.get(MISSIONS, params={"landId": land["id"]}).json()] == [
        mission["id"]
    ]
    assert client.get(f"{MISSIONS}/{mission['id']}").json()["id"] == mission["id"]
    # The action holds the approved version; the mission is not revised behind its back.
    revise = {k: v for k, v in plan_body(project).items() if k != "projectId"}
    conflict = client.put(f"{MISSIONS}/{mission['id']}", json=revise)
    assert conflict.status_code == 409 and "land action" in conflict.json()["detail"]
    status = client.patch(f"{MISSIONS}/{mission['id']}/status", json={"status": "dispatched"})
    assert status.status_code == 200 and status.json()["status"] == "dispatched"
    # The public routes keep their guarantees.
    assert client.get("/api/v1/plans", params={"projectId": project}).json() == []
    assert client.get(f"/api/v1/plans/{mission['id']}").status_code == 404


def test_missions_on_land_are_created_private_and_checked_against_the_land(
    client: TestClient,
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    project = f"land:{land['id']}"
    created = client.post(
        MISSIONS, json={**plan_body(project), "landId": land["id"], "boundaryRevision": 1}
    )
    assert created.status_code == 201, created.text
    plan = created.json()
    assert plan["private"] is True and plan["landId"] == land["id"]
    assert plan["boundaryRevision"] == 1 and plan["landActionId"] is None
    assert plan["areas"][0]["attribution"].startswith("Inventory")
    assert plan["estimates"]["pathM"] == 300
    revise = {
        **{k: v for k, v in plan_body(project).items() if k != "projectId"},
        "title": "Fortnightly fence check",
        "note": "Less often",
    }
    revised = client.put(f"{MISSIONS}/{plan['id']}", json=revise)
    assert revised.status_code == 200 and revised.json()["revision"] == 2
    other = f"land:{uuid.uuid4()}"
    assert client.post(MISSIONS, json={**plan_body(other), "landId": land["id"]}).status_code == 422
    assert client.post(MISSIONS, json=plan_body(project)).status_code == 422
    assert (
        client.post(MISSIONS, json={**plan_body(project), "boundaryRevision": 1}).status_code == 422
    )
    unknown = str(uuid.uuid4())
    assert (
        client.post(MISSIONS, json={**plan_body(f"land:{unknown}"), "landId": unknown}).status_code
        == 404
    )
    assert (
        client.post(
            MISSIONS, json={**plan_body(project), "landId": land["id"], "boundaryRevision": 7}
        ).status_code
        == 404
    )
    # A private plan on an ordinary project (a demo ranch overlapping the land).
    layered = client.post(MISSIONS, json={**plan_body("demo-ranch"), "landId": land["id"]})
    assert layered.status_code == 201, layered.text
    assert [p["id"] for p in client.get(MISSIONS, params={"projectId": "demo-ranch"}).json()] == [
        layered.json()["id"]
    ]
    assert client.get("/api/v1/plans", params={"projectId": "demo-ranch"}).json() == []
    assert client.get("/api/v1/plans", params={"projectId": project}).json() == []


def test_a_limited_mission_list_keeps_the_newest(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    project = f"land:{land['id']}"
    ids = [
        client.post(
            MISSIONS, json={**plan_body(project, title=f"Check {i}"), "landId": land["id"]}
        ).json()["id"]
        for i in range(3)
    ]
    listed = client.get(MISSIONS, params={"projectId": project, "limit": 2})
    assert [p["id"] for p in listed.json()] == ids[1:]
    assert listed.headers["x-truncated"] == "true"
    whole = client.get(MISSIONS, params={"projectId": project})
    assert [p["id"] for p in whole.json()] == ids
    assert "x-truncated" not in whole.headers


def test_missions_follow_workspace_roles_and_isolation(identity_client: IdentityClient) -> None:
    client, token = identity_client
    alice = headers(token())
    workspace = client.post("/api/v1/workspaces", headers=alice, json={"name": "Ranch"}).json()[
        "id"
    ]
    alice = headers(token(), workspace)
    land = client.post("/api/v1/land", json=BODY, headers=alice).json()
    project = f"land:{land['id']}"
    created = client.post(
        MISSIONS, headers=alice, json={**plan_body(project), "landId": land["id"]}
    )
    assert created.status_code == 201, created.text
    plan = created.json()
    from app.services.identity import principal_id
    from tests.test_workspaces import ISSUER

    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=alice,
            json={"principalId": principal_id(ISSUER, "bob"), "role": "viewer"},
        ).status_code
        == 200
    )
    viewer = headers(token("bob"), workspace)
    assert client.get(MISSIONS, headers=viewer, params={"projectId": project}).status_code == 200
    assert client.get(f"{MISSIONS}/{plan['id']}", headers=viewer).status_code == 200
    assert (
        client.post(
            MISSIONS, headers=viewer, json={**plan_body(project), "landId": land["id"]}
        ).status_code
        == 403
    )
    assert (
        client.patch(
            f"{MISSIONS}/{plan['id']}/status", headers=viewer, json={"status": "dispatched"}
        ).status_code
        == 403
    )
    carol = headers(token("carol"))
    elsewhere = client.post("/api/v1/workspaces", headers=carol, json={"name": "Other"}).json()[
        "id"
    ]
    carol = headers(token("carol"), elsewhere)
    assert client.get(MISSIONS, headers=carol, params={"projectId": project}).json() == []
    assert client.get(f"{MISSIONS}/{plan['id']}", headers=carol).status_code == 404
    assert (
        client.post(
            MISSIONS, headers=carol, json={**plan_body(project), "landId": land["id"]}
        ).status_code
        == 404
    )
