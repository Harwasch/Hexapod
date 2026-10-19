from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.services import plans as plan_service

BODY = {
    "projectId": "blackrock-mesa",
    "title": "Mow Z-21 weekly",
    "objective": "Mow the north fence weekly.",
    "goal": "Mow Z-21 weekly with one mower",
    "cadence": "weekly",
    "startDate": "2026-09-17",
    "endDate": None,
    "zoneIds": ["Z-21", "A-01"],
    "machineIds": ["TR-04"],
    "areas": [
        {
            "id": "A-01",
            "name": "Drawn strip",
            "footprint": {
                "type": "Polygon",
                "coordinates": [
                    [[-122.14, 47.64], [-122.13, 47.64], [-122.13, 47.65], [-122.14, 47.64]]
                ],
            },
        }
    ],
    "steps": [
        {
            "title": "Survey",
            "detail": "",
            "machineIds": ["TR-04"],
            "zoneId": "Z-21",
            "when": "Day 1",
            "startDay": 0,
            "days": 1,
        }
    ],
    "estimates": {"acres": 220, "machineHours": 147, "calendarDays": 7},
    "assumptions": ["1.5 acres per machine-hour"],
    "risks": [],
    "questions": [],
    "source": "rules",
    "model": None,
}


def test_plan_lifecycle_and_revisions(client: TestClient) -> None:
    created = client.post("/api/v1/plans", json=BODY)
    assert created.status_code == 201, created.text
    plan = created.json()
    assert plan["status"] == "scheduled" and plan["revision"] == 1
    assert plan["revisions"][0]["note"] == "Approved"
    assert plan["steps"][0]["startDay"] == 0
    assert plan["areas"][0]["id"] == "A-01" and plan["areas"][0]["footprint"]["type"] == "Polygon"

    listed = client.get("/api/v1/plans", params={"projectId": "blackrock-mesa"}).json()
    assert [p["id"] for p in listed] == [plan["id"]]
    assert client.get("/api/v1/plans", params={"projectId": "other"}).json() == []

    dispatched = client.patch(f"/api/v1/plans/{plan['id']}/status", json={"status": "dispatched"})
    assert dispatched.json()["status"] == "dispatched"
    assert (
        client.patch(f"/api/v1/plans/{plan['id']}/status", json={"status": "bogus"}).status_code
        == 422
    )

    revised_body = {k: v for k, v in BODY.items() if k != "projectId"}
    revised_body.update(
        {"title": "Mow Z-21 and Z-14 weekly", "zoneIds": ["Z-21", "Z-14"], "note": "Added Z-14"}
    )
    revised = client.put(f"/api/v1/plans/{plan['id']}", json=revised_body)
    assert revised.status_code == 200, revised.text
    body = revised.json()
    assert body["revision"] == 2 and body["zoneIds"] == ["Z-21", "Z-14"]
    assert body["status"] == "dispatched", "a revision keeps the lifecycle state"
    assert [r["revision"] for r in body["revisions"]] == [1, 2]
    assert (
        body["revisions"][0]["zoneIds"] == ["Z-21", "A-01"]
        and body["revisions"][1]["note"] == "Added Z-14"
    )

    assert client.delete(f"/api/v1/plans/{plan['id']}").status_code == 204
    assert client.get(f"/api/v1/plans/{plan['id']}").status_code == 404


def test_plan_requires_known_site(client: TestClient) -> None:
    response = client.post(
        "/api/v1/plans", json={**BODY, "siteId": "11111111-1111-4111-8111-111111111111"}
    )
    assert response.status_code == 404


def test_a_scout_area_keeps_its_capture_parameters(client: TestClient) -> None:
    capture = {
        "purpose": "lift-knowledge-fog",
        "camera": "dji-m3e-wide",
        "altitudeM": 60,
        "forwardOverlap": 0.75,
        "sideOverlap": 0.7,
        "speedMs": 8,
        "headingDeg": None,
    }
    area: dict[str, Any] = dict(BODY["areas"][0])  # type: ignore[index, arg-type]
    areas = [{**area, "capture": capture}]
    created = client.post("/api/v1/plans", json={**BODY, "areas": areas})
    assert created.status_code == 201, created.text
    stored = client.get(f"/api/v1/plans/{created.json()['id']}").json()
    assert stored["areas"][0]["capture"] == capture
    assert (
        client.get(f"/api/v1/plans/{created.json()['id']}").json()["areas"][0]["attribution"]
        is None
    )
    wrong = {**capture, "altitudeM": 400}
    rejected = client.post(
        "/api/v1/plans", json={**BODY, "areas": [{**areas[0], "capture": wrong}]}
    )
    assert rejected.status_code == 422


def test_a_long_area_name_is_clipped_not_refused(client: TestClient) -> None:
    """Areas are named after a Land's features (up to 200 characters) and derived ground adds
    to that ("…, workable"); the approval keeps a clipped name rather than failing."""
    area: dict[str, Any] = dict(BODY["areas"][0])  # type: ignore[index, arg-type]
    long_name = "North paddock along the seasonal creek " * 4
    created = client.post("/api/v1/plans", json={**BODY, "areas": [{**area, "name": long_name}]})
    assert created.status_code == 201, created.text
    name = created.json()["areas"][0]["name"]
    assert len(name) == 120 and name.endswith("…")
    assert long_name.startswith(name[:-1])


def test_a_full_list_keeps_the_newest_plans_and_says_it_left_some_out(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_service, "LIST_LIMIT", 3)
    ids = [
        client.post("/api/v1/plans", json={**BODY, "title": f"Plan {i}"}).json()["id"]
        for i in range(4)
    ]
    params = {"projectId": "blackrock-mesa"}
    listed = client.get("/api/v1/plans", params=params)
    # The plan just approved is listed; the oldest is the one left out, and the list says so.
    assert [p["id"] for p in listed.json()] == ids[1:]
    assert listed.headers["x-truncated"] == "true"
    assert client.delete(f"/api/v1/plans/{ids[0]}").status_code == 204
    whole = client.get("/api/v1/plans", params=params)
    assert [p["id"] for p in whole.json()] == ids[1:]
    assert "x-truncated" not in whole.headers
