from __future__ import annotations

from fastapi.testclient import TestClient

BODY = {
    "projectId": "blackrock-mesa",
    "title": "Mow Z-21 weekly",
    "objective": "Mow the north fence weekly.",
    "goal": "Mow Z-21 weekly with one mower",
    "cadence": "weekly",
    "startDate": "2026-09-17",
    "endDate": None,
    "zoneIds": ["Z-21"],
    "machineIds": ["TR-04"],
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
        body["revisions"][0]["zoneIds"] == ["Z-21"] and body["revisions"][1]["note"] == "Added Z-14"
    )

    assert client.delete(f"/api/v1/plans/{plan['id']}").status_code == 204
    assert client.get(f"/api/v1/plans/{plan['id']}").status_code == 404


def test_plan_requires_known_site(client: TestClient) -> None:
    response = client.post(
        "/api/v1/plans", json={**BODY, "siteId": "11111111-1111-4111-8111-111111111111"}
    )
    assert response.status_code == 404
