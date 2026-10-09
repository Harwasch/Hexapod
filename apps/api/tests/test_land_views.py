from __future__ import annotations

import uuid
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.research import ResearchArtifact
from app.research import queue
from app.schemas.research import ArtifactContent
from tests.test_land import BODY
from tests.test_research import evidence, start
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client


def payload() -> dict[str, Any]:
    return {
        "name": "View",
        "requestKey": str(uuid.uuid4()),
        "state": {
            "boundaryRevision": 1,
            "camera": {
                "longitude": -77.05,
                "latitude": 38.89,
                "height": 700,
                "heading": 0,
                "pitch": -80,
                "roll": 0,
            },
        },
    }


def test_saved_view_idempotency_rename_conflicts_and_boundary_notice(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/views"
    body = payload()
    response = client.post(path, json=body)
    assert response.status_code == 201, response.text
    view = response.json()
    assert client.post(path + "/recover", json=body).json()["id"] == view["id"]
    assert (
        client.post(path + "/recover", json={**body, "name": "Different capture"}).status_code
        == 409
    )
    assert client.post(path + "/recover", json=payload()).status_code == 404
    assert client.post(path, json=body).json()["id"] == view["id"]
    assert client.post(path, json={**body, "name": "Different"}).status_code == 409
    assert len(client.get(path).json()) == 1
    assert (
        client.patch(
            f"{path}/{view['id']}", json={"name": " Renamed ", "expectedRevision": 1}
        ).json()["name"]
        == "Renamed"
    )
    assert (
        client.patch(
            f"{path}/{view['id']}", json={"name": "Lost edit", "expectedRevision": 1}
        ).status_code
        == 409
    )
    assert client.delete(f"{path}/{view['id']}?expected_revision=1").status_code == 409
    assert client.post(path, json=body).json()["name"] == "Renamed"
    assert client.post(path + "/recover", json=body).json()["name"] == "Renamed"
    assert (
        client.put(f"/api/v1/land/{land['id']}", json={**BODY, "expectedRevision": 1}).status_code
        == 200
    )
    opened = client.get(f"{path}/{view['id']}").json()
    assert opened["state"]["boundaryRevision"] == 1
    assert "current boundary is revision 2" in opened["warnings"][0]
    assert client.delete(f"{path}/{view['id']}?expected_revision=2").status_code == 204
    assert client.get(f"/api/v1/land/{land['id']}").status_code == 200


def test_map_references_are_private_resolved_and_missing_sources_are_reported(
    client: TestClient, db: Session
) -> None:
    land, inv, _, _ = start(client)
    claimed = queue.claim(db)
    assert claimed is not None
    run_id, lease = claimed
    evidence_id = queue.save_evidence(db, run_id, lease, "view-evidence", evidence())
    artifact_id = queue.save_artifact(
        db,
        run_id,
        lease,
        "map",
        ArtifactContent.model_validate(
            {
                "title": "Cited map",
                "method": "Synthetic fixture",
                "evidenceIds": [str(evidence_id)],
                "output": {
                    "kind": "map",
                    "features": [
                        {
                            "label": "Study point",
                            "geometry": {"type": "Point", "coordinates": [-77.05, 38.89]},
                            "value": None,
                        }
                    ],
                    "legend": "Synthetic",
                },
            }
        ),
    )
    body = payload()
    body["state"].update(artifactIds=[str(artifact_id)], investigationId=inv["id"])
    path = f"/api/v1/land/{land['id']}/views"
    saved = client.post(path, json=body)
    assert saved.status_code == 201, saved.text
    url = f"{path}/{saved.json()['id']}"
    opened = client.get(url).json()
    assert opened["maps"][0]["features"][0]["geometry"]["coordinates"] == [-77.05, 38.89]
    other = client.post("/api/v1/land", json={**BODY, "name": "Other land"}).json()
    assert client.post(f"/api/v1/land/{other['id']}/views", json=body).status_code == 422
    artifact = db.get(ResearchArtifact, artifact_id)
    assert artifact is not None
    db.delete(artifact)
    db.commit()
    opened = client.get(url).json()
    assert opened["maps"] == [] and opened["state"]["artifactIds"] == []
    assert any("unavailable" in warning for warning in opened["warnings"])


def test_view_workspace_roles_and_scope(identity_client: IdentityClient) -> None:
    client, token = identity_client
    workspace = client.post(
        "/api/v1/workspaces", headers=headers(token()), json={"name": "One"}
    ).json()["id"]
    owner = headers(token(), workspace)
    land = client.post("/api/v1/land", headers=owner, json=BODY).json()
    path = f"/api/v1/land/{land['id']}/views"
    created = client.post(path, headers=owner, json=payload()).json()
    bob = headers(token("bob"))
    identity = client.get("/api/v1/workspaces/identity", headers=bob).json()["principalId"]
    other = client.post("/api/v1/workspaces", headers=bob, json={"name": "Two"}).json()["id"]
    assert (
        client.get(f"{path}/{created['id']}", headers=headers(token("bob"), other)).status_code
        == 404
    )
    assert (
        client.put(
            "/api/v1/workspaces/members",
            headers=owner,
            json={"principalId": identity, "role": "viewer"},
        ).status_code
        == 200
    )
    viewer = headers(token("bob"), workspace)
    assert client.get(f"{path}/{created['id']}", headers=viewer).status_code == 200
    assert client.post(path, headers=viewer, json=payload()).status_code == 403
    assert (
        client.patch(
            f"{path}/{created['id']}", headers=viewer, json={"name": "No", "expectedRevision": 1}
        ).status_code
        == 403
    )
    assert (
        client.delete(f"{path}/{created['id']}?expected_revision=1", headers=viewer).status_code
        == 403
    )


def test_raster_band_opacity_and_unavailable_reference(client: TestClient, db: Session) -> None:
    from app.analysis.terrain import analyze
    from app.models.land_raster import LandRaster
    from app.schemas.land_rasters import RasterRequest
    from tests.test_terrain import BOUNDARY, fixture_client

    land, _, run, _ = start(client)
    with fixture_client() as source:
        terrain = analyze(BOUNDARY, RasterRequest(), source)
    row = LandRaster(
        land_id=uuid.UUID(land["id"]),
        boundary_revision=1,
        run_id=uuid.UUID(run["id"]),
        request=RasterRequest().model_dump(mode="json"),
        metadata_json=terrain.metadata.model_dump(mode="json"),
        sha256="a" * 64,
        byte_size=1,
    )
    db.add(row)
    db.commit()
    path = f"/api/v1/land/{land['id']}/views"
    body = payload()
    body["state"]["rasters"] = [{"id": str(row.id), "kind": "raster", "band": 2, "opacity": 0.35}]
    saved = client.post(path, json=body)
    assert saved.status_code == 201, saved.text
    url = f"{path}/{saved.json()['id']}"
    opened = client.get(url).json()
    assert opened["rasters"][0]["band"] == 2 and opened["rasters"][0]["opacity"] == 0.35
    assert opened["rasters"][0]["bounds"] == terrain.metadata.bounds
    body["requestKey"] = str(uuid.uuid4())
    body["state"]["rasters"][0]["band"] = 16
    assert client.post(path, json=body).status_code == 422
    db.delete(row)
    db.commit()
    opened = client.get(url).json()
    assert opened["rasters"] == [] and opened["state"]["rasters"] == []
    assert any("unavailable" in warning for warning in opened["warnings"])
