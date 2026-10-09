from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.models.workspace import Workspace
from app.schemas.land_features import FeatureInspectionCreate
from app.services import land_features
from app.services.errors import NotFoundError
from tests.test_land import BODY

FEATURE: dict[str, Any] = {
    "name": "Transmission pole",
    "category": "power",
    "geometry": {"type": "Point", "coordinates": [-122.135, 47.645, 20]},
    "source": {
        "method": "mapped-feature",
        "label": "Mapped pole",
        "url": "https://www.openstreetmap.org/node/9",
        "recordId": "9",
        "meaning": "physical-feature",
    },
    "status": "candidate",
}


def test_feature_identity_revision_inspection_and_private_scope(
    client: TestClient, db: Session
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    payload = {**FEATURE, "requestKey": str(uuid.uuid4())}
    response = client.post(path, json=payload)
    assert response.status_code == 201, response.text
    feature = response.json()
    assert feature["geometry"]["coordinates"] == [-122.135, 47.645]
    assert feature["intersectsLand"] and feature["distanceM"] == 0
    assert client.post(path, json=payload).json()["id"] == feature["id"]
    assert client.post(path, json={**payload, "requestKey": str(uuid.uuid4())}).status_code == 409
    changed = client.put(
        f"{path}/{feature['id']}",
        json={
            **payload,
            "expectedRevision": 1,
            "note": "Confirmed in the field",
            "status": "confirmed",
        },
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["revision"] == 2
    assert (
        client.put(
            f"{path}/{feature['id']}", json={**payload, "expectedRevision": 1, "note": "Stale edit"}
        ).status_code
        == 409
    )
    history = client.get(f"{path}/{feature['id']}/revisions").json()
    assert history[1]["content"]["status"] == "candidate"
    inspection = {
        "requestKey": str(uuid.uuid4()),
        "observedAt": "2026-10-09T00:00:00Z",
        "condition": "fair",
        "notes": "Paint corrosion at base",
        "measurements": {"height": 20},
        "measurementUnits": {"height": "m"},
    }
    inspection_path = f"{path}/{feature['id']}/inspections"
    checked = client.post(inspection_path, json=inspection)
    assert checked.status_code == 201, checked.text
    assert checked.json()["featureRevision"] == 2
    assert client.post(inspection_path, json=inspection).json()["id"] == checked.json()["id"]
    assert len(client.get(inspection_path).json()) == 1
    private = Workspace(name="Private")
    db.add(private)
    db.commit()
    with pytest.raises(NotFoundError):
        land_features.scoped(db, private.id, uuid.UUID(land["id"]), uuid.UUID(feature["id"]))


def test_feature_outside_boundary_is_explicit_and_evidence_is_checked(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    response = client.post(
        path, json={**FEATURE, "geometry": {"type": "Point", "coordinates": [-122.12, 47.645]}}
    )
    assert response.status_code == 201, response.text
    assert not response.json()["intersectsLand"]
    assert response.json()["distanceM"] > 500
    assert (
        client.post(path, json={**FEATURE, "evidenceIds": [str(uuid.uuid4())]}).status_code == 422
    )


def test_inspections_need_units_and_timezone() -> None:
    with pytest.raises(ValidationError, match="unit"):
        FeatureInspectionCreate(
            observed_at="2026-10-09T00:00:00Z",
            condition="unknown",
            notes="Inspection",
            measurements={"distance": 2},
        )
    with pytest.raises(ValidationError, match="timezone"):
        FeatureInspectionCreate(
            observed_at="2026-10-09T00:00:00", condition="unknown", notes="Inspection"
        )


def test_revision_cannot_duplicate_another_source_record(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    assert client.post(path, json=FEATURE).status_code == 201
    alternate = {**FEATURE, "source": {**FEATURE["source"], "recordId": "10"}}
    second = client.post(path, json=alternate).json()
    result = client.put(
        f"{path}/{second['id']}",
        json={**FEATURE, "expectedRevision": 1, "note": "Duplicate identity"},
    )
    assert result.status_code == 409
    assert client.get(f"{path}/{second['id']}").json()["revision"] == 1


def test_inventory_rejects_nonfinite_numeric_attributes() -> None:
    from app.schemas.land_features import LandFeatureCreate

    with pytest.raises(ValidationError, match="finite"):
        LandFeatureCreate.model_validate({**FEATURE, "attributes": {"height": float("inf")}})
