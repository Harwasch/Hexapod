from __future__ import annotations

import copy
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.land import LandArea
from app.models.land_feature import LandFeature, LandFeatureBatch
from app.models.workspace import Workspace
from app.schemas.land_feature_batches import FeatureBatchRequest
from app.services import land_feature_batches, land_features
from app.services.errors import NotFoundError
from tests.test_land import BODY


def batch() -> dict[str, Any]:
    return {
        "requestKey": str(uuid.uuid4()),
        "boundaryRevision": 1,
        "sourceLabel": "Synthetic assets.geojson",
        "sourceFileSha256": "a" * 64,
        "rows": [
            {
                "rowId": str(i),
                "feature": {
                    "requestKey": str(uuid.uuid4()),
                    "name": f"Asset {i}",
                    "category": "equipment",
                    "geometry": {"type": "Point", "coordinates": [-122.135 + i * 0.0001, 47.645]},
                    "source": {"method": "imported", "label": "Synthetic fixture"},
                    "externalRef": {"namespace": " Survey A ", "recordId": str(i)},
                },
            }
            for i in range(3)
        ],
    }


def test_batch_preview_atomic_save_retry_and_duplicates(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    payload = batch()
    payload["rows"][2]["feature"]["externalRef"]["recordId"] = "1"
    preview = client.post(f"{path}/imports/preview", json=payload)
    assert preview.status_code == 200, preview.text
    rows = preview.json()["rows"]
    assert [row["disposition"] for row in rows] == ["created", "created", "duplicate-in-file"]
    assert rows[1]["featureId"] == rows[2]["featureId"]
    assert rows[0]["geometryPreview"]["intersectsLand"]
    assert client.get(path).json() == []
    saved = client.post(f"{path}/imports", json=payload)
    assert saved.status_code == 201, saved.text
    assert client.post(f"{path}/imports", json=payload).json() == saved.json()
    assert len(client.get(path).json()) == 2
    original = client.get(f"{path}/{rows[0]['featureId']}").json()
    assert original["externalRef"]["namespace"] == "survey a"
    # Retrying an import after the feature changed returns the original receipt, not a new write.
    changed = client.put(
        f"{path}/{original['id']}",
        json={
            **payload["rows"][0]["feature"],
            "name": "Field verified label",
            "expectedRevision": 1,
            "note": "Changed label",
        },
    )
    assert changed.status_code == 200, changed.text
    assert client.post(f"{path}/imports", json=payload).json() == saved.json()
    assert client.get(f"{path}/{original['id']}").json()["name"] == "Field verified label"
    payload["requestKey"] = str(uuid.uuid4())
    repeated = client.post(f"{path}/imports", json=payload)
    assert repeated.status_code == 201, repeated.text
    assert all(row["disposition"] == "existing" for row in repeated.json()["rows"])
    assert len(client.get(path).json()) == 2
    assert len(client.get(f"{path}/imports").json()) == 2
    payload["sourceLabel"] = "Changed request"
    assert client.post(f"{path}/imports", json=payload).status_code == 409
    # A completed receipt remains retryable even when its boundary is no longer current.
    original_request = copy.deepcopy(payload)
    original_request["sourceLabel"] = "Synthetic assets.geojson"
    revised = client.put(f"/api/v1/land/{land['id']}", json={**BODY, "expectedRevision": 1})
    assert revised.status_code == 200, revised.text
    assert client.post(f"{path}/imports", json=original_request).json() == repeated.json()
    original_request["requestKey"] = str(uuid.uuid4())
    assert client.post(f"{path}/imports", json=original_request).status_code == 409


def test_batch_rejects_stale_boundary_invalid_rows_and_duplicate_policy(client: TestClient) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    payload = batch()
    payload["boundaryRevision"] = 2
    assert client.post(f"{path}/imports", json=payload).status_code == 409
    payload["boundaryRevision"] = 1
    invalid = copy.deepcopy(payload)
    invalid["rows"][2]["feature"]["evidenceIds"] = [str(uuid.uuid4())]
    assert client.post(f"{path}/imports", json=invalid).status_code == 422
    assert client.get(path).json() == []
    invalid = copy.deepcopy(payload)
    invalid["rows"][2]["feature"]["status"] = "confirmed"
    assert client.post(f"{path}/imports", json=invalid).status_code == 422
    payload["rows"][2]["feature"]["externalRef"]["recordId"] = "1"
    payload["skipDuplicates"] = False
    assert client.post(f"{path}/imports", json=payload).status_code == 409
    assert client.get(path).json() == []


def test_batch_rolls_back_all_rows_on_late_failure_and_scopes_receipts(
    client: TestClient, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    land_id = uuid.UUID(land["id"])
    workspace_id = db.scalars(select(LandArea.workspace_id).where(LandArea.id == land_id)).one()
    payload = FeatureBatchRequest.model_validate(batch())
    create = land_features.create
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Interrupted import")
        return create(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(land_features, "create", interrupted)
        with pytest.raises(RuntimeError, match="Interrupted import"):
            land_feature_batches.create(db, workspace_id, land_id, payload)
    assert db.scalar(select(func.count()).select_from(LandFeature)) == 0
    assert db.scalar(select(func.count()).select_from(LandFeatureBatch)) == 0
    saved = land_feature_batches.create(db, workspace_id, land_id, payload)
    private = Workspace(name="Other")
    db.add(private)
    db.commit()
    with pytest.raises(NotFoundError):
        land_feature_batches.scoped(db, private.id, land_id, saved.id)
    with pytest.raises(NotFoundError):
        land_feature_batches.preview(db, private.id, land_id, payload)


def test_external_identity_cannot_be_duplicated_by_single_feature_writes(
    client: TestClient,
) -> None:
    land = client.post("/api/v1/land", json=BODY).json()
    path = f"/api/v1/land/{land['id']}/features"
    feature = batch()["rows"][0]["feature"]
    assert client.post(path, json=feature).status_code == 201
    feature["requestKey"] = str(uuid.uuid4())
    feature["externalRef"]["namespace"] = "SURVEY A"
    assert client.post(path, json=feature).status_code == 409
    # Record IDs are case-sensitive, dataset namespaces are not.
    feature["externalRef"]["recordId"] = "new"
    second = client.post(path, json=feature).json()
    feature["externalRef"]["recordId"] = "0"
    assert (
        client.put(
            f"{path}/{second['id']}",
            json={**feature, "expectedRevision": 1, "note": "Duplicate external identity"},
        ).status_code
        == 409
    )


def test_concurrent_imports_serialize_retries_and_external_identities(
    client: TestClient, db: Session
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from sqlalchemy.orm import sessionmaker

    land = client.post("/api/v1/land", json=BODY).json()
    land_id = uuid.UUID(land["id"])
    workspace_id = db.scalars(select(LandArea.workspace_id).where(LandArea.id == land_id)).one()
    factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    db.rollback()
    payload = FeatureBatchRequest.model_validate(batch())

    def run(request: FeatureBatchRequest):
        with factory() as session:
            return land_feature_batches.create(session, workspace_id, land_id, request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.submit(run, payload), executor.submit(run, payload)
        assert a.result(timeout=30) == b.result(timeout=30)
    assert db.scalar(select(func.count()).select_from(LandFeature)) == 3
    assert db.scalar(select(func.count()).select_from(LandFeatureBatch)) == 1
    db.rollback()
    alternate = batch()
    for item in alternate["rows"]:
        item["feature"]["externalRef"]["namespace"] = "Other dataset"
    first = FeatureBatchRequest.model_validate(alternate)
    second = first.model_copy(update={"request_key": uuid.uuid4()})
    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.submit(run, first), executor.submit(run, second)
        results = [a.result(timeout=30), b.result(timeout=30)]
    assert sorted(
        sum(row.disposition == "created" for row in result.rows) for result in results
    ) == [0, 3]
    assert db.scalar(select(func.count()).select_from(LandFeature)) == 6
    assert db.scalar(select(func.count()).select_from(LandFeatureBatch)) == 3
