from __future__ import annotations

import io
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.terrain import TerrainResult, analyze
from app.config import Settings
from app.models.land_raster import LandRaster
from app.models.research import ResearchRun
from app.research import queue
from app.research.worker import ResearchWorker
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import RasterRequest
from app.schemas.research import ArtifactContent, RasterOutput
from app.services.errors import InvalidInputError
from app.services.land_rasters import TMS
from tests.test_land import BODY
from tests.test_research import evidence
from tests.test_terrain import BOUNDARY, fixture_client
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client


@pytest.fixture
def terrain(monkeypatch: pytest.MonkeyPatch) -> TerrainResult:
    with fixture_client() as client:
        result = analyze(BOUNDARY, RasterRequest(), client)

    def calculate(
        boundary: Footprint, request: RasterRequest, cancelled: threading.Event | None = None
    ) -> TerrainResult:
        return result

    monkeypatch.setattr("app.research.worker.run_raster", calculate)
    return result


def start(
    client: TestClient, auth: dict[str, str] | None = None
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    land = client.post(
        "/api/v1/land", json={**BODY, "boundary": BOUNDARY.model_dump()}, headers=auth
    ).json()
    inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Terrain", "boundaryRevision": 1},
        headers=auth,
    ).json()["id"]
    body = {
        "requestKey": str(uuid.uuid4()),
        "kind": "raster",
        "question": "Analyze surface elevation and slope",
        "analysis": {"dataset": "cop-dem-glo-30"},
    }
    response = client.post(f"/api/v1/research/investigations/{inv}/runs", json=body, headers=auth)
    assert response.status_code == 201, response.text
    return land, inv, response.json(), body


def test_raster_run_saves_cited_private_cog_tiles_samples_and_revision(
    client: TestClient, db: Session, sessions: sessionmaker[Session], terrain: TerrainResult
) -> None:
    land, inv, run, body = start(client)
    endpoint = f"/api/v1/research/investigations/{inv}/runs"
    assert client.post(endpoint, json=body).json()["id"] == run["id"]
    assert client.post(endpoint, json={**body, "analysis": None}).status_code == 422
    assert client.post(endpoint, json={**body, "kind": "overview"}).status_code == 422
    assert client.post(endpoint, json={**body, "analysis": {"resolutionM": 60}}).status_code == 409
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    assert len(detail["evidence"]) == len(detail["findings"]) == len(detail["artifacts"]) == 1
    assert detail["findings"][0]["evidenceIds"] == [detail["evidence"][0]["id"]]
    raster_id = detail["artifacts"][0]["output"]["rasterId"]
    base = f"/api/v1/land/rasters/{raster_id}"
    saved = client.get(base).json()
    assert saved["metadata"]["bands"][0]["mean"] == 0 and saved["stale"] is False
    assert len(client.get(f"/api/v1/land/{land['id']}/rasters").json()) == 1
    original = client.get(base + "/download")
    assert original.content == terrain.data
    assert original.headers["cache-control"] == "private, no-store"
    assert "authorization" in original.headers["vary"].lower()
    for point, expected in [((-122.132, 47.648), [0, 0]), ((-122.137, 47.643), [None, None])]:
        sample = client.get(base + "/sample", params={"longitude": point[0], "latitude": point[1]})
        assert sample.json()["values"] == expected, sample.text
    assert client.get(base + "/tiles/3/0/0/0.png").status_code == 404
    assert client.get(base + "/tiles/1/30/0/0.png").status_code == 404
    tile = TMS.tile(-122.132, 47.648, 14)
    image = client.get(base + f"/tiles/1/{tile.z}/{tile.x}/{tile.y}.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert sum(Image.open(io.BytesIO(image.content)).getchannel("A").histogram()[1:]) > 0
    global_tile = client.get(base + "/tiles/1/0/0/0.png")
    assert global_tile.status_code == 200
    assert Image.open(io.BytesIO(global_tile.content)).size == (256, 256)
    outside = client.get(base + "/tiles/1/14/0/0.png")
    assert Image.open(io.BytesIO(outside.content)).getchannel("A").getextrema() == (0, 0)
    assert (
        client.put(f"/api/v1/land/{land['id']}", json={**BODY, "expectedRevision": 1}).status_code
        == 200
    )
    assert client.get(base).json()["stale"] is True
    assert client.get(base + "/download").content == original.content


def test_raster_recovery_reuses_saved_bytes_and_rejects_foreign_artifacts(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    terrain: TerrainResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, inv, _, _ = start(client)
    claimed = queue.claim(db)
    assert claimed is not None
    run_id, token = claimed
    worker = ResearchWorker(sessions, Settings(_env_file=None))
    original = queue.save_artifact

    def crash(*args: Any, **kwargs: Any) -> uuid.UUID:
        raise RuntimeError("Simulated worker loss after raster/evidence commit")

    monkeypatch.setattr(queue, "save_artifact", crash)
    with httpx.Client() as http, pytest.raises(RuntimeError, match="Simulated"):
        worker._execute(run_id, token, http)
    monkeypatch.setattr(queue, "save_artifact", original)
    assert db.scalar(select(func.count()).select_from(LandRaster)) == 1

    def no_recompute(*args: Any, **kwargs: Any) -> TerrainResult:
        raise AssertionError("Saved raster must be reused after recovery")

    monkeypatch.setattr("app.research.worker.run_raster", no_recompute)
    row = db.get(ResearchRun, run_id)
    assert row is not None
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    assert worker.run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert len(detail["evidence"]) == len(detail["artifacts"]) == len(detail["findings"]) == 1
    assert db.scalar(select(func.count()).select_from(LandRaster)) == 1
    raster_id = detail["artifacts"][0]["output"]["rasterId"]
    start(client)
    second = queue.claim(db)
    assert second is not None
    local_evidence = queue.save_evidence(db, *second, "local", evidence())
    with pytest.raises(InvalidInputError, match="investigation"):
        queue.save_artifact(
            db,
            *second,
            "foreign",
            ArtifactContent(
                title="Invalid reference",
                method="No source",
                evidence_ids=[local_evidence],
                output=RasterOutput(kind="raster", raster_id=uuid.UUID(raster_id)),
            ),
        )


def test_raster_cancellation_fences_late_bytes_and_quota_is_explicit(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    terrain: TerrainResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, run, _ = start(client)

    def cancelled(*args: Any, **kwargs: Any) -> TerrainResult:
        assert client.post(f"/api/v1/research/runs/{run['id']}/cancel").status_code == 200
        return terrain

    monkeypatch.setattr("app.research.worker.run_raster", cancelled)
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    assert client.get(f"/api/v1/research/runs/{run['id']}").json()["status"] == "cancelled"
    assert db.scalar(select(func.count()).select_from(LandRaster)) == 0
    monkeypatch.setattr("app.research.worker.run_raster", lambda *args: terrain)
    _, _, run, _ = start(client)
    settings = Settings(_env_file=None).model_copy(update={"land_raster_workspace_quota_bytes": 1})
    assert ResearchWorker(sessions, settings).run_once()
    db.expire_all()
    result = client.get(f"/api/v1/research/runs/{run['id']}").json()
    assert result["status"] == "failed" and "allowance" in result["error"]
    assert db.scalar(select(func.count()).select_from(LandRaster)) == 0


def test_private_raster_metadata_tiles_samples_and_download_are_workspace_scoped(
    identity_client: IdentityClient,
    db: Session,
    sessions: sessionmaker[Session],
    terrain: TerrainResult,
) -> None:
    client, token = identity_client
    alice_space = client.post(
        "/api/v1/workspaces", json={"name": "Alice"}, headers=headers(token())
    ).json()["id"]
    alice = headers(token(), alice_space)
    land, _, _, _ = start(client, alice)
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    raster_id = client.get(f"/api/v1/land/{land['id']}/rasters", headers=alice).json()[0]["id"]
    bob_space = client.post(
        "/api/v1/workspaces", json={"name": "Bob"}, headers=headers(token("bob"))
    ).json()["id"]
    bob = headers(token("bob"), bob_space)
    base = f"/api/v1/land/rasters/{raster_id}"
    for tail in ("", "/download", "/sample?longitude=0&latitude=0", "/tiles/1/0/0/0.png"):
        assert client.get(base + tail, headers=bob).status_code == 404
        assert client.get(base + tail).status_code == 401
    assert client.get(f"/api/v1/land/{land['id']}/rasters", headers=bob).status_code == 404


def test_agent_can_request_terrain_and_continue_with_its_saved_evidence(
    client: TestClient, db: Session, sessions: sessionmaker[Session], terrain: TerrainResult
) -> None:
    import json

    from app.research.model import CompleteAction, DecisionResult, RasterAction, ResearchDecision

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            retrieved = json.loads(context)["retrieved"]
            action: RasterAction | CompleteAction
            if not retrieved:
                action = RasterAction(kind="analyze_raster", analysis=RasterRequest())
            else:
                source = next(iter(retrieved.values()))
                assert source["data"]["metadata"]["bands"][0]["mean"] == 0
                action = CompleteAction(
                    kind="complete",
                    summary="The sampled model is level; this is not a field survey.",
                    evidence_ids=source["evidenceIds"],
                )
            return DecisionResult(
                ResearchDecision(progress="Reading the terrain", action=action), 100
            )

    _, inv, run, _ = start(client)
    row = db.get(ResearchRun, uuid.UUID(run["id"]))
    assert row is not None
    row.kind, row.analysis = "investigation", None
    db.commit()
    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded"
    assert detail["artifacts"][0]["output"]["kind"] == "raster"
    assert detail["evidence"][0]["id"] in detail["messages"][-1]["content"]
