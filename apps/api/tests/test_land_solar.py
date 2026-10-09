from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.solar import SolarResult
from app.config import Settings
from app.models.land_solar import LandSolar
from app.models.research import ResearchRun
from app.research import queue
from app.research.worker import ResearchWorker
from app.schemas.geojson import Footprint
from app.schemas.land_solar import SolarRequest
from app.schemas.research import ArtifactContent, SolarOutput
from app.services.errors import InvalidInputError
from tests.test_land import BODY
from tests.test_research import evidence
from tests.test_scenarios import SOLAR
from tests.test_solar import BOUNDARY, REQUEST, result
from tests.test_workspaces import IdentityClient, headers
from tests.test_workspaces import identity_client as _identity_client

identity_client = _identity_client


@pytest.fixture
def solar(monkeypatch: pytest.MonkeyPatch) -> SolarResult:
    output = result(SolarRequest.model_validate(REQUEST))

    def calculate(
        boundary: Footprint, request: SolarRequest, cancelled: threading.Event | None = None
    ) -> SolarResult:
        return output

    monkeypatch.setattr("app.research.worker.run_solar", calculate)
    return output


def start(
    client: TestClient, auth: dict[str, str] | None = None
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    land = client.post(
        "/api/v1/land", json={**BODY, "boundary": BOUNDARY.model_dump()}, headers=auth
    ).json()
    inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Solar", "boundaryRevision": 1},
        headers=auth,
    ).json()["id"]
    body = {
        "requestKey": str(uuid.uuid4()),
        "kind": "solar",
        "question": "Model this array",
        "analysis": REQUEST,
    }
    response = client.post(f"/api/v1/research/investigations/{inv}/runs", json=body, headers=auth)
    assert response.status_code == 201, response.text
    return land, inv, response.json(), body


def financial(assessment: dict[str, Any]) -> dict[str, Any]:
    r, m = assessment["request"], assessment["metadata"]
    return {
        "name": "Hourly solar finance",
        "boundaryRevision": 1,
        "solarAssessmentId": assessment["id"],
        "inputs": {
            **SOLAR,
            "usableRoofAreaM2": r["moduleAreaM2"],
            "moduleEfficiency": r["moduleEfficiency"],
            "tiltDegrees": r["tiltDegrees"],
            "azimuthDegrees": r["azimuthDegrees"],
            "annualPlaneIrradiationKwhM2": m["unshadedPlaneIrradiationKwhM2"],
            "shadeLoss": r["additionalShadeLoss"],
            "systemLoss": r["systemLoss"],
        },
    }


def test_hourly_job_saved_sources_download_finance_and_stale(
    client: TestClient, db: Session, sessions: sessionmaker[Session], solar: SolarResult
) -> None:
    land, inv, run, body = start(client)
    endpoint = f"/api/v1/research/investigations/{inv}/runs"
    assert client.post(endpoint, json=body).json()["id"] == run["id"]
    assert (
        client.post(endpoint, json={**body, "analysis": {**REQUEST, "year": 2022}}).status_code
        == 409
    )
    assert client.post(endpoint, json={**body, "kind": "raster"}).status_code == 422
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    identifier = detail["artifacts"][0]["output"]["assessmentId"]
    base = f"/api/v1/land/solar-assessments/{identifier}"
    saved = client.get(base).json()
    assert saved["metadata"]["completeYear"] and saved["stale"] is False
    assert detail["findings"][0]["evidenceIds"] == [detail["evidence"][0]["id"]]
    assert len(client.get(f"/api/v1/land/{land['id']}/solar-assessments").json()) == 1
    archive = client.get(base + "/download")
    assert archive.content == solar.data and archive.headers["cache-control"] == "private, no-store"
    assert hashlib.sha256(archive.content).hexdigest() == saved["sha256"]
    finance = financial(saved)
    preview = client.post(f"/api/v1/land/{land['id']}/scenarios/preview", json=finance)
    assert preview.status_code == 200, preview.text
    assert (
        preview.json()["summary"]["firstYearGenerationKwh"]
        == saved["metadata"]["annualGenerationKwh"]
    )
    assert preview.json()["summary"]["solarAssessmentSha256"] == saved["sha256"]
    assert preview.json()["algorithm"] == "solar-hourly-cash-flow/1"
    assert (
        client.post(
            f"/api/v1/land/{land['id']}/scenarios/preview",
            json={**finance, "inputs": {**finance["inputs"], "shadeLoss": 0.2}},
        ).status_code
        == 422
    )
    client.put(f"/api/v1/land/{land['id']}", json={**BODY, "expectedRevision": 1})
    assert client.get(base).json()["stale"] is True
    assert client.get(base + "/download").content == solar.data
    assert (
        client.post(
            f"/api/v1/land/{land['id']}/scenarios/preview", json={**finance, "boundaryRevision": 2}
        ).status_code
        == 422
    )


def test_recovery_reuses_archive_and_rejects_foreign_artifact(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    solar: SolarResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, inv, run, _ = start(client)
    original = queue.save_evidence

    def fail(*args: Any, **kwargs: Any) -> uuid.UUID:
        raise queue.LeaseLostError()

    monkeypatch.setattr(queue, "save_evidence", fail)
    db.rollback()
    worker = ResearchWorker(sessions, Settings(_env_file=None))
    worker.run_once()
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(LandSolar)) == 1
    row = db.get(ResearchRun, uuid.UUID(run["id"]))
    assert row is not None
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    monkeypatch.setattr(queue, "save_evidence", original)

    def no_recompute(*args: Any, **kwargs: Any) -> SolarResult:
        raise AssertionError("Saved hourly output must be reused")

    monkeypatch.setattr("app.research.worker.run_solar", no_recompute)
    worker.run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    foreign_id = detail["artifacts"][0]["output"]["assessmentId"]
    _, _, other, _ = start(client)
    db.rollback()
    claimed = queue.claim(db)
    assert claimed is not None
    local_evidence = queue.save_evidence(db, *claimed, "local", evidence())
    with pytest.raises(InvalidInputError, match="this investigation"):
        queue.save_artifact(
            db,
            uuid.UUID(other["id"]),
            claimed[1],
            "foreign",
            ArtifactContent(
                title="Foreign",
                method="Not permitted",
                evidence_ids=[local_evidence],
                output=SolarOutput(kind="solar", assessment_id=uuid.UUID(foreign_id)),
            ),
        )
    db.rollback()


def test_workspace_privacy(
    identity_client: IdentityClient,
    db: Session,
    sessions: sessionmaker[Session],
    solar: SolarResult,
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
    identifier = client.get(f"/api/v1/land/{land['id']}/solar-assessments", headers=alice).json()[
        0
    ]["id"]
    bob_space = client.post(
        "/api/v1/workspaces", json={"name": "Bob"}, headers=headers(token("bob"))
    ).json()["id"]
    bob = headers(token("bob"), bob_space)
    base = f"/api/v1/land/solar-assessments/{identifier}"
    for tail in ("", "/download"):
        assert client.get(base + tail, headers=bob).status_code == 404
        assert client.get(base + tail).status_code == 401
    assert (
        client.get(f"/api/v1/land/{land['id']}/solar-assessments", headers=bob).status_code == 404
    )


def test_cancel_and_quota_leave_no_output(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    solar: SolarResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, run, _ = start(client)

    def cancel(*args: Any, **kwargs: Any) -> SolarResult:
        assert client.post(f"/api/v1/research/runs/{run['id']}/cancel").status_code == 200
        return solar

    monkeypatch.setattr("app.research.worker.run_solar", cancel)
    db.rollback()
    assert ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    assert client.get(f"/api/v1/research/runs/{run['id']}").json()["status"] == "cancelled"
    assert db.scalar(select(func.count()).select_from(LandSolar)) == 0
    monkeypatch.setattr("app.research.worker.run_solar", lambda *args: solar)
    _, _, run, _ = start(client)
    settings = Settings(_env_file=None).model_copy(update={"land_solar_workspace_quota_bytes": 1})
    db.rollback()
    assert ResearchWorker(sessions, settings).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/runs/{run['id']}").json()
    assert detail["status"] == "failed" and "allowance" in detail["error"]
    assert db.scalar(select(func.count()).select_from(LandSolar)) == 0


def test_agent_requests_hourly_analysis_and_receives_saved_evidence(
    client: TestClient, db: Session, sessions: sessionmaker[Session], solar: SolarResult
) -> None:
    import json

    from app.research.model import CompleteAction, DecisionResult, ResearchDecision, SolarAction

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            retrieved = json.loads(context)["retrieved"]
            action: SolarAction | CompleteAction
            if not retrieved:
                action = SolarAction(
                    kind="analyze_solar", analysis=SolarRequest.model_validate(REQUEST)
                )
            else:
                source = next(iter(retrieved.values()))
                assert (
                    source["data"]["metadata"]["annual_generation_kwh"]
                    == solar.metadata.annual_generation_kwh
                )
                action = CompleteAction(
                    kind="complete",
                    summary="A modeled historical year, not measured roof output.",
                    evidence_ids=source["evidenceIds"],
                )
            return DecisionResult(
                ResearchDecision(progress="Analyzing the array", action=action), 100
            )

    _, inv, run, _ = start(client)
    row = db.get(ResearchRun, uuid.UUID(run["id"]))
    assert row is not None
    row.kind, row.analysis = "investigation", None
    db.commit()
    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    assert detail["artifacts"][0]["output"]["kind"] == "solar"
    assert detail["evidence"][0]["id"] in detail["messages"][-1]["content"]


def test_preview_checks_zone_and_partial_year_cannot_feed_finance(
    client: TestClient, db: Session, sessions: sessionmaker[Session], solar: SolarResult
) -> None:
    land, inv, _, _ = start(client)
    endpoint = f"/api/v1/land/{land['id']}/solar-assessments/preview"
    preview = client.post(endpoint, json=REQUEST)
    assert preview.status_code == 200, preview.text
    assert preview.json()["capacityKwDc"] == 20
    assert client.post(endpoint, json={**REQUEST, "moduleAreaM2": 1e7}).status_code == 422
    solar.metadata.complete_year = False
    solar.metadata.annual_generation_kwh = None
    solar.metadata.valid_hours -= 1
    db.rollback()
    ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    identifier = detail["artifacts"][0]["output"]["assessmentId"]
    saved = client.get(f"/api/v1/land/solar-assessments/{identifier}").json()
    response = client.post(f"/api/v1/land/{land['id']}/scenarios/preview", json=financial(saved))
    assert response.status_code == 422 and "Incomplete" in response.text


def test_agent_reads_existing_assessment_without_recomputing(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    solar: SolarResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from app.research.model import CompleteAction, DecisionResult, ResearchDecision, SolarReadAction

    land, _, _, _ = start(client)
    db.rollback()
    ResearchWorker(sessions, Settings(_env_file=None)).run_once()
    db.expire_all()
    saved = client.get(f"/api/v1/land/{land['id']}/solar-assessments").json()[0]
    inv = client.post(
        f"/api/v1/land/{land['id']}/investigations",
        json={"title": "Follow-up", "boundaryRevision": 1},
    ).json()["id"]
    client.post(
        f"/api/v1/research/investigations/{inv}/runs",
        json={"requestKey": str(uuid.uuid4()), "question": "Explain my existing solar assessment"},
    )

    def fail(*args: Any, **kwargs: Any) -> SolarResult:
        raise AssertionError("Existing assessment should not be recomputed")

    monkeypatch.setattr("app.research.worker.run_solar", fail)

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            state = json.loads(context)
            assert state["savedSolarAssessments"][0]["id"] == saved["id"]
            action: SolarReadAction | CompleteAction
            if not state["retrieved"]:
                action = SolarReadAction(
                    kind="read_solar_assessment", assessment_id=uuid.UUID(saved["id"])
                )
            else:
                source = next(iter(state["retrieved"].values()))
                assert source["data"]["sha256"] == saved["sha256"]
                assert source["data"]["request"]["module_area_m2"] == 100
                assert "array_zone" not in source["data"]["request"]
                action = CompleteAction(
                    kind="complete",
                    summary="Reviewed the saved calculation and assumptions.",
                    evidence_ids=source["evidenceIds"],
                )
            return DecisionResult(
                ResearchDecision(progress="Reading saved analysis", action=action), 100
            )

    db.rollback()
    ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    assert detail["evidence"][0]["snapshotHash"] == saved["sha256"]
    assert db.scalar(select(func.count()).select_from(LandSolar)) == 1
