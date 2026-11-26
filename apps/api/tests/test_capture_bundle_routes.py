"""Bundle authorization/dispatch units; PostGIS+moto route lifecycle is in test_capture_api."""

from __future__ import annotations

import time
import uuid
from unittest.mock import Mock

import pytest
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.api.deps import require_write_or_phone_key, require_write_token
from app.api.v1.captures import router
from app.config import Settings
from app.models import Capture, CaptureFile, Job, JobStep
from app.models.enums import CaptureKind, RunStatus, UploadStatus
from app.schemas.capture import CaptureCreate
from app.schemas.job import JobCreate
from app.services import capture_bundles, captures, handoff, jobs
from app.services.errors import ConflictError, InvalidInputError, UnauthorizedError


def test_bundle_routes_require_write_or_phone_key_not_handoff() -> None:
    routes = [
        route for route in router.routes if isinstance(route, APIRoute) and "bundle" in route.path
    ]
    assert len(routes) == 3
    for route in routes:
        assert any(dep.dependency is require_write_or_phone_key for dep in route.dependencies)
    settings = Settings(_env_file=None, api_write_token="test-only-write-token")
    scoped = handoff.issue(handoff.key_for(settings), uuid.uuid4(), now=int(time.time()))
    with pytest.raises(UnauthorizedError):
        require_write_token(settings, None)
    with pytest.raises(UnauthorizedError):
        require_write_token(
            settings, HTTPAuthorizationCredentials(scheme="Bearer", credentials=scoped)
        )
    require_write_token(
        settings, HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-only-write-token")
    )


def test_begin_recovers_client_uuid_without_second_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    capture = Capture(metadata_={"bundleFormat": capture_bundles.FORMAT})
    db.scalar.return_value = capture
    create = Mock()
    monkeypatch.setattr(captures, "create_capture", create)
    payload = CaptureCreate(name="Test", kind=CaptureKind.VIDEO)
    assert capture_bundles.begin(db, uuid.uuid4(), payload) is capture
    create.assert_not_called()
    db.execute.assert_called_once()


def test_begin_marks_only_bundle_and_preserves_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    db.scalar.return_value = None
    create = Mock(return_value=Capture())
    monkeypatch.setattr(captures, "create_capture", create)
    identifier = uuid.uuid4()
    payload = CaptureCreate(name="Test", kind=CaptureKind.VIDEO, metadata={"lat": 37, "lon": -122})
    capture_bundles.begin(db, identifier, payload)
    sent = create.call_args.args[1]
    assert sent.slug == f"hex-{identifier}"
    assert sent.metadata == {"lat": 37, "lon": -122, "bundleFormat": capture_bundles.FORMAT}
    assert "bundleFormat" not in payload.metadata
    with pytest.raises(InvalidInputError):
        capture_bundles.begin(db, uuid.uuid4(), CaptureCreate(name="Bad", kind=CaptureKind.IMAGES))


def test_process_is_idempotent_and_selects_existing_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    capture_id = uuid.uuid4()
    file = CaptureFile(
        filename="capture.hexcap",
        content_type=capture_bundles.CONTENT_TYPE,
        status=UploadStatus.COMPLETE,
    )
    capture = Capture(id=capture_id, files=[file])
    monkeypatch.setattr(capture_bundles, "locked_capture", lambda *_: capture)
    job = Job(params=capture_bundles.PARAMS)
    create = Mock(return_value=job)
    monkeypatch.setattr(jobs, "create_job", create)
    db.scalar.return_value = None
    assert capture_bundles.process(db, capture_id) is job
    payload = create.call_args.args[2]
    assert payload.recipe == "photo-reconstruct"
    assert payload.params == capture_bundles.PARAMS
    db.scalar.return_value = job
    assert capture_bundles.process(db, capture_id) is job
    create.assert_called_once()
    job.params = {}
    with pytest.raises(ConflictError):
        capture_bundles.process(db, capture_id)
    file.status = UploadStatus.IN_PROGRESS
    with pytest.raises(ConflictError):
        capture_bundles.process(db, capture_id)


def test_phone_bundle_uses_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    db = Mock(spec=Session)
    cid = uuid.uuid4()
    monkeypatch.setattr(capture_bundles, "completed_capture", Mock())
    existing = Job()
    create = Mock(return_value=existing)
    monkeypatch.setattr(jobs, "create_on_demand", create)
    assert capture_bundles.process(db, cid, phone=True) is existing
    create.assert_called_once_with(
        db,
        cid,
        JobCreate(recipe="photo-reconstruct", params=capture_bundles.PARAMS),
        retry_failed=True,
    )


@pytest.mark.parametrize("status", [RunStatus.ERROR, RunStatus.CANCELLED])
@pytest.mark.parametrize("with_steps", [False, True])
def test_phone_bundle_refine_retries_dead_job(
    monkeypatch: pytest.MonkeyPatch, status: RunStatus, with_steps: bool
) -> None:
    db = Mock(spec=Session)
    cid = uuid.uuid4()
    steps = [JobStep(stage_id="normalize", ordinal=0, status=status)] if with_steps else []
    job = Job(
        id=uuid.uuid4(), capture_id=cid, status=status, steps=steps, params=capture_bundles.PARAMS
    )
    db.scalar.return_value = job
    db.get.return_value = job
    monkeypatch.setattr(capture_bundles, "completed_capture", Mock())
    monkeypatch.setattr(jobs, "get_capture", Mock())
    result = capture_bundles.process(db, cid, phone=True)
    assert result is job
    assert result.status == RunStatus.NOT_STARTED
    assert all(step.status == RunStatus.NOT_STARTED for step in steps)
    db.commit.assert_called_once()


@pytest.mark.parametrize(
    "status", [RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS, RunStatus.COMPLETE]
)
def test_phone_bundle_refine_keeps_live_or_complete_job(
    monkeypatch: pytest.MonkeyPatch, status: RunStatus
) -> None:
    db = Mock(spec=Session)
    job = Job(id=uuid.uuid4(), status=status)
    db.scalar.return_value = job
    monkeypatch.setattr(capture_bundles, "completed_capture", Mock())
    monkeypatch.setattr(jobs, "get_capture", Mock())
    assert capture_bundles.process(db, uuid.uuid4(), phone=True) is job
    assert job.status == status
    db.commit.assert_not_called()
