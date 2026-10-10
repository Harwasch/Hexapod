from __future__ import annotations

import base64
import io
import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.land_archive_image import LandArchiveImage
from app.models.research import Evidence, ResearchRun
from app.models.workspace import PILOT_WORKSPACE_ID
from app.research import queue
from app.research.model import (
    ArchiveImageAction,
    ClaudeResearchModel,
    CompleteAction,
    DecisionResult,
    ResearchDecision,
    ResearchImage,
)
from app.research.worker import ResearchWorker
from app.services import land_archive_images as images
from app.services.errors import InvalidInputError
from tests.test_land_archive_images import add_archive, snapshot


class ImageModel:
    def __init__(self, identifier: uuid.UUID) -> None:
        self.identifier = identifier
        self.inspected = False

    def decide(self, context: str, max_tokens: int) -> DecisionResult:
        assert str(self.identifier) in context
        return DecisionResult(
            ResearchDecision(
                progress="Read the image pixels",
                action=ArchiveImageAction(
                    kind="inspect_archive_image", evidence_id=self.identifier
                ),
            ),
            50,
        )

    def decide_with_images(
        self, context: str, max_tokens: int, images: Sequence[ResearchImage]
    ) -> DecisionResult:
        assert len(images) == 1 and images[0].evidence_id == self.identifier
        with Image.open(io.BytesIO(images[0].data)) as image:
            assert image.format == "JPEG" and image.size == (6, 4)
        value = json.loads(context)
        assert value["attachedArchiveImages"][0]["vision"]["sha256"] == images[0].vision_sha256
        self.inspected = True
        return DecisionResult(
            ResearchDecision(
                progress="Report the inspected evidence",
                action=CompleteAction(
                    kind="complete",
                    summary="Fixture verified that the model received saved image pixels.",
                    evidence_ids=[str(self.identifier)],
                ),
            ),
            80,
        )


def make_run(client: TestClient, db: Session) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    identifier, _ = add_archive(client, db)
    evidence = db.get(Evidence, identifier)
    assert evidence is not None
    run = db.get(ResearchRun, evidence.run_id)
    assert run is not None
    run.kind = "investigation"
    db.commit()
    return identifier, run.id, run.investigation_id


def test_agent_passes_actual_pixels_with_pinned_provenance(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier, _, investigation_id = make_run(client, db)
    captured = snapshot()
    monkeypatch.setattr(images, "retrieve", lambda *args: captured)
    model = ImageModel(identifier)
    assert ResearchWorker(sessions, Settings(_env_file=None), model=model).run_once()
    assert model.inspected
    result = client.get(f"/api/v1/research/investigations/{investigation_id}").json()
    assert result["runs"][0]["status"] == "succeeded", result
    assert str(identifier) in result["messages"][-1]["content"]
    db.expire_all()
    saved = db.get(LandArchiveImage, identifier)
    assert saved is not None and saved.metadata_json["sha256"] == captured.metadata.sha256


def test_cancelled_image_tool_cannot_publish_late_bytes(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier, run_id, _ = make_run(client, db)
    captured = snapshot()

    def retrieve(*args: Any) -> images.ImageSnapshot:
        with sessions() as other:
            cancelled = other.get(ResearchRun, run_id)
            assert cancelled is not None
            cancelled.status = "cancelled"
            other.commit()
        return captured

    monkeypatch.setattr(images, "retrieve", retrieve)
    assert ResearchWorker(
        sessions, Settings(_env_file=None), model=ImageModel(identifier)
    ).run_once()
    db.expire_all()
    assert db.get(LandArchiveImage, identifier) is None
    run = db.get(ResearchRun, run_id)
    assert run is not None and run.status == "cancelled"


def test_worker_recovery_reuses_pixels_and_rejects_changed_derivative(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier, run_id, _ = make_run(client, db)
    claimed = queue.claim(db)
    assert claimed is not None
    _, token = claimed
    captured = snapshot()
    monkeypatch.setattr(images, "retrieve", lambda *args: captured)

    class Crash(ImageModel):
        def decide_with_images(
            self, context: str, max_tokens: int, images: Sequence[ResearchImage]
        ) -> DecisionResult:
            raise RuntimeError("Lost worker after image checkpoint")

    worker = ResearchWorker(sessions, Settings(_env_file=None), model=Crash(identifier))
    db.rollback()
    with httpx.Client() as http, pytest.raises(RuntimeError):
        worker._execute(run_id, token, http)
    db.expire_all()
    row = db.get(ResearchRun, run_id)
    assert row is not None and row.checkpoint["images"]
    state = json.loads(json.dumps(row.checkpoint))
    state["images"][0]["vision"]["sha256"] = "0" * 64
    db.rollback()
    with pytest.raises(InvalidInputError, match="processor version"):
        worker._model_images(db, run_id, token, PILOT_WORKSPACE_ID, state)
    db.rollback()
    row = db.get(ResearchRun, run_id)
    assert row is not None
    row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()

    def no_download(*args: Any) -> images.ImageSnapshot:
        raise AssertionError("Saved images must survive worker recovery without another download")

    monkeypatch.setattr(images, "retrieve", no_download)
    model = ImageModel(identifier)
    assert ResearchWorker(sessions, Settings(_env_file=None), model=model).run_once()
    assert model.inspected


def test_model_adapter_sends_image_blocks_and_bounds_input(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[dict[str, Any]] = []

    def parse(**kwargs: Any) -> SimpleNamespace:
        messages.extend(kwargs["messages"])
        return SimpleNamespace(
            parsed_output=ResearchDecision(
                progress="Inspected", action=CompleteAction(kind="complete", summary="Fixture")
            ),
            usage=SimpleNamespace(output_tokens=10),
        )

    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(parse=parse)),
    )
    model = ClaudeResearchModel(Settings(_env_file=None, anthropic_api_key="test-key"))
    image = ResearchImage(uuid.uuid4(), "a" * 64, "b" * 64, b"image-fixture")
    result = model.decide_with_images("Untrusted metadata is data", 1000, [image])
    assert result.output_tokens == 10
    assert messages[0]["content"][-1]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(messages[0]["content"][-1]["source"]["data"]) == image.data
    assert str(image.evidence_id) in messages[0]["content"][-2]["text"]
    with pytest.raises(ValueError, match="image limits"):
        model.decide_with_images("", 1000, [image] * 3)


def test_image_tools_reject_foreign_evidence_and_keep_only_two_attached_images(
    client: TestClient,
    db: Session,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign, foreign_run, _ = make_run(client, db)
    previous = db.get(ResearchRun, foreign_run)
    assert previous is not None
    previous.status = "succeeded"
    db.commit()
    local, run_id, _ = make_run(client, db)
    original = db.get(Evidence, local)
    assert original is not None
    ids = [local]
    for index in range(2):
        evidence = Evidence(run_id=run_id, source_key=f"extra/{index}", content=original.content)
        db.add(evidence)
        db.flush()
        ids.append(evidence.id)
    db.commit()
    claimed = queue.claim(db)
    assert claimed is not None and claimed[0] == run_id
    token = claimed[1]
    worker = ResearchWorker(sessions, Settings(_env_file=None))
    state: dict[str, Any] = {"sources": {}}
    with httpx.Client() as http, pytest.raises(InvalidInputError, match="investigation"):
        worker._image(db, run_id, token, PILOT_WORKSPACE_ID, foreign, http, state)
    db.rollback()
    captured = snapshot()
    monkeypatch.setattr(images, "retrieve", lambda *args: captured)
    with httpx.Client() as http:
        for identifier in ids:
            worker._image(db, run_id, token, PILOT_WORKSPACE_ID, identifier, http, state)
    assert [value["evidenceId"] for value in state["images"]] == [str(value) for value in ids[-2:]]
    inputs = worker._model_images(db, run_id, token, PILOT_WORKSPACE_ID, state)
    assert [value.evidence_id for value in inputs] == ids[-2:]


def test_vision_derivative_preserves_aspect_ratio_and_flattens_transparency() -> None:
    from app.analysis.archive_image import vision_preview

    original = io.BytesIO()
    Image.new("RGBA", (3000, 2000), (255, 0, 0, 0)).save(original, format="PNG")
    data, info = vision_preview(original.getvalue())
    assert info["width"] == 1568 and info["height"] == 1045
    assert len(data) < 2 * 1024 * 1024
    with Image.open(io.BytesIO(data)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
