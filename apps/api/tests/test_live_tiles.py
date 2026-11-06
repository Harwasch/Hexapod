"""Live LOD route handlers and worker state, without a database or network."""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response
from sqlalchemy.orm import Session

from app.api.v1.live import capture_live, capture_live_tile, job_live, job_live_tile, router
from app.models import Capture, Job, JobStep
from app.models.enums import RunStatus
from app.services.errors import NotFoundError
from app.services.live import _state
from app.services.live_train import merge_live
from app.storage import ObjectStorage

LiveTiles = tuple[Request, Mock, Mock, Capture, Job, JobStep]


@pytest.fixture
def live_tiles() -> LiveTiles:
    capture = Capture(id=uuid.uuid4(), name="Tree")
    job = Job(
        id=uuid.uuid4(),
        capture_id=capture.id,
        recipe="photo-reconstruct",
        status=RunStatus.IN_PROGRESS,
    )
    prefix = f"runs/{job.id}/train/checkpoint"
    step = JobStep(
        job_id=job.id,
        stage_id="train",
        impl="gsplat",
        ordinal=1,
        checkpoint_key=prefix,
        status=RunStatus.IN_PROGRESS,
        metrics={
            "live": {
                "tiles": {
                    "v": 1,
                    "key": f"{prefix}/live/tiles/30/tileset.json",
                    "step": 30,
                    "final": False,
                }
            }
        },
    )
    db = Mock(spec=Session)
    db.get.side_effect = lambda model, _id: job if model is Job else capture
    db.scalars.return_value.first.return_value = job
    db.scalars.return_value.all.return_value = [step]
    storage = Mock(spec=ObjectStorage)
    storage.presign_get.return_value = "https://private.example/signed?expires=600"
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    request = Request(
        {
            "type": "http",
            "scheme": "https",
            "server": ("api.example", 443),
            "path": f"/api/v1/captures/{capture.id}/live",
            "root_path": "",
            "headers": [],
            "router": app.router,
            "path_params": {"capture_id": capture.id},
        }
    )
    return request, db, storage, capture, job, step


@pytest.mark.parametrize("path", ["tileset.json", "splat_0.glb", "collision.bin", "sub/tile.glb"])
def test_tile_redirect_signs_only_this_runs_live_directory(
    live_tiles: LiveTiles, path: str
) -> None:
    _request, db, storage, capture, _job, step = live_tiles
    response = capture_live_tile(capture.id, 30, path, db, storage)
    assert response.status_code == 302 and response.headers["cache-control"] == "no-store"
    assert response.headers["location"] == storage.presign_get.return_value
    storage.presign_get.assert_called_once_with(
        f"{step.checkpoint_key}/live/tiles/30/{path}", expires_in=600
    )


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "x/../secret",
        "%2e%2e/secret",
        "%252e%252e/secret",
        "/secret",
        "x\\secret",
        "x//secret",
        "x/./secret",
        "",
    ],
)
def test_tile_traversal_is_refused(live_tiles: LiveTiles, path: str) -> None:
    _request, db, storage, capture, job, _step = live_tiles
    with pytest.raises(NotFoundError):
        capture_live_tile(capture.id, 30, path, db, storage)
    with pytest.raises(NotFoundError):
        job_live_tile(job.id, 30, path, db, storage)
    storage.presign_get.assert_not_called()


@pytest.mark.parametrize(
    "key",
    [
        "captures/private/tileset.json",
        "runs/other/train/checkpoint/live/tiles/30/tileset.json",
        "runs/{job}/train/checkpoint/live/tiles/../30/tileset.json",
        "runs/{job}/other/checkpoint/live/tiles/30/tileset.json",
    ],
)
def test_tile_key_outside_this_live_checkpoint_is_refused(live_tiles: LiveTiles, key: str) -> None:
    _request, db, storage, capture, job, step = live_tiles
    step.metrics["live"]["tiles"]["key"] = key.format(job=job.id)
    with pytest.raises(NotFoundError):
        capture_live_tile(capture.id, 30, "tileset.json", db, storage)
    storage.presign_get.assert_not_called()


def test_missing_run_object_or_step_is_not_found(live_tiles: LiveTiles) -> None:
    _request, db, storage, capture, job, _step = live_tiles
    db.scalars.return_value.first.return_value = None
    with pytest.raises(NotFoundError):
        capture_live_tile(capture.id, 30, "tileset.json", db, storage)
    storage.presign_get.assert_not_called()
    db.scalars.return_value.first.return_value = job
    with pytest.raises(NotFoundError):
        capture_live_tile(capture.id, 29, "tileset.json", db, storage)
    storage.head_object.return_value = None
    with pytest.raises(NotFoundError):
        capture_live_tile(capture.id, 30, "tileset.json", db, storage)
    storage.presign_get.assert_not_called()


def test_live_tiles_url_uses_api_origin_and_waits_for_upload(live_tiles: LiveTiles) -> None:
    request, db, storage, capture, job, step = live_tiles
    state = capture_live(capture.id, db, storage, Response(), request)
    assert state.train is not None and state.train.tiles is not None
    assert state.train.tiles.model_dump() == {
        "url": f"https://api.example/api/v1/captures/{capture.id}/live/tiles/30/tileset.json",
        "step": 30,
        "final": False,
        "frame": "solve",
    }
    # A published scan must not replace aligned final tiles with placed ENU coordinates.
    job.status = RunStatus.COMPLETE
    capture.site_id = uuid.uuid4()
    step.metrics["live"]["tiles"]["final"] = True
    train = _state(db, storage, job, capture).train
    assert train is not None and train.tiles is not None and train.tiles.frame == "solve"
    storage.head_object.return_value = None
    job.status = RunStatus.IN_PROGRESS
    assert capture_live(capture.id, db, storage, Response(), request).train is None


def test_job_view_keeps_its_run_even_after_capture_has_a_newer_run(live_tiles: LiveTiles) -> None:
    request, db, storage, _capture, job, _step = live_tiles
    request.scope["path_params"] = {"job_id": job.id}
    state = job_live(job.id, db, storage, Response(), request)
    assert state.train is not None and state.train.tiles is not None
    assert (
        state.train.tiles.url
        == f"https://api.example/api/v1/jobs/{job.id}/live/tiles/30/tileset.json"
    )
    assert job_live_tile(job.id, 30, "splat.glb", db, storage).status_code == 302


def test_tiles_merge_is_bounded_and_resets_on_new_attempt() -> None:
    current: dict[str, Any] = {}
    for step in range(1000):
        current = merge_live(
            current,
            [
                ("plan", {"id": "a", "stepsMax": 1000}),
                (
                    "tiles",
                    {
                        "v": 1,
                        "key": f"runs/r/train/checkpoint/live/tiles/{step}/tileset.json",
                        "step": step,
                        "final": step == 999,
                        "gaussians": 1000000,
                        "tiles": 30,
                        "bytes": 10000000,
                        "seconds": 0.1,
                        "ignored": "x" * 100000,
                    },
                ),
            ],
        )
    assert current["tiles"]["step"] == 999 and current["tiles"]["final"]
    assert len(json.dumps(current)) < 2000
    assert "tiles" not in merge_live(current, [("plan", {"id": "b", "stepsMax": 1000})])


def test_cross_origin_resolver_returns_the_same_signed_url_without_redirect(
    live_tiles: LiveTiles,
) -> None:
    _request, db, storage, capture, job, _step = live_tiles
    for response in (
        capture_live_tile(capture.id, 30, "splat.glb", db, storage, resolve=True),
        job_live_tile(job.id, 30, "collision.bin", db, storage, resolve=True),
    ):
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert json.loads(bytes(response.body)) == {"url": storage.presign_get.return_value}


@pytest.mark.parametrize("part", ["", "/parts/b0"])
def test_retry_tiles_use_this_attempt_and_allow_its_fanout_part(
    live_tiles: LiveTiles, part: str
) -> None:
    _request, db, storage, capture, job, step = live_tiles
    step.attempt = 2
    # The persisted checkpoint_key may still name attempt 1 until completion.
    key = f"runs/{job.id}/train{part}/checkpoint-a2/live/tiles/30/tileset.json"
    step.metrics["live"]["tiles"]["key"] = key
    assert capture_live_tile(capture.id, 30, "tileset.json", db, storage).status_code == 302
    storage.presign_get.assert_called_once_with(key, expires_in=600)
    step.metrics["live"]["tiles"]["key"] = key.replace("checkpoint-a2", "checkpoint")
    with pytest.raises(NotFoundError):
        capture_live_tile(capture.id, 30, "tileset.json", db, storage)
