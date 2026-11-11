"""CPU-only production runner -> worker heartbeat -> HTTP live response, mid-stage.

The loop-shaped trainer writes made-up measurements and tiny PLYs. No database,
network, Modal or GPU is used. The cloud variant deliberately has no provider logs.
"""

from __future__ import annotations

import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response
from sqlalchemy.orm import Session

from app.api.v1.live import capture_live, router
from app.models import Asset, Capture, Epoch, Job, JobStep, Site
from app.models.enums import Representation, RunStatus
from app.services.live import _train
from app.storage import ObjectStorage
from app.worker.pipeline_bridge import (
    CloudRunner,
    LocalTransfer,
    Placement,
    Recipe,
    RunnerSet,
    SubprocessAdapter,
    Workdir,
    execute,
)
from app.worker.runner import JobSupervisor

PIPELINE = Path(__file__).resolve().parents[3] / "tools" / "pipeline"


@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("test_every", [0, 8])
def test_production_live_mid_run_and_published_tiles(
    tmp_path: Path, cloud: bool, test_every: int
) -> None:
    import stages  # noqa: F401 - register real production implementations

    job = Job(id=uuid.uuid4(), recipe="photo-reconstruct", status=RunStatus.IN_PROGRESS, params={})
    capture = Capture(id=uuid.uuid4(), name="Stand-in (simulated)")
    job.capture_id = capture.id
    work = Workdir.create(tmp_path / str(job.id))
    frames, poses = work.input_path("frames"), work.input_path("poses")
    frames.mkdir()
    poses.mkdir()
    for index in range(4):
        (frames / f"{index}.jpg").write_bytes(b"stand-in")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (poses / name).write_bytes(b"stand-in")
    (poses / "poses.json").write_text('{"registered":4}')
    stage: dict[str, Any] = {
        "id": "train",
        "impl": "gsplat",
        "params": {
            "trainer": str(PIPELINE / "tests/loop_stand_in/simple_trainer.py"),
            "python": sys.executable,
            "iterations": 30000,
            "schedule_scale": 0.05,
            "schedule_growth": "off",
            "schedule_full_at": 1,
            "strategy": "mcmc",
            "cap_max": 64,
            "converge": test_every > 0,
            "live": True,
            "test_every": test_every,
            "checkpoint_every": 0,
            "streaming_export": False,
            "extra_args": ["--step-delay", "0.004"],
        },
    }
    transfer = LocalTransfer(tmp_path / "storage")
    if cloud:
        stage["gpu"] = {"tier": "l4"}
        adapter = SubprocessAdapter(transfer, tmp_path / "remote")
        adapter.logs = lambda *_a, **_kw: ()  # type: ignore[method-assign]
        runners = RunnerSet.cloud(
            CloudRunner(
                Placement((adapter,)), transfer, poll_interval_s=0.05, checkpoint_every_s=0.1
            )
        )
    else:
        runners = RunnerSet.local()
    recipe = Recipe.from_dict(
        {"name": "stand-in", "version": 1, "inputs": ["frames", "poses"], "stages": [stage]}
    )
    step = JobStep(
        job_id=job.id,
        stage_id="train",
        impl="gsplat",
        ordinal=1,
        attempt=1,
        status=RunStatus.IN_PROGRESS,
        metrics={},
    )
    db = Mock(spec=Session)

    def scalars(statement: Any) -> Mock:
        result = Mock()
        result.first.return_value = job
        result.all.return_value = [step]
        return result

    db.scalars.side_effect = scalars
    site: Site | None = None
    db.get.side_effect = lambda model, _id: capture if model is Capture else site
    storage = Mock(spec=ObjectStorage)

    def present(key: str) -> bool | None:
        path = (
            transfer.root / key
            if cloud
            else work.checkpoint_dir("train") / key.split("/checkpoint/", 1)[-1]
        )
        return True if path.is_file() else None

    storage.head_object.side_effect = present
    storage.presign_get.side_effect = lambda key, **kw: f"https://storage.example/{key}?signed=1"
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    def response() -> dict[str, Any]:
        request = Request(
            {
                "type": "http",
                "scheme": "http",
                "server": ("test", 80),
                "path": "/",
                "headers": [],
                "root_path": "",
                "router": app.router,
            }
        )
        state = capture_live(capture.id, db, storage, Response(), request)
        return state.model_dump(mode="json", by_alias=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute, recipe, work, runners)
        mid = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not future.done():
            JobSupervisor._report_progress(db, step, work.root)
            body = response()
            train = body.get("train") or {}
            if (
                train.get("series", {}).get("step")
                and train.get("snapshots")
                and train["snapshots"][-1].get("url")
                and train.get("evals")
            ):
                mid = body
                break
            time.sleep(0.05)
        result = future.result(timeout=20)
        assert mid is not None, step.metrics
        assert mid["status"] == "in-progress"
        train = mid["train"]
        assert train["series"]["loss"]["total"] and train["series"]["gs"]
        assert train["series"]["step"][-1] < 1500
        assert train["plan"]["stepsMax"] == 1500
        assert train["events"] and train["snapshots"][-1]["url"]
        assert train["evals"][-1]["split"] == ("train" if test_every == 0 else "val")
        assert not future.cancelled()
        JobSupervisor._report_progress(db, step, work.root)
        assert result.steps
        job.status = RunStatus.COMPLETE
        step.status = RunStatus.COMPLETE
        capture.site_id = uuid.uuid4()
        epoch = Epoch(id=uuid.uuid4(), job_id=job.id, ordinal=1)
        asset = Asset(
            id=uuid.uuid4(),
            epoch_id=epoch.id,
            representation=Representation.GAUSSIAN_SPLAT,
            source_kind="observed",
            created_at=datetime.now(UTC),
            source={"type": "3d-tiles-url", "url": "https://public.example/final/tileset.json"},
        )
        site = Site(id=capture.site_id, epochs=[epoch], assets=[asset])
        final = response()
        assert final["train"]["tiles"] == {
            "url": asset.source["url"],
            "step": 1500,
            "final": True,
            "frame": "enu",
        }


def test_snapshot_signing_is_scoped_and_waits_for_upload() -> None:
    job_id = uuid.uuid4()
    step = JobStep(job_id=job_id, stage_id="train", attempt=2)
    key = f"runs/{job_id}/train/parts/p0/checkpoint-a2/live/splat.spz"
    snapshot = {"step": 200, "count": 10, "key": key}
    storage = Mock(spec=ObjectStorage)
    storage.head_object.return_value = None
    value = {"train": {}, "snapshots": [snapshot]}
    pending = _train(value, storage, step=step)
    assert pending is not None and pending.snapshots[0].url is None
    storage.presign_get.assert_not_called()
    storage.head_object.return_value = True
    storage.presign_get.return_value = "https://signed.example/snapshot"
    ready = _train(value, storage, step=step)
    assert ready is not None and ready.snapshots[0].url
    for forbidden in [
        key.replace(str(job_id), str(uuid.uuid4())),
        key.replace("/train/", "/other/"),
        key.replace("checkpoint-a2", "checkpoint"),
        key.replace("splat.spz", "../secret"),
    ]:
        storage.presign_get.reset_mock()
        assert (
            _train({"train": {}, "snapshots": [{**snapshot, "key": forbidden}]}, storage, step=step)
            is None
        )
        storage.presign_get.assert_not_called()
