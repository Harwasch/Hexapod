"""Preparation is visible before I/O; references fetch the aligner's exact frame sample."""

from __future__ import annotations

import json
import struct
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.models import Capture, CaptureFile, Job, JobStep
from app.models.enums import RunStatus, UploadStatus
from app.storage.base import ObjectPage, ObjectSummary
from app.worker import steps
from app.worker.config import WorkerConfig
from app.worker.loop import Worker
from app.worker.pipeline_bridge import Workdir
from app.worker.runner import JobSupervisor
from app.worker.site_context import _download_tree
from tests.test_worker import config


def test_claim_commits_first_step_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = Job(id=uuid.uuid4(), recipe="t-three")
    db = Mock()
    db.scalar.return_value = None
    sessions = Mock(return_value=db)
    monkeypatch.setattr("app.worker.loop.claim_next", lambda *a, **kw: job)
    worker = Worker(sessions, Mock(), config(tmp_path, min_free_gb=0))

    assert worker.claim() == job.id
    row = db.add.call_args.args[0]
    assert (row.stage_id, row.ordinal, row.status, row.attempt) == (
        "one",
        0,
        RunStatus.IN_PROGRESS,
        1,
    )
    assert row.started_at is not None and row.metrics[steps.PREPARING]
    assert db.commit.call_count >= 1


def test_preparation_progress_is_committed_before_and_after_download(tmp_path: Path) -> None:
    db = Mock()
    capture = Capture(
        id=uuid.uuid4(),
        files=[
            CaptureFile(
                filename="clip.mov",
                storage_key="upload/clip.mov",
                status=UploadStatus.COMPLETE,
            )
        ],
    )
    job = Job(id=uuid.uuid4(), capture_id=capture.id)
    db.get.return_value = capture
    step = JobStep(status=RunStatus.IN_PROGRESS, metrics={steps.PREPARING: True})
    storage = Mock()

    def download(key: str, target: Path) -> int:
        assert step.metrics["progress"]["done"] == 0
        assert db.commit.called
        target.write_bytes(b"video")
        return 5

    storage.download_file.side_effect = download
    supervisor = JobSupervisor(Mock(), storage, WorkerConfig(workdir_root=tmp_path))
    supervisor._seed(db, job, ("upload",), tmp_path / "run", preparation=step)
    assert step.metrics["progress"]["done"] == step.metrics["progress"]["total"] == 1


def test_preparation_does_not_consume_an_attempt_or_reset_resumed_evidence() -> None:
    db = Mock()
    row = JobStep(status=RunStatus.COMPLETE, metrics={"frames": 100}, attempt=2)
    db.scalar.return_value = row
    assert steps.prepare_step(db, uuid.uuid4(), stage_id="normalize", ordinal=0, impl="x") is None
    assert row.metrics == {"frames": 100} and row.attempt == 2
    db.commit.assert_not_called()


def test_actual_stage_keeps_preparation_start_but_resets_its_progress() -> None:
    began = datetime.now(UTC) - timedelta(minutes=24)
    row = JobStep(
        started_at=began,
        status=RunStatus.IN_PROGRESS,
        attempt=1,
        metrics={steps.PREPARING: True, "progress": {"done": 60, "total": 60}},
    )
    db = Mock()
    db.scalar.return_value = row
    actual = steps.start_step(
        db,
        uuid.uuid4(),
        stage_id="normalize",
        ordinal=0,
        impl="ffmpeg_frames",
        attempt=1,
        started_at=datetime.now(UTC),
    )
    assert actual.started_at == began and actual.attempt == 1
    assert actual.metrics == {}


def test_extraction_reports_frame_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = Workdir.create(tmp_path).work_dir("normalize")
    (scratch / "extracted").mkdir(parents=True)
    (scratch / "extraction.json").write_text(json.dumps({"total": 12}))
    for i in range(3):
        (scratch / "extracted" / f"frame_{i:05d}.jpg").write_bytes(b"jpeg")
    report = Mock()
    monkeypatch.setattr(steps, "report_progress", report)
    JobSupervisor._report_progress(
        Mock(), JobStep(stage_id="normalize", impl="ffmpeg_frames"), tmp_path
    )
    assert report.call_args.args[2]["done"] == 3
    assert report.call_args.args[2]["total"] == 12


def test_reference_download_is_bounded_to_registered_sample_across_pages(tmp_path: Path) -> None:
    from app.worker.pipeline_bridge import image_names, sample_evenly

    # Unregistered normalized frames must not shift the sample the aligner will use.
    model = tmp_path / "images.bin"
    names = [f"frame_{i:05d}.jpg" for i in range(100, 1100)]
    with model.open("wb") as handle:
        handle.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(reversed(names)):
            handle.write(struct.pack("<IdddddddI", i + 1, 1, 0, 0, 0, 0, 0, 0, 1))
            handle.write(name.encode() + b"\0")
            handle.write(struct.pack("<Q", 1))
            handle.write(struct.pack("<ddq", 1, 2, -1))
    chosen = set(sample_evenly(image_names(model), 60))
    assert chosen == set(sample_evenly(names, 60))
    objects = tuple(
        ObjectSummary(
            key=f"frames/frame_{i:05d}.jpg", size=1024, etag="etag", last_modified=datetime.now(UTC)
        )
        for i in range(1200)
    )
    storage = Mock()
    storage.list_objects.side_effect = [
        ObjectPage(objects[:1000], "next"),
        ObjectPage(objects[1000:]),
    ]

    def download(key: str, target: Path) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 1024)
        return 1024

    storage.download_file.side_effect = download
    progress = Mock()
    assert (
        _download_tree(storage, "frames", tmp_path / "frames", names=chosen, progress=progress)
        == 60
    )
    assert {p.name for p in (tmp_path / "frames").iterdir()} == chosen
    assert sum(p.stat().st_size for p in (tmp_path / "frames").iterdir()) == 60 * 1024
    assert progress.call_args.args == (60 * 1024, 60 * 1024)
    assert storage.list_objects.call_args.kwargs["continuation_token"] == "next"


@pytest.mark.parametrize("limit", [7, 0])
def test_seed_uses_effective_alignment_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    from app.worker import params, site_context
    from app.worker.child import resolve_recipe
    from app.worker.pipeline_bridge import plan_recipe

    seed = Mock()
    monkeypatch.setattr(site_context, "seed_site_context", seed)
    monkeypatch.setattr(params, "located_point", lambda *args: None)
    db = Mock()
    capture = Capture(id=uuid.uuid4(), files=[])
    db.get.return_value = capture
    plan = plan_recipe(resolve_recipe("photo-reconstruct", None))
    JobSupervisor(Mock(), Mock(), WorkerConfig(workdir_root=tmp_path))._seed(
        db,
        Job(id=uuid.uuid4(), capture_id=capture.id),
        ("site_context",),
        tmp_path / "run",
        plan=plan,
        stage_params={"align": {"max_frames": limit}},
    )
    assert seed.call_args.kwargs["reference_frame_limit"] == limit


def test_preparation_retains_attempt_one_until_stage_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.worker.child import resolve_recipe
    from app.worker.pipeline_bridge import plan_recipe

    plan = plan_recipe(resolve_recipe("photo-reconstruct", None))
    row = JobStep(
        stage_id="normalize",
        attempt=1,
        status=RunStatus.IN_PROGRESS,
        metrics={steps.PREPARING: True},
    )
    monkeypatch.setattr(steps, "steps_of", lambda *args: {"normalize": row})
    supervisor = JobSupervisor(Mock(), Mock(), WorkerConfig(workdir_root=tmp_path))
    assert supervisor._attempts(Mock(), uuid.uuid4(), plan, set())["normalize"] == 1
    row.metrics = {}
    assert supervisor._attempts(Mock(), uuid.uuid4(), plan, set())["normalize"] == 2


def test_reference_fetch_downloads_poses_before_its_exact_frame_sample(tmp_path: Path) -> None:
    from app.services.epochs import ReferenceChoice
    from app.worker.site_context import _fetch_reference

    names = [f"frame_{i:05d}.jpg" for i in range(120)]
    binary = struct.pack("<Q", len(names))
    for i, name in enumerate(names):
        binary += struct.pack("<IdddddddI", i + 1, 1, 0, 0, 0, 0, 0, 0, 1)
        binary += name.encode() + b"\0" + struct.pack("<Q", 0)
    data = {f"frames/{name}": b"jpeg" for name in names}
    data.update(
        {
            "poses/images.bin": binary,
            "poses/points3D.bin": b"geometry",
            "placement.json": b"{}",
            "georef.json": b"{}",
        }
    )
    storage = Mock()

    def listing(prefix: str, **kwargs: object) -> ObjectPage:
        return ObjectPage(
            tuple(
                ObjectSummary(key, len(value), "etag", datetime.now(UTC))
                for key, value in data.items()
                if key.startswith(prefix)
            )
        )

    def download(key: str, path: Path) -> int:
        if key.startswith("frames/"):
            assert (tmp_path / "reference/poses/images.bin").is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data[key])
        return len(data[key])

    storage.list_objects.side_effect = listing
    storage.download_file.side_effect = download
    reference = ReferenceChoice(
        epoch_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        observed_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
        platform="phone-video",
        verdict="coarse",
        uncertainty_m=5,
        keys={
            "frames": "frames",
            "poses": "poses",
            "placement": "placement.json",
            "georef": "georef.json",
        },
    )
    assert _fetch_reference(storage, reference, tmp_path, frame_limit=7) is not None
    assert len(list((tmp_path / "reference/frames").iterdir())) == 7
    assert (tmp_path / "reference/poses/points3D.bin").read_bytes() == b"geometry"
    downloads = storage.download_file.call_count
    assert _fetch_reference(storage, reference, tmp_path, frame_limit=7) is not None
    assert storage.download_file.call_count == downloads


def test_retry_hydrates_a_new_sample_from_the_frozen_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.worker import site_context

    reference = tmp_path / "reference"
    (reference / "poses").mkdir(parents=True)
    names = [f"frame_{i:05d}.jpg" for i in range(10)]
    binary = struct.pack("<Q", len(names))
    for i, name in enumerate(names):
        binary += struct.pack("<IdddddddI", i + 1, 1, 0, 0, 0, 0, 0, 0, 1)
        binary += name.encode() + b"\0" + struct.pack("<Q", 0)
    (reference / "poses/images.bin").write_bytes(binary)
    document = {
        "siteId": "frozen",
        "reference": {
            "epochId": "frozen-epoch",
            "frameLimit": 2,
            "frameSource": "old/frames",
            "poses": "reference/poses",
            "frames": "reference/frames",
        },
    }
    path = tmp_path / "site_context.json"
    path.write_text(json.dumps(document))
    seed_plan = Mock(side_effect=AssertionError("must not choose another reference"))
    monkeypatch.setattr(site_context, "plan_seed", seed_plan)
    fetch = Mock(return_value=7)
    monkeypatch.setattr(site_context, "_download_tree", fetch)
    site_context.seed_site_context(Mock(), Mock(), None, tmp_path, reference_frame_limit=7)
    assert fetch.call_args.args[1] == "old/frames"
    assert len(fetch.call_args.kwargs["names"]) == 7
    saved = json.loads(path.read_text())
    assert saved["reference"]["epochId"] == "frozen-epoch"
    assert saved["reference"]["frameLimit"] == 7
