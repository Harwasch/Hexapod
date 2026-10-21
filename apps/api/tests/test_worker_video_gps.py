"""`worker/video_gps.py`: the video's own location, read before the run is placed.

The capture's video sits in a moto bucket as a synthetic container (`tests/mov_fixtures.
py`); the worker reads its location atoms by ranged GETs, records them on the capture,
and the frame its run is placed in comes from them -- not from where it was uploaded.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.models import Capture, CaptureFile, Job
from app.models.enums import CaptureKind, UploadStatus
from app.schemas.capture import CaptureCreate
from app.services import epochs
from app.storage import S3Storage
from app.worker import video_gps
from app.worker.config import WorkerConfig
from app.worker.pipeline_bridge import load_recipe, plan_recipe
from app.worker.runner import JobSupervisor
from tests.mov_fixtures import mov
from tests.test_epochs import ORIGIN, _north_of, make_site

AT_SITE = f"{ORIGIN[0]:+.4f}{ORIGIN[1]:+.4f}+030.000/"
#: Where the browser was when the clip was sent: 10 km north of the site.
HOME = {"lat": _north_of(10_000), "lon": ORIGIN[1]}


def _capture(
    db: Session,
    storage: S3Storage,
    files: dict[str, bytes],
    metadata: dict[str, Any] | None = None,
) -> Capture:
    capture = Capture(
        slug=f"c-{uuid.uuid4().hex[:8]}",
        name="c",
        kind=CaptureKind.VIDEO,
        metadata_={**HOME, **(metadata or {})},
    )
    db.add(capture)
    db.flush()
    for name, data in files.items():
        key = f"captures/{capture.id}/{name}"
        storage.put_object(key, data, "video/quicktime")
        db.add(
            CaptureFile(
                capture_id=capture.id,
                filename=name,
                bytes=len(data),
                storage_key=key,
                status=UploadStatus.COMPLETE,
            )
        )
    db.commit()
    db.refresh(capture)
    return capture


def test_the_videos_gps_is_recorded_and_the_run_is_placed_in_that_site(
    db: Session, sessions: sessionmaker[Session], storage: S3Storage, tmp_path: Path
) -> None:
    """Recorded on the site, uploaded from 10 km away: the run's frame is the site's."""
    site = make_site(db)
    capture = _capture(db, storage, {"IMG_0001.MOV": mov(iso6709=AT_SITE, accuracy="4.5")})
    job = Job(capture_id=capture.id, recipe="photo-reconstruct", recipe_version="2", params={})
    db.add(job)
    db.commit()
    config = WorkerConfig(workdir_root=tmp_path, worker_id="w", runner="local")

    resolved = JobSupervisor(sessions, storage, config)._stage_params(
        db, job, plan_recipe(load_recipe("photo-reconstruct"))
    )

    db.refresh(capture)
    assert capture.metadata_["videoLocation"] == {
        "lat": ORIGIN[0],
        "lon": ORIGIN[1],
        "accuracyM": 4.5,
        "key": "iso6709",
    }
    assert capture.metadata_["lat"] == HOME["lat"]  # the upload's fix is kept as it was
    frames = {
        stage: p["site_frame"]["siteId"] for stage, p in resolved.items() if "site_frame" in p
    }
    assert frames and set(frames.values()) == {str(site.id)}


def test_a_video_without_location_is_recorded_as_none_and_the_upload_decides(
    db: Session, storage: S3Storage
) -> None:
    make_site(db)
    capture = _capture(db, storage, {"IMG_0002.MOV": mov()})

    video_gps.record_video_location(db, storage, capture)

    assert "videoLocation" in capture.metadata_ and capture.metadata_["videoLocation"] is None
    point = epochs.join_point(capture, None)
    assert point is not None and point.source == "upload" and point.lat == HOME["lat"]
    assert epochs.frame_for_run(db, capture) is None


def test_it_is_read_once(db: Session, storage: S3Storage) -> None:
    capture = _capture(db, storage, {"a.mov": mov(iso6709=AT_SITE)})
    video_gps.record_video_location(db, storage, capture)
    recorded = dict(capture.metadata_)

    class Untouchable:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"the bucket was asked {name} again")

    video_gps.record_video_location(db, Untouchable(), capture)

    assert capture.metadata_ == recorded


def test_a_storage_failure_is_recorded_as_no_location_so_the_frame_never_moves(
    db: Session, storage: S3Storage
) -> None:
    """The frame is chosen again when a stage is retried (`runner._override`). A read that
    failed and was retried there could succeed and move a run's remaining stages into
    another site's frame, halfway: so a failed read is recorded as no location, the upload
    decides, and every later decision for the capture reads that same answer."""
    site = make_site(db)
    capture = _capture(db, storage, {"a.mov": mov(iso6709=AT_SITE)})

    class Down:
        def head_object(self, key: str) -> None:
            raise ConnectionError("the bucket is down")

    video_gps.record_video_location(db, Down(), capture)  # type: ignore[arg-type]
    first = epochs.frame_for_run(db, capture)
    video_gps.record_video_location(db, storage, capture)  # the retry, bucket back

    db.refresh(capture)
    assert "videoLocation" in capture.metadata_ and capture.metadata_["videoLocation"] is None
    point = epochs.join_point(capture, None)
    assert point is not None and point.source == "upload"
    assert first is None and epochs.frame_for_run(db, capture) is None and site.id


def test_the_video_read_is_the_one_the_pipeline_takes(db: Session, storage: S3Storage) -> None:
    """`pick_source` takes the first video by name; a second clip's location is not this
    capture's, and a Matroska file has no QuickTime atoms at all."""
    pair = _capture(db, storage, {"b.mov": mov(iso6709=AT_SITE), "a.mov": mov()})
    mkv = _capture(db, storage, {"a.mkv": b"\x1a\x45\xdf\xa3" + b"\0" * 64})
    photos = _capture(db, storage, {"a.jpg": b"\xff\xd8\xff"})

    for capture in (pair, mkv, photos):
        video_gps.record_video_location(db, storage, capture)

    assert pair.metadata_["videoLocation"] is None
    assert mkv.metadata_["videoLocation"] is None
    assert "videoLocation" not in photos.metadata_


def test_a_request_cannot_claim_a_video_location() -> None:
    """Only the worker writes it, from the bytes: a capture created with one is refused,
    so the join never trusts a location the request made up."""
    with pytest.raises(ValueError, match="videoLocation"):
        CaptureCreate(
            name="c",
            kind=CaptureKind.VIDEO,
            metadata={"videoLocation": {"lat": ORIGIN[0], "lon": ORIGIN[1], "accuracyM": 1}},
        )
