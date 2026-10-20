"""The worker seeds `site_context`: the site's reference epoch, its targets, the clicks.

Against PostGIS and moto: an epoch registered as the worker registers it, its run's
artifacts rowed and in the bucket as `worker/outputs` leaves them, and the seeder asked
for a second capture of the same site.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Artifact, Capture, CaptureFile, Epoch, Job, JobStep
from app.models.enums import RunStatus, UploadStatus
from app.schemas.check_target import CheckTargetSet
from app.services import check_targets, epochs
from app.storage import S3Storage
from app.worker import site_context
from app.worker.outputs import artifact_key, kind_for
from tests.test_registration_epochs import _capture, run

#: What each reference artifact holds in these tests: two frames, a model, two files.
FRAMES = {"frame_0000.jpg": b"jpeg-0", "frame_0001.jpg": b"jpeg-1"}
POSES = {
    "cameras.bin": b"c",
    "images.bin": b"i",
    "points3D.bin": b"p",
    "poses.json": b'{"cameras": 2}',
}
STAGES = {"frames": "normalize", "poses": "pose", "placement.json": "place"}
STAGES["georef.json"] = "georeference"

SURVEYED = {
    "label": "T1",
    "role": "check",
    "lat": 44.7966,
    "lon": -111.1345,
    "height": 1519.2,
    "heightDatum": "ellipsoid-wgs84",
    "sigmaHM": 0.005,
    "sigmaVM": 0.01,
    "source": "rtk-rover",
}


def keep_reference_artifacts(
    db: Session, storage: S3Storage, job_id: uuid.UUID, *, skip: tuple[str, ...] = ()
) -> None:
    """The run's frames, poses, placement and georef, rowed and uploaded as the worker
    leaves a Lane 2 run's outputs (`worker/outputs.upload_artifact`)."""
    for ordinal, (name, stage) in enumerate(STAGES.items()):
        if name in skip:
            continue
        step = JobStep(
            job_id=job_id,
            stage_id=stage,
            ordinal=ordinal,
            impl=stage,
            status=RunStatus.COMPLETE,
            metrics={},
        )
        db.add(step)
        db.flush()
        key = artifact_key(job_id, stage, name)
        members = {"frames": FRAMES, "poses": POSES}.get(name)
        if members is None:
            storage.put_object(key, json.dumps({"name": name}).encode(), "application/json")
        else:
            for member, data in members.items():
                storage.put_object(f"{key}/{member}", data, "application/octet-stream")
        db.add(Artifact(job_step_id=step.id, kind=kind_for(name), storage_key=key))
    db.commit()


def first_epoch(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    *,
    slug: str = "first",
    alignment: dict[str, Any] | None = None,
    platform: str | None = None,
) -> tuple[Capture, Epoch]:
    capture = _capture(db, slug)
    job_id = run(db, storage, tmp_path, capture)
    keep_reference_artifacts(db, storage, job_id)
    epoch = epochs.epoch_of_capture(db, capture.id)
    assert epoch is not None
    epoch.alignment = alignment
    if platform is not None:
        epoch.platform = platform
    db.commit()
    return capture, epoch


def second_capture(db: Session, site_id: uuid.UUID, slug: str = "second") -> Capture:
    return _capture(db, slug, metadata_={"siteId": str(site_id)})


def seeded(db: Session, storage: S3Storage, capture: Capture | None, root: Path) -> dict[str, Any]:
    directory = root / "site_context"
    site_context.seed_site_context(db, storage, capture, directory)
    document: dict[str, Any] = json.loads((directory / "site_context.json").read_text())
    return document


def test_a_capture_of_no_site_gets_the_empty_context(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db, "alone")
    assert seeded(db, storage, capture, tmp_path) == {
        "version": 1,
        "siteId": None,
        "reference": None,
        "targets": [],
        "observations": [],
    }


def test_a_published_reference_is_seeded_with_its_files_and_paths(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    alignment = {"verdict": "coarse", "uncertaintyM": 0.065, "label": "Drone, ±0.065 m"}
    first, epoch = first_epoch(db, storage, tmp_path, alignment=alignment, platform="drone")
    assert first.site_id is not None
    capture = second_capture(db, first.site_id)

    document = seeded(db, storage, capture, tmp_path)

    assert document["siteId"] == str(first.site_id)
    assert document["reference"] == {
        "epochId": str(epoch.id),
        "jobId": str(epoch.job_id),
        "observedAt": epoch.observed_at.isoformat(),
        "platform": "drone",
        "verdict": "coarse",
        "uncertaintyM": 0.065,
        "frames": "reference/frames",
        "poses": "reference/poses",
        "placement": "reference/placement.json",
        "georef": "reference/georef.json",
    }
    root = tmp_path / "site_context" / "reference"
    assert {p.name: p.read_bytes() for p in (root / "frames").iterdir()} == FRAMES
    assert {p.name: p.read_bytes() for p in (root / "poses").iterdir()} == POSES
    assert json.loads((root / "placement.json").read_text()) == {"name": "placement.json"}
    assert json.loads((root / "georef.json").read_text()) == {"name": "georef.json"}


def test_a_resumed_run_keeps_the_context_it_started_with(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, _epoch = first_epoch(db, storage, tmp_path, alignment={"verdict": "coarse"})
    assert first.site_id is not None
    capture = second_capture(db, first.site_id)
    before = seeded(db, storage, capture, tmp_path)

    # The site changes after the run started: a target is surveyed, the reference goes.
    check_targets.replace_active(
        db, first.site_id, CheckTargetSet.model_validate({"targets": [SURVEYED]})
    )
    for row in db.scalars(select(Epoch)):
        row.status = "superseded"
    db.commit()

    assert seeded(db, storage, capture, tmp_path) == before
    assert before["targets"] == [] and before["reference"] is not None


def test_targets_and_the_captures_clicks_are_merged_in(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, _epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None
    check_targets.replace_active(
        db,
        first.site_id,
        CheckTargetSet.model_validate(
            {"targets": [SURVEYED, {**SURVEYED, "label": "T2", "role": "control"}]}
        ),
    )
    capture = second_capture(db, first.site_id)
    clicks = [
        {"target": "T1", "image": "DJI_0042.JPG", "px": [1234.5, 876.0], "imageSize": [5472, 3648]}
    ]
    key = f"captures/{capture.id}/check_observations.json"
    storage.put_object(
        key, json.dumps({"version": 1, "observations": clicks}).encode(), "application/json"
    )
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="check_observations.json",
            storage_key=key,
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()

    document = seeded(db, storage, capture, tmp_path)

    expected = {k: v for k, v in SURVEYED.items() if k != "source"}
    assert document["targets"] == [expected, {**expected, "label": "T2", "role": "control"}]
    assert document["observations"] == clicks


def test_a_malformed_clicks_file_contributes_none(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db, "clicked")
    key = f"captures/{capture.id}/check_observations.json"
    storage.put_object(key, b"not json", "application/json")
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="check_observations.json",
            storage_key=key,
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()
    assert seeded(db, storage, capture, tmp_path)["observations"] == []


def test_an_interrupted_seed_never_mixes_two_references(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A seed that stopped mid-download of another epoch's frames, before it wrote the
    JSON, starts over with the reference it now chooses, and none of the other's files."""
    first, epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None
    stale = tmp_path / "site_context" / "reference"
    (stale / "frames").mkdir(parents=True)
    (stale / "frames" / "frame_0099.jpg").write_bytes(b"another epoch's frame")
    (stale / site_context.REFERENCE_MARKER).write_text(json.dumps({"epochId": "other"}))

    document = seeded(db, storage, second_capture(db, first.site_id), tmp_path)

    assert document["reference"]["epochId"] == str(epoch.id)
    assert sorted(p.name for p in (stale / "frames").iterdir()) == sorted(FRAMES)


def test_a_reference_whose_files_are_gone_is_not_seeded(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None and epoch.job_id is not None
    storage.delete_object(artifact_key(epoch.job_id, "place", "placement.json"))
    document = seeded(db, storage, second_capture(db, first.site_id), tmp_path)
    assert document["reference"] is None
    assert document["siteId"] == str(first.site_id)


# --- which epoch is the reference ---------------------------------------------------------


def _reference(db: Session, site_id: uuid.UUID, capture: Capture) -> epochs.ReferenceChoice | None:
    return epochs.reference_epoch(db, site_id, capture_id=capture.id)


def _join(
    db: Session, storage: S3Storage, tmp_path: Path, site_id: uuid.UUID, slug: str, **fields: Any
) -> Epoch:
    """Another epoch of the site, registered in its frame, with its artifacts kept."""
    from tests.test_registration_epochs import _echoed

    capture = second_capture(db, site_id, slug)
    job_id = run(db, storage, tmp_path, capture, _echoed(site_id))
    keep_reference_artifacts(db, storage, job_id)
    epoch = epochs.epoch_of_capture(db, capture.id)
    assert epoch is not None
    for name, value in fields.items():
        setattr(epoch, name, value)
    db.commit()
    return epoch


def test_the_best_verdict_wins_then_the_oldest(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, one = first_epoch(db, storage, tmp_path, alignment={"verdict": "coarse"})
    assert first.site_id is not None
    two = _join(
        db, storage, tmp_path, first.site_id, "rtk",
        platform="rtk-drone", alignment={"verdict": "survey", "uncertaintyM": 0.03},
    )  # fmt: skip
    newcomer = second_capture(db, first.site_id, "newcomer")
    chosen = _reference(db, first.site_id, newcomer)
    assert chosen is not None and chosen.epoch_id == two.id and chosen.verdict == "survey"

    two.alignment = {"verdict": "coarse"}
    db.commit()
    chosen = _reference(db, first.site_id, newcomer)
    assert chosen is not None and chosen.epoch_id == one.id


def test_only_an_rtk_drone_epoch_is_passed_on_as_survey(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """D6 amendment: "survey" may only be claimed for platform rtk-drone. A phone epoch
    whose record says survey (it never should) reaches the pipeline as coarse, so an RTK
    epoch aligned to it keeps its own frame (`alignment.py`, the M3b review's fix 11)."""
    first, _epoch = first_epoch(
        db, storage, tmp_path, platform="phone-video", alignment={"verdict": "survey"}
    )
    assert first.site_id is not None
    chosen = _reference(db, first.site_id, second_capture(db, first.site_id))
    assert chosen is not None and chosen.verdict == "coarse"
    assert epochs.claimed_verdict({"verdict": "survey"}, "rtk-drone") == "survey"
    assert epochs.claimed_verdict({"verdict": "surveyed"}, "rtk-drone") is None
    assert epochs.claimed_verdict(None, "rtk-drone") is None


def test_an_epoch_unrecorded_takes_its_captures_uncertainty(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, _epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None
    chosen = _reference(db, first.site_id, second_capture(db, first.site_id))
    # `_document()`'s georef: 10 m, what the capture was registered with.
    assert chosen is not None and chosen.verdict is None and chosen.uncertainty_m == 10.0


@pytest.mark.parametrize("why", ["rejected", "running", "no-placement"])
def test_what_cannot_be_a_reference_is_not(
    db: Session, storage: S3Storage, tmp_path: Path, why: str
) -> None:
    first = _capture(db, "first")
    job_id = run(db, storage, tmp_path, first)
    keep_reference_artifacts(
        db, storage, job_id, skip=("placement.json",) if why == "no-placement" else ()
    )
    epoch = epochs.epoch_of_capture(db, first.id)
    assert epoch is not None and first.site_id is not None
    if why == "rejected":
        epoch.status = "rejected"
    if why == "running":
        job = db.get(Job, job_id)
        assert job is not None
        job.status = RunStatus.IN_PROGRESS
    db.commit()
    assert _reference(db, first.site_id, second_capture(db, first.site_id)) is None


def test_a_re_run_never_aligns_to_its_own_or_a_later_epoch(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, one = first_epoch(db, storage, tmp_path, alignment={"verdict": "coarse"})
    assert first.site_id is not None
    two = _join(db, storage, tmp_path, first.site_id, "later", alignment={"verdict": "coarse"})
    assert _reference(db, first.site_id, first) is None
    assert two.capture_id is not None
    later = db.get(Capture, two.capture_id)
    assert later is not None
    chosen = _reference(db, first.site_id, later)
    assert chosen is not None and chosen.epoch_id == one.id
