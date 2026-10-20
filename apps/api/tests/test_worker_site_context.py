"""The worker seeds `site_context`: the site's reference epoch, its targets, the clicks.

Against PostGIS and moto: an epoch registered as the worker registers it, its run's
artifacts rowed and in the bucket as `worker/outputs` leaves them, and the seeder asked
for a second capture of the same site.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
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
    leaves a Lane 2 run's outputs (`worker/outputs.upload_artifact`). Rowed when the run
    started, as the worker rows them before it registers the run."""
    job = db.get(Job, job_id)
    assert job is not None
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
        db.add(
            Artifact(
                job_step_id=step.id,
                kind=kind_for(name),
                storage_key=key,
                created_at=job.created_at,
            )
        )
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

    chosen = _reference(db, first.site_id, capture)
    assert chosen is not None and len(chosen.revision) == 16
    assert document["siteId"] == str(first.site_id)
    assert document["reference"] == {
        "epochId": str(epoch.id),
        "jobId": str(epoch.job_id),
        "observedAt": epoch.observed_at.isoformat(),
        "platform": "drone",
        "verdict": "coarse",
        "uncertaintyM": 0.065,
        "revision": chosen.revision,
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


# --- a Refine of the reference epoch (same job row, same keys) -----------------------------


def _refine(
    db: Session, storage: S3Storage, epoch: Epoch, status: RunStatus, *, placed: bool
) -> None:
    """A Refine of the reference's run (`jobs.retry_job` on the same row, resuming at
    `train`): the job goes back to `status`, and -- when the Refine got as far as `place`
    -- a new `placement.json` row is uploaded over the same key, as `worker/outputs` does."""
    assert epoch.job_id is not None
    job = db.get(Job, epoch.job_id)
    assert job is not None
    job.status = status
    if placed:
        step = db.scalars(
            select(JobStep).where(JobStep.job_id == job.id, JobStep.stage_id == "place")
        ).one()
        key = artifact_key(job.id, "place", "placement.json")
        storage.put_object(key, b'{"refined": true}', "application/json")
        db.add(Artifact(job_step_id=step.id, kind=kind_for("placement.json"), storage_key=key))
    db.commit()


@pytest.mark.parametrize("status", [RunStatus.NOT_STARTED, RunStatus.IN_PROGRESS])
def test_a_refine_in_flight_is_not_read_mid_write(
    db: Session, storage: S3Storage, tmp_path: Path, status: RunStatus
) -> None:
    first, _epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None
    _refine(db, storage, _epoch, status, placed=False)
    assert _reference(db, first.site_id, second_capture(db, first.site_id)) is None


def test_a_refine_that_failed_leaves_the_epoch_the_reference(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The Refine failed before it re-uploaded anything: the bucket still holds the files
    the epoch's shown revision was made from, so it is still the site's scaffold (one
    fixed reference; a later epoch must not silently align to another, or to none)."""
    first, epoch = first_epoch(db, storage, tmp_path, alignment={"verdict": "coarse"})
    assert first.site_id is not None
    _refine(db, storage, epoch, RunStatus.ERROR, placed=False)
    chosen = _reference(db, first.site_id, second_capture(db, first.site_id))
    assert chosen is not None and chosen.epoch_id == epoch.id


def test_a_refine_that_failed_after_replacing_the_placement_is_not_a_reference(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """It got as far as `place` and then failed: the placement under the key is the new
    training's, and the epoch still shows the old tiles. Aligning to that pair would put
    the new epoch about a placement nobody is shown."""
    first, epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None
    _refine(db, storage, epoch, RunStatus.ERROR, placed=True)
    assert _reference(db, first.site_id, second_capture(db, first.site_id)) is None


def test_a_refine_that_succeeded_is_the_reference_with_its_new_files(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The same run row, resumed at `train`, uploads a new placement over the same key and
    registers a new revision of the epoch: the epoch is the reference again, its files the
    Refine's."""
    from app.worker import registration
    from app.worker.carry import CarryPlan
    from tests.test_asset_scale import _document, _registration

    first, epoch = first_epoch(db, storage, tmp_path, alignment={"verdict": "coarse"})
    assert first.site_id is not None and epoch.job_id is not None
    _refine(db, storage, epoch, RunStatus.IN_PROGRESS, placed=True)
    job = db.get(Job, epoch.job_id)
    assert job is not None
    job.status = RunStatus.COMPLETE
    db.commit()
    live = registration.live_tileset_url(db, first)
    registration.register(
        db, storage, capture=first, job_id=job.id,
        registration=_registration(tmp_path, _document()), tiles_stage_id="package",
        published=registration.Published(
            tileset=f"https://cdn.example.com/twin-worker-test/runs/{job.id}/refined/tileset.json",
            thumbnail=None, coverage=None, carry=CarryPlan(based_on=live),
        ),
    )  # fmt: skip
    capture = second_capture(db, first.site_id)
    chosen = _reference(db, first.site_id, capture)
    assert chosen is not None and chosen.epoch_id == epoch.id
    document = seeded(db, storage, capture, tmp_path)
    placement = tmp_path / "site_context" / document["reference"]["placement"]
    assert json.loads(placement.read_text()) == {"refined": True}


def test_a_refine_that_republished_the_same_tiles_is_still_the_reference(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """R2 review of #52: a retry from `place` that produced the very same content-hashed
    tiles keeps the shown asset and its timestamp (no new revision), while its placement
    row is new -- so the files looked newer than the tiles, and the epoch stopped being
    any later capture's reference. A successful publication confirms its files."""
    from app.worker import registration
    from app.worker.carry import CarryPlan
    from tests.test_asset_scale import _document, _registration

    first, epoch = first_epoch(db, storage, tmp_path, alignment={"verdict": "coarse"})
    assert first.site_id is not None and epoch.job_id is not None
    _refine(db, storage, epoch, RunStatus.IN_PROGRESS, placed=True)
    job = db.get(Job, epoch.job_id)
    assert job is not None
    job.status = RunStatus.COMPLETE
    db.commit()
    live = registration.live_tileset_url(db, first)
    assert live is not None
    shown = epochs.current_asset(db, epoch.id)
    assert shown is not None
    registration.register(
        db, storage, capture=first, job_id=job.id,
        registration=_registration(tmp_path, _document()), tiles_stage_id="package",
        published=registration.Published(
            tileset=live, thumbnail=None, coverage=None, carry=CarryPlan(based_on=live),
        ),
    )  # fmt: skip
    assert epochs.current_asset(db, epoch.id) == shown  # the same tiles: no new revision
    capture = second_capture(db, first.site_id)
    chosen = _reference(db, first.site_id, capture)
    assert chosen is not None and chosen.epoch_id == epoch.id


# --- what the bucket may do to a seed ----------------------------------------------------


def test_a_clicks_file_over_the_limit_is_never_read(
    db: Session, storage: S3Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker has 2 GB; a capture file is whatever was uploaded. Clicks are a few kB
    each, so one past `MAX_OBSERVATIONS_BYTES` is not clicks: skipped, saying why, without
    downloading a byte of it."""
    capture = _capture(db, "huge-clicks")
    key = f"captures/{capture.id}/check_observations.json"
    storage.put_object(key, b" " * (site_context.MAX_OBSERVATIONS_BYTES + 1), "application/json")
    db.add(
        CaptureFile(
            capture_id=capture.id,
            filename="check_observations.json",
            storage_key=key,
            status=UploadStatus.COMPLETE,
        )
    )
    db.commit()
    said: list[str] = []
    monkeypatch.setattr(site_context.log, "warning", lambda message, *a: said.append(message % a))
    document = seeded(db, storage, capture, tmp_path)
    assert document["observations"] == []
    assert not (tmp_path / "site_context" / site_context.OBSERVATIONS_FILE).exists()
    assert any("over the 5242880" in line and "not read" in line for line in said), said


@pytest.mark.parametrize("vanishes", ["frames/frame_0001.jpg", "placement.json"])
def test_a_reference_object_that_vanishes_mid_seed_falls_back_to_the_georeference(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vanishes: str,
) -> None:
    """Listed, then gone before its download (a lifecycle rule, a person tidying the
    bucket): the run is aligned by its own georeference, as when it was gone from the
    start -- never dead-lettered on a scaffold that is not there."""
    first, epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None and epoch.job_id is not None
    stage = "place" if vanishes == "placement.json" else "normalize"
    doomed = f"runs/{epoch.job_id}/{stage}/{vanishes}"
    real = storage.download_file

    def download(key: str, target: Path) -> int:
        if key == doomed:
            storage.delete_object(key)
        return real(key, target)

    monkeypatch.setattr(storage, "download_file", download)
    document = seeded(db, storage, second_capture(db, first.site_id), tmp_path)
    assert document["reference"] is None
    assert document["siteId"] == str(first.site_id)


def test_an_interrupted_seed_never_mixes_two_revisions_of_one_reference(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """R2 review of #52: the files a seed had already downloaded were kept by epoch and job
    alone. A retry of the reference's run rewrites the same keys under the same ids, so a
    seed interrupted before the retry resumed with the old revision's files beside the new
    one's. The cache is keyed by the revision -- the artifact rows it was read from."""
    first, epoch = first_epoch(db, storage, tmp_path)
    assert first.site_id is not None and epoch.job_id is not None
    capture = second_capture(db, first.site_id)
    before = _reference(db, first.site_id, capture)
    assert before is not None
    directory = tmp_path / "site_context"
    assert site_context._fetch_reference(storage, before, directory) is not None  # then cut off

    # The reference's run is retried and rewrites its frames under the same key.
    job = db.get(Job, epoch.job_id)
    assert job is not None
    step = db.scalars(
        select(JobStep).where(JobStep.job_id == job.id, JobStep.stage_id == "normalize")
    ).one()
    key = artifact_key(job.id, "normalize", "frames")
    storage.put_object(f"{key}/frame_0000.jpg", b"retried-0", "application/octet-stream")
    db.add(
        Artifact(
            job_step_id=step.id,
            kind=kind_for("frames"),
            storage_key=key,
            # Newer than the first row, and no newer than the tiles the epoch shows.
            created_at=job.created_at + timedelta(milliseconds=1),
        )
    )
    db.commit()
    after = _reference(db, first.site_id, capture)
    assert after is not None and after.revision != before.revision

    document = seeded(db, storage, capture, tmp_path)

    assert document["reference"]["revision"] == after.revision
    frames = directory / document["reference"]["frames"]
    assert (frames / "frame_0000.jpg").read_bytes() == b"retried-0"
