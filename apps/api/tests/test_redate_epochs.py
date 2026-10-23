"""The one-time correction of epochs dated by their upload (`app.scripts.redate_epochs`).

Each test makes the state production was in on 2026-10-22: an epoch whose first run read no
capture time, re-run by a run that did, but still dated by its upload, as `register` left
it before it learned to re-date (the state is set by hand, since `register` now re-dates).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.models import Capture, Epoch, JobStep, Site
from app.models.enums import RunStatus
from app.scripts import redate_epochs
from app.services.sites import site_to_summary
from app.storage import S3Storage
from tests.test_asset_scale import _document
from tests.test_registration_epochs import _assets, _capture, _epochs, run

TAKEN = datetime(2026, 9, 27, 19, 19, 47, tzinfo=UTC)


def _dated_by_its_upload(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    clock: str | None = "2026-09-27T12:19:47-07:00",
) -> tuple[Capture, Epoch]:
    """A capture run twice, the second run reading `clock`, its epoch left on its upload."""
    capture = _capture(db)
    run(db, storage, tmp_path, capture)
    document = _document()
    document["manifest"] = {"capture": {"capturedAtUtc": clock}}
    run(db, storage, tmp_path, capture, document)
    assert capture.site_id is not None
    assets = _assets(db, capture.site_id)
    [epoch] = _epochs(db, capture.site_id)
    epoch.observed_at, epoch.observed_at_source = capture.created_at, "created"
    for asset in assets:
        asset.observed_at = capture.created_at
    db.commit()
    return capture, epoch


def _site(db: Session, capture: Capture) -> Site:
    site = db.get(Site, capture.site_id)
    assert site is not None
    return site


def test_a_dry_run_lists_the_epoch_and_its_runs_date_and_writes_nothing(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)

    [finding] = redate_epochs.find(db)

    assert (finding.epoch_id, finding.old, finding.new) == (epoch.id, capture.created_at, TAKEN)
    assert finding.source is not None and finding.source.value == "manifest"
    assert finding.line().endswith(f"-> {TAKEN.isoformat()} (manifest)")
    db.rollback()
    db.expire_all()
    assert (epoch.observed_at, epoch.observed_at_source) == (capture.created_at, "created")


def test_apply_redates_the_epoch_its_current_asset_and_the_sites_date_and_is_idempotent(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    assert capture.site_id is not None

    [finding] = redate_epochs.apply(db)
    db.commit()

    assert finding.new == TAKEN
    db.expire_all()
    assert (epoch.observed_at, epoch.observed_at_source) == (TAKEN, "manifest")
    old, current = _assets(db, capture.site_id)
    # Only the current revision moves: a superseded one keeps the date it was shown with,
    # and the site's latest date reads the current one.
    assert (current.valid_to, current.observed_at) == (None, TAKEN)
    assert old.valid_to is not None and old.observed_at == capture.created_at
    assert site_to_summary(db, _site(db, capture)).latest_observed_at == TAKEN
    assert redate_epochs.find(db) == []
    assert redate_epochs.apply(db) == []


def test_apply_judges_each_epoch_again_under_its_lock_not_from_a_dry_run(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """What changed after a dry run is what apply sees: an epoch a run dated since is not
    touched, and an asset a run superseded since keeps its date."""
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    assert capture.site_id is not None
    [listed] = redate_epochs.find(db)
    assert listed.new == TAKEN
    said = datetime(2026, 9, 1, tzinfo=UTC)
    epoch.observed_at, epoch.observed_at_source = said, "captured-at"
    db.commit()

    assert redate_epochs.apply(db) == []

    db.expire_all()
    assert (epoch.observed_at, epoch.observed_at_source) == (said, "captured-at")
    _old, current = _assets(db, capture.site_id)
    epoch.observed_at, epoch.observed_at_source = capture.created_at, "created"
    current.valid_to = datetime.now(tz=UTC)
    db.commit()

    redate_epochs.apply(db)
    db.commit()

    db.expire_all()
    assert epoch.observed_at == TAKEN
    assert all(a.observed_at == capture.created_at for a in _assets(db, capture.site_id))


def test_a_document_from_an_older_run_than_the_epochs_is_not_used(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """`jobId` names the latest run, which may have written no document: the document's
    own run (`registrationJobId`) must be the epoch's."""
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    site = _site(db, capture)
    assert (site.metadata_ or {}).get("registrationJobId") == str(epoch.job_id)
    site.metadata_ = {**(site.metadata_ or {}), "registrationJobId": str(uuid.uuid4())}
    db.commit()

    [finding] = redate_epochs.find(db)

    assert (finding.new, finding.skipped) == (None, redate_epochs.NO_DOCUMENT)
    assert redate_epochs.apply(db) == [finding]
    db.expire_all()
    assert epoch.observed_at_source == "created"


def test_a_mirror_from_before_registration_job_id_needs_its_run_to_have_georeferenced(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Production's mirrors predate `registrationJobId`. A run that completed `georeference`
    wrote the document beside its `jobId`; one with no such step may not have."""
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    site = _site(db, capture)
    site.metadata_ = {k: v for k, v in (site.metadata_ or {}).items() if k != "registrationJobId"}
    db.commit()

    [finding] = redate_epochs.find(db)
    assert finding.skipped == redate_epochs.NO_DOCUMENT

    assert epoch.job_id is not None
    db.add(
        JobStep(
            job_id=epoch.job_id,
            stage_id="georeference",
            ordinal=3,
            impl="exif_gps",
            status=RunStatus.COMPLETE,
        )
    )
    db.commit()

    [finding] = redate_epochs.find(db)
    assert finding.new == TAKEN


def test_an_epoch_whose_run_read_no_date_is_left_alone(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, epoch = _dated_by_its_upload(db, storage, tmp_path, clock=None)

    [finding] = redate_epochs.find(db)

    assert (finding.epoch_id, finding.skipped) == (epoch.id, redate_epochs.NO_DATE)
    assert "left as it is" in finding.line()


def test_only_the_epochs_asked_for_are_looked_at(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, epoch = _dated_by_its_upload(db, storage, tmp_path)

    assert redate_epochs.find(db, [uuid.uuid4()]) == []
    assert [f.epoch_id for f in redate_epochs.find(db, [epoch.id])] == [epoch.id]


def test_a_mirror_from_before_registration_job_id_needs_a_georef_in_its_document(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A run that georeferenced always writes `georef`; a document without one is not that
    run's, whatever its step says."""
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    site = _site(db, capture)
    metadata = {k: v for k, v in (site.metadata_ or {}).items() if k != "registrationJobId"}
    metadata["registration"] = {k: v for k, v in metadata["registration"].items() if k != "georef"}
    site.metadata_ = metadata
    assert epoch.job_id is not None
    db.add(
        JobStep(
            job_id=epoch.job_id,
            stage_id="georeference",
            ordinal=3,
            impl="exif_gps",
            status=RunStatus.COMPLETE,
        )
    )
    db.commit()

    [finding] = redate_epochs.find(db)

    assert finding.skipped == redate_epochs.NO_DOCUMENT


def test_apply_judges_each_epoch_under_its_row_lock(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Apply holds each epoch it judges, even one it leaves alone, until its transaction
    ends: a run registering the epoch meanwhile waits for it. Shown with NOWAIT, so the
    test never waits: the epoch here is one whose run read no date, which apply does not
    write, so only the lock it took when it read the row can hold another session off."""
    _, epoch = _dated_by_its_upload(db, storage, tmp_path, clock=None)

    [finding] = redate_epochs.apply(db)

    assert finding.skipped == redate_epochs.NO_DATE
    with Session(db.get_bind()) as other:
        with pytest.raises(OperationalError):
            other.execute(select(Epoch).where(Epoch.id == epoch.id).with_for_update(nowait=True))
        other.rollback()
    db.rollback()
