"""The one-time correction of epochs dated by their upload (`app.scripts.redate_epochs`).

Each test makes the state production was in on 2026-10-22: an epoch whose first run read no
capture time, re-run by a run that did, but still dated by its upload, as `register` left
it before it learned to re-date (the state is set by hand, since `register` now re-dates).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Capture, Epoch, Site
from app.scripts import redate_epochs
from app.storage import S3Storage
from tests.test_asset_scale import _document
from tests.test_registration_epochs import _assets, _capture, _epochs, run

TAKEN = datetime(2026, 9, 27, 19, 19, 47, tzinfo=UTC)


def _dated_by_its_upload(
    db: Session, storage: S3Storage, tmp_path: Path, clock: str | None = "2026-09-27T12:19:47-07:00"
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


def test_a_dry_run_lists_the_epoch_and_its_runs_date_and_writes_nothing(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)

    [finding] = redate_epochs.find(db)

    assert (finding.epoch_id, finding.old, finding.new) == (epoch.id, capture.created_at, TAKEN)
    assert finding.source is not None and finding.source.value == "manifest"
    assert len(finding.assets) == 1
    assert finding.line().endswith(f"-> {TAKEN.isoformat()} (manifest), 1 current asset(s) with it")
    db.rollback()
    db.expire_all()
    assert (epoch.observed_at, epoch.observed_at_source) == (capture.created_at, "created")


def test_apply_redates_the_epoch_and_its_current_asset_and_a_second_run_finds_nothing(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    assert capture.site_id is not None

    assert redate_epochs.apply(db, redate_epochs.find(db)) == 1
    db.commit()

    db.expire_all()
    assert (epoch.observed_at, epoch.observed_at_source) == (TAKEN, "manifest")
    old, current = _assets(db, capture.site_id)
    # Only the current revision moves: a superseded one keeps the date it was shown with.
    assert (current.valid_to, current.observed_at) == (None, TAKEN)
    assert old.valid_to is not None and old.observed_at == capture.created_at
    assert redate_epochs.find(db) == []


def test_an_epoch_its_site_does_not_mirror_is_left_alone(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Without its current run's document in the database there is no evidence to use."""
    capture, epoch = _dated_by_its_upload(db, storage, tmp_path)
    site = db.get(Site, capture.site_id)
    assert site is not None
    site.metadata_ = {**(site.metadata_ or {}), "jobId": "someone-else"}
    db.commit()

    [finding] = redate_epochs.find(db)

    assert (finding.new, finding.skipped) == (None, redate_epochs.NO_DOCUMENT)
    assert redate_epochs.apply(db, [finding]) == 0
    db.expire_all()
    assert epoch.observed_at_source == "created"


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
