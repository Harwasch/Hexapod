"""Every registered capture is an epoch of a site; a re-run is a new revision of it (D3).

Written against `registration.register` the way the worker calls it, with the
`registration.json` documents of `test_asset_scale` and no pipeline: what is checked is
what lands in the database.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models import Asset, Capture, Epoch, Site, SiteFrame
from app.models.enums import CaptureKind
from app.storage import S3Storage
from app.worker import registration
from app.worker.carry import CarryPlan
from tests.test_asset_scale import HEIGHT, LAT, LON, _document, _job, _registration


def _capture(db: Session, slug: str = "spool", **fields: Any) -> Capture:
    capture = Capture(slug=slug, name=slug.title(), kind=CaptureKind.VIDEO, **fields)
    if capture.metadata_ is None:
        capture.metadata_ = {}
    db.add(capture)
    db.commit()
    return capture


def run(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    capture: Capture,
    document: dict[str, Any] | None = None,
    *,
    url: str | None = "new",
    coverage: str | None = None,
    withheld: bool = False,
) -> uuid.UUID:
    """One run of the capture registered as the worker registers it; its job id."""
    job = _job(db, capture)
    tiles = (
        None
        if url is None or withheld
        else (
            f"https://cdn.example.com/twin-worker-test/runs/{job.id}/package/splat/tileset.json"
            if url == "new"
            else url
        )
    )
    live = registration.live_tileset_url(db, capture)
    registration.register(
        db,
        storage,
        capture=capture,
        job_id=job.id,
        registration=_registration(tmp_path, document or _document()),
        tiles_stage_id="package",
        published=registration.Published(
            tileset=tiles,
            thumbnail=None,
            coverage=coverage,
            withheld=withheld,
            carry=CarryPlan(based_on=live),
        ),
    )
    return job.id


def _epochs(db: Session, site_id: uuid.UUID) -> list[Epoch]:
    db.expire_all()
    return list(db.scalars(select(Epoch).where(Epoch.site_id == site_id).order_by(Epoch.ordinal)))


def _assets(db: Session, site_id: uuid.UUID) -> list[Asset]:
    db.expire_all()
    return list(
        db.scalars(select(Asset).where(Asset.site_id == site_id).order_by(Asset.created_at))
    )


def _count(db: Session, model: type[Any]) -> int:
    return int(db.scalar(select(func.count()).select_from(model)) or 0)


def test_a_first_run_makes_a_site_its_frame_and_epoch_1(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    captured = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
    capture = _capture(db, captured_at=captured)

    job_id = run(db, storage, tmp_path, capture)

    db.expire_all()
    assert capture.site_id is not None
    frame = db.get(SiteFrame, capture.site_id)
    assert frame is not None
    assert (frame.lat, frame.lon, frame.height) == (LAT, LON, HEIGHT)
    assert (frame.height_datum, frame.source, frame.epsg) == (
        "msl-unverified",
        "registration",
        32612,
    )
    [epoch] = _epochs(db, capture.site_id)
    assert (epoch.ordinal, epoch.capture_id, epoch.job_id) == (1, capture.id, job_id)
    assert (epoch.observed_at, epoch.observed_at_source) == (captured, "captured-at")
    assert (epoch.platform, epoch.status, epoch.join_method) == (
        "unknown",
        "published",
        "new-site",
    )
    [asset] = _assets(db, capture.site_id)
    assert asset.epoch_id == epoch.id
    assert asset.observed_at == captured
    assert asset.default_visible is True
    assert asset.source_kind == "observed"
    assert asset.render_config["clampToGround"] is True
    site = db.get(Site, capture.site_id)
    assert site is not None and site.metadata_["captureId"] == str(capture.id)


def test_a_re_run_inserts_a_revision_and_supersedes_the_old_one(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    run(db, storage, tmp_path, capture)
    assert capture.site_id is not None
    [first] = _assets(db, capture.site_id)
    snapshot = db.execute(
        text(
            "SELECT to_jsonb(a) - 'valid_to' - 'default_visible' - 'updated_at' "
            "FROM assets a WHERE id = :id"
        ),
        {"id": first.id},
    ).scalar_one()

    second_job = run(db, storage, tmp_path, capture)

    old, new = _assets(db, capture.site_id)
    assert old.id == first.id
    after = db.execute(
        text(
            "SELECT to_jsonb(a) - 'valid_to' - 'default_visible' - 'updated_at' "
            "FROM assets a WHERE id = :id"
        ),
        {"id": first.id},
    ).scalar_one()
    assert after == snapshot
    assert old.valid_to is not None and old.default_visible is False
    assert new.valid_to is None and new.default_visible is True
    assert str(second_job) in new.source["url"] and str(second_job) not in old.source["url"]
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.job_id == second_job
    assert new.epoch_id == epoch.id
    assert registration.splat_asset(db, capture.site_id) == new


def test_a_republish_of_the_same_url_makes_no_revision(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    same = "https://cdn.example.com/twin-worker-test/runs/x/package/splat/tileset.json"
    run(db, storage, tmp_path, capture, url=same)
    assert capture.site_id is not None

    run(db, storage, tmp_path, capture, url=same)

    [asset] = _assets(db, capture.site_id)
    assert asset.valid_to is None and asset.source["url"] == same


@pytest.mark.parametrize(
    ("captured_at", "manifest", "source"),
    [
        (datetime(2026, 9, 12, tzinfo=UTC), "2026-01-01T00:00:00Z", "captured-at"),
        (None, "2026-08-30T14:05:00Z", "manifest"),
        (None, None, "created"),
    ],
)
def test_observed_at_comes_from_the_capture_then_the_manifest_then_the_upload(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    captured_at: datetime | None,
    manifest: str | None,
    source: str,
) -> None:
    capture = _capture(db, captured_at=captured_at)
    document = _document()
    document["manifest"] = {"capture": {"capturedAt": manifest}}

    run(db, storage, tmp_path, capture, document)

    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.observed_at_source == source
    expected = {
        "captured-at": captured_at,
        "manifest": datetime(2026, 8, 30, 14, 5, tzinfo=UTC),
        "created": capture.created_at,
    }[source]
    assert epoch.observed_at == expected


def test_a_run_placed_about_another_origin_is_refused_and_writes_nothing(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    run(db, storage, tmp_path, capture)
    elsewhere = _document()
    elsewhere["georef"]["lat"] = LAT + 0.001
    assets, epochs_before = _count(db, Asset), _count(db, Epoch)

    with pytest.raises(registration.FrameMismatch, match="frame is about"):
        run(db, storage, tmp_path, capture, elsewhere)

    db.rollback()
    assert (_count(db, Asset), _count(db, Epoch)) == (assets, epochs_before)


def test_an_echo_that_disagrees_with_its_georef_is_refused(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    document = _document()
    document["georef"]["siteFrame"] = {
        "siteId": str(uuid.uuid4()),
        "lat": LAT + 0.01,
        "lon": LON,
        "height": HEIGHT,
    }

    with pytest.raises(registration.FrameMismatch, match="georeference is about"):
        run(db, storage, tmp_path, capture, document)
    db.rollback()
    malformed = _document()
    malformed["georef"]["siteFrame"] = {"siteId": "nope"}
    with pytest.raises(registration.FrameMismatch, match="malformed"):
        run(db, storage, tmp_path, capture, malformed)
    db.rollback()
    assert _count(db, Epoch) == 0 and _count(db, Site) == 0


def test_quality_and_coverage_land_on_the_epoch_and_the_mirror(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    document = _document()
    document["quality"] = {"mode": "preview", "bar": "strict", "keepVerifiedPct": 81.5}
    coverage = "https://cdn.example.com/twin-worker-test/runs/x/quality/coverage_enu.ply"

    job_id = run(db, storage, tmp_path, capture, document, coverage=coverage)

    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.quality is not None
    assert epoch.quality["coverageUrl"] == coverage
    assert epoch.quality["keepVerifiedPct"] == 81.5
    assert epoch.quality["jobId"] == str(job_id)
    assert capture.quality == epoch.quality
    site = db.get(Site, capture.site_id)
    assert site is not None
    assert site.metadata_["coverageUrl"] == coverage
    assert site.metadata_["keepVerifiedPct"] == 81.5
    assert site.metadata_["jobId"] == str(job_id)


def test_a_withheld_first_run_still_makes_its_epoch_and_a_later_run_gives_it_tiles(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    run(db, storage, tmp_path, capture, withheld=True)
    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert _assets(db, capture.site_id) == []

    second = run(db, storage, tmp_path, capture)

    [asset] = _assets(db, capture.site_id)
    assert asset.epoch_id == epoch.id and str(second) in asset.source["url"]
    assert _epochs(db, capture.site_id)[0].job_id == second


def test_a_run_with_no_georef_leaves_its_site_frameless(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    document = copy.deepcopy(_document())
    del document["georef"]

    run(db, storage, tmp_path, capture, document)

    assert capture.site_id is not None
    assert db.get(SiteFrame, capture.site_id) is None
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.ordinal == 1
