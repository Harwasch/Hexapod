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


@pytest.mark.parametrize(
    ("utc", "source", "expected"),
    [
        # M3a copies `capturedAtUtc` into `capture.captured_at` too; read first, it keeps
        # the `manifest` label rather than passing for a date a person gave.
        ("2026-08-30T14:05:00+00:00", "manifest", datetime(2026, 8, 30, 14, 5, tzinfo=UTC)),
        ("2026-08-30T07:05:00-07:00", "manifest", datetime(2026, 8, 30, 14, 5, tzinfo=UTC)),
        # Not tz-aware, so not a `capturedAtUtc`: the capture row's date is next.
        ("2026-08-30T14:05:00", "captured-at", datetime(2026, 9, 12, tzinfo=UTC)),
        ("not a time", "captured-at", datetime(2026, 9, 12, tzinfo=UTC)),
    ],
)
def test_the_manifests_utc_capture_time_is_read_first_and_labelled_manifest(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    utc: str,
    source: str,
    expected: datetime,
) -> None:
    capture = _capture(db, captured_at=datetime(2026, 9, 12, tzinfo=UTC))
    document = _document()
    document["manifest"] = {"capture": {"capturedAtUtc": utc, "capturedAt": "2026-01-01"}}

    run(db, storage, tmp_path, capture, document)

    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert (epoch.observed_at, epoch.observed_at_source) == (expected, source)


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


# ── Joining a site (the capture's first run, onto a site that exists) ──────────────────


def _echoed(site_id: uuid.UUID, **georef: Any) -> dict[str, Any]:
    """`_document()` as a run placed in `site_id`'s frame (about LAT, LON, HEIGHT) writes it."""
    document = _document()
    document["georef"].update(georef)
    document["georef"]["siteFrame"] = {
        "siteId": str(site_id),
        "lat": document["georef"]["lat"],
        "lon": document["georef"]["lon"],
        "height": document["georef"]["height"],
        "offsetM": [0.0, 10.0, 0.0],
        "offsetSource": "capture",
        "located": None,
    }
    return document


def test_an_explicit_site_without_a_frame_gets_one_from_its_first_epoch(
    db: Session, storage: S3Storage, tmp_path: Path, client: Any
) -> None:
    from tests.conftest import site_payload

    created = client.post("/api/v1/sites", json=site_payload(slug="ranch"))
    assert created.status_code == 201, created.text
    site_id = uuid.UUID(created.json()["id"])
    [legacy] = _assets(db, site_id)
    legacy_row = db.execute(
        text("SELECT to_jsonb(a) FROM assets a WHERE id = :id"), {"id": legacy.id}
    ).scalar_one()
    capture = _capture(db, metadata_={"siteId": str(site_id)})

    run(db, storage, tmp_path, capture)

    db.expire_all()
    assert capture.site_id == site_id
    frame = db.get(SiteFrame, site_id)
    assert frame is not None and (frame.lat, frame.lon, frame.source) == (
        LAT,
        LON,
        "registration",
    )
    [epoch] = _epochs(db, site_id)
    assert (epoch.ordinal, epoch.join_method) == (1, "explicit")
    # The seeded or hand-made asset is not this epoch's and is not touched.
    after = db.execute(
        text("SELECT to_jsonb(a) FROM assets a WHERE id = :id"), {"id": legacy.id}
    ).scalar_one()
    assert after == legacy_row
    assert {a.epoch_id for a in _assets(db, site_id)} == {None, epoch.id}


def test_an_echoed_frame_joins_that_site_as_the_next_epoch(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    named = _capture(db, "named", metadata_={"siteId": str(first.site_id)})
    matched = _capture(db, "matched")

    run(db, storage, tmp_path, named, _echoed(first.site_id))
    run(db, storage, tmp_path, matched, _echoed(first.site_id))

    assert named.site_id == matched.site_id == first.site_id
    epochs = _epochs(db, first.site_id)
    assert [(e.ordinal, e.join_method) for e in epochs] == [
        (1, "new-site"),
        (2, "explicit"),
        (3, "spatial"),
    ]
    assert [e.capture_id for e in epochs] == [first.id, named.id, matched.id]
    # A later epoch is not the default view until the timeline exists (M2b).
    visible = {a.epoch_id: a.default_visible for a in _assets(db, first.site_id)}
    assert visible == {epochs[0].id: True, epochs[1].id: False, epochs[2].id: False}
    assert _count(db, Site) == 1


def test_an_echo_whose_site_has_another_frame_is_refused(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    other = _capture(db, "other")
    elsewhere = _echoed(first.site_id, lat=LAT + 0.001)

    with pytest.raises(registration.FrameMismatch, match="that site now has"):
        run(db, storage, tmp_path, other, elsewhere)
    db.rollback()
    assert len(_epochs(db, first.site_id)) == 1


def test_an_echo_naming_a_deleted_site_makes_a_new_site_in_that_frame(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    gone = uuid.uuid4()
    capture = _capture(db)

    run(db, storage, tmp_path, capture, _echoed(gone))

    assert capture.site_id is not None and capture.site_id != gone
    frame = db.get(SiteFrame, capture.site_id)
    assert frame is not None
    assert (frame.lat, frame.lon, frame.height, frame.source) == (LAT, LON, HEIGHT, "echo")
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.join_method == "new-site"


def test_a_joined_epoch_grows_the_site_boundary(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    from geoalchemy2.shape import to_shape

    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    site = db.get(Site, first.site_id)
    assert site is not None
    before = to_shape(site.boundary)
    centroid = to_shape(site.centroid)
    north = _echoed(first.site_id)
    north["bboxLocalM"] = {"min": [-4.0, 40.0, -1.0], "max": [6.0, 50.0, 3.0]}
    second = _capture(db, "second")

    run(db, storage, tmp_path, second, north)

    db.expire_all()
    site = db.get(Site, first.site_id)
    assert site is not None
    after = to_shape(site.boundary)
    epochs = _epochs(db, first.site_id)
    assert after.covers(before) and after.area > before.area
    assert all(after.buffer(1e-9).covers(to_shape(e.footprint)) for e in epochs)
    assert to_shape(site.centroid).equals(centroid)


def test_a_later_epoch_leaves_the_mirror_of_the_splat_the_site_shows(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """`view.html` shows the site's first splat (epoch 1, `default_visible`) and reads its
    coverage overlay off the site's metadata. A later epoch's run must not lay its own
    coverage over epoch 1's tiles: the mirror is written for epoch 1 only, and the later
    epoch keeps its coverage on its own row."""
    first = _capture(db, "first")
    epoch1 = "https://cdn.example.com/twin-worker-test/runs/1/quality/coverage_enu.ply"
    epoch2 = "https://cdn.example.com/twin-worker-test/runs/2/quality/coverage_enu.ply"
    measured = _document()
    measured["quality"] = {"mode": "preview", "bar": "strict", "keepVerifiedPct": 81.5}
    first_job = run(db, storage, tmp_path, first, measured, coverage=epoch1)
    assert first.site_id is not None
    second = _capture(db, "second", metadata_={"siteId": str(first.site_id)})
    joined = _echoed(first.site_id)
    joined["quality"] = {"mode": "preview", "bar": "strict", "keepVerifiedPct": 40.0}

    run(db, storage, tmp_path, second, joined, coverage=epoch2)

    db.expire_all()
    site = db.get(Site, first.site_id)
    assert site is not None
    assert site.metadata_["coverageUrl"] == epoch1
    assert site.metadata_["keepVerifiedPct"] == 81.5
    assert site.metadata_["jobId"] == str(first_job)
    later = _epochs(db, first.site_id)[1]
    assert later.quality is not None and later.quality["coverageUrl"] == epoch2

    # A re-run of epoch 1 still refreshes it.
    again = run(db, storage, tmp_path, first, coverage=None)
    db.expire_all()
    site = db.get(Site, first.site_id)
    assert site is not None
    assert "coverageUrl" not in site.metadata_ and site.metadata_["jobId"] == str(again)


def test_a_legacy_capture_that_made_its_site_joins_it_as_new_site(
    db: Session, storage: S3Storage, tmp_path: Path, client: Any
) -> None:
    """A capture registered before 0032 into a site the backfill could not read adopts the
    site's legacy splat on its next run. It made that site, so its epoch is `new-site`, as
    the backfill labels the sites it could read; `explicit` would say it named the site."""
    from tests.conftest import site_payload

    capture = _capture(db)
    created = client.post(
        "/api/v1/sites",
        json={**site_payload(slug="legacy"), "metadata": {"captureId": str(capture.id)}},
    )
    assert created.status_code == 201, created.text
    site_id = uuid.UUID(created.json()["id"])
    capture.site_id = site_id
    db.commit()
    [legacy] = _assets(db, site_id)

    run(db, storage, tmp_path, capture)

    [epoch] = _epochs(db, site_id)
    assert (epoch.ordinal, epoch.join_method) == (1, "new-site")
    assert {a.id: a.epoch_id for a in _assets(db, site_id)}[legacy.id] == epoch.id
