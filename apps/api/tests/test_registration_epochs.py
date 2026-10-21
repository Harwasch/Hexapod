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
    content_addressed: bool = True,
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
            content_addressed=content_addressed,
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


def test_a_re_run_keeps_what_the_epoch_said_before_in_its_history(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A re-run replaces the epoch's run, footprint and quality, which describe its current
    revision. What they said before is not lost: each re-run appends the values it
    replaces to the epoch's `history`, oldest first, and never rewrites an entry."""
    from geoalchemy2.shape import to_shape
    from shapely.geometry import shape

    capture = _capture(db)
    first = _document()
    first["quality"] = {"mode": "preview", "bar": "strict", "keepVerifiedPct": 81.5}
    first_job = run(db, storage, tmp_path, capture, first)
    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    first_footprint, first_quality = to_shape(epoch.footprint), epoch.quality
    assert epoch.history == []
    wider = _document()
    wider["bboxLocalM"] = {"min": [-8.0, -4.0, -1.0], "max": [12.0, 16.0, 3.0]}
    wider["quality"] = {"mode": "full", "bar": "strict", "keepVerifiedPct": 90.0}

    second_job = run(db, storage, tmp_path, capture, wider)
    third_job = run(db, storage, tmp_path, capture)

    [epoch] = _epochs(db, capture.site_id)
    assert epoch.job_id == third_job
    assert [entry["jobId"] for entry in epoch.history] == [str(first_job), str(second_job)]
    assert [entry["replacedByJobId"] for entry in epoch.history] == [
        str(second_job),
        str(third_job),
    ]
    oldest, newer = epoch.history
    assert oldest["quality"] == first_quality and oldest["quality"]["keepVerifiedPct"] == 81.5
    assert shape(oldest["footprint"]).equals(first_footprint)
    assert newer["quality"]["keepVerifiedPct"] == 90.0
    assert shape(newer["footprint"]).area > shape(oldest["footprint"]).area
    assert all(isinstance(entry["replacedAt"], str) for entry in epoch.history)


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


def test_with_one_bucket_the_same_url_is_still_a_new_revision(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """With one bucket nothing is copied into a content-hashed generation: a site points
    at the run's own keys, and a Refine re-runs the same job and rewrites them. The URL
    then says nothing about the bytes, so a republish onto it is a new revision -- with
    the new run's ground samples and provenance -- not "the very same tiles"."""
    capture = _capture(db)
    same = "https://cdn.example.com/twin-worker-test/runs/x/package/splat/tileset.json"
    run(db, storage, tmp_path, capture, url=same, content_addressed=False)
    assert capture.site_id is not None

    run(db, storage, tmp_path, capture, url=same, content_addressed=False)

    old, new = _assets(db, capture.site_id)
    assert old.valid_to is not None and new.valid_to is None
    assert old.source["url"] == new.source["url"] == same
    [epoch] = _epochs(db, capture.site_id)
    assert new.epoch_id == epoch.id and len(epoch.history) == 1


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
        # The files' own clock, with a zone, is read before the capture row's date and
        # keeps the `manifest` label (M3a does not copy it into `capture.captured_at`).
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

    created = client.post("/api/v1/sites", json=site_payload(slug="ranch", boundary=_around()))
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


def _around(half_deg: float = 0.001) -> dict[str, Any]:
    """A boundary about the runs' georeference (`test_asset_scale`'s LAT/LON)."""
    ring = [
        [LON - half_deg, LAT - half_deg],
        [LON + half_deg, LAT - half_deg],
        [LON + half_deg, LAT + half_deg],
        [LON - half_deg, LAT + half_deg],
        [LON - half_deg, LAT - half_deg],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def test_a_capture_far_from_the_frameless_site_it_names_is_a_site_of_its_own(
    db: Session, storage: S3Storage, tmp_path: Path, client: Any
) -> None:
    """Naming a site with no frame fixes its frame where the capture is, for good. A
    capture past the range a frame can hold (2047 m) from that site -- here the test site
    in Seattle and a run placed in Montana -- would fix it a thousand kilometres from
    what the site shows, so it becomes a site of its own and the named one is untouched."""
    from tests.conftest import site_payload

    created = client.post("/api/v1/sites", json=site_payload(slug="seattle"))
    assert created.status_code == 201, created.text
    named = uuid.UUID(created.json()["id"])
    capture = _capture(db, metadata_={"siteId": str(named)})

    run(db, storage, tmp_path, capture)

    db.expire_all()
    assert capture.site_id is not None and capture.site_id != named
    assert db.get(SiteFrame, named) is None
    assert _epochs(db, named) == []
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.join_method == "new-site"


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


def test_a_spatial_epoch_records_which_point_matched_it(
    db: Session, storage: S3Storage, tmp_path: Path, client: Any
) -> None:
    """A capture matched to its site by location says by which point (ADR 0013, "Joining a
    site by the video's GPS"): its video's own GPS, with the accuracy the phone wrote, or
    the location sent with its upload. One that named its site, or made it, was matched by
    neither, and says nothing."""
    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    video = {"lat": LAT, "lon": LON, "accuracyM": 4.5, "key": "iso6709"}
    by_video = _capture(db, "by-video", metadata_={"lat": 1.0, "lon": 2.0, "videoLocation": video})
    by_upload = _capture(db, "by-upload", metadata_={"lat": LAT, "lon": LON, "videoLocation": None})
    named = _capture(db, "named", metadata_={"siteId": str(first.site_id)})

    for capture in (by_video, by_upload, named):
        run(db, storage, tmp_path, capture, _echoed(first.site_id))

    epochs = _epochs(db, first.site_id)
    assert [(e.join_method, e.join_point_source, e.join_point_accuracy_m) for e in epochs] == [
        ("new-site", None, None),
        ("spatial", "video", 4.5),
        ("spatial", "upload", None),
        ("explicit", None, None),
    ]
    listed = client.get(f"/api/v1/sites/{first.site_id}/epochs").json()
    assert [(e["joinPointSource"], e["joinPointAccuracyM"]) for e in listed] == [
        (None, None),
        ("video", 4.5),
        ("upload", None),
        (None, None),
    ]
    # A re-run keeps how the epoch first found its site.
    by_video.metadata_ = {**by_video.metadata_, "videoLocation": None}
    db.commit()
    run(db, storage, tmp_path, by_video, _echoed(first.site_id))
    db.expire_all()
    assert _epochs(db, first.site_id)[1].join_point_source == "video"


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


def test_each_epochs_asset_carries_that_epochs_own_footprint(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The site's boundary grows to the union of its epochs; each epoch's asset is drawn,
    clipped and counted as knowledge (`land_knowledge`) by its own footprint, else its
    site's boundary. So every revision an epoch's run makes carries that run's footprint --
    the epoch's at the time -- and a re-run's new revision carries the new one while the
    old revision keeps what it had."""
    from geoalchemy2.shape import to_shape

    first = _capture(db, "first")
    run(db, storage, tmp_path, first)
    assert first.site_id is not None
    north = _echoed(first.site_id)
    north["bboxLocalM"] = {"min": [-4.0, 40.0, -1.0], "max": [6.0, 50.0, 3.0]}
    run(db, storage, tmp_path, _capture(db, "second"), north)
    epoch1, epoch2 = _epochs(db, first.site_id)
    original = to_shape(epoch1.footprint)
    wider = _document()
    wider["bboxLocalM"] = {"min": [-8.0, -4.0, -1.0], "max": [12.0, 16.0, 3.0]}

    run(db, storage, tmp_path, first, wider)

    epoch1, epoch2 = _epochs(db, first.site_id)
    site = db.get(Site, first.site_id)
    assert site is not None
    by_epoch: dict[uuid.UUID | None, list[Asset]] = {}
    for asset in _assets(db, first.site_id):
        by_epoch.setdefault(asset.epoch_id, []).append(asset)
    old, new = by_epoch[epoch1.id]
    [joined] = by_epoch[epoch2.id]
    assert old.footprint is not None and to_shape(old.footprint).equals(original)
    assert new.footprint is not None and to_shape(new.footprint).equals(to_shape(epoch1.footprint))
    assert joined.footprint is not None
    assert to_shape(joined.footprint).equals(to_shape(epoch2.footprint))
    assert to_shape(joined.footprint).area < to_shape(site.boundary).area


# --- the detected platform (M3a) -------------------------------------------------------


def _detected(platform: object, evidence: str | None = "metadata") -> dict[str, Any]:
    document = _document()
    document["platform"] = platform
    document["platformEvidence"] = evidence
    return document


@pytest.mark.parametrize(
    ("metadata", "platform", "expected"),
    [
        ({}, "rtk-drone", "rtk-drone"),
        ({}, "phone-photo", "phone-photo"),
        # Nothing detected (Lane 1), or a name outside the vocabulary: the row's own guess.
        ({}, None, "unknown"),
        ({"origin": "phone-key"}, None, "phone-video"),
        ({"origin": "phone-key"}, "satellite", "phone-video"),
        # Detection ran and found nothing: no better than the row's guess, never worse.
        ({"origin": "phone-key"}, "unknown", "phone-video"),
        ({}, "unknown", "unknown"),
    ],
)
def test_a_first_run_writes_the_detected_platform_onto_its_epoch(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    metadata: dict[str, Any],
    platform: object,
    expected: str,
) -> None:
    capture = _capture(db, metadata_=metadata)

    run(db, storage, tmp_path, capture, _detected(platform))

    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.platform == expected


def test_a_re_run_writes_what_it_detected_and_never_erases_a_known_platform(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db)
    run(db, storage, tmp_path, capture, _detected("drone"))
    assert capture.site_id is not None
    site_id = capture.site_id

    run(db, storage, tmp_path, capture, _detected("rtk-drone"))
    assert [e.platform for e in _epochs(db, site_id)] == ["rtk-drone"]
    # A re-run that detected nothing, or only `unknown`, keeps what the epoch says.
    run(db, storage, tmp_path, capture, _detected(None))
    run(db, storage, tmp_path, capture, _detected("unknown"))
    assert [e.platform for e in _epochs(db, site_id)] == ["rtk-drone"]
    # A withheld re-run still records it: the platform is the capture's, not the tiles'.
    run(db, storage, tmp_path, capture, _detected("phone-photo"), withheld=True)
    assert [e.platform for e in _epochs(db, site_id)] == ["phone-photo"]


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ("metadata", "rtk-drone"),
        # A write-token client's `metadata.platform`, or a document from before the
        # field: a claim, which never makes the epoch the survey tier.
        ("claimed", "drone"),
        (None, "drone"),
        ("none", "drone"),
    ],
)
def test_only_an_rtk_drone_the_frames_showed_is_the_survey_tier_on_the_epoch(
    db: Session, storage: S3Storage, tmp_path: Path, evidence: str | None, expected: str
) -> None:
    """`epochs.platform` is the tier D4 reads `survey` off, and it has no column for what
    a platform rests on. So `rtk-drone` is written there only when the run's frames
    showed it (`platformEvidence: metadata`); a claimed one is recorded as the `drone` it
    claims to be, without the survey part. Other claimed platforms never upgrade anything
    and are written as they are."""
    capture = _capture(db)

    run(db, storage, tmp_path, capture, _detected("rtk-drone", evidence))
    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.platform == expected

    run(db, storage, tmp_path, capture, _detected("phone-photo", "claimed"))
    assert [e.platform for e in _epochs(db, capture.site_id)] == ["phone-photo"]
    # A re-run whose frames do not bear the claim out cannot raise it to the survey tier.
    run(db, storage, tmp_path, capture, _detected("rtk-drone", "claimed"))
    assert [e.platform for e in _epochs(db, capture.site_id)] == ["drone"]


def test_the_files_utc_time_dates_the_epoch_and_never_becomes_the_captures_own(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """`capture.captured_at` is what a person or the console said. A time read off the
    files dates the epoch, labelled `manifest`, and leaves the column as it was."""
    capture = _capture(db)
    document = _document()
    document["manifest"] = {"capture": {"capturedAtUtc": "2026-08-30T07:05:00-07:00"}}

    run(db, storage, tmp_path, capture, document)

    db.expire_all()
    assert capture.captured_at is None
    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert (epoch.observed_at, epoch.observed_at_source) == (
        datetime(2026, 8, 30, 14, 5, tzinfo=UTC),
        "manifest",
    )
