"""`services/epochs.py`: which epochs came before, and which frame a run is placed in."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, box
from sqlalchemy.orm import Session

from app.models import Asset, Capture, Epoch, Site, SiteFrame
from app.models.enums import AssetProvider, CaptureKind, Representation
from app.services import epochs

ORIGIN = (47.6445, -122.138, 30.0)


def square(lon: float, lat: float, half_deg: float = 0.0005) -> WKBElement:
    """A small square footprint about (lon, lat), stored the way the columns are."""
    return from_shape(
        MultiPolygon([box(lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)]),
        srid=4326,
    )


def make_site(
    db: Session,
    slug: str = "field",
    *,
    at: tuple[float, float] = (ORIGIN[1], ORIGIN[0]),
    framed: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Site:
    lon, lat = at
    site = Site(
        slug=slug,
        name=slug,
        boundary=square(lon, lat),
        centroid=from_shape(box(lon, lat, lon, lat).centroid, srid=4326),
        metadata_=metadata or {},
        attribution=[],
    )
    db.add(site)
    db.flush()
    if framed:
        db.add(
            SiteFrame(
                site_id=site.id,
                lat=lat,
                lon=lon,
                height=ORIGIN[2],
                height_datum="msl-unverified",
                epsg=epochs.utm_epsg(lat, lon),
                source="registration",
            )
        )
        db.flush()
    return site


def make_epoch(
    db: Session,
    site: Site,
    ordinal: int,
    *,
    observed: datetime,
    footprint: WKBElement | None = None,
    status: str = "published",
    quality: dict[str, Any] | None = None,
) -> Epoch:
    epoch = Epoch(
        site_id=site.id,
        ordinal=ordinal,
        observed_at=observed,
        observed_at_source="captured-at",
        footprint=footprint if footprint is not None else site.boundary,
        platform="unknown",
        status=status,
        join_method="new-site" if ordinal == 1 else "explicit",
        quality=quality,
    )
    db.add(epoch)
    db.flush()
    return epoch


def make_asset(
    db: Session,
    site: Site,
    epoch: Epoch | None,
    url: str,
    *,
    created: datetime,
    valid_to: datetime | None = None,
) -> Asset:
    asset = Asset(
        site_id=site.id,
        name="splat",
        representation=Representation.GAUSSIAN_SPLAT,
        provider=AssetProvider.TILES_3D_URL,
        source={"type": "3d-tiles-url", "url": url},
        attribution=[],
        render_config={},
        default_visible=True,
        epoch_id=epoch.id if epoch is not None else None,
        observed_at=epoch.observed_at if epoch is not None else None,
        valid_to=valid_to,
        created_at=created,
        updated_at=created,
    )
    db.add(asset)
    db.flush()
    return asset


T0 = datetime(2026, 9, 12, tzinfo=UTC)


def test_prior_epochs_are_the_published_ones_whose_footprint_intersects(db: Session) -> None:
    site = make_site(db)
    lon, lat = ORIGIN[1], ORIGIN[0]
    first = make_epoch(db, site, 1, observed=T0)
    second = make_epoch(db, site, 2, observed=T0 + timedelta(days=30))
    make_epoch(db, site, 3, observed=T0 + timedelta(days=40), footprint=square(lon + 0.01, lat))
    make_epoch(db, site, 4, observed=T0 + timedelta(days=50), status="rejected")
    other = make_site(db, "other")
    make_epoch(db, other, 1, observed=T0)
    db.commit()

    found = epochs.prior_epochs(db, site_id=site.id, footprint=square(lon, lat, 0.0001))

    assert [epoch.id for epoch in found] == [second.id, first.id]


def test_prior_epochs_can_be_limited_to_before_a_date_and_exclude_one(db: Session) -> None:
    site = make_site(db)
    first = make_epoch(db, site, 1, observed=T0)
    second = make_epoch(db, site, 2, observed=T0 + timedelta(days=30))
    third = make_epoch(db, site, 3, observed=T0 + timedelta(days=60))
    db.commit()

    before = epochs.prior_epochs(
        db, site_id=site.id, footprint=site.boundary, before=third.observed_at
    )
    excluded = epochs.prior_epochs(db, site_id=site.id, footprint=site.boundary, exclude=second.id)

    assert [epoch.id for epoch in before] == [second.id, first.id]
    assert [epoch.id for epoch in excluded] == [third.id, first.id]


def test_utm_epsg_north_south_and_the_antimeridian() -> None:
    assert epochs.utm_epsg(47.6, -122.3) == 32610
    assert epochs.utm_epsg(-33.9, 151.2) == 32756
    assert epochs.utm_epsg(0.0, -180.0) == 32601
    assert epochs.utm_epsg(10.0, 180.0) == 32660
    assert epochs.utm_epsg(-10.0, 179.999) == 32760


def test_the_current_asset_is_the_newest_observed_one_with_no_valid_to(db: Session) -> None:
    site = make_site(db)
    epoch = make_epoch(db, site, 1, observed=T0)
    old = make_asset(
        db,
        site,
        epoch,
        "https://cdn.example.com/a/tileset.json",
        created=T0,
        valid_to=T0 + timedelta(days=1),
    )
    new = make_asset(
        db, site, epoch, "https://cdn.example.com/b/tileset.json", created=T0 + timedelta(days=1)
    )
    legacy = make_asset(
        db, site, None, "https://cdn.example.com/c/tileset.json", created=T0, valid_to=T0
    )
    db.commit()

    assert epochs.current_asset(db, epoch.id) == new
    assert epochs.current_asset_of(epoch.id, [old, new, legacy]) == new
    assert epochs.is_superseded(old) and not epochs.is_superseded(new)
    # A legacy asset's valid_to is a validity window, not a supersession.
    assert not epochs.is_superseded(legacy)
    assert epochs.current_asset(db, uuid.uuid4()) is None


# ── Which frame a run is placed in ─────────────────────────────────────────────────────


def _capture(db: Session, slug: str, metadata: dict[str, Any] | None = None) -> Capture:
    capture = Capture(
        slug=slug, name=slug, kind=CaptureKind.GAUSSIAN_SPLAT, metadata_=metadata or {}
    )
    db.add(capture)
    db.flush()
    return capture


def _north_of(metres: float, lat: float = ORIGIN[0]) -> float:
    return lat + metres / 111_320.0


def test_a_re_run_gets_its_epochs_frame(db: Session) -> None:
    site = make_site(db)
    capture = _capture(db, "c")
    epoch = make_epoch(db, site, 1, observed=T0)
    epoch.capture_id = capture.id
    # Even with metadata that would point elsewhere, a re-run stays in its epoch's frame.
    other = make_site(db, "other", at=(10.0, 10.0))
    capture.metadata_ = {"siteId": str(other.id), "lat": 10.0, "lon": 10.0}
    db.commit()

    frame = epochs.frame_for_run(db, capture)

    assert frame is not None and frame.site_id == site.id
    assert (frame.lat, frame.lon, frame.height) == (ORIGIN[0], ORIGIN[1], ORIGIN[2])
    assert frame.param() == {
        "siteId": str(site.id),
        "lat": ORIGIN[0],
        "lon": ORIGIN[1],
        "height": ORIGIN[2],
        "heightDatum": "msl-unverified",
        "epsg": 32610,
    }


def test_an_explicit_site_wins_over_a_spatial_match(db: Session) -> None:
    near = make_site(db, "near")
    named = make_site(db, "named", at=(ORIGIN[1] + 0.05, ORIGIN[0]))
    capture = _capture(db, "c", {"siteId": str(named.id), "lat": ORIGIN[0], "lon": ORIGIN[1]})
    unframed = make_site(db, "unframed", at=(ORIGIN[1] - 0.05, ORIGIN[0]), framed=False)
    first = _capture(db, "first", {"siteId": str(unframed.id), "lat": ORIGIN[0], "lon": ORIGIN[1]})
    db.commit()

    frame = epochs.frame_for_run(db, capture)

    assert frame is not None and frame.site_id == named.id and frame.site_id != near.id
    # A named site with no frame yet: this run is its first epoch and defines it.
    assert epochs.frame_for_run(db, first) is None


def test_a_spatial_match_needs_a_frame_and_25_m(db: Session) -> None:
    site = make_site(db)  # a 0.001 deg square: its north edge is ORIGIN + 0.0005 deg
    edge = ORIGIN[0] + 0.0005
    inside = _capture(db, "inside", {"lat": _north_of(20, edge), "lon": ORIGIN[1]})
    outside = _capture(db, "outside", {"lat": _north_of(30, edge), "lon": ORIGIN[1]})
    frameless = make_site(db, "frameless", at=(ORIGIN[1] + 1.0, ORIGIN[0]), framed=False)
    on_frameless = _capture(db, "on", {"lat": ORIGIN[0], "lon": ORIGIN[1] + 1.0})
    db.commit()

    found = epochs.frame_for_run(db, inside)
    assert found is not None and found.site_id == site.id
    assert epochs.frame_for_run(db, outside) is None
    assert epochs.frame_for_run(db, on_frameless) is None
    assert frameless.id != site.id


def test_a_spatial_match_uses_the_runs_located_point(db: Session) -> None:
    """A typed place in the job's params wins over the capture's own fix for the join, as
    it does for the placement (`worker/params.located_point`)."""
    site = make_site(db)
    edge = ORIGIN[0] + 0.0005
    far = _capture(db, "far", {"lat": _north_of(500, edge), "lon": ORIGIN[1]})
    db.commit()

    assert epochs.frame_for_run(db, far) is None
    found = epochs.frame_for_run(db, far, (_north_of(20, edge), ORIGIN[1]))
    assert found is not None and found.site_id == site.id


def test_a_phone_capture_joins_by_location_only_a_site_a_phone_made(db: Session) -> None:
    """The phone key reaches only what it created (`phone._check_phone_site`): a phone
    capture that names a desktop-made site is refused, so one standing in that site must
    not join it by location either. Beside a site a phone capture made it joins as before;
    a desktop capture keeps joining any framed site."""
    from app.api.v1.phone import ORIGIN as PHONE

    edge = ORIGIN[0] + 0.0005
    by_desk = _capture(db, "desk-maker")
    desk_site = make_site(db, "desk", metadata={"captureId": str(by_desk.id)})
    by_phone = _capture(db, "phone-maker", {"origin": PHONE})
    phone_site = make_site(
        db, "phone", at=(ORIGIN[1] + 0.05, ORIGIN[0]), metadata={"captureId": str(by_phone.id)}
    )
    seeded = make_site(db, "seeded", at=(ORIGIN[1] - 0.05, ORIGIN[0]))
    near_desk = {"lat": _north_of(20, edge), "lon": ORIGIN[1]}
    near_phone = {"lat": _north_of(20, edge), "lon": ORIGIN[1] + 0.05}
    near_seeded = {"lat": _north_of(20, edge), "lon": ORIGIN[1] - 0.05}
    phone_at_desk = _capture(db, "p1", {"origin": PHONE, **near_desk})
    phone_at_phone = _capture(db, "p2", {"origin": PHONE, **near_phone})
    phone_at_seeded = _capture(db, "p3", {"origin": PHONE, **near_seeded})
    desk_at_desk = _capture(db, "d1", near_desk)
    desk_at_phone = _capture(db, "d2", near_phone)
    desk_at_seeded = _capture(db, "d3", near_seeded)
    db.commit()

    assert epochs.frame_for_run(db, phone_at_desk) is None
    assert epochs.frame_for_run(db, phone_at_seeded) is None
    found = epochs.frame_for_run(db, phone_at_phone)
    assert found is not None and found.site_id == phone_site.id
    # A typed place given to the phone's run is held to the same rule.
    assert epochs.frame_for_run(db, phone_at_phone, (near_desk["lat"], near_desk["lon"])) is None
    for capture, site in (
        (desk_at_desk, desk_site),
        (desk_at_phone, phone_site),
        (desk_at_seeded, seeded),
    ):
        found = epochs.frame_for_run(db, capture)
        assert found is not None and found.site_id == site.id


def _video(lat: float, lon: float, accuracy: float | None = None) -> dict[str, Any]:
    """`metadata.videoLocation` as the worker records it (`worker/video_gps.py`)."""
    return {"videoLocation": {"lat": lat, "lon": lon, "accuracyM": accuracy, "key": "iso6709"}}


def test_a_video_with_gps_uploaded_from_10_km_away_joins_the_site_it_shows(db: Session) -> None:
    """Recorded on the ranch, uploaded from home: the browser's fix is 10 km off, and the
    video's own location is where it was made (ADR 0013, "Joining a site by the video's
    GPS"). The join takes the video's, and says so."""
    site = make_site(db)
    home = {"lat": _north_of(10_000), "lon": ORIGIN[1]}
    capture = _capture(db, "c", {**home, **_video(ORIGIN[0], ORIGIN[1], 5.0)})
    uploaded_only = _capture(db, "u", home)
    db.commit()

    found = epochs.frame_for_run(db, capture)

    assert found is not None and found.site_id == site.id
    assert epochs.frame_for_run(db, uploaded_only) is None
    point = epochs.join_point(capture, None)
    assert point is not None and (point.source, point.accuracy_m) == ("video", 5.0)


def test_a_video_without_gps_falls_back_to_where_it_was_uploaded_from(db: Session) -> None:
    """iOS can strip the location (a video picked through Safari or a Photos share with
    location off): the worker records that it looked and found none (`videoLocation:
    null`), and the join is today's, by the upload-time fix."""
    site = make_site(db)
    edge = ORIGIN[0] + 0.0005
    near = {"lat": _north_of(20, edge), "lon": ORIGIN[1]}
    stripped = _capture(db, "s", {**near, "videoLocation": None})
    garbled = _capture(db, "g", {**near, "videoLocation": {"lat": "north", "lon": 1}})
    db.commit()

    for capture in (stripped, garbled):
        found = epochs.frame_for_run(db, capture)
        assert found is not None and found.site_id == site.id
        point = epochs.join_point(capture, None)
        assert point is not None and (point.source, point.accuracy_m) == ("upload", None)
        assert point.buffer_m == epochs.SPATIAL_JOIN_BUFFER_M


def test_a_video_near_two_sites_joins_the_nearer(db: Session) -> None:
    """A wide fix can reach two sites; the nearer boundary wins, whichever is older."""
    edge = ORIGIN[0] + 0.0005
    far = make_site(db, "far")  # its north edge 35 m south of the point; made first
    # Its south edge 50 m north of `far`'s north edge (a square is 0.001 deg across).
    near = make_site(db, "near", at=(ORIGIN[1], _north_of(50, edge) + 0.0005))
    point = _north_of(35, edge)  # 35 m north of `far`'s edge, 15 m south of `near`'s
    capture = _capture(db, "c", _video(point, ORIGIN[1], 20.0))
    db.commit()

    found = epochs.frame_for_run(db, capture)

    assert found is not None and found.site_id == near.id and far.id != near.id


@pytest.mark.parametrize(
    ("accuracy", "metres", "joins"),
    [
        (None, 20, True),  # no accuracy written: the D3 default, 25 m
        (None, 30, False),
        (20.0, 35, True),  # widened: 2 x 20 = 40 m
        (20.0, 45, False),
        (3.0, 8, True),  # narrowed: 2 x 3 = 6 m, held at the 10 m floor
        (3.0, 20, False),  # inside the default 25 m, outside a good fix's 10 m
        (1000.0, 45, True),  # a fix no better than a kilometre: capped at 50 m
        (1000.0, 60, False),
    ],
)
def test_the_buffer_follows_the_videos_accuracy(
    db: Session, accuracy: float | None, metres: float, joins: bool
) -> None:
    site = make_site(db)
    edge = ORIGIN[0] + 0.0005
    capture = _capture(db, "c", _video(_north_of(metres, edge), ORIGIN[1], accuracy))
    db.commit()

    found = epochs.frame_for_run(db, capture)

    assert (found is not None and found.site_id == site.id) is joins


def test_the_video_buffer_rule() -> None:
    assert epochs.video_join_buffer_m(None) == epochs.SPATIAL_JOIN_BUFFER_M
    assert epochs.video_join_buffer_m(5.0) == 10.0
    assert epochs.video_join_buffer_m(12.5) == 25.0
    assert epochs.video_join_buffer_m(18.0) == 36.0
    assert epochs.video_join_buffer_m(65.0) == epochs.VIDEO_JOIN_MAX_M == 50.0
    assert epochs.video_join_buffer_m(0.5) == epochs.VIDEO_JOIN_MIN_M == 10.0
    assert epochs.VIDEO_JOIN_MAX_M < epochs.EXPLICIT_JOIN_MAX_M


def test_a_phone_capture_joins_by_its_videos_gps_only_a_site_a_phone_made(db: Session) -> None:
    """The video's location is as forgeable as the browser's -- the phone writes both --
    so it is held to the same rule (`phone._check_phone_site`): a phone capture whose
    video says it stands in a desktop-made or seeded site does not join it, even with a
    wide accuracy; one whose video says it stands in a phone-made site does."""
    from app.api.v1.phone import ORIGIN as PHONE

    by_desk = _capture(db, "desk-maker")
    desk_site = make_site(db, "desk", metadata={"captureId": str(by_desk.id)})
    by_phone = _capture(db, "phone-maker", {"origin": PHONE})
    phone_site = make_site(
        db, "phone", at=(ORIGIN[1] + 0.05, ORIGIN[0]), metadata={"captureId": str(by_phone.id)}
    )
    seeded = make_site(db, "seeded", at=(ORIGIN[1] - 0.05, ORIGIN[0]))
    beside_phone = {"lat": ORIGIN[0], "lon": ORIGIN[1] + 0.05}
    at_desk = _capture(
        db, "p1", {"origin": PHONE, **beside_phone, **_video(ORIGIN[0], ORIGIN[1], 1e6)}
    )
    at_seeded = _capture(
        db, "p2", {"origin": PHONE, **beside_phone, **_video(ORIGIN[0], ORIGIN[1] - 0.05)}
    )
    at_phone = _capture(db, "p3", {"origin": PHONE, **_video(ORIGIN[0], ORIGIN[1] + 0.05, 4.0)})
    db.commit()

    assert epochs.frame_for_run(db, at_desk) is None
    assert epochs.frame_for_run(db, at_seeded) is None
    found = epochs.frame_for_run(db, at_phone)
    assert found is not None and found.site_id == phone_site.id
    assert desk_site.id != seeded.id


def test_a_rejected_capture_is_kept_where_its_video_says(db: Session) -> None:
    """`record_rejected` takes the same answer: the video's site, and a footprint about
    the video's point, not the upload's."""
    site = make_site(db)
    home = {"lat": _north_of(10_000), "lon": ORIGIN[1]}
    capture = _capture(db, "c", {**home, **_video(ORIGIN[0], ORIGIN[1], 5.0)})
    db.commit()

    epoch = epochs.record_rejected(
        db, uuid.uuid4(), capture, epochs.Rejection(stage_id="quality", gate={})
    )

    assert epoch is not None and epoch.site_id == site.id and epoch.join_method == "spatial"
    assert (epoch.join_point_source, epoch.join_point_accuracy_m) == ("video", 5.0)
    assert epochs.distance_to_site_m(db, site.id, ORIGIN[0], ORIGIN[1]) == 0.0
    from geoalchemy2.shape import to_shape

    centre = to_shape(epoch.footprint).centroid
    assert abs(centre.y - ORIGIN[0]) < 1e-5 and abs(centre.x - ORIGIN[1]) < 1e-5


def test_the_spatial_match_is_answered_from_an_index_on_the_boundary(db: Session) -> None:
    """The match is a distance on the ellipsoid (`geography(boundary)`), which the plain
    gist index on the geometry column cannot answer: every site was read. Migration 0032
    adds a gist index on that very expression, so the planner can use it (shown with
    sequential scans priced out, since a test database is too small to need it)."""
    from sqlalchemy import text

    make_site(db)
    db.commit()
    statement = epochs.nearest_framed_site_statement(ORIGIN[0], ORIGIN[1])
    sql = statement.compile(dialect=db.get_bind().dialect, compile_kwargs={"literal_binds": True})

    db.execute(text("SET LOCAL enable_seqscan = off"))
    plan = "\n".join(db.scalars(text(f"EXPLAIN {sql}")))
    db.rollback()

    assert "ix_sites_boundary_geography" in plan, plan


def test_no_coordinate_and_no_site_means_no_frame(db: Session) -> None:
    make_site(db)
    capture = _capture(db, "c", {"note": "nowhere"})
    dangling = _capture(db, "d", {"siteId": str(uuid.uuid4())})
    malformed = _capture(db, "m", {"siteId": "not-a-site"})
    db.commit()

    assert epochs.frame_for_run(db, capture) is None
    assert epochs.frame_for_run(db, dangling) is None
    assert epochs.frame_for_run(db, malformed) is None


# ── An epoch's footprint in its site's frame ─────────────────────────────────────────────


def test_a_footprint_in_a_frame_is_placed_by_the_exact_enu_inverse() -> None:
    """A joined epoch's box is metres about its site's frame origin, up to 2 km away. The
    flat 111,320 m a degree is ~0.4% long north-south at 28 deg N -- about 9 m at 2 km --
    so in a frame the corners are placed exactly. Checked against pyproj (geocentric to
    geodetic), an implementation that shares nothing with `placement.enu_to_lonlat`."""
    import math

    import numpy as np
    from pyproj import Transformer

    from app.services import placement

    lat, lon, height = 28.0389, -82.6966, 22.5
    to_geodetic = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = to_ecef.transform(lon, lat, height)
    phi, lam = math.radians(lat), math.radians(lon)
    # East/north/up unit vectors in ECEF.
    axes = np.array(
        [
            [-math.sin(lam), math.cos(lam), 0.0],
            [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)],
        ]
    )

    def truth(east: float, north: float) -> tuple[float, float]:
        x, y, z = np.array([x0, y0, z0]) + east * axes[0] + north * axes[1]
        found_lon, found_lat, _ = to_geodetic.transform(x, y, z)
        return found_lon, found_lat

    box_m = ([1950.0, 1980.0, -2.0], [2010.0, 2040.0, 8.0])
    exact = placement.boundary(lat, lon, box_m, frame_height=height)
    ring = exact.coordinates[0]
    corners = [(1950.0, 1980.0), (2010.0, 1980.0), (2010.0, 2040.0), (1950.0, 2040.0)]
    for (found_lon, found_lat), (east, north) in zip(ring[:4], corners, strict=True):
        want_lon, want_lat = truth(east, north)
        # 1e-8 deg is about a millimetre.
        assert abs(found_lon - want_lon) < 1e-8
        assert abs(found_lat - want_lat) < 1e-8
    assert ring[4] == ring[0]

    # The flat rule it replaces is metres out there; outside a frame it is unchanged.
    flat = placement.boundary(lat, lon, box_m)
    assert abs(flat.coordinates[0][2][1] - ring[2][1]) * placement.METRES_PER_DEGREE > 5.0
    assert flat == placement.rectangle(lat, lon, (1950.0, 2010.0), (1980.0, 2040.0))
