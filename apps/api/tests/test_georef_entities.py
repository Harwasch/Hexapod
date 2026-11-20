import math
import random
from typing import Any

import pytest

from app.services.georef_entities import (
    E2,
    A,
    GeoPoint,
    Similarity,
    TrackPair,
    arkit_to_enu,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_to_ecef,
    fit_heading,
    geodetic_to_ecef,
    pair_bundle_tracks,
    resolve_alignment,
)


def test_known_ecef_axes() -> None:
    assert geodetic_to_ecef(GeoPoint(0, 0, 0)) == (A, 0, 0)
    assert geodetic_to_ecef(GeoPoint(0, 90, 0)) == pytest.approx((0, A, 0), abs=1e-8)
    assert geodetic_to_ecef(GeoPoint(90, 0, 0)) == pytest.approx(
        (0, 0, A * math.sqrt(1 - E2)), abs=1e-8
    )
    assert enu_to_ecef((1, 2, 3), GeoPoint(0, 0, 0)) == (A + 3, 1, 2)


@pytest.mark.parametrize(
    "lat,lon,alt",
    [
        (0, 0, 0),
        (47.6, -122.1, 123),
        (-80, 179.999, -20),
        (90, 0, 5),
        (-90, 90, 0),
        (0, -180, 500000),
    ],
)
def test_ecef_and_enu_roundtrips(lat: float, lon: float, alt: float) -> None:
    origin = GeoPoint(lat, lon, alt)
    restored = ecef_to_geodetic(geodetic_to_ecef(origin))
    # Analytic WGS84 double-precision math: sub-micrometre, not sensor accuracy.
    assert restored.lat == pytest.approx(lat, abs=1e-10)
    assert restored.lon == pytest.approx(lon, abs=1e-10)
    assert restored.alt == pytest.approx(alt, abs=1e-7)
    offset = (1250.5, -2100.25, 31.2)
    assert ecef_to_enu(enu_to_ecef(offset, origin), origin) == pytest.approx(offset, abs=1e-8)


def test_arkit_heading_and_similarity_inverse() -> None:
    assert arkit_to_enu(0).apply((1, 2, -3)) == (1, 3, 2)
    assert arkit_to_enu(math.pi / 2).apply((0, 0, -3)) == pytest.approx((3, 0, 0), abs=1e-12)
    transform = Similarity(arkit_to_enu(0.7).rotation, (3, -4, 8), 2.5)
    assert transform.inverse(transform.apply((2, 8, -4))) == pytest.approx((2, 8, -4))
    with pytest.raises(ValueError, match="handedness"):
        Similarity(((1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))
    with pytest.raises(ValueError, match="orthonormal"):
        Similarity(((2, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))


def test_heading_fit_noisy_walk_and_translation() -> None:
    rng = random.Random(423)
    expected = arkit_to_enu(1.1, (50, -20, 3))
    pairs = []
    for i in range(40):
        world = (float(i), 2.0, -i * 0.5)
        e, n, u = expected.apply(world)
        pairs.append(
            TrackPair(world, (e + rng.uniform(-0.3, 0.3), n + rng.uniform(-0.3, 0.3), u), 0.5)
        )
    fit = fit_heading(pairs)
    assert fit is not None
    # Synthetic noise bounded at .3m across a 40m walk, specified before fitting.
    assert abs(fit.heading_rad - 1.1) < 0.02
    assert fit.transform.apply((20, 2, -10)) == pytest.approx(expected.apply((20, 2, -10)), abs=0.3)
    assert fit.uncertainty_m >= 0.5
    assert fit_heading(pairs[:3]) is None
    assert fit_heading([TrackPair((0, 0, 0), (i, 0, 0), 1) for i in range(10)]) is None
    assert fit_heading([TrackPair((i, 0, 0), (i, 0, 0), 20) for i in range(10)]) is None
    assert fit_heading([TrackPair((i * 3, 0, 0), (i * 6, 0, 0), 1) for i in range(10)]) is None


def test_noisy_inconsistent_walk_rejected() -> None:
    pairs = [TrackPair((i * 10, 0, 0), (i * 10, (-1) ** i * 10, 0), 0.5) for i in range(10)]
    assert fit_heading(pairs) is None


def test_pipeline_priority_preserves_full_transform_and_provenance() -> None:
    georef = {
        "lat": 0,
        "lon": 0,
        "height": 10,
        "frame": {"headingSource": "capture"},
        "uncertaintyM": 3,
    }
    placement = {
        "rotation": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        "translation": (10, 20, 30),
        "scale": 2,
    }
    alignment = resolve_alignment(
        georef=georef,
        placement=placement,
        arkit_training_frame_verified=True,
        height_datum="ellipsoid-wgs84",
        site_frame={"lat": 30, "lon": 40, "height": 99},
    )
    assert alignment is not None and alignment.transform is not None
    assert alignment.source == "pipeline" and alignment.origin.lat == 0
    assert alignment.transform.apply((1, 2, -3)) == (12, 26, 34)
    assert alignment.locate((1, 2, -3)) == ecef_to_geodetic(
        enu_to_ecef((12, 26, 34), alignment.origin)
    )
    assert alignment.uncertainty_at((1, 2, -3)) == 3
    # Neither a render transform with arbitrary heading nor unverified training axes is evidence.
    georef["frame"] = {"headingSource": "none"}
    unknown = resolve_alignment(
        georef=georef, placement=placement, arkit_training_frame_verified=True
    )
    assert unknown is not None and unknown.locate((1, 0, 0)) is None
    georef["frame"] = {"headingSource": "capture"}
    unknown = resolve_alignment(georef=georef, placement=placement)
    assert unknown is not None and unknown.transform is None


def test_site_and_gps_do_not_invent_session_translation_heading_or_altitude() -> None:
    gps = {"lat": 5, "lon": 6, "altitudeM": 20, "horizontalAccuracyM": 4}
    frame = {"lat": 1, "lon": 2, "height": 3, "heightDatum": "ellipsoid-wgs84"}
    alignment = resolve_alignment(site_frame=frame, gps=gps)
    assert alignment is not None and alignment.source == "site"
    assert alignment.locate((0, 0, 0)) is None
    alignment = resolve_alignment(gps=gps)
    assert alignment is not None and alignment.source == "gps"
    assert alignment.height_datum == "msl-unverified" and alignment.locate((0, 0, 0)) is None
    gps["horizontalAccuracyM"] = -1
    assert resolve_alignment(gps=gps) is None
    assert resolve_alignment() is None


def test_pair_clock_interpolation_invalid_fix_and_tracking_gap() -> None:
    def frame(time: float, x: float, state: str = "normal") -> dict[str, Any]:
        return {
            "timestamp": time,
            "trackingState": state,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 2, 3, 1],
        }

    frames = [frame(0, 0), frame(1, 2), frame(2, 4, "limited")]
    fix = {"timestamp": 100.5, "lat": 0, "lon": 0, "horizontalAccuracyM": 2}
    pairs = pair_bundle_tracks(
        frames,
        [
            fix,
            {**fix, "horizontalAccuracyM": -1},
            {**fix, "timestamp": 101.5},
            {**fix, "timestamp": 99},
        ],
        frame_zero_unix=100,
        origin=GeoPoint(0, 0, 0),
    )
    assert len(pairs) == 1 and pairs[0].world == (1, 2, 3)
    assert pairs[0].enu == (0, 0, 0)
    with pytest.raises(ValueError, match="increase"):
        pair_bundle_tracks(frames[::-1], [], frame_zero_unix=100, origin=GeoPoint(0, 0, 0))


@pytest.mark.parametrize("point", [(91, 0, 0), (0, 181, 0), (0, 0, float("nan"))])
def test_invalid_coordinates(point: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        GeoPoint(*point)


def test_bundle_default_and_entity_metadata_preserve_unknowns() -> None:
    from app.services.georef_entities import align_bundle, georeference_entity

    fixes = [
        {"timestamp": 100, "lat": 0, "lon": 0, "horizontalAccuracyM": -1},
        {
            "timestamp": 101,
            "lat": 1,
            "lon": 2,
            "horizontalAccuracyM": 3,
            "altitudeM": 20,
            "verticalAccuracyM": 10,
        },
    ]
    alignment = align_bundle(frames=[], fixes=fixes)
    assert alignment is not None and alignment.origin == GeoPoint(1, 2, 20)
    entity: dict[str, Any] = {
        "anchor": {"world": {"x": 1, "y": 2, "z": 3}, "geo": {"lat": 9, "lon": 9, "alt": 9}},
        "attrs": {"health": "good"},
    }
    result = georeference_entity(entity, alignment)
    assert "geo" not in result["anchor"] and "geo" in entity["anchor"]
    assert result["attrs"]["health"] == "good"
    assert result["attrs"]["georeference"]["headingKnown"] is False
    assert result["attrs"]["georeference"]["originUncertaintyM"] == 3
    assert result["attrs"]["georeference"]["uncertaintyM"] is None


def test_horizontal_fit_never_claims_an_unmeasured_vertical() -> None:
    from app.services.georef_entities import resolve_alignment

    fit = fit_heading(
        [TrackPair((i * 10, 0, 0), (i * 10, 0, 0), 1, vertical_valid=False) for i in range(5)]
    )
    assert fit is not None and fit.vertical_known is False
    alignment = resolve_alignment(
        site_frame={"lat": 0, "lon": 0, "height": 0, "heightDatum": "ellipsoid-wgs84"},
        track_fit=fit,
    )
    assert alignment is not None and alignment.transform is not None
    assert alignment.locate((10, 0, 0)) is None
    near, far = alignment.uncertainty_at((0, 0, 0)), alignment.uncertainty_at((100, 0, 0))
    assert near is not None and far is not None and far > near


def test_georef_without_recentring_can_supply_complete_transform() -> None:
    alignment = resolve_alignment(
        georef={
            "lat": 0,
            "lon": 0,
            "height": 0,
            "frame": {
                "headingSource": "exif-gps",
                "recentre": False,
                "scale": 1,
                "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                "translationM": [1, 2, 3],
            },
        },
        arkit_training_frame_verified=True,
        height_datum="ellipsoid-wgs84",
    )
    assert alignment is not None and alignment.locate((0, 0, 0)) is not None
