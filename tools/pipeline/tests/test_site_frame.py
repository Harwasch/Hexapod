"""A capture placed in its site's frame (decision D3): one origin for every epoch.

The second capture of a field must land in the frame the first one defined, so the two
tilesets share a root transform and differ only where the ground was really captured.
What is checked here is that the frame is honoured and that, without one, nothing moves
by a single byte.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import gps_frames
import numpy as np
import pytest
from test_lane1 import HEIGHT, LAT, LON, PLACEMENT

import gaussians
import site_frame
import splat_stream
import splat_tiles
from conftest import FIXTURE_PLY
from executor import execute
from recipe import load_recipe
from runners import RunnerSet
from workdir import Workdir

#: The frame the second capture is placed in: its origin 30 m south and 20 m west of
#: where the capture is (exactly, by the independent Ferrari inverse in `gps_frames`).
_SOUTH_WEST = gps_frames.enu_to_geodetic(-20.0, -30.0, 0.0, LAT, LON, HEIGHT)
FRAME: dict[str, Any] = {
    "siteId": "site-a",
    "lat": _SOUTH_WEST[0],
    "lon": _SOUTH_WEST[1],
    "height": HEIGHT,
}


def _frame(**override: Any) -> site_frame.SiteFrame:
    parsed = site_frame.from_param({**FRAME, **override})
    assert parsed is not None
    return parsed


def _run(root: Path, params: dict[str, dict[str, Any]]) -> Workdir:
    workdir = Workdir.create(root)
    upload = workdir.input_path("upload")
    upload.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURE_PLY, upload / FIXTURE_PLY.name)
    execute(load_recipe("splat-ingest").with_params(params), workdir, RunnerSet.local())
    return workdir


def _in_frame(frame: dict[str, Any], normalize: dict[str, Any] | None = None) -> dict[str, Any]:
    """PLACEMENT with `site_frame` on the two stages that place, as the worker sends it."""
    return {
        **PLACEMENT,
        "normalize": {
            **PLACEMENT["normalize"],
            "site_frame": frame,
            "lat": LAT,
            "lon": LON,
            **(normalize or {}),
        },
        "georeference": {**PLACEMENT["georeference"], "site_frame": frame},
    }


def _json(workdir: Workdir, stage: str, name: str) -> dict[str, Any]:
    return dict(json.loads((workdir.out_dir(stage) / name).read_text(encoding="utf-8")))


# --- the frame arithmetic --------------------------------------------------------------


def test_the_offset_of_the_origin_is_exactly_zero() -> None:
    frame = _frame()
    assert site_frame.is_zero(site_frame.offset_m(frame, frame.lat, frame.lon))


def test_the_offset_inverts_back_to_the_coordinate() -> None:
    """Forward by `exif`'s WGS84 map, back by Ferrari's closed form (independent code)."""
    frame = _frame()
    for lat, lon in [(LAT, LON), (frame.lat + 0.004, frame.lon - 0.007), (frame.lat, LON)]:
        east, north, up = site_frame.offset_m(frame, lat, lon)
        back_lat, back_lon, _ = gps_frames.enu_to_geodetic(
            east, north, up, frame.lat, frame.lon, frame.height
        )
        assert back_lat == pytest.approx(lat, abs=1e-9)
        assert back_lon == pytest.approx(lon, abs=1e-9)


def test_the_offset_of_the_capture_is_where_the_frame_was_put() -> None:
    east, north, up = site_frame.offset_m(_frame(), LAT, LON)
    # The frame was put 20 m west and 30 m south *in the capture's tangent plane*; read
    # back in the frame's own plane, 36 m away, the two planes differ by d / R (~6e-6 rad)
    # of tilt and turn, which is ~0.2 mm over 36 m. 1 mm is that with margin. The residual
    # `up` is the earth's curvature over 36 m (d^2 / 2R, about 0.1 mm).
    assert east == pytest.approx(20.0, abs=1e-3)
    assert north == pytest.approx(30.0, abs=1e-3)
    assert abs(up) < 1e-3


@pytest.mark.parametrize(
    ("value", "key"),
    [
        ({"lat": 1.0, "lon": 2.0, "height": 3.0}, "siteId"),
        ({**FRAME, "lat": 91.0}, "lat"),
        ({**FRAME, "height": math.nan}, "height"),
        ({**FRAME, "lon": "east"}, "lon"),
        ({**FRAME, "height": True}, "height"),
    ],
)
def test_a_malformed_site_frame_is_refused(value: dict[str, Any], key: str) -> None:
    with pytest.raises(ValueError, match=f"site_frame.{key}"):
        site_frame.from_param(value)


def test_no_site_frame_is_none_and_a_non_object_is_refused() -> None:
    assert site_frame.from_param(None) is None
    with pytest.raises(ValueError, match="must be an object"):
        site_frame.from_param([1, 2, 3])


def test_a_capture_more_than_2047_m_from_its_frame_is_refused() -> None:
    frame = _frame()
    far = gps_frames.enu_to_geodetic(2047.5, 0.0, 0.0, frame.lat, frame.lon, frame.height)
    near = gps_frames.enu_to_geodetic(2046.0, -2046.0, 0.0, frame.lat, frame.lon, frame.height)

    with pytest.raises(ValueError, match="2047 m or more"):
        site_frame.offset_m(frame, far[0], far[1])
    east, north, _ = site_frame.offset_m(frame, near[0], near[1])
    # Not exactly 2046: the point is dropped from the tangent plane onto the frame's
    # height (d^2 / 2R, ~0.65 m down at 2.9 km), which moves it a fraction of a mm.
    assert east == pytest.approx(2046.0, abs=1e-2) and north == pytest.approx(-2046.0, abs=1e-2)


# --- orient_to ---------------------------------------------------------------------


def test_orient_to_without_an_offset_writes_the_bytes_it_always_did(tmp_path: Path) -> None:
    source = splat_stream.open_splat(FIXTURE_PLY)
    absent = splat_stream.orient_to(source, tmp_path / "absent.ply", up_axis="z")
    none = splat_stream.orient_to(source, tmp_path / "none.ply", up_axis="z", offset=None)
    zero = splat_stream.orient_to(
        source, tmp_path / "zero.ply", up_axis="z", offset=[0.0, -0.0, 0.0]
    )

    blob = (tmp_path / "absent.ply").read_bytes()
    assert (tmp_path / "none.ply").read_bytes() == blob
    assert (tmp_path / "zero.ply").read_bytes() == blob
    assert absent.frame is not None and none.frame is not None and zero.frame is not None
    assert absent.frame.to_dict() == none.frame.to_dict() == zero.frame.to_dict()


def test_orient_to_folds_the_offset_into_the_one_recentring_step(tmp_path: Path) -> None:
    source = splat_stream.open_splat(FIXTURE_PLY)
    plain = splat_stream.orient_to(source, tmp_path / "plain.ply", up_axis="z")
    moved = splat_stream.orient_to(
        source, tmp_path / "moved.ply", up_axis="z", offset=(20.0, 30.0, -0.5)
    )
    unrecentred = splat_stream.orient_to(
        source, tmp_path / "flat.ply", up_axis="z", recentre=False, offset=(1.0, 2.0, 3.0)
    )

    assert plain.frame is not None and moved.frame is not None
    assert unrecentred.frame is not None
    total = np.asarray(plain.frame.translation) + np.asarray([20.0, 30.0, -0.5])
    assert np.asarray(moved.frame.translation).tolist() == total.tolist()
    assert np.asarray(unrecentred.frame.translation).tolist() == [1.0, 2.0, 3.0]
    a = gaussians.read_splat(tmp_path / "plain.ply").xyz.astype(np.float64)
    b = gaussians.read_splat(tmp_path / "moved.ply").xyz.astype(np.float64)
    # float32 at ~40 m is ~4e-6 m; 1e-4 m is that with margin, from first principles.
    assert np.abs(b - (a + np.asarray([20.0, 30.0, -0.5]))).max() < 1e-4
    with pytest.raises(ValueError, match="three finite metres"):
        splat_stream.orient_to(source, tmp_path / "bad.ply", up_axis="z", offset=[math.inf, 0, 0])


# --- Lane 1 in a site frame ------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Workdir, Workdir]:
    root = tmp_path_factory.mktemp("site-frame")
    return _run(root / "a", PLACEMENT), _run(root / "b", _in_frame(FRAME))


def test_lane1_in_a_site_frame_keeps_the_origin_and_moves_the_capture(
    runs: tuple[Workdir, Workdir],
) -> None:
    a, b = runs
    georef = _json(b, "georeference", "georef.json")
    offset = site_frame.offset_m(_frame(), LAT, LON)

    # The georeference *is* the frame's origin, exactly, and says how it got there.
    assert (georef["lat"], georef["lon"], georef["height"]) == (
        FRAME["lat"],
        FRAME["lon"],
        FRAME["height"],
    )
    echo = georef["siteFrame"]
    assert echo["siteId"] == "site-a"
    assert (echo["lat"], echo["lon"], echo["height"]) == (FRAME["lat"], FRAME["lon"], HEIGHT)
    assert echo["offsetM"] == list(offset)
    assert echo["offsetSource"] == "capture"
    assert echo["located"] == {"lat": LAT, "lon": LON, "height": HEIGHT}
    assert "in site site-a's frame" in georef["note"]
    assert _json(b, "normalize", "source_meta.json")["siteFrame"] == echo

    # The splat is the same splat, moved by the offset: float32 at ~40 m is ~4e-6 m, so
    # 1e-4 m is that with margin.
    xyz_a = gaussians.read_splat(a.out_dir("normalize") / "canonical.ply").xyz
    xyz_b = gaussians.read_splat(b.out_dir("normalize") / "canonical.ply").xyz
    moved = xyz_a.astype(np.float64) + np.asarray(offset)
    assert np.abs(xyz_b.astype(np.float64) - moved).max() < 1e-4

    # The tiles hang off the frame's origin, not the capture's.
    tileset = _json(b, "package", "splat/tileset.json")
    expected = splat_tiles.enu_to_ecef(FRAME["lat"], FRAME["lon"], HEIGHT)
    assert tileset["root"]["transform"] == pytest.approx(expected, abs=1e-6)

    # And the ground is in the same place on the globe. Not to a centimetre: the ground
    # samples convert metres to degrees by a flat 111,320 m a degree (`gaussians.
    # ground_samples`), which at 28 deg N is ~0.4% long north-south and ~0.06% east-west
    # -- over this 36 m offset, about 12 cm (1.1e-6 deg) of latitude. 2e-6 deg (about
    # 20 cm) is that with margin, from first principles; the exact composition is checked
    # beside it.
    ground_a = _json(a, "ground_samples", "ground_samples.json")
    ground_b = _json(b, "ground_samples", "ground_samples.json")
    assert ground_b["origin"] == {"lat": FRAME["lat"], "lon": FRAME["lon"], "height": HEIGHT}
    assert len(ground_a["samples"]) == len(ground_b["samples"]) > 0
    scale = max(math.cos(math.radians(FRAME["lat"])), 1e-6)
    for one, two in zip(ground_a["samples"], ground_b["samples"], strict=True):
        assert two["lat"] == pytest.approx(one["lat"], abs=2e-6)
        assert two["lon"] == pytest.approx(one["lon"], abs=2e-6)
        # The flat conversion, composed exactly: b's cell is a's, moved by the offset.
        north_a = (one["lat"] - LAT) * gaussians.METRES_PER_DEGREE
        north_b = (two["lat"] - FRAME["lat"]) * gaussians.METRES_PER_DEGREE
        assert north_b - north_a == pytest.approx(offset[1], abs=1e-3)
        east_b = (two["lon"] - FRAME["lon"]) * gaussians.METRES_PER_DEGREE * scale
        assert east_b > 0
        assert two["z"] == pytest.approx(one["z"] + offset[2], abs=1e-3)


def test_lane1_without_a_frame_writes_no_site_frame_key(runs: tuple[Workdir, Workdir]) -> None:
    a, _ = runs
    assert "siteFrame" not in _json(a, "georeference", "georef.json")
    assert "siteFrame" not in _json(a, "normalize", "source_meta.json")


def test_a_capture_with_no_coordinate_sits_at_its_sites_origin(tmp_path: Path) -> None:
    """What the worker sends a capture with no coordinate of its own: `normalize` gets no
    lat/lon (it falls back to the origin), `georeference` gets the frame's own (the recipe
    default of 0, 0 would otherwise be thousands of km away)."""
    params = _in_frame(FRAME)
    params["normalize"] = {k: v for k, v in params["normalize"].items() if k not in ("lat", "lon")}
    params["georeference"] = {
        **params["georeference"],
        "lat": FRAME["lat"],
        "lon": FRAME["lon"],
        "height": FRAME["height"],
    }
    workdir = _run(tmp_path / "run", params)

    echo = _json(workdir, "georeference", "georef.json")["siteFrame"]
    assert echo["offsetM"] == [0.0, 0.0, 0.0]
    assert echo["offsetSource"] == "site-origin"
    assert _json(workdir, "normalize", "source_meta.json")["siteFrame"] == echo


def test_manual_placement_refuses_a_frame_normalize_did_not_use(tmp_path: Path) -> None:
    """A resumed job whose frame changed between attempts must not place silently."""
    other = {**FRAME, "lat": FRAME["lat"] + 0.0001}
    params = _in_frame(FRAME)
    params["georeference"] = {**params["georeference"], "site_frame": other}

    with pytest.raises(Exception, match="re-run from normalize"):
        _run(tmp_path / "run", params)


def test_manual_placement_refuses_a_frame_normalize_was_never_given(tmp_path: Path) -> None:
    params = {**PLACEMENT, "georeference": {**PLACEMENT["georeference"], "site_frame": FRAME}}

    with pytest.raises(Exception, match="re-run from normalize"):
        _run(tmp_path / "run", params)
