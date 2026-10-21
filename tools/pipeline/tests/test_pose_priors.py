"""GNSS pose priors (`priors.py`, `colmap4.py priors`) in the pose stage's bundle adjustment.

The planning is pure and tested on hand-written fixes and `frames_meta.json` records in
M3a's real shape (a list of records keyed by `name`; RTK in `dji.rtk`, fixed is flag 50;
`gps.hAccuracyM`). The bundle adjustment runs on pycolmap 4.2 over the rendered orbit,
tagged with seeded noise of a declared sigma -- and every bound on it is derived from
that sigma, not measured off a run here.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gps_frames
import numpy as np
import pytest
from PIL import Image

import colmap4
import exif
import global_sfm
import priors
import sfm
import site_frame
import stages
import tree_frames
from conftest import FIXTURE_PLY, make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

ORIGIN = (28.0389, -82.6966, 10.8)
FRAMES = 40
#: The RTK-grade sigma the orbit is tagged with, per axis, and declared to the priors.
RTK_SIGMA_M = 0.02
POSE = {
    "matcher": "exhaustive",
    "max_features": 4096,
    "max_image_size": 1600,
    "first_octave": 0,
    "colmap": "4.2",
}

requires_pycolmap = pytest.mark.skipif(
    colmap4.version(colmap4.python()) is None,
    reason=f"needs pycolmap=={global_sfm.PYCOLMAP_VERSION} (or ${global_sfm.PYTHON_ENV})",
)


def _frames(root: Path, fixes: Mapping[str, tuple[float, float, float | None]]) -> Path:
    """Tiny JPEGs with a GPS tag each (`None` altitude: none written)."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (east, north, up) in fixes.items():
        path = root / name
        Image.new("RGB", (16, 12), (90, 90, 90)).save(path, format="JPEG")
        lat, lon, alt = gps_frames.enu_to_geodetic(east, north, up or 0.0, *ORIGIN)
        gps_frames.write_gps(path, lat, lon, alt)
        if up is None:
            _drop_altitude(path)
    return root


def _drop_altitude(path: Path) -> None:
    with Image.open(path) as image:
        tags = image.getexif()
        gps = dict(tags.get_ifd(0x8825))
        pixels = image.copy()
    gps.pop(5, None)
    gps.pop(6, None)
    rewritten = Image.Exif()
    rewritten[0x8825] = gps
    pixels.save(path, format="JPEG", exif=rewritten)


def _plan(frames: Path, **overrides: Any) -> priors.PriorPlan:
    arguments: dict[str, Any] = {
        "requested": "auto",
        "engine": "4.2",
        "frame": None,
        "frames_meta": None,
        "std_m": None,
        "std_v_m": None,
    }
    arguments.update(overrides)
    return priors.plan_priors(frames, **arguments)


FIXES = {
    "frame_0000.jpg": (9.0, 0.0, 3.6),
    "frame_0001.jpg": (0.0, 9.0, 3.6),
    "frame_0002.jpg": (-9.0, 0.0, 3.6),
    "frame_0003.jpg": (0.0, -9.0, 3.6),
}


# --- planning, pure ---------------------------------------------------------------------


def test_fixes_become_enu_priors_about_the_median_fix(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "f", FIXES)
    plan = _plan(frames)
    fixes = exif.read_fixes(frames)
    median = exif.median_fix(fixes)
    expected = exif.enu_offsets(fixes, median)

    assert plan.used and plan.count == 4
    assert plan.origin == median and plan.origin_source == "median-fix"
    assert plan.height_datum == "msl-unverified"
    for prior in plan.priors:
        assert prior.position == expected[prior.name]
        assert (prior.std_h_m, prior.std_v_m, prior.source) == (
            priors.DEFAULT_STD_M,
            priors.DEFAULT_STD_M * priors.VERTICAL_FACTOR,
            "default",
        )
    summary = plan.summary()
    assert summary["vertical"] == "constrained"
    assert summary["coordinateSystem"] == "CARTESIAN" and summary["robustLoss"] is True
    plan.write(tmp_path / "priors.json")
    written = json.loads((tmp_path / "priors.json").read_text())
    assert written["coordinateSystem"] == "CARTESIAN"
    assert written["priors"][0]["covariance"] == [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0, 0, 100]]


def test_frames_meta_accuracy_wins_over_the_param(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "f", FIXES)
    meta = {
        "schema": "frames_meta/1",
        "frames": [
            {
                "name": "frame_0000.jpg",
                "gps": {"hAccuracyM": 0.5, "altitudeDatum": "msl-unverified"},
                "dji": {"rtk": {"flag": 50, "stdLonM": 0.012, "stdLatM": 0.015, "stdHgtM": 0.03}},
            },
            {"name": "frame_0001.jpg", "gps": {"hAccuracyM": 3.0}, "dji": None},
            {
                # An RTK float solution borrows nothing from its RTK figures.
                "name": "frame_0002.jpg",
                "gps": {"hAccuracyM": None},
                "dji": {"rtk": {"flag": 34, "stdLonM": 0.2, "stdLatM": 0.2, "stdHgtM": 0.3}},
            },
        ],
    }
    plan = _plan(frames, frames_meta=meta, std_m=2.0, std_v_m=None)
    by = {p.name: p for p in plan.priors}

    assert (by["frame_0000.jpg"].std_h_m, by["frame_0000.jpg"].std_v_m) == (0.015, 0.03)
    assert by["frame_0000.jpg"].source == "frames_meta:dji-rtk"
    assert (by["frame_0001.jpg"].std_h_m, by["frame_0001.jpg"].std_v_m) == (3.0, 6.0)
    assert by["frame_0001.jpg"].source == "frames_meta:gps-h-accuracy"
    assert (by["frame_0002.jpg"].std_h_m, by["frame_0002.jpg"].source) == (2.0, "param")
    # No record at all: the param, and the vertical factor.
    assert (by["frame_0003.jpg"].std_h_m, by["frame_0003.jpg"].std_v_m) == (2.0, 4.0)


def test_a_datum_mismatch_unconstrains_the_vertical(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "f", {**FIXES, "frame_0004.jpg": (1.0, 1.0, None)})
    plan = _plan(frames)
    by = {p.name: p for p in plan.priors}
    assert by["frame_0004.jpg"].std_v_m == priors.UNCONSTRAINED_STD_M
    assert by["frame_0000.jpg"].std_v_m == priors.DEFAULT_STD_M * priors.VERTICAL_FACTOR
    assert plan.vertical().startswith("unconstrained on 1 of 5: a fix with no altitude")

    frame = site_frame.from_param(
        {
            "siteId": "s",
            "lat": ORIGIN[0],
            "lon": ORIGIN[1],
            "height": ORIGIN[2],
            "heightDatum": "ellipsoid-wgs84",
        }
    )
    framed = _plan(_frames(tmp_path / "g", FIXES), frame=frame)
    assert framed.origin_source == "site-frame" and framed.height_datum == "ellipsoid-wgs84"
    assert {p.std_v_m for p in framed.priors} == {priors.UNCONSTRAINED_STD_M}
    assert framed.vertical() == (
        "unconstrained: fix altitude is msl-unverified, the frame is ellipsoid-wgs84"
    )


def test_3_9_and_off_record_why_no_priors(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "f", FIXES)

    off = _plan(frames, requested="off")
    assert (off.used, off.reason, off.count) == (False, "pose_priors: off", 0)
    old = _plan(frames, engine="3.9")
    assert not old.used and "COLMAP 3.9.1 has no pose-prior bundle adjustment" in old.reason
    few = _plan(_frames(tmp_path / "g", dict(list(FIXES.items())[:2])))
    assert not few.used and few.reason == "2 EXIF fixes, fewer than 3"
    with pytest.raises(ValueError, match="pose_priors must be one of off, auto"):
        _plan(frames, requested="on")
    with pytest.raises(ValueError, match="prior_std_m must be a positive"):
        _plan(frames, std_m=0.0)


def test_the_priors_origin_is_the_georeference_origin(tmp_path: Path) -> None:
    fixes = exif.read_fixes(_frames(tmp_path / "f", FIXES))

    assert priors.frame_origin(fixes, None)[0] == exif.median_fix(fixes)
    assert stages._aligned_origin(fixes, None) == exif.median_fix(fixes)
    frame = site_frame.from_param(
        {"siteId": "s", "lat": ORIGIN[0], "lon": ORIGIN[1], "height": 2.0}
    )
    assert frame is not None
    assert priors.frame_origin(fixes, frame)[0] == stages._aligned_origin(fixes, frame)
    assert priors.frame_origin(fixes, frame)[1:] == ("site-frame", "msl-unverified")


# --- bundle adjustment, pycolmap 4.2 ------------------------------------------------------


@pytest.fixture(scope="module")
def rtk_orbit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, tree_frames.Truth]:
    frames = tmp_path_factory.mktemp("priors") / "frames"
    truth = tree_frames.render_orbit(FIXTURE_PLY, frames, count=FRAMES)
    gps_frames.tag_orbit(
        frames, {p.name: p.centre for p in truth.poses}, ORIGIN, noise_m=RTK_SIGMA_M
    )
    return frames, truth


def _pose(root: Path, frames: Path, **params: object) -> tuple[Path, dict[str, Any]]:
    workdir = Workdir.create(root)
    shutil.copytree(frames, workdir.input_path("frames"))
    execute(
        make_recipe(
            [{"id": "pose", "impl": "colmap", "params": {**POSE, **params}}], inputs=["frames"]
        ),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )
    poses = workdir.artifact_path("pose", "poses")
    return poses, json.loads((poses / "poses.json").read_text())


def _centre_errors(poses: Path, frames: Path, truth: tree_frames.Truth) -> np.ndarray:
    """Each posed centre against its true one, both east/north/up about the priors' origin
    (the median fix), with no fit of any kind in between."""
    median = exif.median_fix(exif.read_fixes(frames))
    known = truth.by_name()
    model = sfm.read_model(poses)
    true = exif.reframe_enu(np.stack([known[i.name].centre for i in model.images]), ORIGIN, median)
    return np.asarray(np.linalg.norm(model.centres() - true, axis=1))


@pytest.fixture(scope="module")
def with_priors(
    tmp_path_factory: pytest.TempPathFactory, rtk_orbit: tuple[Path, tree_frames.Truth]
) -> tuple[Path, dict[str, Any]]:
    if colmap4.version(colmap4.python()) is None:
        pytest.skip("needs pycolmap")
    frames, _ = rtk_orbit
    return _pose(
        tmp_path_factory.mktemp("with") / "run",
        frames,
        pose_priors="auto",
        prior_std_m=RTK_SIGMA_M,
        prior_std_v_m=RTK_SIGMA_M,
    )


@requires_pycolmap
def test_rtk_priors_land_the_model_in_the_site_frame(
    with_priors: tuple[Path, dict[str, Any]], rtk_orbit: tuple[Path, tree_frames.Truth]
) -> None:
    poses, document = with_priors
    frames, truth = rtk_orbit
    errors = _centre_errors(poses, frames, truth)

    assert document["priors"]["used"] is True and document["priors"]["count"] == FRAMES
    assert document["priors"]["sigmaM"]["source"] == "param"
    assert document["priors"]["mapper"] == "incremental"
    assert document["scale"]["metric"] is True
    assert document["scale"]["source"] == "gnss-pose-priors"
    assert document["registered"] == FRAMES
    # Derived, not measured: each fix is the true centre plus N(0, sigma) per axis, so a
    # camera held to its fix is within 3 sigma per axis -- 3*sqrt(3)*sigma = 0.104 m in
    # 3D -- of the truth, with no similarity fitted at all.
    bound = 3.0 * np.sqrt(3.0) * RTK_SIGMA_M
    assert float(np.median(errors)) < bound, (float(np.median(errors)), bound)


@requires_pycolmap
def test_without_priors_the_model_is_not_in_the_site_frame(
    tmp_path: Path, rtk_orbit: tuple[Path, tree_frames.Truth]
) -> None:
    frames, truth = rtk_orbit
    poses, document = _pose(tmp_path / "without", frames)

    assert document["priors"]["used"] is False
    assert document["priors"]["reason"] == "pose_priors: off"
    assert document["scale"]["metric"] is False
    # COLMAP's own frame is arbitrary in origin, rotation and scale: nowhere near the
    # metric frame, by far more than the priors' 3-sigma bound.
    assert float(np.median(_centre_errors(poses, frames, truth))) > 1.0


@requires_pycolmap
def test_a_priors_run_keeps_3_9s_three_file_layout(
    with_priors: tuple[Path, dict[str, Any]],
) -> None:
    poses, _ = with_priors
    assert sorted(p.name for p in poses.iterdir()) == [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "poses.json",
    ]
    assert sfm.read_model(poses).registered == FRAMES


@requires_pycolmap
def test_a_prior_naming_a_frame_the_database_has_not_got_is_refused(tmp_path: Path) -> None:
    import subprocess

    database = tmp_path / "d.db"
    images = tmp_path / "images"
    _frames(images, dict(list(FIXES.items())[:1]))
    subprocess.run(
        colmap4.extract_argv(colmap4.python(), database, images), check=True, capture_output=True
    )
    document = {
        "coordinateSystem": "CARTESIAN",
        "priors": [{"name": "nope.jpg", "position": [0, 0, 0], "covariance": np.eye(3).tolist()}],
    }
    (tmp_path / "p.json").write_text(json.dumps(document))
    done = subprocess.run(
        colmap4.priors_argv(colmap4.python(), database, tmp_path / "p.json"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0 and "nope.jpg" in done.stdout + done.stderr
