"""`check_targets`: triangulate surveyed marks from posed cameras, and score them.

Pure: cameras from `tree_frames.orbit`, a `sfm.Model` built here, clicks computed by the
very camera model under test, so exact data must come back to machine precision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import gps_frames
import numpy as np
import pytest
from PIL import Image

import check_targets
import exif
import sfm
import tree_frames
from check_targets import Observation, Target, Triangulated

ORIGIN = (28.0389, -82.6966, 10.8)
WIDTH, HEIGHT, FOCAL = 640, 480, 520.0


def _model(poses: tuple[tree_frames.Pose, ...], *, k: float | None = None) -> sfm.Model:
    if k is None:
        camera = sfm.Camera(
            1,
            "PINHOLE",
            WIDTH,
            HEIGHT,
            (FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2),
            ("fx", "fy", "cx", "cy"),
        )
    else:
        camera = sfm.Camera(
            1,
            "SIMPLE_RADIAL",
            WIDTH,
            HEIGHT,
            (FOCAL, WIDTH / 2, HEIGHT / 2, k),
            ("f", "cx", "cy", "k"),
        )
    images = tuple(
        sfm.Image(i + 1, p.name, p.qvec, tuple(float(v) for v in p.translation), 1, 0, 0)  # type: ignore[arg-type]
        for i, p in enumerate(poses)
    )
    return sfm.Model(cameras=(camera,), images=images, points3d=0, mean_track_length=0.0)


def _clicks(
    poses: tuple[tree_frames.Pose, ...], point: np.ndarray, label: str, *, k: float = 0.0
) -> list[Observation]:
    out = []
    for pose in poses:
        camera = pose.rotation @ point + pose.translation
        x, y = camera[0] / camera[2], camera[1] / camera[2]
        factor = 1.0 + k * (x * x + y * y)
        u, v = FOCAL * x * factor + WIDTH / 2, FOCAL * y * factor + HEIGHT / 2
        out.append(Observation(label, pose.name, float(u), float(v)))
    return out


def test_a_target_seen_by_known_cameras_triangulates_where_it_is() -> None:
    poses = tree_frames.orbit(12)
    point = np.array([4.0, -4.0, 0.0])
    found = check_targets.triangulate(_model(poses), _clicks(poses, point, "T1"))

    result = found["T1"]
    assert result.status == "ok" and result.views == 12
    assert result.position is not None
    assert np.abs(np.asarray(result.position) - point).max() < 1e-6
    assert result.reprojection_rms_px is not None and result.reprojection_rms_px < 1e-6


def test_radial_distortion_is_undone_before_triangulating() -> None:
    poses = tree_frames.orbit(12)
    point = np.array([-6.0, 0.0, 0.0])
    clicks = _clicks(poses, point, "T7", k=-0.05)

    with_k = check_targets.triangulate(_model(poses, k=-0.05), clicks)["T7"]
    ignored = check_targets.triangulate(_model(poses), clicks)["T7"]

    assert with_k.position is not None and ignored.position is not None
    assert np.abs(np.asarray(with_k.position) - point).max() < 1e-6
    assert float(np.linalg.norm(np.asarray(ignored.position) - point)) > 1e-3


def test_one_view_or_a_narrow_angle_is_not_evaluated() -> None:
    poses = tree_frames.orbit(12)
    point = np.array([4.0, 4.0, 0.0])
    one = check_targets.triangulate(_model(poses), _clicks(poses[:1], point, "T1"))
    assert one["T1"].status == "too-few-views" and one["T1"].position is None

    # Two cameras 5 cm apart, 10 m from the target: rays 0.3 deg apart.
    a = tree_frames.look_at("a.jpg", np.array([10.0, 0.0, 2.0]), point)
    b = tree_frames.look_at("b.jpg", np.array([10.0, 0.05, 2.0]), point)
    narrow = check_targets.triangulate(_model((a, b)), _clicks((a, b), point, "T1"))
    assert narrow["T1"].status == "too-narrow" and narrow["T1"].position is None
    assert narrow["T1"].angle_deg < check_targets.MIN_TRIANGULATION_ANGLE_DEG

    # A click on a frame the model did not register is no view at all.
    unposed = [Observation("T1", "unregistered.jpg", 1.0, 1.0)]
    assert check_targets.triangulate(_model(poses), unposed)["T1"].views == 0


def test_original_pixels_are_scaled_to_the_kept_frame(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("frame_0000.jpg", "frame_0001.jpg"):
        Image.new("RGB", (WIDTH, HEIGHT)).save(frames / name)
    meta = {
        "frames": [
            {"name": "frame_0000.jpg", "original": "DJI_0042.JPG", "originalSize": [5472, 3648]},
            {"name": "frame_0001.jpg", "original": "clip.mov", "originalSize": None},
        ]
    }
    document = {
        "observations": [
            {
                "target": "T1",
                "image": "DJI_0042.JPG",
                "px": [2736.0, 1824.0],
                "imageSize": [5472, 3648],
            },
            # No imageSize: the original's own, from frames_meta.
            {"target": "T1", "image": "DJI_0042.JPG", "px": [5472.0, 0.0]},
            {"target": "T1", "image": "frame_0001.jpg", "uv": [0.5, 0.25]},
            {"target": "T1", "image": "frame_0000.jpg", "px": [10.5, 20.5]},
            {"target": "T1", "image": "nowhere.jpg", "uv": [0.5, 0.5]},
            {"target": "T9", "image": "frame_0000.jpg", "uv": [0.5, 0.5]},
            {"target": "T1", "image": "frame_0000.jpg"},
        ]
    }
    targets = [Target("T1", "check", 0.0, 0.0, 0.0, "msl-unverified", 0.005, 0.01)]
    found, problems = check_targets.load_observations(
        document, frames=frames, frames_meta=meta, targets=targets
    )

    assert [(o.frame, o.u, o.v) for o in found] == [
        ("frame_0000.jpg", 320.0, 240.0),
        ("frame_0000.jpg", 640.0, 0.0),
        ("frame_0001.jpg", 320.0, 120.0),
        ("frame_0000.jpg", 10.5, 20.5),
    ]
    assert len(problems) == 3
    assert "neither a kept frame nor an original" in problems[0]
    assert "unknown target 'T9'" in problems[1]
    assert "needs px" in problems[2]

    # Every frame cut from one video names the same original: ambiguous, so refused.
    video = {
        "frames": [
            {"name": "frame_0000.jpg", "original": "clip.mov"},
            {"name": "frame_0001.jpg", "original": "clip.mov"},
        ]
    }
    _, why = check_targets.load_observations(
        {"observations": [{"target": "T1", "image": "clip.mov", "uv": [0.5, 0.5]}]},
        frames=frames,
        frames_meta=video,
    )
    assert "original of 2 kept frames" in why[0]


def _target(label: str, enu: tuple[float, float, float], datum: str = "msl-unverified") -> Target:
    lat, lon, height = gps_frames.enu_to_geodetic(*enu, *ORIGIN)
    return Target(label, "check", lat, lon, height, datum, 0.005, 0.01)


def _found(
    label: str, position: tuple[float, float, float], images: tuple[str, ...]
) -> Triangulated:
    return Triangulated(label, position, len(images), images, 40.0, 0.5, "ok")


IDENTITY = sfm.Similarity(1.0, np.eye(3), np.zeros(3))


def _add(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def test_rmse_is_horizontal_vertical_total_and_per_target() -> None:
    poses = tree_frames.orbit(8)
    model = _model(poses)
    names = tuple(p.name for p in poses)
    truth = {"T1": (4.0, 4.0, 0.0), "T2": (-4.0, 4.0, 0.0), "T3": (-4.0, -4.0, 0.0)}
    errors = {"T1": (0.03, 0.04, 0.0), "T2": (0.0, 0.0, -0.02), "T3": (0.0, 0.0, 0.0)}
    targets = [_target(label, enu) for label, enu in truth.items()]
    triangulated = {
        label: _found(label, _add(truth[label], errors[label]), names) for label in truth
    }

    block = check_targets.evaluate(
        triangulated,
        targets,
        to_frame=IDENTITY,
        origin=ORIGIN,
        frame_datum="msl-unverified",
        model=model,
        anchor_h=0.02,
        anchor_v=0.02,
    )

    assert block["status"] == "evaluated" and block["evaluated"] == 3 and block["checks"] == 3
    rmse = block["rmseM"]
    assert isinstance(rmse, dict)
    assert rmse["horizontal"] == pytest.approx(math.sqrt(0.05**2 / 3), abs=1e-4)
    assert rmse["vertical"] == pytest.approx(math.sqrt(0.02**2 / 3), abs=1e-4)
    assert rmse["total"] == pytest.approx(math.sqrt((0.05**2 + 0.02**2) / 3), abs=1e-4)
    per = {entry["label"]: entry for entry in block["perTarget"]}  # type: ignore[attr-defined]
    assert per["T1"]["horizontalM"] == pytest.approx(0.05, abs=1e-4)
    assert per["T2"]["verticalM"] == pytest.approx(0.02, abs=1e-4)
    # The footprint is the camera-to-target distance over the focal length, median.
    distances = [float(np.linalg.norm(np.asarray(truth["T3"]) - p.centre)) for p in poses]
    assert per["T3"]["pixelFootprintM"] == pytest.approx(np.median(distances) / FOCAL, abs=1e-4)
    assert block["verticalEvaluated"] is True and block["withinTolerance"] is True
    json.dumps(block)


def test_vertical_is_not_scored_across_height_datums() -> None:
    poses = tree_frames.orbit(8)
    names = tuple(p.name for p in poses)
    target = _target("T1", (4.0, 4.0, 0.0), datum="ellipsoid-wgs84")
    block = check_targets.evaluate(
        {"T1": _found("T1", (4.0, 4.0, 30.0), names)},
        [target],
        to_frame=IDENTITY,
        origin=ORIGIN,
        frame_datum="msl-unverified",
        model=_model(poses),
        anchor_h=0.02,
        anchor_v=0.02,
    )

    assert block["verticalEvaluated"] is False
    per = block["perTarget"][0]  # type: ignore[index]
    assert per["verticalM"] is None and per["horizontalM"] == pytest.approx(0.0, abs=1e-4)
    assert block["withinTolerance"] is True
    assert "height datum" in str(block["reason"])


def test_survey_tolerance_is_three_sigma_of_the_declared_uncertainties() -> None:
    tolerance = check_targets.survey_tolerance(
        anchor_h=0.02, anchor_v=0.02, target_h=0.005, target_v=0.01, pick_m=0.019
    )
    assert tolerance["horizontalM"] == pytest.approx(
        3 * math.sqrt(2) * math.sqrt(0.02**2 + 0.005**2 + 0.019**2)
    )
    assert tolerance["verticalM"] == pytest.approx(3 * math.sqrt(0.02**2 + 0.01**2 + 0.019**2))


def test_targets_are_checked_and_no_targets_is_none() -> None:
    good = {
        "label": "T1",
        "role": "check",
        "lat": 1.0,
        "lon": 2.0,
        "height": 3.0,
        "heightDatum": "msl-unverified",
        "sigmaHM": 0.005,
        "sigmaVM": 0.01,
    }
    assert check_targets.load_targets({"targets": [good]})[0].sigma_v_m == 0.01
    with pytest.raises(ValueError, match="appears twice"):
        check_targets.load_targets({"targets": [good, good]})
    with pytest.raises(ValueError, match="heightDatum"):
        check_targets.load_targets({"targets": [{**good, "heightDatum": "geoid"}]})
    block = check_targets.evaluate(
        {},
        [],
        to_frame=IDENTITY,
        origin=ORIGIN,
        frame_datum="msl-unverified",
        model=_model(()),
        anchor_h=1.0,
        anchor_v=1.0,
    )
    assert block["status"] == "none" and block["evaluated"] == 0
    # The fixture's inverse and the stage's forward map agree, so targets land exactly.
    lat, lon, height = gps_frames.enu_to_geodetic(4.0, 4.0, 0.0, *ORIGIN)
    enu = exif.ecef_to_enu(exif.geodetic_to_ecef(lat, lon, height), *ORIGIN)
    assert np.abs(enu - np.array([4.0, 4.0, 0.0])).max() < 1e-6


def test_a_target_on_another_datum_does_not_silence_the_others_vertical_errors() -> None:
    """R2 review (#51, high): one target surveyed on another height datum cleared
    `verticalEvaluated`, and with it the vertical test of every other target -- three
    marks 0.5 m out vertically passed. The others are still scored, and fail."""
    poses = tree_frames.orbit(8)
    names = tuple(p.name for p in poses)
    truth = {"T1": (4.0, 4.0, 0.0), "T2": (-4.0, 4.0, 0.0), "T3": (-4.0, -4.0, 0.0)}
    targets = [_target(label, enu) for label, enu in truth.items()]
    targets.append(_target("T4", (4.0, -4.0, 0.0), datum="ellipsoid-wgs84"))
    found = {
        label: _found(label, _add(enu, (0.0, 0.0, 0.5)), names) for label, enu in truth.items()
    }
    found["T4"] = _found("T4", (4.0, -4.0, 30.0), names)

    block = check_targets.evaluate(
        found,
        targets,
        to_frame=IDENTITY,
        origin=ORIGIN,
        frame_datum="msl-unverified",
        model=_model(poses),
        anchor_h=0.02,
        anchor_v=0.02,
    )

    assert block["verticalEvaluated"] is False  # not on every target ...
    assert block["verticalChecks"] == 3  # ... but on these three
    rmse = block["rmseM"]
    assert isinstance(rmse, dict) and rmse["vertical"] == pytest.approx(0.5, abs=1e-4)
    assert block["withinTolerance"] is False


def test_check_targets_all_on_another_datum_score_no_vertical_checks() -> None:
    poses = tree_frames.orbit(8)
    names = tuple(p.name for p in poses)
    truth = {"T1": (4.0, 4.0, 0.0), "T2": (-4.0, 4.0, 0.0), "T3": (-4.0, -4.0, 0.0)}
    targets = [_target(label, enu, datum="ellipsoid-wgs84") for label, enu in truth.items()]
    block = check_targets.evaluate(
        {label: _found(label, enu, names) for label, enu in truth.items()},
        targets,
        to_frame=IDENTITY,
        origin=ORIGIN,
        frame_datum="msl-unverified",
        model=_model(poses),
        anchor_h=0.02,
        anchor_v=0.02,
    )
    assert block["checks"] == 3 and block["verticalChecks"] == 0
