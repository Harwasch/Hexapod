"""The milestone-3 demo as a test: two epochs of the tree align, and say what they may.

`registration_scene.build` flies an RTK drone over the synthetic tree (epoch 1, with
eight surveyed check targets), then a second RTK flight and an iPhone-style video on
another orbit, aligns each to epoch 1 on pycolmap 4.2 with the shipped stages, and an
unrelated scene that must not align. Everything here is scored against the truth.

**No bound below is measured off a run here.** Each comes from a sigma the scene
declares (`RTK_SIGMA_M`, the targets' survey sigma, the click noise), from a rule of the
stage itself (the tolerance it computed from those, the uncertainty it claimed), or
from the pose bound `test_pose_colmap4.py` already holds; each assert says which.
Not bit-reproducible: pycolmap's matching and mapping vary with threads, so nothing is
compared byte for byte (the numpy Sim3 alone is, and `test_sim3.py` holds that).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

import check_targets
import colmap4
import exif
import global_sfm
import registration_scene as scene

pytestmark = pytest.mark.skipif(
    colmap4.version(colmap4.python()) is None,
    reason=f"needs pycolmap=={global_sfm.PYCOLMAP_VERSION} (or ${global_sfm.PYTHON_ENV})",
)

SIGMA_K = 3.0


@pytest.fixture(scope="module")
def epochs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, scene.Epoch]:
    return scene.build(tmp_path_factory.mktemp("epochs"))


def test_the_rtk_epoch_1_is_survey_grade_against_its_check_targets(
    epochs: dict[str, scene.Epoch],
) -> None:
    a = epochs["e1"].alignment
    assert (a["platform"], a["platformSource"]) == ("rtk-drone", "frames_meta")
    assert a["method"] == "gnss-prior-ba" and a["verdict"] == "survey"
    assert a["label"].startswith("RTK drone, surveyed, check targets ±")
    checks = a["checkTargets"]
    assert checks["evaluated"] == 8 and checks["verticalEvaluated"] is True
    # The tolerance is three sigma of what was declared: the RTK sigma the XMP carried
    # (and the priors took), the targets' survey sigma, one pick pixel's footprint.
    sigma = checks["sigmaM"]
    assert sigma["anchorH"] == sigma["anchorV"] == scene.RTK_SIGMA_M
    expected = check_targets.survey_tolerance(
        anchor_h=scene.RTK_SIGMA_M,
        anchor_v=scene.RTK_SIGMA_M,
        target_h=scene.TARGET_SIGMA_H_M,
        target_v=scene.TARGET_SIGMA_V_M,
        pick_m=sigma["pick"],
    )
    assert checks["tolerance"]["horizontalM"] == pytest.approx(expected["horizontalM"], abs=1e-4)
    assert checks["tolerance"]["verticalM"] == pytest.approx(expected["verticalM"], abs=1e-4)
    assert checks["rmseM"]["horizontal"] <= checks["tolerance"]["horizontalM"]
    assert checks["rmseM"]["vertical"] <= checks["tolerance"]["verticalM"]
    # Cameras held to fixes with N(0, sigma) per axis are within 3*sqrt(3)*sigma of the
    # truth, with no fit; and within the uncertainty the epoch claims.
    rmse = scene.truth_centre_rmse(epochs["e1"])
    assert rmse is not None
    assert rmse <= SIGMA_K * math.sqrt(3.0) * scene.RTK_SIGMA_M
    assert rmse <= a["uncertaintyM"]


def test_the_rtk_epoch_2_aligns_visually_within_survey_tolerance(
    epochs: dict[str, scene.Epoch],
) -> None:
    a = epochs["e2_drone"].alignment
    assert a["method"] == "visual" and a["verdict"] == "survey"
    assert a["referenceEpochId"] == "e1" and a["frameSource"] == "epoch-alignment"
    registration = a["registration"]
    # The acceptance rule itself: a majority of the sampled frames registered.
    assert (
        registration["registeredFraction"] >= a["tolerance"]["alignment"]["minRegisteredFraction"]
    )
    assert a["inliers"] >= a["tolerance"]["alignment"]["minInliers"]
    rmse = scene.truth_centre_rmse(epochs["e2_drone"])
    assert rmse is not None
    # Within the survey tolerance its declared sigmas give, and within what it claims.
    assert rmse <= a["checkTargets"]["tolerance"]["horizontalM"]
    assert rmse <= a["uncertaintyM"]
    # Never claimed better than the reference it was aligned into.
    assert a["uncertaintyM"] >= epochs["e1"].alignment["uncertaintyM"]
    # Its own GNSS frame agrees with the visual one: both claim centimetres, so camera by
    # camera they are within three sigma of the two claims combined.
    check = a["gnssCheck"]
    assert check["kind"] == "similarity"
    bound = SIGMA_K * math.hypot(math.sqrt(3.0) * scene.RTK_SIGMA_M, a["uncertaintyM"])
    assert check["offsetM"] <= bound, (check, bound)


def test_the_phone_epoch_aligns_visually_at_the_coarse_tier(
    epochs: dict[str, scene.Epoch],
) -> None:
    epoch = epochs["e2_phone"]
    a = epoch.alignment
    assert (a["platform"], a["platformSource"]) == ("phone-video", "frames_meta")
    assert a["method"] == "visual" and a["tier"] == "coarse" and a["verdict"] == "coarse"
    assert a["label"].startswith("Phone video, aligned visually, ±")
    assert "surveyed" not in a["label"]
    rmse = scene.truth_centre_rmse(epoch)
    assert rmse is not None and rmse <= a["uncertaintyM"]
    # Aligning beat what its GPS point alone was worth.
    assert a["uncertaintyM"] < epoch.georef["uncertaintyM"]
    # GNSS checks it, never decides it: the cameras' centroid against the located point
    # is the offset that was written into the clip, to within three sigma of the claim.
    check = a["gnssCheck"]
    assert check["kind"] == "located-point"
    meta = json.loads(epoch.path("normalize", "source_meta.json").read_text())
    written = exif.enu_offsets(
        [exif.Fix(name="p", lat=meta["location"]["lat"], lon=meta["location"]["lon"])],
        epoch.origin,
    )["p"]
    orbit_centre = exif.reframe_enu(np.zeros(3), scene.ORIGIN, epoch.origin)
    truth = float(np.hypot(written[0] - orbit_centre[0], written[1] - orbit_centre[1]))
    assert truth == pytest.approx(scene.gnss_offset_truth(), abs=0.2)  # 6-decimal ISO 6709
    bound = SIGMA_K * math.sqrt(2.0) * a["uncertaintyM"]
    assert abs(check["horizontalOffsetM"] - truth) <= bound, (check, truth, bound)


def test_epoch_1_is_untouched_by_aligning_epoch_2(epochs: dict[str, scene.Epoch]) -> None:
    assert epochs["e1"].fingerprint == scene.fingerprint_of(epochs["e1"])
    for name in ("e2_drone", "e2_phone", "e2_phone_framed", "unrelated"):
        # The fixed-frame guarantee: round-off, not a tolerance.
        assert epochs[name].alignment["registration"]["referenceMovedMaxM"] < 1e-9, name


@pytest.mark.parametrize("name", ["e2_drone", "e2_phone", "e2_phone_framed"])
def test_place_puts_both_epoch_2s_on_the_scene(epochs: dict[str, scene.Epoch], name: str) -> None:
    """`place` applies the alignment's frame: the placed sparse points sit on the tree or
    its ground. `e2_phone_framed` is the review's fix 2 -- the same clip told epoch 1's
    origin as its site frame, so its georeference carries the GPS point's 9.4 m offset
    from it, which `place` must not add on top of the alignment."""
    epoch = epochs[name]
    a = epoch.alignment
    step = json.loads(epoch.workdir.step_path("place").read_text())
    assert step["metrics"]["frameFrom"] == "epoch-alignment"
    if name == "e2_phone_framed":
        offset = epoch.georef["siteFrame"]["offsetM"]
        assert math.hypot(offset[0], offset[1]) > 5.0  # the offset that must not be added
    # Within three sigma of the claimed uncertainty and one pixel's footprint (the sparse
    # points' own triangulation noise), combined.
    bound = SIGMA_K * math.hypot(a["uncertaintyM"], a["tolerance"]["alignment"]["pixelFootprintM"])
    distance = scene.placed_scene_distance(epoch)
    assert distance <= bound, (name, distance, bound)


def test_an_unrelated_capture_is_unaligned_and_keeps_its_gnss_placement(
    epochs: dict[str, scene.Epoch],
) -> None:
    a = epochs["unrelated"].alignment
    assert a["verdict"] == "unaligned" and a["method"] == "none"
    assert a["confidence"] == 0.0 and a["frameSource"] == "georef"
    assert a["frame"] == epochs["unrelated"].georef["frame"]
    assert a["label"].startswith("Drone, not aligned: placed by hand")
    assert (
        a["registration"]["registered"]
        < a["tolerance"]["alignment"]["minRegisteredFraction"] * a["registration"]["newFrames"]
    )
    assert any("fell back" in reason for reason in a["reasons"])


def test_the_demo_report_has_a_row_per_epoch(epochs: dict[str, scene.Epoch]) -> None:
    from experiments import registration_demo

    text = registration_demo.report(epochs)
    rows: list[dict[str, Any]] = registration_demo.rows(epochs)

    assert [row["epoch"] for row in rows] == list(epochs)
    for name, epoch in epochs.items():
        assert name in text and epoch.alignment["label"] in text
    json.dumps(rows)
