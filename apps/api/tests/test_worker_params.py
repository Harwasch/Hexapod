"""What a capture's metadata becomes, per stage, before a run starts.

Pure: an unsaved `Capture` and `Job`, the shipped recipes planned for real, and the
parameter overrides `app.worker.params` derives from them. No database, no bucket.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from app.models import Capture, Job
from app.models.enums import CaptureKind
from app.schemas.capture import UP_AXES, CaptureCreate
from app.worker.params import stage_params
from app.worker.pipeline_bridge import load_recipe, plan_recipe

# Only importable once `pipeline_bridge` has put tools/pipeline on the path, which is why
# they are not ordinary imports: an import sorter would hoist them above the bridge.
gaussians = import_module("gaussians")
import_module("stages")  # registers the shipped implementations


def _params(
    recipe: str,
    metadata: dict[str, Any],
    job_params: dict[str, Any] | None = None,
    frame: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    capture = Capture(slug="c", name="c", kind=CaptureKind.GAUSSIAN_SPLAT, metadata_=metadata)
    job = Job(recipe=recipe, recipe_version="2", params=job_params or {})
    return stage_params(plan_recipe(load_recipe(recipe)), capture, job, frame=frame)


FRAME = {
    "siteId": "0c4f6d36-6f0e-4c4a-9d6b-4cbd7a4a8e11",
    "lat": 47.6,
    "lon": -122.3,
    "height": 30,
    "heightDatum": "msl-unverified",
    "epsg": 32610,
}


def test_up_axis_and_heading_reach_the_normalize_stage() -> None:
    resolved = _params("splat-ingest", {"lat": 1.0, "lon": 2.0, "upAxis": "-y", "headingDeg": 30})

    assert resolved["normalize"]["up_axis"] == "-y"
    assert resolved["normalize"]["heading_deg"] == 30.0
    assert resolved["georeference"]["lat"] == 1.0


def test_no_orientation_means_the_formats_default_decides() -> None:
    resolved = _params("splat-ingest", {"lat": 1.0, "lon": 2.0})

    assert "up_axis" not in resolved.get("normalize", {})


def test_a_job_can_still_override_the_captures_orientation() -> None:
    resolved = _params("splat-ingest", {"upAxis": "y"}, {"normalize": {"up_axis": "z"}})

    assert resolved["normalize"]["up_axis"] == "z"


def test_a_video_capture_hands_its_coordinate_to_the_georeference_as_a_fallback() -> None:
    resolved = _params("photo-reconstruct", {"lat": 51.5, "lon": -0.12, "headingDeg": 90})

    assert resolved["georeference"] == {"lat": 51.5, "lon": -0.12, "heading_deg": 90.0}
    # And nothing orientation-shaped reaches the frame extractor, which has no such knob.
    assert "up_axis" not in resolved.get("normalize", {})


def test_a_video_capture_with_no_coordinate_hands_over_nothing() -> None:
    assert "georeference" not in _params("photo-reconstruct", {})


def test_the_api_and_the_pipeline_agree_on_the_axis_names() -> None:
    assert set(UP_AXES) == set(gaussians.UP_AXES)


def test_an_unknown_up_axis_is_refused_at_capture_creation() -> None:
    with pytest.raises(ValueError, match="upAxis"):
        CaptureCreate(name="x", kind=CaptureKind.GAUSSIAN_SPLAT, metadata={"upAxis": "up"})
    with pytest.raises(ValueError, match="headingDeg"):
        CaptureCreate(name="x", kind=CaptureKind.GAUSSIAN_SPLAT, metadata={"headingDeg": "N"})
    ok = CaptureCreate(name="x", kind=CaptureKind.GAUSSIAN_SPLAT, metadata={"upAxis": "y"})
    assert ok.metadata["upAxis"] == "y"


# --- the site frame (decision D3) -----------------------------------------------------


def test_a_site_frame_reaches_every_stage_that_places_a_capture() -> None:
    lane1 = _params("splat-ingest", {"lat": 47.6001, "lon": -122.3}, frame=FRAME)
    lane2 = _params("photo-reconstruct", {"lat": 47.6001, "lon": -122.3}, frame=FRAME)

    assert lane1["normalize"]["site_frame"] == FRAME
    assert (lane1["normalize"]["lat"], lane1["normalize"]["lon"]) == (47.6001, -122.3)
    assert lane1["georeference"]["site_frame"] == FRAME
    assert (lane1["georeference"]["lat"], lane1["georeference"]["lon"]) == (47.6001, -122.3)
    assert lane2["georeference"]["site_frame"] == FRAME
    assert lane2["georeference"]["lat"] == 47.6001
    # Every stage after them reads the origin from georef.json, so nothing else gets it.
    for resolved in (lane1, lane2):
        placing = {"normalize", "georeference"} if resolved is lane1 else {"georeference"}
        others = {stage for stage, values in resolved.items() if "site_frame" in values}
        assert others == placing


def test_with_a_frame_and_no_coordinate_the_capture_sits_at_its_sites_origin() -> None:
    lane1 = _params("splat-ingest", {}, frame=FRAME)
    lane2 = _params("photo-reconstruct", {}, frame=FRAME)

    assert (lane1["georeference"]["lat"], lane1["georeference"]["lon"]) == (47.6, -122.3)
    assert lane1["georeference"]["height"] == 30
    assert (lane1["normalize"]["lat"], lane1["normalize"]["lon"]) == (47.6, -122.3)
    # Lane 2's stage falls back to the frame's origin itself, after its own evidence.
    assert lane2["georeference"] == {"site_frame": FRAME}


@pytest.mark.parametrize(
    ("recipe", "metadata", "job_params"),
    [
        ("splat-ingest", {"lat": 1.0, "lon": 2.0, "upAxis": "-y", "headingDeg": 30}, None),
        ("splat-ingest", {"lat": 1.0, "lon": 2.0}, None),
        ("splat-ingest", {"upAxis": "y"}, {"normalize": {"up_axis": "z"}}),
        ("photo-reconstruct", {"lat": 51.5, "lon": -0.12, "headingDeg": 90}, None),
        ("photo-reconstruct", {}, None),
    ],
)
def test_no_frame_leaves_the_params_exactly_as_before(
    recipe: str, metadata: dict[str, Any], job_params: dict[str, Any] | None
) -> None:
    capture = Capture(slug="c", name="c", kind=CaptureKind.GAUSSIAN_SPLAT, metadata_=metadata)
    job = Job(recipe=recipe, recipe_version="2", params=job_params or {})
    plan = plan_recipe(load_recipe(recipe))

    assert stage_params(plan, capture, job, frame=None) == stage_params(plan, capture, job)
    assert all("site_frame" not in values for values in stage_params(plan, capture, job).values())


def test_a_typed_place_reaches_normalize_and_georeference_alike() -> None:
    """The upload page's typed place (`georeference: {lat, lon}` in the job's params) wins
    over the phone's fix. In a frame, `normalize` moves the splat by the located point's
    offset and `georeference` refuses any other offset, so both must be told the typed
    place; told the phone's fix and the typed place, the capture could never run."""
    typed = {"georeference": {"lat": 47.6005, "lon": -122.3}}
    with_fix = _params("splat-ingest", {"lat": 47.6001, "lon": -122.3}, typed, FRAME)
    without_fix = _params("splat-ingest", {}, typed, FRAME)

    for resolved in (with_fix, without_fix):
        assert (resolved["normalize"]["lat"], resolved["normalize"]["lon"]) == (47.6005, -122.3)
        assert (resolved["georeference"]["lat"], resolved["georeference"]["lon"]) == (
            47.6005,
            -122.3,
        )
    # With no fix of its own the capture still takes the frame's height as its fallback.
    assert without_fix["georeference"]["height"] == FRAME["height"]


def test_without_a_frame_a_typed_place_leaves_normalize_alone() -> None:
    resolved = _params(
        "splat-ingest", {"lat": 1.0, "lon": 2.0}, {"georeference": {"lat": 3.0, "lon": 4.0}}
    )

    assert "lat" not in resolved.get("normalize", {})
    assert (resolved["georeference"]["lat"], resolved["georeference"]["lon"]) == (3.0, 4.0)


def test_the_located_point_is_the_typed_place_then_the_capture_then_none() -> None:
    from app.worker.params import located_point

    plan = plan_recipe(load_recipe("splat-ingest"))

    def point(metadata: dict[str, Any], job_params: dict[str, Any]) -> object:
        capture = Capture(slug="c", name="c", kind=CaptureKind.GAUSSIAN_SPLAT, metadata_=metadata)
        return located_point(plan, capture, Job(recipe="splat-ingest", params=job_params))

    typed = {"georeference": {"lat": 3.0, "lon": 4.0}}
    assert point({"lat": 1.0, "lon": 2.0}, typed) == (3.0, 4.0)
    assert point({"lat": 1.0, "lon": 2.0}, {}) == (1.0, 2.0)
    assert point({}, {"georeference": {"lat": 3.0}}) is None
    assert point({}, {}) is None


def test_a_job_can_still_override_the_site_frame() -> None:
    other = {**FRAME, "lat": 10.0}
    resolved = _params(
        "splat-ingest", {"lat": 1.0, "lon": 2.0}, {"normalize": {"site_frame": other}}, FRAME
    )

    assert resolved["normalize"]["site_frame"] == other
    assert resolved["georeference"]["site_frame"] == FRAME
