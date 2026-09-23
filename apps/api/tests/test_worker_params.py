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
    recipe: str, metadata: dict[str, Any], job_params: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    capture = Capture(slug="c", name="c", kind=CaptureKind.GAUSSIAN_SPLAT, metadata_=metadata)
    job = Job(recipe=recipe, recipe_version="2", params=job_params or {})
    return stage_params(plan_recipe(load_recipe(recipe)), capture, job)


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


def test_the_api_and_the_pipeline_agree_on_the_axis_names() -> None:
    assert set(UP_AXES) == set(gaussians.UP_AXES)


def test_an_unknown_up_axis_is_refused_at_capture_creation() -> None:
    with pytest.raises(ValueError, match="upAxis"):
        CaptureCreate(name="x", kind=CaptureKind.GAUSSIAN_SPLAT, metadata={"upAxis": "up"})
    with pytest.raises(ValueError, match="headingDeg"):
        CaptureCreate(name="x", kind=CaptureKind.GAUSSIAN_SPLAT, metadata={"headingDeg": "N"})
    ok = CaptureCreate(name="x", kind=CaptureKind.GAUSSIAN_SPLAT, metadata={"upAxis": "y"})
    assert ok.metadata["upAxis"] == "y"
