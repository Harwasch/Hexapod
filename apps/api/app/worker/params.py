"""What a run is told that its recipe file cannot know.

A recipe is a deployment-level document: it says a capture is placed by hand, packaged at
400,000 gaussians and registered. It cannot say *where* this capture was placed, what took
it, or what the site should be called — those are facts about one capture, and they live
in the `captures` row.

So the worker resolves them, per run, and the child merges them over the recipe's own
parameters (:meth:`Recipe.with_params`). Two rules keep this from becoming a second,
shadow recipe format:

* **a stage is found by what it is, not by what it is called.** The coordinate goes to
  whichever stage places captures by hand; the capture's sensor and date go to whichever
  stage produces `source_meta.json`. A deployment that renames its stages, or ships a
  recipe of its own, still gets its capture's coordinate;
* **`jobs.params` wins.** It is a person asking for this run specifically, and it is
  keyed by stage id exactly as it is stored. Until now it was recorded and ignored, which
  is the worst of both: the column said the run was parameterised and nothing read it.
"""

from __future__ import annotations

from typing import Any

from app.models import Capture, Job
from app.worker.pipeline_bridge import Plan

#: The impl that turns an operator's coordinate into `georef.json`.
MANUAL_PLACEMENT = "manual_placement"
#: The impl that turns an uploaded splat's own axes into east/north/up.
INGEST_SPLAT = "ingest_splat"
#: Lane 2's georeference, which takes the capture's coordinate as its last resort.
EXIF_GPS = "exif_gps"
#: The capture-metadata keys that say how a splat file is oriented, and the parameter of
#: `ingest_splat` each becomes. Metadata is camelCase because the console writes it;
#: stage parameters are snake_case because the recipes are.
ORIENTATION_KEYS: dict[str, str] = {"upAxis": "up_axis", "headingDeg": "heading_deg"}
#: The impl that writes what the run should register.
CATALOG = "catalog"
#: The artifact whose producer describes the uploaded bytes.
SOURCE_META = "source_meta.json"


def stage_params(plan: Plan, capture: Capture, job: Job) -> dict[str, dict[str, Any]]:
    """Per-stage parameter overrides for this run, derived then overridden by `job.params`."""
    resolved: dict[str, dict[str, Any]] = {}
    placement = _placement(capture)
    facts = _capture_facts(capture)
    orientation = _orientation(capture)
    for stage in plan.stages:
        if stage.impl.name == MANUAL_PLACEMENT and placement:
            resolved.setdefault(stage.id, {}).update(placement)
        if stage.impl.name == INGEST_SPLAT and orientation:
            resolved.setdefault(stage.id, {}).update(orientation)
        if stage.impl.name == EXIF_GPS:
            # Lane 2's fallback, not its answer: the frames' own GPS and the video's own
            # location both win over it inside the stage. Without it a video that
            # carried no location has nowhere to go -- the stage refuses rather than
            # landing it at (0, 0), and a phone recording with location services off is
            # the ordinary way to get there.
            fallback = {k: v for k, v in placement.items() if k in ("lat", "lon", "height")}
            heading = orientation.get(ORIENTATION_KEYS["headingDeg"])
            if heading is not None:
                fallback["heading_deg"] = heading
            if fallback:
                resolved.setdefault(stage.id, {}).update(fallback)
        if SOURCE_META in stage.impl.produced_names and facts:
            resolved.setdefault(stage.id, {}).update(facts)
        if stage.impl.name == CATALOG:
            # Without this the slug is the run id, so a capture's site is named after the
            # uuid of the job that happened to produce it.
            resolved.setdefault(stage.id, {}).update({"slug": capture.slug, "title": capture.name})
    for stage_id, overrides in _requested(job).items():
        resolved.setdefault(stage_id, {}).update(overrides)
    return resolved


def _requested(job: Job) -> dict[str, dict[str, Any]]:
    """`jobs.params`, checked for shape.

    A malformed `params` fails the job with a message rather than being dropped: a caller
    who sent `{"lat": 51.5}` meaning `{"georeference": {"lat": 51.5}}` has to be told.
    """
    params = job.params or {}
    if not isinstance(params, dict):
        raise ValueError(f"job {job.id}: `params` must be an object keyed by stage id")
    resolved: dict[str, dict[str, Any]] = {}
    for stage_id, overrides in params.items():
        if not isinstance(overrides, dict):
            raise ValueError(
                f"job {job.id}: `params[{stage_id!r}]` must be an object of that stage's "
                f"parameters, for example {{'georeference': {{'lat': 51.5, 'lon': -0.12}}}}"
            )
        resolved[str(stage_id)] = dict(overrides)
    return resolved


def _placement(capture: Capture) -> dict[str, Any]:
    """Where this capture was placed, out of its metadata.

    Lane 1 has no EXIF and no poses — a Scaniverse `.ply` is geometry — so the coordinate
    can only come from whoever dropped it. A capture with no coordinate is left to the
    recipe's default rather than being given one that was made up here.
    """
    metadata = capture.metadata_ or {}
    lat, lon = _number(metadata.get("lat")), _number(metadata.get("lon"))
    if lat is None or lon is None:
        return {}
    placement: dict[str, Any] = {"lat": lat, "lon": lon}
    height = _number(metadata.get("height"))
    if height is not None:
        placement["height"] = height
    uncertainty = _number(metadata.get("uncertaintyM"))
    if uncertainty is not None:
        placement["uncertainty_m"] = uncertainty
    return placement


def _orientation(capture: Capture) -> dict[str, Any]:
    """Which way the uploaded splat's own axes point, when somebody said.

    Unsaid, `ingest_splat` applies the format's evidence-based default (`.spz` y up,
    `.ply` y down -- `tools/pipeline/gaussians.py` records why). Said, it is passed
    through as given, and a value the stage does not know fails the run with the list of
    ones it does rather than being dropped here: an `upAxis` that silently went nowhere is
    a capture that lands on its side with nobody told why.
    """
    metadata = capture.metadata_ or {}
    resolved: dict[str, Any] = {}
    axis = metadata.get("upAxis")
    if isinstance(axis, str) and axis.strip():
        resolved[ORIENTATION_KEYS["upAxis"]] = axis.strip()
    heading = _number(metadata.get("headingDeg"))
    if heading is not None:
        resolved[ORIENTATION_KEYS["headingDeg"]] = heading
    return resolved


def _capture_facts(capture: Capture) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    if capture.sensor:
        facts["sensor"] = capture.sensor
    if capture.device:
        facts["device"] = capture.device
    if capture.captured_at is not None:
        facts["captured_at"] = capture.captured_at.isoformat()
    return facts


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None
