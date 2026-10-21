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
#: Lane 2's pose and alignment, which express GNSS priors, check targets and the
#: reference epoch about the same origin `exif_gps` uses (M3b): with a site frame, its.
COLMAP = "colmap"
ALIGN_EPOCH = "align_epoch"
#: The capture-metadata keys that say how a splat file is oriented, and the parameter of
#: `ingest_splat` each becomes. Metadata is camelCase because the console writes it;
#: stage parameters are snake_case because the recipes are.
ORIENTATION_KEYS: dict[str, str] = {"upAxis": "up_axis", "headingDeg": "heading_deg"}
#: The impl that writes what the run should register.
CATALOG = "catalog"
#: The artifact whose producer describes the uploaded bytes.
SOURCE_META = "source_meta.json"
#: What `api/v1/phone.py` stamps a phone-key capture's `metadata.origin` with
#: (`phone.ORIGIN`), restated because the worker must not import `app.api`;
#: `tests/test_worker_params.py` holds the two equal.
PHONE_ORIGIN = "phone-key"
#: What a phone-key capture tells `normalize` it is (`capture_platform.PHONE_HINT`): the
#: last thing platform detection falls back on when the frames say nothing.
PHONE_HINT = "phone"
#: Lane 2's frame extractor: the stage that reads the clip's container, and records its
#: location in `source_meta.json` for `exif_gps` to place by.
FFMPEG_FRAMES = "ffmpeg_frames"
#: The capture metadata the worker records its reading of the video's location in
#: (`worker/video_gps.py`, `services/epochs.VIDEO_LOCATION`), and the `ffmpeg_frames`
#: parameter it is handed on as (`tools/pipeline/stages.VIDEO_LOCATION_PARAM`): the one
#: reading that both chose the capture's site and places it (the cross review of #72).
VIDEO_LOCATION = "videoLocation"
VIDEO_LOCATION_PARAM = "video_location"
#: The parameter that tells a placing stage which site's frame to place the capture in
#: (`{siteId, lat, lon, height, heightDatum, epsg}`, `services/epochs.RunFrame.param`;
#: decision D3).
SITE_FRAME = "site_frame"


def stage_params(
    plan: Plan, capture: Capture, job: Job, frame: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Per-stage parameter overrides for this run, derived then overridden by `job.params`.

    `frame` is the site frame the run is placed in (`services/epochs.frame_for_run`), or
    None for a run that defines its own. It goes to the stages that write the placement
    -- `manual_placement`, `ingest_splat`, `exif_gps` -- and to the two that must agree
    with it about the origin before `georef.json` exists or about its datum: `colmap`
    (GNSS pose priors about the frame's origin, not the fixes' median) and
    `align_epoch` (the frame's height datum, for check targets). Everything else reads
    the origin from `georef.json`. Without it the output is what it always was.
    """
    resolved: dict[str, dict[str, Any]] = {}
    requested = _requested(job)
    placement = _placement(capture)
    facts = _capture_facts(capture)
    orientation = _orientation(capture)
    point = located_point(plan, capture, job)
    located = {} if point is None else {"lat": point[0], "lon": point[1]}
    for stage in plan.stages:
        if stage.impl.name == MANUAL_PLACEMENT and placement:
            resolved.setdefault(stage.id, {}).update(placement)
        if stage.impl.name == MANUAL_PLACEMENT and frame is not None:
            # A capture with no coordinate of its own sits at its site's origin: the
            # recipe's default of (0, 0) would be an offset of thousands of kilometres.
            here = located or {"lat": frame["lat"], "lon": frame["lon"]}
            fallback = {} if placement else {"height": frame["height"]}
            resolved.setdefault(stage.id, {}).update({SITE_FRAME: frame, **here, **fallback})
        if stage.impl.name == INGEST_SPLAT and orientation:
            resolved.setdefault(stage.id, {}).update(orientation)
        if stage.impl.name == INGEST_SPLAT and frame is not None:
            # `normalize` moves the splat by where the capture is from the frame's origin,
            # and must be told the very point `georeference` will be (`located_point`):
            # `manual_placement` refuses a frame offset `normalize` did not apply.
            here = located or {"lat": frame["lat"], "lon": frame["lon"]}
            resolved.setdefault(stage.id, {}).update({SITE_FRAME: frame, **here})
        if stage.impl.name in (COLMAP, ALIGN_EPOCH) and frame is not None:
            resolved.setdefault(stage.id, {})[SITE_FRAME] = frame
        if stage.impl.name == EXIF_GPS and frame is not None:
            # The stage falls back to the frame's origin itself, after the frames' own GPS,
            # the video's location and the capture's coordinate.
            resolved.setdefault(stage.id, {})[SITE_FRAME] = frame
        if stage.impl.name == EXIF_GPS:
            # Lane 2's fallback, not its answer: the frames' own GPS and the video's own
            # location both win over it inside the stage. Without it a video that
            # carried no location has nowhere to go -- the stage refuses rather than
            # landing it at (0, 0), and a phone recording with location services off is
            # the ordinary way to get there.
            fallback = {
                k: v
                for k, v in placement.items()
                if k in ("lat", "lon", "height", "location_accuracy_m")
            }
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
    for stage_id, overrides in requested.items():
        resolved.setdefault(stage_id, {}).update(overrides)
    if VIDEO_LOCATION in (capture.metadata_ or {}):
        for stage in plan.stages:
            if stage.impl.name == FFMPEG_FRAMES:
                # After `jobs.params`, which wins everywhere else: this reading already
                # chose the capture's site, and a run placed by another point would undo
                # the one source of truth. Present even when null: "the worker found none
                # it could vouch for" must stop the stage reading ffmpeg's instead.
                resolved.setdefault(stage.id, {})[VIDEO_LOCATION_PARAM] = capture.metadata_[
                    VIDEO_LOCATION
                ]
    return resolved


def located_point(plan: Plan, capture: Capture, job: Job) -> tuple[float, float] | None:
    """The one point a run places its capture at, before any stage runs: a hand
    placement's `lat`/`lon` in `jobs.params` (the upload page's typed place, which wins
    over the phone's own fix), else the capture's own coordinate, else None.

    One answer for every consumer that needs it before the run: the spatial join
    (`services/epochs.frame_for_run`) and, in a site frame, both `normalize`
    (`ingest_splat`, which moves the splat by its offset from the frame's origin) and
    `georeference` (`manual_placement`, which refuses a frame offset `normalize` did not
    apply). Two stages told two different points made a capture that could never run.
    """
    requested = _requested(job)
    for stage in plan.stages:
        if stage.impl.name != MANUAL_PLACEMENT:
            continue
        asked = requested.get(stage.id, {})
        lat, lon = _number(asked.get("lat")), _number(asked.get("lon"))
        if lat is not None and lon is not None:
            return lat, lon
    placement = _placement(capture)
    if "lat" in placement and "lon" in placement:
        return float(placement["lat"]), float(placement["lon"])
    return None


#: The artifact a region of interest's coordinates are in: an ROI measured by one run's
#: `quality` stage is a sphere in that run's COLMAP frame, and a pose model computed
#: again is a different, arbitrary frame.
POSES = "poses"
#: The parameters that carry one (the train stage's crop, see the phone's Refine): an ROI
#: sphere, or a support mask of any shape. Both are in the preview's frame.
ROI = "roi"
REGION_PARAMS = frozenset({ROI, "support_mask"})


def without_stale_roi(
    resolved: dict[str, dict[str, Any]], plan: Plan, completed: set[str]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The parameters with every `roi` removed if the poses will be recomputed.

    A Refine resumes a finished run at `train`, keeping its frames and poses, and crops
    to the region the preview measured in exactly those poses. If the poses are *not*
    kept -- the workdir is gone, so nothing can be skipped and the run starts over --
    the region would be applied to a new reconstruction in a different frame and crop
    away the wrong part of it. Training uncropped is then the only safe answer. Returns
    the stage ids an ROI was dropped from, so the caller can say so.
    """
    producer = plan.origins.get(POSES)
    if producer is None or producer in completed:
        return resolved, []
    dropped = sorted(
        stage for stage, values in resolved.items() if REGION_PARAMS.intersection(values)
    )
    if not dropped:
        return resolved, []
    return {
        stage: {k: v for k, v in values.items() if k not in REGION_PARAMS}
        for stage, values in resolved.items()
    }, dropped


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
    # The phone's own horizontal accuracy for its fix (`api/v1/phone.py` writes it). It
    # was recorded and never read, so a 65 m fix was placed as if it were good to 10 m;
    # the placing stages now widen the uncertainty by it.
    accuracy = _number(metadata.get("locationAccuracyM"))
    if accuracy is not None:
        placement["location_accuracy_m"] = accuracy
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
    """What the capture row says about itself, for the stage that describes the upload.

    `platform` (a `metadata.platform` someone set, checked at creation against the epoch
    vocabulary) overrides detection; `platform_hint` says a phone key uploaded it. Both
    also reach `ingest_splat`, which produces `source_meta.json` too and ignores them.
    """
    facts: dict[str, Any] = {}
    metadata = capture.metadata_ or {}
    if capture.sensor:
        facts["sensor"] = capture.sensor
    if capture.device:
        facts["device"] = capture.device
    if capture.captured_at is not None:
        facts["captured_at"] = capture.captured_at.isoformat()
    platform = metadata.get("platform")
    if isinstance(platform, str) and platform.strip():
        facts["platform"] = platform.strip()
    if metadata.get("origin") == PHONE_ORIGIN:
        facts["platform_hint"] = PHONE_HINT
    return facts


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None
