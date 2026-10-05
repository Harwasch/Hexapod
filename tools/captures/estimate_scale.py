"""Estimate an existing phone scan's real-world size, and set it on its asset.

A phone video registered before the pipeline estimated scales has ``scaleSource:
unresolved``: COLMAP normalised its model to about ten units across, and the globe draws one
unit to the metre, two to eight times too big. New runs estimate the scale from how high a
handheld phone is (``tools/pipeline/scale_estimate.py``, ``exif_gps``); this does the same
for a run that already happened, from the pose model it left, and gives the answer to the
API's runtime scale (``PUT /api/v1/assets/{id}/scale``), which resizes the scan where it is
drawn without touching its files.

The estimate is the pipeline's own, in the pipeline's own frame -- not a copy of either:
``sfm.read_model`` and ``sfm.read_points`` read COLMAP's ``cameras.bin``, ``images.bin`` and
``points3D.bin``; ``sfm.camera_up`` and ``sfm.rotation_onto_z`` level the model by how the
phone was held, exactly as ``stages._frame_from_poses`` levels it (the heading turn it adds
is about +z and moves no height); ``scale_estimate.estimate_metres_per_unit`` measures each
camera over the reconstruction's own ground. The two modules are numpy and nothing else, so
they are imported from ``tools/pipeline`` by path, as ``captures_bridge`` imports this
directory from there.

Where the pose model is: the ``pose`` stage's ``poses`` artifact of the run the site shows,
``runs/<job id>/pose/poses/`` in the **private** bucket, which needs that bucket's
credentials (the API's ``GET /api/v1/artifacts?jobId=<job id>&kind=poses`` names the key and
needs none). ``--capture`` looks the rest up over the API's public reads: the capture's site,
its splat asset, the job whose tiles it shows, and the scale the run registered the model
at -- an estimate is metres per *COLMAP* unit, and the asset's scale is relative to the
model as registered, so the factor sent is the estimate over the registered frame's scale.

Usage::

    aws s3 cp --recursive s3://<private bucket>/runs/<job id>/pose/poses work/poses \\
        --endpoint-url <the bucket's endpoint>
    uv run python estimate_scale.py work/poses --capture <capture id>        # a dry run
    uv run python estimate_scale.py work/poses --capture <capture id> --apply \\
        --api-url http://localhost:8000 --token "$API_WRITE_TOKEN"

A dry run (the default) prints the estimate, its evidence and the request it would send.
``--apply`` sends it, and needs ``--api-url`` and ``--token`` given explicitly: the reads
default to the production API, the write never does. It refuses a scan whose scale was
measured (``arkit``, ``exif-gps``, ``manual``) or already estimated, and one somebody has
resized by hand since, unless ``--force``; a capture whose pose model shows too little
ground under its cameras gets no estimate at all, which is the honest answer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"


def _ensure_pipeline_importable() -> None:
    if not PIPELINE_DIR.is_dir():
        raise ModuleNotFoundError(
            f"tools/pipeline is not next to tools/captures (looked in {PIPELINE_DIR})"
        )
    # Appended, not prepended: nothing here may shadow a module of this directory.
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.append(str(PIPELINE_DIR))


_ensure_pipeline_importable()

import scale_estimate
import sfm

#: The API whose public reads `--capture` looks a scan up in, when `--api-url` is not given.
DEFAULT_API = "https://twin-api.fly.dev"
#: Scale sources that already give metres, which an estimate would only make worse
#: (apps/api app/models/enums.py `ScaleSource`).
MEASURED_SCALE_SOURCES = ("arkit", "exif-gps", "manual")
#: `RenderConfig.scale`'s bounds (apps/api app/schemas/asset.py).
MIN_SCALE, MAX_SCALE = 0.01, 100.0
#: How far, in metres at the estimate, the ground the cameras were measured against may be
#: above the lowest ground under them before the dry run says so (`GroundCheck`).
GROUND_GAP_WARN_M = 0.25
#: How far the poses' camera-up may be from the one the registered run levelled by, in
#: degrees, before they are taken for another run's model.
UP_TOLERANCE_DEG = 1.0
USER_AGENT = "hexapod-estimate-scale/1"


@dataclass(frozen=True)
class Found:
    """The estimate, and the camera-up the model was levelled by (in its own frame)."""

    estimate: scale_estimate.ScaleEstimate | None
    up: tuple[float, float, float]
    #: Frames registered in the model, and how many of them agree on `up` (0..1).
    frames: int
    consistency: float
    #: The ground the cameras were measured against, against the lowest ground near them.
    ground: GroundCheck | None = None


@dataclass(frozen=True)
class GroundCheck:
    """Was the "ground" under the cameras the floor? In the levelled model's units.

    The estimate takes the low surface within reach of each camera for the ground, and says
    itself what that cannot know: a table-top orbit measured from the table top comes out
    too big by the table's height over the camera's (``scale_estimate``). This puts a number
    on it: the median surface the cameras were measured against, and the 5th percentile of
    every point under the cameras' footprint -- the lowest surface there. Far apart, the
    cameras stood over something higher than the floor, or over a slope; either way the
    estimate deserves a look before it is applied.
    """

    under_cameras_units: float
    lowest_units: float

    @property
    def gap_units(self) -> float:
        return self.under_cameras_units - self.lowest_units

    def to_dict(self, metres_per_unit: float) -> dict[str, float]:
        return {
            "underCamerasUnits": round(self.under_cameras_units, 4),
            "lowestUnits": round(self.lowest_units, 4),
            "gapUnits": round(self.gap_units, 4),
            "gapM": round(self.gap_units * metres_per_unit, 3),
        }


@dataclass(frozen=True)
class Registered:
    """What the catalog says about the scan: its splat, and the model it was registered as."""

    capture_id: str
    site_id: str
    asset_id: str
    job_id: str | None
    #: The registered provenance's scale source, and the asset's own scale source now.
    registered_scale_source: str
    scale_source: str
    #: Metres per COLMAP unit the run placed the model at (`georef.frame.scale`).
    frame_scale: float
    #: The camera-up the run levelled by, when it recorded one.
    frame_up: tuple[float, float, float] | None
    #: The runtime scale the asset is drawn at now, and how it was set.
    scale: float
    evidence_method: str | None


def estimate(model_dir: Path) -> Found:
    """The camera-height scale of a COLMAP model, levelled as the pipeline levels it."""
    model = sfm.read_model(model_dir)
    up = sfm.camera_up(model)
    if up is None:
        raise SystemExit(f"{model_dir}: the model has no registered frames to level it by")
    level = sfm.rotation_onto_z(up.up)
    cameras = model.centres() @ level.T
    points = sfm.read_points(model_dir) @ level.T
    found = scale_estimate.estimate_metres_per_unit(cameras, points)
    return Found(
        estimate=found,
        up=up.up,
        frames=up.frames,
        consistency=up.consistency,
        ground=ground_check(cameras, points, found) if found is not None else None,
    )


def ground_check(
    cameras: np.ndarray, points: np.ndarray, found: scale_estimate.ScaleEstimate
) -> GroundCheck | None:
    """`GroundCheck` with the estimate's own grid: its cell, its reach, its percentile."""
    cell = found.cell_units
    heights = scale_estimate.camera_heights(cameras, points, cell=cell, radius=2.0 * cell)
    measured = np.isfinite(heights)
    if not bool(measured.any()):
        return None
    under = float(np.median(cameras[measured, 2] - heights[measured]))
    reach = 3.0 * cell
    low = cameras[:, :2].min(axis=0) - reach
    high = cameras[:, :2].max(axis=0) + reach
    near = points[np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)]
    if near.shape[0] == 0:
        return None
    lowest = float(np.percentile(near[:, 2], scale_estimate.GROUND_PERCENTILE))
    return GroundCheck(under_cameras_units=under, lowest_units=lowest)


def lookup(api: str, capture_id: str) -> Registered:
    """The capture's site, splat asset and registered frame, over the API's public reads."""
    base = api.rstrip("/")
    capture = _get_json(f"{base}/api/v1/captures/{urllib.parse.quote(capture_id)}")
    site_id = capture.get("siteId")
    if not site_id:
        raise SystemExit(f"capture {capture_id} has no site: has it finished a run?")
    site = _get_json(f"{base}/api/v1/sites/{urllib.parse.quote(str(site_id))}")
    metadata = site.get("metadata") or {}
    georef = (metadata.get("registration") or {}).get("georef") or {}
    frame = georef.get("frame") or {}
    splats = [a for a in site.get("assets") or [] if a.get("representation") == "gaussian-splat"]
    if not splats:
        raise SystemExit(f"site {site_id} has no gaussian-splat asset")
    # The splat a run registered and repoints is the site's first by creation.
    splat = min(splats, key=lambda a: (str(a.get("createdAt")), str(a.get("id"))))
    render = splat.get("renderConfig") or {}
    evidence = render.get("scaleEvidence") or {}
    provenance = splat.get("provenance") or {}
    registered_source = evidence.get("registeredScaleSource") or georef.get("scaleSource")
    up = (frame.get("up") or {}).get("up")
    return Registered(
        capture_id=capture_id,
        site_id=str(site_id),
        asset_id=str(splat["id"]),
        job_id=str(metadata["jobId"]) if metadata.get("jobId") else None,
        registered_scale_source=str(registered_source or "unresolved"),
        scale_source=str(provenance.get("scaleSource") or "unresolved"),
        frame_scale=_positive(frame.get("scale")) or 1.0,
        frame_up=_vector(up),
        scale=_positive(render.get("scale")) or 1.0,
        evidence_method=str(evidence["method"]) if evidence.get("method") else None,
    )


def request_body(found: scale_estimate.ScaleEstimate, frame_scale: float = 1.0) -> dict[str, Any]:
    """`PUT /assets/{id}/scale`'s body for an estimate, relative to the registered model.

    The estimate is metres per COLMAP unit; the model was registered at `frame_scale` of
    them (1 for an unresolved scale), so it is drawn at `estimate / frame_scale` of itself.
    """
    evidence = found.to_dict()
    note = str(evidence["note"])
    if frame_scale != 1.0:
        note += f"; relative to the model as registered at {frame_scale:g} m per unit"
    return {
        "scale": found.scale / frame_scale,
        "evidence": {
            "method": "camera-height-estimate",
            "cameraHeightM": found.prior_m,
            "cameraHeightUnits": found.camera_height_units,
            "metresPerUnit": found.scale,
            "cameras": found.n_cameras,
            "uncertaintyPct": round(found.uncertainty_pct, 1),
            "note": note[:1000],
        },
    }


def refusal(registered: Registered, body: dict[str, Any]) -> str | None:
    """Why the estimate must not be applied to this scan, or None when it may."""
    source = registered.registered_scale_source
    if source in MEASURED_SCALE_SOURCES:
        return f"its scale was measured when it was registered ({source}); an estimate is worse"
    if source == "camera-height-estimate":
        return "the run that registered it already estimated its scale from the camera height"
    if registered.evidence_method not in (None, "camera-height-estimate"):
        return (
            f"somebody set its scale by hand since ({registered.evidence_method}, now "
            f"{registered.scale:g}); a measurement beats an estimate"
        )
    scale = float(body["scale"])
    if not MIN_SCALE <= scale <= MAX_SCALE:
        return f"a scale of {scale:g} is outside the API's {MIN_SCALE:g}-{MAX_SCALE:g}"
    return None


def up_disagreement_deg(found: Found, registered: Registered) -> float | None:
    """Degrees between the poses' camera-up and the one the registered run levelled by."""
    if registered.frame_up is None:
        return None
    a, b = np.asarray(found.up), np.asarray(registered.frame_up)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def put_scale(api: str, token: str, asset_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """`PUT /api/v1/assets/{asset_id}/scale` with the write token; the asset read back."""
    url = f"{api.rstrip('/')}/api/v1/assets/{urllib.parse.quote(asset_id)}/scale"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            answer = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"PUT {url}: {error.code} {detail}") from error
    return answer if isinstance(answer, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "poses", type=Path, help="a COLMAP sparse model directory (the run's poses)"
    )
    parser.add_argument("--capture", help="the capture id, to look up its asset and frame")
    parser.add_argument(
        "--api-url", help=f"the API (reads default to {DEFAULT_API}; --apply needs it given)"
    )
    parser.add_argument("--token", help="the API's write token, for --apply")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", dest="apply", action="store_false", help="print only (the default)"
    )
    mode.add_argument("--apply", dest="apply", action="store_true", help="PUT the estimate")
    # `--dry-run`'s store_false would otherwise make applying the default.
    parser.set_defaults(apply=False)
    parser.add_argument(
        "--force", action="store_true", help="apply even over a measured or hand-set scale"
    )
    args = parser.parse_args(argv)
    if args.apply and (not args.api_url or not args.token or not args.capture):
        parser.error("--apply needs --capture, --api-url and --token, each given explicitly")

    found = estimate(args.poses)
    registered = lookup(args.api_url or DEFAULT_API, args.capture) if args.capture else None
    report: dict[str, Any] = {
        "poses": str(args.poses),
        "camerasRegistered": found.frames,
        "upConsistency": round(found.consistency, 4),
        "estimate": found.estimate.to_dict() if found.estimate is not None else None,
    }
    if found.estimate is not None and found.ground is not None:
        report["groundCheck"] = found.ground.to_dict(found.estimate.scale)
    body = None
    if found.estimate is not None:
        frame_scale = registered.frame_scale if registered is not None else 1.0
        body = request_body(found.estimate, frame_scale)
        report["request"] = body
    if registered is not None:
        report["capture"] = {
            "id": registered.capture_id,
            "siteId": registered.site_id,
            "assetId": registered.asset_id,
            "jobId": registered.job_id,
            "registeredScaleSource": registered.registered_scale_source,
            "frameScale": registered.frame_scale,
            "scaleNow": registered.scale,
            "scaleSourceNow": registered.scale_source,
        }
        disagreement = up_disagreement_deg(found, registered)
        if disagreement is not None:
            report["upDisagreementDeg"] = round(disagreement, 3)
    print(json.dumps(report, indent=2))

    if body is None:
        print(
            "no estimate: too few cameras with the reconstruction's ground under them, "
            "too many under the surface found, or heights that disagree",
            file=sys.stderr,
        )
        return 1
    gap_m = report.get("groundCheck", {}).get("gapM", 0.0)
    if gap_m > GROUND_GAP_WARN_M:
        print(
            f"check: the cameras were measured against a surface {gap_m:.2f} m (at this "
            f"estimate) above the lowest ground under them -- a table top or a step read as "
            f"the floor makes the estimate too big by that height over the camera's",
            file=sys.stderr,
        )
    if registered is None:
        print("dry run: no --capture, so nothing to apply it to", file=sys.stderr)
        return 0
    why = refusal(registered, body)
    disagreement = report.get("upDisagreementDeg")
    if why is None and disagreement is not None and disagreement > UP_TOLERANCE_DEG:
        why = (
            f"these poses' camera-up is {disagreement:.2f} deg from the one the registered "
            f"run levelled by: they are not that run's model"
        )
    if why is not None:
        print(f"not applicable to asset {registered.asset_id}: {why}", file=sys.stderr)
        if not args.force:
            return 1 if args.apply else 0
    target = f"PUT {args.api_url or DEFAULT_API}/api/v1/assets/{registered.asset_id}/scale"
    if not args.apply:
        print(f"dry run: would {target} with the request above", file=sys.stderr)
        return 0
    answer = put_scale(args.api_url, args.token, registered.asset_id, body)
    render = answer.get("renderConfig") or {}
    print(f"{target}: scale {render.get('scale')}", file=sys.stderr)
    return 0


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read().decode("utf-8"))
    if not isinstance(answer, dict):
        raise SystemExit(f"{url} did not return an object")
    return answer


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _vector(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    numbers = [float(v) for v in value if isinstance(v, int | float) and not isinstance(v, bool)]
    return (numbers[0], numbers[1], numbers[2]) if len(numbers) == 3 else None


if __name__ == "__main__":
    raise SystemExit(main())
