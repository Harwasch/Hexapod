"""`align_epoch`: how this epoch of a site was registered, and what that lets it claim.

Decision D4. Every Lane 2 epoch gets `epoch_alignment.json`: the method that put it in
its site's frame, the similarity, the residuals, its check targets, a tier, a tolerance,
a **verdict** (`survey` | `coarse` | `unaligned`), a confidence and an uncertainty, and a
label the web shows as is. The web reads only `label`, `verdict` and `uncertaintyM`.

**Methods**, in the order they are tried:

* `visual` -- the site has a reference epoch (`site_context`, seeded by the worker): this
  epoch's frames are registered into the reference's **fixed** model
  (`colmap4.py register`; the reference cannot move), the reference model is taken into
  this epoch's east/north/up by its own placement and the exact ECEF reframe between the
  two origins (`reference_frame`), and a seeded LMedS similarity (`sim3.py`) maps this
  epoch's own pose model onto those registered centres. Accepted when a majority of the
  sampled frames registered (`min_registered_fraction`, 0.5), at least `MIN_SIM3_INLIERS`
  (6: twice the 3-point minimum) agree, and they are at least `MIN_INLIER_FRACTION` (0.5:
  a fit most correspondences disagree with is not a fit). GNSS never decides it; it only
  checks it (`gnssCheck`).

  "Agree" is measured against an independent scale, not LMedS's own: LMedS sets its
  threshold from the median residual, so by construction at least half of any set are
  its inliers and a majority test on them could never fail. A correspondence agrees when
  its residual is within `AGREE_K` pixel footprints -- each of the two camera centres
  compared is taken as uncertain by one pixel's footprint at the scene per axis (`g`,
  conservative: a resection averages many points), so their 3D difference has sigma
  `sqrt(6) g`, and three of those is `AGREE_K = 3 sqrt(6)`, about 7.3 `g`. First
  principles, not calibrated. (A deviation from the design, which used LMedS's inliers.)

  The cameras' **orientations** must agree too: the median angle between each agreeing
  camera's rotation through the fit and its rotation as registered within `AGREE_K` of
  one pixel's angle (`1 / focal` radians), by the same reasoning. Centres alone leave a
  rotation about their own line free when they are (nearly) collinear -- a walk along a
  path, a single drone strip -- and a fit wrong about that line would put every camera
  centre right and the scene off to the side; the orientations see it. (Also a
  deviation: the design recorded the rotation residual but did not judge it.)
* `gnss-prior-ba` / `gnss-similarity` -- the georeference's own EXIF frame (pose priors in
  bundle adjustment, or a similarity fitted after it), when it did not recentre.
* `none` -- a located or hand-placed capture: it keeps its placement and is `unaligned`.

**Reference choice when this epoch is the better one** (the M3b review's fix 11; the
default for Harrison's open question): an `rtk-drone` epoch whose priors put it in its
GNSS frame is **not** moved onto a reference that is not itself `survey`. Aligning to a
coarser phone epoch would throw its centimetre GNSS away for the phone's metres; it keeps
`gnss-prior-ba` and the visual fit is recorded beside it (`visualCheck`) as a check.

**Verdict.** `survey` only for the survey tier (`rtk-drone`; `TIER_OF`) **with RTK evidence
in the data** -- an RTK fixed solution on at least half the frames' metadata
(`rtk_evidence`); a platform string, from a param or normalize's override, never suffices
-- aligned, with at
least three evaluated check targets -- three of them scored vertically too, on the frame's
height datum -- within the 3-sigma tolerance of their declared uncertainties
(`check_targets.survey_tolerance`). `coarse` is aligned and not survey -- a
phone epoch that aligns is "Phone video, aligned visually, ±X m", never "surveyed".
`unaligned` is everything else -- including an epoch whose site had a reference it could
not be aligned to, whatever its own GNSS (D4: an epoch that fails alignment is unaligned)
-- labelled as placed by its GPS, its GPS point or by hand, with the georeference's own
uncertainty, and its frame is the georeference's (`frameSource: georef`): it is never
shown as aligned. A registration that crashes is such a failure, never a failed run.

**Confidence is a heuristic score in [0, 1], not a probability**: registered fraction x
inlier fraction x min(1, inliers / 20) x how far the residual is inside its consistency
tolerance x how far the check targets are inside theirs. **uncertaintyM** (`uncertainty`)
starts from what anchors the frame -- for a visual alignment the reference's uncertainty
and three times the fit's RMS in quadrature, for GNSS the georeference's own (floored at
5 m by `exif_gps`) or, with three check targets, the priors' sigma -- and check targets can
only raise it. An anchor with no vertical never claims better than the georeference.

Every gate is a recipe param and **off**: no recorded measurement backs a threshold (D4).

numpy only; pycolmap runs in `colmap4.py`'s subprocess.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import capture_platform
import check_targets
import colmap4
import exif
import frame_meta
import global_sfm
import priors
import sfm
import sim3
import site_frame
from artifacts import ArtifactDecl
from contracts import MetricValue, StageContext, StageOutcome
from errors import QualityGateError, evaluate_gate
from registry import stage_impl

__all__ = [
    "CONFIDENT_INLIERS",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MIN_REGISTERED_FRACTION",
    "EPOCH_ALIGNMENT",
    "MIN_INLIER_FRACTION",
    "MIN_SIM3_INLIERS",
    "PLATFORM_WORDS",
    "SIGMA_K",
    "SITE_CONTEXT",
    "SITE_CONTEXT_FILE",
    "SURVEY_CEILING_M",
    "TIER_OF",
    "Candidate",
    "Reference",
    "SiteContext",
    "align_epoch",
    "confidence",
    "datum_problem",
    "decide",
    "empty_site_context",
    "label",
    "platform_evidence",
    "read_site_context",
    "reference_frame",
    "rtk_evidence",
    "summary_of",
    "uncertainty",
]

F64 = npt.NDArray[np.float64]

SITE_CONTEXT = "site_context"
SITE_CONTEXT_FILE = "site_context.json"
EPOCH_ALIGNMENT = ArtifactDecl(
    "epoch_alignment.json",
    content_type="application/json",
    summary="how this epoch was registered: method, reference epoch, Sim3, residuals, check "
    "targets, tier, verdict, confidence",
)

TIER_OF: dict[str, str] = {
    "rtk-drone": "survey",
    "drone": "coarse",
    "phone-video": "coarse",
    "phone-photo": "coarse",
    "unknown": "coarse",
}
PLATFORM_WORDS: dict[str, str] = {
    "rtk-drone": "RTK drone",
    "drone": "Drone",
    "phone-video": "Phone video",
    "phone-photo": "Phone photos",
    "unknown": "Capture",
}
if set(TIER_OF) != set(capture_platform.PLATFORMS):  # pragma: no cover - import-time guard
    raise RuntimeError("TIER_OF must name every platform in capture_platform.PLATFORMS")
MIN_SIM3_INLIERS = 6
MIN_INLIER_FRACTION = 0.5
DEFAULT_MIN_REGISTERED_FRACTION = 0.5
CONFIDENT_INLIERS = 20
SIGMA_K = 3.0
#: Pixel footprints within which a visual correspondence agrees with the fit (docstring).
AGREE_K = SIGMA_K * math.sqrt(6.0)
DEFAULT_MAX_FRAMES = 60
#: The GPS L1 carrier wavelength, metres: c / 1575.42 MHz.
L1_WAVELENGTH_M = 299_792_458.0 / 1_575.42e6
#: The widest claim (`uncertaintyM`) a `survey` verdict may carry: half an L1 carrier
#: cycle, 0.0951 m. An RTK *fixed* solution has resolved the carrier's whole cycles and is
#: good to a small fraction of one (1-2 cm); a float or wrongly fixed one is off by the
#: order of a cycle or more. A claim of half a cycle or wider cannot tell the two apart,
#: so it is not evidence of survey grade, whatever tolerance the targets passed
#: (first principles, ADR 0015; the R2 review of #51).
SURVEY_CEILING_M = L1_WAVELENGTH_M / 2.0
TOLERANCE_BASIS = "3-sigma of the declared input uncertainties (first principles; not calibrated)"
NOT_EVALUATED = (
    "placement is recentred on the trained splat's footprint after training; an unaligned "
    "epoch has no fixed frame to check against"
)


# --- site context --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """The earlier epoch this one aligns to, with its files under `site_context/`."""

    epoch_id: str | None
    job_id: str | None
    observed_at: str | None
    platform: str | None
    verdict: str | None
    uncertainty_m: float
    frames: Path
    poses: Path
    placement: Path
    georef: Path
    #: The datum its georef origin's height is on (`site_frame.HEIGHT_DATUMS`), when the
    #: site context states it.
    height_datum: str | None = None


@dataclass(frozen=True)
class SiteContext:
    site_id: str | None = None
    reference: Reference | None = None
    document: Mapping[str, object] = field(default_factory=dict)


def empty_site_context(site_id: str | None = None) -> dict[str, object]:
    """What the worker seeds for a run with nothing to align to (design section 4.1)."""
    return {"version": 1, "siteId": site_id, "reference": None, "targets": [], "observations": []}


def read_site_context(directory: Path, log: Callable[[str], None] = print) -> SiteContext:
    """`site_context/site_context.json`; a missing one is an empty context, and says so."""
    path = directory / SITE_CONTEXT_FILE
    if not path.is_file():
        log(f"WARNING: no {SITE_CONTEXT_FILE} in {directory.name}: nothing to align to")
        return SiteContext()
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{SITE_CONTEXT_FILE} must be an object")
    site_id = document.get("siteId")
    raw = document.get("reference")
    reference: Reference | None = None
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(f"{SITE_CONTEXT_FILE} reference must be an object or null")
        paths: dict[str, Path] = {}
        for key in ("frames", "poses", "placement", "georef"):
            relative = raw.get(key)
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{SITE_CONTEXT_FILE} reference.{key} must name a path")
            resolved = (directory / relative).resolve()
            if not resolved.is_relative_to(directory.resolve()):
                raise ValueError(f"reference.{key} must stay inside site_context/")
            if not resolved.exists():
                raise ValueError(
                    f"the reference epoch's {key} ({relative}) is not in site_context/"
                )
            paths[key] = resolved
        datum = raw.get("heightDatum")
        if datum is not None and datum not in site_frame.HEIGHT_DATUMS:
            raise ValueError(
                f"reference.heightDatum must be one of {', '.join(site_frame.HEIGHT_DATUMS)} "
                f"or null, not {datum!r}"
            )
        uncertainty_m = _float(raw.get("uncertaintyM"))
        if uncertainty_m is None or uncertainty_m <= 0:
            raise ValueError("reference.uncertaintyM must be a positive number of metres")
        reference = Reference(
            epoch_id=_str(raw.get("epochId")),
            job_id=_str(raw.get("jobId")),
            observed_at=_str(raw.get("observedAt")),
            platform=_str(raw.get("platform")),
            verdict=_str(raw.get("verdict")),
            uncertainty_m=uncertainty_m,
            height_datum=datum,
            **paths,
        )
    return SiteContext(
        site_id=site_id if isinstance(site_id, str) else None,
        reference=reference,
        document=document,
    )


# --- small helpers ------------------------------------------------------------------------------


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _similarity(block: Mapping[str, Any]) -> sfm.Similarity:
    """A `{scale, rotation, translationM | translation}` block as a similarity."""
    translation = block.get("translationM", block.get("translation"))
    return sfm.Similarity(
        scale=float(block.get("scale") or 1.0),
        rotation=np.asarray(block.get("rotation") or np.eye(3), dtype=np.float64),
        translation=np.zeros(3)
        if translation is None
        else np.asarray(translation, dtype=np.float64),
    )


def _compose(outer: sfm.Similarity, inner: sfm.Similarity) -> sfm.Similarity:
    """`outer(inner(x))`."""
    return sfm.Similarity(
        scale=outer.scale * inner.scale,
        rotation=outer.rotation @ inner.rotation,
        translation=outer.scale * (outer.rotation @ inner.translation) + outer.translation,
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"rms": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "rms": float(np.sqrt(float(np.mean(array**2)))),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(array.max()),
    }


def _rounded(stats: Mapping[str, float], digits: int = 4) -> dict[str, float]:
    return {k: round(v, digits) for k, v in stats.items()}


# --- origin, platform -------------------------------------------------------------------------


def _frame_origin(
    georef: Mapping[str, Any], frame: site_frame.SiteFrame | None
) -> tuple[float, float, float, str]:
    """The M2 adapter: this epoch's origin (georef.json's lat/lon/height -- the site
    frame's when one was given) and its height datum (the frame's, else EXIF's)."""
    lat, lon, height = (_float(georef.get(k)) for k in ("lat", "lon", "height"))
    if lat is None or lon is None:
        raise ValueError("georef.json has no lat/lon to align about")
    datum = (frame.height_datum if frame is not None else None) or "msl-unverified"
    return lat, lon, height or 0.0, datum


#: What a platform rests on (#49's `platformEvidence`, `capture_platform`).
PLATFORM_EVIDENCE = ("metadata", "claimed", "none")


def platform_evidence(block: object) -> str:
    """A platform block's `platformEvidence` (#49): `metadata` (the frames showed it),
    `claimed` (a param or the phone-key hint) or `none`; a missing or unknown value -- a
    block written before #49 -- is `none`, never assumed `metadata`."""
    found = block.get("platformEvidence") if isinstance(block, Mapping) else None
    return found if isinstance(found, str) and found in PLATFORM_EVIDENCE else "none"


def _platform(
    requested: object,
    source_meta: Mapping[str, Any] | None,
    frames_meta: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    """The M3a adapter: (platform, where it came from, what it rests on). The param, then
    normalize's own detection (`frames_meta`, then `source_meta`), then the make: Apple is
    a phone, DJI a `drone` -- never `rtk-drone` without RTK evidence, which only detection
    reads. A param is `claimed` unless the frames showed the same; detection's own block
    says what it rests on (`platform_evidence`)."""
    detected: tuple[str, str, str] | None = None
    for name, document in (("frames_meta", frames_meta), ("source_meta", source_meta)):
        block = document.get("platform") if document is not None else None
        found = block.get("value") if isinstance(block, Mapping) else block
        if isinstance(found, str) and found in capture_platform.PLATFORMS:
            detected = found, name, platform_evidence(block)
            break
    if requested is not None:
        value = str(requested)
        if value not in capture_platform.PLATFORMS:
            raise ValueError(
                f"platform={value!r} is not one of {', '.join(capture_platform.PLATFORMS)}"
            )
        # A param that only repeats what the frames showed keeps their evidence, as
        # `capture_platform.detect` does (#49); any other is a claim.
        shown = detected is not None and detected[0] == value and detected[2] == "metadata"
        return value, "param", "metadata" if shown else "claimed"
    if detected is not None:
        return detected
    if source_meta is not None:
        make = str(source_meta.get("sensor") or "").casefold()
        video = source_meta.get("format") == "video"
        if make == "apple":
            return ("phone-video" if video else "phone-photo"), "source_meta", "metadata"
        if "dji" in make:
            return "drone", "source_meta", "metadata"
    return "unknown", "default", "none"


def rtk_evidence(frames_meta: Mapping[str, Any] | None) -> str | None:
    """What in the frames' own metadata says an RTK solution was fixed, or None.

    Survey is decided on this, never on the platform string: a `platform` param, or
    normalize's own override recorded in `frames_meta`, names a tier but proves nothing
    (the R2 review of #51). Evidence is `dji.rtk.flag == 50` (fixed) on at least half the
    kept frames -- the classification rule `capture_platform` uses, not a tolerance."""
    frames = frames_meta.get("frames") if isinstance(frames_meta, Mapping) else None
    records = [r for r in frames if isinstance(r, Mapping)] if isinstance(frames, list) else []
    fixed = 0
    for record in records:
        dji = record.get("dji")
        rtk = dji.get("rtk") if isinstance(dji, Mapping) else None
        if isinstance(rtk, Mapping) and rtk.get("flag") == frame_meta.RTK_FIXED:
            fixed += 1
    if not records or 2 * fixed < len(records):
        return None
    return f"{fixed}/{len(records)} frames carry an RTK fixed solution (RtkFlag 50)"


# --- candidates ---------------------------------------------------------------------------------


@dataclass
class Candidate:
    """One way this epoch could be in its site's frame."""

    method: str
    aligned: bool
    similarity: sfm.Similarity | None
    correspondences: int = 0
    inliers: int = 0
    residuals: dict[str, float] = field(default_factory=dict)
    inlier_names: tuple[str, ...] = ()
    #: The sigma of whatever anchors this candidate, horizontal and vertical, metres.
    anchor_h: float = 0.0
    anchor_v: float = 0.0
    registered_fraction: float | None = None
    rotation_residual_deg: list[float] = field(default_factory=list)
    #: Why the anchor says nothing about height, when it does not (GNSS priors whose
    #: vertical was left unconstrained): vertical is then never scored or claimed.
    vertical_free: str | None = None
    ransac: dict[str, object] | None = None
    registration: dict[str, object] | None = None
    #: The consistency tolerance and the pixel footprint it is three of (visual).
    tol_align: float | None = None
    footprint: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def inlier_fraction(self) -> float:
        return self.inliers / self.correspondences if self.correspondences else 0.0

    def stats(self) -> dict[str, float]:
        return _stats([self.residuals[n] for n in self.inlier_names if n in self.residuals])


def gnss_candidate(georef: Mapping[str, Any], poses: Mapping[str, Any]) -> Candidate:
    """The georeference's own frame, as a candidate: aligned when it did not recentre and
    a similarity was fitted (`exif_gps`'s aligned branch)."""
    frame = georef.get("frame")
    alignment = georef.get("alignment")
    raw_prior = poses.get("priors")
    prior: dict[str, Any] = raw_prior if isinstance(raw_prior, dict) else {}
    used = bool(prior.get("used"))
    if (
        not isinstance(frame, dict)
        or frame.get("recentre") is not False
        or not isinstance(alignment, dict)
    ):
        return Candidate(method="none", aligned=False, similarity=None)
    per_image = alignment.get("residualM", {}).get("perImage") or {}
    residuals = {str(k): float(v) for k, v in per_image.items()}
    limit = _float(alignment.get("maxErrorM")) or math.inf
    inliers = tuple(sorted(name for name, value in residuals.items() if value <= limit))
    inlier_count = int(alignment.get("inliers") or len(inliers))
    if inlier_count < sim3.MIN_POINTS:
        # A similarity fewer than three fixes agree with is no frame (R2 review of #51):
        # `model_aligner`'s path records the count after the fact, and RANSAC's refuses.
        candidate = Candidate(method="gnss-similarity", aligned=False, similarity=None)
        candidate.reasons.append(
            f"gnss: {inlier_count} of {len(residuals)} fixes agree with the similarity, "
            f"fewer than {sim3.MIN_POINTS}"
        )
        return candidate
    raw_sigma = prior.get("sigmaM")
    sigma: dict[str, Any] = raw_sigma if isinstance(raw_sigma, dict) else {}
    anchor_h = _float(sigma.get("horizontalMedian")) if used else None
    anchor_v = _float(sigma.get("verticalMedian")) if used else None
    floor = _float(georef.get("uncertaintyM")) or 5.0
    return Candidate(
        method="gnss-prior-ba" if used else "gnss-similarity",
        aligned=True,
        similarity=_similarity(frame),
        correspondences=int(alignment.get("images") or len(residuals)),
        inliers=inlier_count,
        residuals=residuals,
        inlier_names=inliers,
        anchor_h=anchor_h if anchor_h is not None else floor,
        anchor_v=anchor_v if anchor_v is not None else floor,
        vertical_free=(
            str(prior.get("vertical") or "unconstrained")
            if anchor_v is not None and anchor_v >= priors.UNCONSTRAINED_STD_M
            else None
        ),
    )


def reference_frame(reference: Reference, origin: tuple[float, float, float]) -> sfm.Similarity:
    """The reference epoch's model frame into **this** epoch's east/north/up.

    Its placement (`place/placement.json`: its model to its canonical east/north/up about
    its georef origin) composed with the exact rigid reframe between the two origins:
    `x_t = R_t R_s^T x_s + R_t (ecef(o_s) - ecef(o_t))` (`exif.enu_rotation`)."""
    placement = _similarity(_read_json(reference.placement))
    georef = _read_json(reference.georef)
    source = tuple(float(georef[k] or 0.0) for k in ("lat", "lon", "height"))
    r_s = exif.enu_rotation(source[0], source[1])
    r_t = exif.enu_rotation(origin[0], origin[1])
    reframe = sfm.Similarity(
        scale=1.0,
        rotation=r_t @ r_s.T,
        translation=r_t @ (exif.geodetic_to_ecef(*source) - exif.geodetic_to_ecef(*origin)),
    )
    return _compose(reframe, placement)


def datum_problem(
    reference: Reference, origin: tuple[float, float, float], datum: str
) -> str | None:
    """Why the reference cannot be reframed into this epoch's east/north/up, or None.

    `reference_frame` moves through ECEF with both origins' numeric heights, which is
    exact only when they are on one height datum: the same point at 100 m above sea level
    and 130 m above the ellipsoid would move the epoch 30 m. No geoid model is available
    here to convert between them, so a different datum is refused. A reference that does
    not state its datum is accepted only at this epoch's own origin (the site frame's,
    shared by every epoch of the site), where no height is converted."""
    if reference.height_datum is not None:
        if reference.height_datum == datum:
            return None
        return (
            f"visual: the reference epoch's heights are on height datum "
            f"{reference.height_datum} and this epoch's on {datum}; with no geoid model to "
            f"convert them the reframe would move it by their separation, so it is refused"
        )
    georef = _read_json(reference.georef)
    theirs = tuple(_float(georef.get(k)) for k in ("lat", "lon", "height"))
    if theirs == tuple(float(v) for v in origin):
        return None
    return (
        "visual: the reference epoch does not state its height datum and its origin is not "
        "this epoch's, so its heights cannot be reframed safely; refused"
    )


def _footprint(model_dir: Path, model: sfm.Model, fit: sfm.Similarity) -> float | None:
    """One pixel's footprint at the scene, metres: the median over this epoch's cameras of
    the distance from the camera to the sparse points' centroid over its focal length."""
    points = sfm.read_points(model_dir)
    if not points.size or not model.cameras:
        return None
    centroid = fit.apply(points).mean(axis=0)
    focal = {camera.id: camera.focal_px for camera in model.cameras}
    values = [
        float(np.linalg.norm(fit.apply(image.centre.reshape(1, 3))[0] - centroid))
        / focal[image.camera_id]
        for image in model.images
        if focal.get(image.camera_id, 0.0) > 0.0
    ]
    return statistics.median(values) if values else None


def visual_candidate(
    registered: Mapping[str, Any],
    *,
    model_dir: Path,
    model: sfm.Model,
    to_target: sfm.Similarity,
    reference: Reference,
    min_registered_fraction: float,
    seed: int,
) -> Candidate:
    """`colmap4 register`'s output as a candidate: this epoch's own model fitted onto its
    frames as registered in the reference, in this epoch's east/north/up."""
    new = registered.get("registeredNew") or {}
    used = int(registered.get("newFramesUsed") or 0)
    fraction = len(new) / used if used else 0.0
    own = model.by_name()
    target = {
        name: tuple(float(v) for v in to_target.apply(np.asarray([entry["centre"]]))[0])
        for name, entry in new.items()
        if name in own
    }
    source = {name: tuple(float(v) for v in own[name].centre) for name in target}
    candidate = Candidate(
        method="visual",
        aligned=False,
        similarity=None,
        correspondences=len(target),
        registered_fraction=fraction,
        anchor_h=reference.uncertainty_m,
        anchor_v=reference.uncertainty_m,
        registration=dict(registered),
    )
    if len(target) < sim3.MIN_POINTS:
        candidate.reasons.append(
            f"visual: {len(new)} of {used} frames registered into the reference; a "
            f"similarity needs {sim3.MIN_POINTS}"
        )
        return candidate
    try:
        fit = sim3.fit_sim3(source, target, max_error_m=None, seed=seed)
    except ValueError as error:
        candidate.reasons.append(f"visual: no similarity ({error})")
        return candidate
    candidate.similarity = fit.similarity
    candidate.inliers = len(fit.inliers)
    candidate.inlier_names = fit.inliers
    candidate.residuals = dict(fit.residuals_m)
    ransac: dict[str, object] = {
        "method": fit.method,
        "seed": fit.seed,
        "iterations": fit.iterations,
        "thresholdM": round(fit.threshold_m, 6),
        "agreeM": None,
    }
    candidate.ransac = ransac
    footprint = _footprint(model_dir, model, fit.similarity)
    candidate.footprint = footprint
    candidate.tol_align = None if footprint is None else SIGMA_K * footprint
    if footprint is not None:
        # Agreement on an independent scale (module docstring): never wider than LMedS's.
        agree = min(fit.threshold_m, AGREE_K * footprint)
        candidate.inlier_names = tuple(n for n in fit.inliers if fit.residuals_m[n] <= agree)
        candidate.inliers = len(candidate.inlier_names)
        ransac["agreeM"] = round(agree, 6)
    else:
        candidate.reasons.append(
            "visual: the pose model has no sparse points to measure a pixel's footprint by; "
            "agreement is LMedS's own threshold"
        )
    for name in candidate.inlier_names:
        mine = own[name].rotation @ fit.similarity.rotation.T
        theirs = np.asarray(new[name]["rotation"], dtype=np.float64) @ to_target.rotation.T
        candidate.rotation_residual_deg.append(sfm.rotation_angle_deg(mine, theirs))
    accepted = True
    if fraction < min_registered_fraction:
        accepted = False
        candidate.reasons.append(
            f"visual: {len(new)} of {used} sampled frames registered into the reference, "
            f"under min_registered_fraction {min_registered_fraction:g}"
        )
    if candidate.inliers < MIN_SIM3_INLIERS:
        accepted = False
        candidate.reasons.append(
            f"visual: {candidate.inliers} inliers, fewer than {MIN_SIM3_INLIERS}"
        )
    if candidate.inlier_fraction < MIN_INLIER_FRACTION:
        accepted = False
        candidate.reasons.append(
            f"visual: {candidate.inlier_fraction:.2f} of correspondences agree with the fit, "
            f"under {MIN_INLIER_FRACTION:g}"
        )
    focal = statistics.median(c.focal_px for c in model.cameras) if model.cameras else 0.0
    if candidate.rotation_residual_deg and focal > 0:
        limit = math.degrees(AGREE_K / focal)
        turned = statistics.median(candidate.rotation_residual_deg)
        if turned > limit:
            accepted = False
            candidate.reasons.append(
                f"visual: the cameras' orientations disagree with the fit by {turned:.2f} deg "
                f"(median), over {limit:.2f} deg ({AGREE_K:.1f} pixels' angle): the camera "
                f"centres do not fix the rotation (nearly collinear?)"
            )
        if candidate.ransac is not None:
            candidate.ransac["agreeDeg"] = round(limit, 4)
    candidate.aligned = accepted
    return candidate


def gnss_check(
    chosen: Candidate,
    georef: Mapping[str, Any],
    gnss: Candidate,
    model: sfm.Model,
) -> dict[str, object]:
    """How far the visual alignment is from what GNSS alone said: the GNSS similarity's
    camera centres against the visual ones, or the located point against the cameras'
    centroid. A check, never an input."""
    if chosen.similarity is None or chosen.method != "visual":
        return {"available": False}
    centres = model.centres()
    visual = chosen.similarity.apply(centres)
    if gnss.similarity is not None:
        placed = gnss.similarity.apply(centres)
        gaps = np.linalg.norm(visual - placed, axis=1)
        horizontal = np.linalg.norm((visual - placed)[:, :2], axis=1)
        return {
            "available": True,
            "kind": "similarity",
            "offsetM": round(float(np.median(gaps)), 4),
            "horizontalOffsetM": round(float(np.median(horizontal)), 4),
            "rotationDeg": round(
                sfm.rotation_angle_deg(chosen.similarity.rotation, gnss.similarity.rotation), 4
            ),
            "scaleRatio": round(chosen.similarity.scale / gnss.similarity.scale, 6),
        }
    method = georef.get("georefMethod")
    echo = georef.get("siteFrame")
    offset = echo.get("offsetM") if isinstance(echo, dict) else None
    located = np.asarray(offset if isinstance(offset, list) else [0.0, 0.0, 0.0], dtype=float)
    centroid = visual.mean(axis=0)
    distance = float(np.linalg.norm(centroid[:2] - located[:2]))
    return {
        "available": True,
        "kind": "located-point" if method == "exif-gps" else "hand-placed-point",
        "offsetM": round(distance, 4),
        "horizontalOffsetM": round(distance, 4),
        "rotationDeg": None,
        "scaleRatio": None,
        "note": "the cameras' centroid as aligned, against the capture's located point "
        "(horizontal; the point carries no height worth comparing)",
    }


# --- the decision ---------------------------------------------------------------------------


def decide(
    *,
    aligned: bool,
    tier: str,
    checks: Mapping[str, Any],
    platform: str,
    visual_failed: bool = False,
    rtk_evidence: str | None = None,
    platform_evidence: str | None = None,
    uncertainty_m: float | None = None,
    method: str = "",
    reference_verdict: str | None = None,
) -> tuple[str, list[str]]:
    """The verdict, and a sentence for every rule that kept it from `survey`.

    `rtk_evidence` (`rtk_evidence()`): what in the frames' metadata shows an RTK fixed
    solution. None -- the default, so a caller that forgets it fails closed -- is never
    survey, whatever the platform string says. `platform_evidence` (#49's
    `platformEvidence`): survey ignores a platform that is not `metadata` -- shown by the
    frames themselves -- and None (missing) is `none`. `uncertainty_m`, the claim the verdict
    would carry: over `SURVEY_CEILING_M`, or None, is never survey. `method` "visual" with
    a `reference_verdict` that is not survey is never survey either: an epoch aligned into
    a reference is no better than that reference.

    `visual_failed`: the site had a reference and this epoch could not be aligned to it.
    D4: an epoch that fails alignment is `unaligned` -- kept, with its GNSS placement,
    and labelled so -- however good its own GNSS frame is; it is not `coarse`."""
    reasons: list[str] = []
    if not aligned:
        return "unaligned", ["not aligned to the site: placed by its georeference only"]
    if visual_failed:
        return "unaligned", [
            "not aligned to the site's reference epoch: placed by its own GNSS only, which "
            "D4 records as unaligned"
        ]
    if tier != "survey":
        reasons.append(
            f"{PLATFORM_WORDS.get(platform, platform)} is the coarse tier: only an RTK drone "
            f"can be surveyed"
        )
        return "coarse", reasons
    if method == "visual" and reference_verdict != "survey":
        reasons.append(
            f"aligned visually to a reference epoch that is not surveyed "
            f"({reference_verdict or 'ungraded'}): it can be no better than that reference"
        )
        return "coarse", reasons
    if rtk_evidence is None:
        reasons.append(
            f"accuracy unverified: the platform is {PLATFORM_WORDS.get(platform, platform)}, "
            f"but no RTK fixed solution is in the frames' metadata, and survey needs that "
            f"evidence in the data, not a platform name"
        )
        return "coarse", reasons
    if platform_evidence != "metadata":
        reasons.append(
            f"accuracy unverified: the {PLATFORM_WORDS.get(platform, platform)} platform rests "
            f"on {platform_evidence or 'none'}, not the frames' metadata, and survey needs "
            f"the frames themselves to show it"
        )
        return "coarse", reasons
    if checks.get("verticalAnchorFree"):
        reasons.append(
            f"accuracy unverified vertically: the GNSS priors' vertical is "
            f"{checks['verticalAnchorFree']}, so no height is surveyed"
        )
        return "coarse", reasons
    evaluated = int(checks.get("checks") or 0)
    if evaluated < check_targets.MIN_CHECK_TARGETS:
        reasons.append(
            f"accuracy unverified: {evaluated} check target(s) evaluated, fewer than "
            f"{check_targets.MIN_CHECK_TARGETS}"
        )
        return "coarse", reasons
    vertical = int(checks.get("verticalChecks") or 0)
    if vertical < check_targets.MIN_CHECK_TARGETS:
        reasons.append(
            f"accuracy unverified vertically: {vertical} check target(s) share the frame's "
            f"height datum and were scored vertically, fewer than "
            f"{check_targets.MIN_CHECK_TARGETS}"
        )
        return "coarse", reasons
    if checks.get("withinTolerance") is not True:
        rmse = checks.get("rmseM") or {}
        tolerance = checks.get("tolerance") or {}
        reasons.append(
            f"check targets out of tolerance: RMSE {rmse.get('horizontal')} m horizontal "
            f"(tolerance {tolerance.get('horizontalM')} m), {rmse.get('vertical')} m vertical "
            f"(tolerance {tolerance.get('verticalM')} m)"
        )
        return "coarse", reasons
    if uncertainty_m is None or not uncertainty_m <= SURVEY_CEILING_M:
        claim = "unknown" if uncertainty_m is None else f"±{uncertainty_m:.3g} m"
        reasons.append(
            f"claim {claim} is over the survey ceiling of ±{SURVEY_CEILING_M:.4f} m (half "
            f"an L1 carrier cycle): too wide to show an RTK fixed solution"
        )
        return "coarse", reasons
    return "survey", reasons


def uncertainty(
    *,
    verdict: str,
    method: str,
    checks: Mapping[str, Any],
    targets: tuple[check_targets.Target, ...],
    reference_m: float | None,
    rms_m: float,
    georef_m: float,
    anchor_m: float = 0.0,
    vertical_free: bool = False,
) -> float:
    """`uncertaintyM` (module docstring).

    The claim without check targets is the floor; targets can only **raise** it, never
    lower it (deviations from the design, which let targets replace it):

    * visually aligned: `sqrt(reference^2 + (3 rms)^2)`, the frame it was aligned into
      and its own fit's error; with GNSS: the anchor's sigma, the larger of horizontal and
      vertical (`anchor_m`, from the caller);
    * with three or more targets: also their RMSE and their own declared sigma (they
      cannot measure better than they were surveyed). A few targets can score below
      what anchors the frame -- 1.3 cm for an epoch aligned into a reference worth 3 cm,
      whose cameras were 1.9 cm from the truth, on one demo run;
    * an anchor with no vertical (`vertical_free`): never better than the georeference's
      own figure, because no height was measured at all;
    * unaligned: the georeference's own figure (its GPS, floored at 5 m by `exif_gps`;
      10 m for a located or hand-placed capture)."""
    if verdict == "unaligned":
        return georef_m
    visual = method == "visual" and reference_m is not None
    floor = (
        math.sqrt(reference_m**2 + (SIGMA_K * rms_m) ** 2)
        if method == "visual" and reference_m is not None
        else anchor_m
    )
    if int(checks.get("evaluated") or 0) >= check_targets.MIN_CHECK_TARGETS:
        rmse = checks.get("rmseM") or {}
        labels = {e.get("label") for e in checks.get("perTarget") or [] if "errorM" in e}
        sigmas = [
            math.sqrt(2.0 * t.sigma_h_m**2 + t.sigma_v_m**2) for t in targets if t.label in labels
        ]
        claim = max(
            floor,
            float(rmse.get("total") or 0.0),
            statistics.median(sigmas) if sigmas else 0.0,
        )
    else:
        claim = floor if visual else georef_m
    return max(claim, georef_m) if vertical_free else claim


def confidence(candidate: Candidate, *, verdict: str, checks: Mapping[str, Any]) -> float:
    """A heuristic consistency score in [0, 1] -- not a probability (module docstring)."""
    if verdict == "unaligned" or not candidate.aligned:
        return 0.0
    rms = candidate.stats()["rms"]
    c_chk = 1.0
    tolerance = checks.get("tolerance")
    rmse = checks.get("rmseM")
    if isinstance(tolerance, Mapping) and isinstance(rmse, Mapping):
        scale = math.hypot(
            float(tolerance["horizontalM"]), float(tolerance.get("verticalM") or 0.0)
        )
        c_chk = max(0.0, 1.0 - float(rmse.get("total") or 0.0) / scale) if scale > 0 else 0.0
    inliers = min(1.0, candidate.inliers / CONFIDENT_INLIERS) * candidate.inlier_fraction
    if candidate.method == "visual":
        tol = candidate.tol_align
        fit = max(0.0, 1.0 - rms / tol) if tol else 0.0
        score = (candidate.registered_fraction or 0.0) * inliers * fit * c_chk
    else:
        limit = SIGMA_K * math.sqrt(3.0) * candidate.anchor_h
        fit = max(0.0, 1.0 - rms / limit) if limit > 0 else 0.0
        score = inliers * fit * c_chk
    return round(min(1.0, max(0.0, score)), 3)


def label(*, platform: str, verdict: str, method: str, uncertainty_m: float, manual: bool) -> str:
    """The deterministic words the web shows (design section 5, Labels)."""
    words = PLATFORM_WORDS.get(platform, "Capture")
    x = f"{uncertainty_m:.2g}"
    if verdict == "survey":
        return f"{words}, surveyed, check targets ±{x} m"
    if verdict == "coarse":
        how = "aligned visually" if method == "visual" else "aligned to its GNSS"
        return f"{words}, {how}, ±{x} m"
    # Placed by its frames' GPS (a similarity), by one GPS point, or by hand.
    by = "hand" if manual else ("its GPS" if method.startswith("gnss") else "its GPS point")
    return f"{words}, not aligned: placed by {by}, ±{x} m"


# --- the stage ------------------------------------------------------------------------------


def _bool(value: object, name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true or false, not {value!r}")


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    number = _float(value)
    if number is None:
        raise ValueError(f"{name} must be a number, not {value!r}")
    return number


def _register(
    ctx: StageContext, reference: Reference, frames: Path, model_dir: Path, seed: int
) -> dict[str, Any]:
    interpreter = colmap4.python()
    if colmap4.version(interpreter) is None:
        raise ValueError(
            f"align: a reference epoch needs pycolmap=={global_sfm.PYCOLMAP_VERSION} in "
            f"{interpreter} (the Modal CPU image has it; elsewhere set "
            f"${global_sfm.PYTHON_ENV}); not falling back to GNSS, which would record a "
            f"visual alignment that never ran"
        )
    output = ctx.work_dir / "register.json"
    first_octave = ctx.param("first_octave")
    threads = ctx.param("threads")
    ctx.run(
        colmap4.register_argv(
            interpreter,
            reference_frames=reference.frames,
            reference_poses=reference.poses,
            frames=frames,
            poses=model_dir,
            work=ctx.work_dir / "register",
            output=output,
            camera_model=str(ctx.param("camera_model", "SIMPLE_RADIAL")),
            max_image_size=int(ctx.param("max_image_size", 1600)),
            max_features=int(ctx.param("max_features", 4096)),
            first_octave=None if first_octave is None else int(first_octave),
            max_frames=int(ctx.param("max_frames", DEFAULT_MAX_FRAMES)),
            random_seed=seed,
            num_threads=None if threads is None else int(threads),
        )
    )
    return _read_json(output)


def _optional_json(ctx: StageContext, name: str) -> dict[str, Any] | None:
    if not ctx.has_input(name):
        return None
    path = ctx.input(name)
    return _read_json(path) if path.is_file() else None


@stage_impl(
    "align_epoch",
    consumes=("frames", "poses", "georef.json", SITE_CONTEXT),
    optional_consumes=("source_meta.json", "frames_meta.json"),
    produces=(EPOCH_ALIGNMENT,),
    summary="register this epoch to its site's reference epoch and record the verdict",
)
def align_epoch(ctx: StageContext) -> StageOutcome:
    """Real: `epoch_alignment.json` for this epoch (the module docstring has the rules).

    Params: `platform` (else detection's), `visual` (`auto` | `off`), `max_frames` (60
    per side), `seed` (0), the extraction settings pose used (`camera_model`,
    `max_image_size`, `max_features`, `first_octave`), `threads`,
    `min_registered_fraction` (0.5), `site_frame`; gates, all off: `fail_unaligned`,
    `max_check_target_rmse_m`, `fail_check_targets_out_of_tolerance`,
    `max_alignment_rms_m`, `max_gnss_visual_offset_m`.
    """
    began = time.monotonic()
    frames = ctx.input("frames")
    model_dir = ctx.input("poses")
    georef = _read_json(ctx.input("georef.json"))
    poses_path = model_dir / "poses.json"
    poses = _read_json(poses_path) if poses_path.is_file() else {}
    context = read_site_context(ctx.input(SITE_CONTEXT), ctx.log)
    source_meta = _optional_json(ctx, "source_meta.json")
    frames_meta = _optional_json(ctx, "frames_meta.json")
    frame = site_frame.from_param(ctx.param("site_frame"))
    lat, lon, height, datum = _frame_origin(georef, frame)
    origin = (lat, lon, height)
    platform, platform_source, platform_rests_on = _platform(
        ctx.param("platform"), source_meta, frames_meta
    )
    tier = TIER_OF[platform]
    evidence = rtk_evidence(frames_meta)
    visual_mode = str(ctx.param("visual", "auto"))
    if visual_mode not in ("auto", "off"):
        raise ValueError(f"visual must be auto or off, not {visual_mode!r}")
    seed = int(ctx.param("seed", sim3.DEFAULT_SEED))
    min_registered = float(ctx.param("min_registered_fraction", DEFAULT_MIN_REGISTERED_FRACTION))
    model = sfm.read_model(model_dir)
    gnss = gnss_candidate(georef, poses)
    reasons: list[str] = list(gnss.reasons)
    reference = context.reference
    visual: Candidate | None = None
    problem = (
        datum_problem(reference, origin, datum)
        if reference is not None and visual_mode == "auto"
        else None
    )
    if problem is not None:
        visual = Candidate(method="visual", aligned=False, similarity=None)
        visual.reasons.append(problem)
    elif reference is not None and visual_mode == "auto":
        try:
            registered = _register(ctx, reference, frames, model_dir, seed)
        except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as error:
            # A registration that crashed is a failed alignment, recorded as one -- never
            # a failed run, which would lose the epoch (D4: unaligned, and labelled).
            visual = Candidate(method="visual", aligned=False, similarity=None)
            visual.reasons.append(
                f"visual: the registration into the reference failed "
                f"({type(error).__name__}: {str(error)[:200]})"
            )
        else:
            visual = visual_candidate(
                registered,
                model_dir=model_dir,
                model=model,
                to_target=reference_frame(reference, origin),
                reference=reference,
                min_registered_fraction=min_registered,
                seed=seed,
            )
    elif reference is not None:
        reasons.append("visual alignment is off (visual: off); placed by its georeference")
    chosen = gnss
    visual_check: dict[str, object] | None = None
    if visual is not None and visual.aligned:
        # Any GNSS frame of a survey-tier epoch (priors in BA or a similarity after it),
        # when it is anchored tighter than a reference that is not itself survey.
        keeps_own = (
            tier == "survey"
            and gnss.method.startswith("gnss")
            and gnss.aligned
            and reference is not None
            and reference.verdict != "survey"
            and max(gnss.anchor_h, gnss.anchor_v) < reference.uncertainty_m
        )
        if keeps_own and reference is not None:
            reasons.append(
                f"kept its own RTK frame: the reference epoch is "
                f"{reference.verdict or 'ungraded'} (±{reference.uncertainty_m:.2g} m), coarser "
                f"than this epoch's GNSS; the visual fit is recorded as a check"
            )
            visual_check = _candidate_block(visual)
        else:
            chosen = visual
    elif visual is not None:
        reasons.extend(visual.reasons)
        reasons.append(f"visual alignment not accepted; fell back to {gnss.method}")
    if not chosen.aligned and chosen.method == "none":
        reasons.append("the georeference recentred (a located or hand-placed capture)")

    targets = check_targets.load_targets(context.document)
    if chosen.aligned and chosen.similarity is not None:
        observations, problems = check_targets.load_observations(
            context.document, frames=frames, frames_meta=frames_meta, targets=targets
        )
        for problem in problems:
            ctx.log(f"check targets: {problem}")
        checks = check_targets.evaluate(
            check_targets.triangulate(model, observations),
            targets,
            to_frame=chosen.similarity,
            origin=origin,
            frame_datum=datum,
            model=model,
            anchor_h=chosen.anchor_h,
            anchor_v=chosen.anchor_v,
            vertical_anchor_free=chosen.vertical_free,
        )
        checks["problems"] = problems
    else:
        checks = {
            "status": "not-evaluated",
            "reason": NOT_EVALUATED,
            "targets": len(targets),
            "evaluated": 0,
            "checks": 0,
        }
    visual_failed = visual is not None and not visual.aligned
    rms = chosen.stats()["rms"] if chosen.aligned else 0.0
    georef_m = _float(georef.get("uncertaintyM")) or 10.0
    # The claim first (it depends only on whether the epoch is aligned): the verdict
    # may not be survey over the ceiling.
    uncertainty_m = uncertainty(
        verdict="unaligned" if not chosen.aligned or visual_failed else "aligned",
        method=chosen.method,
        checks=checks,
        targets=targets,
        reference_m=reference.uncertainty_m if reference is not None else None,
        rms_m=rms,
        georef_m=georef_m,
        # Both axes (R2 review of #51): a scalar claim is never better than either
        # anchor sigma; a vertical left free is the georeference floor's business.
        anchor_m=(
            (chosen.anchor_h if chosen.vertical_free else max(chosen.anchor_h, chosen.anchor_v))
            if chosen.aligned
            else 0.0
        ),
        vertical_free=chosen.vertical_free is not None,
    )
    verdict, why = decide(
        aligned=chosen.aligned,
        tier=tier,
        checks=checks,
        platform=platform,
        visual_failed=visual_failed,
        rtk_evidence=evidence,
        platform_evidence=platform_rests_on,
        uncertainty_m=uncertainty_m,
        method=chosen.method,
        reference_verdict=reference.verdict if reference is not None else None,
    )
    reasons.extend(why)
    score = confidence(chosen, verdict=verdict, checks=checks)
    words = label(
        platform=platform,
        verdict=verdict,
        method=chosen.method,
        uncertainty_m=uncertainty_m,
        manual=georef.get("georefMethod") == "manual",
    )
    check = gnss_check(chosen, georef, gnss, model)
    gates, failure = _gates(ctx, verdict, checks, rms if chosen.aligned else None, check)
    by_epoch = chosen.method == "visual" and chosen.aligned and chosen.similarity is not None
    frame_block: object
    if by_epoch and chosen.similarity is not None:
        frame_block = {
            "source": "visual-epoch",
            **sim3.similarity_dict(chosen.similarity),
            "recentre": False,
            "headingDeg": None,
            "headingSource": "epoch-alignment",
            "up": None,
        }
    else:
        frame_block = georef.get("frame")
    registration = (visual.registration if visual is not None else None) or None
    document: dict[str, object] = {
        "version": 1,
        "method": chosen.method,
        "referenceEpochId": reference.epoch_id if reference is not None else None,
        "referenceJobId": reference.job_id if reference is not None else None,
        "referenceHeightDatum": reference.height_datum if reference is not None else None,
        "platform": platform,
        "platformSource": platform_source,
        "platformEvidence": platform_rests_on,
        "rtkEvidence": evidence,
        "tier": tier,
        "verdict": verdict,
        "confidence": score,
        "uncertaintyM": round(uncertainty_m, 4),
        "label": words,
        "reasons": reasons,
        "origin": {"lat": lat, "lon": lon, "height": height, "heightDatum": datum},
        "sim3": None
        if chosen.similarity is None
        else {
            **sim3.similarity_dict(chosen.similarity),
            "from": "this epoch's pose model (COLMAP frame)",
            "to": "east/north/up metres about origin",
        },
        **_candidate_block(chosen),
        "registration": _registration_block(registration),
        "visualCheck": visual_check,
        "gnssCheck": check,
        "checkTargets": checks,
        "tolerance": {
            "basis": TOLERANCE_BASIS,
            "checkTargets": checks.get("tolerance"),
            "alignment": {
                "rmsM": None if chosen.tol_align is None else round(chosen.tol_align, 4),
                "pixelFootprintM": None if chosen.footprint is None else round(chosen.footprint, 4),
                "minInliers": MIN_SIM3_INLIERS,
                "minInlierFraction": MIN_INLIER_FRACTION,
                "minRegisteredFraction": min_registered,
                "minCheckTargets": check_targets.MIN_CHECK_TARGETS,
            },
        },
        "gates": gates,
        "frame": frame_block,
        "frameSource": "epoch-alignment" if by_epoch else "georef",
    }
    out = ctx.output(EPOCH_ALIGNMENT.name)
    out.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    ctx.log(f"align: {chosen.method}, {verdict} -- {words} (confidence {score})")
    for reason in reasons:
        ctx.log(f"align: {reason}")
    if failure is not None:
        raise failure
    rmse = checks.get("rmseM")
    metrics: dict[str, MetricValue] = {
        "method": chosen.method,
        "verdict": verdict,
        "confidence": score,
        "inliers": chosen.inliers,
        "rmsM": round(rms, 4),
        "checkRmseM": round(float(rmse["total"]), 4) if isinstance(rmse, dict) else -1.0,
        "registered": _registered_count(registration),
        "alignS": round(time.monotonic() - began, 1),
    }
    return StageOutcome(metrics=metrics, summary=words)


def _registered_count(registered: Mapping[str, object] | None) -> int:
    new = (registered or {}).get("registeredNew")
    return len(new) if isinstance(new, dict) else 0


def _candidate_block(candidate: Candidate) -> dict[str, object]:
    stats = candidate.stats()
    rotation = candidate.rotation_residual_deg
    return {
        "correspondences": candidate.correspondences,
        "inliers": candidate.inliers,
        "inlierFraction": round(candidate.inlier_fraction, 4),
        "ransac": candidate.ransac,
        "residualM": {
            **_rounded(stats),
            "perImage": {k: round(v, 4) for k, v in sorted(candidate.residuals.items())},
        },
        "rotationResidualDeg": None
        if not rotation
        else {"median": round(statistics.median(rotation), 4), "max": round(max(rotation), 4)},
    }


def _registration_block(registered: Mapping[str, Any] | None) -> dict[str, object] | None:
    if registered is None:
        return None
    new = registered.get("registeredNew") or {}
    used = int(registered.get("newFramesUsed") or 0)
    return {
        "pycolmap": registered.get("pycolmap"),
        "newFrames": used,
        "registered": len(new),
        "registeredFraction": round(len(new) / used, 4) if used else 0.0,
        "referenceFrames": registered.get("referenceFramesUsed"),
        "registeredReference": registered.get("registeredReference"),
        "referenceFramesUnmatched": registered.get("referenceFramesUnmatched"),
        "exifPosePriorsCleared": registered.get("exifPosePriorsCleared"),
        "crossPairs": registered.get("crossPairs"),
        "verifiedPairs": registered.get("verifiedPairs"),
        "crossInlierMatches": registered.get("crossInlierMatches"),
        "referenceMovedMaxM": registered.get("referenceMovedMaxM"),
        "seconds": registered.get("seconds"),
    }


def _gates(
    ctx: StageContext,
    verdict: str,
    checks: Mapping[str, Any],
    rms: float | None,
    gnss: Mapping[str, Any],
) -> tuple[list[dict[str, object]], QualityGateError | None]:
    """The five gates, every one off unless the run sets it (D4)."""
    rmse = checks.get("rmseM") if isinstance(checks.get("rmseM"), dict) else None
    tolerance = checks.get("tolerance") if isinstance(checks.get("tolerance"), dict) else None
    ratio: float | None = None
    if rmse is not None and tolerance is not None:
        parts = [float(rmse["horizontal"]) / float(tolerance["horizontalM"])]
        if rmse.get("vertical") is not None and tolerance.get("verticalM") is not None:
            parts.append(float(rmse["vertical"]) / float(tolerance["verticalM"]))
        ratio = max(parts)
    specs: list[tuple[str, str, float | None, float | None, str, str]] = [
        (
            "unaligned",
            "fail_unaligned",
            1.0 if verdict == "unaligned" else 0.0,
            0.5 if _bool(ctx.param("fail_unaligned"), "fail_unaligned") else None,
            "",
            "the epoch could not be aligned to its site and fail_unaligned is set",
        ),
        (
            "check_target_rmse_m",
            "max_check_target_rmse_m",
            None if rmse is None else float(rmse["total"]),
            _optional_float(ctx.param("max_check_target_rmse_m"), "max_check_target_rmse_m"),
            "m",
            "the check targets' total RMSE is over max_check_target_rmse_m",
        ),
        (
            "check_targets_tolerance_ratio",
            "fail_check_targets_out_of_tolerance",
            ratio,
            1.0
            if _bool(
                ctx.param("fail_check_targets_out_of_tolerance"),
                "fail_check_targets_out_of_tolerance",
            )
            else None,
            "",
            "the check targets' RMSE is over their 3-sigma tolerance",
        ),
        (
            "alignment_rms_m",
            "max_alignment_rms_m",
            rms,
            _optional_float(ctx.param("max_alignment_rms_m"), "max_alignment_rms_m"),
            "m",
            "the alignment's RMS residual is over max_alignment_rms_m",
        ),
        (
            "gnss_visual_offset_m",
            "max_gnss_visual_offset_m",
            _float(gnss.get("offsetM")) if gnss.get("available") else None,
            _optional_float(ctx.param("max_gnss_visual_offset_m"), "max_gnss_visual_offset_m"),
            "m",
            "the visual alignment is further from the GNSS placement than max_gnss_visual_offset_m",
        ),
    ]
    records: list[dict[str, object]] = []
    first: QualityGateError | None = None
    for name, param, measured, limit, unit, reason in specs:
        record, error = evaluate_gate(
            name, measured, limit, fail_when="above", unit=unit, reason=reason
        )
        record["param"] = param
        records.append(record)
        first = first or error
    return records, first


def summary_of(document: Mapping[str, Any]) -> dict[str, Any]:
    """`epoch_alignment.json` less its per-image residuals: what the catalog carries."""
    out = dict(document)
    residual = out.get("residualM")
    if isinstance(residual, dict):
        out["residualM"] = {k: v for k, v in residual.items() if k != "perImage"}
    return out
