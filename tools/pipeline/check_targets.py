"""Check targets: surveyed ground marks, triangulated from the posed cameras and scored.

A check target is a mark on the ground whose position was surveyed independently of the
capture (an RTK rover, a total station) and that a person -- or, later, a detector (M3c)
-- clicked in some of the capture's frames. Triangulating it from the posed cameras and
moving it by the epoch's placement into east/north/up, then comparing with its survey,
measures the placement's **accuracy** rather than its consistency: a bias common to every
GNSS fix moves the whole model, every alignment residual stays small, and only an
independent mark sees it (LIVING_WORLD.md section 6 reserves 2-3 checkpoints a site).

What is computed, all of it closed-form or a few Gauss-Newton steps, numpy only:

* **observations** come in a frame's pixels (`px` with the `imageSize` they were clicked
  at, COLMAP's continuous convention: the top-left corner is (0, 0), so a click on pixel
  (i, j) is (i + 0.5, j + 0.5)) or as fractions of it (`uv`), on an original upload name
  (mapped to the kept frame through `frames_meta.json`'s `original`) or a kept frame name;
  they are scaled to the kept frame, whose size is read from the frame itself;
* **undistortion**: SIMPLE_RADIAL's `x_d = x (1 + k r^2)` is inverted by ten fixed-point
  steps `x = x_d / (1 + k r^2)`; PINHOLE and SIMPLE_PINHOLE need none; any other camera
  model is `unsupported-camera` rather than triangulated wrongly;
* **triangulation**: the multi-view DLT (Hartley & Zisserman, 12.2) by SVD on the
  undistorted normalised coordinates, then five Gauss-Newton steps on the reprojection
  error in undistorted pixels;
* refused, never scored: fewer than `MIN_VIEWS` views (`too-few-views`), or a largest
  pairwise ray angle under `MIN_TRIANGULATION_ANGLE_DEG` (`too-narrow`; COLMAP's own
  `filter_min_tri_angle`, 1.5 deg), or a point behind a camera that saw it.

**The tolerance** a set of targets is judged against (`survey_tolerance`) is three sigma of
the *declared* uncertainties -- first principles, not calibrated on any capture:
horizontal `3 * sqrt(2) * sqrt(a_h^2 + t_h^2 + p^2)` (the expected 2D RMSE of an isotropic
error is sigma * sqrt(2)), vertical `3 * sqrt(a_v^2 + t_v^2 + p^2)`, where `a` is the
anchor's sigma (the GNSS priors, or the reference epoch's uncertainty when aligned
visually), `t` the targets' declared survey sigma, and `p` one pick pixel's footprint at
the target (`PICK_PX` times the camera-to-target distance over the focal length).
A declared sigma that is wrong (an RTK float solution reported as fixed) makes this
tolerance wrong too: it can only be as honest as what it is told.

Vertical is scored only when the target's height datum is the frame's: an EXIF-defined
frame is `msl-unverified`, a target surveyed on the ellipsoid is tens of metres away from
it vertically for reasons that have nothing to do with the capture. A target on another
datum leaves only *its own* vertical unscored: every same-datum target is still scored
and must pass (`verticalChecks` counts them; a survey verdict needs `MIN_CHECK_TARGETS`).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import PIL.Image

import exif
import sfm
import site_frame

__all__ = [
    "MIN_CHECK_TARGETS",
    "MIN_TRIANGULATION_ANGLE_DEG",
    "MIN_VIEWS",
    "PICK_PX",
    "ROLES",
    "Observation",
    "Target",
    "Triangulated",
    "evaluate",
    "load_observations",
    "load_targets",
    "survey_tolerance",
    "triangulate",
]

F64 = npt.NDArray[np.float64]

MIN_VIEWS = 2
#: COLMAP's `IncrementalMapperOptions.filter_min_tri_angle` default (verified on 4.2).
MIN_TRIANGULATION_ANGLE_DEG = 1.5
#: How many evaluated check targets a `survey` verdict needs (LIVING_WORLD.md section 6).
MIN_CHECK_TARGETS = 3
#: The click's own uncertainty, in pixels of the kept frame (one pixel: first principles).
PICK_PX = 1.0
ROLES = ("check", "control")
#: How many Gauss-Newton steps refine the DLT point, and fixed-point steps undistort.
_GAUSS_NEWTON_STEPS = 5
_UNDISTORT_STEPS = 10
_NO_DISTORTION = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})
_SUPPORTED = _NO_DISTORTION | {"SIMPLE_RADIAL"}


@dataclass(frozen=True)
class Target:
    label: str
    role: str
    lat: float
    lon: float
    height: float
    height_datum: str
    sigma_h_m: float
    sigma_v_m: float


@dataclass(frozen=True)
class Observation:
    """One click: `target` seen in kept frame `frame` at (u, v), kept-frame pixels."""

    target: str
    frame: str
    u: float
    v: float


@dataclass(frozen=True)
class Triangulated:
    target: str
    #: In the pose model's own frame; None unless `status == "ok"`.
    position: tuple[float, float, float] | None
    views: int
    images: tuple[str, ...]
    angle_deg: float
    reprojection_rms_px: float | None
    #: `ok`, `too-few-views`, `too-narrow`, `unsupported-camera` or `behind-camera`.
    status: str


def _number(entry: Mapping[str, object], key: str, label: str) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"check target {label!r}: {key} must be a finite number, not {value!r}")
    return float(value)


def load_targets(document: Mapping[str, object]) -> tuple[Target, ...]:
    """`site_context.json`'s `targets` (design section 4.1), checked; a malformed one is
    refused by label rather than scored."""
    raw = document.get("targets") or []
    if not isinstance(raw, list):
        raise ValueError("site_context targets must be a list")
    out: list[Target] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"a check target must be an object, not {entry!r}")
        label = entry.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"a check target needs a label, not {label!r}")
        if label in seen:
            raise ValueError(f"check target {label!r} appears twice")
        seen.add(label)
        role = entry.get("role", "check")
        if role not in ROLES:
            raise ValueError(f"check target {label!r}: role must be check or control")
        datum = entry.get("heightDatum")
        if datum not in site_frame.HEIGHT_DATUMS:
            raise ValueError(
                f"check target {label!r}: heightDatum must be one of "
                f"{', '.join(site_frame.HEIGHT_DATUMS)}, not {datum!r}"
            )
        sigma_h, sigma_v = _number(entry, "sigmaHM", label), _number(entry, "sigmaVM", label)
        if sigma_h <= 0 or sigma_v <= 0:
            raise ValueError(f"check target {label!r}: sigmas must be positive")
        out.append(
            Target(
                label=label,
                role=str(role),
                lat=_number(entry, "lat", label),
                lon=_number(entry, "lon", label),
                height=_number(entry, "height", label),
                height_datum=str(datum),
                sigma_h_m=sigma_h,
                sigma_v_m=sigma_v,
            )
        )
    return tuple(out)


def _records(frames_meta: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    frames = frames_meta.get("frames") if isinstance(frames_meta, Mapping) else None
    return [r for r in frames if isinstance(r, Mapping)] if isinstance(frames, list) else []


def _frame_of(
    image: str, kept: set[str], records: Sequence[Mapping[str, object]]
) -> tuple[str | None, tuple[int, int] | None, str | None]:
    """The M3a adapter: an observation's `image` as (kept frame, original size, problem).

    A kept frame name is itself. Otherwise the one record whose `original` (or
    `originalPath`) is `image` -- several (every frame cut from one video names it) is
    ambiguous, and the click must name the kept frame instead."""
    if image in kept:
        return image, None, None
    matches = [r for r in records if image in (r.get("original"), r.get("originalPath"))]
    if not matches:
        return None, None, f"image {image!r} is neither a kept frame nor an original upload"
    if len(matches) > 1:
        return (
            None,
            None,
            (
                f"image {image!r} is the original of {len(matches)} kept frames (a video); "
                f"name the kept frame instead"
            ),
        )
    name = matches[0].get("name")
    if not isinstance(name, str) or name not in kept:
        return None, None, f"image {image!r} was not kept"
    size = matches[0].get("originalSize")
    original = (
        (int(size[0]), int(size[1]))
        if isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) for v in size)
        else None
    )
    return name, original, None


def load_observations(
    document: Mapping[str, object],
    *,
    frames: Path,
    frames_meta: Mapping[str, object] | None,
    targets: Sequence[Target] = (),
) -> tuple[tuple[Observation, ...], list[str]]:
    """The clicks, in kept-frame pixels, and what could not be used (never raised: a
    stray click must not fail a capture)."""
    raw = document.get("observations") or []
    problems: list[str] = []
    if not isinstance(raw, list):
        return (), ["site_context observations is not a list"]
    kept = {p.name for p in frames.iterdir() if p.is_file()} if frames.is_dir() else set()
    labels = {t.label for t in targets}
    records = _records(frames_meta)
    sizes: dict[str, tuple[int, int] | None] = {}
    out: list[Observation] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            problems.append(f"observation {index}: not an object")
            continue
        target, image = entry.get("target"), entry.get("image")
        if not isinstance(target, str) or (labels and target not in labels):
            problems.append(f"observation {index}: unknown target {target!r}")
            continue
        if not isinstance(image, str):
            problems.append(f"observation {index}: no image")
            continue
        frame, original, problem = _frame_of(image, kept, records)
        if frame is None:
            problems.append(f"observation {index}: {problem}")
            continue
        if frame not in sizes:
            sizes[frame] = _image_size(frames / frame)
        size = sizes[frame]
        if size is None:
            problems.append(f"observation {index}: kept frame {frame!r} does not open")
            continue
        point = _kept_pixels(entry, size, original)
        if point is None:
            problems.append(f"observation {index}: needs px (with imageSize) or uv")
            continue
        out.append(Observation(target=target, frame=frame, u=point[0], v=point[1]))
    return tuple(out), problems


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with PIL.Image.open(path) as image:
            return int(image.width), int(image.height)
    except OSError:
        return None


def _pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        return None
    a, b = float(value[0]), float(value[1])
    return (a, b) if math.isfinite(a) and math.isfinite(b) else None


def _kept_pixels(
    entry: Mapping[str, object], kept: tuple[int, int], original: tuple[int, int] | None
) -> tuple[float, float] | None:
    """Continuous pixel coordinates scale exactly by size: the corner (0, 0) is fixed."""
    px, uv = _pair(entry.get("px")), _pair(entry.get("uv"))
    if px is not None:
        size = _pair(entry.get("imageSize")) or original or (float(kept[0]), float(kept[1]))
        if size[0] <= 0 or size[1] <= 0:
            return None
        return px[0] * kept[0] / size[0], px[1] * kept[1] / size[1]
    if uv is not None:
        return uv[0] * kept[0], uv[1] * kept[1]
    return None


# --- the camera model ----------------------------------------------------------------------


@dataclass(frozen=True)
class _View:
    name: str
    rotation: F64
    translation: F64
    fx: float
    fy: float
    cx: float
    cy: float
    k: float
    u: float
    v: float

    @property
    def centre(self) -> F64:
        return np.asarray(-self.rotation.T @ self.translation, dtype=np.float64)


def _intrinsics(camera: sfm.Camera) -> tuple[float, float, float, float, float] | None:
    """(fx, fy, cx, cy, k) of a supported camera model, else None."""
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2], 0.0
    if camera.model == "PINHOLE":
        return p[0], p[1], p[2], p[3], 0.0
    if camera.model == "SIMPLE_RADIAL":
        return p[0], p[0], p[1], p[2], p[3]
    return None


def _undistorted(view: _View) -> tuple[float, float]:
    """The normalised image point of the click, radial distortion undone."""
    xd, yd = (view.u - view.cx) / view.fx, (view.v - view.cy) / view.fy
    x, y = xd, yd
    if view.k != 0.0:
        for _ in range(_UNDISTORT_STEPS):
            factor = 1.0 + view.k * (x * x + y * y)
            x, y = xd / factor, yd / factor
    return x, y


def _project(view: _View, point: F64) -> tuple[float, float] | None:
    """`point` (model frame) to distorted pixels, None behind the camera."""
    camera = view.rotation @ point + view.translation
    if camera[2] <= 0.0:
        return None
    x, y = float(camera[0] / camera[2]), float(camera[1] / camera[2])
    factor = 1.0 + view.k * (x * x + y * y)
    return view.fx * x * factor + view.cx, view.fy * y * factor + view.cy


def _dlt(views: Sequence[_View]) -> F64:
    rows = []
    for view in views:
        x, y = _undistorted(view)
        projection = np.hstack([view.rotation, view.translation.reshape(3, 1)])
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vt[-1]
    return np.asarray(homogeneous[:3] / homogeneous[3], dtype=np.float64)


def _refine(views: Sequence[_View], point: F64) -> F64:
    """Gauss-Newton on the reprojection error in undistorted pixels."""
    targets = [_undistorted(view) for view in views]
    for _ in range(_GAUSS_NEWTON_STEPS):
        jacobian, residual = [], []
        for view, (x, y) in zip(views, targets, strict=True):
            camera = view.rotation @ point + view.translation
            z = float(camera[2])
            if z <= 0.0:
                return point
            px, py = float(camera[0]) / z, float(camera[1]) / z
            residual += [view.fx * (px - x), view.fy * (py - y)]
            d_px = (view.rotation[0] - px * view.rotation[2]) / z
            d_py = (view.rotation[1] - py * view.rotation[2]) / z
            jacobian += [view.fx * d_px, view.fy * d_py]
        j, r = np.asarray(jacobian), np.asarray(residual)
        step, *_ = np.linalg.lstsq(j, -r, rcond=None)
        point = point + step
    return point


def _largest_angle_deg(views: Sequence[_View], point: F64) -> float:
    rays = []
    for view in views:
        ray = point - view.centre
        norm = float(np.linalg.norm(ray))
        if norm > 0.0:
            rays.append(ray / norm)
    best = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            cosine = float(np.clip(rays[i] @ rays[j], -1.0, 1.0))
            best = max(best, math.degrees(math.acos(cosine)))
    return best


def triangulate(model: sfm.Model, observations: Sequence[Observation]) -> dict[str, Triangulated]:
    """Each target seen in the model's registered frames, triangulated in the model frame.

    A click on a frame the model did not register is not a view (it has no pose)."""
    images = model.by_name()
    cameras = {camera.id: camera for camera in model.cameras}
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.target, []).append(observation)
    out: dict[str, Triangulated] = {}
    for target in sorted(grouped):
        views: list[_View] = []
        unsupported = False
        for observation in sorted(grouped[target], key=lambda o: o.frame):
            image = images.get(observation.frame)
            if image is None:
                continue
            intrinsics = _intrinsics(cameras[image.camera_id])
            if intrinsics is None:
                unsupported = True
                continue
            fx, fy, cx, cy, k = intrinsics
            views.append(
                _View(
                    name=image.name,
                    rotation=image.rotation,
                    translation=np.asarray(image.tvec, dtype=np.float64),
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    k=k,
                    u=observation.u,
                    v=observation.v,
                )
            )
        names = tuple(view.name for view in views)
        if unsupported and len(views) < MIN_VIEWS:
            out[target] = Triangulated(
                target, None, len(views), names, 0.0, None, "unsupported-camera"
            )
            continue
        if len(views) < MIN_VIEWS:
            out[target] = Triangulated(target, None, len(views), names, 0.0, None, "too-few-views")
            continue
        point = _refine(views, _dlt(views))
        angle = _largest_angle_deg(views, point)
        projected = [_project(view, point) for view in views]
        if any(p is None for p in projected):
            out[target] = Triangulated(
                target, None, len(views), names, angle, None, "behind-camera"
            )
            continue
        errors = [
            (p[0] - view.u) ** 2 + (p[1] - view.v) ** 2
            for p, view in zip(projected, views, strict=True)
            if p is not None
        ]
        rms = math.sqrt(sum(errors) / len(errors))
        status = "ok" if angle >= MIN_TRIANGULATION_ANGLE_DEG else "too-narrow"
        position = (float(point[0]), float(point[1]), float(point[2]))
        out[target] = Triangulated(
            target, position if status == "ok" else None, len(views), names, angle, rms, status
        )
    return out


# --- scoring --------------------------------------------------------------------------------


def survey_tolerance(
    *, anchor_h: float, anchor_v: float, target_h: float, target_v: float, pick_m: float
) -> dict[str, float]:
    """Three sigma of the declared uncertainties (the module docstring has the why)."""
    return {
        "horizontalM": 3.0 * math.sqrt(2.0) * math.sqrt(anchor_h**2 + target_h**2 + pick_m**2),
        "verticalM": 3.0 * math.sqrt(anchor_v**2 + target_v**2 + pick_m**2),
    }


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def evaluate(
    triangulated: Mapping[str, Triangulated],
    targets: Sequence[Target],
    *,
    to_frame: sfm.Similarity,
    origin: tuple[float, float, float],
    frame_datum: str,
    model: sfm.Model,
    anchor_h: float,
    anchor_v: float,
    vertical_anchor_free: str | None = None,
) -> dict[str, object]:
    """The `checkTargets` block of `epoch_alignment.json` (design section 4.4).

    `vertical_anchor_free` (why, when the anchor says nothing about height -- GNSS priors
    whose vertical was unconstrained by a datum mismatch or missing altitudes): vertical
    is then not scored at all and no vertical tolerance is given. A tolerance of three
    times a 1 km "sigma" would pass any vertical error and call it surveyed.

    `to_frame` takes the pose model into east/north/up about `origin` (the epoch's
    placement); each target's survey is put in the same frame through ECEF."""
    by_label = {t.label: t for t in targets}
    images = model.by_name()
    cameras = {camera.id: camera for camera in model.cameras}
    per_target: list[dict[str, object]] = []
    horizontal: list[float] = []
    vertical: list[float] = []
    footprints: list[float] = []
    evaluated: list[Target] = []
    vertical_targets: list[Target] = []
    for label in sorted(triangulated):
        result = triangulated[label]
        target = by_label.get(label)
        if target is None:
            continue
        entry: dict[str, object] = {
            "label": label,
            "role": target.role,
            "views": result.views,
            "angleDeg": _round(result.angle_deg, 2),
            "reprojectionRmsPx": None
            if result.reprojection_rms_px is None
            else _round(result.reprojection_rms_px, 3),
            "status": result.status,
        }
        if result.status != "ok" or result.position is None:
            per_target.append(entry)
            continue
        placed = to_frame.apply(np.asarray([result.position], dtype=np.float64))[0]
        survey = exif.ecef_to_enu(
            exif.geodetic_to_ecef(target.lat, target.lon, target.height), *origin
        )
        error = placed - survey
        h = float(math.hypot(float(error[0]), float(error[1])))
        same_datum = target.height_datum == frame_datum and vertical_anchor_free is None
        footprint = _footprint(result, images, cameras, to_frame, placed)
        footprints.append(footprint)
        horizontal.append(h)
        if same_datum:
            vertical.append(abs(float(error[2])))
            vertical_targets.append(target)
        evaluated.append(target)
        entry.update(
            {
                "errorM": [_round(float(v)) for v in error],
                "horizontalM": _round(h),
                "verticalM": _round(abs(float(error[2]))) if same_datum else None,
                "pixelFootprintM": _round(footprint),
            }
        )
        per_target.append(entry)
    observed = sum(1 for t in triangulated.values() if t.target in by_label)
    block: dict[str, object] = {
        "status": "evaluated" if evaluated else ("none" if not targets else "not-evaluated"),
        "reason": None,
        "targets": len(targets),
        "observations": sum(t.views for t in triangulated.values() if t.target in by_label),
        "observedTargets": observed,
        "evaluated": len(evaluated),
        "checks": sum(1 for t in evaluated if t.role == "check"),
        "verticalEvaluated": bool(evaluated) and len(vertical) == len(evaluated),
        # Check-role targets whose vertical was scored: what a survey verdict counts.
        "verticalChecks": sum(1 for t in vertical_targets if t.role == "check"),
        "verticalAnchorFree": vertical_anchor_free,
        "perTarget": per_target,
        "rmseM": None,
        "tolerance": None,
        "withinTolerance": None,
    }
    if not evaluated:
        block["reason"] = (
            "no check targets for this site"
            if not targets
            else "no target was seen from enough views at a wide enough angle to triangulate"
        )
        return block
    rmse_h = math.sqrt(sum(v * v for v in horizontal) / len(horizontal))
    rmse_v = math.sqrt(sum(v * v for v in vertical) / len(vertical)) if vertical else None
    total = math.sqrt(rmse_h**2 + (rmse_v or 0.0) ** 2)
    tolerance = survey_tolerance(
        anchor_h=anchor_h,
        anchor_v=anchor_v,
        target_h=statistics.median(t.sigma_h_m for t in evaluated),
        target_v=statistics.median(t.sigma_v_m for t in evaluated),
        pick_m=PICK_PX * statistics.median(footprints),
    )
    # Every vertical that was scored must pass: a target on another datum leaves its
    # own height unscored, never the others' (R2 review of #51).
    within = rmse_h <= tolerance["horizontalM"] and (
        rmse_v is None or rmse_v <= tolerance["verticalM"]
    )
    block.update(
        {
            "rmseM": {
                "horizontal": _round(rmse_h),
                "vertical": None if rmse_v is None else _round(rmse_v),
                "total": _round(total),
            },
            "tolerance": {
                "horizontalM": _round(tolerance["horizontalM"]),
                "verticalM": None
                if vertical_anchor_free is not None
                else _round(tolerance["verticalM"]),
            },
            # Unrounded: the tolerance is a closed form of these, and a reader who
            # recomputes it must get the same number.
            "sigmaM": {
                "anchorH": float(anchor_h),
                "anchorV": None if vertical_anchor_free is not None else float(anchor_v),
                "targetH": float(statistics.median(t.sigma_h_m for t in evaluated)),
                "targetV": float(statistics.median(t.sigma_v_m for t in evaluated)),
                "pick": float(PICK_PX * statistics.median(footprints)),
            },
            "withinTolerance": within,
        }
    )
    if vertical_anchor_free is not None:
        block["reason"] = f"vertical not scored: the anchor's vertical is {vertical_anchor_free}"
    elif not block["verticalEvaluated"]:
        block["reason"] = (
            f"vertical scored on {len(vertical)} of {len(evaluated)} targets: the others' "
            f"height datum is not the frame's"
        )
    return block


def _footprint(
    result: Triangulated,
    images: Mapping[str, sfm.Image],
    cameras: Mapping[int, sfm.Camera],
    to_frame: sfm.Similarity,
    placed: F64,
) -> float:
    """The median, over the views, of one pixel's footprint at the target: the camera's
    distance to it in frame metres over its focal length in pixels."""
    values = []
    for name in result.images:
        image = images[name]
        centre = to_frame.apply(np.asarray([image.centre], dtype=np.float64))[0]
        focal = cameras[image.camera_id].focal_px
        if focal > 0.0:
            values.append(float(np.linalg.norm(placed - centre)) / focal)
    return statistics.median(values) if values else 0.0
