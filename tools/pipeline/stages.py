"""The stage implementations, and the artifacts each one declares.

Lane 1 is real here: `ingest_splat`, `manual_placement`, `splat_tiles`, `thumbnail`,
`ground_samples`, `manifest` and `catalog` run for real under LocalRunner, and a dropped
`.ply` or `.spz` becomes a site with no human step. Lane 2's GPU stages are still stubs,
and they say so: they raise rather than pretend, and they name the step that makes them
real. What is *not* a stub anywhere is the contract -- the artifacts each one consumes
and produces. That is what the executor validates, what StubRunner fabricates from, and
what B2 and B3 fill in behind. A recipe that runs green under StubRunner runs the same
stages in the same order against real implementations later, with no executor,
recipe-format or runner change.

Both lanes converge on `canonical.ply`: Lane 1 normalises an already-reconstructed splat
into it, Lane 2's trainer writes it (the plan calls that output "the canonical Gaussians"),
and a single `package` implementation turns either one into the same `splat/` tileset.
"""

from __future__ import annotations

import functools
import hashlib
import json
import struct
import tomllib
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import PIL

import gaussians
from artifacts import ArtifactDecl
from captures_bridge import CAPTURES_DIR, splat_tiles_convert
from contracts import MetricValue, StageContext, StageOutcome
from registry import stage_impl

# ---------------------------------------------------------------------------------------
# The artifact vocabulary. A name is also a path: `frames` is a directory of frames under
# the producing stage's out/, `georef.json` is a file.
# ---------------------------------------------------------------------------------------

SOURCE_META = ArtifactDecl(
    "source_meta.json",
    content_type="application/json",
    summary="sensor, capture date, frame count, and what the uploaded bytes turned out to be",
)
FRAMES = ArtifactDecl(
    "frames",
    kind="dir",
    content_type="inode/directory",
    summary="the selected frames, top-K by variance-of-Laplacian (never a blur threshold)",
    stub_members=("frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg"),
)
POSES = ArtifactDecl(
    "poses",
    kind="dir",
    content_type="inode/directory",
    summary="camera intrinsics and extrinsics for the frames",
    stub_members=("cameras.bin", "images.bin", "points3d.bin"),
)
MASKS = ArtifactDecl(
    "masks",
    kind="dir",
    content_type="inode/directory",
    summary="per-frame transient masks, one per frame",
    stub_members=("mask_0000.png", "mask_0001.png"),
)
CANONICAL_PLY = ArtifactDecl(
    "canonical.ply",
    content_type="application/octet-stream",
    summary="the static splat in a local frame: Lane 1 normalises it, Lane 2 trains it",
    stub_bytes=1024,
)
TRAIN_METRICS = ArtifactDecl(
    "train_metrics.json",
    content_type="application/json",
    summary="iterations, PSNR, gaussian count -- what B3's three-way comparison reads",
)
SCALE = ArtifactDecl(
    "scale.json",
    content_type="application/json",
    summary="metric scale and its source, when the capture carries one",
)
GEOREF = ArtifactDecl(
    "georef.json",
    content_type="application/json",
    summary="lat/lon/height of the local frame, plus georefMethod, scaleSource, uncertaintyM",
)
SPLAT_TILES = ArtifactDecl(
    "splat",
    kind="dir",
    content_type="inode/directory",
    summary="the 3D Tiles tileset the console renders",
    # Pinned to what tools/captures/splat_tiles.convert() actually writes. A stubbed
    # package stage and the real one produce the same file names or a test fails.
    required_members=("tileset.json", "splat.glb"),
    stub_members=("tileset.json", "splat.glb"),
)
THUMBNAIL = ArtifactDecl(
    "thumbnail.jpg",
    content_type="image/jpeg",
    summary="an elevation view of the capture, for the sites list",
    stub_bytes=2048,
)
GROUND_SAMPLES = ArtifactDecl(
    "ground_samples.json",
    content_type="application/json",
    summary="the capture's own ground height per grid cell -- half of B4's height offset",
)
MANIFEST = ArtifactDecl(
    "manifest.json",
    content_type="application/json",
    summary="sensor, date, resolution, georeference method, scale source, uncertainty, tools",
)
REGISTRATION = ArtifactDecl(
    "registration.json",
    content_type="application/json",
    summary="what the pipeline asks the API to register: slug, artifacts, manifest",
)


#: What a hand placement is worth, in metres, until somebody measures it.
#:
#: Ten metres is the order of a person dropping a pin on a globe: the right magnitude, and
#: honest in a way that zero is not.
DEFAULT_MANUAL_UNCERTAINTY_M = 10.0


def _lands_in(step: str, what: str) -> NoReturn:
    raise NotImplementedError(
        f"{what} is not implemented yet -- it lands in {step}. Its artifact contract is "
        f"declared, so the recipe runs end to end under StubRunner until then."
    )


# ---------------------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------------------


@stage_impl(
    "ingest_splat",
    consumes=("upload",),
    produces=(CANONICAL_PLY, SOURCE_META),
    summary="Lane 1: read a Scaniverse/Polycam/Postshot export into a normalised splat",
)
def ingest_splat(ctx: StageContext) -> StageOutcome:
    """Real: the uploaded `.ply` or `.spz` becomes `canonical.ply` and a description of it.

    It normalises and describes; it does not filter. `opacity_min` belongs to `package`,
    where `splat_tiles.convert` is the code that reads it, and a gaussian dropped here
    would make two stages disagree about how many gaussians the capture has. Gaussians
    with a non-finite position are *counted* here and neutralised there, for the same
    reason.
    """
    source = gaussians.pick_splat_file(ctx.input("upload"))
    splat = gaussians.read_splat(source)
    written = gaussians.write_ply(ctx.output(CANONICAL_PLY.name), splat.columns)
    low, high = splat.bbox()
    document: dict[str, object] = {
        "filename": splat.source_name,
        "format": splat.source_format,
        "bytes": splat.source_bytes,
        "checksum": splat.source_checksum,
        "gaussians": splat.count,
        "nonFinite": splat.non_finite,
        "properties": list(splat.properties_in),
        # Spherical-harmonic bands above the DC term, normals, vertex colours already
        # folded into f_dc: read, and deliberately not carried into canonical.ply.
        "dropped": list(splat.dropped),
        "bboxLocalM": {"min": low, "max": high},
        "extentM": _extent(low, high),
        "medianGaussianM": _median_gaussian_m(splat),
        # The capture-level facts the pipeline cannot know by looking at the bytes. The
        # worker passes what the capture row says; a run started by hand leaves them null.
        "sensor": _optional_str(ctx.param("sensor")),
        "device": _optional_str(ctx.param("device")),
        "capturedAt": _optional_str(ctx.param("captured_at")),
    }
    _write_json(ctx.output(SOURCE_META.name), document)
    ctx.log(
        f"{splat.source_format}: {splat.count} gaussians from {splat.source_name} "
        f"({splat.source_bytes} bytes) -> canonical.ply ({written} bytes)"
    )
    if splat.dropped:
        ctx.log(f"dropped {len(splat.dropped)} properties: {', '.join(splat.dropped)}")
    metrics: dict[str, MetricValue] = {
        "gaussians": splat.count,
        "format": splat.source_format,
        "sourceBytes": splat.source_bytes,
        "canonicalBytes": written,
        "nonFinite": splat.non_finite,
        "droppedProperties": len(splat.dropped),
    }
    return StageOutcome(
        metrics=metrics, summary=f"{splat.count} gaussians from a {splat.source_format}"
    )


@stage_impl(
    "ffmpeg_frames",
    consumes=("upload",),
    produces=(FRAMES, SOURCE_META),
    summary="Lane 2: video or image folder to a frame set, top-K by sharpness",
)
def ffmpeg_frames(ctx: StageContext) -> StageOutcome:
    # A0 #6: imageio-ffmpeg's static binary, not a system ffmpeg (ubuntu-latest has none),
    # and no ffprobe -- metadata comes from scraping `ffmpeg -i` stderr.
    _lands_in("B2", "ffmpeg_frames")


# ---------------------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------------------


@stage_impl("colmap", consumes=("frames",), produces=(POSES,), summary="COLMAP SfM poses")
def colmap(ctx: StageContext) -> StageOutcome:
    # A0 #7 measured this as a real integration test rather than a stub: 40/40 registered,
    # 0.422 deg median rotation error, CPU-only, 48.3 s. B2 builds that test, with
    # exhaustive_matcher -- sequential_matcher registers 2/40 on a closed orbit.
    _lands_in("B2", "pose: colmap")


@stage_impl("glomap", consumes=("frames",), produces=(POSES,), summary="GLOMAP global SfM poses")
def glomap(ctx: StageContext) -> StageOutcome:
    _lands_in("B2", "pose: glomap")


@stage_impl(
    "arkit",
    consumes=("upload", "frames"),
    produces=(POSES, SCALE),
    summary="poses and metric scale straight out of an ARKit capture",
)
def arkit(ctx: StageContext) -> StageOutcome:
    _lands_in("B2", "pose: arkit")


# ---------------------------------------------------------------------------------------
# mask / compensate
# ---------------------------------------------------------------------------------------


@stage_impl("none", summary="a deliberate no-op: consumes nothing, produces nothing")
def none(ctx: StageContext) -> StageOutcome:
    """The default for `mask` and `compensate`.

    It is a real stage rather than an absent one so that a run records that the choice was
    made, and so switching to `robust` or `imc` is a one-word recipe edit.
    """
    ctx.log(f"{ctx.stage_id}: none")
    return StageOutcome(metrics={"skipped": True}, summary="no-op")


@stage_impl(
    "robust",
    consumes=("frames",),
    produces=(MASKS,),
    summary="Splatfacto-W-style adaptive residual masking of transients",
)
def robust(ctx: StageContext) -> StageOutcome:
    _lands_in("B3", "mask: robust")


# ---------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------


@stage_impl(
    "gsplat",
    consumes=("frames", "poses"),
    optional_consumes=("masks",),
    produces=(CANONICAL_PLY, TRAIN_METRICS),
    summary="gsplat 3DGS training; consumes masks when a mask stage produced any",
)
def gsplat(ctx: StageContext) -> StageOutcome:
    # The optional `masks` input is how `mask: none` and `mask: robust` are both legal
    # without the trainer or the executor knowing which one ran.
    _lands_in("B2", "train: gsplat")


@stage_impl(
    "opensplat",
    consumes=("frames", "poses"),
    optional_consumes=("masks",),
    produces=(CANONICAL_PLY, TRAIN_METRICS),
    summary="OpenSplat training, the same contract as gsplat",
)
def opensplat(ctx: StageContext) -> StageOutcome:
    _lands_in("B2", "train: opensplat")


# ---------------------------------------------------------------------------------------
# georeference
# ---------------------------------------------------------------------------------------


@stage_impl(
    "manual_placement",
    optional_consumes=("source_meta.json",),
    produces=(GEOREF,),
    summary="georeference from the coordinates the operator placed the capture at",
)
def manual_placement(ctx: StageContext) -> StageOutcome:
    """Real, not a stub: manual placement is parameters, and parameters are here already.

    Lane 1 has no EXIF and no poses -- a Scaniverse `.ply` is geometry and nothing else --
    so the coordinate comes from whoever placed the capture, and this stage's whole job is
    to write down that it was placed rather than measured. Hence the defaults: the scale
    source is `unresolved` until somebody says where it came from, and the uncertainty is
    ten metres rather than zero. A hand-placed capture with `uncertaintyM: 0` would be the
    inspector claiming a survey, which is exactly what the manifest exists to prevent.
    """
    lat = float(ctx.param("lat", 0.0))
    lon = float(ctx.param("lon", 0.0))
    height = float(ctx.param("height", 0.0))
    document = {
        "lat": lat,
        "lon": lon,
        "height": height,
        "georefMethod": "manual",
        "scaleSource": str(ctx.param("scale_source", "unresolved")),
        "uncertaintyM": float(ctx.param("uncertainty_m", DEFAULT_MANUAL_UNCERTAINTY_M)),
        "note": str(ctx.param("note", "placed by hand; not surveyed")),
    }
    _write_json(ctx.output(GEOREF.name), document)
    ctx.log(f"placed at {lat}, {lon}, {height} m")
    return StageOutcome(metrics={"lat": lat, "lon": lon, "height": height})


@stage_impl(
    "exif_gps",
    consumes=("upload",),
    optional_consumes=("poses",),
    produces=(GEOREF,),
    summary="georeference from EXIF GPS, or an iPhone video's location metadata",
)
def exif_gps(ctx: StageContext) -> StageOutcome:
    # A0 sizing note: read both the mdta com.apple.quicktime.location.ISO6709 key and the
    # older (c)xyz atom -- iPhone writes one or the other depending on the capture path.
    _lands_in("B4", "georeference: exif_gps")


# ---------------------------------------------------------------------------------------
# package / register
# ---------------------------------------------------------------------------------------


@stage_impl(
    "splat_tiles",
    consumes=("canonical.ply", "georef.json"),
    produces=(SPLAT_TILES,),
    summary="pack the splat into KHR_gaussian_splatting 3D Tiles (tools/captures, unchanged)",
)
def splat_tiles(ctx: StageContext) -> StageOutcome:
    """Real, and it is the A6 call unchanged: `tools/captures/splat_tiles.convert`.

    Nothing about that function moved for this step. `SPLAT_TILES.required_members`
    already pinned the file names it writes, `tests/test_captures_bridge.py` already ran
    it for real, and the fixture byte-identity gate still covers it -- which is why the
    reader it uses could be fixed here and what it *writes* could not.
    """
    georef = _read_json(ctx.input(GEOREF.name))
    stats = splat_tiles_convert(
        ctx.input(CANONICAL_PLY.name),
        ctx.output(SPLAT_TILES.name),
        lat=float(_number(georef.get("lat"))),
        lon=float(_number(georef.get("lon"))),
        height=float(_number(georef.get("height"))),
        max_gaussians=int(ctx.param("max_gaussians", 400_000)),
        opacity_min=float(ctx.param("opacity_min", 0.02)),
        geometric_error=float(ctx.param("geometric_error", 2.0)),
    )
    ctx.log(
        f"packaged {stats['gaussians']} gaussians ({stats['dropped']} dropped), "
        f"extent {stats['extent_m']:.2f} m"
    )
    metrics: dict[str, MetricValue] = {key: value for key, value in stats.items()}
    return StageOutcome(metrics=metrics, summary=f"{stats['gaussians']} gaussians packaged")


@stage_impl(
    "splat_thumbnail",
    consumes=("canonical.ply",),
    produces=(THUMBNAIL,),
    summary="an elevation view of the splat as a JPEG, for the sites list",
)
def splat_thumbnail(ctx: StageContext) -> StageOutcome:
    """The row of the plan's artifact table that says "the endpoint exists and nothing
    calls it". Now something does."""
    splat = gaussians.read_splat(ctx.input(CANONICAL_PLY.name))
    stats = gaussians.render_thumbnail(
        splat,
        ctx.output(THUMBNAIL.name),
        size=int(ctx.param("size", 512)),
        alpha_min=float(ctx.param("alpha_min", 0.1)),
        quality=int(ctx.param("quality", 82)),
    )
    ctx.log(f"thumbnail: {stats['size']}px from {stats['gaussians']} gaussians")
    metrics: dict[str, MetricValue] = {key: int(value) for key, value in stats.items()}
    return StageOutcome(metrics=metrics, summary=f"{stats['size']}px elevation view")


@stage_impl(
    "splat_ground",
    consumes=("canonical.ply", "georef.json"),
    produces=(GROUND_SAMPLES,),
    summary="the capture's own ground height per grid cell",
)
def splat_ground(ctx: StageContext) -> StageOutcome:
    """Half of the height-offset subtraction a person does by hand today.

    Nothing consumes this yet -- B4 is where the console samples terrain at the same
    longitude and latitude and takes the median difference -- so it is emitted now in the
    shape `tools/captures/ground_samples.py` already prints, `{lon, lat, z, n}`, and B4
    reads one shape rather than two. `z` is the capture's own up axis in metres, relative
    to the placed origin height: the absolute ellipsoid height of a sample is
    `origin.height + z`.
    """
    georef = _read_json(ctx.input(GEOREF.name))
    splat = gaussians.read_splat(ctx.input(CANONICAL_PLY.name))
    cell_m = float(ctx.param("cell_m", 2.0))
    percentile = float(ctx.param("percentile", 5.0))
    samples = gaussians.ground_samples(
        splat,
        lat=float(_number(georef.get("lat"))),
        lon=float(_number(georef.get("lon"))),
        cell_m=cell_m,
        percentile=percentile,
        min_points=int(ctx.param("min_points", 8)),
        max_cells=int(ctx.param("max_cells", 64)),
    )
    median = gaussians.median_of([sample.z for sample in samples])
    median = None if median is None else round(median, 3)
    document: dict[str, object] = {
        "frame": "enu",
        "origin": {
            "lat": _number(georef.get("lat")),
            "lon": _number(georef.get("lon")),
            "height": _number(georef.get("height")),
        },
        "cellM": cell_m,
        "percentile": percentile,
        # Said plainly, because a low percentile of a *tree* is canopy, not ground: this
        # is the capture's own low surface per cell, and it is only the ground where the
        # capture has one.
        "method": f"p{percentile:g} of the gaussian up-coordinate in each {cell_m:g} m cell",
        "medianZ": median,
        "samples": [sample.to_dict() for sample in samples],
    }
    _write_json(ctx.output(GROUND_SAMPLES.name), document)
    ctx.log(f"{len(samples)} ground cells at {cell_m:g} m, median z {median}")
    metrics: dict[str, MetricValue] = {"samples": len(samples), "cellM": cell_m}
    if median is not None:
        metrics["medianZ"] = median
    return StageOutcome(metrics=metrics, summary=f"{len(samples)} ground cells")


@stage_impl(
    "capture_manifest",
    consumes=("source_meta.json", "georef.json", "splat"),
    optional_consumes=("ground_samples.json", "thumbnail.jpg", "train_metrics.json"),
    produces=(MANIFEST,),
    summary="sensor, date, resolution, georeference method, scale source, uncertainty, tools",
)
def capture_manifest(ctx: StageContext) -> StageOutcome:
    """The row that is easiest to skip, and the one that lets the inspector stay honest.

    It carries no wall-clock of its own on purpose. `capturedAt` is the capture's date and
    comes from the capture; *when this ran* is already recorded in `step.json` and in the
    job row, and duplicating it here is the one thing that would stop two runs over the
    same input producing byte-identical outputs.
    """
    source = _read_json(ctx.input(SOURCE_META.name))
    georef = _read_json(ctx.input(GEOREF.name))
    tiles = ctx.input(SPLAT_TILES.name)
    packaged = _glb_summary(tiles / "splat.glb")
    ground = (
        _read_json(ctx.input(GROUND_SAMPLES.name)) if ctx.has_input(GROUND_SAMPLES.name) else {}
    )
    thumbnail = ctx.input(THUMBNAIL.name) if ctx.has_input(THUMBNAIL.name) else None
    document: dict[str, object] = {
        "capture": {
            "sensor": source.get("sensor"),
            "device": source.get("device"),
            "capturedAt": source.get("capturedAt"),
            "license": ctx.param("license"),
            "attribution": list(ctx.param("attribution", []) or []),
        },
        "source": source,
        "georeference": {
            "lat": _number(georef.get("lat")),
            "lon": _number(georef.get("lon")),
            "height": _number(georef.get("height")),
            "georefMethod": georef.get("georefMethod"),
            "scaleSource": georef.get("scaleSource"),
            "uncertaintyM": _number(georef.get("uncertaintyM")),
            "note": georef.get("note"),
        },
        "splat": {
            "gaussiansIn": source.get("gaussians"),
            "gaussiansPackaged": packaged["gaussians"],
            "bytes": packaged["bytes"],
            "bboxLocalM": packaged["bbox"],
            "extentM": _extent(packaged["bbox"]["min"], packaged["bbox"]["max"]),
        },
        # A splat has no ground sample distance: there are no pixels behind it. The
        # honest analogue is how big a gaussian is, and saying the other is unavailable
        # is what stops the inspector inventing "about 2.7 cm per pixel" for a phone scan.
        "resolution": {
            "gsdM": None,
            "medianGaussianM": source.get("medianGaussianM"),
            "note": "no GSD: this lane ingests a reconstruction, not images",
        },
        "ground": {
            "samples": _count(ground.get("samples")),
            "medianZ": ground.get("medianZ"),
            "cellM": ground.get("cellM"),
            "percentile": ground.get("percentile"),
            "method": ground.get("method"),
        },
        "thumbnail": (
            None
            if thumbnail is None
            else {"file": THUMBNAIL.name, "bytes": thumbnail.stat().st_size}
        ),
        "training": _read_json(ctx.input(TRAIN_METRICS.name))
        if ctx.has_input(TRAIN_METRICS.name)
        else None,
        # The recipe, not the run. There is deliberately no run id and no wall clock in
        # here: those are what make two runs over the same bytes differ, and both are
        # already recorded -- the run id by the job row, the wall clock by `step.json`.
        "run": {
            "recipe": ctx.recipe,
            "note": "the full stage list and every parameter are in the run's recipe.json",
        },
        "tools": _tool_versions(),
    }
    _write_json(ctx.output(MANIFEST.name), document)
    ctx.log(f"manifest: {packaged['gaussians']} packaged gaussians, {packaged['bytes']} bytes")
    metrics: dict[str, MetricValue] = {
        "gaussiansPackaged": packaged["gaussians"],
        "tilesetBytes": packaged["bytes"],
    }
    return StageOutcome(metrics=metrics, summary="capture manifest")


@stage_impl(
    "catalog",
    consumes=("splat", "georef.json"),
    optional_consumes=("manifest.json", "thumbnail.jpg", "ground_samples.json"),
    produces=(REGISTRATION,),
    summary="describe the finished capture for the API's registration endpoint",
)
def catalog(ctx: StageContext) -> StageOutcome:
    """Real, not a stub -- and deliberately offline.

    The pipeline is invoked by a worker (A7) that talks to the API over HTTP. This stage
    writes what should be registered; it never imports the API or opens a database
    connection. A7 reads this file and makes the call.
    """
    georef = _read_json(ctx.input(GEOREF.name))
    tiles = sorted(entry.name for entry in ctx.input(SPLAT_TILES.name).iterdir())
    manifest = _read_json(ctx.input(MANIFEST.name)) if ctx.has_input(MANIFEST.name) else {}
    document: dict[str, object] = {
        "slug": str(ctx.param("slug", ctx.run_id)),
        "title": str(ctx.param("title", "")),
        "recipe": ctx.recipe,
        "georef": georef,
        "artifacts": tiles,
        # The site's boundary, in the capture's own frame, so the worker can place a
        # polygon that is the size of the thing rather than A7's 60 m placeholder square.
        "bboxLocalM": _bbox_of(manifest),
        "thumbnail": THUMBNAIL.name if ctx.has_input(THUMBNAIL.name) else None,
        "manifest": manifest or None,
    }
    _write_json(ctx.output(REGISTRATION.name), document)
    return StageOutcome(metrics={"artifacts": len(tiles)})


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _extent(low: list[float], high: list[float]) -> dict[str, float]:
    east, north, up = (float(b - a) for a, b in zip(low, high, strict=True))
    return {"east": east, "north": north, "up": up}


def _bbox_of(manifest: dict[str, object]) -> dict[str, list[float]] | None:
    """The packaged splat's local bounding box, as the manifest recorded it."""
    splat = manifest.get("splat")
    if not isinstance(splat, dict):
        return None
    bbox = splat.get("bboxLocalM")
    if not isinstance(bbox, dict):
        return None
    low, high = bbox.get("min"), bbox.get("max")
    if not isinstance(low, list) or not isinstance(high, list) or len(low) != 3 or len(high) != 3:
        return None
    return {"min": [_number(v) for v in low], "max": [_number(v) for v in high]}


def _median_gaussian_m(splat: gaussians.Splat) -> float:
    """The median gaussian radius in metres -- a splat's answer to "how fine is this?"."""
    scales = np.stack([splat.columns[f"scale_{i}"] for i in range(3)], axis=1)
    finite = np.isfinite(scales).all(axis=1)
    if not bool(finite.any()):
        return 0.0
    return round(float(np.median(np.exp(scales[finite]))), 6)


def _glb_summary(path: Path) -> dict[str, Any]:
    """Count and bounding box read back out of the GLB the packer actually wrote.

    Not out of the packer's return value: this is the one place that checks the tileset on
    disk says what the manifest is about to claim it says.
    """
    blob = path.read_bytes()
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise ValueError(f"{path.name} is not a GLB file")
    length, kind = struct.unpack_from("<II", blob, 12)
    if kind != 0x4E4F534A:
        raise ValueError(f"{path.name}: the first GLB chunk is not JSON")
    document = json.loads(blob[20 : 20 + length].decode("utf-8"))
    position = document["accessors"][0]
    return {
        "gaussians": int(position["count"]),
        "bytes": len(blob),
        "bbox": {
            "min": [float(v) for v in position["min"]],
            "max": [float(v) for v in position["max"]],
        },
    }


@functools.cache
def _tool_versions() -> dict[str, str]:
    """What produced the artifacts, resolved once.

    `tools/captures` is `package = false` and carries no version number, so what is
    recorded for it is the checksum of the file that did the packing -- which is a more
    useful thing to compare two runs on than a version string nobody bumps.
    """
    pyproject = Path(__file__).resolve().parent / "pyproject.toml"
    version = str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])
    packer = (CAPTURES_DIR / "splat_tiles.py").read_bytes()
    return {
        "pipeline": version,
        "splatTiles": f"sha256:{hashlib.sha256(packer).hexdigest()}",
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }
