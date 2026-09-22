"""The stage implementations, and the artifacts each one declares.

Most of these are stubs at A6, and they say so: they raise rather than pretend, and they
name the step that makes them real. What is *not* a stub is the contract -- the artifacts
each one consumes and produces. That is what the executor validates, what StubRunner
fabricates from, and what A8, B2 and B3 fill in behind. A recipe that runs green under
StubRunner today runs the same stages in the same order against real implementations later,
with no executor, recipe-format or runner change.

Both lanes converge on `canonical.ply`: Lane 1 normalises an already-reconstructed splat
into it, Lane 2's trainer writes it (the plan calls that output "the canonical Gaussians"),
and a single `package` implementation turns either one into the same `splat/` tileset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

from artifacts import ArtifactDecl
from contracts import StageContext, StageOutcome
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
REGISTRATION = ArtifactDecl(
    "registration.json",
    content_type="application/json",
    summary="what the pipeline asks the API to register: slug, artifacts, manifest",
)


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
    _lands_in("A8", "ingest_splat")


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
    """Real, not a stub: manual placement is parameters, and parameters are here already."""
    lat = float(ctx.param("lat", 0.0))
    lon = float(ctx.param("lon", 0.0))
    height = float(ctx.param("height", 0.0))
    document = {
        "lat": lat,
        "lon": lon,
        "height": height,
        "georefMethod": "manual",
        "scaleSource": str(ctx.param("scale_source", "source")),
        "uncertaintyM": float(ctx.param("uncertainty_m", 0.0)),
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
    # A8 replaces this body with captures_bridge.splat_tiles_convert(...) and nothing else
    # changes: SPLAT_TILES.required_members already pins the file names that call writes,
    # and tests/test_captures_bridge.py holds the two together.
    _lands_in("A8", "package: splat_tiles")


@stage_impl(
    "catalog",
    consumes=("splat", "georef.json"),
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
    document: dict[str, object] = {
        "slug": str(ctx.param("slug", ctx.run_id)),
        "title": str(ctx.param("title", "")),
        "recipe": ctx.recipe,
        "georef": georef,
        "artifacts": tiles,
    }
    _write_json(ctx.output(REGISTRATION.name), document)
    return StageOutcome(metrics={"artifacts": len(tiles)})


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
