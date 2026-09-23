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
import shutil
import struct
import tomllib
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import PIL
import PIL.Image

import exif
import gaussians
import sfm
import training
import video
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
    summary="a COLMAP sparse model -- intrinsics, extrinsics, points -- plus poses.json",
    # A COLMAP sparse model as COLMAP writes it, which is what gsplat, OpenSplat and
    # nerfstudio all read, beside a JSON summary of it for anything that would rather not
    # parse a binary model. `points3D.bin` is deliberately absent from this list although
    # every real run writes it: artifact names are lowercase by `validate_artifact_name`,
    # so the only spelling this field could carry is one no run ever produces. B2 changed
    # this from A6's ("cameras.bin", "images.bin", "points3d.bin") for that reason.
    stub_members=("cameras.bin", "images.bin", "poses.json"),
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
    summary="the static splat, east/north/up about its placed origin: Lane 1 normalises "
    "an upload into it, Lane 2 places the trained splat into it",
    stub_bytes=1024,
)
TRAINED_PLY = ArtifactDecl(
    "trained.ply",
    content_type="application/octet-stream",
    summary="the trainer's splat, in the reconstruction's own (COLMAP) frame",
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


#: RANSAC inlier threshold for `colmap model_aligner`, in metres.
#:
#: COLMAP 3.9.1 refuses a run with zero here outright, so there is no non-robust mode to
#: fall back to; this is the size of GPS error the alignment will tolerate on one frame
#: before treating it as an outlier. Five metres is the order of a consumer receiver with
#: a clear sky, and a phone's first frame -- taken before the GPS has locked, at the last
#: cell-tower position -- is exactly the outlier this exists to drop.
DEFAULT_ALIGNMENT_MAX_ERROR_M = 5.0

#: The best horizontal uncertainty an EXIF GPS georeference will ever claim, in metres.
#:
#: The alignment residual can be millimetres and the placement still be five metres out:
#: every fix in one capture shares the receiver's bias, and a bias common to all of them
#: moves the whole reconstruction without changing a single residual. Quoting the residual
#: as the uncertainty would be claiming a survey from a phone.
EXIF_GPS_UNCERTAINTY_FLOOR_M = 5.0

#: What an EXIF georeference that could not be aligned is worth, in metres.
#:
#: The same ten metres a hand placement gets, and deliberately so: a coordinate with no
#: orientation and no scale is not a better answer than a person dropping a pin, however
#: the coordinate was obtained.
UNALIGNED_UNCERTAINTY_M = 10.0

#: What `GPSAltitude` is measured from. Not the ellipsoid, which is the point.
_HEIGHT_DATUM = "exif GPSAltitude: metres above mean sea level, not the WGS84 ellipsoid"

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

    It does convert the frame, and this is the stage that must: `canonical.ply` is
    east/north/up, and an exporter's file is whatever its convention is. `up_axis` (the
    capture's `upAxis`) names the file's up; unset, the format's evidence-based default
    applies (`gaussians.DEFAULT_UP_AXIS`). `heading_deg` turns the capture about the
    vertical, and `recentre` puts the origin at the footprint's centre and the capture's
    own ground. What was done is written into `source_meta.json` as `frame`, so a capture
    that lands wrong carries the reason with it.
    """
    source = gaussians.pick_splat_file(ctx.input("upload"))
    read = gaussians.read_splat(source)
    requested = ctx.param("up_axis")
    splat, frame = gaussians.orient(
        read,
        up_axis=None if requested in (None, "") else str(requested),
        heading_deg=float(ctx.param("heading_deg", 0.0) or 0.0),
        recentre=bool(ctx.param("recentre", True)),
    )
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
        # How the file's own axes became east/north/up, and on whose say-so.
        "frame": {
            **frame.to_dict(),
            "evidence": gaussians.UP_AXIS_EVIDENCE.get(splat.source_format),
        },
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
    ctx.log(
        f"frame: up is the file's {frame.up_axis} ({frame.up_axis_source}), heading "
        f"{frame.heading_deg:g} deg, origin moved by "
        f"{', '.join(f'{v:.3f}' for v in frame.translation)} m"
    )
    metrics: dict[str, MetricValue] = {
        "gaussians": splat.count,
        "upAxis": frame.up_axis,
        "format": splat.source_format,
        "sourceBytes": splat.source_bytes,
        "canonicalBytes": written,
        "nonFinite": splat.non_finite,
        "droppedProperties": len(splat.dropped),
    }
    return StageOutcome(
        metrics=metrics, summary=f"{splat.count} gaussians from a {splat.source_format}"
    )


#: `select:` values this stage accepts. There is deliberately no threshold among them.
SELECT_MODES: tuple[str, ...] = ("sharpness", "sharpness-windowed", "all")


@stage_impl(
    "ffmpeg_frames",
    consumes=("upload",),
    produces=(FRAMES, SOURCE_META),
    summary="Lane 2: video or image folder to a frame set, top-K by sharpness",
)
def ffmpeg_frames(ctx: StageContext) -> StageOutcome:
    """Real: the upload becomes a frame set and a description of where it came from.

    Three of A0's findings are load-bearing and each one is a place this could have been
    written wrongly and still passed on the machine it was written on:

    * the ffmpeg binary comes from `imageio_ffmpeg.get_ffmpeg_exe()`. This machine has
      `/usr/bin/ffmpeg`; `ubuntu-latest` has none, so resolving off `PATH` is green here
      and red in CI. `tests/test_normalize.py` reads the argv out of the stage log and
      asserts it points inside the wheel;
    * there is no `ffprobe` in that wheel, so sensor, date and location are scraped from
      `ffmpeg -i` stderr -- including **both** iPhone location keys, the `mdta`
      `com.apple.quicktime.location.ISO6709` and the older `(c)xyz` atom;
    * `select: sharpness` is top-K and cannot be given a cutoff. A0 measured a 101x
      within-clip range in variance-of-Laplacian, so no absolute threshold transfers.
      `sharpness-windowed` is the same rank taken within each of `keep` equal stretches
      of the clip, for when `keep` is a small fraction of the candidates and a blurred
      stretch would otherwise lose a whole side of the capture.

    What it does *not* do is read EXIF off a folder of stills or turn a location into a
    georeference -- `exif_gps` is that stage, and it lands in B4. A location found here is
    recorded in `source_meta.json` and goes no further.
    """
    upload = ctx.input("upload")
    source = video.pick_source(upload)
    fps = float(ctx.param("fps", 4))
    keep = int(ctx.param("keep", 400))
    select = str(ctx.param("select", "sharpness"))
    if select not in SELECT_MODES:
        raise ValueError(
            f"select={select!r} is not one of {', '.join(SELECT_MODES)}. In particular "
            f"there is no blur threshold: A0 measured a 101x within-clip range in "
            f"variance-of-Laplacian, so a cutoff that works on one capture discards a "
            f"whole other capture. Selection is top-K, set by `keep`"
        )

    meta = video.VideoMeta()
    if source.is_video:
        ctx.log(f"$ {' '.join(video.probe_argv(source.path))}")
        text = video.probe_text(source.path)
        for line in text.splitlines():
            ctx.log(line)
        meta = video.parse_probe(text)
        extracted = ctx.work_dir / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        ctx.run(
            video.extract_frames_argv(
                source.path,
                extracted / "frame_%05d.jpg",
                fps=fps,
                quality=int(ctx.param("quality", 2)),
                max_side=_optional_int(ctx.param("max_side")),
            )
        )
        candidates = sorted(extracted.glob("frame_*.jpg"))
    else:
        candidates = list(source.images)
    if not candidates:
        raise ValueError(f"no frames came out of {source.path.name}")

    if select == "sharpness":
        scores = [video.sharpness(path) for path in candidates]
        chosen = video.select_sharpest(scores, keep)
    elif select == "sharpness-windowed":
        scores = [video.sharpness(path) for path in candidates]
        chosen = video.select_sharpest_per_window(scores, keep)
    else:
        scores = []
        chosen = video.evenly_spaced(len(candidates), keep)
    written = video.copy_frames([candidates[i] for i in chosen], ctx.output(FRAMES.name))

    kept_scores = [scores[i] for i in chosen] if scores else []
    dropped_scores = [s for i, s in enumerate(scores) if i not in set(chosen)]
    first = _image_size(written[0])
    document: dict[str, object] = {
        "filename": source.path.name,
        "format": "video" if source.is_video else "images",
        "bytes": _bytes_of(source),
        "checksum": _checksum_of_source(source),
        "frames": {
            "candidates": len(candidates),
            "kept": len(written),
            "fps": fps if source.is_video else None,
            "select": select,
            "keep": keep,
            "width": first[0],
            "height": first[1],
        },
        # Recorded, not thresholded. The numbers are here so a later run can see *why*
        # these frames and not others, which is the only thing a non-linear score is
        # good for.
        "sharpness": {
            "metric": "variance-of-laplacian",
            "selection": (
                "the sharpest of each of K equal stretches; never an absolute cutoff (A0 #6)"
                if select == "sharpness-windowed"
                else "top-K by rank; never an absolute cutoff (A0 #6)"
            ),
            "kept": video.summarise(kept_scores),
            "rejected": video.summarise(dropped_scores),
        },
        "video": meta.to_dict() if source.is_video else None,
        "location": None if meta.location is None else meta.location.to_dict(),
        # The capture-level facts, same convention as `ingest_splat`: what the capture row
        # says wins, and the container's own answer is the fallback rather than the other
        # way round -- a phone that was renamed is still the phone the row names.
        "sensor": _optional_str(ctx.param("sensor")) or meta.make,
        "device": _optional_str(ctx.param("device")) or meta.model,
        "capturedAt": _optional_str(ctx.param("captured_at")) or meta.created_at,
        "tools": {"ffmpeg": video.ffmpeg_version(), "ffmpegFrom": "imageio-ffmpeg"},
    }
    _write_json(ctx.output(SOURCE_META.name), document)
    ctx.log(
        f"{source.kind}: {len(candidates)} candidate frame(s) -> {len(written)} kept "
        f"by {select} ({first[0]}x{first[1]})"
    )
    if meta.location is not None:
        ctx.log(f"location {meta.location.lat}, {meta.location.lon} from {meta.location.source}")
    metrics: dict[str, MetricValue] = {
        "candidates": len(candidates),
        "frames": len(written),
        "select": select,
        "width": first[0],
        "height": first[1],
    }
    if source.is_video:
        metrics["fps"] = fps
        if meta.duration_s is not None:
            metrics["durationS"] = meta.duration_s
    return StageOutcome(metrics=metrics, summary=f"{len(written)} frames by {select}")


# ---------------------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------------------


@stage_impl("colmap", consumes=("frames",), produces=(POSES,), summary="COLMAP SfM poses")
def colmap(ctx: StageContext) -> StageOutcome:
    """Real, and tested against known poses rather than against "a file appeared".

    `tests/test_pose_colmap.py` renders the committed synthetic tree from a known orbit
    and scores what comes back out of here: how many frames registered, and the median
    rotation and translation error after the similarity that any image-only
    reconstruction is defined up to. On that fixture it measures 40/40 registered,
    0.059 deg and 0.092% of scene extent, CPU only, in about 55 s on 4 cores -- better
    than A0 #7's 40/40, 0.422 deg and 0.636%, because a noise-free pinhole render over a
    planar textured ground is an easier scene than whatever A0 measured, not because
    anything improved. The test's thresholds sit where a regression would show.

    Two of A0's findings are enforced here rather than remembered:

    * the matcher defaults to `exhaustive`. `sequential` is 2.5x faster and registered
      **2 of 40** frames on a closed orbit, because without loop detection the last frame
      never meets the first. Asking for it logs that. (On 100 real iPhone frames,
      vocabulary-tree loop closure lifted it only to 76/100, where exhaustive registered
      50/50: the README's "Where pose runs" has the table);
    * the focal length. With no prior COLMAP self-calibrates, and the bias that leaves
      behind is scene-dependent rather than constant -- A0 #7 measured 3.1% low, B2's
      own fixture measured 0.17% high. `poses.json` therefore records the recovered
      focal, whether a prior was held, and both measurements, rather than a correction.
      Pass `focal_px` (EXIF or ARKit) and the prior is held through bundle adjustment.
    """
    frames = ctx.input(FRAMES.name)
    out = ctx.output(POSES.name)
    matcher = str(ctx.param("matcher", "exhaustive"))
    if matcher != "exhaustive":
        ctx.log(
            f"matcher={matcher!r}: A0 #7 measured {matcher}_matcher registering 2 of 40 "
            f"frames on a closed orbit, where exhaustive_matcher registered 40"
        )
    images = sorted(p for p in frames.iterdir() if p.is_file())
    if not images:
        raise ValueError(f"the frames artifact at {frames} is empty")
    width, height = _image_size(images[0])
    focal_prior = _optional_float(ctx.param("focal_px"))
    params = None if focal_prior is None else (focal_prior, width / 2.0, height / 2.0, 0.0)

    database = ctx.work_dir / "database.db"
    sparse = ctx.work_dir / "sparse"
    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    ctx.run(
        sfm.feature_extractor_argv(
            database,
            frames,
            camera_model=str(ctx.param("camera_model", "SIMPLE_RADIAL")),
            camera_params=params,
            max_image_size=int(ctx.param("max_image_size", 2400)),
            max_features=int(ctx.param("max_features", 8192)),
        )
    )
    ctx.run(sfm.matcher_argv(database, matcher))
    ctx.run(sfm.mapper_argv(database, frames, sparse, refine_focal_length=focal_prior is None))
    model_dir = _largest_model(sparse)
    if model_dir is None:
        raise ValueError(
            f"COLMAP's mapper registered no model from {len(images)} frame(s). With "
            f"matcher={matcher!r} that is the failure A0 #7 saw on a closed orbit; "
            f"exhaustive is the matcher that closes one"
        )
    for entry in sorted(model_dir.iterdir()):
        if entry.is_file():
            shutil.copyfile(entry, out / entry.name)
    model = sfm.read_model(out)
    focal = model.cameras[0].focal_px if model.cameras else 0.0
    document: dict[str, object] = {
        "tool": "colmap",
        "version": sfm.colmap_version(),
        "matcher": matcher,
        "frames": len(images),
        "registered": model.registered,
        "registeredFraction": round(model.registered / len(images) if images else 0.0, 4),
        "points3D": model.points3d,
        "meanTrackLength": round(model.mean_track_length, 3),
        "cameras": [camera.to_dict() for camera in model.cameras],
        "images": [image.to_dict() for image in model.images],
        "focal": {
            "px": round(focal, 3),
            "priorPx": focal_prior,
            "refined": focal_prior is None,
            # Said out loud rather than left for somebody to rediscover: a
            # self-calibrated focal that is low makes the whole reconstruction small by
            # about the same fraction, and nothing downstream can see that from the
            # poses alone.
            "biasNote": (
                # Two measurements, both recorded, because they disagree and the
                # disagreement is the finding: the bias is a property of the capture,
                # not a constant to correct by. A0 #7 measured the self-calibrated
                # focal 3.1% LOW on its phone-like orbit; B2's rendered-orbit fixture
                # measured it 0.17% HIGH on 40 frames of the synthetic tree. With no
                # prior, the only honest statement is that the scale is unverified.
                "self-calibrated, so the scale is unverified: A0 #7 measured this 3.1% "
                "low, B2's rendered-orbit fixture measured it 0.17% high. Neither is a "
                "correction to apply -- pass focal_px (EXIF or ARKit) to hold a prior"
                if focal_prior is None
                else "held at the supplied prior through bundle adjustment"
            ),
        },
        "scale": {
            "metric": False,
            "source": "none",
            "note": (
                "a reconstruction from images alone has no metric scale; `arkit` or a "
                "measured baseline is what produces one"
            ),
        },
        # Gravity, as far as the frames can say: how the phone was held. What `place`
        # levels the splat by when nothing better (an EXIF similarity) exists.
        "upEstimate": None if (up := sfm.camera_up(model)) is None else up.to_dict(),
    }
    _write_json(out / "poses.json", document)
    ctx.log(
        f"colmap: {model.registered}/{len(images)} registered, {model.points3d} points, "
        f"mean track {model.mean_track_length:.2f}, focal {focal:.1f} px"
    )
    fraction = model.registered / len(images) if images else 0.0
    warning = partial_registration_warning(model.registered, len(images))
    if warning is not None:
        ctx.log(warning)
    metrics: dict[str, MetricValue] = {
        "frames": len(images),
        "registered": model.registered,
        "registeredFraction": round(fraction, 4),
        "points3D": model.points3d,
        "meanTrackLength": round(model.mean_track_length, 3),
        "focalPx": round(focal, 3),
        "focalPrior": focal_prior is not None,
        "matcher": matcher,
    }
    return StageOutcome(
        metrics=metrics, summary=f"{model.registered}/{len(images)} frames registered"
    )


@stage_impl("glomap", consumes=("frames",), produces=(POSES,), summary="GLOMAP global SfM poses")
def glomap(ctx: StageContext) -> StageOutcome:
    # Still a stub after B2, deliberately: GLOMAP is not in Ubuntu 24.04's archive
    # (`apt-cache policy glomap` finds nothing), so it cannot be installed on this
    # machine or on `ubuntu-latest`, and an implementation nothing can run is a second
    # unverified sketch. It reads the same database `colmap` builds, so when there is a
    # box with one, this is the mapper call and the same `read_model` afterwards.
    _lands_in("B3", "pose: glomap")


@stage_impl(
    "arkit",
    consumes=("upload", "frames"),
    produces=(POSES, SCALE),
    summary="poses and metric scale straight out of an ARKit capture",
)
def arkit(ctx: StageContext) -> StageOutcome:
    # Still a stub after B4, deliberately, and this is the one-line reason: there is no
    # ARKit capture here and no agreed on-disk format to synthesise one against --
    # `ARFrame.camera.transform` is Apple's, but the *file* is Polycam's or Record3D's or
    # Stray Scanner's, and they disagree about layout, handedness and units. A fixture
    # would be inventing the format, and a parser tested against an invented format is a
    # fourth never-executed implementation beside `gsplat`, `opensplat` and ModalAdapter.
    #
    # It is still the only producer of `scale.json`. B4 did not make that the only route
    # to metric scale, though: `georeference: exif_gps` now recovers a metric similarity
    # from GPS and records it in `georef.json`, so a capture can be scaled without a
    # phone -- less precisely, and it says so.
    _lands_in("B4", "pose: arkit")


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
    produces=(TRAINED_PLY, TRAIN_METRICS),
    summary="gsplat 3DGS training; consumes masks when a mask stage produced any",
)
def gsplat(ctx: StageContext) -> StageOutcome:
    """Dispatches a training run. **No training run has ever been executed here.**

    That sentence is the point of this docstring and it is not hedged: `gsplat` needs
    CUDA and no machine this was written on has a GPU. What exists and is tested is
    everything around the trainer -- the COLMAP dataset it is handed, the argv it is
    given, the metrics read back out of its output, and the PLY normalised into
    `trained.ply`. Since 2026-09-23 the argv and the file names are checked against
    gsplat v1.5.3's own `simple_trainer.py` rather than a memory of it; `training.py`
    lists the four things that check corrected. `tests/test_train_gsplat.py` drives the
    stage with a stand-in trainer and says so in its name.

    **A second attempt starts over.** v1.5.3's trainer cannot resume -- its `--ckpt`
    means "evaluate this checkpoint and do not train" -- so nothing here pretends to: the
    result directory is in `work/`, which the executor clears between attempts, and the
    log says the restart out loud. That makes a preemption cost the attempt, which on
    Modal is rare: its client retries a reclaimed input eight times below this seam.

    The output is `trained.ply`, **in COLMAP's frame** (`--no-normalize-world-space`), not
    `canonical.ply`: turning it into east/north/up needs the georeference, which comes
    later, and is the `place` stage's job. Keeping the GPU stage free of placement also
    keeps the GPU box free of the pipeline's georeferencing inputs.

    The optional `masks` input is how `mask: none` and `mask: robust` are both legal
    without the executor knowing which one ran. This trainer has no mask input, so masks
    that arrive are recorded as ignored rather than silently dropped.
    """
    frames = ctx.input(FRAMES.name)
    poses = ctx.input(POSES.name)
    iterations = int(ctx.param("iterations", 30_000))
    trainer = training.trainer_script(ctx.param("trainer"))
    dataset = training.build_dataset(frames, poses, ctx.work_dir / "dataset")
    # In work/, not checkpoint/: see the docstring. Nothing can resume from it.
    result = ctx.work_dir / "gsplat"
    if result.exists():
        shutil.rmtree(result)
    result.mkdir(parents=True)
    if ctx.attempt > 1:
        ctx.log(
            f"attempt {ctx.attempt}: training restarts from step 0 -- gsplat "
            f"{training.GSPLAT_VERSION}'s simple_trainer.py has no resume (its --ckpt "
            f"evaluates a checkpoint instead of continuing one)"
        )
    if ctx.has_input(MASKS.name):
        ctx.log(
            "masks were produced by an earlier stage and this trainer has no mask input; "
            "they are not used. A mask-aware trainer lands in B3"
        )
    ctx.run(
        training.gsplat_argv(
            training.trainer_python(ctx.param("python")),
            trainer,
            dataset,
            result,
            strategy=str(ctx.param("strategy", "default")),
            max_steps=iterations,
            data_factor=int(ctx.param("data_factor", 1)),
            extra=[str(value) for value in (ctx.param("extra_args") or [])],
        )
    )
    ply = training.latest_ply(result)
    if ply is None:
        raise ValueError(
            f"the trainer wrote no .ply under {result}; there is nothing to normalise "
            f"into {TRAINED_PLY.name}"
        )
    splat = gaussians.read_splat(ply)
    written = gaussians.write_ply(ctx.output(TRAINED_PLY.name), splat.columns)
    metrics_document = training.parse_metrics(
        result,
        ctx.log_path.read_text(encoding="utf-8", errors="replace"),
        trainer=f"gsplat:{trainer.name}",
        requested_iterations=iterations,
        resumed_from_step=None,
        attempts=ctx.attempt,
    )
    document = metrics_document.to_dict()
    # Read off the PLY rather than trusted from the stats file: this is the count the
    # artifact actually has, and the two disagreeing is worth being able to see.
    document["gaussiansInPly"] = splat.count
    document["masksIgnored"] = ctx.has_input(MASKS.name)
    document["gsplatVersion"] = training.GSPLAT_VERSION
    _write_json(ctx.output(TRAIN_METRICS.name), document)
    ctx.log(
        f"gsplat: {splat.count} gaussians from {ply.name} -> {TRAINED_PLY.name} "
        f"({written} bytes); metrics from {metrics_document.source}"
    )
    metrics: dict[str, MetricValue] = {
        "gaussians": splat.count,
        "trainedBytes": written,
        "requestedIterations": iterations,
        "metricsSource": metrics_document.source,
    }
    if metrics_document.iterations is not None:
        metrics["iterations"] = metrics_document.iterations
    if metrics_document.psnr is not None:
        metrics["psnr"] = metrics_document.psnr
    return StageOutcome(metrics=metrics, summary=f"{splat.count} gaussians trained")


@stage_impl(
    "opensplat",
    consumes=("frames", "poses"),
    optional_consumes=("masks",),
    produces=(TRAINED_PLY, TRAIN_METRICS),
    summary="OpenSplat training, the same contract as gsplat",
)
def opensplat(ctx: StageContext) -> StageOutcome:
    # Still a stub after B2. OpenSplat needs libtorch with CUDA and there is no GPU here
    # and no reachable GPU provider, so it would be a second trainer nobody has run --
    # and one unrun trainer is already the honest limit of this step. It reads the same
    # COLMAP dataset `training.build_dataset` assembles, so what it needs from this
    # project already exists.
    _lands_in("B3", "train: opensplat")


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
    consumes=("frames",),
    optional_consumes=("poses", "source_meta.json"),
    produces=(GEOREF,),
    summary="georeference from EXIF GPS, or an iPhone video's location metadata",
)
def exif_gps(ctx: StageContext) -> StageOutcome:
    """Real: the frames' own GPS, through `colmap model_aligner`, into a placed frame.

    Two outcomes, and which one happens is a property of the capture rather than a
    setting:

    * **aligned** -- at least `min_images` frames carry an EXIF fix and a `poses` model
      exists, so the fixes define a metric east/north/up frame about their median and
      COLMAP solves for the similarity that takes the reconstruction into it. That
      similarity carries **metric scale**, which is the thing a reconstruction from images
      alone does not have (`poses.json` records `scale: {metric: false}`), so
      `scaleSource` becomes `exif-gps` and the per-image residuals go into the document.
    * **located** -- fixes but no poses, or one location and no per-frame fixes (an iPhone
      video, whose coordinate `ffmpeg_frames` already scraped into `source_meta.json`).
      The capture is placed at that coordinate and nothing else is claimed: no rotation,
      no scale, and an uncertainty no better than a hand placement, because a coordinate
      with no orientation is not a better answer than a person dropping a pin.

    Three things it deliberately does not do:

    * **it does not pretend the altitude is an ellipsoid height.** `GPSAltitude` is metres
      above mean sea level -- the geoid, tens of metres from the WGS84 ellipsoid the globe
      draws. The number is recorded, the datum is recorded beside it, and the viewer's
      clamp is what actually rests the model on the ground.
    * **it does not quote the residual as the accuracy.** Every fix in one capture shares
      the receiver's bias, so a common offset moves the whole reconstruction and changes
      no residual at all. `uncertaintyM` is therefore floored at
      `EXIF_GPS_UNCERTAINTY_FLOOR_M`, and the residual is reported separately as what it
      is: consistency.
    * **it does not transform the splat itself.** It writes `frame` -- the similarity when
      aligned, otherwise a levelling by the camera-up estimate `pose` recorded -- and the
      `place` stage applies it to `trained.ply`. A capture with no EXIF and no location
      falls back to the coordinate on the capture (`lat`/`lon`, which the worker hands
      over), recorded as `manual`; with none of those it refuses rather than landing at
      (0, 0).

    One consequence worth stating: `georef.json` out of the aligned branch is **not**
    byte-reproducible the way the rest of a run's outputs are. `model_aligner` has no
    non-robust mode on 3.9.1, so the similarity comes out of a RANSAC, and the last digits
    of the scale and the residuals move between runs over the same frames. The Lane 1
    byte-identity gate is unaffected -- `manual_placement` writes parameters.
    """
    frames = ctx.input(FRAMES.name)
    fixes = exif.read_fixes(frames)
    min_images = int(ctx.param("min_images", 3))
    ctx.log(f"{len(fixes)} of {_count_files(frames)} frames carry an EXIF GPS fix")

    model_dir = ctx.input(POSES.name) if ctx.has_input(POSES.name) else None
    if len(fixes) >= max(min_images, 3) and model_dir is not None:
        return _exif_gps_aligned(ctx, fixes, model_dir)

    located = _sole_location(ctx, fixes)
    method = "exif-gps"
    if located is None:
        # The capture's own coordinate -- the console sends where its camera was looking
        # -- is the last resort, and it is recorded as the hand placement it is. A video
        # with no location must neither land in the Gulf of Guinea nor fail silently, so
        # with no coordinate at all this is still a refusal, and it says what to do.
        placed = _placed_coordinate(ctx)
        if placed is None:
            raise ValueError(
                f"no EXIF GPS on any of {_count_files(frames)} frames, no location in "
                f"source_meta.json, and no coordinate on the capture, so there is nothing "
                f"to georeference from. Frames stripped of EXIF (anything re-encoded, and "
                f"every frame ffmpeg extracts from a video that carried no location) land "
                f"here. Give the capture a lat/lon -- the console does, from where its "
                f"camera was looking -- or run `georeference: manual_placement`"
            )
        located, method = placed, "manual"
    lat, lon, height = located
    why = "fewer than three EXIF fixes" if model_dir else "no pose model to align against"
    frame = _frame_from_poses(ctx, model_dir)
    source = (
        "the capture's own coordinate (placed by hand)"
        if method == "manual"
        else "EXIF GPS / the video's location"
    )
    document: dict[str, object] = {
        "lat": lat,
        "lon": lon,
        "height": height,
        "georefMethod": method,
        # Not `exif-gps`: a coordinate with no rotation and no similarity says where the
        # capture is and nothing about how big it is.
        "scaleSource": "unresolved",
        "uncertaintyM": float(ctx.param("uncertainty_m", UNALIGNED_UNCERTAINTY_M))
        if method == "manual"
        else UNALIGNED_UNCERTAINTY_M,
        "note": (
            f"located from {source} but not aligned ({why}): levelled by how the camera "
            f"was held, facing an arbitrary heading unless one was given, at an unresolved "
            f"scale -- no better than a hand placement"
        ),
        "fixes": {"frames": len(fixes), "aligned": 0, "heightDatum": _HEIGHT_DATUM},
        "alignment": None,
        "frame": frame,
    }
    _write_json(ctx.output(GEOREF.name), document)
    ctx.log(f"located at {lat}, {lon}, {height} m from {source}; not aligned ({why})")
    ctx.log(f"frame: {frame['source']}, heading {frame['headingDeg']} deg, scale {frame['scale']}")
    return StageOutcome(
        metrics={"fixes": len(fixes), "aligned": 0, "lat": lat, "lon": lon, "method": method},
        summary=f"located at {lat:.6f}, {lon:.6f} ({method}, not aligned)",
    )


@stage_impl(
    "place_splat",
    consumes=("trained.ply", "georef.json"),
    optional_consumes=("poses",),
    produces=(CANONICAL_PLY,),
    summary="Lane 2: the trained splat, turned into east/north/up by the georeference",
)
def place_splat(ctx: StageContext) -> StageOutcome:
    """Real: `trained.ply` (COLMAP's frame) becomes `canonical.ply` (east/north/up).

    The Lane 2 half of what `ingest_splat` does for Lane 1, and the step that makes
    `alignment.applied` true. It applies `georef.json`'s `frame` with
    `gaussians.transform` -- positions, each gaussian's quaternion and its log-scale --
    and, when the frame asks for it (every branch but the EXIF similarity), recentres on
    the splat's own footprint and ground exactly as Lane 1 does, so both lanes put the
    placed coordinate at the middle of the capture.

    A `georef.json` with no `frame` (written by `manual_placement`) is levelled by the
    camera-up estimate in `poses`, if there is one, and otherwise passed through with a
    warning that nothing levelled it.
    """
    georef = _read_json(ctx.input(GEOREF.name))
    trained = gaussians.read_splat(ctx.input(TRAINED_PLY.name))
    frame = georef.get("frame")
    if not isinstance(frame, dict):
        model_dir = ctx.input(POSES.name) if ctx.has_input(POSES.name) else None
        frame = _frame_from_poses(ctx, model_dir)
        if frame["source"] == "none":
            ctx.log("WARNING: no frame in georef.json and no poses: the splat is not levelled")
    rotation = np.asarray(frame.get("rotation") or np.eye(3), dtype=np.float64)
    scale = float(_number(frame.get("scale")) or 1.0)
    offset = frame.get("translationM")
    translation = None if offset is None else np.asarray(offset, dtype=np.float64)
    columns = gaussians.transform(trained.columns, rotation, translation, scale)
    placed = gaussians.Splat(
        columns=columns,
        source_format=trained.source_format,
        source_name=trained.source_name,
        source_bytes=trained.source_bytes,
        source_checksum=trained.source_checksum,
        properties_in=trained.properties_in,
        dropped=trained.dropped,
        non_finite=trained.non_finite,
    )
    moved = [0.0, 0.0, 0.0]
    if frame.get("recentre"):
        placed, recentred = gaussians.orient(placed, up_axis="z")
        moved = [float(v) for v in recentred.translation]
    written = gaussians.write_ply(ctx.output(CANONICAL_PLY.name), placed.columns)
    low, high = placed.bbox()
    extent = _extent(low, high)
    ctx.log(
        f"placed {placed.count} gaussians by {frame.get('source')}: scale {scale:g}, "
        f"recentred by {', '.join(f'{v:.3f}' for v in moved)} m; extent "
        f"{extent['east']:.2f} x {extent['north']:.2f} x {extent['up']:.2f} m"
    )
    metrics: dict[str, MetricValue] = {
        "gaussians": placed.count,
        "canonicalBytes": written,
        "frameSource": str(frame.get("source")),
        "scale": scale,
        "extentUpM": round(extent["up"], 3),
    }
    return StageOutcome(metrics=metrics, summary=f"placed by {frame.get('source')}")


def _placed_coordinate(ctx: StageContext) -> tuple[float, float, float] | None:
    """The coordinate the worker handed over from the capture row, if there was one."""
    lat, lon = _optional_float(ctx.param("lat")), _optional_float(ctx.param("lon"))
    if lat is None or lon is None:
        return None
    return lat, lon, _optional_float(ctx.param("height")) or 0.0


def _frame_from_poses(ctx: StageContext, model_dir: Path | None) -> dict[str, object]:
    """How `place` turns a reconstruction with no GPS similarity into east/north/up.

    Levelled by the camera-up estimate `pose` recorded (`sfm.camera_up`), turned to
    `heading_deg` (the capture's `headingDeg`, else 0 -- which is to say arbitrary, and
    the note says so), scaled by `scale` metres per model unit (else 1, unresolved), and
    recentred on the splat's own footprint by `place`, because nothing here knows where
    in the model the placed coordinate is.
    """
    heading = float(ctx.param("heading_deg", 0.0) or 0.0)
    scale = float(ctx.param("scale", 1.0) or 1.0)
    rotation = np.eye(3)
    up: dict[str, object] | None = None
    source = "none"
    if model_dir is not None:
        estimate = sfm.camera_up(sfm.read_model(model_dir))
        if estimate is not None:
            rotation = gaussians.heading_rotation(heading) @ sfm.rotation_onto_z(estimate.up)
            up = estimate.to_dict()
            source = "camera-up"
    return {
        "source": source,
        "scale": scale,
        "rotation": [[float(v) for v in row] for row in rotation],
        "translationM": None,
        "recentre": True,
        "headingDeg": heading,
        "headingSource": "capture" if ctx.param("heading_deg") is not None else "none",
        "up": up,
    }


def _exif_gps_aligned(
    ctx: StageContext, fixes: tuple[exif.Fix, ...], model_dir: Path
) -> StageOutcome:
    """The aligned branch: fixes define the frame, COLMAP solves for the similarity."""
    lat, lon, height = exif.median_fix(fixes)
    reference = exif.enu_offsets(fixes, (lat, lon, height))
    model = sfm.read_model(model_dir)
    centres = {image.name: tuple(float(v) for v in image.centre) for image in model.images}
    common = {name: value for name, value in reference.items() if name in centres}
    if len(common) < 3:
        raise ValueError(
            f"{len(common)} of {len(fixes)} frames with an EXIF fix are in the pose model "
            f"({model.registered} registered), and a similarity needs three. Either the "
            f"reconstruction dropped the frames that had fixes, or the frame names moved "
            f"between `normalize` and `pose`"
        )

    work = ctx.work_dir / "align"
    work.mkdir(parents=True, exist_ok=True)
    ref_path = work / "ref_positions.txt"
    sfm.write_ref_positions(ref_path, common)
    max_error_m = float(ctx.param("max_error_m", DEFAULT_ALIGNMENT_MAX_ERROR_M))
    # COLMAP 3.9.1 aborts (SIGABRT, no message) rather than creating `--output_path`, and
    # the directory it writes there is never read -- see `sfm`'s module docstring.
    aligned = work / "aligned"
    aligned.mkdir(parents=True, exist_ok=True)
    argv = sfm.model_aligner_argv(
        model_dir,
        aligned,
        ref_path,
        work / "transform.txt",
        max_error_m=max_error_m,
        min_common_images=3,
    )
    ctx.run(argv)
    similarity = sfm.read_similarity(work / "transform.txt")
    residuals = sfm.alignment_residuals(similarity, centres, common)
    values = np.asarray(sorted(residuals.values()), dtype=np.float64)
    rms = float(np.sqrt(float((values**2).mean()))) if values.size else 0.0
    inliers = int((values <= max_error_m).sum())
    # Consistency, floored. The residual cannot see the receiver's own bias, which is
    # common to every fix in the capture and moves all of it together.
    uncertainty = max(rms, EXIF_GPS_UNCERTAINTY_FLOOR_M)

    document: dict[str, object] = {
        "lat": lat,
        "lon": lon,
        "height": height,
        "georefMethod": "exif-gps",
        "scaleSource": "exif-gps",
        "uncertaintyM": round(uncertainty, 3),
        "note": (
            f"aligned to {len(common)} EXIF GPS fixes with colmap model_aligner; "
            f"residual {float(np.median(values)):.2f} m median, {rms:.2f} m rms. The "
            f"reported uncertainty is floored at {EXIF_GPS_UNCERTAINTY_FLOOR_M:g} m "
            f"because a residual measures consistency, not accuracy"
        ),
        "fixes": {
            "frames": len(fixes),
            "aligned": len(common),
            "heightDatum": _HEIGHT_DATUM,
            "dopMedian": _median_or_none([f.dop for f in fixes if f.dop is not None]),
        },
        "alignment": {
            "tool": "colmap model_aligner",
            "version": sfm.colmap_version(),
            "alignmentType": "custom",
            "maxErrorM": max_error_m,
            "images": len(common),
            "inliers": inliers,
            # `scale * R @ x + t` takes a point in the reconstruction's own units into
            # east/north/up metres about (lat, lon, height).
            "scale": similarity.scale,
            "rotation": [[float(v) for v in row] for row in similarity.rotation],
            "translationM": [float(v) for v in similarity.translation],
            "residualM": {
                "median": round(float(np.median(values)), 4),
                "rms": round(rms, 4),
                "max": round(float(values.max()), 4),
                "perImage": {name: round(value, 4) for name, value in sorted(residuals.items())},
            },
            # Applied since the `place` stage exists: `train` writes `trained.ply` in
            # COLMAP's own frame (gsplat's world normalisation is switched off for exactly
            # this), and `place` turns it by this similarity into `canonical.ply`.
            "applied": True,
            "appliedNote": (
                "applied by `place`: trained.ply is in the reconstruction's frame and "
                "canonical.ply is this similarity of it, east/north/up metres about "
                "(lat, lon, height)"
            ),
        },
        # The same similarity, in the shape `place` reads whichever branch wrote it. No
        # recentring: the origin is the median fix, and the similarity already puts the
        # model where the fixes say it is relative to that.
        "frame": {
            "source": "exif-gps-similarity",
            "scale": similarity.scale,
            "rotation": [[float(v) for v in row] for row in similarity.rotation],
            "translationM": [float(v) for v in similarity.translation],
            "recentre": False,
            "headingDeg": None,
            "headingSource": "exif-gps",
            "up": None,
        },
    }
    _write_json(ctx.output(GEOREF.name), document)
    ctx.log(
        f"aligned {len(common)} fixes: scale {similarity.scale:.6g}, residual "
        f"{float(np.median(values)):.3f} m median / {rms:.3f} m rms, {inliers} within "
        f"{max_error_m:g} m"
    )
    metrics: dict[str, MetricValue] = {
        "fixes": len(fixes),
        "aligned": len(common),
        "inliers": inliers,
        "scale": round(similarity.scale, 6),
        "residualMedianM": round(float(np.median(values)), 4),
        "residualRmsM": round(rms, 4),
        "uncertaintyM": round(uncertainty, 3),
    }
    return StageOutcome(
        metrics=metrics,
        summary=f"{len(common)} EXIF fixes, {rms:.2f} m rms residual",
    )


def _sole_location(
    ctx: StageContext, fixes: tuple[exif.Fix, ...]
) -> tuple[float, float, float] | None:
    """One coordinate for the whole capture: the fixes' median, or the video's location.

    A0's sizing note on this stage was to read **both** iPhone location keys -- the `mdta`
    `com.apple.quicktime.location.ISO6709` and the older `(c)xyz` atom. That is already
    done, in `video.parse_probe`, and `ffmpeg_frames` put the answer in
    `source_meta.json`; reading it a second time here would be a second parser of the same
    two keys.
    """
    if fixes:
        return exif.median_fix(fixes)
    if not ctx.has_input(SOURCE_META.name):
        return None
    location = _read_json(ctx.input(SOURCE_META.name)).get("location")
    if not isinstance(location, dict):
        return None
    lat, lon = _optional_float(location.get("lat")), _optional_float(location.get("lon"))
    if lat is None or lon is None:
        return None
    return lat, lon, _optional_float(location.get("alt")) or 0.0


def _count_files(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file())


def _median_or_none(values: list[float]) -> float | None:
    return round(float(np.median(np.asarray(values, dtype=np.float64))), 3) if values else None


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
    ground = (
        _read_json(ctx.input(GROUND_SAMPLES.name)) if ctx.has_input(GROUND_SAMPLES.name) else None
    )
    document: dict[str, object] = {
        "slug": str(ctx.param("slug", ctx.run_id)),
        "title": str(ctx.param("title", "")),
        "recipe": ctx.recipe,
        "georef": georef,
        "artifacts": tiles,
        # `ground_samples.json` in full, not the manifest's five-field summary of it.
        # B4 is where this stops being an artifact nothing reads: the worker turns each
        # cell into an ellipsoid height on the asset, and the viewer rests the capture's
        # own measured ground on the terrain instead of resting its bounding box on it.
        # `optional_consumes` still, so a recipe without a ground stage registers as
        # before and the viewer falls back to the bounding box.
        "ground": ground,
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


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _optional_float(value: object) -> float | None:
    return None if value is None else float(str(value))


def _image_size(path: Path) -> tuple[int, int]:
    with PIL.Image.open(path) as image:
        return int(image.width), int(image.height)


def _bytes_of(source: video.Source) -> int:
    if source.is_video:
        return source.path.stat().st_size
    return sum(path.stat().st_size for path in source.images)


def _checksum_of_source(source: video.Source) -> str:
    """The upload's identity: the file's hash, or a hash over the stills in name order."""
    digest = hashlib.sha256()
    paths = (source.path,) if source.is_video else source.images
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def partial_registration_warning(registered: int, frames: int) -> str | None:
    """Say it as a problem when not every frame joined the reconstruction.

    A partial reconstruction is the ordinary way a real capture disappoints. COLMAP's
    mapper writes one sub-model per connected component and `_largest_model` takes the
    biggest, so frames that did not join it are simply gone -- and everything downstream
    still runs: `train` trains on the subset and the splat covers only what registered.
    The count was always in the metrics and the summary; what was missing was anything
    that reads as "this is not what you asked for" rather than as a number.

    Returns None when every frame registered, so the caller logs nothing on the happy
    path. A pure function because the interesting cases -- a half-registered orbit -- are
    ones a real COLMAP run cannot be made to produce on demand.
    """
    if frames <= 0 or registered >= frames:
        return None
    fraction = registered / frames
    return (
        f"colmap: WARNING -- {frames - registered} of {frames} frames did not register "
        f"({100 * fraction:.1f}% registered). The reconstruction covers only the frames "
        f"that did; a low fraction usually means too little overlap, motion blur, or a "
        f"scene the matcher could not close."
    )


def _largest_model(sparse: Path) -> Path | None:
    """COLMAP's mapper writes one numbered sub-model per connected component.

    The biggest one is the reconstruction; the others are the frames that did not join
    it. Taking the biggest rather than `0` matters because the numbering is the order
    they were found in, not their size.
    """
    models = [entry for entry in sorted(sparse.iterdir()) if (entry / "images.bin").is_file()]
    if not models:
        return None
    return max(models, key=lambda path: (path / "images.bin").stat().st_size)


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
