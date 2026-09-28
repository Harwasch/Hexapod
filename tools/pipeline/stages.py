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
import math
import os
import shutil
import struct
import subprocess
import time
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any, NoReturn

import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image

import blocks
import colmap4
import convergence
import exif
import gaussian_budget
import gaussians
import global_sfm
import holdout
import init_seed
import keyframes
import live
import lod_parents
import quality
import resolution
import sfm
import splat_io
import splat_stream
import support_mask
import training
import video
from artifacts import ArtifactDecl
from captures_bridge import (
    CAPTURES_DIR,
    TILE_GAUSSIANS,
    ParentOverrideError,
    ParentOverrides,
    splat_tiles_convert,
)
from contracts import FANOUT_METRIC, MetricValue, StageContext, StageOutcome, fanout_role
from phases import Phases
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
COVERAGE_ENU = ArtifactDecl(
    "coverage_enu.ply",
    content_type="application/octet-stream",
    summary="quality's coverage.ply, placed exactly as canonical.ply was: gaussian centres "
    "coloured by tier and the camera path, east/north/up; no points without a quality stage",
    stub_bytes=512,
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
    summary="the level-of-detail 3D Tiles tileset the console and the viewer render",
    # Pinned to what tools/captures/splat_tiles.convert() writes for every scan: the
    # tileset and its root tile. A scan within one tile's budget (the stub, the committed
    # tree) is exactly these two; a bigger one adds `splat_<octant path>.glb` children,
    # which `tileset.json` names and tests/test_captures_bridge.py checks it names all of.
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
    # A chunk at a time (`splat_stream`): a phone app's PLY of several million gaussians
    # is placed in the worker's fixed memory, with the same bytes out as `gaussians.orient`.
    splat = splat_stream.open_splat(
        gaussians.pick_splat_file(ctx.input("upload")),
        chunk=int(ctx.param("chunk_gaussians", splat_io.CHUNK)),
    )
    requested = ctx.param("up_axis")
    median_gaussian = splat_stream.median_gaussian_m(splat)
    placed = splat_stream.orient_to(
        splat,
        ctx.output(CANONICAL_PLY.name),
        up_axis=None if requested in (None, "") else str(requested),
        heading_deg=float(ctx.param("heading_deg", 0.0) or 0.0),
        recentre=bool(ctx.param("recentre", True)),
    )
    assert placed.frame is not None
    frame = placed.frame
    written = placed.bytes
    low, high = placed.low, placed.high
    document: dict[str, object] = {
        "filename": splat.source_name,
        "format": splat.source_format,
        "bytes": splat.source_bytes,
        "checksum": splat.checksum(),
        "gaussians": splat.count,
        "nonFinite": placed.non_finite,
        "properties": list(splat.properties_in),
        # Spherical-harmonic bands above the DC term, normals, vertex colours already
        # folded into f_dc: read, and deliberately not carried into canonical.ply.
        "dropped": list(splat.dropped),
        "bboxLocalM": {"min": low, "max": high},
        "extentM": _extent(low, high),
        "medianGaussianM": median_gaussian,
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
        "nonFinite": placed.non_finite,
        "droppedProperties": len(splat.dropped),
    }
    return StageOutcome(
        metrics=metrics, summary=f"{splat.count} gaussians from a {splat.source_format}"
    )


#: `select:` values this stage accepts. There is deliberately no threshold among them.
SELECT_MODES: tuple[str, ...] = ("viewpoint", "sharpness", "sharpness-windowed", "all")


@stage_impl(
    "ffmpeg_frames",
    consumes=("upload",),
    produces=(FRAMES, SOURCE_META),
    summary="Lane 2: video or image folder to a frame set, keyframes by camera motion",
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
    * selection ranks by sharpness and cannot be given a cutoff. A0 measured a 101x
      within-clip range in variance-of-Laplacian, so no absolute threshold transfers.

    Which frames, by `select`:

    * `viewpoint` (the recipe's, for a video): candidates at `fps` (bounded by
      `max_candidates` over the clip, and by the clip's own rate), cut into windows of
      equal camera motion -- `overlap` of the view, `parallax` of the long side -- and
      the sharpest candidate of each kept (`keyframes.py` has the method and the
      derivation). The count follows the capture; `keep_video` is only a ceiling for
      pose time, and when it binds the windows are widened evenly by motion. A photo set
      has no order to measure motion along, and is matched exhaustively (quadratic), so
      it keeps today's rule: `sharpness-windowed` to `keep`;
    * `sharpness`: top-K of the candidates by rank; `sharpness-windowed`: the same rank
      within each of `keep` equal stretches of time; `all`: evenly spaced.

    `max_side` is the long side frames are kept at: a number, or `auto` -- 1600 unless a
    sample of the capture's sharpest frames measurably carries detail above it, then up
    to `auto_ceiling` (`resolution.py`). Absent, frames keep their size.

    What it does *not* do is read EXIF off a folder of stills or turn a location into a
    georeference -- `exif_gps` is that stage, and it lands in B4. A location found here is
    recorded in `source_meta.json` and goes no further.
    """
    upload = ctx.input("upload")
    source = video.pick_source(upload)
    fps = float(ctx.param("fps", 4))
    keep = int(ctx.param("keep", 400))
    select = str(ctx.param("select", "sharpness"))
    # A video's frames are matched sequentially (linear in frames), a photo set's
    # exhaustively (quadratic), so a video can afford more of them: `keep_video`, when
    # set, is a video's cap -- under `viewpoint`, its ceiling -- and `keep` stays the
    # photos'.
    keep_video = _optional_int(ctx.param("keep_video"))
    if select not in SELECT_MODES:
        raise ValueError(
            f"select={select!r} is not one of {', '.join(SELECT_MODES)}. In particular "
            f"there is no blur threshold: A0 measured a 101x within-clip range in "
            f"variance-of-Laplacian, so a cutoff that works on one capture discards a "
            f"whole other capture. Selection is by rank: top-K by `keep`, or the sharpest "
            f"of each window of camera motion"
        )
    size_rule = _frame_size_rule(ctx.param("max_side"))
    base = int(ctx.param("auto_base", resolution.BASE))
    size_ceiling = int(ctx.param("auto_ceiling", resolution.CEILING))
    # Photos keep today's behaviour under `viewpoint`: there is no capture order to
    # measure motion along, and their matching is exhaustive, so `keep` stays their cap.
    applied = "sharpness-windowed" if select == "viewpoint" and not source.is_video else select

    meta = video.VideoMeta()
    rate: float | None = None
    size: resolution.SizeDecision | None = None
    if source.is_video:
        ctx.log(f"$ {' '.join(video.probe_argv(source.path))}")
        text = video.probe_text(source.path)
        for line in text.splitlines():
            ctx.log(line)
        meta = video.parse_probe(text)
        if size_rule == "auto":
            size = _video_frame_size(ctx, source.path, meta, base=base, ceiling=size_ceiling)
        max_side = size.max_side if size is not None else _fixed_side(size_rule)
        rate = _candidate_rate(ctx, fps, meta) if applied == "viewpoint" else fps
        extracted = ctx.work_dir / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        ctx.run(
            video.extract_frames_argv(
                source.path,
                extracted / "frame_%05d.jpg",
                fps=rate,
                quality=int(ctx.param("quality", 2)),
                max_side=max_side,
            )
        )
        candidates = sorted(extracted.glob("frame_*.jpg"))
        if keep_video is not None:
            keep = keep_video
    else:
        candidates = list(source.images)
        if size_rule == "auto" and candidates:
            size = photo_frame_size(candidates, base=base, ceiling=size_ceiling)
        max_side = size.max_side if size is not None else _fixed_side(size_rule)
    if not candidates:
        raise ValueError(f"no frames came out of {source.path.name}")
    if size is not None:
        ctx.log(f"frame size: {size.max_side} px -- {size.reason}")

    viewpoint: dict[str, object] | None = None
    segmentation: keyframes.Segmentation | None = None
    if applied == "viewpoint":
        overlap = float(ctx.param("overlap", 0.9))
        parallax = float(ctx.param("parallax", 0.005))
        if not 0.0 < overlap < 1.0 or parallax <= 0.0:
            raise ValueError(
                f"overlap must be between 0 and 1 and parallax above 0 (got {overlap}, "
                f"{parallax}): they are the motion one window may hold"
            )
        analysis = keyframes.analyse(candidates)
        scores = list(analysis.scores)
        segmentation = keyframes.segment(
            analysis.pairs,
            frame_size=analysis.frame_size,
            overlap=overlap,
            parallax=parallax,
            ceiling=keep,
        )
        chosen = keyframes.select_per_window(scores, segmentation.windows)
        viewpoint = _viewpoint_record(
            analysis, segmentation, chosen, overlap=overlap, parallax=parallax
        )
    elif applied == "sharpness":
        scores = [video.sharpness(path) for path in candidates]
        chosen = video.select_sharpest(scores, keep)
    elif applied == "sharpness-windowed":
        scores = [video.sharpness(path) for path in candidates]
        chosen = video.select_sharpest_per_window(scores, keep)
    else:
        scores = []
        chosen = video.evenly_spaced(len(candidates), keep)
    written = video.copy_frames(
        [candidates[i] for i in chosen],
        ctx.output(FRAMES.name),
        # Photos arrive at full sensor size; a video's frames were bounded by ffmpeg
        # already and pass straight through.
        max_side=max_side,
    )

    kept_scores = [scores[i] for i in chosen] if scores else []
    chosen_set = set(chosen)
    dropped_scores = [s for i, s in enumerate(scores) if i not in chosen_set]
    first = _image_size(written[0])
    selection = {
        "viewpoint": "the sharpest of each window of camera motion; never an absolute "
        "cutoff (A0 #6)",
        "sharpness-windowed": "the sharpest of each of K equal stretches; never an absolute "
        "cutoff (A0 #6)",
    }.get(applied, "top-K by rank; never an absolute cutoff (A0 #6)")
    document: dict[str, object] = {
        "filename": source.path.name,
        "format": "video" if source.is_video else "images",
        "bytes": _bytes_of(source),
        "checksum": _checksum_of_source(source),
        "frames": {
            "candidates": len(candidates),
            "kept": len(written),
            # The rate candidates were taken at: `fps`, or less when `max_candidates`
            # over the clip's length (or the clip's own rate) bounded it.
            "fps": rate,
            "fpsRequested": fps if source.is_video else None,
            "select": applied,
            "selectRequested": select,
            "keep": keep,
            "width": first[0],
            "height": first[1],
            "maxSide": max_side,
        },
        # How big the frames are and why: `auto`'s measurement, or the number asked for.
        "frameSize": (
            size.to_dict()
            if size is not None
            else {"rule": "fixed" if max_side is not None else "source", "maxSide": max_side}
        ),
        "viewpoint": viewpoint,
        # Recorded, not thresholded. The numbers are here so a later run can see *why*
        # these frames and not others, which is the only thing a non-linear score is
        # good for.
        "sharpness": {
            "metric": "variance-of-laplacian",
            "selection": selection,
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
        f"by {applied} ({first[0]}x{first[1]})"
    )
    if segmentation is not None:
        widened = (
            f"widened {segmentation.scale:.2f}x from {segmentation.uncapped_windows} to fit "
            f"the ceiling of {keep}"
            if segmentation.bound
            else f"ceiling {keep} not reached"
        )
        ctx.log(
            f"viewpoint: {len(segmentation.windows)} window(s) of camera motion ({widened}); "
            f"{segmentation.unmeasured_pairs} pair(s) whose motion could not be measured"
        )
    if meta.location is not None:
        ctx.log(f"location {meta.location.lat}, {meta.location.lon} from {meta.location.source}")
    metrics: dict[str, MetricValue] = {
        "candidates": len(candidates),
        "frames": len(written),
        "select": applied,
        "width": first[0],
        "height": first[1],
        "maxSide": max_side if max_side is not None else max(first),
        "maxSideRule": "auto" if size is not None else "fixed" if max_side else "source",
    }
    if source.is_video:
        metrics["fps"] = rate if rate is not None else fps
        if meta.duration_s is not None:
            metrics["durationS"] = meta.duration_s
    if segmentation is not None:
        metrics["keyframeCeilingBound"] = segmentation.bound
        metrics["motionScale"] = round(segmentation.scale, 4)
        metrics["unmeasuredPairs"] = segmentation.unmeasured_pairs
    return StageOutcome(metrics=metrics, summary=f"{len(written)} frames by {applied}")


def _frame_size_rule(value: object) -> int | str | None:
    """`max_side` as asked: absent (keep the source's size), a number, or `auto`."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    if isinstance(value, bool):
        raise ValueError(f"max_side must be a number of pixels or 'auto', not {value!r}")
    try:
        side = int(str(value))
    except ValueError:
        raise ValueError(f"max_side must be a number of pixels or 'auto', not {value!r}") from None
    if side < 64:
        raise ValueError(f"max_side={side} is too small to reconstruct anything from")
    return side


def _fixed_side(rule: int | str | None) -> int | None:
    return rule if isinstance(rule, int) else None


def _candidate_rate(ctx: StageContext, fps: float, meta: video.VideoMeta) -> float:
    """The rate `viewpoint` extracts candidates at: `fps`, bounded two ways.

    By the clip's own frame rate, because ffmpeg's `fps` filter duplicates frames to
    reach a rate the clip does not have, and a duplicate is a candidate with no motion
    and no new sharpness. And by `max_candidates` over the clip's length, because every
    candidate is decoded, scored and tracked on the worker (~50 ms each, plus its JPEG on
    disk): 1200 is two minutes at 10 fps, and a five-minute walk is sampled at 4 fps --
    still several candidates to every window at a walking pace.
    """
    rate = fps
    if meta.fps is not None and meta.fps > 0:
        rate = min(rate, meta.fps)
    max_candidates = int(ctx.param("max_candidates", 1200))
    if meta.duration_s is not None and meta.duration_s > 0 and max_candidates > 0:
        rate = min(rate, max_candidates / meta.duration_s)
    return round(rate, 4)


def _video_frame_size(
    ctx: StageContext, source: Path, meta: video.VideoMeta, *, base: int, ceiling: int
) -> resolution.SizeDecision:
    """`max_side: auto` for a clip: frames sampled across it at the size that would be
    kept, the sharpest half measured. Eight seeks, not a decode of the whole clip."""
    long_side = max(meta.width, meta.height) if meta.width and meta.height else None
    side = resolution.measured_side(long_side, base=base, ceiling=ceiling)
    if long_side is None or side is None:
        return resolution.not_measured(long_side, base=base)
    duration = meta.duration_s if meta.duration_s and meta.duration_s > 0 else None
    count = resolution.SAMPLES
    times = (
        [(index + 0.5) * duration / count for index in range(count)]
        if duration is not None
        else [float(index) for index in range(count)]
    )
    folder = ctx.work_dir / "size-sample"
    folder.mkdir(parents=True, exist_ok=True)
    greys: list[npt.NDArray[np.float32]] = []
    for index, at in enumerate(times):
        target = folder / f"sample_{index:02d}.png"
        try:
            ctx.run(video.grab_frame_argv(source, at, target, max_side=side))
        except subprocess.CalledProcessError:
            continue
        if target.is_file():
            with PIL.Image.open(target) as image:
                greys.append(np.asarray(image.convert("L"), dtype=np.float32))
    return _decide_frame_size(greys, long_side=long_side, base=base, ceiling=ceiling)


def photo_frame_size(
    photos: Sequence[Path | IO[bytes]], *, base: int, ceiling: int
) -> resolution.SizeDecision:
    """`max_side: auto` for a photo set: the same measurement on photos spread over it.

    `photos` may be open binary files as well as paths: a driver that streams a photo set
    it never writes to disk (infra/modal/minnetonka.py) sizes it by this same rule from
    the same sample, `photo_sample(len(photos))` of it, held in memory."""
    picks = [photos[i] for i in photo_sample(len(photos))]

    def opened(photo: Path | IO[bytes]) -> PIL.Image.Image:
        if not isinstance(photo, Path):
            photo.seek(0)
        return PIL.Image.open(photo)

    sizes = []
    for photo in picks:
        with opened(photo) as image:
            sizes.append(max(int(image.width), int(image.height)))
    long_side = min(sizes)
    side = resolution.measured_side(long_side, base=base, ceiling=ceiling)
    if side is None:
        return resolution.not_measured(long_side, base=base)
    greys: list[npt.NDArray[np.float32]] = []
    for photo in picks:
        with opened(photo) as image:
            grey = image.convert("L")
        scale = side / max(grey.size)
        if scale < 1.0:
            size = (round(grey.width * scale), round(grey.height * scale))
            grey = grey.resize(size, PIL.Image.Resampling.LANCZOS)
        greys.append(np.asarray(grey, dtype=np.float32))
    return _decide_frame_size(greys, long_side=long_side, base=base, ceiling=ceiling)


def photo_sample(count: int) -> tuple[int, ...]:
    """Which of `count` photos `max_side: auto` measures: `resolution.SAMPLES` spread
    evenly over the set."""
    return video.evenly_spaced(count, resolution.SAMPLES)


def _decide_frame_size(
    greys: Sequence[npt.NDArray[np.float32]], *, long_side: int, base: int, ceiling: int
) -> resolution.SizeDecision:
    """The sharpest half of the sample, by rank (A0 #6), measured for detail."""
    scores = [video.laplacian_variance(grey) for grey in greys]
    picked = video.select_sharpest(scores, resolution.sample_count(len(greys)))
    return resolution.decide(
        [greys[i] for i in picked], source_side=long_side, base=base, ceiling=ceiling
    )


def _viewpoint_record(
    analysis: keyframes.Analysis,
    segmentation: keyframes.Segmentation,
    chosen: Sequence[int],
    *,
    overlap: float,
    parallax: float,
) -> dict[str, object]:
    """What `source_meta.json` says about the windows: enough to check the budgets held."""
    overlaps, drifts = keyframes.neighbour_motion(
        analysis.pairs, chosen, frame_size=analysis.frame_size
    )
    closed_by: dict[str, int] = {}
    for window in segmentation.windows:
        closed_by[window.closed_by] = closed_by.get(window.closed_by, 0) + 1
    tiles = [float(pair.tiles) for pair in analysis.pairs[1:] if pair is not None]
    return {
        "method": (
            f"phase-correlated tiles on grey frames at {keyframes.ANALYSIS_SIDE} px -> "
            "homography by reweighted DLT; a window closes when the composed homography's "
            "overlap or the accumulated off-homography parallax spends its budget; the "
            "sharpest candidate of each window is kept"
        ),
        "overlap": overlap,
        "parallax": parallax,
        # Each budget as applied: `scale` is 1 unless the ceiling widened them.
        "windowBudget": {
            "viewChanged": round((1.0 - overlap) * segmentation.scale, 4),
            "parallaxOfLongSide": round(parallax * segmentation.scale, 5),
        },
        "scale": round(segmentation.scale, 4),
        "ceiling": segmentation.ceiling,
        "ceilingBound": segmentation.bound,
        "windowsBeforeCeiling": segmentation.uncapped_windows,
        "keyframes": len(chosen),
        "closedBy": closed_by,
        "unmeasuredPairs": segmentation.unmeasured_pairs,
        # Frames that matched nothing but were bridged: the next frame matched across them.
        "heldFrames": sum(1 for pair in analysis.pairs if pair is not None and pair.held),
        "tilesPerPair": video.summarise(tiles),
        "candidatesPerKeyframe": video.summarise(
            [float(w.end - w.start + 1) for w in segmentation.windows]
        ),
        # Between consecutive *kept* frames: what the budgets promise, as it came out.
        "neighbourOverlap": video.summarise(overlaps),
        "neighbourParallax": video.summarise(drifts),
    }


# ---------------------------------------------------------------------------------------
# pose
# ---------------------------------------------------------------------------------------


@stage_impl(
    "colmap",
    consumes=("frames",),
    optional_consumes=("source_meta.json",),
    produces=(POSES,),
    summary="COLMAP SfM poses",
)
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

    **Which pairs get matched** is `sfm.matching_plan`'s decision, from `matcher`
    (`auto` in the recipe) and `source_meta.json`'s `format`: frames cut from a video are
    matched sequentially with vocabulary-tree loop closure, a photo set exhaustively.
    Sequential is a first try, not a verdict. If its best mapping registers fewer than
    `min_registered_fraction` of the frames, the stage matches exhaustively on the same
    database -- which fills in only the pairs sequential skipped -- and maps again, and
    says so in the log and in `poses.json`'s `matching`. A0 #7 is why: sequential without
    loop detection registered **2 of 40** on a closed orbit, and with it, 76 of 100 real
    iPhone frames where exhaustive registered 50 of 50.

    The focal length is the other finding enforced rather than remembered. With no prior
    COLMAP self-calibrates, and the bias that leaves behind is scene-dependent rather
    than constant -- A0 #7 measured 3.1% low, B2's own fixture measured 0.17% high.
    `poses.json` therefore records the recovered focal, whether a prior was held, and
    both measurements, rather than a correction. Pass `focal_px` (EXIF or ARKit) and the
    prior is held through bundle adjustment.

    `mapper: global` maps with GLOMAP (COLMAP 4, through pycolmap -- `global_sfm.py` has
    the why and the measurements) before any incremental seed, on the first matches; a
    result under `min_registered_fraction` falls back to the incremental mapper on the
    same database, and `poses.json`'s `mapper` records which one made the model kept.

    `colmap: "4.2"` runs extraction, matching and incremental mapping on COLMAP 4.2
    through pycolmap (`colmap4.py`) instead of apt's 3.9.1 CLI, with everything else --
    the plan, the fallbacks, the seeds, `poses.json` -- unchanged, so the two can be
    compared on one capture. 3.9.1 stays the default until that comparison decides;
    `poses.json`'s `version` is the COLMAP that ran. A run that asks for 4.2 where there
    is no pycolmap fails rather than quietly running 3.9.1.
    """
    frames = ctx.input(FRAMES.name)
    out = ctx.output(POSES.name)
    requested = str(ctx.param("matcher", "auto"))
    mapper = str(ctx.param("mapper", "incremental"))
    if mapper not in ("incremental", "global"):
        raise ValueError(f"mapper must be incremental or global, not {mapper!r}")
    engine = str(ctx.param("colmap", "3.9"))
    if engine not in colmap4.VERSIONS:
        raise ValueError(f"colmap must be one of {', '.join(colmap4.VERSIONS)}, not {engine!r}")
    four = engine == "4.2"
    interpreter = colmap4.python()
    images = sorted(p for p in frames.iterdir() if p.is_file())
    if not images:
        raise ValueError(f"the frames artifact at {frames} is empty")
    source_format = _source_format(ctx)
    colmap_version = colmap4.version(interpreter) if four else None
    if four and colmap_version is None:
        raise ValueError(
            f"colmap: 4.2 needs pycolmap=={global_sfm.PYCOLMAP_VERSION} in {interpreter} (the "
            f"Modal CPU image has it; elsewhere set ${global_sfm.PYTHON_ENV}); not "
            f"falling back to 3.9.1, which would make a 4.2 run that is not one"
        )
    # 4.2 reads a faiss tree, 3.9.1 a FLANN one (colmap4.py); each has its own variable.
    vocab_tree = (colmap4.vocab_tree_path if four else sfm.vocab_tree_path)(ctx.param("vocab_tree"))
    rematches = max(0, int(ctx.param("rematches", 1)))
    plan = sfm.matching_plan(
        requested,
        source_format=source_format,
        vocab_tree=vocab_tree is not None,
        rematches=rematches,
    )
    ctx.log(
        f"colmap {colmap_version or '3.9'}: matcher={requested}, "
        f"frames from {source_format or 'an unknown source'}, "
        f"vocabulary tree {vocab_tree or 'absent'}; plan: "
        + " -> ".join(
            f"{step.matcher}{' (loop)' if step.loop_detection else ''}"
            f"{' (cleared)' if step.clear else ''}"
            for step in plan
        )
    )
    width, height = _image_size(images[0])
    focal_prior = _optional_float(ctx.param("focal_px"))
    params = None if focal_prior is None else (focal_prior, width / 2.0, height / 2.0, 0.0)
    threads = _optional_int(ctx.param("threads"))
    seconds = {"extract": 0.0, "match": 0.0, "map": 0.0}

    def timed(phase: str, argv: list[str]) -> None:
        began = time.monotonic()
        ctx.run(argv)
        seconds[phase] += time.monotonic() - began

    database = ctx.work_dir / "database.db"
    sparse = ctx.work_dir / "sparse"
    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    extraction: dict[str, Any] = {
        "camera_model": str(ctx.param("camera_model", "SIMPLE_RADIAL")),
        "camera_params": params,
        "max_image_size": int(ctx.param("max_image_size", 2400)),
        "max_features": int(ctx.param("max_features", 8192)),
        "first_octave": _optional_int(ctx.param("first_octave")),
        "num_threads": threads,
    }
    timed(
        "extract",
        colmap4.extract_argv(interpreter, database, frames, **extraction)
        if four
        else sfm.feature_extractor_argv(database, frames, **extraction),
    )
    passes_run: list[dict[str, object]] = []
    pairs = [0]

    def match(step: sfm.MatchPass) -> None:
        if step.clear:
            sfm.clear_matches(database)
        how: dict[str, Any] = {
            "overlap": _optional_int(ctx.param("sequential_overlap")),
            "vocab_tree": vocab_tree if step.loop_detection else None,
            "num_threads": threads,
        }
        timed(
            "match",
            colmap4.match_argv(interpreter, database, step.matcher, **how)
            if four
            else sfm.matcher_argv(database, step.matcher, **how),
        )
        tried, verified = sfm.count_pairs(database)
        pairs[0] = tried
        passes_run.append({**step.to_dict(), "pairs": tried, "verifiedPairs": verified})

    match(plan[0])

    target = float(ctx.param("min_registered_fraction", 0.8))
    enough = math.ceil(target * len(images))
    seeds = tuple(range(max(1, int(ctx.param("mapper_seeds", 3)))))
    # One mapping for a sequential pass, every seed for the rest. A short sequential
    # result is missing pairs -- a loop not closed, a fast pan -- far more often than it
    # is an unlucky initial pair, and the exhaustive pass after it gets every seed anyway.
    schedule = [seeds[:1] if step.matcher == "sequential" else seeds for step in plan]
    # `mapper: global` (global_sfm.py) is tried first, on the first matches; anything
    # short of `enough` falls through to the incremental seeds on the same matches.
    if mapper == "global":
        schedule[0] = (_GLOBAL_SEED, *schedule[0])
    best_so_far = [0]
    round_of: dict[Path, int] = {}
    # On unless the recipe says `live: false`.
    watch = ctx.param("live") is None or _optional_bool(ctx.param("live"), "live")

    def attempt(round_: int, seed: int) -> tuple[Path | None, int]:
        if round_ > 0 and seed == schedule[round_][0]:
            step = plan[round_]
            ctx.log(
                f"colmap: best so far {best_so_far[0]} of {len(images)} registered, short of "
                f"min_registered_fraction {target:g} ({enough}); matching again: "
                f"{step.matcher}{', cleared first' if step.clear else ''} -- {step.why}"
            )
            match(step)
        global_ = seed == _GLOBAL_SEED
        into = sparse / (f"match{round_}-global" if global_ else f"match{round_}-seed{seed}")
        into.mkdir(parents=True)
        began = time.monotonic()
        if global_:
            if not _map_globally(ctx, database, frames, into, focal_prior is None, threads):
                return None, 0
            seconds["global"] = time.monotonic() - began
        else:
            # Cameras as they are solved, for the live viewer (`live.MapperWatch`).
            snapshots = ctx.work_dir / "snapshots" / into.name
            how: dict[str, Any] = {
                "refine_focal_length": focal_prior is None,
                "random_seed": seed,
                "num_threads": threads,
                "snapshot_path": snapshots if watch else None,
                "snapshot_every": live.snapshot_every(len(images)),
            }
            with live.MapperWatch(snapshots, ctx.log, frames=len(images), enabled=watch):
                ctx.run(
                    colmap4.map_argv(interpreter, database, frames, into, **how)
                    if four
                    else sfm.mapper_argv(database, frames, into, **how)
                )
            seconds["map"] += time.monotonic() - began
        found = _largest_model(into)
        registered = 0
        if found is not None:
            round_of[found] = round_
            registered = sfm.read_model(found).registered
        best_so_far[0] = max(best_so_far[0], registered)
        if global_ and registered < enough:
            ctx.log(
                f"colmap: the global mapper registered {registered} of {len(images)}, short "
                f"of {enough}; mapping incrementally on the same matches"
            )
        return found, registered

    model_dir, tries = _best_reconstruction(attempt, rounds=schedule, enough=enough)
    for tried in tries:
        tried["matcher"] = plan[int(tried["match"])].matcher
        tried["mapper"] = "global" if tried["seed"] == _GLOBAL_SEED else "incremental"
    if len(tries) > 1:
        ctx.log(
            "colmap: "
            + ", ".join(
                f"match {t['match']} ({t['matcher']}) "
                + ("global" if t["mapper"] == "global" else f"seed {t['seed']}")
                + f": {t['registered']}"
                for t in tries
            )
            + f" of {len(images)} registered; kept the best"
        )
    if model_dir is None:
        raise ValueError(
            f"COLMAP's mapper registered no model from {len(images)} frame(s) after "
            f"{len(passes_run)} matching pass(es) ending in {plan[len(passes_run) - 1].matcher}"
        )
    # The model kept came from the pass it was mapped after; name the chain up to it.
    matcher = "+".join(dict.fromkeys(step.matcher for step in plan[: round_of[model_dir] + 1]))
    fell_back = plan[0].matcher == "sequential" and len(passes_run) > 1
    mapper_used = "global" if model_dir.parent.name.endswith("-global") else "incremental"
    for entry in sorted(model_dir.iterdir()):
        if entry.is_file():
            shutil.copyfile(entry, out / entry.name)
    model = sfm.read_model(out)
    if watch:
        live.emit_model(ctx.log, out, frames=len(images), final=True)
    focal = model.cameras[0].focal_px if model.cameras else 0.0
    version = colmap_version or sfm.colmap_version()
    document: dict[str, object] = {
        "tool": "colmap",
        # The COLMAP that extracted, matched and mapped: apt's 3.9.1 CLI, or 4.2 through
        # pycolmap (`colmap: "4.2"`, colmap4.py). What an A/B of the two reads first.
        "version": version,
        "colmap": {"requested": engine, "via": "pycolmap" if four else "cli"},
        "matcher": matcher,
        # Which mapper made the model kept: `global` is pycolmap's GLOMAP pipeline
        # (global_sfm.py), tried first when asked, with the incremental one behind it.
        "mapper": {
            "requested": mapper,
            "used": mapper_used,
            "fellBackToIncremental": mapper == "global" and mapper_used != "global",
            "globalMapS": round(seconds["global"], 1) if "global" in seconds else None,
            "pycolmap": global_sfm.PYCOLMAP_VERSION if mapper == "global" else None,
        },
        # What was asked, what the frames were, and every pass that ran with its pair
        # count -- so a fallback, and what it cost, is visible rather than inferred.
        "matching": {
            "requested": requested,
            "sourceFormat": source_format,
            "vocabTree": None if vocab_tree is None else vocab_tree.name,
            "passes": passes_run,
            "fellBackToExhaustive": fell_back,
            "minRegisteredFraction": target,
        },
        # What COLMAP's `num_threads -1` saw, which on a container can be the host's
        # cores rather than the ones the container is billed for.
        "machine": {"cpuCount": os.cpu_count(), "threads": threads},
        "frames": len(images),
        "registered": model.registered,
        "registeredFraction": round(model.registered / len(images) if images else 0.0, 4),
        # Every mapping tried, so a retry is visible rather than silently lucky.
        "mapperAttempts": tries,
        "points3D": model.points3d,
        "meanTrackLength": round(model.mean_track_length, 3),
        # The mean of the points' own errors, as COLMAP's model_analyzer reports it.
        "meanReprojectionErrorPx": round(model.mean_reprojection_error, 4),
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
        "meanReprojectionErrorPx": round(model.mean_reprojection_error, 4),
        "focalPx": round(focal, 3),
        "focalPrior": focal_prior is not None,
        "matcher": matcher,
        "matchedPairs": pairs[0],
        "fellBack": fell_back,
        "extractS": round(seconds["extract"], 1),
        "matchS": round(seconds["match"], 1),
        "mapS": round(seconds["map"], 1),
        "mapper": mapper_used,
        "colmap": version,
    }
    if "global" in seconds:
        metrics["globalMapS"] = round(seconds["global"], 1)
    return StageOutcome(
        metrics=metrics, summary=f"{model.registered}/{len(images)} frames registered"
    )


#: The seed `_best_reconstruction` is handed for the global mapper's one try.
_GLOBAL_SEED = -1


def _map_globally(
    ctx: StageContext,
    database: Path,
    frames: Path,
    into: Path,
    refine_focal_length: bool,
    threads: int | None,
) -> bool:
    """One global mapping (global_sfm.py) into `into`; False, logged, if it could not run.

    A global mapper that cannot run -- pycolmap missing, a crash in it -- is not the
    stage failing: the incremental mapper is right behind it on the same matches.
    """
    try:
        ctx.run(
            global_sfm.argv(
                global_sfm.python(),
                database,
                frames,
                into,
                refine_focal_length=refine_focal_length,
                num_threads=threads,
            )
        )
    except (subprocess.CalledProcessError, OSError) as error:
        ctx.log(f"colmap: the global mapper did not run ({error}); mapping incrementally")
        return False
    return True


def _source_format(ctx: StageContext) -> str | None:
    """`video` or `images`, as `ffmpeg_frames` recorded it; None when nobody said."""
    if not ctx.has_input(SOURCE_META.name):
        return None
    value = _read_json(ctx.input(SOURCE_META.name)).get("format")
    return str(value) if isinstance(value, str) else None


@stage_impl("glomap", consumes=("frames",), produces=(POSES,), summary="GLOMAP global SfM poses")
def glomap(ctx: StageContext) -> StageOutcome:
    # Still a stub, and now for a different reason: GLOMAP is in COLMAP 4 (and pycolmap's
    # wheels), and it runs as the `colmap` impl's `mapper: global` -- on the database that
    # stage builds, with the incremental mapper behind it when it registers too few.
    # A separate impl would duplicate extraction, matching and that fallback.
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
    produces=(TRAINED_PLY, TRAIN_METRICS, holdout.HOLDOUT),
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

    Parameters beyond the recipe's schedule, each off unless asked for, so a run that
    names none of them is the run every earlier capture got:

    * `schedule_scale` (0.05-1.0) replaces the fraction `schedule_full_at` would compute;
    * `train_max_side` shrinks the frames the trainer reads, not the poses
      (`training.build_dataset`);
    * `antialiased`, `opacity_reg`, `depth_loss`, `pose_opt`, `app_opt` and
      `bilateral_grid` are gsplat's own switches (`training.gsplat_argv` says what each
      does);
    * `roi` (`{"center": [x, y, z], "radius": r}`, COLMAP frame) crops the initial points
      to the sphere before training and the trained gaussians to 1.5 radii after it;
    * `variant` is `3dgs`; `2dgs` is refused, and `training.py` says why;
    * `holdout_error` renders the held-out frames after training and writes each
      gaussian's error on them to `holdout/` (`holdout.py`), for `quality` to verify keep
      with. It cannot fail the stage: a script that fails leaves `holdout.json` saying so.
    * `live` (on unless `false`) exports a PLY at 10/25/50/75% of the schedule and packs
      each into a small SPZ under `checkpoint/live/` for the live viewer (`live.py`).
      Only those land in `checkpoint/`, and the full intermediate PLYs are deleted.
    * `init_from: preview` starts from the splat the previous run of this stage left in
      `checkpoint/` (a phone's Refine sets it), on `init_schedule_scale` (1.0, the full
      schedule, measured best) of the schedule; with no usable seed it says why and trains
      as it would have.
      `init_seed.py` has the reasoning.
    * `cap_max: auto` sizes MCMC's gaussian cap to the capture: its supported surface in
      finest-view pixels, from the pose stage's sparse model, times `gaussian_density`
      (x `density_scale`, a phone's quality tier), clamped to `budget_floor` and to the
      smaller of the GPU's memory (the placed tier's, `gaussian_budget.GPU_MEMORY_GB`, or
      `gpu_memory_gb` when a run gives one; at the training frames' real size) and
      `budget_max`. An integer is an override, used as given. `gaussian_budget.py` has the
      formula, the calibration and the memory model; `train_metrics.json`'s `budget` has
      every input.
    * `converge` ends training when held-out PSNR has stopped improving after
      densification (`convergence.py`: `converge_every`, `converge_window`,
      `converge_min_gain_db`), and lets a budget above 1M lengthen the maximum schedule
      by up to 2x (`gaussian_budget.schedule_factor`). The trainer runs through
      `converge_trainer.py`, which ends it with the trainer's own save;
      `train_metrics.json`'s `convergence` has the held-out curve, the steps run of the
      maximum, and why it stopped.

    * `blocks` (`auto`, the default, or a count): a budget more than one GPU trains -- its
      raw count over the smaller of the budget's GPU-memory and `budget_max` ceilings --
      trains as that many blocks and merges them into one `trained.ply` (`blocks.py` has
      the recipe and its sources; `block_epsilon`, `block_min_images`, `block_margin`,
      `block_ring`, `block_ring_outer`, `block_blend`, `block_eval` and the `coarse_*`
      knobs are its parameters). Under a runner that fans out (`cloud.CloudRunner`) the
      blocks train on `block_parallel` (4) GPUs at once, one call each; otherwise one
      after another in this call. Each block's schedule follows its own frames
      (`block_schedule`: `frames`, the default, `share` or `full`). A count forces blocks
      on a capture that fits one GPU, to compare the two. `train_metrics.json`'s `blocks`
      has the partition, each block's cameras, budget, steps and time, and the merge;
      psnr/ssim/lpips are then the merged splat's, by the trainer's own `eval()` on the
      same held-out frames.
    * `batch_size` (1-8, default 1): images per training step, with gsplat's own
      sqrt(batch) learning-rate scaling and the schedule divided by it so the images
      trained on stay the same (`training.gsplat_argv`). Not with `depth_loss`.

    `psnr`, `ssim` and `lpips` are all measured on gsplat's held-out `val` split -- every
    8th registered frame -- and `train_metrics.json` says so and how many frames that is.
    """
    variant = str(ctx.param("variant", "3dgs"))
    if variant != "3dgs":
        raise ValueError(
            training.VARIANT_REFUSALS.get(variant, f"variant={variant!r} is not one of: 3dgs")
        )
    frames = ctx.input(FRAMES.name)
    poses = ctx.input(POSES.name)
    full_iterations = int(ctx.param("iterations", 30_000))
    trainer = training.trainer_script(ctx.param("trainer"))
    data_factor = int(ctx.param("data_factor", 1))
    max_side = _optional_int(ctx.param("train_max_side"))
    if max_side is not None and data_factor != 1:
        raise ValueError(
            f"train_max_side={max_side} and data_factor={data_factor} both shrink the "
            f"training images; give one. train_max_side works at any size, and gsplat's "
            f"parser rescales the intrinsics to match"
        )
    batch_size = training.check_batch_size(ctx.param("batch_size", 1))
    if batch_size > 1 and _optional_bool(ctx.param("depth_loss"), "depth_loss"):
        raise ValueError(
            f"batch_size={batch_size} and depth_loss cannot go together: each frame brings "
            f"its own number of SfM points to the depth loss, and v1.5.3's DataLoader "
            f"stacks a batch with the default collate, which refuses tensors of different "
            f"lengths. Give one of them"
        )
    # The region to train in: a support mask from a preview's quality stage (any shape the
    # well-supported data makes) wins over an ROI sphere; neither means the whole scene.
    mask = support_mask.SupportMask.parse(ctx.param("support_mask"))
    roi: training.Roi | support_mask.SupportMask | None = (
        mask if mask is not None else training.Roi.parse(ctx.param("roi"))
    )
    # Timed, because every call of a block run's fan-out does this again (`phases.py`).
    preamble = Phases()
    with preamble.phase("stageDataset"):
        dataset = training.build_dataset(frames, poses, ctx.work_dir / "dataset", max_side=max_side)
    # A shorter schedule for a smaller capture (`training.schedule_scale`), unless the
    # recipe leaves `schedule_full_at` out, which keeps the full one -- or unless the run
    # names its own `schedule_scale`, which wins over both.
    images = sum(1 for path in (dataset / "images").iterdir() if path.is_file())
    full_at = _optional_int(ctx.param("schedule_full_at"))
    computed = (
        training.schedule_scale(
            images, full_at=full_at, floor=float(ctx.param("schedule_floor", 0.25))
        )
        if full_at
        else 1.0
    )
    requested_scale = _optional_float(ctx.param("schedule_scale"))
    steps_scaler = (
        computed if requested_scale is None else training.check_schedule_scale(requested_scale)
    )
    iterations = training.scaled_steps(full_iterations, steps_scaler)
    ctx.log(
        f"gsplat: {images} frames -> {steps_scaler:g} of the {full_iterations}-step schedule "
        f"= {iterations} steps"
        + (
            ""
            if requested_scale is None
            else f" (schedule_scale given; the capture's own would be {computed:g})"
        )
    )
    train_size: tuple[int, int] | None = None
    if max_side is not None:
        first = min(p for p in (dataset / "images").iterdir() if p.is_file())
        train_size = _image_size(first)
        ctx.log(
            f"gsplat: training images at most {max_side} px on the long side "
            f"({train_size[0]}x{train_size[1]}); poses unchanged"
        )
    strategy = str(ctx.param("strategy", "default"))
    # The gaussian cap: `auto` measures the capture -- from the pose stage's own model,
    # so before the ROI crop below rewrites the dataset's points -- at the size the
    # trainer will really read its frames; an integer is used as given.
    requested_cap = gaussian_budget.parse_cap(ctx.param("cap_max"))
    frame_size, pixel_scale = _training_frame(dataset, data_factor)
    budgeting = time.monotonic()
    budget = gaussian_budget.plan(
        requested_cap,
        model=poses,
        train_size=frame_size,
        pixel_scale=pixel_scale,
        region_contains=_region_test(roi),
        outside_weight=1.0 / training.ROI_OUTSIDE_EVERY,
        density=float(ctx.param("gaussian_density", gaussian_budget.DEFAULT_DENSITY)),
        density_scale=float(ctx.param("density_scale", 1.0)),
        floor=int(ctx.param("budget_floor", gaussian_budget.DEFAULT_FLOOR)),
        budget_max=_optional_int(ctx.param("budget_max")),
        gpu_memory_gb=gaussian_budget.gpu_memory_for(ctx.gpu_tier, ctx.param("gpu_memory_gb")),
        images_per_step=batch_size,
    )
    preamble.add("budget", time.monotonic() - budgeting)
    cap_max = None if budget is None else budget.cap
    if budget is not None:
        ctx.log(f"gsplat: {gaussian_budget.describe(budget)}")
        if (
            budget.mode == "explicit"
            and budget.memory_ceiling is not None
            and budget.cap > budget.memory_ceiling
        ):
            ctx.log(
                f"gsplat: cap_max {budget.cap} is more than {budget.gpu_memory_gb:g} GB is "
                f"modelled to hold at {frame_size} ({budget.memory_ceiling}); used as given"
            )
    crop: training.RoiCrop | None = None
    if roi is not None:
        crop = training.crop_initial_points(dataset / "sparse" / "0", roi)
        where = (
            f"support mask ({roi.voxels} voxels of {roi.voxel:.4g})"
            if isinstance(roi, support_mask.SupportMask)
            else f"roi centre {roi.center} radius {roi.radius:g}"
        )
        ctx.log(
            f"gsplat: {where}: {crop.points_kept} of "
            f"{crop.points_in} initial points kept ({crop.inside} inside, "
            f"{crop.outside_sampled} sampled outside, {crop.kept_for_coverage} kept so "
            f"every frame still sees some)"
        )
    # Blocks (blocks.py): a budget more than one GPU trains -- or `blocks: <n>` -- trains as
    # blocks, on several GPUs at once under a runner that fans out, and merges them into
    # one trained.ply.
    requested_blocks = blocks.parse_blocks(ctx.param("blocks"))
    block_total, block_reason = blocks.block_count(requested_blocks, budget)
    role, _part = fanout_role(ctx.params)
    if role in ("part", "join") and block_total <= 1:
        raise ValueError(
            f"a fan-out {role} call of a stage that plans one block ({block_reason}); only "
            f"a block run fans out, so these params are not the head call's"
        )
    if block_total > 1:
        in_blocks = _train_in_blocks(
            ctx,
            blocks.Settings(
                frames=frames,
                poses=poses,
                dataset=dataset,
                region=roi,
                budget=budget,
                frame_size=frame_size,
                pixel_scale=pixel_scale,
                strategy=strategy,
                full_iterations=full_iterations,
                steps_scaler=_block_schedule(ctx, steps_scaler),
                data_factor=data_factor,
                trainer=trainer,
                python=training.trainer_python(ctx.param("python")),
                switches={
                    "antialiased": _optional_bool(ctx.param("antialiased"), "antialiased"),
                    "opacity_reg": _optional_float(ctx.param("opacity_reg")),
                    "depth_loss": _optional_bool(ctx.param("depth_loss"), "depth_loss"),
                    "pose_opt": _optional_bool(ctx.param("pose_opt"), "pose_opt"),
                    "app_opt": _optional_bool(ctx.param("app_opt"), "app_opt"),
                    "bilateral_grid": _optional_bool(ctx.param("bilateral_grid"), "bilateral_grid"),
                },
                extra=tuple(str(value) for value in (ctx.param("extra_args") or [])),
                converge=_optional_bool(ctx.param("converge"), "converge"),
                rule=convergence.Rule(
                    every=int(ctx.param("converge_every", convergence.DEFAULT_EVERY)),
                    window=int(ctx.param("converge_window", convergence.DEFAULT_WINDOW)),
                    min_gain_db=float(
                        ctx.param("converge_min_gain_db", convergence.DEFAULT_MIN_GAIN_DB)
                    ),
                ),
                live=ctx.param("live") is None or _optional_bool(ctx.param("live"), "live"),
                up=_up_estimate(poses),
                requested=requested_blocks,
                count=block_total,
                reason=block_reason,
                init_max_points=_optional_int(ctx.param("init_max_points")),
                holdout_error=_optional_bool(ctx.param("holdout_error"), "holdout_error"),
                holdout_script=_optional_path(ctx.param("holdout_script")),
                holdout_budget_s=float(ctx.param("holdout_budget_s", holdout.DEFAULT_BUDGET_S)),
                params=dict(ctx.params),
                images=images,
                schedule_full_at=full_at,
                schedule_floor=float(ctx.param("schedule_floor", 0.25)),
                schedule_requested=requested_scale is not None,
                block_schedule=_block_schedule_mode(ctx.param("block_schedule")),
                batch_size=batch_size,
                parallel=_block_parallel(ctx.param("block_parallel")),
                phases=preamble.to_dict(),
            ),
            images=images,
            requested_cap=requested_cap,
            requested_scale=requested_scale,
            max_side=max_side,
            train_size=train_size,
            variant=variant,
        )
        if in_blocks is not None:
            return in_blocks
    elif requested_blocks is None and budget is not None and budget.mode == "auto":
        ctx.log(f"gsplat: one block: {block_reason}")
    # A Refine starts from the preview's splat when it left a seed (init_seed.py).
    seeded, steps_scaler, iterations = _seed_from_preview(
        ctx, poses, dataset, roi, steps_scaler, full_iterations, cap_max
    )
    # Convergence: a held-out curve after densification, a stop when it is flat, and --
    # because the tail can now end early -- a longer maximum for a bigger budget.
    converge = _optional_bool(ctx.param("converge"), "converge")
    rule = convergence.Rule(
        every=int(ctx.param("converge_every", convergence.DEFAULT_EVERY)),
        window=int(ctx.param("converge_window", convergence.DEFAULT_WINDOW)),
        min_gain_db=float(ctx.param("converge_min_gain_db", convergence.DEFAULT_MIN_GAIN_DB)),
    )
    schedule_factor = 1.0
    if converge and strategy == "mcmc" and cap_max is not None:
        schedule_factor = gaussian_budget.schedule_factor(cap_max)
    if schedule_factor != 1.0:
        steps_scaler = round(steps_scaler * schedule_factor, 4)
        iterations = training.scaled_steps(full_iterations, steps_scaler)
        ctx.log(
            f"gsplat: {cap_max} gaussians -> a maximum schedule {schedule_factor:g}x as long "
            f"({steps_scaler:g} of {full_iterations} = {iterations} steps), ended early if "
            f"held-out PSNR goes flat"
        )
    # A batch of images a step: the same images trained on in that many times fewer steps
    # (`training.gsplat_argv`); refine window, evaluations and the convergence window all
    # follow the scaled steps because they are scaled by the same `--steps_scaler`.
    if batch_size > 1:
        steps_scaler = training.batch_steps_scaler(steps_scaler, batch_size)
        iterations = training.scaled_steps(full_iterations, steps_scaler)
        ctx.log(
            f"gsplat: {batch_size} images a step -> {iterations} steps "
            f"({steps_scaler:g} of {full_iterations}), learning rates x{batch_size**0.5:.3g} "
            f"by the trainer"
        )
    refine_stop = convergence.REFINE_STOP_ITER.get(strategy)
    extra_evals = (
        convergence.eval_steps(full_iterations, refine_stop, every=rule.every)
        if converge and refine_stop is not None
        else []
    )
    converge_status = _converge_status(
        converge, strategy, refine_stop, full_iterations, bool(extra_evals)
    )
    if converge:
        ctx.log(
            f"gsplat: convergence {converge_status}"
            + (
                ""
                if not extra_evals
                else f": {len(extra_evals)} extra held-out evaluations; after densification "
                f"(step {refine_stop} unscaled) it stops once PSNR gains under "
                f"{rule.min_gain_db:g} dB over {rule.window} steps"
            )
        )
    registered = _registered_count(poses, images)
    train_frames, val_frames = training.held_out_split(registered)
    ctx.log(
        f"gsplat: {registered} posed frames -> {train_frames} train, {val_frames} held out "
        f"(every {training.TEST_EVERY}th) for psnr/ssim/lpips"
    )
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
    antialiased = _optional_bool(ctx.param("antialiased"), "antialiased")
    depth_loss = _optional_bool(ctx.param("depth_loss"), "depth_loss")
    opacity_reg = _optional_float(ctx.param("opacity_reg"))
    pose_opt = _optional_bool(ctx.param("pose_opt"), "pose_opt")
    app_opt = _optional_bool(ctx.param("app_opt"), "app_opt")
    bilateral_grid = _optional_bool(ctx.param("bilateral_grid"), "bilateral_grid")
    # Intermediate splats for the live viewer, packed small into checkpoint/live/ so the
    # checkpoint syncer uploads them while the stage runs (`live.SplatWatch`).
    live_steps = (
        live.train_ply_steps(full_iterations)
        if ctx.param("live") is None or _optional_bool(ctx.param("live"), "live")
        else []
    )
    argv = training.gsplat_argv(
        training.trainer_python(ctx.param("python")),
        trainer,
        dataset,
        result,
        strategy=strategy,
        max_steps=full_iterations,
        data_factor=data_factor,
        steps_scaler=steps_scaler,
        cap_max=cap_max,
        antialiased=antialiased,
        opacity_reg=opacity_reg,
        depth_loss=depth_loss,
        pose_opt=pose_opt,
        app_opt=app_opt,
        bilateral_grid=bilateral_grid,
        live_steps=live_steps,
        eval_steps=extra_evals,
        batch_size=batch_size,
        extra=[str(value) for value in (ctx.param("extra_args") or [])],
    )
    if extra_evals:
        argv = training.converge_argv(
            argv,
            script=_optional_path(ctx.param("converge_script")) or CONVERGE_SCRIPT,
            every=rule.every,
            window=rule.window,
            min_gain_db=rule.min_gain_db,
        )
    with live.SplatWatch(
        result / "ply",
        ctx.checkpoint_dir / live.LIVE_DIR,
        ctx.log,
        indices=live.ply_indices(live_steps, steps_scaler),
        total=iterations,
        key_prefix=f"{ctx.checkpoint_key}/{live.LIVE_DIR}",
        up=_up_estimate(poses),
    ):
        ctx.run(argv)
    ply = training.latest_ply(result)
    if ply is None:
        raise ValueError(
            f"the trainer wrote no .ply under {result}; there is nothing to normalise "
            f"into {TRAINED_PLY.name}"
        )
    splat = gaussians.read_splat(ply)
    columns = splat.columns
    trained_count = splat.count
    cropped_rows = None
    if roi is not None:
        cropped_rows = training.crop_rows(columns, roi)
        columns, kept = training.crop_splat(columns, roi)
        ctx.log(
            f"gsplat: kept {kept} of {trained_count} trained gaussians inside the "
            + (
                "support mask"
                if isinstance(roi, support_mask.SupportMask)
                else f"roi ({training.ROI_KEEP_RADII:g} radii)"
            )
        )
    written = gaussians.write_ply(ctx.output(TRAINED_PLY.name), columns)
    in_ply = int(columns["x"].shape[0])
    # Per-gaussian error on the held-out frames, in trained.ply's order. After the splat
    # is written, and unable to fail the stage: `holdout.measure` says why.
    held_out = holdout.measure(
        ctx,
        enabled=_optional_bool(ctx.param("holdout_error"), "holdout_error"),
        python=training.trainer_python(ctx.param("python")),
        trainer=trainer,
        dataset=dataset,
        ply=ply,
        rows=trained_count,
        keep=cropped_rows,
        data_factor=data_factor,
        test_every=training.TEST_EVERY,
        antialiased=antialiased,
        script=_optional_path(ctx.param("holdout_script")),
        budget_s=float(ctx.param("holdout_budget_s", holdout.DEFAULT_BUDGET_S)),
    )
    _save_seed(ctx, splat.columns, poses, steps_scaler, iterations, seeded, cap_max)
    metrics_document = training.parse_metrics(
        result,
        ctx.log_path.read_text(encoding="utf-8", errors="replace"),
        trainer=f"gsplat:{trainer.name}",
        requested_iterations=iterations,
        resumed_from_step=None,
        attempts=ctx.attempt,
        registered=registered,
    )
    document = metrics_document.to_dict()
    # Read off the PLY rather than trusted from the stats file: this is the count the
    # artifact actually has, and the two disagreeing is worth being able to see.
    document["gaussiansInPly"] = in_ply
    document["masksIgnored"] = ctx.has_input(MASKS.name)
    document["gsplatVersion"] = training.GSPLAT_VERSION
    # Every knob this run turned, beside the numbers it produced, so two runs of one
    # capture can be compared without reading their recipes.
    document["settings"] = {
        "strategy": strategy,
        "capMax": cap_max,
        "capMaxRequested": requested_cap,
        "scheduleScale": steps_scaler,
        "scheduleFactor": schedule_factor,
        "converge": converge,
        "scheduleScaleRequested": requested_scale,
        "trainMaxSide": max_side,
        "trainImageSize": None if train_size is None else list(train_size),
        "antialiased": antialiased,
        "opacityReg": opacity_reg,
        "depthLoss": depth_loss,
        "poseOpt": pose_opt,
        "appOpt": app_opt,
        "bilateralGrid": bilateral_grid,
        "batchSize": batch_size,
        "variant": variant,
        "initFrom": "sfm" if seeded is None else "preview",
    }
    document["holdout"] = {
        key: held_out[key]
        for key in ("status", "reason", "views", "meanPsnr", "gaussiansMeasured", "seconds")
        if key in held_out
    }
    document["init"] = None if seeded is None else seeded.to_dict()
    # The gaussian budget, every input to it, and the bound that applied.
    document["budget"] = None if budget is None else budget.to_dict()
    # How long it trained and why: the held-out curve, the steps run of the maximum, and
    # whether (and why) the convergence rule ended it early.
    curve = convergence.read_curve(result / "stats")
    report = convergence.read_report(result) if extra_evals else None
    stopped_early = bool(report and report.get("stoppedEarly"))
    document["convergence"] = {
        "enabled": converge,
        "status": converge_status,
        "rule": rule.to_dict() if converge else None,
        "refineStopIter": (
            None if refine_stop is None else training.scaled_steps(refine_stop, steps_scaler)
        ),
        "stepsMax": iterations,
        "stepsRun": metrics_document.iterations,
        "stoppedEarly": stopped_early,
        "reason": _stop_reason(report),
        "hook": report,
        "curve": curve,
    }
    document["roi"] = (
        None
        if roi is None or crop is None
        else {
            **(
                {"kind": "voxels", "voxels": roi.voxels, "voxel": roi.voxel, "dims": list(roi.dims)}
                if isinstance(roi, support_mask.SupportMask)
                else roi.to_dict()
            ),
            **crop.to_dict(),
            "keepRadii": training.ROI_KEEP_RADII,
            "gaussiansTrained": trained_count,
            "gaussiansKept": in_ply,
        }
    )
    _write_json(ctx.output(TRAIN_METRICS.name), document)
    ctx.log(
        f"gsplat: {in_ply} gaussians from {ply.name} -> {TRAINED_PLY.name} "
        f"({written} bytes); metrics from {metrics_document.source}"
    )
    metrics: dict[str, MetricValue] = {
        "gaussians": in_ply,
        "trainedBytes": written,
        "requestedIterations": iterations,
        "scheduleScale": steps_scaler,
        "metricsSource": metrics_document.source,
        # Quality below is on the held-out split, never the training frames.
        "metricsSplit": f"val: every {training.TEST_EVERY}th frame",
        "valFrames": val_frames,
        "trainFrames": train_frames,
        "initFrom": "sfm" if seeded is None else "preview",
    }
    if seeded is not None:
        metrics["seedPoints"] = seeded.seeded
    if budget is not None:
        metrics["gaussianBudget"] = budget.cap
        metrics["budgetMode"] = budget.mode
        metrics["budgetClamp"] = budget.clamp or "none"
        if budget.mode == "auto":
            metrics["budgetFootprints"] = round(budget.footprints)
            metrics["budgetVoxels"] = budget.inside.voxels
    metrics["stepsMax"] = iterations
    if metrics_document.iterations is not None:
        metrics["stepsRun"] = metrics_document.iterations
    if converge:
        metrics["stoppedEarly"] = stopped_early
        metrics["heldOutEvaluations"] = len(curve)
    if metrics_document.iterations is not None:
        metrics["iterations"] = metrics_document.iterations
    for name, value in (
        ("psnr", metrics_document.psnr),
        ("ssim", metrics_document.ssim),
        ("lpips", metrics_document.lpips),
        ("trainSeconds", metrics_document.train_seconds),
        # Only with a bilateral grid: the same val frames after a per-image colour fit.
        ("ccPsnr", metrics_document.cc_psnr),
        ("ccSsim", metrics_document.cc_ssim),
        ("ccLpips", metrics_document.cc_lpips),
    ):
        if value is not None:
            metrics[name] = value
    if max_side is not None:
        metrics["trainMaxSide"] = max_side
    if roi is not None:
        metrics["gaussiansTrained"] = trained_count
    metrics["holdoutError"] = str(held_out.get("status"))
    if batch_size > 1:
        metrics["batchSize"] = batch_size
    return StageOutcome(metrics=metrics, summary=f"{in_ply} gaussians trained")


#: The wrapper that ends a converged run (`convergence.py`), run with the trainer's own
#: interpreter in place of the trainer; beside this file, as `holdout_error.py` is.
CONVERGE_SCRIPT = Path(__file__).resolve().parent / "converge_trainer.py"


def _block_schedule(ctx: StageContext, steps_scaler: float) -> float:
    """The run's schedule, which each block's own is worked out from
    (`blocks.block_schedule`). Each block starts from the prior as a Refine starts from the
    preview's seed, so `init_from: preview` gives it the same `init_schedule_scale`
    (init_seed.py) the one-block Refine it is compared with gets."""
    if str(ctx.param("init_from", "sfm")) == "preview":
        return training.check_schedule_scale(
            init_seed.schedule_for(_optional_float(ctx.param("init_schedule_scale")))
        )
    return steps_scaler


def _block_schedule_mode(value: object) -> str:
    mode = "frames" if value is None else str(value)
    if mode not in blocks.BLOCK_SCHEDULES:
        raise ValueError(
            f"block_schedule must be one of {', '.join(blocks.BLOCK_SCHEDULES)}, not {value!r}"
        )
    return mode


def _block_parallel(value: object) -> int:
    """`block_parallel`: blocks at once under a fanning runner; 1 trains them in turn."""
    if value is None:
        return blocks.BLOCK_PARALLEL
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"block_parallel must be a positive integer, not {value!r}")
    try:
        number = int(float(value))
    except ValueError:
        raise ValueError(f"block_parallel must be a positive integer, not {value!r}") from None
    if number < 1 or number != float(value):
        raise ValueError(f"block_parallel must be a positive integer, not {value!r}")
    return number


def _train_in_blocks(
    ctx: StageContext,
    settings: blocks.Settings,
    *,
    images: int,
    requested_cap: int | str | None,
    requested_scale: float | None,
    max_side: int | None,
    train_size: tuple[int, int] | None,
    variant: str,
) -> StageOutcome | None:
    """`blocks.py`'s run, written as the stage's outputs; None if it came to one block.

    Under a fanning runner (`contracts.FANOUT_PARAM`) this is one of three calls: the head
    plans and, with two or more blocks to train and `block_parallel` above 1, answers with
    a part per block (`blocks.fan_out`) and no outputs; a part trains its one block into
    `checkpoint/`; the join merges and measures, which is what every other call ends in.
    """
    role, part = fanout_role(ctx.params)
    # Where this call's seconds go (`phases.py`): the stage's dataset and budget, the
    # plan (the head renders the prior for the camera test here), then the role's work.
    call = Phases()
    call.update(settings.phases)
    with call.phase("prepare"):
        plan = blocks.prepare(ctx, settings, role=role)
    if plan is None:
        if role in ("part", "join"):
            raise ValueError(f"blocks: the plan came to one block, so there is no {role}")
        return None
    if role == "part":
        index = blocks.part_index(part, plan)
        record = blocks.train_part(ctx, settings, plan, index, phases=call)
        return StageOutcome(
            metrics={
                "block": index,
                "blockSteps": int(record["stepsRun"] or 0),
                "blockSeconds": float(record["seconds"]),
                "blockGaussians": int(record["gaussiansKept"]),
                "callPhases": call.flat(),
            },
            summary=f"block {index + 1} of {plan.partition.count} trained",
        )
    if role == "head":
        todo = blocks.pending(ctx, plan)
        if settings.parallel > 1 and len(todo) > 1:
            spec = blocks.fan_out(settings, plan, todo)
            ctx.log(
                f"blocks: {len(todo)} of {plan.partition.count} blocks to train, "
                f"{min(settings.parallel, len(todo))} at a time on their own GPUs "
                f"({', '.join(p.id for p in spec.parts)}, longest first)"
            )
            return StageOutcome(
                metrics={
                    FANOUT_METRIC: spec.to_json(),
                    "blocks": plan.partition.count,
                    "callPhases": call.flat(),
                },
                summary=f"{len(todo)} blocks to train in parallel",
            )
    run = blocks.train(ctx, settings, plan, ctx.output(TRAINED_PLY.name), phases=call)
    registered = _registered_count(settings.poses, images)
    train_frames, val_frames = training.held_out_split(registered)
    split = training.TrainMetrics(
        trainer=f"gsplat:{settings.trainer.name}", split=(train_frames, val_frames)
    ).to_dict()
    evaluated = run.document.get("evaluationMetrics")
    # psnr/ssim/lpips: the merged splat on the run's own held-out frames, by gsplat's
    # eval() -- the numbers a single run reports, so the two are comparable.
    document: dict[str, Any] = dict(evaluated) if isinstance(evaluated, dict) else dict(split)
    document["heldOut"] = split["heldOut"]
    document["attempts"] = ctx.attempt
    document["gaussians"] = run.gaussians
    document["gaussiansInPly"] = run.gaussians
    document["masksIgnored"] = ctx.has_input(MASKS.name)
    document["gsplatVersion"] = training.GSPLAT_VERSION
    record = run.document["blocks"]
    caps = [block["capMax"] for block in record["blocks"]]
    document["settings"] = {
        "strategy": settings.strategy,
        "capMax": None if any(cap is None for cap in caps) else sum(caps),
        "capMaxRequested": requested_cap,
        "scheduleScale": settings.steps_scaler,
        "scheduleFactor": None,
        "converge": settings.converge,
        "scheduleScaleRequested": requested_scale,
        "trainMaxSide": max_side,
        "trainImageSize": None if train_size is None else list(train_size),
        "antialiased": settings.switches.get("antialiased"),
        "opacityReg": settings.switches.get("opacity_reg"),
        "depthLoss": settings.switches.get("depth_loss"),
        "poseOpt": settings.switches.get("pose_opt"),
        "appOpt": settings.switches.get("app_opt"),
        "bilateralGrid": settings.switches.get("bilateral_grid"),
        "variant": variant,
        "initFrom": "prior",
        "blocks": record["count"],
    }
    held_out = run.document["holdoutSummary"]
    document["holdout"] = {
        key: held_out[key]
        for key in ("status", "reason", "views", "meanPsnr", "gaussiansMeasured", "seconds")
        if key in held_out
    }
    document["init"] = {"from": "prior", "prior": record["prior"]}
    document["budget"] = None if settings.budget is None else settings.budget.to_dict()
    steps_max = sum(int(block["stepsMax"]) for block in record["blocks"])
    steps_run = sum(int(block["stepsRun"] or 0) for block in record["blocks"])
    stopped = any(bool(block["stoppedEarly"]) for block in record["blocks"])
    document["convergence"] = {
        "enabled": settings.converge,
        "status": "per block (blocks[].stoppedEarly)",
        "rule": settings.rule.to_dict() if settings.converge else None,
        "stepsMax": steps_max,
        "stepsRun": steps_run,
        "stoppedEarly": stopped,
    }
    document["roi"] = None
    document["blocks"] = record
    _write_json(ctx.output(TRAIN_METRICS.name), document)
    ctx.log(
        f"gsplat: {run.gaussians} gaussians from {record['count']} blocks -> "
        f"{TRAINED_PLY.name} ({run.written} bytes)"
    )
    metrics: dict[str, MetricValue] = {
        "gaussians": run.gaussians,
        "trainedBytes": run.written,
        "requestedIterations": training.scaled_steps(
            settings.full_iterations, settings.steps_scaler
        ),
        "scheduleScale": settings.steps_scaler,
        "metricsSource": "merged-eval" if isinstance(evaluated, dict) else "none",
        "metricsSplit": f"val: every {training.TEST_EVERY}th frame",
        "valFrames": val_frames,
        "trainFrames": train_frames,
        "initFrom": "prior",
        "stepsMax": steps_max,
        "stepsRun": steps_run,
        "holdoutError": str(held_out.get("status")),
        **run.metrics,
    }
    total_cap = document["settings"]["capMax"]
    if settings.budget is not None:
        metrics["gaussianBudget"] = total_cap if isinstance(total_cap, int) else settings.budget.cap
        metrics["budgetMode"] = settings.budget.mode
        metrics["budgetClamp"] = settings.budget.clamp or "none"
        if settings.budget.raw is not None:
            metrics["budgetRaw"] = settings.budget.raw
    if settings.converge:
        metrics["stoppedEarly"] = stopped
    for name in ("psnr", "ssim", "lpips"):
        value = document.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            metrics[name] = value
    if max_side is not None:
        metrics["trainMaxSide"] = max_side
    ctx.log("gsplat: a block run leaves no seed; the prior stays for a later block run")
    return StageOutcome(
        metrics=metrics,
        summary=f"{run.gaussians} gaussians trained in {record['count']} blocks",
    )


def _training_frame(dataset: Path, data_factor: int) -> tuple[tuple[int, int] | None, float | None]:
    """The size of the frames the trainer reads, and that over the posed camera's width.

    As v1.5.3's parser decides it: the first registered image by name against its own
    COLMAP camera, one ratio for every camera, and `data_factor` dividing it (the parser
    reads `images_<factor>/`). So frames normalize kept at 2,400 px, or shrank to the
    preview's 800, are budgeted -- and their memory ceiling modelled -- at that size.
    (None, None) when either side cannot be read, a hand-made or a test's placeholder
    dataset, which the budget reports as a fallback.
    """
    try:
        model = sfm.read_model(dataset / "sparse" / "0")
        cameras = {camera.id: camera for camera in model.cameras}
        for image in model.images:  # sorted by name, as the parser sorts them
            path = dataset / "images" / image.name
            camera = cameras.get(image.camera_id)
            if path.is_file() and camera is not None and camera.width > 0:
                width, height = _image_size(path)
                factor = max(1, data_factor)
                size = (max(1, round(width / factor)), max(1, round(height / factor)))
                return size, size[0] / camera.width
    except (OSError, ValueError, KeyError, IndexError, struct.error):
        return None, None
    return None, None


def _region_test(
    region: training.Roi | support_mask.SupportMask | None,
) -> Callable[[Any], Any] | None:
    """A Refine's region as the budget's membership test over (n, 3) points."""
    if region is None:
        return None
    return lambda xyz: training.in_region(region, xyz)


def _converge_status(
    converge: bool, strategy: str, refine_stop: int | None, full_iterations: int, active: bool
) -> str:
    if not converge:
        return "off"
    if refine_stop is None:
        return f"not applied: strategy {strategy!r} has no densification end this knows"
    if not active:
        return (
            f"not applied: densification runs to step {refine_stop} of this "
            f"{full_iterations}-step schedule, so there is no tail to end early"
        )
    return "on"


def _stop_reason(report: dict[str, object] | None) -> str | None:
    """Why the run ended where it did, from `converge.json`."""
    if report is None:
        return None
    if report.get("stoppedEarly"):
        return (
            f"held-out PSNR gained {report.get('gainDb')} dB over the last "
            f"{report.get('windowSteps')} steps, under the rule's minimum"
        )
    if not report.get("hooked"):
        return f"ran the full schedule: {report.get('reason')}"
    return f"ran the full schedule (last decision: {report.get('lastDecision')})"


def _registered_count(poses: Path, fallback: int) -> int:
    """How many frames the pose stage registered -- the ones gsplat's parser splits --
    from `poses.json`, or `fallback` when a hand-made model has no summary."""
    summary = poses / "poses.json"
    if summary.is_file():
        value = _read_json(summary).get("registered")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


def _up_estimate(poses: Path) -> list[float] | None:
    """`poses.json`'s camera-up estimate, in COLMAP's frame, or None."""
    summary = poses / "poses.json"
    if not summary.is_file():
        return None
    estimate = _read_json(summary).get("upEstimate")
    up = estimate.get("up") if isinstance(estimate, dict) else None
    if isinstance(up, list) and len(up) == 3 and all(isinstance(v, int | float) for v in up):
        return [float(v) for v in up]
    return None


def _seed_from_preview(
    ctx: StageContext,
    poses: Path,
    dataset: Path,
    region: training.Roi | support_mask.SupportMask | None,
    steps_scaler: float,
    full_iterations: int,
    cap_max: int | None,
) -> tuple[init_seed.SeedApplied | None, float, int]:
    """`init_from: preview`: add the last run's seed to the initial points, and shorten
    the schedule to `init_schedule_scale`. Unchanged when not asked or when no seed fits."""
    init_from = str(ctx.param("init_from", "sfm"))
    if init_from not in ("sfm", "preview"):
        raise ValueError(f"init_from must be sfm or preview, not {init_from!r}")
    iterations = training.scaled_steps(full_iterations, steps_scaler)
    if init_from == "sfm":
        return None, steps_scaler, iterations
    seed, why = init_seed.load(ctx.checkpoint_dir, poses)
    if seed is None:
        ctx.log(
            f"gsplat: init_from=preview, but {why}; starting from COLMAP's points on the "
            f"{steps_scaler:g} schedule instead"
        )
        return None, steps_scaler, iterations
    budget = init_seed.budget_for(cap_max, _optional_int(ctx.param("init_max_points")))
    applied = init_seed.apply(dataset / "sparse" / "0", seed, region, budget=budget)
    scaler = training.check_schedule_scale(
        init_seed.schedule_for(_optional_float(ctx.param("init_schedule_scale")))
    )
    iterations = training.scaled_steps(full_iterations, scaler)
    ctx.log(
        f"gsplat: init_from=preview: {applied.seeded} of the preview's {seed.count} visible "
        f"gaussians added to {applied.sfm_points} SfM points ({applied.inside} in the "
        f"region, {applied.outside_sampled} sampled outside, budget {budget}); schedule "
        f"{steps_scaler:g} -> {scaler:g} = {iterations} steps"
    )
    return applied, scaler, iterations


def _save_seed(
    ctx: StageContext,
    columns: dict[str, Any],
    poses: Path,
    steps_scaler: float,
    iterations: int,
    seeded: init_seed.SeedApplied | None,
    cap_max: int | None,
) -> None:
    """Leave this run's splat in `checkpoint/` for a later Refine. Never fails the run."""
    settings = {
        "scheduleScale": steps_scaler,
        "iterations": iterations,
        "capMax": cap_max,
        "trainMaxSide": _optional_int(ctx.param("train_max_side")),
        "initFrom": "sfm" if seeded is None else "preview",
    }
    try:
        count = init_seed.save(
            ctx.checkpoint_dir,
            columns,
            poses,
            settings=settings,
            sh_c0=gaussians.SH_C0,
            scratch=ctx.work_dir / "seed",
        )
    except (OSError, KeyError, ValueError) as error:
        ctx.log(f"gsplat: could not leave a seed for a later Refine: {error!r}")
        return
    ctx.log(f"gsplat: left a {count}-gaussian seed in checkpoint/ for a later Refine")
    # And the splat's shapes, as the prior a later block run partitions, seeds and rings
    # with (blocks.py) -- the Preview's, in the usual order of runs.
    try:
        prior = blocks.save_prior(
            ctx.checkpoint_dir,
            columns,
            poses,
            scratch=ctx.work_dir / "seed",
            source=f"a run of {iterations} steps, cap {cap_max}",
        )
    except (OSError, KeyError, ValueError) as error:
        ctx.log(f"gsplat: could not leave a prior for a later block run: {error!r}")
        return
    if prior is None:
        ctx.log(
            f"gsplat: no prior left for a later block run (over "
            f"{blocks.PRIOR_MAX_GAUSSIANS} gaussians); an earlier one, if any, stays"
        )
    else:
        ctx.log(f"gsplat: left a {prior}-gaussian prior in checkpoint/ for a block run")


def _optional_bool(value: object, name: str) -> bool:
    """A switch: unset is off, a JSON bool is itself, anything else is refused by name
    rather than read by truthiness -- `"false"` is a truthy string."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true or false, not {value!r}")


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
    optional_consumes=("poses", quality.GATED_PLY.name, quality.COVERAGE_PLY.name),
    produces=(CANONICAL_PLY, COVERAGE_ENU, lod_parents.PLACEMENT),
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

    When a `quality` stage ran, its `gated.ply` -- the same schema, less what the capture
    did not support -- is what is placed, so the footprint it recentres on is the
    footprint that is published. Its `coverage.ply` is moved by the very same similarity
    and recentring (`place_points`) into `coverage_enu.ply`, so a viewer can lay it over
    the splat; with no quality stage that file holds no points.

    A `georef.json` with no `frame` (written by `manual_placement`) is levelled by the
    camera-up estimate in `poses`, if there is one, and otherwise passed through with a
    warning that nothing levelled it.
    """
    georef = _read_json(ctx.input(GEOREF.name))
    source = quality.GATED_PLY.name if ctx.has_input(quality.GATED_PLY.name) else TRAINED_PLY.name
    # A chunk at a time (`splat_stream`), so the worker places a splat of any size in the
    # same memory; canonical.ply is byte-identical to transforming it whole.
    trained = splat_stream.open_splat(
        ctx.input(source), chunk=int(ctx.param("chunk_gaussians", splat_io.CHUNK))
    )
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
    georeferenced = splat_stream.Step(rotation, translation, scale)
    moved = [0.0, 0.0, 0.0]
    recentred: gaussians.Frame | None = None
    if frame.get("recentre"):
        # `gaussians.orient(placed, up_axis="z")` of the georeferenced splat.
        placed = splat_stream.orient_to(
            trained, ctx.output(CANONICAL_PLY.name), before=[georeferenced], up_axis="z"
        )
        recentred = placed.frame
        assert recentred is not None
        moved = [float(v) for v in recentred.translation]
    else:
        placed = splat_stream.transform_to(trained, ctx.output(CANONICAL_PLY.name), [georeferenced])
    written = placed.bytes
    coverage_points = _place_coverage(ctx, rotation, translation, scale, recentred)
    # The same similarity, written down, so the GPU stage that optimises the tileset's
    # parents (`optimise_lod`) can move the cameras exactly as the splat was moved.
    _write_json(
        ctx.output(lod_parents.PLACEMENT.name),
        lod_parents.placement(rotation, translation, scale, recentred),
    )
    low, high = placed.low, placed.high
    extent = _extent(low, high)
    ctx.log(
        f"placed {placed.count} gaussians from {source} by {frame.get('source')}: scale "
        f"{scale:g}, recentred by {', '.join(f'{v:.3f}' for v in moved)} m; extent "
        f"{extent['east']:.2f} x {extent['north']:.2f} x {extent['up']:.2f} m"
    )
    metrics: dict[str, MetricValue] = {
        "gaussians": placed.count,
        "canonicalBytes": written,
        "frameSource": str(frame.get("source")),
        "scale": scale,
        "extentUpM": round(extent["up"], 3),
        "source": source,
        "coveragePoints": coverage_points,
    }
    return StageOutcome(metrics=metrics, summary=f"placed by {frame.get('source')}")


def place_points(
    points: npt.ArrayLike,
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike | None,
    scale: float,
    recentred: gaussians.Frame | None,
) -> npt.NDArray[np.float32]:
    """The similarity `place_splat` applies to gaussian centres, applied to bare points.

    `gaussians.transform` moves a centre by `scale * R @ x + t`, and `orient`'s recentring
    is a second `R' @ x + t'`; this is the same two steps and nothing else, so a point
    that sat on a gaussian's centre before placement sits on it after.
    """
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    moved = scale * (xyz @ np.asarray(rotation, dtype=np.float64).T) + t
    if recentred is not None:
        moved = moved @ np.asarray(recentred.rotation, dtype=np.float64).T + recentred.translation
    return np.ascontiguousarray(moved, dtype=np.float32)


def _place_coverage(
    ctx: StageContext,
    rotation: npt.ArrayLike,
    translation: npt.ArrayLike | None,
    scale: float,
    recentred: gaussians.Frame | None,
) -> int:
    """`coverage.ply` into `coverage_enu.ply`; an empty one when no quality stage ran."""
    if ctx.has_input(quality.COVERAGE_PLY.name):
        xyz, tiers = quality.read_coverage(ctx.input(quality.COVERAGE_PLY.name))
        placed = place_points(xyz, rotation, translation, scale, recentred)
    else:
        placed, tiers = np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.uint8)
    quality.write_coverage(ctx.output(COVERAGE_ENU.name), placed, tiers)
    return int(tiers.shape[0])


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
    optional_consumes=(lod_parents.LOD_PARENTS.name,),
    produces=(SPLAT_TILES,),
    summary="pack every gaussian into a level-of-detail KHR_gaussian_splatting 3D Tileset",
)
def splat_tiles(ctx: StageContext) -> StageOutcome:
    """Real: `tools/captures/splat_tiles.convert`, which writes a level-of-detail hierarchy.

    Until the hierarchy this kept the `max_gaussians` most opaque (400k) in one tile and
    discarded the rest, so a trained scene of 500k+ lost detail at delivery and a bigger
    one lost more. Now every gaussian past `opacity_min` and the floater radius is packed,
    at most `tile_gaussians` to a tile, and how much of it is *drawn* is the viewer's
    budget (the phone's "Detail" choice, apps/web src/lib/detail.ts), not this stage's cut.

    The hierarchy is REPLACE with merged parents: the leaves hold every gaussian once, as
    trained, and each parent holds its subtree merged by Hierarchical 3DGS's moment
    matching, so a distant view is the whole scan at lower resolution rather than a
    thinned subset of it. The parents cost at most ~1/7 more storage (`storage_overhead`
    in the metrics). The packer reads `canonical.ply` in windows and sorts through disk,
    so this stage's memory does not grow with the scan (8M gaussians: under 1.5 GB).

    `SPLAT_TILES.required_members` still pins `tileset.json` and the root tile
    `splat.glb`, which every tileset has; child tiles are named by `tileset.json`. The
    fixture byte-identity gate (tools/captures tests/test_synthetic_tree.py) still covers
    what it writes: the committed tree is one tile, and its `splat.glb` did not change by
    a byte when merged parents arrived -- only `tileset.json`'s `refine` did.
    """
    georef = _read_json(ctx.input(GEOREF.name))
    for retired in ("max_gaussians", "geometric_error"):
        # A run queued before the hierarchy (or a Refine copying a preview's params) may
        # still carry these. Say so, rather than silently honouring or refusing them.
        if ctx.param(retired) is not None:
            ctx.log(f"{retired} is no longer read: every gaussian is packed, in tiles")
    parents = _optimised_parents(ctx)

    def pack(parents: ParentOverrides | None) -> dict[str, float | int]:
        return splat_tiles_convert(
            ctx.input(CANONICAL_PLY.name),
            ctx.output(SPLAT_TILES.name),
            lat=float(_number(georef.get("lat"))),
            lon=float(_number(georef.get("lon"))),
            height=float(_number(georef.get("height"))),
            opacity_min=float(ctx.param("opacity_min", 0.02)),
            tile_gaussians=int(ctx.param("tile_gaussians", TILE_GAUSSIANS)),
            parents=parents,
        )

    try:
        stats = pack(parents)
    except ParentOverrideError as problem:
        # Optimised for another tree: never drawn. The merged parents are what every
        # tileset had before `optimise_lod`, so the capture loses the optimisation only.
        ctx.log(f"WARNING: the optimised parents were not used ({problem}); packing merged")
        shutil.rmtree(ctx.output(SPLAT_TILES.name))
        stats = pack(None)
    ctx.log(
        f"packaged {stats['gaussians']} gaussians ({stats['dropped']} dropped) in "
        f"{stats['tiles']} tiles, {stats['depth']} levels deep, "
        f"extent {stats['extent_m']:.2f} m; merged parents add {stats['parent_gaussians']} "
        f"({float(stats['storage_overhead']):.1%})"
    )
    if stats.get("optimised_parent_gaussians"):
        ctx.log(f"drew {stats['optimised_parent_gaussians']} optimised parents (optimise_lod)")
    metrics: dict[str, MetricValue] = {key: value for key, value in stats.items()}
    return StageOutcome(metrics=metrics, summary=f"{stats['gaussians']} gaussians packaged")


def _optimised_parents(ctx: StageContext) -> ParentOverrides | None:
    """`optimise_lod`'s parents, when it ran and kept them; None otherwise (and why)."""
    if not ctx.has_input(lod_parents.LOD_PARENTS.name):
        return None
    folder = ctx.input(lod_parents.LOD_PARENTS.name)
    path = folder / lod_parents.PARENTS_FILE
    if not path.is_file():
        summary_path = folder / lod_parents.SUMMARY_FILE
        status = "missing"
        if summary_path.is_file():
            try:
                status = str(_read_json(summary_path).get("status"))
            except (OSError, ValueError, AttributeError):
                status = "unreadable"
        ctx.log(f"no optimised parents (optimise_lod: {status}); drawing the merged ones")
        return None
    try:
        return ParentOverrides.load(path)
    except (OSError, ValueError) as problem:
        ctx.log(f"WARNING: {path.name} is unreadable ({problem}); drawing the merged parents")
        return None


@stage_impl(
    "splat_thumbnail",
    consumes=("canonical.ply",),
    produces=(THUMBNAIL,),
    summary="an elevation view of the splat as a JPEG, for the sites list",
)
def splat_thumbnail(ctx: StageContext) -> StageOutcome:
    """The row of the plan's artifact table that says "the endpoint exists and nothing
    calls it". Now something does. Drawn a chunk at a time (`splat_stream.thumbnail`),
    the same JPEG as `gaussians.render_thumbnail`."""
    splat = splat_stream.open_splat(
        ctx.input(CANONICAL_PLY.name), chunk=int(ctx.param("chunk_gaussians", splat_io.CHUNK))
    )
    stats = splat_stream.thumbnail(
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
    # A chunk at a time (`splat_stream.ground_samples`): the same cells and heights as
    # `gaussians.ground_samples`, positions to float32 rounding.
    splat = splat_stream.open_splat(
        ctx.input(CANONICAL_PLY.name), chunk=int(ctx.param("chunk_gaussians", splat_io.CHUNK))
    )
    cell_m = float(ctx.param("cell_m", 2.0))
    percentile = float(ctx.param("percentile", 5.0))
    samples = splat_stream.ground_samples(
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
    packaged = _tileset_summary(tiles)
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
            "tiles": packaged["tiles"],
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
    optional_consumes=(
        "manifest.json",
        "thumbnail.jpg",
        "ground_samples.json",
        quality.QUALITY_JSON.name,
        COVERAGE_ENU.name,
    ),
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
        # The quality bar's verdict, for the phone's forecast and for Refine, which reads
        # the region of interest back out of it (COLMAP frame, the poses this run made).
        "quality": _quality_summary(ctx),
        # The tier-coloured point cloud in the splat's own frame, when there is one to
        # lay over it; the worker publishes it beside the thumbnail.
        "coverage": _coverage_name(ctx),
    }
    _write_json(ctx.output(REGISTRATION.name), document)
    return StageOutcome(metrics={"artifacts": len(tiles)})


#: What of `quality.json` travels to the API. Not the thresholds or the per-bin geometry:
#: those stay in the run's artifact, where the console's Outputs view reaches them.
_QUALITY_SUMMARY_KEYS = (
    "mode",
    "bar",
    "barApplied",
    "roi",
    "gaussians",
    "keepPct",
    "keepVerifiedPct",
    "contextPct",
    "heldOutPsnr",
    "views",
    "spreadDeg",
    "gsd",
    "tips",
)


def _quality_summary(ctx: StageContext) -> dict[str, object] | None:
    if not ctx.has_input(quality.QUALITY_JSON.name):
        return None
    document = _read_json(ctx.input(quality.QUALITY_JSON.name))
    return {key: document.get(key) for key in _QUALITY_SUMMARY_KEYS if key in document}


def _coverage_name(ctx: StageContext) -> str | None:
    if not ctx.has_input(COVERAGE_ENU.name):
        return None
    try:
        _, tiers = quality.read_coverage(ctx.input(COVERAGE_ENU.name))
    except ValueError:
        return None  # a stub's bytes, not a PLY: nothing to lay over the splat
    return COVERAGE_ENU.name if tiers.size else None


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_path(value: object) -> Path | None:
    return Path(str(value)) if value else None


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


def _best_reconstruction(
    attempt: Callable[[int, int], tuple[Path | None, int]],
    *,
    rounds: Sequence[Sequence[int]],
    enough: int,
) -> tuple[Path | None, list[dict[str, int | str]]]:
    """Map until `enough` frames register, keeping the best model seen.

    Why this exists, measured rather than supposed: the same 40 rendered frames that
    register 40/40 on one machine registered **2/40** on a GitHub runner in the first
    real Modal smoke. Feature extraction was identical (the same count on every frame);
    matching was not quite (343 matched pairs against 338), because geometric
    verification runs RANSAC across threads. The mapper then found "no good initial
    image pair" and closed on two frames. So a low result is retried: first the mapper
    with other seeds (cheap -- under a second on 40 frames, against 40 s of matching),
    then, `rounds` allowing, a fresh matching pass. Nothing is thrown away: the best
    model across every attempt is the one returned, and every attempt is reported.

    `rounds` is the seeds to map with after each matching pass, in order; `attempt` is
    told the round and does that round's matching before its first seed.
    """
    best: tuple[Path | None, int] = (None, -1)
    tries: list[dict[str, int | str]] = []
    for round_, seeds in enumerate(rounds):
        for seed in seeds:
            found, registered = attempt(round_, seed)
            tries.append({"match": round_, "seed": seed, "registered": registered})
            if found is not None and registered > best[1]:
                best = (found, registered)
            if best[1] >= enough:
                return best[0], tries
    return best[0], tries


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


def _tileset_summary(tiles: Path) -> dict[str, Any]:
    """`_glb_summary` over every tile `tileset.json` names: counts and bytes summed, boxes
    united.

    The root tile alone is only the coarsest level of a hierarchy -- a tenth of a 1M scene
    -- so reading `splat.glb` by itself, as this did when there was one tile, would have
    the manifest report a scan as a fraction of its size.
    """
    tileset = _read_json(tiles / "tileset.json")
    root = tileset.get("root")
    stack: list[object] = [root]
    summaries: list[dict[str, Any]] = []
    while stack:
        tile = stack.pop()
        if not isinstance(tile, dict):
            continue
        content = tile.get("content")
        if isinstance(content, dict) and isinstance(content.get("uri"), str):
            summaries.append(_glb_summary(tiles / content["uri"]))
        children = tile.get("children")
        if isinstance(children, list):
            stack.extend(children)
    if not summaries:
        raise ValueError(f"{tiles.name}/tileset.json names no tile content")
    return {
        "gaussians": sum(summary["gaussians"] for summary in summaries),
        "bytes": sum(summary["bytes"] for summary in summaries),
        "tiles": len(summaries),
        "bbox": {
            "min": [min(s["bbox"]["min"][axis] for s in summaries) for axis in range(3)],
            "max": [max(s["bbox"]["max"][axis] for s in summaries) for axis in range(3)],
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
