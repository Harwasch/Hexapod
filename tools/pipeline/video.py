"""Frames out of a capture: which ffmpeg, which frames, and what the container says.

Three findings from A0 are baked in here rather than left to whoever edits the stage:

* **the binary is `imageio_ffmpeg.get_ffmpeg_exe()`, never a system `ffmpeg`.** A0 #6
  established that `ubuntu-latest` has no ffmpeg, so a stage that resolves `ffmpeg` off
  `PATH` passes on a developer machine (this one has `/usr/bin/ffmpeg`) and fails in CI.
  `ffmpeg_exe()` is the only place the question is asked, and `tests/test_normalize.py`
  asserts the argv the stage actually ran points inside the wheel;
* **there is no `ffprobe`** in that wheel, so metadata is scraped out of `ffmpeg -i`'s
  stderr. That is not a workaround for a missing feature -- it is the only metadata
  source this project has, and `probe()` is written to be total: an unparseable line
  leaves a field `None` rather than raising;
* **sharpness selection is top-K, never a threshold.** A0 #6 measured
  variance-of-Laplacian at Spearman +0.979 against blur but Pearson +0.680, with a 101x
  within-clip dynamic range -- so the ranking transfers between scenes and no absolute
  cutoff does. `select_sharpest` therefore takes a count and cannot be given a cutoff.

iPhone location is read from **both** places it is written (A0's sizing note): the `mdta`
key `com.apple.quicktime.location.ISO6709` and the older `udta` `(c)xyz` atom, which
ffmpeg surfaces as the tag `location`. Which one a clip carries depends on the capture
path, so reading one of them is reading half the captures.
"""

from __future__ import annotations

import functools
import math
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import numpy.typing as npt
from PIL import Image

__all__ = [
    "ACCURACY_KEY",
    "HDR_TRANSFERS",
    "IMAGE_SUFFIXES",
    "TONEMAP_FILTER",
    "VIDEO_SUFFIXES",
    "VIDEO_TAG_KEYS",
    "Location",
    "Source",
    "VideoMeta",
    "copy_frames",
    "evenly_spaced",
    "extract_frames_argv",
    "ffmpeg_exe",
    "ffmpeg_filters",
    "ffmpeg_version",
    "grab_frame_argv",
    "laplacian_variance",
    "parse_iso6709",
    "pick_source",
    "probe",
    "probe_argv",
    "probe_text",
    "scale_filter",
    "select_sharpest",
    "select_sharpest_per_window",
    "sharpness",
    "summarise",
]

#: What `pick_source` recognises. Lower-case; the comparison lower-cases the suffix.
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")

#: The two keys an iPhone writes its location under, in the order they are preferred.
#: `location` is how ffmpeg reports the `(c)xyz` atom; `location-eng` is the same atom
#: with a language tag, which some writers emit instead.
LOCATION_KEYS: tuple[str, ...] = (
    "com.apple.quicktime.location.iso6709",
    "location",
    "location-eng",
)

#: How far off the iPhone says its fix may be, in metres (horizontal, one sigma as far as
#: Apple documents it at all). Stored lower-cased, as every scraped tag is.
ACCURACY_KEY = "com.apple.quicktime.location.accuracy.horizontal"
#: The QuickTime creation date: local time *with* its UTC offset, which is why it is
#: preferred over `creation_time` (the mvhd time, UTC, sometimes rewritten by an editor).
CREATIONDATE_KEY = "com.apple.quicktime.creationdate"
#: The phone's own 35 mm-equivalent focal length. UNVERIFIED: the key is documented by
#: third parties (Remotion's media-parser, an ExifTool forum thread), not by Apple, and a
#: hugin-devs report says it includes digital zoom. Read when present, never relied on.
FOCAL_35MM_KEY = "com.apple.quicktime.camera.focal_length.35mm_equivalent"
LENS_MODEL_KEY = "com.apple.quicktime.camera.lens_model"

#: The only container tags that leave this module (`VideoMeta.to_dict()["tags"]`). The
#: full scrape can carry device identifiers -- serial numbers, content identifiers, a
#: vendor id -- and `source_meta.json` travels into the manifest and the API, so the
#: subset is named rather than filtered.
VIDEO_TAG_KEYS: tuple[str, ...] = (
    "creation_time",
    "encoder",
    CREATIONDATE_KEY,
    ACCURACY_KEY,
    FOCAL_35MM_KEY,
    LENS_MODEL_KEY,
    "com.apple.quicktime.make",
    "com.apple.quicktime.model",
    "com.apple.quicktime.software",
    "com.android.version",
    "com.android.manufacturer",
    "com.android.model",
)

#: The transfer characteristics that mean HDR: HLG (what an iPhone records "HDR" video
#: in) and PQ (SMPTE ST 2084, Dolby Vision's base layer and HDR10). ffmpeg's names.
HDR_TRANSFERS: tuple[str, ...] = ("arib-std-b67", "smpte2084")

#: HLG or PQ to 8-bit BT.709, before frames are taken: the widely used zimg recipe --
#: linearise (`npl=100`: 100 nits nominal peak), convert primaries to BT.709 in float,
#: Hable's tonemap curve with no desaturation, then BT.709 transfer and matrix, limited
#: range, 4:2:0. `npl=100` is not tuned on any capture (BT.2408's 203-nit HLG reference
#: white is the obvious alternative). Verified 2026-10-19 with the wheel's ffmpeg 7.0.2 on
#: an HLG clip, a PQ clip and a clip with a -90 display rotation (autorotate still
#: applies after it). Without it, frames cut from an iPhone's HDR video are washed out and
#: flat -- the transfer is read as if it were BT.709.
TONEMAP_FILTER = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)
#: The filters `TONEMAP_FILTER` needs, checked against the binary before it is used.
TONEMAP_NEEDS: frozenset[str] = frozenset({"zscale", "tonemap"})
_FILTER_LINE_RE = re.compile(r"^\s*[TSC.|]{2,3}\s+(?P<name>[A-Za-z0-9_]+)\s+\S*->\S*")

#: The pixel formats a video stream's colour group can open with (`yuv420p10le(...)`).
_PIXEL_FORMAT_PREFIXES: tuple[str, ...] = ("yuv", "rgb", "bgr", "gbr", "nv", "p0", "gray")
#: ffmpeg's colour-space names (`av_color_space_name`). A lone token in the colour group
#: is one of these: ffmpeg prints a single name when the matrix, primaries and transfer
#: *enums* are equal, and only for these three do the three *names* coincide too -- a
#: lone `bt470bg` is primaries bt470bg with transfer gamma28, so it says nothing about
#: the transfer and is recorded as the matrix only.
_COLOR_SPACES: frozenset[str] = frozenset(
    {
        "rgb", "bt709", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco", "bt2020nc",
        "bt2020c", "smpte2085", "chroma-derived-nc", "chroma-derived-c", "ictcp",
    }
)  # fmt: skip
_SAME_NAME_SPACES: frozenset[str] = frozenset({"bt709", "smpte170m", "smpte240m"})

_TAG_RE = re.compile(r"^\s{4}([A-Za-z0-9_.\-]+)\s*:\s*(.*?)\s*$")
_DURATION_RE = re.compile(r"^\s*Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_STREAM_RE = re.compile(r"^\s*Stream #\d+:\d+.*?:\s*Video:\s*(?P<rest>.*)$")
_SIZE_RE = re.compile(r"(?<![\dx])(\d{2,5})x(\d{2,5})(?![\dx])")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+fps")
_ROTATION_RE = re.compile(r"^\s*(?:displaymatrix:\s*)?rotation of\s*(-?\d+(?:\.\d+)?)", re.I)
_VERSION_RE = re.compile(r"^ffmpeg version (\S+)")
# +DD.DDDD-DDD.DDDD+AAA.AAA/ -- signs are part of the numbers and are never optional.
_ISO6709_RE = re.compile(
    r"^(?P<lat>[+-]\d+(?:\.\d+)?)(?P<lon>[+-]\d+(?:\.\d+)?)(?P<alt>[+-]\d+(?:\.\d+)?)?/?$"
)


def ffmpeg_exe() -> str:
    """The one place this project decides what `ffmpeg` means.

    Always the binary inside the `imageio-ffmpeg` wheel. Never `shutil.which("ffmpeg")`:
    that is the call that makes a stage pass here and fail on `ubuntu-latest`.
    """
    return str(imageio_ffmpeg.get_ffmpeg_exe())


@functools.cache
def ffmpeg_filters() -> frozenset[str]:
    """The filter names the wheel's ffmpeg was built with (`ffmpeg -filters`)."""
    text = _run([ffmpeg_exe(), "-hide_banner", "-filters"])
    return frozenset(
        match.group("name") for line in text.splitlines() if (match := _FILTER_LINE_RE.match(line))
    )


def ffmpeg_version() -> str:
    """The version string of the binary above, for `source_meta.json`'s `tools`."""
    text = _run([ffmpeg_exe(), "-version"])
    match = _VERSION_RE.search(text)
    return match.group(1) if match else "unknown"


@dataclass(frozen=True)
class Location:
    """A coordinate a container carried, and which key it came out of.

    `source` is kept because the two keys are not interchangeable evidence: the `mdta`
    key is written by the iOS camera, the `(c)xyz` atom by almost everything else, and a
    manifest that says only "lat/lon" cannot tell a reader which.
    """

    lat: float
    lon: float
    altitude_m: float | None
    source: str
    #: The phone's own horizontal accuracy for this fix, in metres (`ACCURACY_KEY`), or
    #: None when it wrote none or wrote something that is not a positive number.
    accuracy_m: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "altitudeM": self.altitude_m,
            "source": self.source,
            "accuracyM": self.accuracy_m,
        }


@dataclass(frozen=True)
class VideoMeta:
    """What `ffmpeg -i` said, with every field optional because stderr is not an API."""

    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    rotation_deg: float | None = None
    make: str | None = None
    model: str | None = None
    software: str | None = None
    created_at: str | None = None
    location: Location | None = None
    tags: dict[str, str] = field(default_factory=dict)
    #: The video stream's colour description, as ffmpeg prints it after the codec:
    #: `yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67)` is pixel format, range, and
    #: matrix/primaries/transfer. None for whatever was not printed (or printed `unknown`).
    pixel_format: str | None = None
    color_range: str | None = None
    color_matrix: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    #: `com.apple.quicktime.creationdate` and the mvhd `creation_time`, verbatim.
    creation_date: str | None = None
    creation_time: str | None = None
    #: The phone's 35 mm-equivalent focal length (`FOCAL_35MM_KEY`, UNVERIFIED key).
    focal_35mm: float | None = None
    lens_model: str | None = None

    @property
    def is_hdr(self) -> bool:
        """An HLG or PQ transfer: frames taken from it straight are washed out and flat."""
        return self.color_transfer in HDR_TRANSFERS

    def started_at(self) -> datetime | None:
        """When recording started, timezone-aware, or None.

        The QuickTime creation date first (local time with its offset), else the mvhd
        `creation_time`. A time without a zone is refused rather than guessed: a naive
        time read as UTC is hours off for most of the world. That the creation date is
        the *start* of recording is an assumption (UNVERIFIED against Apple's docs).
        """
        return self._start()[0]

    def started_from(self) -> str | None:
        """Which tag `started_at` came from: `quicktime-creationdate`,
        `mvhd-creation-time`, or None."""
        return self._start()[1]

    def _start(self) -> tuple[datetime | None, str | None]:
        for raw, source in (
            (self.creation_date, "quicktime-creationdate"),
            (self.creation_time, "mvhd-creation-time"),
        ):
            parsed = _aware_time(raw)
            if parsed is not None:
                return parsed, source
        return None, None

    def to_dict(self) -> dict[str, object]:
        return {
            "durationS": self.duration_s,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "codec": self.codec,
            "rotationDeg": self.rotation_deg,
            "make": self.make,
            "model": self.model,
            "software": self.software,
            "createdAt": self.created_at,
            "location": None if self.location is None else self.location.to_dict(),
            "pixelFormat": self.pixel_format,
            "colorRange": self.color_range,
            "colorMatrix": self.color_matrix,
            "colorPrimaries": self.color_primaries,
            "colorTransfer": self.color_transfer,
            "isHdr": self.is_hdr,
            "creationDate": self.creation_date,
            "creationTime": self.creation_time,
            "focal35mm": self.focal_35mm,
            "lensModel": self.lens_model,
            "tags": {key: self.tags[key] for key in VIDEO_TAG_KEYS if key in self.tags},
        }


@dataclass(frozen=True)
class Source:
    """What the upload turned out to be: one video, or a folder of stills."""

    kind: str
    path: Path
    images: tuple[Path, ...] = ()

    @property
    def is_video(self) -> bool:
        return self.kind == "video"


class SourceError(ValueError):
    """The upload holds nothing this stage can turn into frames."""


def pick_source(upload: Path) -> Source:
    """A video if there is one, otherwise a folder of stills; refuse anything else.

    A video wins over stills that sit beside it, because the one capture path that
    produces both is a phone that also wrote a poster frame.
    """
    files = sorted(p for p in upload.rglob("*") if p.is_file())
    videos = [p for p in files if p.suffix.lower() in VIDEO_SUFFIXES]
    if videos:
        return Source(kind="video", path=videos[0])
    images = tuple(p for p in files if p.suffix.lower() in IMAGE_SUFFIXES)
    if images:
        return Source(kind="images", path=upload, images=images)
    seen = ", ".join(sorted({p.suffix.lower() or "<none>" for p in files})) or "nothing"
    raise SourceError(
        f"{upload} holds no video ({', '.join(VIDEO_SUFFIXES)}) and no images "
        f"({', '.join(IMAGE_SUFFIXES)}); it holds {seen}"
    )


def probe_argv(source: Path) -> list[str]:
    """The argv that asks the container what it is. In one place so a stage can log it."""
    return [ffmpeg_exe(), "-hide_banner", "-i", str(source)]


def probe_text(source: Path) -> str:
    """`ffmpeg -i <source>` stderr, verbatim. Exits 1 with no output file; that is fine."""
    return _run(probe_argv(source))


def probe(source: Path) -> VideoMeta:
    """Scrape what the container says. Never raises on unexpected text."""
    return parse_probe(probe_text(source))


def parse_probe(text: str) -> VideoMeta:
    """The parser, separated from the subprocess so it can be tested on real stderr."""
    tags: dict[str, str] = {}
    duration: float | None = None
    width = height = None
    fps: float | None = None
    codec: str | None = None
    rotation: float | None = None
    colour: _Colour | None = None
    for line in text.splitlines():
        if duration is None and (match := _DURATION_RE.match(line)):
            hours, minutes, seconds = match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if (match := _ROTATION_RE.match(line)) and rotation is None:
            rotation = float(match.group(1))
        if stream := _VIDEO_STREAM_RE.match(line):
            rest = stream.group("rest")
            if codec is None:
                codec = rest.split()[0].strip(",")
            if width is None and (size := _SIZE_RE.search(rest)):
                width, height = int(size.group(1)), int(size.group(2))
            if fps is None and (rate := _FPS_RE.search(rest)):
                fps = float(rate.group(1))
            if colour is None:
                colour = _colour(rest)
        # Container and stream tags are indented four spaces under a `Metadata:` line.
        # Collected flat and first-wins, so a container-level tag is not overwritten by
        # a per-stream one that means something narrower.
        if (tag := _TAG_RE.match(line)) and tag.group(1).lower() not in tags:
            tags[tag.group(1).lower()] = tag.group(2)
    return VideoMeta(
        duration_s=duration,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        rotation_deg=rotation,
        make=_first(tags, ("com.apple.quicktime.make", "make", "manufacturer")),
        model=_first(tags, ("com.apple.quicktime.model", "model")),
        software=_first(tags, ("com.apple.quicktime.software", "encoder", "software")),
        created_at=_first(tags, ("com.apple.quicktime.creationdate", "creation_time", "date")),
        location=_location(tags),
        tags=tags,
        pixel_format=colour.pixel_format if colour else None,
        color_range=colour.range if colour else None,
        color_matrix=colour.matrix if colour else None,
        color_primaries=colour.primaries if colour else None,
        color_transfer=colour.transfer if colour else None,
        creation_date=tags.get(CREATIONDATE_KEY) or None,
        creation_time=tags.get("creation_time") or None,
        focal_35mm=_positive(tags.get(FOCAL_35MM_KEY)),
        lens_model=tags.get(LENS_MODEL_KEY) or None,
    )


@dataclass(frozen=True)
class _Colour:
    pixel_format: str
    range: str | None = None
    matrix: str | None = None
    primaries: str | None = None
    transfer: str | None = None


def _top_level(text: str) -> list[str]:
    """`text` split on the commas outside parentheses: `a (b, c), d(e, f), g` -> three."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def _colour(rest: str) -> _Colour | None:
    """The pixel format and colour description out of a video stream line, or None.

    The first top-level item that is a pixel-format word (`yuv420p10le`, `yuvj420p`,
    `nv12`, `p010le`, `gray`, ...) optionally followed by its parenthesised group. The
    codec's own brackets (`hevc (Main 10) (hvc1 / 0x31637668)`) come first and are a
    different item, so they are never read as one. Total: never raises.
    """
    for item in _top_level(rest)[1:]:
        match = re.fullmatch(r"(?P<pix>[a-z0-9_]+)(?:\((?P<inner>[^)]*)\))?", item)
        if match is None or not match.group("pix").startswith(_PIXEL_FORMAT_PREFIXES):
            continue
        colour = _Colour(pixel_format=match.group("pix"))
        values: dict[str, str | None] = {}
        for token in (match.group("inner") or "").split(","):
            token = token.strip()
            if token in ("tv", "pc"):
                values["range"] = token
            elif token.count("/") == 2:
                matrix, primaries, transfer = (_known(part) for part in token.split("/"))
                values.update(matrix=matrix, primaries=primaries, transfer=transfer)
            elif token in _COLOR_SPACES:
                values["matrix"] = token
                if token in _SAME_NAME_SPACES:
                    values.update(primaries=token, transfer=token)
        return _Colour(
            pixel_format=colour.pixel_format,
            range=values.get("range"),
            matrix=values.get("matrix"),
            primaries=values.get("primaries"),
            transfer=values.get("transfer"),
        )
    return None


def _known(name: str) -> str | None:
    name = name.strip()
    return None if not name or name in ("unknown", "reserved") else name


def _positive(raw: str | None) -> float | None:
    """A positive, finite number out of a tag, or None."""
    if raw is None:
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _aware_time(raw: str | None) -> datetime | None:
    """An ISO 8601 time with a zone (`Z`, `+01:00` or `+0100`), or None."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _first(tags: dict[str, str], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = tags.get(key)
        if value:
            return value
    return None


def _location(tags: dict[str, str]) -> Location | None:
    """Both iPhone location keys, in preference order. A0: read both, not one."""
    for key in LOCATION_KEYS:
        raw = tags.get(key)
        if not raw:
            continue
        parsed = parse_iso6709(raw)
        if parsed is not None:
            lat, lon, alt = parsed
            return Location(
                lat=lat,
                lon=lon,
                altitude_m=alt,
                source=key,
                accuracy_m=_positive(tags.get(ACCURACY_KEY)),
            )
    return None


def parse_iso6709(text: str) -> tuple[float, float, float | None] | None:
    """`+37.7749-122.4194+010.000/` -> (37.7749, -122.4194, 10.0).

    ISO 6709 signs are part of the numbers, which is what makes the string parseable
    without separators at all. The compact forms -- degrees and minutes `+DDMM.M`,
    degrees, minutes and seconds `+DDMMSS.S` -- are told apart by their count of integer
    digits and converted (`_iso6709_component`). Anything that does not match, or is not a
    point on the globe, returns None rather than a coordinate nobody can vouch for. The
    API's pre-run reader (`apps/api/app/services/video_location.py`) parses the same way,
    and a test there holds the two to one answer per string.
    """
    match = _ISO6709_RE.match(text.strip())
    if match is None:
        return None
    lat = _iso6709_component(match.group("lat"), 2)
    lon = _iso6709_component(match.group("lon"), 3)
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    altitude = match.group("alt")
    return lat, lon, None if altitude is None else float(altitude)


def _iso6709_component(text: str, degree_digits: int) -> float | None:
    """One signed ISO 6709 component in degrees, by its count of integer digits: up to
    `degree_digits` (2 for latitude, 3 for longitude) is decimal degrees; two more is
    degrees and decimal minutes (+DDMM.M); four more is degrees, minutes and decimal
    seconds (+DDMMSS.S). Any other count, or minutes or seconds of 60 or more, is None.
    Read as decimal degrees, +0030.0 (half a degree) was thirty."""
    sign = -1.0 if text[0] == "-" else 1.0
    whole, _, fraction = text[1:].partition(".")
    tail = f".{fraction}" if fraction else ""
    count = len(whole)
    if count <= degree_digits:
        return sign * float(text[1:])
    if count == degree_digits + 2:
        minutes = float(whole[degree_digits:] + tail)
        if minutes >= 60:
            return None
        return sign * (int(whole[:degree_digits]) + minutes / 60)
    if count == degree_digits + 4:
        minutes = int(whole[degree_digits : degree_digits + 2])
        seconds = float(whole[degree_digits + 2 :] + tail)
        if minutes >= 60 or seconds >= 60:
            return None
        return sign * (int(whole[:degree_digits]) + minutes / 60 + seconds / 3600)
    return None


def extract_frames_argv(
    source: Path,
    pattern: Path,
    *,
    fps: float,
    quality: int = 2,
    max_side: int | None = None,
    tonemap: str | None = None,
) -> list[str]:
    """The argv that turns a clip into numbered JPEGs. Built here so a test can read it.

    `-vsync vfr` with an `fps` filter is deliberately *not* used: constant-rate sampling
    is what makes "4 fps" mean the same number of frames for the same clip on any
    machine, which is what the frame count in `source_meta.json` is worth anything for.
    `tonemap` (`TONEMAP_FILTER` for an HDR clip) sits between the rate and the size, so
    only the frames kept are converted and they are scaled in 8-bit BT.709.
    """
    chain = [f"fps={fps:g}"]
    if tonemap:
        chain.append(tonemap)
    if max_side is not None:
        chain.append(scale_filter(max_side))
    return [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vf",
        ",".join(chain),
        "-q:v",
        str(quality),
        "-start_number",
        "0",
        str(pattern),
    ]


def scale_filter(max_side: int) -> str:
    """The ffmpeg filter that bounds the *long* side, whichever it is, to `max_side`.

    Aspect preserved, even dimensions (-2). Bounding the width alone, as this once did,
    left a portrait phone clip -- whose width is its short side -- at nearly full height.
    """
    m = max_side
    return f"scale=w='if(gte(iw,ih),min({m},iw),-2)':h='if(gte(iw,ih),-2,min({m},ih))'"


def grab_frame_argv(
    source: Path, at_s: float, target: Path, *, max_side: int, tonemap: str | None = None
) -> list[str]:
    """The argv that writes the one frame at `at_s` seconds, losslessly, to `target`.

    `-ss` before `-i` seeks the demuxer to the keyframe before `at_s` and decodes only
    from there, so sampling eight frames across a two-minute 4K clip decodes eight GOPs
    rather than the whole clip. PNG, not JPEG: the frames are measured for fine detail
    (`resolution.py`), and a JPEG's own quantisation is exactly the band being measured.
    `tonemap` comes first, so `max_side: auto` measures the frames training will see.
    """
    chain = [tonemap, scale_filter(max_side)] if tonemap else [scale_filter(max_side)]
    return [
        ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{max(0.0, at_s):.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        ",".join(chain),
        str(target),
    ]


def sharpness(path: Path) -> float:
    """Variance of the Laplacian of the luminance, at full resolution.

    A0 #6 ranked blur with this at Spearman +0.979 and Pearson +0.680: the order is
    right, the values are not linear in anything, and the within-clip range was 101x.
    That is exactly the evidence for ranking and against a cutoff -- see
    `select_sharpest`, which cannot be given one.
    """
    with Image.open(path) as image:
        grey = np.asarray(image.convert("L"), dtype=np.float64)
    return laplacian_variance(grey)


def laplacian_variance(grey: npt.NDArray[np.floating[Any]]) -> float:
    """`sharpness` of a frame already decoded to grey, for a caller that needs the pixels
    for something else too (`select: viewpoint` measures motion on the same decode)."""
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    grey = grey.astype(np.float64, copy=False)
    laplacian = (
        4.0 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    )
    return float(laplacian.var())


def select_sharpest(scores: Sequence[float], keep: int) -> tuple[int, ...]:
    """The indices of the `keep` sharpest, returned in their original (temporal) order.

    Top-K and nothing else: there is no threshold parameter to pass, because A0 measured
    that no absolute cutoff transfers between scenes. Ties break on the earlier index, so
    two runs over the same frames select the same frames.
    """
    if keep <= 0:
        return ()
    if keep >= len(scores):
        return tuple(range(len(scores)))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return tuple(sorted(order[:keep]))


def select_sharpest_per_window(scores: Sequence[float], keep: int) -> tuple[int, ...]:
    """The sharpest frame of each of `keep` equal stretches of the clip, in temporal order.

    Still top-K by rank and still no cutoff -- the rank is taken within a window rather
    than across the clip. Global top-K is right when it keeps most candidates; when it
    keeps a quarter of them, a stretch of motion blur (walking faster round one side of
    the object) can lose every frame of that side, and COLMAP cannot register what it
    was never given. Windows are `evenly_spaced`'s boundaries; ties break on the earlier
    index, as `select_sharpest`'s do.
    """
    count = len(scores)
    if keep <= 0 or count == 0:
        return ()
    if keep >= count:
        return tuple(range(count))
    edges = [round(i * count / keep) for i in range(keep + 1)]
    return tuple(
        max(range(start, end), key=lambda i: (scores[i], -i))
        for start, end in pairwise(edges)
        if end > start
    )


def evenly_spaced(count: int, keep: int) -> tuple[int, ...]:
    """`keep` indices spread across `count`, endpoints included. The `select: all` path.

    Used when the recipe asks for no sharpness ranking but the clip is longer than the
    frame budget: dropping the tail of a capture loses a whole side of the object,
    whereas thinning it loses resolution evenly.
    """
    if keep <= 0 or count <= 0:
        return ()
    if keep >= count:
        return tuple(range(count))
    if keep == 1:
        return (0,)
    step = (count - 1) / (keep - 1)
    return tuple(sorted({round(i * step) for i in range(keep)}))


def copy_frames(
    sources: Sequence[Path],
    out_dir: Path,
    *,
    stem: str = "frame",
    max_side: int | None = None,
) -> tuple[Path, ...]:
    """Copy the selected frames into `out_dir`, renumbered from zero in order.

    Renumbered rather than keeping the extraction numbers: the frame *set* is the
    artifact, and a gap in it would make the numbering mean "when in the clip" for the
    pose stage, which is a fact it must not be able to read off a filename.

    `max_side` shrinks a frame whose long side is bigger. An iPhone photo is 5712x4284;
    training on that ran 30,000 gsplat steps on 24 MP images, many times the work of the
    same scene at 1600 px for no detail a splat can hold. The EXIF block is carried
    across unchanged, because `georeference` reads each frame's GPS from it and the focal
    length in 35 mm terms is independent of the pixel count.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, source in enumerate(sources):
        target = out_dir / f"{stem}_{index:04d}{source.suffix.lower()}"
        if max_side is None or not _shrink(source, target, max_side):
            shutil.copyfile(source, target)
        written.append(target)
    return tuple(written)


def _shrink(source: Path, target: Path, max_side: int) -> bool:
    """Write `source` into `target` with its long side at most `max_side`.

    False (and nothing written) when it is already small enough, so the caller copies the
    bytes untouched rather than re-encoding a frame that did not need it.
    """
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size
        if max(width, height) <= max_side:
            return False
        scale = max_side / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        exif = image.info.get("exif")
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        options: dict[str, object] = {"quality": 95}
        if exif:
            options["exif"] = exif
        kind = "JPEG" if target.suffix in {".jpg", ".jpeg"} else None
        resized.save(target, format=kind, **options)
    return True


def _run(argv: Sequence[str]) -> str:
    """Run a tool and return stdout+stderr. `ffmpeg -i` exits 1 by design, so no check."""
    completed = subprocess.run(  # noqa: S603 - fixed argv from this module, never a shell
        list(argv), capture_output=True, text=True, check=False
    )
    return completed.stdout + completed.stderr


def summarise(values: Sequence[float]) -> dict[str, float | None]:
    """min/median/max of a score list, or nulls -- what `source_meta.json` records."""
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return {
        "min": _round(ordered[0]),
        "median": _round(median),
        "max": _round(ordered[-1]),
    }


def _round(value: float) -> float:
    return round(value, 4) if math.isfinite(value) else 0.0
