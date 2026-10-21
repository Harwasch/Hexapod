"""Where a phone says its video was made, read from the container's own location atoms.

A capture joins an existing site by where it was made (decision D3), and that is decided
before its run starts -- the pipeline places the capture in the site's frame -- so before
`tools/pipeline/video.py` has scraped the clip with ffmpeg. The browser's fix at upload
time is often the wrong answer: a video recorded on a ranch and uploaded from home is
kilometres from the site it shows. So the worker reads the video's own location first
(`worker/video_gps.py`), and this module is how: a reader of the QuickTime / ISO base
media box tree that walks the top-level boxes by their sizes, then `moov`'s children,
and reads only two of them whole --

* `moov/meta`, the QuickTime metadata the iOS camera writes: `hdlr` `mdta`, `keys`,
  `ilst`, with `com.apple.quicktime.location.ISO6709` and
  `com.apple.quicktime.location.accuracy.horizontal`;
* `moov/udta/(c)xyz`, the older user-data atom most other writers (Android, ffmpeg's
  QuickTime muxer) use: a 16-bit length, a 16-bit language code, then the ISO 6709
  string; and `moov/udta/loci`, the 3GPP location box ffmpeg's MP4 muxer writes, which
  ffmpeg reads back as the same `location` tag.

The places, in the order, that the pipeline reads (`video.LOCATION_KEYS`; A0's sizing
note: reading one of them is reading half the captures). Nothing is decoded
and the `mdat` -- the gigabytes -- is skipped by its size (the top-level walk reads box
headers only, so not one byte of it is fetched): a 12 GB clip costs a handful of
ranged GETs, 8 bytes each at the top level and at most 64 KiB inside `moov`. Pure
Python, no dependency: ffprobe is not on the worker (the pipeline's ffmpeg is a wheel in
another project) and would need the whole file local.

Total: anything that is not a well-formed container, or a location that is not a point
on the globe, is `None`, never an exception; reads are capped (`MAX_READS`, `MAX_READ_BYTES`,
`MAX_METADATA_BYTES`, `MAX_BOXES`) so a crafted file costs no more than an honest one.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.storage import ObjectStorage

__all__ = [
    "QUICKTIME_SUFFIXES",
    "VIDEO_SUFFIXES",
    "ReadAt",
    "VideoLocation",
    "location_in_storage",
    "location_of_bytes",
    "read_location",
]

#: `(offset, length) -> bytes`: at most `length` bytes from `offset`, fewer at the end.
ReadAt = Callable[[int, int], bytes]

#: What the pipeline takes as a video (`tools/pipeline/video.VIDEO_SUFFIXES`), and the ones
#: of them that are QuickTime / ISO base media files with location atoms to read.
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
QUICKTIME_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v")

#: The iOS camera's keys, compared lower-cased (ffmpeg lower-cases them too).
ISO6709_KEY = "com.apple.quicktime.location.iso6709"
ACCURACY_KEY = "com.apple.quicktime.location.accuracy.horizontal"
XYZ = b"\xa9xyz"
LOCI = b"loci"
#: `VideoLocation.key`'s values, by which atom the point came out of.
KEYS: tuple[str, ...] = ("iso6709", "xyz", "loci")

#: Bytes fetched per read: one GET covers a box header and, usually, its neighbours.
CHUNK = 64 * 1024
#: The most reads one file may cost. An iPhone clip takes three or four (the start, the
#: `moov` header, each `trak` skipped, the `meta`); a file that needs more is not one.
MAX_READS = 64
#: The most bytes one file may cost, all reads together: a few metadata boxes' worth. A
#: moov of many boxes each under `MAX_METADATA_BYTES` stops here, not at 64 of them.
MAX_READ_BYTES = 4 * 1024 * 1024
#: The most a `meta` or `udta` box may be and still be read whole: an iPhone's is under a
#: kilobyte (a few dozen keys). A larger one is skipped, not fetched.
MAX_METADATA_BYTES = 1024 * 1024
#: The most boxes walked at one level, so a file of a million empty boxes ends quickly.
MAX_BOXES = 1024
#: The most `keys` entries read.
MAX_KEYS = 1024

# +DD.DDDD-DDD.DDDD+AAA.AAA/ -- the grammar `tools/pipeline/video.parse_iso6709` accepts.
_ISO6709_RE = re.compile(
    r"^(?P<lat>[+-]\d+(?:\.\d+)?)(?P<lon>[+-]\d+(?:\.\d+)?)(?P<alt>[+-]\d+(?:\.\d+)?)?/?$"
)
#: QuickTime well-known data types (the `data` atom's type indicator).
_UTF8, _SIGNED, _UNSIGNED, _FLOAT32, _FLOAT64 = 1, 21, 22, 23, 24


@dataclass(frozen=True)
class VideoLocation:
    """A point the video's container carried, its horizontal accuracy, and which atom.

    `key` is `iso6709` (the iOS camera's `mdta` key), `xyz` (the `udta` atom) or `loci`
    (the 3GPP box), because they are not the same evidence: only the first comes with an
    accuracy.
    """

    lat: float
    lon: float
    #: The phone's own horizontal accuracy for the fix, in metres; None when it wrote none
    #: or wrote something that is not a positive number.
    accuracy_m: float | None
    key: str

    def to_metadata(self) -> dict[str, Any]:
        """The shape kept on the capture (`metadata.videoLocation`)."""
        return {"lat": self.lat, "lon": self.lon, "accuracyM": self.accuracy_m, "key": self.key}

    @staticmethod
    def from_metadata(value: object) -> VideoLocation | None:
        """`metadata.videoLocation` read back, checked again; None for anything else."""
        if not isinstance(value, dict):
            return None
        lat, lon = _finite(value.get("lat")), _finite(value.get("lon"))
        if lat is None or lon is None or not _on_globe(lat, lon):
            return None
        accuracy = _finite(value.get("accuracyM"))
        key = value.get("key")
        return VideoLocation(
            lat=lat,
            lon=lon,
            accuracy_m=accuracy if accuracy is not None and accuracy > 0 else None,
            key=key if isinstance(key, str) and key in KEYS else "iso6709",
        )


class _Exhausted(Exception):  # noqa: N818 - a budget running out, not an error
    pass


class _Reader:
    """`ReadAt` behind a small cache and a budget of reads."""

    def __init__(self, read_at: ReadAt, size: int, chunk: int) -> None:
        self._read_at = read_at
        self._size = size
        self._chunk = chunk
        self._reads = 0
        self._bytes = 0
        self._cached: list[tuple[int, bytes]] = []

    def read(self, offset: int, length: int, *, ahead_to: int | None = None) -> bytes:
        """`length` bytes from `offset`, fetched with up to a chunk of read-ahead, but
        never past `ahead_to` (the end of the box being walked; the file's end unless
        said): the top-level walk reads headers only, so it never fetches the `mdat`."""
        if offset < 0 or length <= 0 or offset >= self._size:
            return b""
        length = min(length, self._size - offset)
        limit = self._size if ahead_to is None else min(ahead_to, self._size)
        for start, data in self._cached:
            if start <= offset and offset + length <= start + len(data):
                return data[offset - start : offset - start + length]
        wanted = max(length, min(self._chunk, limit - offset))
        if self._reads >= MAX_READS or self._bytes + wanted > MAX_READ_BYTES:
            raise _Exhausted
        self._reads += 1
        self._bytes += wanted
        data = self._read_at(offset, wanted)
        self._cached = [(offset, data), *self._cached[:3]]
        return data[:length]


def _boxes(read: ReadAt, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """`(type, payload start, box end)` of each box in [start, end), until one does not
    parse: a size under its own header, or past its container, ends the walk."""
    offset = start
    for _ in range(MAX_BOXES):
        if offset + 8 > end:
            return
        # Eight bytes, then the 64-bit size only when the box has one: sixteen up front
        # would be eight bytes of an ordinary `mdat`'s payload.
        head = read(offset, 8)
        if len(head) < 8:
            return
        size, kind = struct.unpack(">I4s", head)
        header = 8
        if size == 1:
            large = read(offset + 8, 8)
            if len(large) < 8:
                return
            (size,) = struct.unpack(">Q", large)
            header = 16
        elif size == 0:
            size = end - offset  # "to the end of its container"
        if size < header or offset + size > end:
            return
        yield kind, offset + header, offset + size
        offset += size


def _memory(data: bytes) -> ReadAt:
    return lambda offset, length: data[offset : offset + length]


def read_location(read_at: ReadAt, size: int, *, chunk: int = CHUNK) -> VideoLocation | None:
    """The location in a container of `size` bytes read through `read_at`, or None."""
    reader = _Reader(read_at, size, chunk)
    try:
        # Headers only at the top level: the box after one may be a 12 GB `mdat`.
        headers: ReadAt = lambda offset, length: reader.read(  # noqa: E731
            offset, length, ahead_to=offset + length
        )
        for kind, start, end in _boxes(headers, 0, size):
            if kind == b"moov":
                return _in_moov(reader, start, end)
    except _Exhausted:
        return None
    return None


def location_of_bytes(data: bytes) -> VideoLocation | None:
    """`read_location` over a container held in memory."""
    return read_location(_memory(data), len(data))


def location_in_storage(storage: ObjectStorage, key: str) -> VideoLocation | None:
    """`read_location` over an object in the bucket, by ranged GETs; None when it is gone.
    A storage failure is raised: the caller decides whether that is "no location"."""
    found = storage.head_object(key)
    if found is None:
        return None
    return read_location(
        lambda offset, length: storage.get_object_range(key, offset, length), found.size
    )


def _in_moov(reader: _Reader, start: int, end: int) -> VideoLocation | None:
    keyed: VideoLocation | None = None
    xyz: VideoLocation | None = None
    # Read ahead freely inside `moov`: it holds no media, only boxes that describe it.
    within: ReadAt = lambda offset, length: reader.read(offset, length, ahead_to=end)  # noqa: E731
    for kind, child_start, child_end in _boxes(within, start, end):
        if kind not in (b"meta", b"udta") or child_end - child_start > MAX_METADATA_BYTES:
            continue
        payload = reader.read(child_start, child_end - child_start)
        if len(payload) != child_end - child_start:
            continue
        if kind == b"meta":
            found = _meta(payload)
            keyed, xyz = keyed or found[0], xyz or found[1]
            continue
        for inner, inner_start, inner_end in _boxes(_memory(payload), 0, len(payload)):
            if inner == XYZ:
                xyz = xyz or _xyz(payload[inner_start:inner_end])
            elif inner == LOCI:
                xyz = xyz or _loci(payload[inner_start:inner_end])
            elif inner == b"meta":
                found = _meta(payload[inner_start:inner_end])
                keyed, xyz = keyed or found[0], xyz or found[1]
    return keyed or xyz


def _meta(payload: bytes) -> tuple[VideoLocation | None, VideoLocation | None]:
    """The `mdta` location (with its accuracy) and any iTunes-style `(c)xyz` item in a
    `meta` box. QuickTime's `meta` is a plain atom; ISO's is a full box, whose first four
    bytes are a version and flags -- zero, where a plain atom's first child has its size."""
    offset = 4 if payload[:4] == b"\0\0\0\0" else 0
    keys: list[str] = []
    items: dict[int | bytes, tuple[int, bytes]] = {}
    for kind, start, end in _boxes(_memory(payload), offset, len(payload)):
        if kind == b"keys":
            keys = _keys(payload[start:end])
        elif kind == b"ilst":
            items = _ilst(payload[start:end])
    named = {name: index for index, name in enumerate(keys, start=1)}
    keyed = None
    point = _text(items.get(named.get(ISO6709_KEY, 0)))
    if point is not None:
        parsed = _parse_iso6709(point)
        if parsed is not None:
            accuracy = _accuracy(items.get(named.get(ACCURACY_KEY, 0)))
            keyed = VideoLocation(lat=parsed[0], lon=parsed[1], accuracy_m=accuracy, key="iso6709")
    xyz = None
    text = _text(items.get(XYZ))
    if text is not None:
        parsed = _parse_iso6709(text)
        if parsed is not None:
            xyz = VideoLocation(lat=parsed[0], lon=parsed[1], accuracy_m=None, key="xyz")
    return keyed, xyz


def _keys(payload: bytes) -> list[str]:
    """`keys`: version and flags, an entry count, then (size, namespace, name) entries."""
    if len(payload) < 8:
        return []
    (count,) = struct.unpack(">I", payload[4:8])
    names: list[str] = []
    offset = 8
    for _ in range(min(count, MAX_KEYS)):
        if offset + 8 > len(payload):
            break
        (size,) = struct.unpack(">I", payload[offset : offset + 4])
        if size < 8 or offset + size > len(payload):
            break
        names.append(payload[offset + 8 : offset + size].decode("utf-8", "replace").lower())
        offset += size
    return names


def _ilst(payload: bytes) -> dict[int | bytes, tuple[int, bytes]]:
    """`ilst`: one item per key, typed by the key's 1-based index (or, iTunes-style, by a
    four-character code), each holding a `data` atom: type, locale, value."""
    items: dict[int | bytes, tuple[int, bytes]] = {}
    for kind, start, end in _boxes(_memory(payload), 0, len(payload)):
        item = payload[start:end]
        for inner, inner_start, inner_end in _boxes(_memory(item), 0, len(item)):
            if inner != b"data" or inner_end - inner_start < 8:
                continue
            (indicator,) = struct.unpack(">I", item[inner_start : inner_start + 4])
            name: int | bytes = kind if kind == XYZ else struct.unpack(">I", kind)[0]
            items.setdefault(name, (indicator & 0xFFFFFF, item[inner_start + 8 : inner_end]))
            break
    return items


def _text(item: tuple[int, bytes] | None) -> str | None:
    if item is None or item[0] != _UTF8:
        return None
    try:
        return item[1].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _accuracy(item: tuple[int, bytes] | None) -> float | None:
    """The horizontal accuracy, written as text (what an iPhone writes) or as a number."""
    if item is None:
        return None
    kind, raw = item
    value: float | None = None
    try:
        if kind == _UTF8:
            value = float(raw.decode("utf-8").strip())
        elif kind == _FLOAT32 and len(raw) == 4:
            (value,) = struct.unpack(">f", raw)
        elif kind == _FLOAT64 and len(raw) == 8:
            (value,) = struct.unpack(">d", raw)
        elif kind in (_SIGNED, _UNSIGNED) and len(raw) in (1, 2, 4, 8):
            value = float(int.from_bytes(raw, "big", signed=kind == _SIGNED))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if value is not None and math.isfinite(value) and value > 0 else None


def _xyz(payload: bytes) -> VideoLocation | None:
    """`(c)xyz`: a 16-bit length and a 16-bit language code before the string."""
    if len(payload) < 4:
        return None
    (length,) = struct.unpack(">H", payload[:2])
    if 4 + length > len(payload):
        return None  # a string that says it is longer than the atom holding it
    try:
        text = payload[4 : 4 + length].decode("utf-8")
    except UnicodeDecodeError:
        return None
    parsed = _parse_iso6709(text)
    if parsed is None:
        return None
    return VideoLocation(lat=parsed[0], lon=parsed[1], accuracy_m=None, key="xyz")


def _loci(payload: bytes) -> VideoLocation | None:
    """`loci` (3GPP TS 26.244 8.2.3): version and flags, a language code, a NUL-terminated
    name, a role byte, then longitude, latitude, altitude as signed 16.16 fixed point."""
    end = payload.find(b"\0", 6)
    if len(payload) < 6 or end < 0 or end + 1 + 1 + 8 > len(payload):
        return None
    at = end + 2  # past the name's NUL and the role
    lon, lat = (value / 65536 for value in struct.unpack(">ii", payload[at : at + 8]))
    if not _on_globe(lat, lon):
        return None
    return VideoLocation(lat=lat, lon=lon, accuracy_m=None, key="loci")


def _parse_iso6709(text: str) -> tuple[float, float] | None:
    match = _ISO6709_RE.match(text.strip())
    if match is None:
        return None
    lat, lon = float(match.group("lat")), float(match.group("lon"))
    return (lat, lon) if _on_globe(lat, lon) else None


def _on_globe(lat: float, lon: float) -> bool:
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
