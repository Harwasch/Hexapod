"""One site cell lattice, and the one binary container every per-cell map travels in.

**The lattice.** Every per-cell map of a site -- an epoch's confidence today, its height
surfaces later -- is a grid of axis-aligned 2.5D columns in the **tileset frame**: the
east/north/up frame `root.transform` places on the globe, which is the site frame whenever
the run had one. A cell at `level` is `LATTICE_BASE_M * 2**level` metres on a side, and a
point `(x, y)` (x east, y north) is in cell `i = floor(x / cell)`, `j = floor(y / cell)`,
counted from the frame origin. So the cells of every level nest -- cell `(i, j)` at
`level + 1` is the four cells `(2i + a, 2j + b)` at `level` -- and two epochs of a site
compare at the coarser of their two levels with no resampling. (`x / cell` divides by a
power of two, which is exact in float64, so `floor(x / 2c) == floor(i / 2)` exactly too.)

**The container**, `hexapod.cellgrid` v1, is a gzip (level 9, `mtime=0`, so the bytes are
stable) of the level planes concatenated, finest first. A plane is `nj x ni` cells, row
major (j outer, i inner), each cell its channels interleaved, little-endian. Everything a
reader needs is in a JSON descriptor -- the value of the tileset's root `extras` key
(`extras.confidence`) -- so a decoder in any language is a gunzip and a few slices:

    {"format": "hexapod.cellgrid", "version": 1, "kind": "confidence", "uri": "confidence.bin",
     "frame": "tileset", "lattice": {"baseM": 0.25, "level": 2}, "cellM": 1.0,
     "channels": [{"name": "score", "type": "u1"}, ...],
     "levels": [{"level": 2, "cellM": 1.0, "i0": -40, "j0": -35, "ni": 80, "nj": 70,
                 "offset": 0, "bytes": 16800}, ...],
     ...what the kind adds (reasonBits, policy, summary, ...)...,
     "sha256": "<of the uncompressed planes>"}

`offset` and `bytes` are into the *uncompressed* planes. numpy and the stdlib only, so the
pipeline, the API and the tests all read it with what they already have.

**Confidence** is the first kind (tools/pipeline confidence.py writes it): channels
`score` (0 = no evidence, else 1..255), `reasons` (`REASON_BITS`) and `relief` (the
height range of the cell's gaussians, in `RELIEF_UNIT_M`, saturating), and the
`weak-v1` policy (`WEAK_BELOW`). Its coarser levels are a **min-of-children** pyramid
(`min_pyramid`): a coarse cell is as weak as its weakest child, so a viewer that draws
the coarse level never shows a weak patch as strong.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CONFIDENCE_CHANNELS",
    "FORMAT",
    "LATTICE_BASE_M",
    "REASON_BITS",
    "RELIEF_UNIT_M",
    "VERSION",
    "WEAK_BELOW",
    "WEAK_POLICY",
    "CellGrid",
    "CellGridError",
    "CellPlane",
    "Channel",
    "Lattice",
    "decode",
    "descriptor_of",
    "encode",
    "from_tileset",
    "lookup",
    "min_pyramid",
    "pack",
    "weak_mask",
]

FORMAT = "hexapod.cellgrid"
VERSION = 1
#: The finest cell of the site lattice, metres. Level `k` is `LATTICE_BASE_M * 2**k`.
LATTICE_BASE_M = 0.25
#: Levels a plane may be at. 30 is a 268,000 km cell: past it nothing on Earth is a cell.
MAX_LEVEL = 30
#: The frame every plane is in: the tileset's (see the module docstring).
FRAME = "tileset"

#: The channel types a descriptor may name, and their little-endian numpy codes.
TYPES: Mapping[str, str] = {
    "u1": "<u1",
    "i1": "<i1",
    "u2": "<u2",
    "i2": "<i2",
    "u4": "<u4",
    "i4": "<i4",
}

#: `confidence`'s reason bits: why a cell's gaussians fell short of keep, plus `unseen`
#: (inside the capture's region of interest with no evidence at all) and `alignment`
#: (the epoch's placement uncertainty capped its score).
REASON_BITS: Mapping[str, int] = {
    "views": 1,
    "spread": 2,
    "gsd": 4,
    "accuracy": 8,
    "unseen": 16,
    "alignment": 32,
}
#: `relief`'s unit: a cell's gaussians span `relief * RELIEF_UNIT_M` metres of height.
RELIEF_UNIT_M = 0.25
#: The policy `confidence` follows, and its threshold: a cell is weak when its score is
#: under `WEAK_BELOW` (fewer than half of its visible gaussians, by alpha, passed keep --
#: a majority rule, stated from first principles, not calibrated) or it is unseen.
WEAK_POLICY = "weak-v1"
WEAK_BELOW = 128

#: Keys of a descriptor that are the container's own; every other key is the kind's.
_CONTAINER_KEYS = (
    "format",
    "version",
    "kind",
    "uri",
    "frame",
    "lattice",
    "cellM",
    "channels",
    "levels",
    "sha256",
)
_EMPTY_SCORE = np.int64(1) << 40


class CellGridError(ValueError):
    """A descriptor or a blob that is not a `hexapod.cellgrid` v1 this reader can trust."""


@dataclass(frozen=True)
class Lattice:
    """One level of the site lattice."""

    level: int

    def __post_init__(self) -> None:
        if not 0 <= self.level <= MAX_LEVEL:
            raise CellGridError(f"lattice level {self.level} is outside 0..{MAX_LEVEL}")

    @property
    def cell_m(self) -> float:
        return math.ldexp(LATTICE_BASE_M, self.level)

    def index(self, xy: Any) -> tuple[np.ndarray, np.ndarray]:
        """Each point's cell `(i, j)` from its first two coordinates, int64. A point that
        is not finite gets the minimum int64 on both, which is inside no plane."""
        array = np.asarray(xy, dtype=np.float64)
        points = array.reshape(-1, array.shape[-1]) if array.size else np.zeros((0, 2))
        cell = self.cell_m
        with np.errstate(invalid="ignore", over="ignore"):
            fi = np.floor(points[:, 0] / cell)
            fj = np.floor(points[:, 1] / cell)
        limit = float(1 << 62)
        finite = np.isfinite(fi) & np.isfinite(fj) & (np.abs(fi) < limit) & (np.abs(fj) < limit)
        i = np.where(finite, fi, 0.0).astype(np.int64)
        j = np.where(finite, fj, 0.0).astype(np.int64)
        low = np.iinfo(np.int64).min
        i[~finite] = low
        j[~finite] = low
        return i, j


@dataclass(frozen=True)
class Channel:
    """One per-cell value: its name, its type (`TYPES`) and, when it is a length, its unit."""

    name: str
    type: str
    unit_m: float | None = None

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise CellGridError(f"channel {self.name!r} has unknown type {self.type!r}")

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"name": self.name, "type": self.type}
        if self.unit_m is not None:
            out["unitM"] = self.unit_m
        return out

    @staticmethod
    def from_dict(document: Mapping[str, Any]) -> Channel:
        unit = document.get("unitM")
        return Channel(
            name=str(document["name"]),
            type=str(document["type"]),
            unit_m=None if unit is None else float(unit),
        )


#: `confidence`'s channels, in the order they are interleaved.
CONFIDENCE_CHANNELS: tuple[Channel, ...] = (
    Channel("score", "u1"),
    Channel("reasons", "u1"),
    Channel("relief", "u1", unit_m=RELIEF_UNIT_M),
)


@dataclass(frozen=True)
class CellPlane:
    """The cells `i0 .. i0+ni-1` by `j0 .. j0+nj-1` of one level, `data[j - j0, i - i0, c]`."""

    level: int
    i0: int
    j0: int
    ni: int
    nj: int
    data: np.ndarray

    def __post_init__(self) -> None:
        Lattice(self.level)
        if self.ni < 0 or self.nj < 0:
            raise CellGridError("a plane cannot have a negative size")
        if self.data.ndim != 3 or self.data.shape[:2] != (self.nj, self.ni):
            raise CellGridError(
                f"plane data has shape {self.data.shape}; ({self.nj}, {self.ni}, C) was declared"
            )

    @property
    def cell_m(self) -> float:
        return Lattice(self.level).cell_m

    @property
    def cells(self) -> int:
        return self.ni * self.nj


@dataclass(frozen=True)
class CellGrid:
    """A kind of per-cell map: its channels, its planes (finest first) and its kind's keys."""

    kind: str
    channels: tuple[Channel, ...]
    planes: tuple[CellPlane, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.planes:
            raise CellGridError("a cell grid has at least one plane")
        levels = [plane.level for plane in self.planes]
        if levels != sorted(set(levels)):
            raise CellGridError(f"planes must be finest first, one per level: {levels}")
        for plane in self.planes:
            if plane.data.shape[2] != len(self.channels):
                raise CellGridError(
                    f"level {plane.level} has {plane.data.shape[2]} channels; "
                    f"{len(self.channels)} are declared"
                )

    @property
    def level(self) -> int:
        return self.planes[0].level

    def plane(self, level: int | None = None) -> CellPlane:
        if level is None:
            return self.planes[0]
        for plane in self.planes:
            if plane.level == level:
                return plane
        raise KeyError(f"no level {level} in this grid (levels {[p.level for p in self.planes]})")

    def channel(self, name: str) -> int:
        for index, channel in enumerate(self.channels):
            if channel.name == name:
                return index
        raise KeyError(f"no channel {name!r} in this {self.kind} grid")


def _record(channels: Sequence[Channel]) -> np.dtype[Any]:
    return np.dtype([(channel.name, TYPES[channel.type]) for channel in channels])


def _plane_bytes(plane: CellPlane, channels: Sequence[Channel]) -> bytes:
    record = _record(channels)
    out = np.zeros((plane.nj, plane.ni), dtype=record)
    for index, channel in enumerate(channels):
        values = plane.data[:, :, index]
        info = np.iinfo(np.dtype(TYPES[channel.type]))
        if values.size and (int(values.min()) < info.min or int(values.max()) > info.max):
            raise CellGridError(
                f"level {plane.level} channel {channel.name!r} holds values outside "
                f"{channel.type} ({int(values.min())}..{int(values.max())})"
            )
        out[channel.name] = values
    return out.tobytes()


def descriptor_of(
    grid: CellGrid, raw_sha256: str, uri: str, offsets: Sequence[int]
) -> dict[str, Any]:
    """The descriptor of `grid`'s planes, written at `offsets` in `uri` (module docstring)."""
    itemsize = _record(grid.channels).itemsize
    levels = [
        {
            "level": plane.level,
            "cellM": plane.cell_m,
            "i0": plane.i0,
            "j0": plane.j0,
            "ni": plane.ni,
            "nj": plane.nj,
            "offset": int(offset),
            "bytes": plane.cells * itemsize,
        }
        for plane, offset in zip(grid.planes, offsets, strict=True)
    ]
    out: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "kind": grid.kind,
        "uri": uri,
        "frame": FRAME,
        "lattice": {"baseM": LATTICE_BASE_M, "level": grid.level},
        "cellM": grid.planes[0].cell_m,
        "channels": [channel.to_dict() for channel in grid.channels],
        "levels": levels,
    }
    for key, value in grid.meta.items():
        if key in _CONTAINER_KEYS:
            raise CellGridError(f"{key!r} is the container's own key, not the kind's")
        out[key] = value
    out["sha256"] = raw_sha256
    return out


def pack(grid: CellGrid, uri: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """The gzipped planes and their descriptor, in memory."""
    parts: list[bytes] = []
    offsets: list[int] = []
    at = 0
    for plane in grid.planes:
        offsets.append(at)
        part = _plane_bytes(plane, grid.channels)
        parts.append(part)
        at += len(part)
    raw = b"".join(parts)
    digest = hashlib.sha256(raw).hexdigest()
    descriptor = descriptor_of(grid, digest, uri or f"{grid.kind}.bin", offsets)
    return gzip.compress(raw, compresslevel=9, mtime=0), descriptor


def encode(grid: CellGrid, out_dir: Path, uri: str | None = None) -> dict[str, Any]:
    """Write `grid` to `out_dir / uri` (default `<kind>.bin`); return its descriptor."""
    blob, descriptor = pack(grid, uri)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / str(descriptor["uri"])).write_bytes(blob)
    return descriptor


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CellGridError(f"{what} must be an integer, not {value!r}")
    return value


def decode(descriptor: Mapping[str, Any], blob: bytes) -> CellGrid:
    """The grid `descriptor` describes, from its file's bytes. Checks everything it can:
    the format and version, every level's size against its slice, and the sha256."""
    if descriptor.get("format") != FORMAT or descriptor.get("version") != VERSION:
        raise CellGridError(
            f"not a {FORMAT} v{VERSION}: format {descriptor.get('format')!r}, "
            f"version {descriptor.get('version')!r}"
        )
    if descriptor.get("frame") != FRAME:
        raise CellGridError(f"a cell grid is in the {FRAME} frame, not {descriptor.get('frame')!r}")
    channels = tuple(Channel.from_dict(c) for c in descriptor.get("channels") or ())
    if not channels:
        raise CellGridError("a cell grid declares at least one channel")
    record = _record(channels)
    try:
        raw = gzip.decompress(blob)
    except (OSError, EOFError) as problem:
        raise CellGridError(f"{descriptor.get('uri')!r} is not gzip: {problem}") from problem
    expected = descriptor.get("sha256")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise CellGridError(f"{descriptor.get('uri')!r} does not match its sha256")
    # One dtype for every channel: the channels' own when they share one, else int32
    # (wide enough for every type `TYPES` allows but u4, which no kind uses yet).
    types = {channel.type for channel in channels}
    common = np.dtype(TYPES[types.pop()]).newbyteorder("=") if len(types) == 1 else np.dtype("i4")
    planes: list[CellPlane] = []
    for entry in descriptor.get("levels") or ():
        level = _int(entry.get("level"), "level")
        ni, nj = _int(entry.get("ni"), "ni"), _int(entry.get("nj"), "nj")
        offset, size = _int(entry.get("offset"), "offset"), _int(entry.get("bytes"), "bytes")
        if size != ni * nj * record.itemsize or offset < 0 or offset + size > len(raw):
            raise CellGridError(f"level {level}: {size} bytes at {offset} is not {nj}x{ni} cells")
        cells = np.frombuffer(raw, dtype=record, count=ni * nj, offset=offset).reshape(nj, ni)
        data = np.stack([cells[c.name].astype(common) for c in channels], axis=-1)
        planes.append(
            CellPlane(
                level=level,
                i0=_int(entry.get("i0"), "i0"),
                j0=_int(entry.get("j0"), "j0"),
                ni=ni,
                nj=nj,
                data=data,
            )
        )
    lattice = descriptor.get("lattice") or {}
    if planes and lattice.get("level") != planes[0].level:
        raise CellGridError("lattice.level is not the finest plane's level")
    meta = {k: v for k, v in descriptor.items() if k not in _CONTAINER_KEYS}
    return CellGrid(
        kind=str(descriptor.get("kind")), channels=channels, planes=tuple(planes), meta=meta
    )


def _coarsen(plane: CellPlane, score: int, reasons: int, relief: int) -> CellPlane:
    """One level up: score the minimum of the children that have one (0 when none does),
    reasons their OR, relief their maximum. A plane with any other channel is refused:
    how to reduce it is a decision this function does not make for it."""
    channels = plane.data.shape[2]
    if sorted((score, reasons, relief)) != list(range(channels)):
        raise CellGridError("min_pyramid needs exactly the score, reasons and relief channels")
    # Pad to whole parents: an odd first index borrows an empty cell before it, an odd
    # last one an empty cell after it. Empty is score 0 (no evidence), no reasons, relief 0.
    # `>>` on int is floor division by two, negative indices included.
    pi0, pj0 = plane.i0 >> 1, plane.j0 >> 1
    pi1, pj1 = (plane.i0 + plane.ni - 1) >> 1, (plane.j0 + plane.nj - 1) >> 1
    ni, nj = (pi1 - pi0 + 1, pj1 - pj0 + 1) if plane.cells else (0, 0)
    padded = np.zeros((2 * nj, 2 * ni, channels), dtype=plane.data.dtype)
    di, dj = plane.i0 - 2 * pi0, plane.j0 - 2 * pj0
    padded[dj : dj + plane.nj, di : di + plane.ni] = plane.data
    quads = padded.reshape(nj, 2, ni, 2, channels)
    out = np.zeros((nj, ni, channels), dtype=plane.data.dtype)
    s = quads[..., score].astype(np.int64)
    best = np.where(s > 0, s, _EMPTY_SCORE).min(axis=(1, 3), initial=_EMPTY_SCORE)
    out[..., score] = np.where(best == _EMPTY_SCORE, 0, best)
    r = quads[..., reasons]
    out[..., reasons] = r[:, 0, :, 0] | r[:, 0, :, 1] | r[:, 1, :, 0] | r[:, 1, :, 1]
    out[..., relief] = quads[..., relief].max(axis=(1, 3), initial=0)
    return CellPlane(plane.level + 1, pi0, pj0, ni, nj, out)


def min_pyramid(
    plane: CellPlane, levels: int, *, score: int = 0, reasons: int = 1, relief: int = 2
) -> list[CellPlane]:
    """`plane` and up to `levels` coarser planes above it, min-of-children (module
    docstring). Stops early once a plane is a single cell (or empty): there is nothing
    coarser to say. (A plane that straddles the frame origin never becomes one: cells -1
    and 0 have different parents at every level.) An unseen child carries its bit up in `reasons`, so a coarse cell over
    any unseen one is weak (`weak_mask`) however strong its other children are."""
    out = [plane]
    while len(out) <= levels and out[-1].level < MAX_LEVEL and out[-1].cells > 1:
        out.append(_coarsen(out[-1], score, reasons, relief))
    return out


def lookup(grid: CellGrid, xyz: Any, level: int | None = None) -> np.ndarray:
    """Each point's cell values at `level` (default: the finest), `(n, channels)`; zeros
    for a point outside the plane (no evidence), as a reader of a missing cell must."""
    plane = grid.plane(level)
    array = np.asarray(xyz, dtype=np.float64)
    points = array.reshape(-1, array.shape[-1]) if array.size else np.zeros((0, 3))
    i, j = Lattice(plane.level).index(points)
    di, dj = i - plane.i0, j - plane.j0
    inside = (di >= 0) & (di < plane.ni) & (dj >= 0) & (dj < plane.nj)
    out = np.zeros((points.shape[0], plane.data.shape[2]), dtype=plane.data.dtype)
    out[inside] = plane.data[dj[inside], di[inside]]
    return out


def weak_mask(plane: CellPlane, *, score: int = 0, reasons: int = 1) -> np.ndarray:
    """`weak-v1` per cell, `(nj, ni)` bool: scored under `WEAK_BELOW`, or unseen. A cell
    with no evidence that is not unseen (outside what the capture aimed at) is not weak:
    it is simply not this epoch's."""
    s = plane.data[..., score].astype(np.int64)
    unseen = (plane.data[..., reasons].astype(np.int64) & REASON_BITS["unseen"]) > 0
    weak: np.ndarray = ((s > 0) & (s < WEAK_BELOW)) | unseen
    return weak


def from_tileset(tileset_path: Path) -> dict[str, CellGrid]:
    """Every cell grid a `tileset.json` declares on its root `extras`, by key, read from
    the files beside it."""
    document = json.loads(tileset_path.read_text(encoding="utf-8"))
    extras = (document.get("root") or {}).get("extras") or {}
    out: dict[str, CellGrid] = {}
    folder = tileset_path.parent.resolve()
    for key, value in extras.items():
        if isinstance(value, dict) and value.get("format") == FORMAT:
            uri = str(value.get("uri") or "")
            target = (folder / uri).resolve()
            if not uri or not target.is_relative_to(folder):
                raise CellGridError(f"extras.{key}.uri {uri!r} must name a file beside the tileset")
            out[key] = decode(value, target.read_bytes())
    return out
