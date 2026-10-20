"""Reading a `hexapod.cellgrid` v1 in the API: numpy and the stdlib, nothing else.

The container (ADR 0016; written by tools/captures/cell_grid.py, which the API does not
import) is a gzip of level planes, finest first, each `nj x ni` cells row-major (j outer,
i inner) with its channels interleaved little-endian, all described by a JSON descriptor
(the tileset's root `extras.confidence`, stored as the epoch map's `descriptor`). A level's
cell is `LATTICE_BASE_M * 2**level` metres in the tileset frame, east (`i`) and north
(`j`) from the frame origin, so the levels nest and one epoch's cells can be read at
another's level without resampling.

This reader is the second of three (Python pipeline, API, TypeScript web), held to the
pipeline's bytes by the committed cross-language fixture `data/tiles/synthetic-yard-epochs`
(`tests/test_cell_grid_decode.py`). It is stricter about size than the pipeline's: the
planes are decompressed no further than the descriptor says they reach, so a hostile or
broken file cannot inflate past what its descriptor declared, and nothing is decoded past
`MAX_RAW_BYTES`.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FORMAT = "hexapod.cellgrid"
VERSION = 1
LATTICE_BASE_M = 0.25
MAX_LEVEL = 30
#: The confidence reason bits (tools/captures/cell_grid.py `REASON_BITS`).
REASON_BITS: Mapping[str, int] = {
    "views": 1,
    "spread": 2,
    "gsd": 4,
    "accuracy": 8,
    "unseen": 16,
    "alignment": 32,
}
UNSEEN = REASON_BITS["unseen"]
#: weak-v1: a scored cell under this is weak (ADR 0016).
WEAK_BELOW = 128
#: The pipeline caps a grid at 2^22 cells a level with eight coarser levels over it, three
#: bytes a cell: ~17 MB of planes. Four times that is not a grid it wrote.
MAX_RAW_BYTES = 64 * 1024 * 1024

TYPES: Mapping[str, str] = {
    "u1": "<u1",
    "i1": "<i1",
    "u2": "<u2",
    "i2": "<i2",
    "u4": "<u4",
    "i4": "<i4",
}


class CellGridError(ValueError):
    """A descriptor or file that is not a `hexapod.cellgrid` v1 this reader can trust."""


@dataclass(frozen=True)
class Plane:
    """One level's cells `i0 .. i0+ni-1` by `j0 .. j0+nj-1`, each channel `(nj, ni)`."""

    level: int
    i0: int
    j0: int
    ni: int
    nj: int
    channels: Mapping[str, NDArray[Any]]

    @property
    def cell_m(self) -> float:
        return math.ldexp(LATTICE_BASE_M, self.level)

    @property
    def cells(self) -> int:
        return self.ni * self.nj

    def channel(self, name: str) -> NDArray[np.int64]:
        try:
            return self.channels[name].astype(np.int64)
        except KeyError as error:
            raise CellGridError(f"no channel {name!r} at level {self.level}") from error


@dataclass(frozen=True)
class Grid:
    kind: str
    planes: tuple[Plane, ...]
    descriptor: Mapping[str, Any]

    @property
    def finest(self) -> Plane:
        return self.planes[0]

    def plane(self, level: int) -> Plane | None:
        return next((p for p in self.planes if p.level == level), None)

    def at(self, level: int) -> Plane:
        """The plane at `level`: the grid's own when it has one, else its finest level
        coarsened up to it (`coarsen`), as its pyramid would have been."""
        found = self.plane(level)
        if found is not None:
            return found
        plane = self.finest
        if level < plane.level:
            raise CellGridError(f"level {level} is finer than the grid's finest ({plane.level})")
        while plane.level < level:
            plane = coarsen(plane)
        return plane


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CellGridError(f"{what} must be an integer, not {value!r}")
    return value


def _inflate(blob: bytes, limit: int) -> bytes:
    """`blob` gunzipped, but never past `limit` bytes: one more is a refusal."""
    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        raw = inflater.decompress(blob, limit + 1)
    except zlib.error as error:
        raise CellGridError(f"not gzip: {error}") from error
    if len(raw) > limit or inflater.unconsumed_tail:
        raise CellGridError(f"the planes are larger than the {limit} bytes the descriptor declares")
    if not inflater.eof:
        raise CellGridError("the gzip stream is truncated")
    return raw


def decode(descriptor: Mapping[str, Any], blob: bytes) -> Grid:
    """The grid `descriptor` describes, from its file's bytes: the format and version, the
    channels, every level's window against its slice, the planes' size and their sha256
    checked. Anything wrong is a `CellGridError`."""
    if descriptor.get("format") != FORMAT or descriptor.get("version") != VERSION:
        raise CellGridError(
            f"not a {FORMAT} v{VERSION}: {descriptor.get('format')!r} "
            f"v{descriptor.get('version')!r}"
        )
    listed = descriptor.get("channels")
    if not isinstance(listed, list) or not listed:
        raise CellGridError("channels must be a non-empty list")
    names: list[str] = []
    types: list[str] = []
    for entry in listed:
        if not isinstance(entry, Mapping) or entry.get("type") not in TYPES:
            raise CellGridError(f"channel {entry!r} is not one this reader knows")
        names.append(str(entry.get("name")))
        types.append(TYPES[str(entry["type"])])
    if len(set(names)) != len(names):
        raise CellGridError(f"channel names must be unique: {names}")
    record = np.dtype(list(zip(names, types, strict=True)))
    levels = descriptor.get("levels")
    if not isinstance(levels, list) or not levels:
        raise CellGridError("levels must be a non-empty list")
    windows: list[tuple[int, int, int, int, int, int, int]] = []
    declared = 0
    for entry in levels:
        if not isinstance(entry, Mapping):
            raise CellGridError("each level is an object")
        level = _int(entry.get("level"), "level")
        i0, j0 = _int(entry.get("i0"), "i0"), _int(entry.get("j0"), "j0")
        ni, nj = _int(entry.get("ni"), "ni"), _int(entry.get("nj"), "nj")
        offset, size = _int(entry.get("offset"), "offset"), _int(entry.get("bytes"), "bytes")
        if not 0 <= level <= MAX_LEVEL or ni < 0 or nj < 0 or offset < 0:
            raise CellGridError(f"level {level}: not a window this reader can read")
        if size != ni * nj * record.itemsize:
            raise CellGridError(f"level {level}: {size} bytes is not {nj}x{ni} cells")
        declared = max(declared, offset + size)
        windows.append((level, i0, j0, ni, nj, offset, size))
    if [w[0] for w in windows] != sorted({w[0] for w in windows}):
        raise CellGridError("levels must be finest first, one per level")
    if declared > MAX_RAW_BYTES:
        raise CellGridError(f"the planes would be {declared} bytes, over {MAX_RAW_BYTES}")
    raw = _inflate(blob, declared)
    expected = descriptor.get("sha256")
    if not isinstance(expected, str) or hashlib.sha256(raw).hexdigest() != expected:
        raise CellGridError("the planes do not match the descriptor's sha256")
    planes = []
    for level, i0, j0, ni, nj, offset, size in windows:
        if offset + size > len(raw):
            raise CellGridError(f"level {level} reaches past the planes")
        cells = np.frombuffer(raw, dtype=record, count=ni * nj, offset=offset).reshape(nj, ni)
        planes.append(Plane(level, i0, j0, ni, nj, {name: cells[name].copy() for name in names}))
    return Grid(kind=str(descriptor.get("kind")), planes=tuple(planes), descriptor=descriptor)


def weak_mask(plane: Plane) -> NDArray[np.bool_]:
    """weak-v1 per cell, `(nj, ni)`: scored under `WEAK_BELOW`, or unseen. A cell with no
    evidence that is not unseen is not this epoch's, so not weak."""
    score = plane.channel("score")
    unseen = (plane.channel("reasons") & UNSEEN) > 0
    weak: NDArray[np.bool_] = ((score > 0) & (score < WEAK_BELOW)) | unseen
    return weak


def covered_mask(plane: Plane) -> NDArray[np.bool_]:
    """The cells the epoch speaks for: scored, or expected and unseen."""
    covered: NDArray[np.bool_] = (plane.channel("score") > 0) | (
        (plane.channel("reasons") & UNSEEN) > 0
    )
    return covered


_EMPTY = np.int64(1) << 40


def coarsen(plane: Plane) -> Plane:
    """One level up by min-of-children, as the pipeline's pyramid is built: score the
    minimum of the children that have one (0 when none does), reasons their OR, relief
    their maximum. Cells outside the plane are empty (no evidence)."""
    pi0, pj0 = plane.i0 >> 1, plane.j0 >> 1
    pi1, pj1 = (plane.i0 + plane.ni - 1) >> 1, (plane.j0 + plane.nj - 1) >> 1
    ni, nj = (pi1 - pi0 + 1, pj1 - pj0 + 1) if plane.cells else (0, 0)
    di, dj = plane.i0 - 2 * pi0, plane.j0 - 2 * pj0

    def quads(name: str) -> NDArray[np.int64]:
        padded = np.zeros((2 * nj, 2 * ni), dtype=np.int64)
        padded[dj : dj + plane.nj, di : di + plane.ni] = plane.channel(name)
        return padded.reshape(nj, 2, ni, 2)

    score = quads("score")
    best = np.where(score > 0, score, _EMPTY).min(axis=(1, 3), initial=_EMPTY)
    reasons = quads("reasons")
    out = {
        "score": np.where(best == _EMPTY, 0, best),
        "reasons": reasons[:, 0, :, 0]
        | reasons[:, 0, :, 1]
        | reasons[:, 1, :, 0]
        | reasons[:, 1, :, 1],
        "relief": quads("relief").max(axis=(1, 3), initial=0),
    }
    return Plane(plane.level + 1, pi0, pj0, ni, nj, out)


def window(
    plane: Plane, i0: int, j0: int, ni: int, nj: int, mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """`mask` (over `plane`) placed in the window `i0, j0, ni, nj` of the same level;
    False outside the plane."""
    out = np.zeros((nj, ni), dtype=bool)
    a0, a1 = max(i0, plane.i0), min(i0 + ni, plane.i0 + plane.ni)
    b0, b1 = max(j0, plane.j0), min(j0 + nj, plane.j0 + plane.nj)
    if a0 < a1 and b0 < b1:
        out[b0 - j0 : b1 - j0, a0 - i0 : a1 - i0] = mask[
            b0 - plane.j0 : b1 - plane.j0, a0 - plane.i0 : a1 - plane.i0
        ]
    return out


def union_window(planes: Sequence[Plane]) -> tuple[int, int, int, int]:
    """`(i0, j0, ni, nj)` covering every plane (all at one level)."""
    i0 = min(p.i0 for p in planes)
    j0 = min(p.j0 for p in planes)
    i1 = max(p.i0 + p.ni for p in planes)
    j1 = max(p.j0 + p.nj for p in planes)
    return i0, j0, i1 - i0, j1 - j0
