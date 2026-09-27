"""Where the data supports a high-quality reconstruction, as a voxel mask of any shape.

The quality stage decides per gaussian whether a splat has earned its place (views,
angular spread, pixel size). This module turns that verdict into a *region*: the voxels
that hold enough well-supported splats, dilated a little so a surface's own thickness and
a refine's densification have room. A Refine trains inside it and drops what falls outside
it -- so the good data decides the shape, whether that is one object, an L of two walls,
or three separate things on a lawn. No sphere, box or picked subject is involved.

**Voxel size comes from the data's own resolution, not the scene's extent.** A fixed
number of cells across the extent made a 2 m table's voxels 2.6 cm and a 50 m building's
65 cm: blocky and loose on the large scene, needlessly fine on a tiny one. The size is
instead a multiple of the keep tier's *pixel footprint* (its GSD: model units per pixel
in the sharpest view, from `quality.measure_support`), so it is in model units and the
same shape at any scale gets the same voxels. The multiple, `FOOTPRINT_MULTIPLE`:

* A well-trained surface carries a gaussian about every `s` = 2 px of footprint (gsplat
  densifies until a gaussian's image-space gradient is small, which is a few pixels).
* A voxel of side `v` that a surface crosses holds, on average over orientations, a
  patch of `2/3 v^2` of it (Cauchy: a cube's mean cross-section is its volume over its
  mean width, `v^3 / 1.5 v`); so `lambda = (2/3) (v/s)^2` gaussians, about Poisson.
* `MIN_PER_VOXEL` = 2 of them make a voxel supported. For 95% of a surface's voxels to
  meet that, `P(N >= 2) = 1 - e^-lambda (1 + lambda) >= 0.95` needs `lambda >= 4.74`, so
  `v >= 2.67 s`, or 5.3 px; 6 px leaves a margin, and the 2-voxel dilation closes the
  odd hole that is left.

**The footprint is not one number across a scene** -- near things are sampled finer than
far ones. Connectivity is decided at the coarse end (sparser gaussians per voxel), so the
size is taken at the keep tier's 90th percentile, not its median. That percentile is
bounded: keep *requires* a GSD within `keep_max_gsd_ratio` (2x) of the region of
interest's median, so the coarse end cannot run away, and the fine end only gets voxels
somewhat coarser than it could have had -- a slightly looser crop there, never a hole. A
per-region (octree) size would buy that tightness back at the cost of a much more complex
format, and the dilation margin is already two voxels.

**The gaussians' own spacing can be coarser than the footprint says** (a gaussian budget
spread over a big scene). So the footprint size is checked against the data: if fewer
than `RETAINED_SHARE` of the keep gaussians land in supported voxels, the gaussians are
sparser than 2 px apart and the voxel grows by sqrt(2) until they do (`spacingSteps`).
Counts per voxel are dimensionless, so this is scale-free too.

**Bounds.** At least `MIN_CELLS_ACROSS` voxels span the extent (a scene imaged at a very
coarse footprint still has a shape, not one blob), at most `MAX_DIM` per axis; and the
dilated mask holds at most `MAX_VOXELS` voxels and encodes into at most
`MAX_ENCODED_BYTES`. Only when one of those binds is the voxel coarsened, and `sizing.bound`
says which.

It travels as JSON (a job parameter), **sparsely**: the occupied voxels' linear indices
(C order over `dims`) as maximal runs, written as LEB128 varints -- the gap before each
run, then its length -- zlib-compressed and base64'd (`kind` "voxels.v2"). A surface
dilated into a shell is mostly runs, so a table at millimetre voxels is a few kilobytes
and a building's hundreds of thousands of voxels fit the limit. The dense packed-bit
format earlier runs wrote (`kind` "voxels") is still read. Held in memory as the runs'
starts and stops, `contains` is a binary search: O(log runs) per point, no dense grid.
"""

from __future__ import annotations

import base64
import binascii
import math
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

import outofcore

__all__ = [
    "FOOTPRINT_MULTIPLE",
    "KIND",
    "LEGACY_KIND",
    "MAX_ENCODED_BYTES",
    "ArrayPoints",
    "Occupancy",
    "PointSet",
    "Sizing",
    "StreamedPoints",
    "SupportMask",
    "build",
    "build_from",
    "resolution",
    "resolution_of",
]

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]

#: The format `to_dict` writes, and the dense one earlier runs wrote (still parsed).
KIND = "voxels.v2"
LEGACY_KIND = "voxels"
#: Voxel side in pixel footprints (model units per pixel); derived in the module docstring.
FOOTPRINT_MULTIPLE = 6.0
#: Which percentile of the keep tier's footprint sizes the voxel: the coarse end, where
#: connectivity is decided (see the module docstring).
FOOTPRINT_PERCENTILE = 90.0
#: A voxel counts as supported when it holds at least this many keep splats: one stray
#: keep splat is not a surface, and two in one voxel usually are part of one.
MIN_PER_VOXEL = 2
#: The share of keep gaussians that must land in supported voxels; below it the gaussians
#: are sparser than the footprint implies and the voxel grows. Not 1: a few keep strays
#: are always alone.
RETAINED_SHARE = 0.9
#: Voxels of margin added round the supported ones.
DILATE = 2
#: At least this many voxels across the extent's longest side, however coarse the footprint.
MIN_CELLS_ACROSS = 24
#: At most this many voxels along any axis (keeps linear indices far inside int64).
MAX_DIM = 1 << 16
#: At most this many voxels in the dilated mask: memory in `build` and the encoded size
#: are both about proportional to it.
MAX_VOXELS = 2_000_000
#: The most encoded bytes accepted when parsing, so a job parameter cannot be made huge;
#: `build` coarsens until it fits.
MAX_ENCODED_BYTES = 256 * 1024
#: The most runs a parsed mask may hold (memory in `parse` and `contains` is 16 B a run).
MAX_RUNS = 4_000_000
#: The voxel count of the old extent rule, used only when no footprint is given.
EXTENT_CELLS = 80
#: The dense format's per-axis limit (4x its 80 cells), which bounds its decompression.
LEGACY_MAX_DIM = 320

#: An LEB128 varint of a value below MAX_DIM**3 = 2**48 is at most 7 bytes.
_MAX_VARINT_BYTES = 7
_MIN_POINTS = 32
#: How many times sizing may coarsen before giving up (each step is at least 1.1x).
_MAX_STEPS = 40


@dataclass(frozen=True)
class Sizing:
    """How the voxel size was chosen, for quality.json and the logs."""

    #: "footprint" (the rule in the module docstring) or "extent" (no footprint given).
    rule: str
    #: The footprint percentile used, in model units per pixel; None under "extent".
    footprint: float | None
    #: The keep tier's 90th over 10th percentile footprint: how uneven the resolution is.
    footprint_spread: float | None
    #: Voxel side in footprints; None under "extent".
    multiple: float | None
    #: How many sqrt(2) steps the gaussians' own spacing asked for beyond the footprint.
    spacing_steps: int
    #: The voxel the data supports -- footprint x multiple, spacing-checked, and within
    #: `MIN_CELLS_ACROSS` -- before any memory or byte cap. keepPct is measured at this.
    resolution: float
    #: The voxel the mask uses: `resolution`, or coarser when `bound` says a cap bound.
    voxel: float
    #: None, or which bound set the size: "min-cells", "max-cells", "max-voxels",
    #: "max-bytes".
    bound: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "footprint": self.footprint,
            "footprintPercentile": FOOTPRINT_PERCENTILE if self.rule == "footprint" else None,
            "footprintSpread": (
                None if self.footprint_spread is None else round(self.footprint_spread, 3)
            ),
            "multiple": self.multiple,
            "spacingSteps": self.spacing_steps,
            "resolution": self.resolution,
            "voxel": self.voxel,
            "coarsenedBy": round(self.voxel / self.resolution, 4),
            "bound": self.bound,
        }


@dataclass(frozen=True)
class SupportMask:
    """Occupied voxels of a grid, as sorted, disjoint runs of C-order linear indices."""

    origin: tuple[float, float, float]
    voxel: float
    dims: tuple[int, int, int]
    #: Each run's first linear index and one past its last; sorted, disjoint, non-adjacent.
    starts: I64
    stops: I64
    #: How `voxel` was chosen; None for a parsed mask.
    sizing: Sizing | None = None

    @property
    def voxels(self) -> int:
        return int((self.stops - self.starts).sum())

    @property
    def runs(self) -> int:
        return int(self.starts.size)

    def indices(self) -> I64:
        """Every occupied voxel's linear index, sorted. Memory is 8 B a voxel."""
        if self.starts.size == 0:
            return np.zeros(0, dtype=np.int64)
        lengths = self.stops - self.starts
        offsets = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        out: I64 = np.repeat(self.starts, lengths) + offsets
        return out

    def contains(self, xyz: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Whether each point falls in a supported voxel. Non-finite points never do.

        A binary search of the runs per point, so millions of points against hundreds of
        thousands of runs is a fraction of a second, with no dense grid in memory.
        """
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        ijk = np.floor((np.where(finite[:, None], points, 0.0) - self.origin) / self.voxel)
        dims = np.asarray(self.dims)
        inside = finite & (ijk >= 0).all(axis=1) & (ijk < dims).all(axis=1)
        result = np.zeros(points.shape[0], dtype=bool)
        if self.starts.size == 0 or not inside.any():
            return result
        linear = _linear(ijk[inside].astype(np.int64), self.dims)
        run = np.searchsorted(self.starts, linear, side="right") - 1
        hit = run >= 0
        hit[hit] = linear[hit] < self.stops[run[hit]]
        result[inside] = hit
        return result

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "kind": KIND,
            "frame": "colmap",
            # Full precision: a rounded origin or voxel would move boundaries, so a point
            # inside before the round trip could be outside after it.
            "origin": [float(v) for v in self.origin],
            "voxel": float(self.voxel),
            "dims": list(self.dims),
            "voxels": self.voxels,
            "runs": self.runs,
            "encoding": "zlib+base64 of LEB128 varints: per run, the gap then the length",
            "data": _encode_runs(self.starts, self.stops),
        }
        if self.sizing is not None:
            document["sizing"] = self.sizing.to_dict()
        return document

    @staticmethod
    def parse(value: object) -> SupportMask | None:
        """The dict `to_dict` wrote (or an earlier run's dense one), checked; None for no
        mask. Anything malformed is refused by name, because a mask silently ignored is
        an uncropped run."""
        if value is None:
            return None
        if not isinstance(value, Mapping) or value.get("kind") not in (KIND, LEGACY_KIND):
            raise ValueError("support_mask must be a voxel mask written by the quality stage")
        legacy = value.get("kind") == LEGACY_KIND
        origin = value.get("origin")
        dims = value.get("dims")
        voxel: Any = value.get("voxel")
        payload = value.get("bits" if legacy else "data")
        limit = LEGACY_MAX_DIM if legacy else MAX_DIM
        if (
            not isinstance(origin, list)
            or len(origin) != 3
            or not all(_finite(v) for v in origin)
            or not isinstance(dims, list)
            or len(dims) != 3
            or not all(_int(d) and 0 < d <= limit for d in dims)
            or not _finite(voxel)
            or float(voxel) <= 0
            or not isinstance(payload, str)
            or len(payload) > MAX_ENCODED_BYTES
            or value.get("frame", "colmap") != "colmap"
        ):
            raise ValueError("support_mask has a malformed origin, voxel, dims, frame or data")
        shape = (int(dims[0]), int(dims[1]), int(dims[2]))
        cells = shape[0] * shape[1] * shape[2]
        starts, stops = _decode_dense(payload, cells) if legacy else _decode_runs(payload, cells)
        if starts.size == 0:
            raise ValueError("support_mask is empty: it would crop everything away")
        declared = value.get("voxels")
        if declared is not None and (not _int(declared) or declared != int((stops - starts).sum())):
            raise ValueError("support_mask's voxel count does not match its data")
        runs = value.get("runs")
        if runs is not None and (not _int(runs) or runs != starts.size):
            raise ValueError("support_mask's run count does not match its data")
        return SupportMask(
            origin=(float(origin[0]), float(origin[1]), float(origin[2])),
            voxel=float(voxel),
            dims=shape,
            starts=starts,
            stops=stops,
        )


# ---------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Occupancy:
    """What the sizing and the mask need to know of the points in one grid: how many are
    inside it, how many are in voxels holding at least `min_per_voxel`, and those voxels."""

    total: int
    retained: int
    seed_count: int
    #: The supported voxels' linear indices, sorted; None when there were more than the
    #: caller's limit (then only `seed_count` is known).
    seeds: I64 | None


class PointSet(Protocol):
    """The points a mask is built from, as the statistics building needs of them.

    Two implementations: `ArrayPoints` over arrays in memory (what `build(xyz, footprint)`
    has always done, with the same numpy calls) and `StreamedPoints` over chunks read
    from disk (`outofcore.py`), which computes the same numbers exactly without holding
    the points -- the quality stage's keep tier can be most of a many-million-gaussian
    splat.
    """

    def size(self) -> int: ...

    def has_footprint(self) -> bool: ...

    def box(self) -> tuple[F64, F64]:
        """The 0.5 and 99.5 percentiles of each axis, as `np.percentile(points, q, axis=0)`."""
        ...

    def footprint_percentiles(self, qs: tuple[float, ...]) -> tuple[int, list[float]]:
        """How many footprints are usable (finite and positive), and their percentiles
        (none when fewer than `_MIN_POINTS` are)."""
        ...

    def occupancy(
        self, origin: F64, dims: tuple[int, int, int], voxel: float, min_per_voxel: int, limit: int
    ) -> Occupancy: ...


class ArrayPoints:
    """`PointSet` over arrays: the finite points of `xyz`, and their footprints."""

    def __init__(self, xyz: npt.ArrayLike, footprint: npt.ArrayLike | None) -> None:
        self.points, self.prints = _finite_points(xyz, footprint)

    def size(self) -> int:
        return int(self.points.shape[0])

    def has_footprint(self) -> bool:
        return self.prints is not None

    def box(self) -> tuple[F64, F64]:
        return np.percentile(self.points, 0.5, axis=0), np.percentile(self.points, 99.5, axis=0)

    def footprint_percentiles(self, qs: tuple[float, ...]) -> tuple[int, list[float]]:
        assert self.prints is not None
        usable = self.prints[np.isfinite(self.prints) & (self.prints > 0)]
        if usable.size < _MIN_POINTS:
            return int(usable.size), []
        return int(usable.size), [float(np.percentile(usable, q)) for q in qs]

    def occupancy(
        self, origin: F64, dims: tuple[int, int, int], voxel: float, min_per_voxel: int, limit: int
    ) -> Occupancy:
        linear, counts = _occupancy(self.points, origin, dims, voxel)
        supported = counts >= min_per_voxel
        seed_count = int(supported.sum())
        return Occupancy(
            total=int(counts.sum()),
            retained=int(counts[supported].sum()),
            seed_count=seed_count,
            seeds=linear[supported] if seed_count <= limit else None,
        )


#: A stream of `(points, footprints)` chunks: points (m, 3) float64, footprints (m,)
#: float64 or None. Called again for every pass.
PointChunks = Callable[[], Iterable[tuple[F64, F64 | None]]]


class StreamedPoints:
    """`PointSet` over a chunked stream, with the exact statistics of `ArrayPoints`.

    Non-finite points are dropped chunk by chunk, as `_finite_points` drops them. The box
    and the footprint percentiles are numpy's own `np.percentile` (`outofcore.percentile`,
    bit for bit); occupancy counts voxels a partition at a time (`outofcore.group_counts`),
    so memory is bounded by the partition budget and the seed limit, not by the number of
    points.
    """

    def __init__(self, chunks: PointChunks) -> None:
        self._chunks = chunks
        self._size: int | None = None
        self._has_footprint: bool | None = None

    def _finite(self) -> Iterator[tuple[F64, F64 | None]]:
        for points, prints in self._chunks():
            xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            finite = np.isfinite(xyz).all(axis=1)
            footprint = None if prints is None else np.asarray(prints, dtype=np.float64)[finite]
            yield xyz[finite], footprint

    def _describe(self) -> None:
        size, has = 0, False
        for points, prints in self._finite():
            size += int(points.shape[0])
            has = has or prints is not None
        self._size, self._has_footprint = size, has

    def size(self) -> int:
        if self._size is None:
            self._describe()
        assert self._size is not None
        return self._size

    def has_footprint(self) -> bool:
        if self._has_footprint is None:
            self._describe()
        return bool(self._has_footprint)

    def box(self) -> tuple[F64, F64]:
        def stream() -> Iterator[tuple[F64, I64]]:
            for points, _ in self._finite():
                count = points.shape[0]
                yield points.T.reshape(-1), np.repeat(np.arange(3, dtype=np.int64), count)

        found = outofcore.grouped_percentile(stream, [0, 1, 2], [0.5, 99.5])
        low = np.asarray([found[axis][0] for axis in range(3)], dtype=np.float64)
        high = np.asarray([found[axis][1] for axis in range(3)], dtype=np.float64)
        return low, high

    def footprint_percentiles(self, qs: tuple[float, ...]) -> tuple[int, list[float]]:
        def stream() -> Iterator[F64]:
            for _, prints in self._finite():
                if prints is not None:
                    yield prints[np.isfinite(prints) & (prints > 0)]

        usable = sum(int(chunk.shape[0]) for chunk in stream())
        if usable < _MIN_POINTS:
            return usable, []
        found = outofcore.percentile(stream, list(qs))
        assert found is not None
        return usable, [float(value) for value in found]

    def occupancy(
        self, origin: F64, dims: tuple[int, int, int], voxel: float, min_per_voxel: int, limit: int
    ) -> Occupancy:
        def stream() -> Iterator[tuple[I64, None]]:
            for points, _ in self._finite():
                ijk = np.floor((points - origin) / voxel)
                inside = (ijk >= 0).all(axis=1) & (ijk < np.asarray(dims)).all(axis=1)
                yield _linear(ijk[inside].astype(np.int64), dims), None

        total = retained = seed_count = 0
        seeds: list[I64] | None = []
        for keys, counts, _ in outofcore.group_counts(stream, rows=self.size()):
            supported = counts >= min_per_voxel
            total += int(counts.sum())
            retained += int(counts[supported].sum())
            seed_count += int(supported.sum())
            if seeds is not None:
                if seed_count > limit:
                    seeds = None
                else:
                    seeds.append(keys[supported])
        return Occupancy(
            total=total,
            retained=retained,
            seed_count=seed_count,
            seeds=None if seeds is None else np.sort(np.concatenate([_EMPTY, *seeds])),
        )


_EMPTY = np.zeros(0, dtype=np.int64)


def resolution(
    xyz: npt.ArrayLike,
    footprint: npt.ArrayLike,
    *,
    multiple: float = FOOTPRINT_MULTIPLE,
    percentile: float = FOOTPRINT_PERCENTILE,
    min_per_voxel: int = MIN_PER_VOXEL,
    retained_share: float = RETAINED_SHARE,
    min_cells: int = MIN_CELLS_ACROSS,
) -> Sizing | None:
    """The voxel size the points' pixel footprint supports, in model units; None when
    there are too few points or no usable footprint. See the module docstring."""
    return resolution_of(
        ArrayPoints(xyz, footprint),
        multiple=multiple,
        percentile=percentile,
        min_per_voxel=min_per_voxel,
        retained_share=retained_share,
        min_cells=min_cells,
    )


def resolution_of(
    points: PointSet,
    *,
    multiple: float = FOOTPRINT_MULTIPLE,
    percentile: float = FOOTPRINT_PERCENTILE,
    min_per_voxel: int = MIN_PER_VOXEL,
    retained_share: float = RETAINED_SHARE,
    min_cells: int = MIN_CELLS_ACROSS,
) -> Sizing | None:
    """`resolution` of any `PointSet`."""
    box = _box(points)
    if box is None or not points.has_footprint():
        return None
    return _resolution(points, box, multiple, percentile, min_per_voxel, retained_share, min_cells)


def _resolution(
    points: PointSet,
    box: tuple[F64, F64, float],
    multiple: float,
    percentile: float,
    min_per_voxel: int,
    retained_share: float,
    min_cells: int,
) -> Sizing | None:
    usable, found = points.footprint_percentiles((percentile, 10.0))
    if usable < _MIN_POINTS:
        return None
    low, high, span = box
    size, p10 = found
    ceiling = span / max(1, min_cells)
    voxel = size * multiple
    steps = 0
    bound: str | None = None
    if voxel < span / MAX_DIM:
        # A footprint absurdly fine for the extent: start where the grid can hold it, so
        # the spacing check below reaches a real answer in its bounded number of steps.
        voxel, bound = span / MAX_DIM, "max-cells"
    if voxel >= ceiling:
        voxel, bound = ceiling, "min-cells"
    else:
        while _retained(points, low, high, voxel, min_per_voxel) < retained_share:
            if steps >= _MAX_STEPS or voxel * math.sqrt(2.0) >= ceiling:
                voxel, bound = ceiling, "min-cells"
                break
            voxel *= math.sqrt(2.0)
            steps += 1
    return Sizing(
        rule="footprint",
        footprint=size,
        footprint_spread=size / p10 if p10 > 0 else None,
        multiple=multiple,
        spacing_steps=steps,
        resolution=voxel,
        voxel=voxel,
        bound=bound,
    )


def build(
    xyz: npt.ArrayLike,
    footprint: npt.ArrayLike | None = None,
    *,
    multiple: float = FOOTPRINT_MULTIPLE,
    min_per_voxel: int = MIN_PER_VOXEL,
    dilate: int = DILATE,
    max_voxels: int = MAX_VOXELS,
    max_bytes: int = MAX_ENCODED_BYTES,
) -> SupportMask | None:
    """The supported region of `xyz` (the keep tier's centres), or None when too few.

    `footprint` is each point's pixel footprint (model units per pixel, the quality
    stage's GSD); the voxel is sized from it (`resolution`). Without one the old extent
    rule applies -- `EXTENT_CELLS` across -- and `sizing.rule` says so.

    The grid spans the points' 0.5-99.5 percentile box (plus the dilation), so one far
    stray does not stretch it; strays outside it are simply not supported.
    """
    return build_from(
        ArrayPoints(xyz, footprint),
        multiple=multiple,
        min_per_voxel=min_per_voxel,
        dilate=dilate,
        max_voxels=max_voxels,
        max_bytes=max_bytes,
    )


def build_from(
    points: PointSet,
    *,
    multiple: float = FOOTPRINT_MULTIPLE,
    min_per_voxel: int = MIN_PER_VOXEL,
    dilate: int = DILATE,
    max_voxels: int = MAX_VOXELS,
    max_bytes: int = MAX_ENCODED_BYTES,
) -> SupportMask | None:
    """`build` of any `PointSet` -- the quality stage's streamed keep tier among them."""
    box = _box(points)
    if box is None:
        return None
    low, high, span = box
    sizing = (
        None
        if not points.has_footprint()
        else _resolution(
            points,
            box,
            multiple,
            FOOTPRINT_PERCENTILE,
            min_per_voxel,
            RETAINED_SHARE,
            MIN_CELLS_ACROSS,
        )
    )
    if sizing is None:
        voxel = span / max(1, EXTENT_CELLS - 2 * dilate)
        sizing = Sizing("extent", None, None, None, 0, voxel, voxel, None)
    voxel = sizing.voxel
    bound = sizing.bound
    # The per-axis cap: the grid's longest side in voxels, margin included.
    floor = span / (MAX_DIM - 2 * dilate - 2)
    if voxel < floor:
        voxel, bound = floor, "max-cells"
    # A surface's shell after dilating by `dilate` either side is about 2 * dilate + 1
    # voxels thick, which is what one supported voxel becomes.
    per_seed = 2 * max(0, dilate) + 1
    for _ in range(_MAX_STEPS):
        origin = low - dilate * voxel
        dims = _dims(low, high, voxel, dilate)
        occupancy = points.occupancy(origin, dims, voxel, min_per_voxel, max_voxels // per_seed)
        if occupancy.seed_count == 0:
            return None
        if occupancy.seed_count * per_seed > max_voxels or occupancy.seeds is None:
            # Coarsen before dilating: dilation's candidates cost an L1 ball of int64s per
            # seed (25 at `dilate` 2), which is the memory this cap is for.
            voxel *= _step(occupancy.seed_count * per_seed / max_voxels)
            bound = "max-voxels"
            continue
        occupied = _dilate(occupancy.seeds, dims, dilate)
        if occupied.size > max_voxels:
            voxel *= _step(occupied.size / max_voxels)
            bound = "max-voxels"
            continue
        starts, stops = _runs(occupied)
        encoded = len(_encode_runs(starts, stops))
        if encoded > max_bytes:
            voxel *= _step(encoded / max_bytes)
            bound = "max-bytes"
            continue
        return SupportMask(
            origin=(float(origin[0]), float(origin[1]), float(origin[2])),
            voxel=float(voxel),
            dims=dims,
            starts=starts,
            stops=stops,
            sizing=Sizing(
                rule=sizing.rule,
                footprint=sizing.footprint,
                footprint_spread=sizing.footprint_spread,
                multiple=sizing.multiple,
                spacing_steps=sizing.spacing_steps,
                resolution=sizing.resolution,
                voxel=float(voxel),
                bound=bound,
            ),
        )
    return None


def _step(ratio: float) -> float:
    """How much to grow the voxel to shrink a surface-like count by `ratio`: counts on a
    surface go as 1/voxel^2, with 5% to spare and never less than a 10% step."""
    return max(1.1, 1.05 * math.sqrt(ratio))


def _finite_points(xyz: npt.ArrayLike, footprint: npt.ArrayLike | None) -> tuple[F64, F64 | None]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    if footprint is None:
        return points[finite], None
    prints = np.asarray(footprint, dtype=np.float64).reshape(-1)
    if prints.shape[0] != points.shape[0]:
        raise ValueError("support_mask: one footprint per point is needed")
    return points[finite], prints[finite]


def _box(points: PointSet) -> tuple[F64, F64, float] | None:
    """The 0.5-99.5 percentile box and its longest side; None for too few points or none."""
    if points.size() < _MIN_POINTS:
        return None
    low, high = points.box()
    span = float(np.max(high - low))
    if not math.isfinite(span) or span <= 0:
        return None
    return low, high, span


def _dims(low: F64, high: F64, voxel: float, dilate: int) -> tuple[int, int, int]:
    sides = np.ceil((high - low) / voxel).astype(np.int64) + 1 + 2 * max(0, dilate)
    sides = np.clip(sides, 1, MAX_DIM)
    return int(sides[0]), int(sides[1]), int(sides[2])


def _linear(ijk: I64, dims: tuple[int, int, int]) -> I64:
    """C-order linear indices, as `np.ravel_multi_index` gives, which the dense format
    used too (its bits were `occupied.reshape(-1)`)."""
    return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]


def _occupancy(
    points: F64, origin: F64, dims: tuple[int, int, int], voxel: float
) -> tuple[I64, I64]:
    """The occupied voxels' linear indices, sorted, and how many points each holds."""
    ijk = np.floor((points - origin) / voxel)
    inside = (ijk >= 0).all(axis=1) & (ijk < np.asarray(dims)).all(axis=1)
    linear = _linear(ijk[inside].astype(np.int64), dims)
    unique, counts = np.unique(linear, return_counts=True)
    return unique.astype(np.int64), counts.astype(np.int64)


def _retained(points: PointSet, low: F64, high: F64, voxel: float, min_per_voxel: int) -> float:
    """The share of the points (in the percentile box) that land in supported voxels."""
    occupied = points.occupancy(low, _dims(low, high, voxel, 0), voxel, min_per_voxel, 0)
    return float(occupied.retained) / occupied.total if occupied.total else 0.0


def _dilate(seeds: I64, dims: tuple[int, int, int], radius: int) -> I64:
    """`radius` steps of 6-neighbour dilation -- the L1 ball of that radius round every
    seed -- on sorted linear indices, without a dense grid. Sorted and unique out."""
    if radius <= 0:
        return seeds
    ny, nz = dims[1], dims[2]
    i, rest = np.divmod(seeds, ny * nz)
    j, k = np.divmod(rest, nz)
    span = range(-radius, radius + 1)
    offsets = [
        (a, b, c) for a in span for b in span for c in span if abs(a) + abs(b) + abs(c) <= radius
    ]
    grown: list[I64] = []
    for a, b, c in offsets:
        ii, jj, kk = i + a, j + b, k + c
        ok = (ii >= 0) & (ii < dims[0]) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
        grown.append((ii[ok] * ny + jj[ok]) * nz + kk[ok])
    return _sorted_unique(np.concatenate(grown))


def _sorted_unique(values: I64) -> I64:
    """`np.unique` by sorting. numpy 2's default for plain `unique` is a hash table,
    which measured 100x slower than a sort on the tens of millions of dilation
    candidates here."""
    ordered = np.sort(values)
    if ordered.size == 0:
        return ordered.astype(np.int64)
    out: I64 = ordered[np.concatenate([[True], ordered[1:] != ordered[:-1]])].astype(np.int64)
    return out


def _runs(indices: I64) -> tuple[I64, I64]:
    """Maximal runs of consecutive values in sorted unique `indices`: starts, stops."""
    if indices.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = indices[np.concatenate([[0], breaks + 1])]
    stops = indices[np.concatenate([breaks, [indices.size - 1]])] + 1
    return starts.astype(np.int64), stops.astype(np.int64)


# ---------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------


def _encode_runs(starts: I64, stops: I64) -> str:
    """Per run the gap from the previous run's stop (the first from 0), then its length."""
    values = np.empty(2 * starts.size, dtype=np.uint64)
    previous = np.concatenate([[0], stops[:-1]]).astype(np.int64)
    values[0::2] = (starts - previous).astype(np.uint64)
    values[1::2] = (stops - starts).astype(np.uint64)
    return base64.b64encode(zlib.compress(_varints(values), 9)).decode("ascii")


def _decode_runs(data: str, cells: int) -> tuple[I64, I64]:
    raw = _inflate(data, 2 * MAX_RUNS * _MAX_VARINT_BYTES)
    values = _unvarints(raw)
    if values.size % 2:
        raise ValueError("support_mask's runs are not (gap, length) pairs")
    gaps, lengths = values[0::2], values[1::2]
    # Each value is at most `cells` before anything is summed, so the sums below stay far
    # inside int64 (runs <= MAX_RUNS, cells <= 2**48).
    if (
        values.size // 2 > MAX_RUNS
        or (values > np.uint64(cells)).any()
        or (lengths < 1).any()
        or (gaps[1:] < 1).any()
    ):
        raise ValueError("support_mask's runs are not sorted, disjoint and non-empty")
    ends = np.cumsum(gaps.astype(np.int64) + lengths.astype(np.int64))
    stops = ends
    starts = ends - lengths.astype(np.int64)
    if stops.size and int(stops[-1]) > cells:
        raise ValueError("support_mask's runs run past the end of its dims")
    return starts, stops


def _decode_dense(bits: str, cells: int) -> tuple[I64, I64]:
    """The dense format: packed bits of every cell, C order."""
    packed = _inflate(bits, (cells + 7) // 8)
    if len(packed) != (cells + 7) // 8:
        raise ValueError("support_mask's bits do not fill its dims")
    occupied = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:cells]
    return _runs(np.flatnonzero(occupied).astype(np.int64))


def _inflate(data: str, limit: int) -> bytes:
    """base64 then zlib, refusing more than `limit` bytes out: a parameter cannot expand
    into gigabytes, and trailing or truncated data is refused rather than guessed at."""
    try:
        compressed = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("support_mask's data is not base64") from error
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(compressed, limit + 1)
    except zlib.error as error:
        raise ValueError("support_mask's data is not zlib") from error
    if len(out) > limit or inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
        raise ValueError("support_mask's data is too large, truncated or has trailing bytes")
    return out


def _varints(values: npt.NDArray[np.uint64]) -> bytes:
    """Unsigned LEB128, vectorised: seven bits a byte, low first, high bit = more."""
    if values.size == 0:
        return b""
    widths = np.ones(values.size, dtype=np.int64)
    rest = values >> np.uint64(7)
    while rest.any():
        widths += rest > 0
        rest = rest >> np.uint64(7)
    width = int(widths.max())
    shifts = np.uint64(7) * np.arange(width, dtype=np.uint64)
    groups = ((values[:, None] >> shifts[None, :]) & np.uint64(0x7F)).astype(np.uint8)
    position = np.arange(width)[None, :]
    groups |= np.where(position < widths[:, None] - 1, np.uint8(0x80), np.uint8(0))
    # Row-major boolean selection keeps each value's bytes together and in order.
    return groups[position < widths[:, None]].tobytes()


def _unvarints(raw: bytes) -> npt.NDArray[np.uint64]:
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size == 0:
        return np.zeros(0, dtype=np.uint64)
    last = data < 0x80
    if not last[-1]:
        raise ValueError("support_mask's runs end inside a number")
    ends = np.flatnonzero(last)
    firsts = np.concatenate([[0], ends[:-1] + 1])
    widths = ends - firsts + 1
    # A longer number than any index needs, or a padded one (a final zero byte after
    # others), is not something `to_dict` writes.
    if widths.max() > _MAX_VARINT_BYTES or ((widths > 1) & (data[ends] == 0)).any():
        raise ValueError("support_mask's runs hold a malformed number")
    position = np.arange(data.size) - np.repeat(firsts, widths)
    parts = (data & 0x7F).astype(np.uint64) << (np.uint64(7) * position.astype(np.uint64))
    return np.add.reduceat(parts, firsts).astype(np.uint64)


def _int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
