"""`confidence_grid`: how well each part of an epoch was seen, per cell of its site's lattice.

The quality stage measured, per trained gaussian, whether the capture supports it (its
tier) and, when not, which keep criterion it failed (`support/`, `quality.SUPPORT`). This
stage moves those gaussians into the tileset frame by the very similarity `place` applied
(`placement.json`) and bins them into the site lattice (tools/captures `cell_grid.py`:
2.5D columns, `0.25 m x 2^level`, `i = floor(x / cell)` from the frame origin), so every
epoch of a site -- which shares one frame (D3) -- lands on cells that nest with every other
epoch's.

**Score** (policy `weak-v1`; first principles, not calibrated -- docs/DECISIONS/0016). Over
the gaussians in a cell at least 0.05 opaque (the coverage-candidate rule; as stored,
`round(255 alpha) >= 13`):

    keepShare = sum(alpha [tier == keep]) / sum(alpha)
    score     = 1 + round(254 keepShare f)          (0: no evidence at all)

where `f` is the alignment factor below. A cell is **weak** when `score < 128` -- fewer
than half of what is visible in it, by alpha, passed keep: a majority rule -- or when it
is **unseen**: inside the disc the cameras were pointed at (the quality stage's region of
interest, placed like the gaussians) with no evidence. **Reasons**: bit `k` is set when
more than half of the cell's non-keep alpha failed criterion `k` (views, spread, GSD,
held-out accuracy -- the recipe's own thresholds, as `support/` recorded them), and
`alignment` when `f < 1`. **Relief**: the cell's gaussians' height range in 0.25 m.

**The alignment cap.** A cell can only claim its evidence when the epoch's placement
error is under half the cell: `f = min(1, cell / (2 u_rel))`, where `u_rel` is the
epoch's uncertainty *relative to the site frame* -- 0 when `site_context.json` says no
epoch came before it (this epoch defines the frame), else `epoch_alignment.json`'s
`uncertaintyM`, else `georef.json`'s. So an unaligned drone epoch placed by its GPS
(+/-10 m) scores `f = 0.05` on 1 m cells and is weak everywhere, with the `alignment`
reason, which is the honest answer: nobody knows where its evidence is to a metre.
`uncertaintyM` of a visual alignment includes its reference's own uncertainty
(alignment.py `uncertainty`), so the cap is conservative there (ADR 0016, open questions).

**Level.** The smallest level at which the median occupied cell holds at least
`MIN_CELL_GAUSSIANS` (16) visible gaussians -- a binomial share from 16 has sigma <= 0.125,
enough to tell under a half from three quarters -- and the plane holds at most `MAX_CELLS`.
Above it, `PYRAMID_LEVELS` coarser levels, min-of-children (`cell_grid.min_pyramid`).

**Extent.** The plane covers the visible gaussians within the packer's own floater rule
(1.5x the 99.5th percentile distance from their median, here in plan) and the pointed
disc; what lies past it is counted (`summary.outside`) and not binned, so one sky floater
cannot make every cell coarse.

**Memory and determinism.** `support/` is read a chunk at a time, in five passes: the
median, the floater radius, the extent, the counts that choose the level, and the sums
(integers: alpha in 1/255ths, so nothing depends on the order they are added in), in
bands of at most `BAND_CELLS` cells -- so the stage holds a chunk and a few planes,
whatever the size of the splat. No matrix products: the placement is written out per
component, so no BLAS kernel can change a bit of it.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import alignment
import outofcore
import quality
import splat_io
from artifacts import ArtifactDecl
from captures_bridge import cell_grid
from contracts import MetricValue, StageContext, StageOutcome
from placement import PLACEMENT
from registry import stage_impl

__all__ = [
    "BAND_CELLS",
    "CONFIDENCE",
    "MAX_CELLS",
    "MIN_CELL_GAUSSIANS",
    "PYRAMID_LEVELS",
    "WEAK_BELOW",
    "Binned",
    "Placement",
    "alignment_factor",
    "bin_support",
    "choose_level",
    "read_placement",
    "relative_uncertainty",
    "score_plane",
    "unseen_cells",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

DESCRIPTOR_FILE = "confidence.json"
BIN_FILE = "confidence.bin"
CONFIDENCE = ArtifactDecl(
    "confidence",
    kind="dir",
    content_type="inode/directory",
    summary="how well each part of this epoch was seen, per cell of the site lattice in the "
    "tileset frame (hexapod.cellgrid: score, reasons, relief; weak-v1), capped by its "
    "alignment uncertainty; confidence.json is the descriptor package declares on the root",
    required_members=(DESCRIPTOR_FILE, BIN_FILE),
    stub_members=(DESCRIPTOR_FILE, BIN_FILE),
)

WEAK_BELOW = cell_grid.WEAK_BELOW
MIN_CELL_GAUSSIANS = 16
MAX_CELLS = 1 << 22
PYRAMID_LEVELS = 8
#: Cells summed in one pass over `support/`: a band of rows of the plane. 2^20 cells of
#: the accumulators (seven int64 and two float64) is 72 MB.
BAND_CELLS = 1 << 20
#: Visible: at least 0.05 opaque, as `support/` stores alpha (`round(255 alpha)`).
VISIBLE_ALPHA = 13
#: The packer's floater rule (tools/captures splat_tiles `_filter_rows`), in plan.
FLOATER_PERCENTILE = 99.5
FLOATER_FACTOR = 1.5
#: What an epoch is worth relative to its frame when nothing measured it: the same ten
#: metres a hand placement is (stages.DEFAULT_MANUAL_UNCERTAINTY_M).
UNKNOWN_UNCERTAINTY_M = 10.0

_KEEP_BITS = tuple((name, bit) for name, bit in quality.SUPPORT_REASON_BITS.items())
if any(cell_grid.REASON_BITS[name] != bit for name, bit in _KEEP_BITS):  # pragma: no cover
    raise RuntimeError("quality.SUPPORT_REASON_BITS must be cell_grid.REASON_BITS' first bits")
UNSEEN = cell_grid.REASON_BITS["unseen"]
ALIGNMENT = cell_grid.REASON_BITS["alignment"]


# --- inputs ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """`placement.json`: `x' = scale * rotation @ x + translation`, COLMAP to tileset frame."""

    scale: float
    rotation: F64
    translation: F64

    def apply(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike) -> F64:
        """Each point placed, `(n, 3)` float64, component by component (no BLAS)."""
        px, py, pz = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
        r, s, t = self.rotation, self.scale, self.translation
        out = np.empty((px.shape[0], 3), dtype=np.float64)
        for axis in range(3):
            out[:, axis] = s * (r[axis, 0] * px + r[axis, 1] * py + r[axis, 2] * pz) + t[axis]
        return out


def read_placement(path: Path) -> Placement:
    """The inverse of `placement.compose`: its similarity back as arrays, checked."""
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        scale = float(document["scale"])
        rotation = np.asarray(document["rotation"], dtype=np.float64)
        translation = np.asarray(document["translation"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as problem:
        raise ValueError(f"{path.name} is not a placement: {problem}") from problem
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not (math.isfinite(scale) and scale > 0)
        or not (np.isfinite(rotation).all() and np.isfinite(translation).all())
    ):
        raise ValueError(f"{path.name} is not a similarity (scale, 3x3 rotation, translation)")
    return Placement(scale, rotation, translation)


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def prior_epochs(site_context: Mapping[str, Any] | None) -> int:
    """`site_context.json`'s `priorEpochs`, or what it implies: none without a reference
    epoch, one with (the worker writes the key from A4 on)."""
    if site_context is None:
        return 0
    value = site_context.get("priorEpochs")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0 if site_context.get("reference") is None else 1


def relative_uncertainty(
    alignment_doc: Mapping[str, Any] | None,
    georef: Mapping[str, Any] | None,
    site_context: Mapping[str, Any] | None,
) -> tuple[float, str, int]:
    """`u_rel` (module docstring), where it came from, and the prior epochs it assumed."""
    prior = prior_epochs(site_context)
    if prior == 0:
        return 0.0, alignment.SITE_CONTEXT_FILE, prior
    for document, source in (
        (alignment_doc, alignment.EPOCH_ALIGNMENT.name),
        (georef, "georef.json"),
    ):
        value = None if document is None else _positive(document.get("uncertaintyM"))
        if value is not None:
            return value, source, prior
    return UNKNOWN_UNCERTAINTY_M, "unknown", prior


def alignment_factor(cell_m: float, u_rel: float) -> float:
    """`f = min(1, cell / (2 u_rel))`: 1 when the placement error is under half a cell."""
    if u_rel <= 0:
        return 1.0
    return min(1.0, cell_m / (2.0 * u_rel))


# --- binning -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Support:
    """`support/`'s columns, read a chunk at a time."""

    folder: Path
    document: Mapping[str, Any]
    chunk: int

    @property
    def rows(self) -> int:
        return int(self.document["rows"])

    def column(self, name: str) -> splat_io.Column:
        spec = self.document["columns"][name]
        return splat_io.Column(self.folder / str(spec["file"]), str(spec["dtype"]))

    def chunks(self, *names: str) -> Iterator[list[npt.NDArray[Any]]]:
        columns = [self.column(name) for name in names]
        for start, stop in splat_io.ranges(self.rows, self.chunk):
            yield [column.read(start, stop) for column in columns]


def read_support(folder: Path, chunk: int = splat_io.CHUNK) -> _Support:
    document = json.loads((folder / quality.SUPPORT_FILE).read_text(encoding="utf-8"))
    if document.get("status") != "measured" or document.get("frame") != "colmap":
        raise ValueError(
            f"support/ is not measured evidence in the COLMAP frame "
            f"(status {document.get('status')!r}, frame {document.get('frame')!r})"
        )
    support = _Support(folder, document, chunk)
    for name in ("x", "y", "z", "alpha", "tier", "reasons"):
        if support.column(name).size != support.rows:
            raise ValueError(
                f"support/{name} holds {support.column(name).size} rows, not {support.rows}"
            )
    return support


@dataclass(frozen=True)
class _Extent:
    """The plane's rectangle in metres, and the floater rule that bounded it."""

    x0: float
    x1: float
    y0: float
    y1: float
    centre: tuple[float, float] | None
    radius: float | None

    def cells(self, level: int) -> tuple[int, int, int, int]:
        """`(i0, j0, ni, nj)` covering the rectangle at `level`."""
        corners = np.array([[self.x0, self.y0], [self.x1, self.y1]])
        i, j = cell_grid.Lattice(level).index(corners)
        return int(i[0]), int(j[0]), int(i[1] - i[0]) + 1, int(j[1] - j[0]) + 1


@dataclass(frozen=True)
class Disc:
    """The placed region of interest in plan: what the capture should have seen."""

    x: float
    y: float
    radius: float


def _placed_roi(document: Mapping[str, Any], placement: Placement) -> Disc | None:
    roi = document.get("roi")
    if not isinstance(roi, Mapping):
        return None
    try:
        centre = [float(v) for v in roi["center"]]
        radius = float(roi["radius"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(centre) != 3 or not all(math.isfinite(v) for v in (*centre, radius)) or radius <= 0:
        return None
    placed = placement.apply([centre[0]], [centre[1]], [centre[2]])[0]
    return Disc(float(placed[0]), float(placed[1]), radius * placement.scale)


Visible = Callable[[], Iterator[tuple[F64, npt.NDArray[Any], npt.NDArray[Any]]]]


def _visible(support: _Support, placement: Placement, extra: tuple[str, ...] = ()) -> Visible:
    """A stream of the visible rows: placed positions and the columns asked for."""

    def stream() -> Iterator[tuple[F64, npt.NDArray[Any], npt.NDArray[Any]]]:
        for x, y, z, alpha, *rest in support.chunks("x", "y", "z", "alpha", *extra):
            placed = placement.apply(x, y, z)
            seen = (alpha >= VISIBLE_ALPHA) & np.isfinite(placed).all(axis=1)
            others = np.zeros((int(seen.sum()), len(rest)), dtype=np.int64)
            for column, values in enumerate(rest):
                others[:, column] = values[seen]
            yield placed[seen], alpha[seen], others

    return stream


def _extent(visible: Visible, roi: Disc | None) -> tuple[_Extent | None, int]:
    """The rectangle the plane covers (module docstring), and how many gaussians are
    visible at all."""
    medians = outofcore.grouped_median(
        lambda: (
            (
                np.concatenate([placed[:, 0], placed[:, 1]]),
                np.repeat(np.arange(2, dtype=np.int64), placed.shape[0]),
            )
            for placed, _, _ in visible()
        ),
        [0, 1],
    )
    if 0 not in medians:
        if roi is None:
            return None, 0
        r = roi.radius
        return _Extent(roi.x - r, roi.x + r, roi.y - r, roi.y + r, None, None), 0
    cx, cy = float(medians[0]), float(medians[1])

    def distances() -> Iterator[F64]:
        for placed, _, _ in visible():
            yield np.hypot(placed[:, 0] - cx, placed[:, 1] - cy)

    found = outofcore.percentile(distances, [FLOATER_PERCENTILE])
    assert found is not None
    radius = FLOATER_FACTOR * float(found[0])
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    total = 0
    for placed, _, _ in visible():
        near = np.hypot(placed[:, 0] - cx, placed[:, 1] - cy) <= radius
        total += int(placed.shape[0])
        if near.any():
            low = np.minimum(low, placed[near, :2].min(axis=0))
            high = np.maximum(high, placed[near, :2].max(axis=0))
    if roi is not None:
        low = np.minimum(low, [roi.x - roi.radius, roi.y - roi.radius])
        high = np.maximum(high, [roi.x + roi.radius, roi.y + roi.radius])
    rectangle = _Extent(
        float(low[0]), float(high[0]), float(low[1]), float(high[1]), (cx, cy), radius
    )
    return rectangle, total


def _finest_level(extent: _Extent, max_cells: int) -> int:
    for level in range(cell_grid.MAX_LEVEL + 1):
        _, _, ni, nj = extent.cells(level)
        if ni * nj <= max_cells:
            return level
    return cell_grid.MAX_LEVEL


def _keys(
    placed: F64, level: int, i0: int, j0: int, ni: int, nj: int
) -> tuple[I64, npt.NDArray[np.bool_]]:
    i, j = cell_grid.Lattice(level).index(placed)
    di, dj = i - i0, j - j0
    inside = (di >= 0) & (di < ni) & (dj >= 0) & (dj < nj)
    return np.where(inside, dj * ni + di, 0), inside


def choose_level(
    counts: npt.NDArray[np.integer[Any]],
    level: int,
    i0: int,
    j0: int,
    *,
    min_cell_gaussians: int = MIN_CELL_GAUSSIANS,
) -> int:
    """The smallest level, from `level` up, at which the median occupied cell holds at
    least `min_cell_gaussians` -- counted exactly at each level from the finest plane's
    counts `(nj, ni)`, since every coarser cell is a union of finer ones. With nothing
    occupied, `level` itself."""
    plane = cell_grid.CellPlane(
        level, i0, j0, counts.shape[1], counts.shape[0], counts.astype(np.int64)[..., None]
    )
    while True:
        occupied = plane.data[..., 0][plane.data[..., 0] > 0]
        if occupied.size == 0:
            return plane.level
        middle = np.sort(occupied)
        n = middle.size
        median2 = int(middle[(n - 1) // 2]) + int(middle[n // 2])  # twice the median
        if median2 >= 2 * min_cell_gaussians:
            return plane.level
        if plane.level >= cell_grid.MAX_LEVEL or (plane.ni <= 2 and plane.nj <= 2):
            return plane.level
        plane = _sum_up(plane)


def _sum_up(plane: cell_grid.CellPlane) -> cell_grid.CellPlane:
    """One level up by sums: the counts of a coarser lattice, exactly."""
    pi0, pj0 = plane.i0 >> 1, plane.j0 >> 1
    ni = ((plane.i0 + plane.ni - 1) >> 1) - pi0 + 1
    nj = ((plane.j0 + plane.nj - 1) >> 1) - pj0 + 1
    padded = np.zeros((2 * nj, 2 * ni, 1), dtype=np.int64)
    di, dj = plane.i0 - 2 * pi0, plane.j0 - 2 * pj0
    padded[dj : dj + plane.nj, di : di + plane.ni] = plane.data
    summed = padded.reshape(nj, 2, ni, 2, 1).sum(axis=(1, 3))
    return cell_grid.CellPlane(plane.level + 1, pi0, pj0, ni, nj, summed)


@dataclass(frozen=True)
class Binned:
    """A level's per-cell sums: what `score_plane` turns into channels."""

    level: int
    i0: int
    j0: int
    ni: int
    nj: int
    count: I64
    alpha: I64
    keep: I64
    #: Per keep criterion (`quality.SUPPORT_REASON_BITS` order), non-keep alpha failing it.
    failed: I64
    zmin: F64
    zmax: F64
    #: Visible gaussians the plane does not hold (past the floater radius), and those it does.
    outside: int
    binned: int


def bin_support(
    support: _Support,
    placement: Placement,
    roi: Disc | None,
    *,
    min_cell_gaussians: int = MIN_CELL_GAUSSIANS,
    max_cells: int = MAX_CELLS,
    band_cells: int = BAND_CELLS,
) -> Binned | None:
    """The sums of every visible gaussian per cell, at the level `choose_level` picks.
    None when there is nothing to bin and no disc to expect anything in."""
    visible = _visible(support, placement, ("tier", "reasons"))
    extent, total = _extent(visible, roi)
    if extent is None:
        return None
    finest = _finest_level(extent, max_cells)
    i0, j0, ni, nj = extent.cells(finest)
    counts = np.zeros(ni * nj, dtype=np.int64)
    for placed, _, _ in visible():
        keys, inside = _keys(placed, finest, i0, j0, ni, nj)
        counts += np.bincount(keys[inside], minlength=ni * nj).astype(np.int64)
    level = choose_level(
        counts.reshape(nj, ni), finest, i0, j0, min_cell_gaussians=min_cell_gaussians
    )
    i0, j0, ni, nj = extent.cells(level)
    cells = ni * nj
    count = np.zeros(cells, dtype=np.int64)
    alpha_sum = np.zeros(cells, dtype=np.int64)
    keep_sum = np.zeros(cells, dtype=np.int64)
    failed = np.zeros((cells, len(_KEEP_BITS)), dtype=np.int64)
    zmin = np.full(cells, np.inf)
    zmax = np.full(cells, -np.inf)
    rows_per_band = max(1, band_cells // max(1, ni))
    binned = 0
    for band0 in range(0, nj, rows_per_band):
        band1 = min(nj, band0 + rows_per_band)
        low, high = band0 * ni, band1 * ni
        size = high - low
        for placed, alpha, rest in visible():
            keys, inside = _keys(placed, level, i0, j0, ni, nj)
            mine = inside & (keys >= low) & (keys < high)
            if not mine.any():
                continue
            local = keys[mine] - low
            weight = alpha[mine].astype(np.int64)
            tier, reasons = rest[mine, 0], rest[mine, 1]
            binned += int(local.size)
            count[low:high] += np.bincount(local, minlength=size)
            # Integer weights summed by bincount in float64: exact below 2^53, so the
            # sums are the same whatever order the chunks or rows come in.
            alpha_sum[low:high] += np.rint(
                np.bincount(local, weights=weight, minlength=size)
            ).astype(np.int64)
            kept = tier == quality.TIER_KEEP
            keep_sum[low:high] += np.rint(
                np.bincount(local[kept], weights=weight[kept], minlength=size)
            ).astype(np.int64)
            for column, (_, bit) in enumerate(_KEEP_BITS):
                hit = ~kept & ((reasons & bit) > 0)
                failed[low:high, column] += np.rint(
                    np.bincount(local[hit], weights=weight[hit], minlength=size)
                ).astype(np.int64)
            z = placed[mine, 2]
            np.minimum.at(zmin, local + low, z)
            np.maximum.at(zmax, local + low, z)
    # What the plane's rectangle left out: the floaters past the radius, in plan.
    outside = total - binned
    return Binned(
        level, i0, j0, ni, nj, count, alpha_sum, keep_sum, failed, zmin, zmax, outside, binned
    )


def _empty() -> Binned:
    none = np.zeros(1, dtype=np.int64)
    failed = np.zeros((1, len(_KEEP_BITS)), dtype=np.int64)
    return Binned(0, 0, 0, 1, 1, none, none, none, failed, np.zeros(1), np.zeros(1), 0, 0)


def unseen_cells(
    roi: Disc | None,
    level: int,
    i0: int,
    j0: int,
    ni: int,
    nj: int,
    occupied: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Cells `(nj, ni)` whose centre is inside the placed disc and that hold nothing."""
    if roi is None:
        return np.zeros((nj, ni), dtype=bool)
    cell = cell_grid.Lattice(level).cell_m
    cx = (np.arange(i0, i0 + ni, dtype=np.float64) + 0.5) * cell
    cy = (np.arange(j0, j0 + nj, dtype=np.float64) + 0.5) * cell
    inside = (cx[None, :] - roi.x) ** 2 + (cy[:, None] - roi.y) ** 2 <= roi.radius**2
    unseen: npt.NDArray[np.bool_] = inside & ~occupied.reshape(nj, ni)
    return unseen


def score_plane(
    binned: Binned, factor: float, unseen: npt.NDArray[np.bool_]
) -> cell_grid.CellPlane:
    """The channels (`cell_grid.CONFIDENCE_CHANNELS`) of the binned level (module docstring)."""
    occupied = binned.count > 0
    alpha = binned.alpha.astype(np.float64)
    share = np.divide(
        binned.keep.astype(np.float64), alpha, out=np.zeros_like(alpha), where=alpha > 0
    )
    score = np.where(occupied, 1 + np.floor(254.0 * share * factor + 0.5), 0).astype(np.int64)
    reasons = np.zeros(binned.count.shape, dtype=np.int64)
    short = binned.alpha - binned.keep
    for column, (_, bit) in enumerate(_KEEP_BITS):
        reasons |= np.where(2 * binned.failed[:, column] > short, bit, 0)
    if factor < 1.0:
        reasons |= np.where(occupied, ALIGNMENT, 0)
    reasons = reasons.reshape(binned.nj, binned.ni)
    reasons |= np.where(unseen, UNSEEN, 0)
    span = np.where(occupied, binned.zmax - binned.zmin, 0.0)
    relief = np.clip(np.floor(span / cell_grid.RELIEF_UNIT_M + 0.5), 0, 255).astype(np.int64)
    data = np.stack(
        [score.reshape(binned.nj, binned.ni), reasons, relief.reshape(binned.nj, binned.ni)],
        axis=-1,
    ).astype(np.uint8)
    return cell_grid.CellPlane(binned.level, binned.i0, binned.j0, binned.ni, binned.nj, data)


def summarise(
    plane: cell_grid.CellPlane, binned: Binned, metres: float | None, median_gsd: float | None
) -> dict[str, Any]:
    """`summary` of the finest level: counts, the weak share and why."""
    score = plane.data[..., 0].astype(np.int64)
    reasons = plane.data[..., 1].astype(np.int64)
    weak = cell_grid.weak_mask(plane)
    unseen = (reasons & UNSEEN) > 0
    scored = int((score > 0).sum())
    expected = scored + int(unseen.sum())
    gsd_mm = None
    if metres is not None and median_gsd is not None and math.isfinite(median_gsd):
        gsd_mm = round(median_gsd * metres * 1000.0, 2)
    return {
        "cells": plane.cells,
        "scored": scored,
        "weak": int(weak.sum()),
        "unseen": int(unseen.sum()),
        # Of the cells this epoch speaks for -- scored, or expected and unseen.
        "weakShare": round(int(weak.sum()) / expected, 4) if expected else 0.0,
        "reasons": {
            name: int((weak & ((reasons & bit) > 0)).sum())
            for name, bit in cell_grid.REASON_BITS.items()
        },
        "gaussians": binned.binned,
        "outside": binned.outside,
        "medianGsdMm": gsd_mm,
        "metric": metres is not None,
    }


def _read_optional(
    ctx: StageContext, name: str, member: str | None = None
) -> dict[str, Any] | None:
    if not ctx.has_input(name):
        return None
    path = ctx.input(name)
    if member is not None:
        path = path / member
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


@stage_impl(
    "confidence_grid",
    consumes=(quality.SUPPORT.name, PLACEMENT.name),
    optional_consumes=(
        alignment.EPOCH_ALIGNMENT.name,
        alignment.SITE_CONTEXT,
        "georef.json",
    ),
    produces=(CONFIDENCE,),
    summary="bin the quality stage's evidence into the site's cells: a confidence map per "
    "epoch, capped by its alignment",
)
def confidence_grid(ctx: StageContext) -> StageOutcome:
    """See the module. Writes `confidence/confidence.json` (the descriptor) and
    `confidence/confidence.bin`; never fails a run for a weak map."""
    started = time.perf_counter()
    base = float(ctx.param("base_cell_m", cell_grid.LATTICE_BASE_M))
    if base != cell_grid.LATTICE_BASE_M:
        raise ValueError(
            f"confidence.base_cell_m is the site lattice's, {cell_grid.LATTICE_BASE_M} m; "
            f"{base} would put this epoch's cells on no other epoch's"
        )
    min_cell = int(ctx.param("min_cell_gaussians", MIN_CELL_GAUSSIANS))
    max_cells = int(ctx.param("max_cells", MAX_CELLS))
    levels = int(ctx.param("pyramid_levels", PYRAMID_LEVELS))
    chunk = int(ctx.param("chunk_gaussians", splat_io.CHUNK))
    band = int(ctx.param("band_cells", BAND_CELLS))
    support = read_support(ctx.input(quality.SUPPORT.name), chunk)
    placement = read_placement(ctx.input(PLACEMENT.name))
    roi = _placed_roi(support.document, placement)
    aligned = _read_optional(ctx, alignment.EPOCH_ALIGNMENT.name)
    georef = _read_optional(ctx, "georef.json")
    context = _read_optional(ctx, alignment.SITE_CONTEXT, alignment.SITE_CONTEXT_FILE)
    u_rel, source, prior = relative_uncertainty(aligned, georef, context)
    binned = bin_support(
        support, placement, roi, min_cell_gaussians=min_cell, max_cells=max_cells, band_cells=band
    )
    if binned is None:
        # Nothing visible and nowhere it was expected: one empty cell at the origin, so
        # the artifact still has its shape and says it measured nothing.
        binned = _empty()
    cell_m = cell_grid.Lattice(binned.level).cell_m
    factor = alignment_factor(cell_m, u_rel)
    unseen = unseen_cells(
        roi, binned.level, binned.i0, binned.j0, binned.ni, binned.nj, binned.count > 0
    )
    plane = score_plane(binned, factor, unseen)
    planes = cell_grid.min_pyramid(plane, levels)
    metres = _positive(support.document.get("metresPerUnit"))
    median_gsd = _positive(support.document.get("medianGsd"))
    summary = summarise(plane, binned, metres, median_gsd)
    meta: dict[str, object] = {
        "reasonBits": dict(cell_grid.REASON_BITS),
        "policy": cell_grid.WEAK_POLICY,
        "weakBelow": WEAK_BELOW,
        "alignmentCap": {
            "uncertaintyM": round(u_rel, 4),
            "priorEpochs": prior,
            "factor": round(factor, 6),
            "source": source,
        },
        # Where `unseen` was expected: the placed region of interest, in the tileset frame.
        "roiDisc": None
        if roi is None
        else {"centerM": [round(roi.x, 4), round(roi.y, 4)], "radiusM": round(roi.radius, 4)},
        "summary": summary,
    }
    grid = cell_grid.CellGrid("confidence", cell_grid.CONFIDENCE_CHANNELS, tuple(planes), meta)
    out = ctx.output(CONFIDENCE.name)
    descriptor = cell_grid.encode(grid, out, BIN_FILE)
    (out / DESCRIPTOR_FILE).write_text(json.dumps(descriptor, indent=1) + "\n", encoding="utf-8")
    seconds = time.perf_counter() - started
    ctx.log(
        f"confidence: level {binned.level} ({cell_m:g} m cells), {plane.ni}x{plane.nj}, "
        f"{summary['scored']} scored, {summary['weak']} weak ({summary['weakShare']:.1%}), "
        f"{summary['unseen']} unseen; cap f={factor:.3g} from u={u_rel:g} m ({source}, "
        f"{prior} prior epoch(s)); {len(planes)} levels; {seconds:.1f} s"
    )
    metrics: dict[str, MetricValue] = {
        "level": binned.level,
        "cellM": cell_m,
        "cells": plane.cells,
        "scored": int(summary["scored"]),
        "weak": int(summary["weak"]),
        "weakShare": float(summary["weakShare"]),
        "alignmentFactor": round(factor, 6),
        "uncertaintyM": round(u_rel, 4),
        "levels": len(planes),
        "seconds": round(seconds, 2),
    }
    return StageOutcome(
        metrics=metrics,
        summary=f"{float(summary['weakShare']):.0%} of the cells it speaks for are weak",
    )
