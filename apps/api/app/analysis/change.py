"""The change between two epochs, from their height surfaces (M5, ADR 0018 § API).

**Geometry first.** Each epoch's run measured a height surface per cell (P5, `surface.bin`:
its top surface `dsm`, its bare `ground` and the spread `sigma` of its highest samples, in
centimetres above the site frame's origin, on its confidence grid's plane). Two epochs of
one site share that frame, and their cells nest (`0.25 m * 2**level`), so the change is a
comparison cell by cell -- never of splat pixels, which would read a season or a sky as
geometry. Appearance is not compared at all (`appearance-only` is a reserved class that is
never emitted): that needs the new photos compared with the old epoch at fixed geometry.

**Resampling.** Both surfaces are brought to the coarser of their two levels, and each
coarse cell is taken over **only the children both epochs measured**: of those, `dsm` the
highest, `ground` the lowest and `sigma` the largest (P5's demo `coarsen_surface` rule,
tools/pipeline `experiments/confidence_demo.py`, which `coarsen_surface` here matches byte
for byte on the committed fixture `data/tiles/synthetic-yard-surfaces`). Coarsening each
epoch on its own would let one side's single ground child stand for a whole cell against
the other side's crate in a sibling, or set one side's uphill children against the other's
downhill ones: a change that is only a difference in what was measured. So the finer epoch
is first brought to the coarser one's level over the children whose coarse cell the coarser
one measured, then both go up together over the children both measured. A coarse cell where
each side measured something but no child is shared is `insufficient`; one whose shared
children are fewer than either side measured (the coarser epoch counts as having measured
all of its cell) is compared and flagged `lowConfidence`. If the two windows together would
still be over `MAX_CHANGE_CELLS` cells, both go up further until they are not (and say so).

**The rule** (`change-v1`). Where both epochs have a surface, `delta = dsm_to - dsm_from`
(computed in 32 bits: two i2 heights can differ by more than i2 holds) against `t = 3 *
sqrt(sigma_from^2 + sigma_to^2 + u_from^2 + u_to^2)`, where `u` is each epoch's
uncertainty from the site frame (its confidence grid's `alignmentCap.uncertaintyM`: 0 for
the epoch that made the frame, its alignment's for one aligned to it, its GPS's for one
nobody could align). `raised` when `delta > t`, `lowered` when `delta < -t`, else
`unchanged`; `insufficient` where either epoch measured something but they share no child;
no class where neither measured anything. The 3 is a first-principles choice (three
standard deviations of the sum of two independent errors), not a measurement: the map says
`uncalibrated` until a site scanned twice measures it. `minDetectableM` is the least `t`
over the compared cells: a change smaller than it cannot be seen anywhere between these
two dates -- an epoch placed by GPS alone at +/-10 m makes it about 30 m, and so finds no
change rather than false ones.

**Low confidence is flagged, not hidden.** A compared cell is `lowConfidence` when either
epoch's confidence grid does not vouch for it at the compared level (weak-v1: scored under
128, or unseen, or no evidence at all; the grid's own min-of-children pyramid, so a coarse
cell is as weak as its weakest child), when it was compared over fewer children than either
side measured (above), or when either sigma is saturated (255 cm means at least 2.55 m: the
threshold is then a lower bound). It is still classified: P5's demo found
the weak cells of a real change were its crate's own sides, so hiding them would hide the
change.

**Areas.** The changed and insufficient cells become outlines in the site frame
(`change_areas`): every 4-connected region of one class (raised and lowered split further
by low confidence) is a polygon with its exact cell edges, through
`rasterio.features.shapes` as weak areas are outlined, then to the globe through the
frame's own tangent plane (`weak_areas.site_projection`). Unchanged cells are counted, not
outlined: an area of them would read as confident wherever the comparison could not see.
The outline is bounded as weak areas' are: a map over `weak_areas.MAX_ZONE_CELLS` cells or
over `weak_areas.MAX_ZONE_EDGES` cell edges of outline is classified again a level up (from
the surfaces, by the same rule) and drawn there, and its areas' numbers are of the cells
they were drawn from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from rasterio.features import rasterize, shapes
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from app.analysis import cell_grid
from app.analysis.weak_areas import (
    MAX_ZONE_CELLS,
    MAX_ZONE_EDGES,
    SiteProjection,
    cell_affine,
    polygon_parts,
)

POLICY = "change-v1"
#: The threshold's multiple of the combined standard deviation (module docstring).
THRESHOLD_SIGMA = 3.0
#: The two windows together are compared on at most this many cells (2048 x 2048, the
#: pipeline's own grid ceiling); past it both go up a level. `change_planes` holds about
#: 30 bytes a cell: both surfaces over the cells they share (12), its outputs (10), what
#: each side measured and the vouched cells, plus a band of the classification.
MAX_CHANGE_CELLS = 1 << 22
ChangeClass = Literal["unchanged", "raised", "lowered", "insufficient"]
#: Codes of the class plane, and no class (neither epoch has a surface there).
UNCHANGED, RAISED, LOWERED, INSUFFICIENT, NONE = 0, 1, 2, 3, -1
CLASSES: tuple[ChangeClass, ...] = ("unchanged", "raised", "lowered", "insufficient")
#: Reserved: a change seen only in appearance. Never emitted (module docstring).
APPEARANCE_ONLY = "appearance-only"
NO = cell_grid.NO_HEIGHT
SATURATED = cell_grid.SIGMA_SATURATED
CM = cell_grid.HEIGHT_UNIT_M


@dataclass(frozen=True)
class Surface:
    """One level of a height surface, cells `i0 .. i0+ni-1` by `j0 .. j0+nj-1`, each array
    `(nj, ni)`: heights in centimetres above the frame origin, `NO` where there is none."""

    level: int
    i0: int
    j0: int
    ni: int
    nj: int
    #: The container's own types, never widened here: `dsm` and `ground` i2, `sigma` and
    #: `support` u1. Differences are taken wide (`change_planes`), never in these.
    dsm: NDArray[np.int16]
    ground: NDArray[np.int16]
    sigma: NDArray[np.uint8]
    support: NDArray[np.uint8]

    @property
    def cells(self) -> int:
        return self.ni * self.nj

    @property
    def has(self) -> NDArray[np.bool_]:
        """Cells with a surface: a height for both and a spread (P5 writes sigma 0 and
        `NO` together; a cell that says otherwise is not believed)."""
        found: NDArray[np.bool_] = (self.sigma > 0) & (self.dsm != NO) & (self.ground != NO)
        return found

    @staticmethod
    def of(plane: cell_grid.Plane) -> Surface:
        """A decoded surface grid's plane (`cell_grid.decode` has checked its channels and
        their types; the arrays are its own, not copied)."""
        try:
            dsm, ground = plane.channels["dsm"], plane.channels["ground"]
            sigma, support = plane.channels["sigma"], plane.channels["support"]
        except KeyError as error:
            raise cell_grid.CellGridError(f"not a surface plane: no {error}") from error
        for name, values, kind in (
            ("dsm", dsm, np.int16),
            ("ground", ground, np.int16),
            ("sigma", sigma, np.uint8),
            ("support", support, np.uint8),
        ):
            if values.dtype != kind:
                raise cell_grid.CellGridError(f"surface channel {name} is {values.dtype}")
        return Surface(
            plane.level, plane.i0, plane.j0, plane.ni, plane.nj, dsm, ground, sigma, support
        )


def _blocks(values: NDArray[Any], i0: int, j0: int, fill: Any) -> NDArray[Any]:
    """`values` (a plane at `i0, j0`) padded to whole parents with `fill`, shaped
    `(nj, 2, ni, 2)` over its parents."""
    vj, vi = values.shape
    pi0, pj0 = i0 >> 1, j0 >> 1
    if vi == 0 or vj == 0:
        return np.full((0, 2, 0, 2), fill, dtype=values.dtype)
    ni = ((i0 + vi - 1) >> 1) - pi0 + 1
    nj = ((j0 + vj - 1) >> 1) - pj0 + 1
    padded = np.full((2 * nj, 2 * ni), fill, dtype=values.dtype)
    di, dj = i0 - 2 * pi0, j0 - 2 * pj0
    padded[dj : dj + vj, di : di + vi] = values
    out: NDArray[Any] = padded.reshape(nj, 2, ni, 2)
    return out


def _quads(values: NDArray[Any], s: Surface, fill: Any) -> NDArray[Any]:
    """`values` (over `s`) padded to whole parents with `fill`, shaped `(nj, 2, ni, 2)`."""
    return _blocks(values, s.i0, s.j0, fill)


def coarsen_surface(s: Surface) -> Surface:
    """`s` one level up (module docstring): of the children with a surface, `dsm` the
    highest, `ground` the lowest, `sigma` the largest; `support`'s source bits OR'd over the
    children and its count nibble the largest child's. None where no child has one."""
    has = _quads(s.has, s, False)
    nj, _, ni, _ = has.shape
    top = np.int16(np.iinfo(np.int16).max)
    dsm = np.where(has, _quads(s.dsm, s, NO), np.int16(NO)).max(axis=(1, 3), initial=NO)
    ground = np.where(has, _quads(s.ground, s, NO), top).min(axis=(1, 3), initial=top)
    sigma = np.where(has, _quads(s.sigma, s, 0), np.uint8(0)).max(axis=(1, 3), initial=0)
    any_has = has.any(axis=(1, 3))
    support = _quads(s.support, s, 0)
    bits = np.bitwise_or.reduce(np.bitwise_or.reduce(support & 0x0F, axis=3), axis=1)
    count = (support >> 4).max(axis=(1, 3), initial=0)
    return Surface(
        s.level + 1,
        s.i0 >> 1,
        s.j0 >> 1,
        ni,
        nj,
        np.where(any_has, dsm, np.int16(NO)).astype(np.int16),
        np.where(any_has, ground, np.int16(NO)).astype(np.int16),
        np.where(any_has, sigma, np.uint8(0)).astype(np.uint8),
        (bits | (count << 4)).astype(np.uint8),
    )


@dataclass(frozen=True)
class _Cells:
    """One array over the cells `i0 ..`, `j0 ..` of one level."""

    level: int
    i0: int
    j0: int
    values: NDArray[Any]

    @property
    def window(self) -> tuple[int, int, int, int]:
        nj, ni = self.values.shape
        return self.i0, self.j0, ni, nj

    def up(self, how: Literal["any", "sum"]) -> _Cells:
        """One level up: whether any child is set, or how many children's values sum to."""
        q = _blocks(self.values, self.i0, self.j0, 0)
        out = q.any(axis=(1, 3)) if how == "any" else q.sum(axis=(1, 3), dtype=np.int32)
        return _Cells(self.level + 1, self.i0 >> 1, self.j0 >> 1, out)


def _masked(s: Surface, keep: NDArray[np.bool_]) -> Surface:
    """`s` with a surface only where `keep` (and it had one)."""
    keep = keep & s.has
    return Surface(
        s.level,
        s.i0,
        s.j0,
        s.ni,
        s.nj,
        np.where(keep, s.dsm, np.int16(NO)).astype(np.int16),
        np.where(keep, s.ground, np.int16(NO)).astype(np.int16),
        np.where(keep, s.sigma, np.uint8(0)).astype(np.uint8),
        np.where(keep, s.support, np.uint8(0)).astype(np.uint8),
    )


def _under(fine: Surface, coarse: Surface) -> tuple[Surface, _Cells]:
    """`fine` brought up to `coarse`'s level over only the children whose coarse cell
    `coarse` measured, and per coarse cell how many of its finest children that is."""
    k = coarse.level - fine.level
    keep = np.zeros((fine.nj, fine.ni), dtype=bool)
    if coarse.cells and fine.cells:
        rows = (np.arange(fine.j0, fine.j0 + fine.nj) >> k) - coarse.j0
        columns = (np.arange(fine.i0, fine.i0 + fine.ni) >> k) - coarse.i0
        inside = ((rows >= 0) & (rows < coarse.nj))[:, None] & (
            (columns >= 0) & (columns < coarse.ni)
        )[None, :]
        lookup = coarse.has[
            np.ix_(np.clip(rows, 0, coarse.nj - 1), np.clip(columns, 0, coarse.ni - 1))
        ]
        keep = lookup & inside
    masked = _masked(fine, keep)
    count = _Cells(fine.level, fine.i0, fine.j0, masked.has.astype(np.int32))
    for _ in range(k):
        masked = coarsen_surface(masked)
        count = count.up("sum")
    return masked, count


@dataclass(frozen=True)
class _Joint:
    """Both epochs over one window of one level, each surface taken over the children both
    measured only (`shared`); `partial` where that is fewer than either measured."""

    level: int
    i0: int
    j0: int
    a: tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.uint8]]
    b: tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.uint8]]
    shared: NDArray[np.bool_]
    partial: NDArray[np.bool_]

    @property
    def window(self) -> tuple[int, int, int, int]:
        nj, ni = self.shared.shape
        return self.i0, self.j0, ni, nj


def _intersection(*windows: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    i0 = max(w[0] for w in windows)
    j0 = max(w[1] for w in windows)
    i1 = min(w[0] + w[2] for w in windows)
    j1 = min(w[1] + w[3] for w in windows)
    return i0, j0, max(0, i1 - i0), max(0, j1 - j0)


def _window(s: Surface) -> tuple[int, int, int, int]:
    return s.i0, s.j0, s.ni, s.nj


def _joint(pa: Surface, pb: Surface, partial: _Cells | None) -> _Joint:
    """`pa` and `pb` (one level) over the intersection of their windows."""
    w = _intersection(_window(pa), _window(pb))

    def side(s: Surface) -> tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.uint8]]:
        return (
            _place(s.dsm, s.i0, s.j0, w, np.int16(NO)),
            _place(s.ground, s.i0, s.j0, w, np.int16(NO)),
            _place(s.sigma, s.i0, s.j0, w, np.uint8(0)),
        )

    shared = _place(pa.has, pa.i0, pa.j0, w, False) & _place(pb.has, pb.i0, pb.j0, w, False)
    part = (
        _place(partial.values, partial.i0, partial.j0, w, False) & shared
        if partial is not None
        else np.zeros(shared.shape, dtype=bool)
    )
    return _Joint(pa.level, w[0], w[1], side(pa), side(pb), shared, part)


def _joint_up(joint: _Joint, measured_a: _Cells, measured_b: _Cells) -> _Joint:
    """`joint` one level up, each side's dsm (max), ground (min) and sigma (max) over the
    children both measured; partial where a parent's shared children are fewer than the
    children either side measured (`measured_*`: per child, whether that side measured
    anything under it), or where a shared child was already partial."""
    i0, j0 = joint.i0, joint.j0
    sq = _blocks(joint.shared, i0, j0, False)
    shared = sq.any(axis=(1, 3))
    top = np.int16(np.iinfo(np.int16).max)

    def side(
        values: tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.uint8]],
    ) -> tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.uint8]]:
        dsm, ground, sigma = values
        high = np.where(sq, _blocks(dsm, i0, j0, np.int16(NO)), np.int16(NO)).max(
            axis=(1, 3), initial=NO
        )
        low = np.where(sq, _blocks(ground, i0, j0, np.int16(NO)), top).min(axis=(1, 3), initial=top)
        spread = np.where(sq, _blocks(sigma, i0, j0, np.uint8(0)), np.uint8(0)).max(
            axis=(1, 3), initial=0
        )
        return (
            np.where(shared, high, np.int16(NO)).astype(np.int16),
            np.where(shared, low, np.int16(NO)).astype(np.int16),
            np.where(shared, spread, np.uint8(0)).astype(np.uint8),
        )

    window = (i0 >> 1, j0 >> 1, shared.shape[1], shared.shape[0])
    both = sq.sum(axis=(1, 3), dtype=np.int32)
    fewer = np.zeros(shared.shape, dtype=bool)
    for measured in (measured_a, measured_b):
        counts = _Cells(
            measured.level, measured.i0, measured.j0, measured.values.astype(np.int32)
        ).up("sum")
        fewer |= both < _place(counts.values, counts.i0, counts.j0, window, np.int32(0))
    inherited = _blocks(joint.partial & joint.shared, i0, j0, False).any(axis=(1, 3))
    return _Joint(
        joint.level + 1,
        window[0],
        window[1],
        side(joint.a),
        side(joint.b),
        shared,
        shared & (inherited | fewer),
    )


def _place(
    values: NDArray[Any], i0: int, j0: int, window: tuple[int, int, int, int], fill: Any
) -> NDArray[Any]:
    """`values` (a plane at `i0, j0`) in `window` (`i0, j0, ni, nj` of the same level)."""
    wi0, wj0, ni, nj = window
    out = np.full((nj, ni), fill, dtype=values.dtype)
    vj, vi = values.shape
    a0, a1 = max(wi0, i0), min(wi0 + ni, i0 + vi)
    b0, b1 = max(wj0, j0), min(wj0 + nj, j0 + vj)
    if a0 < a1 and b0 < b1:
        out[b0 - wj0 : b1 - wj0, a0 - wi0 : a1 - wi0] = values[b0 - j0 : b1 - j0, a0 - i0 : a1 - i0]
    return out


def _union_cells(a: Surface, b: Surface, level: int) -> int:
    """How many cells the two windows' union holds at `level` (>= both), without
    coarsening anything: the window's corners halve with each level."""

    def span(lo: int, n: int, k: int) -> tuple[int, int]:
        return lo >> k, (lo + n - 1) >> k

    ia = span(a.i0, a.ni, level - a.level)
    ib = span(b.i0, b.ni, level - b.level)
    ja = span(a.j0, a.nj, level - a.level)
    jb = span(b.j0, b.nj, level - b.level)
    ni = max(ia[1], ib[1]) - min(ia[0], ib[0]) + 1
    nj = max(ja[1], jb[1]) - min(ja[0], jb[0]) + 1
    return ni * nj


def vouched(grid: cell_grid.Grid, level: int) -> cell_grid.Plane:
    """`grid` (a confidence grid) at `level` through its own min-of-children pyramid, as a
    plane of one channel, `strong`: scored at least `WEAK_BELOW` and not unseen."""
    plane = grid.at(level)
    score, reasons = plane.channels["score"], plane.channels["reasons"]
    strong = (score >= cell_grid.WEAK_BELOW) & ((reasons & cell_grid.UNSEEN) == 0)
    return cell_grid.Plane(level, plane.i0, plane.j0, plane.ni, plane.nj, {"strong": strong})


#: `change_planes` classifies a band of rows at a time, of about this many cells.
BAND_CELLS = 1 << 16


@dataclass(frozen=True)
class ChangePlanes:
    """The change between two surfaces at one level, over the union of their windows:
    `classes` (`NONE` where neither has a surface), the dsm and ground differences in
    centimetres (0 where not both), and `low` (low confidence, compared cells only)."""

    level: int
    #: The coarser of the two surfaces' own levels; `level` is above it only when the
    #: windows were too large (`MAX_CHANGE_CELLS`) or the outline was (`change_areas`).
    native_level: int
    i0: int
    j0: int
    ni: int
    nj: int
    classes: NDArray[np.int8]
    delta_cm: NDArray[np.int32]
    ground_delta_cm: NDArray[np.int32]
    low: NDArray[np.bool_]
    #: The least threshold over the compared cells, metres; None when none was compared.
    min_detectable_m: float | None
    uncertainty_from_m: float
    uncertainty_to_m: float

    @property
    def cell_m(self) -> float:
        return math.ldexp(cell_grid.LATTICE_BASE_M, self.level)

    def counts(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(self.classes == code)) for code, name in enumerate(CLASSES)
        }

    def low_counts(self) -> dict[str, int]:
        """Low-confidence cells of each compared class (insufficient has none to state)."""
        return {
            name: int(np.count_nonzero((self.classes == code) & self.low))
            for code, name in enumerate(CLASSES)
            if code != INSUFFICIENT
        }


def change_planes(
    a: Surface,
    b: Surface,
    ua: float,
    ub: float,
    *,
    confidence_a: cell_grid.Grid | None = None,
    confidence_b: cell_grid.Grid | None = None,
    level: int | None = None,
    sigmas: float = THRESHOLD_SIGMA,
) -> ChangePlanes:
    """The change from `a` to `b` (module docstring), with `ua` and `ub` each epoch's
    uncertainty from the site frame in metres, at the coarser of their levels (or `level`
    when that is coarser still, or coarser yet when the windows would hold over
    `MAX_CHANGE_CELLS`). A cell is low confidence when a confidence grid given does not
    vouch for it, or a grid is not given (nothing vouches), or either sigma saturates."""
    for name, value in (("ua", ua), ("ub", ub)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite uncertainty >= 0, not {value!r}")
    native = max(a.level, b.level)
    at = max(native, level if level is not None else native)
    while _union_cells(a, b, at) > MAX_CHANGE_CELLS and at < cell_grid.MAX_LEVEL:
        at += 1
    # The finer epoch is brought to the coarser one's level over the children whose coarse
    # cell the coarser one measured; then both go up together over the children both
    # measured (module docstring), knowing per cell what each side measured at all.
    swap = a.level < b.level
    fine, coarse = (a, b) if swap else (b, a)
    if fine.level < coarse.level:
        lifted, count = _under(fine, coarse)
        whole = 1 << (2 * (coarse.level - fine.level))
        partial = _Cells(
            count.level, count.i0, count.j0, (count.values > 0) & (count.values < whole)
        )
        seen = _Cells(fine.level, fine.i0, fine.j0, fine.has)
        while seen.level < coarse.level:
            seen = seen.up("any")
        ca, cb = (lifted, coarse) if swap else (coarse, lifted)
        seen_a, seen_b = (
            (seen, _Cells(coarse.level, coarse.i0, coarse.j0, coarse.has))
            if swap
            else (
                _Cells(coarse.level, coarse.i0, coarse.j0, coarse.has),
                seen,
            )
        )
        joint = _joint(ca, cb, partial)
    else:
        joint = _joint(a, b, None)
        seen_a = _Cells(a.level, a.i0, a.j0, a.has)
        seen_b = _Cells(b.level, b.i0, b.j0, b.has)
    while joint.level < at:
        joint = _joint_up(joint, seen_a, seen_b)
        seen_a, seen_b = seen_a.up("any"), seen_b.up("any")
    i0, j0 = min(seen_a.i0, seen_b.i0), min(seen_a.j0, seen_b.j0)
    ni = max(seen_a.i0 + seen_a.window[2], seen_b.i0 + seen_b.window[2]) - i0
    nj = max(seen_a.j0 + seen_a.window[3], seen_b.j0 + seen_b.window[3]) - j0
    union = (i0, j0, ni, nj)
    # Into the union window: insufficient where one side measured something and the two
    # share no child; no class where neither measured anything.
    anywhere = _place(seen_a.values, seen_a.i0, seen_a.j0, union, False) | _place(
        seen_b.values, seen_b.i0, seen_b.j0, union, False
    )
    classes = np.where(anywhere, np.int8(INSUFFICIENT), np.int8(NONE)).astype(np.int8)
    del anywhere
    delta_cm = np.zeros((nj, ni), dtype=np.int32)
    ground_delta_cm = np.zeros((nj, ni), dtype=np.int32)
    low = np.zeros((nj, ni), dtype=bool)
    # Classified on the joint window (where alone both can have a surface), a band of rows
    # at a time, so nothing but the outputs is held for the whole window.
    jw = joint.window
    di, dj = jw[0] - i0, jw[1] - j0
    (da, ga, sa), (db, gb, sb) = joint.a, joint.b
    grids = [vouched(g, at) if g is not None else None for g in (confidence_a, confidence_b)]
    least = math.inf
    step = max(1, BAND_CELLS // max(jw[2], 1))
    for r0 in range(0, jw[3], step):
        r1 = min(jw[3], r0 + step)
        band = (jw[0], jw[1] + r0, jw[2], r1 - r0)
        both = joint.shared[r0:r1]
        delta = db[r0:r1].astype(np.int32) - da[r0:r1].astype(np.int32)
        ground = gb[r0:r1].astype(np.int32) - ga[r0:r1].astype(np.int32)
        s_a, s_b = sa[r0:r1], sb[r0:r1]
        spread = (s_a.astype(np.float64) * CM) ** 2 + (s_b.astype(np.float64) * CM) ** 2
        threshold = sigmas * np.sqrt(spread + ua * ua + ub * ub)
        change_m = delta * CM
        out = classes[dj + r0 : dj + r1, di : di + jw[2]]
        out[both] = UNCHANGED
        out[both & (change_m > threshold)] = RAISED
        out[both & (change_m < -threshold)] = LOWERED
        delta_cm[dj + r0 : dj + r1, di : di + jw[2]] = np.where(both, delta, 0)
        ground_delta_cm[dj + r0 : dj + r1, di : di + jw[2]] = np.where(both, ground, 0)
        if both.any():
            least = min(least, float(threshold[both].min()))
        strong = np.ones(both.shape, dtype=bool)
        for grid in grids:
            if grid is None:
                strong[...] = False
            else:
                strong &= _place(grid.channels["strong"], grid.i0, grid.j0, band, False)
        low[dj + r0 : dj + r1, di : di + jw[2]] = both & (
            joint.partial[r0:r1] | ~strong | (s_a >= SATURATED) | (s_b >= SATURATED)
        )
    return ChangePlanes(
        level=at,
        native_level=native,
        i0=i0,
        j0=j0,
        ni=ni,
        nj=nj,
        classes=classes,
        delta_cm=delta_cm,
        ground_delta_cm=ground_delta_cm,
        low=low,
        min_detectable_m=least if math.isfinite(least) else None,
        uncertainty_from_m=ua,
        uncertainty_to_m=ub,
    )


# --- areas --------------------------------------------------------------------------------

#: What an area is drawn by: its class and, for raised and lowered, its confidence.
#: Unchanged cells are counted, never outlined: they are most of a map and its outline's
#: cost, and an area of them would read as confident wherever the comparison could not see
#: (an epoch placed by GPS alone is unchanged everywhere).
AreaClass = Literal["raised", "lowered", "insufficient"]
_GROUPS: tuple[tuple[AreaClass, bool | None], ...] = (
    ("raised", False),
    ("raised", True),
    ("lowered", False),
    ("lowered", True),
    ("insufficient", None),
)


def group_codes(planes: ChangePlanes) -> NDArray[np.uint8]:
    """Per cell, 1 + the index of its `_GROUPS` entry; 0 where it has no class."""
    codes = np.zeros(planes.classes.shape, dtype=np.uint8)
    for index, (name, low) in enumerate(_GROUPS):
        mask = planes.classes == CLASSES.index(name)
        if low is not None:
            mask &= planes.low == low
        codes[mask] = index + 1
    return codes


def outline_edges(codes: NDArray[np.uint8]) -> int:
    """Cell edges between two cells of different groups, or between a classed cell and the
    window's outside: what outlining every area costs, holes included."""
    padded = np.pad(codes, 1)
    across = padded[:, 1:] != padded[:, :-1]
    up = padded[1:, :] != padded[:-1, :]
    return int(np.count_nonzero(across) + np.count_nonzero(up))


@dataclass(frozen=True)
class AreaShape:
    """One change area: its outline in the frame and on the globe, and what is in it."""

    change: AreaClass
    metric: Polygon
    footprint: MultiPolygon
    cells: int
    low_confidence_cells: int
    mean_delta_m: float | None
    max_abs_delta_m: float | None
    mean_ground_delta_m: float | None

    @property
    def low_confidence(self) -> bool:
        """Every cell of it is low confidence (raised and lowered areas are all or none)."""
        return self.cells > 0 and self.low_confidence_cells == self.cells


def needs_coarser(planes: ChangePlanes) -> bool:
    """Whether `planes` is too large to outline (module docstring)."""
    if planes.ni * planes.nj > MAX_ZONE_CELLS:
        return True
    return outline_edges(group_codes(planes)) > MAX_ZONE_EDGES


def _globe(polygons: list[Any], projection: SiteProjection) -> list[MultiPolygon]:
    """Every outline's vertices through the site's tangent plane, in one call."""

    def apply(points: NDArray[np.float64]) -> NDArray[np.float64]:
        lon, lat = projection.to_geographic(points[:, 0], points[:, 1])
        return np.column_stack([lon, lat])

    moved = shapely.transform(np.array(polygons, dtype=object), apply)
    out: list[MultiPolygon] = []
    for geometry in moved:
        parts = [orient(part) for part in polygon_parts(make_valid(geometry))]
        if not parts:
            raise ValueError("a change area did not survive projection to the globe")
        out.append(MultiPolygon(parts))
    return out


def change_areas(planes: ChangePlanes, projection: SiteProjection) -> list[AreaShape]:
    """Every 4-connected region of one group (`_GROUPS`) of `planes` as an area, raised and
    lowered first, then insufficient, largest first within each (unchanged cells are
    counted in `planes`, not outlined). Each area's numbers are of its own cells (its
    region's, at `planes.level`)."""
    codes = group_codes(planes)
    if not codes.any():
        return []
    affine = cell_affine(planes.cell_m, planes.i0, planes.j0)
    regions: list[tuple[int, Polygon]] = []
    for geometry, value in shapes(codes, mask=codes > 0, transform=affine, connectivity=4):
        region = shape(geometry)
        if isinstance(region, Polygon):
            regions.append((int(value), region))
    # Each region's cells: its polygon is exactly its cells' squares, so burning the
    # polygons back with their index recovers the labels (cell centres, rasterize's rule).
    labels = rasterize(
        ((polygon, index + 1) for index, (_, polygon) in enumerate(regions)),
        out_shape=codes.shape,
        transform=affine,
        dtype=np.int32,
        fill=0,
    )
    count = len(regions) + 1
    flat = labels.ravel()
    compared = np.isin(planes.classes.ravel(), (UNCHANGED, RAISED, LOWERED))
    delta = planes.delta_cm.ravel().astype(np.float64)
    ground = planes.ground_delta_cm.ravel().astype(np.float64)
    cells = np.bincount(flat, minlength=count)
    lows = np.bincount(flat, weights=planes.low.ravel(), minlength=count)
    measured = np.bincount(flat, weights=compared, minlength=count)
    delta_sum = np.bincount(flat, weights=np.where(compared, delta, 0.0), minlength=count)
    ground_sum = np.bincount(flat, weights=np.where(compared, ground, 0.0), minlength=count)
    peak = np.zeros(count, dtype=np.float64)
    np.maximum.at(peak, flat, np.where(compared, np.abs(delta), 0.0))
    # Exact cell edges; `simplify(0)` only drops the vertices along a straight run of them.
    globes = _globe([make_valid(polygon.simplify(0)) for _, polygon in regions], projection)
    out: list[AreaShape] = []
    for index, (value, polygon) in enumerate(regions):
        label = index + 1
        name, _low = _GROUPS[value - 1]
        n = int(cells[label])
        has_delta = name != "insufficient" and measured[label] > 0
        out.append(
            AreaShape(
                change=name,
                metric=polygon,
                footprint=globes[index],
                cells=n,
                low_confidence_cells=int(lows[label]),
                mean_delta_m=float(delta_sum[label] / measured[label] * CM) if has_delta else None,
                max_abs_delta_m=float(peak[label] * CM) if has_delta else None,
                mean_ground_delta_m=float(ground_sum[label] / measured[label] * CM)
                if has_delta
                else None,
            )
        )
    order: Mapping[str, int] = {"raised": 0, "lowered": 1, "insufficient": 2}
    return sorted(out, key=lambda area: (order[area.change], -area.cells))


@dataclass(frozen=True)
class ChangeMap:
    """The change at its compared level (`planes`, whose counts the map states) and the
    areas drawn at `drawn.level` (the same planes unless they were too large to outline)."""

    planes: ChangePlanes
    drawn: ChangePlanes
    areas: list[AreaShape]


def change_map(
    a: Surface,
    b: Surface,
    ua: float,
    ub: float,
    projection: SiteProjection,
    *,
    confidence_a: cell_grid.Grid | None = None,
    confidence_b: cell_grid.Grid | None = None,
) -> ChangeMap:
    """`change_planes` and its areas, a level up at a time while they are too large to
    outline (module docstring)."""
    planes = change_planes(a, b, ua, ub, confidence_a=confidence_a, confidence_b=confidence_b)
    drawn = planes
    while needs_coarser(drawn) and drawn.level < cell_grid.MAX_LEVEL:
        drawn = change_planes(
            a,
            b,
            ua,
            ub,
            confidence_a=confidence_a,
            confidence_b=confidence_b,
            level=drawn.level + 1,
        )
    return ChangeMap(planes=planes, drawn=drawn, areas=change_areas(drawn, projection))
