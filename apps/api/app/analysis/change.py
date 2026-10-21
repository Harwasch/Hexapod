"""The change between two epochs, from their height surfaces (M5, ADR 0018 § API).

**Geometry first.** Each epoch's run measured a height surface per cell (P5, `surface.bin`:
its top surface `dsm` and that one's `sigma`, its bare `ground` and that one's
`groundSigma`, in centimetres above the site frame's origin, on its confidence grid's
plane; each sigma carries its estimator's bias). Two epochs of one site share that frame,
and their cells nest (`0.25 m * 2**level`), so the change is a comparison of cells --
never of splat pixels, which would read a season or a sky as geometry. Appearance is not
compared at all (`appearance-only` is a reserved class that is never emitted): that needs
the new photos compared with the old epoch at fixed geometry.

**Two layers** (review B5a's ruling). **Ground** -- the bare earth, `ground` and
`groundSigma` -- is the primary product. **Vegetation** -- the top surface, `dsm` and
`sigma`: canopy, crops, and anything else standing on the ground -- is a separate layer
with its own, higher threshold, so that wind in a crown or a crop grown is never read as
the ground moving. Each layer is classified on its own: `raised`, `lowered`, `unchanged`,
`insufficient`. A change of the ground wider than P5's ground window shows in both; one
narrower (a crate, a spoil heap), and anything that grows, only in vegetation.

**Resampling.** Both surfaces are brought to the coarser of their two levels. The finer
epoch is first brought to the coarser one's level over the children whose coarse cell the
coarser one measured, then both go up together over the children both measured: of
those, the top surface the highest, the ground the lowest, each sigma the largest (P5's
demo `coarsen_surface` rule, tools/pipeline `experiments/confidence_demo.py`, which
`coarsen_surface` here matches byte for byte on the committed fixture
`data/tiles/synthetic-yard-surfaces`). A coarse cell where each side measured something
but no child is shared is `insufficient`, and so is one whose shared children are fewer
than either side measured (the coarser epoch counts as having measured all of its cell):
its value would be those children's alone against the whole, a difference in what was
measured (review B5a; it was compared and only flagged). If the two windows together would
still be over `MAX_CHANGE_CELLS` cells, both go up further until they are not (and say so).

**The rule** (`change-v1`). Where both epochs have the layer's height, `delta = h_to -
h_from` (computed in 32 bits: two i2 heights can differ by more than i2 holds) against

    t = 3 sqrt(s_from^2 + s_to^2 + u_from^2 + u_to^2 + 2 w^2) + drift

with `s` each side's sigma for the layer, `u` each epoch's placement uncertainty (the
larger of its uncertainty from the site frame and its alignment's own claim, so a coarse
epoch brings its metres: `services/change_maps.py` decides it), `w` the layer's own motion
between visits (0 for the ground; `VEGETATION_MOTION_M` for vegetation, below) and `drift`
the ground envelope's on a slope (0 for vegetation; below). `insufficient` where either
epoch measured the layer but the two cannot be compared; no class where neither did. The 3
is a first-principles choice (three standard deviations of the sum of independent
errors), not a measurement: the map says `uncalibrated` until a site scanned twice
measures it. `minDetectableM` is the least `t` over a layer's compared cells: a change
smaller than it cannot be seen anywhere between these two dates -- an epoch placed at
+/-10 m makes it about 30 m, and so finds no change rather than false ones.

**Horizontal uncertainty.** Placement error moves a whole epoch sideways, not only up: a
roof edge one cell off reads as metres of change cell for cell. So each cell is compared
with the other epoch's neighbourhood within the horizontal radius `r` (the least whole
number of cells covering `3 sqrt(u_from^2 + u_to^2 + 2 w^2)`): `raised` when the later
height is above everything the earlier epoch had nearby by more than `t`, or the earlier is
below everything the later has nearby by more than `t` (a hole filled); `lowered` the
mirror. A pure shift within `r` explains every cell, so it is never change; a post put up
or taken away still is. `delta` is still the cell's own (review B5a).

**Vegetation's threshold** (derived, uncalibrated). Between two visits a canopy's top
moves without anything having grown: a crown sways in the wind by decimetres at its top,
and the leaves the 2nd-highest sample lands on flutter by their own size. `w = 0.25 m` per
epoch -- the lattice's finest cell -- is taken as one standard deviation of that,
vertically and horizontally, in each epoch independently: it adds `2 w^2` under the root,
so with perfectly placed, noiseless epochs the least vegetation change seen is `3 sqrt(2)
w`, about 1.06 m, and the horizontal radius about as much (5 cells at 0.25 m). The ground
has no such term: what sways is above it. A season (leaf-on against leaf-off) is real
vegetation change and reads as such.

**The ground envelope's drift** (derived). P5's ground is the least 2nd-lowest sample over
a window of cells, so on a slope of grade `g` it reads up to `g R` low, `R` the window's
half-diagonal, and which cell of the window it reads depends on what else stands in it: a
bush grown at a window's downhill corner moves the ground of every cell within its reach.
That bound, `g R`, is added to the ground's threshold, so vegetation can never move the
ground layer: `g` is the steeper of the two epochs' local grades over the window's own
half-width (one-sided differences of the ground, the larger; `UNKNOWN_GRADE` where no cell
that far has a ground) and `R` the larger of the two epochs' reaches (`ground_reach_m`; 0
where the window is the cell alone). On flat ground it is nothing.

**What is not compared.** A cell where either sigma saturates (255 cm: at least 2.55 m, so
the threshold would be a lower bound only) is `insufficient`, as is one where either epoch
has no height for the layer -- for the ground, also where it was never observed (P5's
`NO_HEIGHT` ground under a surface: canopy or roof wider than the window).

**Low confidence is flagged, not hidden.** A compared cell is `lowConfidence` when either
epoch's confidence grid does not vouch for it at the compared level (weak-v1: scored under
128, or unseen, or no evidence at all; the grid's own min-of-children pyramid, so a coarse
cell is as weak as its weakest child). It is still classified: P5's demo found the weak
cells of a real change were its crate's own sides, so hiding them would hide the change.

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
#: A canopy top's own motion between visits, per epoch, metres: one standard deviation,
#: vertically and horizontally (module docstring, "Vegetation's threshold").
VEGETATION_MOTION_M = 0.25
#: The grade the ground's drift is bounded by where no neighbour gives one: the steepest
#: slope P5 takes as ground at all (tools/pipeline surface.py `GROUND_MAX_GRADE`).
UNKNOWN_GRADE = 0.5
#: The two windows together are compared on at most this many cells (2048 x 2048, the
#: pipeline's own grid ceiling); past it both go up a level. `change_planes` holds about
#: 30 bytes a cell: both sides over the cells they share, its outputs, what each side
#: measured and the vouched cells, plus a band of the classification; and, with a
#: horizontal radius, the two sides' neighbourhoods (8 more) and, for the ground, its
#: local rise (8 more at the peak, 4 kept).
MAX_CHANGE_CELLS = 1 << 22
Layer = Literal["ground", "vegetation"]
LAYERS: tuple[Layer, ...] = ("ground", "vegetation")
ChangeClass = Literal["unchanged", "raised", "lowered", "insufficient"]
#: Codes of the class plane, and no class (neither epoch has the layer's height there).
UNCHANGED, RAISED, LOWERED, INSUFFICIENT, NONE = 0, 1, 2, 3, -1
CLASSES: tuple[ChangeClass, ...] = ("unchanged", "raised", "lowered", "insufficient")
#: Reserved: a change seen only in appearance. Never emitted (module docstring).
APPEARANCE_ONLY = "appearance-only"
NO = cell_grid.NO_HEIGHT
SATURATED = cell_grid.SIGMA_SATURATED
CM = cell_grid.HEIGHT_UNIT_M
_I16_MAX = np.int16(np.iinfo(np.int16).max)


@dataclass(frozen=True)
class Surface:
    """One level of a height surface, cells `i0 .. i0+ni-1` by `j0 .. j0+nj-1`, each array
    `(nj, ni)`: heights in centimetres above the frame origin, `NO` where there is none."""

    level: int
    i0: int
    j0: int
    ni: int
    nj: int
    #: The container's own types, never widened here: `dsm` and `ground` i2, the sigmas and
    #: `support` u1. Differences are taken wide (`change_planes`), never in these.
    dsm: NDArray[np.int16]
    ground: NDArray[np.int16]
    sigma: NDArray[np.uint8]
    support: NDArray[np.uint8]
    #: The ground's own sigma (P5): 0 where the ground is `NO`, observed or not.
    ground_sigma: NDArray[np.uint8]
    #: How far from a cell its ground may have come, metres (`ground_reach_m`; 0: the
    #: cell alone).
    ground_reach_m: float = 0.0

    @property
    def cells(self) -> int:
        return self.ni * self.nj

    @property
    def has(self) -> NDArray[np.bool_]:
        """Cells with a top surface: a height and its spread (P5 writes sigma 0 and `NO`
        together; a cell that says otherwise is not believed)."""
        found: NDArray[np.bool_] = (self.sigma > 0) & (self.dsm != NO)
        return found

    @property
    def has_ground(self) -> NDArray[np.bool_]:
        """Cells with a surface whose ground was observed (P5 writes `NO` ground and
        groundSigma 0 under a surface where the window saw only canopy or roof)."""
        found: NDArray[np.bool_] = self.has & (self.ground != NO) & (self.ground_sigma > 0)
        return found

    def heights(self, layer: Layer) -> Heights:
        """The layer's heights and sigma (module docstring): the ground's, or the top's."""
        how: Literal["max", "min"]
        if layer == "vegetation":
            keep, height, sigma, how = self.has, self.dsm, self.sigma, "max"
        else:
            keep, height, sigma, how = self.has_ground, self.ground, self.ground_sigma, "min"
        return Heights(
            self.level,
            self.i0,
            self.j0,
            np.where(keep, height, np.int16(NO)).astype(np.int16),
            np.where(keep, sigma, np.uint8(0)).astype(np.uint8),
            how,
        )

    @staticmethod
    def of(plane: cell_grid.Plane, *, ground_reach_m: float = 0.0) -> Surface:
        """A decoded surface grid's plane (`cell_grid.decode` has checked its channels and
        their types; the arrays are its own, not copied)."""
        try:
            dsm, ground = plane.channels["dsm"], plane.channels["ground"]
            sigma, support = plane.channels["sigma"], plane.channels["support"]
            ground_sigma = plane.channels["groundSigma"]
        except KeyError as error:
            raise cell_grid.CellGridError(f"not a surface plane: no {error}") from error
        for name, values, kind in (
            ("dsm", dsm, np.int16),
            ("ground", ground, np.int16),
            ("sigma", sigma, np.uint8),
            ("support", support, np.uint8),
            ("groundSigma", ground_sigma, np.uint8),
        ):
            if values.dtype != kind:
                raise cell_grid.CellGridError(f"surface channel {name} is {values.dtype}")
        return Surface(
            plane.level,
            plane.i0,
            plane.j0,
            plane.ni,
            plane.nj,
            dsm,
            ground,
            sigma,
            support,
            ground_sigma,
            ground_reach_m,
        )


def ground_reach_m(descriptor: Mapping[str, Any], level: int) -> float:
    """How far from a cell its ground may have come, metres: the half-diagonal of the window
    P5 took the least over (`estimators.groundWindowCells` cells a side at the surface's
    own level), or, where the descriptor does not say, of P5's own 5 m window at that level
    (the cells whose centres are within 2.5 m along each axis)."""
    cell = math.ldexp(cell_grid.LATTICE_BASE_M, level)
    estimators = descriptor.get("estimators")
    cells = estimators.get("groundWindowCells") if isinstance(estimators, Mapping) else None
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        half = math.floor(2.5 / cell + 1e-9)
    else:
        half = (cells - 1) // 2
    return half * cell * math.sqrt(2.0)


@dataclass(frozen=True)
class Heights:
    """One layer of a surface at one level: heights (`NO` where none) and their sigma,
    and how its children come together a level up (`max`: a top; `min`: a ground)."""

    level: int
    i0: int
    j0: int
    height: NDArray[np.int16]
    sigma: NDArray[np.uint8]
    how: Literal["max", "min"]

    @property
    def ni(self) -> int:
        return int(self.height.shape[1])

    @property
    def nj(self) -> int:
        return int(self.height.shape[0])

    @property
    def cells(self) -> int:
        return self.ni * self.nj

    @property
    def window(self) -> tuple[int, int, int, int]:
        return self.i0, self.j0, self.ni, self.nj

    @property
    def has(self) -> NDArray[np.bool_]:
        found: NDArray[np.bool_] = (self.sigma > 0) & (self.height != NO)
        return found


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


def _reduce(
    values: NDArray[np.int16], keep: NDArray[np.bool_], i0: int, j0: int, how: str
) -> NDArray[np.int16]:
    """Each parent's highest (`max`) or lowest (`min`) of the children `keep` holds; `NO`
    where it holds none."""
    q, k = _blocks(values, i0, j0, np.int16(NO)), _blocks(keep, i0, j0, False)
    if how == "max":
        out = np.where(k, q, np.int16(NO)).max(axis=(1, 3), initial=NO)
    else:
        out = np.where(k, q, _I16_MAX).min(axis=(1, 3), initial=_I16_MAX)
    found: NDArray[np.int16] = np.where(k.any(axis=(1, 3)), out, np.int16(NO)).astype(np.int16)
    return found


def _widest(
    sigma: NDArray[np.uint8], keep: NDArray[np.bool_], i0: int, j0: int
) -> NDArray[np.uint8]:
    """Each parent's largest sigma of the children `keep` holds; 0 where it holds none."""
    q, k = _blocks(sigma, i0, j0, np.uint8(0)), _blocks(keep, i0, j0, False)
    out: NDArray[np.uint8] = (
        np.where(k, q, np.uint8(0)).max(axis=(1, 3), initial=0).astype(np.uint8)
    )
    return out


def coarsen_heights(h: Heights) -> Heights:
    """`h` one level up over the children that have a height: its own way for the height,
    the largest sigma. None where no child has one."""
    keep = h.has
    return Heights(
        h.level + 1,
        h.i0 >> 1,
        h.j0 >> 1,
        _reduce(h.height, keep, h.i0, h.j0, h.how),
        _widest(h.sigma, keep, h.i0, h.j0),
        h.how,
    )


def coarsen_surface(s: Surface) -> Surface:
    """`s` one level up (module docstring): of the children with a surface, `dsm` the
    highest and `sigma` the largest; of those whose ground was observed, `ground` the
    lowest and `groundSigma` the largest; `support`'s source bits OR'd over the children
    and its count nibble the largest child's. None where no child has one."""
    has, grounded = s.has, s.has_ground
    support = _blocks(s.support, s.i0, s.j0, np.uint8(0))
    bits = np.bitwise_or.reduce(np.bitwise_or.reduce(support & 0x0F, axis=3), axis=1)
    count = (support >> 4).max(axis=(1, 3), initial=0)
    dsm = _reduce(s.dsm, has, s.i0, s.j0, "max")
    nj, ni = dsm.shape
    return Surface(
        s.level + 1,
        s.i0 >> 1,
        s.j0 >> 1,
        ni,
        nj,
        dsm,
        _reduce(s.ground, grounded, s.i0, s.j0, "min"),
        _widest(s.sigma, has, s.i0, s.j0),
        (bits | (count << 4)).astype(np.uint8),
        _widest(s.ground_sigma, grounded, s.i0, s.j0),
        s.ground_reach_m,
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


def _masked(h: Heights, keep: NDArray[np.bool_]) -> Heights:
    """`h` with a height only where `keep` (and it had one)."""
    keep = keep & h.has
    return Heights(
        h.level,
        h.i0,
        h.j0,
        np.where(keep, h.height, np.int16(NO)).astype(np.int16),
        np.where(keep, h.sigma, np.uint8(0)).astype(np.uint8),
        h.how,
    )


def _under(fine: Heights, coarse: Heights) -> tuple[Heights, _Cells]:
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
        masked = coarsen_heights(masked)
        count = count.up("sum")
    return masked, count


@dataclass(frozen=True)
class _Joint:
    """Both epochs over one window of one level, each taken over the children both
    measured only (`shared`); `partial` where that is fewer than either measured."""

    level: int
    i0: int
    j0: int
    a: tuple[NDArray[np.int16], NDArray[np.uint8]]
    b: tuple[NDArray[np.int16], NDArray[np.uint8]]
    how: Literal["max", "min"]
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


def _joint(pa: Heights, pb: Heights, partial: _Cells | None) -> _Joint:
    """`pa` and `pb` (one level) over the intersection of their windows."""
    w = _intersection(pa.window, pb.window)

    def side(h: Heights) -> tuple[NDArray[np.int16], NDArray[np.uint8]]:
        return (
            _place(h.height, h.i0, h.j0, w, np.int16(NO)),
            _place(h.sigma, h.i0, h.j0, w, np.uint8(0)),
        )

    shared = _place(pa.has, pa.i0, pa.j0, w, False) & _place(pb.has, pb.i0, pb.j0, w, False)
    part = (
        _place(partial.values, partial.i0, partial.j0, w, False) & shared
        if partial is not None
        else np.zeros(shared.shape, dtype=bool)
    )
    return _Joint(pa.level, w[0], w[1], side(pa), side(pb), pa.how, shared, part)


def _joint_up(joint: _Joint, measured_a: _Cells, measured_b: _Cells) -> _Joint:
    """`joint` one level up, each side over the children both measured; partial where a
    parent's shared children are fewer than the children either side measured
    (`measured_*`: per child, whether that side measured anything under it), or where a
    shared child was already partial."""
    i0, j0, children = joint.i0, joint.j0, joint.shared

    def side(
        values: tuple[NDArray[np.int16], NDArray[np.uint8]],
    ) -> tuple[NDArray[np.int16], NDArray[np.uint8]]:
        height, sigma = values
        return (
            _reduce(height, children, i0, j0, joint.how),
            _widest(sigma, children, i0, j0),
        )

    sq = _blocks(children, i0, j0, False)
    shared = sq.any(axis=(1, 3))
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
        joint.how,
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


# --- horizontal uncertainty and the ground's drift -----------------------------------------


def _slide(values: NDArray[np.int16], half: int, axis: int, op: Any, fill: np.int16) -> Any:
    """`op` (np.maximum or np.minimum) over each cell's `2 half + 1` cells along `axis`,
    cells past the edge `fill`: by doubling (runs of 1, 2, 4 ... cells), so the cost is
    `log2(half)` passes however wide, and exact (a maximum or a minimum is)."""
    if half <= 0:
        return values
    width = 2 * half + 1
    pad = [(0, 0), (0, 0)]
    pad[axis] = (half, half)
    run = np.pad(values, pad, constant_values=fill)
    span = 1
    while 2 * span <= width:
        head, tail = [slice(None)] * 2, [slice(None)] * 2
        head[axis], tail[axis] = slice(0, -span), slice(span, None)
        run = op(run[tuple(head)], run[tuple(tail)])
        span *= 2
    n = values.shape[axis]
    first, second = [slice(None)] * 2, [slice(None)] * 2
    first[axis], second[axis] = slice(0, n), slice(width - span, width - span + n)
    return op(run[tuple(first)], run[tuple(second)])


def neighbourhood(
    height: NDArray[np.int16], keep: NDArray[np.bool_], half: int
) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    """Each cell's highest and lowest height among the `keep` cells of its `(2 half + 1)^2`
    square (`NO` and the i2 ceiling where it holds none): the heights a horizontal error
    of up to `half` cells along each axis could have put there."""
    if half <= 0:
        return height, height
    top = np.where(keep, height, np.int16(NO)).astype(np.int16)
    low = np.where(keep, height, _I16_MAX).astype(np.int16)
    for axis in (0, 1):
        top = _slide(top, half, axis, np.maximum, np.int16(NO))
        low = _slide(low, half, axis, np.minimum, _I16_MAX)
    return top, low


def horizontal_cells(ua: float, ub: float, motion_m: float, cell_m: float, sigmas: float) -> int:
    """The horizontal radius in whole cells: the least number covering
    `sigmas * sqrt(ua^2 + ub^2 + 2 motion^2)` (module docstring)."""
    reach = sigmas * math.sqrt(ua * ua + ub * ub + 2.0 * motion_m * motion_m)
    return math.ceil(round(reach / cell_m, 9))


def local_rise_cm(
    height: NDArray[np.int16], keep: NDArray[np.bool_], k: int, unknown_cm: int
) -> NDArray[np.int32]:
    """Per cell, the steepest rise (centimetres, either way) to a cell `k` away along either
    axis that also has a height -- one-sided differences, the largest -- or `unknown_cm`
    where none has (module docstring, "The ground envelope's drift"). Whole centimetres,
    so exact, and 4 bytes a cell."""
    nj, ni = height.shape
    best = np.full((nj, ni), -1, dtype=np.int32)
    for dj, di in ((0, k), (0, -k), (k, 0), (-k, 0)):
        if abs(dj) >= nj or abs(di) >= ni:
            continue
        # here[j, i] against there[j + dj, i + di], where both are inside the plane.
        here = (slice(max(0, -dj), nj - max(0, dj)), slice(max(0, -di), ni - max(0, di)))
        there = (slice(max(0, dj), nj - max(0, -dj)), slice(max(0, di), ni - max(0, -di)))
        rise = np.abs(height[there].astype(np.int32) - height[here])
        view = best[here]
        np.maximum(view, rise, out=view, where=keep[here] & keep[there])
    best[best < 0] = unknown_cm
    return best


# --- the change ----------------------------------------------------------------------------

#: `change_planes` classifies a band of rows at a time, of about this many cells.
BAND_CELLS = 1 << 16


@dataclass(frozen=True)
class ChangePlanes:
    """The change of one layer between two surfaces at one level, over the union of their
    windows: `classes` (`NONE` where neither has the layer's height), the difference in
    centimetres (0 where not compared), and `low` (low confidence, compared cells only)."""

    layer: Layer
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
    low: NDArray[np.bool_]
    #: The least threshold over the compared cells, metres; None when none was compared.
    min_detectable_m: float | None
    uncertainty_from_m: float
    uncertainty_to_m: float
    #: The layer's own motion between visits (`w`, metres), the horizontal radius each cell
    #: was compared within (cells), and the reach of the ground's drift (`R`, metres; 0
    #: for vegetation).
    motion_m: float = 0.0
    horizontal_cells: int = 0
    drift_reach_m: float = 0.0

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
    layer: Layer,
    confidence_a: cell_grid.Grid | None = None,
    confidence_b: cell_grid.Grid | None = None,
    level: int | None = None,
    sigmas: float = THRESHOLD_SIGMA,
    motion_m: float | None = None,
    horizontal: bool = True,
) -> ChangePlanes:
    """The change of `layer` from `a` to `b` (module docstring), with `ua` and `ub` each
    epoch's placement uncertainty in metres, at the coarser of their levels (or `level`
    when that is coarser still, or coarser yet when the windows would hold over
    `MAX_CHANGE_CELLS`). `motion_m` is the layer's own motion (by default 0 for the
    ground, `VEGETATION_MOTION_M` for vegetation); `horizontal=False` compares cell for
    cell, as P5's demo does (the committed fixture's reference). A cell is low confidence
    when a confidence grid given does not vouch for it, or a grid is not given."""
    for name, value in (("ua", ua), ("ub", ub)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite uncertainty >= 0, not {value!r}")
    motion = (0.0 if layer == "ground" else VEGETATION_MOTION_M) if motion_m is None else motion_m
    native = max(a.level, b.level)
    at = max(native, level if level is not None else native)
    while _union_cells(a, b, at) > MAX_CHANGE_CELLS and at < cell_grid.MAX_LEVEL:
        at += 1
    ha, hb = a.heights(layer), b.heights(layer)
    # The finer epoch is brought to the coarser one's level over the children whose coarse
    # cell the coarser one measured; then both go up together over the children both
    # measured (module docstring), knowing per cell what each side measured at all.
    swap = ha.level < hb.level
    fine, coarse = (ha, hb) if swap else (hb, ha)
    if fine.level < coarse.level:
        lifted, count = _under(fine, coarse)
        whole = 1 << (2 * (coarse.level - fine.level))
        partial = _Cells(
            count.level, count.i0, count.j0, (count.values > 0) & (count.values < whole)
        )
        seen = _Cells(fine.level, fine.i0, fine.j0, fine.has)
        while seen.level < coarse.level:
            seen = seen.up("any")
        coarse_seen = _Cells(coarse.level, coarse.i0, coarse.j0, coarse.has)
        ca, cb = (lifted, coarse) if swap else (coarse, lifted)
        seen_a, seen_b = (seen, coarse_seen) if swap else (coarse_seen, seen)
        joint = _joint(ca, cb, partial)
    else:
        joint = _joint(ha, hb, None)
        seen_a = _Cells(ha.level, ha.i0, ha.j0, ha.has)
        seen_b = _Cells(hb.level, hb.i0, hb.j0, hb.has)
    while joint.level < at:
        joint = _joint_up(joint, seen_a, seen_b)
        seen_a, seen_b = seen_a.up("any"), seen_b.up("any")
    i0, j0 = min(seen_a.i0, seen_b.i0), min(seen_a.j0, seen_b.j0)
    ni = max(seen_a.i0 + seen_a.window[2], seen_b.i0 + seen_b.window[2]) - i0
    nj = max(seen_a.j0 + seen_a.window[3], seen_b.j0 + seen_b.window[3]) - j0
    union = (i0, j0, ni, nj)
    # Into the union window: insufficient where one side measured something and the two
    # cannot be compared; no class where neither measured anything.
    anywhere = _place(seen_a.values, seen_a.i0, seen_a.j0, union, False) | _place(
        seen_b.values, seen_b.i0, seen_b.j0, union, False
    )
    classes = np.where(anywhere, np.int8(INSUFFICIENT), np.int8(NONE)).astype(np.int8)
    del anywhere
    delta_cm = np.zeros((nj, ni), dtype=np.int32)
    low = np.zeros((nj, ni), dtype=bool)
    # Classified on the joint window (where alone both can have a height), a band of rows
    # at a time; the neighbourhoods and the grade are of the whole joint window.
    jw = joint.window
    di, dj = jw[0] - i0, jw[1] - j0
    (da, sa), (db, sb) = joint.a, joint.b
    # Compared only over every child the finer epoch (or, coarser, either) measured, and
    # only where neither sigma saturates (its threshold would be a lower bound); each
    # against the other epoch's neighbourhood within the horizontal radius.
    compared = joint.shared & ~joint.partial & (sa < SATURATED) & (sb < SATURATED)
    cell_m = math.ldexp(cell_grid.LATTICE_BASE_M, at)
    half = horizontal_cells(ua, ub, motion, cell_m, sigmas) if horizontal else 0
    top_a, low_a = neighbourhood(da, compared, half)
    top_b, low_b = neighbourhood(db, compared, half)
    reach = max(a.ground_reach_m, b.ground_reach_m) if layer == "ground" else 0.0
    rise: NDArray[np.int32] | None = None
    run_cm = 0.0
    if reach > 0:
        k = max(1, round(reach / math.sqrt(2.0) / cell_m))
        run_cm = k * cell_m / CM
        unknown = math.ceil(UNKNOWN_GRADE * run_cm)
        rise = local_rise_cm(da, compared, k, unknown)
        np.maximum(rise, local_rise_cm(db, compared, k, unknown), out=rise)
    grids = [vouched(g, at) if g is not None else None for g in (confidence_a, confidence_b)]
    least = math.inf
    floor = ua * ua + ub * ub + 2.0 * motion * motion
    step = max(1, BAND_CELLS // max(jw[2], 1))
    for r0 in range(0, jw[3], step):
        r1 = min(jw[3], r0 + step)
        band = (jw[0], jw[1] + r0, jw[2], r1 - r0)
        both = compared[r0:r1]
        here_a, here_b = da[r0:r1].astype(np.int32), db[r0:r1].astype(np.int32)
        delta = here_b - here_a
        spread = (sa[r0:r1].astype(np.float64) * CM) ** 2 + (sb[r0:r1].astype(np.float64) * CM) ** 2
        threshold = sigmas * np.sqrt(spread + floor)
        if rise is not None:
            threshold = threshold + rise[r0:r1] / run_cm * reach
        # Raised: the later height is above all the earlier had nearby, or the earlier one
        # below all the later has nearby (a hole filled); lowered the mirror. A shift within
        # the radius explains every cell, a new height none (module docstring).
        up = np.maximum(here_b - top_a[r0:r1], low_b[r0:r1].astype(np.int32) - here_a)
        down = np.maximum(low_a[r0:r1].astype(np.int32) - here_b, here_a - top_b[r0:r1])
        out = classes[dj + r0 : dj + r1, di : di + jw[2]]
        out[both] = UNCHANGED
        out[both & (up * CM > threshold)] = RAISED
        out[both & (down * CM > threshold)] = LOWERED
        delta_cm[dj + r0 : dj + r1, di : di + jw[2]] = np.where(both, delta, 0)
        if both.any():
            least = min(least, float(threshold[both].min()))
        strong = np.ones(both.shape, dtype=bool)
        for grid in grids:
            if grid is None:
                strong[...] = False
            else:
                strong &= _place(grid.channels["strong"], grid.i0, grid.j0, band, False)
        low[dj + r0 : dj + r1, di : di + jw[2]] = both & ~strong
    return ChangePlanes(
        layer=layer,
        level=at,
        native_level=native,
        i0=i0,
        j0=j0,
        ni=ni,
        nj=nj,
        classes=classes,
        delta_cm=delta_cm,
        low=low,
        min_detectable_m=least if math.isfinite(least) else None,
        uncertainty_from_m=ua,
        uncertainty_to_m=ub,
        motion_m=motion,
        horizontal_cells=half,
        drift_reach_m=reach,
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
    cells = np.bincount(flat, minlength=count)
    lows = np.bincount(flat, weights=planes.low.ravel(), minlength=count)
    measured = np.bincount(flat, weights=compared, minlength=count)
    delta_sum = np.bincount(flat, weights=np.where(compared, delta, 0.0), minlength=count)
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
            )
        )
    order: Mapping[str, int] = {"raised": 0, "lowered": 1, "insufficient": 2}
    return sorted(out, key=lambda area: (order[area.change], -area.cells))


@dataclass(frozen=True)
class ChangeMap:
    """One layer's change at its compared level (`planes`, whose counts the map states)
    and the areas drawn at `drawn.level` (the same planes unless they were too large to
    outline)."""

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
    layer: Layer,
    confidence_a: cell_grid.Grid | None = None,
    confidence_b: cell_grid.Grid | None = None,
) -> ChangeMap:
    """`change_planes` of `layer` and its areas, a level up at a time while they are too
    large to outline (module docstring)."""
    planes = change_planes(
        a, b, ua, ub, layer=layer, confidence_a=confidence_a, confidence_b=confidence_b
    )
    drawn = planes
    while needs_coarser(drawn) and drawn.level < cell_grid.MAX_LEVEL:
        drawn = change_planes(
            a,
            b,
            ua,
            ub,
            layer=layer,
            confidence_a=confidence_a,
            confidence_b=confidence_b,
            level=drawn.level + 1,
        )
    return ChangeMap(planes=planes, drawn=drawn, areas=change_areas(drawn, projection))
