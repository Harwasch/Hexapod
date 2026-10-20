"""The change between two epochs, from their height surfaces (M5, ADR 0018 § API).

**Geometry first.** Each epoch's run measured a height surface per cell (P5, `surface.bin`:
its top surface `dsm`, its bare `ground` and the spread `sigma` of its highest samples, in
centimetres above the site frame's origin, on its confidence grid's plane). Two epochs of
one site share that frame, and their cells nest (`0.25 m * 2**level`), so the change is a
comparison cell by cell -- never of splat pixels, which would read a season or a sky as
geometry. Appearance is not compared at all (`appearance-only` is a reserved class that is
never emitted): that needs the new photos compared with the old epoch at fixed geometry.

**Resampling.** Both surfaces are brought to the coarser of their two levels
(`coarsen_surface`): of the children that have a surface, `dsm` the highest, `ground` the
lowest and `sigma` the largest -- a coarse cell is as tall as its tallest part, as low as
its lowest, and as uncertain as its least certain; a cell none of whose children has a
surface has none. This is P5's demo `coarsen_surface` (tools/pipeline
`experiments/confidence_demo.py`), held to it by the committed fixture
`data/tiles/synthetic-yard-surfaces`. If the two windows together would still be over
`MAX_CHANGE_CELLS` cells, both go up further until they are not (and say so).

**The rule** (`change-v1`). Where both epochs have a surface, `delta = dsm_to - dsm_from`
(computed in 32 bits: two i2 heights can differ by more than i2 holds) against
`t = 3 * sqrt(sigma_from^2 + sigma_to^2 + u_from^2 + u_to^2)`, where `u` is each epoch's
uncertainty from the site frame (its confidence grid's `alignmentCap.uncertaintyM`: 0 for
the epoch that made the frame, its alignment's for one aligned to it, its GPS's for one
nobody could align). `raised` when `delta > t`, `lowered` when `delta < -t`, else
`unchanged`; `insufficient` where exactly one epoch has a surface; no class where neither
has. The 3 is a first-principles choice (three standard deviations of the sum of two
independent errors), not a measurement: the map says `uncalibrated` until a site scanned
twice measures it. `minDetectableM` is the least `t` over the compared cells: a change
smaller than it cannot be seen anywhere between these two dates -- an epoch placed by GPS
alone at +/-10 m makes it about 30 m, and so finds no change rather than false ones.

**Low confidence is flagged, not hidden.** A compared cell is `lowConfidence` when either
epoch's confidence grid does not vouch for it at the compared level (weak-v1: scored under
128, or unseen, or no evidence at all; the grid's own min-of-children pyramid, so a coarse
cell is as weak as its weakest child) or when either sigma is saturated (255 cm means at
least 2.55 m: the threshold is then a lower bound). It is still classified: P5's demo found
the weak cells of a real change were its crate's own sides, so hiding them would hide the
change.

**Areas.** The classified cells become outlines in the site frame (`change_areas`): every
4-connected region of one class (raised and lowered split further by low confidence) is a
polygon with its exact cell edges, through `rasterio.features.shapes` as weak areas are
outlined, then to the globe through the frame's own tangent plane
(`weak_areas.site_projection`). The outline is bounded as weak areas' are: a map over
`weak_areas.MAX_ZONE_CELLS` cells or over `weak_areas.MAX_ZONE_EDGES` cell edges of
outline is classified again a level up (from the surfaces, by the same rule) and drawn
there, and its areas' numbers are of the cells they were drawn from.
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
#: 12 bytes a cell (its outputs and each confidence grid's vouched cells) plus a band.
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


def _quads(values: NDArray[Any], s: Surface, fill: int) -> NDArray[Any]:
    """`values` (over `s`) padded to whole parents with `fill`, shaped `(nj, 2, ni, 2)`."""
    pi0, pj0 = s.i0 >> 1, s.j0 >> 1
    ni = ((s.i0 + s.ni - 1) >> 1) - pi0 + 1 if s.cells else 0
    nj = ((s.j0 + s.nj - 1) >> 1) - pj0 + 1 if s.cells else 0
    padded = np.full((2 * nj, 2 * ni), fill, dtype=values.dtype)
    di, dj = s.i0 - 2 * pi0, s.j0 - 2 * pj0
    padded[dj : dj + s.nj, di : di + s.ni] = values
    out: NDArray[Any] = padded.reshape(nj, 2, ni, 2)
    return out


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


def surface_at(s: Surface, level: int) -> Surface:
    if level < s.level:
        raise cell_grid.CellGridError(f"level {level} is finer than the surface's ({s.level})")
    while s.level < level:
        s = coarsen_surface(s)
    return s


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


def _window_of(a: Surface, b: Surface) -> tuple[int, int, int, int]:
    i0, j0 = min(a.i0, b.i0), min(a.j0, b.j0)
    return (
        i0,
        j0,
        max(a.i0 + a.ni, b.i0 + b.ni) - i0,
        max(a.j0 + a.nj, b.j0 + b.nj) - j0,
    )


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


#: `change_planes` works a band of rows at a time, of about this many cells, so nothing
#: but its outputs (10 bytes a cell) is ever held for the whole window.
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
    pa, pb = surface_at(a, at), surface_at(b, at)
    i0, j0, ni, nj = _window_of(pa, pb)
    grids = [vouched(g, at) if g is not None else None for g in (confidence_a, confidence_b)]
    classes = np.full((nj, ni), NONE, dtype=np.int8)
    delta_cm = np.zeros((nj, ni), dtype=np.int32)
    ground_delta_cm = np.zeros((nj, ni), dtype=np.int32)
    low = np.zeros((nj, ni), dtype=bool)
    least = math.inf
    rows = max(1, BAND_CELLS // max(ni, 1))
    for r0 in range(0, nj, rows):
        r1 = min(nj, r0 + rows)
        band = (i0, j0 + r0, ni, r1 - r0)
        da, ga = _place(pa.dsm, pa.i0, pa.j0, band, NO), _place(pa.ground, pa.i0, pa.j0, band, NO)
        db, gb = _place(pb.dsm, pb.i0, pb.j0, band, NO), _place(pb.ground, pb.i0, pb.j0, band, NO)
        sa, sb = _place(pa.sigma, pa.i0, pa.j0, band, 0), _place(pb.sigma, pb.i0, pb.j0, band, 0)
        ha = (sa > 0) & (da != NO) & (ga != NO)
        hb = (sb > 0) & (db != NO) & (gb != NO)
        both = ha & hb
        out = classes[r0:r1]
        out[ha ^ hb] = INSUFFICIENT
        # Wide: two i2 heights can differ by more than an i2 holds.
        delta = db.astype(np.int32) - da.astype(np.int32)
        delta_cm[r0:r1] = np.where(both, delta, 0)
        ground_delta_cm[r0:r1] = np.where(both, gb.astype(np.int32) - ga.astype(np.int32), 0)
        spread = (sa.astype(np.float64) * CM) ** 2 + (sb.astype(np.float64) * CM) ** 2
        threshold = sigmas * np.sqrt(spread + ua * ua + ub * ub)
        change_m = delta * CM
        out[both] = UNCHANGED
        out[both & (change_m > threshold)] = RAISED
        out[both & (change_m < -threshold)] = LOWERED
        if both.any():
            least = min(least, float(threshold[both].min()))
        strong = np.ones(both.shape, dtype=bool)
        for grid in grids:
            if grid is None:
                strong[...] = False
            else:
                strong &= _place(grid.channels["strong"], grid.i0, grid.j0, band, False)
        low[r0:r1] = both & (~strong | (sa >= SATURATED) | (sb >= SATURATED))
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

#: What an area is drawn by: its class and, for raised and lowered, its confidence. (An
#: unchanged or insufficient area is drawn whole; it says how many of its cells are low.)
_GROUPS: tuple[tuple[ChangeClass, bool | None], ...] = (
    ("raised", False),
    ("raised", True),
    ("lowered", False),
    ("lowered", True),
    ("insufficient", None),
    ("unchanged", None),
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

    change: ChangeClass
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
    lowered first, then insufficient, then unchanged, largest first within each. Each
    area's numbers are of its own cells (its region's, at `planes.level`)."""
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
    order: Mapping[str, int] = {"raised": 0, "lowered": 1, "insufficient": 2, "unchanged": 3}
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
