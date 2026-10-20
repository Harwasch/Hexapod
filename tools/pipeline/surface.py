"""`height_surface`: an epoch's measured heights, per cell of its site's lattice.

Change between two epochs of a site is measured as change in **geometry**, never in splat
pixels (ADR 0018, docs/DECISIONS/0018-change-detection.md). This stage gives each epoch
the geometry to compare: per cell, the height of its top surface (`dsm`), of the bare
ground under it (`ground`), and how far the top surface's own samples disagree (`sigma`).

**Samples.** Two sets of points, both in the tileset frame:

- the pose model's sparse SfM points (`poses/points3D.bin`, COLMAP's frame), placed by the
  similarity `place` applied to the trained splat (`placement.json`: the composition of
  `stages.place_points`' two steps), applied component by component as `confidence.py`
  does, so no BLAS kernel can move a bit;
- the centres of `canonical.ply`'s gaussians at least half opaque (`opacity >= 0` in the
  file's logit, so no exponential either); canonical.ply is already in the tileset frame.

**The lattice.** The epoch's confidence grid (`confidence/confidence.json`) has already
chosen the level and the window of cells; the surface is written on exactly that plane --
the same level, `i0`, `j0`, `ni`, `nj` -- so every surface cell *is* a confidence cell, and
two epochs' surfaces nest exactly as their confidence grids do (cell_grid.py, C1). Samples
outside that window are counted (`summary.outside`) and not binned.

**Estimators** (first principles, **uncalibrated**: no measurement on a real site backs
them yet). Per cell, streaming, the four highest and the four lowest sample heights,
`z1 >= z2 >= z3 >= z4` from the top:

    dsm    = z2                        the 2nd highest: one floater cannot raise it
    ground = the least 2nd lowest over a 5 m square of cells
                                       the bare ground under a crown or a roof
    sigma  = max(1 cm, (z1 - z4) / 2)  half the spread of the top four

`ground` is the 2nd-lowest height (one sample under the surface cannot lower it) of every
cell whose centre is within 2.5 m of this one's along each axis (`GROUND_WINDOW_M`): under
a crown or a roof no wider than that, some neighbour saw the ground. It is a lower
envelope, so it is biased **low, never high**: on a slope of grade `g` by up to `2.5 g`
metres along each axis, and a low outlier that survives the 2nd-lowest rule spreads over
its whole square. A cell with fewer than `MIN_SAMPLES` (3) samples has no surface
(`NO_HEIGHT`, sigma 0): with two, the 2nd highest is the lower and the 2nd lowest the
higher of them, so the ground would stand above the top. From three up,
`ground <= 2nd lowest <= dsm`. `sigma` is `dsm`'s: a floater does not move `dsm` but widens
`sigma`, which makes the cell slower to call changed rather than wrongly changed. Only `dsm`
and `sigma` enter the change threshold.

**Heights** are `z` of the tileset frame in centimetres: above the frame origin's height
(georef.json `height`, in whatever datum the site frame names), along the origin's up. That
is a tangent plane, not a height above any datum -- a kilometre from the origin it is about
`d^2 / 2R` (8 cm) above the ellipsoid there. Two epochs of a site share the frame, so their
difference is unaffected. `i2` centimetres hold +/-327.67 m; a cell whose surface is further
from the origin's height than that has none (`summary.outOfRange`), never a clipped one. A
run whose scale was never resolved is not in metres at all, and says so (`metric`, from the
confidence grid's summary).

**Memory and determinism.** One pass over the samples per band of at most `BAND_CELLS`
cells, whose extremes (73 bytes a cell: 76 MB at 2^20) are that band's alone; the plane is
8 bytes a cell (32 MB at 2^22) and the ground filter two more 4-byte planes, whatever the
number of samples. The four highest and lowest of a set do not depend on the order it is
read in, and every other step is a rounding of them, so the bytes depend on neither the
chunk nor the band.
"""

from __future__ import annotations

import json
import math
import struct
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import confidence
import splat_io
from artifacts import ArtifactDecl
from captures_bridge import cell_grid
from contracts import MetricValue, StageContext, StageOutcome
from placement import PLACEMENT
from registry import stage_impl

__all__ = [
    "BAND_CELLS",
    "GROUND_WINDOW_M",
    "MIN_SAMPLES",
    "SIGMA_FLOOR_M",
    "SURFACE",
    "TOP_K",
    "Surfaced",
    "Window",
    "ground_filter",
    "ground_half_cells",
    "read_window",
    "sfm_points",
    "splat_centres",
    "surface_plane",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

DESCRIPTOR_FILE = "surface.json"
BIN_FILE = "surface.bin"
SURFACE = ArtifactDecl(
    "surface",
    kind="dir",
    content_type="inode/directory",
    summary="this epoch's measured heights per cell, on its confidence grid's plane in the "
    "tileset frame (hexapod.cellgrid: dsm, ground, sigma, support; uncalibrated "
    "estimators); surface.json is the descriptor package declares on the root",
    required_members=(DESCRIPTOR_FILE, BIN_FILE),
    stub_members=(DESCRIPTOR_FILE, BIN_FILE),
)

#: Heights kept per cell at each end: what `z2` and `z4` need.
TOP_K = 4
#: Fewer samples than this in a cell: no surface (module docstring).
MIN_SAMPLES = 3
#: `sigma` is never under a centimetre, the unit it is stored in.
SIGMA_FLOOR_M = 0.01
#: The side of the square of cell centres `ground` is the least over (module docstring).
GROUND_WINDOW_M = 5.0
#: Opaque enough to be a surface: a gaussian at least half opaque.
MIN_SPLAT_ALPHA = 0.5
#: Cells whose extremes are held in one pass: 2^20 (73 bytes each) is 76 MB.
BAND_CELLS = 1 << 20
#: The most cells the plane may have: the confidence grid's own cap.
MAX_CELLS = confidence.MAX_CELLS
#: SfM points read from `points3D.bin` at a time.
POINTS_CHUNK = 1 << 16
#: Where a sample came from (`support`'s low nibble).
SFM = cell_grid.SURFACE_SOURCE_BITS["sfm"]
SPLAT = cell_grid.SURFACE_SOURCE_BITS["splat"]
#: i2's range in centimetres, less the sentinel.
HEIGHT_LIMIT_CM = 32767
_NO_GROUND = np.iinfo(np.int32).max
#: One SfM point's fixed part in `points3D.bin`: id, xyz, rgb, error, track length.
_POINT = struct.Struct("<Q3d3BdQ")

# --- inputs ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The confidence grid's finest plane: the cells the surface is written on."""

    level: int
    i0: int
    j0: int
    ni: int
    nj: int
    sha256: str | None = None

    @property
    def cell_m(self) -> float:
        return cell_grid.Lattice(self.level).cell_m

    @property
    def cells(self) -> int:
        return self.ni * self.nj


def _integer(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"confidence.json: {what} must be an integer, not {value!r}")
    return value


def read_window(descriptor: Mapping[str, Any]) -> Window:
    """The finest plane of a confidence grid's descriptor, checked: a confidence
    `hexapod.cellgrid` v1 in the tileset frame, whose lattice is its finest level's."""
    if (
        descriptor.get("format") != cell_grid.FORMAT
        or descriptor.get("version") != cell_grid.VERSION
        or descriptor.get("kind") != "confidence"
        or descriptor.get("frame") != cell_grid.FRAME
    ):
        raise ValueError(
            "confidence.json is not a confidence grid in the tileset frame "
            f"(format {descriptor.get('format')!r}, kind {descriptor.get('kind')!r}, "
            f"frame {descriptor.get('frame')!r}): the surface has no plane to align to"
        )
    levels = descriptor.get("levels")
    lattice = descriptor.get("lattice")
    if not isinstance(levels, list) or not levels or not isinstance(levels[0], Mapping):
        raise ValueError("confidence.json declares no levels")
    if not isinstance(lattice, Mapping):
        raise ValueError("confidence.json has no lattice")
    finest = levels[0]
    sha = descriptor.get("sha256")
    window = Window(
        level=_integer(finest.get("level"), "levels[0].level"),
        i0=_integer(finest.get("i0"), "levels[0].i0"),
        j0=_integer(finest.get("j0"), "levels[0].j0"),
        ni=_integer(finest.get("ni"), "levels[0].ni"),
        nj=_integer(finest.get("nj"), "levels[0].nj"),
        sha256=sha if isinstance(sha, str) else None,
    )
    if lattice.get("level") != window.level or lattice.get("baseM") != cell_grid.LATTICE_BASE_M:
        raise ValueError("confidence.json's lattice is not its finest level's")
    if window.ni < 1 or window.nj < 1:
        raise ValueError(f"confidence.json's finest plane is {window.ni}x{window.nj} cells")
    cell_grid.Lattice(window.level)
    return window


#: A stream of samples: positions `(n, 3)` float64 in the tileset frame, and the source.
Samples = Callable[[], Iterator[tuple[F64, int]]]


def splat_centres(path: Path, chunk: int, min_alpha: float = MIN_SPLAT_ALPHA) -> Samples:
    """The centres of `path`'s gaussians at least `min_alpha` opaque, a chunk at a time.
    The test is on the stored logit (`opacity >= logit(min_alpha)`: 0 for a half)."""
    if not 0.0 < min_alpha < 1.0:
        raise ValueError(f"min_splat_alpha must be inside (0, 1), not {min_alpha}")
    threshold = math.log(min_alpha / (1.0 - min_alpha))
    reader = splat_io.SplatReader(path, chunk=chunk)

    def stream() -> Iterator[tuple[F64, int]]:
        for _, columns in reader.chunks(("x", "y", "z", "opacity")):
            opaque = columns["opacity"].astype(np.float64) >= threshold
            xyz = np.stack([columns[name][opaque] for name in ("x", "y", "z")], axis=1)
            yield xyz.astype(np.float64), SPLAT

    return stream


def sfm_points(path: Path, placement: confidence.Placement, chunk: int = POINTS_CHUNK) -> Samples:
    """`points3D.bin`'s positions placed into the tileset frame, `chunk` points at a time.
    Streamed -- each point's fixed part read, its track skipped by its length -- so the
    memory is a chunk's whatever the model's size. Nothing when the file is absent."""

    def stream() -> Iterator[tuple[F64, int]]:
        if not path.is_file():
            return
        with path.open("rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                raise ValueError(f"{path.name} is too short to be a points3D.bin")
            (count,) = struct.unpack("<Q", head)
            done = 0
            while done < count:
                size = min(chunk, count - done)
                xyz = np.empty((size, 3), dtype=np.float64)
                for row in range(size):
                    record = handle.read(_POINT.size)
                    if len(record) != _POINT.size:
                        raise ValueError(f"{path.name} ends inside point {done + row}")
                    _, x, y, z, _, _, _, _, length = _POINT.unpack(record)
                    xyz[row] = (x, y, z)
                    handle.seek(8 * length, 1)
                done += size
                # A non-finite point stays non-finite and is counted, not binned.
                with np.errstate(invalid="ignore", over="ignore"):
                    placed = placement.apply(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                yield placed, SFM

    return stream


def chain(*streams: Samples) -> Samples:
    """One stream of several, in order."""

    def stream() -> Iterator[tuple[F64, int]]:
        for each in streams:
            yield from each()

    return stream


# --- the extremes -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Band:
    """Rows `rows` of the plane: per cell its sample count, the sources that reached it,
    and its `TOP_K` highest (descending) and lowest (ascending) heights; and how many
    samples the pass saw at all (finite, and not)."""

    count: I64
    sources: npt.NDArray[np.uint8]
    top: F64
    bottom: F64
    binned: dict[int, int]
    finite: int
    non_finite: int


def _merge(held: F64, cells: I64, found: F64, *, highest: bool) -> None:
    """`held[cells]` becomes the `TOP_K` most extreme of itself and `found`, row by row."""
    both = np.sort(np.concatenate([held[cells], found], axis=1), axis=1)
    held[cells] = both[:, ::-1][:, :TOP_K] if highest else both[:, :TOP_K]


def _band(samples: Samples, window: Window, rows: tuple[int, int]) -> _Band:
    """One pass over the samples, keeping the extremes of the band's cells alone."""
    band0, band1 = rows
    ni = window.ni
    size = (band1 - band0) * ni
    count = np.zeros(size, dtype=np.int64)
    sources = np.zeros(size, dtype=np.uint8)
    top = np.full((size, TOP_K), -np.inf)
    bottom = np.full((size, TOP_K), np.inf)
    binned: dict[int, int] = {SFM: 0, SPLAT: 0}
    lattice = cell_grid.Lattice(window.level)
    finite_total = non_finite_total = 0
    for xyz, source in samples():
        finite = np.isfinite(xyz).all(axis=1)
        seen = int(finite.sum())
        finite_total += seen
        non_finite_total += int(xyz.shape[0]) - seen
        i, j = lattice.index(xyz)
        di, dj = i - window.i0, j - window.j0 - band0
        mine = finite & (di >= 0) & (di < ni) & (dj >= 0) & (dj < band1 - band0)
        if not mine.any():
            continue
        local = dj[mine] * ni + di[mine]
        z = xyz[mine, 2]
        binned[source] += int(local.size)
        count += np.bincount(local, minlength=size)
        order = np.lexsort((z, local))
        keys, heights = local[order], z[order]
        cells, start, many = np.unique(keys, return_index=True, return_counts=True)
        high = np.full((cells.size, TOP_K), -np.inf)
        low = np.full((cells.size, TOP_K), np.inf)
        for rank in range(TOP_K):
            has = many > rank
            low[has, rank] = heights[start[has] + rank]
            high[has, rank] = heights[start[has] + many[has] - 1 - rank]
        _merge(top, cells, high, highest=True)
        _merge(bottom, cells, low, highest=False)
        sources[cells] |= np.uint8(source)
    return _Band(count, sources, top, bottom, binned, finite_total, non_finite_total)


def _centimetres(metres: F64) -> I64:
    """Whole centimetres, half up; anything not finite is past every range."""
    with np.errstate(invalid="ignore", over="ignore"):
        cm = np.floor(metres * 100.0 + 0.5)
    out: I64 = np.where(np.isfinite(cm), cm, float(1 << 40)).astype(np.int64)
    return out


def support_of(count: I64, sources: npt.NDArray[np.uint8]) -> I64:
    """`support` (cell_grid.py): the source bits, and `floor(log2 count)` -- exact, from
    the float exponent -- in the high nibble, at most 15."""
    _, exponent = np.frexp(np.maximum(count, 1).astype(np.float64))
    log2 = np.where(count > 0, np.minimum(exponent.astype(np.int64) - 1, 15), 0)
    out: I64 = (log2 << 4) | sources.astype(np.int64)
    return out


@dataclass(frozen=True)
class Surfaced:
    """The plane, and what binning it counted."""

    plane: cell_grid.CellPlane
    binned: dict[int, int]
    #: Finite samples outside the plane's window, and samples with a non-finite coordinate.
    outside: int
    non_finite: int
    #: Cells with samples but fewer than `min_samples`; cells whose heights `i2` cannot hold;
    #: cells whose sigma is past 255 cm (stored as 255: "at least").
    thin: int
    out_of_range: int
    sigma_saturated: int
    ground_half: int


def ground_half_cells(cell_m: float, window_m: float = GROUND_WINDOW_M) -> int:
    """The ground filter's half-width in cells: the cells whose centres are within
    `window_m / 2` of this one's along an axis (0: the cell alone)."""
    return math.floor((window_m / 2.0) / cell_m + 1e-9)


def ground_filter(ground: npt.NDArray[np.int32], half: int) -> npt.NDArray[np.int32]:
    """The least of each cell's `(2 half + 1)^2` square of cells; a cell without a ground
    (`_NO_GROUND`) takes no part and gets none. Separable -- a row pass, then a column pass,
    each the minimum of `2 half + 1` shifted copies -- and exact, since a minimum is."""
    if half <= 0:
        return ground.copy()
    nj, ni = ground.shape
    rows = np.full((nj, ni + 2 * half), _NO_GROUND, dtype=np.int32)
    rows[:, half : half + ni] = ground
    across = rows[:, 0:ni].copy()
    for shift in range(1, 2 * half + 1):
        np.minimum(across, rows[:, shift : shift + ni], out=across)
    del rows
    columns = np.full((nj + 2 * half, ni), _NO_GROUND, dtype=np.int32)
    columns[half : half + nj] = across
    del across
    out = columns[0:nj].copy()
    for shift in range(1, 2 * half + 1):
        np.minimum(out, columns[shift : shift + nj], out=out)
    del columns
    filtered: npt.NDArray[np.int32] = np.where(ground == _NO_GROUND, _NO_GROUND, out).astype(
        np.int32
    )
    return filtered


def surface_plane(
    samples: Samples,
    window: Window,
    *,
    min_samples: int = MIN_SAMPLES,
    sigma_floor_m: float = SIGMA_FLOOR_M,
    ground_window_m: float = GROUND_WINDOW_M,
    band_cells: int = BAND_CELLS,
) -> Surfaced:
    """Every sample binned on `window`, and the four channels (module docstring).

    Memory: one band of extremes at a time (`band_cells` cells, 73 bytes each), the plane
    (8 bytes a cell) and, while the ground is filtered, two 4-byte planes."""
    if min_samples < 3:
        raise ValueError(
            f"min_samples is at least 3, not {min_samples}: with two samples the 2nd lowest "
            "is the higher of them, and the ground would stand above the surface"
        )
    if not (math.isfinite(sigma_floor_m) and sigma_floor_m > 0):
        raise ValueError(f"sigma_floor_m must be a positive length, not {sigma_floor_m}")
    if not (math.isfinite(ground_window_m) and ground_window_m >= 0):
        raise ValueError(f"ground_window_m must be a length, not {ground_window_m}")
    ni, nj = window.ni, window.nj
    dsm = np.full((nj, ni), cell_grid.NO_HEIGHT, dtype=np.int16)
    ground = np.full((nj, ni), _NO_GROUND, dtype=np.int32)
    sigma = np.zeros((nj, ni), dtype=np.uint8)
    support = np.zeros((nj, ni), dtype=np.uint8)
    rows_per_band = max(1, band_cells // max(1, ni))
    binned = {SFM: 0, SPLAT: 0}
    thin = out_of_range = saturated = 0
    finite = non_finite = 0
    for band0 in range(0, nj, rows_per_band):
        band1 = min(nj, band0 + rows_per_band)
        band = _band(samples, window, (band0, band1))
        finite, non_finite = band.finite, band.non_finite
        for source, many in band.binned.items():
            binned[source] += many
        count = band.count
        enough = count >= min_samples
        thin += int(((count > 0) & ~enough).sum())
        last = np.maximum(np.minimum(count, TOP_K) - 1, 0)
        z1, zk = band.top[:, 0], band.top[np.arange(count.size), last]
        with np.errstate(invalid="ignore"):
            spread = np.maximum(sigma_floor_m, (z1 - zk) / 2.0)
            # Rounded up -- never understated -- after a micro-rounding that keeps a
            # floating error from turning exactly 1 cm into 2.
            sigma_cm = np.ceil(np.round(spread / cell_grid.HEIGHT_UNIT_M, 6))
        top_cm = _centimetres(band.top[:, 1])
        low_cm = _centimetres(band.bottom[:, 1])
        fits = (np.abs(top_cm) <= HEIGHT_LIMIT_CM) & (np.abs(low_cm) <= HEIGHT_LIMIT_CM)
        surfaced = enough & fits
        out_of_range += int((enough & ~fits).sum())
        saturated += int((surfaced & (sigma_cm > 255)).sum())
        shape = (band1 - band0, ni)
        dsm[band0:band1] = np.where(surfaced, top_cm, cell_grid.NO_HEIGHT).reshape(shape)
        ground[band0:band1] = np.where(surfaced, low_cm, _NO_GROUND).reshape(shape)
        sigma[band0:band1] = np.where(
            surfaced, np.clip(np.nan_to_num(sigma_cm, nan=255.0), 1, 255), 0
        ).reshape(shape)
        support[band0:band1] = support_of(count, band.sources).reshape(shape)
        del band
    half = ground_half_cells(window.cell_m, ground_window_m)
    filtered = ground_filter(ground, half)
    del ground
    data = np.empty((nj, ni, len(cell_grid.SURFACE_CHANNELS)), dtype=np.int16)
    data[..., 0] = dsm
    del dsm
    data[..., 1] = np.where(filtered == _NO_GROUND, cell_grid.NO_HEIGHT, filtered)
    del filtered
    data[..., 2] = sigma
    data[..., 3] = support
    plane = cell_grid.CellPlane(window.level, window.i0, window.j0, ni, nj, data)
    return Surfaced(
        plane=plane,
        binned=binned,
        outside=finite - binned[SFM] - binned[SPLAT],
        non_finite=non_finite,
        thin=thin,
        out_of_range=out_of_range,
        sigma_saturated=saturated,
        ground_half=half,
    )


# --- the stage ---------------------------------------------------------------------------


def summarise(surfaced: Surfaced, metric: bool | None) -> dict[str, Any]:
    """`summary`: what was surfaced, from what, and what was left out and why."""
    plane = surfaced.plane
    has = plane.data[..., 2] > 0
    sigmas = np.sort(plane.data[..., 2][has].astype(np.int64))
    median = None
    if sigmas.size:
        median = (int(sigmas[(sigmas.size - 1) // 2]) + int(sigmas[sigmas.size // 2])) / 2
    return {
        "cells": plane.cells,
        "surfaced": int(has.sum()),
        "thin": surfaced.thin,
        "samples": {"sfm": surfaced.binned[SFM], "splat": surfaced.binned[SPLAT]},
        "outside": surfaced.outside,
        "nonFinite": surfaced.non_finite,
        "outOfRange": surfaced.out_of_range,
        "sigmaSaturated": surfaced.sigma_saturated,
        "medianSigmaCm": median,
        "metric": metric,
    }


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return loaded


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None


@stage_impl(
    "height_surface",
    consumes=("canonical.ply", "poses", PLACEMENT.name, confidence.CONFIDENCE.name),
    optional_consumes=("georef.json",),
    produces=(SURFACE,),
    summary="measure the epoch's top surface, bare ground and their spread per cell of its "
    "confidence grid: the geometry two epochs' change is measured in",
)
def height_surface(ctx: StageContext) -> StageOutcome:
    """See the module. Writes `surface/surface.json` (the descriptor) and
    `surface/surface.bin`; the plane is the confidence grid's finest, cell for cell."""
    started = time.perf_counter()
    chunk = int(ctx.param("chunk_gaussians", splat_io.CHUNK))
    band = int(ctx.param("band_cells", BAND_CELLS))
    min_samples = int(ctx.param("min_samples", MIN_SAMPLES))
    sigma_floor = float(ctx.param("sigma_floor_m", SIGMA_FLOOR_M))
    window_m = float(ctx.param("ground_window_m", GROUND_WINDOW_M))
    min_alpha = float(ctx.param("min_splat_alpha", MIN_SPLAT_ALPHA))
    max_cells = int(ctx.param("max_cells", MAX_CELLS))
    graded = _read_json(ctx.input(confidence.CONFIDENCE.name) / confidence.DESCRIPTOR_FILE)
    window = read_window(graded)
    if window.cells > max_cells:
        raise ValueError(
            f"the confidence plane is {window.ni}x{window.nj} cells, more than max_cells "
            f"({max_cells}); the surface is written on that plane or not at all"
        )
    placement = confidence.read_placement(ctx.input(PLACEMENT.name))
    points = ctx.input("poses") / "points3D.bin"
    if not points.is_file():
        ctx.log("surface: poses/ has no points3D.bin; the splat's centres alone")
    samples = chain(
        sfm_points(points, placement), splat_centres(ctx.input("canonical.ply"), chunk, min_alpha)
    )
    surfaced = surface_plane(
        samples,
        window,
        min_samples=min_samples,
        sigma_floor_m=sigma_floor,
        ground_window_m=window_m,
        band_cells=band,
    )
    grid_summary = graded.get("summary")
    metric = grid_summary.get("metric") if isinstance(grid_summary, Mapping) else None
    summary = summarise(surfaced, metric if isinstance(metric, bool) else None)
    georef = _read_json(ctx.input("georef.json")) if ctx.has_input("georef.json") else {}
    plane = surfaced.plane
    cells = 2 * surfaced.ground_half + 1
    meta: dict[str, object] = {
        "policy": "surface-v1",
        "status": "uncalibrated",
        "estimators": {
            "dsm": "the 2nd highest sample in the cell",
            "ground": "the 2nd lowest sample, then the least of it over the cells whose "
            f"centres are within {window_m / 2:g} m along each axis ({cells}x{cells} cells)",
            "sigma": f"max({sigma_floor:g} m, (z1 - z4) / 2) of the 4 highest samples",
            "minSamples": min_samples,
            "groundWindowM": window_m,
            "groundWindowCells": cells,
            "samples": "SfM points3D placed by placement.json, and canonical.ply centres "
            f"at least {min_alpha:g} opaque",
        },
        "noData": cell_grid.NO_HEIGHT,
        "sourceBits": dict(cell_grid.SURFACE_SOURCE_BITS),
        "sigmaSaturatesAt": 255,
        # What a height is measured from (module docstring).
        "vertical": {
            "zero": "the tileset frame's origin",
            "originHeightM": _number(georef.get("height")),
            "surface": "the tangent plane at the origin, not a height above a datum",
        },
        # The confidence plane this one is, cell for cell.
        "alignedTo": {"kind": "confidence", "level": window.level, "sha256": window.sha256},
        "summary": summary,
    }
    grid = cell_grid.CellGrid("surface", cell_grid.SURFACE_CHANNELS, (plane,), meta)
    out = ctx.output(SURFACE.name)
    descriptor = cell_grid.encode(grid, out, BIN_FILE)
    (out / DESCRIPTOR_FILE).write_text(json.dumps(descriptor, indent=1) + "\n", encoding="utf-8")
    seconds = time.perf_counter() - started
    ctx.log(
        f"surface: level {plane.level} ({plane.cell_m:g} m cells), {plane.ni}x{plane.nj}, the "
        f"confidence plane; {summary['surfaced']} cells surfaced, {summary['thin']} too thin, "
        f"{summary['outOfRange']} past i2; samples {summary['samples']}, {summary['outside']} "
        f"outside, {summary['nonFinite']} not finite; median sigma {summary['medianSigmaCm']} "
        f"cm; ground over {cells}x{cells} cells; {seconds:.1f} s (uncalibrated estimators)"
    )
    metrics: dict[str, MetricValue] = {
        "level": plane.level,
        "cellM": plane.cell_m,
        "cells": plane.cells,
        "surfaced": int(summary["surfaced"]),
        "thin": int(summary["thin"]),
        "outOfRange": int(summary["outOfRange"]),
        "sfmSamples": surfaced.binned[SFM],
        "splatSamples": surfaced.binned[SPLAT],
        "seconds": round(seconds, 2),
    }
    return StageOutcome(
        metrics=metrics, summary=f"{summary['surfaced']} of {plane.cells} cells surfaced"
    )
