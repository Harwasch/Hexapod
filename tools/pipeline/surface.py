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
them yet). Per cell, streaming, the eight highest and the eight lowest sample heights,
`z1 >= z2 >= ... >= z8` from the top, and the count `n`:

    dsm    = z2                        the 2nd highest: one floater cannot raise it
    ground = the least 2nd lowest over a 5 m square of cells
                                       the bare ground under a crown or a roof
    sigma  = max(1 cm, (z1 - zm) RMS2(n) / SPAN(n)),  m = min(n, 8)
                                       how far z2 is from the surface, bias included
    groundSigma = max(1 cm, s_low RMS2(N))
                                       how far the ground is from the bare surface

`ground` is the 2nd-lowest height (one sample under the surface cannot lower it) of every
cell whose centre is within 2.5 m of this one's along each axis (`GROUND_WINDOW_M`): under
a crown or a roof no wider than that, some neighbour saw the ground. That needs cells of
2 m or less (levels 0-3); from 4 m cells up the window is the cell alone and the ground is
its own 2nd-lowest sample (`groundWindowCells` in the descriptor says which). It is a lower
envelope, so it is biased **low, never high**: on a slope of grade `g` by up to `2.5 g`
metres along each axis, and a low outlier that survives the 2nd-lowest rule spreads over
its whole square.

**The ground's sigma** (`groundSigma`) is the envelope's own, as `sigma` is dsm's. The
least 2nd-lowest over a window holding `N` samples in all is never lower than the 2nd
lowest of those `N` together, so its distance below the ground is at most that order
statistic's: `s_low RMS2(N)`, with `N` the window's samples (a box sum) and `s_low` the
noise scale **of the cell whose 2nd-lowest set the minimum**, `(z_m - z_1) / SPAN(n)` from
its bottom `m` samples (a tie goes to the larger). Never an average over the window: a
minimum is set by the noisiest lows, and a window of 1 cm cells with one 10 cm cell reads
some 30 cm low with an averaged sigma of 5-7 cm (review B5a round 2: 28.7% of unchanged
cells lowered). Far larger than dsm's at the same noise (at 1 cm and 100 samples a cell,
about 5 cm against dsm's 2 cm): a lower envelope over 441 cells is a biased estimator, and
its sigma says so. 0 where the ground is `NO_HEIGHT`.

**Ground that was never seen.** Under a crown or a roof wider than the window, every cell
of the window sees only it, and the least of them is the canopy, not the ground. That is
told from the ground around it: a cell's ground is **unobserved** -- `NO_HEIGHT` in
`ground`, with its `dsm` still there -- when it is higher than any slope up to
`GROUND_MAX_GRADE` (1 in 2) from a lower 2nd-lowest sample anywhere allows
(`grade_envelope`: the least of `low(c') + grade * L1 distance`). A slope that steep or
gentler never trips it (its own samples are such a slope); a 4 m canopy does, wherever the
ground beside it is closer than 8 m. Ground steeper than 1 in 2 beside a drop reads as
unobserved (a loss, never a wrong height).

**Under a wide canopy** (`under_canopy`, review B5a round 2). Deeper inside a crown than
its height over the grade, the slope test lets the canopy stand as ground (397 cm at the
centre of a 4 m crown 20 m wide), so its growth read as the ground rising. What gives a
canopy away is its edge: a cell that stands above the ground its own window saw by more
than any slope of the grade could explain across the window (`grade * R`, R the window's
half-diagonal: 1.77 m at the 5 m window, plus 3 sigma of its top and ground) is a **seed**,
on top of something, and keeps the ground it saw. From the seeds that ground is carried
across the canopy's top -- through adjacent cells that never drop by more than a slope of
the grade could (plus 3 sigma; any rise, a taller crown on a canopy being canopy still) --
and every cell it reaches whose own window ground stands more than the same margin above
it has **no ground**: never canopy height. Ground beside the canopy has its own window
ground at the ground's level, so it takes nothing. Limits, each a loss rather than a wrong
height: ground that rises from a canopy's foot to above its top is unobserved where the
canopy reaches it; and a cliff's top (it stands above its own foot) is unobserved across
the flat ground it is joined to. Not caught: a canopy or roof lower than that margin
(about 1.8 m) and wider than the window, whose edge no slope test can tell from a bank.
`summary.groundUnobserved` counts every unobserved cell.

A cell with fewer than `MIN_SAMPLES` (3) samples has no surface
(`NO_HEIGHT`, sigma 0): with two, the 2nd highest is the lower and the 2nd lowest the
higher of them, so the ground would stand above the top. From three up,
`ground <= 2nd lowest <= dsm`. `sigma` is `dsm`'s: the root-mean-square distance of `z2`
from the surface under Gaussian noise of the scale the top eight's spread shows. The 2nd
highest of `n` noisy samples reads high by a bias that grows with `n` (about 1.4 noise
scales at 20 samples, 3.2 at 2,000), so sigma carries that bias as well as the scatter
("the noise model" below): two scans of unchanged ground with very different sample counts
or noise then differ by less than the change threshold. A floater does not move `dsm` but
widens `sigma`, which makes the cell slower to call changed rather than wrongly changed.

**Heights** are `z` of the tileset frame in centimetres: above the frame origin's height
(georef.json `height`, in whatever datum the site frame names), along the origin's up. That
is a tangent plane, not a height above any datum -- a kilometre from the origin it is about
`d^2 / 2R` (8 cm) above the ellipsoid there. Two epochs of a site share the frame, so their
difference is unaffected. `i2` centimetres hold +/-327.67 m; a cell whose surface is further
from the origin's height than that has none (`summary.outOfRange`), never a clipped one. A
run whose scale was never resolved is not in metres at all, and says so (`metric`, from the
confidence grid's summary).

**Memory and determinism.** One pass over the samples per band of at most `BAND_CELLS`
cells, whose extremes (137 bytes a cell: 72 MB at 2^19) are that band's alone; the plane is
10 bytes a cell (40 MB at 2^22), and making the ground about 40 more (its filter, envelope,
box sums and the canopy test's 24), whatever the number of samples. The eight highest and
lowest of a set do not depend on the order it is read in, and every other step is a
rounding of them or exact integer arithmetic, so the bytes depend on neither the chunk nor
the band.
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
    "FEW_SAMPLES",
    "GROUND_MAX_GRADE",
    "GROUND_WINDOW_M",
    "MIN_SAMPLES",
    "SIGMA_FLOOR_M",
    "SURFACE",
    "TOP_K",
    "Surfaced",
    "Window",
    "box_sum",
    "grade_envelope",
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

#: Heights kept per cell at each end: `z2`, and the eight the noise is read from.
TOP_K = 8
#: Fewer samples than this in a cell: no surface (module docstring).
MIN_SAMPLES = 3
#: `sigma` is never under a centimetre, the unit it is stored in.
SIGMA_FLOOR_M = 0.01
#: The side of the square of cell centres `ground` is the least over (module docstring).
GROUND_WINDOW_M = 5.0
#: The steepest ground the unobserved test takes as ground, rise over run (module
#: docstring): 1 in 2, about 27 degrees.
GROUND_MAX_GRADE = 0.5
#: Opaque enough to be a surface: a gaussian at least half opaque.
MIN_SPLAT_ALPHA = 0.5
#: Cells whose extremes are held in one pass: 2^19 (137 bytes each) is 72 MB.
BAND_CELLS = 1 << 19
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
#: A millimetre, and the most a cell's ground noise is held as (32 m): 15 bits of the
#: ground filter's keys, below the height's 16 (`surface_plane`).
_MM = 0.001
_NOISE_MM_MAX = (1 << 15) - 1
#: One SfM point's fixed part in `points3D.bin`: id, xyz, rgb, error, track length.
_POINT = struct.Struct("<Q3d3BdQ")

# --- the noise model: order statistics of the normal ------------------------------------
#
# `dsm` is the 2nd highest of a cell's `n` samples. Under height noise of scale `s` about a
# surface at `mu`, it reads `mu + s E[Z(2:n)]` on average, where `Z(k:n)` is the k-th
# highest of `n` standard normals: about 1.4 s high at 20 samples and 3.2 s at 2,000. So
# two scans of unchanged ground with different counts or noise differ by that bias, and
# sigma must carry it. `s` is read from the spread of the top `m = min(n, 8)` samples,
# whose expectation is `s (E[Z(1:n)] - E[Z(m:n)])`. So
#
#     sigma = (z1 - zm) * RMS2(n) / SPAN(n),   RMS2(n) = sqrt(E[Z(2:n)^2]),
#                                              SPAN(n) = E[Z(1:n)] - E[Z(m:n)]
#
# the root-mean-square distance of `z2` from `mu` (bias and scatter together) at the noise
# the top eight show. Two epochs' biases have the same sign, so their difference is never
# more than the larger, and `3 sqrt(sigma_a^2 + sigma_b^2)` covers it and the scatter.
#
# The tables are the exact normal values for n = 3 .. 64, then one per half-octave
# (n in [2^e, 1.5 2^e) and [1.5 2^e, 2^(e+1))) taken at its top: RMS2 grows with n and,
# from n = 8, SPAN shrinks, so a bucket's value gives the larger sigma. Computed by
# integrating the order statistic's density (V = 1 - Phi(Z(k:n)) is Beta(k, n - k + 1)),
# RMS2 rounded up and SPAN down at the 4th decimal; `tests/test_surface_stage.py` checks
# them against a second, independent integration. Fixed numbers in the source, so the
# stored bytes depend on no machine's libm. Gaussian noise is the model, not a measurement:
# heavier tails read further high, and that is uncalibrated (ADR 0018).
#
# **Few samples.** From three to six samples the spread is a poor estimate of the noise
# (a range of two to five degrees of freedom), and an under-read sigma calls noise a
# change: against a noiseless other side, a cell of three samples read at the mean span
# called 6.3% of unchanged cells changed, four 2.9%, five 1.2%, six 0.46%, against the
# rule's 0.27% (review B5a round 2 found it on the demo's flat scans: three samples at 1
# and at 10 cm). The scale is widened by `FEW_SAMPLES` there, to the rule's rate at worst:
# a cell of three samples is five times less sure of itself, and says so.

#: Counts the tables hold exactly; above, half-octaves to 2^48 (beyond, the last).
_EXACT_MAX = 64
# fmt: off
_RMS2 = (
    0.6699, 0.6699, 0.7461, 0.8316, 0.9113, 0.9827, 1.0464, 1.1033, 1.1545, 1.2009, 1.2432,
    1.2820, 1.3178, 1.3510, 1.3818, 1.4107, 1.4378, 1.4632, 1.4873, 1.5100, 1.5316, 1.5522,
    1.5718, 1.5904, 1.6083, 1.6255, 1.6419, 1.6577, 1.6729, 1.6875, 1.7016, 1.7153, 1.7285,
    1.7412, 1.7536, 1.7655, 1.7772, 1.7884, 1.7994, 1.8101, 1.8204, 1.8305, 1.8404, 1.8500,
    1.8593, 1.8684, 1.8774, 1.8861, 1.8946, 1.9029, 1.9110, 1.9190, 1.9268, 1.9344, 1.9419,
    1.9493, 1.9564, 1.9635, 1.9704, 1.9772, 1.9839, 1.9904, 2.1502, 2.2625, 2.4138, 2.5167,
    2.6562, 2.7517, 2.8817, 2.9711, 3.0934, 3.1777, 3.2933, 3.3733, 3.4833, 3.5595, 3.6646,
    3.7375, 3.8382, 3.9083, 4.0052, 4.0726, 4.1660, 4.2312, 4.3215, 4.3845, 4.4719, 4.5330,
    4.6179, 4.6773, 4.7598, 4.8175, 4.8978, 4.9540, 5.0323, 5.0871, 5.1635, 5.2171, 5.2917,
    5.3441, 5.4170, 5.4683, 5.5397, 5.5899, 5.6599, 5.7090, 5.7777, 5.8259, 5.8932, 5.9406,
    6.0067, 6.0532, 6.1181, 6.1638, 6.2277, 6.2726, 6.3354, 6.3796, 6.4414, 6.4849, 6.5458,
    6.5886, 6.6486, 6.6908, 6.7499, 6.7915, 6.8497, 6.8908, 6.9482, 6.9887, 7.0454, 7.0853,
    7.1413, 7.1807, 7.2359, 7.2748, 7.3294, 7.3678, 7.4217, 7.4597, 7.5130, 7.5505, 7.6031,
    7.6403, 7.6923, 7.7290,
)
_SPAN = (
    1.6925, 2.0587, 2.3259, 2.5344, 2.7043, 2.8472, 2.4173, 2.1948, 2.0484, 1.9414, 1.8585,
    1.7915, 1.7359, 1.6887, 1.6479, 1.6122, 1.5807, 1.5525, 1.5271, 1.5041, 1.4830, 1.4637,
    1.4459, 1.4294, 1.4141, 1.3998, 1.3864, 1.3739, 1.3620, 1.3509, 1.3403, 1.3303, 1.3208,
    1.3118, 1.3032, 1.2950, 1.2871, 1.2796, 1.2724, 1.2655, 1.2588, 1.2524, 1.2463, 1.2404,
    1.2346, 1.2291, 1.2238, 1.2186, 1.2136, 1.2088, 1.2041, 1.1995, 1.1951, 1.1908, 1.1866,
    1.1826, 1.1786, 1.1748, 1.1711, 1.1674, 1.1639, 1.1604, 1.0824, 1.0342, 0.9760, 0.9403,
    0.8960, 0.8681, 0.8328, 0.8102, 0.7813, 0.7625, 0.7381, 0.7222, 0.7014, 0.6877, 0.6696,
    0.6576, 0.6417, 0.6311, 0.6170, 0.6075, 0.5949, 0.5863, 0.5749, 0.5672, 0.5569, 0.5499,
    0.5404, 0.5340, 0.5253, 0.5193, 0.5113, 0.5059, 0.4984, 0.4934, 0.4865, 0.4817, 0.4753,
    0.4709, 0.4648, 0.4607, 0.4551, 0.4512, 0.4459, 0.4422, 0.4372, 0.4337, 0.4290, 0.4257,
    0.4212, 0.4181, 0.4139, 0.4109, 0.4069, 0.4041, 0.4003, 0.3976, 0.3939, 0.3914, 0.3879,
    0.3855, 0.3821, 0.3798, 0.3766, 0.3744, 0.3713, 0.3691, 0.3662, 0.3641, 0.3613, 0.3593,
    0.3566, 0.3547, 0.3521, 0.3503, 0.3478, 0.3460, 0.3436, 0.3419, 0.3395, 0.3379, 0.3356,
    0.3340, 0.3318, 0.3303,
)
# fmt: on


def _order_index(count: I64) -> I64:
    """Each count's (>= 3) row of `_RMS2` and `_SPAN`: exact to 64, then its half-octave
    (the bit after the leading one), from integer arithmetic alone."""
    n = np.maximum(count, 3)
    _, exponent = np.frexp(n.astype(np.float64))  # n < 2^53: exact; n = m 2^exponent
    top = exponent.astype(np.int64) - 1  # floor(log2 n)
    half = (n >> np.maximum(top - 1, 0)) & 1
    bucket = (_EXACT_MAX - 2) + 2 * (top - 6) + half
    index = np.where(n <= _EXACT_MAX, n - 3, bucket)
    out: I64 = np.minimum(index, len(_RMS2) - 1)
    return out


def rms_second(count: I64) -> F64:
    """`RMS2(n)`: sqrt(E[Z(2:n)^2]) for standard normal noise (the noise model above)."""
    out: F64 = np.asarray(_RMS2, dtype=np.float64)[_order_index(count)]
    return out


#: How much wider the noise scale is read from three to six samples (module docstring,
#: "few samples"): the least factor keeping a cell's false-change rate at the 3 sigma
#: rule's 0.27% against a noiseless other side (the worst case: its noise is all there is),
#: by Monte Carlo (4,000,000 draws: 4.955, 2.186, 1.399, 1.094 at 3, 4, 5, 6 samples),
#: with about 3% to spare, rounded up to 0.05. From seven samples none is needed (0.17%
#: at seven unscaled, falling).
FEW_SAMPLES: Mapping[int, float] = {3: 5.1, 4: 2.25, 5: 1.45, 6: 1.15}


def few_samples(count: I64) -> F64:
    """The factor the noise scale read from `count` samples is widened by (`FEW_SAMPLES`;
    1 from five up)."""
    out: F64 = np.ones(np.shape(count), dtype=np.float64)
    for n, factor in FEW_SAMPLES.items():
        out = np.where(count == n, factor, out)
    return out


def top_span(count: I64) -> F64:
    """`SPAN(n)`: E[Z(1:n) - Z(m:n)], m = min(n, TOP_K) (the noise model above)."""
    out: F64 = np.asarray(_SPAN, dtype=np.float64)[_order_index(count)]
    return out


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


#: What `_centimetres` gives a height it cannot hold: past i2's range by far, and an int64.
_PAST_CM = 1 << 40


def _centimetres(metres: F64) -> I64:
    """Whole centimetres, half up. Anything not finite, or too large for the int64 cast to
    be exact (|cm| >= 2^40: eleven million km), becomes `_PAST_CM` *before* the cast, so it
    is past every range rather than wrapped into a plausible height."""
    with np.errstate(invalid="ignore", over="ignore"):
        cm = np.floor(metres * 100.0 + 0.5)
        held = np.isfinite(cm) & (np.abs(cm) < _PAST_CM)
    out: I64 = np.where(held, cm, float(_PAST_CM)).astype(np.int64)
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
    #: Surfaced cells whose ground was never observed (module docstring): `NO_HEIGHT` ground.
    ground_unobserved: int = 0
    #: Cells whose groundSigma is past 255 cm (stored as 255: "at least").
    ground_sigma_saturated: int = 0


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


def box_sum(values: I64, half: int) -> I64:
    """The sum of each cell's `(2 half + 1)^2` square of cells (cells past the plane's edge
    are none): two passes of exact integer cumulative sums."""
    if half <= 0:
        return values.copy()
    out = values
    for axis in (1, 0):
        n = out.shape[axis]
        total = np.cumsum(out, axis=axis)
        zero = np.zeros_like(np.take(total, [0], axis=axis))
        padded = np.concatenate([zero, total], axis=axis)  # padded[k] = sum of the first k
        hi = np.minimum(np.arange(n) + half + 1, n)
        lo = np.maximum(np.arange(n) - half, 0)
        out = np.take(padded, hi, axis=axis) - np.take(padded, lo, axis=axis)
    result: I64 = out
    return result


def grade_envelope(low: npt.NDArray[np.int32], step: float) -> F64:
    """Per cell, the least over every cell `c'` with a ground (not `_NO_GROUND`) of
    `low(c') + step * (|di| + |dj|)`: the highest a slope of `step` per cell from any
    lower sample anywhere lets this cell's ground be. Infinite where no cell has one.

    The L1 distance is separable, so it is two 1D passes, each a forward and a backward
    running minimum (`h[i] = min_k (f[k] + step |i - k|)` is `step i + cummin(f - step k)`
    from the left, and its mirror from the right). Every value is a whole centimetre plus
    a multiple of `step` (12.5 cm at 0.25 m cells and a grade of 1 in 2), exact in float64,
    so the result depends on no order of operations. L1 is at least the straight-line
    distance, so a real slope up to the grade in any direction stays under it."""
    out: F64 = np.where(low == _NO_GROUND, np.inf, low.astype(np.float64))
    for axis in (1, 0):
        n = out.shape[axis]
        ramp = np.arange(n, dtype=np.float64) * step
        ramp = ramp[None, :] if axis == 1 else ramp[:, None]
        with np.errstate(invalid="ignore"):
            out = np.minimum.accumulate(out - ramp, axis=axis) + ramp
            back = np.flip(np.minimum.accumulate(np.flip(out + ramp, axis), axis=axis), axis)
            out = back - ramp
    return out


#: Larger than any height difference the canopy test meets, in centimetres.
_FAR_CM = 1 << 30


def under_canopy(
    dsm: npt.NDArray[np.int16],
    ground: npt.NDArray[np.int32],
    sigma: npt.NDArray[np.uint8],
    ground_sigma_cm: F64,
    has_ground: npt.NDArray[np.bool_],
    unobserved: npt.NDArray[np.bool_],
    *,
    step_cm: float,
    stand_cm: float,
) -> npt.NDArray[np.bool_]:
    """The cells with a ground whose ground is a canopy's or a roof's (module docstring,
    "under a wide canopy"): reached across a surface that never drops by more than a slope
    of the grade could (adjacent tops: a fall of at most `step_cm` plus 3 sigma; any
    rise, a taller crown on a canopy being canopy still), from a cell that stands
    more than `stand_cm` plus 3 sigma above the ground its window saw, carrying the lowest
    such ground `v`, and whose own window ground stands that far above `v`. Cells the slope
    test already left unobserved relay `v` too. Whole centimetres, so exact; four sweeps a
    round (east, west, north, south), vectorised across the other axis, until nothing
    changes (`v` only falls, over a finite set of values)."""
    nj, ni = dsm.shape
    has = dsm != cell_grid.NO_HEIGHT
    # int32 and float32 throughout (heights are i2 centimetres): 24 bytes a cell.
    top = dsm.astype(np.int32)
    low = ground.astype(np.int32)
    spread = sigma.astype(np.float32) ** 2
    above = np.ceil(stand_cm + 3.0 * np.sqrt(spread + ground_sigma_cm.astype(np.float32) ** 2))
    above = above.astype(np.int32)
    seed = has_ground & (top - low > above)
    value = np.where(seed, low, _FAR_CM).astype(np.int32)
    # What a cell will take: any value if the slope test left it unobserved; else one
    # its own window ground stands more than `above` over (`low - v > above`).
    takes = np.where(
        unobserved & has, _FAR_CM, np.where(has_ground & ~seed, low - above, -_FAR_CM)
    ).astype(np.int32)

    # Across a surface the value may climb any step (a taller crown on a canopy is still
    # canopy) but drop no more than a slope of the grade could, plus 3 sigma of the two tops.
    rise = np.where(has, top, -_FAR_CM).astype(np.int32)

    def sweep(axis: int, forward: bool) -> bool:
        changed = False
        n = ni if axis == 1 else nj
        steps = range(1, n) if forward else range(n - 2, -1, -1)
        for k in steps:
            src = k - 1 if forward else k + 1
            pick = (slice(None), k) if axis == 1 else (k, slice(None))
            from_ = (slice(None), src) if axis == 1 else (src, slice(None))
            tolerance = step_cm + 3.0 * np.sqrt(spread[pick] + spread[from_])
            link = has[pick] & has[from_] & (rise[pick] - rise[from_] >= -tolerance)
            incoming, here = value[from_], value[pick]
            better = link & (incoming < here) & (incoming < takes[pick])
            if better.any():
                here[better] = incoming[better]
                changed = True
        return changed

    while True:
        changed = False
        for axis, forward in ((1, True), (1, False), (0, True), (0, False)):
            changed |= sweep(axis, forward)
        if not changed:
            break
    out: npt.NDArray[np.bool_] = has_ground & ~seed & (value < _FAR_CM)
    return out


def surface_plane(
    samples: Samples,
    window: Window,
    *,
    min_samples: int = MIN_SAMPLES,
    sigma_floor_m: float = SIGMA_FLOOR_M,
    ground_window_m: float = GROUND_WINDOW_M,
    ground_max_grade: float = GROUND_MAX_GRADE,
    band_cells: int = BAND_CELLS,
) -> Surfaced:
    """Every sample binned on `window`, and the five channels (module docstring).

    Memory: one band of extremes at a time (`band_cells` cells, 137 bytes each), the plane
    (10 bytes a cell), and while the ground is made, about 40 bytes a cell more: the
    filter's two 4-byte planes, the envelope's 8-byte one and the box sums' 8-byte ones."""
    if min_samples < 3:
        raise ValueError(
            f"min_samples is at least 3, not {min_samples}: with two samples the 2nd lowest "
            "is the higher of them, and the ground would stand above the surface"
        )
    if not (math.isfinite(sigma_floor_m) and sigma_floor_m > 0):
        raise ValueError(f"sigma_floor_m must be a positive length, not {sigma_floor_m}")
    if not (math.isfinite(ground_window_m) and ground_window_m >= 0):
        raise ValueError(f"ground_window_m must be a length, not {ground_window_m}")
    if not (math.isfinite(ground_max_grade) and ground_max_grade > 0):
        raise ValueError(f"ground_max_grade must be a positive grade, not {ground_max_grade}")
    ni, nj = window.ni, window.nj
    dsm = np.full((nj, ni), cell_grid.NO_HEIGHT, dtype=np.int16)
    ground = np.full((nj, ni), _NO_GROUND, dtype=np.int32)
    sigma = np.zeros((nj, ni), dtype=np.uint8)
    support = np.zeros((nj, ni), dtype=np.uint8)
    # The ground's own noise per cell (the bottom eight's spread over SPAN, whole mm) and
    # its sample count: what groundSigma is made of once the window is known.
    noise_mm = np.zeros((nj, ni), dtype=np.int32)
    counts = np.zeros((nj, ni), dtype=np.int64)
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
        z1, zm = band.top[:, 0], band.top[np.arange(count.size), last]
        with np.errstate(invalid="ignore"):
            scale = (z1 - zm) * (few_samples(count) / top_span(count))
            spread = np.maximum(sigma_floor_m, scale * rms_second(count))
            # Rounded up -- never understated -- after a micro-rounding that keeps a
            # floating error from turning exactly 1 cm into 2.
            sigma_cm = np.ceil(np.round(spread / cell_grid.HEIGHT_UNIT_M, 6))
        zl1, zlm = band.bottom[:, 0], band.bottom[np.arange(count.size), last]
        with np.errstate(invalid="ignore"):
            low_noise = np.ceil(
                np.round((zlm - zl1) * (few_samples(count) / top_span(count)) / _MM, 6)
            )
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
        noise_mm[band0:band1] = np.where(
            surfaced, np.clip(np.nan_to_num(low_noise, nan=_NOISE_MM_MAX), 1, _NOISE_MM_MAX), 0
        ).reshape(shape)
        counts[band0:band1] = count.reshape(shape)
        del band
    half = ground_half_cells(window.cell_m, ground_window_m)
    # The least 2nd-lowest over the window, and the noise scale of the cell it came from
    # (module docstring): one exact filter over keys that sort by the height, then by the
    # larger noise, so a tie is resolved towards the larger sigma.
    keys = np.where(
        ground == _NO_GROUND,
        _NO_GROUND,
        ((ground.astype(np.int64) + 32768) << 15) | (_NOISE_MM_MAX - noise_mm),
    ).astype(np.int32)
    del noise_mm
    least = ground_filter(keys, half)
    del keys
    filtered = np.where(least == _NO_GROUND, _NO_GROUND, (least >> 15) - 32768).astype(np.int32)
    noise_at = np.where(least == _NO_GROUND, 0, _NOISE_MM_MAX - (least & _NOISE_MM_MAX))
    del least
    # Ground the window saw only canopy or roof for: higher than any slope up to
    # GROUND_MAX_GRADE from a lower 2nd-lowest anywhere allows (module docstring).
    step_cm = ground_max_grade * window.cell_m / cell_grid.HEIGHT_UNIT_M
    unobserved = (filtered != _NO_GROUND) & (filtered > grade_envelope(ground, step_cm))
    del ground
    has_ground = (filtered != _NO_GROUND) & ~unobserved
    pooled = box_sum(counts, half)
    del counts
    with np.errstate(invalid="ignore"):
        spread = np.maximum(sigma_floor_m, noise_at * _MM * rms_second(pooled))
        ground_sigma_cm = np.ceil(np.round(spread / cell_grid.HEIGHT_UNIT_M, 6))
    del pooled, noise_at, spread
    # Ground under a canopy too wide for the window or the slope test: reached from the
    # canopy's edge step across its top (module docstring, "under a wide canopy").
    reach_m = half * window.cell_m * math.sqrt(2.0)
    canopy = under_canopy(
        dsm,
        filtered,
        sigma,
        ground_sigma_cm,
        has_ground,
        unobserved,
        step_cm=ground_max_grade * window.cell_m / cell_grid.HEIGHT_UNIT_M,
        stand_cm=ground_max_grade * reach_m / cell_grid.HEIGHT_UNIT_M,
    )
    has_ground &= ~canopy
    unobserved |= canopy
    del canopy
    ground_saturated = int((has_ground & (ground_sigma_cm > 255)).sum())
    data = np.empty((nj, ni, len(cell_grid.SURFACE_CHANNELS)), dtype=np.int16)
    data[..., 0] = dsm
    del dsm
    data[..., 1] = np.where(has_ground, filtered, cell_grid.NO_HEIGHT)
    del filtered
    data[..., 2] = sigma
    data[..., 3] = support
    data[..., 4] = np.where(has_ground, np.clip(ground_sigma_cm, 1, 255), 0)
    del ground_sigma_cm
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
        ground_unobserved=int(unobserved.sum()),
        ground_sigma_saturated=ground_saturated,
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
        "groundUnobserved": surfaced.ground_unobserved,
        "groundSigmaSaturated": surfaced.ground_sigma_saturated,
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
    max_grade = float(ctx.param("ground_max_grade", GROUND_MAX_GRADE))
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
        ground_max_grade=max_grade,
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
            "sigma": f"max({sigma_floor:g} m, (z1 - zm) RMS2(n) / SPAN(n)), m = min(n, "
            f"{TOP_K}): the RMS distance of the 2nd highest from the surface, its bias "
            "included, under normal noise of the scale the top samples' spread shows",
            "minSamples": min_samples,
            "groundWindowM": window_m,
            "groundWindowCells": cells,
            "groundUnobserved": f"no ground ({cell_grid.NO_HEIGHT}) under a surface where "
            "the window's least is higher than a slope of "
            f"{max_grade:g} from any lower 2nd-lowest sample allows: canopy or roof "
            "wider than the window, the ground under it never seen",
            "groundMaxGrade": max_grade,
            "groundSigma": f"max({sigma_floor:g} m, s_low RMS2(N)): N the window's samples, "
            "s_low the (z_m - z_1) / SPAN(n) of the lowest samples of the cell that set the "
            "minimum: "
            "the 2nd lowest of N samples bounds the envelope's distance below the ground",
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
