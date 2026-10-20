"""The geometry of block training: the partition, the camera rule, the frozen ring, the
cross-fade at the seams and each block's held-out frames.

Pure numpy and the standard library, importable by CPython 3.10 (the trainer's
interpreter). The ideas are the papers': testing a camera by how much a block changes the
coarse model's render (CityGaussian, arXiv 2404.01133), a margin round each block and a
visibility fallback (VastGaussian, arXiv 2402.17427), and a frozen ring of surroundings
(Hierarchical 3DGS, arXiv 2406.12080). The rules below -- quantile grid, SSIM boundary,
blend width and its cap, merge order -- are this project's own, stated as maths.

* **Partition.** A grid in the ground plane of the cameras' up: axes from the principal
  direction of the points, cell edges at per-axis quantiles (so each column and each row
  holds as many points as the others, within one), the outer edges open.
* **Cameras.** A camera trains a block when it stands in it, or when removing the block
  from the coarse model's render changes the view: 1 - SSIM > epsilon (Wang, Bovik,
  Sheikh and Simoncelli, IEEE TIP 13(4), 2004), with local statistics renormalised to
  the in-frame pixels so a block seen only at the frame border counts.
* **Ring.** The prior's gaussians within a square annulus round the block (block-
  normalised L-infinity distance <= outer), outside its trainable region.
* **Blend.** Across each interior edge, a linear ramp of half-width h = min(blend * D,
  margin * w, w / 2): D the spacing of the adjacent cell centres, w the narrower cell's
  width. The per-cell weights telescope, so the blocks' weights sum to exactly 1
  everywhere, and the cap keeps the band inside both blocks' trained margins.

Derivations (the ring and the blend, written from the papers' prose and this module's
stated rules, not from any implementation):

* Ring. Hierarchical 3DGS (arXiv 2406.12080) trains each chunk against frozen content
  round it, so a chunk does not invent its own surroundings. Measure "round it" in the
  block's own size, so one rule fits a long block and a square one: with centre c and
  size s per axis, d(p) = max_a |p_a - c_a| / s_a, the L-infinity distance in block
  units, whose level sets are squares like the grid's cells. The ring is
  {p : d(p) <= outer} minus the trainable region (`in_expanded`): a point is either
  optimised or held fixed, never both. Tested as |p_a - c_a| <= outer * s_a, the same
  set without a division.
* Blend. Where two blocks meet, each point's contribution must be a convex combination of
  the blocks' gaussians, weights >= 0 summing to 1, or the band doubles (or loses)
  density. Take per edge j a step-like S_j(x) in [0, 1], "how far x has crossed edge j",
  with S_0 = 1 and S_K = 0 on the open outer edges, and give cell k the weight
  S_k - S_{k+1}: the sum over cells telescopes to S_0 - S_K = 1 whatever the S_j are,
  and each weight is >= 0 when S_k >= S_{k+1}. The fewest-assumption S_j is a linear ramp
  from 0 at e_j - h to 1 at e_j + h, centred on the edge so neither side is favoured.
  h = 0 is the hard crop of VastGaussian (arXiv 2402.17427) and of the CityGaussian paper
  (arXiv 2404.01133); a point on the edge then goes to the cell after it, as in
  `cell_of`. Scaling h with the centre spacing D keeps the band the same fraction of a
  block on any grid; capping it at margin * w keeps the fading gaussians inside the
  region both neighbours trained; capping it at w / 2 means the two ramps bounding a
  cell never overlap (h_k + h_{k+1} <= w_k), so S_k >= S_{k+1} and no weight is negative
  for any margin. Two axes: the weight of cell (i, j) is the product of its per-axis
  weights, a product of two partitions of unity is one, and a block's weight is the sum
  over its cells.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B = npt.NDArray[np.bool_]

#: The trainable region grows by this fraction of each cell's width per side (VastGaussian,
#: arXiv 2402.17427, expands by 20%; 10% is what the measured 2-block spool run used:
#: 26.42 dB merged against 26.39 whole).
DEFAULT_MARGIN = 0.1
#: The frozen ring reaches this many block sizes from the block's centre: one full block
#: width beyond each edge, so a camera anywhere in the block sees at least a block width
#: of frozen surroundings in every ground direction. The frozen surroundings are
#: Hierarchical 3DGS's (arXiv 2406.12080). The value 1.5 is from that project's released
#: code, not the paper: its scaffold loads coarse gaussians up to 1.5 chunk sizes from the
#: chunk's centre. Only the value is kept; ADR 0012 names the file.
DEFAULT_RING_OUTER = 1.5
#: Half-width of the cross-fade, as a fraction of the spacing of adjacent cell centres.
#: The value is from Hierarchical 3DGS's (arXiv 2406.12080) released code, not the paper:
#: its renderer fades over a +-5% band of the distance ratio (falloff 0.05), on a uniform
#: grid a half-width of about falloff / 4 of the centre spacing. The code this replaced
#: used exactly 0.05 / 4, and the measured 2-block spool run used it. Only the value is
#: kept; ADR 0012 names the file. VastGaussian (arXiv 2402.17427) and CityGaussian (arXiv
#: 2404.01133) crop each block to its bounds, the limit blend = 0.
DEFAULT_BLEND = 0.0125
#: The camera test's 1 - SSIM threshold (`block_epsilon` is the knob): Table S3 of
#: CityGaussian, arXiv 2404.01133, which uses 0.05 for MatrixCity and Sci-Art.
DEFAULT_EPSILON = 0.05
#: The fewest training cameras a block may have; fewer and it joins a neighbour. The value
#: is from CityGaussian V2's (arXiv 2411.00771) released code, not the paper: its block
#: data parser asserts more than 50 images per block. Only the value is kept; ADR 0012
#: names the file. None of the three papers states a minimum per block.
MIN_IMAGES = 50
#: Robust outer bounds of the points, per axis.
OUTER_QUANTILES = (0.01, 0.99)
#: SSIM's window (Wang et al. 2004): 11 taps, sigma 1.5, K1 0.01 and K2 0.03 at L = 1.
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K = (0.01, 0.03)


def _unit(vector: F64) -> F64:
    out: F64 = vector / np.linalg.norm(vector)
    return out


def horizontal_axes(up: npt.ArrayLike, xyz: npt.ArrayLike) -> tuple[F64, F64]:
    """Two orthonormal ground-plane axes for `up`: the first along the points' principal
    horizontal direction (sign fixed), the second up x first."""
    u_hat = _unit(np.asarray(up, dtype=np.float64).reshape(3))
    seed = np.array([1.0, 0.0, 0.0]) if abs(u_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(seed - float(seed @ u_hat) * u_hat)
    e2: F64 = np.cross(u_hat, e1)
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return e1, e2
    planar = np.stack([points @ e1, points @ e2], axis=1)
    planar -= planar.mean(axis=0)
    scatter = planar.T @ planar
    _values, vectors = np.linalg.eigh(scatter)
    m = vectors[:, -1]
    if m[0] < 0 or (m[0] == 0 and m[1] < 0):
        m = -m
    a1 = _unit(m[0] * e1 + m[1] * e2)
    a2: F64 = np.cross(u_hat, a1)
    return a1, a2


def grid_shape(count: float, extent_u: float, extent_v: float, *, exact: bool) -> tuple[int, int]:
    """(columns, rows) for `count` blocks over extents (u, v): cells as square as possible,
    with no more cells than needed (`exact`: exactly `count`; else at least `count`)."""
    n = max(1, math.floor(count))
    lu, lv = max(float(extent_u), 1e-12), max(float(extent_v), 1e-12)
    best: tuple[tuple[float, int, int], tuple[int, int]] | None = None
    for m in range(1, n + 1):
        k = n // m if exact else -(-n // m)
        if k < 1 or (exact and m * k != n):
            continue
        score = (round(abs(math.log((lu / m) / (lv / k))) + math.log(m * k / n), 12), m * k, -m)
        if best is None or score < best[0]:
            best = (score, (m, k))
    return (n, 1) if best is None else best[1]


@dataclass(frozen=True)
class Partition:
    """A grid in the ground plane: cells (iu, iv) numbered iu * rows + iv, grouped into
    blocks. Persisted as `to_dict` (`checkpoint/blocks/plan.json`, `partition.json`)."""

    origin: tuple[float, float, float]
    axis_u: tuple[float, float, float]
    axis_v: tuple[float, float, float]
    u_edges: tuple[float, ...]
    v_edges: tuple[float, ...]
    blocks: tuple[tuple[int, ...], ...]

    @property
    def columns(self) -> int:
        return len(self.u_edges) - 1

    @property
    def rows(self) -> int:
        return len(self.v_edges) - 1

    @property
    def count(self) -> int:
        return len(self.blocks)

    def project(self, xyz: npt.ArrayLike) -> F64:
        """(n, 2) ground-plane coordinates (u, v) of points, about the origin."""
        p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        d = p - np.asarray(self.origin, dtype=np.float64)
        au, av = self.axis_u, self.axis_v
        # Element-wise, not a matrix product: the same bits on every BLAS kernel.
        u = d[:, 0] * au[0] + d[:, 1] * au[1] + d[:, 2] * au[2]
        v = d[:, 0] * av[0] + d[:, 1] * av[1] + d[:, 2] * av[2]
        out: F64 = np.stack([u, v], axis=1)
        return out

    def cell_of(self, uv: npt.ArrayLike) -> I64:
        """Each point's cell. A point on an inner edge goes to the cell after it; the outer
        edges are open, so every point (a non-finite coordinate read as 0) has one."""
        coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        coords = np.where(np.isfinite(coords), coords, 0.0)
        inner_u = np.asarray(self.u_edges[1:-1], dtype=np.float64)
        inner_v = np.asarray(self.v_edges[1:-1], dtype=np.float64)
        iu = np.searchsorted(inner_u, coords[:, 0], side="right")
        iv = np.searchsorted(inner_v, coords[:, 1], side="right")
        out: I64 = (iu * self.rows + iv).astype(np.int64)
        return out

    def block_of_cell(self) -> I64:
        """The block of each cell, -1 for a cell no block holds."""
        table = np.full(self.columns * self.rows, -1, dtype=np.int64)
        for index, cells in enumerate(self.blocks):
            for cell in cells:
                table[cell] = index
        return table

    def block_of(self, uv: npt.ArrayLike) -> I64:
        out: I64 = self.block_of_cell()[self.cell_of(uv)]
        return out

    def cell_rect(self, cell: int) -> tuple[float, float, float, float]:
        """(u0, u1, v0, v1) of a cell."""
        iu, iv = divmod(int(cell), self.rows)
        return (self.u_edges[iu], self.u_edges[iu + 1], self.v_edges[iv], self.v_edges[iv + 1])

    def rect(self, block: int) -> tuple[float, float, float, float]:
        """The bounding rectangle (u0, u1, v0, v1) of a block's cells."""
        rects = [self.cell_rect(cell) for cell in self.blocks[block]]
        return (
            min(r[0] for r in rects),
            max(r[1] for r in rects),
            min(r[2] for r in rects),
            max(r[3] for r in rects),
        )

    def centre(self, block: int) -> tuple[float, float]:
        u0, u1, v0, v1 = self.rect(block)
        return (0.5 * (u0 + u1), 0.5 * (v0 + v1))

    def in_expanded(self, uv: npt.ArrayLike, block: int, margin: float) -> B:
        """Whether each point lies in one of the block's cells grown by `margin` times that
        cell's width on each side (closed); a side on the grid's outer edge is unbounded."""
        coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        u, v = coords[:, 0], coords[:, 1]
        inside = np.zeros(coords.shape[0], dtype=bool)
        for cell in self.blocks[block]:
            iu, iv = divmod(int(cell), self.rows)
            u0, u1, v0, v1 = self.cell_rect(cell)
            gu, gv = margin * (u1 - u0), margin * (v1 - v0)
            low_u = -np.inf if iu == 0 else u0 - gu
            high_u = np.inf if iu == self.columns - 1 else u1 + gu
            low_v = -np.inf if iv == 0 else v0 - gv
            high_v = np.inf if iv == self.rows - 1 else v1 + gv
            inside |= (u >= low_u) & (u <= high_u) & (v >= low_v) & (v <= high_v)
        return inside

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": list(self.origin),
            "axisU": list(self.axis_u),
            "axisV": list(self.axis_v),
            "uEdges": list(self.u_edges),
            "vEdges": list(self.v_edges),
            "blocks": [list(cells) for cells in self.blocks],
        }

    @staticmethod
    def from_dict(document: dict[str, Any]) -> Partition:
        def triple(values: Sequence[Any]) -> tuple[float, float, float]:
            a, b, c = (float(x) for x in values)
            return (a, b, c)

        return Partition(
            origin=triple(document["origin"]),
            axis_u=triple(document["axisU"]),
            axis_v=triple(document["axisV"]),
            u_edges=tuple(float(x) for x in document["uEdges"]),
            v_edges=tuple(float(x) for x in document["vEdges"]),
            blocks=tuple(tuple(int(c) for c in cells) for cells in document["blocks"]),
        )


def _edges(values: F64, cells: int) -> tuple[float, ...]:
    low, high = (float(q) for q in np.quantile(values, OUTER_QUANTILES))
    inner = [float(np.quantile(values, j / cells)) for j in range(1, cells)]
    edges = [min([low, *inner]), *inner, max([high, *inner])]
    edges = [float(e) for e in np.maximum.accumulate(np.asarray(edges, dtype=np.float64))]
    if edges[-1] == edges[0]:
        edges[-1] = edges[0] + 1e-9
    return tuple(edges)


def partition(xyz: npt.ArrayLike, up: npt.ArrayLike, count: float, *, exact: bool) -> Partition:
    """The grid of `count` blocks (exactly, or at least) over the points' ground plane,
    one block a cell. Independent of the points' order."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        raise ValueError("no finite positions to partition")
    a1, a2 = horizontal_axes(up, points)
    axis_u = tuple(round(float(c), 9) for c in a1)
    axis_v = tuple(round(float(c), 9) for c in a2)
    origin = tuple(float(c) for c in np.median(points, axis=0))
    probe = Partition(
        origin=(origin[0], origin[1], origin[2]),
        axis_u=(axis_u[0], axis_u[1], axis_u[2]),
        axis_v=(axis_v[0], axis_v[1], axis_v[2]),
        u_edges=(0.0, 1.0),
        v_edges=(0.0, 1.0),
        blocks=((0,),),
    )
    uv = probe.project(points)
    lo_u, hi_u = np.quantile(uv[:, 0], OUTER_QUANTILES)
    lo_v, hi_v = np.quantile(uv[:, 1], OUTER_QUANTILES)
    columns, rows = grid_shape(count, float(hi_u - lo_u), float(hi_v - lo_v), exact=exact)
    return Partition(
        origin=probe.origin,
        axis_u=probe.axis_u,
        axis_v=probe.axis_v,
        u_edges=_edges(uv[:, 0], columns),
        v_edges=_edges(uv[:, 1], rows),
        blocks=tuple((cell,) for cell in range(columns * rows)),
    )


def cameras_inside(part: Partition, centres: npt.ArrayLike) -> B:
    """(cameras, blocks): True where a camera's centre stands in the block."""
    which = part.block_of(part.project(centres))
    out: B = which[:, None] == np.arange(part.count)[None, :]
    return out


# --- the camera test ------------------------------------------------------------------


def _window() -> F64:
    offsets = np.arange(SSIM_WINDOW, dtype=np.float64) - (SSIM_WINDOW - 1) / 2
    taps = np.exp(-(offsets**2) / (2.0 * SSIM_SIGMA**2))
    out: F64 = taps / taps.sum()
    return out


def _blur(image: F64, taps: F64) -> F64:
    """Separable correlation of (H, W, C) with `taps` along H and W, zero outside."""
    half = taps.size // 2
    out = image
    for axis in (0, 1):
        size = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (half, half)
        padded = np.pad(out, pad)
        total = np.zeros_like(out)
        for k, weight in enumerate(taps):
            total += weight * np.take(padded, np.arange(k, k + size), axis=axis)
        out = total
    return out


def ssim(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Mean SSIM of two images (H, W, C) or (H, W) in [0, 1] (Wang et al. 2004), local
    statistics by the 11-tap gaussian renormalised to the in-frame pixels: every local
    average is (G * x) / (G * 1) with zero outside the frame, so a border window uses only
    real pixels, weighted to sum to 1."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[:, :, None], y[:, :, None]
    taps = _window()
    norm = _blur(np.ones((*x.shape[:2], 1), dtype=np.float64), taps)

    def mean(z: F64) -> F64:
        out: F64 = _blur(z, taps) / norm
        return out

    mx, my = mean(x), mean(y)
    sxx = mean(x * x) - mx * mx
    syy = mean(y * y) - my * my
    sxy = mean(x * y) - mx * my
    c1, c2 = SSIM_K[0] ** 2, SSIM_K[1] ** 2
    top = (2.0 * mx * my + c1) * (2.0 * sxy + c2)
    bottom = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float((top / bottom).mean())


def contribution(
    render: Callable[[int, Any], Any],
    cameras: int,
    members: Sequence[Any],
    *,
    skip: npt.ArrayLike | None = None,
    ssim_fn: Callable[[Any, Any], float] = ssim,
) -> F64:
    """(cameras, blocks): 1 - SSIM of each camera's render without the block's gaussians
    (`render(c, members[b])`, handed that very object) against its full render
    (`render(c, None)`, at most once a camera). 0, unrendered, where `skip` is set or a
    block has no gaussians."""
    blocks = len(members)
    loss = np.zeros((cameras, blocks), dtype=np.float64)
    skipped = (
        np.zeros((cameras, blocks), dtype=bool) if skip is None else np.asarray(skip, dtype=bool)
    )
    present = [bool(np.asarray(member).any()) for member in members]
    for c in range(cameras):
        full: Any = None
        for b in range(blocks):
            if skipped[c, b] or not present[b]:
                continue
            if full is None:
                full = render(c, None)
            loss[c, b] = 1.0 - float(ssim_fn(render(c, members[b]), full))
    return loss


def assign(inside: npt.ArrayLike, loss: npt.ArrayLike | None, epsilon: float) -> B:
    """The cameras each block trains on: those standing in it, or whose view it changes
    by more than `epsilon`."""
    stands = np.array(inside, dtype=bool, copy=True)
    if loss is None:
        return stands
    out: B = stands | (np.asarray(loss, dtype=np.float64) > epsilon)
    return out


def _adjacent(part: Partition, a: Sequence[int], b: Sequence[int]) -> bool:
    for ca in a:
        ua, va = divmod(int(ca), part.rows)
        for cb in b:
            ub, vb = divmod(int(cb), part.rows)
            if abs(ua - ub) + abs(va - vb) == 1:
                return True
    return False


def merge_small(
    part: Partition,
    assigned: npt.ArrayLike,
    train: npt.ArrayLike,
    gaussians: Sequence[int],
    minimum: int = MIN_IMAGES,
) -> tuple[Partition, B, list[tuple[int, int, int]]]:
    """Blocks with fewer than `minimum` training cameras join a neighbour, smallest first,
    each into the edge-adjacent block with the fewest gaussians. Returns the partition with
    the merged blocks (ordered by their smallest cell), its camera matrix and the merges
    as (into, absorbed, its training cameras) in the indices of the list at that step."""
    cells = [list(c) for c in part.blocks]
    columns = [np.asarray(col, dtype=bool) for col in np.asarray(assigned, dtype=bool).T]
    mass = [int(g) for g in gaussians]
    is_train = np.asarray(train, dtype=bool)
    merges: list[tuple[int, int, int]] = []
    while len(cells) > 1:
        cams = [int((col & is_train).sum()) for col in columns]
        s = min(range(len(cells)), key=lambda b: (cams[b], b))
        if cams[s] >= minimum:
            break
        candidates = [
            o for o in range(len(cells)) if o != s and _adjacent(part, cells[o], cells[s])
        ]
        if not candidates:
            candidates = [o for o in range(len(cells)) if o != s]
        o = min(candidates, key=lambda b: (mass[b], b))
        merges.append((o, s, cams[s]))
        cells[o] = sorted(set(cells[o]) | set(cells[s]))
        columns[o] = columns[o] | columns[s]
        mass[o] += mass[s]
        del cells[s], columns[s], mass[s]
    order = sorted(range(len(cells)), key=lambda b: min(cells[b]))
    matrix: B = (
        np.stack([columns[b] for b in order], axis=1)
        if columns
        else np.zeros((np.asarray(assigned).shape[0], 0), dtype=bool)
    )
    merged = Partition(
        origin=part.origin,
        axis_u=part.axis_u,
        axis_v=part.axis_v,
        u_edges=part.u_edges,
        v_edges=part.v_edges,
        blocks=tuple(tuple(cells[b]) for b in order),
    )
    return merged, matrix, merges


# --- the frozen ring and the blend ----------------------------------------------------


def ring(part: Partition, uv: npt.ArrayLike, block: int, *, margin: float, outer: float) -> B:
    """The frozen ring: points within `outer` block sizes of the block's centre (block-
    normalised L-infinity distance) that are not in its trainable region."""
    coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    lo_u, hi_u, lo_v, hi_v = part.rect(block)
    mid_u, mid_v = part.centre(block)
    # max(|du| / size_u, |dv| / size_v) <= outer, written without the division so that a
    # block of zero width along an axis is a limit, not a NaN.
    reach_u = outer * (hi_u - lo_u)
    reach_v = outer * (hi_v - lo_v)
    in_square = (np.abs(coords[:, 0] - mid_u) <= reach_u) & (
        np.abs(coords[:, 1] - mid_v) <= reach_v
    )
    frozen: B = in_square & ~part.in_expanded(coords, block, margin)
    return frozen


def _cell_weights(edges: Sequence[float], x: F64, blend: float, margin: float) -> F64:
    """(n, cells) weights of one axis: S_j - S_{j+1}, S_j the ramp across edge j."""
    e = np.asarray(edges, dtype=np.float64)
    cells = e.size - 1
    # Read a non-finite coordinate as 0, as `Partition.cell_of` does.
    at = np.where(np.isfinite(x), np.asarray(x, dtype=np.float64), 0.0)
    # past[:, j] = S_j(x), how far x has crossed edge j. The outer edges are open: every
    # point is past edge 0 and none past the last, so the columns telescope to exactly 1.
    past = np.zeros((at.size, cells + 1), dtype=np.float64)
    past[:, 0] = 1.0
    for j in range(1, cells):
        narrower = min(e[j] - e[j - 1], e[j + 1] - e[j])
        spacing = 0.5 * (e[j + 1] - e[j - 1])
        half = min(blend * spacing, margin * narrower, 0.5 * narrower)
        if half > 0.0:
            foot = e[j] - half
            past[:, j] = np.clip((at - foot) / (2.0 * half), 0.0, 1.0)
        else:
            past[:, j] = at >= e[j]
    share: F64 = past[:, :-1] - past[:, 1:]
    return share


def weights(
    part: Partition,
    uv: npt.ArrayLike,
    block: int,
    *,
    blend: float = DEFAULT_BLEND,
    margin: float = DEFAULT_MARGIN,
) -> F64:
    """Each point's weight in `block`, in [0, 1]: the sum over its cells of the product of
    the per-axis cell weights. Summed over every block it is exactly 1."""
    coords = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    wu = _cell_weights(part.u_edges, coords[:, 0], blend, margin)
    wv = _cell_weights(part.v_edges, coords[:, 1], blend, margin)
    total = np.zeros(coords.shape[0], dtype=np.float64)
    for cell in part.blocks[block]:
        iu, iv = divmod(int(cell), part.rows)
        total += wu[:, iu] * wv[:, iv]
    out: F64 = np.clip(total, 0.0, 1.0)
    return out


def blended_opacity(logits: npt.ArrayLike, weight: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Opacity logits with alpha scaled by `weight` (clipped to (1e-7, 1 - 1e-7)); where
    the weight is at least 1 the input logit, bit for bit."""
    raw = np.asarray(logits, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float64)
    with np.errstate(over="ignore"):
        alpha = 1.0 / (1.0 + np.exp(-raw.astype(np.float64)))
    faded = np.clip(alpha * w, 1e-7, 1.0 - 1e-7)
    out = np.log(faded / (1.0 - faded)).astype(np.float32)
    full = w >= 1.0
    out[full] = raw[full]
    return out


# --- a block's held-out frames --------------------------------------------------------


def val_order(train: Sequence[str], val: Sequence[str]) -> tuple[list[tuple[str, bool]], int]:
    """The frames of a block in the order its dataset renames them (`<position>_<name>`),
    each with whether it is held out, and the `test_every` that holds out exactly the val
    ones: gsplat 1.5.3's parser sorts the names and holds out index % every == 0."""
    if not val:
        return [(name, index == 0) for index, name in enumerate(train)], max(2, len(train) + 1)
    for every in range(2, len(train) + 3):
        order: list[tuple[str, bool]] = []
        t = v = 0
        failed = False
        position = 0
        while t < len(train):
            if position % every == 0:
                if v >= len(val):
                    failed = True
                    break
                order.append((val[v], True))
                v += 1
            else:
                order.append((train[t], False))
                t += 1
            position += 1
        if not failed:
            return order, every
    raise AssertionError("val_order: no hold-out spacing fits, which cannot happen")
