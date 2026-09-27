"""Block training's geometry: the partition, the camera rule, the frozen ring, the blend.

Plain numpy and nothing newer than CPython 3.10, because two interpreters import it -- the
pipeline's (3.12), which plans and merges the blocks (`blocks.py`), and the trainer's
(gsplat's CUDA wheels are 3.10 only), where `block_views.py` renders the prior to decide
which cameras each block needs. The same arrangement as `holdout_maths.py`.

Every rule here is taken from a primary source's code, read at its tag, and adapted only
where our data differs; each function says which and how.

* **Partition** (`partition`): CityGaussian V2's `build_partition_coordinates`
  (`internal/utils/citygs_partitioning_utils.py`): the grid's edges are *per-axis
  quantiles* of the coarse model's gaussian positions -- `torch.quantile(points[:, 0],
  linspace(0, 1, block_dim[0] + 1))`, the same for y -- so every column and every row
  holds the same number of gaussians. In the ground plane, after rotating the scene so the
  cameras' mean up is +z (their `partition_citygs.py --reorient`: `up = -mean(c2w[:, :3,
  1])`, which is `sfm.camera_up`). Two differences, both because our captures are phone
  walks and orbits rather than aerial blocks: the two horizontal axes are the principal
  axes of the gaussians' footprint (so a long walk is cut across its length, not across a
  diagonal), and the outer bounds are the 1st and 99th percentiles rather than the min and
  max, because a phone capture's coarse model has floaters and far background that would
  otherwise stretch the outer blocks' extents (which set their margin, ring and centre).
  The outer edges are *open* for membership: every gaussian and every camera belongs to
  exactly one block.
* **Cameras** (`assign`): CityGS V1 `data_partition.py` / V2
  `projection_based_partition_assignment`: a camera belongs to a block if its centre lies
  in it, or if rendering the coarse model with the block's gaussians' opacity zeroed
  changes its render by `1 - SSIM > epsilon`. Their `ssim` is the standard 11x11,
  sigma 1.5 gaussian window with zero padding (`utils/loss_utils.py`), which `ssim` here
  reproduces.
* **Minimum views** (`merge_small`): V2's block dataparser asserts `len(images) > 50`
  ("Easy to overfit"); a block with fewer is merged into its grid neighbour.
* **Margin**: VastGaussian §3.2 expands each cell by 20% for data selection --
  `(l_h + 0.2 l_h) x (l_w + 0.2 l_w)`, 10% a side -- which is `DEFAULT_MARGIN`.
* **Frozen ring** (`ring`): Hierarchical 3DGS `scene/gaussian_model.py`
  (`create_from_pcd`, the scaffold): a chunk loads the coarse gaussians whose Chebyshev
  distance from its centre, `max(|dx|, |dy|)`, is between 0.5 and 1.5 chunk sizes, and
  excludes them from densification. Here the inner bound is the trainable region's edge
  (the block plus its margin), so no surface is both trainable and frozen.
* **Blend** (`weights`): H3DGS `hierarchy_explicit_loader.cpp::getWeight`: a gaussian of
  chunk `c` keeps weight 1 while `dist(c) <= 0.95 d_other`, 0 past `1.05 d_other`, linear
  between, and its opacity is multiplied by it. On H3DGS's uniform grid that band is the
  shared edge +- `falloff * D / 4` (D the distance between the two centres: `x = f D /
  (2 (2 + f))`, 0.0122 D at f = 0.05). A quantile grid is not uniform, and there the
  bisector between centres is *not* the shared edge -- it can lie inside the neighbour's
  core, where this block trained nothing. So the ramp is applied per axis at the grid's
  own edges, with that same half-width, and a block's weight is the sum over its cells of
  the product of the two axes' ramps. That is a partition of unity: the weights of all
  blocks sum to exactly 1 everywhere, so the band never doubles the density. `falloff 0`
  is the hard crop CityGS and VastGS use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: VastGaussian §3.2: each cell expanded by 20% (10% a side) for its training data.
DEFAULT_MARGIN = 0.1
#: H3DGS scaffold: the frozen ring reaches 1.5 block sizes from the block's centre.
DEFAULT_RING_OUTER = 1.5
#: H3DGS getWeight's `falloff`: the +-5% band of the distance ratio.
DEFAULT_FALLOFF = 0.05
#: CityGS's `1 - SSIM` threshold. Their released configs use 0.05-0.12 (V1: Sci-Art and
#: MatrixCity 0.05, Residence 0.08, Building 0.1, Rubble 0.12; V2: 0.05 for every aerial
#: scene and 0.01 for MatrixCity's street views, with 0.08 as the code default). The
#: lowest non-street value: a lower threshold gives a block more cameras, and CityGS
#: Tab. 4 shows the camera test is the largest single factor (25.77 dB with it, 23.43
#: without), while an extra camera only costs training time -- the frozen ring already
#: renders what it sees outside the block. Ours are ground-level phone captures, closer to
#: their street case than to aerial; calibrating on our captures is the plan's open
#: question, and `block_epsilon` is the knob.
DEFAULT_EPSILON = 0.05
#: CityGS V2 `colmap_block_dataparser.py`: `assert len(new_images) > 50`.
MIN_IMAGES = 50
#: The outer bounds, as quantiles (see the module docstring).
OUTER_QUANTILES = (0.01, 0.99)

#: CityGS `utils/loss_utils.py::ssim`: an 11-tap gaussian of sigma 1.5, and the standard
#: constants for images in [0, 1].
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
_C1 = 0.01**2
_C2 = 0.03**2


# --- the partition ----------------------------------------------------------------------


def horizontal_axes(up: Sequence[float], xyz: Any) -> tuple[Any, Any]:
    """Two orthonormal axes perpendicular to `up`: the principal axis of the positions'
    footprint in the ground plane first, then the one across it."""
    u = np.asarray(up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    # Any vector not parallel to up, then Gram-Schmidt.
    seed = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return e1, e2
    flat = np.stack([points @ e1, points @ e2], axis=1)
    flat -= flat.mean(axis=0)
    values, vectors = np.linalg.eigh(flat.T @ flat)
    major = vectors[:, int(np.argmax(values))]
    # A deterministic sign, so the same scene always numbers its blocks the same way.
    if major[0] < 0 or (major[0] == 0 and major[1] < 0):
        major = -major
    a1 = major[0] * e1 + major[1] * e2
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(u, a1)
    return a1, a2


def grid_shape(count: int, extent_u: float, extent_v: float, *, exact: bool) -> tuple[int, int]:
    """Columns along the long axis and rows across it, for `count` blocks.

    CityGS fixes `block_dim` by hand per scene (2x2 to 6x6). Here it follows the count:
    the grid whose cells are closest to square, with `exact` meaning `m * k == count` (a
    forced block count is honoured as given) and otherwise the smallest `m * k >= count`
    that is, penalised by the extra blocks (a count from the budget is a minimum).
    """
    count = max(1, int(count))
    lu = max(float(extent_u), 1e-12)
    lv = max(float(extent_v), 1e-12)
    # Ties (square extents) go to more columns: cut across the principal axis first.
    best: tuple[float, int, int] | None = None
    shape = (count, 1)
    for m in range(1, count + 1):
        k = count // m if exact else math.ceil(count / m)
        if k < 1 or (exact and m * k != count):
            continue
        aspect = abs(math.log((lu / m) / (lv / k)))
        key = (round(aspect + math.log(m * k / count), 12), m * k, -m)
        if best is None or key < best:
            best, shape = key, (m, k)
    return shape


@dataclass(frozen=True)
class Partition:
    """A grid over the ground plane, and which cells each block is.

    `u_edges` / `v_edges` are the grid's edges including the (finite, robust) outer ones;
    membership treats the outer edges as open. A cell is `iu * rows + iv`. `blocks` are
    tuples of cells: one each, until `merge_small` joins a block too thin in cameras to
    its neighbour.
    """

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

    def project(self, xyz: Any) -> Any:
        """(n, 3) world positions as (n, 2) ground-plane coordinates."""
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - np.asarray(self.origin)
        return np.stack([points @ np.asarray(self.axis_u), points @ np.asarray(self.axis_v)], 1)

    def cell_of(self, uv: Any) -> Any:
        """Each point's cell, the outer edges open (non-finite points go to cell 0)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        u = np.nan_to_num(uv[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(uv[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
        iu = np.searchsorted(np.asarray(self.u_edges[1:-1]), u, side="right")
        iv = np.searchsorted(np.asarray(self.v_edges[1:-1]), v, side="right")
        return iu * self.rows + iv

    def block_of_cell(self) -> Any:
        table = np.full(self.columns * self.rows, -1, dtype=np.int64)
        for index, cells in enumerate(self.blocks):
            table[list(cells)] = index
        return table

    def block_of(self, uv: Any) -> Any:
        """Each point's block (by its centre; outer edges open)."""
        return self.block_of_cell()[self.cell_of(uv)]

    def cell_rect(self, cell: int) -> tuple[float, float, float, float]:
        iu, iv = divmod(int(cell), self.rows)
        return (self.u_edges[iu], self.u_edges[iu + 1], self.v_edges[iv], self.v_edges[iv + 1])

    def rect(self, block: int) -> tuple[float, float, float, float]:
        """The block's finite bounding rectangle, (u0, u1, v0, v1)."""
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

    def in_expanded(self, uv: Any, block: int, margin: float) -> Any:
        """In the block grown by `margin` of each cell's size on each side (VastGS's 10%),
        the grid's outer edges staying open."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        inside = np.zeros(uv.shape[0], dtype=bool)
        for cell in self.blocks[block]:
            iu, iv = divmod(int(cell), self.rows)
            u0, u1, v0, v1 = self.cell_rect(cell)
            du, dv = margin * (u1 - u0), margin * (v1 - v0)
            lo_u = -np.inf if iu == 0 else u0 - du
            hi_u = np.inf if iu == self.columns - 1 else u1 + du
            lo_v = -np.inf if iv == 0 else v0 - dv
            hi_v = np.inf if iv == self.rows - 1 else v1 + dv
            inside |= (
                (uv[:, 0] >= lo_u) & (uv[:, 0] <= hi_u) & (uv[:, 1] >= lo_v) & (uv[:, 1] <= hi_v)
            )
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
        return Partition(
            origin=_three(document["origin"]),
            axis_u=_three(document["axisU"]),
            axis_v=_three(document["axisV"]),
            u_edges=tuple(float(v) for v in document["uEdges"]),
            v_edges=tuple(float(v) for v in document["vEdges"]),
            blocks=tuple(tuple(int(c) for c in cells) for cells in document["blocks"]),
        )


def _three(values: Any) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _edges(values: Any, count: int) -> tuple[float, ...]:
    """`count` cells' edges: inner ones at equal-count quantiles (CityGS V2), outer ones at
    `OUTER_QUANTILES`, made non-decreasing so a degenerate axis still yields a grid."""
    inner = [float(q) for q in np.quantile(values, np.linspace(0.0, 1.0, count + 1)[1:-1])]
    low, high = (float(q) for q in np.quantile(values, OUTER_QUANTILES))
    edges = [min(low, *inner) if inner else low, *inner, max(high, *inner) if inner else high]
    for index in range(1, len(edges)):
        edges[index] = max(edges[index], edges[index - 1])
    if edges[-1] == edges[0]:
        edges[-1] = edges[0] + 1e-9
    return tuple(edges)


def partition(xyz: Any, up: Sequence[float], count: int, *, exact: bool) -> Partition:
    """Split `xyz` (the prior's gaussian centres in the region to train) into `count`
    blocks of equal gaussian counts per column and per row. Deterministic."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        raise ValueError("no finite positions to partition")
    # Rounded, so that the same points in another order (a sum's rounding differs in the
    # last bit) give the same partition, byte for byte.
    a1, a2 = (np.round(axis, 9) for axis in horizontal_axes(up, points))
    origin = np.median(points, axis=0)
    rel = points - origin
    u, v = rel @ a1, rel @ a2
    lo_u, hi_u = np.quantile(u, OUTER_QUANTILES)
    lo_v, hi_v = np.quantile(v, OUTER_QUANTILES)
    columns, rows = grid_shape(count, float(hi_u - lo_u), float(hi_v - lo_v), exact=exact)
    return Partition(
        origin=_three(origin),
        axis_u=_three(a1),
        axis_v=_three(a2),
        u_edges=_edges(u, columns),
        v_edges=_edges(v, rows),
        blocks=tuple((cell,) for cell in range(columns * rows)),
    )


# --- which cameras each block trains on -------------------------------------------------


def cameras_inside(part: Partition, centres: Any) -> Any:
    """(cameras, blocks): the block each camera centre stands in (outer edges open)."""
    block = part.block_of(part.project(centres))
    out = np.zeros((block.shape[0], part.count), dtype=bool)
    out[np.arange(block.shape[0]), block] = True
    return out


def assign(inside: Any, loss: Any | None, epsilon: float) -> Any:
    """CityGS's union: inside the block, or `1 - SSIM > epsilon` without it."""
    assigned = np.asarray(inside, dtype=bool).copy()
    if loss is not None:
        assigned |= np.asarray(loss, dtype=np.float64) > epsilon
    return assigned


def _gaussian_window(size: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA) -> Any:
    x = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def _filter(image: Any, window: Any) -> Any:
    """Separable 2D filter with zero padding ('same'), per channel: (H, W, C)."""
    pad = window.shape[0] // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)))
    rows = sum(w * padded[i : i + image.shape[0], :, :] for i, w in enumerate(window))
    return sum(w * rows[:, i : i + image.shape[1], :] for i, w in enumerate(window))


def ssim(a: Any, b: Any) -> float:
    """Mean SSIM of two (H, W, 3) images in [0, 1], as CityGS's `ssim` computes it."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    window = _gaussian_window()
    mu_x, mu_y = _filter(x, window), _filter(y, window)
    sxx = _filter(x * x, window) - mu_x**2
    syy = _filter(y * y, window) - mu_y**2
    sxy = _filter(x * y, window) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + _C1) * (2 * sxy + _C2)) / (
        (mu_x**2 + mu_y**2 + _C1) * (sxx + syy + _C2)
    )
    return float(value.mean())


def contribution(
    render: Callable[[int, Any | None], Any],
    cameras: int,
    members: Sequence[Any],
    *,
    skip: Any | None = None,
    ssim_fn: Callable[[Any, Any], float] = ssim,
) -> Any:
    """(cameras, blocks) of `1 - SSIM(render without block, render with it)`.

    `render(camera, drop)` renders the prior from `camera`, with the gaussians where
    `drop` is true made transparent (None: all of them drawn). `members[b]` is block `b`'s
    gaussians. `skip[c, b]` true means the answer is already known (the camera stands in
    the block, or none of the block's gaussians are in its view) and 0 is recorded without
    rendering -- CityGS V2 likewise renders only for cameras not assigned by position.
    """
    loss = np.zeros((cameras, len(members)), dtype=np.float64)
    for camera in range(cameras):
        full = None
        for block, member in enumerate(members):
            if skip is not None and bool(skip[camera, block]):
                continue
            if not np.any(member):
                continue
            if full is None:
                full = render(camera, None)
            loss[camera, block] = 1.0 - ssim_fn(render(camera, member), full)
    return loss


def merge_small(
    part: Partition,
    assigned: Any,
    train: Any,
    gaussians: Sequence[int],
    minimum: int = MIN_IMAGES,
) -> tuple[Partition, Any, list[tuple[int, int, int]]]:
    """Merge blocks with fewer than `minimum` training cameras into a grid neighbour.

    Repeatedly: the block with the fewest training cameras, if under `minimum`, joins
    the edge-adjacent block holding the fewest gaussians (the least likely to push the
    merged block over the GPU); its cameras are the union of both. Returns the new
    partition, the new assignment, and each merge as (kept, absorbed, cameras before).
    The union is a subset of what the SSIM test would give the merged block (removing
    both blocks changes a render at least as much as removing either), so it is never
    more cameras than a re-render would pick.
    """
    blocks = [tuple(cells) for cells in part.blocks]
    columns = [np.asarray(assigned[:, b], dtype=bool) for b in range(len(blocks))]
    counts = list(int(g) for g in gaussians)
    train = np.asarray(train, dtype=bool)
    merges: list[tuple[int, int, int]] = []
    while len(blocks) > 1:
        cams = [int((column & train).sum()) for column in columns]
        small = min(range(len(blocks)), key=lambda b: (cams[b], b))
        if cams[small] >= minimum:
            break
        neighbours = [
            b
            for b in range(len(blocks))
            if b != small and _adjacent(part, blocks[small], blocks[b])
        ]
        if not neighbours:
            neighbours = [b for b in range(len(blocks)) if b != small]
        other = min(neighbours, key=lambda b: (counts[b], b))
        merges.append((other, small, cams[small]))
        blocks[other] = tuple(sorted(blocks[other] + blocks[small]))
        columns[other] = columns[other] | columns[small]
        counts[other] += counts[small]
        del blocks[small], columns[small], counts[small]
    order = sorted(range(len(blocks)), key=lambda b: min(blocks[b]))
    merged = Partition(
        origin=part.origin,
        axis_u=part.axis_u,
        axis_v=part.axis_v,
        u_edges=part.u_edges,
        v_edges=part.v_edges,
        blocks=tuple(blocks[b] for b in order),
    )
    matrix = np.stack([columns[b] for b in order], axis=1) if blocks else assigned
    return merged, matrix, merges


def _adjacent(part: Partition, a: Sequence[int], b: Sequence[int]) -> bool:
    for cell in a:
        iu, iv = divmod(int(cell), part.rows)
        for other in b:
            ju, jv = divmod(int(other), part.rows)
            if abs(iu - ju) + abs(iv - jv) == 1:
                return True
    return False


# --- the frozen ring and the blend ------------------------------------------------------


def ring(part: Partition, uv: Any, block: int, *, margin: float, outer: float) -> Any:
    """The prior's gaussians to freeze round `block`: within `outer` block sizes of its
    centre (Chebyshev, per axis, as H3DGS's scaffold) and outside its trainable region."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    u0, u1, v0, v1 = part.rect(block)
    cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    wu, wv = max(u1 - u0, 1e-12), max(v1 - v0, 1e-12)
    chebyshev = np.maximum(np.abs(uv[:, 0] - cu) / wu, np.abs(uv[:, 1] - cv) / wv)
    return (chebyshev <= outer) & ~part.in_expanded(uv, block, margin)


def _axis_weights(edge_values: Sequence[float], values: Any, falloff: float) -> Any:
    """(n, cells) one axis's partition of unity: ramps of half-width `falloff * D / 4` at
    each interior edge (D the distance between the two cells' centres), open outside."""
    edges = np.asarray(edge_values, dtype=np.float64)
    cells = edges.shape[0] - 1
    x = np.asarray(values, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    steps = np.ones((x.shape[0], cells + 1), dtype=np.float64)
    steps[:, cells] = 0.0
    centres = 0.5 * (edges[:-1] + edges[1:])
    for k in range(1, cells):
        half = falloff * (centres[k] - centres[k - 1]) / 4.0
        if half <= 0.0:
            steps[:, k] = (x >= edges[k]).astype(np.float64)
        else:
            # 1 to the right of the edge (inside cell k), 0 to the left, linear across.
            steps[:, k] = np.clip(0.5 + (x - edges[k]) / (2.0 * half), 0.0, 1.0)
    # weight of cell i = (right of edge i) - (right of edge i + 1).
    return steps[:, :-1] - steps[:, 1:]


def weights(part: Partition, uv: Any, block: int, *, falloff: float = DEFAULT_FALLOFF) -> Any:
    """Each point's weight in `block`: the sum over its cells of the product of the two
    axes' ramps (see the module docstring). 1 deep inside, 0 outside past the band."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    wu = _axis_weights(part.u_edges, uv[:, 0], falloff)
    wv = _axis_weights(part.v_edges, uv[:, 1], falloff)
    total = np.zeros(uv.shape[0], dtype=np.float64)
    for cell in part.blocks[block]:
        iu, iv = divmod(int(cell), part.rows)
        total += wu[:, iu] * wv[:, iv]
    return np.clip(total, 0.0, 1.0)


def blended_opacity(logits: Any, weight: Any) -> Any:
    """`alpha' = alpha * weight`, in the PLY's logit space (getWeight's `opacity * w`)."""
    logits = np.asarray(logits, dtype=np.float64)
    with np.errstate(over="ignore"):
        alpha = 1.0 / (1.0 + np.exp(-logits))
    scaled = np.clip(alpha * np.asarray(weight, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    out = np.log(scaled / (1.0 - scaled))
    # A weight of exactly 1 leaves the stored value as it was, bit for bit.
    return np.where(np.asarray(weight) >= 1.0, logits, out).astype(np.float32)


def val_order(train: Sequence[str], val: Sequence[str]) -> tuple[list[tuple[str, bool]], int]:
    """Names in the order a block's dataset lists them, and the `test_every` that makes
    gsplat's parser hold out exactly the val ones.

    v1.5.3's parser sorts image names and holds out `index % test_every == 0`. A block's
    frames are renamed `<position>_<name>` in this order, so the block trains on its
    training frames and is evaluated on the global held-out frames it was assigned --
    never on a frame another block trained on. The smallest `test_every` whose slots
    the val frames fill, so as many of them as possible are used; any left over are not
    in the block's dataset at all. Index 0 is always held out, so a block with no val
    frame gives up its first training frame to it.
    """
    train, val = list(train), list(val)
    if not val:
        return [(name, index == 0) for index, name in enumerate(train)], max(2, len(train) + 1)
    for every in range(2, len(train) + 3):
        order: list[tuple[str, bool]] = []
        t = v = 0
        ok = True
        while t < len(train):
            if len(order) % every == 0:
                if v >= len(val):
                    ok = False
                    break
                order.append((val[v], True))
                v += 1
            else:
                order.append((train[t], False))
                t += 1
        if ok:
            return order, every
    raise AssertionError("unreachable: every = len(train) + 1 always fits one val frame")
