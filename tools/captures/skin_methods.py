"""Skin methods and handle policies for the motion-skins bake-off (docs/SCENE_OBJECTS.md §9).

Every method here writes the same contract as `skin_scene.py` (hexapod.skin v1: per splat,
signed int8 weights on `m` handles, handle 0 the constant field), so the viewer, the wind
(`skinWind.ts`) and the poke driver (`skinPoke.ts`) need nothing per method. What differs is
how the weight fields are found:

* **freeform** -- `skin_scene.fit_skin`: FreeForm/RKPM skinning eigenmodes (NVIDIA Kaolin,
  Apache-2.0, vendored in `kaolin_rkpm.py`) integrated over the splat centres. The splats are
  surface samples, so the object behaves as a shell, and its modes are free: the wind keeps
  only the combinations that leave its base still.
* **pinned** -- the same basis with the object's base held (`kaolin_rkpm` `pinned`): a penalty
  on the lowest tenth of its height (the band the wind anchors, `skin_scene.anchor_mask`), so
  every learned mode is a cantilever mode that leaves the base still by itself. A rooted
  plant keeps all its handles under the wind's anchor instead of the few combinations of free
  modes that happen to vanish there (the yard's 1 m shrub: 2 of 8).
* **tetfem** -- linear FEM on a tetrahedral mesh of the object's **filled occupancy**: the
  splat centres voxelised (a voxel `TET_SPACINGS` splat spacings), grown by a voxel, holes
  filled (the inside the capture never saw is taken as solid), the largest face-connected
  piece cut into six tetrahedra a voxel (Kuhn), P1 elements, and the same skinning
  eigenproblem `H c = λ M c` (uniform material, `H` the Laplacian scaled by `λ + 4μ` as
  Kaolin's RKPM path assembles it) solved over the volume with the nodes of the base band
  held (Dirichlet), as `pinned` holds the splats there. Splats take their weights
  barycentrically from the tetrahedron they sit in (one outside the mesh from the nearest
  voxel of it). The mass and the anchor (`dynamics`) are integrated over the volume, not the
  surface: a solid object's inside carries its mass. Measured on the synthetic tree: free
  modes on this mesh kept no wind direction for the yard's shrubs (every mix moved their
  base), and a mesh closed back to the occupancy (dilate, fill, erode) left twigs a voxel
  thin that hinged (kNN stretch p99 1.26 at 2% handles); grown and held, p99 1.02, as
  FreeForm's.
* **rigid** -- one handle, the constant: no weights at all (a skin of one handle takes no rows
  of `skin.bin`); the object moves only as a whole (`Z_0`), or not at all.

The eigenvalues of every method are on one scale -- the object mapped into the unit box
`skin_scene.fit_skin` uses, `E = 1` -- so the wind's `ω_j = c·√λ_j / scale` and a material's
`c` mean the same under each.

**Handle policies** (`size_policy`, `stiffness_policy`): how many handles an object gets.
Today's rule is size alone (`skin_scene.handle_count`, 8 to 16). The stiffness-aware rule
reads a class from the object's category, its names (top tags) or a fitted `materials.json`
stiffness, then its property scores (`stiffness_class`): rigid things get one handle, firm
compact things (a pumpkin) `FIRM_HANDLES`, plants the size rule, and big trees
`TREE_HANDLES` (two texels a splat).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

import skin_scene
from kaolin_rkpm import SimplicitsRKPM, to_lame

__all__ = [
    "FIRM_HANDLES",
    "TREE_HANDLES",
    "HandlePolicy",
    "fit_pinned",
    "fit_rigid",
    "fit_tetfem",
    "fitter",
    "size_policy",
    "stiffness_class",
    "stiffness_policy",
    "tet_mesh",
    "traits_of",
]

#: Handles of a firm compact object (the constant one included): a few squash-and-sway modes.
FIRM_HANDLES = 4
#: Handles of a big tree under the stiffness-aware rule: 31 learned weights, two texels a splat.
TREE_HANDLES = 32
#: A plant at least this tall (metres) is a big tree to the stiffness-aware rule.
TREE_HEIGHT_M = 6.0
#: The pinned basis's penalty, relative to the Hessian (kaolin_rkpm `pin_stiffness`).
PIN_STIFFNESS = 1e4
#: The volume mesh: voxels along the longest side of the object's box (bounds), and at most
#: this many filled voxels (the grid is coarsened until it fits).
TET_RESOLUTION_MIN = 12
TET_RESOLUTION_MAX = 64
TET_MAX_VOXELS = 24_000
#: ... the voxel is this many median splat spacings.
TET_SPACINGS = 4.0
#: Voxels the occupancy is grown by before its holes are filled: what joins the splat centres
#: into one solid (kept, not eroded back: a twig a voxel thin would hinge the mesh).
TET_GROW = 1

# ------------------------------------------------------------------------------- policies

#: Categories (segment_scene's broad categories, data/categories.json) by stiffness class.
PLANT_CATEGORIES = frozenset({"trees", "shrubs", "grass", "flowers"})
FIRM_CATEGORIES = frozenset({"produce", "animals", "clothing"})
RIGID_CATEGORIES = frozenset(
    {"fixtures", "furniture", "vehicles", "equipment", "buildings", "walls", "rock", "wood"}
)
#: Words in an object's top names, by class (checked after the category).
PLANT_WORDS = (
    "tree", "bush", "shrub", "plant", "fern", "grass", "conifer", "pine", "hedge", "flower",
    "leaves", "branch", "vine", "palm", "reed",
)  # fmt: skip
FIRM_WORDS = (
    "pumpkin", "gourd", "squash", "melon", "fruit", "vegetable", "cushion", "pillow", "bag",
    "hay", "ball",
)  # fmt: skip
RIGID_WORDS = (
    "rock", "stone", "boulder", "spool", "reel", "table", "bench", "car", "truck", "barrel",
    "crate", "box", "pole", "post", "sign", "wall", "fence", "machine", "manhole", "cistern",
    "vehicle", "chair",
)  # fmt: skip
#: A fitted wave speed (materials.json `stiffness`, m/s) up to these is a plant, firm; above,
#: rigid. The property prior gives 3.5 for foliage and 14 for the stiffest.
PLANT_MAX_WAVE_SPEED = 7.0
FIRM_MAX_WAVE_SPEED = 20.0


def stiffness_class(instance: Mapping, material: Mapping | None = None) -> str:
    """`rigid`, `firm` or `plant` (a plant `TREE_HEIGHT_M` tall or more: `tree`).

    In order: a `materials.json` record's fitted `stiffness`; the instance's category; its top
    three names; its property scores (vegetation over a half is a plant, rigid over 0.8 with
    little vegetation or elasticity is rigid, anything else firm).
    """
    height = _height(instance)
    plant = "tree" if height >= TREE_HEIGHT_M else "plant"
    c = (material or {}).get("stiffness")
    if isinstance(c, (int, float)) and math.isfinite(c) and c > 0:
        return (
            plant if c <= PLANT_MAX_WAVE_SPEED else "firm" if c <= FIRM_MAX_WAVE_SPEED else "rigid"
        )
    category = instance.get("category") or ""
    if category in PLANT_CATEGORIES:
        return plant
    if category in FIRM_CATEGORIES:
        return "firm"
    if category in RIGID_CATEGORIES:
        return "rigid"
    names = " ".join(str(t.get("label", "")) for t in (instance.get("tags") or [])[:3]).lower()
    for words, found in ((PLANT_WORDS, plant), (FIRM_WORDS, "firm"), (RIGID_WORDS, "rigid")):
        if any(w in names for w in words):
            return found
    p = instance.get("properties") or {}
    vegetation = float(p.get("vegetation", 0) or 0)
    rigid = float(p.get("rigid", 0) or 0)
    elastic = float(p.get("elastic", 0) or 0)
    if vegetation >= 0.5:
        return plant
    if rigid >= 0.8 and max(vegetation, elastic) < 0.3:
        return "rigid"
    return "firm"


def _height(instance: Mapping) -> float:
    b = instance.get("bounds") or {}
    try:
        return float(b["max"][2]) - float(b["min"][2])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class HandlePolicy:
    """How many handles each object gets: `handles(instance, points)` (the constant one
    included), and the class it records (`skin.json`'s `class`, or None)."""

    name: str
    about: str
    choose: Callable[[Mapping, np.ndarray], tuple[int, str | None]]

    def handles(self, instance: Mapping, points: np.ndarray) -> tuple[int, str | None]:
        return self.choose(instance, points)


def _diagonal(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(0) - points.min(0)))


def size_policy() -> HandlePolicy:
    """Today's rule: `m = clamp(round(8 + 2·log2(d / 2 m)), 8, 16)`, size alone."""
    return HandlePolicy(
        "size",
        "handles by size alone: 8 at 2 m, 12 at 8 m, 16 from 32 m",
        lambda instance, points: (skin_scene.handle_count(_diagonal(points)), None),
    )


def stiffness_policy(
    materials: Mapping[int, Mapping] | None = None, *, wide: bool = True
) -> HandlePolicy:
    """The stiffness-aware rule: rigid 1, firm `FIRM_HANDLES`, plants by size, big trees
    `TREE_HANDLES` (16 when the viewer cannot carry wide rows: `wide` False)."""
    tree = TREE_HANDLES if wide else skin_scene.MAX_HANDLES

    def choose(instance: Mapping, points: np.ndarray) -> tuple[int, str]:
        cls = stiffness_class(instance, (materials or {}).get(int(instance.get("id", 0))))
        if cls == "rigid":
            return 1, cls
        if cls == "firm":
            return FIRM_HANDLES, cls
        if cls == "tree":
            return tree, cls
        return skin_scene.handle_count(_diagonal(points)), cls

    return HandlePolicy(
        "stiffness",
        f"rigid things 1 handle (no weights), firm compact things {FIRM_HANDLES}, plants by "
        f"size, trees {tree}",
        choose,
    )


# ------------------------------------------------------------------------------- common


def _frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The fit's box (skin_scene.fit_skin): its min and max corners, the origin (the bounds'
    base centre) and the half extent."""
    low, high = points.min(0), points.max(0)
    centre = (low + high) / 2
    half = float(max((high - low).max() / 2, 1e-3))
    pad = half * 1.02
    return centre - pad, centre + pad, np.array([centre[0], centre[1], low[2]]), half


def _finish(
    *,
    points: np.ndarray,
    index: int,
    instance: int,
    raw: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray] | None,
    model: SimplicitsRKPM | None,
    eigenvalues: np.ndarray,
    nodes: int,
    started: float,
    grams: tuple[np.ndarray, np.ndarray, int, float] | None = None,
    extra: dict | None = None,
    splats: int | None = None,
    frame_of: np.ndarray | None = None,
) -> skin_scene.Skin:
    """A `Skin` from raw learned weights at the fit points: scaled to `max |w| = 1` (positive),
    supports and (unless given) the splat-integrated grams, as `fit_skin` does. `frame_of`: the
    points whose bounds give the rest frame (all the object's, when `points` is its pool)."""
    _, _, origin, half = _frame(points if frame_of is None else frame_of)
    learned = raw.shape[1]
    peak = np.abs(raw).argmax(0)
    peak_value = raw[peak, np.arange(learned)]
    norm = 1.0 / np.where(np.abs(peak_value) > 0, peak_value, 1.0)
    w = np.abs(raw * norm[None, :])
    local = points - origin
    mass = w.sum(0).clip(1e-12)
    centres = (w.T @ local) / mass[:, None]
    radii = np.sqrt((w * ((local[:, None, :] - centres[None]) ** 2).sum(-1)).sum(0) / mass)
    gram, anchor_gram, anchors, band = grams or skin_scene.modal_grams(points, raw * norm[None, :])
    if grams is not None:
        # Grams integrated elsewhere are of the raw fields: bring them to the scaled ones.
        s = np.r_[1.0, norm]
        gram = gram * s[:, None] * s[None, :]
        anchor_gram = anchor_gram * s[:, None] * s[None, :]
    return skin_scene.Skin(
        index=index,
        instance=instance,
        origin=origin,
        scale=half,
        handles=learned + 1,
        nodes=nodes,
        splats=len(points) if splats is None else splats,
        model=model,  # type: ignore[arg-type]
        norm=norm,
        eigenvalues=np.asarray(eigenvalues, np.float64),
        centres=centres,
        radii=radii,
        seconds=time.perf_counter() - started,
        mass=gram,
        anchor_gram=anchor_gram,
        anchor_splats=anchors,
        anchor_band=band,
        evaluate=evaluate,
        extra=dict(extra or {}),
    )


def fit_rigid(
    points: np.ndarray, index: int, instance: int, *, seed: int = 0, extra: dict | None = None
) -> skin_scene.Skin:
    """One handle, the constant field: the object moves only as a whole. No weights."""
    started = time.perf_counter()
    points = np.asarray(points, np.float64)
    _, _, origin, half = _frame(points)
    mask, band = skin_scene.anchor_mask(points)
    return skin_scene.Skin(
        index=index,
        instance=instance,
        origin=origin,
        scale=half,
        handles=1,
        nodes=0,
        splats=len(points),
        model=None,  # type: ignore[arg-type]
        norm=np.zeros(0),
        eigenvalues=np.zeros(0),
        centres=np.zeros((0, 3)),
        radii=np.zeros(0),
        seconds=time.perf_counter() - started,
        mass=np.ones((1, 1)),
        anchor_gram=np.ones((1, 1)),
        anchor_splats=int(mask.sum()),
        anchor_band=band,
        extra=dict(extra or {}),
    )


# ------------------------------------------------------------------------------- pinned


def fit_pinned(
    points: np.ndarray,
    index: int,
    instance: int,
    *,
    seed: int = 0,
    handles: int | None = None,
    extra: dict | None = None,
) -> skin_scene.Skin:
    """FreeForm/RKPM eigenmodes with the object's base held (the anchor band): every learned
    handle leaves the base still by itself."""
    started = time.perf_counter()
    points = np.asarray(points, np.float64)
    if handles is None:
        handles = skin_scene.handle_count(_diagonal(points))
    if handles <= 1:
        return fit_rigid(points, index, instance, seed=seed, extra=extra)
    nodes = skin_scene.node_count(len(points))
    handles = max(2, min(handles, nodes))
    bb_min, bb_max, _, _ = _frame(points)
    # `handles` counts the constant handle; the pinned problem has no constant mode, so all
    # `handles - 1` eigenvectors it keeps are learned ones (num_handles - 1 of them).
    model = SimplicitsRKPM(
        handles,
        nodes,
        num_points=skin_scene.FIT_POINTS,
        bb_min=bb_min,
        bb_max=bb_max,
        seed=seed,
    )
    pinned, band = skin_scene.anchor_mask(points)
    pool = skin_scene.fit_pool(points, seed)
    held = skin_scene.fit_pool(np.c_[points, pinned], seed)[:, 3] > 0.5
    n = len(pool)
    model.init(
        pool, np.ones(n), np.full(n, skin_scene.POISSON), pinned=held,
        pin_stiffness=PIN_STIFFNESS,
    )  # fmt: skip
    raw = model.forward(model._offset_scale(pool))
    return _finish(
        points=pool,
        splats=len(points),
        frame_of=points,
        index=index,
        instance=instance,
        raw=raw,
        evaluate=None,
        model=model,
        eigenvalues=model.evals,
        nodes=model.rkpm.num_nodes if model.rkpm else nodes,
        started=started,
        extra={"pinned": {"band": round(band, 4), "splats": int(pinned.sum())}, **(extra or {})},
    )


# ------------------------------------------------------------------------------- tet FEM

#: The six tetrahedra of a cube (Kuhn): for each order of the axes, the path 000 → 111.
_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


@dataclass
class TetMesh:
    """A voxel tetrahedral mesh in the fit's unit box: node coordinates (unit box), tets (node
    indices), the voxel size (unit box), the grid, which voxels are meshed, and the node index
    of every grid corner (-1: none)."""

    nodes: np.ndarray
    tets: np.ndarray
    voxel: float
    shape: tuple[int, int, int]
    solid: np.ndarray
    corner: np.ndarray
    nearest: np.ndarray  # per voxel, the (i, j, k) of the nearest meshed voxel


def tet_mesh(
    unit: np.ndarray,
    *,
    resolution: int | None = None,
    max_voxels: int = TET_MAX_VOXELS,
    grow: int | None = None,
) -> TetMesh:
    """The filled occupancy of points `unit` (already in the unit box) as tetrahedra.

    The voxel is `TET_SPACINGS` median splat spacings (so neighbouring splats share or touch
    voxels), grown by `TET_GROW` voxels and its holes filled, between a `TET_RESOLUTION_MAX`th and a
    `TET_RESOLUTION_MIN`th of the box, and coarser while more than `max_voxels` are filled.
    `resolution` fixes the voxels along the box instead.
    """
    unit = np.asarray(unit, np.float64)
    grow = TET_GROW if grow is None else grow
    if resolution is None:
        # The median spacing, from up to 20k of the points to their nearest neighbour of all.
        query = unit[:: max(1, len(unit) // 20_000)]
        spacing = (
            float(np.median(cKDTree(unit).query(query, k=2)[0][:, 1])) if len(unit) > 1 else 0.1
        )
        res = int(
            np.clip(
                round(1.0 / max(TET_SPACINGS * spacing, 1e-9)),
                TET_RESOLUTION_MIN,
                TET_RESOLUTION_MAX,
            )
        )
    else:
        res = resolution
    while True:
        h = 1.0 / res
        shape = (res, res, res)
        idx = np.clip(np.floor(unit / h).astype(np.int64), 0, res - 1)
        occupied = np.zeros(shape, bool)
        occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        grown = ndimage.binary_dilation(occupied, iterations=grow) if grow > 0 else occupied
        solid = ndimage.binary_fill_holes(grown)
        labels, count = ndimage.label(solid)  # face-connected
        if count > 1:
            sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
            solid = labels == (1 + int(np.argmax(sizes)))
        if solid.sum() <= max_voxels or res <= 8:
            break
        res = int(res * 0.85)
    vox = np.argwhere(solid)
    # Corner grid (res + 1)^3 -> node ids for the corners meshed voxels use.
    corner = -np.ones((res + 1,) * 3, np.int64)
    offsets = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    corners = (vox[:, None, :] + offsets[None]).reshape(-1, 3)
    used = np.unique(corners, axis=0)
    corner[used[:, 0], used[:, 1], used[:, 2]] = np.arange(len(used))
    nodes = used.astype(np.float64) * h
    tets = []
    for perm in _PERMUTATIONS:
        path = [np.zeros(3, np.int64)]
        for axis in perm:
            step = path[-1].copy()
            step[axis] += 1
            path.append(step)
        at = [vox + p for p in path]
        tets.append(np.stack([corner[a[:, 0], a[:, 1], a[:, 2]] for a in at], axis=1))
    # Per voxel, the nearest meshed one (itself when meshed): where an outside splat reads.
    _, nearest = ndimage.distance_transform_edt(~solid, return_indices=True)
    return TetMesh(
        nodes=nodes,
        tets=np.concatenate(tets),
        voxel=h,
        shape=shape,
        solid=solid,
        corner=corner,
        nearest=np.moveaxis(nearest, 0, -1),
    )


def _p1_matrices(mesh: TetMesh, coeff: float) -> tuple[coo_matrix, coo_matrix, np.ndarray]:
    """P1 stiffness `∫ coeff ∇φ_a·∇φ_b` and consistent mass `∫ φ_a φ_b` over the mesh, and
    each node's lumped mass."""
    x = mesh.nodes[mesh.tets]  # (t, 4, 3)
    d = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)  # (t,3,3)
    det = np.linalg.det(d)
    vol = np.abs(det) / 6.0
    inv = np.linalg.inv(d)  # rows: ∇λ_1..3
    grads = np.concatenate([-inv.sum(1, keepdims=True), inv], axis=1)  # (t, 4, 3)
    ke = coeff * vol[:, None, None] * np.einsum("tai,tbi->tab", grads, grads)
    me = (vol / 20.0)[:, None, None] * (np.ones((4, 4)) + np.eye(4))[None]
    rows = np.repeat(mesh.tets, 4, axis=1).reshape(-1)
    cols = np.tile(mesh.tets, (1, 4)).reshape(-1)
    n = len(mesh.nodes)
    k = coo_matrix((ke.reshape(-1), (rows, cols)), shape=(n, n)).tocsc()
    m = coo_matrix((me.reshape(-1), (rows, cols)), shape=(n, n)).tocsc()
    lumped = np.asarray(m.sum(axis=1)).reshape(-1)
    return k, m, lumped


def _interpolate(mesh: TetMesh, unit: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nodal `values` (nodes, c) at points `unit` (unit box): barycentric in the Kuhn tet that
    holds each point, a point outside the mesh read at the closest point of its nearest
    meshed voxel."""
    res = mesh.shape[0]
    h = mesh.voxel
    g = np.asarray(unit, np.float64) / h
    v = np.clip(np.floor(g).astype(np.int64), 0, res - 1)
    near = mesh.nearest[v[:, 0], v[:, 1], v[:, 2]]
    f = np.clip(g - near, 0.0, 1.0)
    order = np.argsort(-f, axis=1, kind="stable")
    fs = np.take_along_axis(f, order, axis=1)
    lam = np.stack([1 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2]], axis=1)
    out = np.zeros((len(unit), values.shape[1]))
    step = np.zeros((len(unit), 3), np.int64)
    for k in range(4):
        if k > 0:
            np.put_along_axis(step, order[:, k - 1 : k], 1, axis=1)
        c = near + step
        ids = mesh.corner[c[:, 0], c[:, 1], c[:, 2]]
        out += lam[:, k : k + 1] * values[np.maximum(ids, 0)]
    return out


def fit_tetfem(
    points: np.ndarray,
    index: int,
    instance: int,
    *,
    seed: int = 0,
    handles: int | None = None,
    extra: dict | None = None,
) -> skin_scene.Skin:
    """Skinning eigenmodes by linear FEM over the object's filled occupancy (module docstring)."""
    started = time.perf_counter()
    points = np.asarray(points, np.float64)
    if handles is None:
        handles = skin_scene.handle_count(_diagonal(points))
    if handles <= 1:
        return fit_rigid(points, index, instance, seed=seed, extra=extra)
    bb_min, bb_max, _, _ = _frame(points)
    span = bb_max - bb_min
    unit = (points - bb_min) / span
    mesh = tet_mesh(unit)
    n = len(mesh.nodes)
    handles = max(2, min(handles, n - 1))
    mu, lam = to_lame(np.array(1.0), np.array(skin_scene.POISSON))
    coeff = float(lam + 4 * mu)  # kaolin_rkpm's reparameterised Lamé coefficient
    k, m, lumped = _p1_matrices(mesh, coeff)
    # The base held: the nodes in the anchor band (the wind's, skin_scene.anchor_mask) are
    # fixed, so every mode is one of the rooted object's and none is the constant field.
    _, band = skin_scene.anchor_mask(points)
    z = mesh.nodes[:, 2] * span[2] + bb_min[2]
    low = float(points[:, 2].min())
    held = z <= low + band
    if not held.any():
        held = z <= z.min() + 1e-9
    free = np.flatnonzero(~held)
    want = min(handles - 1, len(free) - 1)
    kf = k[free][:, free]
    mf = m[free][:, free]
    vals, vecs = eigsh(kf, k=want, M=mf, sigma=0.0, which="LM")
    order = np.argsort(vals)
    evals = vals[order]
    evecs = np.zeros((n, want))
    evecs[free] = vecs[:, order]

    def evaluate(positions: np.ndarray) -> np.ndarray:
        return _interpolate(mesh, (np.asarray(positions, np.float64) - bb_min) / span, evecs)

    pool = skin_scene.fit_pool(points, seed)
    raw = evaluate(pool)
    # The grams over the volume (lumped node masses), the anchor the nodes in the base band.
    w = np.c_[np.ones(n), evecs]
    gram = (w * lumped[:, None]).T @ w / lumped.sum()
    anchor = (w[held] * lumped[held, None]).T @ w[held] / lumped[held].sum()
    anchors_mask, _ = skin_scene.anchor_mask(points)
    return _finish(
        points=pool,
        splats=len(points),
        frame_of=points,
        index=index,
        instance=instance,
        raw=raw,
        evaluate=evaluate,
        model=None,
        eigenvalues=evals,
        nodes=n,
        started=started,
        grams=(gram, anchor, int(anchors_mask.sum()), band),
        extra={
            "mesh": {
                "voxelM": round(float(mesh.voxel * span.max()), 4),
                "voxels": int(mesh.solid.sum()),
                "tets": len(mesh.tets),
                "nodes": int(n),
                "filled": True,
                "heldNodes": int(held.sum()),
            },
            "dynamicsOver": "volume",
            **(extra or {}),
        },
    )


def fitter(method: str, policy: HandlePolicy, instances: Sequence[Mapping]) -> skin_scene.Fitter:
    """`skin_scene.build`'s `fit` for `method` under `policy`: each object's handle count (and
    class) from the policy, its fields from the method."""
    by_id = {int(i["id"]): i for i in instances}
    fits = {
        "freeform": lambda p, i, k, *, seed, handles, extra: _with_extra(
            skin_scene.fit_skin(p, i, k, seed=seed, handles=handles), extra
        ),
        "pinned": fit_pinned,
        "tetfem": fit_tetfem,
    }
    if method not in fits:
        raise ValueError(f"unknown skin method {method!r}; one of {sorted(fits)}")
    fit = fits[method]

    def run(points: np.ndarray, index: int, instance: int, *, seed: int = 0) -> skin_scene.Skin:
        record = by_id.get(instance, {"id": instance})
        handles, cls = policy.handles(record, points)
        extra = {**({"class": cls} if cls else {}), "traits": traits_of(record)}
        if handles <= 1:
            return fit_rigid(points, index, instance, seed=seed, extra=extra)
        return fit(points, index, instance, seed=seed, handles=handles, extra=extra)

    return run


#: The property scores a skin carries for its drivers' priors (skinWind.ts `materialPrior`).
TRAIT_PROPERTIES = ("movable", "rigid", "elastic", "static", "vegetation")


def traits_of(instance: Mapping) -> dict:
    """What a skin carries of the instance it moves, for a viewer without its `instances.json`
    (a variant made from another segmentation, a scan with none): a name, its category, its
    behaviour and the property scores the wind's prior reads."""
    tags = instance.get("tags") or []
    label = str(tags[0].get("label") or "") if tags else ""
    label = label or str(instance.get("category") or "")
    p = instance.get("properties") or {}
    out: dict = {"label": label} if label else {}
    if instance.get("category"):
        out["category"] = instance["category"]
    if instance.get("behaviour"):
        out["behaviour"] = instance["behaviour"]
    props = {
        k: round(float(p[k]), 3) for k in TRAIT_PROPERTIES if isinstance(p.get(k), (int, float))
    }
    if props:
        out["properties"] = props
    return out


def _with_extra(skin: skin_scene.Skin, extra: dict | None) -> skin_scene.Skin:
    skin.extra.update(extra or {})
    return skin
