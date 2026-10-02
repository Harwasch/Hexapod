"""Skins for scene objects: per splat, signed weights on a few handles (docs/SCENE_OBJECTS.md).

Step B2 of docs/LIVING_PLAN.md. For every instance that can move (`behaviour` in-place or
movable, `instances.json`), a skin is computed from its splat centres alone with Simplicits'
FreeForm/RKPM basis (`kaolin_rkpm.py`, vendored from NVIDIA Kaolin, Apache-2.0; no training,
NumPy/SciPy on the CPU), and written beside the tiles as `skin.json` + `skin.bin`, keyed by the
same per-tile checksums as `instances.json`. The viewer displaces each splat by

    x' = x + Σ_j w_j(x) · Z_j · [x − origin; 1]

with `Z_j` a 3×4 affine per handle per frame (`apps/web/src/cesium/splatSkin.ts`). Handle 0 is
the constant field (`w_0 ≡ 1`, not stored): `Z_0 = [R − I | t]` moves the object rigidly and
exactly. Handles `1..m−1` are the smallest elastic eigenmodes of the shape. Weights are signed
and never normalised to sum to one.

**Which instance owns a splat.** Ids in the tiles are leaf-level. A splat is skinned by the
coarsest instance on its ancestor chain whose behaviour moves (a tree, not each of its
branches): one skin per object, smooth across its parts.

**Fit** (`fit_skin`). The object's leaf-tile splat centres (merged level-of-detail parents are
evaluated, not fitted) are mapped into a cube around the object (proportions kept), `nodes`
kernels are placed by farthest point sampling (`node_count`: an eighth of the splats, 48 to
600), at most `FIT_POINTS` integration points are sampled the same way, and the generalised
eigenproblem `H c = λ M c` gives the modes, for a uniform material (`POISSON`; Young's modulus
cancels). Each field is scaled to `max |w| = 1` over the object's splats and its sign fixed so
that maximum is positive, so `Z_j` means "the displacement where handle j acts fully".

**Handle count** (`handle_count`): `m = clamp(round(8 + 2·log2(d / 2 m)), 8, 16)` for an
object whose bounds' diagonal is `d` -- 8 for a 2 m shrub, 12 at 8 m, 16 from 32 m -- and no
more than its node count allows.

**Weights on disk** (`quantise`): dense, one 16-byte row per skinned splat, byte `k` the
int8 weight of handle `k + 1` (`round(127·w)`), unused bytes 0. Dense rather than top-k:
the eigenmodes are global (every mode is non-zero almost everywhere), so dropping all but the
k largest per splat switches modes on and off between neighbours and tears the surface
(`sparsity_report`; the numbers are in SCENE_OBJECTS.md §4). 16 bytes is one RGBA32UI texel,
the viewer's upload unit.

Usage:
    python skin_scene.py TILES_DIR INSTANCES_JSON [--out DIR] [--link]
    python skin_scene.py data/tiles/synthetic-yard/splat \\
        data/tiles/synthetic-yard/instances/instances.json --out data/tiles/synthetic-yard/skin
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from kaolin_rkpm import SimplicitsRKPM

__all__ = [
    "Skin",
    "TileSplats",
    "build",
    "decode_tiles",
    "deform",
    "edge_stretch",
    "fit_skin",
    "handle_count",
    "link_skin",
    "local_jacobians",
    "node_count",
    "quantise",
    "read_tiles",
    "skin_owners",
    "sparsity_report",
    "write_skin",
]

FORMAT = "hexapod.skin"
VERSION = 1
FRAME = "tileset local ENU (the root transform's frame), metres"
FORMULA = (
    "x' = x + sum_j w_j(x) * Z_j * [x - origin; 1]; Z_j is 3x4 (row-major), j = 0 is the "
    "constant handle (w_0 = 1, not stored), w_j for j >= 1 is byte j - 1 of the splat's row "
    "times weights.scale; signed, not normalised"
)
#: Behaviours that get a skin.
SKINNED = ("in-place", "movable")
#: Bytes a splat's weight row takes: one RGBA32UI texel.
ROW_BYTES = 16
#: Handles an object may have, the constant one included (the learned ones fill a row).
MIN_HANDLES = 8
MAX_HANDLES = ROW_BYTES
#: int8 steps per unit weight.
QUANT = 127
#: Integration points for the eigenproblem, at most.
FIT_POINTS = 4000
#: Kernel nodes: an eighth of the splats, within these.
NODES_MIN = 48
NODES_MAX = 600
#: A uniform material: only Poisson's ratio shapes the modes.
POISSON = 0.45
#: Anchor splats (`anchor_mask`): the lowest tenth of an object's height ...
ANCHOR_BAND = 0.1
#: ... and at least this many median splat spacings.
ANCHOR_SPACINGS = 3


def handle_count(diagonal_m: float) -> int:
    """Handles for an object whose bounds' diagonal is `diagonal_m`, the constant included."""
    if not diagonal_m > 0:
        return MIN_HANDLES
    m = round(8 + 2 * math.log2(diagonal_m / 2.0))
    return int(min(MAX_HANDLES, max(MIN_HANDLES, m)))


def node_count(splats: int) -> int:
    """RKPM nodes for an object of `splats` points (never more than its points)."""
    return int(min(splats, max(NODES_MIN, min(NODES_MAX, splats // 8))))


def skin_owners(instances: Sequence[dict]) -> np.ndarray:
    """Per id (index 0: none), the id of the instance whose skin moves it, or 0.

    The coarsest instance on the ancestor chain whose behaviour is in `SKINNED`.
    """
    by_id = {int(i["id"]): i for i in instances}
    top = max(by_id, default=0)
    owner = np.zeros(top + 1, np.int64)
    for k in by_id:
        chain = []
        at: int | None = k
        while at is not None and at in by_id:
            chain.append(at)
            parent = by_id[at].get("parent")
            at = None if parent is None else int(parent)
        moving = [c for c in chain if by_id[c].get("behaviour") in SKINNED]
        owner[k] = moving[-1] if moving else 0
    return owner


# ------------------------------------------------------------------------------------ tiles


@dataclass
class TileSplats:
    """One tile: its uri, checksum, canonical positions, leaf-level ids and whether it is a
    leaf of the tileset (holds originals rather than merged parents)."""

    uri: str
    checksum: str
    positions: np.ndarray
    ids: np.ndarray
    leaf: bool


def decode_runs(runs: Sequence[int]) -> np.ndarray:
    """`[value, count, ...]` run-length pairs to the values."""
    pairs = np.asarray(runs, np.int64).reshape(-1, 2)
    return np.repeat(pairs[:, 0], pairs[:, 1])


def _leaf_uris(tileset: dict) -> set[str]:
    leaves: set[str] = set()

    def walk(tile: dict) -> None:
        uri = tile.get("content", {}).get("uri")
        if uri and not tile.get("children"):
            leaves.add(uri)
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    return leaves


def read_tiles(tiles_dir: Path, instances_doc: dict) -> list[TileSplats]:
    """Every tile's positions and ids, refusing a tile `instances.json` does not list (or
    lists with another count): skins are bound by the same checksums."""
    from rig_tiles import tile_positions, tile_uris
    from synthetic_tree import checksum_positions

    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    leaves = _leaf_uris(tileset)
    out: list[TileSplats] = []
    for uri in tile_uris(tileset):
        positions = tile_positions(tiles_dir / uri)
        checksum = checksum_positions(positions)
        runs = instances_doc["tiles"].get(checksum)
        if runs is None:
            raise SystemExit(f"{uri} ({checksum}) is not in instances.json: segment these tiles")
        ids = decode_runs(runs)
        if ids.size != positions.shape[0]:
            raise SystemExit(f"{uri}: {positions.shape[0]} gaussians, {ids.size} ids")
        out.append(TileSplats(uri, checksum, positions, ids, uri in leaves))
    return out


# -------------------------------------------------------------------------------------- fit


@dataclass
class Skin:
    """One object's skin: the fitted basis and what the contract records about it."""

    index: int  # 1-based skin id
    instance: int
    origin: np.ndarray  # rest frame origin: the bounds' base centre
    scale: float  # half the largest extent, metres
    handles: int  # the constant one included
    nodes: int
    splats: int  # fit points
    model: SimplicitsRKPM
    norm: np.ndarray  # per learned field: multiply the eigenmode by this
    eigenvalues: np.ndarray
    centres: np.ndarray  # per learned handle, |w|-weighted centre, rest frame
    radii: np.ndarray  # per learned handle, |w|-weighted rms distance from it
    seconds: float
    mass: np.ndarray  # (m, m): mean over the object's splats of w_i w_j, w_0 = 1
    anchor_gram: np.ndarray  # (m, m): the same over its anchor splats (`anchor_mask`)
    anchor_splats: int
    anchor_band: float  # metres above its lowest splat

    def weights(self, positions: np.ndarray) -> np.ndarray:
        """Learned weights at `positions` (tileset frame), (n, handles − 1), normalised."""
        if len(positions) == 0:
            return np.zeros((0, self.handles - 1))
        return self.model.forward(self.model._offset_scale(positions)) * self.norm[None, :]

    def weight_gradients(self, positions: np.ndarray) -> np.ndarray:
        """`∇w` of the learned weights at `positions`, (n, handles − 1, 3), per metre."""
        g = self.model.grad(self.model._offset_scale(positions))
        g = g / (self.model.bb_max - self.model.bb_min)[None, None]
        return g * self.norm[None, :, None]


def fit_skin(points: np.ndarray, index: int, instance: int, *, seed: int = 0) -> Skin:
    """The skin of one object from its splat centres (tileset frame, metres)."""
    started = time.perf_counter()
    points = np.asarray(points, np.float64)
    low, high = points.min(0), points.max(0)
    centre = (low + high) / 2
    half = float(max((high - low).max() / 2, 1e-3))
    pad = half * 1.02
    handles = handle_count(float(np.linalg.norm(high - low)))
    nodes = node_count(len(points))
    handles = max(2, min(handles, nodes - 1))
    model = SimplicitsRKPM(
        handles,
        nodes,
        num_points=FIT_POINTS,
        bb_min=centre - pad,
        bb_max=centre + pad,
        seed=seed,
    )
    n = len(points)
    model.init(points, np.ones(n), np.full(n, POISSON))
    learned = model.num_handles
    raw = model.forward(model._offset_scale(points))
    peak = np.abs(raw).argmax(0)
    peak_value = raw[peak, np.arange(learned)]
    norm = 1.0 / np.where(np.abs(peak_value) > 0, peak_value, 1.0)  # max |w| = 1, positive
    w = np.abs(raw * norm[None, :])
    origin = np.array([centre[0], centre[1], low[2]])
    local = points - origin
    mass = w.sum(0).clip(1e-12)
    centres = (w.T @ local) / mass[:, None]
    radii = np.sqrt((w * ((local[:, None, :] - centres[None]) ** 2).sum(-1)).sum(0) / mass)
    gram, anchor_gram, anchors, band = modal_grams(points, raw * norm[None, :])
    return Skin(
        index=index,
        instance=instance,
        origin=origin,
        scale=half,
        handles=learned + 1,
        nodes=model.rkpm.num_nodes if model.rkpm else nodes,
        splats=n,
        model=model,
        norm=norm,
        eigenvalues=model.evals.copy(),
        centres=centres,
        radii=radii,
        seconds=time.perf_counter() - started,
        mass=gram,
        anchor_gram=anchor_gram,
        anchor_splats=anchors,
        anchor_band=band,
    )


def anchor_mask(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Where an object meets what holds it: its splats within `ANCHOR_BAND` of its height (at
    least `ANCHOR_SPACINGS` median splat spacings) above its lowest one. Returns the mask and
    the band in metres.

    Measured on the yard: contact with unskinned neighbours (splats within two spacings of a
    static splat) anchors a shrub wherever it touches the next shrub, up to its top, and leaves
    it no direction to move in; the lowest band leaves the tree 12 of its 14 handle directions
    and a 1 m shrub 2 of 8 that keep the base within 5% of still (`ANCHOR_TOLERANCE` in
    packages/world/src/skinWind.ts)."""
    points = np.asarray(points, np.float64)
    if len(points) < 2:
        return np.ones(len(points), bool), 0.0
    distance, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(distance[:, 1]))
    low = float(points[:, 2].min())
    height = float(points[:, 2].max()) - low
    band = max(ANCHOR_BAND * height, ANCHOR_SPACINGS * spacing)
    return points[:, 2] <= low + band, band


def modal_grams(
    points: np.ndarray, learned: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """What a modal driver needs beyond the eigenvalues (SCENE_OBJECTS.md §4, `dynamics`): the
    weights' Gram over the object's splats, `M_ij = mean(w_i w_j)` with `w_0 = 1` -- the
    handles' mass matrix for translations; its row 0 is each handle's mean weight, how much a
    uniform force drives it -- and the same over its anchor splats (`anchor_mask`), with
    their count and band."""
    w = np.c_[np.ones(len(learned)), learned]
    gram = w.T @ w / max(len(w), 1)
    mask, band = anchor_mask(points)
    a = w[mask]
    anchor_gram = a.T @ a / max(len(a), 1)
    return gram, anchor_gram, int(mask.sum()), band


def _upper(matrix: np.ndarray) -> list[float]:
    """A symmetric matrix's upper triangle, row by row, to five significant digits."""
    rows, cols = np.triu_indices(matrix.shape[0])
    return [float(f"{v:.5g}") + 0.0 for v in matrix[rows, cols]]


# ------------------------------------------------------------------------- quantise, deform


def quantise(weights: np.ndarray) -> np.ndarray:
    """Learned weights (n, ≤ 16) to int8 rows (n, 16): `round(127·w)`, clipped, rest 0."""
    rows = np.zeros((len(weights), ROW_BYTES), np.int8)
    q = np.clip(np.round(np.asarray(weights) * QUANT), -QUANT, QUANT)
    rows[:, : q.shape[1]] = q.astype(np.int8)
    return rows


def dequantise(rows: np.ndarray, learned: int) -> np.ndarray:
    """int8 rows back to the first `learned` weights."""
    return rows[:, :learned].astype(np.float64) / QUANT


def deform(
    positions: np.ndarray, origin: np.ndarray, learned: np.ndarray, handles: np.ndarray
) -> np.ndarray:
    """`x + Σ_j w_j Z_j [x − origin; 1]` with `w_0 = 1`; `handles` (m, 3, 4), `learned`
    (n, ≥ m − 1) -- the shader's arithmetic, in float64."""
    local = np.asarray(positions, np.float64) - origin
    xh = np.concatenate([local, np.ones((len(local), 1))], axis=1)
    m = handles.shape[0]
    w = np.concatenate([np.ones((len(local), 1)), learned[:, : m - 1]], axis=1)
    return positions + np.einsum("nj,jab,nb->na", w, handles, xh)


def random_handles(
    rng: np.random.Generator,
    handles: int,
    scale: float,
    *,
    constant: bool = False,
) -> np.ndarray:
    """Random affine handles (m, 3, 4) for an object of half-size `scale`: linear parts and
    translations of comparable effect, the constant handle at rest unless `constant`."""
    z = rng.normal(size=(handles, 3, 4))
    z[:, :, 3] *= scale
    if not constant:
        z[0] = 0
    return z


def scaled_to(
    positions: np.ndarray, origin: np.ndarray, learned: np.ndarray, z: np.ndarray, peak: float
) -> np.ndarray:
    """`z` scaled so its largest displacement over `positions` is `peak` metres."""
    moved = deform(positions, origin, learned, z)
    top = float(np.abs(moved - positions).max())
    return z * (peak / top) if top > 0 else z


def edge_stretch(rest: np.ndarray, moved: np.ndarray, k: int = 8) -> np.ndarray:
    """Every k-nearest-neighbour edge's length after over before."""
    k = min(k, len(rest) - 1)
    if k < 1:
        return np.ones(0)
    _, nb = cKDTree(rest).query(rest, k=k + 1)
    nb = nb[:, 1:]
    before = np.linalg.norm(rest[:, None] - rest[nb], axis=-1)
    after = np.linalg.norm(moved[:, None] - moved[nb], axis=-1)
    keep = before > 1e-6
    return after[keep] / before[keep]


def local_jacobians(rest: np.ndarray, moved: np.ndarray, k: int = 12) -> np.ndarray:
    """Per point, the least-squares linear map of its k neighbours' offsets, (n, 3, 3)."""
    k = min(k, len(rest) - 1)
    _, nb = cKDTree(rest).query(rest, k=k + 1)
    a = rest[nb[:, 1:]] - rest[:, None]  # (n, k, 3)
    b = moved[nb[:, 1:]] - moved[:, None]
    # F aᵀ ≈ bᵀ  →  F = (bᵀ a)(aᵀ a)⁻¹, regularised for flat neighbourhoods.
    ata = np.einsum("nki,nkj->nij", a, a)
    bta = np.einsum("nki,nkj->nij", b, a)
    reg = np.eye(3)[None] * (np.trace(ata, axis1=1, axis2=2)[:, None, None] * 1e-3 + 1e-12)
    return bta @ np.linalg.inv(ata + reg)


def skin_jacobians(skin: Skin, positions: np.ndarray, handles: np.ndarray) -> np.ndarray:
    """The deformation's exact Jacobian at `positions`, (n, 3, 3):
    `I + Σ_j (w_j A_j + Z_j [x − o; 1] ∇w_jᵀ)`. The viewer draws covariances through the
    first sum only (it drops the `∇w` term)."""
    m = handles.shape[0]
    w = np.c_[np.ones(len(positions)), skin.weights(positions)][:, :m]
    grads = np.concatenate(
        [np.zeros((len(positions), 1, 3)), skin.weight_gradients(positions)], axis=1
    )[:, :m]
    local = np.c_[positions - skin.origin, np.ones(len(positions))]
    disp = np.einsum("jab,nb->nja", handles, local)
    jac = np.eye(3)[None] + np.einsum("nj,jab->nab", w, handles[:, :, :3])
    return jac + np.einsum("nja,njb->nab", disp, grads)


def sparsity_report(
    skin: Skin,
    positions: np.ndarray,
    *,
    ks: Sequence[int] = (2, 4, 8),
    trials: int = 5,
    peak_fraction: float = 0.05,
    seed: int = 1,
) -> dict[str, dict[str, float]]:
    """Dense int8 and top-k int8 against float64 weights, under random handle transforms
    whose largest displacement is `peak_fraction` of the object's half-size: displacement
    error (rms and max, as a share of the rms displacement) and kNN edge stretch."""
    rng = np.random.default_rng(seed)
    exact = skin.weights(positions)
    learned = exact.shape[1]
    dense = dequantise(quantise(exact), learned)
    variants: dict[str, np.ndarray] = {"float": exact, "int8": dense}
    order = np.argsort(-np.abs(exact), axis=1)
    for k in ks:
        if k >= learned:
            continue
        keep = np.zeros_like(exact, bool)
        np.put_along_axis(keep, order[:, :k], True, axis=1)
        variants[f"top{k}-int8"] = np.where(keep, dense, 0.0)
    out: dict[str, dict[str, float]] = {}
    acc: dict[str, list[tuple[float, float, float, float]]] = {name: [] for name in variants}
    for _ in range(trials):
        z = random_handles(rng, skin.handles, skin.scale)
        z = scaled_to(positions, skin.origin, exact, z, peak_fraction * skin.scale)
        reference = deform(positions, skin.origin, exact, z)
        disp = np.sqrt(((reference - positions) ** 2).sum(1).mean())
        for name, w in variants.items():
            moved = deform(positions, skin.origin, w, z)
            err = np.linalg.norm(moved - reference, axis=1)
            stretch = edge_stretch(positions, moved)
            acc[name].append(
                (
                    float(np.sqrt((err**2).mean()) / disp),
                    float(err.max() / disp),
                    float(np.percentile(stretch, 99)),
                    float(stretch.max()),
                )
            )
    for name, rows in acc.items():
        a = np.array(rows)
        out[name] = {
            "rmsError": float(a[:, 0].mean()),
            "maxError": float(a[:, 1].mean()),
            "stretchP99": float(a[:, 2].mean()),
            "stretchMax": float(a[:, 3].mean()),
            # float32 dense; int8 rows; top-k: an int8 weight and a 4-bit handle index each.
            "bytesPerSplat": float(
                ROW_BYTES
                if name == "int8"
                else 4 * learned
                if name == "float"
                else int(name[3]) * 1.5
            ),
        }
    return out


# ------------------------------------------------------------------------------------ build


def _rle(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    change = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    lengths = np.diff(np.r_[change, values.size])
    out = np.empty(change.size * 2, np.int64)
    out[0::2] = values[change]
    out[1::2] = lengths
    return [int(v) for v in out]


def _round(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values).reshape(-1)]


@dataclass
class Built:
    document: dict
    blob: bytes
    skins: list[Skin]
    clipped: int  # weights beyond ±1 at evaluation, clipped by the int8 rows


def build(
    tiles: Sequence[TileSplats],
    instances: Sequence[dict],
    *,
    only: Sequence[int] | None = None,
    log: bool = False,
) -> Built:
    """`skin.json` and `skin.bin` for these tiles and instances (`only`: skin just these
    owners, the rest stay unskinned -- for small fixtures)."""
    owner = skin_owners(instances)
    if only is not None:
        owner = np.where(np.isin(owner, list(only)), owner, 0)
    by_owner: dict[int, list[np.ndarray]] = {}
    for tile in tiles:
        if not tile.leaf:
            continue
        own = owner[np.clip(tile.ids, 0, owner.size - 1)] * (tile.ids < owner.size)
        for k in np.unique(own[own > 0]):
            by_owner.setdefault(int(k), []).append(tile.positions[own == k])
    skins: list[Skin] = []
    skin_of_instance = np.zeros(owner.size, np.int64)
    for k in sorted(by_owner):
        points = np.unique(np.concatenate(by_owner[k]).astype(np.float64), axis=0)
        if len(points) < 4:
            continue
        skin = fit_skin(points, len(skins) + 1, k, seed=k)
        skins.append(skin)
        skin_of_instance[k] = skin.index
        if log:
            print(
                f"skin {skin.index}: instance {k}, {len(points)} splats, {skin.handles} handles, "
                f"{skin.nodes} nodes, {skin.seconds:.1f} s",
                flush=True,
            )
    rows: list[np.ndarray] = []
    row = 0
    clipped = 0
    tile_doc: dict[str, dict] = {}
    for tile in sorted(tiles, key=lambda t: t.checksum):
        own = owner[np.clip(tile.ids, 0, owner.size - 1)] * (tile.ids < owner.size)
        which = skin_of_instance[own]
        block = np.zeros((tile.ids.size, ROW_BYTES), np.int8)
        for s in np.unique(which[which > 0]):
            skin = skins[int(s) - 1]
            at = which == s
            w = skin.weights(tile.positions[at].astype(np.float64))
            clipped += int((np.abs(w) > 1.0 + 0.5 / QUANT).sum())
            block[at] = quantise(w)
        tile_rows = block[which > 0]
        tile_doc[tile.checksum] = {"skins": _rle(which), "row": row}
        rows.append(tile_rows)
        row += len(tile_rows)
    blob = (np.concatenate(rows) if rows else np.zeros((0, ROW_BYTES), np.int8)).tobytes()
    document = {
        "format": FORMAT,
        "version": VERSION,
        "frame": FRAME,
        "formula": FORMULA,
        "method": {
            "name": "simplicits-rkpm",
            "source": "NVIDIA Kaolin (Apache-2.0), vendored in tools/captures/kaolin_rkpm.py",
            "material": {"uniform": True, "poisson": POISSON},
        },
        "weights": {
            "file": "skin.bin",
            "dtype": "int8",
            "rowBytes": ROW_BYTES,
            "scale": 1.0 / QUANT,
            "rows": row,
        },
        "skins": [
            {
                "id": s.index,
                "instance": s.instance,
                "handles": s.handles,
                "origin": _round(s.origin),
                "scale": round(s.scale, 4),
                "splats": s.splats,
                "nodes": s.nodes,
                "eigenvalues": [float(f"{v:.6g}") for v in s.eigenvalues],
                "support": [
                    {"centre": _round(c), "radius": round(float(r), 4)}
                    for c, r in zip(s.centres, s.radii, strict=True)
                ],
                "dynamics": {
                    "mass": _upper(s.mass),
                    "anchor": {
                        "source": "base",
                        "splats": s.anchor_splats,
                        "band": round(s.anchor_band, 4),
                        "gram": _upper(s.anchor_gram),
                    },
                },
            }
            for s in skins
        ],
        "tiles": tile_doc,
        "tilesEncoding": "rle [skin, count, ...] in the tile's own order (0: no skin); "
        "skinned splats take consecutive rows of skin.bin from `row`",
    }
    return Built(document, blob, skins, clipped)


def write_skin(out_dir: Path, built: Built) -> None:
    """`skin.json` and `skin.bin` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skin.bin").write_bytes(built.blob)
    (out_dir / "skin.json").write_text(
        json.dumps(built.document, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def link_skin(tileset: Path, count: int, uri: str = "skin.json") -> None:
    """`root.extras.skin = {uri, count}` on a tileset.json, in place if already there."""
    document = json.loads(tileset.read_text(encoding="utf-8"))
    extras = document["root"].setdefault("extras", {})
    extras["skin"] = {"uri": uri, "count": count}
    tileset.write_text(json.dumps(document, indent=1), encoding="utf-8")


def decode_tiles(document: dict, blob: bytes) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per tile checksum, every splat's skin id and int8 row (zeros where unskinned): what the
    viewer decodes, for round-trip checks."""
    rows = np.frombuffer(blob, np.int8).reshape(-1, ROW_BYTES)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for checksum, entry in document["tiles"].items():
        which = decode_runs(entry["skins"])
        block = np.zeros((which.size, ROW_BYTES), np.int8)
        skinned = which > 0
        start = int(entry["row"])
        block[skinned] = rows[start : start + int(skinned.sum())]
        out[checksum] = (which, block)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tiles", type=Path, help="the tileset directory (tileset.json)")
    parser.add_argument("instances", type=Path, help="its instances.json")
    parser.add_argument("--out", type=Path, default=None, help="default: beside instances.json")
    parser.add_argument(
        "--link", action="store_true", help="declare it in the tileset's root.extras.skin"
    )
    parser.add_argument("--report", type=Path, default=None, help="sparsity report (JSON)")
    parser.add_argument(
        "--only", default=None, help="comma-separated instance ids: skin just these objects"
    )
    args = parser.parse_args()
    doc = json.loads(args.instances.read_text(encoding="utf-8"))
    tiles = read_tiles(args.tiles, doc)
    only = None if args.only is None else [int(v) for v in args.only.split(",")]
    built = build(tiles, doc["instances"], only=only, log=True)
    out = args.out or args.instances.parent
    write_skin(out, built)
    splats = sum(t.ids.size for t in tiles)
    print(
        f"{len(built.skins)} skins, {built.document['weights']['rows']} of {splats} tile "
        f"splats skinned, skin.bin {len(built.blob)} B, {built.clipped} weights clipped -> {out}"
    )
    if args.link:
        link_skin(
            args.tiles / "tileset.json",
            len(built.skins),
            os.path.relpath(out / "skin.json", args.tiles),
        )
    if args.report is not None:
        report = {}
        for skin in built.skins:
            points = np.concatenate(
                [
                    t.positions[np.isin(t.ids, _members(doc["instances"], skin.instance))]
                    for t in tiles
                    if t.leaf
                ]
            ).astype(np.float64)
            report[str(skin.instance)] = sparsity_report(skin, points)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")


def _members(instances: Sequence[dict], root: int) -> list[int]:
    """`root` and every instance below it."""
    children: dict[int, list[int]] = {}
    for i in instances:
        if i.get("parent") is not None:
            children.setdefault(int(i["parent"]), []).append(int(i["id"]))
    out, stack = [], [root]
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(children.get(k, []))
    return out


if __name__ == "__main__":
    main()
