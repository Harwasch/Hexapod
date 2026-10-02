"""Movable objects out of a scan's spatial tiles into tilesets of their own, and the hole
each leaves filled (docs/SCENE_OBJECTS.md §4 "Split objects", §5; LIVING_PLAN.md C4).

A movable object stored in the spatial tiles cannot move: its gaussians would leave their
tile's bounds and the tile would be culled from under them. `split` takes chosen instances
(by default the coarsest `movable` ones of a useful size; or explicit ids) and writes, into a
**new** directory (the input is never written to):

1. **objects/<id>/** -- each object's leaf gaussians as a one-tile tileset of its own, in its
   own frame: positions relative to its **origin** (the base centre of its bounds, on the SPZ
   grid, so the shift is exact), the root transform the scan's times the origin's
   translation, and view cones from the scan's observers. Moving it is one matrix.
2. **the scan's tiles without it**: every tile that held any of its gaussians is rewritten
   without them, byte for byte what was there minus the removed records (the SPZ is sliced,
   never re-quantised), under a new name; untouched tiles are copied unchanged. A merged
   parent's gaussian goes with the object when its id is the object's or, unbound (it merged
   several ids), when most of its `BIND_NEIGHBOURS` nearest leaves are the object's.
3. **instances.json** re-bound: the rewritten tiles' checksums changed, so their runs are the
   old runs without the removed gaussians, under the new checksum; the object tiles' runs are
   added, so hide, highlight and search treat an object as the same ids wherever it is drawn.
   `skin.json` (when linked) is re-bound the same way.
4. **fills/<id>/**: the hole behind the object filled by Teacher B (`teacher_fill.fill_hole`:
   views around where it stood, the pixels that now see through to nothing or to what lies
   behind the surface around it, filled, lifted onto that surface, optionally distilled) as an
   inferred layer, declared in `root.extras.inferredLayers` like any other.
5. **root.extras.objects**: per object its tileset's uri, instance id, origin and pose (the
   rest pose: identity), and its fill's uri -- how the viewer finds and places them.

Re-running with the same arguments writes the same bytes (idempotent); a directory not
written by `split` is refused as an output, and so is the input.

Usage:
    python split_objects.py split TILES_DIR OUT_DIR [--ids 3,7] [--instances PATH]
        [--filler telea|module:Class] [--renderer cpu|gsplat] [--views 6]
        [--distill N --distill-on local|modal] [--no-fill] [--save DIR]
    python split_objects.py candidates TILES_DIR [--select movable|loose]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import scene_plants
from rig_tiles import glb_spz
from skin_scene import decode_runs
from splat_tiles import SH_DIMS, SPZ_MAGIC, build_glb
from synthetic_tree import checksum_positions

FORMAT = "hexapod.split"
VERSION = 1
OBJECT_FORMAT = "hexapod.object"
#: Where the object tilesets and their fills go, inside the output directory.
OBJECTS_DIR = "objects"
FILLS_DIR = "fills"
#: An object's tile content.
OBJECT_CONTENT = "object.glb"
#: Default choice: an object needs this many gaussians (itself and its descendants)...
MIN_SPLATS = 500
#: ...and its bounds' largest side at most this (m): what moves is vehicle-sized at most.
MAX_EXTENT_M = 8.0
#: Neighbours an unbound merged parent's gaussian consults (as `segment_scene` binds them).
BIND_NEIGHBOURS = 8
#: Proposed selection rule (`--select loose`), A6's pumpkins: see `loose`.
LOOSE_MOVABLE = 0.4
LOOSE_MARGIN = 0.1
LOOSE_SHARE = 0.4
#: `--absorb`: an id goes with an object when this share of its gaussians lies in the
#: object's box (its gaussians' 3rd-97th percentiles, padded by `ABSORB_PAD` of its size)...
ABSORB_INSIDE = 0.8
ABSORB_PAD = 0.05
#: ...and it has at most this share of the object's gaussians: a fragment, not a neighbour.
ABSORB_SHARE = 0.2
POSE_REST = {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}


# ------------------------------------------------------------------------------- SPZ records


@dataclass
class Spz:
    """An SPZ (v2 or v3) as its records, so gaussians can be taken out and moved without
    re-quantising anything: fixed-point positions and the byte fields as stored."""

    version: int
    degree: int
    fractional_bits: int
    flags: int
    fixed: np.ndarray  # (n, 3) int64
    alpha: np.ndarray  # (n,) uint8
    colour: np.ndarray  # (n, 3) uint8
    scale: np.ndarray  # (n, 3) uint8
    rotation: np.ndarray  # (n, 3 or 4) uint8
    sh: np.ndarray  # (n, 3 * K) uint8

    def __len__(self) -> int:
        return int(self.fixed.shape[0])

    def take(self, index: np.ndarray) -> Spz:
        return Spz(
            self.version,
            self.degree,
            self.fractional_bits,
            self.flags,
            self.fixed[index],
            self.alpha[index],
            self.colour[index],
            self.scale[index],
            self.rotation[index],
            self.sh[index],
        )

    def positions(self) -> np.ndarray:
        """Float32 positions as `rig_tiles.tile_positions` reads them (the checksum's input)."""
        out = (self.fixed / float(1 << self.fractional_bits)).astype(np.float32)
        out[out == 0.0] = 0.0
        return out

    def scales(self) -> np.ndarray:
        """Linear scales (m)."""
        return np.exp(self.scale.astype(np.float64) / 16.0 - 10.0)

    @staticmethod
    def concat(parts: Sequence[Spz]) -> Spz:
        first = parts[0]
        for p in parts[1:]:
            if (p.version, p.degree, p.fractional_bits) != (
                first.version,
                first.degree,
                first.fractional_bits,
            ):
                raise ValueError("tiles disagree on SPZ version, SH degree or fixed-point bits")
        return Spz(
            first.version,
            first.degree,
            first.fractional_bits,
            first.flags,
            *(
                np.concatenate([getattr(p, name) for p in parts])
                for name in ("fixed", "alpha", "colour", "scale", "rotation", "sh")
            ),
        )


def read_spz(blob: bytes) -> Spz:
    """The records of a gzip-framed SPZ (what `splat_tiles.pack_spz` writes)."""
    raw = gzip.decompress(blob)
    magic, version, count, degree, bits, flags, _reserved = struct.unpack_from("<IIIBBBB", raw)
    if magic != SPZ_MAGIC or version not in (2, 3):
        raise ValueError(f"not an SPZ v2/v3 (magic 0x{magic:08X}, version {version})")
    rb = 3 if version == 2 else 4
    k = 3 * SH_DIMS[degree]
    body = np.frombuffer(raw, np.uint8, offset=16)
    o = 0

    def cut(width: int) -> np.ndarray:
        nonlocal o
        part = body[o : o + count * width].reshape(count, width)
        o += count * width
        return part

    packed = cut(9).reshape(count, 3, 3).astype(np.int64)
    fixed = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    fixed = (fixed ^ 0x800000) - 0x800000
    alpha = cut(1)[:, 0]
    colour, scale, rotation = cut(3), cut(3), cut(rb)
    sh = cut(k) if k else np.zeros((count, 0), np.uint8)
    return Spz(version, degree, bits, flags, fixed, alpha, colour, scale, rotation, sh)


def write_spz(spz: Spz) -> bytes:
    """The inverse of `read_spz`, framed as `splat_tiles.pack_spz` frames it (gzip level 6,
    mtime 0: the same records are the same bytes)."""
    n = len(spz)
    header = struct.pack(
        "<IIIBBBB", SPZ_MAGIC, spz.version, n, spz.degree, spz.fractional_bits, spz.flags, 0
    )
    u = (spz.fixed & 0xFFFFFF).astype(np.uint32)
    pos = np.stack([u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF], axis=2).astype(np.uint8)
    raw = header + b"".join(
        np.ascontiguousarray(a, np.uint8).tobytes()
        for a in (pos, spz.alpha, spz.colour, spz.scale, spz.rotation, spz.sh)
    )
    return gzip.compress(raw, compresslevel=6, mtime=0)


def read_tile(path: Path) -> Spz:
    return read_spz(glb_spz(path))


def tile_bytes(spz: Spz) -> bytes:
    """A GLB as `splat_tiles` writes a tile."""
    xyz = spz.positions()
    return build_glb(
        len(spz), xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist(), write_spz(spz), spz.degree
    )


# ------------------------------------------------------------------------------- choosing


@dataclass
class Choice:
    instance: int
    #: Every id that belongs to it: itself and its descendants.
    ids: list[int]
    splats: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    @property
    def extent(self) -> float:
        return float(np.max(self.bounds_max - self.bounds_min))


def _children(instances: Sequence[dict]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i in instances:
        if i.get("parent"):
            out.setdefault(int(i["parent"]), []).append(int(i["id"]))
    return out


def _subtree(root: int, children: dict[int, list[int]]) -> list[int]:
    out, stack = [], [root]
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(children.get(k, []))
    return sorted(out)


def _choice(doc: dict, instance: int) -> Choice:
    by_id = {int(i["id"]): i for i in doc["instances"]}
    if instance not in by_id:
        raise ValueError(f"instance {instance} is not in instances.json")
    ids = _subtree(instance, _children(doc["instances"]))
    rows = [by_id[k] for k in ids]
    return Choice(
        instance,
        ids,
        int(sum(int(r["splats"]) for r in rows)),
        np.min([r["bounds"]["min"] for r in rows], axis=0),
        np.max([r["bounds"]["max"] for r in rows], axis=0),
    )


def loose(instance: dict, scene_extent: float) -> bool:
    """A proposed rule for what `segment_scene.BEHAVIOUR_RULE` calls in-place but is loose
    (A6: the pumpkins scored vegetation 0.85-0.96, so `in-place` took them, though nothing
    roots them): movable when the movable score is at least `LOOSE_MOVABLE` and within
    `LOOSE_MARGIN` of the static one, and the object is compact -- its bounds' largest side
    at most `LOOSE_SHARE` of the scan's -- whatever its vegetation score. Size, not a class,
    keeps the ground and the bushes out."""
    p = instance.get("properties", {})
    if instance.get("behaviour") == "movable":
        return True
    side = float(np.max(np.subtract(instance["bounds"]["max"], instance["bounds"]["min"])))
    return (
        p.get("movable", 0.0) >= LOOSE_MOVABLE
        and p.get("movable", 0.0) >= p.get("static", 0.0) - LOOSE_MARGIN
        and side <= LOOSE_SHARE * scene_extent
    )


def choose(
    doc: dict,
    ids: Sequence[int] | None = None,
    *,
    min_splats: int = MIN_SPLATS,
    max_extent_m: float = MAX_EXTENT_M,
    select: str = "movable",
    scene_extent: float | None = None,
) -> list[Choice]:
    """The objects to split: `ids` as given (an id inside another chosen one is dropped), or
    the coarsest instances `select` picks (`movable`: behaviour movable; `loose`: `loose`)
    with at least `min_splats` gaussians and bounds no larger than `max_extent_m`."""
    by_id = {int(i["id"]): i for i in doc["instances"]}
    if ids:
        chosen = [_choice(doc, int(k)) for k in dict.fromkeys(ids)]
    else:
        if select == "loose":
            extent = scene_extent or float(
                np.max(
                    np.max([i["bounds"]["max"] for i in doc["instances"]], axis=0)
                    - np.min([i["bounds"]["min"] for i in doc["instances"]], axis=0)
                )
            )
            picks = {k for k, i in by_id.items() if loose(i, extent)}
        elif select == "movable":
            picks = {k for k, i in by_id.items() if i.get("behaviour") == "movable"}
        else:
            raise ValueError(f"select {select!r}: 'movable' or 'loose'")

        def coarsest(k: int) -> bool:
            p = by_id[k].get("parent")
            while p:
                if p in picks:
                    return False
                p = by_id[p].get("parent")
            return True

        chosen = [_choice(doc, k) for k in sorted(picks) if coarsest(k)]
        chosen = [c for c in chosen if c.splats >= min_splats and c.extent <= max_extent_m]
    owned: set[int] = set()
    out = []
    for c in sorted(chosen, key=lambda c: (-len(c.ids), c.instance)):
        if c.instance in owned:
            continue
        owned.update(c.ids)
        out.append(c)
    return sorted(out, key=lambda c: c.instance)


# ------------------------------------------------------------------------------- the tiles


@dataclass
class TileRef:
    node: dict
    uri: str
    leaf: bool
    spz: Spz
    checksum: str
    labels: np.ndarray
    #: Per gaussian: the object it goes with (index into the choices), or -1.
    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


def _nodes(tileset: dict) -> list[tuple[dict, bool]]:
    out: list[tuple[dict, bool]] = []

    def walk(tile: dict) -> None:
        if tile.get("content", {}).get("uri"):
            out.append((tile, not tile.get("children")))
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    return out


def read_tiles(tiles_dir: Path, tileset: dict, doc: dict) -> list[TileRef]:
    """Every tile's records and instance ids (refused when instances.json does not list it)."""
    out = []
    for node, leaf in _nodes(tileset):
        uri = node["content"]["uri"]
        spz = read_tile(tiles_dir / uri)
        checksum = checksum_positions(spz.positions())
        runs = doc["tiles"].get(checksum)
        if runs is None:
            raise ValueError(f"tile {uri} ({checksum}) is not in instances.json: segment first")
        labels = decode_runs(runs)
        if labels.size != len(spz):
            raise ValueError(f"tile {uri}: {labels.size} ids for {len(spz)} gaussians")
        out.append(TileRef(node, uri, leaf, spz, checksum, labels))
    return out


def absorb(tiles: Sequence[TileRef], choices: Sequence[Choice]) -> dict[int, list[int]]:
    """Fragments of each chosen object that segmentation left under other ids (A6: a pumpkin
    in a dozen small top-level instances): every other id with `ABSORB_INSIDE` of its leaf
    gaussians inside the object's box and at most `ABSORB_SHARE` of its size joins it.
    Geometry only. Returns, per chosen instance, the ids it took (they are added to its
    `ids`)."""
    leaves = [t for t in tiles if t.leaf]
    positions = np.concatenate([t.spz.positions() for t in leaves]).astype(np.float64)
    labels = np.concatenate([t.labels for t in leaves])
    counts = np.bincount(labels)
    taken: set[int] = {k for c in choices for k in c.ids}
    out: dict[int, list[int]] = {}
    for c in choices:
        mine = np.isin(labels, c.ids)
        if not mine.any():
            continue
        low, high = np.percentile(positions[mine], [3, 97], axis=0)
        pad = ABSORB_PAD * float(np.max(high - low))
        inside = np.all((positions >= low - pad) & (positions <= high + pad), axis=1)
        within = np.bincount(labels[inside], minlength=counts.size)
        share = within / np.maximum(counts, 1)
        size = int(mine.sum())
        joined = [
            int(k)
            for k in np.flatnonzero((share >= ABSORB_INSIDE) & (counts <= ABSORB_SHARE * size))
            if k > 0 and int(k) not in taken
        ]
        taken.update(joined)
        c.ids = sorted(c.ids + joined)
        out[c.instance] = joined
    return out


def assign(tiles: Sequence[TileRef], choices: Sequence[Choice], max_id: int) -> None:
    """Each gaussian's owner: a leaf's by its id; a merged parent's by its id when bound,
    else by the majority of its `BIND_NEIGHBOURS` nearest leaves."""
    of_id = np.full(max_id + 1, -1, np.int64)
    for k, c in enumerate(choices):
        of_id[c.ids] = k
    for t in tiles:
        t.owner = of_id[t.labels]
    leaves = [t for t in tiles if t.leaf]
    positions = np.concatenate([t.spz.positions() for t in leaves]).astype(np.float64)
    owners = np.concatenate([t.owner for t in leaves])
    if not (owners >= 0).any():
        return
    tree = cKDTree(positions)
    k = min(BIND_NEIGHBOURS, len(positions))
    for t in tiles:
        if t.leaf:
            continue
        unbound = np.flatnonzero(t.labels == 0)
        if unbound.size == 0:
            continue
        _, index = tree.query(t.spz.positions()[unbound].astype(np.float64), k=k)
        near = owners[np.asarray(index).reshape(unbound.size, k)]
        for j in range(len(choices)):
            votes = (near == j).sum(axis=1)
            t.owner[unbound[votes * 2 > k]] = j


def _content_name(uri: str, checksum: str) -> str:
    """A rewritten tile's name: its old stem and its new checksum's digest, so a cache that
    holds the old tile can never serve it for the new one."""
    stem, _, suffix = uri.rpartition(".")
    return f"{stem}.{checksum.rsplit(':', 1)[-1]}.{suffix}"


def _matrix(transform: Sequence[float] | None) -> np.ndarray:
    """A 3D Tiles column-major transform as a row-major 4x4."""
    return np.eye(4) if transform is None else np.asarray(transform, np.float64).reshape(4, 4).T


def _transform(matrix: np.ndarray) -> list[float]:
    return [float(v) for v in matrix.T.reshape(-1)]


def _box(low: np.ndarray, high: np.ndarray) -> list[float]:
    mid, half = (low + high) / 2, (high - low) / 2
    return [*map(float, mid), float(half[0]), 0, 0, 0, float(half[1]), 0, 0, 0, float(half[2])]


@dataclass
class SplitObject:
    choice: Choice
    spz: Spz  # in the object's frame
    labels: np.ndarray
    origin: np.ndarray  # scene frame, m (on the SPZ grid)
    uri: str  # its tileset.json, relative to the scene's
    checksum: str
    from_tiles: int
    fill: dict | None = None

    def scene_spz(self) -> Spz:
        """Its gaussians back in the scene's frame."""
        shift = np.round(self.origin * (1 << self.spz.fractional_bits)).astype(np.int64)
        moved = self.spz.take(np.arange(len(self.spz)))
        moved.fixed = self.spz.fixed + shift
        return moved


def gather(tiles: Sequence[TileRef], k: int, choice: Choice, uri: str) -> SplitObject:
    """Object `k`'s leaf gaussians, in tile order, moved into its own frame."""
    parts, labels, sources = [], [], 0
    for t in tiles:
        if not t.leaf:
            continue
        mine = np.flatnonzero(t.owner == k)
        if mine.size:
            parts.append(t.spz.take(mine))
            labels.append(t.labels[mine])
            sources += 1
    if not parts:
        raise ValueError(f"instance {choice.instance} has no gaussians in the leaf tiles")
    spz = Spz.concat(parts)
    unit = float(1 << spz.fractional_bits)
    low, high = spz.fixed.min(axis=0), spz.fixed.max(axis=0)
    origin_fixed = np.array([(low[0] + high[0]) // 2, (low[1] + high[1]) // 2, low[2]], np.int64)
    spz.fixed = spz.fixed - origin_fixed
    return SplitObject(
        choice,
        spz,
        np.concatenate(labels),
        origin_fixed / unit,
        uri,
        checksum_positions(spz.positions()),
        sources,
    )


def object_tileset(obj: SplitObject, scene: dict, scene_uri: str) -> dict:
    """The object's one-tile tileset: the scene's root transform times its origin's."""
    xyz = obj.spz.positions().astype(np.float64)
    reach = 3.0 * obj.spz.scales().max(axis=1)
    low = (xyz - reach[:, None]).min(axis=0)
    high = (xyz + reach[:, None]).max(axis=0)
    translate = np.eye(4)
    translate[:3, 3] = obj.origin
    root_matrix = _matrix(scene["root"].get("transform")) @ translate
    document = {
        "asset": {"version": "1.1"},
        "geometricError": float(np.linalg.norm(high - low)),
        **{k: scene[k] for k in ("extensionsUsed", "extensions") if k in scene},
        "root": {
            "transform": _transform(root_matrix),
            "refine": "REPLACE",
            "boundingVolume": {"box": _box(low, high)},
            "geometricError": 0.0,
            "content": {"uri": OBJECT_CONTENT},
            "extras": {
                "gaussians": len(obj.spz),
                "object": {
                    "format": OBJECT_FORMAT,
                    "version": VERSION,
                    "instance": obj.choice.instance,
                    "ids": obj.choice.ids,
                    "origin": [float(v) for v in obj.origin],
                    "frame": "the scene's local ENU, translated to origin (metres)",
                    "scene": scene_uri,
                    "fromTiles": obj.from_tiles,
                },
            },
        },
    }
    return document


# ------------------------------------------------------------------------------- re-binding


def rebind_skin(
    skin: dict, blob: bytes, changes: dict[str, tuple[str, np.ndarray]]
) -> tuple[dict, bytes]:
    """`skin.json` + `skin.bin` with every changed tile (old checksum -> new checksum, kept
    gaussians) re-bound: its runs without the removed gaussians and its rows without theirs.
    Rows are re-packed in the new checksums' order, as `skin_scene` writes them."""
    from skin_scene import ROW_BYTES

    rows = np.frombuffer(blob, np.uint8).reshape(-1, ROW_BYTES)
    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for checksum, entry in skin["tiles"].items():
        which = decode_runs(entry["skins"])
        start = int(entry["row"])
        mine = rows[start : start + int((which > 0).sum())]
        if checksum in changes:
            new, keep = changes[checksum]
            kept_rows = mine[np.flatnonzero(keep[which > 0])] if mine.size else mine
            which = which[keep]
            checksum, mine = new, kept_rows
            if which.size == 0:
                continue
        blocks[checksum] = (which, mine)
    tiles, out, row = {}, [], 0
    for checksum in sorted(blocks):
        which, mine = blocks[checksum]
        tiles[checksum] = {"skins": scene_plants._rle(which), "row": row}
        out.append(mine)
        row += len(mine)
    document = {**skin, "tiles": tiles}
    document["weights"] = {**skin["weights"], "rows": row}
    data = np.concatenate(out).tobytes() if out else b""
    return document, data


# ------------------------------------------------------------------------------- the run


def _refuse_output(tiles_dir: Path, out_dir: Path) -> None:
    src, dst = tiles_dir.resolve(), out_dir.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError(f"{out_dir}: the output must be a directory of its own, not the input's")
    existing = out_dir / "tileset.json"
    if existing.exists():
        extras = json.loads(existing.read_text(encoding="utf-8"))["root"].get("extras", {})
        if extras.get("split", {}).get("format") != FORMAT:
            raise ValueError(f"{out_dir} holds a tileset split did not write; not overwriting it")


def _instances_path(tiles_dir: Path, tileset: dict, given: Path | None) -> Path:
    if given is not None:
        return given
    ref = tileset["root"].get("extras", {}).get("instances", {}).get("uri")
    if not ref:
        raise ValueError("the tileset declares no instances: pass --instances")
    return tiles_dir / ref


@dataclass
class Split:
    out_dir: Path
    objects: list[SplitObject]
    tiles: list[TileRef]
    report: dict


def split(
    tiles_dir: Path,
    out_dir: Path,
    *,
    ids: Sequence[int] | None = None,
    instances: Path | None = None,
    min_splats: int = MIN_SPLATS,
    max_extent_m: float = MAX_EXTENT_M,
    select: str = "movable",
    absorb_fragments: bool = False,
    fill: Callable[[Split], None] | None = None,
) -> Split:
    """Steps 1-3 and 5 of the module docstring into `out_dir`; `fill` (step 4) runs on the
    result before the tileset is written (`fill_holes`)."""
    _refuse_output(tiles_dir, out_dir)
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    instances_path = _instances_path(tiles_dir, tileset, instances)
    doc = json.loads(instances_path.read_text(encoding="utf-8"))
    choices = choose(doc, ids, min_splats=min_splats, max_extent_m=max_extent_m, select=select)
    if not choices:
        raise ValueError("no instance chosen: pass --ids, or loosen --min-splats/--max-extent-m")
    already = {int(o["instance"]) for o in tileset["root"].get("extras", {}).get("objects", [])}
    if clash := already.intersection(c.instance for c in choices):
        raise ValueError(f"instances {sorted(clash)} are already split in this tileset")
    max_id = max(int(i["id"]) for i in doc["instances"])
    tiles = read_tiles(tiles_dir, tileset, doc)
    absorbed = absorb(tiles, choices) if absorb_fragments else {}
    assign(tiles, choices, max_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    objects = [
        gather(tiles, k, c, f"{OBJECTS_DIR}/{c.instance}/tileset.json")
        for k, c in enumerate(choices)
    ]
    # The scene's tiles: unchanged ones copied, changed ones rewritten under a new name.
    replaced: set[str] = set()
    changes: dict[str, tuple[str, np.ndarray]] = {}
    bound = dict(doc["tiles"])
    removed = {"leaves": 0, "parents": 0}
    for t in tiles:
        keep = t.owner < 0
        if keep.all():
            continue
        removed["leaves" if t.leaf else "parents"] += int((~keep).sum())
        replaced.add(t.uri)
        bound.pop(t.checksum, None)
        kept = t.spz.take(np.flatnonzero(keep))
        if len(kept) == 0:
            del t.node["content"]
            changes[t.checksum] = ("", keep)
        else:
            checksum = checksum_positions(kept.positions())
            uri = _content_name(t.uri, checksum)
            (out_dir / uri).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / uri).write_bytes(tile_bytes(kept))
            t.node["content"]["uri"] = uri
            bound[checksum] = scene_plants._rle(t.labels[keep])
            changes[t.checksum] = (checksum, keep)
        if "gaussians" in t.node.get("extras", {}):
            t.node["extras"]["gaussians"] = len(kept)
    for obj in objects:
        bound[obj.checksum] = scene_plants._rle(obj.labels)

    # Everything else the scan carries comes along unchanged (sidecars, earlier objects).
    for path in sorted(tiles_dir.rglob("*")):
        rel = path.relative_to(tiles_dir).as_posix()
        if not path.is_file() or rel in replaced or rel == "tileset.json":
            continue
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    # instances.json (+ .emb), re-bound, beside the tiles.
    new_doc = {**doc, "tiles": dict(sorted(bound.items()))}
    (out_dir / "instances.json").write_text(
        json.dumps(new_doc, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    emb = instances_path.parent / doc.get("embedding", {}).get("file", "instances.emb")
    if emb.exists() and emb.resolve() != (out_dir / emb.name).resolve():
        shutil.copyfile(emb, out_dir / emb.name)
    extras = tileset["root"].setdefault("extras", {})
    extras["instances"] = {"uri": "instances.json", "count": len(doc["instances"])}

    # skin.json, when linked: the changed tiles re-bound (object tiles carry no skin yet).
    skin_ref = extras.get("skin", {}).get("uri")
    if skin_ref and (tiles_dir / skin_ref).exists():
        skin = json.loads((tiles_dir / skin_ref).read_text(encoding="utf-8"))
        blob = (tiles_dir / skin_ref).parent.joinpath(skin["weights"]["file"]).read_bytes()
        skin_doc, skin_bin = rebind_skin(skin, blob, {k: v for k, v in changes.items() if v[0]})
        (out_dir / skin_ref).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / skin_ref).write_text(json.dumps(skin_doc, indent=1), encoding="utf-8")
        (out_dir / skin_ref).parent.joinpath(skin["weights"]["file"]).write_bytes(skin_bin)

    for obj in objects:
        folder = out_dir / OBJECTS_DIR / str(obj.choice.instance)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / OBJECT_CONTENT).write_bytes(tile_bytes(obj.spz))
        document = object_tileset(obj, tileset, "../../tileset.json")
        document["root"]["extras"]["instances"] = {
            "uri": "../../instances.json",
            "count": len(doc["instances"]),
        }
        (folder / "tileset.json").write_text(json.dumps(document, indent=1), encoding="utf-8")

    report = {
        "objects": [
            {
                "instance": o.choice.instance,
                "ids": len(o.choice.ids),
                "splats": len(o.spz),
                "fromTiles": o.from_tiles,
                "origin": [round(float(v), 4) for v in o.origin],
                **({"absorbed": absorbed[o.choice.instance]} if absorbed else {}),
            }
            for o in objects
        ],
        "removed": removed,
        "tilesRewritten": len([c for c in changes.values() if c[0]]),
        "tilesEmptied": len([c for c in changes.values() if not c[0]]),
    }
    result = Split(out_dir, objects, tiles, report)
    # The scene's tileset, as it stands without the objects: what the fill reads.
    _write_tileset(out_dir, tileset, objects, tiles_dir, report)
    if fill is not None:
        fill(result)
        _write_tileset(out_dir, tileset, objects, tiles_dir, report)
    return result


def _write_tileset(
    out_dir: Path, tileset: dict, objects: Sequence[SplitObject], tiles_dir: Path, report: dict
) -> None:
    extras = tileset["root"].setdefault("extras", {})
    earlier = [
        o
        for o in extras.get("objects", [])
        if o["instance"] not in {x.choice.instance for x in objects}
    ]
    extras["objects"] = earlier + [
        {
            "uri": o.uri,
            "instance": o.choice.instance,
            "origin": [float(v) for v in o.origin],
            "pose": POSE_REST,
            "splats": len(o.spz),
            **({"fill": o.fill["uri"]} if o.fill else {}),
        }
        for o in objects
    ]
    layers = [
        layer
        for layer in extras.get("inferredLayers", [])
        if layer["uri"] not in {o.fill["uri"] for o in objects if o.fill}
    ]
    for o in objects:
        if o.fill:
            layers.append({"uri": o.fill["uri"], "evidence": o.fill["evidence"]})
    if layers:
        extras["inferredLayers"] = layers
    extras["split"] = {
        "format": FORMAT,
        "version": VERSION,
        "source": tiles_dir.resolve().name,
        "removed": report["removed"],
        "rule": (
            "leaf gaussians by instance id; a merged parent's by its id, or unbound by the "
            f"majority of its {BIND_NEIGHBOURS} nearest leaves; tiles sliced, never re-quantised"
        ),
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------- the fill


def hole_context(
    instances: Sequence[dict], positions: np.ndarray, obj: SplitObject
) -> dict[str, object]:
    """What a generative filler is told about an object's hole: the tags of what lies around
    its footprint at its base (`teacher_fill.describe_surroundings` over the box the support
    plane is fitted in), never its own or its fragments' (they are the negative prompt)."""
    import teacher_fill as tf

    low, high = positions.min(axis=0), positions.max(axis=0)
    centre, half = (low + high) / 2, float(np.max(high - low)) / 2
    reach = tf.SURROUND_SCALE * half
    box_low = np.array([centre[0] - reach, centre[1] - reach, low[2] - half])
    box_high = np.array(
        [centre[0] + reach, centre[1] + reach, low[2] + tf.HOLE_BASE_SHARE * (high[2] - low[2])]
    )
    fragments = sorted(int(k) for k in np.unique(obj.labels) if k > 0)
    return tf.describe_surroundings(
        instances, box_low, box_high, object_ids=obj.choice.ids, exclude_ids=fragments
    )


def fill_holes(
    filler_spec: str = "telea",
    *,
    renderer: str = "cpu",
    views: int = 6,
    width: int = 480,
    height: int = 360,
    stride: int = 2,
    distill: int = 0,
    distill_on: str = "local",
    max_scale_m: float | None = None,
    save_dir: Path | None = None,
) -> Callable[[Split], None]:
    """Step 4 as `split`'s `fill`: per object, `teacher_fill.fill_hole` on the scene without
    it, packaged as an inferred layer in `fills/<id>/`."""

    def run(result: Split) -> None:
        import teacher_fill as tf
        import view_cones as vc
        from splat_render import Splats, _from_columns, load_tileset
        from splat_tiles import unpack_spz

        out = result.out_dir
        scene_tileset = out / "tileset.json"
        kept = load_tileset(scene_tileset)
        if max_scale_m is not None:
            kept = kept.take(np.flatnonzero(kept.scales.max(axis=1) <= max_scale_m))
        grid = vc.cone_grid_from_tileset(scene_tileset)
        filler = tf.make_filler(filler_spec)
        instances = tf.read_instances(scene_tileset)
        runner = None
        if distill:
            runner = tf._distill_runner(distill_on)
        for obj in result.objects:
            columns = unpack_spz(write_spz(obj.scene_spz()))
            removed: Splats = _from_columns(columns)
            save = save_dir / str(obj.choice.instance) if save_dir else None
            context = None
            if hasattr(filler, "context"):
                context = hole_context(instances, removed.positions, obj)
            lifted, confidence, cameras, report = tf.fill_hole(
                kept,
                removed,
                filler,
                grid,
                views=views,
                width=width,
                height=height,
                stride=stride,
                renderer=tf.make_renderer(renderer),
                save_dir=save,
                distill_iterations=distill,
                distill_runner=runner,
                context=context,
            )
            obj.fill = {"report": report}
            if len(lifted) == 0:
                continue
            uri = f"{FILLS_DIR}/{obj.choice.instance}/tileset.json"
            evidence = tf.package_inferred(
                lifted,
                confidence,
                cameras,
                scene_tileset,
                out / FILLS_DIR / str(obj.choice.instance),
                filler.name,
                rule=tf.HOLE_RULE,
                extra={"hole": obj.choice.instance},
            )
            obj.fill.update({"uri": uri, "evidence": evidence})
        result.report["fills"] = {
            str(o.choice.instance): (o.fill or {}).get("report") for o in result.objects
        }

    return run


# ------------------------------------------------------------------------------- the CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cand = sub.add_parser("candidates", help="list what split would choose")
    cand.add_argument("tiles", type=Path, help="the tileset directory")
    run = sub.add_parser("split", help="split chosen objects out, fill their holes")
    run.add_argument("tiles", type=Path, help="the tileset directory (never written)")
    run.add_argument("out", type=Path, help="a new directory for the split tileset")
    for p in (cand, run):
        p.add_argument("--instances", type=Path, default=None, help="default: root.extras")
        p.add_argument("--ids", default=None, help="comma-separated instance ids")
        p.add_argument("--min-splats", type=int, default=MIN_SPLATS)
        p.add_argument("--max-extent-m", type=float, default=MAX_EXTENT_M)
        p.add_argument("--select", choices=("movable", "loose"), default="movable")
    run.add_argument("--no-fill", action="store_true", help="leave the holes")
    run.add_argument(
        "--absorb", action="store_true", help="take fragments under other ids (`absorb`)"
    )
    run.add_argument("--filler", default="telea", help="'telea' or module:Class[?k=v]")
    run.add_argument("--renderer", choices=("cpu", "gsplat"), default="cpu")
    run.add_argument("--views", type=int, default=6)
    run.add_argument("--width", type=int, default=480)
    run.add_argument("--height", type=int, default=360)
    run.add_argument("--stride", type=int, default=2)
    run.add_argument("--distill", type=int, default=0, help="refine the fill this many steps")
    run.add_argument("--distill-on", choices=("local", "modal"), default="local")
    run.add_argument("--max-scale-m", type=float, default=None, help="condition without floaters")
    run.add_argument("--save", type=Path, default=None, help="strips per object")
    args = parser.parse_args(argv)
    ids = [int(v) for v in args.ids.split(",") if v.strip()] if args.ids else None
    if args.command == "candidates":
        tileset = json.loads((args.tiles / "tileset.json").read_text(encoding="utf-8"))
        doc = json.loads(_instances_path(args.tiles, tileset, args.instances).read_text())
        chosen = choose(
            doc, ids, min_splats=args.min_splats, max_extent_m=args.max_extent_m,
            select=args.select,
        )  # fmt: skip
        print(
            json.dumps(
                [
                    {"instance": c.instance, "ids": len(c.ids), "splats": c.splats,
                     "extentM": round(c.extent, 3)}
                    for c in chosen
                ],
                indent=1,
            )
        )  # fmt: skip
        return 0
    fill = None
    if not args.no_fill:
        fill = fill_holes(
            args.filler,
            renderer=args.renderer,
            views=args.views,
            width=args.width,
            height=args.height,
            stride=args.stride,
            distill=args.distill,
            distill_on=args.distill_on,
            max_scale_m=args.max_scale_m,
            save_dir=args.save,
        )
    result = split(
        args.tiles,
        args.out,
        ids=ids,
        instances=args.instances,
        min_splats=args.min_splats,
        max_extent_m=args.max_extent_m,
        select=args.select,
        absorb_fragments=args.absorb,
        fill=fill,
    )
    print(json.dumps(result.report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
