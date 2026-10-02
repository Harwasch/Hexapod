"""What a segmentation covers, and how it looks, against another (docs/SCENE_OBJECTS.md §3).

For a scan's tileset and one or more `instances.json` (a published one and a new run, say):

* `metrics` -- the share of splats without an instance over every tile, over the leaf tiles,
  per tile depth, and over the **rim** (leaf tiles at least `RIM_LEVELS` shallower than the
  median leaf splat's: the coarse edge of a capture, which a view of the whole scan is mostly
  made of); and each category's share of the leaf splats. Nothing in it knows the scene.
* `contact_sheet` -- per camera, the scan, then each document coloured by object (an
  instance's top-level ancestor, `segment_scene._colours`) and by category (the category's
  own colour), splats without an instance in magenta; drawn by any `SplatRenderer` (gsplat on
  a GPU, as a viewer draws it).

    python instances_report.py TILES_DIR before.json after.json [--sheet out.png] [--gsplat]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import rebind_instances
import scene_categories
import segment_scene
from rig_tiles import tile_positions
from splat_render import Camera, Splats, load_tileset
from synthetic_tree import checksum_positions

#: The rim: leaf tiles at least this many levels shallower than the median leaf splat's.
RIM_LEVELS = 2
#: Splats without an instance, in the sheets.
UNASSIGNED = (1.0, 0.0, 1.0)
SHEET_SIZE = (480, 360)


def tile_depths(tileset: dict) -> list[tuple[str, int, bool]]:
    """Every content uri with its depth and whether it is a leaf, depth first."""
    out: list[tuple[str, int, bool]] = []

    def has_content(tile: dict) -> bool:
        return bool(tile.get("content", {}).get("uri")) or any(
            has_content(c) for c in tile.get("children", [])
        )

    def walk(tile: dict, depth: int) -> None:
        uri = tile.get("content", {}).get("uri")
        if uri:
            out.append((uri, depth, not any(has_content(c) for c in tile.get("children", []))))
        for child in tile.get("children", []):
            walk(child, depth + 1)

    walk(tileset["root"], 0)
    return out


def tile_checksums(tiles_dir: Path) -> dict[str, str]:
    """Per content uri, its position checksum (the binding's key)."""
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    return {
        uri: checksum_positions(tile_positions(tiles_dir / uri))
        for uri, _, _ in tile_depths(tileset)
    }


def categories_of(document: dict) -> dict[int, str]:
    """Per instance id, its category: the file's own, else the tags' rule."""
    records = document["instances"]
    if all(r.get("category") for r in records):
        return {int(r["id"]): str(r["category"]) for r in records}
    return scene_categories.instance_categories(records, scene_categories.load())


def metrics(tiles_dir: Path, document: dict, checksums: dict[str, str] | None = None) -> dict:
    """Unassigned shares (all tiles, leaves, per depth, rim) and category shares of leaves."""
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    checksums = checksums or tile_checksums(tiles_dir)
    category = categories_of(document)
    rows = []
    for uri, depth, leaf in tile_depths(tileset):
        ids = rebind_instances.decode_runs(document["tiles"][checksums[uri]])
        rows.append((depth, leaf, ids))
    leaf_total = np.array([[d, ids.size] for d, leaf, ids in rows if leaf])
    order = np.argsort(leaf_total[:, 0], kind="stable")
    cumulative = np.cumsum(leaf_total[order, 1])
    median_depth = int(leaf_total[order][np.searchsorted(cumulative, cumulative[-1] / 2), 0])
    rim_depth = median_depth - RIM_LEVELS

    def share(selected) -> dict[str, float]:
        total = sum(ids.size for ids in selected)
        zero = sum(int((ids == 0).sum()) for ids in selected)
        return {"splats": int(total), "unassigned": round(zero / max(total, 1), 4)}

    by_depth = {}
    for depth in sorted({d for d, _, _ in rows}):
        for leaf in (True, False):
            chosen = [ids for d, lf, ids in rows if d == depth and lf == leaf]
            if chosen:
                by_depth[f"{depth}{'L' if leaf else 'P'}"] = share(chosen)
    leaves = [ids for _, leaf, ids in rows if leaf]
    counts: dict[str, int] = {}
    for ids in leaves:
        values, n = np.unique(ids, return_counts=True)
        for v, c in zip(values.tolist(), n.tolist(), strict=True):
            key = category.get(int(v), "none") if v else "none"
            counts[key] = counts.get(key, 0) + c
    total = max(sum(counts.values()), 1)
    return {
        "instances": len(document["instances"]),
        "all": share([ids for _, _, ids in rows]),
        "leaves": share(leaves),
        "rim": {
            "depthAtMost": rim_depth,
            **share([i for d, lf, i in rows if lf and d <= rim_depth]),
        },
        "byDepth": by_depth,
        "categories": {
            k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        },
    }


def leaf_ids(tiles_dir: Path, document: dict, checksums: dict[str, str]) -> np.ndarray:
    """Per leaf splat, in `splat_render.load_tileset`'s order (leaf uris sorted), its id."""
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    leaves = sorted(uri for uri, _, leaf in tile_depths(tileset) if leaf)
    return np.concatenate(
        [rebind_instances.decode_runs(document["tiles"][checksums[uri]]) for uri in leaves]
    )


def _hex(colour: str) -> tuple[float, float, float]:
    return tuple(int(colour[k : k + 2], 16) / 255 for k in (1, 3, 5))  # type: ignore[return-value]


def colourings(document: dict, ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per splat, its colour by object (top-level instance) and by category."""
    parent = np.array([int(r["parent"] or 0) for r in document["instances"]], np.int64)
    top = segment_scene.top_level(parent)[ids]
    by_object = segment_scene._colours(top)
    category = categories_of(document)
    palette = {c.id: _hex(c.color) for c in scene_categories.CATEGORIES}
    table = np.zeros((len(document["instances"]) + 1, 3))
    table[0] = UNASSIGNED
    for k, c in category.items():
        table[k] = palette.get(c, palette[scene_categories.OTHER])
    by_object[ids == 0] = UNASSIGNED
    return {"objects": by_object, "categories": table[ids]}


def sheet_cameras(positions: np.ndarray, size: tuple[int, int] = SHEET_SIZE) -> list[Camera]:
    """The whole scan from two sides (45 and 30 degrees up), and its middle from closer."""
    _, lo, hi = segment_scene._extent(positions)
    centre = (lo + hi) / 2
    radius = max(0.5 * float(np.linalg.norm((hi - lo)[:2])), 1e-3)
    distance = radius / math.tan(math.radians(30))
    out = []
    for azimuth, elevation, scale in ((-90, 45, 1.0), (45, 30, 1.0), (200, 60, 0.45)):
        a, e = math.radians(azimuth), math.radians(elevation)
        direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        out.append(
            Camera.look_at(
                centre + distance * scale * direction, centre, fov_deg=60,
                width=size[0], height=size[1],
            )
        )  # fmt: skip
    return out


def contact_sheet(
    splats: Splats,
    panels: Sequence[tuple[str, np.ndarray]],
    cameras: Sequence[Camera],
    renderer,
    out: Path,
) -> None:
    """One row per camera: the scan, then each `(title, per-splat colours)` panel."""
    from PIL import Image, ImageDraw

    titled = [("scan", splats.colours), *panels]
    background = (0.12, 0.12, 0.12)
    rows = []
    for title, colours in titled:
        painted = Splats(
            splats.positions, splats.rotations, splats.scales, colours, splats.opacities
        )
        column = []
        for camera in cameras:
            frame = renderer(painted, camera, background=background)
            column.append(np.round(np.clip(frame.rgb, 0, 1) * 255).astype(np.uint8))
        image = Image.fromarray(np.concatenate(column, axis=0))
        ImageDraw.Draw(image).text((8, 6), title, fill=(255, 255, 255))
        rows.append(np.asarray(image))
    Image.fromarray(np.concatenate(rows, axis=1)).save(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tiles", type=Path, help="the scan's tileset directory")
    parser.add_argument("documents", type=Path, nargs="+", help="instances.json files")
    parser.add_argument("--sheet", type=Path, default=None, help="a contact sheet PNG")
    parser.add_argument("--gsplat", action="store_true", help="draw the sheet with gsplat")
    parser.add_argument("--out", type=Path, default=None, help="the metrics as JSON")
    args = parser.parse_args(argv)
    checksums = tile_checksums(args.tiles)
    documents = [json.loads(p.read_text(encoding="utf-8")) for p in args.documents]
    report = {str(p): metrics(args.tiles, d, checksums) for p, d in zip(args.documents, documents)}
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.sheet:
        splats = load_tileset(args.tiles / "tileset.json")
        panels = []
        for path, document in zip(args.documents, documents, strict=True):
            painted = colourings(document, leaf_ids(args.tiles, document, checksums))
            name = path.parent.name or path.stem
            panels += [(f"{name}: objects", painted["objects"])]
            panels += [(f"{name}: categories", painted["categories"])]
        renderer = segment_scene.make_renderer("gsplat") if args.gsplat else None
        contact_sheet(
            splats, panels, sheet_cameras(splats.positions), renderer or segment_scene.CpuRenderer(),
            args.sheet,
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
