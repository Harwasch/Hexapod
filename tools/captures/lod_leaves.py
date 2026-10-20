"""The leaves of a published splat tileset, for splat-transform to merge into PlayCanvas's
streamed level of detail (.github/workflows/streamed-lod-backfill.yml).

The leaves hold every packaged gaussian once. Their SPZ positions are each in its tile's own
frame: the site frame for every tile of a scan under ~4 km, and for a tile of a larger site
its own origin, a translation-only `transform` relative to its parent's
(`splat_tiles.tile_frame`). splat-transform knows nothing of tile transforms, so a leaf with
an offset is moved by its own `--translate` before the merge (`plan`), and the merged PLY is
checked leaf by leaf against the leaves' site-frame boxes before anything is built from it
(`check`).

**The sign of the translate.** splat-transform 3.8.0 (@playcanvas/splat-transform, MIT) reads
an SPZ as data in `Transform.PLY` space -- a 180-degree turn about z (dist/index.mjs
`Transform.PLY`, `readSpz`); applies `--translate t` in its own space, as `T(t) * transform`
(`processDataTable`); and every writer bakes back into PLY space (`bakeTransform`). So the
stored positions move by `Rz(180) t = (-tx, -ty, tz)`, and to move a leaf by its offset `o`
in our east-north-up frame the flag is `--translate=-ox,-oy,oz`. The `=` form, because a value
starting with `-` would otherwise be read as an option. Derived from the source, then run
(2026-10-19, splat-transform 3.8.0 from npm, the workflow's own commands on
data/tiles/synthetic-far): the merge held every one of the 7,901 gaussians where the scene
put it (within 6e-13 m: a half turn's sine is not exactly 0), in input order. `check` stays
in the workflow and checks every leaf's own run of rows, so a later splat-transform that
changes its frames or its order fails the run instead of publishing a scrambled site.

Usage:
    python lod_leaves.py fetch SOURCE OUT_DIR    # SOURCE: https://.../tileset.json or a dir
    python lod_leaves.py check OUT_DIR FULL_PLY
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from splat_tiles import read_ply, tile_offsets

USER_AGENT = "streamed-lod-backfill"
#: How far a leaf's run of merged rows may stand from that leaf's site-frame box (m): the
#: accessor bounds are float32 of the stored positions, and splat-transform writes float32
#: too, so the two agree to within a float32 step at 5 km (~0.0005 m), by construction and
#: not by calibration. A leaf in the wrong frame is off by its whole-metre offset, or twice it.
CHECK_TOLERANCE_M = 0.01


def _leaves(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        if not tile.get("children"):
            out.append(tile)
    return out


def _name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].removesuffix(".glb")


def _translate(offset: np.ndarray) -> str | None:
    """splat-transform's `--translate` value that moves a leaf by `offset` (see the module
    docstring): x and y negated, z as it is; None for a leaf in the site frame."""
    if not offset.any():
        return None

    def number(v: float) -> str:
        v += 0.0  # no "-0"
        return str(int(v)) if v.is_integer() else repr(v)

    x, y, z = (float(v) for v in offset)
    return f"{number(-x)},{number(-y)},{number(z)}"


def plan(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every leaf, sorted by name (the order `spz/*.spz` globbed in): its uri, the file name
    it is fetched to, its composed offset and the translate that puts it in the site frame."""
    offsets = tile_offsets(document)
    rows = []
    for tile in _leaves(document):
        uri = tile["content"]["uri"].removeprefix("./")
        offset = offsets.get(uri, np.zeros(3))
        rows.append(
            {
                "uri": uri,
                "name": _name(uri),
                "offset": [float(v) + 0.0 for v in offset],
                "translate": _translate(offset),
                "gaussians": int(tile.get("extras", {}).get("gaussians", 0)),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _glb_parts(data: bytes) -> tuple[dict[str, Any], bytes]:
    """A splat GLB's JSON and its SPZ block (as `splat_tiles.build_glb` lays it out)."""
    if data[:4] != b"glTF":
        raise ValueError("not a GLB")
    length = struct.unpack_from("<I", data, 12)[0]
    doc: dict[str, Any] = json.loads(data[20 : 20 + length])
    offset = 20 + length
    blob = data[offset + 8 : offset + 8 + struct.unpack_from("<I", data, offset)[0]]
    ext = doc["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"]
    view = doc["bufferViews"][
        ext["extensions"]["KHR_gaussian_splatting_compression_spz_2"]["bufferView"]
    ]
    start = view.get("byteOffset", 0)
    return doc, blob[start : start + view["byteLength"]]


def _reader(source: str) -> tuple[Callable[[str], bytes], str]:
    """How to read a file beside the tileset, and the tileset's own name, for a URL or a
    directory."""
    if source.startswith("https://"):
        base = source.rsplit("/", 1)[0] + "/"

        def get(rel: str) -> bytes:
            request = urllib.request.Request(base + rel, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=600) as response:
                data: bytes = response.read()
                return data

        return get, source.rsplit("/", 1)[1]
    folder = Path(source)
    if folder.is_file():
        folder, name = folder.parent, folder.name
    else:
        name = "tileset.json"
    return (lambda rel: (folder / rel).read_bytes()), name


def fetch(source: str, out_dir: Path) -> dict[str, Any]:
    """Every leaf's SPZ to `out_dir/spz/<name>.spz`, and `out_dir/leaves.json`: the plan, and
    each leaf's site-frame box (its accessor bounds plus its offset), which `check` reads."""
    get, name = _reader(source)
    document = json.loads(get(name))
    rows = plan(document)
    (out_dir / "spz").mkdir(parents=True, exist_ok=True)
    for row in rows:
        doc, spz = _glb_parts(get(row["uri"]))
        (out_dir / "spz" / f"{row['name']}.spz").write_bytes(spz)
        position = doc["accessors"][0]
        offset = np.asarray(row["offset"], np.float64)
        row["min"] = [float(v) for v in np.asarray(position["min"], np.float64) + offset]
        row["max"] = [float(v) for v in np.asarray(position["max"], np.float64) + offset]
        row["count"] = int(position["count"])
    summary = {"tileset": source, "leaves": rows}
    (out_dir / "leaves.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def check(out_dir: Path, full_ply: Path) -> dict[str, Any]:
    """Refuses (`SystemExit`) a merged PLY that does not hold every leaf where its tile's
    transform puts it. splat-transform concatenates its inputs in the order given (the
    order of leaves.json, `plan`'s) and keeps every gaussian, so the merge must have exactly
    the leaves' count, and each leaf's run of rows must have that leaf's own site-frame box.
    One leaf merged in the wrong frame moves its run's box by its offset (at least
    `TILE_FIT_M`); a leaf missing or duplicated changes the count. A splat-transform that
    ever reorders its inputs is refused too: the check fails safe, it never passes a merge
    it cannot place."""
    leaves = json.loads((out_dir / "leaves.json").read_text(encoding="utf-8"))["leaves"]
    want_low = np.min([row["min"] for row in leaves], axis=0)
    want_high = np.max([row["max"] for row in leaves], axis=0)
    columns = read_ply(full_ply)
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
    counts = [int(row.get("count", row["gaussians"])) for row in leaves]
    report: dict[str, Any] = {
        "gaussians": int(xyz.shape[0]),
        "leavesGaussians": int(sum(counts)),
        "min": xyz.min(axis=0).tolist() if len(xyz) else None,
        "max": xyz.max(axis=0).tolist() if len(xyz) else None,
        "leavesMin": want_low.tolist(),
        "leavesMax": want_high.tolist(),
    }
    if xyz.shape[0] != sum(counts):
        raise SystemExit(
            f"{full_ply.name} holds {xyz.shape[0]} gaussians, the leaves {sum(counts)}: "
            "a leaf is missing from the merge, or merged twice"
        )
    error, worst = 0.0, ""
    start = 0
    for row, count in zip(leaves, counts, strict=True):
        part = xyz[start : start + count]
        start += count
        if not count:
            continue
        off = float(
            max(
                np.abs(part.min(axis=0) - np.asarray(row["min"], np.float64)).max(),
                np.abs(part.max(axis=0) - np.asarray(row["max"], np.float64)).max(),
            )
        )
        if off > error:
            error, worst = off, row["name"]
    report["errorM"] = error
    report["worstLeaf"] = worst or None
    if error > CHECK_TOLERANCE_M:
        raise SystemExit(
            f"{full_ply.name}: leaf {worst}'s gaussians are {error:.3f} m from its box in the "
            "site frame: a leaf was merged in the wrong frame (the --translate of a tile "
            "with an origin), or out of order"
        )
    return report


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetching = sub.add_parser("fetch")
    fetching.add_argument("source")
    fetching.add_argument("out_dir", type=Path)
    checking = sub.add_parser("check")
    checking.add_argument("out_dir", type=Path)
    checking.add_argument("full_ply", type=Path)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        rows = fetch(args.source, args.out_dir)["leaves"]
        moved = sum(1 for row in rows if row["translate"])
        count = sum(row["gaussians"] for row in rows)
        print(f"{len(rows)} leaves, {count} gaussians, {moved} translated into the site frame")
    else:
        print(json.dumps(check(args.out_dir, args.full_ply)))


if __name__ == "__main__":
    main(sys.argv[1:])
