"""A synthetic site 10.4 km across, so per-tile origins have a fixture to be held to.

SPZ stores positions as 24-bit fixed point: +/-2047.99976 m from a tile's frame at 12
fractional bits. Every scan this repo had packed fitted, so nothing exercised what happens
past it -- which, until per-tile origins, was a silent wrap (docs/DECISIONS/
0011-splat-tile-origins.md). This scene is the smallest one that does: a track 10.4 km
east-west, a spur 1.8 km north from its east end, and three small dense plots, one at the
site origin, one 2.6 km west, one 2.6 km east and 1.8 km north, which a viewer can look for.

Deterministic with no RNG: every gaussian is placed by formula (the plots on golden-angle
discs) and snapped to the 1/4096 m grid, and every coordinate is under 4096 m or a whole
number, so float32 holds each one exactly.

It emits, into ``<out_dir>``::

    splat/tileset.json, splat/*.glb, collision.bin, viewcones.bin   splat_tiles.convert
    source/expected.json   what a reader must find: per tile its composed offset, its
                           fractional bits, its gaussian count and the site-frame checksum
                           of its positions; the plots and the leaves that draw them; the
                           site-frame box of everything

The PLY is written to a temporary directory and never committed: `scene()` is its source.
Parents go through `eigh`, so the bytes depend on the OpenBLAS kernel; tests/conftest.py
pins Sandybridge, and the fixture is written under the same pin.

Usage:
    OPENBLAS_CORETYPE=Sandybridge python synthetic_far.py ../../data/tiles/synthetic-far
    python synthetic_far.py report ../../data/tiles/synthetic-far/splat
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import splat_tiles
from rig_tiles import glb_spz, tile_uris
from splat_render import Splats, save_ply
from synthetic_tree import checksum_positions

FORMAT = 1
GENERATOR = "tools/captures/synthetic_far.py"

LAT, LON, HEIGHT = 28.0389, -82.6966, 0.0
#: Gaussians a tile: small, so a 10.4 km site is a tree several levels deep.
TILE_GAUSSIANS = 1000
#: Where the plots stand (east, north, up), metres in the site frame.
PLOTS: dict[str, tuple[float, float, float]] = {
    "origin": (0.0, 0.0, 1.0),
    "west": (-2600.0, 0.0, 1.0),
    "north-east": (2600.0, 1800.0, 1.0),
}
PLOT_GAUSSIANS = 600
PLOT_RADIUS_M = 3.0
PLOT_COLOURS: dict[str, tuple[float, float, float]] = {
    "origin": (0.95, 0.15, 0.15),
    "west": (0.15, 0.85, 0.2),
    "north-east": (0.2, 0.4, 0.95),
}
TRACK_COLOUR = (0.5, 0.5, 0.5)
SCALE_M = 0.4
OPACITY = 0.9
#: SPZ's grid at 12 fractional bits.
QUANTUM = 1.0 / 4096


def _disc(centre: tuple[float, float, float]) -> np.ndarray:
    """`PLOT_GAUSSIANS` points on a golden-angle (Vogel) disc of `PLOT_RADIUS_M`."""
    k = np.arange(PLOT_GAUSSIANS, dtype=np.float64)
    radius = PLOT_RADIUS_M * np.sqrt((k + 0.5) / PLOT_GAUSSIANS)
    angle = k * math.pi * (3.0 - math.sqrt(5.0))
    x, y, z = centre
    return np.stack(
        [x + radius * np.cos(angle), y + radius * np.sin(angle), np.full_like(k, z)], axis=1
    )


def scene() -> Splats:
    """The track, the spur and the plots, in that order."""
    east = np.arange(-5200.0, 5200.0 + 1.0, 2.0)
    track = np.stack([east, np.zeros_like(east), np.zeros_like(east)], axis=1)
    north = np.arange(2.0, 1800.0 + 1.0, 2.0)
    spur = np.stack([np.full_like(north, 2600.0), north, np.zeros_like(north)], axis=1)
    discs = [_disc(centre) for centre in PLOTS.values()]
    positions = np.concatenate([track, spur, *discs])
    positions = np.round(positions / QUANTUM) * QUANTUM
    colours = np.concatenate(
        [
            np.tile(TRACK_COLOUR, (track.shape[0] + spur.shape[0], 1)),
            *(np.tile(PLOT_COLOURS[name], (PLOT_GAUSSIANS, 1)) for name in PLOTS),
        ]
    )
    n = positions.shape[0]
    return Splats(
        positions.astype(np.float32),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        np.full((n, 3), SCALE_M, np.float32),
        colours.astype(np.float32),
        np.full(n, OPACITY, np.float32),
    )


def _decoded(path: Path) -> tuple[np.ndarray, int]:
    """A tile's stored (tile-local) positions, float32, and its header's fractional bits."""
    blob = glb_spz(path)
    columns = splat_tiles.unpack_spz(blob)
    local = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype("<f4")
    local[local == 0.0] = 0.0
    return local, gzip.decompress(blob)[13]


def _plot_leaves(tiles_dir: Path, tileset: dict, offsets: dict[str, np.ndarray]) -> dict:
    """Per plot, the leaves that draw any of its gaussians (site frame, at its height)."""
    leaves: list[str] = []
    stack = [tileset["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
        else:
            leaves.append(tile["content"]["uri"])
    out: dict[str, list[str]] = {}
    for name, (x, y, z) in PLOTS.items():
        found = []
        for uri in sorted(leaves):
            local, _bits = _decoded(tiles_dir / uri)
            site = local.astype(np.float64) + offsets[uri]
            near = (np.hypot(site[:, 0] - x, site[:, 1] - y) <= PLOT_RADIUS_M + QUANTUM) & (
                np.abs(site[:, 2] - z) < QUANTUM
            )
            if near.any():
                found.append(uri)
        out[name] = found
    return out


def generate(out_dir: Path) -> dict:
    """Write the fixture (see the module docstring) and return `splat_tiles.convert`'s stats."""
    splat_dir = out_dir / "splat"
    source = out_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    splats = scene()
    with tempfile.TemporaryDirectory(prefix=".synthetic_far-") as scratch:
        ply = Path(scratch) / "far.ply"
        save_ply(ply, splats, comment=GENERATOR)
        stats = splat_tiles.convert(
            ply,
            splat_dir,
            LAT,
            LON,
            HEIGHT,
            tile_gaussians=TILE_GAUSSIANS,
            work_dir=Path(scratch),
        )
    if stats["dropped"]:
        raise SystemExit(f"the packer dropped {stats['dropped']} gaussians; nothing should go")
    tileset = json.loads((splat_dir / "tileset.json").read_text(encoding="utf-8"))
    offsets = splat_tiles.tile_offsets(tileset)
    tiles = {}
    for uri in tile_uris(tileset):
        local, bits = _decoded(splat_dir / uri)
        site = splat_tiles.site_positions(local, offsets[uri])
        tiles[uri] = {
            "offset": [float(v) for v in offsets[uri]],
            "fractionalBits": bits,
            "gaussians": int(local.shape[0]),
            "checksum": checksum_positions(site),
        }
    positions = splats.positions.astype(np.float64)
    expected = {
        "format": FORMAT,
        "generator": GENERATOR,
        "lat": LAT,
        "lon": LON,
        "height": HEIGHT,
        "tileGaussians": TILE_GAUSSIANS,
        "plots": {name: list(centre) for name, centre in PLOTS.items()},
        "plotLeaves": _plot_leaves(splat_dir, tileset, offsets),
        "bboxSiteM": {
            "min": positions.min(axis=0).tolist(),
            "max": positions.max(axis=0).tolist(),
        },
        "tiles": tiles,
    }
    (source / "expected.json").write_text(
        json.dumps(expected, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def report(tiles_dir: Path) -> list[dict]:
    """Per tile of a packed far scene: where its frame is, how precisely it stores
    positions, and how far its decoded positions land from what the packer was given (the
    tree `splat_tiles.hierarchy` builds from `scene()`, in the site frame). The last line
    is the summary. Every `maxDecodeErrorM` is at most half its tile's `quantumM`."""
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    offsets = splat_tiles.tile_offsets(tileset)
    own: dict[str, list[float] | None] = {}
    stack = [tileset["root"]]
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        transform = tile.get("transform") if tile is not tileset["root"] else None
        own[tile["content"]["uri"]] = None if transform is None else transform[12:15]
    truth: dict[str, np.ndarray] = {}

    def emit(tile: splat_tiles.Tile, gaussians: splat_tiles.Gaussians, _keys: object) -> None:
        truth[tile.uri] = gaussians.xyz.astype(np.float64)

    with tempfile.TemporaryDirectory(prefix=".synthetic_far-") as scratch:
        ply = Path(scratch) / "far.ply"
        save_ply(ply, scene(), comment=GENERATOR)
        splat_tiles.hierarchy(ply, 0.02, TILE_GAUSSIANS, Path(scratch), emit)
    rows = []
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for uri in tile_uris(tileset):
        local, bits = _decoded(tiles_dir / uri)
        site = local.astype(np.float64) + offsets[uri]
        low, high = np.minimum(low, site.min(axis=0)), np.maximum(high, site.max(axis=0))
        rows.append(
            {
                "uri": uri,
                "origin": [float(v) for v in offsets[uri]],
                "transform": own[uri],
                "bits": bits,
                "quantumM": 2.0**-bits,
                "gaussians": int(local.shape[0]),
                "maxDecodeErrorM": float(np.abs(site - truth[uri]).max()),
            }
        )
    rows.append(
        {
            # Of the decoded centres in the site frame, as `convert`'s `extent_m` is.
            "extentM": float((high - low).max()),
            "offsetTiles": sum(1 for row in rows if any(row["origin"])),
            "minBits": min(row["bits"] for row in rows),
            "worstErrorOverHalfQuantum": max(
                row["maxDecodeErrorM"] / (row["quantumM"] / 2) for row in rows
            ),
        }
    )
    return rows


def main(argv: list[str]) -> None:
    if argv[:1] == ["report"]:
        parser = argparse.ArgumentParser(prog="synthetic_far.py report")
        parser.add_argument("tiles_dir", type=Path)
        args = parser.parse_args(argv[1:])
        for row in report(args.tiles_dir):
            print(json.dumps(row))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.out_dir)))


if __name__ == "__main__":
    main(sys.argv[1:])
