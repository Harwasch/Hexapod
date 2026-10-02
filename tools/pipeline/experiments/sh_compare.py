"""One site packaged three ways -- SH degree 0, 1 and 3 -- side by side, to look at.

**Why.** gsplat trains spherical harmonics to degree 3 and, until `ship_sh_degree` existed,
every capture shipped degree 0: one colour from every side. Whether the view-dependent
colour is worth its cost is a question of what it looks like and what it costs *on the
devices that draw it* -- frame rate, GPU memory, how long the tiles take to arrive -- and
none of that can be measured on this repository's machines (headless GL is SwiftShader).
So this packs one splat at each degree with the `package` stage's own packer and
parameters, writes the three tilesets beside each other with a small manifest of what
each costs on disk, and (with `--site`) as three sites a few scan-widths apart on the
globe, so the owner can fly between them in the console and read the developer panel.

**What it needs: a PLY that still has its `f_rest_*`.** Run outputs do not keep one
(tools/pipeline/README.md, "Shipping SH"): gsplat's own export is in the training
container's `work/`, deleted with it, and a block run's parts leave `checkpoint/` once
merged. So the usual source is a run trained with `ship_sh_degree: 3`
(`{"train": {"ship_sh_degree": 3}}`), whose `stages/place/out/canonical.ply` is already in
east/north/up with its SH turned there -- pass it as is, with `--georef` for where it
goes. Two other sources work, both in COLMAP's frame and so placed first with
`--place` -- which runs the `place` stage itself on it, the georeference's frame and its
SH rotation included, exactly as a run would:

* that run's `stages/train/out/trained.ply` (SH degree 3, COLMAP's frame), with its
  `stages/georeference/out/georef.json` (and `--poses stages/pose/out/poses` for a
  capture placed by camera-up, which needs them);
* a gsplat export (`point_cloud_<step>.ply`) kept from a training container by hand.

A Lane 1 upload with SH (a phone app's gsplat-style PLY) is `normalize`d with
`ship_sh_degree: 3` by its own run; its `canonical.ply` is the first case.

Usage (from tools/pipeline):

    uv run python experiments/sh_compare.py canonical.ply out/ --georef georef.json
    uv run python experiments/sh_compare.py trained.ply out/ --georef georef.json --place \\
        [--poses poses/]
    # ... and as three sites the dev API serves from the checkout:
    uv run python experiments/sh_compare.py canonical.ply ../../data/tiles \\
        --georef georef.json --site spool --name "Spool table"

Writes `out/sh0/`, `out/sh1/`, `out/sh3/` (or `out/<site>-sh<d>/splat/` and a `site.json`
each) and `out/sh_compare.json`: per degree, tiles, bytes, bytes a gaussian, the ratio to
degree 0, and seconds to pack.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import harmonics  # noqa: E402
import splat_io  # noqa: E402
from captures_bridge import TILE_GAUSSIANS, splat_tiles_convert  # noqa: E402

#: The degrees compared: what ships today, the cheap band, and everything gsplat trains.
DEGREES: tuple[int, ...] = (0, 1, 3)
#: The `package` stage's own parameters (recipes/photo-reconstruct.yaml), so each tileset
#: is the one a run at that degree would publish.
OPACITY_MIN = 0.02
#: Metres per degree of latitude, as the pipeline registers with (gaussians.py).
METRES_PER_DEGREE = 111_320.0
#: Sites this many scan-widths apart: far enough that no gaussian of one overlaps the
#: next, near enough that all three load together (SiteManager loads every site nearby).
SPACING_WIDTHS = 1.5


def place(source: Path, georef: Path, poses: Path | None, work: Path) -> Path:
    """`source` (COLMAP's frame) through the real `place` stage: `canonical.ply`, with
    every SH band it carries turned into east/north/up by the georeference's frame."""
    from executor import execute
    from recipe import Recipe
    from runners import RunnerSet
    from workdir import Workdir

    if work.exists():
        shutil.rmtree(work)
    workdir = Workdir.create(work)
    inputs = ["trained.ply", "georef.json"]
    shutil.copyfile(source, workdir.input_path("trained.ply"))
    shutil.copyfile(georef, workdir.input_path("georef.json"))
    if poses is not None:
        shutil.copytree(poses, workdir.input_path("poses"))
        inputs.append("poses")
    recipe = Recipe.from_dict(
        {
            "name": "sh-compare-place",
            "version": 1,
            "inputs": inputs,
            "stages": [{"id": "place", "impl": "place_splat", "params": {}}],
        }
    )
    execute(recipe, workdir, RunnerSet.local())
    return workdir.artifact_path("place", "canonical.ply")


def tileset_files(folder: Path) -> list[Path]:
    """Every file a viewer fetches: `tileset.json` and every tile it names, parents too."""
    document = json.loads((folder / "tileset.json").read_text(encoding="utf-8"))
    files = [folder / "tileset.json"]
    stack = [document["root"]]
    while stack:
        tile = stack.pop()
        uri = (tile.get("content") or {}).get("uri")
        if uri:
            files.append(folder / uri)
        stack.extend(tile.get("children") or [])
    return files


def measure(folder: Path) -> dict[str, Any]:
    """What a viewer downloads: the tiles (GLBs) and everything else the folder holds."""
    tiles = [path for path in tileset_files(folder) if path.suffix == ".glb"]
    every = [path for path in folder.rglob("*") if path.is_file()]
    return {
        "tiles": len(tiles),
        "tileBytes": sum(path.stat().st_size for path in tiles),
        "rootTileBytes": (folder / "splat.glb").stat().st_size,
        "folderBytes": sum(path.stat().st_size for path in every),
    }


def compare(
    ply: Path,
    out: Path,
    *,
    lat: float,
    lon: float,
    height: float,
    degrees: Sequence[int] = DEGREES,
    tile_gaussians: int | None = TILE_GAUSSIANS,
    opacity_min: float = OPACITY_MIN,
    site: str | None = None,
    spacing_m: float | None = None,
) -> dict[str, Any]:
    """Pack `ply` once per degree into `out` and return the manifest (also written there).

    With `site`, each lands as `out/<site>-sh<d>/splat/` and is placed `spacing_m` east of
    the one before (by default `SPACING_WIDTHS` times the scan's width): the tiles are the
    same bytes wherever the root transform puts them, so the sizes compare exactly.
    """
    layout = splat_io.read_layout(ply)
    carried = harmonics.degree_of(layout.properties, source=ply.name)
    missing = [d for d in degrees if d > carried]
    if missing:
        raise SystemExit(
            f"{ply.name} carries SH degree {carried}, so degree {max(missing)} cannot be "
            f"packed from it: use a PLY that kept its f_rest_* (this module's docstring "
            f"says which), or a run trained with ship_sh_degree: 3"
        )
    out.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    spacing = 0.0
    for index, degree in enumerate(degrees):
        target = out / f"sh{degree}" if site is None else out / f"{site}-sh{degree}" / "splat"
        if target.exists():
            shutil.rmtree(target)
        east = index * spacing
        here = lon + east / (METRES_PER_DEGREE * max(math.cos(math.radians(lat)), 1e-6))
        began = time.perf_counter()
        stats = splat_tiles_convert(
            ply,
            target,
            lat=lat,
            lon=here,
            height=height,
            opacity_min=opacity_min,
            tile_gaussians=tile_gaussians,
            sh_degree=degree,
        )
        seconds = time.perf_counter() - began
        if site is not None and index == 0:
            # Known only once the first is packed: the scan's own width.
            spacing = (
                spacing_m if spacing_m is not None else SPACING_WIDTHS * float(stats["extent_m"])
            )
        sizes = measure(target)
        variant: dict[str, Any] = {
            "shDegree": int(stats["sh_degree"]),
            "dir": str(target.relative_to(out)),
            "gaussians": int(stats["gaussians"]),
            "parentGaussians": int(stats["parent_gaussians"]),
            **sizes,
            "bytesPerGaussian": round(sizes["tileBytes"] / max(1, int(stats["gaussians"])), 2),
            "packSeconds": round(seconds, 2),
            "lon": here,
            "eastOffsetM": round(east, 3),
        }
        if site is not None:
            variant["site"] = _write_site(
                target.parent, site, degree, lat=lat, lon=here, height=height, stats=stats
            )
        variants.append(variant)
    base = next((v for v in variants if v["shDegree"] == 0), variants[0])
    for variant in variants:
        variant["tileBytesRatio"] = round(variant["tileBytes"] / max(1, base["tileBytes"]), 3)
    manifest: dict[str, Any] = {
        "source": {
            "file": ply.name,
            "sha256": splat_io.file_checksum(ply),
            "gaussians": layout.count,
            "shDegree": carried,
            "bytes": ply.stat().st_size,
        },
        "origin": {"lat": lat, "lon": lon, "height": height},
        "packer": {"tileGaussians": tile_gaussians, "opacityMin": opacity_min},
        "variants": variants,
        "toMeasure": (
            "on the devices that draw them, per variant: frame rate and GPU memory in the "
            "developer panel (D) with the variant alone in view, the first tile's arrival "
            "and the whole tileset's load in the browser's network panel, and the look -- "
            "shine and colour that change as the view goes round -- against the photos"
        ),
    }
    (out / "sh_compare.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _write_site(
    folder: Path,
    site: str,
    degree: int,
    *,
    lat: float,
    lon: float,
    height: float,
    stats: dict[str, float | int],
) -> str:
    """A `site.json` the seeder reads (apps/api/app/seed/captures.py `Capture`): the
    variant as a site of its own, its boundary the tileset's footprint."""
    tileset = json.loads((folder / "splat" / "tileset.json").read_text(encoding="utf-8"))
    box = tileset["root"]["boundingVolume"]["box"]
    cx, cy = float(box[0]), float(box[1])
    hx = abs(float(box[3])) + abs(float(box[6]))
    hy = abs(float(box[4])) + abs(float(box[7]))
    scale = METRES_PER_DEGREE * max(math.cos(math.radians(lat)), 1e-6)

    def corner(east: float, north: float) -> list[float]:
        return [round(lon + east / scale, 8), round(lat + north / METRES_PER_DEGREE, 8)]

    ring = [
        corner(cx - hx, cy - hy),
        corner(cx + hx, cy - hy),
        corner(cx + hx, cy + hy),
        corner(cx - hx, cy + hy),
    ]
    slug = f"{site}-sh{degree}"
    document = {
        "slug": slug,
        "name": f"{site} -- SH degree {degree}",
        "description": (
            f"The same splat packaged at spherical-harmonic degree {degree} "
            f"({int(stats['gaussians'])} gaussians, {int(stats['tiles'])} tiles), for "
            f"comparing view-dependent colour against its cost "
            f"(tools/pipeline/experiments/sh_compare.py)"
        ),
        "boundary": [*ring, ring[0]],
        "center": [lon, lat, height],
        "attribution": "Hexapod capture (SH comparison)",
        "license_name": "All rights reserved",
        "pipeline": f"tools/pipeline/experiments/sh_compare.py, sh_degree {degree}",
        "assets": [
            {
                "representation": "gaussian-splat",
                "name": f"Gaussian splat (SH {degree})",
                "path": "splat/tileset.json",
                "clamp_to_ground": True,
                "default": True,
            }
        ],
    }
    (folder / "site.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return slug


def _georef(path: Path) -> tuple[float, float, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return float(document["lat"]), float(document["lon"]), float(document.get("height") or 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("ply", type=Path, help="a splat PLY with f_rest_* (see the docstring)")
    parser.add_argument("out", type=Path, help="where the tilesets and sh_compare.json go")
    parser.add_argument("--georef", type=Path, help="the run's georef.json: where it goes")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--height", type=float, default=0.0)
    parser.add_argument(
        "--place",
        action="store_true",
        help="the PLY is in COLMAP's frame (trained.ply, a gsplat export): run the place "
        "stage on it with --georef's frame first, SH turned with it",
    )
    parser.add_argument("--poses", type=Path, help="the run's poses/, for a camera-up frame")
    parser.add_argument(
        "--degrees", default=",".join(map(str, DEGREES)), help="comma-separated, default 0,1,3"
    )
    parser.add_argument("--tile-gaussians", type=int, default=TILE_GAUSSIANS)
    parser.add_argument("--site", help="also write each as a site: <out>/<site>-sh<d>/")
    parser.add_argument("--spacing-m", type=float, help="metres between the sites (east)")
    args = parser.parse_args(argv)
    if args.georef is not None:
        lat, lon, height = _georef(args.georef)
    elif args.lat is not None and args.lon is not None:
        lat, lon, height = args.lat, args.lon, args.height
    else:
        parser.error("give --georef, or --lat and --lon")
    if args.place and args.georef is None:
        parser.error("--place needs --georef: its frame is what places the splat")
    degrees = [harmonics.check_degree(int(d)) for d in args.degrees.split(",") if d.strip()]
    ply = args.ply
    if args.place:
        ply = place(args.ply, args.georef, args.poses, args.out / "sh-compare-place")
    manifest = compare(
        ply,
        args.out,
        lat=lat,
        lon=lon,
        height=height,
        degrees=degrees,
        tile_gaussians=args.tile_gaussians or None,
        site=args.site,
        spacing_m=args.spacing_m,
    )
    for variant in manifest["variants"]:
        sys.stdout.write(
            f"SH {variant['shDegree']}: {variant['tiles']} tiles, {variant['tileBytes']:,} "
            f"bytes ({variant['tileBytesRatio']:.3g}x degree 0), "
            f"{variant['bytesPerGaussian']} B/gaussian, packed in {variant['packSeconds']} s"
            f" -> {variant['dir']}\n"
        )
    sys.stdout.write(f"wrote {args.out / 'sh_compare.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
