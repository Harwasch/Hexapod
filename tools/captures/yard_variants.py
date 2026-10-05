"""Method variants for the synthetic yard (the viewer's e2e fixture), for comparing methods.

The owner judges a bake-off by switching between methods on the same scan in the app
(apps/web/src/lib/variants.ts, docs/SCENE_OBJECTS.md "Variants"). The synthetic yard is where
the viewer's switching is tested, so it gets at least two variants per system that differ at a
glance, in today's file formats, under ``variants/<system>/<name>/`` beside the yard's tiles,
and the ``extras.variants`` block that declares them, as the measured tileset's root would
(``variants/variants.json``, its paths relative to ``splat/tileset.json``):

objects/whole   the yard's 24 top-level objects with every part folded into its object, each
                with a broad category (the trees, the snags as dead wood, the shed, the shrubs,
                lawn, path);
objects/parts   every one of today's 103 instances, the trees' parts as dead wood and the
                shed's as walls, so the objects panel lists more, and other, categories;
fill/hedge      a hedge of inferred splats beyond the yard's west edge;
fill/mound      a mound of inferred splats beyond its east edge;
skins/tree      the big tree's skin alone (``skin_scene.build(only=[1])``);
skins/small     the snag's and two shrubs' skins (``only=[9, 10, 12]``): other objects sway.

Both objects variants keep today's ids for the top-level objects, so the skins (which name
instance 1, 9, 10 and 12) find their objects under either. The fills are packaged as
``teacher_fill.package_inferred`` packages a real one (their own tileset in the yard's frame,
``extras.evidence``), without view cones: a fixture is seen from wherever a test looks.

Only the yard's committed files are read; nothing under ``splat/`` is written (its listing is
held to a fresh run's, tests/test_scene_plants.py). Written once and committed, not
byte-reproducible (the skins' eigensolve is the CPU's).

Usage:
    python yard_variants.py ../../data/tiles/synthetic-yard
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import skin_scene
import teacher_fill
from rebind_instances import decode_runs, encode_runs
from splat_render import Camera, Splats

#: A broad category (lib/categories.ts ids) per top-level object of the yard, by its id in
#: data/tiles/synthetic-yard/instances/instances.json, with a word to search it by.
YARD_OBJECTS: dict[int, tuple[str, str]] = {
    **{i: ("trees", "tree") for i in (1, 3, 5)},
    **{i: ("wood", "snag") for i in (7, 9)},
    8: ("buildings", "shed"),
    **{i: ("shrubs", "shrub") for i in (10, 11, 12, 13, 16)},
    **{i: ("grass", "lawn") for i in (2, 4, 6)},
    **{i: ("paths", "gravel path") for i in (14, 15, *range(17, 25))},
}

#: What the parts of an object are, when they are something else (`objects/parts`).
PART_CATEGORIES: dict[str, tuple[str, str]] = {
    "trees": ("wood", "branch"),
    "buildings": ("walls", "wall"),
}

TILESET = "splat/tileset.json"


def _rel(path: str) -> str:
    """A path under the yard as the measured tileset (``splat/tileset.json``) names it."""
    return f"../{path}"


def _subtree(instances: list[dict]) -> dict[int, int]:
    """Every instance id to its top-level object's id."""
    parent = {int(i["id"]): i.get("parent") for i in instances}
    top: dict[int, int] = {}
    for k in parent:
        j = k
        while parent.get(j) is not None:
            j = int(parent[j])
        top[k] = j
    return top


def _union(boxes: list[dict]) -> dict:
    return {
        "min": [round(min(b["min"][a] for b in boxes), 4) for a in range(3)],
        "max": [round(max(b["max"][a] for b in boxes), 4) for a in range(3)],
    }


def objects_whole(doc: dict) -> dict:
    """The top-level objects only, every part's splats folded into its object."""
    top = _subtree(doc["instances"])
    members: dict[int, list[dict]] = {}
    for instance in doc["instances"]:
        members.setdefault(top[int(instance["id"])], []).append(instance)
    instances = []
    for root in sorted(members):
        group = members[root]
        head = next(i for i in group if int(i["id"]) == root)
        splats = sum(int(i["splats"]) for i in group)
        weights = np.array([max(1, int(i["splats"])) for i in group], float)
        centroid = np.average(
            np.array([i["centroid"] for i in group], float), axis=0, weights=weights
        )
        category, word = YARD_OBJECTS[root]
        instances.append(
            {
                **head,
                "splats": splats,
                "bounds": _union([i["bounds"] for i in group]),
                "centroid": [round(float(c), 4) for c in centroid],
                "tags": [{"label": word, "score": 0.9}],
                "category": category,
            }
        )
    lookup = np.zeros(max(top) + 1, np.int64)
    for k, root in top.items():
        lookup[k] = root
    tiles = {
        checksum: encode_runs(lookup[decode_runs(runs)]) for checksum, runs in doc["tiles"].items()
    }
    return _document(doc, instances, tiles)


def objects_parts(doc: dict) -> dict:
    """Every instance, the parts of trees and of the shed in categories of their own."""
    top = _subtree(doc["instances"])
    instances = []
    for instance in doc["instances"]:
        k = int(instance["id"])
        category, word = YARD_OBJECTS[top[k]]
        if k != top[k]:
            category, word = PART_CATEGORIES.get(category, (category, word))
        instances.append(
            {**instance, "tags": [{"label": word, "score": 0.8}], "category": category}
        )
    return _document(doc, instances, doc["tiles"])


def _document(doc: dict, instances: list[dict], tiles: dict) -> dict:
    out = {k: v for k, v in doc.items() if k not in ("instances", "tiles", "embedding")}
    return {**out, "instances": instances, "tiles": tiles}


def _discs(
    rng: np.random.Generator, positions: np.ndarray, normals: np.ndarray, colour, jitter: float
) -> Splats:
    n = len(positions)
    rotations = teacher_fill._disc_rotations(normals)
    scales = np.column_stack([np.full(n, 0.14), np.full(n, 0.14), np.full(n, 0.02)])
    colours = np.clip(np.asarray(colour, float) + rng.normal(0, jitter, size=(n, 3)), 0, 1)
    return Splats(positions, rotations, scales, colours, np.full(n, 0.92))


def hedge(rng: np.random.Generator) -> Splats:
    """A box hedge, 12 m long, 1.6 m high, beyond the yard's west edge (x < 0)."""
    n = 2400
    face = rng.integers(0, 3, size=n)
    u = rng.uniform(0, 1, size=n)
    v = rng.uniform(0, 1, size=n)
    x = np.where(face == 0, -1.8, np.where(face == 1, -3.0, -3.0 + 1.2 * u))
    y = 6.0 + 12.0 * np.where(face == 2, v, u)
    z = np.where(face == 2, 1.6, 1.6 * v)
    normals = np.zeros((n, 3))
    normals[face == 0] = [1, 0, 0]
    normals[face == 1] = [-1, 0, 0]
    normals[face == 2] = [0, 0, 1]
    return _discs(rng, np.column_stack([x, y, z]), normals, [0.16, 0.42, 0.12], 0.05)


def mound(rng: np.random.Generator) -> Splats:
    """A mound of earth 2.5 m in radius, beyond the yard's east edge (x > 32)."""
    n = 2400
    theta = rng.uniform(0, 2 * np.pi, size=n)
    up = rng.uniform(0.05, 1.0, size=n)
    ring = np.sqrt(1 - up**2)
    normals = np.column_stack([ring * np.cos(theta), ring * np.sin(theta), up])
    positions = np.array([35.5, 12.0, -0.2]) + 2.5 * normals * np.array([1, 1, 0.7])
    return _discs(rng, positions, normals, [0.55, 0.38, 0.22], 0.05)


def package_fill(yard: Path, name: str, splats: Splats, filler: str) -> dict:
    """The fill as an inferred layer under ``variants/fill/<name>/``; its declaration."""
    out = yard / "variants" / "fill" / name
    shutil.rmtree(out, ignore_errors=True)
    centre = splats.positions.mean(axis=0)
    camera = Camera.look_at(
        (centre + [0.0, -12.0, 8.0]).tolist(), centre.tolist(), width=64, height=48
    )
    confidence = np.full(len(splats), 0.6)
    evidence = teacher_fill.package_inferred(
        splats,
        confidence,
        [camera],
        yard / TILESET,
        out,
        filler,
        rule="a fixture: generated by tools/captures/yard_variants.py, not by an image model",
    )
    # Seen from wherever a test looks: no view cones.
    document = json.loads((out / "tileset.json").read_text(encoding="utf-8"))
    document["root"]["extras"].pop("viewCones", None)
    (out / "tileset.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    for cones in out.glob("viewcones*.bin"):
        cones.unlink()
    return {"uri": _rel(f"variants/fill/{name}/tileset.json"), "evidence": evidence}


def build_skin(yard: Path, name: str, doc: dict, only: list[int]) -> None:
    """Today's skin method for just the `only` objects, under ``variants/skins/<name>/``."""
    tiles = skin_scene.read_tiles(yard / "splat", doc)
    built = skin_scene.build(tiles, doc["instances"], only=only)
    out = yard / "variants" / "skins" / name
    shutil.rmtree(out, ignore_errors=True)
    skin_scene.write_skin(out, built)


def write_variants(yard: Path) -> dict:
    """Every variant's files, and ``variants/variants.json``: the block to declare."""
    doc = json.loads((yard / "instances" / "instances.json").read_text(encoding="utf-8"))
    for name, made in (("whole", objects_whole(doc)), ("parts", objects_parts(doc))):
        out = yard / "variants" / "objects" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "instances.json").write_text(
            json.dumps(made, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    rng = np.random.default_rng(20261005)
    hedge_layer = package_fill(yard, "hedge", hedge(rng), "fixture-hedge")
    mound_layer = package_fill(yard, "mound", mound(rng), "fixture-mound")
    build_skin(yard, "tree", doc, [1])
    build_skin(yard, "small", doc, [9, 10, 12])
    variants = {
        "objects": [
            {
                "name": "whole",
                "label": "A · Whole objects",
                "about": "Each thing in the yard is one object: a tree with its branches, the shed with its walls.",
                "instances": _rel("variants/objects/whole/instances.json"),
            },
            {
                "name": "parts",
                "label": "B · Parts",
                "about": "Every part is its own object: branches apart from their trees, the shed's walls apart from it.",
                "instances": _rel("variants/objects/parts/instances.json"),
            },
        ],
        "fill": [
            {
                "name": "hedge",
                "label": "Hedge",
                "about": "A fixture fill: a hedge generated beyond the yard's west edge.",
                "inferredLayers": [hedge_layer],
            },
            {
                "name": "mound",
                "label": "Mound",
                "about": "A fixture fill: a mound of earth generated beyond the yard's east edge.",
                "inferredLayers": [mound_layer],
            },
        ],
        "skins": [
            {
                "name": "tree",
                "label": "Tree only",
                "about": "Only the big tree is skinned: it sways in the wind, the rest stands still.",
                "skin": _rel("variants/skins/tree/skin.json"),
            },
            {
                "name": "small",
                "label": "Snag and shrubs",
                "about": "The snag and two shrubs are skinned: they sway, the big tree stands still.",
                "skin": _rel("variants/skins/small/skin.json"),
            },
        ],
    }
    (yard / "variants" / "variants.json").write_text(
        json.dumps({"variants": variants}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("yard", type=Path, help="data/tiles/synthetic-yard")
    args = parser.parse_args(argv)
    variants = write_variants(args.yard)
    for system, entries in variants.items():
        print(system, [e["name"] for e in entries])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
