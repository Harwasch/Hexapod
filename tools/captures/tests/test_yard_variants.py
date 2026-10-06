"""The synthetic yard's method variants (`yard_variants.py`, data/tiles/synthetic-yard/variants):
the viewer's e2e fixture for comparing methods (apps/web/e2e/variants.spec.ts). What the
committed files must hold for that spec to mean anything."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import yard_variants as yv
from rebind_instances import decode_runs

YARD = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard"
SPLAT = YARD / "splat"


def _declared() -> dict:
    return json.loads((YARD / "variants" / "variants.json").read_text(encoding="utf-8"))["variants"]


def _today() -> dict:
    return json.loads((YARD / "instances" / "instances.json").read_text(encoding="utf-8"))


def test_every_declared_file_is_there_two_per_system() -> None:
    declared = _declared()
    assert {system: len(entries) for system, entries in declared.items()} == {
        "objects": 2,
        "fill": 2,
        "skins": 2,
    }
    for entry in declared["objects"]:
        assert (SPLAT / entry["instances"]).resolve().is_file()
    for entry in declared["skins"]:
        skin = (SPLAT / entry["skin"]).resolve()
        assert skin.is_file() and (skin.parent / "skin.bin").is_file()
    for entry in declared["fill"]:
        (layer,) = entry["inferredLayers"]
        assert (SPLAT / layer["uri"]).resolve().is_file()
        assert layer["evidence"]["kind"] == "inferred"
    for entries in declared.values():
        for entry in entries:
            assert entry["name"] and entry["label"] and entry["about"]


def test_whole_folds_every_part_into_its_object() -> None:
    today = _today()
    whole = yv.objects_whole(today)
    roots = {int(i["id"]) for i in today["instances"] if i.get("parent") is None}
    assert {int(i["id"]) for i in whole["instances"]} == roots
    assert sum(i["splats"] for i in whole["instances"]) == sum(
        i["splats"] for i in today["instances"]
    )
    assert all(i.get("category") for i in whole["instances"])
    for checksum, runs in today["tiles"].items():
        before = decode_runs(runs)
        after = decode_runs(whole["tiles"][checksum])
        assert after.size == before.size
        assert set(np.unique(after)) <= roots | {0}
        assert np.array_equal(after == 0, before == 0)


def test_parts_lists_other_categories_than_whole() -> None:
    today = _today()
    whole = {i["category"] for i in yv.objects_whole(today)["instances"]}
    parts = yv.objects_parts(today)
    assert len(parts["instances"]) == len(today["instances"])
    assert parts["tiles"] == today["tiles"]
    assert {i["category"] for i in parts["instances"]} - whole == {"walls"}


def test_the_fills_lie_beyond_the_yard_in_its_frame() -> None:
    measured = json.loads((SPLAT / "tileset.json").read_text(encoding="utf-8"))
    centres = {}
    for entry in _declared()["fill"]:
        layer = json.loads((SPLAT / entry["inferredLayers"][0]["uri"]).read_text("utf-8"))
        assert layer["root"]["transform"] == measured["root"]["transform"]
        assert "viewCones" not in layer["root"]["extras"]
        centres[entry["name"]] = layer["root"]["boundingVolume"]["box"][:3]
    assert centres["hedge"][0] < 0
    assert centres["mound"][0] > 32


def test_the_skins_move_other_objects() -> None:
    moved = {}
    for entry in _declared()["skins"]:
        skin = json.loads((SPLAT / entry["skin"]).resolve().read_text(encoding="utf-8"))
        moved[entry["name"]] = sorted(s["instance"] for s in skin["skins"])
    assert moved == {"tree": [1], "small": [9, 10, 12]}
