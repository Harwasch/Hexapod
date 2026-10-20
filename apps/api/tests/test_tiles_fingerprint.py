"""`sidecars.tiles_fingerprint`: who the tiles are, and what is not part of that.

A tile's `extras.confidenceMin` (ADR 0016) summarises the epoch's confidence grid over the
tile's box; it is written on every tile, not only the root. A re-pack whose grid changed
and whose tiles did not must keep the fingerprint, or every sidecar bound to the tiles
(`nativeLod`, the rig, objects) is dropped as if the splats had moved.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

from app.services import sidecars
from app.storage.base import ObjectSummary

WHEN = datetime(2026, 10, 20, tzinfo=UTC)


def _tile(uri: str, box: list[float], confidence: int | None, **more: Any) -> dict[str, Any]:
    extras: dict[str, Any] = {"gaussians": 100}
    if confidence is not None:
        extras["confidenceMin"] = confidence
    return {
        "boundingVolume": {"box": box},
        "geometricError": 0.0,
        "content": {"uri": uri},
        "extras": extras,
        **more,
    }


def _tileset(child: int | None, grandchild: int | None, root: int | None) -> dict[str, Any]:
    box = [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    leaf = _tile("splat_0-1.glb", box, grandchild)
    middle = _tile("splat_0.glb", box, child, children=[leaf])
    middle["geometricError"] = 0.5
    top = _tile("splat.glb", box, root, children=[middle])
    top["extras"]["collision"] = {"uri": "collision.bin"}
    top["geometricError"] = 1.0
    return {"asset": {"version": "1.1"}, "geometricError": 2.0, "root": top}


OBJECTS = {
    uri: ObjectSummary(key=uri, size=10, etag=f'"{uri}"', last_modified=WHEN)
    for uri in ("splat.glb", "splat_0.glb", "splat_0-1.glb")
}


def test_a_confidence_only_change_on_any_tile_keeps_the_fingerprint() -> None:
    plain = sidecars.tiles_fingerprint(_tileset(None, None, None), OBJECTS)
    measured = sidecars.tiles_fingerprint(_tileset(40, 7, 7), OBJECTS)
    regraded = sidecars.tiles_fingerprint(_tileset(200, 0, 0), OBJECTS)
    assert plain is not None
    assert plain == measured == regraded


def test_what_the_tiles_are_still_changes_it() -> None:
    base = _tileset(40, 7, 7)
    before = sidecars.tiles_fingerprint(base, OBJECTS)
    moved = copy.deepcopy(base)
    moved["root"]["children"][0]["children"][0]["boundingVolume"]["box"][0] = 0.5
    recounted = copy.deepcopy(base)
    recounted["root"]["children"][0]["extras"]["gaussians"] = 101
    assert sidecars.tiles_fingerprint(moved, OBJECTS) != before
    assert sidecars.tiles_fingerprint(recounted, OBJECTS) != before
    # The document the caller passed is not touched.
    assert base == _tileset(40, 7, 7)
