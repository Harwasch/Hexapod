"""The manifest's read-back of a packaged tileset (`stages._tileset_summary`) is in the site
frame: what `registration._boundary` draws a site's outline from.

A tile of a site past ~4 km is written in a frame of its own (tools/captures
`splat_tiles.tile_frame`), and its GLB accessor bounds are relative to it. The committed
synthetic-far fixture has fifteen such tiles; source/expected.json records the site box.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from stages import _tileset_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
FAR = REPO_ROOT / "data" / "tiles" / "synthetic-far"
LOD = REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod"


def _accessor(path: Path) -> dict[str, Any]:
    blob = path.read_bytes()
    (length,) = struct.unpack_from("<I", blob, 12)
    document: dict[str, Any] = json.loads(blob[20 : 20 + length])
    position: dict[str, Any] = document["accessors"][0]
    return position


def _uris(tileset: dict[str, Any]) -> list[str]:
    out: list[str] = []
    stack: list[dict[str, Any]] = [tileset["root"]]
    while stack:
        tile = stack.pop()
        out.append(tile["content"]["uri"])
        stack.extend(tile.get("children", []))
    return out


def test_tileset_summary_reads_far_tiles_in_the_site_frame() -> None:
    expected = json.loads((FAR / "source" / "expected.json").read_text(encoding="utf-8"))
    summary = _tileset_summary(FAR / "splat")
    box = expected["bboxSiteM"]
    assert summary["bbox"]["min"] == pytest.approx(box["min"], abs=1e-3)
    assert summary["bbox"]["max"] == pytest.approx(box["max"], abs=1e-3)
    assert summary["tiles"] == len(expected["tiles"])
    assert summary["gaussians"] == sum(row["gaussians"] for row in expected["tiles"].values())
    # Read where it is stored, no tile reaches the far ends of the site.
    stored = [_accessor(FAR / "splat" / uri) for uri in expected["tiles"]]
    assert max(max(abs(v) for v in a["max"] + a["min"]) for a in stored) < 4100
    assert summary["bbox"]["max"][0] - summary["bbox"]["min"][0] == pytest.approx(10400.0)


def test_tileset_summary_of_the_lod_fixture_is_unchanged() -> None:
    tileset = json.loads((LOD / "tileset.json").read_text(encoding="utf-8"))
    accessors = [_accessor(LOD / uri) for uri in _uris(tileset)]
    summary = _tileset_summary(LOD)
    assert summary["tiles"] == len(accessors)
    assert summary["gaussians"] == sum(int(a["count"]) for a in accessors)
    assert summary["bbox"] == {
        "min": [min(float(a["min"][axis]) for a in accessors) for axis in range(3)],
        "max": [max(float(a["max"][axis]) for a in accessors) for axis in range(3)],
    }
