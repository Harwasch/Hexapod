"""The fill gate reads only what the packer writes (ADR 0016 § Fill; `app/worker/positions.py`
`packer_placements`, `packer_spz`).

Codex rounds B5b and R4 on #60 found, twice, something Cesium honours that the gate did not
read: a second copy of a tile, a node matrix, then `content.url`, 3DTILES_multiple_contents
and an SPZ block in a second, data-URI buffer. The ruling: no denylist. A layer's tileset and
every GLB in it must be laid out exactly as tools/captures' tiler and `build_glb` write them,
and anything else is refused. Each case below is one refusal (the R4 probes by their names);
the packer's own output -- the test helper's `build_glb` layout and every committed fixture
tile -- passes. tools/captures/tests/test_rescan_gate.py holds the publish pre-check to the
same cases.
"""

from __future__ import annotations

import base64
import copy
import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.config import REPO_ROOT
from app.worker import positions
from tests.sidecar_fixtures import a_tileset, splat_glb

TILES = REPO_ROOT / "data" / "tiles"
#: Every committed tileset the tiler wrote, scans and fill layers alike.
FIXTURES = sorted(TILES.glob("**/tileset.json"))
POINT = [[0.5, 0.5, 0.5]]


def _chunks(data: bytes) -> tuple[dict[str, Any], bytes]:
    (length,) = struct.unpack_from("<I", data, 12)
    gltf: dict[str, Any] = json.loads(data[20 : 20 + length])
    return gltf, data[28 + length :]


def _glb(gltf: dict[str, Any], binary: bytes) -> bytes:
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (
        struct.pack("<4sII", b"glTF", 2, 28 + len(text) + len(binary))
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


# --- the packer's own output passes ------------------------------------------------------


def test_the_helpers_build_glb_layout_passes_and_reads_the_same_block() -> None:
    data = splat_glb(POINT)
    assert positions.packer_spz(data) == positions.glb_spz(data)


@pytest.mark.parametrize("tileset", FIXTURES, ids=lambda p: str(p.parent.relative_to(TILES)))
def test_every_committed_fixture_tileset_and_tile_passes(tileset: Path) -> None:
    document = json.loads(tileset.read_text(encoding="utf-8"))
    placements = positions.packer_placements(document)
    expected = positions.tile_placements(document)
    assert [uri for uri, _ in placements] == [uri for uri, _ in expected]
    assert all((a == b).all() for (_, a), (_, b) in zip(placements, expected, strict=True))
    for uri, _ in placements:
        data = (tileset.parent / uri).read_bytes()
        assert positions.packer_spz(data) == positions.glb_spz(data)


# --- the tileset: content by `uri` alone, plain names, the tiler's keys --------------------


def _child(document: dict[str, Any]) -> dict[str, Any]:
    child: dict[str, Any] = document["root"]["children"][0]
    return child


def _content(document: dict[str, Any]) -> dict[str, Any]:
    content: dict[str, Any] = _child(document)["content"]
    return content


TILESET_REFUSALS: dict[str, Callable[[dict[str, Any]], object]] = {
    # R4: Cesium loads `content.url` (the 1.0 spelling), and prefers it over `uri`.
    "legacy_url_child": lambda d: _child(d).update(content={"url": "f1.glb"}),
    "legacy_url_override": lambda d: _content(d).update(url="bad.glb"),
    # R4: an extension's contents are drawn too.
    "legacy_multiple_contents": lambda d: _child(d).update(
        extensions={"3DTILES_multiple_contents": {"contents": [{"uri": "f1.glb"}]}}
    ),
    "contents_array": lambda d: _child(d).update(contents=[{"uri": "f1.glb"}]),
    "implicit_tiling": lambda d: _child(d).update(implicitTiling={"subtreeLevels": 2}),
    "tile_metadata": lambda d: _child(d).update(metadata={"class": "x"}),
    "viewer_request_volume": lambda d: _child(d).update(viewerRequestVolume={"box": [0] * 12}),
    "content_group": lambda d: _content(d).update(group=0),
    "content_bounding_volume": lambda d: _content(d).update(boundingVolume={}),
    "top_level_extensions_required": lambda d: d.update(extensionsRequired=["X"]),
    "top_level_unknown_extension": lambda d: d.update(
        extensionsUsed=["3DTILES_metadata"], extensions={"3DTILES_metadata": {}}
    ),
    "top_level_schema": lambda d: d.update(schema={}),
    "asset_version_1_0": lambda d: d.update(asset={"version": "1.0"}),
    "bounding_region": lambda d: _child(d).update(boundingVolume={"region": [0] * 6}),
    "refine_unknown": lambda d: _child(d).update(refine="MERGE"),
    "children_not_a_list": lambda d: _child(d).update(children={"0": {}}),
    "rotating_child_transform": lambda d: _child(d).update(
        transform=[0.0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ),
    # R4 MEDIUM: a literal file `f0.glb?v=1` is another file than the one HTTP fetches.
    "uri_query_literal": lambda d: _content(d).update(uri="f1.glb?v=1"),
    "uri_fragment": lambda d: _content(d).update(uri="f1.glb#x"),
    "uri_parent": lambda d: _content(d).update(uri="../f1.glb"),
    "uri_dot": lambda d: _content(d).update(uri="./f1.glb"),
    "uri_scheme": lambda d: _content(d).update(uri="https://evil.example/f1.glb"),
    "uri_absolute": lambda d: _content(d).update(uri="/f1.glb"),
    "uri_backslash": lambda d: _content(d).update(uri="a\\f1.glb"),
    "uri_encoded": lambda d: _content(d).update(uri="f1%2Eglb"),
    "uri_not_a_glb": lambda d: _content(d).update(uri="nested.json"),
}


@pytest.mark.parametrize("case", sorted(TILESET_REFUSALS))
def test_a_tileset_not_written_as_the_tiler_writes_one_is_refused(case: str) -> None:
    document = a_tileset(("f0.glb", "f1.glb"))
    TILESET_REFUSALS[case](document)
    with pytest.raises(positions.TileFormatError):
        positions.packer_placements(document)


def test_the_tilers_own_keys_pass() -> None:
    """What `splat_tiles.write_tiles` writes on top of the helper's tileset."""
    document = a_tileset(("f0.glb", "f1.glb"))
    document["extensionsUsed"] = ["3DTILES_content_gltf"]
    used = ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"]
    document["extensions"] = {
        "3DTILES_content_gltf": {"extensionsUsed": used, "extensionsRequired": used}
    }
    _child(document)["transform"] = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 0, 1]
    _child(document)["extras"] = {"gaussians": 1, "confidenceMin": 40}
    placed = positions.packer_placements(document)
    assert [(uri, offset.tolist()) for uri, offset in placed] == [
        ("f0.glb", [0.0, 0.0, 0.0]),
        ("f1.glb", [3.0, 4.0, 0.0]),
    ]


# --- the GLB: build_glb's layout, one buffer, the packer's extensions ----------------------


def _edit(change: Callable[[dict[str, Any]], object]) -> Callable[[], bytes]:
    return lambda: splat_glb(POINT, edit=change)


def _external_spz_buffer() -> bytes:
    """R4: `bufferViews[0].buffer = 1`, a data-URI buffer holding an SPZ at (30, .5, .5);
    the BIN chunk keeps the safe one the gate decoded."""
    gltf, binary = _chunks(splat_glb(POINT))
    evil = positions.glb_spz(splat_glb([[30.0, 0.5, 0.5]]))
    uri = "data:application/octet-stream;base64," + base64.b64encode(evil).decode()
    gltf["buffers"].append({"byteLength": len(evil), "uri": uri})
    gltf["bufferViews"][0].update(buffer=1, byteLength=len(evil))
    return _glb(gltf, binary)


def _external_file_buffer() -> bytes:
    gltf, binary = _chunks(splat_glb(POINT))
    gltf["buffers"][0]["uri"] = "other.bin"
    return _glb(gltf, binary)


def _sparse_accessor() -> bytes:
    """R4: a sparse POSITION overwriting gaussian 0 with (30, .5, .5)."""
    gltf, binary = _chunks(splat_glb(POINT))
    offset = len(binary)
    binary += b"\0\0\0\0" + struct.pack("<3f", 30, 0.5, 0.5)
    gltf["bufferViews"] += [
        {"buffer": 0, "byteOffset": offset, "byteLength": 1},
        {"buffer": 0, "byteOffset": offset + 4, "byteLength": 12},
    ]
    gltf["buffers"][0]["byteLength"] = len(binary)
    gltf["accessors"][0]["sparse"] = {
        "count": 1,
        "indices": {"bufferView": 1, "componentType": 5121},
        "values": {"bufferView": 2},
    }
    return _glb(gltf, binary)


def _trailing_chunk() -> bytes:
    data = splat_glb(POINT)
    extra = struct.pack("<II", 4, 0x12345678) + b"\0\0\0\0"
    return struct.pack("<4sII", b"glTF", 2, len(data) + len(extra)) + data[12:] + extra


def _matrix(g: dict[str, Any], index: int, value: object) -> None:
    g["nodes"][0]["matrix"][index] = value


def _primitive(g: dict[str, Any]) -> dict[str, Any]:
    primitive: dict[str, Any] = g["meshes"][0]["primitives"][0]
    return primitive


GLB_REFUSALS: dict[str, Callable[[], bytes]] = {
    "external_spz_buffer": _external_spz_buffer,
    "external_file_buffer": _external_file_buffer,
    "sparse_accessor": _sparse_accessor,
    "trailing_chunk": _trailing_chunk,
    "second_primitive": lambda: splat_glb(POINT, also=([[30.0, 0.5, 0.5]],)),
    "meshopt_buffer_view": _edit(
        lambda g: g["bufferViews"][0].update(extensions={"EXT_meshopt_compression": {}})
    ),
    "second_buffer_view": _edit(lambda g: g["bufferViews"].append(dict(g["bufferViews"][0]))),
    "accessor_buffer_view": _edit(lambda g: g["accessors"][0].update(bufferView=0)),
    # R4: an extension the gate cannot read, required or merely used.
    "unknown_required": _edit(
        lambda g: (
            g["extensionsUsed"].append("VENDOR_unknown_positions"),
            g["extensionsRequired"].append("VENDOR_unknown_positions"),
        )
    ),
    "quantization_used": _edit(lambda g: g["extensionsUsed"].append("KHR_mesh_quantization")),
    "quantization_required": _edit(
        lambda g: (
            g["extensionsUsed"].append("KHR_mesh_quantization"),
            g["extensionsRequired"].append("KHR_mesh_quantization"),
        )
    ),
    "spz_not_required": _edit(lambda g: g.update(extensionsRequired=[])),
    "morph": _edit(lambda g: _primitive(g).update(targets=[{"POSITION": 0}])),
    "skin": _edit(lambda g: g.update(skins=[{"joints": [0]}])),
    "animation": _edit(lambda g: g.update(animations=[{"channels": [], "samplers": []}])),
    "instancing": _edit(
        lambda g: g["nodes"][0].update(
            extensions={"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 0}}}
        )
    ),
    "node_matrix": _edit(lambda g: _matrix(g, 12, 29.5)),
    "node_matrix_bool": _edit(lambda g: _matrix(g, 0, True)),
    "node_trs": _edit(lambda g: g["nodes"].__setitem__(0, {"mesh": 0, "translation": [0, 0, 0]})),
    "node_children": _edit(
        lambda g: (g["nodes"][0].update(children=[1]), g["nodes"].append({"mesh": 0}))
    ),
    "second_scene_node": _edit(
        lambda g: (
            g["nodes"].append(copy.deepcopy(g["nodes"][0])),
            g["scenes"][0]["nodes"].append(1),
        )
    ),
    "second_scene": _edit(lambda g: g["scenes"].append({"nodes": [0]})),
    "scene_one": _edit(lambda g: g.update(scene=1, scenes=[{"nodes": [0]}, {"nodes": [0]}])),
    "no_scene": _edit(lambda g: (g.pop("scene"), g.pop("scenes"))),
    "second_mesh": _edit(lambda g: g["meshes"].append(copy.deepcopy(g["meshes"][0]))),
    "uncompressed_splat": _edit(lambda g: _primitive(g).pop("extensions")),
    "spz_extra_key": _edit(
        lambda g: _primitive(g)["extensions"]["KHR_gaussian_splatting"].update(kernel="x")
    ),
    "primitive_indices": _edit(lambda g: _primitive(g).update(indices=0)),
    "primitive_mode_triangles": _edit(lambda g: _primitive(g).update(mode=4)),
    "unknown_attribute": _edit(lambda g: _primitive(g)["attributes"].update(_OFFSET=0)),
    "unknown_top_level_key": _edit(lambda g: g.update(cameras=[])),
    "gltf_version_1": _edit(lambda g: g.update(asset={"version": "1.0"})),
}


@pytest.mark.parametrize("case", sorted(GLB_REFUSALS))
def test_a_glb_not_written_as_build_glb_writes_one_is_refused(case: str) -> None:
    data = GLB_REFUSALS[case]()
    with pytest.raises(positions.TileFormatError):
        positions.packer_spz(data)
