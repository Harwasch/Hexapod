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


# --- the SPZ stream: one gzip member of exactly the header and body the packer writes -----
#
# Codex R5: Python and Cesium's SPZ loader (@spz-loader/core) disagree on a header the
# packer never writes. At 32 fractional bits Python decodes 2048 / 2^32 m, inside a target;
# Cesium draws it at 2048 m. Every header field the packer fixes is held to its value, the
# fractional bits to the tiler's own rule (12, or the most a tile reaching past 2048 m still
# fits at, `splat_tiles.fractional_bits_for`), and the stream to one gzip member holding
# exactly what the header says.

SPZ_HEADER = "<IIIBBBB"


def _raw(points: Any = POINT, bits: int = 12) -> bytearray:
    import gzip

    return bytearray(gzip.decompress(positions.glb_spz(splat_glb(points, fractional_bits=bits))))


def _with_spz(spz: bytes) -> bytes:
    """`splat_glb(POINT)` with its SPZ block replaced by `spz`."""
    gltf, _ = _chunks(splat_glb(POINT))
    gltf["bufferViews"][0].update(byteOffset=0, byteLength=len(spz))
    gltf["buffers"][0]["byteLength"] = len(spz) + (-len(spz) % 4)
    return _glb(gltf, spz)


def _gz(raw: bytes | bytearray) -> bytes:
    import gzip

    return gzip.compress(bytes(raw), mtime=0)


def _header(index: int, value: int) -> Callable[[], bytes]:
    """The packer's SPZ with header field `index` (of magic, version, count, degree, bits,
    flags, reserved) set to `value`."""

    def make() -> bytes:
        raw = _raw()
        fields = list(struct.unpack_from(SPZ_HEADER, raw, 0))
        fields[index] = value
        struct.pack_into(SPZ_HEADER, raw, 0, *fields)
        return _with_spz(_gz(raw))

    return make


SPZ_REFUSALS: dict[str, Callable[[], bytes]] = {
    "fractional_bits_13": _header(4, 13),
    "fractional_bits_23": _header(4, 23),
    "fractional_bits_31": _header(4, 31),
    "fractional_bits_32": _header(4, 32),
    "fractional_bits_33": _header(4, 33),
    "fractional_bits_255": _header(4, 255),
    # Fewer bits than the tiler gives a tile this close to its frame.
    "fractional_bits_11_small_tile": lambda: splat_glb(POINT, fractional_bits=11),
    "spz_magic": _header(0, 0x12345678),
    "spz_version_3": _header(1, 3),
    "spz_sh_degree_4": _header(3, 4),
    "spz_flags": _header(5, 1),
    "spz_reserved": _header(6, 1),
    "header_count_two": _header(2, 2),
    "gzip_split": lambda: _with_spz(_gz(_raw()[:16]) + _gz(_raw()[16:])),
    "gzip_concat": lambda: _with_spz(_gz(_raw()) + _gz(_raw([[30.0, 0.5, 0.5]]))),
    "gzip_trailing_zero": lambda: _with_spz(_gz(_raw()) + b"\0" * 8),
    "gzip_trailing_junk": lambda: _with_spz(_gz(_raw()) + b"JUNK"),
    "spz_raw_trailing": lambda: _with_spz(_gz(_raw() + _raw([[30.0, 0.5, 0.5]]))),
    "spz_raw_short": lambda: _with_spz(_gz(_raw()[:-1])),
    "not_gzip": lambda: _with_spz(bytes(_raw())),
}


@pytest.mark.parametrize("case", sorted(SPZ_REFUSALS))
def test_an_spz_stream_not_written_as_the_packer_writes_one_is_refused(case: str) -> None:
    with pytest.raises(positions.TileFormatError):
        positions.packer_spz(SPZ_REFUSALS[case]())


def test_a_tile_reaching_past_2048_m_at_the_tilers_fewer_bits_passes() -> None:
    """The tiler's rule (`fractional_bits_for`): 12 bits to 2047.99 m, 11 to 4095.99 m."""
    data = splat_glb([[3000.0, 0.5, 0.5]], fractional_bits=11)
    assert positions.packer_spz(data) == positions.glb_spz(data)
    with pytest.raises(positions.TileFormatError):
        positions.packer_spz(splat_glb([[3000.0, 0.5, 0.5]], fractional_bits=10))


# --- the accessors: build_glb's exact values (Codex R5) -------------------------------------
#
# The SPZ path keeps each accessor's componentType, type, normalized and count: POSITION as
# 5125 (unsigned int) makes Cesium read the float bytes of 0.5 as 1056964608. So every
# accessor is build_glb's, value for value, its count the SPZ's, and the attributes map to
# them in build_glb's order, SH ones exactly those of the SPZ's degree.


def _accessor(index: int, **change: Any) -> Callable[[], bytes]:
    return _edit(lambda g: g["accessors"][index].update(**change))


ACCESSOR_REFUSALS: dict[str, Callable[[], bytes]] = {
    "accessor_uint": _accessor(0, componentType=5125),
    "accessor_type": _accessor(0, componentType=5121, type="SCALAR"),
    "accessor_vec4_position": _accessor(0, type="VEC4"),
    "accessor_normalized": _accessor(0, normalized=True),
    "accessor_count_zero": _accessor(0, count=0),
    "accessor_count_two": _accessor(0, count=2),
    "accessor_count_huge": _accessor(0, count=100000000),
    "accessor_minmax": _accessor(0, min=[30, 30, 30], max=[31, 31, 31]),
    "accessor_no_minmax": _edit(
        lambda g: (g["accessors"][0].pop("min"), g["accessors"][0].pop("max"))
    ),
    "accessor_minmax_short": _accessor(0, min=[0.5, 0.5]),
    "accessor_color_float": _accessor(1, componentType=5126),
    "accessor_color_not_normalized": _edit(lambda g: g["accessors"][1].pop("normalized")),
    "accessor_scale_count": _accessor(2, count=2),
    "accessor_rotation_vec3": _accessor(3, type="VEC3"),
    "accessor_extra": _edit(lambda g: g["accessors"].append(dict(g["accessors"][2]))),
    "attributes_swapped": _edit(
        lambda g: _primitive(g)["attributes"].update(
            POSITION=2, **{"KHR_gaussian_splatting:SCALE": 0}
        )
    ),
    "attributes_missing_rotation": _edit(
        lambda g: _primitive(g)["attributes"].pop("KHR_gaussian_splatting:ROTATION")
    ),
    "attributes_sh_without_degree": _edit(
        lambda g: (
            g["accessors"].append(dict(g["accessors"][2])),
            _primitive(g)["attributes"].update({"KHR_gaussian_splatting:SH_DEGREE_1_COEF_0": 4}),
        )
    ),
}


@pytest.mark.parametrize("case", sorted(ACCESSOR_REFUSALS))
def test_an_accessor_not_build_glbs_value_for_value_is_refused(case: str) -> None:
    with pytest.raises(positions.TileFormatError):
        positions.packer_spz(ACCESSOR_REFUSALS[case]())


def test_an_sh_tile_s_attributes_are_its_degree_s() -> None:
    """build_glb at SH degree 1 declares three SH accessors; the committed SH fixture is
    checked in `test_every_committed_fixture_tileset_and_tile_passes`."""
    from tests.sidecar_fixtures import splat_glb as glb

    names = [f"KHR_gaussian_splatting:SH_DEGREE_1_COEF_{n}" for n in range(3)]

    def declare(g: dict[str, Any]) -> None:
        for n, name in enumerate(names):
            g["accessors"].append({"componentType": 5126, "count": 1, "type": "VEC3"})
            _primitive(g)["attributes"][name] = 4 + n

    data = glb(POINT, sh_degree=1, edit=declare)
    assert positions.packer_spz(data) == positions.glb_spz(data)
    with pytest.raises(positions.TileFormatError):
        positions.packer_spz(glb(POINT, sh_degree=1))


# --- the tileset's numbers: whole-metre translations within a site, finite errors (R5) ------
#
# Codex R5: child translations of +1e300 then -1e300 cancel in the gate's float64 sum of
# offsets but not in Cesium's Matrix4 products, which lose the root's 100 m. The tiler
# writes whole-metre origins (`splat_tiles.tile_frame`) of a site, so each translation and
# each composed offset is a whole number of metres within 2^23 m (SPZ's whole-metre reach):
# then every partial sum is exact in float64 and in float32, in any order.


def _translate(*values: float) -> Callable[[dict[str, Any]], object]:
    """The child moved by `values[0]` east, a grandchild under it by each next one."""

    def change(d: dict[str, Any]) -> None:
        tile = _child(d)
        for index, value in enumerate(values):
            tile["transform"] = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, value, 0, 0, 1]
            if index + 1 < len(values):
                inner = {
                    "boundingVolume": {"box": [0.0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5]},
                    "geometricError": 0,
                    "content": {"uri": "f1.glb"},
                }
                tile.pop("content", None)
                tile["children"] = [inner]
                tile = inner

    return change


NUMBER_REFUSALS: dict[str, Callable[[dict[str, Any]], object]] = {
    "translation_cancel_1e300": _translate(1e300, -1e300),
    "translation_cancel_1e9": _translate(1e9, -1e9),
    "translation_1e300": _translate(1e300),
    "translation_past_site": _translate(float(2**23 + 1)),
    "translation_sum_past_site": _translate(float(2**23), float(2**23)),
    "translation_fraction": _translate(0.25),
    "translation_nested_fraction": _translate(0.25, 0.75),
    "translation_nan": _translate(float("nan")),
    "error_nan": lambda d: d["root"].update(geometricError=float("nan")),
    "error_inf": lambda d: _child(d).update(geometricError=float("inf")),
    "error_negative": lambda d: _child(d).update(geometricError=-1.0),
    "error_not_a_number": lambda d: _child(d).update(geometricError="1"),
    "top_error_nan": lambda d: d.update(geometricError=float("nan")),
    "box_nan": lambda d: _child(d)["boundingVolume"].update(box=[float("nan")] * 12),
}


@pytest.mark.parametrize("case", sorted(NUMBER_REFUSALS))
def test_a_tileset_number_the_tiler_never_writes_is_refused(case: str) -> None:
    document = a_tileset(("f0.glb", "f1.glb"))
    NUMBER_REFUSALS[case](document)
    with pytest.raises(positions.TileFormatError):
        positions.packer_placements(document)


def test_whole_metre_offsets_compose_exactly_in_any_float_order() -> None:
    """The tolerance is zero: whole metres under 2^23 m sum exactly in float64 and float32,
    so the gate's offsets (float64, root down) are Cesium's (Matrix4 products) and the
    GPU's (float32) bit for bit."""
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(200):
        steps = rng.integers(-(2**20), 2**20, size=(8, 3)).astype(np.float64)
        document = a_tileset(("f0.glb", "f1.glb"))
        _translate(*steps[:, 0])(document)
        (_, f0), (_, f1) = positions.packer_placements(document)
        exact = int(steps[:, 0].astype(np.int64).sum())
        assert f1[0] == exact
        assert float(np.float32(steps[::-1, 0]).sum(dtype=np.float32)) == exact
        assert not f0.any()
    document = a_tileset(("f0.glb", "f1.glb"))
    _translate(float(2**23 - 1), 1.0)(document)
    assert positions.packer_placements(document)[-1][1][0] == 2**23
