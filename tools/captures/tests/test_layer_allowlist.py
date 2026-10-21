"""The publish pre-check reads only what the packer writes (`rescan_gate.layer_positions`).

The same allowlist, and the same refusals by name, as the API's gate
(apps/api tests/test_layer_allowlist.py; Codex R4 on #60): a layer's tileset and every GLB in
it must be laid out exactly as `splat_tiles.write_tiles` and `build_glb` write them. The
packer's own output -- `build_glb` and every committed fixture tileset -- passes.
"""

from __future__ import annotations

import base64
import copy
import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import rescan_gate as rg
from splat_tiles import build_glb, pack_spz, tile_placements

REPO = Path(__file__).resolve().parents[3]
TILES = REPO / "data" / "tiles"
FIXTURES = sorted(TILES.glob("**/tileset.json"))
BOX = [0.0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5]


def glb(points: Any) -> bytes:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = p.shape[0]
    spz = pack_spz(
        p, np.zeros((n, 3)), np.zeros(n), np.zeros((n, 3)), np.tile([0, 0, 0, 1.0], (n, 1))
    )
    return build_glb(n, p.min(axis=0).tolist(), p.max(axis=0).tolist(), spz)


def chunks(data: bytes) -> tuple[dict[str, Any], bytes]:
    (length,) = struct.unpack_from("<I", data, 12)
    gltf: dict[str, Any] = json.loads(data[20 : 20 + length])
    return gltf, data[28 + length :]


def repack(gltf: dict[str, Any], binary: bytes) -> bytes:
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


SAFE = glb([[0.5, 0.5, 0.5]])
BAD = glb([[30.0, 0.5, 0.5]])


def tileset() -> dict[str, Any]:
    child = {"boundingVolume": {"box": BOX}, "geometricError": 0, "content": {"uri": "f1.glb"}}
    return {
        "asset": {"version": "1.1"},
        "geometricError": 10,
        "root": {
            "boundingVolume": {"box": BOX},
            "geometricError": 1,
            "refine": "REPLACE",
            "content": {"uri": "f0.glb"},
            "children": [child],
            "extras": {"gaussians": 1},
        },
    }


def write(folder: Path, document: dict[str, Any], leaf: bytes = SAFE) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "f0.glb").write_bytes(SAFE)
    (folder / "f1.glb").write_bytes(leaf)
    (folder / "bad.glb").write_bytes(BAD)
    (folder / "tileset.json").write_text(json.dumps(document), encoding="utf-8")
    return folder / "tileset.json"


# --- the packer's own output passes ------------------------------------------------------


def test_build_glb_output_passes(tmp_path: Path) -> None:
    assert rg.layer_positions(write(tmp_path, tileset())).shape == (2, 3)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: str(p.parent.relative_to(TILES)))
def test_every_committed_fixture_tileset_and_tile_passes(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    every = rg.layer_positions(path)
    assert every.shape[0] > 0
    assert len({uri for uri, _ in tile_placements(document)}) >= 1


def test_the_tilers_own_keys_pass(tmp_path: Path) -> None:
    document = tileset()
    document["extensionsUsed"] = ["3DTILES_content_gltf"]
    used = ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"]
    document["extensions"] = {
        "3DTILES_content_gltf": {"extensionsUsed": used, "extensionsRequired": used}
    }
    child = document["root"]["children"][0]
    child["transform"] = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 0, 1]
    child["extras"] = {"gaussians": 1, "confidenceMin": 40}
    xyz = rg.layer_positions(write(tmp_path, document))
    assert sorted(xyz[:, 0].tolist()) == [0.5, 3.5]


# --- the tileset ---------------------------------------------------------------------------


def _child(d: dict[str, Any]) -> dict[str, Any]:
    child: dict[str, Any] = d["root"]["children"][0]
    return child


def _content(d: dict[str, Any]) -> dict[str, Any]:
    content: dict[str, Any] = _child(d)["content"]
    return content


SHIFT = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 29.5, 0, 0, 1]

TILESET_REFUSALS: dict[str, Callable[[dict[str, Any]], object]] = {
    "legacy_url_child": lambda d: _child(d).update(content={"url": "f1.glb"}, transform=SHIFT),
    "legacy_url_override": lambda d: _content(d).update(url="bad.glb"),
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
    # R4 MEDIUM: the literal file `f1.glb?v=1` exists; HTTP would fetch `f1.glb`.
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
def test_a_tileset_not_written_as_the_tiler_writes_one_is_refused(
    tmp_path: Path, case: str
) -> None:
    document = tileset()
    TILESET_REFUSALS[case](document)
    path = write(tmp_path, document)
    # Whatever file a uri names exists, so only the layout can refuse it.
    for name in ("f1.glb?v=1", "f1.glb#x", "f1%2Eglb", "nested.json"):
        (tmp_path / name).write_bytes(SAFE)
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(path)


# --- the GLB -------------------------------------------------------------------------------


def _edit(change: Callable[[dict[str, Any]], object]) -> Callable[[], bytes]:
    def make() -> bytes:
        gltf, binary = chunks(SAFE)
        change(gltf)
        return repack(gltf, binary)

    return make


def _external_spz_buffer() -> bytes:
    gltf, binary = chunks(SAFE)
    evil = glb_spz_bytes(BAD)
    uri = "data:application/octet-stream;base64," + base64.b64encode(evil).decode()
    gltf["buffers"].append({"byteLength": len(evil), "uri": uri})
    gltf["bufferViews"][0].update(buffer=1, byteLength=len(evil))
    return repack(gltf, binary)


def glb_spz_bytes(data: bytes) -> bytes:
    gltf, binary = chunks(data)
    view = gltf["bufferViews"][0]
    return binary[view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]]


def _external_file_buffer() -> bytes:
    gltf, binary = chunks(SAFE)
    gltf["buffers"][0]["uri"] = "other.bin"
    return repack(gltf, binary)


def _sparse_accessor() -> bytes:
    gltf, binary = chunks(SAFE)
    binary += b"\0" * (-len(binary) % 4)
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
    return repack(gltf, binary)


def _second_primitive() -> bytes:
    gltf, binary = chunks(SAFE)
    other = glb_spz_bytes(BAD)
    binary += b"\0" * (-len(binary) % 4)
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(other)})
    binary += other
    gltf["buffers"][0]["byteLength"] = len(binary)
    primitive = copy.deepcopy(gltf["meshes"][0]["primitives"][0])
    primitive["extensions"]["KHR_gaussian_splatting"]["extensions"][
        "KHR_gaussian_splatting_compression_spz_2"
    ]["bufferView"] = 1
    gltf["meshes"][0]["primitives"].append(primitive)
    return repack(gltf, binary)


def _trailing_chunk() -> bytes:
    extra = struct.pack("<II", 4, 0x12345678) + b"\0\0\0\0"
    return struct.pack("<4sII", b"glTF", 2, len(SAFE) + len(extra)) + SAFE[12:] + extra


def _primitive(g: dict[str, Any]) -> dict[str, Any]:
    primitive: dict[str, Any] = g["meshes"][0]["primitives"][0]
    return primitive


def _matrix(g: dict[str, Any], index: int, value: object) -> None:
    g["nodes"][0]["matrix"][index] = value


GLB_REFUSALS: dict[str, Callable[[], bytes]] = {
    "external_spz_buffer": _external_spz_buffer,
    "external_file_buffer": _external_file_buffer,
    "sparse_accessor": _sparse_accessor,
    "trailing_chunk": _trailing_chunk,
    "second_primitive": _second_primitive,
    "meshopt_buffer_view": _edit(
        lambda g: g["bufferViews"][0].update(extensions={"EXT_meshopt_compression": {}})
    ),
    "second_buffer_view": _edit(lambda g: g["bufferViews"].append(dict(g["bufferViews"][0]))),
    "accessor_buffer_view": _edit(lambda g: g["accessors"][0].update(bufferView=0)),
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
def test_a_glb_not_written_as_build_glb_writes_one_is_refused(tmp_path: Path, case: str) -> None:
    path = write(tmp_path, tileset(), leaf=GLB_REFUSALS[case]())
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(path)


# --- the SPZ stream (Codex R5: the header's values, the gzip framing) ----------------------

SPZ_HEADER = "<IIIBBBB"


def _raw(points: Any = ((0.5, 0.5, 0.5),), bits: int = 12) -> bytearray:
    import gzip

    return bytearray(gzip.decompress(glb_spz_bytes(glb_at(points, bits))))


def glb_at(points: Any, bits: int) -> bytes:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = p.shape[0]
    quats = np.tile([0, 0, 0, 1.0], (n, 1))
    spz = pack_spz(p, np.zeros((n, 3)), np.zeros(n), np.zeros((n, 3)), quats, fractional_bits=bits)
    return build_glb(n, p.min(axis=0).tolist(), p.max(axis=0).tolist(), spz)


def _with_spz(spz: bytes) -> bytes:
    gltf, _ = chunks(SAFE)
    gltf["bufferViews"][0].update(byteOffset=0, byteLength=len(spz))
    gltf["buffers"][0]["byteLength"] = len(spz) + (-len(spz) % 4)
    return repack(gltf, spz)


def _gz(raw: bytes | bytearray) -> bytes:
    import gzip

    return gzip.compress(bytes(raw), mtime=0)


def _header(index: int, value: int) -> Callable[[], bytes]:
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
    "fractional_bits_11_small_tile": lambda: glb_at([[0.5, 0.5, 0.5]], 11),
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
def test_an_spz_stream_not_written_as_the_packer_writes_one_is_refused(
    tmp_path: Path, case: str
) -> None:
    path = write(tmp_path, tileset(), leaf=SPZ_REFUSALS[case]())
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(path)


def test_a_tile_reaching_past_2048_m_at_the_tilers_fewer_bits_passes(tmp_path: Path) -> None:
    """`splat_tiles.fractional_bits_for` is the rule: 12 bits to 2047.99 m, 11 to 4095.99 m."""
    from splat_tiles import fractional_bits_for

    assert fractional_bits_for(3000.0) == 11
    path = write(tmp_path, tileset(), leaf=glb_at([[3000.0, 0.5, 0.5]], 11))
    assert rg.layer_positions(path).shape == (2, 3)
    path = write(tmp_path, tileset(), leaf=glb_at([[3000.0, 0.5, 0.5]], 10))
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(path)


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
def test_an_accessor_not_build_glbs_value_for_value_is_refused(tmp_path: Path, case: str) -> None:
    path = write(tmp_path, tileset(), leaf=ACCESSOR_REFUSALS[case]())
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(path)


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
def test_a_tileset_number_the_tiler_never_writes_is_refused(tmp_path: Path, case: str) -> None:
    document = tileset()
    NUMBER_REFUSALS[case](document)
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(write(tmp_path, document))


def test_whole_metre_nested_translations_pass(tmp_path: Path) -> None:
    document = tileset()
    _translate(3.0, -2.0, 1.0)(document)
    xyz = rg.layer_positions(write(tmp_path, document))
    assert sorted(xyz[:, 0].tolist()) == [0.5, 2.5]


API_TESTS = REPO / "apps" / "api" / "tests" / "test_layer_allowlist.py"
API_POSITIONS = REPO / "apps" / "api" / "app" / "worker" / "positions.py"


def _case_names(path: Path) -> dict[str, set[str]]:
    """Each `*_REFUSALS` dict literal's keys in a test file, read without importing it."""
    import ast

    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        value = getattr(node, "value", None)
        if (
            isinstance(target, ast.Name)
            and target.id.endswith("_REFUSALS")
            and isinstance(value, ast.Dict)
        ):
            keys = {k.value for k in value.keys if isinstance(k, ast.Constant)}
            assert len(keys) == len(value.keys), f"{target.id}: every key a plain string"
            found[target.id] = keys
    return found


def test_the_cases_are_the_apis_and_the_apis_are_these() -> None:
    """One allowlist, two transcriptions, one list of refusals: the same `*_REFUSALS`
    dicts with the same names in apps/api tests/test_layer_allowlist.py and here, both
    ways (Codex R5: the check went one way, and a captures-only case passed it)."""
    here, there = _case_names(Path(__file__)), _case_names(API_TESTS)
    assert (
        set(here)
        == set(there)
        == {
            "TILESET_REFUSALS",
            "GLB_REFUSALS",
            "SPZ_REFUSALS",
            "ACCESSOR_REFUSALS",
            "NUMBER_REFUSALS",
        }
    )
    for name in here:
        assert here[name] == there[name], (name, here[name] ^ there[name])


def _api_positions() -> Any:
    """The API gate's own module (apps/api app/worker/positions.py: numpy and the standard
    library), loaded from its file without the API's environment."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("api_positions", API_POSITIONS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _every_case() -> list[tuple[str, Callable[[], tuple[dict[str, Any], bytes]]]]:
    cases: list[tuple[str, Callable[[], tuple[dict[str, Any], bytes]]]] = []
    for table in (TILESET_REFUSALS, NUMBER_REFUSALS):
        for name, change in table.items():

            def tiles(
                change: Callable[[dict[str, Any]], object] = change,
            ) -> tuple[dict[str, Any], bytes]:
                document = tileset()
                change(document)
                return document, SAFE

            cases.append((name, tiles))
    for table in (GLB_REFUSALS, SPZ_REFUSALS, ACCESSOR_REFUSALS):
        for name, make in table.items():
            cases.append((name, lambda make=make: (tileset(), make())))
    return cases


@pytest.mark.parametrize(("case", "build"), _every_case(), ids=[n for n, _ in _every_case()])
def test_every_case_is_refused_by_both_validators(
    tmp_path: Path, case: str, build: Callable[[], tuple[dict[str, Any], bytes]]
) -> None:
    """Each refusal, run through the publish pre-check (`rescan_gate.layer_positions`) and
    through the API gate's own functions (`packer_placements`, then `packer_spz` of every
    tile it places) on the very same files: both refuse."""
    api = _api_positions()
    document, leaf = build()
    path = write(tmp_path, document, leaf=leaf)
    for name in ("f1.glb?v=1", "f1.glb#x", "f1%2Eglb", "nested.json"):
        (tmp_path / name).write_bytes(SAFE)
    with pytest.raises(rg.RescanGateError):
        rg.layer_positions(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(api.TileFormatError):
        for uri, _ in api.packer_placements(loaded):
            api.packer_spz((tmp_path / uri).read_bytes())


def test_the_packers_output_passes_both_validators() -> None:
    api = _api_positions()
    for path in FIXTURES:
        document = json.loads(path.read_text(encoding="utf-8"))
        for uri, _ in api.packer_placements(document):
            api.packer_spz((path.parent / uri).read_bytes())
        assert rg.layer_positions(path).shape[0] > 0
