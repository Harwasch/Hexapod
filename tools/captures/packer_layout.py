"""What a published fill layer may be: exactly what the tiler writes (ADR 0016 § Fill).

The publish pre-check (`rescan_gate.layer_positions`) reads a tile's SPZ block and each
tile's translation, which is the whole placement only on the layout
`splat_tiles.write_tiles` and `splat_tiles.build_glb` write. Cesium honours more than that
-- `content.url`, multiple contents, a second (data-URI) buffer, node transforms,
extensions -- so a layer that uses any of it is refused, never read (Codex R4 on #60: no
denylist). A transcription of the API gate's allowlist (apps/api app/worker/positions.py,
`packer_placements` and `packer_spz`), held to the same cases by name
(tests/test_layer_allowlist.py in both projects).
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from splat_tiles import SplatFormatError, tile_placements


class LayoutError(ValueError):
    """A layer tileset or tile GLB not laid out as the tiler writes one; the message says why."""


#: The top-level keys of a layer's tileset, and of each tile and its content, as the tiler
#: writes them.
TILESET_KEYS = frozenset({"asset", "geometricError", "extensionsUsed", "extensions", "root"})
TILE_KEYS = frozenset(
    {"transform", "boundingVolume", "geometricError", "refine", "content", "extras", "children"}
)
CONTENT_KEYS = frozenset({"uri"})
#: The one tileset extension the tiler declares, and what it says.
CONTENT_GLTF = "3DTILES_content_gltf"
SPLAT_EXTENSIONS = frozenset({"KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"})
#: A content uri: a plain relative path of plain names (sidecars.check_relative's rule) to a
#: `.glb`, so no `?`, `#`, `%`, `..`, `./`, scheme, absolute path or backslash.
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_DEPTH = 6

#: `build_glb`'s glTF: its top-level keys, extensions, scene, node and primitive.
GLB_KEYS = frozenset(
    {
        "asset",
        "extensionsUsed",
        "extensionsRequired",
        "scene",
        "scenes",
        "nodes",
        "meshes",
        "materials",
        "accessors",
        "bufferViews",
        "buffers",
    }
)
GLB_USED = frozenset({"KHR_materials_unlit", *SPLAT_EXTENSIONS})
PACKER_NODE: dict[str, Any] = {
    "mesh": 0,
    "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
}
PACKER_MATERIALS: list[dict[str, Any]] = [{"extensions": {"KHR_materials_unlit": {}}}]
PACKER_SPLAT: dict[str, Any] = {
    "KHR_gaussian_splatting": {
        "extensions": {"KHR_gaussian_splatting_compression_spz_2": {"bufferView": 0}}
    }
}
PRIMITIVE_KEYS = frozenset({"mode", "material", "attributes", "extensions"})
ACCESSOR_KEYS = frozenset({"componentType", "count", "type", "min", "max", "normalized"})
_ATTRIBUTE = re.compile(
    r"POSITION|COLOR_0|KHR_gaussian_splatting:(SCALE|ROTATION|SH_DEGREE_[1-3]_COEF_[0-9]{1,2})"
)
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942


def _same(a: object, b: object) -> bool:
    """JSON equality that tells `true` from `1` (Python does not)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    return type(a) is type(b) and a == b


def _index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _names(value: object, expected: frozenset[str], what: str) -> None:
    if (
        not isinstance(value, list)
        or not all(isinstance(v, str) for v in value)
        or len(set(value)) != len(value)
        or set(value) != expected
    ):
        raise LayoutError(f"{what} {value!r} is not the packer's {sorted(expected)}")


def _keys(value: object, allowed: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LayoutError(f"{what} is not an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise LayoutError(f"{what} has {unknown}, which the packer never writes")
    return value


def content_uri(uri: object) -> str:
    """`uri` if it is a plain relative path to a `.glb` beside the tileset, else
    `LayoutError`."""
    parts = uri.split("/") if isinstance(uri, str) else []
    if (
        not parts
        or len(parts) > MAX_DEPTH
        or not all(_SEGMENT.fullmatch(part) for part in parts)
        or not parts[-1].endswith(".glb")
    ):
        raise LayoutError(
            f"content uri {uri!r} is not a plain relative path to a .glb (letters, digits, "
            f"'.', '_' and '-' in at most {MAX_DEPTH} segments, none starting with '.')"
        )
    return str(uri)


def packer_placements(document: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    """`tile_placements` of a tileset laid out as the tiler writes one, else
    `LayoutError`: the tiler's keys only, at the top, on each tile and on each content;
    content named by `uri` alone (never `url`, `contents` or an extension), each a plain
    relative `.glb`; a box for each bounding volume; child transforms translations."""
    _keys(document, TILESET_KEYS, "the tileset")
    if not _same(document.get("asset"), {"version": "1.1"}):
        raise LayoutError(f"tileset asset {document.get('asset')!r} is not 3D Tiles 1.1")
    if "extensionsUsed" in document or "extensions" in document:
        _names(document.get("extensionsUsed"), frozenset({CONTENT_GLTF}), "extensionsUsed")
        extensions = _keys(document.get("extensions"), frozenset({CONTENT_GLTF}), "extensions")
        declared = _keys(
            extensions.get(CONTENT_GLTF),
            frozenset({"extensionsUsed", "extensionsRequired"}),
            CONTENT_GLTF,
        )
        _names(declared.get("extensionsUsed"), SPLAT_EXTENSIONS, f"{CONTENT_GLTF} used")
        _names(declared.get("extensionsRequired"), SPLAT_EXTENSIONS, f"{CONTENT_GLTF} required")
    stack: list[object] = [document.get("root")]
    while stack:
        tile = _keys(stack.pop(), TILE_KEYS, "a tile")
        volume = _keys(tile.get("boundingVolume"), frozenset({"box"}), "a bounding volume")
        _numbers(volume.get("box"), 12, "a bounding box")
        if tile.get("refine", "REPLACE") not in ("REPLACE", "ADD"):
            raise LayoutError(f"refine {tile.get('refine')!r} is not REPLACE or ADD")
        if "content" in tile:
            content_uri(_keys(tile["content"], CONTENT_KEYS, "a tile's content").get("uri"))
        children = tile.get("children", [])
        if not isinstance(children, list):
            raise LayoutError("a tile's children are not a list")
        stack.extend(children)
    try:
        return [(uri, offset) for uri, offset in tile_placements(document)]
    except SplatFormatError as error:
        raise LayoutError(str(error)) from error


def _numbers(value: object, count: int, what: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != count
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        raise LayoutError(f"{what} is not {count} finite numbers")


def packer_spz(data: bytes) -> bytes:
    """The SPZ block of a GLB laid out exactly as `splat_tiles.build_glb` writes one, else
    `LayoutError`: a JSON chunk then a BIN chunk and nothing after; one buffer, the BIN
    chunk itself (no `uri`: no data-URI or external buffer); one buffer view in it; the
    packer's extensions exactly, used and required; one scene of one node, that node
    exactly `PACKER_NODE`; one mesh of one point primitive whose splats are that view's SPZ
    (no indices, morph targets, other attributes); accessors with no data of their own (no
    buffer view, no sparse); no skins, animations, instancing or anything else."""
    if len(data) < 28:
        raise LayoutError("not a GLB: shorter than its header")
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise LayoutError("not a whole binary glTF 2.0")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    bin_at = 20 + json_length
    if json_type != _JSON_CHUNK or bin_at + 8 > len(data):
        raise LayoutError("not a GLB of a JSON chunk then a BIN chunk")
    bin_length, bin_type = struct.unpack_from("<II", data, bin_at)
    if bin_type != _BIN_CHUNK or bin_at + 8 + bin_length != len(data):
        raise LayoutError("not a GLB of a JSON chunk then a BIN chunk, and nothing after")
    try:
        gltf = json.loads(data[20:bin_at])
    except ValueError as error:
        raise LayoutError(f"not a splat GLB: {error!r}") from error
    _keys(gltf, GLB_KEYS, "the glTF")
    asset = _keys(gltf.get("asset"), frozenset({"version", "generator"}), "the glTF asset")
    if asset.get("version") != "2.0":
        raise LayoutError(f"glTF version {asset.get('version')!r} is not 2.0")
    _names(gltf.get("extensionsUsed"), GLB_USED, "extensionsUsed")
    _names(gltf.get("extensionsRequired"), SPLAT_EXTENSIONS, "extensionsRequired")
    if not (
        _same(gltf.get("scene"), 0)
        and _same(gltf.get("scenes"), [{"nodes": [0]}])
        and _same(gltf.get("nodes"), [PACKER_NODE])
    ):
        raise LayoutError("not build_glb's one scene of one node with its matrix")
    if "materials" in gltf and not _same(gltf["materials"], PACKER_MATERIALS):
        raise LayoutError("not build_glb's one unlit material")
    meshes = gltf.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise LayoutError("not one mesh")
    primitives = _keys(meshes[0], frozenset({"primitives"}), "the mesh").get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise LayoutError("not one primitive")
    primitive = _keys(primitives[0], PRIMITIVE_KEYS, "the primitive")
    if not _same(primitive.get("extensions"), PACKER_SPLAT):
        raise LayoutError("the primitive's splats are not build_glb's SPZ in view 0")
    if not _same(primitive.get("mode", 0), 0):
        raise LayoutError("the primitive is not points")
    if "material" in primitive and not ("materials" in gltf and _same(primitive["material"], 0)):
        raise LayoutError("the primitive's material is not the unlit one")
    accessors = gltf.get("accessors")
    if not isinstance(accessors, list):
        raise LayoutError("no accessors")
    for accessor in accessors:
        _keys(accessor, ACCESSOR_KEYS, "an accessor")
    attributes = primitive.get("attributes")
    if (
        not isinstance(attributes, dict)
        or "POSITION" not in attributes
        or not all(_ATTRIBUTE.fullmatch(name) for name in attributes)
        or not all(_index(v) and v < len(accessors) for v in attributes.values())
    ):
        raise LayoutError(f"attributes {attributes!r} are not build_glb's")
    buffers = gltf.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise LayoutError("not one buffer")
    size = _keys(buffers[0], frozenset({"byteLength"}), "the buffer").get("byteLength")
    if not _index(size) or size > bin_length:
        raise LayoutError("the buffer is not the BIN chunk")
    views = gltf.get("bufferViews")
    if not isinstance(views, list) or len(views) != 1:
        raise LayoutError("not one buffer view")
    view = _keys(views[0], frozenset({"buffer", "byteOffset", "byteLength"}), "the view")
    offset, length = view.get("byteOffset", 0), view.get("byteLength")
    if not (_same(view.get("buffer"), 0) and _index(offset) and _index(length)):
        raise LayoutError("the buffer view is not one into the BIN chunk")
    if offset + length > size:
        raise LayoutError("the SPZ block runs past the buffer")
    start = bin_at + 8 + offset
    return data[start : start + length]
