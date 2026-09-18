"""Shrink the textures inside ODM's b3dm mesh tiles.

ODM writes each tile's texture as one PNG of the full atlas (about 13 MB per tile), which is
far more than a browser needs for a tile that covers tens of metres. This rewrites every
b3dm under a tileset folder with the same geometry and a JPEG texture capped at a maximum
edge length, so a whole site fits in a repository.

Usage:
    python shrink_b3dm.py <tileset_dir> <out_dir> [--max-edge 2048] [--quality 85]
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
from pathlib import Path

from PIL import Image

# ODM writes 16k-square atlases; that is the input, not an attack.
Image.MAX_IMAGE_PIXELS = None

B3DM_HEADER = struct.Struct("<4sIIIIII")
GLB_HEADER = struct.Struct("<4sII")
CHUNK = struct.Struct("<II")
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def split_b3dm(raw: bytes) -> tuple[bytes, bytes]:
    magic, _version, _length, ftj, ftb, btj, btb = B3DM_HEADER.unpack_from(raw)
    if magic != b"b3dm":
        raise ValueError("not a b3dm")
    tables = raw[B3DM_HEADER.size : B3DM_HEADER.size + ftj + ftb + btj + btb]
    return tables, raw[B3DM_HEADER.size + ftj + ftb + btj + btb :]


def split_glb(glb: bytes) -> tuple[dict, bytes]:
    magic, _version, length = GLB_HEADER.unpack_from(glb)
    if magic != b"glTF":
        raise ValueError("not a glb")
    # A b3dm may carry padding after the glb; trust the glb's own length.
    end = min(length, len(glb))
    offset = GLB_HEADER.size
    gltf: dict | None = None
    binary = b""
    while offset + CHUNK.size <= end:
        length, kind = CHUNK.unpack_from(glb, offset)
        body = glb[offset + CHUNK.size : offset + CHUNK.size + length]
        if kind == JSON_CHUNK:
            gltf = json.loads(body)
        elif kind == BIN_CHUNK:
            binary = body
        offset += CHUNK.size + length
    if gltf is None:
        raise ValueError("glb without a JSON chunk")
    return gltf, binary


def join_glb(gltf: dict, binary: bytes) -> bytes:
    json_bytes = pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    binary = pad4(binary)
    total = GLB_HEADER.size + CHUNK.size * 2 + len(json_bytes) + len(binary)
    return (
        GLB_HEADER.pack(b"glTF", 2, total)
        + CHUNK.pack(len(json_bytes), JSON_CHUNK)
        + json_bytes
        + CHUNK.pack(len(binary), BIN_CHUNK)
        + binary
    )


def shrink_image(data: bytes, max_edge: int, quality: int) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True, subsampling=1)
    return out.getvalue()


def shrink_glb(glb: bytes, max_edge: int, quality: int) -> bytes:
    gltf, binary = split_glb(glb)
    views = gltf.get("bufferViews", [])
    image_views = {img["bufferView"] for img in gltf.get("images", []) if "bufferView" in img}
    new_binary = bytearray()
    for index, view in enumerate(views):
        start = view.get("byteOffset", 0)
        chunk = binary[start : start + view["byteLength"]]
        if index in image_views:
            chunk = shrink_image(chunk, max_edge, quality)
        view["byteOffset"] = len(new_binary)
        view["byteLength"] = len(chunk)
        new_binary += pad4(chunk)
    for img in gltf.get("images", []):
        if "bufferView" in img:
            img["mimeType"] = "image/jpeg"
            if "name" in img:
                img["name"] = Path(img["name"]).with_suffix(".jpg").name
    gltf["buffers"][0]["byteLength"] = len(new_binary)
    return join_glb(gltf, bytes(new_binary))


def shrink_b3dm(raw: bytes, max_edge: int, quality: int) -> bytes:
    tables, glb = split_b3dm(raw)
    magic, version, _length, ftj, ftb, btj, btb = B3DM_HEADER.unpack_from(raw)
    new_glb = shrink_glb(glb, max_edge, quality)
    total = B3DM_HEADER.size + len(tables) + len(new_glb)
    return B3DM_HEADER.pack(magic, version, total, ftj, ftb, btj, btb) + tables + new_glb


def shrink_tileset(src: Path, dst: Path, max_edge: int, quality: int) -> dict[str, int]:
    before = after = tiles = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".b3dm":
            raw = path.read_bytes()
            out = shrink_b3dm(raw, max_edge, quality)
            target.write_bytes(out)
            before += len(raw)
            after += len(out)
            tiles += 1
        else:
            shutil.copyfile(path, target)
    return {"tiles": tiles, "bytes_before": before, "bytes_after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tileset_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--max-edge", type=int, default=2048)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args()
    print(json.dumps(shrink_tileset(args.tileset_dir, args.out_dir, args.max_edge, args.quality)))


if __name__ == "__main__":
    main()
