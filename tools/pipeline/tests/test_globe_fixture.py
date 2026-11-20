"""The iOS/web fixture must be actual output of the existing production packager."""

import json
import struct
from pathlib import Path

from scripts.make_globe_fixture import OUTPUT, generate


def test_tiny_khr_fixture(tmp_path: Path) -> None:
    generate(tmp_path / "splat")
    for filename in ("tileset.json", "splat.glb", "collision.bin", "viewcones.bin"):
        assert (tmp_path / "splat" / filename).read_bytes() == (OUTPUT / filename).read_bytes()
    tile = json.loads((OUTPUT / "tileset.json").read_text())
    assert (
        "KHR_gaussian_splatting" in tile["extensions"]["3DTILES_content_gltf"]["extensionsRequired"]
    )
    assert tile["root"]["content"]["uri"] == "splat.glb"
    assert len(tile["root"]["transform"]) == 16
    raw = (OUTPUT / "splat.glb").read_bytes()
    assert len(raw) < 4096
    assert struct.unpack_from("<4sII", raw) == (b"glTF", 2, len(raw))
    size = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20 : 20 + size])
    assert "KHR_gaussian_splatting" in gltf["extensionsRequired"]
    assert gltf["accessors"][0]["count"] == 4
