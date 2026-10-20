"""Per-tile origins (docs/DECISIONS/0011-splat-tile-origins.md): SPZ stores positions as
24-bit fixed point, so at 12 fractional bits a tile spans +/-2047.99976 m and no more. The
packer refuses what does not fit instead of wrapping it, moves a non-root tile that does not
fit to an origin of its own (a translation-only `transform`, relative to its parent), and
gives a tile that still does not fit fewer fractional bits."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import splat_tiles
import synthetic_far
from rig_tiles import glb_spz
from splat_render import save_ply
from splat_tiles import SPZ_FIXED_MAX, SplatFormatError, pack_spz, unpack_spz

REPO_ROOT = Path(__file__).resolve().parents[3]
LOD_FIXTURE = REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod"


def _pack(xyz: np.ndarray, **kwargs: int) -> bytes:
    n = xyz.shape[0]
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (n, 1))
    return pack_spz(
        xyz.astype(np.float32),
        np.zeros((n, 3), np.float32),
        np.zeros(n, np.float32),
        np.full((n, 3), -3.0, np.float32),
        quat,
        **kwargs,
    )


def _xyz(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_pack_spz_refuses_a_position_past_the_24_bit_range() -> None:
    # Before the refusal, 3000 m wrapped to 3000 - 4096 = -1096 m without a word.
    with pytest.raises(SplatFormatError, match=r"2047\.99"):
        _pack(_xyz((3000.0, 0.0, 0.0)))


def test_pack_spz_packs_the_last_step_inside_the_range() -> None:
    edge = SPZ_FIXED_MAX / 4096
    xyz = _xyz((edge, -edge, 0.0), (-edge, edge, edge))
    back = unpack_spz(_pack(xyz))
    assert np.array_equal(back["x"], xyz[:, 0])
    assert np.array_equal(back["y"], xyz[:, 1])
    assert np.array_equal(back["z"], xyz[:, 2])


def test_pack_spz_refuses_non_finite_positions() -> None:
    with pytest.raises(SplatFormatError, match="finite"):
        _pack(_xyz((np.nan, 0.0, 0.0)))
    with pytest.raises(SplatFormatError, match="finite"):
        _pack(_xyz((np.inf, 0.0, 0.0)))


def test_pack_spz_writes_the_fractional_bits_it_is_given() -> None:
    blob = _pack(_xyz((3000.5, -3000.5, 1.0)), fractional_bits=10)
    raw = gzip.decompress(blob)
    assert raw[13] == 10
    back = unpack_spz(blob)
    assert float(back["x"][0]) == 3000.5
    assert float(back["y"][0]) == -3000.5


def test_pack_spz_refuses_fractional_bits_outside_the_header_byte() -> None:
    with pytest.raises(ValueError, match="fractional"):
        _pack(_xyz((1.0, 0.0, 0.0)), fractional_bits=24)
    with pytest.raises(ValueError, match="fractional"):
        _pack(_xyz((1.0, 0.0, 0.0)), fractional_bits=-1)


def test_the_default_still_packs_twelve_fractional_bits() -> None:
    assert gzip.decompress(_pack(_xyz((1.0, 2.0, 3.0))))[13] == splat_tiles.SPZ_FRACTIONAL_BITS


# ------------------------------------------------------------------ the frame a tile is in


def test_tile_frame_keeps_origin_zero_when_it_fits() -> None:
    xyz = _xyz((2047.0, -2047.0, 0.0), (-100.0, 5.0, 3.0))
    origin, bits = splat_tiles.tile_frame(xyz, may_move=True)
    assert not origin.any() and bits == 12


def test_tile_frame_snaps_to_whole_metres_without_negative_zero() -> None:
    xyz = _xyz((2999.6, -0.4, -0.2), (3100.3, 0.3, 0.1))
    origin, bits = splat_tiles.tile_frame(xyz, may_move=True)
    assert origin.tolist() == [3050.0, 0.0, 0.0]
    assert not np.signbit(origin).any()
    assert bits == 12


def test_fractional_bits_for_steps_down_at_each_power_of_two() -> None:
    assert splat_tiles.fractional_bits_for(2047.9) == 12
    assert splat_tiles.fractional_bits_for(2048.1) == 11
    assert splat_tiles.fractional_bits_for(4095.9) == 11
    assert splat_tiles.fractional_bits_for(4096.1) == 10
    with pytest.raises(SplatFormatError, match="whole metres"):
        splat_tiles.fractional_bits_for(1e7)


def test_the_root_never_takes_an_origin() -> None:
    xyz = _xyz((-5200.0, 0.0, 0.0), (5200.0, 1800.0, 1.0))
    origin, bits = splat_tiles.tile_frame(xyz, may_move=False)
    assert not origin.any() and bits == 10  # 5200 m needs 10 bits; 11 holds 4095.99
    moved, _ = splat_tiles.tile_frame(xyz, may_move=True)
    assert moved.tolist() == [0.0, 900.0, 0.0]


def _tiles(tileset: dict) -> list[dict]:
    out, stack = [], [tileset["root"]]
    while stack:
        tile = stack.pop()
        out.append(tile)
        stack.extend(tile.get("children", []))
    return out


def _header_bits(path: Path) -> int:
    return gzip.decompress(glb_spz(path))[13]


def test_a_scan_that_fits_writes_no_tile_transform() -> None:
    tileset = json.loads((LOD_FIXTURE / "tileset.json").read_text(encoding="utf-8"))
    tiles = _tiles(tileset)
    assert len(tiles) > 10
    assert all("transform" not in tile for tile in tiles if tile is not tileset["root"])
    assert {_header_bits(LOD_FIXTURE / tile["content"]["uri"]) for tile in tiles} == {12}
    assert not any(v.any() for v in splat_tiles.tile_offsets(tileset).values())


def _translate(x: float, y: float, z: float) -> list[float]:
    return [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, x, y, z, 1.0]


def test_tile_offsets_composes_nested_translations() -> None:
    tileset = {
        "root": {
            # The globe placement is the root's own, and no part of any tile's offset.
            "transform": _translate(1e6, 2e6, 3e6),
            "content": {"uri": "splat.glb"},
            "children": [
                {
                    "transform": _translate(2548.0, 900.0, 0.0),
                    "content": {"uri": "./splat_4.glb"},
                    "children": [
                        {
                            "transform": _translate(1342.0, 0.0, 0.0),
                            "contents": [{"uri": "splat_4-4.glb"}],
                        },
                        {"content": {"uri": "splat_4-0.glb"}},
                    ],
                },
                {"content": {"uri": "splat_0.glb"}},
            ],
        }
    }
    offsets = {k: v.tolist() for k, v in splat_tiles.tile_offsets(tileset).items()}
    assert offsets == {
        "splat.glb": [0.0, 0.0, 0.0],
        "splat_4.glb": [2548.0, 900.0, 0.0],
        "splat_4-4.glb": [3890.0, 900.0, 0.0],
        "splat_4-0.glb": [2548.0, 900.0, 0.0],
        "splat_0.glb": [0.0, 0.0, 0.0],
    }


def test_tile_offsets_refuses_a_rotating_child_transform() -> None:
    turned = _translate(10.0, 0.0, 0.0)
    turned[0], turned[1], turned[4], turned[5] = 0.0, 1.0, -1.0, 0.0
    tileset = {
        "root": {
            "content": {"uri": "splat.glb"},
            "children": [{"transform": turned, "content": {"uri": "a.glb"}}],
        }
    }
    with pytest.raises(SplatFormatError, match="translation"):
        splat_tiles.tile_offsets(tileset)
    projective = _translate(0.0, 0.0, 0.0)
    projective[3] = 0.5
    tileset["root"]["children"][0]["transform"] = projective
    with pytest.raises(SplatFormatError, match="translation"):
        splat_tiles.tile_offsets(tileset)


def test_site_positions_is_the_identity_without_an_offset() -> None:
    local = _xyz((1.0, -0.0, 2.0))
    assert splat_tiles.site_positions(local, None) is local
    assert splat_tiles.site_positions(local, np.zeros(3)) is local
    moved = splat_tiles.site_positions(_xyz((-5.0, 0.25, 0.0)), np.array([5.0, 3000.0, 0.0]))
    assert moved.dtype == np.dtype("<f4") and moved.tolist() == [[0.0, 3000.25, 0.0]]
    assert not np.signbit(moved).any()


# --------------------------------------------------------------- a site 10.4 km across


@pytest.fixture(scope="module")
def far(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("far")
    ply = root / "far.ply"
    save_ply(ply, synthetic_far.scene())
    stats = splat_tiles.convert(
        ply,
        root / "splat",
        synthetic_far.LAT,
        synthetic_far.LON,
        synthetic_far.HEIGHT,
        tile_gaussians=synthetic_far.TILE_GAUSSIANS,
    )
    return ply, root / "splat", stats


def test_a_ten_kilometre_site_decodes_back_within_each_tiles_quantum(
    far: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    ply, tiles_dir, stats = far
    assert stats["dropped"] == 0
    assert stats["extent_m"] > 4000
    assert stats["offset_tiles"] >= 4
    assert stats["min_fractional_bits"] <= 11
    tileset = json.loads((tiles_dir / "tileset.json").read_text(encoding="utf-8"))
    root = tileset["root"]
    offsets = splat_tiles.tile_offsets(tileset)

    nested = [
        child
        for tile in _tiles(tileset)
        if tile is not root and "transform" in tile
        for child in tile.get("children", [])
        if "transform" in child
    ]
    assert nested, "no tile with a transform sits under another"

    truth: dict[str, tuple[splat_tiles.Tile, np.ndarray]] = {}

    def emit(tile: splat_tiles.Tile, gaussians: splat_tiles.Gaussians, _keys: object) -> None:
        truth[tile.uri] = (tile, gaussians.xyz.astype(np.float64))

    splat_tiles.hierarchy(ply, 0.02, synthetic_far.TILE_GAUSSIANS, tmp_path, emit)
    assert sorted(truth) == sorted(offsets)
    source = splat_tiles.read_ply(ply)
    snapped = np.stack([source["x"], source["y"], source["z"]], axis=1).astype(np.float64)
    leaves = {t["content"]["uri"] for t in _tiles(tileset) if not t.get("children")}

    boxes = {t["content"]["uri"]: t["boundingVolume"]["box"] for t in _tiles(tileset)}
    for uri, (tile, site) in truth.items():
        path = tiles_dir / uri
        bits = _header_bits(path)
        assert bits <= 12
        columns = unpack_spz(glb_spz(path))
        decoded = np.stack([columns["x"], columns["y"], columns["z"]], axis=1).astype(np.float64)
        composed = decoded + offsets[uri]
        error = float(np.abs(composed - site).max())
        assert error <= 2.0 ** -(bits + 1) + 1e-9, (uri, bits, error)
        if uri in leaves and bits == 12:
            # A leaf is its originals: on the 1/4096 grid, so back exactly.
            rows = {tuple(p) for p in snapped}
            assert all(tuple(p) in rows for p in composed), uri
        box = np.asarray(boxes[uri], dtype=np.float64)
        box[:3] += offsets[uri]
        assert np.allclose(box, splat_tiles._box(tile.low, tile.high), rtol=0, atol=1e-9), uri
    # The root is the tileset's frame: never moved, whatever it holds. Its transform is the
    # globe placement alone, and it holds more than 12 bits can reach, so it has fewer.
    assert root["transform"] == splat_tiles.enu_to_ecef(
        synthetic_far.LAT, synthetic_far.LON, synthetic_far.HEIGHT
    )
    assert not offsets["splat.glb"].any() and _header_bits(tiles_dir / "splat.glb") < 12
    exact_leaves = sum(1 for uri in leaves if _header_bits(tiles_dir / uri) == 12)
    assert exact_leaves == len(leaves)
