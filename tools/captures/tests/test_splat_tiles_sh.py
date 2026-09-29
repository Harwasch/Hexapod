"""Spherical harmonics in the tiles: a trained PLY's view-dependent colour, carried.

What has to hold for CesiumJS 1.145 (and Spark) to draw it, and for the bytes to be SPZ's:

* the SH bytes are nianticlabs/spz's: quantised by its `quantizeSH` (5 bits for degree 1,
  4 for 2 and 3, rounding half away from zero), coefficient-major with the colour innermost,
  read from the PLY's channel-major `f_rest_*`, and the header's degree byte says which;
* the glTF declares them as `KHR_gaussian_splatting:SH_DEGREE_l_COEF_n`, 3, 8 or 15 of
  them and never `SH_DEGREE_0_COEF_0` -- CesiumJS *counts* them to learn the degree;
* every tile of a tileset has the same degree (CesiumJS draws one primitive at the first
  tile's degree), and a merged parent's SH is its members' weighted mean, like its colour;
* `sh_degree` caps the bands, and a PLY without SH -- every committed fixture -- packs to
  exactly the bytes it did before SH existed;
* `data/tiles/synthetic-tree-sh`, the SH-3 fixture e2e/splatLod.spec.ts renders, is the
  packer's own output.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_splat_tiles_lod import PROPERTIES, scene, tiles_of

import splat_tiles

REST = [f"f_rest_{i}" for i in range(45)]


def write_sh_ply(path: Path, columns: dict[str, np.ndarray], rest: int = 45) -> Path:
    """A 3DGS PLY in the trainers' column order: x y z, f_dc, f_rest, opacity, scale, rot."""
    names = [*PROPERTIES[:6], *REST[:rest], *PROPERTIES[6:]]
    count = columns["x"].shape[0]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in names] + ["end_header"]
    rows = np.zeros(count, dtype=[(name, "<f4") for name in names])
    for name in names:
        rows[name] = columns[name]
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())
    return path


def with_sh(columns: dict[str, np.ndarray], seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = columns["x"].shape[0]
    out = dict(columns)
    out.update({name: rng.normal(0.0, 0.3, count).astype(np.float32) for name in REST})
    return out


def sh_of(columns: dict[str, np.ndarray], degree: int = 3) -> np.ndarray:
    """The PLY's f_rest_* as (n, K, 3), coefficient-major: f_rest_{c * 15 + j} -> [:, j, c]."""
    dim = splat_tiles.SH_DIMS[degree]
    return np.stack(
        [np.stack([columns[f"f_rest_{c * 15 + j}"] for c in range(3)], axis=1) for j in range(dim)],
        axis=1,
    )


def glb_parts(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    offset, chunks = 12, {}
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunks[kind] = raw[offset + 8 : offset + 8 + length]
        offset += 8 + length
    gltf = json.loads(chunks[0x4E4F534A])
    view = gltf["bufferViews"][0]
    return gltf, chunks[0x004E4942][view.get("byteOffset", 0) :][: view["byteLength"]]


def cesium_sh_degree(gltf: dict[str, Any]) -> int:
    """GaussianSplat3DTileContent.js `degreeAndCoefFromAttributes`, as CesiumJS 1.145 has it:
    the number of attributes whose name holds SH_DEGREE_, and 3/8/15 -> 1/2/3, else 0."""
    names = gltf["meshes"][0]["primitives"][0]["attributes"]
    return {3: 1, 8: 2, 15: 3}.get(sum("SH_DEGREE_" in name for name in names), 0)


# ------------------------------------------------------------------------ the SPZ bytes


def spz_quantize(x: float, bucket: int) -> int:
    """nianticlabs/spz `quantizeSH`, one float32 at a time, the C way: `std::round` half
    away from zero, integer division truncating toward zero, then a clamp to a byte."""
    scaled = float(np.float32(x) * np.float32(128.0))
    rounded = math.floor(abs(scaled) + 0.5) * (1 if scaled >= 0 else -1)
    q = int(rounded) + 128
    q = int((q + bucket // 2) / bucket) * bucket
    return min(max(q, 0), 255)


def test_quantisation_is_spz_quantize_sh() -> None:
    rng = np.random.default_rng(1)
    edges = np.array([k / 256 for k in range(-600, 601)], dtype=np.float32)  # every half step
    values = np.concatenate(
        [edges, rng.normal(0, 0.4, 3000), rng.normal(0, 3.0, 300), [0.0, -0.0, 1.0, -1.0]]
    ).astype(np.float32)
    values = values[: values.size // 45 * 45].reshape(-1, 15, 3)
    got = splat_tiles.quantize_sh(values)
    for j in range(15):
        bucket = 8 if j < 3 else 16
        expected = [spz_quantize(float(x), bucket) for x in values[:, j, :].reshape(-1)]
        assert got[:, j, :].reshape(-1).tolist() == expected, f"coefficient {j}"
    # 0 is a bucket centre in every band; half steps round away from zero, as std::round.
    zero = splat_tiles.quantize_sh(np.zeros((1, 15, 3), np.float32))
    assert (zero == 128).all()
    halves = splat_tiles.quantize_sh(np.full((1, 3, 3), 2.5 / 128, np.float32))
    assert (halves == 128).all()  # 131 in a bucket of 8 -> 128
    assert splat_tiles.quantize_sh(np.full((1, 3, 3), 3.5 / 128, np.float32))[0, 0, 0] == 136


def test_non_finite_coefficients_are_written_as_zero() -> None:
    sh = np.zeros((1, 8, 3), np.float32)
    sh[0, 0, 0], sh[0, 3, 1], sh[0, 5, 2] = np.nan, np.inf, -np.inf
    assert (splat_tiles.quantize_sh(sh) == 128).all()


def test_pack_spz_writes_sh_after_the_rotations_coefficient_major() -> None:
    n = 4
    zeros3 = np.zeros((n, 3), np.float32)
    quat = np.tile(np.float32([0, 0, 0, 1]), (n, 1))
    # A distinct value per (point, coefficient, colour) that survives quantisation.
    sh = np.zeros((n, 8, 3), np.float32)
    for i in range(n):
        for j in range(8):
            for c in range(3):
                sh[i, j, c] = ((i * 24 + j * 3 + c) % 15 - 7) * 16 / 128
    raw = gzip.decompress(
        splat_tiles.pack_spz(zeros3, zeros3, np.zeros(n, np.float32), zeros3, quat, sh)
    )
    magic, version, count, degree, bits, flags, _ = struct.unpack_from("<IIIBBBB", raw)
    assert (magic, version, count, degree, bits, flags) == (splat_tiles.SPZ_MAGIC, 2, n, 2, 12, 0)
    body = np.frombuffer(raw, np.uint8, offset=16 + 19 * n)
    assert body.size == n * 8 * 3
    assert np.array_equal(body.reshape(n, 8, 3), splat_tiles.quantize_sh(sh))
    back = splat_tiles.unpack_spz(
        splat_tiles.pack_spz(zeros3, zeros3, zeros3[:, 0], zeros3, quat, sh), sh=True
    )
    for j in range(8):
        for c in range(3):
            assert np.allclose(back[f"f_rest_{c * 8 + j}"], sh[:, j, c], atol=1e-6)


def test_without_sh_nothing_changes() -> None:
    """Degree 0 is the header byte 0 and no SH bytes: the pre-SH packer's output."""
    columns = scene(500, seed=2)
    args = (
        np.stack([columns[a] for a in "xyz"], axis=1),
        np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1),
        columns["opacity"],
        np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1),
        np.stack([columns[f"rot_{i}"] for i in (1, 2, 3, 0)], axis=1),
    )
    raw = gzip.decompress(splat_tiles.pack_spz(*args))
    assert raw[12] == 0 and len(raw) == 16 + 19 * 500
    assert set(splat_tiles.unpack_spz(splat_tiles.pack_spz(*args), sh=True)) == set(
        splat_tiles.unpack_spz(splat_tiles.pack_spz(*args))
    )


# ------------------------------------------------------------------------ the PLY side


def test_the_ply_degree_is_read_from_f_rest(tmp_path: Path) -> None:
    columns = with_sh(scene(200))
    for rest, degree in ((0, 0), (9, 1), (24, 2), (45, 3), (30, 2)):
        ply = write_sh_ply(tmp_path / f"r{rest}.ply", columns, rest)
        assert splat_tiles.ply_sh_degree(splat_tiles.ply_layout(ply)) == degree
    # f_rest_0..f_rest_8 is 3 a channel: red 0-2, green 3-5, blue 6-8, read colour-innermost.
    layout = splat_tiles.ply_layout(tmp_path / "r9.ply")
    assert splat_tiles.ply_sh_columns(layout, 1) == tuple(
        f"f_rest_{i}" for i in (0, 3, 6, 1, 4, 7, 2, 5, 8)
    )


def test_a_ply_with_a_broken_f_rest_block_is_refused(tmp_path: Path) -> None:
    columns = with_sh(scene(200))
    ply = write_sh_ply(tmp_path / "odd.ply", columns, 10)
    with pytest.raises(splat_tiles.SplatFormatError, match="f_rest"):
        splat_tiles.convert(ply, tmp_path / "out", 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------ the tileset


def test_a_single_tile_carries_the_plys_sh_in_ply_order(tmp_path: Path) -> None:
    columns = with_sh(scene(3_000))
    ply = write_sh_ply(tmp_path / "s.ply", columns)
    stats = splat_tiles.convert(ply, tmp_path / "out", 46.84, -91.99, 0.0)
    assert stats["sh_degree"] == 3 and stats["tiles"] == 1
    gltf, spz = glb_parts(tmp_path / "out" / "splat.glb")
    assert cesium_sh_degree(gltf) == 3
    primitive = gltf["meshes"][0]["primitives"][0]
    names = [name for name in primitive["attributes"] if "SH_DEGREE_" in name]
    assert names == splat_tiles.sh_attributes(3)
    assert not any("SH_DEGREE_0" in name for name in names)
    for name in names:
        accessor = gltf["accessors"][primitive["attributes"][name]]
        assert accessor == {"componentType": 5126, "count": 3_000, "type": "VEC3"}
    raw = gzip.decompress(spz)
    assert raw[12] == 3
    body = np.frombuffer(raw, np.uint8, offset=16 + 19 * 3_000).reshape(3_000, 15, 3)
    assert np.array_equal(body, splat_tiles.quantize_sh(sh_of(columns)))


def test_sh_degree_caps_the_bands(tmp_path: Path) -> None:
    columns = with_sh(scene(3_000))
    ply = write_sh_ply(tmp_path / "s.ply", columns)
    for cap in (1, 2):
        splat_tiles.convert(ply, tmp_path / f"d{cap}", 0.0, 0.0, 0.0, sh_degree=cap)
        gltf, spz = glb_parts(tmp_path / f"d{cap}" / "splat.glb")
        raw = gzip.decompress(spz)
        assert raw[12] == cap == cesium_sh_degree(gltf)
        body = np.frombuffer(raw, np.uint8, offset=16 + 19 * 3_000)
        assert np.array_equal(body, splat_tiles.quantize_sh(sh_of(columns, cap)).reshape(-1))
    # Capped at 0, an SH PLY packs to exactly the bytes of the same PLY without f_rest_*.
    plain = write_sh_ply(tmp_path / "plain.ply", columns, 0)
    splat_tiles.convert(ply, tmp_path / "d0", 0.0, 0.0, 0.0, sh_degree=0, tile_gaussians=700)
    splat_tiles.convert(plain, tmp_path / "p", 0.0, 0.0, 0.0, tile_gaussians=700)
    capped = {p.name: p.read_bytes() for p in (tmp_path / "d0").iterdir()}
    assert capped == {p.name: p.read_bytes() for p in (tmp_path / "p").iterdir()}
    # A cap above the PLY's degree is the PLY's degree; outside 0-3 is refused.
    stats = splat_tiles.convert(plain, tmp_path / "p3", 0.0, 0.0, 0.0, sh_degree=3)
    assert stats["sh_degree"] == 0
    with pytest.raises(ValueError, match="sh_degree"):
        splat_tiles.convert(ply, tmp_path / "bad", 0.0, 0.0, 0.0, sh_degree=4)


def test_the_disk_sort_carries_sh_whatever_the_buckets(tmp_path: Path, monkeypatch) -> None:
    ply = write_sh_ply(tmp_path / "s.ply", with_sh(scene(6_000, seed=9)))
    splat_tiles.convert(ply, tmp_path / "one", 0.0, 0.0, 0.0, tile_gaussians=500)
    monkeypatch.setattr(splat_tiles, "BUCKET_RECORDS", 700)
    monkeypatch.setattr(splat_tiles, "CHUNK_BYTES", 236 * 999)
    splat_tiles.convert(ply, tmp_path / "many", 0.0, 0.0, 0.0, tile_gaussians=500)
    one = {path.name: path.read_bytes() for path in (tmp_path / "one").iterdir()}
    many = {path.name: path.read_bytes() for path in (tmp_path / "many").iterdir()}
    assert one == many and len(one) > 3


BUDGET = 700


@pytest.fixture(scope="module")
def sh_tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], dict]:
    root = tmp_path_factory.mktemp("sh")
    columns = with_sh(scene(12_000))
    ply = write_sh_ply(root / "scene.ply", columns)
    out = root / "splat"
    stats = splat_tiles.convert(ply, out, 46.84, -91.99, 0.0, tile_gaussians=BUDGET)
    return out, json.loads((out / "tileset.json").read_text()), stats | {"columns": columns}


def test_every_tile_has_the_leaves_degree(sh_tree: tuple[Path, dict, dict]) -> None:
    out, tileset, stats = sh_tree
    tiles = tiles_of(tileset)
    assert stats["tiles"] == len(tiles) > 10
    for tile, _, _ in tiles:
        gltf, spz = glb_parts(out / tile["content"]["uri"])
        assert cesium_sh_degree(gltf) == 3 == gzip.decompress(spz)[12], tile["content"]["uri"]


def test_leaves_carry_every_gaussians_sh_unchanged(sh_tree: tuple[Path, dict, dict]) -> None:
    out, tileset, stats = sh_tree
    columns = stats["columns"]
    found = []
    for tile, _, _ in tiles_of(tileset):
        if "children" in tile:
            continue
        decoded = splat_tiles.unpack_spz(glb_parts(out / tile["content"]["uri"])[1], sh=True)
        found.append(np.stack([decoded[name] for name in ("x", "y", "z", *REST)], axis=1))
    got = np.concatenate(found)
    expected_sh = (splat_tiles.quantize_sh(sh_of(columns)).astype(np.float32) - 128) / 128
    channel_major = expected_sh.transpose(0, 2, 1).reshape(-1, 45)
    xyz = np.stack([columns[a] for a in "xyz"], axis=1)
    expected = np.concatenate([xyz, channel_major], axis=1)
    assert np.array_equal(got[np.lexsort(got.T[::-1])], expected[np.lexsort(expected.T[::-1])])


def test_a_merged_parents_sh_is_its_members_weighted_mean(
    sh_tree: tuple[Path, dict, dict],
) -> None:
    """The root against a brute-force weighted mean, over each cell, of the SH the leaves
    are drawn with (SPZ-quantised: the packer keeps SH as the bytes it will write)."""
    out, _, stats = sh_tree
    columns = stats["columns"]
    xyz = np.stack([columns[a] for a in "xyz"], axis=1)
    origin = xyz.min(axis=0).astype(np.float64)
    edge = max(float((xyz.max(axis=0).astype(np.float64) - origin).max()), 1e-3)
    codes = splat_tiles.morton_codes(xyz, origin, edge)
    plan = splat_tiles.plan_tree(np.sort(codes), BUDGET)
    keys = codes >> np.uint64(3 * (splat_tiles.GRID_BITS - plan.level))
    cells = np.unique(keys)
    raw = gzip.decompress(glb_parts(out / "splat.glb")[1])
    got = np.frombuffer(raw, np.uint8, offset=16 + 19 * cells.size).reshape(cells.size, 15, 3)
    scales = np.exp(np.stack([columns[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    surface = (
        scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2] + scales[:, 1] * scales[:, 2]
    )
    weight = splat_tiles.sigmoid(columns["opacity"].astype(np.float64)) * surface
    drawn = splat_tiles.unquantize_sh(splat_tiles.quantize_sh(sh_of(columns))).astype(np.float64)
    expected = np.empty_like(got)
    for index, key in enumerate(cells):
        rows = keys == key
        mean = (weight[rows, None, None] * drawn[rows]).sum(axis=0) / weight[rows].sum()
        expected[index] = splat_tiles.quantize_sh(mean[None].astype(np.float32))[0]
    # The same bytes, but where a mean sits on a rounding edge and float summation order
    # tips it: then one bucket (8 for degree 1, 16 above), no more.
    difference = np.abs(got.astype(int) - expected.astype(int))
    assert (difference <= 16).all() and (difference[:, :3] <= 8).all()
    assert (difference == 0).mean() > 0.999
    assert (got != 128).mean() > 0.2  # a real comparison, not zeros against zeros


def test_optimised_parents_keep_the_merged_sh(tmp_path: Path) -> None:
    """lod_optimise.py optimises parents in DC colour only; the bands stay the merge's."""
    columns = with_sh(scene(6_000))
    ply = write_sh_ply(tmp_path / "s.ply", columns)
    emitted: dict[str, Any] = {}

    def emit(tile: splat_tiles.Tile, gaussians: splat_tiles.Gaussians, keys: Any) -> None:
        if keys is not None:
            emitted[tile.uri] = (gaussians, keys)

    splat_tiles.hierarchy(ply, 0.02, BUDGET, tmp_path, emit)
    uris = list(emitted)
    offsets = np.cumsum([0, *(emitted[uri][1].size for uri in uris)])
    cat = {
        name: np.concatenate([getattr(emitted[uri][0], name) for uri in uris])
        for name in ("xyz", "sh0", "opacity_logit", "log_scales", "quat_xyzw")
    }
    assert all(emitted[uri][0].sh_rest is None for uri in uris)  # the GPU's view: DC only
    overrides = splat_tiles.ParentOverrides(
        source_sha256=splat_tiles.file_sha256(ply),
        opacity_min=0.02,
        tile_gaussians=BUDGET,
        uris=uris,
        offsets=offsets,
        keys=np.concatenate([emitted[uri][1] for uri in uris]),
        gaussians=splat_tiles.Gaussians(**{**cat, "sh0": cat["sh0"] + np.float32(0.2)}),
    )
    splat_tiles.convert(ply, tmp_path / "merged", 0.0, 0.0, 0.0, tile_gaussians=BUDGET)
    stats = splat_tiles.convert(
        ply, tmp_path / "opt", 0.0, 0.0, 0.0, tile_gaussians=BUDGET, parents=overrides
    )
    assert stats["optimised_parent_gaussians"] == offsets[-1] > 0
    for uri in uris:
        merged = gzip.decompress(glb_parts(tmp_path / "merged" / uri)[1])
        optimised = gzip.decompress(glb_parts(tmp_path / "opt" / uri)[1])
        n = struct.unpack_from("<I", merged, 8)[0]
        assert optimised[12] == 3
        assert merged[16 + 19 * n :] == optimised[16 + 19 * n :]  # the SH bytes
        assert merged[16 + 10 * n : 16 + 13 * n] != optimised[16 + 10 * n : 16 + 13 * n]


# ------------------------------------------------------------------------ the SH fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
TREE_PLY = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"
#: The SH-3 fixture e2e/splatLod.spec.ts loads in CesiumJS: the committed synthetic tree with
#: a view-dependent tint, packed as the LOD fixture is (1,500 gaussians a tile).
#: Regenerated by `PYTHONPATH=. uv run python tests/test_splat_tiles_sh.py`.
SH_FIXTURE = REPO_ROOT / "data" / "tiles" / "synthetic-tree-sh"
#: The tint, in degree-1 coefficients of the capture's frame (x east, y north, z up), which
#: CesiumJS and 3DGS evaluate as `-C1 y sh[0] + C1 z sh[1] - C1 x sh[2]` of the direction from
#: the camera to the gaussian. Red -0.6 and blue +0.6 on sh[2] make every gaussian redder seen
#: looking east and bluer looking west; green -0.6 on sh[0] makes it greener looking north.
#: After quantisation each is C1 x 0.625 ~ 0.3 of full scale either way -- a colour change no
#: geometry of the tree could fake, which is what lets the e2e test see that CesiumJS
#: evaluates the bands, and in the capture's own frame (a y/z mix-up would lose the green).
TINT = {"f_rest_2": -0.6, "f_rest_32": 0.6, "f_rest_15": -0.6}


def tinted_tree(path: Path) -> Path:
    """The committed synthetic tree, plus SH degree 3: the tint, and zeros elsewhere."""
    columns = splat_tiles.read_ply(TREE_PLY)
    count = columns["x"].shape[0]
    columns.update({name: np.full(count, TINT.get(name, 0.0), np.float32) for name in REST})
    return write_sh_ply(path, columns)


def pack_sh_fixture(out: Path, work: Path) -> dict[str, float | int]:
    return splat_tiles.convert(
        tinted_tree(work / "tinted.ply"), out, 28.0389, -82.6966, 0.0, 0.02, 1500
    )


def test_committed_sh_fixture_matches_the_packer(tmp_path: Path) -> None:
    stats = pack_sh_fixture(tmp_path / "sh", tmp_path)
    assert stats["sh_degree"] == 3 and stats["tiles"] > 10
    fresh = {path.name: path.read_bytes() for path in (tmp_path / "sh").iterdir()}
    committed = {path.name: path.read_bytes() for path in SH_FIXTURE.iterdir()}
    assert sorted(fresh) == sorted(committed)
    assert fresh == committed
    # The same tree, the same tiles as the degree-0 LOD fixture: SH plays no part in the plan.
    lod = json.loads(
        (REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod" / "tileset.json").read_text()
    )
    assert json.loads(fresh["tileset.json"]) == lod


if __name__ == "__main__":
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        shutil.rmtree(SH_FIXTURE, ignore_errors=True)
        print(json.dumps(pack_sh_fixture(SH_FIXTURE, Path(scratch))))
