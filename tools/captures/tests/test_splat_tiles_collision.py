"""The collision grid the packer writes beside the tiles (splat_tiles `COLLISION_FORMAT`).

The web clients load `collision.bin` instead of building an occupancy grid on the main
thread, and they read it by the format alone -- so these tests pin the format (the brick
records, the bit order, the extras), the rule that makes a cell solid (the runtime's own,
apps/web src/lib/occupancy.ts), the cell size, and that reruns write the same bytes.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_splat_tiles_lod import write_ply

import splat_tiles

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURES = Path(__file__).resolve().parents[1]
TREE_PLY = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "splat.ply"


def logit(alpha: float) -> float:
    return math.log(alpha / (1 - alpha))


def splats(xyz: np.ndarray, alpha: np.ndarray | float) -> dict[str, np.ndarray]:
    count = xyz.shape[0]
    opacity = np.broadcast_to(np.vectorize(logit)(np.asarray(alpha, np.float64)), (count,))
    columns = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "opacity": opacity}
    columns.update({f"f_dc_{i}": np.zeros(count) for i in range(3)})
    columns.update({f"scale_{i}": np.full(count, -5.0) for i in range(3)})
    columns.update({"rot_0": np.ones(count), **{f"rot_{i}": np.zeros(count) for i in (1, 2, 3)}})
    return {name: np.asarray(value, dtype=np.float32) for name, value in columns.items()}


def join(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def wall(x: float = 1.03, size: float = 2.0, spacing: float = 0.01) -> np.ndarray:
    """A vertical wall at `x`, splats every `spacing` over `size` x `size` (y, z)."""
    steps = np.arange(0.0, size, spacing) + spacing / 2
    y, z = np.meshgrid(steps, steps, indexing="ij")
    return np.c_[np.full(y.size, x), y.ravel(), z.ravel()]


def pack(tmp_path: Path, columns: dict[str, np.ndarray], name: str = "splat") -> Path:
    ply = write_ply(tmp_path / f"{name}.ply", columns)
    out = tmp_path / name
    splat_tiles.convert(ply, out, 46.84, -91.99, 0.0, tile_gaussians=4000)
    return out


# ---------------------------------------------------------------------------- geometry


def test_a_wall_of_splats_is_a_wall_of_solid_cells(tmp_path: Path) -> None:
    points = wall()
    grid = splat_tiles.read_collision(pack(tmp_path, splats(points, 0.9)) / "collision.bin")
    # 1 cm apart, a 2^-6 m cell averages 1.56^2 = 2.4 splats, a 2^-5 m one 3.1^2 = 9.8: the
    # smallest power of two holding COLLISION_SPLATS_PER_CELL (3) is 2^-5.
    assert grid.cell == 2**-5
    origin = np.asarray(grid.origin)
    assert np.all(origin <= points.min(axis=0))
    assert np.all(np.mod(origin, grid.cell * 8) == 0), "origin is on a brick boundary"
    x_index = math.floor((1.03 - origin[0]) / grid.cell)
    # Every cell the wall passes through, and nothing else: each holds >= 9 splats of 0.9.
    assert set(grid.cells[:, 0].tolist()) == {x_index}
    assert grid.solid == {tuple(cell) for cell in grid.cell_of(points).tolist()}
    across = math.ceil(2.0 / grid.cell)
    assert len(grid.solid) == across * across


def test_the_opacity_rule_is_the_runtimes(tmp_path: Path) -> None:
    """Solid at a summed 0.6 (occupancy.ts SOLID), counting splats of >= 0.2 (MIN_OPACITY)."""
    assert (splat_tiles.COLLISION_SOLID, splat_tiles.COLLISION_MIN_OPACITY) == (0.6, 0.2)
    cell = 2**-5  # the wall's (above); each cluster sits at the centre of its own cell

    def centre(i: int) -> np.ndarray:
        return np.array([[48.5 * cell, (8 * i + 0.5) * cell, 32.5 * cell]])

    cases = {
        "one opaque splat": ([0.7], True),
        "one faint splat": ([0.5], False),
        "three faint ones adding up": ([0.25, 0.25, 0.25], True),
        "haze that adds up but never counts": ([0.15] * 6, False),
        "just over": ([0.31, 0.31], True),
        "just under": ([0.29, 0.29], False),
    }
    parts = [splats(wall(), 0.9)]
    for index, (alphas, _) in enumerate(cases.values()):
        at = np.repeat(centre(index), len(alphas), axis=0)
        parts.append(splats(at, np.asarray(alphas)))
    grid = splat_tiles.read_collision(pack(tmp_path, join(*parts)) / "collision.bin")
    assert grid.cell == cell
    for index, (name, (_, solid)) in enumerate(cases.items()):
        cell_index = tuple(grid.cell_of(centre(index))[0].tolist())
        assert (cell_index in grid.solid) is solid, name


def test_every_solid_cell_is_the_brute_force_sum(tmp_path: Path) -> None:
    """The committed tree, against a dict over every counted splat, as occupancy.ts adds."""
    out = tmp_path / "tree"
    stats = splat_tiles.convert(TREE_PLY, out, 28.0389, -82.6966, 0.0, 0.02, 1500)
    grid = splat_tiles.read_collision(out)
    data = splat_tiles.read_ply(TREE_PLY)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    alpha = splat_tiles.sigmoid(data["opacity"].astype(np.float64))
    counted = alpha >= 0.2
    sums: dict[tuple[int, int, int], float] = {}
    for cell, value in zip(grid.cell_of(xyz[counted]).tolist(), alpha[counted], strict=True):
        sums[tuple(cell)] = sums.get(tuple(cell), 0.0) + float(value)
    assert grid.solid == {cell for cell, total in sums.items() if total >= 0.6}
    # The cell, by occupancy.ts `cellFor`'s rule counted over every splat, then floored.
    expected = 2.0**5
    for exponent in range(-8, 5):
        occupied = np.unique(np.floor(xyz[counted] / 2.0**exponent), axis=0).shape[0]
        if counted.sum() / occupied >= 3:
            expected = 2.0**exponent
            break
    assert grid.cell == expected == stats["collision_cell_m"]
    assert stats["collision_solid_cells"] == len(grid.solid) > 100


def test_a_big_site_gets_no_more_than_4096_cells_a_side(tmp_path: Path) -> None:
    """Two dense patches 200 m apart: dense enough for 2^-5 m, held to 200 / 4096 -> 2^-4."""
    a = wall(x=0.0, size=0.5)
    b = wall(x=200.0, size=0.5)
    grid = splat_tiles.read_collision(pack(tmp_path, splats(np.r_[a, b], 0.9)) / "collision.bin")
    assert grid.cell == 2**-4
    assert grid.solid == {tuple(cell) for cell in grid.cell_of(np.r_[a, b]).tolist()}


# ------------------------------------------------------------------------------ format


def test_one_cell_is_one_record_with_one_bit(tmp_path: Path) -> None:
    """Cell (9, 17, 250): brick (1, 2, 31), local (1, 1, 2), bit 1 + 8 + 128 = 137."""
    blob, bricks = splat_tiles.encode_collision(np.array([[9, 17, 250]]))
    raw = gzip.decompress(blob)
    assert bricks == 1 and len(raw) == 76
    assert struct.unpack_from("<3i", raw, 0) == (1, 2, 31)
    mask = raw[12:]
    assert mask[137 >> 3] == 1 << (137 & 7)
    assert sum(mask) == mask[137 >> 3]


def test_bricks_round_trip_in_z_y_x_order() -> None:
    rng = np.random.default_rng(5)
    cells = np.unique(rng.integers(0, 70, size=(5000, 3)), axis=0)
    blob, bricks = splat_tiles.encode_collision(cells[rng.permutation(cells.shape[0])])
    records = np.frombuffer(gzip.decompress(blob), dtype=splat_tiles.COLLISION_RECORD)
    assert records.shape[0] == bricks == np.unique(cells >> 3, axis=0).shape[0]
    order = [(int(b[2]), int(b[1]), int(b[0])) for b in records["brick"]]
    assert order == sorted(order) and len(set(order)) == len(order)
    assert (records["mask"] != 0).any(axis=1).all(), "only bricks with a solid cell"
    decoded = splat_tiles.decode_collision(blob)
    assert {tuple(c) for c in decoded.tolist()} == {tuple(c) for c in cells.tolist()}
    # Bit by bit, as a reader in another language does it.
    solid = {tuple(c) for c in cells.tolist()}
    for record in records[:20]:
        for n in range(512):
            local = (n & 7, (n >> 3) & 7, n >> 6)
            cell = tuple(int(record["brick"][k]) * 8 + local[k] for k in range(3))
            bit = (int(record["mask"][n >> 3]) >> (n & 7)) & 1
            assert bool(bit) == (cell in solid)


def test_nothing_solid_is_an_empty_file(tmp_path: Path) -> None:
    grid = splat_tiles.read_collision(pack(tmp_path, splats(wall(), 0.1)) / "collision.bin")
    assert grid.cells.shape == (0, 3)
    assert gzip.decompress((tmp_path / "splat" / "collision.bin").read_bytes()) == b""


def test_the_tileset_declares_it_on_the_root(tmp_path: Path) -> None:
    out = pack(tmp_path, splats(wall(), 0.9))
    tileset = json.loads((out / "tileset.json").read_text())
    extras = tileset["root"]["extras"]["collision"]
    assert set(extras) == {
        "format",
        "version",
        "uri",
        "cell",
        "origin",
        "brick",
        "bricks",
        "solidCells",
        "rule",
    }
    assert extras["format"] == "hexapod.collision" and extras["version"] == 1
    assert extras["uri"] == "collision.bin" and (out / extras["uri"]).is_file()
    assert extras["brick"] == 8 and isinstance(extras["cell"], float)
    assert math.log2(extras["cell"]).is_integer() and len(extras["origin"]) == 3
    raw = gzip.decompress((out / "collision.bin").read_bytes())
    assert extras["bricks"] == len(raw) // 76 and len(raw) % 76 == 0
    assert (
        extras["solidCells"]
        == splat_tiles.decode_collision((out / "collision.bin").read_bytes()).shape[0]
    )
    assert isinstance(extras["rule"], str) and "\n" not in extras["rule"]
    # Only the root: a viewer has it before any tile loads, and children say nothing of it.
    stack = list(tileset["root"].get("children", []))
    assert stack, "the wall is packed as a hierarchy"
    while stack:
        tile = stack.pop()
        assert "collision" not in tile.get("extras", {})
        stack.extend(tile.get("children", []))


# --------------------------------------------------------------------------- stability


def test_reruns_write_the_same_bytes_whatever_the_windows(tmp_path: Path, monkeypatch) -> None:
    ply = TREE_PLY
    splat_tiles.convert(ply, tmp_path / "a", 28.0389, -82.6966, 0.0, 0.02, 1500)
    splat_tiles.convert(ply, tmp_path / "b", 28.0389, -82.6966, 0.0, 0.02, 1500)
    monkeypatch.setattr(splat_tiles, "CHUNK_BYTES", 56 * 777)
    splat_tiles.convert(ply, tmp_path / "c", 28.0389, -82.6966, 0.0, 0.02, 1500)
    first = (tmp_path / "a" / "collision.bin").read_bytes()
    assert first == (tmp_path / "b" / "collision.bin").read_bytes()
    assert first == (tmp_path / "c" / "collision.bin").read_bytes()
    assert first[4:8] == b"\0\0\0\0", "gzip mtime pinned to 0"


# ----------------------------------------------------------------------------- backfill


def test_the_cli_backfills_a_published_tileset(tmp_path: Path) -> None:
    """`splat_tiles.py collision` on a tileset packed without the grid: the same two files
    `convert` writes today, byte for byte."""
    fresh = tmp_path / "fresh"
    splat_tiles.convert(TREE_PLY, fresh, 28.0389, -82.6966, 0.0, 0.02, 1500)
    old = tmp_path / "old"
    shutil.copytree(fresh, old)
    (old / "collision.bin").unlink()
    tileset = json.loads((old / "tileset.json").read_text())
    del tileset["root"]["extras"]["collision"]
    (old / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "splat_tiles.py", "collision", str(TREE_PLY), str(old)]
        + ["--tileset", str(old / "tileset.json")],
        cwd=CAPTURES,
        capture_output=True,
        text=True,
        check=True,
    )
    printed = json.loads(result.stdout)
    assert printed["bytes"] == (fresh / "collision.bin").stat().st_size
    for name in ("collision.bin", "tileset.json"):
        assert (old / name).read_bytes() == (fresh / name).read_bytes(), name


def test_the_backfill_refuses_a_tileset_from_another_ply(tmp_path: Path) -> None:
    out = pack(tmp_path, splats(wall(), 0.9))
    with pytest.raises(splat_tiles.SplatFormatError, match="not the PLY it was packed from"):
        splat_tiles.build_collision(TREE_PLY, tmp_path / "x", tileset=out / "tileset.json")
