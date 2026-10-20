"""The site cell lattice and its container (`cell_grid.py`), against brute force."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import cell_grid
from cell_grid import CellGrid, CellGridError, CellPlane, Channel, Lattice


def _confidence_plane(seed: int, level: int = 2, i0: int = -7, j0: int = 5) -> CellPlane:
    rng = np.random.default_rng(seed)
    nj, ni = 9, 13
    data = np.zeros((nj, ni, 3), dtype=np.uint8)
    scored = rng.random((nj, ni)) < 0.7
    data[..., 0] = np.where(scored, rng.integers(1, 256, (nj, ni)), 0)
    data[..., 1] = rng.integers(0, 64, (nj, ni))
    data[..., 2] = rng.integers(0, 256, (nj, ni))
    return CellPlane(level, i0, j0, ni, nj, data)


def _grid(plane: CellPlane, levels: int = 8) -> CellGrid:
    return CellGrid(
        "confidence",
        cell_grid.CONFIDENCE_CHANNELS,
        tuple(cell_grid.min_pyramid(plane, levels)),
        {"policy": cell_grid.WEAK_POLICY, "weakBelow": cell_grid.WEAK_BELOW},
    )


def test_a_grid_round_trips_through_its_descriptor_and_bytes(tmp_path: Path) -> None:
    grid = _grid(_confidence_plane(1))

    descriptor = cell_grid.encode(grid, tmp_path)
    back = cell_grid.decode(descriptor, (tmp_path / "confidence.bin").read_bytes())

    assert descriptor["format"] == "hexapod.cellgrid" and descriptor["version"] == 1
    assert descriptor["uri"] == "confidence.bin" and descriptor["frame"] == "tileset"
    assert descriptor["lattice"] == {"baseM": 0.25, "level": 2} and descriptor["cellM"] == 1.0
    # The container's keys first, the kind's after them, the digest last.
    assert list(descriptor)[:9] == [
        "format", "version", "kind", "uri", "frame", "lattice", "cellM", "channels", "levels",
    ]  # fmt: skip
    assert list(descriptor)[-1] == "sha256"
    assert back.kind == "confidence" and back.channels == grid.channels
    assert back.meta == {"policy": "weak-v1", "weakBelow": 128}
    assert [p.level for p in back.planes] == [p.level for p in grid.planes]
    for got, want in zip(back.planes, grid.planes, strict=True):
        assert (got.i0, got.j0, got.ni, got.nj) == (want.i0, want.j0, want.ni, want.nj)
        np.testing.assert_array_equal(got.data, want.data)
    # The levels' offsets and sizes are into the uncompressed planes, which the digest is of.
    raw = gzip.decompress((tmp_path / "confidence.bin").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    ends = [entry["offset"] + entry["bytes"] for entry in descriptor["levels"]]
    assert ends[-1] == len(raw)
    assert [e["offset"] for e in descriptor["levels"]] == [0, *ends[:-1]]


def test_the_bytes_are_stable_and_carry_no_timestamp(tmp_path: Path) -> None:
    grid = _grid(_confidence_plane(2))
    first = cell_grid.encode(grid, tmp_path / "a")
    second = cell_grid.encode(grid, tmp_path / "b")

    blob = (tmp_path / "a" / "confidence.bin").read_bytes()
    assert blob == (tmp_path / "b" / "confidence.bin").read_bytes()
    assert first == second
    # gzip's MTIME field (bytes 4..7) is zero, so the same planes are the same file.
    assert blob[:2] == b"\x1f\x8b" and blob[4:8] == b"\x00\x00\x00\x00"


def test_a_tampered_or_foreign_blob_is_refused(tmp_path: Path) -> None:
    grid = _grid(_confidence_plane(3))
    descriptor = cell_grid.encode(grid, tmp_path)
    blob = (tmp_path / "confidence.bin").read_bytes()
    raw = bytearray(gzip.decompress(blob))
    raw[0] ^= 1
    with pytest.raises(CellGridError, match="sha256"):
        cell_grid.decode(descriptor, gzip.compress(bytes(raw), mtime=0))
    with pytest.raises(CellGridError, match="not a hexapod.cellgrid"):
        cell_grid.decode({**descriptor, "version": 2}, blob)
    with pytest.raises(CellGridError, match="not gzip"):
        cell_grid.decode(descriptor, b"plain")
    short = dict(descriptor)
    short["levels"] = [{**descriptor["levels"][0], "bytes": 3}]
    with pytest.raises(CellGridError, match="is not"):
        cell_grid.decode(short, blob)


def _brute_parent(plane: CellPlane) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Every parent cell, from every child, one at a time."""
    children: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for jj in range(plane.nj):
        for ii in range(plane.ni):
            i, j = plane.i0 + ii, plane.j0 + jj
            key = (i // 2, j // 2)
            score, reasons, relief = (int(v) for v in plane.data[jj, ii])
            children.setdefault(key, []).append((score, reasons, relief))
    out = {}
    for key, cells in children.items():
        scores = [s for s, _, _ in cells if s > 0]
        reasons = 0
        for _, r, _ in cells:
            reasons |= r
        out[key] = (min(scores) if scores else 0, reasons, max(r for _, _, r in cells))
    return out


@pytest.mark.parametrize("seed, i0, j0", [(4, -7, 5), (5, 0, 0), (6, 3, -12), (7, -1, -1)])
def test_the_pyramid_is_min_or_and_max_of_children(seed: int, i0: int, j0: int) -> None:
    pyramid = cell_grid.min_pyramid(_confidence_plane(seed, i0=i0, j0=j0), 8)

    for child, parent in itertools.pairwise(pyramid):
        assert parent.level == child.level + 1
        expected = _brute_parent(child)
        got = {
            (parent.i0 + ii, parent.j0 + jj): tuple(int(v) for v in parent.data[jj, ii])
            for jj in range(parent.nj)
            for ii in range(parent.ni)
        }
        assert got == expected
    # Eight levels at most, fewer once a plane is a single cell (nothing coarser to say).
    # A plane straddling the origin never gets there: cells -1 and 0 have different
    # parents at every level, which is the lattice being a lattice.
    assert len(pyramid) == 9 or (len(pyramid) < 9 and pyramid[-1].cells == 1)
    if i0 >= 0 and j0 >= 0:
        assert pyramid[-1].cells == 1 and len(pyramid) < 9


def test_a_weak_child_makes_its_parent_weak_and_unseen_carries_up() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)
    data[..., 0] = [[250, 240], [90, 0]]  # one weak child, one with no evidence
    plane = CellPlane(0, 0, 0, 2, 2, data)
    parent = cell_grid.min_pyramid(plane, 1)[1]
    assert int(parent.data[0, 0, 0]) == 90
    assert cell_grid.weak_mask(parent).tolist() == [[True]]

    data[..., 0] = [[250, 240], [200, 0]]
    data[1, 1, 1] = cell_grid.REASON_BITS["unseen"]
    parent = cell_grid.min_pyramid(CellPlane(0, 0, 0, 2, 2, data), 1)[1]
    assert int(parent.data[0, 0, 0]) == 200
    assert int(parent.data[0, 0, 1]) == cell_grid.REASON_BITS["unseen"]
    assert cell_grid.weak_mask(parent).tolist() == [[True]]


def test_cells_nest_across_levels() -> None:
    """Property: a point's cell one level up is its cell's parent, for points anywhere
    (negative, on cell edges, far out) and every level -- the reason two epochs of a site
    compare with no resampling."""
    rng = np.random.default_rng(11)
    points = np.concatenate(
        [
            rng.uniform(-5000, 5000, (2000, 2)),
            rng.integers(-400, 400, (500, 2)) * 0.25,  # exactly on base-cell edges
            rng.normal(0, 0.3, (500, 2)),
        ]
    )
    for level in range(14):
        i, j = Lattice(level).index(points)
        up_i, up_j = Lattice(level + 1).index(points)
        np.testing.assert_array_equal(up_i, i >> 1)
        np.testing.assert_array_equal(up_j, j >> 1)
        cell = Lattice(level).cell_m
        assert np.all(i * cell <= points[:, 0]) and np.all(points[:, 0] < (i + 1) * cell)
    # A point that is not finite is in no cell.
    i, _ = Lattice(0).index(np.array([[np.nan, 0.0], [np.inf, 1.0]]))
    assert (i == np.iinfo(np.int64).min).all()


def test_two_byte_channels_are_little_endian_and_signed(tmp_path: Path) -> None:
    channels = (Channel("dsm", "i2"), Channel("support", "u1"))
    data = np.zeros((1, 2, 2), dtype=np.int32)
    data[0, 0] = [-2, 7]
    data[0, 1] = [0x1234, 255]
    # A kind of no fixed channels: the container's own rules only (a `surface` grid must
    # have all four of its channels, `test_a_surface_grid_needs_its_four_channels`).
    grid = CellGrid("heights", channels, (CellPlane(3, 4, -1, 2, 1, data),))

    descriptor = cell_grid.encode(grid, tmp_path)
    raw = gzip.decompress((tmp_path / "heights.bin").read_bytes())

    # Interleaved per cell: dsm (2 bytes, little-endian), then support.
    assert raw == b"\xfe\xff\x07\x34\x12\xff"
    back = cell_grid.decode(descriptor, (tmp_path / "heights.bin").read_bytes())
    assert back.planes[0].data.dtype == np.int32
    np.testing.assert_array_equal(back.planes[0].data, data)
    with pytest.raises(CellGridError, match="outside i2"):
        data[0, 0, 0] = 40_000
        cell_grid.encode(
            CellGrid("heights", channels, (CellPlane(3, 4, -1, 2, 1, data),)), tmp_path
        )


def test_a_surface_grid_needs_its_four_channels(tmp_path: Path) -> None:
    """`surface` (ADR 0018): dsm and ground i2 centimetres, sigma and support u1, in that
    order; the no-height sentinel is i2's least value and survives the round trip."""
    assert [(c.name, c.type, c.unit_m) for c in cell_grid.SURFACE_CHANNELS] == [
        ("dsm", "i2", 0.01),
        ("ground", "i2", 0.01),
        ("sigma", "u1", 0.01),
        ("support", "u1", None),
    ]
    data = np.zeros((1, 2, 4), dtype=np.int16)
    data[0, 0] = [cell_grid.NO_HEIGHT, cell_grid.NO_HEIGHT, 0, 0x11]
    data[0, 1] = [32767, -32767, 255, 0xF3]
    grid = CellGrid("surface", cell_grid.SURFACE_CHANNELS, (CellPlane(2, -1, 0, 2, 1, data),))
    descriptor = cell_grid.encode(grid, tmp_path)
    assert descriptor["uri"] == "surface.bin"
    assert descriptor["channels"][0] == {"name": "dsm", "type": "i2", "unitM": 0.01}
    back = cell_grid.decode(descriptor, (tmp_path / "surface.bin").read_bytes())
    np.testing.assert_array_equal(back.planes[0].data, data)
    short = dict(descriptor, channels=descriptor["channels"][:2])
    with pytest.raises(CellGridError, match="needs channels"):
        cell_grid.decode(short, (tmp_path / "surface.bin").read_bytes())


def test_lookup_finds_each_points_cell_and_zero_outside(tmp_path: Path) -> None:
    plane = _confidence_plane(8, level=2, i0=-7, j0=5)
    grid = _grid(plane)
    cell = plane.cell_m
    inside = np.array([[(-7 + 3 + 0.5) * cell, (5 + 2 + 0.25) * cell, 9.0]])
    outside = np.array([[(-7 - 1 + 0.5) * cell, 5.5 * cell, 0.0], [np.nan, 0.0, 0.0]])

    values = cell_grid.lookup(grid, np.concatenate([inside, outside]))

    np.testing.assert_array_equal(values[0], plane.data[2, 3])
    assert not values[1:].any()
    # A coarser level is looked up in its own plane.
    coarse = cell_grid.lookup(grid, inside, level=3)
    parent = grid.plane(3)
    i, j = Lattice(3).index(inside)
    np.testing.assert_array_equal(
        coarse[0], parent.data[int(j[0]) - parent.j0, int(i[0]) - parent.i0]
    )


def test_a_tileset_names_its_grids_on_the_root(tmp_path: Path) -> None:
    grid = _grid(_confidence_plane(9))
    descriptor = cell_grid.encode(grid, tmp_path)
    tileset = {
        "root": {
            "extras": {"gaussians": 3, "collision": {"uri": "c.bin"}, "confidence": descriptor}
        }
    }
    (tmp_path / "tileset.json").write_text(json.dumps(tileset))

    grids = cell_grid.from_tileset(tmp_path / "tileset.json")

    assert list(grids) == ["confidence"]
    np.testing.assert_array_equal(grids["confidence"].planes[0].data, grid.planes[0].data)
    tileset["root"]["extras"]["confidence"] = {**descriptor, "uri": "../escape.bin"}
    (tmp_path / "tileset.json").write_text(json.dumps(tileset))
    with pytest.raises(CellGridError, match="beside the tileset"):
        cell_grid.from_tileset(tmp_path / "tileset.json")


def test_a_corrupt_stream_or_a_malformed_descriptor_is_a_cell_grid_error() -> None:
    """Whatever is wrong with a grid, a reader gets `CellGridError` and nothing else, so
    `package` can leave a bad grid out rather than fail the run."""
    folder = cell_grid_fixture()
    descriptor = json.loads((folder / "grids.json").read_text())["epochs"]["e2"]["descriptor"]
    blob = bytearray((folder / "e2" / "confidence.bin").read_bytes())
    blob[40] ^= 0xFF
    blob[41] ^= 0xFF  # inside the deflate stream: zlib's own error, or a wrong digest
    with pytest.raises(CellGridError):
        cell_grid.decode(descriptor, bytes(blob))
    good = (folder / "e2" / "confidence.bin").read_bytes()
    channels = descriptor["channels"]
    malformed = [
        {**descriptor, "channels": [channels[0], channels[0], channels[2]]},  # duplicate
        {**descriptor, "channels": ["score", "reasons", "relief"]},  # not objects
        {**descriptor, "channels": [{"type": "u1"}, *channels[1:]]},  # no name
        {**descriptor, "lattice": [0.25, 1]},  # not an object
        {**descriptor, "levels": "all of them"},
        {**descriptor, "levels": [["level", 1]]},
        {**descriptor, "channels": [channels[1], channels[2], {"name": "x", "type": "u1"}]},
    ]
    for bad in malformed:
        with pytest.raises(CellGridError):
            cell_grid.decode(bad, good)


def cell_grid_fixture() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard-epochs"


def test_plane_at_reads_a_level_the_pyramid_stopped_short_of() -> None:
    """`plane_at` is the stored plane where there is one, else the nearest finer one
    coarsened min-of-children up to the level -- what a longer pyramid would have held."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, (6, 9, 3)).astype(np.uint8)
    data[..., 1] &= 31
    plane = cell_grid.CellPlane(1, -5, 3, 9, 6, data)
    short = cell_grid.CellGrid(
        "confidence", cell_grid.CONFIDENCE_CHANNELS, tuple(cell_grid.min_pyramid(plane, 1))
    )
    full = cell_grid.min_pyramid(plane, 3)
    assert cell_grid.plane_at(short, 2) is short.planes[1]
    found = cell_grid.plane_at(short, 4)
    assert (found.level, found.i0, found.j0, found.ni, found.nj) == (
        full[3].level,
        full[3].i0,
        full[3].j0,
        full[3].ni,
        full[3].nj,
    )
    assert np.array_equal(found.data, full[3].data)
    with pytest.raises(cell_grid.CellGridError, match="finer"):
        cell_grid.plane_at(short, 0)
    covered = cell_grid.covered_mask(plane)
    assert np.array_equal(covered, (data[..., 0] > 0) | ((data[..., 1] & 16) > 0))
    assert not (cell_grid.weak_mask(plane) & ~covered).any()
