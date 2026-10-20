"""The API's `hexapod.cellgrid` reader against the pipeline's bytes (cross-language).

`data/tiles/synthetic-yard-epochs` was written by tools/captures (`synthetic_cell_grids.py`,
through `cell_grid.pack`); `grids.json` holds each level's sha256 of its planes as the
pipeline lays them out. The API decodes the same files with its own reader and must land on
the same planes, level by level.
"""

from __future__ import annotations

import copy
import gzip
import hashlib

import numpy as np
import pytest

from app.analysis import cell_grid
from tests.confidence_fixtures import GRIDS, blob, descriptor


def _plane_sha256(plane: cell_grid.Plane, channels: list[dict[str, str]]) -> str:
    record = np.dtype([(c["name"], cell_grid.TYPES[c["type"]]) for c in channels])
    rows = np.zeros((plane.nj, plane.ni), dtype=record)
    for channel in channels:
        rows[channel["name"]] = plane.channels[channel["name"]]
    return hashlib.sha256(rows.tobytes()).hexdigest()


@pytest.mark.parametrize("epoch", ["e1", "e2"])
def test_the_api_reads_the_pipeline_s_planes(epoch: str) -> None:
    data = blob(epoch)
    assert hashlib.sha256(data).hexdigest() == GRIDS["epochs"][epoch]["fileSha256"]
    grid = cell_grid.decode(descriptor(epoch), data)
    expected = {entry["level"]: entry["sha256"] for entry in GRIDS["epochs"][epoch]["planes"]}
    assert [plane.level for plane in grid.planes] == sorted(expected)
    for plane in grid.planes:
        assert _plane_sha256(plane, descriptor(epoch)["channels"]) == expected[plane.level]


@pytest.mark.parametrize("epoch", ["e1", "e2"])
def test_coarsening_is_the_pipeline_s_pyramid(epoch: str) -> None:
    """`coarsen` (what `Grid.at` falls back on) gives the very planes the pipeline's
    min-of-children pyramid wrote."""
    grid = cell_grid.decode(descriptor(epoch), blob(epoch))
    for finer, coarser in zip(grid.planes, grid.planes[1:], strict=False):
        made = cell_grid.coarsen(finer)
        assert (made.level, made.i0, made.j0, made.ni, made.nj) == (
            coarser.level,
            coarser.i0,
            coarser.j0,
            coarser.ni,
            coarser.nj,
        )
        for name in ("score", "reasons", "relief"):
            assert np.array_equal(made.channel(name), coarser.channel(name)), (epoch, name)


def test_the_weak_count_is_the_summary_s() -> None:
    for epoch in ("e1", "e2"):
        grid = cell_grid.decode(descriptor(epoch), blob(epoch))
        summary = descriptor(epoch)["summary"]
        assert int(cell_grid.weak_mask(grid.finest).sum()) == summary["weak"]
        covered = int(cell_grid.covered_mask(grid.finest).sum())
        assert covered == summary["scored"] + summary["unseen"]


def test_a_level_past_the_pyramid_is_coarsened_from_the_finest() -> None:
    grid = cell_grid.decode(descriptor("e1"), blob("e1"))
    top = grid.planes[-1].level
    beyond = grid.at(top + 1)
    assert beyond.level == top + 1
    assert grid.at(grid.finest.level) is grid.finest
    with pytest.raises(cell_grid.CellGridError):
        grid.at(grid.finest.level - 1)


def test_a_damaged_or_inflated_file_is_refused() -> None:
    good = descriptor("e1")
    data = blob("e1")
    raw = gzip.decompress(data)
    with pytest.raises(cell_grid.CellGridError, match="sha256"):
        tampered = bytearray(raw)
        tampered[0] ^= 1
        cell_grid.decode(good, gzip.compress(bytes(tampered)))
    # More bytes than every level declares: refused before they are all inflated.
    with pytest.raises(cell_grid.CellGridError, match="larger than"):
        cell_grid.decode(good, gzip.compress(raw + b"\0" * 1_000_000))
    with pytest.raises(cell_grid.CellGridError, match="truncated"):
        cell_grid.decode(good, data[: len(data) // 2])
    with pytest.raises(cell_grid.CellGridError, match="not gzip"):
        cell_grid.decode(good, b"not a gzip stream")
    changes: list[dict[str, object]] = [
        {"format": "x"},
        {"channels": [{"name": "score", "type": "f8"}]},
        {"levels": []},
    ]
    for change in changes:
        broken = copy.deepcopy(good)
        broken.update(change)
        with pytest.raises(cell_grid.CellGridError):
            cell_grid.decode(broken, data)
    huge = copy.deepcopy(good)
    huge["levels"][0].update({"ni": 1 << 13, "nj": 1 << 13, "bytes": 3 << 26})
    with pytest.raises(cell_grid.CellGridError, match="over"):
        cell_grid.decode(huge, data)


def _decoded_bytes(grid: cell_grid.Grid) -> int:
    return sum(int(a.nbytes) for plane in grid.planes for a in plane.channels.values())


def test_levels_that_share_bytes_are_refused() -> None:
    """Every level copies its own slice, so slices that overlap would decode the same bytes
    again per level: thirty levels over one 48 MiB slice would be ~1.4 GiB from a file under
    the 64 MiB cap. A level's slice must start where the one before it ended, or later."""
    from tests.confidence_fixtures import encode_grid

    data, good = encode_grid(1, 0, 0, np.full((4, 4), 200), np.zeros((4, 4)))
    entry = good["levels"][0]
    stacked = copy.deepcopy(good)
    stacked["levels"] = [{**entry, "level": level} for level in range(1, 31)]
    with pytest.raises(cell_grid.CellGridError, match="overlap"):
        cell_grid.decode(stacked, data)
    half = copy.deepcopy(good)
    half["levels"] = [
        {**entry, "ni": 2, "nj": 2, "bytes": 12, "offset": 0},
        {**entry, "level": 2, "ni": 2, "nj": 2, "bytes": 12, "offset": 6},
    ]
    with pytest.raises(cell_grid.CellGridError, match="overlap"):
        cell_grid.decode(half, data)


@pytest.mark.parametrize("epoch", ["e1", "e2"])
def test_a_decoded_grid_holds_no_more_than_its_planes(epoch: str) -> None:
    """The bound the cap rests on: what decode keeps is at most the inflated planes, which
    are at most `MAX_RAW_BYTES`."""
    grid = cell_grid.decode(descriptor(epoch), blob(epoch))
    raw = gzip.decompress(blob(epoch))
    assert _decoded_bytes(grid) <= len(raw) <= cell_grid.MAX_RAW_BYTES


def test_a_confidence_grid_must_carry_its_three_channels() -> None:
    for index, name in enumerate(("score", "reasons", "relief")):
        renamed = descriptor("e1")
        renamed["channels"][index]["name"] = "other"
        with pytest.raises(cell_grid.CellGridError, match=name):
            cell_grid.decode(renamed, blob("e1"))
