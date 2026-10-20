"""The committed cross-language cell-grid fixture is what its generator writes, byte for byte."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import cell_grid
import synthetic_cell_grids

COMMITTED = synthetic_cell_grids.OUT


def test_the_committed_grids_are_regenerated_byte_for_byte(tmp_path: Path) -> None:
    synthetic_cell_grids.write(tmp_path)
    for name in ("e1/confidence.bin", "e2/confidence.bin", "grids.json"):
        assert (tmp_path / name).read_bytes() == (COMMITTED / name).read_bytes(), name


def test_every_level_decodes_to_the_digest_grids_json_states() -> None:
    document = json.loads((COMMITTED / "grids.json").read_text(encoding="utf-8"))
    assert sorted(document["epochs"]) == ["e1", "e2"]
    decoded: dict[str, cell_grid.CellGrid] = {}
    for name, epoch in document["epochs"].items():
        blob = (COMMITTED / epoch["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == epoch["fileSha256"]
        grid = cell_grid.decode(epoch["descriptor"], blob)
        record = np.dtype([(c.name, cell_grid.TYPES[c.type]) for c in grid.channels])
        for plane, expected in zip(grid.planes, epoch["planes"], strict=True):
            rows = np.zeros((plane.nj, plane.ni), dtype=record)
            for index, channel in enumerate(grid.channels):
                rows[channel.name] = plane.data[..., index]
            assert plane.level == expected["level"]
            assert hashlib.sha256(rows.tobytes()).hexdigest() == expected["sha256"]
        decoded[name] = grid
    e1, e2 = decoded["e1"], decoded["e2"]
    # Two levels that nest: e2 (0.5 m) coarsened is on e1's 1 m cells, same window.
    assert (e1.level, e2.level) == (2, 1)
    assert (e1.plane(2).i0, e1.plane(2).j0) == (e2.plane(2).i0, e2.plane(2).j0)
    # The story the fixture tells: e1 weak in the north, e2 weak in a north-east patch.
    weak1 = cell_grid.weak_mask(e1.planes[0])
    assert weak1[e1.planes[0].nj // 2 :].all() and not weak1[: e1.planes[0].nj // 2].any()
    assert 0 < e2.meta["summary"]["weakShare"] < 0.2
    assert e2.meta["alignmentCap"]["uncertaintyM"] == 0.062


def test_the_fixture_covers_the_yard_root_box() -> None:
    box = json.loads(synthetic_cell_grids.YARD.read_text())["root"]["boundingVolume"]["box"]
    i0, j0, ni, nj = synthetic_cell_grids.yard_window(2)
    assert i0 <= np.floor((box[0] - box[3]) / 1.0) and (i0 + ni) * 1.0 >= box[0] + box[3]
    assert j0 <= np.floor((box[1] - box[7]) / 1.0) and (j0 + nj) * 1.0 >= box[1] + box[7]
