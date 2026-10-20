"""The API's `hexapod.cellgrid` reader against P5's surface bytes (cross-language).

`data/tiles/synthetic-yard-surfaces` was written by tools/pipeline with P5's own encoder
(`cell_grid.pack` of kind `surface`); `surfaces.json` holds each plane's sha256 as the
pipeline lays it out. The API decodes the same files with its own reader and must land on
the same planes.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from app.analysis import cell_grid
from tests.surface_fixtures import DOCUMENT, surface_blob, surface_descriptor


def plane_sha256(plane: cell_grid.Plane) -> str:
    record = np.dtype(
        [
            (name, cell_grid.TYPES[kind])
            for name, kind in cell_grid.REQUIRED_CHANNELS["surface"].items()
        ]
    )
    rows = np.zeros((plane.nj, plane.ni), dtype=record)
    for name in cell_grid.REQUIRED_CHANNELS["surface"]:
        rows[name] = plane.channels[name]
    return hashlib.sha256(rows.tobytes()).hexdigest()


@pytest.mark.parametrize("epoch", ["e1", "e2"])
def test_the_api_reads_p5_s_surface_planes(epoch: str) -> None:
    data = surface_blob(epoch)
    expected = DOCUMENT["epochs"][epoch]
    assert hashlib.sha256(data).hexdigest() == expected["fileSha256"]
    grid = cell_grid.decode(surface_descriptor(epoch), data)
    assert grid.kind == "surface"
    [plane] = grid.planes
    assert plane.level == expected["plane"]["level"]
    assert plane_sha256(plane) == expected["plane"]["sha256"]
    # i2 heights come back signed, the no-surface sentinel included, never wrapped.
    dsm = plane.channels["dsm"]
    assert dsm.dtype == np.dtype("<i2")
    holes = dsm == cell_grid.NO_HEIGHT
    assert int(holes.sum()) == expected["descriptor"]["summary"]["thin"]
    assert (plane.channels["sigma"][holes] == 0).all()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["channels"].pop(1), "'ground'"),
        (lambda d: d["channels"][0].update(type="u2"), "'dsm' of type i2"),
    ],
)
def test_a_surface_grid_without_its_channels_is_refused(
    change: Callable[[dict[str, Any]], object], message: str
) -> None:
    descriptor = copy.deepcopy(surface_descriptor("e1"))
    change(descriptor)
    with pytest.raises(cell_grid.CellGridError, match=message):
        cell_grid.decode(descriptor, surface_blob("e1"))
