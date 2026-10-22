"""The committed height surfaces (data/tiles/synthetic-yard-surfaces), as API tests use them.

Two epochs of the synthetic yard written by tools/pipeline `experiments/surface_fixture.py`
with P5's own encoder, each on the plane of the matching confidence grid of
`data/tiles/synthetic-yard-epochs`: e1 at level 2 (1 m cells), e2 at level 1 (0.5 m). e2
has a shed (raised) where both saw well and a stack (raised) where e1's confidence is weak;
e1 has a spoil heap e2 no longer has (lowered); a hedge stands in both. `surfaces.json`
also holds P5's demo reference change between them. Fixtures, not measurements.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

SURFACES = Path(__file__).resolve().parents[3] / "data" / "tiles" / "synthetic-yard-surfaces"
DOCUMENT: dict[str, Any] = json.loads((SURFACES / "surfaces.json").read_text(encoding="utf-8"))


def surface_descriptor(epoch: str, origin_height_m: float | None = None) -> dict[str, Any]:
    """The epoch's descriptor, as its tileset's `extras.surface` would hold it; with the
    frame origin's height a test registers it under, when given (the planes are heights
    above that origin, whatever it is: only the descriptor names it)."""
    found: dict[str, Any] = copy.deepcopy(DOCUMENT["epochs"][epoch]["descriptor"])
    if origin_height_m is not None:
        found["vertical"]["originHeightM"] = origin_height_m
    return found


def surface_blob(epoch: str) -> bytes:
    return (SURFACES / epoch / "surface.bin").read_bytes()
