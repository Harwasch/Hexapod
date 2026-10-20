"""Read models of an epoch's confidence map (M4, ADR 0016).

What the API says about how well each part of an epoch was seen: the summary of the grid
its current revision shipped (`EpochConfidenceRead`, on every epoch), the weak cells as
re-scan zones (`WeakAreasRead`), and the cells a re-scan left weak (`FillTargetsRead`).
The grid itself is a `hexapod.cellgrid` file beside the tiles; these are what the API
computes from it.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import CamelModel

Finite = Annotated[float, Field(allow_inf_nan=False)]


class AlignmentCapRead(CamelModel):
    """How the epoch's placement capped its scores (ADR 0016, C4): a cell may only claim
    evidence when the epoch's uncertainty from the site frame is under half a cell.

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    #: The epoch's uncertainty relative to the site frame, metres (0 for the epoch that
    #: made the frame).
    uncertainty_m: Finite
    #: `min(1, cell / (2 * uncertaintyM))`: what every score was multiplied by.
    factor: Finite
    #: The site's published epochs before this one when it was measured.
    prior_epochs: int
    #: Which file the uncertainty was read from (`epoch_alignment.json`, `georef.json`, ...).
    source: str


class EpochConfidenceRead(CamelModel):
    """The confidence map an epoch's current revision shipped, summarised at its finest
    level. Null on an epoch means not measured: the run that made its current revision
    measured none (a splat upload carries no capture evidence).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    status: Literal["measured"]
    #: The revision (asset) the map was measured on.
    asset_id: uuid.UUID
    #: The `hexapod.cellgrid` file beside the tiles.
    url: str
    #: The finest level's cell, metres, and its lattice level (`0.25 m * 2**level`).
    cell_m: Finite
    level: int
    #: Which cells count as weak: `weak-v1` (score under 128, or unseen).
    policy: str
    #: Weak cells over the cells this epoch speaks for (scored, or expected and unseen).
    weak_share: Finite
    weak_cells: int
    scored_cells: int
    unseen_cells: int
    #: Per reason (`views`, `spread`, `gsd`, `accuracy`, `unseen`, `alignment`), how many
    #: weak cells carry it.
    reasons: dict[str, int]
    alignment_cap: AlignmentCapRead
