"""Read models of an epoch's height surface and of the change between two dates (M5, ADR 0018).

What the API says about the heights an epoch's current revision measured
(`EpochSurfaceRead`, on every epoch). The surface itself is a `hexapod.cellgrid` file
beside the tiles; this is its summary.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import CamelModel

Finite = Annotated[float, Field(allow_inf_nan=False)]


class EpochSurfaceRead(CamelModel):
    """The height surface an epoch's current revision shipped: per cell of its confidence
    grid's finest plane, its top surface (`dsm`), bare ground and spread, in centimetres
    above the site frame's origin. Null on an epoch means not measured: the run that made
    its current revision measured none (a splat upload, or a run before the surface stage).

    Read model: every field is explicit (no defaults) so the OpenAPI contract marks it required.
    """

    status: Literal["measured"]
    #: The revision (asset) the surface was measured on.
    asset_id: uuid.UUID
    #: The `hexapod.cellgrid` file beside the tiles.
    url: str
    #: Its cell, metres, and lattice level (`0.25 m * 2**level`): its confidence grid's own.
    cell_m: Finite
    level: int
    #: How it was estimated (`surface-v1`), and whether that was ever calibrated on a site
    #: scanned twice: `uncalibrated` until one is.
    policy: str
    calibration: str
    #: Cells of its plane, and those with a surface (enough samples to estimate one).
    cells: int
    surfaced_cells: int
    #: The median spread of its surfaced cells, metres (null when none is surfaced).
    median_sigma_m: Finite | None
    #: The ellipsoidal height of the frame origin its heights are above, metres.
    origin_height_m: Finite
