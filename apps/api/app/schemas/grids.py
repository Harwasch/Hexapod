"""One wire format for a per-cell answer drawn over a Land: the lon/lat grid.

Cells are equal steps of longitude and latitude, row 0 is the northern row and column 0 the
western column, so the console can drape the whole grid as one textured rectangle whose
texels are exactly the cells (apps/web/src/cesium/GroundGridLayer.ts). `app.analysis.
lonlat_grid` builds and encodes them. The Fog of Knowing sends its density and best-record
grids this way; similarity and terrain grids (batch 3) reuse it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import CamelModel

Finite = Annotated[float, Field(allow_inf_nan=False)]


class LonLatGrid(CamelModel):
    """`value = raw * scale + offset`, except raw == `nodata`. `data` is base64 of the raw
    values, little-endian, row-major from the north-west cell."""

    west: Finite = Field(ge=-180, le=180)
    south: Finite = Field(ge=-90, le=90)
    east: Finite = Field(ge=-180, le=180)
    north: Finite = Field(ge=-90, le=90)
    columns: int = Field(ge=1, le=4096)
    rows: int = Field(ge=1, le=4096)
    #: Nominal cell size on the ground at the grid's centre, metres.
    cell_m: Finite = Field(gt=0)
    encoding: Literal["u8", "u16", "f32"]
    scale: Finite = 1.0
    offset: Finite = 0.0
    #: The raw value that means "no value here"; for f32 grids NaN is also no value.
    nodata: Finite | None = None
    data: str
