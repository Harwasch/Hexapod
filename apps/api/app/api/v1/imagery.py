"""Dated public imagery for an area, and tiles of USGS historical maps: the Time Lantern's data.

Both are public reads, like /layers: the imagery is public data, and the Descent and the
lantern work before anyone signs in. Both are rate-limited (app/services/ratelimit.py), since
each asks outside services on the caller's behalf. Nothing is stored.
"""

from __future__ import annotations

import logging
import threading

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response
from rasterio.errors import RasterioError

from app.api.deps import LimitTopoTile, SettingsDep, limit_imagery_catalog
from app.research.providers import dated_imagery
from app.schemas.dated_imagery import DatedImageryCatalog
from app.services import usgs_topo_tiles

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dated imagery"])

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    """One client for every catalog: connections to the same few hosts are reused."""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                headers={"User-Agent": "LivingWorld-DatedImagery/1.0"}, timeout=25
            )
        return _client


@router.get(
    "/imagery/dated",
    response_model=DatedImageryCatalog,
    summary="Dated public imagery for an area",
)
def dated(
    request: Request,
    response: Response,
    settings: SettingsDep,
    west: float = Query(ge=-180, le=180),
    south: float = Query(ge=-90, le=90),
    east: float = Query(ge=-180, le=180),
    north: float = Query(ge=-90, le=90),
) -> DatedImageryCatalog:
    """NAIP years, dry-season Sentinel-2 scenes and USGS historical map editions over an area
    (each side at most 0.25°), newest first, each with its date's meaning, its licence and its
    stated positional accuracy. A provider that fails is listed as unavailable; the others
    still answer. Cached on this server for 6 h per area rounded to 0.01°; a catalog missing
    something that may answer if asked again is cached for seconds, and not by browsers."""
    bbox = dated_imagery.check_bbox((west, south, east, north))
    catalog = dated_imagery.cached(bbox)
    if catalog is None:
        limit_imagery_catalog(request, settings)
        catalog = dated_imagery.discover(bbox, _http())
    response.headers["Cache-Control"] = (
        "no-store" if dated_imagery.incomplete(catalog) else "public, max-age=600"
    )
    return catalog


@router.get(
    "/imagery/usgs-topo/tiles/{z}/{x}/{y}.png",
    responses={
        200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
        204: {"description": "The client left before the tile was drawn; nothing was read."},
        503: {"description": "Every sheet reader is busy: ask again after Retry-After seconds."},
    },
    dependencies=[LimitTopoTile],
    summary="A tile of a USGS historical topographic map",
)
async def usgs_topo_tile(
    request: Request,
    z: int,
    x: int,
    y: int,
    sheet: str = Query(max_length=120, description="STATE/STATE_name_scanid_year_scale_geo"),
    w: float = Query(ge=-180, le=180),
    s: float = Query(ge=-90, le=90),
    e: float = Query(ge=-180, le=180),
    n: float = Query(ge=-90, le=90),
) -> Response:
    """A 256 px PNG on the WGS1984Quad grid (the grid of the land raster tiles), cut from the
    sheet's USGS GeoTIFF and transparent outside its neatline (`w`, `s`, `e`, `n`, NAD27).

    An `async def`, so a tile waiting for one of the tiler's few sheet readers waits on the event
    loop and holds none of the threads every other route runs on; the read itself runs on the
    tiler's own threads (usgs_topo_tiles.render_tile)."""
    tile = usgs_topo_tiles.plan(sheet, (w, s, e, n), z, x, y)
    try:
        png = await usgs_topo_tiles.render_tile(tile, gone=request.is_disconnected)
    except usgs_topo_tiles.RenderBusy as error:
        raise HTTPException(
            503,
            "Historical map tiles are busy. Try again in a few seconds.",
            headers={
                "Retry-After": str(usgs_topo_tiles.RETRY_AFTER_S),
                "Cache-Control": "no-store",
            },
        ) from error
    except (OSError, RasterioError, ValueError) as error:
        logger.warning("usgs topo tile failed: %s %s/%s/%s: %s", sheet, z, x, y, error)
        raise HTTPException(
            502, "The USGS map sheet could not be read. Try again shortly."
        ) from error
    if png is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(
        png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=604800", "X-Content-Type-Options": "nosniff"},
    )
