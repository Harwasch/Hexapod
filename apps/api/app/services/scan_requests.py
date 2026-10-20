"""Site scans requested from the landing page: stored as sent, read with the write token.

Open to anyone, so what it does with a request is bounded: an outline is checked for size
before PostGIS sees it (app/schemas/scan_request.py), dissolved under a statement timeout,
reduced to the precision it is stored and read at, and refused if nothing of it survives; the
day's requests are capped across every client; and a database refusal is answered without the
values that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import DataError, OperationalError
from sqlalchemy.orm import Session

from app.models.scan_request import ScanRequest
from app.schemas.geojson import Footprint
from app.schemas.scan_request import ScanRequestCreate, ScanRequestRead
from app.services.errors import InvalidInputError

FOOTPRINT: TypeAdapter[Footprint] = TypeAdapter(Footprint)

#: The largest area a request may outline (m2): 50 km2, about 12,000 acres, far beyond any one
#: site the fleet works and well short of a county.
MAX_AREA_M2 = 50_000_000
#: The smallest (m2): less is a sliver, not land.
MIN_AREA_M2 = 1.0
#: The grid an outline is stored on (degrees): ST_AsGeoJSON's nine decimals, so what is stored
#: is exactly what a read returns, and a sliver thinner than the grid is seen as no area now
#: rather than as an unreadable row later.
PRECISION_DEG = 1e-9

#: The visitor's outline as stored: made valid, neighbouring parcels (a MultiPolygon whose parts
#: share edges, which is not a valid MultiPolygon) dissolved into one outline, and put on the
#: stored grid.
CLEAN = text(
    "SELECT ST_AsEWKT(g), ST_Area(g::geography), ST_IsEmpty(g), ST_AsGeoJSON(g) FROM ("
    "SELECT ST_Multi(ST_CollectionExtract(ST_ReducePrecision(ST_UnaryUnion(ST_MakeValid("
    "ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(:boundary), 4326)))), :grid), 3)) AS g) AS t"
)

UNPROCESSABLE = (
    "The outline could not be processed. Choose fewer parcels, or remove the land and enter "
    "its acres."
)


class DailyCapReached(Exception):  # noqa: N818 - a condition, answered 503, not an error
    """The day's requests are all in: the form is told to try later."""


@dataclass(frozen=True)
class CleanOutline:
    ewkt: str
    area_m2: float


def clean_outline(db: Session, boundary: str, timeout_ms: int) -> CleanOutline:
    """The outline as it will be stored, or InvalidInputError. Bounded by `timeout_ms`: two
    crossing comb-shaped parcels of a few hundred corners took 16 s to dissolve unbounded."""
    db.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
    try:
        ewkt, area, empty, geojson = db.execute(
            CLEAN, {"boundary": boundary, "grid": PRECISION_DEG}
        ).one()
    except OperationalError as error:  # the timeout (QueryCanceled), or the geometry refused
        db.rollback()
        raise InvalidInputError(UNPROCESSABLE) from error
    if empty or not area or area < MIN_AREA_M2:
        raise InvalidInputError("The outline does not enclose an area.")
    try:
        FOOTPRINT.validate_json(geojson)
    except ValidationError as error:
        raise InvalidInputError(UNPROCESSABLE) from error
    if area > MAX_AREA_M2:
        raise InvalidInputError(
            f"The outline covers {area / 1e6:,.0f} km2; requests may cover at most "
            f"{MAX_AREA_M2 / 1e6:,.0f} km2. Choose fewer parcels, or describe the site in the "
            "notes."
        )
    return CleanOutline(ewkt=ewkt, area_m2=float(area))


def stored_today(db: Session) -> int:
    """Requests stored in the last 24 hours, by every client."""
    since = func.now() - text("interval '1 day'")
    return db.scalar(select(func.count()).where(ScanRequest.created_at > since)) or 0


def create_scan_request(
    db: Session, payload: ScanRequestCreate, *, daily_cap: int, clean_timeout_ms: int
) -> None:
    """Stores a request, or quietly drops one whose honeypot was filled in."""
    if payload.website.strip():
        return
    if stored_today(db) >= daily_cap:
        raise DailyCapReached
    boundary = None
    if payload.boundary is not None:
        boundary = clean_outline(db, payload.boundary.model_dump_json(), clean_timeout_ms).ewkt
    try:
        db.add(
            ScanRequest(
                name=payload.name,
                email=payload.email,
                notes=payload.notes.strip(),
                boundary=boundary,
                parcel_ids=payload.parcel_ids,
            )
        )
        db.commit()
    except DataError as error:
        # Raised from the values themselves; its message (parameters hidden by the engine
        # anyway) is not repeated to the visitor or the log.
        db.rollback()
        raise InvalidInputError(
            "Part of this request could not be stored. Check it and send it again."
        ) from error


def list_scan_requests(db: Session, limit: int) -> list[ScanRequestRead]:
    """The most recent requests first. A stored outline that cannot be read back is left out
    of its row (and flagged), never allowed to fail the whole list."""
    rows = db.execute(
        select(
            ScanRequest,
            func.ST_AsGeoJSON(ScanRequest.boundary),
            func.ST_Area(func.geography(ScanRequest.boundary)),
        )
        .order_by(ScanRequest.created_at.desc(), ScanRequest.id)
        .limit(limit)
    ).all()
    result = []
    for row, geojson, area in rows:
        boundary: Footprint | None = None
        unreadable = False
        if geojson:
            try:
                boundary = FOOTPRINT.validate_json(geojson)
            except ValidationError:
                unreadable = True
        result.append(
            ScanRequestRead(
                id=row.id,
                created_at=row.created_at,
                name=row.name,
                email=row.email,
                notes=row.notes,
                boundary=boundary,
                boundary_unreadable=unreadable,
                area_m2=float(area) if area is not None and boundary is not None else None,
                parcel_ids=list(row.parcel_ids),
            )
        )
    return result
