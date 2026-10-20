"""Site scans requested from the landing page: stored as sent, read with the write token."""

from __future__ import annotations

from pydantic import TypeAdapter
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.scan_request import ScanRequest
from app.schemas.geojson import Footprint
from app.schemas.scan_request import ScanRequestCreate, ScanRequestRead
from app.services.errors import InvalidInputError

FOOTPRINT: TypeAdapter[Footprint] = TypeAdapter(Footprint)

#: The largest area a request may outline (m2): 50 km2, about 12,000 acres, far beyond any one
#: site the fleet works and well short of a county.
MAX_AREA_M2 = 50_000_000

#: The visitor's outline as stored: made valid, and adjacent parcels (a MultiPolygon whose parts
#: share edges, which is not a valid MultiPolygon) dissolved into one outline.
CLEAN = text(
    "SELECT ST_AsEWKT(g), ST_Area(g::geography), ST_IsEmpty(g) FROM ("
    "SELECT ST_Multi(ST_CollectionExtract(ST_UnaryUnion(ST_MakeValid("
    "ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(:boundary), 4326)))), 3)) AS g) AS t"
)


def create_scan_request(db: Session, payload: ScanRequestCreate) -> None:
    """Stores a request, or quietly drops one whose honeypot was filled in."""
    if payload.website.strip():
        return
    boundary = None
    if payload.boundary is not None:
        ewkt, area, empty = db.execute(
            CLEAN, {"boundary": payload.boundary.model_dump_json()}
        ).one()
        if empty or not area:
            raise InvalidInputError("The outline does not enclose an area.")
        if area > MAX_AREA_M2:
            raise InvalidInputError(
                f"The outline covers {area / 1e6:,.0f} km2; requests may cover at most "
                f"{MAX_AREA_M2 / 1e6:,.0f} km2. Choose fewer parcels, or describe the site in the "
                "notes."
            )
        boundary = ewkt
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


def list_scan_requests(db: Session, limit: int) -> list[ScanRequestRead]:
    """The most recent requests first."""
    rows = db.execute(
        select(
            ScanRequest,
            func.ST_AsGeoJSON(ScanRequest.boundary),
            func.ST_Area(func.geography(ScanRequest.boundary)),
        )
        .order_by(ScanRequest.created_at.desc(), ScanRequest.id)
        .limit(limit)
    ).all()
    return [
        ScanRequestRead(
            id=row.id,
            created_at=row.created_at,
            name=row.name,
            email=row.email,
            notes=row.notes,
            boundary=FOOTPRINT.validate_json(geojson) if geojson else None,
            area_m2=float(area) if area is not None else None,
            parcel_ids=list(row.parcel_ids),
        )
        for row, geojson, area in rows
    ]
