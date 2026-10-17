"""A saved Land as mission ground.

Mission planning works on zones; a Land is a boundary plus an inventory of mapped features.
This turns one into the other, read-only: the boundary is a zone, every polygon feature is a
zone, and every line feature (a fence, a ditch, a track) is a corridor around the line, with
the line kept so the planner measures along it and the map draws it. Points cannot be worked
as ground and are only counted. Nothing is stored: the ground is always the Land as saved.
"""

from __future__ import annotations

import uuid
from typing import Any

from geoalchemy2 import Geometry
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import case, cast, func, select
from sqlalchemy.orm import Session

from app.models.land_feature import LandFeature
from app.schemas.agent import ZONE_NAME_MAX
from app.schemas.geojson import LineString
from app.schemas.land import BoundarySource
from app.schemas.land_ground import LandGround, LandGroundItem
from app.services.land import FOOTPRINT, get_land

#: Width of the work corridor assumed around a mapped line (5 m each side). Stated in every
#: line zone's source text, so it is never read as a surveyed width.
GROUND_LINE_WIDTH_M = 10.0
GROUND_FEATURE_LIMIT = 200
LINE: TypeAdapter[LineString] = TypeAdapter(LineString)

MEANING = {
    "study-area": "study area, not a verified property boundary",
    "recorded-parcel": "recorded parcel, not verified title",
    "physical-feature": "physical feature outline",
}


def zone_id(identifier: uuid.UUID) -> str:
    return f"L-{identifier.hex[:8]}"


def _clip(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _feature_source(content: dict[str, Any], revision: int) -> str:
    source = content.get("source")
    label = source.get("label") if isinstance(source, dict) else None
    status = content.get("status") or "candidate"
    return f"Inventory · {status} · revision {revision} · {label or 'unlabelled source'}"


def ground(db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID) -> LandGround:
    land = get_land(db, workspace_id, land_id)
    source = BoundarySource.model_validate(land.source)
    items = [
        LandGroundItem(
            zone_id=zone_id(land.id),
            kind="boundary",
            feature_id=None,
            revision=land.revision,
            # Land names run to 200 characters; a zone's name stops at 120 (ZoneName), as the
            # feature names below do.
            name=_clip(land.name, ZONE_NAME_MAX),
            category=None,
            status=None,
            footprint=land.boundary,
            line=None,
            area_m2=land.area_m2,
            length_m=None,
            buffer_m=None,
            source=_clip(
                f"Saved boundary, revision {land.revision} · {source.label} · "
                f"{MEANING.get(source.meaning, source.meaning)}"
            ),
        )
    ]
    kind = func.ST_GeometryType(LandFeature.geometry)
    footprint = case(
        (
            kind == "ST_LineString",
            cast(
                func.ST_Buffer(
                    func.geography(LandFeature.geometry),
                    GROUND_LINE_WIDTH_M / 2,
                    "endcap=flat join=round",
                ),
                Geometry(srid=4326),
            ),
        ),
        else_=LandFeature.geometry,
    )
    rows = db.execute(
        select(
            LandFeature.id,
            LandFeature.revision,
            LandFeature.content,
            kind,
            func.ST_AsGeoJSON(LandFeature.geometry),
            func.ST_AsGeoJSON(footprint),
            func.ST_Area(func.geography(footprint)),
            func.ST_Length(func.geography(LandFeature.geometry)),
        )
        .where(LandFeature.land_id == land_id)
        .order_by(LandFeature.updated_at.desc(), LandFeature.id)
        .limit(GROUND_FEATURE_LIMIT)
    ).all()
    skipped = 0
    for identifier, revision, content, geometry_type, raw, outline, area, length in rows:
        if content.get("status") == "retired":
            continue
        if geometry_type not in {"ST_LineString", "ST_Polygon", "ST_MultiPolygon"}:
            skipped += 1
            continue
        is_line = geometry_type == "ST_LineString"
        try:
            shaped = FOOTPRINT.validate_json(outline)
            line = LINE.validate_json(raw) if is_line else None
        except ValidationError:
            # A corridor PostGIS cannot make valid (a degenerate line) is not ground.
            skipped += 1
            continue
        detail = (
            f" · line planned as an assumed {GROUND_LINE_WIDTH_M:g} m work corridor"
            if is_line
            else ""
        )
        items.append(
            LandGroundItem(
                zone_id=zone_id(identifier),
                kind="feature",
                feature_id=identifier,
                revision=revision,
                name=str(content.get("name") or "Unnamed feature")[:120],
                category=content.get("category"),
                status=content.get("status"),
                footprint=shaped,
                line=line,
                area_m2=float(area or 0),
                length_m=float(length) if is_line else None,
                buffer_m=GROUND_LINE_WIDTH_M if is_line else None,
                source=_clip(_feature_source(content, revision) + detail),
            )
        )
    return LandGround(
        land_id=land.id,
        name=land.name,
        boundary_revision=land.revision,
        items=items,
        skipped=skipped,
    )
