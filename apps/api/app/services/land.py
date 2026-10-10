from __future__ import annotations

import json
import uuid
from typing import Any

from geoalchemy2 import Geometry
from pydantic import TypeAdapter
from sqlalchemy import ColumnElement, cast, func, select, text
from sqlalchemy.orm import Session

from app.models.land import LandArea, LandBoundaryRevision
from app.schemas.geojson import Footprint
from app.schemas.land import (
    BoundaryOperation,
    BoundaryResult,
    BoundaryRevisionRead,
    BoundarySource,
    BoundarySplit,
    BoundarySplitResult,
    CorridorRequest,
    LandCreate,
    LandRead,
    LandRevise,
)
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.geometry import footprint_to_wkb

FOOTPRINT: TypeAdapter[Footprint] = TypeAdapter(Footprint)


def _snapshot(row: LandArea, payload: LandCreate, note: str) -> LandBoundaryRevision:
    return LandBoundaryRevision(
        land_id=row.id,
        revision=row.revision,
        boundary=payload.boundary.model_dump(mode="json"),
        source=payload.source.model_dump(mode="json", by_alias=True),
        note=note,
    )


def _read(row: LandArea, boundary: str, area: float, perimeter: float) -> LandRead:
    return LandRead(
        id=row.id,
        name=row.name,
        description=row.description,
        boundary=FOOTPRINT.validate_json(boundary),
        source=BoundarySource.model_validate(row.source),
        revision=row.revision,
        area_m2=area,
        perimeter_m=perimeter,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def list_land(db: Session, workspace_id: uuid.UUID, limit: int, offset: int) -> list[LandRead]:
    statement = (
        select(
            LandArea,
            func.ST_AsGeoJSON(LandArea.boundary),
            func.ST_Area(func.geography(LandArea.boundary)),
            func.ST_Perimeter(func.geography(LandArea.boundary)),
        )
        .where(LandArea.workspace_id == workspace_id)
        .order_by(LandArea.updated_at.desc(), LandArea.id)
        .limit(limit)
        .offset(offset)
    )
    return [
        _read(row, boundary, area, perimeter)
        for row, boundary, area, perimeter in db.execute(statement)
    ]


def get_land(db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID) -> LandRead:
    result = db.execute(
        select(
            LandArea,
            func.ST_AsGeoJSON(LandArea.boundary),
            func.ST_Area(func.geography(LandArea.boundary)),
            func.ST_Perimeter(func.geography(LandArea.boundary)),
        ).where(LandArea.id == land_id, LandArea.workspace_id == workspace_id)
    ).first()
    if result is None:
        raise NotFoundError("land area", land_id)
    return _read(*result)


def create_land(db: Session, workspace_id: uuid.UUID, payload: LandCreate) -> LandRead:
    row = LandArea(
        workspace_id=workspace_id,
        name=payload.name,
        description=payload.description,
        boundary=footprint_to_wkb(payload.boundary),
        source=payload.source.model_dump(mode="json", by_alias=True),
        revision=1,
    )
    db.add(row)
    db.flush()
    db.add(_snapshot(row, payload, "Area established"))
    db.commit()
    return get_land(db, workspace_id, row.id)


def revise_land(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, payload: LandRevise
) -> LandRead:
    # Serialize writers, then compare the revision: two tabs cannot overwrite each other.
    row = db.scalar(
        select(LandArea)
        .where(LandArea.id == land_id, LandArea.workspace_id == workspace_id)
        .with_for_update()
    )
    if row is None:
        raise NotFoundError("land area", land_id)
    if row.revision != payload.expected_revision:
        raise ConflictError("This land area changed. Reload it before saving your boundary.")
    row.name = payload.name
    row.description = payload.description
    row.boundary = footprint_to_wkb(payload.boundary)
    row.source = payload.source.model_dump(mode="json", by_alias=True)
    row.revision += 1
    db.add(_snapshot(row, payload, payload.note))
    db.commit()
    return get_land(db, workspace_id, land_id)


def revisions(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID
) -> list[BoundaryRevisionRead]:
    get_land(db, workspace_id, land_id)
    rows = db.scalars(
        select(LandBoundaryRevision)
        .where(LandBoundaryRevision.land_id == land_id)
        .order_by(LandBoundaryRevision.revision.desc())
    )
    return [BoundaryRevisionRead.model_validate(row) for row in rows]


def delete_land(db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID) -> None:
    row = db.scalar(
        select(LandArea)
        .where(LandArea.id == land_id, LandArea.workspace_id == workspace_id)
        .with_for_update()
    )
    if row is None:
        raise NotFoundError("land area", land_id)
    db.delete(row)
    db.commit()


def _result(db: Session, expression: ColumnElement[Any]) -> BoundaryResult:
    geometry = select(
        func.ST_Multi(func.ST_CollectionExtract(expression, 3)).label("geom")
    ).subquery()
    geom = geometry.c.geom
    result = db.execute(
        select(
            func.ST_AsGeoJSON(geom),
            func.ST_Area(func.geography(geom)),
            func.ST_Perimeter(func.geography(geom)),
            func.ST_IsEmpty(geom),
            func.ST_IsValid(geom),
        )
    ).one()
    boundary, area, perimeter, empty, valid = result
    if empty or not valid or area <= 0:
        raise InvalidInputError("The operation does not produce a valid land area.")
    return BoundaryResult(
        boundary=FOOTPRINT.validate_json(boundary), area_m2=area, perimeter_m=perimeter
    )


def corridor(db: Session, payload: CorridorRequest) -> BoundaryResult:
    # Geography buffering chooses a local metric projection. Bound its extent rather
    # than silently promise accurate buffering across continents or the dateline.
    longitudes = [point[0] for point in payload.coordinates]
    latitudes = [point[1] for point in payload.coordinates]
    if max(longitudes) - min(longitudes) > 5 or max(latitudes) - min(latitudes) > 5:
        raise InvalidInputError(
            "Split corridors spanning more than 5 degrees into shorter sections."
        )
    line = func.ST_SetSRID(
        func.ST_GeomFromGeoJSON(
            json.dumps(
                {
                    "type": "LineString",
                    "coordinates": payload.coordinates,
                }
            )
        ),
        4326,
    )
    return _result(
        db,
        cast(
            func.ST_Buffer(
                func.geography(line),
                payload.width_m / 2,
                f"endcap={payload.cap} join=round",
            ),
            Geometry(srid=4326),
        ),
    )


def operate(db: Session, payload: BoundaryOperation) -> BoundaryResult:
    operation = {
        "union": func.ST_Union,
        "difference": func.ST_Difference,
        "intersection": func.ST_Intersection,
    }[payload.operation]
    return _result(
        db,
        operation(
            func.ST_SetSRID(func.ST_GeomFromGeoJSON(payload.left.model_dump_json()), 4326),
            func.ST_SetSRID(func.ST_GeomFromGeoJSON(payload.right.model_dump_json()), 4326),
        ),
    )


def split_boundary(db: Session, payload: BoundarySplit) -> BoundarySplitResult:
    """Preview a planar GeoJSON cut; only an explicit land save changes a record."""
    rows = db.execute(
        text("""
        WITH cut AS (
            SELECT ST_Split(
                ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(:boundary), 4326)),
                ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(:line), 4326))
            ) AS geom
        ), pieces AS (
            SELECT (ST_Dump(ST_CollectionExtract(geom, 3))).geom AS geom FROM cut
        )
        SELECT ST_AsGeoJSON(geom, 15), ST_Area(geom::geography),
               ST_Perimeter(geom::geography), ST_NPoints(geom)
        FROM pieces ORDER BY ST_AsEWKB(ST_Normalize(geom)) LIMIT 101
        """),
        {
            "boundary": payload.boundary.model_dump_json(),
            "line": json.dumps({"type": "LineString", "coordinates": payload.coordinates}),
        },
    ).all()
    original_parts = 1 if payload.boundary.type == "Polygon" else len(payload.boundary.coordinates)
    if len(rows) <= original_parts:
        raise InvalidInputError(
            "The line did not divide the land. Draw from outside one edge to outside another."
        )
    if len(rows) > 100 or sum(row[3] for row in rows) > 20_000:
        raise InvalidInputError("This cut creates too many pieces or vertices. Use a simpler line.")
    parts = [
        BoundaryResult(
            boundary=FOOTPRINT.validate_json(geometry), area_m2=area, perimeter_m=perimeter
        )
        for geometry, area, perimeter, _ in rows
    ]
    selection = None
    if payload.keep_parts is not None:
        indices = sorted(set(payload.keep_parts))
        if indices[-1] >= len(parts):
            raise InvalidInputError("A selected part is not in this split preview.")
        geometries = [parts[index].boundary.model_dump(mode="json") for index in indices]
        # Merge adjacent selected pieces before validating the result. A MultiPolygon
        # whose members share a cut edge would otherwise be invalid.
        collection = func.ST_SetSRID(
            func.ST_GeomFromGeoJSON(
                json.dumps({"type": "GeometryCollection", "geometries": geometries})
            ),
            4326,
        )
        selection = _result(db, func.ST_UnaryUnion(collection))
    return BoundarySplitResult(parts=parts, selection=selection)
