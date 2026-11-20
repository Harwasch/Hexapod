"""Capture scoping and persistence, kept separate from network/tool execution."""

import hashlib
import json
import uuid
from typing import Any

from geoalchemy2 import WKTElement
from pydantic import TypeAdapter
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import AskRun, Capture, ScanEntity
from app.schemas.ask import AskEvent, AskRequest, Entity
from app.services.errors import ConflictError, InvalidInputError, NotFoundError

EVENT: TypeAdapter[AskEvent] = TypeAdapter(AskEvent)


def capture_scope(db: Session, scan_id: str | None) -> Capture | None:
    if scan_id is None:
        return None
    try:
        capture_id = uuid.UUID(scan_id)
    except ValueError as exc:
        raise InvalidInputError("scanId must be a capture UUID") from exc
    capture = db.get(Capture, capture_id)
    if capture is None:
        raise NotFoundError("Capture", scan_id)
    return capture


def check_site(capture: Capture, site_id: uuid.UUID | None) -> None:
    if site_id is not None and capture.site_id != site_id:
        raise NotFoundError("Capture in site", capture.id)


def rows(
    db: Session, capture: Capture, limit: int = 100, after: uuid.UUID | None = None
) -> list[ScanEntity]:
    query = select(ScanEntity).where(ScanEntity.capture_id == capture.id)
    if after is not None:
        query = query.where(ScanEntity.id > after)
    return list(db.scalars(query.order_by(ScanEntity.id).limit(limit)))


def get_row(db: Session, capture: Capture, entity_id: uuid.UUID) -> ScanEntity:
    row = db.get(ScanEntity, entity_id)
    if row is None or row.capture_id != capture.id:
        raise NotFoundError("Entity in capture", entity_id)
    return row


def assign(row: ScanEntity, entity: Entity) -> None:
    world = entity.anchor.world
    row.world_anchor = WKTElement(f"POINT Z ({world.x} {world.y} {world.z})", srid=0)
    geo = entity.anchor.geo
    row.geo_anchor = (
        None if geo is None else WKTElement(f"POINT Z ({geo.lon} {geo.lat} {geo.alt})", srid=4326)
    )
    row.entity = entity.model_dump(mode="json", by_alias=True, exclude_none=True)


def add(db: Session, capture: Capture, entity: Entity, ask_id: str | None = None) -> ScanEntity:
    row = ScanEntity(capture_id=capture.id, site_id=capture.site_id, ask_id=ask_id)
    assign(row, entity)
    db.add(row)
    return row


def claim(db: Session, request: AskRequest, capture: Capture | None) -> tuple[AskRun, bool]:
    serialized = json.dumps(
        request.model_dump(mode="json", by_alias=True, exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    key = int.from_bytes(hashlib.sha256(request.id.encode()).digest()[:8], "big", signed=True)
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
    run = db.get(AskRun, request.id)
    if run is not None:
        if run.request_hash != digest:
            raise ConflictError("ask id already belongs to a different request")
        if run.state == "running":
            raise ConflictError("ask is already running", code="ask_running")
        if run.state == "complete":
            db.commit()
            return run, True
        run.state = "running"
        run.events = []
    else:
        run = AskRun(
            id=request.id,
            request_hash=digest,
            capture_id=capture.id if capture else None,
            state="running",
            events=[],
        )
        db.add(run)
    db.commit()
    return run, False


def finish(db: Session, run: AskRun, events: list[dict[str, Any]], capture: Capture | None) -> None:
    for event in events:
        parsed = EVENT.validate_python(event)
        if parsed.type == "entity":
            if capture is None:
                raise InvalidInputError("entities require a capture coordinate frame")
            add(db, capture, parsed.entity, run.id)
    run.events = events
    run.state = "complete"
    db.commit()


def fail(db: Session, run: AskRun) -> None:
    db.rollback()
    run.state = "failed"
    db.add(run)
    db.commit()
