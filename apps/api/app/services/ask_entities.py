"""Capture scoping and persistence, kept separate from network/tool execution."""

import hashlib
import json
import math
import uuid
from datetime import timedelta
from typing import Any

from geoalchemy2 import WKTElement
from pydantic import TypeAdapter
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import AskRun, Capture, Epoch, ScanEntity, Site
from app.models.base import utcnow
from app.schemas.ask import AskEvent, AskRequest, Entity
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.georef_entities import georeference_entity, resolve_alignment
from app.services.property_memory import MemoryQuery, Observation, PostgresMemory, dedupe_candidates

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
    row = ScanEntity(id=uuid.uuid4(), capture_id=capture.id, site_id=capture.site_id, ask_id=ask_id)
    prepared = prepare(capture, entity)
    assign(row, prepared)
    row.created_at = utcnow()
    link_identity(db, capture, row)
    db.add(row)
    return row


def prepare(capture: Capture, entity: Entity) -> Entity:
    # Stored per-capture by catalog registration, never inferred from a site's origin.
    placement = (capture.metadata_ or {}).get("entityPlacement", {})
    alignment = resolve_alignment(
        georef=placement.get("georef"),
        placement=placement.get("placement"),
        arkit_training_frame_verified=placement.get("arkitTrainingFrameVerified") is True,
        height_datum=placement.get("heightDatum", "msl-unverified"),
    )
    prepared = Entity.model_validate(
        georeference_entity(
            entity.model_dump(mode="json", by_alias=True, exclude_none=True), alignment
        )
    )
    prepared.attrs.pop("identity", None)
    return prepared


def link_identity(db: Session, capture: Capture, row: ScanEntity) -> None:
    geo = row.entity["anchor"].get("geo")
    uncertainty = row.entity["attrs"]["georeference"]["uncertaintyM"]
    label = row.entity["label"]
    if (
        capture.site_id is None
        or geo is None
        or uncertainty is None
        or uncertainty > 5
        or not label.strip()
        or len(label) > 200
    ):
        return
    # Create and update use the same property lock before discovering identity links.
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": capture.site_id.int % (2**63)})
    observed = (
        db.scalar(select(Epoch.observed_at).where(Epoch.capture_id == capture.id))
        or capture.captured_at
        or row.created_at
    )
    new = Observation(
        id=row.id,
        site_id=capture.site_id,
        scan_id=capture.id,
        label=row.entity["label"],
        lat=geo["lat"],
        lon=geo["lon"],
        uncertainty_m=uncertainty,
        observed_at=observed,
    )
    candidates = []
    offset = 0
    repository = PostgresMemory(db)
    while True:
        page = repository.entities(
            capture.site_id,
            MemoryQuery(
                lat=new.lat,
                lon=new.lon,
                radius_m=1 + math.hypot(uncertainty, 5),
                since=observed - timedelta(days=30),
                until=observed + timedelta(days=30),
                limit=500,
                offset=offset,
            ),
        )
        for item in page:
            anchor = item["entity"]["anchor"].get("geo")
            label = item["entity"]["label"]
            if (
                anchor is not None
                and item["uncertaintyM"] is not None
                and 0 <= item["uncertaintyM"] <= 5
                and label.strip()
                and len(label) <= 200
            ):
                candidates.append(
                    Observation(
                        id=item["id"],
                        site_id=capture.site_id,
                        scan_id=item["scanId"],
                        label=item["entity"]["label"],
                        lat=anchor["lat"],
                        lon=anchor["lon"],
                        uncertainty_m=item["uncertaintyM"],
                        observed_at=item["observedAt"],
                    )
                )
        if len(page) < 500:
            break
        offset += 500
        if offset > 100000:
            raise InvalidInputError("identity search exceeds property limit; no link saved")
    # Links preserve every scan's evidence and never average anchors or confidence.
    row.entity = {
        **row.entity,
        "attrs": {**row.entity["attrs"], "identity": dedupe_candidates(new, candidates)},
    }


def request_site(db: Session, request: AskRequest, capture: Capture | None) -> None:
    if request.site_id is not None:
        site_id = uuid.UUID(request.site_id)
        if db.get(Site, site_id) is None:
            raise NotFoundError("Site", site_id)
        if capture is not None:
            check_site(capture, site_id)


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
            site_id=uuid.UUID(request.site_id) if request.site_id else None,
            request=request.model_dump(mode="json", by_alias=True, exclude_none=True),
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
