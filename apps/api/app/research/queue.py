from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import timedelta
from typing import Any, Literal

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.land_raster import LandRaster
from app.models.land_solar import LandSolar
from app.models.research import (
    Evidence,
    Finding,
    ResearchArtifact,
    ResearchMessage,
    ResearchRun,
    ResearchWorkerPresence,
)
from app.research.calculations import METHOD, calculate
from app.research.locations import geometry_location, media_location, union_location
from app.schemas.calculations import CalculationOutput
from app.schemas.research import (
    ArtifactContent,
    EventLocation,
    EventLocationRef,
    EvidenceContent,
    FindingContent,
    GalleryOutput,
    MapOutput,
    RasterOutput,
    SolarOutput,
    TimelineOutput,
)
from app.services.errors import InvalidInputError
from app.services.research import event, now

#: How often an active worker records presence (ResearchWorker._check_in).
CHECK_IN_EVERY_S = 15
#: A worker seen within this long, and not stopped, is "online" in /research/status: three
#: missed check-ins, so one slow database round trip never reads as an outage.
ONLINE_WITHIN_S = 45


class LeaseLostError(Exception):
    """The result belongs to a cancelled or superseded attempt and cannot be published."""


def check_in(
    db: Session,
    worker_id: uuid.UUID,
    *,
    hostname: str,
    pid: int,
    current_run_id: uuid.UUID | None,
    stopping: bool = False,
) -> None:
    """Record that this worker is alive (or, with `stopping`, that it has stopped).

    Times come from the database clock, as every research age does (services.research.now),
    so the API and the worker never disagree about how long ago "now" was."""
    clock = func.clock_timestamp()
    statement = insert(ResearchWorkerPresence).values(
        id=worker_id,
        hostname=hostname[:255],
        pid=pid,
        started_at=clock,
        last_seen_at=clock,
        current_run_id=current_run_id,
        stopped_at=clock if stopping else None,
    )
    db.execute(
        statement.on_conflict_do_update(
            index_elements=[ResearchWorkerPresence.id],
            set_={
                "last_seen_at": statement.excluded.last_seen_at,
                "current_run_id": statement.excluded.current_run_id,
                "stopped_at": statement.excluded.stopped_at,
            },
        )
    )
    db.commit()


def prune_presence(db: Session, older_than: timedelta = timedelta(days=7)) -> None:
    """Forget workers not seen for a while, so restarts do not pile up rows. A worker calls
    this once at start-up, just before its own first check-in."""
    db.execute(
        delete(ResearchWorkerPresence).where(
            ResearchWorkerPresence.last_seen_at < now(db) - older_than
        )
    )
    db.commit()


def anything_claimable(db: Session) -> bool:
    """Final idle-exit check, including runs whose recovery lease has expired."""
    return bool(
        db.scalar(
            select(
                select(ResearchRun.id)
                .where(
                    or_(
                        ResearchRun.status == "queued",
                        and_(
                            ResearchRun.status == "running",
                            ResearchRun.lease_expires_at <= func.clock_timestamp(),
                        ),
                    )
                )
                .exists()
            )
        )
    )


def claim(
    db: Session, lease_seconds: int = 45, max_attempts: int = 3
) -> tuple[uuid.UUID, uuid.UUID] | None:
    timestamp = now(db)
    run = db.scalar(
        select(ResearchRun)
        .where(
            or_(
                ResearchRun.status == "queued",
                and_(ResearchRun.status == "running", ResearchRun.lease_expires_at <= timestamp),
            )
        )
        .order_by(ResearchRun.created_at, ResearchRun.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if run is None:
        db.rollback()
        return None
    if run.attempt >= max_attempts:
        run.status = "failed"
        run.error = "The research worker exhausted its recovery attempts. Start a new run to retry."
        run.finished_at = timestamp
        run.lease_token = None
        run.lease_expires_at = None
        event(db, run, "failed", {"message": run.error})
        db.commit()
        return None
    run.attempt += 1
    run.status = "running"
    run.lease_token = uuid.uuid4()
    run.lease_expires_at = timestamp + timedelta(seconds=lease_seconds)
    run.started_at = run.started_at or timestamp
    event(db, run, "started", {"attempt": run.attempt, "resumed": bool(run.checkpoint)})
    result = (run.id, run.lease_token)
    db.commit()
    return result


def locked(db: Session, run_id: uuid.UUID, token: uuid.UUID) -> ResearchRun:
    run = db.scalar(
        select(ResearchRun)
        .where(ResearchRun.id == run_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if (
        run is None
        or run.status != "running"
        or run.lease_token != token
        or run.lease_expires_at is None
        or run.lease_expires_at <= now(db)
    ):
        db.rollback()
        raise LeaseLostError()
    return run


def heartbeat(db: Session, run_id: uuid.UUID, token: uuid.UUID, lease_seconds: int = 45) -> None:
    run = locked(db, run_id, token)
    run.lease_expires_at = now(db) + timedelta(seconds=lease_seconds)
    db.commit()


def checkpoint(
    db: Session,
    run_id: uuid.UUID,
    token: uuid.UUID,
    state: dict[str, Any],
    *,
    kind: str = "progress",
    payload: dict[str, Any] | None = None,
    location: EventLocation | None = None,
) -> None:
    run = locked(db, run_id, token)
    run.checkpoint = deepcopy(state)
    event(db, run, kind, payload or {}, location)
    db.commit()


def finish(
    db: Session,
    run_id: uuid.UUID,
    token: uuid.UUID,
    status: Literal["succeeded", "partial", "failed"],
    message: str,
) -> None:
    run = locked(db, run_id, token)
    run.status = status
    run.finished_at = now(db)
    run.error = message if status == "failed" else None
    run.lease_token = None
    run.lease_expires_at = None
    db.add(
        ResearchMessage(investigation_id=run.investigation_id, role="assistant", content=message)
    )
    event(db, run, status, {"message": message})
    db.commit()


def save_evidence(
    db: Session, run_id: uuid.UUID, token: uuid.UUID, key: str, content: EvidenceContent
) -> uuid.UUID:
    locked(db, run_id, token)
    row = db.scalar(select(Evidence).where(Evidence.run_id == run_id, Evidence.source_key == key))
    if row is None:
        row = Evidence(run_id=run_id, source_key=key, content=content.model_dump(mode="json"))
        db.add(row)
        db.flush()
    identifier = row.id
    db.commit()
    return identifier


def validate_citations(db: Session, run: ResearchRun, ids: list[uuid.UUID]) -> None:
    found = set(
        db.scalars(
            select(Evidence.id)
            .join(ResearchRun, ResearchRun.id == Evidence.run_id)
            .where(ResearchRun.investigation_id == run.investigation_id, Evidence.id.in_(ids))
        )
    )
    if found != set(ids):
        raise InvalidInputError(
            "Every citation must reference retrieved evidence in this investigation."
        )


def save_finding(
    db: Session,
    run_id: uuid.UUID,
    token: uuid.UUID,
    key: str,
    content: FindingContent,
    *,
    location: EventLocation | None = None,
) -> uuid.UUID:
    """`location` is where the finding came from when it has no boundary of its own (the
    source's sample point or query box), so the card can still pin it on the map."""
    run = locked(db, run_id, token)
    validate_citations(db, run, content.evidence_ids)
    row = db.scalar(select(Finding).where(Finding.run_id == run_id, Finding.output_key == key))
    if row is None:
        row = Finding(run_id=run_id, output_key=key, content=content.model_dump(mode="json"))
        db.add(row)
        db.flush()
        event(
            db,
            run,
            "finding",
            {"id": str(row.id), "title": content.title},
            geometry_location(content.boundary, "result") or location,
        )
    identifier = row.id
    db.commit()
    return identifier


def save_artifact(
    db: Session,
    run_id: uuid.UUID,
    token: uuid.UUID,
    key: str,
    content: ArtifactContent,
    *,
    location: EventLocation | None = None,
) -> uuid.UUID:
    """Artifacts with a place of their own (a map, a raster, an array zone, a gallery of
    located sources) are located by it; `location` covers the rest (a chart from one point)."""
    run = locked(db, run_id, token)
    derived: EventLocation | None = None
    if isinstance(content.output, CalculationOutput):
        expected = calculate(content.output.request)
        if content.output != expected:
            raise InvalidInputError("Calculation results must match the saved recipe exactly.")
        content = content.model_copy(update={"method": METHOD})
    ids = content.evidence_ids[:]
    if isinstance(content.output, TimelineOutput):
        ids.extend(
            identifier for entry in content.output.entries for identifier in entry.evidence_ids
        )
    if isinstance(content.output, GalleryOutput):
        ids.extend(content.output.evidence_ids)
    validate_citations(db, run, ids)
    if isinstance(content.output, GalleryOutput):
        located: list[EventLocation] = []
        for identifier in content.output.evidence_ids:
            item = db.get(Evidence, identifier)
            if item is None or not item.content.get("media"):
                raise InvalidInputError(
                    "Gallery entries need licensed archive media from this investigation."
                )
            place = media_location(item.content["media"])
            if place is not None:
                located.append(place)
        footprints = any(place.meaning == "catalog-footprint" for place in located)
        derived = union_location(
            (_as_geometry(place) for place in located),
            "catalog-footprint" if footprints else "catalog-coordinate",
        )
    if isinstance(content.output, RasterOutput):
        raster = db.scalar(
            select(LandRaster)
            .join(ResearchRun, ResearchRun.id == LandRaster.run_id)
            .where(
                LandRaster.id == content.output.raster_id,
                ResearchRun.investigation_id == run.investigation_id,
            )
        )
        if raster is None:
            raise InvalidInputError("Raster outputs must belong to this investigation.")
        bounds = raster.metadata_json.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            derived = _bounds_location(
                bounds,
                EventLocationRef(raster_id=content.output.raster_id),
            )
    if isinstance(content.output, SolarOutput):
        assessment = db.scalar(
            select(LandSolar)
            .join(ResearchRun, ResearchRun.id == LandSolar.run_id)
            .where(
                LandSolar.id == content.output.assessment_id,
                ResearchRun.investigation_id == run.investigation_id,
            )
        )
        if assessment is None:
            raise InvalidInputError("Solar outputs must belong to this investigation.")
        derived = geometry_location(
            assessment.request.get("array_zone"),
            "array-zone",
            ref=EventLocationRef(assessment_id=content.output.assessment_id),
        )
    row = db.scalar(
        select(ResearchArtifact).where(
            ResearchArtifact.run_id == run_id, ResearchArtifact.output_key == key
        )
    )
    if row is None:
        row = ResearchArtifact(
            run_id=run_id, output_key=key, content=content.model_dump(mode="json")
        )
        db.add(row)
        db.flush()
        if isinstance(content.output, MapOutput):
            derived = union_location(
                (feature.geometry for feature in content.output.features),
                "result",
                ref=EventLocationRef(artifact_id=row.id),
            )
        event(
            db,
            run,
            "artifact",
            {"id": str(row.id), "title": content.title, "kind": content.output.kind},
            derived or location,
        )
    identifier = row.id
    db.commit()
    return identifier


def _as_geometry(place: EventLocation) -> dict[str, Any]:
    if place.point is not None:
        return {"type": "Point", "coordinates": list(place.point)}
    west, south, east, north = place.bbox or (0.0, 0.0, 0.0, 0.0)
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }


def _bounds_location(bounds: list[Any], ref: EventLocationRef) -> EventLocation | None:
    try:
        west, south, east, north = (float(value) for value in bounds)
        return EventLocation(
            kind="bbox", bbox=(west, south, east, north), ref=ref, meaning="analysis-extent"
        )
    except (TypeError, ValueError):
        return None
