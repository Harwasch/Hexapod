from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Header, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import DbSession, SettingsDep
from app.api.workspace_deps import WorkspaceDep, workspace_access
from app.models.research import Evidence, Finding, ResearchEvent
from app.schemas.research import (
    EventRead,
    EvidenceRead,
    FindingDisposition,
    FindingRead,
    InvestigationCreate,
    InvestigationDetail,
    InvestigationRead,
    OverviewRead,
    OverviewRequest,
    ResearchStatus,
    RunCreate,
    RunRead,
    SourceRead,
)
from app.services import research
from app.services.errors import NotFoundError

router = APIRouter(tags=["land research"])


@router.get("/research/status", response_model=ResearchStatus)
def status(settings: SettingsDep, scope: WorkspaceDep) -> ResearchStatus:
    return ResearchStatus(
        model_configured=bool(settings.anthropic_api_key),
        model=settings.anthropic_model if settings.anthropic_api_key else None,
    )


@router.get("/research/sources", response_model=list[SourceRead])
def sources(scope: WorkspaceDep) -> list[SourceRead]:
    from dataclasses import asdict

    from app.research.providers.open_data import SOURCES

    return [
        SourceRead.model_validate(
            {key: value for key, value in asdict(source).items() if key != "endpoint"}
        )
        for source in SOURCES.values()
    ]


@router.post("/land/{land_id}/investigations", response_model=InvestigationRead, status_code=201)
def create_investigation(
    land_id: uuid.UUID, payload: InvestigationCreate, db: DbSession, scope: WorkspaceDep
) -> InvestigationRead:
    scope.require("owner", "editor")
    return research.create_investigation(db, scope.id, land_id, scope.principal.id, payload)


@router.get("/land/{land_id}/investigations", response_model=list[InvestigationRead])
def list_investigations(
    land_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[InvestigationRead]:
    return research.list_investigations(db, scope.id, land_id, limit, offset)


@router.get("/research/investigations/{investigation_id}", response_model=InvestigationDetail)
def detail(
    investigation_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> InvestigationDetail:
    return research.detail(db, scope.id, investigation_id, limit, offset)


@router.post(
    "/research/investigations/{investigation_id}/runs", response_model=RunRead, status_code=201
)
def start_run(
    investigation_id: uuid.UUID, payload: RunCreate, db: DbSession, scope: WorkspaceDep
) -> RunRead:
    scope.require("owner", "editor")
    return research.create_run(db, scope.id, investigation_id, payload)


@router.get("/research/runs/{run_id}", response_model=RunRead)
def get_run(run_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> RunRead:
    return RunRead.model_validate(research.get_run(db, scope.id, run_id))


@router.post("/research/runs/{run_id}/cancel", response_model=RunRead)
def cancel_run(run_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> RunRead:
    scope.require("owner", "editor")
    return research.cancel_run(db, scope.id, run_id)


@router.get("/research/runs/{run_id}/events", response_model=list[EventRead])
def events(
    run_id: uuid.UUID, db: DbSession, scope: WorkspaceDep, after: int = Query(0, ge=0)
) -> list[EventRead]:
    research.get_run(db, scope.id, run_id)
    rows = db.scalars(
        select(ResearchEvent)
        .where(ResearchEvent.run_id == run_id, ResearchEvent.sequence > after)
        .order_by(ResearchEvent.sequence)
        .limit(200)
    )
    return [EventRead.model_validate(row) for row in rows]


@router.get("/research/runs/{run_id}/stream", response_class=StreamingResponse)
def stream(
    run_id: uuid.UUID,
    db: DbSession,
    scope: WorkspaceDep,
    after: int = Query(0, ge=0),
    last_event_id: Annotated[int | None, Header(ge=0)] = None,
) -> StreamingResponse:
    research.get_run(db, scope.id, run_id)
    db.rollback()

    def generate() -> Iterator[str]:
        cursor = max(after, last_event_id or 0)
        deadline = time.monotonic() + 25
        try:
            while time.monotonic() < deadline:
                if (
                    scope.principal.expires_at is not None
                    and scope.principal.expires_at <= time.time()
                ):
                    yield "event: access-expired\ndata: {}\n\n"
                    return
                try:
                    workspace_access(scope.principal, db, scope.id)
                    run = research.get_run(db, scope.id, run_id)
                except NotFoundError:
                    yield "event: access-revoked\ndata: {}\n\n"
                    return
                batch = list(
                    db.scalars(
                        select(ResearchEvent)
                        .where(ResearchEvent.run_id == run_id, ResearchEvent.sequence > cursor)
                        .order_by(ResearchEvent.sequence)
                        .limit(200)
                    )
                )
                terminal = run.status in research.TERMINAL
                encoded = [
                    (
                        item.sequence,
                        EventRead.model_validate(item).model_dump(mode="json", by_alias=True),
                    )
                    for item in batch
                ]
                db.rollback()  # Never hold a database transaction across client/network waits.
                for sequence, payload in encoded:
                    cursor = sequence
                    yield f"id: {sequence}\nevent: research\ndata: {json.dumps(payload)}\n\n"
                if terminal and len(batch) < 200:
                    return
                if not batch:
                    yield ": keepalive\n\n"
                    time.sleep(0.5)
        finally:
            db.rollback()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/research/evidence/{evidence_id}", response_model=EvidenceRead)
def get_evidence(evidence_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> EvidenceRead:
    row = db.get(Evidence, evidence_id)
    if row is None:
        raise NotFoundError("evidence", evidence_id)
    research.get_run(db, scope.id, row.run_id)
    return EvidenceRead(id=row.id, run_id=row.run_id, **row.content)


@router.put("/research/findings/{finding_id}", response_model=FindingRead)
def disposition(
    finding_id: uuid.UUID, payload: FindingDisposition, db: DbSession, scope: WorkspaceDep
) -> FindingRead:
    scope.require("owner", "editor")
    row = db.get(Finding, finding_id)
    if row is None:
        raise NotFoundError("finding", finding_id)
    research.get_run(db, scope.id, row.run_id)
    row.disposition = payload.disposition
    db.commit()
    return FindingRead(id=row.id, run_id=row.run_id, disposition=row.disposition, **row.content)


@router.post("/land/{land_id}/overview", response_model=OverviewRead)
def overview(
    land_id: uuid.UUID, payload: OverviewRequest, db: DbSession, scope: WorkspaceDep
) -> OverviewRead:
    scope.require("owner", "editor")
    investigation, run = research.ensure_overview(
        db, scope.id, land_id, scope.principal.id, payload.boundary_revision
    )
    return OverviewRead(investigation=investigation, run=run)
