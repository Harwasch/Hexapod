from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.land import LandArea, LandBoundaryRevision
from app.models.research import (
    Evidence,
    Finding,
    Investigation,
    ResearchArtifact,
    ResearchEvent,
    ResearchMessage,
    ResearchRun,
)
from app.schemas.research import (
    EvidenceRead,
    FindingRead,
    InvestigationCreate,
    InvestigationDetail,
    InvestigationRead,
    MessageRead,
    ResearchArtifactRead,
    ResearchBudget,
    ResearchPage,
    RunCreate,
    RunRead,
)
from app.services.errors import ConflictError, NotFoundError
from app.services.land import get_land

TERMINAL = {"succeeded", "partial", "failed", "cancelled"}


def now(db: Session) -> datetime:
    return db.execute(select(func.clock_timestamp())).scalar_one()  # type: ignore[no-any-return]


def event(db: Session, run: ResearchRun, kind: str, payload: dict[str, Any]) -> None:
    """Caller holds the run lock (or has just created the run)."""
    db.add(
        ResearchEvent(
            run_id=run.id,
            sequence=run.next_sequence,
            kind=kind,
            payload=payload,
            created_at=now(db),
        )
    )
    run.next_sequence += 1


def investigation(
    db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID, *, lock: bool = False
) -> Investigation:
    statement = (
        select(Investigation)
        .join(LandArea, LandArea.id == Investigation.land_id)
        .where(Investigation.id == identifier, LandArea.workspace_id == workspace_id)
    )
    if lock:
        statement = statement.with_for_update(of=Investigation)
    row = db.scalar(statement)
    if row is None:
        raise NotFoundError("investigation", identifier)
    return row


def read_investigation(db: Session, row: Investigation) -> InvestigationRead:
    revision = db.scalar(select(LandArea.revision).where(LandArea.id == row.land_id))
    return InvestigationRead(
        id=row.id,
        land_id=row.land_id,
        title=row.title,
        question=row.question,
        boundary_revision=row.boundary_revision,
        created_at=row.created_at,
        stale=row.boundary_revision != revision,
    )


def create_investigation(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    principal_id: str,
    payload: InvestigationCreate,
) -> InvestigationRead:
    get_land(db, workspace_id, land_id)
    revision = db.scalar(
        select(LandBoundaryRevision).where(
            LandBoundaryRevision.land_id == land_id,
            LandBoundaryRevision.revision == payload.boundary_revision,
        )
    )
    if revision is None:
        raise NotFoundError("boundary revision", payload.boundary_revision)
    row = Investigation(land_id=land_id, created_by=principal_id, **payload.model_dump())
    db.add(row)
    db.commit()
    return read_investigation(db, row)


def list_investigations(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, limit: int, offset: int
) -> list[InvestigationRead]:
    get_land(db, workspace_id, land_id)
    rows = db.scalars(
        select(Investigation)
        .where(Investigation.land_id == land_id)
        .order_by(Investigation.created_at.desc(), Investigation.id)
        .limit(limit)
        .offset(offset)
    )
    return [read_investigation(db, row) for row in rows]


def create_run(
    db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID, payload: RunCreate
) -> RunRead:
    investigation(db, workspace_id, identifier, lock=True)
    existing = db.scalar(
        select(ResearchRun).where(
            ResearchRun.investigation_id == identifier,
            ResearchRun.request_key == payload.request_key,
        )
    )
    if existing:
        if (
            existing.question != payload.question
            or existing.kind != payload.kind
            or ResearchBudget.model_validate(existing.budget) != payload.budget
            or existing.analysis != (payload.analysis.model_dump() if payload.analysis else None)
        ):
            raise ConflictError(
                "This request key was already used for a different research request."
            )
        return RunRead.model_validate(existing)
    run = ResearchRun(investigation_id=identifier, **payload.model_dump())
    db.add(run)
    db.flush()
    db.add(ResearchMessage(investigation_id=identifier, role="user", content=payload.question))
    event(db, run, "queued", {"kind": payload.kind})
    db.commit()
    return RunRead.model_validate(run)


def get_run(
    db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID, *, lock: bool = False
) -> ResearchRun:
    statement = (
        select(ResearchRun)
        .join(Investigation, Investigation.id == ResearchRun.investigation_id)
        .join(LandArea, LandArea.id == Investigation.land_id)
        .where(ResearchRun.id == identifier, LandArea.workspace_id == workspace_id)
    )
    if lock:
        statement = statement.with_for_update(of=ResearchRun)
    row = db.scalar(statement)
    if row is None:
        raise NotFoundError("research run", identifier)
    return row


def cancel_run(db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID) -> RunRead:
    run = get_run(db, workspace_id, identifier, lock=True)
    if run.status not in TERMINAL:
        run.status = "cancelled"
        run.finished_at = now(db)
        run.lease_token = None
        run.lease_expires_at = None
        event(db, run, "cancelled", {})
        db.commit()
    return RunRead.model_validate(run)


def detail(
    db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID, limit: int = 100, offset: int = 0
) -> InvestigationDetail:
    row = investigation(db, workspace_id, identifier)
    run_ids = select(ResearchRun.id).where(ResearchRun.investigation_id == identifier)
    statements: dict[str, Select[Any]] = {
        "runs": select(ResearchRun)
        .where(ResearchRun.investigation_id == identifier)
        .order_by(ResearchRun.created_at, ResearchRun.id),
        "messages": select(ResearchMessage)
        .where(ResearchMessage.investigation_id == identifier)
        .order_by(ResearchMessage.created_at, ResearchMessage.id),
        "findings": select(Finding)
        .where(Finding.run_id.in_(run_ids))
        .order_by(Finding.created_at, Finding.id),
        "evidence": select(Evidence)
        .where(Evidence.run_id.in_(run_ids))
        .order_by(Evidence.created_at, Evidence.id),
        "artifacts": select(ResearchArtifact)
        .where(ResearchArtifact.run_id.in_(run_ids))
        .order_by(ResearchArtifact.created_at, ResearchArtifact.id),
    }
    totals = {
        name: db.scalar(select(func.count()).select_from(statement.order_by(None).subquery())) or 0
        for name, statement in statements.items()
    }
    rows = {
        name: db.scalars(statement.limit(limit).offset(offset)).all()
        for name, statement in statements.items()
    }
    return InvestigationDetail(
        page=ResearchPage(offset=offset, limit=limit, totals=totals),
        investigation=read_investigation(db, row),
        messages=[MessageRead.model_validate(message) for message in rows["messages"]],
        runs=[RunRead.model_validate(run) for run in rows["runs"]],
        findings=[
            FindingRead(id=f.id, run_id=f.run_id, disposition=f.disposition, **f.content)
            for f in rows["findings"]
        ],
        evidence=[EvidenceRead(id=e.id, run_id=e.run_id, **e.content) for e in rows["evidence"]],
        artifacts=[
            ResearchArtifactRead(id=a.id, run_id=a.run_id, **a.content) for a in rows["artifacts"]
        ],
    )


def ensure_overview(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, principal_id: str, revision: int
) -> tuple[InvestigationRead, RunRead]:
    area = db.scalar(
        select(LandArea)
        .where(LandArea.id == land_id, LandArea.workspace_id == workspace_id)
        .with_for_update()
    )
    if area is None:
        raise NotFoundError("land area", land_id)
    if area.revision != revision:
        raise ConflictError("The boundary changed. Reload before starting its overview.")
    existing = db.execute(
        select(Investigation, ResearchRun)
        .join(ResearchRun, ResearchRun.investigation_id == Investigation.id)
        .where(
            Investigation.land_id == land_id,
            Investigation.boundary_revision == revision,
            Investigation.title == "Land overview",
            ResearchRun.kind == "overview",
        )
        .order_by(ResearchRun.created_at.desc())
        .limit(1)
    ).first()
    if existing:
        return read_investigation(db, existing[0]), RunRead.model_validate(existing[1])
    question = "What is useful and interesting about this land?"
    row = Investigation(
        land_id=land_id,
        boundary_revision=revision,
        created_by=principal_id,
        title="Land overview",
        question=question,
    )
    db.add(row)
    db.flush()
    run = create_run(
        db, workspace_id, row.id, RunCreate(request_key=land_id, kind="overview", question=question)
    )
    return read_investigation(db, row), run
