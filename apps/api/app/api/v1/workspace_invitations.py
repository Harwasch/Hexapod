from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.api.workspace_deps import Identity, WorkspaceDep
from app.models.base import utcnow
from app.models.workspace import Membership, Workspace, WorkspaceInvitation, WorkspaceProfile
from app.schemas.workspace import (
    InvitationCreate,
    InvitationCreated,
    InvitationPreview,
    InvitationRead,
    InvitationToken,
    WorkspaceRead,
)
from app.services.errors import ConflictError, NotFoundError

router = APIRouter(prefix="/invitations", tags=["workspaces"])


def _lock_owner(db: Session, workspace_id: uuid.UUID, principal_id: str) -> None:
    db.scalar(select(Workspace).where(Workspace.id == workspace_id).with_for_update())
    actor = db.get(Membership, (workspace_id, principal_id), populate_existing=True)
    if actor is None or actor.role != "owner":
        raise HTTPException(status_code=403, detail="Only an owner can manage invitations.")


@router.get("", response_model=list[InvitationRead])
def invitations(
    db: DbSession,
    scope: WorkspaceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[InvitationRead]:
    scope.require("owner")
    rows = db.scalars(
        select(WorkspaceInvitation)
        .where(WorkspaceInvitation.workspace_id == scope.id)
        .order_by(WorkspaceInvitation.created_at.desc(), WorkspaceInvitation.id)
        .offset(offset)
        .limit(limit)
    )
    return [InvitationRead.model_validate(row) for row in rows]


@router.post("", response_model=InvitationCreated, status_code=201)
def create_invitation(
    payload: InvitationCreate, db: DbSession, scope: WorkspaceDep, response: Response
) -> InvitationCreated:
    scope.require("owner")
    if scope.principal.pilot:
        raise ConflictError("Invitations require OIDC identity.")
    _lock_owner(db, scope.id, scope.principal.id)
    token = secrets.token_urlsafe(32)
    row = WorkspaceInvitation(
        workspace_id=scope.id,
        token_hash=hashlib.sha256(token.encode()).hexdigest(),
        label=payload.label,
        role=payload.role,
        created_by=scope.principal.id,
        expires_at=utcnow() + timedelta(days=payload.expires_in_days),
    )
    db.add(row)
    db.commit()
    response.headers["Cache-Control"] = "no-store"
    return InvitationCreated(**InvitationRead.model_validate(row).model_dump(), token=token)


@router.delete("/{invitation_id}", status_code=204)
def revoke_invitation(invitation_id: uuid.UUID, db: DbSession, scope: WorkspaceDep) -> Response:
    scope.require("owner")
    _lock_owner(db, scope.id, scope.principal.id)
    row = db.scalar(
        select(WorkspaceInvitation)
        .where(
            WorkspaceInvitation.id == invitation_id, WorkspaceInvitation.workspace_id == scope.id
        )
        .with_for_update()
    )
    if row is None:
        raise NotFoundError("invitation", str(invitation_id))
    if row.revoked_at is None:
        row.revoked_at = utcnow()
    db.commit()
    return Response(status_code=204)


def _invitation(
    db: Session, token: str, principal_id: str, *, lock: bool = False
) -> tuple[WorkspaceInvitation, Membership | None]:
    query = select(WorkspaceInvitation).where(
        WorkspaceInvitation.token_hash == hashlib.sha256(token.encode()).hexdigest()
    )
    row = db.scalar(query)
    if row is None:
        raise HTTPException(status_code=404, detail="Invitation is unavailable.")
    if lock:
        # Every membership/invitation writer takes the workspace lock first.
        db.scalar(select(Workspace).where(Workspace.id == row.workspace_id).with_for_update())
        row = db.scalar(query.with_for_update().execution_options(populate_existing=True))
        if row is None:
            raise HTTPException(status_code=404, detail="Invitation is unavailable.")
    member = db.get(Membership, (row.workspace_id, principal_id), populate_existing=True)
    # A lost acceptance response can be retried, but a removed member cannot reuse it.
    if row.accepted_by == principal_id and member is not None:
        return row, member
    if row.revoked_at or row.accepted_at or row.expires_at <= utcnow():
        raise HTTPException(
            status_code=410, detail="Invitation has expired, was revoked, or was already used."
        )
    return row, member


@router.post("/inspect", response_model=InvitationPreview)
def inspect_invitation(
    payload: InvitationToken, db: DbSession, who: Identity, response: Response
) -> InvitationPreview:
    if who.pilot:
        raise ConflictError("Sign in with an individual account to use an invitation.")
    row, member = _invitation(db, payload.token, who.id)
    workspace = db.get(Workspace, row.workspace_id)
    assert workspace is not None
    response.headers["Cache-Control"] = "no-store"
    return InvitationPreview(
        workspace_name=workspace.name,
        role=row.role,
        expires_at=row.expires_at,
        already_member=member is not None,
    )


@router.post("/accept", response_model=WorkspaceRead)
def accept_invitation(
    payload: InvitationToken, db: DbSession, who: Identity, response: Response
) -> WorkspaceRead:
    if who.pilot:
        raise ConflictError("Sign in with an individual account to use an invitation.")
    row, member = _invitation(db, payload.token, who.id, lock=True)
    if member is None:
        if db.get(WorkspaceProfile, who.id) is None:
            raise ConflictError("Add your display name before joining the workspace.")
        member = Membership(workspace_id=row.workspace_id, principal_id=who.id, role=row.role)
        db.add(member)
        row.accepted_at, row.accepted_by = utcnow(), who.id
    # Existing members keep their role and do not consume someone else's invitation.
    workspace = db.get(Workspace, row.workspace_id)
    assert workspace is not None
    result = WorkspaceRead(id=workspace.id, name=workspace.name, role=member.role)
    db.commit()
    response.headers["Cache-Control"] = "no-store"
    return result
