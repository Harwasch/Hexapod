from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import select

from app.api.deps import DbSession
from app.api.workspace_deps import Identity, WorkspaceDep
from app.models.workspace import PILOT_WORKSPACE_ID, Membership, Workspace
from app.schemas.workspace import IdentityRead, MembershipWrite, WorkspaceCreate, WorkspaceRead
from app.services.errors import ConflictError, NotFoundError

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("/identity", response_model=IdentityRead)
def identity(who: Identity) -> IdentityRead:
    return IdentityRead(principal_id=who.id, mode="pilot" if who.pilot else "oidc")


@router.get("", response_model=list[WorkspaceRead])
def list_workspaces(db: DbSession, who: Identity) -> list[WorkspaceRead]:
    if who.pilot:
        return [WorkspaceRead(id=PILOT_WORKSPACE_ID, name="Pilot workspace", role="owner")]
    rows = db.execute(
        select(Workspace, Membership.role)
        .join(Membership, Membership.workspace_id == Workspace.id)
        .where(Membership.principal_id == who.id, Workspace.id != PILOT_WORKSPACE_ID)
        .order_by(Workspace.created_at, Workspace.id)
    )
    return [WorkspaceRead(id=row.id, name=row.name, role=role) for row, role in rows]


@router.post("", response_model=WorkspaceRead, status_code=201)
def create_workspace(payload: WorkspaceCreate, db: DbSession, who: Identity) -> WorkspaceRead:
    if who.pilot:
        raise HTTPException(status_code=409, detail="Multiple workspaces require OIDC identity.")
    row = Workspace(name=payload.name)
    db.add(row)
    db.flush()
    db.add(Membership(workspace_id=row.id, principal_id=who.id, role="owner"))
    db.commit()
    return WorkspaceRead(id=row.id, name=row.name, role="owner")


@router.get("/members", response_model=list[MembershipWrite])
def members(db: DbSession, scope: WorkspaceDep) -> list[MembershipWrite]:
    scope.require("owner")
    rows = db.scalars(select(Membership).where(Membership.workspace_id == scope.id))
    return [MembershipWrite.model_validate(row) for row in rows]


@router.put("/members", response_model=MembershipWrite)
def set_member(payload: MembershipWrite, db: DbSession, scope: WorkspaceDep) -> MembershipWrite:
    scope.require("owner")
    if scope.principal.pilot:
        raise ConflictError("Pilot access has no individual memberships.")
    # Serialize membership changes and recheck ownership after acquiring the lock.
    db.scalar(select(Workspace).where(Workspace.id == scope.id).with_for_update())
    actor = db.get(Membership, (scope.id, scope.principal.id), populate_existing=True)
    if actor is None or actor.role != "owner":
        raise HTTPException(status_code=403, detail="Only an owner can change membership.")
    row = db.get(Membership, (scope.id, payload.principal_id))
    if row and row.role == "owner" and payload.role != "owner":
        owners = db.scalars(
            select(Membership).where(
                Membership.workspace_id == scope.id, Membership.role == "owner"
            )
        ).all()
        if len(owners) == 1:
            raise ConflictError("A workspace must retain at least one owner.")
    if row:
        row.role = payload.role
    else:
        db.add(Membership(workspace_id=scope.id, **payload.model_dump()))
    db.commit()
    return payload


@router.delete("/members/{principal_id}", status_code=204)
def remove_member(principal_id: str, db: DbSession, scope: WorkspaceDep) -> Response:
    scope.require("owner")
    db.scalar(select(Workspace).where(Workspace.id == scope.id).with_for_update())
    actor = db.get(Membership, (scope.id, scope.principal.id), populate_existing=True)
    if actor is None or actor.role != "owner":
        raise HTTPException(status_code=403, detail="Only an owner can change membership.")
    row = db.get(Membership, (scope.id, principal_id))
    if row is None:
        raise NotFoundError("membership", principal_id)
    if row.role == "owner":
        owners = db.scalars(
            select(Membership).where(
                Membership.workspace_id == scope.id, Membership.role == "owner"
            )
        ).all()
        if len(owners) == 1:
            raise ConflictError("A workspace must retain at least one owner.")
    db.delete(row)
    db.commit()
    return Response(status_code=204)
