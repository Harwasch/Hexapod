from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.dialects.postgresql import insert

from app.api.deps import (
    DbSession,
    SettingsDep,
    require_write_token,
    write_token_scheme,
)
from app.models.workspace import PILOT_WORKSPACE_ID, Membership, Workspace
from app.services.errors import NotFoundError
from app.services.identity import Principal, principal_id, verify_token


def current_principal(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(write_token_scheme)],
) -> Principal:
    if settings.land_auth_mode == "pilot":
        require_write_token(settings, credentials)
        return Principal(principal_id("pilot", "operator"), pilot=True)
    return verify_token(credentials.credentials if credentials else "", settings)


Identity = Annotated[Principal, Depends(current_principal)]


@dataclass(frozen=True)
class WorkspaceAccess:
    id: uuid.UUID
    principal: Principal
    role: str

    def require(self, *roles: str) -> None:
        if self.role not in roles:
            raise HTTPException(status_code=403, detail="Your workspace role cannot do this.")


def workspace_access(
    identity: Identity,
    db: DbSession,
    workspace_id: Annotated[uuid.UUID | None, Header(alias="X-Workspace-ID")] = None,
) -> WorkspaceAccess:
    if identity.pilot:
        if workspace_id is not None and workspace_id != PILOT_WORKSPACE_ID:
            raise NotFoundError("workspace", workspace_id)
        db.execute(
            insert(Workspace)
            .values(id=PILOT_WORKSPACE_ID, name="Pilot workspace")
            .on_conflict_do_nothing(index_elements=[Workspace.id])
        )
        return WorkspaceAccess(PILOT_WORKSPACE_ID, identity, "owner")
    if workspace_id is None:
        raise HTTPException(status_code=400, detail="Choose a workspace using X-Workspace-ID.")
    # The pilot's records are never implicitly claimed by the first OIDC user.
    member = db.get(Membership, (workspace_id, identity.id))
    if member is None or workspace_id == PILOT_WORKSPACE_ID:
        raise NotFoundError("workspace", workspace_id)
    return WorkspaceAccess(workspace_id, identity, member.role)


WorkspaceDep = Annotated[WorkspaceAccess, Depends(workspace_access)]
