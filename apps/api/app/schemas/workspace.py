from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.schemas.base import CamelModel

Role = Literal["owner", "editor", "viewer"]


class WorkspaceCreate(CamelModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class WorkspaceRead(WorkspaceCreate):
    id: uuid.UUID
    role: Role


class MembershipWrite(CamelModel):
    principal_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    role: Role


class IdentityRead(CamelModel):
    principal_id: str
    mode: Literal["pilot", "oidc"]


class ProfileWrite(CamelModel):
    display_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
    ]


class ProfileRead(IdentityRead):
    display_name: str | None = None


class MemberRead(MembershipWrite):
    display_name: str | None = None


class InvitationCreate(CamelModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    role: Literal["editor", "viewer"] = "viewer"
    expires_in_days: int = Field(default=7, ge=1, le=30)


class InvitationRead(CamelModel):
    id: uuid.UUID
    label: str
    role: Literal["editor", "viewer"]
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    accepted_at: datetime | None


class InvitationCreated(InvitationRead):
    token: str


class InvitationToken(CamelModel):
    token: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")


class InvitationPreview(CamelModel):
    workspace_name: str
    role: Literal["editor", "viewer"]
    expires_at: datetime
    already_member: bool
