from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel

Role = Literal["owner", "editor", "viewer"]


class WorkspaceCreate(CamelModel):
    name: str = Field(min_length=1, max_length=200)


class WorkspaceRead(WorkspaceCreate):
    id: uuid.UUID
    role: Role


class MembershipWrite(CamelModel):
    principal_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    role: Role


class IdentityRead(CamelModel):
    principal_id: str
    mode: Literal["pilot", "oidc"]
