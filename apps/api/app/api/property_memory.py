"""Catalog observations across scans of a property; historical answers stay authenticated."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import DbSession, PhoneRead, RequireWriteOrPhoneKey, RequireWriteToken
from app.services import phone_access
from app.services.property_memory import (
    MemoryQuery,
    Observation,
    PostgresMemory,
    dedupe_candidates,
)

router = APIRouter(prefix="/properties", tags=["property-memory"])


@router.get("/{site_id}/entities", dependencies=[PhoneRead])
def entities(
    site_id: uuid.UUID, db: DbSession, query: Annotated[MemoryQuery, Query()], request: Request
) -> dict[str, Any]:
    if getattr(request.state, "api_credential", None) == "phone":
        phone_access.site(db, site_id)
    return _page(PostgresMemory(db).entities(site_id, query), query)


@router.post("/{site_id}/entities/search", dependencies=[RequireWriteOrPhoneKey])
def search(
    site_id: uuid.UUID, query: MemoryQuery, db: DbSession, request: Request
) -> dict[str, Any]:
    """Read-only POST permits polygon/JSON attribute filters without URL-size limits."""
    if getattr(request.state, "api_credential", None) == "phone":
        phone_access.site(db, site_id)
    return _page(PostgresMemory(db).entities(site_id, query), query)


def _page(items: list[dict[str, Any]], query: MemoryQuery) -> dict[str, Any]:
    return {
        "items": items,
        "nextOffset": query.offset + query.limit if len(items) == query.limit else None,
    }


class DedupeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Observation
    candidates: list[Observation] = Field(max_length=500)


@router.post("/{site_id}/entities/dedupe", dependencies=[RequireWriteToken])
def dedupe(site_id: uuid.UUID, request: Annotated[DedupeRequest, Body()]) -> dict[str, Any]:
    """Pure proposal against supplied observations. No database writes or identity claims."""
    if request.observation.site_id != site_id or any(
        c.site_id != site_id for c in request.candidates
    ):
        raise HTTPException(422, "all observations must belong to the requested property")
    return dedupe_candidates(request.observation, request.candidates)
