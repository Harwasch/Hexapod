"""Site scans requested from the landing page's form.

`POST` is open to anyone, on purpose (Harrison's decision, 2026-10-20): it is how a visitor who
has not signed in asks for a scan. What bounds it: a tight per-client rate limit, a body-size
cap (app/main.py), a honeypot field, an area cap on the outline, an email check, and one
generic answer for every accepted submission, so the route tells a prober nothing. `GET`
holds names and emails, so it takes the write token: the admin console reads it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from app.api.deps import DbSession, LimitScanRequests, RequireWriteToken
from app.schemas.scan_request import ScanRequestCreate, ScanRequestRead, ScanRequestReceived
from app.services import scan_requests

router = APIRouter(prefix="/scan-requests", tags=["scan-requests"])


@router.post(
    "",
    response_model=ScanRequestReceived,
    status_code=202,
    dependencies=[LimitScanRequests],
    responses={
        413: {"description": "The request body is larger than this route accepts"},
        429: {"description": "Too many requests from this client; see Retry-After"},
    },
)
def create(payload: ScanRequestCreate, db: DbSession) -> ScanRequestReceived:
    """Request a site scan. No token. Answered `received` whether stored or not."""
    scan_requests.create_scan_request(db, payload)
    return ScanRequestReceived()


@router.get("", response_model=list[ScanRequestRead], dependencies=[RequireWriteToken])
def list_requests(
    db: DbSession,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[ScanRequestRead]:
    """The most recent scan requests first. Write token required: they hold contact details."""
    response.headers["Cache-Control"] = "private, no-store"
    return scan_requests.list_scan_requests(db, limit)
