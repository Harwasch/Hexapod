"""Site scans requested from the landing page's form.

`POST` is open to anyone, on purpose (Harrison's decision, 2026-10-20): it is how a visitor who
has not signed in asks for a scan. What bounds it: a tight per-client rate limit and a body-size
cap, both before the body is read (app/main.py); a cap on the day's requests across every
client; a honeypot field; size, precision, time and area bounds on the outline; strict email and
control-character checks; and one generic answer for every accepted submission, so the route
tells a prober nothing. `GET` holds names and emails, so it always takes the write token, and
refuses outright on a server with none configured: the admin console reads it. Requests are kept
indefinitely, with no automatic deletion, until the person who sent one asks for it to be deleted
(Harrison's decision, 2026-10-20).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.security import HTTPAuthorizationCredentials

from app.api.deps import DbSession, SettingsDep, require_write_token, write_token_scheme
from app.schemas.scan_request import ScanRequestCreate, ScanRequestRead, ScanRequestReceived
from app.services import scan_requests

router = APIRouter(prefix="/scan-requests", tags=["scan-requests"])

#: What a request beyond the day's cap is told to wait (s).
DAILY_CAP_RETRY_S = 3600


def require_list_token(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(write_token_scheme)],
) -> None:
    """The write token, always. `require_write_token` lets everything through on a server with
    no token configured (a fresh checkout); a list of names and emails may not be read there."""
    if not settings.api_write_token:
        raise HTTPException(
            status_code=403,
            detail="Scan requests can be read only with API_WRITE_TOKEN set on the server, "
            "and none is.",
        )
    require_write_token(settings, credentials)


@router.post(
    "",
    response_model=ScanRequestReceived,
    status_code=202,
    responses={
        413: {"description": "The request body is larger than this route accepts"},
        429: {"description": "Too many requests from this client; see Retry-After"},
        503: {"description": "The day's requests are all in; see Retry-After"},
    },
)
def create(payload: ScanRequestCreate, db: DbSession, settings: SettingsDep) -> ScanRequestReceived:
    """Request a site scan. No token. Answered `received` whether stored or not."""
    try:
        scan_requests.create_scan_request(
            db,
            payload,
            daily_cap=settings.scan_requests_daily_cap,
            clean_timeout_ms=settings.scan_request_clean_timeout_ms,
        )
    except scan_requests.DailyCapReached as capped:
        raise HTTPException(
            status_code=503,
            detail="We cannot take more requests right now. Please try again later.",
            headers={"Retry-After": str(DAILY_CAP_RETRY_S)},
        ) from capped
    return ScanRequestReceived()


@router.get(
    "",
    response_model=list[ScanRequestRead],
    dependencies=[Depends(require_list_token)],
    responses={403: {"description": "No write token is configured on this server"}},
)
def list_requests(
    db: DbSession,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[ScanRequestRead]:
    """The most recent scan requests first. Write token required: they hold contact details."""
    response.headers["Cache-Control"] = "private, no-store"
    return scan_requests.list_scan_requests(db, limit)
