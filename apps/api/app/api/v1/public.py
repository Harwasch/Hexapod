"""Routes for visitors who have not signed in: today, finding their land on the landing page.

Nothing here reads or writes workspace data, and nothing takes a token. What bounds them: a
per-client rate limit (app/services/ratelimit.py), a body-size cap for the whole `/public`
prefix (app/main.py), and for the parcel lookup a short county timeout, an answer cache and a
cap on lookups running at once (app/services/public_parcels.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import DbSession, LimitPublicParcels
from app.schemas.land_selection import CandidateResult, PublicParcelRequest
from app.services import land_selection
from app.services.public_parcels import BUSY_RETRY_S, Busy, PublicParcels

router = APIRouter(prefix="/public", tags=["public"])


def parcel_http() -> Iterator[httpx.Client]:
    """The client county parcel services are asked with (a dependency, so tests can stub it)."""
    timeout = httpx.Timeout(land_selection.PUBLIC_TIMEOUT_S, connect=3.0)
    with httpx.Client(
        headers={"User-Agent": "LivingWorld-PublicParcels/1.0"}, timeout=timeout
    ) as client:
        yield client


ParcelHttp = Annotated[httpx.Client, Depends(parcel_http)]


@router.post(
    "/parcels/near",
    response_model=CandidateResult,
    dependencies=[LimitPublicParcels],
    responses={
        413: {"description": "The request body is larger than this route accepts"},
        429: {"description": "Too many lookups from this client; see Retry-After"},
        503: {"description": "Too many lookups running at once; see Retry-After"},
    },
)
def parcels_near(
    payload: PublicParcelRequest,
    request: Request,
    db: DbSession,
    client: ParcelHttp,
) -> CandidateResult:
    """County assessor parcels near a point, with their source, attribution and any notice
    their terms require.

    California only (elsewhere the answer is `uncovered`), and only the counties whose terms
    were verified (`public_counties`; docs/PUBLIC_PARCELS.md).
    """
    shared: PublicParcels = request.app.state.public_parcels
    try:
        return shared.lookup(
            PublicParcels.key(payload),
            lambda: land_selection.public_parcels(db, payload, client),
        )
    except Busy as busy:
        raise HTTPException(
            status_code=503,
            detail="Parcel lookups are busy. Try again in a few seconds.",
            headers={"Retry-After": str(BUSY_RETRY_S)},
        ) from busy
