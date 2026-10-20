"""Routes for visitors who have not signed in: today, finding their land on the landing page.

Nothing here reads or writes workspace data, and nothing takes a token. Each route that calls
a provider on the visitor's behalf is rate-limited per client (app/services/ratelimit.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends

from app.api.deps import DbSession, LimitPublicParcels
from app.schemas.land_selection import CandidateResult, PublicParcelRequest
from app.services import land_selection

router = APIRouter(prefix="/public", tags=["public"])


def parcel_http() -> Iterator[httpx.Client]:
    """The client county parcel services are asked with (a dependency, so tests can stub it)."""
    with httpx.Client(headers={"User-Agent": "LivingWorld-PublicParcels/1.0"}) as client:
        yield client


ParcelHttp = Annotated[httpx.Client, Depends(parcel_http)]


@router.post(
    "/parcels/near",
    response_model=CandidateResult,
    dependencies=[LimitPublicParcels],
    responses={429: {"description": "Too many lookups from this client; see Retry-After"}},
)
def parcels_near(
    payload: PublicParcelRequest, db: DbSession, client: ParcelHttp
) -> CandidateResult:
    """County assessor parcels near a point, with their source and attribution.

    California only (elsewhere the answer is `uncovered`), and only counties whose published
    terms allow the outlines on a public page (`NO_PUBLIC_REUSE`).
    """
    return land_selection.public_parcels(db, payload, client)
