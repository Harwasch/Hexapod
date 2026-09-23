from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import DbSession
from app.models.enums import ArtifactKind
from app.schemas.job import ArtifactRow
from app.services import artifacts as artifact_service

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get(
    "",
    response_model=list[ArtifactRow],
    summary="List artifacts, newest first",
    description=(
        "Every output the pipeline has written, with the run that produced it and "
        "whatever points at it. `unreferenced=true` is the cleanup list: artifacts no "
        "site asset or thumbnail URL refers to."
    ),
)
def list_artifacts(
    db: DbSession,
    kind: Annotated[ArtifactKind | None, Query()] = None,
    job_id: Annotated[uuid.UUID | None, Query(alias="jobId")] = None,
    capture_id: Annotated[uuid.UUID | None, Query(alias="captureId")] = None,
    unreferenced: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ArtifactRow]:
    return artifact_service.list_artifacts(
        db,
        kind=kind,
        job_id=job_id,
        capture_id=capture_id,
        unreferenced=unreferenced,
        limit=limit,
        offset=offset,
    )
