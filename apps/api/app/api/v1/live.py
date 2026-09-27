from __future__ import annotations

import uuid

from fastapi import APIRouter, Response

from app.api.deps import DbSession, Storage
from app.schemas.live import LiveState
from app.services import live as live_service

router = APIRouter(tags=["live"])

_DESCRIPTION = (
    "What a run looks like while it runs, for the live viewer: the pose stage's cameras "
    "and a sample of its sparse points as they are solved (quantised, base64; see the "
    "schema), and training's newest intermediate splat with a short-lived signed URL for "
    "its SPZ (null until the checkpoint syncer has uploaded it). Coordinates are the pose "
    "solve's own frame, not east/north/up. Open, like every read: poll it every few seconds."
)


@router.get(
    "/captures/{capture_id}/live",
    response_model=LiveState,
    summary="The live state of a capture's newest run",
    description=_DESCRIPTION,
)
def capture_live(
    capture_id: uuid.UUID, db: DbSession, storage: Storage, response: Response
) -> LiveState:
    response.headers["Cache-Control"] = "no-store"
    return live_service.live_for_capture(db, storage, capture_id)


@router.get(
    "/jobs/{job_id}/live",
    response_model=LiveState,
    summary="The live state of one run",
    description=_DESCRIPTION,
)
def job_live(job_id: uuid.UUID, db: DbSession, storage: Storage, response: Response) -> LiveState:
    response.headers["Cache-Control"] = "no-store"
    return live_service.live_for_job(db, storage, job_id)
