from __future__ import annotations

import uuid

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

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
    capture_id: uuid.UUID, db: DbSession, storage: Storage, response: Response, request: Request
) -> LiveState:
    response.headers["Cache-Control"] = "no-store"
    return _absolute_tiles(live_service.live_for_capture(db, storage, capture_id), request)


@router.get(
    "/jobs/{job_id}/live",
    response_model=LiveState,
    summary="The live state of one run",
    description=_DESCRIPTION,
)
def job_live(
    job_id: uuid.UUID, db: DbSession, storage: Storage, response: Response, request: Request
) -> LiveState:
    response.headers["Cache-Control"] = "no-store"
    return _absolute_tiles(live_service.live_for_job(db, storage, job_id), request)


def _absolute_tiles(state: LiveState, request: Request) -> LiveState:
    if state.train and state.train.tiles and state.train.tiles.url.startswith("/api/v1/captures/"):
        # The console and API use different origins in production. Keep relative GLBs
        # on this route, not on the web origin or the signed tileset's bucket URL.
        route = "capture_live_tile"
        params: dict[str, str | int | uuid.UUID] = {"capture_id": state.capture_id}
        if "job_id" in request.path_params:
            route, params = "job_live_tile", {"job_id": state.job_id}
        state.train.tiles.url = str(
            request.url_for(
                route,
                **params,
                step=state.train.tiles.step,
                path="tileset.json",
            )
        )
    return state


@router.get(
    "/captures/{capture_id}/live/tiles/{step}/{path:path}",
    status_code=302,
    response_class=RedirectResponse,
    summary="Fetch a file from the newest run's live LOD tiles",
    responses={
        200: {"description": "With resolve=true, the signed URL as JSON for cross-origin browsers."}
    },
)
def capture_live_tile(
    capture_id: uuid.UUID,
    step: int,
    path: str,
    db: DbSession,
    storage: Storage,
    resolve: bool = False,
) -> Response:
    return _tile_response(
        live_service.tile_for_capture(db, storage, capture_id, step, path), resolve
    )


@router.get(
    "/jobs/{job_id}/live/tiles/{step}/{path:path}",
    status_code=302,
    response_class=RedirectResponse,
    summary="Fetch a file from one run's live LOD tiles",
    responses={
        200: {"description": "With resolve=true, the signed URL as JSON for cross-origin browsers."}
    },
)
def job_live_tile(
    job_id: uuid.UUID, step: int, path: str, db: DbSession, storage: Storage, resolve: bool = False
) -> Response:
    return _tile_response(live_service.tile_for_job(db, storage, job_id, step, path), resolve)


def _tile_response(url: str, resolve: bool) -> Response:
    headers = {"Cache-Control": "no-store"}
    if resolve:
        # Pages -> API -> R2 is a redirect-tainted fetch (Origin: null). A fresh
        # browser fetch keeps the Pages origin allowed by the existing bucket CORS.
        return JSONResponse({"url": url}, headers=headers)
    return RedirectResponse(url, status_code=302, headers=headers)
