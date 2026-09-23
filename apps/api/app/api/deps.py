from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import get_db
from app.services import handoff
from app.services.errors import UnauthorizedError
from app.services.ion import IonClient
from app.services.planner import Planner, build_planner
from app.services.vision import Outliner, build_outliner
from app.storage import ObjectStorage, get_storage


def _db() -> Generator[Session, None, None]:
    yield from get_db()


def _settings(request: Request) -> Settings:
    """The settings this app was built with.

    Read off `app.state` rather than `get_settings()` so that an app created with
    explicit settings -- `create_app(Settings(api_write_token=...))` -- is actually
    governed by them, instead of by the process-wide lru_cached environment.
    """
    settings: Settings = request.app.state.settings
    return settings


def _ion() -> IonClient:
    return IonClient()


def _planner() -> Planner:
    return build_planner()


def _outliner() -> Outliner | None:
    return build_outliner()


DbSession = Annotated[Session, Depends(_db)]
SettingsDep = Annotated[Settings, Depends(_settings)]
Ion = Annotated[IonClient, Depends(_ion)]
PlannerDep = Annotated[Planner, Depends(_planner)]
OutlinerDep = Annotated[Outliner | None, Depends(_outliner)]
Storage = Annotated[ObjectStorage, Depends(get_storage)]

#: `Authorization: Bearer <API_WRITE_TOKEN>`. auto_error=False so a missing header
#: reaches the check below and is answered with the same RFC-7807 Problem as a wrong
#: one, rather than FastAPI's own 403 body.
write_token_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="writeToken",
    description="The shared write token (`API_WRITE_TOKEN`), as `Authorization: Bearer <token>`.",
)


def require_write_token(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(write_token_scheme)],
) -> None:
    """Gate every mutating endpoint on the single shared write token.

    Reads stay open on purpose: the world is meant to be viewable without a token.

    With no token configured this is a no-op, which is what keeps a fresh checkout and
    the test suite working with zero configuration. That is safe only because
    `create_app` refuses to start a production environment without a token -- the
    unset case cannot reach production, it can only stay local.
    """
    expected = settings.api_write_token
    if not expected:
        return
    supplied = credentials.credentials if credentials is not None else ""
    # compare_digest, not `==`: a plain comparison leaks the length of the shared
    # prefix through timing, and ruff's S105-family rules flag it. Bytes, not str,
    # because compare_digest raises TypeError on a non-ASCII str.
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise UnauthorizedError(
            "This endpoint needs the write token: send Authorization: Bearer <API_WRITE_TOKEN>."
        )


#: Spelled once so every mutating route reads `dependencies=[RequireWriteToken]`.
RequireWriteToken = Depends(require_write_token)


#: `Authorization: Bearer <write token | handoff token>` on the upload endpoints.
#:
#: A second scheme rather than a second header: both tokens are bearer credentials for
#: the same API, and the header a phone sends must not depend on which one it holds. The
#: schemes are separate in the document because they are not interchangeable anywhere
#: else -- every other mutating route takes `writeToken` and only `writeToken`.
upload_token_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="uploadToken",
    description=(
        "Either the shared write token (`API_WRITE_TOKEN`), or a capture handoff token "
        "from `POST /captures/{capture_id}/handoff`. A handoff token is accepted only on "
        "this capture's upload endpoints and only until it expires."
    ),
)

#: Header carrying the next token in a handoff chain. See app/services/handoff.py.
HANDOFF_RENEWAL_HEADER = "X-Handoff-Token"


def require_upload_token(
    capture_id: uuid.UUID,
    settings: SettingsDep,
    response: Response,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(upload_token_scheme)],
) -> None:
    """Gate one capture's upload endpoints on the write token *or* a handoff token for it.

    `capture_id` is declared here as a path parameter, so it is the id of the capture in
    *this* request's URL, resolved by FastAPI before the route function runs. That is the
    whole scope check: a handoff token whose signed capture id is not this one is refused
    here, every request, with no cooperation required from the client. There is no code
    path on which a handoff token reaches a route whose capture id it does not name.

    Everything outside these four endpoints keeps `require_write_token`, which does not
    know handoff tokens exist -- so creating a capture, queueing a job, deleting a site
    or touching another capture is closed to a handoff token by construction, not by a
    check that could be forgotten.
    """
    supplied = credentials.credentials if credentials is not None else ""
    expected = settings.api_write_token
    if not expected:
        # Writes are open on this deployment (see require_write_token). Not a place to
        # start enforcing handoff tokens: it would make the phone page the *only* thing
        # needing credentials on an API that has none.
        return
    if hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        return

    now = int(time.time())
    key = handoff.key_for(settings)
    # Raises UnauthorizedError -- the same 401 Problem as a missing write token, saying
    # no more than that this link is not valid here.
    claims = handoff.verify(key, supplied, capture_id=capture_id, now=now)
    # A capture-sized upload outlives a handoff-sized token, so every authorised response
    # carries the next one. The ceiling inside `claims` rides along unchanged, which is
    # what stops the chain running forever.
    renewed = handoff.renew(key, claims, now=now)
    if renewed is not None:
        response.headers[HANDOFF_RENEWAL_HEADER] = renewed


#: Spelled once so the upload routes read `dependencies=[RequireUploadToken]`.
RequireUploadToken = Depends(require_upload_token)
