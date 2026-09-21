from __future__ import annotations

import hmac
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import get_db
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
