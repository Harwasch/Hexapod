from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from starlette.routing import compile_path

from app.config import Settings
from app.db import get_db
from app.services import handoff, phone_access, phone_key
from app.services.errors import UnauthorizedError
from app.services.ion import IonClient
from app.services.planner import Planner, build_planner
from app.services.ratelimit import client_key, limits_of
from app.services.vision import Outliner, build_outliner
from app.storage import ObjectStorage, get_public_storage, get_storage


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
#: Where something a browser fetches by URL is written: the public bucket where there are
#: two, the only bucket where there is one. Never `Storage` for that -- in production
#: `Storage` is the private bucket, which no browser can read.
PublicStorage = Annotated[ObjectStorage, Depends(get_public_storage)]

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


write_or_phone_key_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="writeOrPhoneKey",
    description=(
        "The write token or a PBKDF2-checked phone key. Phone keys reach only their "
        "own capture bundles and questions, and property reads for their captures' sites."
    ),
)


def check_phone_key(request: Request, settings: Settings, supplied: str) -> None:
    bucket = limits_of(request).phone_key
    client = client_key(request, settings)
    bucket.take(client)
    phone_key.check(settings, supplied)
    bucket.give_back(client)
    request.state.api_credential = "phone"


def require_write_or_phone_key(
    request: Request,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(write_or_phone_key_scheme)],
) -> None:
    supplied = credentials.credentials if credentials is not None else ""
    if settings.api_write_token and hmac.compare_digest(
        supplied.encode("utf-8"), settings.api_write_token.encode("utf-8")
    ):
        request.state.api_credential = "write"
        return
    # Preserve open writes in development, while recognizing explicit phone keys.
    if not settings.api_write_token:
        if supplied and phone_key.enabled(settings):
            try:
                check_phone_key(request, settings, supplied)
                return
            except UnauthorizedError:
                pass
        request.state.api_credential = "write"
        return
    check_phone_key(request, settings, supplied)


RequireWriteOrPhoneKey = Depends(require_write_or_phone_key)


def scoped_phone_capture(capture_id: uuid.UUID, request: Request, db: DbSession) -> None:
    if getattr(request.state, "api_credential", None) == "phone":
        phone_access.capture(db, capture_id)


PhoneCaptureScope = Depends(scoped_phone_capture)


def optional_phone_read(
    request: Request,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(write_or_phone_key_scheme)],
) -> None:
    # Public reads remain available without a credential.
    if credentials is not None:
        require_write_or_phone_key(request, settings, credentials)


PhoneRead = Depends(optional_phone_read)


# Exact route templates, including the non-schema Ask alias. A recognized phone key
# cannot use unrelated public reads either. Endpoint dependencies still enforce auth
# and resource ownership; this deny-by-default guard never grants access on its own.
PHONE_OPERATIONS = frozenset(
    {
        ("POST", "/api/v1/phone/check"),
        ("POST", "/api/v1/phone/captures"),
        *(
            ("POST", f"/api/v1/phone/captures/{{capture_id}}/{action}")
            for action in ("process", "refine", "stop")
        ),
        ("PUT", "/api/v1/captures/bundles/{bundle_id}"),
        ("PUT", "/api/v1/captures/{capture_id}/bundle-upload"),
        ("POST", "/api/v1/captures/{capture_id}/process-bundle"),
        ("POST", "/api/v1/captures/{capture_id}/files/{file_id}/parts"),
        ("POST", "/api/v1/captures/{capture_id}/files/{file_id}/complete"),
        ("GET", "/api/v1/captures/{capture_id}"),
        ("POST", "/api/v1/ask"),
        ("POST", "/v1/ask"),
        ("GET", "/api/v1/properties/{site_id}/entities"),
        ("POST", "/api/v1/properties/{site_id}/entities/search"),
    }
)
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_PHONE_OPERATION_PATTERNS = tuple(
    (method, compile_path(template)[0]) for method, template in PHONE_OPERATIONS
)


def _allowed_for_phone(method: str, path: str) -> bool:
    """Whether this method and full path match an operation in PHONE_OPERATIONS.

    An app-level dependency sees only the router-relative route in `scope["route"]`
    (`/phone/check`), so the full request path is matched against the allowlist's own
    templates, compiled the way Starlette compiles route paths.
    """
    return any(m == method and regex.fullmatch(path) for m, regex in _PHONE_OPERATION_PATTERNS)


def restrict_phone_routes(request: Request, settings: SettingsDep) -> None:
    # Reads are public or carry their own credential checks, and a phone key grants nothing
    # there that an anonymous request lacks. Checking them would run the deliberately slow
    # key hash, and spend the phone-key rate limit, on every bearer read.
    if request.method not in _MUTATING:
        return
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        return
    if settings.api_write_token and hmac.compare_digest(
        supplied.encode("utf-8"), settings.api_write_token.encode("utf-8")
    ):
        return
    if not phone_key.enabled(settings):
        return
    if _allowed_for_phone(request.method, request.scope.get("path", "")):
        return
    # Bound password hashing on public endpoints, just as on /phone/check.
    bucket = limits_of(request).phone_key
    client = client_key(request, settings)
    bucket.take(client)
    try:
        phone_key.check(settings, supplied)
    except UnauthorizedError:
        return
    bucket.give_back(client)
    raise UnauthorizedError("That phone key is not right.")


#: `Authorization: Bearer <write token | handoff token>` on the upload endpoints.
#:
#: A second scheme rather than a second header: both tokens are bearer credentials for
#: the same API, and the header a phone sends must not depend on which one it holds. The
#: schemes are separate in the document because they are not interchangeable anywhere
#: else. The app bundle routes use `writeOrPhoneKey`; part/complete also retain this
#: handoff scheme via require_bundle_part_token.
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

    Only these four upload endpoints admit a handoff token. Part/complete wrap this
    dependency to also admit scoped phone keys; the other routes never accept handoff
    tokens for creation, processing, deletion or another capture.
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


def require_bundle_part_token(
    capture_id: uuid.UUID,
    file_id: uuid.UUID,
    request: Request,
    settings: SettingsDep,
    db: DbSession,
    response: Response,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(upload_token_scheme)],
    phone_credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(write_or_phone_key_scheme)
    ],
) -> None:
    # Preserve handoff renewal and existing write-token behavior on shared routes.
    if settings.api_write_token:
        try:
            require_upload_token(capture_id, settings, response, credentials)
            supplied = credentials.credentials if credentials else ""
            request.state.api_credential = (
                "write"
                if hmac.compare_digest(
                    supplied.encode("utf-8"), settings.api_write_token.encode("utf-8")
                )
                else "handoff"
            )
            return
        except UnauthorizedError:
            pass
    require_write_or_phone_key(request, settings, phone_credentials)
    if getattr(request.state, "api_credential", None) == "phone":
        row = phone_access.capture(db, capture_id)
        if row.metadata_.get("bundleFormat") != "hex-capture-v1" or not any(
            file.id == file_id
            and file.filename == "capture.hexcap"
            and file.content_type == "application/x-hex-capture"
            for file in row.files
        ):
            raise UnauthorizedError("That phone key is not right.")


RequireBundlePartToken = Depends(require_bundle_part_token)


def limit_reconcile(request: Request, settings: SettingsDep) -> None:
    """One bucket walk per token: see app/services/ratelimit.py. Listed after the write
    token on its route, so a request without one is refused before it spends anything."""
    limits_of(request).reconcile.take(client_key(request, settings))


def limit_step_log(request: Request, settings: SettingsDep) -> None:
    """One object-storage read per token: see app/services/ratelimit.py."""
    limits_of(request).step_log.take(client_key(request, settings))


def limit_place(request: Request, settings: SettingsDep) -> None:
    """One public-agency lookup per token: see app/services/ratelimit.py."""
    limits_of(request).place.take(client_key(request, settings))


def limit_topo_tile(request: Request, settings: SettingsDep) -> None:
    """A few byte-range reads of a USGS sheet per token: see app/services/ratelimit.py."""
    limits_of(request).topo_tile.take(client_key(request, settings))


def limit_imagery_catalog(request: Request, settings: SettingsDep) -> None:
    """One dated-imagery catalog per token. Called by the route itself, only on a cache miss:
    a catalog this server holds costs nothing to answer."""
    limits_of(request).imagery_catalog.take(client_key(request, settings))


def limit_cell_map(request: Request, settings: SettingsDep) -> None:
    """One confidence-map computation per token: see app/services/ratelimit.py."""
    limits_of(request).cell_map.take(client_key(request, settings))


LimitReconcile = Depends(limit_reconcile)
LimitStepLog = Depends(limit_step_log)
LimitPlace = Depends(limit_place)
LimitTopoTile = Depends(limit_topo_tile)
LimitCellMap = Depends(limit_cell_map)
