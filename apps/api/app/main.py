"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_v1
from app.config import REPO_ROOT, Settings, get_settings
from app.schemas.common import Problem
from app.services.errors import ConflictError, NotFoundError, UnauthorizedError
from app.services.urls import UrlValidationError
from app.storage import StorageUnavailableError

logger = logging.getLogger("twin.api")

API_DESCRIPTION = """
Catalog service for the geospatial digital twin.

* **Sites** — physical places with reality captures, footprints and camera bookmarks.
* **Assets** — derived, renderable delivery assets (3D Tiles) with provenance.
* **Layers** — composable world layers from open data or the user's own sources.
* **Captures** — uploads in progress: source files go browser → object storage over
  presigned multipart URLs, never through this API.
* **Jobs** — pipeline runs over a capture, queued here and executed by a worker.

All geometry is GeoJSON (WGS 84, RFC 7946). All timestamps are ISO 8601.

Reads are open. Every mutating endpoint requires the shared write token as
`Authorization: Bearer <API_WRITE_TOKEN>`, unless the deployment has no token
configured — which production refuses to start without.
"""


def _problem(
    status_code: int,
    title: str,
    detail: str | None = None,
    errors: list[dict[str, object]] | None = None,
) -> JSONResponse:
    payload = Problem(title=title, status=status_code, detail=detail, errors=errors)
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json", by_alias=True, exclude_none=True),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Fail at startup, not at the first unauthenticated POST. An unset API_WRITE_TOKEN
    # means "writes are open", which is how a fresh checkout and the test suite run with
    # no configuration; this line is what stops that convenience reaching production.
    if settings.is_production and not settings.api_write_token:
        raise RuntimeError(
            "API_WRITE_TOKEN must be set when APP_ENV=production: refusing to start an "
            "internet-facing API whose writes are open."
        )
    # The same instinct, one line later. Production with neither TILES_BASE_URL nor a
    # bucket used to log an error and carry on seeding capture URLs against the static
    # mount -- which production disables and this image has no data/tiles to serve
    # anyway. Every one of those URLs 404s in a browser, hours after the deploy, and the
    # only trace is a line in a log nobody is reading. A deployment that cannot serve the
    # thing it exists to serve should not come up.
    if settings.is_production and not settings.tiles_source_configured:
        raise RuntimeError(
            "APP_ENV=production needs somewhere to serve capture tiles from, and has "
            "neither: set OBJECT_STORAGE_ENDPOINT_URL, OBJECT_STORAGE_BUCKET, "
            "OBJECT_STORAGE_ACCESS_KEY and OBJECT_STORAGE_SECRET_KEY (uploads need them "
            "regardless), or set TILES_BASE_URL to the public prefix the tiles are "
            "published under. With neither, every capture would be seeded against the "
            "/api/v1/tiles static mount, which production disables and this image does "
            "not carry -- so every tileset URL would 404."
        )
    # The third refusal, and the one with the widest blast radius. Enabling public access
    # on an R2 bucket exposes the whole bucket -- Cloudflare's own words are that it
    # "allows users to expose the contents of their R2 buckets directly to the Internet",
    # with no prefix scoping, and a custom domain behaves identically. This API keeps raw
    # uploads under `captures/` and every run's frames, logs and checkpoints under
    # `runs/`, so a single bucket with a public URL on it publishes all of that along with
    # the tiles it meant to publish. There is no symptom: the globe works perfectly, and
    # the exposure is visible only to somebody who already has a key. Set
    # OBJECT_STORAGE_PUBLIC_BUCKET to a second bucket and only the objects `register`
    # copies into it are reachable; see app/worker/publish.py.
    if settings.is_production and settings.object_storage_configured:
        if not settings.publish_bucket_configured:
            raise RuntimeError(
                "APP_ENV=production with one bucket in both roles: set "
                "OBJECT_STORAGE_PUBLIC_BUCKET to a bucket other than "
                f"{settings.object_storage_bucket!r}. Making tiles readable means making "
                "the bucket readable -- R2 has no per-prefix public access -- and this "
                "bucket also holds every raw upload under captures/ and every run's "
                "frames, logs and checkpoints under runs/."
            )
        if not settings.object_storage_public_url:
            raise RuntimeError(
                "APP_ENV=production has OBJECT_STORAGE_PUBLIC_BUCKET but no "
                "OBJECT_STORAGE_PUBLIC_URL. Published tiles would be addressed against "
                "the S3 API endpoint, which is not the public hostname and is not a CDN; "
                "set the custom domain the public bucket is served on."
            )
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=API_DESCRIPTION,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        # Authorization carries the write token, so the preflight has to allow it or
        # every browser write fails before it is sent.
        allow_headers=["Content-Type", "Accept", "Authorization"],
        # A handoff-authorised response carries the next token in this header, and a
        # cross-origin phone page cannot read a header the server does not expose --
        # without this line the renewal chain silently breaks the moment the web app is
        # served from an origin other than the API's.
        expose_headers=["X-Handoff-Token"],
        max_age=600,
    )

    @app.middleware("http")
    async def timing_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
        if elapsed_ms > 500:
            logger.warning(
                "slow request %s %s took %.0f ms", request.method, request.url.path, elapsed_ms
            )
        return response

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return _problem(status.HTTP_404_NOT_FOUND, "Not found", str(exc))

    @app.exception_handler(ConflictError)
    async def conflict_handler(_: Request, exc: ConflictError) -> JSONResponse:
        return _problem(status.HTTP_409_CONFLICT, "Conflict", str(exc))

    @app.exception_handler(UnauthorizedError)
    async def unauthorized_handler(_: Request, exc: UnauthorizedError) -> JSONResponse:
        response = _problem(status.HTTP_401_UNAUTHORIZED, "Unauthorized", str(exc))
        # RFC 9110 requires this on a 401, and it tells a client which scheme to use.
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(StorageUnavailableError)
    async def storage_handler(_: Request, exc: StorageUnavailableError) -> JSONResponse:
        return _problem(status.HTTP_503_SERVICE_UNAVAILABLE, "Object storage unavailable", str(exc))

    @app.exception_handler(UrlValidationError)
    async def url_handler(_: Request, exc: UrlValidationError) -> JSONResponse:
        return _problem(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid URL", str(exc))

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return _problem(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid input", str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors: list[dict[str, object]] = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _problem(status.HTTP_422_UNPROCESSABLE_CONTENT, "Validation error", None, errors)

    # Read by app.api.deps._settings, so a test app built with explicit settings is
    # governed by them rather than by the process-wide lru_cached environment.
    app.state.settings = settings
    app.include_router(api_v1)
    # **Development only.** Capture tiles kept on a developer's disk
    # (data/tiles/<slug>/<representation>/tileset.json) are served as static 3D Tiles under
    # the API prefix, so the web app's /api proxy covers them with no bucket configured.
    #
    # This mount used to be the *only* way a local capture was reachable, and
    # infra/api.Dockerfile copies apps/api/ and nothing else -- so in the one deployment
    # path the docs describe, `data/tiles` did not exist, this mount was silently absent,
    # and every local capture 404'd. A9 did not patch that by copying tiles into the image;
    # it removed the dependency. A deployment with object storage configured seeds tileset
    # URLs that point at the bucket (app/seed/captures.tiles_base_url), so whether this
    # directory exists no longer decides whether a capture loads.
    #
    # Production is refused the mount outright rather than being allowed to depend on it by
    # accident: an image that happened to carry a stale data/tiles would otherwise serve it.
    tiles_dir = Path(settings.tiles_dir)
    if not tiles_dir.is_absolute():
        tiles_dir = REPO_ROOT / tiles_dir
    if settings.is_production:
        logger.info("tiles are served from object storage; the local static mount is disabled")
    elif tiles_dir.is_dir():
        app.mount("/api/v1/tiles", StaticFiles(directory=str(tiles_dir)), name="tiles")
    return app


app = create_app()
