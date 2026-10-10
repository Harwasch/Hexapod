"""Authenticate and bound JSON wire bodies before FastAPI parses model input."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.routing import APIRoute

from app.api.deps import require_write_token, write_token_scheme


class AuthenticatedBodyRoute(APIRoute):
    def body_limit(self) -> int:
        return 20 * 1024 * 1024

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def bounded(context: Request) -> Response:
            if context.method in {"POST", "PUT", "PATCH"}:
                settings = getattr(context.app.state, "settings", None)
                if settings is not None:
                    require_write_token(settings, await write_token_scheme(context))
                # Ordinary route dependencies remain in force for all requests,
                # including test applications using explicit dependency overrides.
                limit = self.body_limit()
                length = context.headers.get("content-length")
                if length is not None:
                    try:
                        declared = int(length)
                    except ValueError:
                        raise HTTPException(400, "Invalid content length.") from None
                    if declared < 0:
                        raise HTTPException(400, "Invalid content length.")
                    if declared > limit:
                        raise HTTPException(413, "Worlds request body exceeds the size limit.")
                body = bytearray()
                async for chunk in context.stream():
                    if len(body) + len(chunk) > limit:
                        raise HTTPException(413, "Worlds request body exceeds the size limit.")
                    body.extend(chunk)
                context._body = bytes(body)
            return await original(context)

        return bounded
