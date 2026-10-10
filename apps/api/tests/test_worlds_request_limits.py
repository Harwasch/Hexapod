from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.services.errors import UnauthorizedError
from app.worlds import intelligence
from app.worlds.request_limits import AuthenticatedBodyRoute


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.state.settings = SimpleNamespace(api_write_token="private-test-token")
    app.add_exception_handler(
        UnauthorizedError,
        lambda request, error: JSONResponse(status_code=401, content={"detail": "Unauthorized"}),
    )
    app.include_router(intelligence.router)
    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/worlds/characters/analyze",
        "/worlds/characters/synthesize",
        "/worlds/references/generate",
        "/worlds/intelligence/command",
        "/worlds/intelligence/director",
        "/worlds/intelligence/highlights",
    ],
)
def test_image_routes_authenticate_before_consuming_media(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    async def unread(_self: Any) -> AsyncIterator[bytes]:
        pytest.fail("Unauthorized inline image body must not be consumed")
        yield b""

    monkeypatch.setattr(Request, "stream", unread)
    response = client.post(path, content=b"{}", headers={"Content-Type": "application/json"})
    assert response.status_code == 401


def test_image_routes_reject_declared_body_before_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unread(_self: Any) -> AsyncIterator[bytes]:
        pytest.fail("Oversized inline image body must not be consumed")
        yield b""

    monkeypatch.setattr(Request, "stream", unread)
    response = client.post(
        "/worlds/characters/analyze",
        content=b"{}",
        headers={
            "Authorization": "Bearer private-test-token",
            "Content-Length": str(20 * 1024 * 1024 + 1),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413


def test_chunked_or_underdeclared_image_body_is_bounded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AuthenticatedBodyRoute, "body_limit", lambda _: 32)
    response = client.post(
        "/worlds/intelligence/highlights",
        content=(b"x" * 20 for _ in range(3)),
        headers={"Authorization": "Bearer private-test-token", "Content-Type": "application/json"},
    )
    assert response.status_code == 413
