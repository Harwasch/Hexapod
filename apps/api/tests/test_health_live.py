import asyncio
from unittest.mock import Mock

import httpx
import pytest
from sqlalchemy import Engine

from app.api.deps import _db
from app.main import create_app


def test_liveness_never_opens_database(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = Mock(side_effect=AssertionError("liveness connected to the database"))
    monkeypatch.setattr(Engine, "connect", connect)
    app = create_app()

    def unavailable() -> None:
        raise AssertionError("liveness requested a database session")

    app.dependency_overrides[_db] = unavailable

    async def request() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get("/api/v1/health/live")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": app.version}
    connect.assert_not_called()
