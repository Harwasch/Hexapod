"""Shared typed fixtures for Worlds manager contract tests."""

from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.worlds.config import WorldsSettings
from app.worlds.lifecycle import Lifecycle
from app.worlds.store import Store

WorldsSetup = tuple[TestClient, WorldsSettings, list[httpx.Request]]
LifecycleSetup = tuple[Lifecycle, Store, list[tuple[str, str]]]


def require_record(value: dict[str, Any] | None) -> dict[str, Any]:
    assert value is not None, "Expected persisted Worlds record"
    return value
