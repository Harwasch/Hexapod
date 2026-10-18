from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_write_token
from app.worlds import intelligence


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("WORLDS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("WORLDS_LLM_MODEL", raising=False)
    app = FastAPI()
    app.include_router(intelligence.router)
    app.dependency_overrides[require_write_token] = lambda: None
    with TestClient(app) as client:
        yield client


def test_offline_optimizer_preserves_original_and_discloses_source(client: TestClient) -> None:
    response = client.post(
        "/worlds/prompts/optimize",
        json={
            "prompt": "forest at night",
            "modelId": "matrix",
            "capabilities": {"input": {"text": False}, "control": {"wasd": True}},
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["original"] == "forest at night"
    assert result["enhanced"].startswith("forest at night")
    assert result["source"] == "local-guide"
    assert "visual conditioning" in result["notes"][1]


def test_generated_controls_filter_invented_and_unsupported_actions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def completion(*_: Any) -> dict[str, Any]:
        return {
            "bindings": [
                {
                    "id": "1",
                    "key": "KeyW",
                    "label": "Forward",
                    "type": "native",
                    "action": "forward",
                },
                {"id": "2", "key": "KeyF", "label": "Fly", "type": "native", "action": "fly"},
                {"id": "3", "key": "KeyE", "label": "Talk", "type": "semantic", "prompt": "talk"},
                {"id": "4", "key": "KeyR", "label": "Rain", "type": "prompt", "prompt": "rain"},
                {
                    "id": "5",
                    "key": "KeyW",
                    "label": "Duplicate",
                    "type": "native",
                    "action": "forward",
                },
            ]
        }

    monkeypatch.setattr(intelligence, "_complete", completion)
    result = client.post(
        "/worlds/controls/generate",
        json={
            "prompt": "forest",
            "modelId": "matrix",
            "nativeActions": ["forward"],
            "capabilities": {"control": {"promptDuringRollout": True}},
        },
    ).json()
    assert [b["id"] for b in result["bindings"]] == ["1", "4"]
    assert result["bindings"][1]["experimental"] is True


def test_games_cannot_invent_live_prompt_support(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def completion(*_: Any) -> dict[str, Any]:
        return {
            "name": "Forest",
            "premise": "Explore",
            "objective": "Find the tower",
            "prompt": "Forest",
            "events": [{"atSeconds": 20, "prompt": "rain"}],
        }

    monkeypatch.setattr(intelligence, "_complete", completion)
    result = client.post("/worlds/games/create", json={"prompt": "forest", "modelId": "matrix"})
    assert result.status_code == 200
    assert result.json()["events"] == []


def test_invalid_llm_response_is_actionable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def completion(*_: Any) -> dict[str, Any]:
        return {"enhanced": []}

    monkeypatch.setattr(intelligence, "_complete", completion)
    response = client.post(
        "/worlds/prompts/optimize", json={"prompt": "forest", "modelId": "matrix"}
    )
    assert response.status_code == 502
    assert "invalid prompt" in response.json()["detail"]


def test_remote_plaintext_language_model_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLDS_LLM_BASE_URL", "http://remote.example/v1")
    monkeypatch.setenv("WORLDS_LLM_MODEL", "test")
    response = client.post(
        "/worlds/prompts/optimize", json={"prompt": "forest", "modelId": "matrix"}
    )
    assert response.status_code == 503
    assert "HTTPS" in response.json()["detail"]


def test_long_valid_prompt_and_live_worker_capabilities_remain_valid(client: TestClient) -> None:
    body = {
        "prompt": "a" * 8000,
        "modelId": "astronex-world",
        "capabilities": {"nativeActions": ["forward"], "resumeKind": "visual-checkpoint"},
    }
    for route in ["/worlds/prompts/optimize", "/worlds/games/create"]:
        response = client.post(route, json=body)
        assert response.status_code == 200
        assert response.json()["source"] == "local-guide"
