from __future__ import annotations

import asyncio
import base64
import io

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from app.api.deps import require_write_token
from app.worlds import intelligence, vision


def picture():
    buffer = io.BytesIO()
    image = Image.new("RGB", (12, 12), "red")
    exif = Image.Exif()
    exif[270] = "private local path and user metadata"
    image.save(buffer, "JPEG", exif=exif)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class AsyncRouteClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def execute():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://test"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(execute())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client(monkeypatch):
    for key in [
        "WORLDS_VISION_BASE_URL",
        "WORLDS_VISION_MODEL",
        "WORLDS_IMAGE_BASE_URL",
        "WORLDS_IMAGE_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    app = FastAPI()
    app.include_router(intelligence.router)

    async def authorized():
        pass

    app.dependency_overrides[require_write_token] = authorized
    return AsyncRouteClient(app)


def test_vision_media_strips_metadata_and_rejects_urls():
    cleaned = vision.clean_image(picture())
    with Image.open(io.BytesIO(cleaned)) as image:
        assert not image.getexif()
    for invalid in [
        "https://example.com/private.jpg",
        "data:image/svg+xml;base64,PHN2Zz4=",
        "data:image/jpeg;base64,bm90LWltYWdl",
    ]:
        with pytest.raises(Exception) as failure:
            vision.clean_image(invalid)
        assert failure.value.status_code == 422


def test_vision_media_is_bounded_before_decode():
    with pytest.raises(Exception) as failure:
        vision.clean_image("a" * 2_800_000)
    assert failure.value.status_code == 413


def test_unconfigured_vision_and_image_services_fail_honestly(client):
    status = client.get("/worlds/intelligence/status").json()
    assert status["visionConfigured"] is False
    assert status["imageConfigured"] is False
    response = client.post(
        "/worlds/intelligence/command",
        json={"prompt": "forest", "modelId": "test", "frame": picture(), "command": "pick that up"},
    )
    assert response.status_code == 503
    response = client.post(
        "/worlds/characters/synthesize", json={"description": "character", "images": [picture()]}
    )
    assert response.status_code == 503


def test_grounded_commands_filter_invented_native_actions(client, monkeypatch):
    async def complete(payload, instruction, images):
        assert payload["command"] == "fly"
        assert "frame" not in payload
        assert len(images) == 1
        return {
            "observation": "A rocky path",
            "explanation": "Flight",
            "action": {"type": "native", "action": "fly"},
        }

    monkeypatch.setattr(vision, "complete", complete)
    response = client.post(
        "/worlds/intelligence/command",
        json={
            "prompt": "forest",
            "modelId": "test",
            "frame": picture(),
            "command": "fly",
            "nativeActions": ["forward"],
        },
    )
    assert response.status_code == 200
    assert response.json()["action"] is None
    assert "not supported" in response.json()["explanation"]


def test_director_never_sends_native_player_movement_or_claims_unset_objective(client, monkeypatch):
    async def complete(*args):
        return {
            "observation": "A trail",
            "objective": {"progress": "appears-complete", "confidence": 1, "evidence": "A trail"},
            "event": {"type": "native", "action": "forward"},
            "reason": "Move player",
        }

    monkeypatch.setattr(vision, "complete", complete)
    response = client.post(
        "/worlds/intelligence/director",
        json={
            "prompt": "forest",
            "modelId": "test",
            "frame": picture(),
            "nativeActions": ["forward"],
        },
    )
    assert response.status_code == 200
    assert response.json()["event"] is None
    assert response.json()["objective"]["progress"] == "unknown"


def test_director_permits_only_declared_live_prompt_support(client, monkeypatch):
    async def complete(*args):
        return {
            "observation": "A trail",
            "objective": {
                "progress": "in-progress",
                "confidence": 0.6,
                "evidence": "The transmitter is visible",
            },
            "event": {"type": "prompt", "prompt": "Fog rolls across the trail"},
            "reason": "Atmosphere",
        }

    monkeypatch.setattr(vision, "complete", complete)
    body = {
        "prompt": "forest",
        "modelId": "test",
        "frame": picture(),
        "objective": "Reach the transmitter",
    }
    assert client.post("/worlds/intelligence/director", json=body).json()["event"] is None
    body["capabilities"] = {"control": {"promptSwitching": True}}
    assert (
        client.post("/worlds/intelligence/director", json=body).json()["event"]["prompt"]
        == "Fog rolls across the trail"
    )


def test_highlight_ranges_must_fit_replay(client, monkeypatch):
    async def complete(*args):
        return {
            "highlights": [
                {"startSeconds": 1, "endSeconds": 3, "title": "Light", "evidence": "Bright frame"},
                {
                    "startSeconds": 4,
                    "endSeconds": 50,
                    "title": "Invented",
                    "evidence": "Outside recording",
                },
            ]
        }

    monkeypatch.setattr(vision, "complete", complete)
    body = {
        "durationSeconds": 10,
        "samples": [{"atSeconds": 1, "image": picture()}, {"atSeconds": 8, "image": picture()}],
    }
    response = client.post("/worlds/intelligence/highlights", json=body)
    assert response.status_code == 200
    assert len(response.json()["highlights"]) == 1
    body["samples"].reverse()
    assert client.post("/worlds/intelligence/highlights", json=body).status_code == 422


def test_character_assessment_names_the_qualitative_metric(client, monkeypatch):
    async def complete(*args):
        return {
            "score": 0.7,
            "confidence": 0.6,
            "evidence": ["Coat colors match"],
            "differences": ["Silhouette changed"],
        }

    monkeypatch.setattr(vision, "complete", complete)
    response = client.post(
        "/worlds/characters/evaluate", json={"references": [picture()], "candidates": [picture()]}
    )
    assert response.status_code == 200
    assert response.json()["metric"] == "qualitative-appearance-consistency"
    assert "not a biometric" in response.json()["note"]


def test_untrusted_payload_cannot_set_ai_endpoint(client):
    response = client.post(
        "/worlds/characters/analyze",
        json={"images": [picture()], "endpoint": "https://attacker.example"},
    )
    assert response.status_code == 422


def test_image_synthesis_strips_reference_metadata_and_returns_real_pixels(client, monkeypatch):
    monkeypatch.setenv("WORLDS_IMAGE_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("WORLDS_IMAGE_MODEL", "configured-edit-model")
    monkeypatch.setenv("WORLDS_IMAGE_API_KEY", "server-only-secret")
    original_client = httpx.AsyncClient
    seen = []

    def image_service(request):
        seen.append(request)
        assert request.url.path == "/v1/images/edits"
        assert request.headers["authorization"] == "Bearer server-only-secret"
        assert b"private local path" not in request.content
        assert b"reference-1.jpg" in request.content
        return httpx.Response(200, json={"data": [{"b64_json": picture().split(",", 1)[1]}]})

    def network_client(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(image_service))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", network_client)
    response = client.post(
        "/worlds/characters/synthesize",
        json={"images": [picture()], "description": "A red-coated explorer"},
    )
    assert response.status_code == 200
    assert response.json()["image"].startswith("data:image/jpeg;base64,")
    assert response.json()["conditioningMethod"] == "reference-image-edit"
    assert "server-only-secret" not in response.text
    assert len(seen) == 1


def test_image_service_urls_are_not_fetched(client, monkeypatch):
    monkeypatch.setenv("WORLDS_IMAGE_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("WORLDS_IMAGE_MODEL", "configured-edit-model")
    original_client = httpx.AsyncClient
    seen = []

    def image_service(request):
        seen.append(request)
        return httpx.Response(
            200, json={"data": [{"url": "http://169.254.169.254/latest/meta-data"}]}
        )

    def network_client(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(image_service))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", network_client)
    response = client.post("/worlds/characters/synthesize", json={"images": [picture()]})
    assert response.status_code == 502
    assert len(seen) == 1


@pytest.mark.parametrize(
    "route,body,expected",
    [
        ("/worlds/references/generate", {"prompt": "A red forest"}, "text-to-image"),
        (
            "/worlds/references/generate",
            {"prompt": "A forest", "images": [picture(), picture()]},
            "reference-image-edit",
        ),
        ("/worlds/characters/synthesize", {"description": "An explorer in red"}, "text-to-image"),
    ],
)
def test_starting_image_and_text_only_character_use_real_image_endpoint(
    client, monkeypatch, route, body, expected
):
    monkeypatch.setenv("WORLDS_IMAGE_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("WORLDS_IMAGE_MODEL", "gpt-image-1")
    original_client = httpx.AsyncClient
    seen = []

    def image_service(request):
        seen.append(request)
        assert request.url.path == (
            "/v1/images/generations" if expected == "text-to-image" else "/v1/images/edits"
        )
        assert b"response_format" not in request.content
        assert b"private local path" not in request.content
        return httpx.Response(200, json={"data": [{"b64_json": picture().split(",", 1)[1]}]})

    def network_client(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(image_service))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", network_client)
    response = client.post(route, json=body)
    assert response.status_code == 200
    assert response.json()["conditioningMethod"] == expected
    assert len(seen) == 1
