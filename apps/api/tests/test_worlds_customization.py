import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.worlds import customization


def test_configured_proxy_passes_payload_and_keeps_token_server_side(monkeypatch):
    monkeypatch.setenv("WORLD_CUSTOMIZATION_GATEWAY_URL", "https://training.example")
    monkeypatch.setenv("WORLD_CUSTOMIZATION_GATEWAY_TOKEN", "private-gateway-token")
    calls = []

    def request(method, url, *, token, payload, timeout):
        calls.append((method, url, token, payload, timeout))
        return httpx.Response(200, json={"safe": True})

    monkeypatch.setattr(customization, "request", request)
    assert customization.forward("POST", "/jobs", {"profileId": "approved"}) == {"safe": True}
    assert calls == [
        (
            "POST",
            "https://training.example/jobs",
            "private-gateway-token",
            {"profileId": "approved"},
            30,
        )
    ]


def test_customization_rejects_arbitrary_paths_and_unconfirmed_training():
    with pytest.raises(ValidationError):
        customization.Prepare(profileId="../../private")
    with pytest.raises(ValidationError):
        customization.Prepare(profileId="safe", source="/arbitrary/source")
    with pytest.raises(ValidationError):
        customization.Run(confirmTraining=False)
    with pytest.raises(HTTPException):
        customization.job_id("../escape")


def test_malformed_worker_response_fails_closed():
    with pytest.raises(HTTPException) as failure:
        customization.parsed_job({"id": "anything", "status": "completed"})
    assert failure.value.status_code == 502
