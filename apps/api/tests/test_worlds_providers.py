"""Provider contracts verified without credentials, allocation or outbound calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException

from app.worlds.config import ModelProfile, WorldsSettings
from app.worlds.providers import LambdaProvider, ModalProvider, hardware
from app.worlds.router import Quote, provider_quote, usage
from app.worlds.store import Store
from tests import test_worlds as fixtures
from tests.test_worlds import connect

setup = fixtures.setup


def configured(tmp_path, **extra):
    script = tmp_path / "boot.yaml"
    script.write_text("#cloud-config\n# {{WORLD_MODEL_ID}} {{WORLD_GATEWAY_TOKEN_BASE64}}")
    return WorldsSettings(
        _env_file=None,
        data_dir=tmp_path,
        data_persistent=True,
        gateway_token="g" * 32,
        max_worker_hourly_cost=5,
        lambda_allow_provision=True,
        lambda_api_key="lambda-private",
        lambda_region="us-east-1",
        lambda_ssh_key_names=["worlds"],
        lambda_image_id="approved-image",
        lambda_bootstrap_file=script,
        lambda_gateway_template="https://{instance_id}.worlds.example.test",
        **extra,
    )


def inventory(cents=199):
    return {
        "data": {
            "gpu_1x_a100_sxm4": {
                "instance_type": {"price_cents_per_hour": cents, "specs": {"memory_gib": 240}},
                "regions_with_capacity_available": [{"name": "us-east-1"}],
            }
        }
    }


def test_lambda_live_quote_does_not_confuse_ram_or_actual_billing(tmp_path, monkeypatch):
    config = configured(tmp_path)
    monkeypatch.setattr(
        "app.worlds.providers.request", lambda *a, **kw: httpx.Response(200, json=inventory())
    )
    assert hardware("lambda", config)[0]["memoryGB"] is None
    result = provider_quote("lambda", Quote(durationMinutes=30), config)
    assert result["estimatedCostUSD"] == 0.995
    assert result["hourlyCost"] == 1.99
    assert result["billedCostUSD"] is None
    assert result["pricingSource"] == "provider-quote"


def test_lambda_launch_uses_only_approved_inputs(tmp_path, monkeypatch):
    config = configured(tmp_path)
    calls = []

    def request(method, url, **kw):
        calls.append((method, url, kw))
        if url.endswith("instance-types"):
            return httpx.Response(200, json=inventory())
        return httpx.Response(200, json={"data": {"instance_ids": ["instance-1"]}})

    monkeypatch.setattr("app.worlds.providers.request", request)
    result = LambdaProvider(config).create_worker(name="worlds-owned")
    assert result["gatewayUrl"] == "https://instance-1.worlds.example.test"
    assert result["estimatedHourlyCost"] == 1.99
    body = calls[-1][2]["payload"]
    assert body["image"] == {"id": "approved-image"}
    assert body["name"] == "worlds-owned"
    assert "lambda-private" not in str(body)
    assert body["user_data"].startswith("#cloud-config")


def test_lambda_preflight_rejects_over_budget_without_allocation(tmp_path, monkeypatch):
    call = Mock(return_value=httpx.Response(200, json=inventory(999)))
    monkeypatch.setattr("app.worlds.providers.request", call)
    with pytest.raises(HTTPException) as error:
        LambdaProvider(configured(tmp_path)).create_worker()
    assert error.value.status_code == 422
    assert call.call_count == 1


def test_lambda_pending_termination_never_claims_destroyed(tmp_path, monkeypatch):
    provider = LambdaProvider(configured(tmp_path))
    monkeypatch.setattr(provider, "status", lambda worker: {"status": "terminating"})
    call = Mock(return_value=httpx.Response(200, json={"data": {}}))
    monkeypatch.setattr(provider, "api", call)
    with pytest.raises(HTTPException) as error:
        provider.destroy({"managed": True, "providerId": "owned-id"})
    assert error.value.status_code == 503
    assert call.call_args.args[2] == {"instance_ids": ["owned-id"]}


def test_model_profile_cannot_inherit_other_model_runtime():
    config = WorldsSettings(
        _env_file=None,
        runpod_template_id="astronex-only",
        local_gateway_url="http://localhost:8789",
        model_profiles_json={"matrix-game-3": ModelProfile(modal_image="matrix")},
    )
    selected = config.for_model("matrix-game-3")
    assert selected.runpod_template_id is None
    assert selected.local_gateway_url is None
    assert selected.modal_image == "matrix"
    with pytest.raises(ValueError):
        config.for_model("unknown-model")


def test_model_session_mismatch_rejected_before_gateway(setup):
    client, _, calls = setup
    worker = connect(client)
    calls.clear()
    response = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": worker["id"], "modelId": "matrix-game-3", "prompt": "world"},
    )
    assert response.status_code == 409
    assert calls == []


def test_profile_gateway_persists_for_status(setup, monkeypatch):
    client, config, calls = setup
    config.model_profiles_json = {
        "matrix-game-3": ModelProfile(local_gateway_url="http://127.0.0.1:9999")
    }

    def request(method, url, **kw):
        calls.append(SimpleNamespace(url=httpx.URL(url)))
        return httpx.Response(200, json={"status": "ready", "models": [{"id": "matrix-game-3"}]})

    monkeypatch.setattr("app.worlds.providers.request", request)
    response = client.post(
        "/api/v1/worlds/workers", json={"provider": "local", "modelId": "matrix-game-3"}
    )
    assert response.status_code == 201
    worker = response.json()
    assert worker["modelId"] == "matrix-game-3"
    assert client.get(f"/api/v1/worlds/workers/{worker['id']}").status_code == 200
    assert calls[-1].url.port == 9999


def test_runpod_hardware_uses_bearer_and_secure_cloud(monkeypatch):
    calls = []

    def request(*args, **kw):
        calls.append((args, kw))
        return httpx.Response(
            200,
            json={
                "data": {
                    "gpuTypes": [
                        {
                            "id": "gpu",
                            "memoryInGb": 48,
                            "lowestPrice": {"stockStatus": "Low", "uninterruptablePrice": 0.75},
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr("app.worlds.providers.request", request)
    rows = hardware("runpod", WorldsSettings(_env_file=None, runpod_api_key="private"))
    assert rows[0]["available"] is True
    assert rows[0]["hourlyCost"] == 0.75
    assert "private" not in calls[0][0][1]
    assert calls[0][1]["token"] == "private"
    assert "secureCloud: true" in calls[0][1]["payload"]["query"]


class AsyncMethod:
    def __init__(self, value):
        self.value = value
        self.calls = []

    async def aio(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.value


def test_modal_sandbox_has_dedicated_credentials_and_hard_timeout(tmp_path, monkeypatch):
    config = configured(
        tmp_path,
        modal_allow_provision=True,
        modal_token_id="dedicated-id",
        modal_token_secret="dedicated-secret",
        modal_image="registry/model:approved",
        modal_hourly_cost=2,
    )
    box = SimpleNamespace(object_id="sb-owned")
    create = AsyncMethod(box)
    sdk = SimpleNamespace(
        Client=SimpleNamespace(from_credentials=AsyncMethod("client")),
        App=SimpleNamespace(lookup=AsyncMethod("app")),
        Sandbox=SimpleNamespace(create=create),
        Image=SimpleNamespace(from_registry=lambda image: image),
        Secret=SimpleNamespace(from_dict=lambda value: value),
    )
    provider = ModalProvider(config)
    monkeypatch.setattr(provider, "sdk", lambda: sdk)
    value = provider.create_worker(name="worlds-private-operation")
    assert value["providerId"] == "sb-owned"
    assert value["gatewayUrl"] is None  # tunnel discovery must not lose allocation identity
    assert value["pricingSource"] == "operator-estimate"
    args = create.calls[0][1]
    assert args["timeout"] == config.worker_max_lifetime_seconds
    assert args["encrypted_ports"] == [8789]
    assert "dedicated-secret" not in str(args)
    assert sdk.Client.from_credentials.calls[0][0] == ("dedicated-id", "dedicated-secret")


def test_usage_marks_elapsed_estimates_and_no_billed_total(tmp_path):
    db = Store(tmp_path)
    db.put(
        "worker",
        {
            "id": "owned",
            "provider": "lambda",
            "managed": True,
            "status": "destroyed",
            "createdAt": "2026-01-01T00:00:00+00:00",
            "endedAt": "2026-01-01T00:30:00+00:00",
            "estimatedHourlyCost": 2,
        },
    )
    value = usage(db)
    assert value["estimatedCostUSD"] == 1
    assert value["billedCostUSD"] is None
    assert value["workers"][0]["durationSeconds"] == 1800


def test_image_only_session_can_use_empty_prompt(setup):
    client, config, _ = setup
    worker = connect(client)
    db = Store(config.data_dir)
    db.patch("worker", worker["id"], {"capabilities": {"input": {"text": False}}})
    result = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": worker["id"], "modelId": "astronex-world", "prompt": ""},
    )
    assert result.status_code == 201


def test_image_only_worker_rejects_text_before_session_creation(setup):
    client, config, calls = setup
    worker = connect(client)
    Store(config.data_dir).patch(
        "worker", worker["id"], {"capabilities": {"input": {"text": False}}}
    )
    calls.clear()
    result = client.post(
        "/api/v1/worlds/sessions",
        json={"workerId": worker["id"], "modelId": "astronex-world", "prompt": "unsupported"},
    )
    assert result.status_code == 422
    assert calls == []


def test_lambda_recovery_only_adopts_exact_operation_name(tmp_path, monkeypatch):
    provider = LambdaProvider(configured(tmp_path))
    monkeypatch.setattr(
        provider,
        "api",
        lambda *a, **kw: httpx.Response(
            200,
            json={
                "data": [
                    {"id": "unrelated", "name": "not-owned", "status": "active"},
                    {"id": "owned", "name": "worlds-specific-operation", "status": "active"},
                ]
            },
        ),
    )
    assert provider.find_worker("worlds-specific-operation")["providerId"] == "owned"
    assert provider.find_worker("missing") is None


def test_modal_sdk_operations_share_one_loop_and_close_cleanly(tmp_path):
    import asyncio

    from app.worlds.providers import close_modal_transport

    class LoopMethod:
        async def aio(self):
            return asyncio.get_running_loop()

    provider = ModalProvider(configured(tmp_path))
    first = provider.call(LoopMethod())
    assert provider.call(LoopMethod()) is first
    close_modal_transport()
    assert first.is_closed()


def test_readiness_distinguishes_model_gateway_and_provisioning_modes(setup):
    client, config, _ = setup
    config.model_profiles_json = {
        "forgewm": ModelProfile(local_gateway_url="http://127.0.0.1:9999")
    }
    data = client.get("/api/v1/worlds/readiness").json()
    profile = next(item for item in data["modelProfiles"] if item["modelId"] == "forgewm")
    assert profile["gatewayProviders"] == ["local"]
    assert profile["provisioningProviders"] == []
    assert any(
        item["provider"] == "local" and item["modelId"] == "forgewm" for item in data["gateways"]
    )
