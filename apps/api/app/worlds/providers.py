from __future__ import annotations

import asyncio
import math
import re
import time
from contextlib import nullcontext, suppress
from threading import Lock, Thread
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException

from app.worlds.config import WorldsSettings

RUNPOD_API = "https://rest.runpod.io/v1"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_POOL: httpx.Client | None = None
_POOL_USERS = 0
_POOL_LOCK = Lock()
_MODAL_LOCK = Lock()
_MODAL_LOOP: asyncio.AbstractEventLoop | None = None
_MODAL_THREAD: Thread | None = None
_MODAL_CLIENT: tuple[str, str, Any] | None = None


def modal_loop() -> asyncio.AbstractEventLoop:
    # SDK clients/channels must outlive an individual asyncio.run call and stay
    # on one event loop across FastAPI worker threads and the cleanup thread.
    global _MODAL_LOOP, _MODAL_THREAD
    with _MODAL_LOCK:
        if _MODAL_LOOP is None:
            loop = asyncio.new_event_loop()
            _MODAL_LOOP = loop
            _MODAL_THREAD = Thread(target=loop.run_forever, name="worlds-modal-sdk", daemon=True)
            _MODAL_THREAD.start()
        return _MODAL_LOOP


def close_modal_transport() -> None:
    global _MODAL_LOOP, _MODAL_THREAD, _MODAL_CLIENT
    with _MODAL_LOCK:
        loop, thread = _MODAL_LOOP, _MODAL_THREAD
        if loop is None:
            return

        async def cancel_pending() -> None:
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(cancel_pending(), loop).result(timeout=2)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=2)
        if not loop.is_running():
            loop.close()
        _MODAL_LOOP = None
        _MODAL_THREAD = None
        _MODAL_CLIENT = None


def start_transport() -> None:
    global _POOL, _POOL_USERS
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = httpx.Client(
                timeout=httpx.Timeout(30, connect=10),
                follow_redirects=False,
                limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
            )
        _POOL_USERS += 1


def stop_transport() -> None:
    global _POOL, _POOL_USERS
    with _POOL_LOCK:
        _POOL_USERS = max(0, _POOL_USERS - 1)
        if _POOL_USERS == 0 and _POOL is not None:
            _POOL.close()
            _POOL = None
            close_modal_transport()


def request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> httpx.Response:
    """Bounded, nonredirecting transport; upstream bodies never enter error messages."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    deadline = time.monotonic() + timeout
    try:
        with (
            (
                nullcontext(_POOL)
                if _POOL is not None
                else httpx.Client(
                    timeout=httpx.Timeout(timeout, connect=min(10, timeout)), follow_redirects=False
                )
            ) as client,
            client.stream(
                method,
                url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(timeout, connect=min(10, timeout)),
            ) as response,
        ):
            if response.status_code >= 300:
                status = response.status_code
                code = status if status in {404, 409, 422, 429, 501, 503} else 502
                raise HTTPException(
                    code,
                    f"Compute service returned HTTP {status}. "
                    "Check worker health and server configuration.",
                )
            chunks = []
            count = 0
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise HTTPException(504, "Compute service exceeded the response deadline.")
                count += len(chunk)
                if count > MAX_RESPONSE_BYTES:
                    raise HTTPException(502, "Compute service response exceeded the size limit.")
                chunks.append(chunk)
            return httpx.Response(
                response.status_code, headers=response.headers, content=b"".join(chunks)
            )
    except httpx.TimeoutException:
        raise HTTPException(
            504, "Compute service timed out; check status before retrying."
        ) from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise HTTPException(
            502, "Compute service is unreachable; check worker connection."
        ) from None


def response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("object expected")
        return value
    except (ValueError, TypeError):
        raise HTTPException(502, "Compute service returned an invalid JSON response.") from None


class ComputeProvider(Protocol):
    def create_worker(
        self, gpu: str | None = None, *, name: str = "worlds-worker"
    ) -> dict[str, Any]: ...
    def find_worker(self, name: str) -> dict[str, Any] | None: ...
    def status(self, worker: dict[str, Any]) -> dict[str, Any]: ...
    def stop(self, worker: dict[str, Any]) -> None: ...
    def destroy(self, worker: dict[str, Any]) -> None: ...


class GatewayProvider:
    """Connects to an explicitly configured native, Docker or hosted worker.

    This adapter does not claim to own the process/container or its billing lifecycle.
    """

    def __init__(self, name: str, settings: WorldsSettings) -> None:
        self.name = name
        self.settings = settings

    def create_worker(
        self, gpu: str | None = None, *, name: str = "worlds-worker"
    ) -> dict[str, Any]:
        try:
            url = self.settings.gateway_url(self.name)
        except ValueError:
            raise HTTPException(
                503, "Configured gateway URL is invalid; remote workers need HTTPS."
            ) from None
        if not url:
            raise HTTPException(
                503, f"Configure WORLD_{self.name.upper()}_GATEWAY_URL on the API server."
            )
        token = self.settings.gateway_token
        if self.name != "local" and not token:
            raise HTTPException(
                503, "WORLD_GATEWAY_TOKEN is required for hosted inference workers."
            )
        health = response_json(
            request(
                "GET",
                url.rstrip("/") + "/health",
                token=token.get_secret_value() if token else None,
            )
        )
        if not isinstance(health.get("status"), str) or health.get("status") not in {
            "ok",
            "ready",
            "healthy",
        }:
            raise HTTPException(
                503,
                "Worker gateway is reachable but its model is not ready. Check model installation.",
            )
        models = health.get("models")
        selected = (
            next(
                (
                    item
                    for item in models
                    if isinstance(item, dict) and item.get("id") == self.settings.model_id
                ),
                None,
            )
            if isinstance(models, list)
            else None
        )
        if isinstance(models, list) and selected is None:
            raise HTTPException(422, "Gateway does not advertise the selected model runtime.")
        capabilities = selected.get("capabilities", {}) if selected else {}
        return {
            "capabilities": capabilities if isinstance(capabilities, dict) else {},
            "gatewayUrl": url.rstrip("/"),
            "managed": False,
            "status": "ready" if health.get("status") in {"ok", "ready", "healthy"} else "starting",
            "estimatedHourlyCost": None,
        }

    def status(self, worker: dict[str, Any]) -> dict[str, Any]:
        config = self.settings.model_copy(
            update={
                f"{self.name}_gateway_url": worker.get("gatewayUrl"),
                "model_id": worker.get("modelId", "astronex-world"),
            }
        )
        result = GatewayProvider(self.name, config).create_worker()
        return {"status": result["status"], "capabilities": result.get("capabilities", {})}

    def find_worker(self, name: str) -> dict[str, Any] | None:
        raise HTTPException(409, "Externally managed gateways cannot reconcile allocations.")

    def stop(self, worker: dict[str, Any]) -> None:
        raise HTTPException(
            409,
            "This worker is externally managed. End sessions here, "
            "then stop compute with its owner or provider console.",
        )

    def destroy(self, worker: dict[str, Any]) -> None:
        self.stop(worker)


class RunPodProvider(GatewayProvider):
    """RunPod REST Pods API. Provision only an operator approved template.

    References: https://docs.runpod.io/api-reference/pods/POST/pods
    https://docs.runpod.io/api-reference/pods/POST/pods-podId-stop
    Recovery may list pods, but only an exact private operation name from the
    durable ledger can be adopted. Clients cannot supply external pod identifiers.
    """

    def __init__(self, settings: WorldsSettings) -> None:
        super().__init__("runpod", settings)

    def api(self, method: str, path: str, payload: dict[str, Any] | None = None) -> httpx.Response:
        key = self.settings.runpod_api_key
        if not key:
            raise HTTPException(503, "Configure WORLD_RUNPOD_API_KEY on the API server.")
        return request(method, RUNPOD_API + path, token=key.get_secret_value(), payload=payload)

    def create_worker(
        self, gpu: str | None = None, *, name: str = "worlds-worker"
    ) -> dict[str, Any]:
        if self.settings.runpod_gateway_url:
            return super().create_worker(gpu, name=name)
        if not self.settings.runpod_allow_provision or not self.settings.runpod_template_id:
            raise HTTPException(
                503,
                "RunPod provisioning is disabled. Configure a gateway URL or "
                "an approved WORLD_RUNPOD_TEMPLATE_ID and WORLD_RUNPOD_ALLOW_PROVISION.",
            )
        token = self.settings.gateway_token
        if not token:
            raise HTTPException(
                503, "WORLD_GATEWAY_TOKEN is required for hosted inference workers."
            )
        pod = response_json(
            self.api(
                "POST",
                "/pods",
                {
                    "name": name,
                    "templateId": self.settings.runpod_template_id,
                    "computeType": "GPU",
                    "cloudType": "SECURE",
                    "gpuCount": 1,
                    "gpuTypeIds": [gpu or self.settings.runpod_gpu_type],
                    "ports": [f"{self.settings.runpod_port}/http"],
                    "env": {
                        "WORLD_GATEWAY_TOKEN": token.get_secret_value(),
                        "PORT": str(self.settings.runpod_port),
                        "WORLD_MODEL_ID": self.settings.model_id,
                        "WORLD_BIND": "0.0.0.0",  # noqa: S104 - authenticated container proxy endpoint
                    },
                },
            )
        )
        pod_id = str(pod.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", pod_id):
            raise HTTPException(
                502,
                "RunPod did not return a valid pod ID. Check the provider console "
                "before retrying; a worker may have been created.",
            )
        return {
            "providerId": pod_id,
            "managed": True,
            "status": "starting",
            "gatewayUrl": f"https://{pod_id}-{self.settings.runpod_port}.proxy.runpod.net",
            "estimatedHourlyCost": hourly_cost(pod),
        }

    def billing(self, worker: dict[str, Any], start: str, end: str) -> list[dict[str, Any]]:
        """Verified public v1 OpenAPI: amount is USD, grouped by one owned Pod."""
        from datetime import UTC, datetime
        from decimal import Decimal, InvalidOperation

        query = urlencode(
            {
                "podId": worker["providerId"],
                "grouping": "podId",
                "bucketSize": "day",
                "startTime": start,
                "endTime": end,
            }
        )
        response = self.api("GET", "/billing/pods?" + query)
        try:
            data = response.json()
            if not isinstance(data, list) or len(data) > 64:
                raise ValueError("bounded array expected")
            rows = []
            seen = set()
            lower = (
                datetime.fromisoformat(start)
                .astimezone(UTC)
                .replace(hour=0, minute=0, second=0, microsecond=0)
            )
            upper = datetime.fromisoformat(end).astimezone(UTC)
            for item in data:
                if not isinstance(item, dict) or item.get("podId") != worker["providerId"]:
                    raise ValueError("billing target mismatch")
                if isinstance(item.get("amount"), bool):
                    raise ValueError("invalid amount")
                amount = Decimal(str(item.get("amount")))
                if not amount.is_finite() or abs(amount) > Decimal(1000000000):
                    raise ValueError("invalid amount")
                timestamp = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timezone required")
                timestamp = timestamp.astimezone(UTC)
                if timestamp < lower or timestamp > upper or timestamp in seen:
                    raise ValueError("unexpected or duplicate time bucket")
                seen.add(timestamp)
                row: dict[str, Any] = {
                    "time": timestamp.isoformat(),
                    "amount": float(amount),
                    "currency": "USD",
                }
                for key in ("timeBilledMs", "diskSpaceBilledGb"):
                    if key in item:
                        value = item[key]
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise ValueError("invalid billing metric")
                        row[key] = value
                rows.append(row)
            return rows
        except (ValueError, TypeError, KeyError, AttributeError, InvalidOperation):
            raise HTTPException(
                502, "RunPod returned invalid or mismatched billing data."
            ) from None

    def find_worker(self, name: str) -> dict[str, Any] | None:
        """Reconcile only the exact random name persisted before a create call."""
        response = self.api("GET", "/pods")
        try:
            pods = response.json()
        except ValueError:
            raise HTTPException(502, "RunPod reconciliation returned invalid data.") from None
        if not isinstance(pods, list):
            raise HTTPException(502, "RunPod reconciliation returned invalid data.")
        matches = [pod for pod in pods if isinstance(pod, dict) and pod.get("name") == name]
        if len(matches) > 1:
            raise HTTPException(
                409, "Duplicate operation names require provider-console reconciliation."
            )
        if not matches:
            return None
        pod = matches[0]
        identity = str(pod.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identity):
            raise HTTPException(502, "RunPod reconciliation returned an invalid ID.")
        return {
            "providerId": identity,
            "managed": True,
            "status": "starting",
            "gatewayUrl": f"https://{identity}-{self.settings.runpod_port}.proxy.runpod.net",
            "estimatedHourlyCost": hourly_cost(pod),
        }

    def status(self, worker: dict[str, Any]) -> dict[str, Any]:
        if not worker.get("managed"):
            return super().status(worker)
        pod = response_json(self.api("GET", f"/pods/{worker['providerId']}"))
        desired = pod.get("desiredStatus", "UNKNOWN")
        if not isinstance(desired, str):
            raise HTTPException(502, "RunPod returned an invalid worker status.")
        return {
            "status": {"RUNNING": "starting", "EXITED": "stopped", "TERMINATED": "destroyed"}.get(
                desired, "unknown"
            ),
            "estimatedHourlyCost": hourly_cost(pod),
        }

    def stop(self, worker: dict[str, Any]) -> None:
        if not worker.get("managed"):
            return super().stop(worker)
        self.api("POST", f"/pods/{worker['providerId']}/stop")

    def destroy(self, worker: dict[str, Any]) -> None:
        if not worker.get("managed"):
            return super().destroy(worker)
        try:
            self.api("DELETE", f"/pods/{worker['providerId']}")
        except HTTPException as exc:
            if exc.status_code != 404:
                raise


def provider(name: str, settings: WorldsSettings) -> ComputeProvider:
    if name == "runpod":
        return RunPodProvider(settings)
    if name == "lambda":
        return LambdaProvider(settings)
    if name == "modal":
        return ModalProvider(settings)
    if name == "local":
        return GatewayProvider(name, settings)
    raise HTTPException(422, "Unknown compute provider.")


def hourly_cost(pod: dict[str, Any]) -> float | None:
    if isinstance(pod.get("costPerHr"), bool):
        return None
    try:
        value = float(pod.get("costPerHr", "nan"))
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def hardware(name: str, config: WorldsSettings) -> list[dict[str, Any]]:
    """Live read-only inventory. Rates are quotes, never billed usage."""
    rows: list[dict[str, Any]] = []
    if name == "runpod" and config.runpod_api_key:
        data = response_json(
            request(
                "POST",
                "https://api.runpod.io/graphql",
                token=config.runpod_api_key.get_secret_value(),
                payload={
                    "query": (
                        "query { gpuTypes { id displayName memoryInGb "
                        "lowestPrice(input: {gpuCount: 1, secureCloud: true}) "
                        "{ stockStatus uninterruptablePrice } } }"
                    )
                },
            )
        )
        if data.get("errors") or not isinstance(data.get("data"), dict):
            raise HTTPException(502, "RunPod inventory query failed.")
        for item in data["data"].get("gpuTypes", []):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            price = item.get("lowestPrice") or {}
            if not isinstance(price, dict):
                continue
            rows.append(
                {
                    "id": item["id"],
                    "name": item.get("displayName", item["id"]),
                    "memoryGB": number(item.get("memoryInGb")),
                    "regions": [],
                    "available": price.get("stockStatus") != "None"
                    if price.get("stockStatus") in {"High", "Medium", "Low", "None"}
                    else None,
                    "hourlyCost": number(price.get("uninterruptablePrice")),
                    "pricingSource": "provider-quote",
                }
            )
    elif name == "lambda" and config.lambda_api_key:
        lambda_data = response_json(LambdaProvider(config).api("GET", "/instance-types")).get(
            "data"
        )
        if not isinstance(lambda_data, dict):
            raise HTTPException(502, "Lambda inventory returned invalid data.")
        for key, value in lambda_data.items():
            if not isinstance(value, dict) or not isinstance(value.get("instance_type"), dict):
                continue
            item = value["instance_type"]
            regions = [
                region["name"]
                for region in value.get("regions_with_capacity_available", [])
                if isinstance(region, dict) and isinstance(region.get("name"), str)
            ]
            cents = number(item.get("price_cents_per_hour"))
            rows.append(
                {
                    "id": key,
                    "name": item.get("description", key),
                    "memoryGB": None,
                    "regions": regions,
                    "available": bool(regions),
                    "hourlyCost": cents / 100 if cents is not None else None,
                    "pricingSource": "provider-quote",
                }
            )
    else:
        rows.append(
            {
                "id": config.gpu_type(name) or "configured-gateway",
                "name": config.gpu_type(name) or "Configured gateway",
                "memoryGB": None,
                "regions": [],
                "available": None,
                "hourlyCost": config.modal_hourly_cost if name == "modal" else None,
                "pricingSource": "operator-estimate"
                if name == "modal" and config.modal_hourly_cost is not None
                else "unavailable",
            }
        )
    return rows


class LambdaProvider(GatewayProvider):
    """Lambda Cloud v1; bootstrapping/TLS are explicitly approved operator inputs."""

    def __init__(self, settings: WorldsSettings) -> None:
        super().__init__("lambda", settings)

    def api(self, method: str, path: str, payload: dict[str, Any] | None = None) -> httpx.Response:
        key = self.settings.lambda_api_key
        if key is None:
            raise HTTPException(503, "Configure dedicated WORLD_LAMBDA_API_KEY.")
        return request(
            method,
            "https://cloud.lambda.ai/api/v1" + path,
            token=key.get_secret_value(),
            payload=payload,
        )

    def record(self, item: dict[str, Any]) -> dict[str, Any]:
        identity = str(item.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identity):
            raise HTTPException(502, "Lambda returned an invalid instance ID.")
        url = (self.settings.lambda_gateway_template or "").replace("{instance_id}", identity)
        from app.worlds.config import validate_url

        try:
            validate_url(url)
        except ValueError:
            raise HTTPException(
                503, "Lambda gateway template must produce a valid HTTPS URL."
            ) from None
        kind = item.get("instance_type") or {}
        cents = number(kind.get("price_cents_per_hour")) if isinstance(kind, dict) else None
        return {
            "providerId": identity,
            "managed": True,
            "gatewayUrl": url.rstrip("/"),
            "status": {
                "active": "starting",
                "booting": "starting",
                "terminated": "destroyed",
                "preempted": "destroyed",
                "terminating": "terminating",
                "unhealthy": "unknown",
            }.get(str(item.get("status")), "unknown"),
            "estimatedHourlyCost": cents / 100 if cents is not None else None,
            "pricingSource": "provider-quote",
        }

    def create_worker(
        self, gpu: str | None = None, *, name: str = "worlds-worker"
    ) -> dict[str, Any]:
        if self.settings.lambda_gateway_url:
            return super().create_worker(gpu, name=name)
        config = self.settings
        if not config.lambda_allow_provision or config.provisioning_problem("lambda"):
            raise HTTPException(503, "Lambda provisioning is disabled or incomplete.")
        # All preflight failures are definite (422); reserve no uncertain billable intent.
        try:
            choices = hardware("lambda", config)
        except HTTPException:
            raise HTTPException(
                422, "Lambda inventory preflight failed; no allocation was requested."
            ) from None
        chosen = next(
            (row for row in choices if row["id"] == (gpu or config.lambda_instance_type)), None
        )
        if chosen is None or config.lambda_region not in chosen["regions"]:
            raise HTTPException(
                422, "Approved Lambda hardware is not available in the selected region."
            )
        rate = chosen["hourlyCost"]
        if (
            rate is None
            or config.max_worker_hourly_cost is None
            or rate > config.max_worker_hourly_cost
        ):
            raise HTTPException(422, "Lambda quoted rate is unknown or exceeds the cost ceiling.")
        try:
            assert config.lambda_bootstrap_file is not None
            script = config.lambda_bootstrap_file.read_text()
            assert config.gateway_token is not None
            # Base64 allows safe interpolation into operator-owned cloud-init without shell injection.
            import base64

            script = script.replace(
                "{{WORLD_GATEWAY_TOKEN_BASE64}}",
                base64.b64encode(config.gateway_token.get_secret_value().encode()).decode(),
            )
            script = script.replace("{{WORLD_MODEL_ID}}", config.model_id)
            if len(script.encode()) > 1_000_000:
                raise ValueError("bootstrap too large")
            self.record({"id": "preflight", "status": "booting"})
        except (OSError, ValueError, HTTPException):
            raise HTTPException(
                422, "Lambda approved bootstrap or HTTPS routing configuration is invalid."
            ) from None
        result = response_json(
            self.api(
                "POST",
                "/instance-operations/launch",
                {
                    "region_name": config.lambda_region,
                    "instance_type_name": gpu or config.lambda_instance_type,
                    "ssh_key_names": config.lambda_ssh_key_names,
                    "image": {"id": config.lambda_image_id},
                    "name": name,
                    "user_data": script,
                },
            )
        ).get("data", {})
        ids = result.get("instance_ids") if isinstance(result, dict) else None
        if not isinstance(ids, list) or len(ids) != 1:
            raise HTTPException(
                502, "Lambda allocation outcome is uncertain; reconciliation is required."
            )
        value = self.record({"id": ids[0], "status": "booting"})
        value["estimatedHourlyCost"] = rate
        return value

    def find_worker(self, name: str) -> dict[str, Any] | None:
        data = response_json(self.api("GET", "/instances")).get("data")
        if not isinstance(data, list):
            raise HTTPException(502, "Lambda reconciliation returned invalid data.")
        found = [item for item in data if isinstance(item, dict) and item.get("name") == name]
        if len(found) > 1:
            raise HTTPException(
                409, "Duplicate Lambda operation names require manual reconciliation."
            )
        return self.record(found[0]) if found else None

    def status(self, worker: dict[str, Any]) -> dict[str, Any]:
        if not worker.get("managed"):
            return super().status(worker)
        data = response_json(self.api("GET", f"/instances/{worker['providerId']}")).get("data")
        if not isinstance(data, dict):
            raise HTTPException(502, "Lambda status returned invalid data.")
        value = self.record(data)
        if value["estimatedHourlyCost"] is None:
            value["estimatedHourlyCost"] = worker.get("estimatedHourlyCost")
        return value

    def stop(self, worker: dict[str, Any]) -> None:
        raise HTTPException(409, "Lambda does not suspend billing. Terminate this worker instead.")

    def destroy(self, worker: dict[str, Any]) -> None:
        if not worker.get("managed"):
            return super().destroy(worker)
        try:
            if self.status(worker)["status"] == "destroyed":
                return
            self.api(
                "POST", "/instance-operations/terminate", {"instance_ids": [worker["providerId"]]}
            )
            if self.status(worker)["status"] != "destroyed":
                raise HTTPException(503, "Lambda termination is pending; cleanup will retry.")
        except HTTPException as exc:
            if exc.status_code != 404:
                raise


class ModalProvider(GatewayProvider):
    """Dedicated Modal Sandbox app; never imports the Earth deployment or its credentials."""

    def __init__(self, settings: WorldsSettings) -> None:
        super().__init__("modal", settings)

    def sdk(self) -> Any:
        import importlib

        try:
            return importlib.import_module("modal")
        except ImportError:
            raise HTTPException(
                503, "Install the API modal optional dependency to use standalone Sandboxes."
            ) from None

    def call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        async def invoke() -> Any:
            return await asyncio.wait_for(function.aio(*args, **kwargs), timeout=15)

        future = asyncio.run_coroutine_threadsafe(invoke(), modal_loop())
        try:
            return future.result(timeout=16)
        except Exception:
            future.cancel()
            raise HTTPException(
                502, "Modal operation failed or timed out; check status before retrying."
            ) from None

    def client(self) -> Any:
        config = self.settings
        if not config.modal_token_id or not config.modal_token_secret:
            raise HTTPException(
                503, "Configure dedicated WORLD_MODAL_TOKEN_ID and WORLD_MODAL_TOKEN_SECRET."
            )
        global _MODAL_CLIENT
        identity = config.modal_token_id.get_secret_value()
        secret = config.modal_token_secret.get_secret_value()
        # Configuration is fixed for a lifespan. Do not inherit Modal's global
        # environment credential cache or share a client with another product.
        if _MODAL_CLIENT is not None and _MODAL_CLIENT[:2] == (identity, secret):
            return _MODAL_CLIENT[2]
        value = self.call(self.sdk().Client.from_credentials, identity, secret)
        _MODAL_CLIENT = (identity, secret, value)
        return value

    def sandbox(self, worker: dict[str, Any]) -> Any:
        return self.call(self.sdk().Sandbox.from_id, worker["providerId"], client=self.client())

    def record(self, sandbox: Any) -> dict[str, Any]:
        return {
            "providerId": sandbox.object_id,
            "gatewayUrl": None,
            "managed": True,
            "status": "starting",
            "estimatedHourlyCost": self.settings.modal_hourly_cost,
            "pricingSource": "operator-estimate",
        }

    def create_worker(
        self, gpu: str | None = None, *, name: str = "worlds-worker"
    ) -> dict[str, Any]:
        if self.settings.modal_gateway_url:
            return super().create_worker(gpu, name=name)
        config = self.settings
        if not config.modal_allow_provision or config.provisioning_problem("modal"):
            raise HTTPException(503, "Standalone Modal provisioning is disabled or incomplete.")
        try:
            sdk = self.sdk()
            client = self.client()
            app = self.call(
                sdk.App.lookup, config.modal_app_name, create_if_missing=True, client=client
            )
            image = sdk.Image.from_registry(config.modal_image)
        except Exception:
            raise HTTPException(
                422, "Modal preflight failed; no Sandbox allocation was requested."
            ) from None
        assert config.gateway_token is not None
        box = self.call(
            sdk.Sandbox.create,
            "/opt/gateway-venv/bin/python",
            "/opt/worlds/gateway.py",
            app=app,
            name=name,
            image=image,
            gpu=gpu or config.modal_gpu_type,
            workdir="/opt/worlds",
            env={
                "PORT": str(config.runpod_port),
                "WORLD_BIND": "0.0.0.0",  # noqa: S104 - authenticated sandbox tunnel
                "WORLD_MODEL_ID": config.model_id,
            },
            secrets=[
                sdk.Secret.from_dict(
                    {"WORLD_GATEWAY_TOKEN": config.gateway_token.get_secret_value()}
                )
            ],
            encrypted_ports=[config.runpod_port],
            timeout=config.worker_max_lifetime_seconds,
            client=client,
        )
        return self.record(box)

    def find_worker(self, name: str) -> dict[str, Any] | None:
        box = self.call(
            self.sdk().Sandbox.from_name, self.settings.modal_app_name, name, client=self.client()
        )
        # No not-found guessing: unresolved allocations continue reserving fleet capacity.
        return self.record(box)

    def status(self, worker: dict[str, Any]) -> dict[str, Any]:
        if not worker.get("managed"):
            return super().status(worker)
        box = self.sandbox(worker)
        if self.call(box.poll) is not None:
            return {"status": "destroyed"}
        value = {"status": "starting", "estimatedHourlyCost": worker.get("estimatedHourlyCost")}
        if not worker.get("gatewayUrl"):
            tunnels = self.call(box.tunnels, timeout=15)
            tunnel = tunnels.get(self.settings.runpod_port)
            if tunnel is not None:
                from app.worlds.config import validate_url

                try:
                    validate_url(tunnel.url)
                except ValueError:
                    raise HTTPException(502, "Modal returned an invalid tunnel URL.") from None
                value["gatewayUrl"] = tunnel.url.rstrip("/")
        return value

    def stop(self, worker: dict[str, Any]) -> None:
        raise HTTPException(409, "Modal Sandboxes cannot suspend. Terminate this worker instead.")

    def destroy(self, worker: dict[str, Any]) -> None:
        if not worker.get("managed"):
            return super().destroy(worker)
        box = self.sandbox(worker)
        self.call(box.terminate)
        if self.call(box.poll) is None:
            raise HTTPException(503, "Modal termination is pending; cleanup will retry.")
