from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.api.deps import RequireWriteToken
from app.worlds.config import WorldsSettings
from app.worlds.lifecycle import Lifecycle
from app.worlds.providers import RunPodProvider, hardware, provider, request, response_json
from app.worlds.request_limits import AuthenticatedBodyRoute
from app.worlds.store import Store


class BoundedWorldsRoute(AuthenticatedBodyRoute):
    def body_limit(self) -> int:
        return 7 * 1024 * 1024 if self.path.endswith("/sessions") else 128 * 1024


router = APIRouter(
    prefix="/worlds",
    tags=["worlds"],
    dependencies=[RequireWriteToken],
    route_class=BoundedWorldsRoute,
)


def settings(context: Request) -> WorldsSettings:
    return getattr(context.app.state, "worlds_settings", None) or WorldsSettings.load()


Settings = Annotated[WorldsSettings, Depends(settings)]


def store(config: Settings) -> Store:
    return Store(config.data_dir)


Database = Annotated[Store, Depends(store)]


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerCreate(Body):
    provider: Literal["runpod", "local", "modal", "lambda"] = "runpod"
    model_id: str = Field(
        default="astronex-world", alias="modelId", pattern=r"^[a-zA-Z0-9_-]{1,100}$"
    )
    gpu_type_id: str | None = Field(default=None, alias="gpuTypeId", min_length=1, max_length=100)


class SessionCreate(Body):
    worker_id: str = Field(alias="workerId", min_length=1, max_length=100)
    model_id: str = Field(
        alias="modelId", min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    prompt: str = Field(default="", max_length=16000)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    quality: Literal["quality", "balanced", "low-latency", "max-fps"] = "balanced"
    resolution: str | None = Field(default=None, max_length=30, pattern=r"^\d{2,4}x\d{2,4}$")
    inputs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_inputs(self) -> SessionCreate:
        if len(json.dumps(self.inputs)) > 6 * 1024 * 1024:
            raise ValueError("Input media exceeds 6 MB; resize or shorten the input first")
        allowed = {"images", "video", "characterDescription"}
        if set(self.inputs) - allowed:
            raise ValueError("Unsupported creation input")
        images = self.inputs.get("images", [])
        if not isinstance(images, list) or len(images) > 8:
            raise ValueError("At most eight reference images are supported")
        for value in images:
            if not isinstance(value, str) or not re.fullmatch(
                r"data:image/(?:jpeg|png|webp);base64,[A-Za-z0-9+/=\s]+", value
            ):
                raise ValueError("Images must be inline JPEG, PNG or WebP data URLs")
        video = self.inputs.get("video")
        if video is not None and (
            not isinstance(video, str)
            or not re.fullmatch(r"data:video/(?:mp4|webm);base64,[A-Za-z0-9+/=\s]+", video)
        ):
            raise ValueError("Video must be an inline MP4 or WebM data URL")
        description = self.inputs.get("characterDescription", "")
        if not isinstance(description, str) or len(description) > 8000:
            raise ValueError("Character description is too long")
        return self


class Action(Body):
    type: Literal["native", "prompt", "semantic", "pause", "resume"]
    action: str | None = Field(default=None, max_length=500)
    prompt: str | None = Field(default=None, max_length=16000)
    values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded(self) -> Action:
        if len(json.dumps(self.values)) > 16000:
            raise ValueError("Action payload exceeds the size limit")
        if self.type in {"prompt", "semantic"} and not (self.prompt or self.action):
            raise ValueError("A prompt or semantic action is required")
        if self.type == "native" and not self.action and not self.values:
            raise ValueError("A native action or control vector is required")
        return self


class Heartbeat(Body):
    active: bool = True


class Offer(Body):
    type: Literal["offer"]
    sdp: str = Field(min_length=1, max_length=100000)


def now() -> str:
    return datetime.now(UTC).isoformat()


def get_record(db: Store, kind: str, identity: str) -> dict[str, Any]:
    item = db.get(kind, identity)
    if item is None:
        raise HTTPException(404, f"Worlds {kind} not found.")
    return item


def public_worker(worker: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in worker.items()
        if key not in {"gatewayUrl", "providerId", "operationName", "cleanupUntil"}
    }


def gateway(
    config: WorldsSettings,
    worker: dict[str, Any],
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> httpx.Response:
    if worker["status"] in {
        "stopped",
        "destroyed",
        "detached",
        "terminating",
        "stopping",
        "detaching",
    }:
        raise HTTPException(409, "Worker is no longer active. Connect a new worker.")
    token = config.gateway_token
    return request(
        method,
        worker["gatewayUrl"] + path,
        token=token.get_secret_value() if token else None,
        payload=payload,
    )


def session_context(db: Store, identity: str) -> tuple[dict[str, Any], dict[str, Any]]:
    session = get_record(db, "session", identity)
    return session, get_record(db, "worker", session["workerId"])


def active_context(db: Store, identity: str) -> tuple[dict[str, Any], dict[str, Any]]:
    session, worker = session_context(db, identity)
    if (
        session["status"] in {"stopped", "terminating"}
        or session.get("leaseExpiresAt", 0) <= time.time()
    ):
        raise HTTPException(409, "Session lease has ended. Start a new session.")
    return session, worker


@router.get("/providers")
def providers(config: Settings, context: Request) -> dict[str, Any]:
    profiles = [
        config.for_model(identity)
        for identity in dict.fromkeys([config.model_id, *config.model_profiles_json])
    ]
    result = []
    for identity, name in [
        ("runpod", "RunPod"),
        ("local", "Local GPU"),
        ("modal", "Modal"),
        ("lambda", "Lambda"),
    ]:
        configured = any(
            bool(getattr(profile, f"{identity}_gateway_url", None))
            and (identity == "local" or bool(config.gateway_token))
            for profile in profiles
        )
        provision = bool(
            any(
                profile.managed(identity) and profile.provisioning_problem(identity) is None
                for profile in profiles
            )
            and getattr(getattr(context.app.state, "worlds_lifecycle", None), "running", False)
            and context.app.state.settings.api_write_token
        )
        result.append(
            {
                "id": identity,
                "name": name,
                "configured": configured or provision,
                "canProvision": provision,
                "message": "Gateway configured; health checked when connecting."
                if configured
                else "Approved runtime configured; starting compute incurs charges."
                if provision
                else "Configure an approved runtime or gateway on the API server.",
                "gpuTypeId": config.gpu_type(identity),
            }
        )
    for identity, name in [
        ("coreweave", "CoreWeave"),
        ("aws", "AWS"),
        ("gcp", "GCP"),
        ("azure", "Azure"),
    ]:
        result.append(
            {
                "id": identity,
                "name": name,
                "configured": False,
                "canProvision": False,
                "message": "Provider adapter not installed. Connect via a compatible gateway.",
            }
        )
    return {
        "providers": result,
        "primaryProvider": "runpod",
        "modelProfiles": [
            {
                "modelId": identity,
                "gatewayProviders": [
                    name
                    for name in ("runpod", "local", "modal", "lambda")
                    if getattr(config.for_model(identity), f"{name}_gateway_url", None)
                ],
                "provisioningProviders": [
                    name
                    for name in ("runpod", "local", "modal", "lambda")
                    if config.for_model(identity).managed(name)
                    and config.for_model(identity).provisioning_problem(name) is None
                ],
                "providers": [
                    name
                    for name in ("runpod", "local", "modal", "lambda")
                    if getattr(config.for_model(identity), f"{name}_gateway_url", None)
                    or (
                        config.for_model(identity).managed(name)
                        and config.for_model(identity).provisioning_problem(name) is None
                    )
                ],
            }
            for identity in dict.fromkeys([config.model_id, *config.model_profiles_json])
        ],
    }


@router.get("/catalog")
def catalog(config: Settings, context: Request) -> dict[str, Any]:
    """Live capability inspection shares the bounded readiness probes."""
    report = readiness(config, context)
    return {key: report[key] for key in ("gateways", "providers", "primaryProvider")}


@router.post("/workers", status_code=201)
def create_worker(
    body: WorkerCreate, config: Settings, db: Database, context: Request
) -> dict[str, Any]:
    # Persist intent before the irreversible provider call. A timeout leaves an explicit
    # unknown operation, preventing clients from assuming no billable worker exists.
    try:
        config = config.for_model(body.model_id)
    except ValueError:
        raise HTTPException(422, "This model has no operator-approved runtime profile.") from None
    managed = config.managed(body.provider)
    if managed:
        problem = config.provisioning_problem(body.provider)
        if problem:
            raise HTTPException(503, problem)
        if not context.app.state.settings.api_write_token or not getattr(
            getattr(context.app.state, "worlds_lifecycle", None), "running", False
        ):
            raise HTTPException(
                503,
                "Managed provisioning requires authenticated access and a healthy lifecycle reaper.",
            )
        if body.gpu_type_id is not None and body.gpu_type_id != config.gpu_type(body.provider):
            raise HTTPException(422, "This GPU is not in the operator-approved configuration.")
    timestamp = time.time()
    worker: dict[str, Any] = {
        "id": str(uuid4()),
        "provider": body.provider,
        "modelId": body.model_id,
        "createdAt": now(),
        "status": "provisioning",
        "estimatedHourlyCost": None,
        "managed": managed,
        "hardDeadline": timestamp + config.worker_max_lifetime_seconds,
        "idleDeadline": timestamp
        + (config.worker_startup_seconds if managed else config.worker_idle_seconds),
    }
    if managed:
        worker["operationName"] = "worlds-" + worker["id"]
        if not db.reserve_worker(worker, config.max_managed_workers):
            raise HTTPException(
                409,
                (
                    "Managed worker capacity is reserved. Reuse or terminate an existing worker; "
                    "unresolved creations also consume capacity."
                ),
            )
    else:
        db.put("worker", worker)
    try:
        worker.update(
            provider(body.provider, config).create_worker(
                body.gpu_type_id, name=worker.get("operationName", "worlds-worker")
            )
        )
    except HTTPException as exc:
        may_have_provisioned = managed and exc.status_code not in {409, 422, 429}
        worker["status"] = "unknown" if may_have_provisioned else "error"
        worker["message"] = str(exc.detail)
        if managed and not may_have_provisioned:
            worker["managed"] = False
        db.patch("worker", worker["id"], worker)
        raise
    worker = db.patch("worker", worker["id"], worker)
    if managed:
        cost = worker.get("estimatedHourlyCost")
        if (
            not isinstance(cost, (int, float))
            or config.max_worker_hourly_cost is None
            or cost > config.max_worker_hourly_cost
        ):
            try:
                Lifecycle(config, db).terminate_worker(worker, "cost-limit")
            except HTTPException:
                raise HTTPException(
                    502,
                    "The worker price exceeded policy or was unknown; termination is pending and will retry.",
                ) from None
            raise HTTPException(
                409,
                "Worker price exceeded policy or was unavailable. The new worker was terminated.",
            )
    return public_worker(worker)


@router.get("/workers")
def list_workers(db: Database) -> dict[str, Any]:
    return {"workers": [public_worker(item) for item in db.list("worker")]}


@router.get("/workers/{identity}")
def worker_status(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    worker = get_record(db, "worker", identity)
    if worker["status"] in {
        "destroyed",
        "detached",
        "error",
        "unknown",
        "terminating",
        "stopping",
        "detaching",
    }:
        return public_worker(worker)
    worker.update(provider(worker["provider"], config).status(worker))
    if worker["status"] == "starting" and worker.get("gatewayUrl"):
        try:
            health = response_json(gateway(config, worker, "GET", "/health"))
            if isinstance(health.get("status"), str) and health["status"] in {
                "ok",
                "ready",
                "healthy",
            }:
                models = health.get("models", [])
                selected = (
                    next(
                        (
                            item
                            for item in models
                            if isinstance(item, dict)
                            and item.get("id") == worker.get("modelId", "astronex-world")
                        ),
                        None,
                    )
                    if isinstance(models, list)
                    else None
                )
                if selected is not None and selected.get("status", "ready") == "ready":
                    worker["status"] = "ready"
                    caps = selected.get("capabilities", {})
                    worker["capabilities"] = caps if isinstance(caps, dict) else {}
        except HTTPException:
            pass  # a provisioned pod is often booting/model-loading
    worker = db.patch(
        "worker",
        identity,
        {
            key: worker[key]
            for key in ("status", "estimatedHourlyCost", "gatewayUrl", "capabilities")
            if key in worker
        },
    )
    if worker["status"] == "destroyed":
        db.end_worker_sessions(identity, now())
        worker = db.patch("worker", identity, {"endedAt": worker.get("endedAt") or now()})
    return public_worker(worker)


@router.post("/workers/{identity}/stop")
def stop_worker(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    worker = get_record(db, "worker", identity)
    if not worker.get("managed"):
        raise HTTPException(409, "Externally managed compute must be stopped by its owner.")
    if worker["provider"] in {"lambda", "modal"}:
        raise HTTPException(
            409, "This provider cannot suspend billing. Terminate the worker instead."
        )
    if worker["status"] in {"unknown", "provisioning"}:
        raise HTTPException(
            409, "Wait for provisioning reconciliation before stopping this worker."
        )
    outcome, current = db.begin_shutdown(identity, "stopping", require_idle=True)
    if outcome == "occupied":
        raise HTTPException(
            409, "End active sessions or pending deletion before stopping this worker."
        )
    if current is None:
        raise HTTPException(404, "Worlds worker not found.")
    if outcome == "terminal":
        return public_worker(current)
    try:
        provider(current["provider"], config).stop(current)
    except HTTPException:
        db.patch(
            "worker", identity, {"cleanupError": "Provider stop not confirmed; cleanup will retry."}
        )
        raise
    return public_worker(db.patch("worker", identity, {"status": "stopped", "cleanupError": None}))


@router.delete("/workers/{identity}")
def destroy_worker(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    worker = get_record(db, "worker", identity)
    if worker["status"] in {"unknown", "provisioning"}:
        raise HTTPException(
            409,
            "Worker creation outcome is unknown. Await reconciliation or check the provider console.",
        )
    outcome, current = db.begin_shutdown(
        identity,
        "terminating" if worker.get("managed") else "detaching",
        require_idle=not worker.get("managed"),
    )
    if outcome == "occupied":
        raise HTTPException(409, "End this worker's sessions before removing its connection.")
    if current is None:
        raise HTTPException(404, "Worlds worker not found.")
    if outcome == "terminal":
        return public_worker(current)
    if current.get("managed"):
        current = Lifecycle(config, db).terminate_worker(current, "user-request")
    else:
        current = db.patch(
            "worker",
            identity,
            {
                "status": "detached",
                "message": "Connection removed. Externally managed compute continues until its owner stops it.",
            },
        )
    return public_worker(current)


def session_update(session: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    # Do not relay internal paths, upstream credentials or arbitrary echoed input fields.
    for key in (
        "status",
        "seed",
        "capabilities",
        "progress",
        "stage",
        "fps",
        "latencyMs",
        "frameId",
        "resumeKind",
        "frameIndex",
        "chunks",
        "generationSeconds",
        "generatedFPS",
        "interactionMode",
        "queuedRevision",
        "appliedRevision",
        "generatingRevision",
        "bufferedChunks",
        "totalGeneratedFrames",
        "totalGenerationSeconds",
        "modelLoadSeconds",
        "modelLoadCount",
        "transformerLoadCount",
        "promptEncodingCount",
        "continuity",
        "residency",
        "renderedWindowCount",
        "samplingSeconds",
        "promptTruncated",
        "peakVRAMBytes",
        "audioAvailable",
    ):
        if key in result:
            session[key] = result[key]
    if result.get("error"):
        session["error"] = (
            "Model worker reported a generation failure. Check worker health or retry."
        )
    return session


@router.post("/sessions", status_code=201)
def create_session(body: SessionCreate, config: Settings, db: Database) -> dict[str, Any]:
    worker = get_record(db, "worker", body.worker_id)
    if worker.get("modelId", "astronex-world") != body.model_id:
        raise HTTPException(
            409, "Worker runtime does not match the selected model. Connect a matching worker."
        )
    if not worker.get("gatewayUrl") or worker["status"] != "ready":
        raise HTTPException(
            409, "Worker is not ready. Poll worker status before starting a session."
        )
    capabilities = worker.get("capabilities", {})
    input_caps = capabilities.get("input", {}) if isinstance(capabilities, dict) else {}
    if isinstance(input_caps, dict):
        if input_caps.get("text") is False and body.prompt.strip():
            raise HTTPException(422, "Selected worker does not accept text conditioning.")
        if input_caps.get("text", True) and not body.prompt.strip():
            raise HTTPException(422, "Selected worker requires a prompt.")
    session = {
        "id": str(uuid4()),
        "workerId": worker["id"],
        "modelId": body.model_id,
        "status": "starting",
        "createdAt": now(),
        "leaseExpiresAt": min(
            time.time() + config.session_lease_seconds, worker.get("hardDeadline", float("inf"))
        ),
        "hardDeadline": worker.get("hardDeadline"),
        "heartbeatIntervalSeconds": 20,
    }
    claim = db.claim_session(session, idle_seconds=config.worker_idle_seconds)
    if claim == "occupied":
        raise HTTPException(
            409,
            "This worker already has an active or unresolved session. Choose another worker or end that session first.",
        )
    if claim == "not-ready":
        raise HTTPException(
            409, "Worker is no longer ready. Refresh its status before starting a session."
        )
    payload = {
        "id": session["id"],
        "modelId": body.model_id,
        "input": {"prompt": body.prompt, **body.inputs},
        "seed": body.seed,
        "quality": body.quality,
        "resolution": body.resolution,
    }
    try:
        result = response_json(gateway(config, worker, "POST", "/sessions", payload))
        if result.get("id") != session["id"]:
            raise HTTPException(502, "Worker did not preserve the requested session ID.")
    except HTTPException:
        session["status"] = "error"
        db.patch("session", session["id"], {"status": "error"})
        raise
    return db.patch(
        "session",
        session["id"],
        session_update({}, result),
    )


@router.get("/sessions")
def sessions(db: Database) -> dict[str, Any]:
    return {"sessions": db.list("session")}


@router.get("/sessions/{identity}")
def session_status(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    session, worker = session_context(db, identity)
    if session["status"] == "stopped":
        return session
    result = response_json(gateway(config, worker, "GET", f"/sessions/{identity}"))
    return db.patch(
        "session",
        identity,
        session_update({}, result),
    )


@router.post("/sessions/{identity}/actions")
def action(identity: str, body: Action, config: Settings, db: Database) -> dict[str, Any]:
    _, worker = active_context(db, identity)
    result = response_json(
        gateway(
            config,
            worker,
            "POST",
            f"/sessions/{identity}/actions",
            body.model_dump(exclude_none=True),
        )
    )
    return {
        key: value
        for key, value in result.items()
        if key in {"accepted", "status", "mechanism", "sequence", "appliesAt", "message", "revision"}
    }


@router.post("/sessions/{identity}/offer")
def offer(identity: str, body: Offer, config: Settings, db: Database) -> dict[str, Any]:
    _, worker = active_context(db, identity)
    result = response_json(
        gateway(config, worker, "POST", f"/sessions/{identity}/offer", body.model_dump())
    )
    if result.get("type") != "answer" or not isinstance(result.get("sdp"), str):
        raise HTTPException(502, "Worker returned an invalid WebRTC answer.")
    return {"type": "answer", "sdp": result["sdp"]}


@router.get("/sessions/{identity}/frame")
def frame(identity: str, config: Settings, db: Database) -> Response:
    _, worker = active_context(db, identity)
    result = gateway(config, worker, "GET", f"/sessions/{identity}/frame")
    if result.status_code == 204:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    media = result.headers.get("content-type", "").split(";")[0]
    if media not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(502, "Worker did not return a supported image frame.")
    headers = {
        key: result.headers[key]
        for key in ("x-generated-fps", "x-frame-id", "x-frame-index", "x-latency-ms")
        if key in result.headers
    }
    headers["Cache-Control"] = "no-store"
    return Response(content=result.content, media_type=media, headers=headers)


@router.post("/sessions/{identity}/snapshot")
def snapshot(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    _, worker = active_context(db, identity)
    result = response_json(gateway(config, worker, "POST", f"/sessions/{identity}/snapshot"))
    allowed = {"exact", "approximate", "visual-checkpoint", "exact-resume", "approximate-resume"}
    if result.get("resumeKind") not in allowed:
        raise HTTPException(502, "Worker did not classify snapshot fidelity.")
    return {
        key: value
        for key, value in result.items()
        if key in {"resumeKind", "state", "seed", "frameId", "modelId", "frame"}
    }


@router.delete("/sessions/{identity}")
def stop_session(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    session, worker = session_context(db, identity)
    if session["status"] != "stopped":
        try:
            gateway(config, worker, "DELETE", f"/sessions/{identity}")
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
        session["status"] = "stopped"
        session["endedAt"] = now()
        session = db.patch(
            "session", identity, {"status": "stopped", "endedAt": session["endedAt"]}
        )
    return session


@router.post("/sessions/{identity}/heartbeat")
def heartbeat(identity: str, body: Heartbeat, config: Settings, db: Database) -> dict[str, Any]:
    get_record(db, "session", identity)
    value = db.heartbeat(
        identity,
        time.time(),
        config.session_lease_seconds,
        config.worker_idle_seconds,
        active=body.active,
    )
    if value is None:
        raise HTTPException(409, "Session lease expired or cleanup started; it cannot be renewed.")
    worker = get_record(db, "worker", value["workerId"])
    gateway(config, worker, "POST", f"/sessions/{identity}/heartbeat", {"active": body.active})
    return {
        key: value.get(key)
        for key in ("id", "status", "leaseExpiresAt", "hardDeadline", "heartbeatIntervalSeconds")
    }


@router.get("/sessions/{identity}/transport-config")
def transport_config(identity: str, config: Settings, db: Database) -> dict[str, Any]:
    _, worker = active_context(db, identity)
    result = response_json(gateway(config, worker, "GET", f"/sessions/{identity}/transport-config"))
    servers = result.get("iceServers", [])
    if not isinstance(servers, list) or len(servers) > 10:
        raise HTTPException(502, "Worker returned invalid transport configuration.")
    return {
        "iceServers": [
            {
                key: value
                for key, value in server.items()
                if key in {"urls", "username", "credential"}
            }
            for server in servers
            if isinstance(server, dict)
        ],
        "iceTransportPolicy": result.get("iceTransportPolicy", "all"),
    }


@router.get("/readiness")
def readiness(config: Settings, context: Request) -> dict[str, Any]:
    def inspect(pair: tuple[str, WorldsSettings]) -> dict[str, Any]:
        name, selected = pair
        result: dict[str, Any] = {
            "provider": name,
            "modelId": selected.model_id,
            "status": "unconfigured",
            "models": [],
        }
        try:
            url = selected.gateway_url(name)
            if not url:
                return result
            token = config.gateway_token
            health = response_json(
                request(
                    "GET",
                    url.rstrip("/") + "/health",
                    token=token.get_secret_value() if token else None,
                    timeout=3,
                )
            )
            raw_models = health.get("models", [])
            models = []
            if isinstance(raw_models, list):
                for item in raw_models[:30]:
                    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                        continue
                    models.append(
                        {
                            "id": item["id"][:100],
                            "status": item.get("status")
                            if isinstance(item.get("status"), str)
                            and item.get("status") in {"ready", "unavailable", "loading"}
                            else "unknown",
                            "reason": item.get("reason", "")[:500]
                            if isinstance(item.get("reason", ""), str)
                            else "",
                            "capabilities": item.get("capabilities", {})
                            if isinstance(item.get("capabilities", {}), dict)
                            else {},
                        }
                    )
            result.update(
                status=health.get("status")
                if isinstance(health.get("status"), str)
                and health.get("status") in {"ready", "ok", "healthy", "unavailable", "loading"}
                else "unknown",
                models=models,
            )
            if result["status"] not in {"ready", "ok", "healthy"}:
                result["message"] = "Gateway connected; install and verify the selected model."
        except (HTTPException, ValueError):
            result.update(
                status="unavailable",
                message="Gateway health check failed. Verify HTTPS, authentication and worker startup.",
            )
        return result

    probes = [(name, config) for name in ("runpod", "local", "modal", "lambda")]
    seen = {
        (name, getattr(config, f"{name}_gateway_url", None), config.model_id) for name, _ in probes
    }
    for model_id in config.model_profiles_json:
        selected = config.for_model(model_id)
        for name in ("runpod", "local", "modal", "lambda"):
            url = getattr(selected, f"{name}_gateway_url", None)
            if url and (name, url, model_id) not in seen:
                probes.append((name, selected))
                seen.add((name, url, model_id))
    with ThreadPoolExecutor(max_workers=16) as executor:
        gateways = list(executor.map(inspect, probes))
    provider_state = providers(config, context)
    running = bool(getattr(getattr(context.app.state, "worlds_lifecycle", None), "running", False))
    return {
        **provider_state,
        "gateways": gateways,
        "provisioningEnabled": any(item["canProvision"] for item in provider_state["providers"]),
        "configurationIssue": config.provisioning_problem()
        if config.runpod_allow_provision and not config.runpod_gateway_url
        else None,
        "lifecycle": {
            "enabled": config.lifecycle_enabled,
            "running": running,
            "sessionLeaseSeconds": config.session_lease_seconds,
            "heartbeatIntervalSeconds": 20,
            "workerIdleSeconds": config.worker_idle_seconds,
            "workerMaxLifetimeSeconds": config.worker_max_lifetime_seconds,
            "workerStartupSeconds": config.worker_startup_seconds,
            "maxManagedWorkers": config.max_managed_workers,
            "maxWorkerHourlyCost": config.max_worker_hourly_cost,
        },
    }


class Quote(Body):
    model_id: str = Field(
        default="astronex-world", alias="modelId", pattern=r"^[a-zA-Z0-9_-]{1,100}$"
    )
    gpu_type_id: str | None = Field(default=None, alias="gpuTypeId", max_length=100)
    duration_minutes: float = Field(
        default=10, alias="durationMinutes", gt=0, le=240, allow_inf_nan=False
    )


def runtime(config: WorldsSettings, model_id: str) -> WorldsSettings:
    try:
        return config.for_model(model_id)
    except ValueError:
        raise HTTPException(422, "Model has no approved runtime profile.") from None


@router.get("/providers/{name}/hardware")
def provider_hardware(
    name: Literal["runpod", "local", "modal", "lambda"], config: Settings
) -> dict[str, Any]:
    observed = now()
    return {
        "provider": name,
        "hardware": [{**row, "observedAt": observed} for row in hardware(name, config)],
    }


@router.post("/providers/{name}/quote")
def provider_quote(
    name: Literal["runpod", "local", "modal", "lambda"], body: Quote, config: Settings
) -> dict[str, Any]:
    config = runtime(config, body.model_id)
    gpu = body.gpu_type_id or config.gpu_type(name) or "configured-gateway"
    chosen = next((row for row in hardware(name, config) if row["id"] == gpu), {})
    rate = chosen.get("hourlyCost")
    return {
        "provider": name,
        "modelId": body.model_id,
        "gpuTypeId": gpu,
        "hourlyCost": rate,
        "estimatedCostUSD": rate * body.duration_minutes / 60 if rate is not None else None,
        "pricingSource": chosen.get("pricingSource", "unavailable"),
        "available": chosen.get("available"),
        "observedAt": now(),
        "billedCostUSD": None,
        "message": (
            "Compute estimate excludes storage, network, tax and rounding. "
            "Provider invoice is authoritative."
        ),
    }


@router.get("/usage")
def usage(db: Database) -> dict[str, Any]:
    rows = []
    timestamp = time.time()
    for worker in db.records("worker"):
        try:
            start = datetime.fromisoformat(worker["createdAt"]).timestamp()
            end = (
                datetime.fromisoformat(worker["endedAt"]).timestamp()
                if worker.get("endedAt")
                else timestamp
            )
        except (KeyError, ValueError, TypeError):
            continue
        rate = worker.get("estimatedHourlyCost")
        duration = max(0, end - start)
        rows.append(
            {
                "workerId": worker["id"],
                "provider": worker["provider"],
                "modelId": worker.get("modelId", "astronex-world"),
                "status": worker["status"],
                "startedAt": worker["createdAt"],
                "endedAt": worker.get("endedAt"),
                "durationSeconds": duration,
                "hourlyCost": rate,
                "estimatedCostUSD": duration / 3600 * rate
                if rate is not None and worker.get("managed")
                else None,
                "billedCostUSD": None,
                "pricingSource": worker.get(
                    "pricingSource", "provider-quote" if rate is not None else "unavailable"
                ),
            }
        )
    reconciliation = billing_summary(db)
    imported_usd = {
        row["workerId"]: row for row in reconciliation["workerTotals"] if row["currency"] == "USD"
    }
    for row in rows:
        actual = imported_usd.get(row["workerId"], {})
        row["importedActualCostUSD"] = actual.get("importedActual")
        row["importedComputeCostUSD"] = actual.get("importedCompute")
        row["estimatedImportedPeriodComputeCostUSD"] = actual.get("estimatedComputeCostUSD")
        row["computeDeltaUSD"] = actual.get("computeDeltaUSD")
        row["billingSource"] = "operator-imported" if actual else None
    estimates = [row["estimatedCostUSD"] for row in rows if row["estimatedCostUSD"] is not None]
    deltas = [row["computeDeltaUSD"] for row in rows if row["computeDeltaUSD"] is not None]
    return {
        "workers": rows,
        "importedActualCostUSD": sum(row["importedActual"] for row in imported_usd.values())
        if imported_usd
        else None,
        "computeDeltaUSD": sum(deltas) if deltas else None,
        "unmatchedBillingRows": reconciliation["unmatchedCount"],
        "estimatedCostUSD": sum(estimates) if estimates else None,
        "billedCostUSD": None,
        "message": "Elapsed estimate only. Disk charges and price changes are unmeasured; billed totals unavailable.",
    }


class BillingLine(Body):
    provider: Literal["runpod", "lambda", "modal", "local"]
    reference: str = Field(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$")
    source: str = Field(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$")
    worker_id: str | None = Field(default=None, alias="workerId", min_length=1, max_length=100)
    provider_id: str | None = Field(default=None, alias="providerId", min_length=1, max_length=100)
    amount: Decimal = Field(
        ge=-1000000000, le=1000000000, max_digits=16, decimal_places=6, allow_inf_nan=False
    )
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    period_start: AwareDatetime = Field(alias="periodStart")
    period_end: AwareDatetime = Field(alias="periodEnd")
    kind: Literal["charge", "credit"]
    category: Literal["compute", "storage", "network", "tax", "other"] = "other"
    description: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_line(self) -> BillingLine:
        if (self.worker_id is None) == (self.provider_id is None):
            raise ValueError("Specify exactly one owned workerId or providerId target")
        if not self.reference.strip() or not self.source.strip():
            raise ValueError("Reference and source must not be blank")
        if (
            self.period_end <= self.period_start
            or (self.period_end - self.period_start).total_seconds() > 366 * 86400
        ):
            raise ValueError("Billing period must be positive and at most 366 days")
        if self.kind == "credit" and self.amount >= 0:
            raise ValueError("Credits must have a negative amount")
        if self.kind == "charge" and self.amount < 0:
            raise ValueError("Charges cannot be negative; use kind credit")
        return self


class BillingImport(Body):
    rows: list[BillingLine] = Field(min_length=1, max_length=100)


def billing_summary(db: Store, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    workers = db.records("worker")
    by_id = {worker["id"]: worker for worker in workers}
    by_provider: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for owned_worker in workers:
        if owned_worker.get("providerId"):
            by_provider.setdefault(
                (owned_worker["provider"], owned_worker["providerId"]), []
            ).append(owned_worker)
    records = []
    totals: dict[str, dict[str, Decimal]] = {}
    worker_totals: dict[tuple[str, str], dict[str, Any]] = {}
    unmatched = 0
    for item in db.records("billing"):
        if item.get("requestedWorkerId") is not None:
            candidate = by_id.get(item["requestedWorkerId"])
            candidates = (
                [candidate]
                if candidate is not None and candidate["provider"] == item["provider"]
                else []
            )
        else:
            candidates = by_provider.get((item["provider"], item["requestedProviderId"]), [])
        worker = candidates[0] if len(candidates) == 1 else None
        amount = Decimal(item["amount"])
        currency = item["currency"]
        currency_totals = totals.setdefault(
            currency,
            {
                "importedActual": Decimal(0),
                "matchedActual": Decimal(0),
                "unmatchedActual": Decimal(0),
            },
        )
        currency_totals["importedActual"] += amount
        currency_totals["matchedActual" if worker else "unmatchedActual"] += amount
        # Provider identifiers are operator supplied/imported, never credentials.
        records.append(
            {
                **item,
                "amount": float(amount),
                "workerId": worker["id"] if worker else None,
                "matchStatus": "matched" if worker else "unmatched",
                "unmatchedReason": None
                if worker
                else "Target is absent, provider mismatched, or provider ID is ambiguous.",
            }
        )
        if worker is None:
            unmatched += 1
            continue
        group = worker_totals.setdefault(
            (worker["id"], currency),
            {
                "workerId": worker["id"],
                "provider": worker["provider"],
                "currency": currency,
                "importedActual": Decimal(0),
                "importedCompute": Decimal(0),
                "computePeriods": [],
                "lineCount": 0,
            },
        )
        group["importedActual"] += amount
        group["lineCount"] += 1
        if item["category"] == "compute":
            group["importedCompute"] += amount
            group["computePeriods"].append(
                (
                    datetime.fromisoformat(item["periodStart"]).timestamp(),
                    datetime.fromisoformat(item["periodEnd"]).timestamp(),
                )
            )
    summaries = []
    for group in worker_totals.values():
        worker = by_id[group["workerId"]]
        estimate = None
        duration = 0.0
        periods = sorted(group.pop("computePeriods"))
        rate = worker.get("estimatedHourlyCost")
        if group["currency"] == "USD" and periods and rate is not None and worker.get("managed"):
            try:
                start = datetime.fromisoformat(worker["createdAt"]).timestamp()
                end = (
                    datetime.fromisoformat(worker["endedAt"]).timestamp()
                    if worker.get("endedAt")
                    else time.time()
                )
                merged: list[list[float]] = []
                for left, right in periods:
                    left, right = max(left, start), min(right, end)
                    if right <= left:
                        continue
                    if merged and left <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], right)
                    else:
                        merged.append([left, right])
                duration = sum(right - left for left, right in merged)
                estimate = Decimal(str(duration)) / Decimal(3600) * Decimal(str(rate))
            except (KeyError, ValueError, TypeError):
                estimate = None
        summaries.append(
            {
                **group,
                "importedActual": float(group["importedActual"]),
                "importedCompute": float(group["importedCompute"]),
                "estimatedComputeCostUSD": float(estimate) if estimate is not None else None,
                "computeDeltaUSD": float(group["importedCompute"] - estimate)
                if estimate is not None
                else None,
                "comparedComputeSeconds": duration if estimate is not None else None,
                "sourceType": "operator-imported",
            }
        )
    records.sort(key=lambda row: (row["importedAt"], row["id"]), reverse=True)
    return {
        "records": records[offset : offset + limit],
        "totalRecords": len(records),
        "unmatchedCount": unmatched,
        "offset": offset,
        "limit": limit,
        "workerTotals": summaries,
        "totals": [
            {"currency": currency, **{key: float(value) for key, value in values.items()}}
            for currency, values in sorted(totals.items())
        ],
        "sourceType": "operator-imported",
        "providerReportedTotals": [],
        "message": (
            "Imported amounts are operator assertions, not API-verified totals. No currency conversion. "
            "Compute delta compares imported compute periods with elapsed-rate estimates; other fees excluded."
        ),
    }


@router.get("/billing")
def billing(
    db: Database,
    offset: int = Query(default=0, ge=0, le=10000),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    return billing_summary(db, offset=offset, limit=limit)


@router.post("/billing/import")
def import_billing(body: BillingImport, db: Database) -> dict[str, Any]:
    rows = []
    timestamp = now()
    for item in body.rows:
        rows.append(
            {
                "id": hashlib.sha256((item.provider + "\0" + item.reference).encode()).hexdigest(),
                "provider": item.provider,
                "reference": item.reference,
                "source": item.source,
                "requestedWorkerId": item.worker_id,
                "requestedProviderId": item.provider_id,
                "amount": "0" if item.amount == 0 else str(item.amount.normalize()),
                "currency": item.currency,
                "periodStart": item.period_start.astimezone(UTC).isoformat(),
                "periodEnd": item.period_end.astimezone(UTC).isoformat(),
                "kind": item.kind,
                "category": item.category,
                "description": item.description,
                "sourceType": "operator-imported",
                "importedAt": timestamp,
            }
        )
    try:
        inserted, duplicates = db.import_billing(rows)
    except ValueError as exc:
        if str(exc) == "ledger-limit":
            raise HTTPException(
                409, "Billing ledger is limited to 10,000 immutable imported lines."
            ) from None
        raise HTTPException(
            409,
            "A provider reference already exists with different content. No rows imported. "
            "Use a distinct credit/adjustment reference.",
        ) from None
    return {"inserted": inserted, "duplicates": duplicates, **billing_summary(db)}


@router.get("/billing/runpod/{identity}")
def runpod_billing(
    identity: str,
    config: Settings,
    db: Database,
    start_time: Annotated[AwareDatetime, Query(alias="startTime")],
    end_time: Annotated[AwareDatetime, Query(alias="endTime")],
) -> dict[str, Any]:
    if end_time <= start_time or (end_time - start_time).total_seconds() > 31 * 86400:
        raise HTTPException(
            422, "Provider billing queries require a positive period of at most 31 days."
        )
    worker = get_record(db, "worker", identity)
    if worker["provider"] != "runpod" or not worker.get("managed") or not worker.get("providerId"):
        raise HTTPException(
            409, "Live billing is supported only for an owned managed RunPod worker."
        )
    start, end = start_time.astimezone(UTC).isoformat(), end_time.astimezone(UTC).isoformat()
    rows = RunPodProvider(config).billing(worker, start, end)
    amount = sum((Decimal(str(row["amount"])) for row in rows), Decimal(0))
    return {
        "workerId": identity,
        "provider": "runpod",
        "sourceType": "provider-reported",
        "source": "RunPod REST v1 /billing/pods",
        "periodStart": start,
        "periodEnd": end,
        "bucketSize": "day",
        "currency": "USD",
        "records": rows,
        "reportedAmountUSD": float(amount) if rows else None,
        "observedAt": now(),
        "message": (
            "Live RunPod billing buckets, separate from imported invoices and not compute-only. "
            "Empty response is unknown, not zero. Results are not added to import totals."
        ),
    }
