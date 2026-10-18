"""Durable Worlds leases and provider cleanup, independent from browser polling."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException

from app.worlds.config import WorldsSettings
from app.worlds.providers import provider, request, start_transport, stop_transport
from app.worlds.store import Store

logger = logging.getLogger("twin.worlds.lifecycle")


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def created_timestamp(record: dict[str, Any], fallback: float) -> float:
    try:
        return datetime.fromisoformat(record["createdAt"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return fallback


class Lifecycle:
    def __init__(self, config: WorldsSettings, db: Store) -> None:
        self.config = config
        self.db = db
        self.closed = threading.Event()
        self.last_tick = 0.0
        self.last_success = 0.0
        self.thread = threading.Thread(target=self.run, name="worlds-lease-reaper", daemon=True)

    @property
    def running(self) -> bool:
        return self.thread.is_alive() and time.time() - self.last_success < 180

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.closed.set()
        self.thread.join(timeout=65)

    def run(self) -> None:
        while not self.closed.is_set():
            self.last_tick = time.time()
            try:
                self.tick(self.last_tick)
                self.last_success = time.time()
            except Exception:
                # No exception body: transports can contain URLs/credentials.
                logger.error("Worlds cleanup cycle failed; retrying on the next cycle")
            self.closed.wait(self.config.reaper_interval_seconds)

    def terminate_worker(self, worker: dict[str, Any], reason: str) -> dict[str, Any]:
        self.db.patch(
            "worker", worker["id"], {"status": "terminating", "terminationReason": reason}
        )
        try:
            provider(worker["provider"], self.config).destroy(worker)
        except HTTPException:
            self.db.patch(
                "worker",
                worker["id"],
                {"cleanupError": "Provider teardown not confirmed; cleanup will retry."},
            )
            raise
        self.db.end_worker_sessions(worker["id"], iso_now())
        return self.db.patch(
            "worker",
            worker["id"],
            {"status": "destroyed", "endedAt": iso_now(), "cleanupError": None, "cleanupUntil": 0},
        )

    def tick(self, timestamp: float) -> None:
        # Recover interrupted provisioning using exact random operation names. Never
        # retry POST /pods: an upstream timeout can still have allocated a billable pod.
        for worker in self.db.records("worker"):
            if self.closed.is_set():
                return
            if not worker.get("managed") or worker.get("status") == "destroyed":
                continue
            created = created_timestamp(worker, timestamp)
            if "hardDeadline" not in worker:
                worker = self.db.patch(
                    "worker",
                    worker["id"],
                    {
                        "hardDeadline": created + self.config.worker_max_lifetime_seconds,
                        "idleDeadline": created + self.config.worker_startup_seconds,
                    },
                )
            if not worker.get("providerId"):
                if timestamp < created + 60 or not worker.get("operationName"):
                    continue
                claimed = self.db.claim_cleanup("worker", worker["id"], timestamp)
                if claimed is None:
                    continue
                try:
                    recovered = provider(worker["provider"], self.config).find_worker(
                        worker["operationName"]
                    )
                    if recovered is None:
                        self.db.patch(
                            "worker",
                            worker["id"],
                            {
                                "status": "unknown",
                                "cleanupError": (
                                    "Creation outcome unresolved; capacity remains reserved "
                                    "until reconciliation."
                                ),
                            },
                        )
                        continue
                    worker = self.db.patch("worker", worker["id"], recovered)
                except HTTPException:
                    self.db.patch(
                        "worker",
                        worker["id"],
                        {"cleanupError": "Provider reconciliation failed; cleanup will retry."},
                    )
                    continue
                # Unknown create responses are always cleaned up after recovery; the
                # requesting browser cannot safely depend on their late allocation.
                with suppress(HTTPException):
                    self.terminate_worker(worker, "recovered-provisioning")
                continue
            cost = worker.get("estimatedHourlyCost")
            cap = self.config.max_worker_hourly_cost
            reason = (
                "hard-lifetime"
                if timestamp >= worker["hardDeadline"]
                else "idle-timeout"
                if timestamp >= worker.get("idleDeadline", 0)
                else "cost-limit"
                if cap is not None and (not isinstance(cost, (int, float)) or cost > cap)
                else "retry-cleanup"
                if worker.get("status") == "terminating"
                else "retry-stop"
                if worker.get("status") == "stopping"
                else None
            )
            if (
                reason is None
                or self.db.claim_cleanup(
                    "worker",
                    worker["id"],
                    timestamp,
                    expired_field={
                        "hard-lifetime": "hardDeadline",
                        "idle-timeout": "idleDeadline",
                    }.get(reason),
                )
                is None
            ):
                continue
            with suppress(HTTPException):
                if reason == "retry-stop":
                    provider(worker["provider"], self.config).stop(worker)
                    self.db.patch(
                        "worker",
                        worker["id"],
                        {"status": "stopped", "cleanupUntil": 0, "cleanupError": None},
                    )
                else:
                    self.terminate_worker(worker, reason)
        for session in self.db.records("session"):
            if self.closed.is_set():
                return
            if session.get("status") == "stopped" or session.get("leaseExpiresAt", 0) > timestamp:
                continue
            claimed = self.db.claim_cleanup(
                "session", session["id"], timestamp, expired_field="leaseExpiresAt"
            )
            if claimed is None:
                continue
            session_worker = self.db.get("worker", session["workerId"])
            if session_worker is None or session_worker.get("status") in {"destroyed", "detached"}:
                self.db.patch("session", session["id"], {"status": "stopped", "endedAt": iso_now()})
                continue
            self.db.patch("session", session["id"], {"status": "terminating"})
            try:
                token = self.config.gateway_token
                request(
                    "DELETE",
                    session_worker["gatewayUrl"] + f"/sessions/{session['id']}",
                    token=token.get_secret_value() if token else None,
                    timeout=15,
                )
            except HTTPException as exc:
                if exc.status_code != 404:
                    self.db.patch(
                        "session",
                        session["id"],
                        {"cleanupError": "Session cleanup not confirmed; retry pending."},
                    )
                    continue
            self.db.patch(
                "session",
                session["id"],
                {"status": "stopped", "endedAt": iso_now(), "cleanupError": None},
            )


@asynccontextmanager
async def worlds_lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = WorldsSettings.load()
    app.state.worlds_settings = config
    profiles = [
        config.for_model(model_id)
        for model_id in dict.fromkeys([config.model_id, *config.model_profiles_json])
    ]
    for name, field in (
        ("runpod", "runpod_template_id"),
        ("lambda", "lambda_image_id"),
        ("modal", "modal_image"),
    ):
        candidates = [
            selected for selected in profiles if selected.managed(name) and getattr(selected, field)
        ]
        if not candidates and config.managed(name):
            candidates = [config]  # Explicitly enabled but incomplete config must fail closed.
        for selected in candidates:
            problem = selected.provisioning_problem(name)
            if problem or not app.state.settings.api_write_token:
                raise RuntimeError(
                    problem or "Managed Worlds provisioning requires API_WRITE_TOKEN."
                )
    lifecycle: Lifecycle | None = None
    start_transport()
    try:
        if config.lifecycle_enabled and config.lifecycle_needed:
            lifecycle = Lifecycle(config, Store(config.data_dir))
            app.state.worlds_lifecycle = lifecycle
            lifecycle.start()
        yield
    finally:
        if lifecycle is not None:
            await asyncio.to_thread(lifecycle.close)
        stop_transport()
