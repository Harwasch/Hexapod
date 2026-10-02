"""Starting the worker when a job is queued: the other half of its idle exit.

The worker exits 0 once it has had nothing to do for `WORKER_IDLE_EXIT_S`
(`app/worker/loop.py`), its Fly machine stops (fly.toml restarts the group only on a
failure), and Neon's compute, with nobody polling it, scales to zero. Something then has
to start the worker again, and the only thing that knows a job has just been queued is
the request that queued it. So after a route commits a job, this asks Fly's Machines API
to start the app's stopped `worker` machines.

The rules it keeps, and why:

* **after the commit.** A worker started before the row exists polls an empty queue, and
  a worker that is already polling finds the row on its next tick anyway. The routes add
  this as a FastAPI background task, which runs once the response has been sent -- so only
  after a handler that returned, never after one that raised.
* **off the request path, briefly.** A background task, every call with a 5 s timeout. A
  phone's "Process" does not wait on Fly's API.
* **never fatal.** A wake that fails is a log line. The job is queued whatever happens
  here: the next enqueue wakes the worker, as does `fly machine start`, and
  `QUEUE_CHECK_URL` is what makes a job nobody claims loud rather than silent.
* **nothing to do in development.** No `FLY_API_TOKEN` (or no `FLY_APP_NAME`, which Fly
  sets on its machines and nothing else does) and the call is a no-op.

**The race at the edge of the idle exit.** The worker asks the queue once more just
before it exits (`claim.anything_claimable`), which leaves only the moment between that
question and the process being gone. A job committed then finds the machine still
`started` -- and starting a started machine does nothing -- after which the machine stops
with the job queued. So when this finds a worker machine it did not start (running, or on
its way down), it looks again `RECHECK_AFTER_S` later and starts any that have stopped by
then. A worker that took the job cannot be one of them: it exits only after
`WORKER_IDLE_EXIT_S` of idleness, far longer than the recheck. A machine stopped by then
left without the job.

`QUEUE_CHECK_URL`, when set, is a healthchecks.io-style check: queueing pings
`<url>/start`, and the worker pings `<url>` itself when it claims a job, so a job that is
queued and never claimed -- a token that has expired, a machine that will not boot --
raises an alert after the check's grace time instead of waiting for someone to notice.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import BackgroundTasks

from app.config import Settings, get_settings

log = logging.getLogger("app.worker_wake")

#: Fly's Machines API. The app is named in the path; the token decides what it may do.
FLY_MACHINES_API = "https://api.machines.dev/v1"
#: The process group fly.toml's `[processes]` names the worker; Fly records it on every
#: machine as `config.metadata.fly_process_group`.
WORKER_PROCESS_GROUP = "worker"
#: Per request. Fly answers a start in well under a second; anything slower is an outage
#: this call should not be part of.
TIMEOUT_S = 5.0
#: How long after finding a worker machine up to look at it again (see the module note).
RECHECK_AFTER_S = 20.0
#: The states a start request moves to `started`.
_STARTABLE = frozenset({"stopped", "suspended"})
#: States in which there is no machine to start or wait for.
_GONE = frozenset({"destroying", "destroyed"})


@dataclass(frozen=True)
class Wake:
    """What one look at the worker machines did."""

    #: Machines this call started.
    started: tuple[str, ...] = ()
    #: Machines found in any other live state: up, starting, or on their way down.
    up: tuple[str, ...] = ()


def schedule(background: BackgroundTasks) -> None:
    """What a route that has just committed a job calls: wake the worker after the
    response has gone. One line in each enqueue path, so none of them can forget the
    ordering."""
    background.add_task(after_enqueue)


def after_enqueue(
    settings: Settings | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    later: Callable[[float, Callable[[], object]], None] | None = None,
) -> Wake | None:
    """Ping the queue check, start the stopped worker machines, and look again later if
    one was found already up. Never raises."""
    resolved = settings or get_settings()
    ping_queue_check(resolved, transport=transport)
    wake = wake_workers(resolved, transport=transport)
    if wake is not None and wake.up:
        (later or _later)(RECHECK_AFTER_S, lambda: wake_workers(resolved, transport=transport))
    return wake


def wake_workers(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> Wake | None:
    """Start every stopped or suspended machine in the worker's process group.

    None when there is nothing to call (no token or no app name: development) or the
    listing failed; a start that fails is logged and the others are still started.
    """
    token, app_name = settings.fly_api_token, settings.fly_app_name
    if not token or not app_name:
        return None
    started: list[str] = []
    up: list[str] = []
    try:
        with httpx.Client(
            base_url=FLY_MACHINES_API,
            headers={"Authorization": _authorization(token)},
            timeout=TIMEOUT_S,
            transport=transport,
        ) as client:
            listing = client.get(f"/apps/{app_name}/machines")
            listing.raise_for_status()
            machines = [m for m in _machines(listing.json()) if _group(m) == WORKER_PROCESS_GROUP]
            if not machines:
                log.warning(
                    "wake: app %s has no machine in process group %r; nothing to start",
                    app_name,
                    WORKER_PROCESS_GROUP,
                )
            for machine in machines:
                machine_id, state = str(machine.get("id", "")), str(machine.get("state", ""))
                if state in _GONE or not machine_id:
                    continue
                if state not in _STARTABLE:
                    up.append(machine_id)
                    continue
                try:
                    client.post(f"/apps/{app_name}/machines/{machine_id}/start").raise_for_status()
                except httpx.HTTPError as error:
                    log.warning("wake: could not start worker machine %s: %s", machine_id, error)
                    continue
                started.append(machine_id)
    except (httpx.HTTPError, ValueError) as error:
        # ValueError: a body that is not the JSON the API documents.
        log.warning("wake: could not list app %s's machines: %s", app_name, error)
        return None
    if started:
        log.info("wake: started worker machine(s) %s for a queued job", ", ".join(started))
    return Wake(started=tuple(started), up=tuple(up))


def ping_queue_check(settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
    """`<QUEUE_CHECK_URL>/start`: a job has been queued, and its claim should follow."""
    url = settings.queue_check_url
    if not url:
        return
    try:
        with httpx.Client(timeout=TIMEOUT_S, transport=transport) as client:
            client.get(f"{url.rstrip('/')}/start").raise_for_status()
    except httpx.HTTPError as error:
        log.warning("wake: the queue check did not take the start ping: %s", error)


def _authorization(token: str) -> str:
    """Fly's macaroon tokens (`fly tokens create ...`) are printed with their scheme,
    `FlyV1 fm2_...`, and go in the header as they are; an older personal token is a
    bearer token."""
    stripped = token.strip()
    return stripped if stripped.startswith("FlyV1 ") else f"Bearer {stripped}"


def _machines(document: object) -> list[dict[str, Any]]:
    if not isinstance(document, list):
        raise ValueError(f"expected a list of machines, got {type(document).__name__}")
    return [machine for machine in document if isinstance(machine, dict)]


def _group(machine: dict[str, Any]) -> str | None:
    config = machine.get("config")
    metadata = config.get("metadata") if isinstance(config, dict) else None
    group = metadata.get("fly_process_group") if isinstance(metadata, dict) else None
    return str(group) if group is not None else None


def _later(delay_s: float, call: Callable[[], object]) -> None:
    """Run `call` once, `delay_s` from now, on a daemon thread of its own."""
    timer = threading.Timer(delay_s, call)
    timer.daemon = True
    timer.start()
