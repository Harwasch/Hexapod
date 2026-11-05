"""Read-only Modal bench watcher and local live-console server; no database or settings.

cd apps/api && uv run --with modal==1.6.2 python -m app.scripts.bench_live RUN_ID
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import mimetypes
import re
import shutil
import sys
import threading
import uuid
from collections.abc import Collection
from datetime import UTC, datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from app.models.enums import RunStatus
from app.schemas.live import LiveCameras, LiveSplat, LiveStage, LiveState, LiveTrain
from app.services.live_train import derive_events, merge_live

# Deliberately avoid pipeline_bridge: it imports app.config and the worker's settings.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools/pipeline"))
from live import records

LOG = logging.getLogger(__name__)
STATE = Path.home() / ".local/share/hexapod-bench"


def snapshot_name(name: str, cached: Collection[str]) -> bool:
    """Only immutable previews actually downloaded from the volume may be served."""
    return bool(re.fullmatch(r"splat_[a-zA-Z0-9_-]+\.spz", name)) and name in cached


def state_from_log(
    run_id: str,
    text: str,
    result: dict[str, Any] | None,
    *,
    cached: Collection[str] = (),
    started_at: datetime | None = None,
    plan: dict[str, Any] | None = None,
) -> LiveState:
    """Pure log-to-console projection, using exactly the worker's record/merge semantics."""
    parsed = records(text)
    # A stage-plan UUID resets training history. Frozen cameras precede that line
    # and belong to the scene, so retain them just as the API retains a pose step.
    frozen = next((value for kind, value in reversed(parsed) if kind == "cameras"), None)
    merged = merge_live({"cameras": frozen} if frozen else {}, parsed)
    stage_plan = next((value for kind, value in reversed(parsed) if kind == "plan"), {})
    metadata = {**stage_plan, **(plan or {}), **(result or {})}
    scene = metadata.get("scene", metadata.get("benchScene"))
    variant = metadata.get("variant")
    if isinstance(variant, dict):
        variant = variant.get("name")
    if not isinstance(variant, str) or not variant:
        variant = metadata.get("benchVariant")
    name = f"{scene} · {variant}" if scene and variant else run_id
    if metadata.get("rehearsal", metadata.get("benchRehearsal")):
        name += " (Simulated rehearsal)"
    status = (
        RunStatus.IN_PROGRESS
        if result is None
        else RunStatus.COMPLETE
        if result.get("status") == "complete"
        else RunStatus.ERROR
    )
    started = started_at or datetime.now(UTC)
    if stage_plan.get("startedAt"):
        with contextlib.suppress(ValueError):
            started = datetime.fromisoformat(stage_plan["startedAt"])
    train = None
    if value := merged.get("train"):
        try:
            train = LiveTrain.model_validate({**value, "snapshots": merged.get("snapshots", [])})
            document = train.model_dump(by_alias=True)
            document["relocations"] = value.get("relocations", [])
            document["events"] = derive_events(document)
            train = LiveTrain.model_validate(document)
            for snapshot in train.snapshots:
                name_part = snapshot.key.rsplit("/", 1)[-1]
                snapshot.url = (
                    f"/snapshots/{name_part}" if snapshot_name(name_part, cached) else None
                )
        except (ValidationError, ValueError, TypeError, KeyError, IndexError):
            train = None
    cameras = None
    if value := merged.get("cameras"):
        with contextlib.suppress(ValidationError):
            cameras = LiveCameras.model_validate(
                {
                    **{
                        field.alias or key: value.get(field.alias or key)
                        for key, field in LiveCameras.model_fields.items()
                        if key != "stage_id"
                    },
                    "stageId": "train",
                }
            )
    splat = None
    if value := merged.get("splat"):
        name_part = value["key"].rsplit("/", 1)[-1]
        with contextlib.suppress(ValidationError):
            splat = LiveSplat.model_validate(
                {
                    **{
                        field.alias or key: value.get(field.alias or key)
                        for key, field in LiveSplat.model_fields.items()
                        if key not in ("stage_id", "name", "url")
                    },
                    "stageId": "train",
                    "name": name_part,
                    "url": f"/snapshots/{name_part}" if snapshot_name(name_part, cached) else None,
                }
            )
    return LiveState(
        job_id=uuid.UUID(run_id),
        capture_id=uuid.UUID(run_id),
        capture_name=name,
        recipe="frozen-bench",
        status=status,
        error=(result or {}).get("error") or "Bench run failed"
        if status == RunStatus.ERROR
        else None,
        site_id=None,
        stage=LiveStage(
            stage_id="train",
            impl="gsplat",
            status=status,
            ordinal=0,
            started_at=started,
            progress=None,
        ),
        steps_done=int(status == RunStatus.COMPLETE),
        steps_started=1,
        cameras=cameras,
        splat=splat,
        train=train,
    )


class Watch:
    """One poller owns Modal I/O; request handlers only read an immutable state snapshot."""

    def __init__(self, run_id: str, cache: Path, plan: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.cache = cache
        self.plan = plan
        self.started_at = datetime.now(UTC)
        self.text = ""
        self.result: dict[str, Any] | None = None
        self.finished = False
        self.cached: frozenset[str] = frozenset()
        self.state = state_from_log(run_id, "", None, started_at=self.started_at, plan=plan)
        self.stop = threading.Event()
        self.errors: set[str] = set()

    def _read(self, volume: Any, path: str) -> bytes:
        return b"".join(volume.read_file(path))

    def poll(self, volume: Any) -> None:
        prefix = f"runs/{self.run_id}"
        # `result.json` is the run's last write, so nothing changes after the pass that saw it.
        if self.finished:
            return
        self.finished = self.result is not None
        for filename in ("live/train.log", "result.json"):
            try:
                raw = self._read(volume, f"{prefix}/{filename}")
                if filename == "result.json":
                    result = json.loads(raw)
                    if not isinstance(result, dict):
                        raise ValueError("result must be an object")
                    self.result = result
                else:
                    self.text = raw.decode("utf-8", errors="replace")
            except FileNotFoundError:
                pass  # A run without --live, or a file that has not been committed yet.
            except Exception as error:
                self._warn(filename, error)
        try:
            for entry in volume.listdir(f"{prefix}/live"):
                name = entry.path.rsplit("/", 1)[-1]
                if not snapshot_name(name, {name}) or name in self.cached:
                    continue
                self.cache.mkdir(parents=True, exist_ok=True)
                target = self.cache / name
                partial_path = target.with_suffix(".spz.part")
                try:
                    with partial_path.open("wb") as output:
                        for chunk in volume.read_file(f"{prefix}/live/{name}"):
                            output.write(chunk)
                    partial_path.replace(target)
                finally:
                    partial_path.unlink(missing_ok=True)
                self.cached = self.cached | {name}
        except FileNotFoundError:
            pass
        except Exception as error:
            # Modal's `listdir` on a missing folder: expected until the first sync lands.
            if type(error).__name__ != "NotFoundError":
                self._warn("snapshots", error)
        self.state = state_from_log(
            self.run_id,
            self.text,
            self.result,
            cached=self.cached,
            started_at=self.started_at,
            plan=self.plan,
        )

    def _warn(self, kind: str, error: Exception) -> None:
        if kind not in self.errors:
            self.errors.add(kind)
            LOG.warning("bench watch: %s: %s", kind, error)

    def run(self) -> None:
        # Importing the script, projecting logs and HTTP tests never import Modal.
        try:
            volume = importlib.import_module("modal").Volume.from_name("hexapod-bench")
            while not self.stop.is_set():
                try:
                    self.poll(volume)
                except Exception as error:
                    self._warn("poll", error)
                self.stop.wait(15)
        except Exception as error:
            self._warn("Modal", error)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, watch: Watch, web_dist: Path | None, **kwargs: Any) -> None:
        self.watch = watch
        self.web_dist = web_dist.resolve() if web_dist else None
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == f"/api/v1/captures/{self.watch.run_id}/live":
            state = self.watch.state.model_copy(deep=True)
            # Host is the address the browser used for this API, including over Tailscale.
            # Absolute URLs also work when a separately hosted bundle polls this origin.
            host = self.headers.get("Host")
            if not host:
                server = cast(ThreadingHTTPServer, self.server)
                host = f"{server.server_name}:{server.server_port}"
            origin = "http://" + host
            if state.splat and state.splat.url:
                state.splat.url = origin + state.splat.url
            if state.train:
                for snapshot in state.train.snapshots:
                    if snapshot.url:
                        snapshot.url = origin + snapshot.url
            self._reply(200, state.model_dump_json(by_alias=True).encode(), "application/json")
            return
        if path.startswith("/snapshots/"):
            name = path.removeprefix("/snapshots/")
            if snapshot_name(name, self.watch.cached):
                self._file(self.watch.cache / name, "application/octet-stream")
                return
        elif self.web_dist and not path.startswith("/api/"):
            candidate = (
                self.web_dist / ("view.html" if path == "/" else path.lstrip("/"))
            ).resolve()
            if candidate.is_relative_to(self.web_dist) and candidate.is_file():
                self._file(
                    candidate, mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                )
                return
        self._reply(404, b"Not found", "text/plain")

    def _file(self, path: Path, content_type: str) -> None:
        try:
            with path.open("rb") as source:
                self.send_response(200)
                self._headers(content_type, path.stat().st_size)
                shutil.copyfileobj(source, self.wfile)
        except FileNotFoundError:
            self._reply(404, b"Not found", "text/plain")

    def _headers(self, content_type: str, size: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._headers(content_type, len(body))
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def local_plan(run_id: str) -> dict[str, Any] | None:
    """The default ledger is optional; a custom ledger's name comes from the synced plan."""
    try:
        for line in (STATE / "ledger.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("runId") == run_id and row.get("event") == "planned":
                return dict(row)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web-dist", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-fA-F0-9-]{32,36}", args.run_id):
        parser.error("run_id must be a bench UUID")
    try:
        uuid.UUID(args.run_id)
    except ValueError:
        parser.error("run_id must be a bench UUID")
    if args.web_dist and not (args.web_dist / "view.html").is_file():
        parser.error("--web-dist must contain view.html; build apps/web first")
    logging.basicConfig(level=logging.INFO)
    watch = Watch(args.run_id, STATE / "live" / args.run_id, local_plan(args.run_id))
    handler = partial(Handler, watch=watch, web_dist=args.web_dist)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    thread = threading.Thread(target=watch.run, daemon=True, name="bench-volume-watch")
    thread.start()
    host = args.host if args.host not in ("0.0.0.0", "::") else "<this-machine>"  # noqa: S104
    sys.stdout.write(f"Open http://{host}:{server.server_port}/view.html#live/{args.run_id}\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        watch.stop.set()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
