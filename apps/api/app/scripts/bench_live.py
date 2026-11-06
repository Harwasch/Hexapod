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
import subprocess
import sys
import threading
import uuid
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from app.models.enums import RunStatus
from app.schemas.live import (
    LiveCameras,
    LiveSplat,
    LiveStage,
    LiveState,
    LiveTrain,
    LiveTrainEnded,
    LiveTrainTiles,
)
from app.services.live_train import derive_events, end_reason, final_metrics, merge_live

# Deliberately avoid pipeline_bridge: it imports app.config and the worker's settings.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools/pipeline"))
from live import MAX_LINE, parse_line

LOG = logging.getLogger(__name__)
STATE = Path.home() / ".local/share/hexapod-bench"


def snapshot_name(name: str, cached: Collection[str]) -> bool:
    """Only immutable previews actually downloaded from the volume may be served."""
    allowed = name == "seeds.bin" or bool(re.fullmatch(r"splat_[a-zA-Z0-9_-]+\.spz", name))
    return allowed and name in cached


def _metrics_record(line: str, following: Sequence[str]) -> tuple[str, dict[str, Any]] | None:
    """Decode the quality branch's optional tag without importing that branch's code."""
    _prefix, tag, text = line.partition("live-metrics: ")
    if not tag:
        return None
    for extra in ("", *following):
        text += extra
        if len(text) > MAX_LINE:
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            return None
        step = payload.get("step")
        if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
            return None
        if any(
            payload.get(key) is not None and not isinstance(payload[key], list)
            for key in ("trainMemoryCurve", "valPsnrCurve")
        ):
            return None
        return "metrics", payload
    return None


def bench_records(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Keep live.records order and split-line semantics, with a metrics-only fallback."""
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        if "live-" not in line:
            continue
        following = lines[index + 1 : index + 8]
        parsed = parse_line(line, following)
        if parsed is None:
            parsed = _metrics_record(line, following)
        if parsed is not None:
            found.append(parsed)
    return found


def bench_final(result: dict[str, Any] | None) -> dict[str, int | float] | None:
    """Test scores from parse_result; corrected validation scores from its embedded report."""
    if result is None:
        return None
    scores = result.get("test")
    artifacts = result.get("artifacts")
    metrics = artifacts.get("train_metrics.json") if isinstance(artifacts, dict) else None
    corrected = metrics.get("colorCorrected") if isinstance(metrics, dict) else None
    fields = (
        {key: scores.get(key) for key in ("psnr", "ssim", "lpips")}
        if isinstance(scores, dict)
        else {}
    )
    if isinstance(corrected, dict):
        fields.update({f"cc{key.title()}": corrected.get(key) for key in ("psnr", "ssim", "lpips")})
    return final_metrics(fields)


def state_from_log(
    run_id: str,
    text: str,
    result: dict[str, Any] | None,
    *,
    cached: Collection[str] = (),
    started_at: datetime | None = None,
    plan: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
    replaced_by: str | None = None,
    error_texts: Sequence[str] = (),
) -> LiveState:
    """Pure log-to-console projection, using exactly the worker's record/merge semantics."""
    parsed = bench_records(text)
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
    outcome = result if result is not None else failure
    error = (outcome or {}).get("error") or "Bench run failed"
    status = (
        RunStatus.IN_PROGRESS
        if outcome is None
        else RunStatus.COMPLETE
        if (outcome or {}).get("status") == "complete"
        else RunStatus.ERROR
    )
    started = started_at or datetime.now(UTC)
    if stage_plan.get("startedAt"):
        with contextlib.suppress(ValueError):
            started = datetime.fromisoformat(stage_plan["startedAt"])
    final = bench_final(result)
    train = None
    value = merged.get("train")
    if value or final or status == RunStatus.ERROR or replaced_by:
        value = value or {}
        try:
            train = LiveTrain.model_validate({**value, "snapshots": merged.get("snapshots", [])})
            document = train.model_dump(by_alias=True)
            document["final"] = final
            document["relocations"] = value.get("relocations", [])
            document["events"] = derive_events(document)
            train = LiveTrain.model_validate(document)
            train.replaced_by = replaced_by
            if status == RunStatus.ERROR and train.stop is None and train.final is None:
                train.ended = LiveTrainEnded(
                    reason=end_reason(
                        [(result or {}).get("error"), (failure or {}).get("error"), *error_texts],
                        preempted=False,
                    ),
                    step=max([0, *train.series.step, *[e.step for e in train.evals]]),
                )
            for snapshot in train.snapshots:
                name_part = snapshot.key.rsplit("/", 1)[-1]
                snapshot.url = (
                    f"/snapshots/{run_id}/{name_part}" if snapshot_name(name_part, cached) else None
                )
            if train.seeds is not None:
                train.seeds.url = (
                    f"/snapshots/{run_id}/seeds.bin" if snapshot_name("seeds.bin", cached) else None
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
                    "url": f"/snapshots/{run_id}/{name_part}"
                    if snapshot_name(name_part, cached)
                    else None,
                }
            )
    return LiveState(
        job_id=uuid.UUID(run_id),
        capture_id=uuid.UUID(run_id),
        capture_name=name,
        recipe="frozen-bench",
        status=status,
        error=error if status == RunStatus.ERROR else None,
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


def app_oom_lines(app_id: str) -> tuple[str, ...]:
    """Read the app's stdout for at most 20 seconds; never retain unrelated logs.

    `modal app logs` can follow indefinitely. TimeoutExpired carries the stdout
    collected before subprocess.run kills and reaps the CLI. Stderr is discarded.
    Use the shared classifier itself to recognize its exact OOM markers.
    """
    output: str | bytes
    try:
        output = subprocess.run(  # noqa: S603
            ["modal", "app", "logs", app_id],  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        ).stdout
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ()
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return tuple(
        line for line in output.splitlines() if end_reason([line], preempted=False) == "oom"
    )


class Watch:
    """One poller owns Modal I/O; request handlers only read an immutable state snapshot."""

    def __init__(
        self,
        run_id: str,
        cache: Path,
        plan: dict[str, Any] | None = None,
        *,
        app_log_causes: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.run_id = run_id
        self.cache = cache
        self.plan = plan
        self.app_log_causes = app_log_causes if app_log_causes is not None else {}
        self.started_at = datetime.now(UTC)
        self.text = ""
        self.result: dict[str, Any] | None = None
        self.finished = False
        self.cached: frozenset[str] = frozenset()
        # Do not expose a ledger failure before the first volume read: an ended
        # console stops polling, so that would strand it at step zero without its log.
        self.plan = plan or local_plan(run_id)
        self.state = state_from_log(run_id, "", None, started_at=self.started_at, plan=self.plan)
        self.poll_lock = threading.Lock()
        self.stop = threading.Event()
        self.errors: set[str] = set()

    def _read(self, volume: Any, path: str) -> bytes:
        return b"".join(volume.read_file(path))

    def poll(self, volume: Any) -> None:
        prefix = f"runs/{self.run_id}"
        # `result.json` is the run's last write, so nothing changes after the pass that saw it.
        if self.finished:
            self.refresh()
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
                partial_path = target.with_name(target.name + ".part")
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
        self.refresh()

    def refresh(self) -> None:
        ledger = LocalLedger.read()
        self.plan = ledger.plans.get(self.run_id, self.plan)
        failure = ledger.failures.get(self.run_id)
        outcome = self.result if self.result is not None else failure
        failed = outcome is not None and outcome.get("status") != "complete"
        app_id = (failure or {}).get("appId")
        if (
            failed
            and isinstance(app_id, str)
            and re.fullmatch(r"ap-[a-zA-Z0-9]+", app_id)
            and ledger.app_runs.get(app_id) == {self.run_id}
            and self.run_id not in self.app_log_causes
        ):
            # refresh runs only in the poller. Cache even unavailable/empty logs,
            # including across idle-watch eviction, so failures never retry in a loop.
            self.app_log_causes[self.run_id] = app_oom_lines(app_id)
        self.state = state_from_log(
            self.run_id,
            self.text,
            self.result,
            cached=self.cached,
            started_at=self.started_at,
            plan=self.plan,
            failure=failure,
            replaced_by=ledger.replacement(self.run_id),
            error_texts=self.app_log_causes.get(self.run_id, ()),
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
                    with self.poll_lock:
                        if self.stop.is_set():
                            break
                        self.poll(volume)
                except Exception as error:
                    self._warn("poll", error)
                self.stop.wait(15)
        except Exception as error:
            self._warn("Modal", error)


class Watches:
    """At most eight retained pollers. Active HTTP readers pin their watch; idle
    entries are evicted in least-recently-used order, including the primary run.
    """

    def __init__(self, primary: Watch) -> None:
        self.primary = primary.run_id
        self.app_log_causes = primary.app_log_causes
        self.watches = {primary.run_id: primary}
        self.users = {primary.run_id: 0}
        self.used = {primary.run_id: monotonic()}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    def start(self, watch: Watch) -> None:
        thread = threading.Thread(target=watch.run, daemon=True, name=f"bench-watch-{watch.run_id}")
        self.threads[watch.run_id] = thread
        thread.start()

    @contextlib.contextmanager
    def acquire(self, run_id: str) -> Iterator[Watch | None]:
        with self.lock:
            watch = self.watches.get(run_id)
            if watch is None:
                ledger = LocalLedger.read()
                if run_id == self.primary or run_id in ledger.ids:
                    if len(self.watches) >= 8:
                        for oldest in sorted(self.watches, key=self.used.__getitem__):
                            idle = self.watches[oldest]
                            if self.users[oldest] or not idle.poll_lock.acquire(blocking=False):
                                continue
                            try:
                                idle.stop.set()
                                self.watches.pop(oldest)
                                self.users.pop(oldest)
                                self.used.pop(oldest)
                                self.threads.pop(oldest, None)
                            finally:
                                idle.poll_lock.release()
                            break
                    if len(self.watches) < 8:
                        watch = Watch(
                            run_id,
                            STATE / "live" / run_id,
                            ledger.plans.get(run_id),
                            app_log_causes=self.app_log_causes,
                        )
                        self.watches[run_id] = watch
                        self.users[run_id] = 0
                        self.start(watch)
            if watch is not None:
                self.users[run_id] += 1
                self.used[run_id] = monotonic()
        try:
            yield watch
        finally:
            if watch is not None:
                with self.lock:
                    self.users[run_id] -= 1
                    self.used[run_id] = monotonic()

    def close(self) -> None:
        for watch in self.watches.values():
            watch.stop.set()
        for thread in self.threads.values():
            thread.join(timeout=2)


class Handler(BaseHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        watches: Watches,
        web_dist: Path | None,
        tiles: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.watches = watches
        self.web_dist = web_dist.resolve() if web_dist else None
        # A locally packaged solve-local tileset of the primary run's final splat.
        self.tiles = tiles.resolve() if tiles else None
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if match := re.fullmatch(r"/api/v1/captures/([^/]+)/live", path):
            run_id = match[1]
            with self.watches.acquire(run_id) as watch:
                if watch is not None:
                    self._state(watch)
                else:
                    status = (
                        503
                        if run_id in LocalLedger.read().ids or run_id == self.watches.primary
                        else 404
                    )
                    self._reply(status, b"Watch unavailable", "text/plain")
            return
        if self.tiles and path.startswith("/tiles/"):
            candidate = (self.tiles / path.removeprefix("/tiles/")).resolve()
            if candidate.is_relative_to(self.tiles) and candidate.is_file():
                kind = "application/json" if candidate.suffix == ".json" else "model/gltf-binary"
                self._file(candidate, kind)
                return
        elif path.startswith("/snapshots/"):
            parts = path.removeprefix("/snapshots/").split("/")
            run_id, name = (
                (self.watches.primary, parts[0])
                if len(parts) == 1
                else (parts if len(parts) == 2 else ("", ""))
            )
            # Reject malformed names before looking up or starting any poller.
            if snapshot_name(name, {name}):
                with self.watches.acquire(run_id) as watch:
                    if watch is not None and snapshot_name(name, watch.cached):
                        self._file(watch.cache / name, "application/octet-stream")
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

    def _state(self, watch: Watch) -> None:
        state = watch.state.model_copy(deep=True)
        # Absolute URLs also work when a separately hosted bundle polls this origin.
        host = self.headers.get("Host")
        if not host:
            server = cast(ThreadingHTTPServer, self.server)
            host = f"{server.server_name}:{server.server_port}"
        origin = "http://" + host
        if state.splat and state.splat.url:
            state.splat.url = origin + state.splat.url
        if state.train:
            if state.train.seeds and state.train.seeds.url:
                state.train.seeds.url = origin + state.train.seeds.url
            for snapshot in state.train.snapshots:
                if snapshot.url:
                    snapshot.url = origin + snapshot.url
            if self.tiles and watch.run_id == self.watches.primary:
                step = state.train.series.step[-1] if state.train.series.step else 0
                state.train.tiles = LiveTrainTiles(
                    url=origin + "/tiles/tileset.json", step=step, final=True
                )
        self._reply(200, state.model_dump_json(by_alias=True).encode(), "application/json")

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


@dataclass(frozen=True)
class LedgerRow:
    value: dict[str, Any]
    file: Path
    line: int

    @property
    def timestamp(self) -> datetime | None:
        for key in ("timestamp", "createdAt"):
            value = self.value.get(key)
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    stamp = datetime.fromisoformat(value)
                    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp
        return None


class LocalLedger:
    """Read-only ledger view. Existing writers have no timestamps: across files use
    mtime/name order, within each file always use line order. Never infer from tags.
    A partial append is ignored until the next poll, without hiding other files/rows.
    """

    def __init__(self, rows: list[LedgerRow]) -> None:
        self.rows = rows
        self.ids = {r.value["runId"] for r in rows}
        self.plans = {r.value["runId"]: r.value for r in rows if r.value.get("event") == "planned"}
        self.failures = {
            r.value["runId"]: r.value for r in rows if r.value.get("event") == "failed"
        }
        #: Runs per Modal app. One `bench run` batch shares an app, and its log does not say
        #: which run a line came from, so an OOM line is only evidence for an app's only run.
        self.app_runs: dict[str, set[str]] = {}
        for r in rows:
            app = r.value.get("appId")
            if isinstance(app, str):
                self.app_runs.setdefault(app, set()).add(r.value["runId"])

    @classmethod
    def read(cls) -> LocalLedger:
        files = []
        for path in STATE.glob("*.jsonl"):
            with contextlib.suppress(OSError):
                files.append((path.stat().st_mtime_ns, path.name, path))
        rows = []
        for _, _, path in sorted(files):
            try:
                lines = path.read_text().splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("runId"), str):
                        continue
                    # Ledger ids become cache directories and URL path components.
                    run_id = row["runId"]
                    if not re.fullmatch(r"[a-fA-F0-9-]{32,36}", run_id):
                        continue
                    uuid.UUID(run_id)
                    rows.append(LedgerRow(row, path, index))
                except (ValueError, TypeError):
                    continue
        return cls(rows)

    @staticmethod
    def variant(row: dict[str, Any]) -> tuple[Any, ...] | None:
        variant = row.get("variant")
        name = variant.get("name") if isinstance(variant, dict) else variant
        identity = (row.get("scene"), name, row.get("seed"), row.get("schedule"))
        return identity if all(value is not None for value in identity) else None

    def replacement(self, run_id: str) -> str | None:
        current = run_id
        seen = {run_id}
        while (next_run := self._successor(current)) is not None:
            if next_run in seen:
                return None
            seen.add(next_run)
            current = next_run
        return current if current != run_id else None

    def _successor(self, run_id: str) -> str | None:
        plans = [r for r in self.rows if r.value.get("event") == "planned"]
        source = next((r for r in plans if r.value["runId"] == run_id), None)
        if source is None or self.variant(source.value) is None:
            return None
        for candidate in plans:
            if candidate.value["runId"] == run_id:
                continue
            if self.variant(candidate.value) != self.variant(source.value):
                continue
            if source.file == candidate.file:
                later = candidate.line > source.line
            elif source.timestamp is not None and candidate.timestamp is not None:
                later = candidate.timestamp > source.timestamp
            else:
                later = plans.index(candidate) > plans.index(source)
            if later:
                return str(candidate.value["runId"])
        return None


def local_plan(run_id: str) -> dict[str, Any] | None:
    return LocalLedger.read().plans.get(run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web-dist", type=Path)
    parser.add_argument("--tiles", type=Path, help="tileset directory of the run's final splat")
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
    watches = Watches(watch)
    if args.tiles and not (args.tiles / "tileset.json").is_file():
        parser.error("--tiles must contain tileset.json")
    handler = partial(Handler, watches=watches, web_dist=args.web_dist, tiles=args.tiles)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    watches.start(watch)
    host = args.host if args.host not in ("0.0.0.0", "::") else "<this-machine>"  # noqa: S104
    sys.stdout.write(f"Open http://{host}:{server.server_port}/view.html#live/{args.run_id}\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        watches.close()


if __name__ == "__main__":
    main()
