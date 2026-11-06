"""Best-effort bench telemetry export. Never changes the recipe or trainer parameters."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import live
from workdir import Workdir

from .ledger import hourly_rate

LOG = logging.getLogger(__name__)
TAGS = (
    live.CAMERAS_TAG,
    live.SPLAT_TAG,
    live.TRAIN_TAG,
    live.PLAN_TAG,
    live.SEEDS_TAG,
    live.METRICS_TAG,
)


class CommitLock:
    """Serialize flushes, coalescing requests behind a stuck telemetry commit.

    Telemetry coalesces overlapping requests. Final flushes wait for ownership and
    always commit on their caller's thread, before a single-use container can exit.
    Callback errors propagate (LiveSync guards its own telemetry calls).
    """

    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.lock = threading.Condition()
        self.running = False
        self.pending = False
        self.warned = False

    def __call__(self) -> bool:
        with self.lock:
            if self.running:
                self.pending = True
                if not self.warned:
                    self.warned = True
                    LOG.warning("bench live: commit still in flight; queued another flush")
                return False
            self.running = True
        return self._flush()

    def final(self, *, timeout: float = 120) -> None:
        """Wait boundedly for telemetry, then perform a real synchronous final flush.

        A stuck telemetry commit must not fail a run that has already been paid for, so
        after the wait this commits anyway, beside it.
        """
        with self.lock:
            if not self.lock.wait_for(lambda: not self.running, timeout=timeout):
                LOG.warning("bench live: commit still in flight after final wait; committing")
            self.running = True
        self._flush()

    def _flush(self) -> bool:
        try:
            while True:
                with self.lock:
                    self.pending = False
                self.callback()
                with self.lock:
                    if not self.pending:
                        self.running = False
                        self.lock.notify_all()
                        return True
        except BaseException:
            with self.lock:
                self.running = False
                self.lock.notify_all()
            raise


def atomic_copy(source: Path, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    try:
        shutil.copyfile(source, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


class LiveSync:
    def __init__(
        self,
        work: Workdir,
        scene: Path,
        destination: Path,
        plan: dict[str, Any],
        commit: Callable[[], object],
        *,
        interval: float = 60,
        join_timeout: float = 60,
    ) -> None:
        self.work = work
        self.scene = scene
        self.directory = destination / "live"
        self.plan = plan
        self.commit = commit
        self.interval = interval
        self.join_timeout = join_timeout
        self.started_at = datetime.now(UTC).isoformat()
        self.offset = 0
        self.dirty = False
        self.errors: set[str] = set()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _guard(self, kind: str, action: Callable[[], object]) -> None:
        try:
            action()
        except Exception as error:
            if kind not in self.errors:
                self.errors.add(kind)
                LOG.warning("bench live: %s: %s", kind, error)

    def _append(self, text: str) -> None:
        if not text:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "train.log"
        partial = target.with_suffix(".log.part")
        try:
            with partial.open("wb") as output:
                if target.exists():
                    with target.open("rb") as previous:
                        shutil.copyfileobj(previous, output)
                output.write(text.encode("utf-8"))
            partial.replace(target)
            self.dirty = True
        finally:
            partial.unlink(missing_ok=True)

    def _cameras(self) -> None:
        live.emit_model(
            lambda line: self._append(line + "\n") if live.CAMERAS_TAG in line else None,
            self.scene / "poses",
            final=True,
        )

    def _logs(self) -> None:
        path = self.work.log_path("train")
        if not path.is_file():
            return
        lines = []
        with path.open("rb") as source:
            source.seek(self.offset)
            position = self.offset
            while raw := source.readline():
                if not raw.endswith(b"\n"):
                    break  # Retry the incomplete line next time.
                position = source.tell()
                line = raw.decode("utf-8", errors="replace")
                tag = next((tag for tag in TAGS if tag in line), None)
                if tag is None:
                    continue
                line = line[line.index(tag) :].rstrip("\r\n")
                if tag == live.PLAN_TAG:

                    def enrich() -> None:
                        nonlocal line
                        payload = json.loads(line[len(live.PLAN_TAG) :])
                        # Display metadata only: results.py compares the original params.
                        payload.update(
                            usdPerHour=hourly_rate(
                                self.plan["tier"], self.plan.get("hostMemoryGiB")
                            ),
                            benchScene=self.plan["scene"],
                            benchVariant=self.plan["variant"]["name"],
                            startedAt=self.started_at,
                            benchRehearsal=bool(self.plan.get("rehearsal")),
                        )
                        line = live.PLAN_TAG + json.dumps(payload, separators=(",", ":"))

                    self._guard("plan", enrich)
                lines.append(line + "\n")
        self._append("".join(lines))
        self.offset = position

    def _snapshots(self) -> None:
        directory = self.work.checkpoint_dir("train") / live.LIVE_DIR
        sources = sorted(directory.glob("splat_*.spz"))
        if (directory / "seeds.bin").is_file():
            sources.append(directory / "seeds.bin")
        for source in sources:
            target = self.directory / source.name
            if target.exists():
                continue
            self.directory.mkdir(parents=True, exist_ok=True)
            atomic_copy(source, target)
            self.dirty = True

    def poll(self) -> None:
        self._guard("logs", self._logs)
        self._guard("snapshots", self._snapshots)
        if self.dirty:

            def flush() -> None:
                if self.commit() is not False:
                    self.dirty = False

            self._guard("commit", flush)

    def _run(self) -> None:
        self._guard("cameras", self._cameras)
        self.poll()
        while not self.stop_event.wait(self.interval):
            self.poll()
        self.poll()  # Last copy happens on the daemon too: stop never waits on I/O forever.

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True, name="bench-live-sync")
        self._guard("start", self.thread.start)

    def stop(self) -> None:
        def finish() -> None:
            self.stop_event.set()
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(self.join_timeout)

        self._guard("stop", finish)


@contextlib.contextmanager
def watching(
    plan: dict[str, Any],
    work: Workdir,
    scene: Path,
    destination: Path,
    commit: Callable[[], object] | None,
) -> Iterator[None]:
    sync = None
    if plan.get("live"):
        try:
            sync = LiveSync(work, scene, destination, plan, commit or (lambda: None))
            sync.start()
        except Exception:
            LOG.warning("bench live: could not start telemetry sync", exc_info=True)
    try:
        yield
    finally:
        if sync is not None:
            sync.stop()
