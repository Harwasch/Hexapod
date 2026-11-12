"""Bounded, atomic training observations beside the existing checkpoint snapshots.

The provider's log service is diagnostic, not a reliable live transport. The stage
writes this journal through its log observer; checkpoint sync uploads it with SPZs.
CloudRunner fetches only this small file while training, never the recovery checkpoint.
The worker merges its validated records with the same reducer used by bench live.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import live

if TYPE_CHECKING:
    from contracts import StageContext

NAME = "telemetry.json"
LOCAL_NAME = "live-telemetry.json"
MAX_BYTES = 256_000
MAX_SAMPLES = 300
GET_TIMEOUT_S = 1.0
Record = tuple[str, dict[str, Any]]


def read(path: Path) -> list[Record]:
    """Validate untrusted storage bytes before the API worker sees them."""
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return []
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("v") != 1:
            return []
        rows = value.get("records")
        if not isinstance(rows, list) or len(rows) > 1024:
            return []
        found: list[Record] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                return []
            kind, payload = row
            tag = next((tag for name, tag, _valid in live._TAGS if name == kind), None)
            if tag is None:
                return []
            parsed = live.parse_line(tag + json.dumps(payload))
            if parsed is None:
                return []
            found.append(parsed)
        return found
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return []


def checkpoint_keys(path: Path, *, stage_attempt: int | None = None) -> list[str]:
    """Runner-owned scope, stamped from CallRecord.request, never a remote journal."""
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        value = json.loads(raw) if len(raw) <= MAX_BYTES else {}
        if stage_attempt is not None and value.get("stageAttempt") != stage_attempt:
            return []
        keys = value.get("checkpointKeys", [])
        if isinstance(keys, list) and len(keys) <= 64:
            return [key for key in keys if isinstance(key, str) and len(key) <= 512]
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        pass
    return []


def write(
    path: Path,
    records: list[Record],
    *,
    checkpoint_key: str | None = None,
    stage_attempt: int | None = None,
) -> None:
    """Telemetry persistence must never turn a running GPU call into a failure."""
    with contextlib.suppress(Exception):
        keys = (
            list(
                dict.fromkeys(
                    [
                        *checkpoint_keys(path, stage_attempt=stage_attempt),
                        checkpoint_key,
                    ]
                )
            )[-64:]
            if checkpoint_key is not None
            else []
        )
        _write(path, records, keys, stage_attempt)


def clear(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _write(path: Path, records: list[Record], keys: list[str], stage_attempt: int | None) -> None:
    raw = json.dumps(
        {"v": 1, "records": records, "checkpointKeys": keys, "stageAttempt": stage_attempt},
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(raw.encode()) > MAX_BYTES:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(raw, encoding="utf-8")
    partial.replace(path)


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[Record] = []
        self.lock = threading.Lock()

    def line(self, line: str) -> None:
        # Observations may never fail the paid stage, including a broken local disk.
        with contextlib.suppress(Exception):
            parsed = live.parse_line(line)
            if parsed is None:
                return
            with self.lock:
                kind, _payload = parsed
                if kind == "plan":
                    self.records.clear()  # a sequential block or a new trainer
                self.records.append(parsed)
                self._bound()
                write(self.path, self.records)

    def _bound(self) -> None:
        groups: dict[str, list[Record]] = {}
        for kind, value in self.records:
            key = kind + ":" + str(value.get("k", ""))
            groups.setdefault(key, []).append((kind, value))
        retained: set[int] = set()
        for key, rows in groups.items():
            limit = (
                MAX_SAMPLES
                if key == "train:s"
                else 256
                if key == "train:e"
                else 64
                if key in ("splat:", "train:r")
                else 1
            )
            if len(rows) > limit:
                if key == "train:s":
                    width = max(1, math.ceil(rows[-1][1]["step"] / (limit - 2)))
                    buckets = {value["step"] // width: (kind, value) for kind, value in rows[1:-1]}
                    rows = [rows[0], *buckets.values(), rows[-1]]
                else:
                    rows = rows[-limit:]
            retained.update(id(value) for _kind, value in rows)
        self.records = [(kind, value) for kind, value in self.records if id(value) in retained]


def observe(ctx: StageContext) -> StageContext:
    # Viewer state is not recovery state; never replay a previous attempt as current.
    shutil.rmtree(ctx.checkpoint_dir / live.LIVE_DIR, ignore_errors=True)
    if ctx.param("live") is False:
        return ctx
    journal = Journal(ctx.checkpoint_dir / live.LIVE_DIR / NAME)
    return replace(ctx, _observe_log=journal.line)


class Poller:
    """At most one small GET per checkpoint interval, plus a final GET at completion."""

    def __init__(self) -> None:
        self.last = float("-inf")
        self._pending: threading.Thread | None = None

    def poll(
        self, request: Any, transfer: Any, work: Path, now: float, *, final: bool = False
    ) -> list[Record]:
        if request is None or request.impl != "gsplat" or request.params.get("live") is False:
            return []
        if not final and now - self.last < max(0.05, request.checkpoint_every_s):
            return []
        if self._pending is not None and self._pending.is_alive():
            return []  # One stuck GET cannot accumulate threads or late observations.
        self.last = now
        records: list[Record] = []

        def fetch() -> None:
            with contextlib.suppress(Exception):
                work.mkdir(parents=True, exist_ok=True)
                # Each call owns its file: a late GET cannot overwrite a newer poll
                # or another block's journal. Cleanup belongs to the fetch thread.
                with tempfile.TemporaryDirectory(prefix="live-get-", dir=work) as directory:
                    incoming = Path(directory) / NAME
                    key = f"{request.checkpoint_key}/{live.LIVE_DIR}/{NAME}"
                    if transfer.get(key, incoming):
                        records.extend(read(incoming))

        with contextlib.suppress(Exception):
            # A daemon, not a pool: shutdown must not wait forever on a broken bucket.
            self._pending = threading.Thread(target=fetch, daemon=True)
            self._pending.start()
            self._pending.join(GET_TIMEOUT_S)
            if not self._pending.is_alive():
                return records
        return []
