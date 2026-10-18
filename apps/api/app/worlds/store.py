from __future__ import annotations

import builtins
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal


class Store:
    """Small local metadata ledger, independent from the globe's PostgreSQL schema.

    Credentials, prompts, media and model state are deliberately not persisted here.
    SQLite serializes writes across API workers. Lifecycle operations only touch
    recorded IDs or exact private operation names reserved before provisioning.
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / "metadata.sqlite3"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS records "
                "(kind TEXT NOT NULL, id TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY(kind,id))"
            )
        self.path.chmod(0o600)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, kind: str, value: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as db:
            db.execute(
                "INSERT INTO records VALUES (?,?,?) ON CONFLICT(kind,id) "
                "DO UPDATE SET value=excluded.value",
                (kind, value["id"], json.dumps(value)),
            )
        return value

    def get(self, kind: str, identity: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT value FROM records WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT value FROM records WHERE kind=? ORDER BY rowid DESC LIMIT 500", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def has_active_sessions(self, worker_id: str) -> bool:
        with self.connection() as db:
            return (
                db.execute(
                    "SELECT 1 FROM records WHERE kind=? AND json_extract(value, '$.workerId')=? "
                    "AND json_extract(value, '$.status') != ? LIMIT 1",
                    ("session", worker_id, "stopped"),
                ).fetchone()
                is not None
            )

    def end_worker_sessions(self, worker_id: str, ended_at: str) -> None:
        """A confirmed provider deletion terminates sessions even if its gateway crashed."""
        with self.connection() as db:
            db.execute(
                "UPDATE records SET value=json_set(value, '$.status', 'stopped', '$.endedAt', ?) "
                "WHERE kind='session' AND json_extract(value, '$.workerId')=?",
                (ended_at, worker_id),
            )

    def claim_session(
        self, value: dict[str, Any], *, idle_seconds: int = 300
    ) -> Literal["claimed", "occupied", "not-ready"]:
        """Reserve one worker before inference with an inter-process SQLite write lock.

        Starting and errored sessions retain the claim until explicit cleanup, since
        a failed upstream response does not prove that no inference was started.
        """
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = db.execute(
                "SELECT value FROM records WHERE kind='worker' AND id=?", (value["workerId"],)
            ).fetchone()
            if worker is None or json.loads(worker[0]).get("status") != "ready":
                return "not-ready"
            worker_data = json.loads(worker[0])
            timestamp = time.time()
            if (
                worker_data.get("hardDeadline", float("inf")) <= timestamp
                or worker_data.get("cleanupUntil", 0) > timestamp
            ):
                return "not-ready"
            active = db.execute(
                "SELECT 1 FROM records WHERE kind='session' "
                "AND json_extract(value, '$.workerId')=? "
                "AND json_extract(value, '$.status') != 'stopped' LIMIT 1",
                (value["workerId"],),
            ).fetchone()
            if active is not None:
                return "occupied"
            worker_data["idleDeadline"] = min(
                timestamp + idle_seconds, worker_data.get("hardDeadline", timestamp + idle_seconds)
            )
            db.execute(
                "UPDATE records SET value=? WHERE kind='worker' AND id=?",
                (json.dumps(worker_data), value["workerId"]),
            )
            db.execute(
                "INSERT INTO records VALUES ('session',?,?)", (value["id"], json.dumps(value))
            )
            return "claimed"

    def patch(self, kind: str, identity: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Merge without overwriting concurrent lease renewal or reviving terminal work."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM records WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
            if row is None:
                raise KeyError(identity)
            value: dict[str, Any] = json.loads(row[0])
            if value.get("status") in {
                "terminating",
                "stopping",
                "detaching",
                "destroyed",
                "detached",
                "stopped",
            } and changes.get("status") not in {
                None,
                "stopped",
                "destroyed",
                "detached",
                "terminating",
            }:
                changes = {key: item for key, item in changes.items() if key != "status"}
            value.update(changes)
            db.execute(
                "UPDATE records SET value=? WHERE kind=? AND id=?",
                (json.dumps(value), kind, identity),
            )
            return value

    def reserve_worker(self, value: dict[str, Any], limit: int) -> bool:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM records WHERE kind='worker' "
                "AND json_extract(value, '$.managed')=1 "
                "AND json_extract(value, '$.status') NOT IN ('destroyed','detached')"
            ).fetchone()[0]
            if count >= limit:
                return False
            db.execute(
                "INSERT INTO records VALUES ('worker',?,?)", (value["id"], json.dumps(value))
            )
            return True

    def records(self, kind: str) -> builtins.list[dict[str, Any]]:
        """Lifecycle scans must cover all records, independent of UI pagination."""
        with self.connection() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT value FROM records WHERE kind=?", (kind,)).fetchall()
            ]

    def heartbeat(
        self, identity: str, timestamp: float, duration: int, idle: int, *, active: bool
    ) -> dict[str, Any] | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM records WHERE kind='session' AND id=?", (identity,)
            ).fetchone()
            if row is None:
                return None
            session: dict[str, Any] = json.loads(row[0])
            worker_row = db.execute(
                "SELECT value FROM records WHERE kind='worker' AND id=?", (session["workerId"],)
            ).fetchone()
            if worker_row is None:
                return None
            worker = json.loads(worker_row[0])
            if (
                session.get("status") in {"stopped", "terminating", "error"}
                or worker.get("status")
                in {"stopped", "terminating", "stopping", "detaching", "destroyed", "detached"}
                or session.get("leaseExpiresAt", 0) <= timestamp
                or worker.get("hardDeadline", float("inf")) <= timestamp
                or worker.get("cleanupUntil", 0) > timestamp
            ):
                return None
            if active:
                session["leaseExpiresAt"] = min(
                    timestamp + duration, worker.get("hardDeadline", timestamp + duration)
                )
                worker["idleDeadline"] = min(
                    timestamp + idle, worker.get("hardDeadline", timestamp + idle)
                )
                db.execute(
                    "UPDATE records SET value=? WHERE kind='session' AND id=?",
                    (json.dumps(session), identity),
                )
                db.execute(
                    "UPDATE records SET value=? WHERE kind='worker' AND id=?",
                    (json.dumps(worker), worker["id"]),
                )
            return session

    def claim_cleanup(
        self, kind: str, identity: str, timestamp: float, *, expired_field: str | None = None
    ) -> dict[str, Any] | None:
        """A bounded database lease prevents multiple API processes repeating cleanup."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM records WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
            if row is None:
                return None
            value: dict[str, Any] = json.loads(row[0])
            if value.get("cleanupUntil", 0) > timestamp or value.get("status") in {
                "destroyed",
                "detached",
            }:
                return None
            if expired_field is not None and value.get(expired_field, 0) > timestamp:
                return None
            # Recovery plus confirmed termination can make multiple bounded provider
            # requests. Keep ownership longer than that full operation chain.
            value["cleanupUntil"] = timestamp + 150
            db.execute(
                "UPDATE records SET value=? WHERE kind=? AND id=?",
                (json.dumps(value), kind, identity),
            )
            return value

    def begin_shutdown(
        self,
        identity: str,
        target: Literal["stopping", "terminating", "detaching"],
        *,
        require_idle: bool,
    ) -> tuple[Literal["claimed", "occupied", "terminal", "missing"], dict[str, Any] | None]:
        """Serialize a shutdown decision against the same transaction as session claims."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM records WHERE kind='worker' AND id=?", (identity,)
            ).fetchone()
            if row is None:
                return "missing", None
            worker: dict[str, Any] = json.loads(row[0])
            if worker.get("status") in {"destroyed", "detached"} or (
                target == "stopping" and worker.get("status") == "stopped"
            ):
                return "terminal", worker
            if (
                require_idle
                and db.execute(
                    "SELECT 1 FROM records WHERE kind='session' AND json_extract(value, '$.workerId')=? "
                    "AND json_extract(value, '$.status') != 'stopped' LIMIT 1",
                    (identity,),
                ).fetchone()
                is not None
            ):
                return "occupied", worker
            # A pending irreversible deletion cannot be downgraded to a stop.
            if worker.get("status") == "terminating" and target == "stopping":
                return "occupied", worker
            worker["status"] = target
            db.execute(
                "UPDATE records SET value=? WHERE kind='worker' AND id=?",
                (json.dumps(worker), identity),
            )
            return "claimed", worker

    def import_billing(self, rows: builtins.list[dict[str, Any]]) -> tuple[int, int]:
        """Immutable, atomic invoice-line insertion; duplicates cannot alter prior imports."""
        inserted = duplicates = 0
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM records WHERE kind='billing'").fetchone()[0]
            for item in rows:
                previous = db.execute(
                    "SELECT value FROM records WHERE kind='billing' AND id=?", (item["id"],)
                ).fetchone()
                if previous is not None:
                    stored = json.loads(previous[0])
                    comparable = {key: value for key, value in item.items() if key != "importedAt"}
                    if {
                        key: value for key, value in stored.items() if key != "importedAt"
                    } != comparable:
                        raise ValueError("reference-conflict")
                    duplicates += 1
                    continue
                if count + inserted >= 10000:
                    raise ValueError("ledger-limit")
                db.execute(
                    "INSERT INTO records VALUES ('billing', ?, ?)", (item["id"], json.dumps(item))
                )
                inserted += 1
        return inserted, duplicates
