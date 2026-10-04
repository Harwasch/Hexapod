"""`modal`, as far as `tools/pipeline/modal_adapter.py` uses it. See `tests/fake_modal`.

State lives under `$FAKE_MODAL_DIR`, one directory per call: `request.json`, the
container's `log.txt`, its `pid`, and in the end `result.json`, `error.txt` or
`cancelled`. Three ledgers beside them say what was done to the calls, one id per line:
`spawned.txt`, `from_id.txt`, `cancelled.txt`.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

CONTAINER = Path(__file__).resolve().parent.parent / "container.py"


def _state() -> Path:
    return Path(os.environ["FAKE_MODAL_DIR"])


def _note(ledger: str, call_id: str) -> None:
    with (_state() / ledger).open("a", encoding="utf-8") as handle:
        handle.write(call_id + "\n")


class Function:
    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def from_name(app_name: str, name: str, environment_name: str | None = None) -> Function:
        return Function(name)

    def spawn(self, payload: dict[str, Any]) -> FunctionCall:
        call_id = f"fc-{uuid.uuid4().hex[:12]}"
        directory = _state() / call_id
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(json.dumps(payload), encoding="utf-8")
        log = (directory / "log.txt").open("ab")
        # A session of its own: the recipe process that spawned it can be signalled or
        # killed, and this goes on, as a container on somebody else's machine does.
        process = subprocess.Popen(
            [sys.executable, "-u", str(CONTAINER), str(directory)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        (directory / "pid").write_text(str(process.pid), encoding="utf-8")
        _note("spawned.txt", call_id)
        return FunctionCall(call_id)


class _Entry:
    def __init__(self, message: str) -> None:
        self.message = message


class _Logs:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def fetch(self) -> list[_Entry]:
        path = self._directory / "log.txt"
        if not path.is_file():
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        return [_Entry(line) for line in text.splitlines()]


class FunctionCall:
    def __init__(self, call_id: str) -> None:
        self.object_id = call_id
        self._directory = _state() / call_id
        self.logs = _Logs(self._directory)

    @staticmethod
    def from_id(call_id: str) -> FunctionCall:
        _note("from_id.txt", call_id)
        return FunctionCall(call_id)

    def get(self, timeout: float | None = None, *, index: int = 0) -> Any:
        if (self._directory / "result.json").is_file():
            return json.loads((self._directory / "result.json").read_text(encoding="utf-8"))
        if (self._directory / "error.txt").is_file():
            raise RuntimeError((self._directory / "error.txt").read_text(encoding="utf-8"))
        if (self._directory / "cancelled").is_file():
            raise RuntimeError("the call was cancelled")
        raise TimeoutError()

    def cancel(self, terminate_containers: bool = False) -> None:
        (self._directory / "cancelled").write_text(str(terminate_containers), encoding="utf-8")
        _note("cancelled.txt", self.object_id)
        if terminate_containers:
            pid = int((self._directory / "pid").read_text(encoding="utf-8"))
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
