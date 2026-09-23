"""The two provider adapters that actually run here, and a transfer over a directory.

`ModalAdapter` lives in `modal_adapter.py` and has never been executed. These two have:

* **`FakeAdapter`** is deterministic and drives every cloud test, preemption included. Its
  "remote" is a generator the test supplies -- one `yield` per unit of work -- so taking
  the machine away is simply not advancing it again. No threads, no clock, no sleeping:
  a preemption happens after exactly N polls or exactly N simulated seconds, every run.
* **`SubprocessAdapter`** runs the stage in a **local subprocess** through the same five
  methods. It is the second implementation, and it is the reason the protocol above is a
  contract rather than a wish: it was written against the same `submit`/`poll`/`logs`/
  `cancel` and it works, including preemption, which for a process is what it is for a
  container -- a signal. It is also useful on its own, as the way to drive a GPU box you
  have a shell on.

Both hold a `Transfer` because that is how a remote gets its bytes; the protocol itself
does not mention one, since a real provider's container fetches its own.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from artifacts import ArtifactDecl
from cloud import Poll, RemoteHandle, RemoteState, StageRequest, Transfer
from contracts import MetricValue
from providers import Rate, provider

__all__ = [
    "FakeAdapter",
    "LocalTransfer",
    "RemoteExecution",
    "RemoteScript",
    "SubprocessAdapter",
    "finishes_immediately",
]


# --- a transfer over a directory ---------------------------------------------------


@dataclass(frozen=True)
class LocalTransfer:
    """`Transfer` over a directory. What the tests use, and what a GPU box on the same
    filesystem (an NFS mount, a workstation with two cards) can use for real.

    The same encoding the worker's S3 implementation uses, so the two behave alike: a
    directory's members live under the key, and a file lives *at* the key.
    """

    root: Path

    def _at(self, key: str) -> Path:
        return self.root / key

    def put(self, key: str, source: Path) -> int:
        target = self._at(key)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return _bytes_under(target)

    def get(self, key: str, target: Path) -> int:
        source = self._at(key)
        if not source.exists():
            return 0
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return _bytes_under(source)

    def exists(self, key: str) -> bool:
        return self._at(key).exists()

    def delete(self, key: str) -> None:
        target = self._at(key)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _bytes_under(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


# --- the fake --------------------------------------------------------------------


@dataclass
class RemoteExecution:
    """The sandbox a scripted remote runs in: the same four directories a stage gets.

    A script is a generator and **one `yield` is one unit of work**: the adapter advances
    it a fixed number of units per poll and moves a simulated clock on by the same
    amount. Taking the machine away is then simply not advancing it again -- which is
    what a killed container looks like from inside a training loop, except that here it
    happens at the same iteration every time the test runs.
    """

    request: StageRequest
    root: Path
    seconds_per_tick: float = 1.0
    elapsed_s: float = 0.0
    lines: list[str] = field(default_factory=list)

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoint"

    def input(self, name: str) -> Path:
        return self.inputs_dir / name

    def output(self, name: str) -> Path:
        decl = self._decl(name)
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if decl is not None and decl.kind == "dir":
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _decl(self, name: str) -> ArtifactDecl | None:
        for decl in self.request.produces:
            if decl.name == name:
                return decl
        return None

    def log(self, message: str) -> None:
        self.lines.append(message)


#: What a fake remote is: a generator that yields once per unit of work.
RemoteScript = Callable[[RemoteExecution], Iterator[None]]


def finishes_immediately(run: RemoteExecution) -> Iterator[None]:
    """The default remote: write every declared artifact once and stop.

    Enough for the tests that are about placement, cost or transfer rather than about
    what the stage computes. A test about being interrupted supplies its own script.
    """
    run.log(f"fake: running {run.request.impl} for stage {run.request.stage_id}")
    yield
    for decl in run.request.produces:
        path = run.output(decl.name)
        if decl.kind == "dir":
            for member in decl.members_to_stub or ("member.bin",):
                target = path / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fake")
        else:
            path.write_bytes(b"fake")


@dataclass
class _FakeRun:
    execution: RemoteExecution
    steps: Iterator[None]
    polls: int = 0
    state: RemoteState = "pending"
    detail: str = ""
    synced_at_s: float = 0.0


class FakeAdapter:
    """A provider that exists only here, and always does the same thing twice.

    Told `preempt_after_polls=2, preempt_attempts=(1,)`, the first attempt at a stage is
    taken away on its second poll and every later attempt runs to the end -- which is
    what a preemption-and-resume test needs to be about the resume rather than about
    timing. Told `preempt_after_s`, it uses the script's own simulated clock, so "killed
    after 90 seconds of a 300-second stage" is a fact, not a race.
    """

    def __init__(
        self,
        transfer: Transfer,
        sandbox: Path,
        *,
        name: str = "fake",
        interruptible: bool = True,
        rates: Mapping[str, Rate] | None = None,
        script: RemoteScript = finishes_immediately,
        preempt_after_polls: int | None = None,
        preempt_after_s: float | None = None,
        preempt_attempts: Collection[int] | None = None,
        fail_with: str | None = None,
        ticks_per_poll: int = 1,
        seconds_per_tick: float = 1.0,
    ) -> None:
        self.name = name
        self.interruptible = interruptible
        self._transfer = transfer
        self._rates = dict(rates or {})
        self._script = script
        self._sandbox = sandbox
        self._preempt_after_polls = preempt_after_polls
        self._preempt_after_s = preempt_after_s
        self._preempt_attempts = None if preempt_attempts is None else set(preempt_attempts)
        self._fail_with = fail_with
        self._ticks_per_poll = ticks_per_poll
        self._seconds_per_tick = seconds_per_tick
        self._runs: dict[str, _FakeRun] = {}
        #: Every request this adapter was handed, in order. A test asserts on placement
        #: with it -- which provider got which attempt.
        self.submitted: list[StageRequest] = []

    # --- the protocol -------------------------------------------------------------

    def rate(self, tier: str) -> Rate | None:
        return self._rates.get(tier)

    def submit(self, request: StageRequest) -> RemoteHandle:
        self.submitted.append(request)
        handle = RemoteHandle(
            id=f"{self.name}-{request.stage_id}-{request.attempt}-{len(self.submitted)}",
            provider=self.name,
            tier=request.tier,
        )
        root = self._sandbox / handle.id
        execution = RemoteExecution(
            request=request, root=root, seconds_per_tick=self._seconds_per_tick
        )
        for directory in (
            execution.inputs_dir,
            execution.out_dir,
            execution.work_dir,
            execution.checkpoint_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for artifact, key in request.inputs.items():
            self._transfer.get(key, execution.input(artifact))
        self._transfer.get(request.checkpoint_key, execution.checkpoint_dir)
        execution.log(f"fake: started on {self.name} ({request.tier})")
        self._runs[handle.id] = _FakeRun(execution=execution, steps=self._script(execution))
        return handle

    def poll(self, handle: RemoteHandle) -> Poll:
        run = self._runs[handle.id]
        if run.state in ("succeeded", "failed", "preempted"):
            return self._result(run)
        run.polls += 1
        run.state = "running"
        for _ in range(self._ticks_per_poll):
            if self._taken(run):
                # The box is gone. Nothing more is synced: what survives is what the
                # last interval sync already put in object storage, which is exactly
                # what makes the checkpoint interval a decision with a cost.
                run.state = "preempted"
                run.detail = f"{self.name} reclaimed the machine"
                return self._result(run)
            run.execution.elapsed_s += self._seconds_per_tick
            try:
                next(run.steps)
            except StopIteration:
                self._sync(run)
                run.state = "failed" if self._fail_with else "succeeded"
                run.detail = self._fail_with or ""
                if run.state == "succeeded":
                    self._transfer.put(run.execution.request.outputs_key, run.execution.out_dir)
                return self._result(run)
        self._sync(run)
        return self._result(run)

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        return tuple(self._runs[handle.id].execution.lines[since:])

    def cancel(self, handle: RemoteHandle) -> None:
        run = self._runs.get(handle.id)
        if run is None or run.state in ("succeeded", "failed", "preempted"):
            return
        run.state = "failed"
        run.detail = "cancelled"

    # --- the bits that make it deterministic --------------------------------------

    def _taken(self, run: _FakeRun) -> bool:
        attempt = run.execution.request.attempt
        if self._preempt_attempts is not None and attempt not in self._preempt_attempts:
            return False
        if self._preempt_after_polls is not None and run.polls >= self._preempt_after_polls:
            return True
        return (
            self._preempt_after_s is not None and run.execution.elapsed_s >= self._preempt_after_s
        )

    def _sync(self, run: _FakeRun) -> None:
        """The interval checkpoint sync, on the script's own clock."""
        every = run.execution.request.checkpoint_every_s
        if run.execution.elapsed_s - run.synced_at_s < every:
            return
        run.synced_at_s = run.execution.elapsed_s
        if any(run.execution.checkpoint_dir.iterdir()):
            self._transfer.put(run.execution.request.checkpoint_key, run.execution.checkpoint_dir)
            run.execution.log(f"fake: synced checkpoint at {run.execution.elapsed_s:.0f}s")

    @staticmethod
    def _result(run: _FakeRun) -> Poll:
        metrics: Mapping[str, MetricValue] | None = None
        if run.state == "succeeded":
            metrics = {"fake": True, "ticks": int(run.execution.elapsed_s)}
        return Poll(
            state=run.state,
            billed_s=run.execution.elapsed_s,
            detail=run.detail,
            metrics=metrics,
            summary=f"{run.execution.request.impl} on fake" if run.state == "succeeded" else "",
        )


# --- a local subprocess, through the same five methods -----------------------------


@dataclass
class _Process:
    request: StageRequest
    root: Path
    process: subprocess.Popen[bytes]
    started_at: float
    stop: threading.Event
    syncer: threading.Thread
    log_handle: IO[bytes]
    finished: bool = False
    state: RemoteState = "running"
    detail: str = ""
    billed_s: float = 0.0


class SubprocessAdapter:
    """The stage, in a process on this machine, behind the provider protocol.

    This is the second implementation of `ProviderAdapter`, and it was written to find
    out whether the first one's shape was real. It was: nothing here needed the protocol
    changed. What a container gets from a provider -- a sandbox, its inputs, its
    checkpoint, a log, a signal when the machine is wanted back -- a process gets from
    here, and `poll` distinguishes *preempted* (killed by a signal) from *failed*
    (exited non-zero) the same way a provider's API does, because for a container those
    really are the same two cases.

    A checkpoint syncer runs beside the process on `checkpoint_every_s`, exactly as a
    sidecar would: the process writes into `checkpoint/` and knows nothing about where
    that goes, which is the only arrangement under which a stage that is killed without
    warning still leaves something behind.
    """

    def __init__(
        self,
        transfer: Transfer,
        sandbox: Path,
        *,
        name: str = "subprocess",
        interruptible: bool = False,
        rates: Mapping[str, Rate] | None = None,
        impl_modules: Sequence[str] = (),
        python: str | None = None,
        pipeline_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.interruptible = interruptible
        self._transfer = transfer
        self._sandbox = sandbox
        self._rates = dict(rates or {})
        self._impl_modules = tuple(impl_modules)
        self._python = python or sys.executable
        self._pipeline_dir = pipeline_dir or Path(__file__).resolve().parent
        self._env = dict(env or {})
        self._runs: dict[str, _Process] = {}

    def rate(self, tier: str) -> Rate | None:
        recorded = self._rates.get(tier)
        if recorded is not None:
            return recorded
        listed = provider(self.name)
        return None if listed is None else listed.rate(tier)

    def submit(self, request: StageRequest) -> RemoteHandle:
        handle = RemoteHandle(id=uuid.uuid4().hex, provider=self.name, tier=request.tier)
        root = self._sandbox / handle.id
        for name in ("inputs", "out", "work", "checkpoint"):
            (root / name).mkdir(parents=True, exist_ok=True)
        for artifact, key in request.inputs.items():
            self._transfer.get(key, root / "inputs" / artifact)
        self._transfer.get(request.checkpoint_key, root / "checkpoint")
        spec = root / "stage.json"
        spec.write_text(
            json.dumps(
                {
                    "request": request.to_dict(),
                    "sandbox": str(root),
                    "implModules": list(self._impl_modules),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        log = root / "log.txt"
        log.touch()
        handle_out = log.open("ab")
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell, our own module
            [self._python, "-u", str(self._pipeline_dir / "run_stage.py"), str(spec)],
            stdout=handle_out,
            stderr=subprocess.STDOUT,
            cwd=str(self._pipeline_dir),
            env={**os.environ, **self._env},
        )
        stop = threading.Event()
        syncer = threading.Thread(
            target=self._sync_forever,
            args=(request, root / "checkpoint", stop, request.checkpoint_every_s),
            daemon=True,
        )
        syncer.start()
        self._runs[handle.id] = _Process(
            request=request,
            root=root,
            process=process,
            started_at=time.monotonic(),
            stop=stop,
            syncer=syncer,
            log_handle=handle_out,
        )
        return handle

    def poll(self, handle: RemoteHandle) -> Poll:
        run = self._runs[handle.id]
        if run.finished:
            return self._result(run)
        code = run.process.poll()
        if code is None:
            return Poll(state="running", billed_s=time.monotonic() - run.started_at)
        run.finished = True
        run.billed_s = time.monotonic() - run.started_at
        run.stop.set()
        run.syncer.join(timeout=5.0)
        run.log_handle.close()
        # One last sync, so a stage that finished between syncs does not lose the tail
        # of its checkpoint. A killed one gets nothing extra, which is the point.
        if code == 0:
            self._sync(run.request, run.root / "checkpoint")
            self._transfer.put(run.request.outputs_key, run.root / "out")
            run.state = "succeeded"
        elif code < 0:
            # Killed by a signal: the machine was wanted back. This is the same case a
            # provider reports as an interruption, and it is not a failure.
            run.state = "preempted"
            run.detail = f"the process was killed by signal {-code}"
        else:
            run.state = "failed"
            run.detail = self._tail(run)
        return self._result(run)

    def logs(self, handle: RemoteHandle, *, since: int = 0) -> Sequence[str]:
        path = self._runs[handle.id].root / "log.txt"
        if not path.is_file():
            return ()
        return tuple(path.read_text(encoding="utf-8", errors="replace").splitlines()[since:])

    def cancel(self, handle: RemoteHandle) -> None:
        run = self._runs.get(handle.id)
        if run is None:
            return
        run.stop.set()
        if run.process.poll() is None:
            run.process.terminate()
            try:
                run.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                run.process.kill()
                run.process.wait()
        if not run.log_handle.closed:
            run.log_handle.close()

    # --- helpers -------------------------------------------------------------------

    def _sync_forever(
        self, request: StageRequest, checkpoint: Path, stop: threading.Event, every: float
    ) -> None:
        while not stop.wait(max(0.05, every)):
            self._sync(request, checkpoint)

    def _sync(self, request: StageRequest, checkpoint: Path) -> None:
        if checkpoint.is_dir() and any(checkpoint.iterdir()):
            self._transfer.put(request.checkpoint_key, checkpoint)

    def _result(self, run: _Process) -> Poll:
        metrics: Mapping[str, MetricValue] | None = None
        summary = ""
        if run.state == "succeeded":
            reported = _read_result(run.root / "result.json")
            metrics = reported[0]
            summary = reported[1]
        return Poll(
            state=run.state,
            billed_s=run.billed_s,
            detail=run.detail,
            metrics=metrics,
            summary=summary,
        )

    @staticmethod
    def _tail(run: _Process, lines: int = 20) -> str:
        path = run.root / "log.txt"
        if not path.is_file():
            return f"the stage exited with code {run.process.returncode}"
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        return "\n".join(text) or f"the stage exited with code {run.process.returncode}"


def _read_result(path: Path) -> tuple[dict[str, MetricValue], str]:
    if not path.is_file():
        return {}, ""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}, ""
    metrics: dict[str, MetricValue] = {}
    for key, value in dict(document.get("metrics") or {}).items():
        metrics[str(key)] = value if isinstance(value, bool | int | float | str) else str(value)
    return metrics, str(document.get("summary", ""))
