"""The machine a remote call ran on, and what a call leaves behind on it.

Two jobs, both for `remote.execute`, both best-effort (nothing here may fail a stage):

* **What the machine was doing while the stage ran** (`HostWatch`). Two block parts of one
  fan-out, same code, same cameras, similar gaussian counts, trained at 18 and 3 it/s on
  two L4 containers (job 33bc1bff), and nothing recorded could say whether the slow one's
  GPU was slow, its CPU was starved, or it did different work. A GPU function that asks
  Modal for no CPU is reserved 0.125 of a core and bursts into whatever its neighbours
  leave; gsplat's trainer feeds every step from four DataLoader workers decoding JPEGs,
  and the main process launches the step's kernels from Python -- a starved container
  trains at the speed of its CPU share, not its GPU. So a sampler beside the stage reads,
  every `every_s`: the GPU (`nvidia-smi`: name, utilisation, SM clock, temperature,
  power, throttle reasons), the load average, the cgroup's CPU throttling and pressure,
  and the process tree's CPU seconds. A starved call shows low GPU utilisation with
  throttled or pressured CPU; a slow GPU shows high utilisation at a low clock; a call
  that did more work shows neither.
* **What a call leaves behind** (`Leftovers`). Modal keeps a container warm between
  calls, so a child process or a thread a call forgot is still running in the next call
  -- a block part landing on the head's container would share its CPU with whatever the
  head left. `execute` snapshots the process's children and threads when it starts and,
  when it ends, kills every child process that was not there before and reports every
  thread that was not there before and is still alive.

Linux `/proc` and cgroup files, read if they are there; anywhere else (macOS, a sandbox
without them) the figures are simply absent.
"""

from __future__ import annotations

import contextlib
import os
import resource
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["HostWatch", "Leftovers", "cpu_seconds", "sample"]

Sample = Mapping[str, float | str]

#: `nvidia-smi --query-gpu` fields, in the order `_gpu` reads them back.
GPU_FIELDS = (
    "name",
    "utilization.gpu",
    "clocks.sm",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
)
#: Asked for separately: the field was renamed (`clocks_event_reasons.active`) and a
#: query naming a field the driver does not know fails whole.
THROTTLE_FIELDS = ("clocks_throttle_reasons.active", "clocks_event_reasons.active")


def _number(text: str) -> float | None:
    text = text.strip()
    try:
        return float(int(text, 16)) if text.lower().startswith("0x") else float(text)
    except ValueError:
        return None


def _smi(query: str) -> list[str] | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - absolute path from which, fixed argv
            [smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return [cell.strip() for cell in done.stdout.strip().splitlines()[0].split(",")]


def _gpu() -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    cells = _smi(",".join(GPU_FIELDS))
    if cells is None or len(cells) != len(GPU_FIELDS):
        return out
    out["gpu"] = cells[0]
    for name, cell in zip(
        ("gpuUtil", "smMHz", "smMaxMHz", "tempC", "powerW"), cells[1:], strict=True
    ):
        value = _number(cell)
        if value is not None:
            out[name] = value
    for query in THROTTLE_FIELDS:
        reasons = _smi(query)
        if reasons:
            value = _number(reasons[0])
            if value is not None:
                out["throttle"] = value
                break
    return out


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _cgroup() -> dict[str, float]:
    """cgroup v2 (`cpu.stat`, `cpu.max`) or v1 (`cpu/cpu.stat`, `cfs_quota_us`)."""
    out: dict[str, float] = {}
    stat = _read("/sys/fs/cgroup/cpu.stat")
    if stat is not None:
        fields = dict(line.split(None, 1) for line in stat.splitlines() if " " in line)
        if "throttled_usec" in fields:
            out["throttledS"] = float(fields["throttled_usec"]) / 1e6
        if "usage_usec" in fields:
            out["cgroupCpuS"] = float(fields["usage_usec"]) / 1e6
        quota = (_read("/sys/fs/cgroup/cpu.max") or "").split()
        if len(quota) == 2 and quota[0] != "max":
            out["cpuQuota"] = float(quota[0]) / float(quota[1])
        return out
    stat = _read("/sys/fs/cgroup/cpu/cpu.stat")
    if stat is not None:
        fields = dict(line.split(None, 1) for line in stat.splitlines() if " " in line)
        if "throttled_time" in fields:
            out["throttledS"] = float(fields["throttled_time"]) / 1e9
        limit = _number(_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or "")
        period = _number(_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or "")
        if limit is not None and period and limit > 0:
            out["cpuQuota"] = limit / period
    return out


def _pressure() -> float | None:
    """CPU pressure (`some avg10`): the share of the last 10 s something waited for CPU."""
    text = _read("/proc/pressure/cpu")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("some"):
            for item in line.split()[1:]:
                key, _, value = item.partition("=")
                if key == "avg10":
                    return _number(value)
    return None


def sample() -> dict[str, float | str]:
    """One look at the machine. Every key is optional."""
    out: dict[str, float | str] = {}
    with contextlib.suppress(OSError):
        out["load"] = os.getloadavg()[0]
    out.update(_cgroup())
    psi = _pressure()
    if psi is not None:
        out["psiCpu"] = psi
    out.update(_gpu())
    return out


def cpu_seconds() -> float:
    """This process's CPU seconds and its waited-for children's (the trainer, once it has
    exited), user and system."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


@dataclass
class _Series:
    count: int = 0
    total: float = 0.0
    low: float = float("inf")
    high: float = float("-inf")
    first: float = 0.0
    last: float = 0.0

    def add(self, value: float) -> None:
        if self.count == 0:
            self.first = value
        self.count += 1
        self.total += value
        self.low = min(self.low, value)
        self.high = max(self.high, value)
        self.last = value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class HostWatch:
    """Samples the machine beside a stage, on its own thread; stopped and joined by the
    `with` that started it, however the stage ends.

    `summary()` is flat numbers (`phases.flat`'s form) plus the GPU's name:

    * `cpus` -- what `os.cpu_count()` says, which in a container is often the *host's*
      cores (every thread pool sized from it oversubscribes a small share);
    * `cpuQuota` -- the cgroup's hard limit in cores, when it has one;
    * `cpuS`, `cores` -- CPU seconds of this process and its exited children over the
      watch, and that divided by the wall time: the cores the call actually got;
      `cgroupCores` the same from the cgroup's own count, running children included;
    * `throttledS` -- seconds the cgroup was throttled over the watch;
    * `psiCpu`, `load` (mean), `loadMax`;
    * `gpuUtil` (mean), `gpuUtilMin`, `smMHz` (mean), `smMHzMin`, `smMaxMHz`, `tempC`
      (max), `powerW` (mean), `throttle` (nvidia-smi's clock throttle reasons seen,
      OR-ed: 0x1 idle, 0x4 software power cap, 0x8 hardware slowdown, 0x20/0x40
      software/hardware thermal, 0x80 power brake);
    * `samples`.
    """

    def __init__(
        self,
        every_s: float = 30.0,
        *,
        probe: Callable[[], Sample] = sample,
        clock: Callable[[], float] = time.monotonic,
        cpu: Callable[[], float] = cpu_seconds,
    ) -> None:
        self._every_s = max(0.05, every_s)
        self._probe = probe
        self._clock = clock
        self._cpu = cpu
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._series: dict[str, _Series] = {}
        self._gpu = ""
        self._throttle = 0
        self._lock = threading.Lock()
        self._began = 0.0
        self._ended: float | None = None
        self._cpu_began = 0.0
        self._cpu_ended: float | None = None

    def __enter__(self) -> HostWatch:
        self._began = self._clock()
        self._cpu_began = self._cpu()
        self._take()
        self._thread = threading.Thread(target=self._run, daemon=True, name="host-watch")
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        self._take()
        self._ended = self._clock()
        self._cpu_ended = self._cpu()

    def _run(self) -> None:
        while not self._stop.wait(self._every_s):
            self._take()

    def _take(self) -> None:
        try:
            taken = dict(self._probe())
        except Exception:  # watching must never fail the stage
            return
        with self._lock:
            for name, value in taken.items():
                if name == "gpu":
                    self._gpu = self._gpu or str(value)
                elif name == "throttle" and isinstance(value, int | float):
                    self._throttle |= int(value)
                elif isinstance(value, int | float) and not isinstance(value, bool):
                    self._series.setdefault(name, _Series()).add(float(value))

    @property
    def gpu(self) -> str:
        return self._gpu

    def summary(self) -> dict[str, float]:
        with self._lock:
            series = dict(self._series)
        out: dict[str, float] = {}
        cpus = os.cpu_count()
        if cpus:
            out["cpus"] = float(cpus)
        if "cpuQuota" in series:
            out["cpuQuota"] = series["cpuQuota"].last
        wall = (self._ended if self._ended is not None else self._clock()) - self._began
        used = (self._cpu_ended if self._cpu_ended is not None else self._cpu()) - self._cpu_began
        out["cpuS"] = max(0.0, used)
        if wall > 0:
            out["cores"] = max(0.0, used) / wall
        if "throttledS" in series:
            out["throttledS"] = max(0.0, series["throttledS"].last - series["throttledS"].first)
        if "cgroupCpuS" in series and wall > 0:
            # The whole container's, running children included (`cores` counts a child
            # only once it has exited and been waited for).
            spent = series["cgroupCpuS"].last - series["cgroupCpuS"].first
            out["cgroupCores"] = max(0.0, spent) / wall
        for name, key, how in (
            ("psiCpu", "psiCpu", "mean"),
            ("load", "load", "mean"),
            ("loadMax", "load", "max"),
            ("gpuUtil", "gpuUtil", "mean"),
            ("gpuUtilMin", "gpuUtil", "min"),
            ("smMHz", "smMHz", "mean"),
            ("smMHzMin", "smMHz", "min"),
            ("smMaxMHz", "smMaxMHz", "max"),
            ("tempC", "tempC", "max"),
            ("powerW", "powerW", "mean"),
        ):
            if key in series:
                one = series[key]
                out[name] = one.mean if how == "mean" else one.low if how == "min" else one.high
        if self._throttle:
            out["throttle"] = float(self._throttle)
        out["samples"] = float(max((s.count for s in series.values()), default=0))
        return {name: round(value, 2) for name, value in out.items()}


# --- what a call leaves behind -------------------------------------------------------


def _children(parent: int) -> set[int]:
    """Every live descendant of `parent`, from `/proc/*/stat`; empty without `/proc`."""
    parents: dict[int, int] = {}
    with contextlib.suppress(OSError):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            stat = _read(f"/proc/{entry.name}/stat")
            if not stat:
                continue
            # `pid (comm) state ppid ...`: comm may hold spaces and parentheses.
            rest = stat.rsplit(")", 1)[-1].split()
            if len(rest) >= 2 and rest[0] != "Z":
                with contextlib.suppress(ValueError):
                    parents[int(entry.name)] = int(rest[1])
    found: set[int] = set()
    frontier = [parent]
    while frontier:
        here = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == here and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


@dataclass
class Leftovers:
    """What was running when a call began, so what it leaves can be told apart."""

    processes: set[int] = field(default_factory=set)
    #: The thread objects themselves, not their idents: an ident is reused once its
    #: thread has ended, and a new thread must not pass for an old one.
    threads: set[threading.Thread] = field(default_factory=set)

    @staticmethod
    def now() -> Leftovers:
        return Leftovers(processes=_children(os.getpid()), threads=set(threading.enumerate()))

    def reap(self, *, grace_s: float = 5.0) -> tuple[list[int], list[str]]:
        """Kill every descendant process started since `now()`; name every thread started
        since then that is still alive (a thread cannot be killed from outside, so the
        fix for one is at its source). Returns (killed pids, live thread names)."""
        extra = sorted(_children(os.getpid()) - self.processes)
        for pid in extra:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while extra and time.monotonic() < deadline:
            if not (_children(os.getpid()) & set(extra)):
                break
            time.sleep(0.05)
        for pid in sorted(_children(os.getpid()) & set(extra)):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while extra and _children(os.getpid()) & set(extra) and time.monotonic() < deadline:
            time.sleep(0.05)
        for pid in extra:
            # Our own children stay zombies until waited; a grandchild is reaped by init.
            # Neither blocks: WNOHANG, and ChildProcessError for one that is not ours.
            with contextlib.suppress(ChildProcessError, OSError):
                os.waitpid(pid, os.WNOHANG)
        threads = [t.name for t in threading.enumerate() if t not in self.threads and t.is_alive()]
        return extra, threads
