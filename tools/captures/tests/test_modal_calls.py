"""`modal_calls`: the calls a run spawned, cancelled when it stops short, against fakes.

No Modal here: a fake call answers `get` and `cancel` as `modal.FunctionCall` does (a bare
`TimeoutError` when it has not finished, the call's own exception when it failed), and a
call that is "running" blocks in `get` on a lock, as Modal's blocking `get` waits on a
future, so a real SIGINT or SIGTERM lands while the run is waiting, as on GitHub.
"""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from modal_calls import RunStopped, SpawnedCalls, Stopped

TERMINATE = [{"terminate_containers": True}]


class FakeCall:
    def __init__(
        self,
        result: object = None,
        error: BaseException | None = None,
        running: bool = False,
        cancel_error: Exception | None = None,
        cancel_hangs: bool = False,
    ) -> None:
        self.result, self.error, self.running = result, error, running
        self.cancel_error, self.cancel_hangs = cancel_error, cancel_hangs
        self.cancels: list[dict] = []
        self.release = threading.Event()

    def get(self, timeout: float | None = None) -> object:
        if self.running and not self.release.wait(10):
            raise AssertionError("waited on and never interrupted")
        if self.error is not None:
            raise self.error
        return self.result

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancels.append({"terminate_containers": terminate_containers})
        if self.cancel_hangs:
            self.release.wait(10)
        if self.cancel_error is not None:
            raise self.cancel_error


class FakeFunction:
    def __init__(self, *calls: FakeCall) -> None:
        self.calls = list(calls)
        self.args: list[tuple] = []

    def spawn(self, *args: object) -> FakeCall:
        self.args.append(args)
        return self.calls.pop(0)


def _signal_soon(signum: int, delay: float = 0.2) -> threading.Timer:
    timer = threading.Timer(delay, os.kill, (os.getpid(), signum))
    timer.start()
    return timer


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_cancels_every_outstanding_call_and_exits_non_zero(signum: int) -> None:
    done, waited, queued = FakeCall(result="clip"), FakeCall(running=True), FakeCall(running=True)
    fn = FakeFunction(done, waited, queued)
    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append)
    before = signal.getsignal(signum)
    name = signal.Signals(signum).name
    with pytest.raises(RunStopped, match=f"stopped by {name}; 2 of 2") as raised, calls.guard():
        a, b = calls.spawn("a", fn, 1), calls.spawn("b", fn, 2)
        calls.spawn("c", fn, 3)
        assert calls.get(a) == "clip"
        timer = _signal_soon(signum)
        calls.get(b)  # waits, as the run does, until the signal
    timer.join()
    # A stop, not a KeyboardInterrupt that slipped past: `modal run` exits 0 on those.
    assert isinstance(raised.value, Exception) and isinstance(raised.value.__cause__, Stopped)
    assert done.cancels == []
    assert waited.cancels == queued.cancels == TERMINATE
    assert [r["state"] for r in calls.report()] == ["succeeded", "cancelled", "cancelled"]
    assert calls.outstanding() == []
    assert f"{name}: stopping" in lines
    assert signal.getsignal(signum) is before


def test_a_second_signal_while_cancelling_does_not_cut_the_cancels_short() -> None:
    class Resignals(FakeCall):
        def cancel(self, terminate_containers: bool = False) -> None:
            os.kill(os.getpid(), signal.SIGTERM)  # GitHub's SIGTERM, 7.5 s after its SIGINT
            time.sleep(0.2)
            super().cancel(terminate_containers)

    waited, other = Resignals(running=True), FakeCall(running=True)
    fn = FakeFunction(waited, other)
    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append)
    with pytest.raises(RunStopped, match="SIGINT; 2 of 2"), calls.guard():
        first = calls.spawn("waited", fn)
        calls.spawn("other", fn)
        timer = _signal_soon(signal.SIGINT)
        calls.get(first)
    timer.join()
    assert waited.cancels == other.cancels == TERMINATE
    assert "SIGTERM: already stopping; the cancels go on" in lines


def test_an_exception_cancels_the_outstanding_calls_and_is_raised_as_it_came() -> None:
    collected, running = FakeCall(result=1), FakeCall(running=True)
    fn = FakeFunction(collected, running)
    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append)
    error = KeyError("views")
    with pytest.raises(KeyError) as raised, calls.guard():
        a = calls.spawn("starts camp", fn)
        calls.spawn("clip camp-gentle", fn)
        calls.get(a)
        raise error
    assert raised.value is error
    assert collected.cancels == [] and running.cancels == TERMINATE
    assert lines[0] == "cancelling 1 outstanding Modal call(s) (the run raised KeyError: 'views')"
    assert "  clip camp-gentle: cancelled" in lines


def test_success_cancels_nothing() -> None:
    a, b, never = FakeCall(result=1), FakeCall(result=2), FakeCall(running=True)
    fn = FakeFunction(a, b, never)
    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append)
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    with calls.guard():
        x, y = calls.spawn("x", fn), calls.spawn("y", fn)
        calls.spawn("never collected", fn)
        assert (calls.get(x), calls.get(y)) == (1, 2)
    assert a.cancels == b.cancels == never.cancels == []
    assert calls.failures() == []
    assert lines == ["1 call(s) never collected, left running: never collected"]
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_a_calls_own_error_is_raised_as_it_came_and_recorded() -> None:
    own = RuntimeError("Cosmos: the container exited during its start")
    failing = FakeCall(error=own)
    # The other call's cancel fails too: that is logged, and never raised in place of `own`.
    other = FakeCall(running=True, cancel_error=ConnectionError("modal unreachable"))
    fn = FakeFunction(failing, other)
    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append)
    with pytest.raises(RuntimeError) as raised, calls.guard():
        f = calls.spawn("clip cosmos", fn)
        calls.spawn("clip wan", fn)
        calls.get(f)  # not caught: the run stops on it
    assert raised.value is own
    assert failing.cancels == []  # it finished, on its own; reported, not cancelled
    assert other.cancels == TERMINATE
    detail = "RuntimeError('Cosmos: the container exited during its start')"
    assert calls.failures() == [{"call": "clip cosmos", "detail": detail}]
    assert f"clip cosmos failed: {own!r}" in lines
    assert calls.report()[1]["state"] == "running"  # its cancel was not confirmed
    assert "cancel failed: ConnectionError('modal unreachable')" in calls.report()[1]["detail"]


def test_a_caught_failure_lets_the_run_go_on_and_stays_recorded() -> None:
    fn = FakeFunction(FakeCall(error=ValueError("no frames")), FakeCall(result=b"png"))
    calls = SpawnedCalls(log=lambda line: None)
    with calls.guard():
        bad, good = calls.spawn("sheet a", fn), calls.spawn("sheet b", fn)
        with pytest.raises(ValueError, match="no frames"):
            calls.get(bad)
        assert calls.get(good) == b"png"
    assert calls.failures() == [{"call": "sheet a", "detail": "ValueError('no frames')"}]


def test_a_call_with_no_result_in_time_is_cancelled_and_reported() -> None:
    slow = FakeCall(error=TimeoutError())  # Modal's get: a bare TimeoutError, not finished
    own = FakeCall(error=TimeoutError("The read operation timed out"))  # the call's own
    fn = FakeFunction(slow, own)
    calls = SpawnedCalls(log=lambda line: None)
    with calls.guard():
        s, o = calls.spawn("clip slow", fn), calls.spawn("starts pumpkin", fn)
        with pytest.raises(TimeoutError, match="clip slow: no result within 3600 s; cancelled"):
            calls.get(s, timeout=3600)
        with pytest.raises(TimeoutError, match="The read operation timed out"):
            calls.get(o, timeout=3600)
    assert slow.cancels == TERMINATE and own.cancels == []
    assert [f["call"] for f in calls.failures()] == ["clip slow", "starts pumpkin"]
    assert calls.failures()[0]["detail"] == "no result within 3600 s"


def test_cancelling_is_bounded_in_time() -> None:
    hung, quick = FakeCall(running=True, cancel_hangs=True), FakeCall(running=True)
    fn = FakeFunction(hung, quick)
    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append, cancel_timeout_s=0.3)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="boom"), calls.guard():
        calls.spawn("hung", fn)
        calls.spawn("quick", fn)
        raise RuntimeError("boom")
    assert time.monotonic() - started < 2.0
    hung.release.set()
    assert hung.cancels == quick.cancels == TERMINATE
    assert calls.outstanding() == ["hung"]
    assert "  hung: no answer to the cancel within 0.3 s" in lines


def test_a_signal_during_a_spawn_lands_once_the_call_is_tracked() -> None:
    call = FakeCall(running=True)

    class SignalledMidSpawn(FakeFunction):
        def spawn(self, *args: object) -> FakeCall:
            os.kill(os.getpid(), signal.SIGTERM)  # while Modal is still creating the call
            time.sleep(0.05)
            return super().spawn(*args)

    lines: list[str] = []
    calls = SpawnedCalls(log=lines.append)
    with pytest.raises(RunStopped, match="SIGTERM; 1 of 1"), calls.guard():
        calls.spawn("clip", SignalledMidSpawn(call))
        pytest.fail("the stop lands as the spawn returns")
    assert call.cancels == TERMINATE
    assert "SIGTERM: stopping once the call being spawned is tracked" in lines


def test_a_stop_swallowed_on_the_way_out_still_stops_the_run() -> None:
    """Something between the handler and the guard catching the stop (a library's own
    `except BaseException`) does not turn it into a success."""
    waited, queued = FakeCall(running=True), FakeCall(running=True)
    fn = FakeFunction(waited, queued)
    calls = SpawnedCalls(log=lambda line: None)
    with pytest.raises(RunStopped, match="SIGTERM"), calls.guard():
        first = calls.spawn("first", fn)
        calls.spawn("second", fn)
        timer = _signal_soon(signal.SIGTERM)
        try:
            calls.get(first)
        except BaseException:  # noqa: BLE001 - the swallow under test
            swallowed = True
    timer.join()
    assert swallowed
    assert waited.cancels == queued.cancels == TERMINATE
