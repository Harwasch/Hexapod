"""What `ModalAdapter` can be tested on without a Modal account.

These tests do not prove the adapter works -- nothing in this repository can, and the
module says so in its own docstring. They pin down the part that is testable from here:
that a Modal outcome is classified into the right `RemoteState`. That is worth pinning
because the previous version of this file got it exactly backwards, reporting a healthy
running stage as `failed` on its very first poll, and no amount of reading will stop that
returning by accident.

The fakes below mirror the real hierarchy rather than merely borrowing its names:
`modal.exception.FunctionTimeoutError` and `OutputExpiredError` really are subclasses of
`modal.exception.TimeoutError`, which really does *not* inherit from the builtin. Both
facts are the whole reason `_STATES` matches on the exact qualified class name, so a fake
that flattened them would test nothing.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from artifacts import ArtifactDecl
from cloud import RemoteHandle, StageRequest
from modal_adapter import GPU_NAMES, MAX_LOG_LINES, ModalAdapter


def modal_exception(name: str, base: type[BaseException] = Exception) -> type[BaseException]:
    """A stand-in whose qualified name is the one `_STATES` keys on."""
    kind = type(name, (base,), {})
    kind.__module__ = "modal.exception"
    return kind


#: Deliberately built in the SDK's own shape: a `modal.Error` subclass, not a builtin one.
ModalTimeout = modal_exception("TimeoutError")
FunctionTimeout = modal_exception("FunctionTimeoutError", ModalTimeout)
OutputExpired = modal_exception("OutputExpiredError", ModalTimeout)
InternalFailure = modal_exception("InternalFailure")


class FakeLogs:
    def __init__(self, entries: list[Any] | Exception) -> None:
        self._entries = entries

    def fetch(self) -> list[Any]:
        if isinstance(self._entries, Exception):
            raise self._entries
        return self._entries


class FakeEntry:
    """A `LogEntry` as far as this adapter uses one: a `.message`."""

    def __init__(self, message: str) -> None:
        self.message = message


class FakeCall:
    """One `FunctionCall`: an outcome to hand back, or an exception to raise."""

    def __init__(self, outcome: Any = None, raises: BaseException | None = None) -> None:
        self.object_id = "fc-1"
        self._outcome = outcome
        self._raises = raises
        self.logs = FakeLogs([])
        self.cancelled_with: dict[str, Any] | None = None

    def get(self, timeout: float | None = None, *, index: int = 0) -> Any:
        assert timeout == 0, "poll must not block the supervisor"
        if self._raises is not None:
            raise self._raises
        return self._outcome

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled_with = {"terminate_containers": terminate_containers}


class FakeFunction:
    def __init__(self, call: FakeCall) -> None:
        self._call = call
        self.spawned: Any = None

    def spawn(self, payload: Any) -> FakeCall:
        self.spawned = payload
        return self._call


def request() -> StageRequest:
    return StageRequest(
        recipe="r",
        run_id="run",
        stage_id="train",
        impl="gsplat",
        attempt=1,
        tier="l4",
        preemptible=True,
        params={},
        inputs={},
        produces=(ArtifactDecl("out.ply"),),
        checkpoint_key="runs/run/train/checkpoint",
        outputs_key="runs/run/train/transfer/out",
    )


def adapter_over(call: FakeCall) -> tuple[ModalAdapter, RemoteHandle, FakeFunction]:
    """A submitted stage, with the Modal lookup replaced by a fake."""
    function = FakeFunction(call)
    adapter = ModalAdapter("twin")
    adapter._function = lambda: function  # type: ignore[method-assign]
    return adapter, adapter.submit(request()), function


def test_submit_sends_the_request_dict_and_keeps_the_call_id() -> None:
    call = FakeCall()
    _, handle, function = adapter_over(call)
    assert handle.id == "fc-1"
    assert handle.provider == "modal"
    assert handle.tier == "l4"
    assert function.spawned["stageId"] == "train"


def test_a_running_stage_is_running_and_not_failed() -> None:
    """The regression that matters: `modal.exception.TimeoutError` means "not yet".

    It does not inherit from the builtin `TimeoutError`, so the old `except TimeoutError`
    never caught it and every healthy stage was dead-lettered on its first poll.
    """
    adapter, handle, _ = adapter_over(FakeCall(raises=ModalTimeout()))
    poll = adapter.poll(handle)
    assert poll.state == "running"
    assert poll.billed_s >= 0.0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # Both subclass Modal's TimeoutError; an isinstance check would call them running.
        (FunctionTimeout("6h"), "failed"),
        (OutputExpired(), "failed"),
        (InternalFailure("worker lost"), "preempted"),
        # A stage whose own code times out. Same class name as Modal's, different module,
        # and it must terminate the poll loop rather than be mistaken for "no result yet".
        (TimeoutError("the stage's own timeout"), "failed"),
        (RuntimeError("boom"), "failed"),
    ],
)
def test_terminal_outcomes_are_classified_by_exact_class(
    error: BaseException, expected: str
) -> None:
    adapter, handle, _ = adapter_over(FakeCall(raises=error))
    assert adapter.poll(handle).state == expected


def test_a_verdict_is_remembered_rather_than_re_polled() -> None:
    call = FakeCall(raises=InternalFailure("worker lost"))
    adapter, handle, _ = adapter_over(call)
    assert adapter.poll(handle).state == "preempted"
    call._raises = None  # a second get() would now "succeed"; it must not be asked
    assert adapter.poll(handle).state == "preempted"


def test_success_carries_metrics_and_summary_and_drops_unserialisable_ones() -> None:
    outcome = {"metrics": {"psnr": 27.5, "steps": 7000, "curve": [1, 2]}, "summary": "7k steps"}
    adapter, handle, _ = adapter_over(FakeCall(outcome=outcome))
    poll = adapter.poll(handle)
    assert poll.state == "succeeded"
    assert poll.summary == "7k steps"
    assert poll.metrics == {"psnr": 27.5, "steps": 7000}


def test_cancel_terminates_the_container_rather_than_leaving_it_billing() -> None:
    call = FakeCall(raises=ModalTimeout())
    adapter, handle, _ = adapter_over(call)
    adapter.cancel(handle)
    assert call.cancelled_with == {"terminate_containers": True}
    assert adapter.poll(handle).state == "failed"


def test_cancel_is_safe_on_a_finished_stage() -> None:
    call = FakeCall(outcome={})
    adapter, handle, _ = adapter_over(call)
    assert adapter.poll(handle).state == "succeeded"
    adapter.cancel(handle)
    assert call.cancelled_with is None


def test_logs_are_lines_and_since_is_an_index_into_them() -> None:
    call = FakeCall(raises=ModalTimeout())
    adapter, handle, _ = adapter_over(call)
    # One entry carrying three lines, because a LogEntry message is not one line.
    call.logs = FakeLogs([FakeEntry("a\nb\n"), FakeEntry("c")])
    assert list(adapter.logs(handle)) == ["a", "b", "c"]
    assert list(adapter.logs(handle, since=2)) == ["c"]
    assert list(adapter.logs(handle, since=9)) == []


def test_a_failed_log_fetch_is_not_a_failed_stage() -> None:
    call = FakeCall(raises=ModalTimeout())
    adapter, handle, _ = adapter_over(call)
    call.logs = FakeLogs([FakeEntry("kept")])
    assert list(adapter.logs(handle)) == ["kept"]
    call.logs = FakeLogs(RuntimeError("log service down"))
    assert list(adapter.logs(handle)) == ["kept"]
    assert adapter.poll(handle).state == "running"


def test_logs_are_bounded() -> None:
    call = FakeCall(raises=ModalTimeout())
    adapter, handle, _ = adapter_over(call)
    call.logs = FakeLogs([FakeEntry("\n".join(str(n) for n in range(MAX_LOG_LINES * 3)))])
    lines = adapter.logs(handle)
    assert len(lines) == MAX_LOG_LINES
    assert lines[-1] == str(MAX_LOG_LINES * 3 - 1)


def test_logs_of_an_unknown_handle_are_empty() -> None:
    adapter = ModalAdapter("twin")
    assert adapter.logs(RemoteHandle(id="nope", provider="modal", tier="l4")) == ()


def test_gpu_names_are_modals_identifiers() -> None:
    """`A10G` is AWS's name for the instance; Modal's tier is `A10`.

    Kept as a test rather than a comment because the mapping is what the remote half
    passes to `gpu=`, and a wrong string there fails at deploy time on a machine that is
    not this one.
    """
    assert "a10g" not in GPU_NAMES
    assert GPU_NAMES["a10"] == "A10"
    assert GPU_NAMES["l4"] == "L4"
    # Bare "A100" is Modal's 40 GB part; the table's $2.50 is the 80 GB price.
    assert GPU_NAMES["a100"] == "A100-80GB"
    assert "A100" not in set(GPU_NAMES.values())
    assert set(GPU_NAMES.values()) == {
        "T4",
        "L4",
        "A10",
        "L40S",
        "A100-40GB",
        "A100-80GB",
        "H100",
        "H200",
        "B200",
        "B300",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("modal") is not None,
    reason="modal is installed here, so the missing-dependency path cannot be exercised",
)
def test_reaching_for_modal_without_the_package_explains_itself() -> None:
    """The library is deliberately not a dependency; the failure must say why.

    Skipped rather than faked where `modal` happens to be installed, because the point is
    the real import failing. Calling `submit` there would hit the live API, which is
    exactly the thing this repository has never done and must not start doing in a test.
    """
    adapter = ModalAdapter("twin")
    with pytest.raises(RuntimeError, match="deliberately not a dependency"):
        adapter.submit(request())
