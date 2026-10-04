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
import sys
import time
import types
from dataclasses import replace
from typing import Any

import pytest

from artifacts import ArtifactDecl
from cloud import Reattachable, RemoteHandle, StageRequest
from contracts import FANOUT_PARAM
from modal_adapter import (
    GPU_FALLBACKS,
    GPU_NAMES,
    GPU_RESERVATION,
    MAX_LOG_LINES,
    ModalAdapter,
    fallback_tier,
    reservation_rate,
    tier_of_gpu,
)
from providers import Rate, provider


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
    adapter._function = lambda tier: function  # type: ignore[method-assign]
    return adapter, adapter.submit(request()), function


def test_the_function_looked_up_is_the_one_for_the_request_s_tier() -> None:
    """One deployed function per tier, because Modal fixes a function's GPU at
    decoration time. `infra/modal/app.py` registers exactly these names."""
    asked: list[str] = []

    def lookup(tier: str) -> FakeFunction:
        asked.append(f"run_stage_{tier}")
        return FakeFunction(FakeCall())

    adapter = ModalAdapter("twin")
    adapter._function = lookup  # type: ignore[method-assign]
    adapter.submit(request())
    assert asked == ["run_stage_l4"]


def test_submit_sends_the_request_dict_and_keeps_the_call_id() -> None:
    call = FakeCall()
    _, handle, function = adapter_over(call)
    assert handle.id == "fc-1"
    assert handle.provider == "modal"
    assert handle.tier == "l4"
    assert function.spawned["stageId"] == "train"


def test_a_running_stage_is_running_and_not_failed() -> None:
    """The regression that matters, and it happened: a bare builtin `TimeoutError()` is
    what Modal 1.5.5's zero-timeout `get` raises for a call with no output yet
    (`modal/_functions.py`, `poll_function`). Classifying it as a failure dead-lettered
    the first real GPU run on its first poll."""
    adapter, handle, _ = adapter_over(FakeCall(raises=TimeoutError()))
    poll = adapter.poll(handle)
    assert poll.state == "running"
    assert poll.billed_s >= 0.0


def test_a_call_whose_stage_has_not_printed_its_first_line_is_pending() -> None:
    """A container that crash-loops on import looks exactly like a running stage to
    `get`; only the log tells them apart, and `CloudRunner` gives up on `pending`."""
    call = FakeCall(raises=TimeoutError())
    adapter, handle, _ = adapter_over(call)
    call.logs = FakeLogs([FakeEntry("Traceback (most recent call last):\nIndexError: 2")])
    adapter.logs(handle)
    assert adapter.poll(handle).state == "pending"
    call.logs = FakeLogs([FakeEntry("run_stage: gsplat for stage 'train', attempt 1")])
    adapter.logs(handle)
    assert adapter.poll(handle).state == "running"


def test_started_is_remembered_after_the_start_line_scrolls_away() -> None:
    call = FakeCall(raises=TimeoutError())
    adapter, handle, _ = adapter_over(call)
    noise = "\n".join(str(n) for n in range(MAX_LOG_LINES * 3))
    call.logs = FakeLogs([FakeEntry("run_stage: gsplat for stage 'train'"), FakeEntry(noise)])
    adapter.logs(handle)
    assert adapter.poll(handle).state == "running"
    call.logs = FakeLogs([FakeEntry(noise)])
    adapter.logs(handle)
    assert adapter.poll(handle).state == "running"


def test_an_unreadable_log_never_makes_a_stage_look_stuck() -> None:
    call = FakeCall(raises=TimeoutError())
    adapter, handle, _ = adapter_over(call)
    call.logs = FakeLogs(RuntimeError("log service down"))
    adapter.logs(handle)
    assert adapter.poll(handle).state == "running"


def test_modals_own_timeout_also_means_not_yet() -> None:
    adapter, handle, _ = adapter_over(FakeCall(raises=ModalTimeout()))
    assert adapter.poll(handle).state == "running"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # Both subclass Modal's TimeoutError; an isinstance check would call them running.
        (FunctionTimeout("6h"), "failed"),
        (OutputExpired(), "failed"),
        (InternalFailure("worker lost"), "preempted"),
        # A stage whose own code times out arrives as this, because `remote.execute`
        # converts it in the container (see test_remote); it must end the poll loop.
        (RuntimeError("the stage timed out: TimeoutError('read')"), "failed"),
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
    # Not a verdict either way: "kept" is no start line, so the call is still pending.
    assert adapter.poll(handle).state not in ("failed", "succeeded", "preempted")


def test_logs_are_bounded_per_call_and_never_skip_a_line() -> None:
    call = FakeCall(raises=ModalTimeout())
    adapter, handle, _ = adapter_over(call)
    call.logs = FakeLogs([FakeEntry("\n".join(str(n) for n in range(MAX_LOG_LINES * 3)))])
    first = adapter.logs(handle)
    assert len(first) == MAX_LOG_LINES
    assert first[0] == "0"
    # The caller's cursor advances by what it got; the backlog arrives in order.
    second = adapter.logs(handle, since=len(first))
    assert second[0] == str(MAX_LOG_LINES)


def test_since_keeps_indexing_the_whole_log_once_it_outgrows_the_held_tail() -> None:
    # A pose solve logs far more than MAX_LOG_LINES; the lines after that must still
    # reach the caller (the finished cameras' live line is one of the last).
    call = FakeCall(raises=ModalTimeout())
    adapter, handle, _ = adapter_over(call)
    history = [str(n) for n in range(MAX_LOG_LINES + 10)]
    call.logs = FakeLogs([FakeEntry("\n".join(history))])
    cursor = 0
    while lines := adapter.logs(handle, since=cursor):
        cursor += len(lines)
    assert cursor == len(history)
    history += ["live-cameras: final"]
    call.logs = FakeLogs([FakeEntry("\n".join(history))])
    assert list(adapter.logs(handle, since=cursor)) == ["live-cameras: final"]
    # A failed fetch answers from the held tail, still by the whole log's index.
    call.logs = FakeLogs(RuntimeError("log service down"))
    assert list(adapter.logs(handle, since=cursor)) == ["live-cameras: final"]
    assert list(adapter.logs(handle, since=cursor + 1)) == []


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


# --- what a call is billed, and on which GPU ---------------------------------------------


def finished(adapter: ModalAdapter, handle: RemoteHandle, *, wall_s: float) -> None:
    """Make the call `wall_s` old by the runner's clocks: submitted that long ago."""
    run = adapter._calls[handle.id]
    run.started_at -= wall_s
    run.submitted_at = 1_000_000.0


def test_a_cold_call_is_billed_from_its_container_s_start_not_from_the_submit() -> None:
    """Job 33bc1bff's b1 waited 1,697 s for an L4 and the wall-time proxy billed all of it.
    Modal bills from the container's start; the wait is reported apart, as `queue_s`."""
    outcome = {
        "metrics": {
            "containerCall": 1,
            "containerStartedAt": 1_001_697.0,
            "containerBootedAt": 1_001_700.0,
            "remoteEnteredAt": 1_001_705.0,
            "remoteFinishedAt": 1_003_365.0,
        }
    }
    adapter, handle, _ = adapter_over(FakeCall(outcome=outcome))
    finished(adapter, handle, wall_s=3_372.0)

    poll = adapter.poll(handle)

    assert poll.billing == "container"
    assert poll.billed_s == pytest.approx(1_668.0)
    assert poll.queue_s == pytest.approx(1_697.0)


def test_without_the_sandbox_s_start_a_cold_call_is_billed_from_its_boot() -> None:
    outcome = {
        "metrics": {
            "containerCall": 1,
            "containerBootedAt": 1_000_010.0,
            "remoteEnteredAt": 1_000_012.0,
            "remoteFinishedAt": 1_000_110.0,
        }
    }
    adapter, handle, _ = adapter_over(FakeCall(outcome=outcome))
    finished(adapter, handle, wall_s=115.0)

    poll = adapter.poll(handle)

    assert poll.billed_s == pytest.approx(100.0) and poll.queue_s == pytest.approx(10.0)


def test_a_warm_call_is_billed_from_entering_the_function() -> None:
    """The container's start was the call before's; this one starts at its own entry."""
    outcome = {
        "metrics": {
            "containerCall": 2,
            "containerStartedAt": 999_000.0,
            "remoteEnteredAt": 1_000_001.0,
            "remoteFinishedAt": 1_000_051.0,
        }
    }
    adapter, handle, _ = adapter_over(FakeCall(outcome=outcome))
    finished(adapter, handle, wall_s=60.0)

    poll = adapter.poll(handle)

    assert poll.billed_s == pytest.approx(50.0) and poll.queue_s == pytest.approx(1.0)


def test_the_container_s_figure_never_exceeds_the_runner_s_wall_time() -> None:
    """Clock skew between the two machines can only shrink a bill, never grow it."""
    outcome = {
        "metrics": {
            "containerCall": 1,
            "containerStartedAt": 999_000.0,
            "remoteEnteredAt": 999_010.0,
            "remoteFinishedAt": 1_000_500.0,
        }
    }
    adapter, handle, _ = adapter_over(FakeCall(outcome=outcome))
    finished(adapter, handle, wall_s=400.0)

    poll = adapter.poll(handle)

    assert poll.billed_s == pytest.approx(400.0) and poll.queue_s == 0.0


def test_a_call_that_reports_no_clocks_keeps_the_proxy_and_says_so() -> None:
    adapter, handle, _ = adapter_over(FakeCall(outcome={"metrics": {"psnr": 27.0}}))
    finished(adapter, handle, wall_s=42.0)

    poll = adapter.poll(handle)

    assert poll.billing == "wall-proxy"
    assert poll.billed_s == pytest.approx(42.0, abs=1.0) and poll.queue_s is None
    running, running_handle, _ = adapter_over(FakeCall(raises=TimeoutError()))
    assert running.poll(running_handle).billing == "wall-proxy"


def part_request(tier: str = "l4") -> StageRequest:
    return replace(request(), tier=tier, params={FANOUT_PARAM: {"role": "part", "part": "b0"}})


def test_a_part_may_start_on_the_first_free_gpu_of_its_fallback_list() -> None:
    """A part waits on no one GPU type: `run_stage_l4_fallback` is deployed with
    `gpu=["L4", "L40S"]`, and the call is priced at what the container says it got."""
    looked_up: list[str] = []
    call = FakeCall(
        outcome={"metrics": {"remoteGpu": "NVIDIA L40S", "containerCall": 1}, "summary": ""}
    )

    def lookup(tier: str) -> FakeFunction:
        looked_up.append(tier)
        return FakeFunction(call)

    adapter = ModalAdapter("twin")
    adapter._function = lookup  # type: ignore[method-assign]
    handle = adapter.submit(part_request())
    poll = adapter.poll(handle)

    assert looked_up == [fallback_tier("l4")] == ["l4_fallback"]
    assert poll.tier == "l40s"
    # The head, the join and a single run keep the tier they asked for.
    adapter.submit(request())
    assert looked_up[-1] == "l4"
    assert GPU_FALLBACKS["l4"][0] == "l4", "the asked-for tier stays first choice"


def test_a_part_on_its_own_tier_is_priced_at_it_and_an_unknown_gpu_at_the_dearest() -> None:
    for gpu, expected in (("NVIDIA L4", "l4"), ("Some Future GPU", "l40s")):
        call = FakeCall(outcome={"metrics": {"remoteGpu": gpu}})
        adapter = ModalAdapter("twin")
        adapter._function = lambda tier, call=call: FakeFunction(call)  # type: ignore[method-assign,misc]
        assert adapter.poll(adapter.submit(part_request())).tier == expected


def test_a_deployment_without_the_fallback_functions_runs_the_part_on_its_tier() -> None:
    not_found = modal_exception("NotFoundError")
    looked_up: list[str] = []

    class Missing:
        def spawn(self, payload: Any) -> Any:
            raise not_found("Lookup failed for Function 'run_stage_l4_fallback'")

    def lookup(tier: str) -> Any:
        looked_up.append(tier)
        return Missing() if tier.endswith("_fallback") else FakeFunction(FakeCall())

    adapter = ModalAdapter("twin")
    adapter._function = lookup  # type: ignore[method-assign]
    adapter.submit(part_request())
    assert looked_up == ["l4_fallback", "l4"]


def test_fallback_can_be_turned_off() -> None:
    looked_up: list[str] = []
    adapter = ModalAdapter("twin", part_fallback=False)
    adapter._function = lambda tier: looked_up.append(tier) or FakeFunction(FakeCall())  # type: ignore[method-assign,func-returns-value]
    adapter.submit(part_request())
    assert looked_up == ["l4"]


@pytest.mark.parametrize(
    ("name", "chain", "tier"),
    [
        ("NVIDIA L4", ("l4", "l40s"), "l4"),
        ("NVIDIA L40S", ("l4", "l40s"), "l40s"),
        ("NVIDIA A10", ("l4", "a10"), "a10"),
        ("NVIDIA A100-SXM4-80GB", ("a100-40gb", "a100"), "a100"),
        ("NVIDIA A100-SXM4-40GB", ("a100", "a100-40gb"), "a100-40gb"),
        ("NVIDIA H100 80GB HBM3", ("l4", "l40s"), None),
    ],
)
def test_a_gpu_s_name_is_its_tier(name: str, chain: tuple[str, ...], tier: str | None) -> None:
    assert tier_of_gpu(name, chain) == tier


def test_every_fallback_list_is_of_deployed_priced_tiers() -> None:
    """Each tier in a list is one `providers.py` offers Modal with and prices, so the
    call can always be priced at what it ran on."""
    modal = provider("modal")
    assert modal is not None
    for tier, chain in GPU_FALLBACKS.items():
        assert chain[0] == tier
        for one in chain:
            assert one in modal.tiers and one in GPU_NAMES and modal.rate(one) is not None


# --- a call picked up by another process, and what an hour of it costs ----------------


def test_a_call_is_re_attached_to_by_its_id_and_polls_and_cancels_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker that comes up after a deploy rebuilds the call the last one left running
    with `FunctionCall.from_id` -- no `spawn`, so no second call -- and the proxy for its
    billed time keeps counting from the original submit."""
    call = FakeCall(raises=TimeoutError())
    looked_up: list[str] = []

    def from_id(call_id: str) -> FakeCall:
        looked_up.append(call_id)
        return call

    namespace = types.SimpleNamespace(FunctionCall=types.SimpleNamespace(from_id=from_id))
    monkeypatch.setitem(sys.modules, "modal", namespace)
    adapter = ModalAdapter("twin")
    assert isinstance(adapter, Reattachable)
    handle = RemoteHandle(id="fc-left-running", provider="modal", tier="l4")

    adapter.reattach(handle, request(), submitted_at=time.time() - 600.0)

    assert looked_up == ["fc-left-running"]
    poll = adapter.poll(handle)
    assert poll.state == "running"
    assert poll.billed_s >= 600.0
    adapter.cancel(handle)
    assert call.cancelled_with == {"terminate_containers": True}


def test_a_call_known_only_by_id_can_still_be_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call whose workdir is gone is known from the database by id alone, with no
    request: enough to stop paying for it."""
    call = FakeCall(raises=TimeoutError())
    namespace = types.SimpleNamespace(
        FunctionCall=types.SimpleNamespace(from_id=lambda _call_id: call)
    )
    monkeypatch.setitem(sys.modules, "modal", namespace)
    adapter = ModalAdapter("twin")
    handle = RemoteHandle(id="fc-orphan", provider="modal", tier="l4")

    adapter.reattach(handle, None, submitted_at=0.0)
    adapter.cancel(handle)

    assert call.cancelled_with == {"terminate_containers": True}


def test_outrunning_the_function_s_limit_is_marked_as_a_timeout() -> None:
    """The worker does not retry a timeout: the next attempt would outrun the same six
    hours. Every other failure is not one."""
    adapter, handle, _ = adapter_over(FakeCall(raises=FunctionTimeout("6h")))
    poll = adapter.poll(handle)
    assert poll.state == "failed" and poll.timed_out
    adapter, handle, _ = adapter_over(FakeCall(raises=OutputExpired()))
    assert not adapter.poll(handle).timed_out
    adapter, handle, _ = adapter_over(FakeCall(raises=RuntimeError("boom")))
    assert not adapter.poll(handle).timed_out


def test_a_gpu_hour_is_priced_with_the_cpu_and_memory_its_function_reserves() -> None:
    """2 cores and 8 GiB reserved beside every GPU, billed at Modal's list prices: +$0.158
    an hour on top of the L4's $0.80 -- a fifth of the bill that `costUsd` left out."""
    assert GPU_RESERVATION == (2.0, 8192)
    reserved = reservation_rate("l4")
    assert reserved is not None
    assert reserved.usd_per_hour == pytest.approx(2 * 0.04716 + 8 * 0.007992)
    adapter = ModalAdapter("twin")
    l4 = adapter.rate("l4")
    assert l4 is not None
    assert l4.usd_per_hour == pytest.approx(0.7992 + 0.158256)
    assert "reserved" in l4.source
    # The CPU box's rate already *is* its cores and memory; nothing is added to it.
    cpu = adapter.rate("cpu4")
    assert cpu is not None and cpu.usd_per_hour == pytest.approx(0.2526)
    assert reservation_rate("cpu4") is None
    # A deployment's own GPU price gets the reservation too: Modal bills it either way.
    own = ModalAdapter("twin", rates={"l4": Rate(0.50, "PIPELINE_GPU_RATES")}).rate("l4")
    assert own is not None and own.usd_per_hour == pytest.approx(0.50 + 0.158256)
    # And an unpriced GPU stays unpriced, rather than priced at its reservation alone.
    assert ModalAdapter("twin").rate("t4") is None
