"""Production telemetry survives a provider whose log service returns nothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import live_payload


def test_hanging_get_is_bounded_and_never_overlaps_or_replays_late_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time
    from types import SimpleNamespace

    monkeypatch.setattr(live_payload, "GET_TIMEOUT_S", 0.02, raising=False)
    release = threading.Event()
    calls = []

    class SlowTransfer:
        def get(self, key: str, target: Path) -> int:
            calls.append(key)
            assert release.wait(5)
            live_payload.write(target, [("train", {"v": 1, "k": "s", "step": 200})])
            return target.stat().st_size

    request = SimpleNamespace(
        impl="gsplat", params={}, checkpoint_key="runs/r/train/checkpoint", checkpoint_every_s=1
    )
    poller = live_payload.Poller()
    # Release the fake even on the unfixed synchronous implementation's failure.
    rescue = threading.Timer(0.5, release.set)
    rescue.start()
    try:
        started = time.monotonic()
        assert poller.poll(request, SlowTransfer(), tmp_path, 0) == []
        assert time.monotonic() - started < 0.3
        for timestamp in range(1, 5):
            assert poller.poll(request, SlowTransfer(), tmp_path, timestamp, final=True) == []
        assert len(calls) == 1
    finally:
        release.set()
        rescue.cancel()
        pending = getattr(poller, "_pending", None)
        if pending is not None:
            pending.join(2)
    assert not list(tmp_path.iterdir()), "late GETs must clean up their isolated temporary file"


def test_cloud_poll_records_the_reattached_calls_checkpoint_scope(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from cloud import CallBook, CallRecord, CloudRunner, Placement, Poll, RemoteHandle, StageRequest
    from contracts import StageContext

    request = StageRequest(
        recipe="r",
        run_id="r",
        stage_id="train",
        impl="gsplat",
        attempt=1,
        tier="l4",
        preemptible=False,
        params={},
        inputs={},
        produces=(),
        checkpoint_key="runs/r/train/checkpoint",
        outputs_key="runs/r/train/transfer/out",
    )
    record = CallRecord(RemoteHandle("old-call", "fake", "l4"), 0, 1, request)
    book = CallBook(tmp_path / "calls.json", calls={"": record})
    book.save()
    book = CallBook.read(book.path)
    adapter = Mock()
    adapter.poll.return_value = Poll("succeeded")
    adapter.logs.return_value = []
    records = [("train", {"v": 1, "k": "s", "step": 200})]
    transfer = Mock()

    def get(key: str, target: Path) -> int:
        live_payload.write(target, records)
        return target.stat().st_size

    transfer.get.side_effect = get
    runner = CloudRunner(Placement((adapter,)), transfer)
    context = cast("StageContext", SimpleNamespace(attempt=2, work_dir=tmp_path))
    runner._watch(adapter, record.handle, context, book, "", "l4", 0)
    document = json.loads((tmp_path / live_payload.LOCAL_NAME).read_text())
    assert document["checkpointKeys"] == [request.checkpoint_key]
    assert document["stageAttempt"] == 2
    assert live_payload.read(tmp_path / live_payload.LOCAL_NAME) == records


def test_payload_keeps_plan_and_even_history_bounded(tmp_path: Path) -> None:
    target = tmp_path / "telemetry.json"
    journal = live_payload.Journal(target)
    journal.line('live-plan: {"v":1,"id":"plan","stepsMax":1000000}')
    for step in range(1000):
        journal.line(
            "live-train: "
            + json.dumps(
                {
                    "v": 1,
                    "k": "s",
                    "run": "trainer",
                    "step": step,
                    "loss": {"total": step / 1000},
                    "gs": 64,
                }
            )
        )
    records = live_payload.read(target)
    assert records[0][0] == "plan"
    samples = [value["step"] for kind, value in records if kind == "train"]
    assert samples[0] == 0 and samples[-1] == 999
    assert len(samples) <= live_payload.MAX_SAMPLES
    assert target.stat().st_size <= live_payload.MAX_BYTES


def test_invalid_or_oversized_payloads_are_ignored(tmp_path: Path) -> None:
    target = tmp_path / "telemetry.json"
    for raw in [
        "{",
        "[]",
        '{"v":1,"records":[["train",{"v":1,"k":"s","step":-1}]]}',
        " " * (live_payload.MAX_BYTES + 1),
        '{"v":1,"records":' + "[" * 20_000 + "0" + "]" * 20_000 + "}",
        json.dumps(
            {"v": 1, "records": [["train", {"v": 1, "k": "e", "step": 1, "psnr": 10**400}]]}
        ),
    ]:
        target.write_text(raw)
        assert live_payload.read(target) == []


def test_recipe_opt_out_writes_no_live_payload(tmp_path: Path) -> None:
    from executor import execute
    from runners import RunnerSet
    from test_train_gsplat import seed_inputs, stand_in_params, train_recipe
    from workdir import Workdir

    work = Workdir.create(tmp_path / "run")
    seed_inputs(work)
    previous = work.checkpoint_dir("train") / "live"
    previous.mkdir(parents=True)
    (previous / "telemetry.json").write_text("stale observations")
    recovery = work.checkpoint_dir("train") / "recovery.bin"
    recovery.write_bytes(b"recovery state")
    execute(train_recipe(stand_in_params(live=False)), work, RunnerSet.local())
    assert recovery.read_bytes() == b"recovery state"
    assert not list(work.checkpoint_dir("train").rglob("telemetry.json"))
    import live

    assert live.records(work.log_path("train").read_text()) == []


def test_poll_cost_is_bounded_and_disabled_stages_never_read(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    request = SimpleNamespace(
        impl="gsplat", params={}, checkpoint_key="runs/r/train/checkpoint-a2", checkpoint_every_s=60
    )
    transfer = Mock()
    transfer.get.return_value = 0
    poller = live_payload.Poller()
    for timestamp in range(180):
        poller.poll(request, transfer, tmp_path, timestamp)
    assert transfer.get.call_count == 3
    assert transfer.get.call_args.args[0] == "runs/r/train/checkpoint-a2/live/telemetry.json"
    poller.poll(request, transfer, tmp_path, 180, final=True)
    assert transfer.get.call_count == 4
    transfer.reset_mock()
    request.params["live"] = False
    poller.poll(request, transfer, tmp_path, 240, final=True)
    transfer.get.assert_not_called()


def test_journal_reset_and_failed_writes_are_best_effort(tmp_path: Path) -> None:
    target = tmp_path / "telemetry.json"
    journal = live_payload.Journal(target)
    journal.line('live-train: {"v":1,"k":"s","step":200,"run":"old"}')
    journal.line('live-plan: {"v":1,"id":"new","stepsMax":1000}')
    assert len(live_payload.read(target)) == 1
    target.unlink()
    target.mkdir()
    live_payload.write(target, [])  # even the runner's local write is best effort
    journal.line('live-train: {"v":1,"k":"s","step":1,"run":"new"}')
