"""Production telemetry survives a provider whose log service returns nothing."""

from __future__ import annotations

import json
from pathlib import Path

import live_payload


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
