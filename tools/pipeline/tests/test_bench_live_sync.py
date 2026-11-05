"""Opt-in volume telemetry, exercised without Modal, GPUs or network."""

import ast
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

import live
from experiments.bench import __main__ as cli
from experiments.bench import live_sync, rehearsal, worker
from experiments.bench.ledger import hourly_rate
from workdir import Workdir


def setup_sync(tmp_path: Path, commit: Any = lambda: None, **kwargs: Any) -> live_sync.LiveSync:
    work = Workdir.create(tmp_path / "work")
    return live_sync.LiveSync(
        work,
        tmp_path / "scene",
        tmp_path / "run",
        {"tier": "l4", "scene": "spool", "variant": {"name": "baseline"}},
        commit,
        **kwargs,
    )


def write_log(sync: live_sync.LiveSync, text: str) -> None:
    path = sync.work.log_path("train")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(text)


def test_incremental_filtered_atomic_copies_and_change_only_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commits: list[bool] = []
    sync = setup_sync(tmp_path, lambda: commits.append(True))
    write_log(sync, 'noise\n[train] live-train: {"v":1,"k":"s","step":1}\n')
    source = sync.work.checkpoint_dir("train") / "live/splat_000001.spz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"preview")
    source.with_suffix(".spz.part").write_bytes(b"unfinished")
    original = Path.replace
    renames: list[str] = []

    def replace(path: Path, target: Any) -> Path:
        assert path.name.endswith(".part")
        renames.append(Path(target).name)
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    sync.poll()
    log = sync.directory / "train.log"
    assert log.read_text() == 'live-train: {"v":1,"k":"s","step":1}\n'
    assert (sync.directory / source.name).read_bytes() == b"preview"
    assert sorted(renames) == ["splat_000001.spz", "train.log"]
    sync.poll()
    assert commits == [True] and len(renames) == 2
    write_log(sync, 'ignored\nlive-plan: {"v":1,"stepsMax":10}\nlive-train: ')
    sync.poll()
    enriched = live.records(log.read_text())[-1][1]
    assert enriched["usdPerHour"] == hourly_rate("l4")
    assert enriched["benchScene"] == "spool" and enriched["benchVariant"] == "baseline"
    write_log(sync, '{"v":1,"k":"s","step":2}\n')
    sync.poll()
    assert [row[1].get("step") for row in live.records(log.read_text())] == [1, None, 2]
    assert len(commits) == 3
    assert not list(sync.directory.glob("*.part"))


def test_commit_failure_retries_without_duplicate_copies_and_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[Any] = []

    def commit() -> None:
        calls.append(True)
        if len(calls) < 3:
            raise RuntimeError("offline")

    sync = setup_sync(tmp_path, commit)
    sync.poll()  # Missing stage/checkpoint directories are normal at startup.
    assert not calls and not sync.directory.exists()
    write_log(sync, "live-train: {}\n")
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            sync.poll()
    assert len(calls) == 3
    assert sync.dirty is False
    assert (sync.directory / "train.log").read_text() == "live-train: {}\n"
    assert sum("commit:" in row.message for row in caplog.records) == 1


@pytest.mark.parametrize("name", ["splat_1.spz", "seeds.bin"])
def test_failed_atomic_copy_never_publishes_partial_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls: list[Any] = []
    sync = setup_sync(tmp_path, lambda: calls.append(True))
    source = sync.work.checkpoint_dir("train") / "live" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"preview")
    original = shutil.copyfile

    def fail(source: Any, target: Any) -> None:
        Path(target).write_bytes(b"half")
        raise OSError("interrupted")

    monkeypatch.setattr(shutil, "copyfile", fail)
    sync.poll()
    assert not list(sync.directory.iterdir()) and not calls
    monkeypatch.setattr(shutil, "copyfile", original)
    sync.poll()
    assert (sync.directory / source.name).read_bytes() == b"preview" and calls == [True]


def test_stop_is_bounded_even_when_commit_is_stuck_and_commits_are_serial(tmp_path: Path) -> None:
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    calls: list[Any] = []

    def commit() -> None:
        calls.append(True)
        entered.set()
        assert release.wait(5)

    guarded = live_sync.CommitLock(commit)
    sync = setup_sync(tmp_path, guarded, join_timeout=0.01)
    write_log(sync, "live-train: {}\n")
    sync.start()
    try:
        assert entered.wait(2)
        # Overlapping telemetry flushes coalesce behind the daemon's volume call.
        assert guarded() is False and len(calls) == 1

        def stop_sync() -> None:
            sync.stop()
            stopped.set()

        thread = threading.Thread(target=stop_sync, daemon=True)
        thread.start()
        assert stopped.wait(0.5)
    finally:
        release.set()
        if sync.thread:
            sync.thread.join(2)
    assert sync.thread is not None and not sync.thread.is_alive()


def test_stop_performs_final_copy_and_emits_frozen_cameras_only_in_synced_log(
    tmp_path: Path,
) -> None:
    sync = setup_sync(tmp_path, interval=3600)
    rehearsal.synthetic_scene(sync.scene, "spool", "test")
    sync.start()
    write_log(sync, "live-train: {}\n")
    sync.stop()
    text = (sync.directory / "train.log").read_text()
    assert text.count("live-cameras:") == 1 and "live-train:" in text
    assert "live-cameras:" not in sync.work.log_path("train").read_text()


@pytest.mark.parametrize("enabled", [False, True])
def test_rehearsal_opt_in_preserves_trainer_params_and_comparison_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    args = cli.parser().parse_args(
        ["run", "--scene", "spool", "--rehearse", *(["--live"] if enabled else [])]
    )
    plan = cli.plans_for(args)[0]
    plan["timeoutSeconds"] = 1800
    assert ("live" in plan) == enabled
    assert "live" not in plan["trainParams"]
    params = rehearsal.stand_in_params()
    # Fabricated telemetry from the stand-in, just as the real trainer hook emits it.
    params["extra_args"] += ["--emit-live"]
    monkeypatch.setattr(rehearsal, "stand_in_params", lambda: params)
    scene = tmp_path / "scene"
    rehearsal.synthetic_scene(scene, "spool", "test")
    run = tmp_path / "run"
    commits: list[bool] = []
    result = worker.run_training(scene, run, plan, live_commit=lambda: commits.append(True))
    assert result["status"] == "complete", result["error"]
    assert result["trainParams"] == {**plan["trainParams"], **params}
    assert not any(key.startswith("_live") for key in result["trainParams"])
    if enabled:
        text = (run / "live/train.log").read_text()
        assert all(tag in text for tag in (live.TRAIN_TAG, live.PLAN_TAG, live.CAMERAS_TAG))
        assert list((run / "live").glob("splat_*.spz")) and commits
    else:
        assert not (run / "live").exists() and not commits


@pytest.mark.parametrize("enabled", [False, True])
def test_modal_boundary_commits_marker_first_and_keeps_default_commit_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    # Execute only _train's AST with a fake volume and stand-in worker: no Modal import.
    path = cli.REPO / "infra/modal/bench.py"
    tree = ast.parse(path.read_text())
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_train"
    )
    calls: list[Any] = []

    def train(scene: Any, run: Any, plan: Any, **kwargs: Any) -> dict[str, Any]:
        assert calls == ["reload", "commit"]
        if plan.get("live"):
            kwargs["live_commit"]()
        return {"status": "complete"}

    class Volume:
        def reload(self) -> None:
            calls.append("reload")

        def commit(self) -> None:
            calls.append("commit")

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": lambda _: tmp_path,
        "volume": Volume(),
        "_billed_since": lambda: (0, "test"),
        "time": __import__("time"),
        "hourly_rate": lambda _: 1,
        "write_json": lambda *_: None,
    }
    monkeypatch.setattr(worker, "run_training", train)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    plan = {
        "runId": "one",
        "scene": "spool",
        "tier": "l4",
        "schedule": "screen",
        **({"live": True} if enabled else {}),
    }
    namespace["_train"](plan)
    assert calls.count("commit") == (4 if enabled else 3)
    namespace["_train"](plan)  # A live flush cannot permit a second training attempt.
    assert calls.count("commit") == (5 if enabled else 4)


def test_queued_telemetry_commit_is_drained_after_active_flush() -> None:
    entered, release = threading.Event(), threading.Event()
    observed: list[str] = []
    result = "training"

    def callback() -> None:
        observed.append(result)
        if len(observed) == 1:
            entered.set()
            assert release.wait(2)

    commit = live_sync.CommitLock(callback)
    thread = threading.Thread(target=commit, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        result = "complete"
        assert commit() is False  # Queued, never overlapping and never blocking the run.
    finally:
        release.set()
        thread.join(2)
    assert observed == ["training", "complete"]


def test_malformed_plan_does_not_block_later_records(tmp_path: Path) -> None:
    sync = setup_sync(tmp_path)
    write_log(sync, 'live-plan: invalid\nlive-train: {"v":1,"k":"s","step":2}\n')
    sync.poll()
    sync.poll()
    assert live.records((sync.directory / "train.log").read_text())[0][1]["step"] == 2
    assert (sync.directory / "train.log").read_text().count("live-train:") == 1


def test_start_and_copy_errors_cannot_fail_training_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_start(self: threading.Thread) -> None:
        raise RuntimeError("cannot start a thread")

    monkeypatch.setattr(threading.Thread, "start", failed_start)
    sync = setup_sync(tmp_path)
    with live_sync.watching(sync.plan, sync.work, sync.scene, tmp_path / "run", None):
        pass
    assert not sync.directory.exists()


def test_final_commit_waits_and_flushes_on_its_callers_thread() -> None:
    entered, release = threading.Event(), threading.Event()
    final_started, final_done = threading.Event(), threading.Event()
    observed: list[int] = []
    errors: list[Exception] = []

    def callback() -> None:
        observed.append(threading.get_ident())
        if len(observed) == 1:
            entered.set()
            assert release.wait(2)

    commit = live_sync.CommitLock(callback)
    telemetry = threading.Thread(target=commit, daemon=True)
    telemetry.start()

    def finish() -> None:
        final_started.set()
        try:
            commit.final(timeout=2)
        except Exception as error:
            errors.append(error)
        finally:
            final_done.set()

    caller = threading.Thread(target=finish, daemon=True)
    try:
        assert entered.wait(1)
        caller.start()
        assert final_started.wait(1)
        assert not final_done.wait(0.02) and len(observed) == 1
    finally:
        release.set()
        telemetry.join(2)
        if caller.ident is not None:
            caller.join(2)
    assert not errors
    assert final_done.is_set() and observed == [telemetry.ident, caller.ident]


def test_final_commit_timeout_is_bounded_and_callback_errors_propagate() -> None:
    def fail() -> None:
        raise RuntimeError("final flush failed")

    commit = live_sync.CommitLock(fail)
    with pytest.raises(RuntimeError, match="final flush failed"):
        commit.final()
    # A stuck telemetry commit must not fail a run that has been paid for: it commits anyway.
    calls: list[int] = []
    commit = live_sync.CommitLock(lambda: calls.append(1))
    commit.running = True
    commit.final(timeout=0.01)
    assert calls == [1]
    assert commit.running is False


def test_stop_uses_sixty_second_join_and_never_raises(tmp_path: Path) -> None:
    from types import SimpleNamespace

    timeouts: list[float] = []

    def join(timeout: float) -> None:
        timeouts.append(timeout)
        raise RuntimeError("join failed")

    sync = setup_sync(tmp_path)
    thread: Any = SimpleNamespace(ident=1, join=join)
    sync.thread = thread
    sync.stop()
    assert timeouts == [60] and sync.stop_event.is_set()


def test_seeds_and_twenty_snapshots_sync_once_atomically_with_quality_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    commits: list[bool] = []
    sync = setup_sync(tmp_path, lambda: commits.append(True))
    directory = sync.work.checkpoint_dir("train") / "live"
    payload = live.write_seeds(
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[255, 0, 0], [0, 255, 0]]),
        directory,
        lambda line: write_log(sync, line + "\n"),
        key_prefix="runs/r/train/checkpoint/live",
    )
    assert payload is not None
    metrics = 'live-metrics: {"step":1, "farFieldShare":0.25}'
    write_log(sync, metrics + "\nignored\n")
    for step in range(1, 21):
        (directory / f"splat_{step:06d}.spz").write_bytes(bytes([step]))
    (directory / "seeds.bin.part").write_bytes(b"unfinished")
    (directory / "other.bin").write_bytes(b"not telemetry")
    original = Path.replace
    renamed: list[str] = []

    def replace(source: Path, target: Any) -> Path:
        assert source.name.endswith(".part")
        renamed.append(Path(target).name)
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    sync.poll()
    sync.poll()
    assert (sync.directory / "seeds.bin").read_bytes() == (directory / "seeds.bin").read_bytes()
    assert (sync.directory / "seeds.bin").stat().st_size == 20
    text = (sync.directory / "train.log").read_text()
    assert live.records(text) == [("seeds", payload)]
    assert metrics + "\n" in text and "ignored" not in text
    assert len(list(sync.directory.glob("splat_*.spz"))) == 20
    assert len(renamed) == 22 and renamed.count("seeds.bin") == 1
    assert not list(sync.directory.glob("*.part")) and not (sync.directory / "other.bin").exists()
    assert commits == [True]
