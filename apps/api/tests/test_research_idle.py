import uuid
from contextlib import nullcontext
from unittest.mock import Mock

import httpx
import pytest

from app.config import Settings
from app.research import __main__ as entry
from app.research import queue
from app.research.worker import ResearchWorker


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.delays: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.delays.append(delay)
        self.now += delay


def test_empty_queue_backs_off_and_exits_at_configured_deadline() -> None:
    clock = Clock()
    worker = Mock()
    worker.run_once.return_value = False
    worker.has_pending_work.return_value = False
    entry.poll(worker, idle_exit_s=300, clock=clock.time, sleep=clock.sleep)
    assert clock.now == 300
    assert clock.delays[0] == 2
    assert max(clock.delays) == 30
    assert worker.run_once.call_count == len(clock.delays) + 1


def test_run_resets_idle_deadline_and_backoff() -> None:
    clock = Clock()
    worker = Mock()
    worker.has_pending_work.return_value = False
    worked = False

    def run_once() -> bool:
        nonlocal worked
        if clock.now >= 240 and not worked:
            worked = True
            clock.now += 500  # Time spent working must not count as idle.
            return True
        return False

    worker.run_once.side_effect = run_once
    entry.poll(worker, idle_exit_s=300, clock=clock.time, sleep=clock.sleep)
    assert clock.now >= 1040
    assert clock.delays.count(2) >= 60


def test_zero_disables_exit_and_once_does_not_sleep() -> None:
    clock = Clock()
    worker = Mock()
    worker.run_once.side_effect = [False] * 200 + [KeyboardInterrupt]
    with pytest.raises(KeyboardInterrupt):
        entry.poll(worker, idle_exit_s=0, clock=clock.time, sleep=clock.sleep)
    assert clock.now > 300
    clock.delays.clear()
    worker.run_once.side_effect = None
    entry.poll(worker, idle_exit_s=300, once=True, clock=clock.time, sleep=clock.sleep)
    assert not clock.delays


def test_research_idle_exit_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_IDLE_EXIT_S", raising=False)
    assert Settings(_env_file=None).research_idle_exit_s == 300
    monkeypatch.setenv("RESEARCH_IDLE_EXIT_S", "42")
    assert Settings(_env_file=None).research_idle_exit_s == 42


def test_final_empty_claim_does_not_exit_with_more_pending_work() -> None:
    clock = Clock()
    worker = Mock()
    worker.run_once.return_value = False
    # claim() may return None after failing an exhausted run; the next queued run
    # must still be served, even if the idle deadline has passed.
    worker.has_pending_work.side_effect = [True, False]
    entry.poll(worker, idle_exit_s=4, clock=clock.time, sleep=clock.sleep)
    assert worker.has_pending_work.call_count == 2
    assert worker.run_once.call_count == 4


def test_idle_polls_and_stop_never_write_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = Mock(return_value=nullcontext(Mock()))
    check_in = Mock()
    monkeypatch.setattr(queue, "check_in", check_in)
    claim = Mock(return_value=None)
    monkeypatch.setattr(queue, "claim", claim)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as http:
        worker = ResearchWorker(sessions, Settings(_env_file=None), client=http)
        for _ in range(3):
            assert not worker.run_once()
        worker.stop()
        check_in.assert_not_called()
        run_id, token = uuid.uuid4(), uuid.uuid4()
        claim.return_value = (run_id, token)
        execute = Mock()
        monkeypatch.setattr(worker, "_execute", execute)
        assert worker.run_once()
        execute.assert_called_once()
        assert check_in.call_count == 2
        assert check_in.call_args_list[0].kwargs["current_run_id"] == run_id
        assert check_in.call_args_list[1].kwargs["stopping"] is True
        claim.return_value = None
        assert not worker.run_once()
        worker.stop()
        assert check_in.call_count == 2
