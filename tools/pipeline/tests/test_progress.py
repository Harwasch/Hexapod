"""Reading a long tool's progress while it runs (`progress.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import progress

GSPLAT_LINE = "loss=0.041| sh degree=3| :  37%|███▋      | 11100/30000 [40:12<1:08:30,  4.60it/s]"


def test_parse_reads_tqdm_counts_and_times() -> None:
    found = progress.parse(GSPLAT_LINE)
    assert found == progress.Progress(
        done=11100, total=30000, elapsed_s=40 * 60 + 12, remaining_s=3600 + 8 * 60 + 30
    )
    assert found.to_dict() == {"done": 11100, "total": 30000, "elapsedS": 2412, "remainingS": 4110}


def test_parse_has_no_remaining_time_before_a_rate() -> None:
    found = progress.parse("  0%|          | 0/30000 [00:00<?, ?it/s]")
    assert found is not None and found.remaining_s is None


def test_parse_ignores_lines_that_merely_contain_a_fraction() -> None:
    assert progress.parse("registered 4/4 frames") is None
    assert progress.parse("Step:  29999 {'num_GS': 1}") is None


def test_latest_is_the_newest_progress_line() -> None:
    log = "\n".join(
        ["$ trainer", "  1%| | 100/30000 [00:10<50:00, 10it/s]", "note", GSPLAT_LINE, "tail"]
    )
    found = progress.latest(log)
    assert found is not None and found.done == 11100
    assert progress.latest("nothing here") is None


def test_stream_logs_as_it_goes_and_thins_redraws(tmp_path: Path) -> None:
    script = tmp_path / "tool.py"
    script.write_text(
        "import sys\n"
        "print('starting')\n"
        "for i in range(1, 201):\n"
        "    sys.stderr.write(f'\\r{i//2}%|#| {i}/200 [00:01<00:01, 99it/s]')\n"
        "sys.stderr.write('\\n')\n"
        "print('done')\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    ticks = iter(range(10_000))
    code = progress.stream(
        [sys.executable, str(script)],
        cwd=tmp_path,
        log=lines.append,
        every_s=50,
        clock=lambda: float(next(ticks)),
    )
    assert code == 0
    assert lines[0] == "starting" and lines[-1] == "done"
    kept = [line for line in lines if progress.parse(line)]
    # 200 redraws, one kept per 50 ticks, and the last one always.
    assert 3 <= len(kept) <= 6
    assert progress.parse(kept[-1]) == progress.Progress(200, 200, 1, 1)


def test_stream_returns_the_exit_code(tmp_path: Path) -> None:
    lines: list[str] = []
    code = progress.stream(
        [sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"],
        cwd=tmp_path,
        log=lines.append,
    )
    assert code == 3 and lines == ["bad"]


def test_tail_reads_the_end_of_a_long_file(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("x\n" * 50_000 + GSPLAT_LINE + "\n", encoding="utf-8")
    assert progress.latest(progress.tail(path)) is not None
    assert progress.tail(tmp_path / "missing") == ""
