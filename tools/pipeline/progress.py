"""How far along a long-running tool is, read from what it prints.

A stage that shells out used to collect the tool's output when the tool exited, so a
two-hour training run said nothing at all for two hours -- not in the stage log, not in
Modal's logs, not on the phone page. `stream` runs the tool with its output going into
the stage log as it is printed; `latest` reads the newest progress line back out of a
log, which is how the worker turns a stage log into "iteration 11,100 of 30,000".

The progress line is tqdm's, because that is what the trainers print (gsplat's
`simple_trainer.py` wraps its loop in `tqdm.tqdm(range(init_step, max_steps))`):

    loss=0.041| sh degree=3| :  37%|███▋      | 11100/30000 [40:12<1:08:30,  4.60it/s]

tqdm redraws that line with a carriage return several times a second. Logging each
redraw would be tens of thousands of lines, so a progress line is kept at most once per
`every_s`, plus the last one.
"""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Progress", "latest", "parse", "stream"]

#: `done/total [elapsed<remaining`. The remaining time is `?` before tqdm has a rate.
_TQDM_RE = re.compile(r"(\d+)/(\d+) \[(\d+(?::\d+)+)<(\?|\d+(?::\d+)+)")


@dataclass(frozen=True)
class Progress:
    done: int
    total: int
    elapsed_s: int
    #: None until the tool has measured a rate.
    remaining_s: int | None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "done": self.done,
            "total": self.total,
            "elapsedS": self.elapsed_s,
            "remainingS": self.remaining_s,
        }


def _seconds(clock: str) -> int:
    total = 0
    for part in clock.split(":"):
        total = total * 60 + int(part)
    return total


def parse(line: str) -> Progress | None:
    """The progress in one line, or None when the line is not a progress line."""
    match = _TQDM_RE.search(line)
    if match is None:
        return None
    done, total, elapsed, remaining = match.groups()
    return Progress(
        done=int(done),
        total=int(total),
        elapsed_s=_seconds(elapsed),
        remaining_s=None if remaining == "?" else _seconds(remaining),
    )


def latest(text: str) -> Progress | None:
    """The newest progress line in a log, or None when it has none."""
    for line in reversed(text.splitlines()):
        found = parse(line)
        if found is not None:
            return found
    return None


def tail(path: Path, size: int = 16_384) -> str:
    """The end of a file, for reading the newest progress out of a long log cheaply."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - size))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def stream(
    argv: Sequence[str],
    *,
    cwd: Path,
    log: Callable[[str], None],
    every_s: float = 15.0,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run `argv`, logging its output (stdout and stderr) line by line as it arrives.

    Returns the exit code. A line ends at `\\n` or `\\r`, because tqdm ends its redraws
    with the second; progress lines are thinned to one per `every_s`.
    """
    process = subprocess.Popen(  # noqa: S603 - argv comes from stage code, never a shell
        list(argv),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # A Python tool writing into a pipe buffers its prints until exit otherwise.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    last_kept = float("-inf")
    held: str | None = None

    def emit(line: str) -> None:
        nonlocal last_kept, held
        if not line.strip():
            return
        if parse(line) is None:
            # The redraw that was held back is the last word on the progress before
            # this line, so it goes first: a log reads in the order things happened.
            if held is not None:
                log(held)
                held = None
            log(line)
            return
        now = clock()
        if now - last_kept >= every_s:
            log(line)
            last_kept = now
            held = None
        else:
            held = line

    fd = process.stdout.fileno()
    while True:
        chunk = os.read(fd, 65_536)
        if not chunk:
            break
        pending += decoder.decode(chunk)
        *lines, pending = re.split(r"\r\n|\r|\n", pending)
        for line in lines:
            emit(line)
    pending += decoder.decode(b"", final=True)
    emit(pending)
    if held is not None:
        log(held)
    return process.wait()
