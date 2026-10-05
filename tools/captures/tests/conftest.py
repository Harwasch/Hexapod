"""Pins OpenBLAS to one CPU kernel and one thread before numpy loads, and keeps the pytest
process from forking.

The committed tiles are checked byte for byte against a fresh run, and packaging goes
through BLAS (the batched covariance matmul, `eigh` on the merged parents). OpenBLAS picks
its kernels by CPU, and the last bits of those floats differ between them: the yard written
with SkylakeX kernels did not match one written on a CI runner's Haswell/Zen kernels.
Haswell's run on any AVX2 machine, so every checkout and CI writes the same bytes;
regenerate the committed fixtures the same way (synthetic_yard.py's usage).

**One thread**, because Haswell's kernels, forced, crash numpy's OpenBLAS (0.3.31) on the
AMD EPYC 9V74 and 9V45 (Zen 4, Zen 5) runners: its worker threads die of SIGSEGV at one
instruction of `dgemm_kernel_HASWELL`, in test_skin_scene's threaded `f @ pp` (kaolin_rkpm).
Caught under gdb on CI in 2026-10: 8 of 8 runs on Zen 4/5 crashed there, while 4 of 4 on EPYC
7763 (Zen 3) and every local run on Sapphire Rapids passed, so the runner a job drew
decided whether Captures went red. With the kernel pinned, a single-threaded run is also
the same bytes on every machine.

The fork guard is the earlier, wrong suspect's, kept because it is still right: this
process holds three OpenBLAS thread pools (numpy's, scipy's, and the 0.3.15 that OpenCV's
wheel bundles) and OpenCV's own, and a fork copies only the forking thread. So a test that
needs a fork runs it in a new interpreter (`fresh_process`: `subprocess`, which execs), and
a test that forks this process fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

CAPTURES = Path(__file__).resolve().parents[1]
#: Forks of this process (`os.fork`, which multiprocessing's fork context calls).
_FORKS = [0]


def _count_fork() -> None:
    _FORKS[0] += 1


os.register_at_fork(before=_count_fork)


@pytest.fixture(autouse=True)
def _the_pytest_process_does_not_fork() -> Iterator[None]:
    before = _FORKS[0]
    yield
    assert _FORKS[0] == before, (
        "this test forked the pytest process; fork in a new interpreter instead "
        "(the `fresh_process` fixture, tests/conftest.py)"
    )


@pytest.fixture
def fresh_process() -> Callable[[str], dict]:
    """Runs Python `code` in a new interpreter (in tools/captures, which it imports from) and
    returns the JSON object it prints last."""

    def run(code: str) -> dict:
        command = [sys.executable, "-c", code]
        done = subprocess.run(
            command, cwd=CAPTURES, capture_output=True, text=True, timeout=600, check=False
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    return run
