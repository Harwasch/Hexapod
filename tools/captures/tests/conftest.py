"""Pins OpenBLAS to one CPU kernel before numpy loads, and keeps the pytest process from
forking.

The committed tiles are checked byte for byte against a fresh run, and packaging goes
through BLAS (the batched covariance matmul, `eigh` on the merged parents). OpenBLAS picks
its kernels by CPU, and the last bits of those floats differ between them: the yard written
with SkylakeX kernels did not match one written on a CI runner's Haswell/Zen kernels.
Haswell's run on any AVX2 machine, so every checkout and CI writes the same bytes;
regenerate the committed fixtures the same way (synthetic_yard.py's usage).

The whole suite runs in this one process, which by the segmentation tests holds three
OpenBLAS thread pools (numpy's, scipy's, and the 0.3.15 that OpenCV's wheel bundles) and
OpenCV's own. A fork copies the process but only the forking thread; each OpenBLAS tears
its pool down in this process before the fork (its pthread_atfork handler) and rebuilds it
on next use. On CI (Python 3.12), with 24 forks of this process by segment_scene's render
workers behind it -- 22 of them with OpenCV's threads running, which Python 3.12 warns
about -- the next heavy LAPACK call (`np.linalg.solve` in kaolin_rkpm, test_skin_scene)
died of SIGBUS, twice in a row (SIGSEGV with no traceback the second time). So a test that
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
