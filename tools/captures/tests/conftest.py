"""Pins OpenBLAS to one CPU kernel before numpy loads, and keeps the pytest process from
forking.

OpenBLAS picks its kernels by CPU, and the last bits of floats that go through BLAS differ
between them, so a suite that compares bytes pins one. **Sandybridge's**, because it is the
one kernel that runs on every CI runner: Haswell's, pinned until 2026-10, crash numpy's
OpenBLAS (0.3.31) on the AMD EPYC 9V74 and 9V45 runners that expose AVX-512 -- SIGSEGV in
`dgemm_kernel_HASWELL`, at one thread or four, in test_skin_scene's large matmuls
(kaolin_rkpm) and in a bare script of the same matmul shapes -- and SkylakeX's are illegal
instructions on the runners without AVX-512. Measured on CI with each pin on each kind of
runner (EPYC 7763; 9V74/9V45 with and without AVX-512): only Sandybridge ran that script and
test_skin_scene clean on all of them. Which runner a job drew had decided whether Captures
went red.

The committed synthetic yard (data/tiles/synthetic-yard/splat) was written with Haswell's
kernels, and its bytes differ under Sandybridge's, so test_scene_plants writes its fresh yard
in a new interpreter pinned to Haswell (`synthetic_yard.FIXTURE_CORETYPE`). That is safe on
every runner: the yard check passed under Haswell on the crashing ones, which died later, in
the large matmuls.

A fork copies the process but only the forking thread, and this process holds three OpenBLAS
thread pools (numpy's, scipy's, and the 0.3.15 that OpenCV's wheel bundles) and OpenCV's own.
So a test that needs a fork runs it in a new interpreter (`fresh_process`: `subprocess`,
which execs), and a test that forks this process fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_CORETYPE", "Sandybridge")

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
