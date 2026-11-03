import subprocess
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.bench.gpu import GpuSampler, summarise_samples


def test_gpu_samples_parse_units_invalid_rows_and_statistics() -> None:
    samples = [
        "10, 123\n30, 456\n",
        "90, 789",
        "0, 0",
        "N/A, N/A",
        "nan, 12",
        "5, inf",
        "-1, 10",
        "101, 10",
        "10, -1",
        "bad",
        "1,2,3",
    ]
    assert summarise_samples(samples) == {
        "gpuUtilMean": 32.5,
        "gpuUtilP50": 20.0,
        "gpuMemMaxMiB": 789.0,
    }
    assert all(v is None for v in summarise_samples([]).values())


@pytest.mark.parametrize(
    "failure", [None, FileNotFoundError(), subprocess.TimeoutExpired("smi", 2)]
)
def test_sampler_stops_and_joins_without_waiting_five_seconds(
    monkeypatch: pytest.MonkeyPatch, failure: Exception | None
) -> None:
    called = threading.Event()

    def probe(argv: list[str], **kwargs: Any) -> Any:
        assert argv == [
            "smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] == 2 and kwargs["check"]
        called.set()
        if failure:
            raise failure
        return SimpleNamespace(stdout="80, 2048\n")

    monkeypatch.setattr(subprocess, "run", probe)
    sampler = GpuSampler("smi", interval=60)
    sampler.start()
    assert called.wait(2)
    sampler.stop()
    sampler.stop()  # also safe from the worker's outer finally
    assert sampler._thread is not None and not sampler._thread.is_alive()
    assert sampler.samples == ([] if failure else ["80, 2048\n"])


def test_no_smi_or_thread_start_failure_is_harmless(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no threads")

    monkeypatch.setattr(threading.Thread, "start", fail)
    for executable in (None, "smi"):
        sampler = GpuSampler(executable)
        sampler.start()
        sampler.stop()
        assert sampler.samples == []
