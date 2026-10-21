"""`experiments/ingest_demo.py` runs end to end and describes all three captures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "experiments" / "ingest_demo.py"


def test_the_demo_describes_both_captures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location("ingest_demo", DEMO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--out", str(tmp_path)]) == 0

    report = capsys.readouterr().out
    assert "PLATFORM DETECTION TABLE" in report
    # One of its six frames lost the RTK fix: a drone, not the survey tier.
    assert "run: drone                         drone        drone-xmp        metadata" in report
    assert "run: phone-photos                  phone-photo" in report
    assert "run: phone-video                   phone-video" in report
    assert '"tonemapped": true' in report
    assert '"accuracyM": 4.766775' in report
