"""Paid-trainer failures must retain the diagnostic, on CPU as well as on Modal."""

import subprocess
import sys
from pathlib import Path

import pytest

from contracts import StageContext
from errors import StageFailedError
from workdir import Workdir


def failed_command(work: Workdir) -> None:
    work.prepare_stage("train")
    context = StageContext(
        recipe="test",
        run_id="test",
        stage_id="train",
        impl="gsplat",
        params={},
        attempt=1,
        gpu_tier=None,
        out_dir=work.out_dir("train"),
        work_dir=work.work_dir("train"),
        checkpoint_dir=work.checkpoint_dir("train"),
        log_path=work.log_path("train"),
        checkpoint_key="test/train",
        attempts_path=work.attempts_path("train"),
        _inputs={},
        _produces={},
    )
    script = work.work_dir("train") / "failed.py"
    script.write_text(
        "import sys\n"
        "for i in range(70): print(f'output line {i}', flush=True)\n"
        "print('CUDA out of memory: diagnostic sentinel', file=sys.stderr, flush=True)\n"
        "sys.exit(1)\n"
    )
    try:
        context.run([sys.executable, str(script)])
    except subprocess.CalledProcessError as exc:
        raise StageFailedError("test", "train", "gsplat", exc) from exc


def test_failed_command_keeps_last_forty_lines_and_full_stage_log(tmp_path: Path) -> None:
    work = Workdir.create(tmp_path / "work")
    with pytest.raises(StageFailedError) as failure:
        failed_command(work)
    message = str(failure.value)
    assert "CUDA out of memory: diagnostic sentinel" in message
    assert "output line 31\n" in message
    assert "output line 30\n" not in message
    assert "output line 0\n" in work.log_path("train").read_text()
