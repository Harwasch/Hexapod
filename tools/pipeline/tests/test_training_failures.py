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


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("line_length", [20, 200])
def test_trainer_diagnostic_survives_worker_message_cut(as_bytes: bool, line_length: int) -> None:
    argv = ["python", "trainer.py", "--data_dir", "/dataset/" + "x" * 2500]
    diagnostic = "RuntimeError: CUDA out of memory: diagnostic sentinel"
    output = "\n".join(["traceback context " + "x" * line_length] * 40 + [diagnostic])
    cause = subprocess.CalledProcessError(1, argv, output=output.encode() if as_bytes else output)
    message = str(StageFailedError("photo-reconstruct", "train", "gsplat", cause))
    # apps/api/app/worker/child.py::_message keeps this prefix, including its ellipsis.
    assert len(message) > 2000
    worker_message = message[:1997] + "..."
    assert diagnostic in worker_message
    assert argv[-1] in message
