"""`train` / `gsplat`: everything around the trainer, and nothing that needs a GPU.

**No training run has been executed anywhere in this repository.** `gsplat` needs CUDA,
and no machine this was written on has a GPU. So what is tested here is the dispatch:
the COLMAP dataset the trainer is handed, the argv it is given, the metrics read back
out of its output, and the PLY normalised into `trained.ply`.

`gsplat_stand_in.py` plays the trainer. It is named that way in every test below so no
reader can mistake one of these for a training measurement -- it writes made-up numbers
into the file layout gsplat v1.5.3 uses, and the only thing it is evidence about is this
project's half of the arrangement.

What the stand-in imitates was read from gsplat v1.5.3's `examples/simple_trainer.py`,
and the argv `training.gsplat_argv` builds was parsed by that file's own CLI in a CPU venv
with the training image's exact wheel -- see `training.py`, which lists what that check
corrected. The image build in `infra/modal/app.py` repeats the parse against the
installed trainer, so an argv that stops parsing fails the deploy rather than a run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import training
from adapters import LocalTransfer, SubprocessAdapter
from cloud import AttemptLedger, CloudRunner, Placement
from conftest import make_recipe
from errors import PreemptedError
from executor import execute
from providers import Rate
from recipe import Recipe
from runners import LocalRunner, RunnerSet
from workdir import Workdir

STAND_IN = Path(__file__).resolve().parent / "gsplat_stand_in.py"

#: A `stats/val_step<i:04d>.json` shape, as v1.5.3's `eval()` writes it: the metrics
#: dict plus `ellipse_time` and `num_GS` (and `cc_*` when bilateral grids are on).
REAL_FORMAT_STATS = {
    "psnr": 28.417469024658203,
    "ssim": 0.9124583005905151,
    "lpips": 0.10422705858945847,
    "ellipse_time": 0.008542060852050781,
    "num_GS": 412733,
}


def seed_inputs(workdir: Workdir, *, frames: int = 4) -> None:
    """A frame set and a COLMAP model, in the shape the two real stages produce."""
    frame_dir = workdir.input_path("frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        (frame_dir / f"frame_{index:04d}.jpg").write_bytes(b"\xff\xd8\xff not a real jpeg")
    poses = workdir.input_path("poses")
    poses.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (poses / name).write_bytes(b"colmap")
    (poses / "poses.json").write_text(json.dumps({"registered": frames}), encoding="utf-8")


def train_recipe(params: dict[str, object], *, gpu: dict[str, object] | None = None) -> Recipe:
    stage: dict[str, object] = {"id": "train", "impl": "gsplat", "params": params}
    if gpu is not None:
        stage["gpu"] = gpu
    return make_recipe([stage], inputs=["frames", "poses"])


def stand_in_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "iterations": 300,
        "trainer": str(STAND_IN),
        "python": sys.executable,
        "extra_args": ["--ckpt-every", "100", "--gaussians", "64"],
    }
    params.update(overrides)
    return params


# --- argv and dataset, which need neither a trainer nor a GPU ------------------------


def test_the_argv_is_gsplat_1_5_3s_and_carries_the_four_switches_it_cannot_do_without() -> None:
    """Each of these was learned by reading v1.5.3, and each one missing costs a GPU run:

    no `--disable_viewer` and the trainer sleeps for a million seconds after training; no
    `--save_ply` and there is no PLY; no `--no-normalize-world-space` and the PLY is in a
    rotated, rescaled frame nothing downstream knows; and `--ckpt`, if it were ever
    passed, would evaluate a checkpoint instead of training.
    """
    argv = training.gsplat_argv(
        "python3", Path("/t/simple_trainer.py"), Path("/d"), Path("/r"), max_steps=7000
    )

    assert argv[:3] == ["python3", "/t/simple_trainer.py", "default"]
    assert argv[argv.index("--data_dir") + 1] == "/d"
    assert argv[argv.index("--result_dir") + 1] == "/r"
    assert argv[argv.index("--max_steps") + 1] == "7000"
    for switch in ("--disable_viewer", "--save_ply", "--no-normalize-world-space"):
        assert switch in argv
    for steps in ("--ply_steps", "--save_steps", "--eval_steps"):
        assert argv[argv.index(steps) + 1] == "7000"
    assert "--ckpt" not in argv


def test_extra_arguments_come_last_so_they_can_override() -> None:
    argv = training.gsplat_argv(
        "python3", Path("/t/s.py"), Path("/d"), Path("/r"), extra=["--data_factor", "2"]
    )

    assert argv[-2:] == ["--data_factor", "2"]


def test_the_trainer_runs_under_its_own_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    """gsplat's prebuilt CUDA wheels are CPython 3.10 only; this project is 3.12."""
    monkeypatch.setenv("GSPLAT_PYTHON", "/opt/trainer/bin/python")

    assert training.trainer_python(None) == "/opt/trainer/bin/python"
    assert training.trainer_python("/other/python") == "/other/python"
    monkeypatch.delenv("GSPLAT_PYTHON")
    assert training.trainer_python(None) == sys.executable


def test_the_ply_is_the_furthest_step_and_not_the_last_name(tmp_path: Path) -> None:
    """`point_cloud_999.ply` sorts after `point_cloud_29999.ply` as text."""
    plys = tmp_path / "ply"
    plys.mkdir()
    for step in (999, 29999, 6999):
        (plys / f"point_cloud_{step}.ply").write_text("x")

    newest = training.latest_ply(tmp_path)
    assert newest is not None and newest.name == "point_cloud_29999.ply"
    assert training.latest_ply(tmp_path / "nothing") is None


def test_the_dataset_is_colmaps_own_layout(tmp_path: Path) -> None:
    frames, poses = tmp_path / "frames", tmp_path / "poses"
    frames.mkdir()
    poses.mkdir()
    (frames / "frame_0000.jpg").write_bytes(b"a")
    (poses / "cameras.bin").write_bytes(b"b")

    dataset = training.build_dataset(frames, poses, tmp_path / "work")

    assert (dataset / "images" / "frame_0000.jpg").read_bytes() == b"a"
    assert (dataset / "sparse" / "0" / "cameras.bin").read_bytes() == b"b"
    # Copied, not linked: the trainer may see this directory through a mount where a
    # symlink out of it resolves to nothing.
    assert not (dataset / "images" / "frame_0000.jpg").is_symlink()


def test_a_missing_trainer_says_where_one_comes_from() -> None:
    with pytest.raises(training.TrainerMissingError, match=r"simple_trainer\.py"):
        training.trainer_script(None)
    with pytest.raises(training.TrainerMissingError, match="does not exist"):
        training.trainer_script("/no/such/trainer.py")


# --- reading the trainer's own output ------------------------------------------------


def test_metrics_come_out_of_the_stats_file_gsplat_writes(tmp_path: Path) -> None:
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "val_step6999.json").write_text(json.dumps({"psnr": 1.0, "num_GS": 1}))
    (stats / "val_step29999.json").write_text(json.dumps(REAL_FORMAT_STATS))
    # The per-save training stats sit beside them and carry no quality metrics.
    (stats / "train_step29999_rank0.json").write_text(json.dumps({"mem": 3.1, "num_GS": 9}))

    metrics = training.parse_metrics(tmp_path, requested_iterations=30000)

    assert metrics.source == "stats"
    # The furthest-along step, as a count: `val_step29999` is the zero-based index of
    # the 30,000th step.
    assert metrics.iterations == 30000
    assert metrics.psnr == pytest.approx(28.417469)
    assert metrics.ssim == pytest.approx(0.912458)
    assert metrics.lpips == pytest.approx(0.104227)
    assert metrics.gaussians == 412733
    assert metrics.requested_iterations == 30000


def test_metrics_fall_back_to_the_trainers_stdout(tmp_path: Path) -> None:
    """v1.5.3 prints `print("Step: ", step, stats)` and an eval line with PSNR."""
    log = (
        "Step:  6999 {'mem': 2.1, 'ellipse_time': 310.2, 'num_GS': 412733}\n"
        "Step:  13999 {'mem': 2.6, 'ellipse_time': 640.8, 'num_GS': 508120}\n"
        "PSNR: 26.110, SSIM: 0.8410, LPIPS: 0.172 Time: 0.011s/image Number of GS: 508120\n"
    )

    metrics = training.parse_metrics(tmp_path, log)

    assert metrics.source == "stdout"
    assert metrics.iterations == 14000
    assert metrics.gaussians == 508120
    assert metrics.psnr == pytest.approx(26.11)


def test_an_output_nobody_recognises_records_nulls_rather_than_a_number(
    tmp_path: Path,
) -> None:
    """The honest failure. A plausible PSNR that no trainer produced is worse than none."""
    metrics = training.parse_metrics(tmp_path, "the trainer said something else entirely")

    assert metrics.source == "none"
    assert metrics.psnr is None and metrics.iterations is None and metrics.gaussians is None
    assert metrics.to_dict()["psnr"] is None


# --- the stage, driven by a stand-in trainer -----------------------------------------


def test_the_stage_dispatches_and_normalises_what_the_stand_in_trainer_wrote(
    tmp_path: Path,
) -> None:
    """End to end through `LocalRunner` -- with a stand-in, not a trainer."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))

    out = workdir.out_dir("train")
    assert (out / "trained.ply").read_bytes()[:3] == b"ply"
    metrics = json.loads((out / "train_metrics.json").read_text())
    assert metrics["source"] == "stats"
    assert metrics["iterations"] == 300
    assert metrics["requestedIterations"] == 300
    assert metrics["gaussiansInPly"] == 64
    assert metrics["resumedFromStep"] is None
    assert metrics["masksIgnored"] is False
    assert metrics["gsplatVersion"] == training.GSPLAT_VERSION
    assert metrics["trainer"] == "gsplat:gsplat_stand_in.py"
    log = workdir.log_path("train").read_text()
    assert "--disable_viewer" in log and "--ckpt " not in log


def test_the_result_directory_is_scratch_because_nothing_can_resume_from_it(
    tmp_path: Path,
) -> None:
    """In `work/`, not `checkpoint/`: v1.5.3 cannot continue a checkpoint, so syncing its
    checkpoints out every minute would be paying to move bytes nothing reads."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))

    assert any(workdir.work_dir("train").rglob("ckpt_*.pt"))
    assert not any(workdir.checkpoint_dir("train").rglob("*"))


def test_a_trainer_that_writes_no_ply_is_a_failure_rather_than_an_empty_artifact(
    tmp_path: Path,
) -> None:
    """If the trainer puts its splat somewhere this project does not look, the stage
    raises and says the splat is missing, instead of producing a `trained.ply` of
    nothing."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    params = stand_in_params(extra_args=["--ckpt-every", "100", "--gaussians", "64", "--no-ply"])

    with pytest.raises(Exception, match=r"wrote no \.ply"):
        execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    assert not (workdir.out_dir("train") / "trained.ply").exists()


def test_masks_that_this_trainer_cannot_use_are_recorded_as_ignored(tmp_path: Path) -> None:
    """`mask: robust` lands in B3; until then the contract says so out loud."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    masks = workdir.input_path("masks")
    masks.mkdir(parents=True)
    (masks / "mask_0000.png").write_bytes(b"\x89PNG")
    recipe = make_recipe(
        [{"id": "train", "impl": "gsplat", "params": stand_in_params()}],
        inputs=["frames", "poses", "masks"],
    )

    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["masksIgnored"] is True
    assert "not used" in workdir.log_path("train").read_text()


# --- preemption, through the cloud seam ----------------------------------------------


def test_a_killed_training_attempt_starts_over_and_says_so(tmp_path: Path) -> None:
    """A genuine SIGTERM mid-run, through `SubprocessAdapter` and `CloudRunner`.

    The stand-in is killed at step 150 of 300. The second attempt trains from step 0 --
    v1.5.3 has no resume, and pretending otherwise was the bug this replaces -- and the
    log says why. Both attempts are in the ledger, because both were paid for.
    """
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = SubprocessAdapter(transfer, tmp_path / "sandbox", rates={"l4": Rate(0.80, "test")})
    cloud = CloudRunner(
        Placement((adapter,)), transfer, poll_interval_s=0.02, checkpoint_every_s=0.05
    )
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    marker = tmp_path / "died-once"
    params = stand_in_params(
        extra_args=[
            "--ckpt-every",
            "100",
            "--gaussians",
            "64",
            "--die-at",
            "150",
            "--die-marker",
            str(marker),
            "--kill-parent",
        ]
    )
    recipe = train_recipe(params, gpu={"tier": "l4", "preemptible": True})

    with pytest.raises(PreemptedError):
        execute(recipe, workdir, RunnerSet.cloud(cloud), attempts={"train": 1})

    assert marker.exists()
    assert not (workdir.out_dir("train") / "trained.ply").exists()

    execute(recipe, workdir, RunnerSet.cloud(cloud), attempts={"train": 2})

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["resumedFromStep"] is None
    assert metrics["iterations"] == 300
    assert metrics["attempts"] == 2
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    assert [entry.state for entry in ledger.entries] == ["preempted", "succeeded"]
    assert ledger.usd is not None and ledger.usd > 0
    log = workdir.log_path("train").read_text()
    assert "restarts from step 0" in log
    assert "starting at step 0" in log
