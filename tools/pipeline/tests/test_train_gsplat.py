"""`train` / `gsplat`: everything around the trainer, and nothing that needs a GPU.

**No training run has been executed anywhere in this repository.** `gsplat` needs CUDA,
the machine this was written on has none, and no GPU provider was reachable from it. So
what is tested here is the dispatch: the COLMAP dataset the trainer is handed, the argv it
is given, the checkpoint layout that lets a preempted attempt continue, the metrics read
back out of its output, and the PLY normalised into `canonical.ply`.

`gsplat_stand_in.py` plays the trainer. It is named that way in every test below so no
reader can mistake one of these for a training measurement -- it writes made-up numbers
into the file layout gsplat uses, and the only thing it is evidence about is this
project's half of the arrangement.

The two facts it stands in for -- the trainer's CLI and its output file names -- are
transcribed from `gsplat/examples/simple_trainer.py` rather than observed, and
`training.py` says so. If they are wrong, `parse_metrics` records nulls and the stage
raises on the missing PLY; neither invents a result.
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

#: A real `stats/val_step<n>_rank0.json` shape, as gsplat writes it.
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


def test_the_argv_is_gsplats_and_disables_the_viewer() -> None:
    """`--disable_viewer` is not cosmetic: the default opens a server and blocks, which
    on a headless preemptible box is a stage that is billed until the tier runs out."""
    argv = training.gsplat_argv(
        "python3", Path("/t/simple_trainer.py"), Path("/d"), Path("/r"), max_steps=7000
    )

    assert argv[:3] == ["python3", "/t/simple_trainer.py", "default"]
    assert argv[argv.index("--data_dir") + 1] == "/d"
    assert argv[argv.index("--result_dir") + 1] == "/r"
    assert argv[argv.index("--max_steps") + 1] == "7000"
    assert "--disable_viewer" in argv
    assert "--ckpt" not in argv


def test_a_checkpoint_turns_into_the_resume_flag() -> None:
    argv = training.gsplat_argv(
        "python3",
        Path("/t/simple_trainer.py"),
        Path("/d"),
        Path("/r"),
        checkpoint=Path("/r/ckpts/ckpt_6999_rank0.pt"),
        extra=["--data_factor", "2"],
    )

    assert argv[argv.index("--ckpt") + 1] == "/r/ckpts/ckpt_6999_rank0.pt"
    assert argv[-2:] == ["--data_factor", "2"]


def test_checkpoints_are_ordered_by_step_and_not_by_mtime(tmp_path: Path) -> None:
    """A checkpoint pulled back out of object storage has whatever mtime the copy gave
    it, so "newest file" would be a fact about the transfer rather than the training."""
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    for step in (500, 29000, 7000):
        (ckpts / f"ckpt_{step}_rank0.pt").write_text("x")
    (ckpts / "ckpt_500_rank0.pt").touch()  # the oldest step, the newest mtime

    newest = training.latest_checkpoint(tmp_path)
    assert newest is not None and newest.name == "ckpt_29000_rank0.pt"
    assert training.latest_checkpoint(tmp_path / "nothing") is None


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
    (stats / "val_step6999_rank0.json").write_text(json.dumps({"psnr": 1.0, "num_GS": 1}))
    (stats / "val_step29999_rank0.json").write_text(json.dumps(REAL_FORMAT_STATS))

    metrics = training.parse_metrics(tmp_path, requested_iterations=30000)

    assert metrics.source == "stats"
    # The furthest-along step, not the last file the glob happened to return.
    assert metrics.iterations == 29999
    assert metrics.psnr == pytest.approx(28.417469)
    assert metrics.ssim == pytest.approx(0.912458)
    assert metrics.lpips == pytest.approx(0.104227)
    assert metrics.gaussians == 412733
    assert metrics.requested_iterations == 30000


def test_metrics_fall_back_to_the_trainers_stdout(tmp_path: Path) -> None:
    log = "Step 6999: 412,733 GSs\nStep 13999: 508,120 GSs\npsnr: 26.11\n"

    metrics = training.parse_metrics(tmp_path, log)

    assert metrics.source == "stdout"
    assert metrics.iterations == 13999
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
    assert (out / "canonical.ply").read_bytes()[:3] == b"ply"
    metrics = json.loads((out / "train_metrics.json").read_text())
    assert metrics["source"] == "stats"
    assert metrics["iterations"] == 300
    assert metrics["requestedIterations"] == 300
    assert metrics["gaussiansInPly"] == 64
    assert metrics["resumedFromStep"] is None
    assert metrics["masksIgnored"] is False
    assert metrics["trainer"] == "gsplat:gsplat_stand_in.py"
    log = workdir.log_path("train").read_text()
    assert "--disable_viewer" in log and "--ckpt " not in log


def test_the_result_directory_lives_inside_the_checkpoint_so_a_kill_keeps_it(
    tmp_path: Path,
) -> None:
    """The one design decision in this stage, asserted rather than described.

    `checkpoint/` is the only directory the executor keeps between attempts and the only
    one B1b's syncer copies out while the stage is still running. A trainer whose
    `--result_dir` is anywhere else loses everything the moment the box goes.
    """
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))

    checkpoints = sorted((workdir.checkpoint_dir("train") / "gsplat" / "ckpts").iterdir())
    assert [p.name for p in checkpoints][-1] == "ckpt_300_rank0.pt"
    assert not any(workdir.work_dir("train").rglob("ckpt_*.pt"))


def test_a_trainer_that_writes_no_ply_is_a_failure_rather_than_an_empty_artifact(
    tmp_path: Path,
) -> None:
    """The failure mode `training.py` names out loud.

    gsplat's output file names are transcribed from its repository rather than observed
    here. If they are wrong, this is what happens: the stage raises and says the splat is
    missing, instead of producing a `canonical.ply` of nothing.
    """
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    params = stand_in_params(extra_args=["--ckpt-every", "100", "--gaussians", "64", "--no-ply"])

    with pytest.raises(Exception, match=r"wrote no \.ply"):
        execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    assert not (workdir.out_dir("train") / "canonical.ply").exists()


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


# --- preemption and resume, through the cloud seam -----------------------------------


def test_a_killed_training_attempt_resumes_from_its_checkpoint(tmp_path: Path) -> None:
    """A genuine SIGTERM mid-run, through `SubprocessAdapter` and `CloudRunner`.

    The stand-in is killed at step 150 of 300. The second attempt is handed `--ckpt` and
    starts at 100 -- the last checkpoint that was written and synced -- rather than at
    zero, which is the whole reason `--result_dir` is inside `checkpoint/`.
    """
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = SubprocessAdapter(transfer, tmp_path / "sandbox", rates={"l4": Rate(0.80, "test")})
    cloud = CloudRunner(
        Placement((adapter,)), transfer, poll_interval_s=0.02, checkpoint_every_s=0.05
    )
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    params = stand_in_params(
        extra_args=["--ckpt-every", "100", "--gaussians", "64", "--die-at", "150", "--kill-parent"]
    )
    recipe = train_recipe(params, gpu={"tier": "l4", "preemptible": True})

    with pytest.raises(PreemptedError):
        execute(recipe, workdir, RunnerSet.cloud(cloud), attempts={"train": 1})

    kept = sorted((workdir.checkpoint_dir("train") / "gsplat" / "ckpts").iterdir())
    assert [p.name for p in kept] == ["ckpt_100_rank0.pt"]
    assert not (workdir.out_dir("train") / "canonical.ply").exists()

    execute(recipe, workdir, RunnerSet.cloud(cloud), attempts={"train": 2})

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["resumedFromStep"] == 100
    assert metrics["iterations"] == 300
    assert metrics["attempts"] == 2
    ledger = AttemptLedger.read(workdir.attempts_path("train"))
    assert [entry.state for entry in ledger.entries] == ["preempted", "succeeded"]
    assert ledger.usd is not None and ledger.usd > 0
    assert "resuming at step 100" in workdir.log_path("train").read_text()
