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
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import gaussians
import progress
import sfm
import support_mask
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
    # The trainer's progress bar reaches the stage log, so the worker can read it back.
    assert progress.latest(log) == progress.Progress(300, 300, 9, 1)


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


# --- a schedule sized to the capture ----------------------------------------------------


def test_a_small_capture_gets_a_proportionally_shorter_schedule_with_a_floor() -> None:
    assert training.schedule_scale(4, full_at=60, floor=0.25) == 0.25
    assert training.schedule_scale(30, full_at=60, floor=0.25) == 0.5
    assert training.schedule_scale(60, full_at=60, floor=0.25) == 1.0
    assert training.schedule_scale(400, full_at=60, floor=0.25) == 1.0
    assert training.schedule_scale(4, full_at=0, floor=0.25) == 1.0
    assert training.scaled_steps(30_000, 0.25) == 7_500


def test_the_schedule_is_scaled_by_the_trainer_and_the_cap_only_goes_to_mcmc() -> None:
    argv = training.gsplat_argv(
        "python",
        Path("t.py"),
        Path("d"),
        Path("r"),
        strategy="mcmc",
        steps_scaler=0.25,
        cap_max=500_000,
    )
    # The step lists stay unscaled: `adjust_steps` scales them alongside max_steps.
    assert argv[argv.index("--max_steps") + 1] == "30000"
    assert argv[argv.index("--ply_steps") + 1] == "30000"
    assert argv[argv.index("--steps_scaler") + 1] == "0.25"
    assert argv[argv.index("--strategy.cap-max") + 1] == "500000"
    assert "--steps_scaler" not in training.gsplat_argv("p", Path("t"), Path("d"), Path("r"))
    with pytest.raises(ValueError, match="cap_max"):
        training.gsplat_argv("p", Path("t"), Path("d"), Path("r"), cap_max=10)


def test_a_four_frame_capture_trains_a_quarter_schedule_under_the_cap(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir, frames=4)

    params = stand_in_params(schedule_full_at=16, strategy="mcmc", cap_max=32)
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["requestedIterations"] == 75
    assert metrics["iterations"] == 75
    assert metrics["gaussiansInPly"] == 32
    log = workdir.log_path("train").read_text()
    assert "4 frames -> 0.25 of the 300-step schedule = 75 steps" in log
    last = progress.latest(log)
    assert last is not None and last.total == 75


# --- the quality parameters (the preview / refine contract) ----------------------------


def write_cameras_bin(path: Path, width: int, height: int, focal: float) -> None:
    """One SIMPLE_RADIAL camera in COLMAP's binary layout, which `sfm.read_model` reads."""
    path.write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<IiQQ", 1, 2, width, height)
        + struct.pack("<4d", focal, width / 2.0, height / 2.0, 0.0)
    )


def seed_real_frames(
    workdir: Workdir, *, frames: int = 4, size: tuple[int, int] = (64, 48)
) -> None:
    """Real JPEGs and a real `cameras.bin` of the same size, where the fakes will not do."""
    seed_inputs(workdir, frames=frames)
    frame_dir = workdir.input_path("frames")
    for index in range(frames):
        Image.new("RGB", size, (40 * index, 90, 160)).save(frame_dir / f"frame_{index:04d}.jpg")
    write_cameras_bin(workdir.input_path("poses") / "cameras.bin", *size, focal=50.0)


def ring_points(count: int, *, images: int, radius: float, first_image: int = 1) -> sfm.Points3D:
    """`count` points on a ring of `radius` about the origin, point `i` seen by image
    `first_image + i % images` alone, in `points3D.bin`'s own shape."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    xyz = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)], axis=1)
    image_ids = np.arange(count) % images + first_image
    track = np.stack([image_ids, np.arange(count)], axis=1).astype(np.uint32)
    return sfm.Points3D(
        ids=np.arange(1, count + 1, dtype=np.uint64),
        xyz=xyz,
        rgb=np.full((count, 3), 128, dtype=np.uint8),
        error=np.full(count, 0.5),
        track=track,
        track_offsets=np.arange(count + 1, dtype=np.int64),
    )


def test_the_default_argv_carries_none_of_the_quality_switches() -> None:
    """A run that names none of them is the run every earlier capture got."""
    plain = training.gsplat_argv("p", Path("t"), Path("d"), Path("r"), strategy="mcmc")
    tuned = training.gsplat_argv(
        "p",
        Path("t"),
        Path("d"),
        Path("r"),
        strategy="mcmc",
        antialiased=True,
        opacity_reg=0.001,
        depth_loss=True,
    )

    for switch in ("--antialiased", "--opacity_reg", "--depth_loss"):
        assert switch not in plain
        assert switch in tuned
    assert tuned[tuned.index("--opacity_reg") + 1] == "0.001"


def test_a_requested_schedule_scale_wins_over_the_captures_own(tmp_path: Path) -> None:
    """The preview preset: `schedule_scale` 0.1 on a capture that would get the full one."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir, frames=4)

    params = stand_in_params(schedule_full_at=4, schedule_scale=0.1)
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["requestedIterations"] == 30
    assert metrics["iterations"] == 30
    assert metrics["settings"]["scheduleScale"] == 0.1
    assert metrics["settings"]["scheduleScaleRequested"] == 0.1
    assert "the capture's own would be 1" in workdir.log_path("train").read_text()


@pytest.mark.parametrize("scale", [0.01, 1.5])
def test_a_schedule_scale_outside_its_range_is_refused(tmp_path: Path, scale: float) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    with pytest.raises(Exception, match="schedule_scale"):
        execute(
            train_recipe(stand_in_params(schedule_scale=scale)),
            workdir,
            RunnerSet(cpu=LocalRunner()),
        )


def test_train_max_side_shrinks_the_images_and_not_the_model(tmp_path: Path) -> None:
    """The images shrink, `sparse/0/` is the pose stage's bytes, and the trainer -- here the
    stand-in, mirroring v1.5.3's parser -- would scale the intrinsics by exactly 1/2."""
    workdir = Workdir.create(tmp_path / "run")
    seed_real_frames(workdir, size=(64, 48))
    cameras = (workdir.input_path("poses") / "cameras.bin").read_bytes()

    execute(train_recipe(stand_in_params(train_max_side=32)), workdir, RunnerSet(cpu=LocalRunner()))

    dataset = workdir.work_dir("train") / "dataset"
    with Image.open(dataset / "images" / "frame_0000.jpg") as image:
        assert image.size == (32, 24)
    assert (dataset / "sparse" / "0" / "cameras.bin").read_bytes() == cameras
    cfg = json.loads((workdir.work_dir("train") / "gsplat" / "cfg.yml").read_text())
    assert cfg["image_scale"] == [0.5, 0.5]
    assert cfg["data_factor"] == 1
    metrics = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert metrics["settings"]["trainMaxSide"] == 32
    assert metrics["settings"]["trainImageSize"] == [32, 24]


def test_train_max_side_leaves_frames_that_already_fit_byte_for_byte(tmp_path: Path) -> None:
    frames, poses = tmp_path / "frames", tmp_path / "poses"
    frames.mkdir()
    poses.mkdir()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(frames / "frame_0000.jpg")
    original = (frames / "frame_0000.jpg").read_bytes()

    dataset = training.build_dataset(frames, poses, tmp_path / "work", max_side=64)

    assert (dataset / "images" / "frame_0000.jpg").read_bytes() == original


def test_train_max_side_and_data_factor_are_not_both_given(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    with pytest.raises(Exception, match="give one"):
        execute(
            train_recipe(stand_in_params(train_max_side=32, data_factor=2)),
            workdir,
            RunnerSet(cpu=LocalRunner()),
        )


def test_the_quality_switches_reach_the_trainer_with_the_presets_semantics(
    tmp_path: Path,
) -> None:
    """The refine shape: mcmc, antialiased, opacity_reg over the preset's 0.01, depth loss."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    params = stand_in_params(
        strategy="mcmc", cap_max=48, antialiased=True, opacity_reg=0.001, depth_loss=True
    )
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    cfg = json.loads((workdir.work_dir("train") / "gsplat" / "cfg.yml").read_text())
    assert cfg["antialiased"] is True
    assert cfg["opacity_reg"] == 0.001
    assert cfg["depth_loss"] is True
    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert document["settings"] == {
        "strategy": "mcmc",
        "capMax": 48,
        "scheduleScale": 1.0,
        "scheduleScaleRequested": None,
        "trainMaxSide": None,
        "trainImageSize": None,
        "antialiased": True,
        "opacityReg": 0.001,
        "depthLoss": True,
        "variant": "3dgs",
    }


def test_the_mcmc_preset_keeps_its_own_opacity_reg_when_none_is_given(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(train_recipe(stand_in_params(strategy="mcmc")), workdir, RunnerSet(cpu=LocalRunner()))

    cfg = json.loads((workdir.work_dir("train") / "gsplat" / "cfg.yml").read_text())
    assert cfg["opacity_reg"] == 0.01
    assert cfg["antialiased"] is False and cfg["depth_loss"] is False


def test_a_switch_given_as_a_string_is_refused_rather_than_read_as_true(tmp_path: Path) -> None:
    """`"false"` is a truthy string; a switch is a JSON bool or nothing."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    with pytest.raises(Exception, match="antialiased must be true or false"):
        execute(
            train_recipe(stand_in_params(antialiased="false")),
            workdir,
            RunnerSet(cpu=LocalRunner()),
        )


def test_2dgs_is_refused_by_name_with_the_reason(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    with pytest.raises(Exception, match="writes no PLY"):
        execute(
            train_recipe(stand_in_params(variant="2dgs")), workdir, RunnerSet(cpu=LocalRunner())
        )


# --- held-out metrics ------------------------------------------------------------------


def test_the_held_out_split_is_every_eighth_registered_frame() -> None:
    """v1.5.3: `indices % test_every == 0` are `val` -- 1 of 4, 11 of 87, 13 of 100."""
    assert training.held_out_split(4) == (3, 1)
    assert training.held_out_split(8) == (7, 1)
    assert training.held_out_split(9) == (7, 2)
    assert training.held_out_split(87) == (76, 11)
    assert training.held_out_split(100) == (87, 13)
    assert training.held_out_split(0) == (0, 0)


def test_the_metrics_say_they_are_held_out_and_on_how_many_frames(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir, frames=9)

    execute(train_recipe(stand_in_params()), workdir, RunnerSet(cpu=LocalRunner()))

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert document["heldOut"]["split"] == "val"
    assert document["heldOut"]["testEvery"] == 8
    assert document["heldOut"]["valFrames"] == 2
    assert document["heldOut"]["trainFrames"] == 7
    assert "never trains on" in document["heldOut"]["note"]
    assert document["ssim"] == pytest.approx(0.8712)
    assert document["lpips"] == pytest.approx(0.1431)
    step = json.loads(workdir.step_path("train").read_text())
    assert step["metrics"]["valFrames"] == 2
    assert step["metrics"]["ssim"] == pytest.approx(0.8712)
    assert step["metrics"]["lpips"] == pytest.approx(0.1431)
    assert step["metrics"]["metricsSplit"].startswith("val")


def test_train_seconds_come_from_the_training_stats_not_the_eval_render_time(
    tmp_path: Path,
) -> None:
    """v1.5.3 uses `ellipse_time` in both files: seconds since training began in
    `train_step*`, mean seconds to render one val image in `val_step*`."""
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "val_step29999.json").write_text(json.dumps(REAL_FORMAT_STATS))
    (stats / "train_step29999_rank0.json").write_text(
        json.dumps({"mem": 3.1, "ellipse_time": 1328.4, "num_GS": 412733})
    )

    metrics = training.parse_metrics(tmp_path, registered=87)

    assert metrics.train_seconds == pytest.approx(1328.4)
    assert metrics.eval_seconds_per_image == pytest.approx(0.008542)
    assert metrics.peak_memory_gb == pytest.approx(3.1)
    held_out = metrics.to_dict()["heldOut"]
    assert isinstance(held_out, dict) and held_out["valFrames"] == 11


def test_the_stdout_fallback_reads_ssim_and_lpips_too(tmp_path: Path) -> None:
    log = "PSNR: 26.110, SSIM: 0.8410, LPIPS: 0.172 Time: 0.011s/image Number of GS: 5\n"

    metrics = training.parse_metrics(tmp_path, log)

    assert metrics.ssim == pytest.approx(0.841)
    assert metrics.lpips == pytest.approx(0.172)


# --- the region of interest ------------------------------------------------------------


def test_points3d_survives_a_round_trip_through_this_projects_writer(tmp_path: Path) -> None:
    points = ring_points(12, images=3, radius=2.0)
    path = tmp_path / "points3D.bin"

    sfm.write_points3d(path, points)
    back = sfm.read_points3d(path)

    assert np.array_equal(back.ids, points.ids)
    assert np.allclose(back.xyz, points.xyz)
    assert np.array_equal(back.track, points.track)
    assert np.array_equal(back.track_offsets, points.track_offsets)


def test_an_roi_keeps_its_inside_a_sample_of_the_outside_and_every_frames_view(
    tmp_path: Path,
) -> None:
    """Inside: every point. Outside: one in ten. And no frame is left with fewer than the
    minimum of its own points, since gsplat's depth loss looks every frame up."""
    inner = ring_points(40, images=2, radius=1.0)
    # Images 3-6 see only the outer ring, 50 points each.
    outer = ring_points(200, images=4, radius=10.0, first_image=3)
    merged = sfm.Points3D(
        ids=np.concatenate([inner.ids, outer.ids + 1000]),
        xyz=np.concatenate([inner.xyz, outer.xyz]),
        rgb=np.concatenate([inner.rgb, outer.rgb]),
        error=np.concatenate([inner.error, outer.error]),
        track=np.concatenate([inner.track, outer.track]),
        track_offsets=np.arange(241, dtype=np.int64),
    )
    sfm.write_points3d(tmp_path / "points3D.bin", merged)
    roi = training.Roi(center=(0.0, 0.0, 0.0), radius=2.0)

    crop = training.crop_initial_points(tmp_path, roi, outside_every=10, min_per_image=30)

    assert crop.points_in == 240
    assert crop.inside == 40
    assert crop.outside_sampled == 20
    kept = sfm.read_points3d(tmp_path / "points3D.bin")
    per_image = np.bincount(kept.track[:, 0].astype(np.int64))
    assert all(per_image[image] >= 30 for image in (3, 4, 5, 6))
    # Each outer image had 5 sampled points and is topped up to 30 with its own.
    assert crop.kept_for_coverage == 4 * 25
    assert crop.points_kept == len(kept) == 40 + 20 + 100


def test_a_malformed_roi_is_refused_rather_than_ignored() -> None:
    assert training.Roi.parse(None) is None
    assert training.Roi.parse({"center": [1, 2, 3], "radius": 4}) == training.Roi(
        (1.0, 2.0, 3.0), 4.0
    )
    for bad in (
        {"center": [1, 2], "radius": 1},
        {"center": [1, 2, "x"], "radius": 1},
        {"center": [1, 2, 3], "radius": 0},
        {"center": [1, 2, 3], "radius": True},
        [0, 0, 0, 1],
    ):
        with pytest.raises(ValueError, match="roi"):
            training.Roi.parse(bad)


def test_the_roi_crops_the_initial_points_and_the_trained_splat(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    sfm.write_points3d(
        workdir.input_path("poses") / "points3D.bin", ring_points(100, images=4, radius=3.0)
    )
    params = stand_in_params(roi={"center": [0.0, 0.0, 0.0], "radius": 1.0})

    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    roi = document["roi"]
    assert roi["center"] == [0.0, 0.0, 0.0] and roi["radius"] == 1.0
    assert roi["initialPoints"] == 100
    assert roi["insideRoi"] == 0
    dataset_points = workdir.work_dir("train") / "dataset" / "sparse" / "0" / "points3D.bin"
    assert roi["initialPointsKept"] == len(sfm.read_points3d(dataset_points))
    # The poses artifact is untouched: the crop is the trainer's dataset only.
    assert len(sfm.read_points3d(workdir.input_path("poses") / "points3D.bin")) == 100
    trained = gaussians.read_splat(workdir.out_dir("train") / "trained.ply")
    assert float(np.linalg.norm(trained.xyz, axis=1).max()) <= 1.5
    assert roi["gaussiansTrained"] == 64
    assert roi["gaussiansKept"] == trained.count == document["gaussiansInPly"]
    assert 0 < trained.count < 64


def test_a_support_mask_crops_training_to_whatever_shape_the_data_has(tmp_path: Path) -> None:
    """The mask, not a sphere: here the supported data is one side of the scene only."""
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)
    sfm.write_points3d(
        workdir.input_path("poses") / "points3D.bin", ring_points(100, images=4, radius=3.0)
    )
    rng = np.random.default_rng(3)
    supported = rng.uniform([0.5, -4.0, -4.0], [4.0, 4.0, 4.0], size=(60_000, 3))
    mask = support_mask.build(supported)
    assert mask is not None
    params = stand_in_params(
        support_mask=mask.to_dict(),
        # A mask wins over a sphere given alongside it.
        roi={"center": [0.0, 0.0, 0.0], "radius": 100.0},
    )

    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    document = json.loads((workdir.out_dir("train") / "train_metrics.json").read_text())
    assert document["roi"]["kind"] == "voxels"
    trained = gaussians.read_splat(workdir.out_dir("train") / "trained.ply")
    assert trained.count > 0
    assert mask.contains(trained.xyz).all()
    # Nothing from the unsupported side survives, beyond the mask's own margin.
    assert float(trained.xyz[:, 0].min()) >= 0.5 - 3 * mask.voxel
    assert document["roi"]["gaussiansKept"] == trained.count
