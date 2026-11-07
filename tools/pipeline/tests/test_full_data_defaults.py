"""Production keeps the capture's data; fixtures below are dispatch tests, not GPU wins."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from test_remote import request_for

import converge_trainer
import remote
import resume_trainer
import train_split
import training
import video
from adapters import LocalTransfer, SubprocessAdapter
from cloud import CloudRunner, Placement
from conftest import make_recipe
from errors import PreemptedError
from executor import execute
from recipe import load_recipe
from runners import RunnerSet
from test_train_gsplat import seed_inputs, stand_in_params, train_recipe
from workdir import Workdir


def test_recipe_preserves_all_native_data_and_recovers_training() -> None:
    stages = {s.id: s.params for s in load_recipe("photo-reconstruct").stages}
    normalize, train = stages["normalize"], stages["train"]
    assert normalize["select"] == "usable"
    assert normalize["keep"] is normalize["keep_video"] is normalize["max_side"] is None
    assert normalize["max_candidates"] == 0
    assert train["test_every"] == 0
    assert train["checkpoint_every"] == 5000 and train["streaming_export"] is True
    assert train["holdout_error"] is False
    assert training.frame_schedule(2036, train) * train["iterations"] >= 60000


def test_normalize_keeps_more_than_100_distinct_native_photos_and_records_reductions(
    tmp_path: Path,
) -> None:
    work = Workdir.create(tmp_path / "run")
    upload = work.input_path("upload")
    upload.mkdir(parents=True)
    for i in range(103):
        Image.new("RGB", (120, 80), (i, 2 * i, 0)).save(upload / f"p{i:03d}.png")
    (upload / "duplicate.png").write_bytes((upload / "p000.png").read_bytes())
    recipe = make_recipe([{"id": "normalize", "impl": "ffmpeg_frames"}], inputs=["upload"])
    execute(recipe, work, RunnerSet.local())
    meta = json.loads((work.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["frames"]["candidates"] == 104
    assert meta["frames"]["kept"] == 103
    assert meta["frames"]["selectionCounts"] == {"duplicates": 1, "capped": 0, "unreadable": 0}
    assert meta["frameSize"]["rule"] == "source"


def test_video_native_decode_has_neither_sampling_nor_shrink_filter() -> None:
    argv = video.extract_frames_argv(Path("clip.mov"), Path("frame_%05d.jpg"), fps=None)
    assert "-vf" not in argv and argv[argv.index("-vsync") + 1] == "0"
    limited = video.extract_frames_argv(Path("clip.mov"), Path("f%d.jpg"), fps=15, max_side=1600)
    assert "fps=15" in limited[limited.index("-vf") + 1]


def test_near_duplicate_rule_keeps_motion_and_optional_cap(tmp_path: Path) -> None:
    paths = []
    for i, value in enumerate([0, 1, 3, 9, 12]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (20, 20), (value, value, value)).save(path)
        paths.append(path)
    selected, counts = video.usable_frames(paths)
    assert selected == [0, 2, 3, 4] and counts["duplicates"] == 1
    selected, counts = video.usable_frames(paths, cap=2)
    assert len(selected) == 2 and counts["capped"] == 2


@pytest.mark.parametrize("frames", [2036, 4000, 10000])
def test_large_scan_floor_survives_schedule_ceiling_and_batching(frames: int) -> None:
    params = {"schedule_max_scale": 1.0}
    assert training.frame_schedule(4, {"schedule_full_at": 0}) == 1.0
    assert training.frame_schedule(frames, params) >= 2.0
    assert training.frame_schedule(frames, {"schedule_scale": 1.0}) >= 2.0
    for batch in (1, 2, 4, 8):
        scaler = training.batch_steps_scaler(training.frame_schedule(frames, params), batch)
        assert training.scaled_steps(30000, scaler) * batch >= 60000
    assert training.frame_schedule(frames, {"schedule_scale": 0.1}) == 0.1


def test_no_holdout_adapter_uses_every_index_and_never_calls_validation() -> None:
    class Dataset:
        def __init__(self, parser: Any, split: str = "train", **kwargs: Any) -> None:
            self.indices = np.arange(len(parser.image_names))
            self.indices = self.indices[self.indices % parser.test_every != 0]

    class Runner:
        def eval(self, *args: Any) -> None:
            raise AssertionError("validation must be skipped")

    parser = SimpleNamespace(test_every=0, image_names=list("abcdefghijk"))
    module = {"Dataset": Dataset}
    train_split.install(module, Runner, SimpleNamespace(test_every=0))
    assert module["Dataset"](parser).indices.tolist() == list(range(11))
    assert len(module["Dataset"](parser, split="val").indices) == 0
    assert parser.test_every == 0
    Runner().eval(3)
    assert training.held_out_split(2036, 0) == (2036, 0)


def test_no_holdout_dispatch_reports_unavailable_scores(tmp_path: Path) -> None:
    work = Workdir.create(tmp_path / "run")
    seed_inputs(work)
    params = stand_in_params(test_every=0, checkpoint_every=5, iterations=19, live=False)
    execute(train_recipe(params), work, RunnerSet.local())
    document = json.loads((work.out_dir("train") / "train_metrics.json").read_text())
    assert document["heldOut"]["trainFrames"] == 4 and document["heldOut"]["valFrames"] == 0
    assert all(document[k] is None for k in ("psnr", "ssim", "lpips"))
    assert resume_trainer.newest(work.checkpoint_dir("train") / "resume") is not None


def test_cloud_preemption_resumes_the_synced_optimizer_checkpoint(tmp_path: Path) -> None:
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = SubprocessAdapter(transfer, tmp_path / "sandbox")
    runner = CloudRunner(
        Placement((adapter,)), transfer, poll_interval_s=0.02, checkpoint_every_s=0.05
    )
    work = Workdir.create(tmp_path / "run")
    seed_inputs(work)
    params = stand_in_params(
        test_every=0,
        checkpoint_every=5,
        iterations=19,
        live=False,
        extra_args=[
            "--gaussians",
            "64",
            "--die-at",
            "8",
            "--die-marker",
            str(tmp_path / "died"),
            "--kill-parent",
            "--sync-grace-s",
            "0.3",
        ],
    )
    recipe = train_recipe(params, gpu={"tier": "l4", "preemptible": True})
    with pytest.raises(PreemptedError):
        execute(recipe, work, RunnerSet.cloud(runner), attempts={"train": 1})
    execute(recipe, work, RunnerSet.cloud(runner), attempts={"train": 2})
    doc = json.loads((work.out_dir("train") / "train_metrics.json").read_text())
    assert doc["resumedFromStep"] == 5 and doc["iterations"] == 19
    assert doc["psnr"] is None


def test_sync_pins_files_during_rotation_and_excludes_partial_writes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "resume_000000005.pt").write_bytes(b"old state")
    (checkpoint / ".resume_000000010.pt.part").write_bytes(b"partial")

    class RotatingTransfer(LocalTransfer):
        def put(self, key: str, source: Path) -> int:
            (checkpoint / "resume_000000005.pt").unlink()
            assert not list(source.glob("*.part"))
            assert (source / "resume_000000005.pt").read_bytes() == b"old state"
            return super().put(key, source)

    transfer = RotatingTransfer(tmp_path / "bucket")
    remote._sync(request_for("unused", produces=()), transfer, checkpoint)


def test_gpu_side_modules_still_parse_on_python_310() -> None:
    for module in (training, resume_trainer, train_split, converge_trainer):
        assert module.__file__ is not None
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))


def test_blocks_train_every_assigned_camera_without_a_hidden_val_frame(tmp_path: Path) -> None:
    from test_blocks import block_params, run_blocks, seed_prior, seed_walk

    work = Workdir.create(tmp_path / "run")
    seed_walk(work)
    seed_prior(work, tmp_path / "scratch")
    doc = run_blocks(work, block_params(test_every=0, checkpoint_every=5, iterations=19))
    assert doc["heldOut"]["valFrames"] == 0 and doc["psnr"] is None
    assert doc["blocks"]["evaluation"]["status"] == "off"
    for block in doc["blocks"]["blocks"]:
        assert block["cameras"]["val"] == 0 and block["cameras"]["testEvery"] == 0


def test_convergence_cannot_save_before_its_minimum_floor(tmp_path: Path) -> None:
    import convergence

    class Runner:
        stats_dir = str(tmp_path)
        render_dir = str(tmp_path / "renders")

        def train(self) -> None:
            pass

        def eval(self, step: int, stage: str = "val") -> None:
            (tmp_path / f"{stage}_step{step:04d}.json").write_text(json.dumps({"psnr": 20.0}))

    cfg = SimpleNamespace(
        max_steps=20,
        strategy=SimpleNamespace(refine_stop_iter=0),
        ply_steps=[],
        save_steps=[],
        eval_steps=[],
    )
    state = converge_trainer._State(convergence.Rule(every=1, window=2), tmp_path)
    state.min_steps = 8
    converge_trainer.install(Runner, cfg, state)
    runner = Runner()
    for step in range(7):
        runner.eval(step)
    assert cfg.save_steps == [8]
    with pytest.raises(converge_trainer._ConvergedError):
        runner.eval(7)
    assert state.stopped_at is not None and state.stopped_at + 1 == 8


def test_block_share_schedule_cannot_shorten_a_beach_sized_block_below_floor() -> None:
    from test_blocks import settings_for

    import blocks

    settings = settings_for(10000, block_schedule="share", steps_scaler=2.0)
    assert blocks.block_schedule(settings, 2036, 10000)[0] >= 2.0


def test_large_scan_floor_also_covers_blocks_with_smaller_assigned_pools() -> None:
    from test_blocks import settings_for

    import blocks

    settings = settings_for(2036, steps_scaler=1.0)
    assert blocks.block_schedule(settings, 1000, 2036)[0] >= 2.0


def test_merged_bench_evaluation_keeps_metric_flags_without_resuming_training() -> None:
    extra = training.train_extra({"test_every": 8}, Path("/checkpoint/resume"))
    extra += ["--checkpoint-token", "dataset", "--lpips_net", "vgg"]
    assert training.evaluation_extra(extra) == ["--test_every", "8", "--lpips_net", "vgg"]


@pytest.mark.parametrize("factor", [1, 2])
def test_native_memory_budget_covers_the_largest_registered_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factor: int
) -> None:
    import sfm
    import stages

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 48)).save(images / "a.png")
    Image.new("RGB", (128, 96)).save(images / "b.png")
    model = SimpleNamespace(
        cameras=[
            SimpleNamespace(id=1, width=64, height=48),
            SimpleNamespace(id=2, width=128, height=96),
        ],
        images=[
            SimpleNamespace(name="a.png", camera_id=1),
            SimpleNamespace(name="b.png", camera_id=2),
        ],
    )
    monkeypatch.setattr(sfm, "read_model", lambda _: model)
    size, scale = stages._training_frame(tmp_path, factor)
    assert size == (128 // factor, 96 // factor)
    assert scale == 1 / factor  # Intrinsics still follow the pinned parser's reference image.
