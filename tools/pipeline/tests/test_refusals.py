"""Everything the executor refuses, and whether it refuses before running anything."""

from __future__ import annotations

from pathlib import Path

import pytest

from artifacts import ArtifactDecl
from conftest import make_recipe, seeded_workdir
from contracts import StageContext, StageOutcome
from errors import (
    DuplicateArtifactError,
    MissingArtifactError,
    MissingInputError,
    UndeclaredArtifactError,
    UnknownImplError,
    UnresolvedArtifactError,
)
from executor import execute
from registry import stage_impl
from runners import LocalRunner, RunnerSet


@stage_impl("t_forgets_its_output", produces=(ArtifactDecl("forgotten.json"),))
def _forgets_its_output(ctx: StageContext) -> StageOutcome:
    ctx.log("writing nothing at all")
    return StageOutcome()


@stage_impl("t_writes_undeclared", produces=(ArtifactDecl("declared.json"),))
def _writes_undeclared(ctx: StageContext) -> StageOutcome:
    ctx.output("declared.json").write_text("{}\n", encoding="utf-8")
    (ctx.out_dir / "surprise.bin").write_bytes(b"not declared anywhere")
    return StageOutcome()


@stage_impl("t_makes_a_thing", produces=(ArtifactDecl("thing.json"),))
def _makes_a_thing(ctx: StageContext) -> StageOutcome:
    ctx.output("thing.json").write_text("{}\n", encoding="utf-8")
    return StageOutcome()


def _local() -> RunnerSet:
    return RunnerSet(cpu=LocalRunner())


def test_an_unproduced_input_is_refused_before_anything_runs(tmp_path: Path) -> None:
    recipe = make_recipe(
        # package consumes canonical.ply and georef.json; nothing here produces either.
        [{"id": "package", "impl": "splat_tiles"}],
        inputs=["upload"],
    )
    workdir = seeded_workdir(tmp_path / "run")

    with pytest.raises(UnresolvedArtifactError) as raised:
        execute(recipe, workdir, RunnerSet.stubbed())

    message = str(raised.value)
    assert "recipe 'test'" in message
    assert "stage 'package'" in message
    assert "canonical.ply" in message
    # Nothing ran, and nothing was even created.
    assert not workdir.stages_dir.exists()
    assert not workdir.recipe_path.exists()


def test_an_unknown_impl_names_the_recipe_and_the_stage(tmp_path: Path) -> None:
    recipe = make_recipe(
        [{"id": "train", "impl": "nerfstudio"}], name="photo-reconstruct", inputs=["upload"]
    )
    workdir = seeded_workdir(tmp_path / "run")

    with pytest.raises(UnknownImplError) as raised:
        execute(recipe, workdir, RunnerSet.stubbed())

    message = str(raised.value)
    assert "recipe 'photo-reconstruct'" in message
    assert "stage 'train'" in message
    assert "unknown impl 'nerfstudio'" in message
    # And it says what it does know, so the fix is in the message.
    assert "gsplat" in message
    assert not workdir.stages_dir.exists()


def test_a_stage_that_does_not_write_what_it_declared_fails_loudly(tmp_path: Path) -> None:
    recipe = make_recipe([{"id": "broken", "impl": "t_forgets_its_output"}], inputs=[])
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    with pytest.raises(MissingArtifactError) as raised:
        execute(recipe, workdir, _local())

    message = str(raised.value)
    assert "stage 'broken'" in message
    assert "forgotten.json" in message
    assert "wrote nothing at" in message
    # The stage did run -- the failure is at the end of it, not instead of it.
    assert workdir.log_path("broken").read_text().strip() == "writing nothing at all"


def test_a_stage_that_writes_an_undeclared_output_fails_loudly(tmp_path: Path) -> None:
    recipe = make_recipe([{"id": "sloppy", "impl": "t_writes_undeclared"}], inputs=[])
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    with pytest.raises(UndeclaredArtifactError, match=r"surprise\.bin"):
        execute(recipe, workdir, _local())


def test_two_stages_producing_the_same_artifact_are_refused(tmp_path: Path) -> None:
    recipe = make_recipe(
        [
            {"id": "first", "impl": "t_makes_a_thing"},
            {"id": "second", "impl": "t_makes_a_thing"},
        ],
        inputs=[],
    )

    with pytest.raises(DuplicateArtifactError, match="already produced by stage 'first'"):
        execute(recipe, seeded_workdir(tmp_path / "run", upload=False), _local())


def test_a_recipe_input_that_was_never_seeded_is_refused(tmp_path: Path) -> None:
    recipe = make_recipe([{"id": "normalize", "impl": "ingest_splat"}], inputs=["upload"])
    workdir = seeded_workdir(tmp_path / "run", upload=False)

    with pytest.raises(MissingInputError, match="input 'upload' is missing"):
        execute(recipe, workdir, RunnerSet.stubbed())

    assert not workdir.stages_dir.exists()


def test_an_unimplemented_stage_says_which_step_lands_it(tmp_path: Path) -> None:
    """The stub that is still a stub, asked to run, refuses by name.

    This has followed the frontier of what is implemented: it asked `ffmpeg_frames`
    until B2 made that real, then `exif_gps` until B4 made *that* real. It now asks
    `pose: arkit`, which B4 deliberately left stubbed -- there is no ARKit capture in
    this repository, and the on-disk format is the capture app's rather than Apple's.
    """
    from errors import StageFailedError

    workdir = seeded_workdir(tmp_path / "run")
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "frame_0000.jpg").write_bytes(b"not really a frame, but deterministic bytes")
    recipe = make_recipe([{"id": "pose", "impl": "arkit"}], inputs=["upload", "frames"])

    with pytest.raises(StageFailedError, match="lands in B4"):
        execute(recipe, workdir, _local())
