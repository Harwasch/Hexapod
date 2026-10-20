"""`pose` / `colmap`, `focal_from: exif`: which focal the camera starts from (pure).

The run that proves the seed is refined rather than held is
`test_pose_colmap4.py::test_a_seeded_focal_is_refined_from_where_it_starts`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import stages
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir


class _Context:
    """The three things `_focal_seed` asks of a StageContext."""

    def __init__(self, document: dict[str, Any] | None, root: Path) -> None:
        self.path = root / "frames_meta.json"
        self.logged: list[str] = []
        if document is not None:
            self.path.write_text(json.dumps(document), encoding="utf-8")

    def has_input(self, name: str) -> bool:
        return name == "frames_meta.json" and self.path.is_file()

    def input(self, name: str) -> Path:
        return self.path

    def log(self, line: str) -> None:
        self.logged.append(line)


def frames_meta(px: float | None, source: str, size: list[int]) -> dict[str, Any]:
    return {
        "schema": "frames_meta/1",
        "focal": {"px": px, "from": source},
        "frames": [{"name": "frame_0000.jpg", "frameSize": size, "focal": {"px": px}}],
    }


def _seed(document: dict[str, Any] | None, tmp_path: Path, focal_from: str = "exif") -> Any:
    context = _Context(document, tmp_path)
    return stages._focal_seed(context, focal_from, 2400, 1798)  # type: ignore[arg-type]


def test_the_exif_focal_seeds_the_camera_without_holding_it(tmp_path: Path) -> None:
    calibrated = frames_meta(1687.86, "dji-calibrated", [2400, 1798])

    assert _seed(calibrated, tmp_path) == (1687.86, "dji-calibrated")
    # Off unless asked.
    assert _seed(calibrated, tmp_path, "none") is None
    # The frames disagreed (capture_focal refused): nothing to seed from.
    assert _seed(frames_meta(None, "mixed", [2400, 1798]), tmp_path) is None
    # A focal in pixels of other pixels is not a focal of these frames.
    assert _seed(frames_meta(1687.86, "dji-calibrated", [1600, 1199]), tmp_path) is None
    # No frames_meta.json at all (an older normalize, or splat-ingest).
    assert _seed(None, tmp_path) is None


def test_a_planned_frames_meta_that_is_not_there_seeds_nothing(tmp_path: Path) -> None:
    """A job resumed across a deploy: normalize finished before it wrote frames_meta.json,
    and the new recipe plans one for pose. Nothing to seed from, and no crash."""
    context = _Context(None, tmp_path)
    context.has_input = lambda name: True  # type: ignore[method-assign]

    assert stages._focal_seed(context, "exif", 2400, 1798) is None  # type: ignore[arg-type]
    assert "wrote no frames_meta.json" in context.logged[0]


def test_an_unknown_focal_source_is_refused(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (64, 48)).save(frames / "frame_0000.jpg")
    recipe = make_recipe(
        [{"id": "pose", "impl": "colmap", "params": {"focal_from": "guess"}}], inputs=["frames"]
    )

    with pytest.raises(Exception, match="focal_from must be one of none, exif"):
        execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))


def test_pose_reads_frames_meta_when_a_recipe_hands_it_over() -> None:
    from registry import lookup

    impl = lookup("colmap")
    assert impl is not None
    assert "frames_meta.json" in impl.optional_consumes
