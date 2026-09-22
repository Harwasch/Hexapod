"""Shared helpers. The pipeline is imported by bare module name, as in tools/captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recipe import Recipe
from workdir import Workdir


def make_recipe(stages: list[dict[str, Any]], *, name: str = "test", inputs: list[str]) -> Recipe:
    return Recipe.from_dict({"name": name, "version": 1, "inputs": inputs, "stages": stages})


def seeded_workdir(root: Path, *, upload: bool = True) -> Workdir:
    workdir = Workdir.create(root)
    if upload:
        target = workdir.input_path("upload")
        target.mkdir(parents=True, exist_ok=True)
        (target / "capture.ply").write_bytes(b"not really a splat, but deterministic bytes")
    return workdir


def tree(root: Path) -> dict[str, bytes]:
    """Every file under `root`, keyed by relative path. The unit of a determinism check."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def workdir(tmp_path: Path) -> Workdir:
    return seeded_workdir(tmp_path / "run")
