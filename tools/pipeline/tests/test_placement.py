"""`placement.json`: `place` records the similarity it applied, as one step."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gaussians
import placement
import stages
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir


def test_the_placement_is_places_two_steps_as_one() -> None:
    rng = np.random.default_rng(2)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotation *= np.sign(np.linalg.det(rotation))
    recentred = SimpleNamespace(
        rotation=np.eye(3), translation=np.array([0.3, -0.2, 1.5]), scale=1.0
    )
    xyz = rng.normal(size=(20, 3))
    expected = stages.place_points(xyz, rotation, [1.0, 2.0, 3.0], 2.5, recentred)  # type: ignore[arg-type]
    document = placement.compose(rotation, [1.0, 2.0, 3.0], 2.5, recentred)
    r, t = np.array(document["rotation"]), np.array(document["translation"])
    got = float(document["scale"]) * xyz @ r.T + t  # type: ignore[arg-type]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def _splat_columns(count: int, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(count, 3)) * [2.0, 2.0, 0.5]
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    columns = {axis: xyz[:, i] for i, axis in enumerate("xyz")}
    columns.update({f"f_dc_{i}": rng.normal(0, 0.5, count) for i in range(3)})
    columns["opacity"] = rng.uniform(0.0, 4.0, count)
    columns.update({f"scale_{i}": rng.normal(-3.5, 0.3, count) for i in range(3)})
    columns.update({f"rot_{i}": quat[:, i] for i in range(4)})
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


def test_place_writes_the_similarity_it_applied(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    columns = _splat_columns(3000)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, columns)
    angle = math.radians(40.0)
    frame = {
        "source": "test",
        "scale": 1.8,
        "rotation": [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        "translationM": [2.0, 1.0, -0.5],
        "recentre": True,
    }
    workdir.input_path("georef.json").write_text(
        json.dumps({"lat": 1.0, "lon": 2.0, "height": 0.0, "frame": frame})
    )
    recipe = make_recipe(
        [{"id": "place", "impl": "place_splat"}], inputs=["trained.ply", "georef.json"]
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    placed = gaussians.read_splat(workdir.artifact_path("place", "canonical.ply"))
    document = json.loads(workdir.artifact_path("place", "placement.json").read_text())
    r, t = np.array(document["rotation"]), np.array(document["translation"])
    xyz = np.stack([columns[a] for a in "xyz"], 1).astype(np.float64)
    moved = document["scale"] * xyz @ r.T + t
    assert document["scale"] == pytest.approx(1.8)
    np.testing.assert_allclose(moved, placed.xyz, atol=1e-4)
