from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.bench.recipe import PIPELINE, make_recipe, patch_seed, train_params


def test_full_is_production_screen_does_not_change_capacity_overrides_last() -> None:
    source = yaml.safe_load((PIPELINE / "recipes/photo-reconstruct.yaml").read_text())
    production = next(s["params"] for s in source["stages"] if s["id"] == "train")
    assert train_params("full", {}) == production
    screen = train_params("screen", {})
    assert screen == {**production, "schedule_scale": 0.1, "train_max_side": 800}
    changed = train_params("screen", {"train": {"cap_max": 123456, "schedule_scale": 0.2}})
    assert changed["cap_max"] == 123456 and changed["schedule_scale"] == 0.2
    assert train_params("full", {"train": {"pose_opt": True}})["pose_opt"] is True
    recipe = make_recipe(changed, "l4")
    assert recipe.inputs == ("frames", "poses")
    assert len(recipe.stages) == 1 and recipe.stages[0].id == "train"


def test_seed_patch_once_and_next_to_original(tmp_path: Path) -> None:
    original = tmp_path / "simple_trainer.py"
    source = "def init() -> None:\n    set_random_seed(42 + local_rank)\n    another()\n"
    original.write_text(source)
    patched = patch_seed(original, 43)
    assert patched.parent == original.parent
    assert patched.name == "simple_trainer_bench_seed43.py"
    assert patched.read_text() == source.replace("42 +", "43 +")
    assert original.read_text() == source
    assert patch_seed(original, 43).read_bytes() == patched.read_bytes()


@pytest.mark.parametrize("count", [0, 2])
def test_seed_patch_fails_loudly_on_drift(tmp_path: Path, count: int) -> None:
    original = tmp_path / "trainer.py"
    original.write_text("set_random_seed(42 + local_rank)\n" * count)
    with pytest.raises(ValueError, match="exactly one"):
        patch_seed(original, 43)
    assert not original.with_name("trainer_bench_seed43.py").exists()


def test_seed_42_is_unpatched_even_if_source_shape_changes(tmp_path: Path) -> None:
    original = tmp_path / "trainer.py"
    original.write_text("production")
    assert patch_seed(original, 42) == original
    assert list(tmp_path.iterdir()) == [original]


@pytest.mark.parametrize(
    "override",
    [
        {"pose": {}},
        {"train": {"extra_args": ["--data_dir", "/test"]}},
        {"train": {"data_factor": 2}},
    ],
)
def test_recipe_refuses_input_split_or_pose_bypasses(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        train_params("full", override)
