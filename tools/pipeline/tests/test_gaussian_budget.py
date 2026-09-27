"""The gaussian budget: a capture's supported surface, counted in its own finest pixels.

Synthetic models with known answers -- flat patches of SfM points under cameras at known
distances -- so the footprint (depth / focal) and the area are known in closed form and
the budget's arithmetic can be checked rather than trusted. What the constant *should* be
is not testable here; `gaussian_budget.py` says what it was calibrated on (3DGS's Truck
model, measured 2026-09-27 at 1.52M) and `train_metrics.json`'s `budget` is where a real
capture's numbers land.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import gaussian_budget
import sfm
import support_mask
from executor import execute
from recipe import load_recipe
from runners import LocalRunner, RunnerSet
from synthetic_scene import look_at, write_model
from test_train_gsplat import seed_inputs, stand_in_params, train_recipe
from workdir import Workdir

#: Points 2 cm apart, cameras 1,000 px of focal length.
SPACING = 0.02
FOCAL = 1000.0


def patch(centre: tuple[float, float, float], side: float) -> np.ndarray:
    """A square of points in the z = centre[2] plane, `SPACING` apart."""
    ticks = np.arange(-side / 2, side / 2 + 1e-9, SPACING)
    x, y = np.meshgrid(ticks + centre[0], ticks + centre[1])
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, centre[2])], axis=1)


def model(
    directory: Path,
    xyz: np.ndarray,
    heights: list[float],
    *,
    scale: float = 1.0,
    width: int = 1600,
    height: int = 1200,
    focal: float = FOCAL,
) -> Path:
    """Cameras straight above the origin at each of `heights`, looking down, all of them
    observing every point -- so each point's finest footprint is (nearest camera's depth to
    its plane) / focal. `scale` multiplies every length, as a model in other units would."""
    cameras = []
    for level in heights:
        centre = np.array([0.0, 0.0, level]) * scale
        cameras.append((look_at(centre, np.zeros(3), up=np.array([0.0, 1.0, 0.0])), centre))
    write_model(directory, cameras, width=width, height=height, focal=focal)
    count = xyz.shape[0]
    track = np.stack(
        [
            np.tile(np.arange(1, len(cameras) + 1), count),
            np.repeat(np.arange(count), len(cameras)),
        ],
        axis=1,
    ).astype(np.uint32)
    sfm.write_points3d(
        directory / "points3D.bin",
        sfm.Points3D(
            ids=np.arange(1, count + 1, dtype=np.uint64),
            xyz=xyz * scale,
            rgb=np.full((count, 3), 128, dtype=np.uint8),
            error=np.full(count, 0.5),
            track=track,
            track_offsets=np.arange(0, count * len(cameras) + 1, len(cameras), dtype=np.int64),
        ),
    )
    return directory


def auto(directory: Path, **overrides: Any) -> gaussian_budget.Budget:
    kwargs: dict[str, Any] = {
        "model": directory,
        "train_size": (1600, 1200),
        "pixel_scale": 1.0,
        "floor": 1,
    }
    kwargs.update(overrides)
    budget = gaussian_budget.plan("auto", **kwargs)
    assert budget is not None
    return budget


# --- the footprint and the surface ---------------------------------------------------------


def test_a_points_footprint_is_its_nearest_cameras_depth_over_focal(tmp_path: Path) -> None:
    directory = model(tmp_path / "m", patch((0.0, 0.0, 0.0), 0.2), heights=[2.0, 5.0])

    _xyz, finest = gaussian_budget.finest_footprints(directory)
    _xyz, halved = gaussian_budget.finest_footprints(directory, pixel_scale=0.5)

    assert np.allclose(finest, 2.0 / FOCAL)
    # Frames trained at half the size have half the focal length in pixels.
    assert np.allclose(halved, 2.0 * 2.0 / FOCAL)


def test_a_flat_patch_counts_its_area_in_footprints(tmp_path: Path) -> None:
    """1 m^2 seen from 2 m at 1,000 px: a 2 mm footprint, 250,000 footprints^2 --
    give or take the voxels a square's edges only partly fill."""
    directory = model(tmp_path / "m", patch((0.0, 0.0, 0.0), 1.0), heights=[2.0])

    budget = auto(directory, density=1.0)

    assert budget.mode == "auto"
    assert budget.inside.median_footprint == pytest.approx(0.002)
    assert budget.footprints == pytest.approx(250_000, rel=0.15)
    assert budget.raw == round(budget.footprints)


def test_the_units_of_the_model_do_not_matter(tmp_path: Path) -> None:
    """COLMAP's scale is arbitrary: the same scene in millimetres gets the same budget."""
    xyz = patch((0.0, 0.0, 0.0), 1.0)
    metres = auto(model(tmp_path / "m", xyz, heights=[2.0]))
    millimetres = auto(model(tmp_path / "mm", xyz, heights=[2.0], scale=1000.0))

    assert millimetres.cap == metres.cap
    assert millimetres.inside.area == pytest.approx(metres.inside.area * 1e6)


def test_frames_trained_larger_get_more_gaussians_for_the_same_surface(tmp_path: Path) -> None:
    """2,400 px frames against 1,600 px ones: (2400 / 1600)^2 = 2.25x the footprints."""
    directory = model(tmp_path / "m", patch((0.0, 0.0, 0.0), 1.6), heights=[2.0])

    at_1600 = auto(directory, train_size=(1600, 1200), pixel_scale=1.0)
    at_2400 = auto(directory, train_size=(2400, 1800), pixel_scale=1.5)

    assert at_2400.footprints / at_1600.footprints == pytest.approx(2.25, rel=0.2)


def test_near_surface_outweighs_far_surface_by_the_square_of_its_resolution(
    tmp_path: Path,
) -> None:
    """Two equal patches, one 1 m below the cameras and one 4 m: the near one is seen at a
    quarter of the footprint and so holds sixteen times the pixels. Each octave of footprint
    has its own voxel size, so neither is counted in the other's cells."""
    near = patch((-3.0, 0.0, 1.0), 1.0)  # cameras at z = 2
    far = patch((3.0, 0.0, -2.0), 1.0)
    directory = model(tmp_path / "m", np.concatenate([near, far]), heights=[2.0])

    both = auto(directory, density=1.0)
    near_only = auto(
        directory, density=1.0, region_contains=lambda xyz: xyz[:, 0] < 0, outside_weight=0.0
    )

    far_share = both.footprints - near_only.footprints
    assert near_only.footprints / far_share == pytest.approx(16.0, rel=0.25)


def test_a_refines_region_counts_in_full_and_the_rest_at_its_weight(tmp_path: Path) -> None:
    near = patch((-3.0, 0.0, 0.0), 1.0)
    far = patch((3.0, 0.0, 0.0), 1.0)
    directory = model(tmp_path / "m", np.concatenate([near, far]), heights=[2.0])

    whole = auto(directory, density=1.0)
    region = auto(
        directory, density=1.0, region_contains=lambda xyz: xyz[:, 0] < 0, outside_weight=0.1
    )

    assert region.outside is not None and region.outside_weight == 0.1
    assert region.inside.footprints == pytest.approx(whole.footprints / 2, rel=0.05)
    assert region.footprints == pytest.approx(
        region.inside.footprints + 0.1 * region.outside.footprints
    )
    document = region.to_dict()
    assert document["outsideWeight"] == 0.1 and document["outside"] is not None


# --- the clamp -------------------------------------------------------------------------------


def test_the_floor_and_both_ceilings_say_which_one_applied(tmp_path: Path) -> None:
    directory = model(tmp_path / "m", patch((0.0, 0.0, 0.0), 1.0), heights=[2.0])

    raw = auto(directory, density=1.0)
    floored = auto(directory, density=1.0, floor=10 * (raw.raw or 0))
    capped = auto(directory, density=1.0, budget_max=1_000)
    tiny_gpu = auto(directory, density=1.0, gpu_memory_gb=3.0)  # ~115k at 1600x1200

    assert (raw.clamp, raw.cap) == (None, raw.raw)
    assert (floored.clamp, floored.cap) == ("floor", floored.floor)
    assert (capped.clamp, capped.cap) == ("budget-max", 1_000)
    assert tiny_gpu.clamp == "gpu-memory" and tiny_gpu.cap == tiny_gpu.memory_ceiling


def test_the_memory_ceiling_falls_with_frame_size_and_rises_with_the_card() -> None:
    """The model in `gaussian_budget.py`: on the L4's 24 GB, ~8.7M at 1600x900, ~5.4M at
    2400x1350, and 1616 bytes a gaussian at gsplat's own benchmark resolution."""
    at_1600 = gaussian_budget.memory_ceiling(1600 * 900)
    at_2400 = gaussian_budget.memory_ceiling(2400 * 1350)

    assert 8_000_000 < at_1600 < 9_500_000
    assert 5_000_000 < at_2400 < 6_000_000
    assert gaussian_budget.memory_ceiling(1600 * 900, 48.0) > 2 * at_1600
    assert (
        gaussian_budget.PARAMETER_BYTES + gaussian_budget.RASTER_BYTES == 1616
    )  # gsplat's measured slope


def test_the_recipe_is_bounded_by_the_gpu_not_by_the_worker(tmp_path: Path) -> None:
    """photo-reconstruct no longer sets the 2M `budget_max` the whole-splat stages after
    training needed: a surface big enough gets the L4's ceiling (~8.7M at 1600x900), and a
    run's own `budget_max` still caps below it."""
    train = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "train").params
    assert "budget_max" not in train
    directory = model(tmp_path / "m", patch((0.0, 0.0, 0.0), 1.0), heights=[2.0])
    params = {
        "floor": int(train["budget_floor"]),
        "gpu_memory_gb": float(train["gpu_memory_gb"]),
        "train_size": (1600, 900),
    }
    huge = auto(directory, density=1e6, **params)
    assert huge.clamp == "gpu-memory"
    assert huge.cap == gaussian_budget.memory_ceiling(1600 * 900) > 8_000_000
    assert auto(directory, density=1e6, budget_max=3_000_000, **params).cap == 3_000_000


def test_an_explicit_cap_is_used_as_given() -> None:
    budget = gaussian_budget.plan(
        123_456, model=Path("/nowhere"), train_size=(1600, 900), pixel_scale=1.0
    )

    assert budget is not None and (budget.mode, budget.cap, budget.raw) == (
        "explicit",
        123_456,
        None,
    )
    assert (
        gaussian_budget.plan(None, model=Path("/nowhere"), train_size=None, pixel_scale=None)
        is None
    )


def test_a_model_that_cannot_be_read_falls_back_to_the_old_cap(tmp_path: Path) -> None:
    (tmp_path / "cameras.bin").write_bytes(b"colmap")

    unreadable = auto(tmp_path)
    unsized = auto(tmp_path, train_size=None, pixel_scale=None)

    assert (unreadable.mode, unreadable.cap) == ("fallback", gaussian_budget.FALLBACK_CAP)
    assert "could not be read" in unreadable.reason
    assert unsized.mode == "fallback" and "size is unknown" in unsized.reason


@pytest.mark.parametrize(
    ("given", "parsed"), [(None, None), ("auto", "auto"), ("AUTO", "auto"), (500_000, 500_000)]
)
def test_cap_max_is_auto_or_a_positive_integer(given: object, parsed: object) -> None:
    assert gaussian_budget.parse_cap(given) == parsed


@pytest.mark.parametrize("given", [0, -5, "lots", True, 1.5e9 + 0.5])
def test_anything_else_is_refused_by_name(given: object) -> None:
    with pytest.raises(ValueError, match="cap_max"):
        gaussian_budget.parse_cap(given)


def test_a_bigger_budget_may_have_a_longer_maximum_schedule_up_to_twice() -> None:
    assert gaussian_budget.schedule_factor(500_000) == 1.0
    assert gaussian_budget.schedule_factor(1_000_000) == 1.0
    assert gaussian_budget.schedule_factor(2_000_000) == 1.41
    assert gaussian_budget.schedule_factor(4_000_000) == 2.0
    assert gaussian_budget.schedule_factor(40_000_000) == 2.0


# --- in the train stage -----------------------------------------------------------------------


def seed_measurable(workdir: Workdir, *, size: tuple[int, int] = (64, 48)) -> None:
    """Real frames, and a real model of the same size whose points the budget can measure."""
    seed_inputs(workdir, frames=4)
    poses = workdir.input_path("poses")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (poses / name).unlink()
    # Off-centre by half a spacing, so no point sits exactly on a support mask's face.
    model(
        poses,
        patch((0.01, 0.01, 0.0), 1.0),
        heights=[2.0, 2.5, 3.0, 3.5],
        width=size[0],
        height=size[1],
        focal=50.0,
    )
    frames = workdir.input_path("frames")
    for index in range(4):
        Image.new("RGB", size, (60 * index, 90, 160)).save(frames / f"frame_{index:04d}.jpg")


def train_metrics(workdir: Workdir) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        (workdir.out_dir("train") / "train_metrics.json").read_text()
    )
    return document


def test_auto_measures_the_capture_and_hands_the_trainer_the_cap(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir)

    params = stand_in_params(strategy="mcmc", cap_max="auto", budget_floor=1, gaussian_density=2.0)
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    document = train_metrics(workdir)
    budget = document["budget"]
    cfg = json.loads((workdir.work_dir("train") / "gsplat" / "cfg.yml").read_text())
    assert budget["mode"] == "auto" and budget["clamp"] is None
    # 1 m^2 seen from 2 m by a 50 px focal at 64 px wide: a 4 cm footprint, ~625 of them.
    assert budget["surface"]["medianFootprint"] == pytest.approx(0.04)
    assert budget["capMax"] == round(2.0 * budget["footprints"])
    assert budget["trainImageSize"] == [64, 48] and budget["pixelScale"] == 1.0
    assert cfg["cap_max"] == budget["capMax"] == document["settings"]["capMax"]
    assert document["settings"]["capMaxRequested"] == "auto"
    step = json.loads(workdir.step_path("train").read_text())
    assert step["metrics"]["gaussianBudget"] == budget["capMax"]
    assert step["metrics"]["budgetClamp"] == "none"
    assert step["metrics"]["budgetVoxels"] == budget["surface"]["voxels"]
    assert "cap_max auto ->" in workdir.log_path("train").read_text()


def test_auto_reads_the_frames_at_the_size_training_will(tmp_path: Path) -> None:
    """`train_max_side` halves the frames, so the same surface is a quarter the pixels."""
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(64, 48))

    params = stand_in_params(
        strategy="mcmc", cap_max="auto", budget_floor=1, gaussian_density=2.0, train_max_side=32
    )
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    budget = train_metrics(workdir)["budget"]
    assert budget["trainImageSize"] == [32, 24] and budget["pixelScale"] == 0.5
    assert budget["surface"]["medianFootprint"] == pytest.approx(0.08)


def test_a_support_mask_budgets_only_what_it_holds_in_full(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir)
    # Occupied: the patch's x < 0 half only -- voxels (0, 0, 0) and (0, 1, 0), which are
    # C-order linear indices 0 and 1, one run [0, 2).
    mask = support_mask.SupportMask(
        origin=(-1.0, -1.0, -0.5),
        voxel=1.0,
        dims=(2, 2, 1),
        starts=np.array([0], dtype=np.int64),
        stops=np.array([2], dtype=np.int64),
    )

    params = stand_in_params(
        strategy="mcmc",
        cap_max="auto",
        budget_floor=1,
        gaussian_density=2.0,
        support_mask=mask.to_dict(),
    )
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))

    budget = train_metrics(workdir)["budget"]
    assert budget["outside"] is not None and budget["outsideWeight"] == 0.1
    # 25 of the patch's 51 columns are at x < 0.
    assert (budget["surface"]["points"], budget["outside"]["points"]) == (25 * 51, 26 * 51)
    assert budget["footprints"] == pytest.approx(
        budget["surface"]["footprints"] + 0.1 * budget["outside"]["footprints"], abs=1
    )


def test_auto_on_a_model_it_cannot_read_trains_under_the_old_cap(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_inputs(workdir)

    execute(
        train_recipe(stand_in_params(strategy="mcmc", cap_max="auto")),
        workdir,
        RunnerSet(cpu=LocalRunner()),
    )

    document = train_metrics(workdir)
    assert document["budget"]["mode"] == "fallback"
    assert document["settings"]["capMax"] == gaussian_budget.FALLBACK_CAP
    assert "fallback" in workdir.log_path("train").read_text()


def test_a_phones_density_scale_multiplies_the_measured_budget(tmp_path: Path) -> None:
    runs = {}
    for scale in (1.0, 2.0):
        workdir = Workdir.create(tmp_path / f"run{scale:g}")
        seed_measurable(workdir)
        params = stand_in_params(
            strategy="mcmc",
            cap_max="auto",
            budget_floor=1,
            gaussian_density=2.0,
            density_scale=scale,
        )
        execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))
        runs[scale] = train_metrics(workdir)["budget"]

    assert runs[2.0]["densityScale"] == 2.0
    assert runs[2.0]["capMax"] == pytest.approx(2 * runs[1.0]["capMax"], abs=1)
