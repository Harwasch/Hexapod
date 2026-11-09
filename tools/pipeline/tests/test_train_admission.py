"""Admission chooses capacity before paying for a remote training call."""

from pathlib import Path
from typing import Any

import pytest
from test_gaussian_budget import seed_measurable, train_metrics

import blocks
import gaussian_budget as gb
import train_admission
from executor import execute
from recipe import load_recipe
from runners import LocalRunner, RunnerSet
from test_train_gsplat import stand_in_params, train_recipe
from workdir import Workdir

PIXELS = 1080 * 1920


@pytest.mark.parametrize(
    ("maximum", "count", "tier", "parts"),
    [
        ("l40s", 5_000_000, "l4", 1),
        ("l40s", 10_000_000, "l40s", 1),
        ("l40s", 20_000_000, "l40s", 2),
        ("l40s", 30_000_000, "l40s", 3),
        ("a100", 5_000_000, "l4", 1),
        ("a100", 10_000_000, "l40s", 1),
        ("a100", 20_000_000, "a100", 1),
        ("a100", 30_000_000, "a100", 2),
    ],
)
def test_choose_capacity_before_blocks(maximum: str, count: int, tier: str, parts: int) -> None:
    budget = gb.Budget(mode="auto", cap=count, raw=count, train_size=(1080, 1920))
    admitted = train_admission.admit(budget, "l4", maximum)
    assert admitted.gpu_tier == tier
    assert admitted.tier_reason
    assert admitted.memory_ceiling == gb.memory_ceiling(PIXELS, gb.GPU_MEMORY_GB[tier])
    assert blocks.block_count(None, admitted)[0] == parts


def test_density_default_and_gpu_max_are_distinct(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    params = {"cap_max": "auto", "budget_floor": 1}
    density = train_admission.plan(params, "l4", workdir.input_path("poses"), (1080, 1920), 1.0)
    maximum = train_admission.plan(
        {**params, "budget_mode": "gpu-max"}, "l4", workdir.input_path("poses"), (1080, 1920), 1.0
    )
    assert density is not None and maximum is not None
    assert density.gpu_tier == "l4" and density.cap < maximum.cap
    assert maximum.gpu_tier == "l40s"
    assert maximum.cap == maximum.raw == gb.memory_ceiling(PIXELS, 48)
    assert blocks.block_count(None, maximum)[0] == 1


def test_gpu_max_still_obeys_budget_max(tmp_path: Path) -> None:
    budget = train_admission.plan(
        {"cap_max": "auto", "budget_mode": "gpu-max", "budget_max": 1_000_000},
        "l4",
        tmp_path,
        (1080, 1920),
        1.0,
    )
    assert budget is not None and budget.cap == budget.raw == 1_000_000
    assert budget.gpu_tier == "l4"


@pytest.mark.parametrize("params", [{"max_gpu_tier": "typo"}, {"budget_mode": "typo"}])
def test_bad_policy_is_refused(params: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train_admission.plan({"cap_max": "auto", **params}, "l4", tmp_path, (1080, 1920), 1.0)


def test_explicit_preview_cap_stays_fixed_and_inherits_opacity(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir)
    shipped = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "train")
    assert shipped.params["budget_mode"] == "density"
    assert shipped.params["max_gpu_tier"] == "l40s"
    params = stand_in_params(**{**shipped.params, "cap_max": 200_000, "schedule_scale": 0.1})
    execute(train_recipe(params), workdir, RunnerSet(cpu=LocalRunner()))
    document = train_metrics(workdir)
    assert document["settings"]["opacityReg"] == 0.002
    assert document["budget"]["capMax"] == 200_000
    assert document["budget"]["gpuTier"] == "l4"
    assert document["budget"]["tierReason"]
    argv = (workdir.log_path("train")).read_text()
    assert "--opacity_reg 0.002" in argv


@pytest.mark.parametrize(
    ("maximum", "count", "tier"),
    [
        ("l40s", 5_000_000, "l4"),
        ("l40s", 10_000_000, "l40s"),
        ("l40s", 20_000_000, "l40s"),
        ("a100", 20_000_000, "a100"),
        ("a100", 30_000_000, "a100"),
    ],
)
def test_selected_tier_reaches_request_step_and_cost_ledger(
    tmp_path: Path, maximum: str, count: int, tier: str
) -> None:
    from adapters import FakeAdapter, LocalTransfer
    from cloud import AttemptLedger, CloudRunner, Placement, run_cost
    from modal_adapter import ModalAdapter

    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    poses = workdir.input_path("poses")
    measured = gb.plan("auto", model=poses, train_size=(1080, 1920), pixel_scale=1.0)
    assert measured is not None
    params = {
        "cap_max": "auto",
        "strategy": "mcmc",
        "max_gpu_tier": maximum,
        "gaussian_density": count / measured.footprints,
    }
    transfer = LocalTransfer(tmp_path / "bucket")
    prices = ModalAdapter("unused")
    rate = prices.rate(tier)
    assert rate is not None
    adapter = FakeAdapter(transfer, tmp_path / "remote", rates={tier: rate})
    runner = CloudRunner(Placement((adapter,)), transfer, poll_interval_s=0, sleep=lambda _: None)
    result = execute(
        train_recipe(params, gpu={"tier": "l4"}), workdir, RunnerSet(cpu=LocalRunner(), gpu=runner)
    )
    req = adapter.submitted[0]
    assert req.tier == tier
    assert req.params[train_admission.PLAN_PARAM]["recipeTier"] == "l4"
    assert req.params[train_admission.PLAN_PARAM]["pixels"] == PIXELS
    entries = AttemptLedger.read(workdir.attempts_path("train")).entries
    assert entries and all(entry.tier == tier for entry in entries)
    assert all(entry.usd == round(rate.usd_for(entry.billed_s), 6) for entry in entries)
    assert run_cost(workdir).tier == tier
    assert result.steps[0].gpu_tier == tier


def test_preflight_geometry_matches_assembled_dataset(tmp_path: Path) -> None:
    import training
    from stages import _training_frame

    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    frames, poses = workdir.input_path("frames"), workdir.input_path("poses")
    predicted = train_admission.frame_geometry(frames, poses, max_side=800)
    dataset = training.build_dataset(frames, poses, tmp_path / "dataset", max_side=800)
    assert predicted == _training_frame(dataset, 1)
    assert predicted[0] == (450, 800)


def test_promoted_call_and_container_agree_on_budget(tmp_path: Path) -> None:
    from adapters import LocalTransfer, SubprocessAdapter
    from cloud import CloudRunner, Placement
    from modal_adapter import ModalAdapter

    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    measured = gb.plan(
        "auto", model=workdir.input_path("poses"), train_size=(1080, 1920), pixel_scale=1.0
    )
    assert measured is not None
    params = stand_in_params(
        strategy="mcmc",
        cap_max="auto",
        gaussian_density=10_000_000 / measured.footprints,
        opacity_reg=0.002,
    )
    transfer = LocalTransfer(tmp_path / "bucket")
    rate = ModalAdapter("unused").rate("l40s")
    assert rate is not None
    adapter = SubprocessAdapter(transfer, tmp_path / "remote", rates={"l40s": rate})
    runner = CloudRunner(Placement((adapter,)), transfer, poll_interval_s=0.02)
    result = execute(
        train_recipe(params, gpu={"tier": "l4"}), workdir, RunnerSet(cpu=LocalRunner(), gpu=runner)
    )
    document = train_metrics(workdir)
    assert result.steps[0].gpu_tier == "l40s"
    assert document["budget"]["gpuTier"] == "l40s"
    assert document["budget"]["recipeGpuTier"] == "l4"
    assert document["budget"]["tierReason"]
    assert document["settings"]["capMax"] == 10_000_000
    assert document["settings"]["opacityReg"] == 0.002
    assert "blocks" not in document


def test_batch_and_explicit_memory_restriction_affect_admission() -> None:
    budget = gb.Budget(mode="auto", cap=10_000_000, raw=10_000_000, train_size=(1080, 1920))
    batched = train_admission.admit(budget, "l4", "a100", images_per_step=2)
    assert batched.gpu_tier == "a100"
    assert batched.memory_ceiling == gb.memory_ceiling(2 * PIXELS, 80)
    restricted = train_admission.admit(budget, "l4", "l40s", memory_override=24)
    assert restricted.gpu_memory_gb == 24
    assert restricted.memory_ceiling == gb.memory_ceiling(PIXELS, 24)
    assert blocks.block_count(None, restricted)[0] == 2


def test_checkpoint_identity_ignores_admission_annotations_but_keeps_policy(tmp_path: Path) -> None:
    import training

    params = {"cap_max": "auto", "max_gpu_tier": "l40s", "opacity_reg": 0.002}
    token = training.checkpoint_token(params, tmp_path)
    annotated = {**params, train_admission.PLAN_PARAM: {"tier": "l40s", "reason": "promoted"}}
    assert training.checkpoint_token(annotated, tmp_path) == token
    assert training.checkpoint_token({**annotated, "max_gpu_tier": "a100"}, tmp_path) != token


def test_an_already_allocated_local_gpu_cannot_promote_itself(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    measured = gb.plan(
        "auto", model=workdir.input_path("poses"), train_size=(1080, 1920), pixel_scale=1.0
    )
    assert measured is not None
    params = stand_in_params(
        strategy="mcmc",
        cap_max="auto",
        gaussian_density=10_000_000 / measured.footprints,
        blocks=1,
    )
    execute(
        train_recipe(params, gpu={"tier": "l4"}),
        workdir,
        RunnerSet(cpu=LocalRunner(), gpu=LocalRunner()),
    )
    budget = train_metrics(workdir)["budget"]
    assert budget["gpuTier"] == "l4"
    assert budget["capMax"] == gb.memory_ceiling(PIXELS, 24)
    assert budget["raw"] == 10_000_000
