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
        ("l40s", 500_000, "l4", 1),
        ("l40s", 5_000_000, "l4", 1),
        ("l40s", 6_720_000, "l4", 1),
        ("l40s", 3_000_000, "l4", 1),
        ("l40s", 50_000_000, "l40s", 4),
        ("l40s", 8_050_000, "l40s", 1),
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
    assert admitted.raw == count
    assert admitted.cap == (count if tier == "l4" else admitted.memory_ceiling)


def test_auto_default_density_opt_out_and_gpu_max_are_distinct(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    params = {"cap_max": "auto", "budget_floor": 1, "max_gpu_tier": "l40s"}
    automatic = train_admission.plan(params, "l4", workdir.input_path("poses"), (1080, 1920), 1.0)
    density = train_admission.plan(
        {**params, "budget_mode": "density"}, "l4", workdir.input_path("poses"), (1080, 1920), 1.0
    )
    maximum = train_admission.plan(
        {**params, "budget_mode": "gpu-max"}, "l4", workdir.input_path("poses"), (1080, 1920), 1.0
    )
    assert automatic is not None and density is not None and maximum is not None
    assert automatic.budget_mode == "auto"
    assert automatic.gpu_tier == "l4"
    assert automatic.raw is not None
    assert automatic.cap == automatic.raw
    assert automatic.to_dict()["budgetHeadroom"] is None
    assert automatic.raw == density.raw == density.cap
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
    assert shipped.params["budget_mode"] == "auto"
    assert shipped.params.get("budget_headroom") is None
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
        max_gpu_tier="l40s",
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
    assert document["settings"]["capMax"] == gb.memory_ceiling(PIXELS, 48)
    assert document["budget"]["raw"] == document["budget"]["densityTarget"] == 10_000_000
    assert document["budget"]["capMax"] == document["budget"]["memoryCeiling"]
    assert "rule B" in document["budget"]["tierReason"]
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
    assert training.checkpoint_token({**annotated, "budget_headroom": 1.0}, tmp_path) != token


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


@pytest.mark.parametrize(
    "count,tier", [(2_000_000, "l4"), (8_050_000, "l40s"), (20_000_000, "l40s")]
)
def test_density_opt_out_preserves_old_caps(count: int, tier: str) -> None:
    budget = gb.Budget(
        mode="auto", budget_mode="density", cap=count, raw=count, train_size=(1080, 1920)
    )
    admitted = train_admission.admit(budget, "l4", "l40s")
    assert admitted.gpu_tier == tier
    assert admitted.cap == min(count, gb.memory_ceiling(PIXELS, gb.GPU_MEMORY_GB[tier]))


@pytest.mark.parametrize("mode", ["auto", "density"])
def test_budget_max_and_integer_override_win(tmp_path: Path, mode: str) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    params = {"cap_max": "auto", "budget_mode": mode, "budget_max": 1_000_000}
    budget = train_admission.plan(params, "l4", workdir.input_path("poses"), (1080, 1920), 1.0)
    assert budget is not None and budget.raw is not None
    expected = min(1_000_000, max(budget.floor, budget.raw))
    assert budget.cap == expected
    assert budget.gpu_tier == "l4"
    explicit = train_admission.plan(
        {**params, "cap_max": 3_000_000}, "l4", workdir.input_path("poses"), (1080, 1920), 1.0
    )
    assert explicit is not None and explicit.cap == 3_000_000 and explicit.mode == "explicit"


@pytest.mark.parametrize("headroom", [None, 1.0, 2.5])
@pytest.mark.parametrize("count", [2_000_000, 20_000_000, 50_000_000])
@pytest.mark.parametrize("mode", ["auto", "density"])
def test_each_block_uses_its_tier_policy(
    tmp_path: Path, mode: str, count: int, headroom: float | None
) -> None:
    from test_blocks import settings_for

    import block_maths

    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    poses = workdir.input_path("poses")
    measured = gb.plan("auto", model=poses, train_size=(1080, 1920), pixel_scale=1.0)
    assert measured is not None
    budget = train_admission.plan(
        {
            "cap_max": "auto",
            "budget_mode": mode,
            "budget_headroom": headroom,
            "gaussian_density": count / measured.footprints,
            "max_gpu_tier": "l40s",
        },
        "l4",
        poses,
        (1080, 1920),
        1.0,
    )
    assert budget is not None and budget.raw == count
    assert blocks.block_count(None, budget)[0] == (
        4 if count == 50_000_000 else 2 if count == 20_000_000 else 1
    )
    parts = 4 if count == 50_000_000 else 2
    settings = settings_for(
        parts,
        poses=poses,
        budget=budget,
        frame_size=(1080, 1920),
        pixel_scale=1.0,
    )
    xyz, _ = gb.finest_footprints(poses)
    partition = block_maths.partition(xyz, [0.0, 0.0, 1.0], parts, exact=True)
    for index in range(parts):
        region = blocks.BlockRegion(partition, index, margin=0.0)
        cap, part = blocks._block_cap(settings, region, 1 / parts)
        assert part is not None and part.raw is not None
        assert part.memory_ceiling is not None
        expected = min(
            part.memory_ceiling,
            (
                part.memory_ceiling
                if mode == "auto" and headroom is None and part.gpu_tier != part.recipe_gpu_tier
                else max(
                    part.floor,
                    round(part.raw * (headroom if mode == "auto" and headroom is not None else 1)),
                )
            ),
        )
        assert cap == expected
        assert cap <= gb.memory_ceiling(PIXELS, 48)
        assert part.gpu_tier == budget.gpu_tier
        assert part.budget_mode == mode
        assert part.to_dict()["densityTarget"] == part.raw
        assert part.tier_reason


def test_fixed_a100_accepts_a_lower_recipe_maximum(tmp_path: Path) -> None:
    budget = train_admission.plan(
        {"cap_max": "auto", "budget_mode": "gpu-max", "max_gpu_tier": "l40s"},
        "a100",
        tmp_path,
        (1080, 1920),
        1.0,
        fixed_tier=True,
    )
    assert budget is not None and budget.gpu_tier == "a100"
    assert budget.cap == gb.memory_ceiling(PIXELS, 80)


@pytest.mark.parametrize("tier", ["l4", "l40s", "a100"])
def test_omitted_maximum_never_promotes_old_jobs(tmp_path: Path, tier: str) -> None:
    budget = train_admission.plan(
        {"cap_max": "auto", "budget_mode": "gpu-max"},
        tier,
        tmp_path,
        (1080, 1920),
        1.0,
    )
    assert budget is not None and budget.gpu_tier == budget.max_gpu_tier == tier
    assert budget.cap == gb.memory_ceiling(PIXELS, gb.gpu_memory_for(tier))


@pytest.mark.parametrize("maximum,expected", [("l40s", "l4"), ("a100", "a100"), ("a100", "l4")])
@pytest.mark.parametrize("failure", ["pending", "poll-capacity", "submit-capacity"])
def test_unavailable_l40s_replans_without_unsafe_budgets(
    tmp_path: Path, maximum: str, expected: str, failure: str
) -> None:
    from adapters import FakeAdapter, LocalTransfer
    from cloud import AttemptLedger, CloudRunner, Placement, Poll, RemoteHandle, StageRequest

    unavailable = ["l40s", "a100"] if maximum == "a100" and expected == "l4" else ["l40s"]

    class NoL40S(FakeAdapter):
        def submit(self, request: StageRequest) -> RemoteHandle:
            if request.tier in unavailable and failure == "submit-capacity":
                self.submitted.append(request)
                raise RuntimeError("ResourceExhaustedError: no GPU capacity")
            return super().submit(request)

        def poll(self, handle: RemoteHandle) -> Poll:
            if handle.tier in unavailable:
                if failure == "poll-capacity":
                    return Poll(state="failed", detail=f"no {handle.tier} capacity")
                return Poll(state="pending")
            return super().poll(handle)

    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    poses = workdir.input_path("poses")
    measured = gb.plan("auto", model=poses, train_size=(1080, 1920), pixel_scale=1.0)
    assert measured is not None
    params = {
        "cap_max": "auto",
        "strategy": "mcmc",
        "max_gpu_tier": maximum,
        "gaussian_density": 10_000_000 / measured.footprints,
    }
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = NoL40S(transfer, tmp_path / "remote")
    ticks = iter(range(10000))
    runner = CloudRunner(
        Placement((adapter,)),
        transfer,
        poll_interval_s=0,
        sleep=lambda _: None,
        max_pending_s=2,
        clock=lambda: float(next(ticks)),
    )
    recipe = train_recipe(params, gpu={"tier": "l4"})
    result = execute(recipe, workdir, RunnerSet(cpu=LocalRunner(), gpu=runner))
    assert [request.tier for request in adapter.submitted] == [*unavailable, expected]
    request = adapter.submitted[-1]
    record = request.params[train_admission.PLAN_PARAM]
    assert record["unavailableTiers"] == unavailable
    assert "capacity" in record["replanReason"]
    budget = train_admission.plan(
        request.params, request.tier, poses, (1080, 1920), 1.0, fixed_tier=True
    )
    assert budget is not None
    assert budget.cap == min(25_000_000, gb.memory_ceiling(PIXELS, gb.gpu_memory_for(expected)))
    assert record["cap"] == budget.cap
    assert blocks.block_count(None, budget)[0] == (2 if expected == "l4" else 1)
    assert result.steps[0].gpu_tier == expected
    entries = AttemptLedger.read(workdir.attempts_path("train")).entries
    assert [entry.tier for entry in entries] == [*unavailable, expected]
    assert "capacity unavailable" in entries[0].detail
    # A worker retry/deploy retains the exclusion, rather than waiting on L40S again.
    from plan import plan_recipe

    context = runner._context(plan_recipe(recipe).stages[0], workdir, 2)
    assert context.gpu_tier == expected


def test_capacity_downgrade_executes_and_merges_l4_blocks(tmp_path: Path) -> None:
    from test_blocks import block_params, seed_prior, seed_walk

    from adapters import LocalTransfer, SubprocessAdapter
    from cloud import AttemptLedger, CloudRunner, Placement, RemoteHandle, StageRequest

    class NoL40S(SubprocessAdapter):
        def submit(self, request: StageRequest) -> RemoteHandle:
            if request.tier == "l40s":
                raise RuntimeError("no L40S capacity")
            return super().submit(request)

    workdir = Workdir.create(tmp_path / "run")
    seed_walk(workdir)
    seed_prior(workdir, tmp_path / "scratch")
    poses = workdir.input_path("poses")
    measured = gb.plan("auto", model=poses, train_size=(64, 48), pixel_scale=1.0)
    assert measured is not None
    target = gb.memory_ceiling(64 * 48, 24) + 1_000_000
    params = block_params(
        cap_max="auto",
        blocks="auto",
        max_gpu_tier="l40s",
        gaussian_density=target / measured.footprints,
    )
    transfer = LocalTransfer(tmp_path / "bucket")
    adapter = NoL40S(transfer, tmp_path / "remote")
    runner = CloudRunner(Placement((adapter,)), transfer, poll_interval_s=0.02)
    result = execute(
        train_recipe(params, gpu={"tier": "l4"}),
        workdir,
        RunnerSet(cpu=LocalRunner(), gpu=runner),
    )
    document = train_metrics(workdir)
    assert document["budget"]["gpuTier"] == "l4"
    assert document["budget"]["raw"] == target
    assert document["budget"]["capMax"] == gb.memory_ceiling(64 * 48, 24)
    assert result.steps[0].gpu_tier == "l4"
    entries = AttemptLedger.read(workdir.attempts_path("train")).entries
    assert [entry.tier for entry in entries] == ["l40s", "l4", "l4", "l4", "l4"]
    assert sorted(entry.part for entry in entries[2:-1]) == ["b0", "b1"]
    import json

    record = json.loads(str(result.steps[0].metrics["trainAdmission"]))
    assert record["unavailableTiers"] == ["l40s"]


@pytest.mark.parametrize("headroom", [1.0, 1.5, 2.5])
def test_recipe_headroom_and_density_opt_out(tmp_path: Path, headroom: float) -> None:
    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    params = {"cap_max": "auto", "budget_floor": 1, "budget_headroom": headroom}
    auto = train_admission.plan(params, "l4", workdir.input_path("poses"), (1080, 1920), 1.0)
    density = train_admission.plan(
        {**params, "budget_mode": "density"}, "l4", workdir.input_path("poses"), (1080, 1920), 1.0
    )
    assert auto is not None and density is not None and auto.raw is not None
    assert auto.raw == density.raw == density.cap
    assert auto.cap == round(auto.raw * headroom)
    assert auto.to_dict()["budgetHeadroom"] == headroom
    assert density.to_dict()["budgetHeadroom"] == 1.0


@pytest.mark.parametrize("headroom", [0.0, 0.5, float("nan"), float("inf")])
def test_invalid_headroom_is_refused(tmp_path: Path, headroom: float) -> None:
    with pytest.raises(ValueError, match="budget_headroom"):
        train_admission.plan(
            {"cap_max": "auto", "budget_headroom": headroom}, "l4", tmp_path, (1080, 1920), 1.0
        )


@pytest.mark.parametrize("headroom", [None, 2.5])
def test_replanned_l4_blocks_reapply_policy_to_their_own_targets(
    tmp_path: Path, headroom: float | None
) -> None:
    from test_blocks import settings_for

    import block_maths

    workdir = Workdir.create(tmp_path / "run")
    seed_measurable(workdir, size=(1080, 1920))
    poses = workdir.input_path("poses")
    from test_gaussian_budget import model, patch

    model(poses, patch((0.01, 0.01, 0.0), 1.0), [2.0, 2.5, 3.0, 3.5], width=1080, height=1920)
    measured = gb.plan("auto", model=poses, train_size=(1080, 1920), pixel_scale=1.0)
    assert measured is not None
    params = {
        "cap_max": "auto",
        "budget_headroom": headroom,
        "max_gpu_tier": "l40s",
        "gaussian_density": 8_050_000 / measured.footprints,
    }
    budget = train_admission.plan(params, "l4", poses, (1080, 1920), 1.0, unavailable=("l40s",))
    assert budget is not None and budget.gpu_tier == "l4"
    assert blocks.block_count(None, budget)[0] == 2
    # Explicitly finer blocks exercise unsaturated caps on the replanned smaller tier.
    parts = 8
    settings = settings_for(
        parts, poses=poses, budget=budget, frame_size=(1080, 1920), pixel_scale=1.0
    )
    xyz, _ = gb.finest_footprints(poses)
    partition = block_maths.partition(xyz, [0.0, 0.0, 1.0], parts, exact=True)
    unsaturated = 0
    for index in range(parts):
        cap, part = blocks._block_cap(
            settings, blocks.BlockRegion(partition, index, 0.0), 1 / parts
        )
        assert part is not None and part.raw is not None and part.memory_ceiling is not None
        assert part.gpu_tier == "l4" and part.budget_headroom == headroom
        assert cap == min(part.memory_ceiling, max(part.floor, round((headroom or 1) * part.raw)))
        unsaturated += int(cap < part.memory_ceiling)
    assert unsaturated > 0


@pytest.mark.parametrize("count, expected", [(500_000, 1_250_000), (8_050_000, None)])
def test_optional_headroom_on_base_and_promoted_tiers(count: int, expected: int | None) -> None:
    budget = gb.Budget(
        mode="auto", cap=count, raw=count, train_size=(1080, 1920), budget_headroom=2.5
    )
    admitted = train_admission.admit(budget, "l4", "l40s")
    assert admitted.cap == (expected if expected is not None else gb.memory_ceiling(PIXELS, 48))


def test_density_opt_out_on_promoted_tier() -> None:
    budget = gb.Budget(
        mode="auto", cap=8_050_000, raw=8_050_000, train_size=(1080, 1920), budget_mode="density"
    )
    admitted = train_admission.admit(budget, "l4", "l40s")
    assert admitted.gpu_tier == "l40s"
    assert admitted.cap == 8_050_000
