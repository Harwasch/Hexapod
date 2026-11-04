import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sfm
from experiments.bench import __main__ as cli
from experiments.bench import ledger
from experiments.bench.recipe import PIPELINE
from experiments.bench.scene import read_json


def test_rehearsal_end_to_end_split_pool_recipe_execute_parse_ledger_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_modal() -> None:
        raise AssertionError("rehearsal must never import Modal")

    monkeypatch.setattr(cli, "load_modal", no_modal)
    # tmp_path is outside the repository in normal CI; this sandbox uses workspace disk.
    monkeypatch.setattr(cli, "REPO", tmp_path / "not-the-repo")
    monkeypatch.setattr(cli, "git_commit", lambda: "test-commit")
    common = ["--rehearse", "--state-dir", str(tmp_path)]
    assert cli.main(["run", "--scene", "spool", "--seeds", "42,43", *common]) == 0
    state = tmp_path / "rehearsal"
    split = read_json(state / "scenes/spool/split.json")
    rows = [read_json(path) for path in (state / "runs").glob("*/result.json")]
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "complete", row["error"]
        assert row["rehearsal"] and row["costUsd"] == 0
        assert row["splitDigest"] == split["digest"]
        assert {r["name"] for r in row["test"]["perFrame"]} == set(split["test"])
        assert row["test"]["frames"] == 2 and row["splats"] == 64
        assert row["trainParams"]["train_max_side"] == 800
        assert row["trainParams"]["cap_max"] == "auto"
        assert row["val"]["psnr"] is not None
        assert row["trainSeconds"] > 0
        assert not (state / "runs" / row["runId"] / "trained.ply").exists()
        assert (state / "runs" / row["runId"] / "train_metrics.json").is_file()
    assert len(sfm.read_model(state / "scenes/spool/poses").images) == 16
    entries = ledger.read(state / "ledger.jsonl")
    assert len(entries) == 4
    assert [e["event"] for e in entries[:2]] == ["planned", "planned"]
    assert cli.main(["report", *common]) == 0
    output = capsys.readouterr().out
    assert "REHEARSAL" in output and "| psnr | 2 | 20.0000" in output
    assert cli.main(["ledger", *common]) == 0
    assert json.loads(capsys.readouterr().out)["totalUsd"] == 0


def test_dry_run_never_loads_modal_or_writes_and_cap_refuses_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "REPO", tmp_path / "not-the-repo")
    monkeypatch.setattr(cli, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("no remote work"))
    state = tmp_path / "state"
    assert cli.main(["freeze", "--scene", "spool", "--dry-run", "--state-dir", str(state)]) == 0
    assert cli.main(["run", "--scene", "spool", "--dry-run", "--state-dir", str(state)]) == 0
    assert not state.exists()
    assert cli.main(["run", "--scene", "spool", "--cap", "0.01", "--state-dir", str(state)]) == 2
    assert not state.exists()


def test_remote_batch_spawns_at_most_four_before_waiting_and_pins_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = 0
    peak = 0
    seen = []

    class Call:
        def __init__(self, plan: dict[str, Any]) -> None:
            self.plan = plan

        def get(self) -> dict[str, Any]:
            nonlocal active
            active -= 1
            return {"runId": self.plan["runId"], "status": "complete", "costUsd": 0.01}

    class Function:
        def spawn(self, plan: dict[str, Any]) -> Any:
            nonlocal active, peak
            # Every plan was persisted before any launch.
            assert len(ledger.read(tmp_path / "ledger.jsonl")) >= 7
            assert plan["manifestDigest"] == "frozen" and plan["splitDigest"] == "split"
            active += 1
            peak = max(peak, active)
            seen.append(plan["runId"])
            return Call(plan)

    module = SimpleNamespace(
        app=SimpleNamespace(run=contextlib.nullcontext),
        volume=SimpleNamespace(read_file=lambda _: [b'{"digest":"frozen","splitDigest":"split"}']),
        FUNCTIONS={("l4", "screen"): Function()},
    )
    monkeypatch.setattr(cli, "load_modal", lambda: module)
    plans = [
        {"runId": str(i), "scene": "spool", "schedule": "screen", "tier": "l4"} for i in range(7)
    ]
    results = cli.remote_batch(plans, tmp_path, tmp_path / "ledger.jsonl", 10)
    assert len(results) == 7 and peak == 4 and active == 0
    assert seen == [str(i) for i in range(7)]
    assert ledger.summary(ledger.read(tmp_path / "ledger.jsonl"))["totalUsd"] == pytest.approx(0.07)


def test_modal_app_reuses_image_and_gpu_has_no_secret() -> None:
    # Structural guard without importing Modal or contacting any service.
    source = (Path(__file__).resolve().parents[3] / "infra/modal/bench.py").read_text()
    assert 'modal.App("hexapod-bench")' in source
    assert 'modal.Volume.from_name("hexapod-bench", create_if_missing=True)' in source
    assert "image = shared.image" in source
    assert "shared.GPU_CPU_CORES" in source and "shared.GPU_MEMORY_MIB" in source
    gpu_definitions = source.split("FUNCTIONS =", 1)[1]
    assert "secrets=" not in gpu_definitions
    assert "timeout=TIMEOUTS[schedule]" in gpu_definitions
    # The restart marker is committed before any training, so a preempted input that
    # Modal re-runs is refused rather than paid for twice.
    train = source.split("def _train", 1)[1].split("FUNCTIONS =", 1)[0]
    assert train.index("claim(") < train.index("volume.commit()") < train.index("run_training(")


@pytest.mark.parametrize("keep", [False, True])
def test_keep_splat_retains_raw_full_sh_beside_shipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: bool
) -> None:
    from experiments.bench import worker
    from experiments.bench.rehearsal import synthetic_scene
    from holdout_error import read_ply

    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    # Shipping degree 0 (not the recipe's 3), so the shipped PLY lacks the bands the
    # kept full-SH export has.
    args = cli.parser().parse_args(
        [
            "run",
            "--scene",
            "spool",
            "--rehearse",
            "--renders",
            "3",
            "--set",
            '{"train": {"ship_sh_degree": 0}}',
        ]
    )
    plan = cli.plans_for(args)[0]
    plan.update(timeoutSeconds=1800, keepSplat=keep)
    assert plan["renders"] == 3
    scene, run = tmp_path / "scene", tmp_path / "run"
    synthetic_scene(scene, "spool", "test")
    result = worker.run_training(scene, run, plan)
    assert result["status"] == "complete", result["error"]
    assert result["gpuUtilMean"] is None
    assert (run / "gpu_samples.json").is_file()
    assert (run / "trained.ply").exists() == keep
    assert (run / "full_sh.ply").exists() == keep
    if keep:
        assert "f_rest_44" in read_ply(run / "full_sh.ply")
        assert "f_rest_0" not in read_ply(run / "trained.ply")


def test_negative_renders_refused_before_any_launch() -> None:
    args = cli.parser().parse_args(["run", "--scene", "spool", "--renders", "-1"])
    with pytest.raises(ValueError, match="nonnegative"):
        cli.plans_for(args)


def test_quality_stack_only_baseline_enabled_and_public_freeze_plan() -> None:
    variants = PIPELINE / "experiments/variants/quality-stack.yaml"
    args = cli.parser().parse_args(
        [
            "run",
            "--scene",
            "truck",
            "--variants",
            str(variants),
            "--variant",
            "baseline-main",
            "--topdown",
            "--tag",
            "quality-stack",
        ]
    )
    plan = cli.plans_for(args)[0]
    assert plan["variant"]["name"] == "baseline-main" and plan["topdown"]
    args.variant = "upgraded"
    with pytest.raises(ValueError, match="pending implementation"):
        cli.plans_for(args)
    args = cli.parser().parse_args(["freeze", "--scene", "truck"])
    assert cli.plans_for(args)[0]["sourceType"] == "public"
    args.run = "4dd02171"
    with pytest.raises(ValueError, match="pinned archive"):
        cli.plans_for(args)
    args = cli.parser().parse_args(["freeze", "--scene", "beach"])
    plan = cli.plans_for(args)[0]
    assert plan["sourceType"] == "r2" and plan["sourceRun"] == "4dd02171"


def test_total_cap_refused_before_modal_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_modal() -> None:
        raise AssertionError("must not import Modal")

    monkeypatch.setattr(cli, "load_modal", no_modal)
    monkeypatch.setattr(cli, "REPO", tmp_path / "not-repo")
    assert (
        cli.main(
            [
                "run",
                "--scene",
                "truck",
                "--state-dir",
                str(tmp_path),
                "--tag",
                "quality",
                "--total-cap",
                "0.01",
            ]
        )
        == 2
    )
    assert not (tmp_path / "ledger.jsonl").exists()


def test_public_freeze_dispatch_has_no_storage_secret_or_private_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ast

    source = (PIPELINE.parents[1] / "infra/modal/bench.py").read_text()
    public = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "freeze_public"
    )
    decorator = public.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert "secrets" not in {k.arg for k in decorator.keywords}
    calls = []

    def spawn(plan: dict[str, Any]) -> Any:
        calls.append(plan)
        return SimpleNamespace(
            get=lambda: {"runId": plan["runId"], "status": "complete", "costUsd": 0.01}
        )

    # No private function, bucket or volume-read API exists on this stand-in.
    module = SimpleNamespace(
        app=SimpleNamespace(run=contextlib.nullcontext), freeze_public=SimpleNamespace(spawn=spawn)
    )
    monkeypatch.setattr(cli, "load_modal", lambda: module)
    plan = cli.plans_for(cli.parser().parse_args(["freeze", "--scene", "truck"]))[0]
    cli.remote_batch([plan], tmp_path, tmp_path / "ledger.jsonl", 2)
    assert calls == [plan]


def test_results_cli_filters_tag_and_reports_incomplete_flip_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from experiments.bench.scene import write_json

    monkeypatch.setattr(cli, "REPO", tmp_path / "not-repo")
    for run, tag in (("yes", "quality-stack"), ("no", "other")):
        write_json(
            tmp_path / f"runs/{run}/result.json",
            {
                "schemaVersion": 1,
                "runId": run,
                "tag": tag,
                "scene": "truck",
                "variant": {"name": "baseline-main"},
                "status": "complete",
                "schedule": "screen",
            },
        )
    assert (
        cli.main(
            [
                "results",
                "--state-dir",
                str(tmp_path),
                "--tag",
                "quality-stack",
                "--flip-rule",
                "--candidate",
                "upgraded",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "yes / screen" in text and "no / screen" not in text
    assert "incomplete" in text
