import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sfm
from experiments.bench import __main__ as cli
from experiments.bench import ledger
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
