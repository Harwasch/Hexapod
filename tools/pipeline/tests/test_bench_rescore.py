"""Rescore uses one capped call and shared frozen geometry, including in rehearsal."""

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_bench_matched import A, B, pair

from experiments.bench import __main__ as cli
from experiments.bench import ledger, rescore
from experiments.bench.scene import read_json, write_json

C, D = "c" * 32, "d" * 32


def inputs(state: Path) -> dict[str, Any]:
    manifest, _ = pair(state)
    shutil.copytree(state / "runs" / A, state / "runs" / C)
    shutil.copytree(state / "runs" / A, state / "runs" / D)
    (state / "runs" / D / "trained.ply").unlink()
    shutil.copyfile(state / "runs" / B / "trained.ply", state / "runs" / B / "faded.ply")
    for run in (A, B, C, D):
        path = state / "runs" / run / "result.json"
        data = read_json(path)
        data.update(runId=run, variant={"name": "test-" + run[0]})
        if run == B:
            data["trainParams"]["step_max_side"] = 32
        write_json(path, data)
    return manifest


def test_rehearsal_rescores_four_kept_plys_and_reports_missing_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "rehearsal"
    inputs(state)
    original = (state / "runs" / A / "result.json").read_bytes()
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("rehearsal imported Modal"))
    monkeypatch.setattr(cli, "run_training", lambda *a, **kw: pytest.fail("rescore trained"))
    assert (
        cli.main(
            [
                "rescore",
                "--scene",
                "beach",
                "--runs",
                A,
                C,
                B,
                D,
                "--rehearse",
                "--state-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    batches = list((state / "runs").glob("rescore-*/rescore.json"))
    assert len(batches) == 1
    document = read_json(batches[0])
    assert document["status"] == "complete" and document["costUsd"] == 0
    assert len(document["rows"]) == 4
    assert document["unavailable"][0]["runId"] == D
    assert "classic" in document["renderer"]
    assert (batches[0].parent / "comparison-overhead-90.png").is_file()
    assert (batches[0].parent / "cameras.json").is_file()
    assert len(read_json(state / "runs" / B / "rescore.json")["rows"]) == 2
    assert (state / "runs" / B / "rescore-renders" / "row-2-overhead-90.png").is_file()
    assert (state / "runs" / A / "result.json").read_bytes() == original
    assert (
        cli.main(["rescore-table", "--runs", A, B, C, "--rehearse", "--state-dir", str(tmp_path)])
        == 0
    )
    output = capsys.readouterr().out
    assert "(differs)" in output and "classic" in output.lower()
    report = read_json(state / "rescore-table.json")
    assert len(report["tables"]["full"]) == len(report["tables"]["half"]) == 4
    assert report["unavailable"]
    entries = ledger.read(state / "ledger.jsonl")
    assert [e["event"] for e in entries] == ["planned", "finished"]
    assert entries[0]["timeoutSeconds"] == 2400 and entries[0]["tag"] == "rescore"


def test_inventory_selects_exact_ply_and_refuses_mismatched_frames(tmp_path: Path) -> None:
    manifest = inputs(tmp_path)
    plan = {"scene": "beach", "runs": [A, B], "ply": ["faded"], "rehearsal": True}
    rows, missing = rescore.inventory(tmp_path / "runs", plan, manifest)
    assert [row["runId"] for row in rows] == [B]
    assert [row["runId"] for row in missing] == [A]
    with pytest.raises(ValueError, match="distinct resolved"):
        rescore.inventory(tmp_path / "runs", {**plan, "runs": [A, A[:8]]}, manifest)
    bad = copy.deepcopy(manifest)
    bad["digest"] = "changed"
    with pytest.raises(ValueError, match="identical"):
        rescore.inventory(tmp_path / "runs", plan, bad)


def test_rescore_reservation_and_total_cap_fail_before_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "REPO", tmp_path / "not-repo")
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("cap must precede Modal"))
    args = cli.parser().parse_args(["rescore", "--scene", "beach", "--runs", A, B])
    plan = cli.plans_for(args)
    priced = ledger.price_plan([], plan, 1, 5)
    assert priced
    assert ledger.TIMEOUTS["rescore"] == 2400
    assert (
        ledger.TIMEOUTS["rescore"] + ledger.OVERHEAD_SECONDS + ledger.RESTART_SECONDS
    ) * ledger.hourly_rate("l4") / 3600 < 1
    assert (
        cli.main(
            [
                "rescore",
                "--scene",
                "beach",
                "--runs",
                A,
                "--total-cap",
                ".01",
                "--state-dir",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_transport_rejects_paths_and_report_refuses_mixed_yardsticks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch ID"):
        rescore.save_local(tmp_path, {"runId": "../../escape"})
    with pytest.raises(ValueError, match="PNG name"):
        rescore.save_local(tmp_path, {"runId": "rescore-" + A, "pngs": {"../oops.png": "AAAA"}})
    with pytest.raises(ValueError, match="identical"):
        rescore.table([{"manifestDigest": "a"}, {"manifestDigest": "b"}])


def test_no_training_and_classic_renderer_are_structural_contracts() -> None:
    source = (Path(__file__).parents[1] / "experiments/bench/rescore_gpu.py").read_text()
    assert 'rasterize_mode="classic"' in source
    assert "with_ut=False" in source and "with_eval3d=False" in source
    assert "value[~far]" in source
    assert "optimizer" not in source.split("def score_all", 1)[1].split("def cpu_self_check", 1)[0]
    modal = (Path(__file__).parents[3] / "infra/modal/bench.py").read_text()
    assert 'timeout=TIMEOUTS["rescore"]' in modal
    assert "run_rescore(" in modal


def test_training_report_skips_rescore_batches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "rehearsal"
    write_json(
        state / "runs" / ("rescore-" + A) / "result.json",
        {
            "schemaVersion": 1,
            "schedule": "rescore",
            "status": "complete",
            "rows": [],
        },
    )
    assert cli.main(["report", "--rehearse", "--state-dir", str(tmp_path)]) == 0
    assert "Frozen-input bench" in capsys.readouterr().out
