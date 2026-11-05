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
    assert entries[0]["timeoutSeconds"] == 7200 and entries[0]["tag"] == "rescore"


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
    priced = ledger.price_plan([], plan, 3, 5)
    assert priced
    assert ledger.TIMEOUTS["rescore"] == 7200
    assert (
        ledger.TIMEOUTS["rescore"] + ledger.OVERHEAD_SECONDS + ledger.RESTART_SECONDS
    ) * ledger.hourly_rate("l4") / 3600 < 3
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


def test_oversized_preview_payload_is_reported_before_returning_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs(tmp_path)
    monkeypatch.setattr(rescore, "MAX_PNG_BASE64_BYTES", 64, raising=False)
    result = rescore.run_rescore(
        tmp_path,
        tmp_path / "runs" / ("rescore-" + C),
        {
            "runId": "rescore-" + C,
            "scene": "beach",
            "runs": [A, B],
            "rehearsal": True,
            "benchCommit": "test",
            "timeoutSeconds": 7200,
        },
    )
    assert result["status"] == "failed"
    assert "PNG payload" in result["error"]
    assert result["pngs"] == {}
    assert (tmp_path / "runs" / ("rescore-" + C) / "scores.json").exists()


def test_recorded_training_size_wins_over_frozen_size(tmp_path: Path) -> None:
    from experiments.bench.rescore_maths import row_base

    manifest = inputs(tmp_path)
    write_json(
        tmp_path / "runs" / B / "train_memory.json",
        {"samples": [{"width": 450, "height": 800, "step": 50000}]},
    )
    sources, _ = rescore.inventory(
        tmp_path / "runs",
        {"scene": "beach", "runs": [B], "rehearsal": True, "ply": ["trained"]},
        manifest,
    )
    row = row_base(sources[0], [1600, 900], [1599, 899])
    assert row["trainedAt"] == {
        "width": 450,
        "height": 800,
        "source": "train_memory.json: last recorded training step",
    }


def test_rehearsal_odd_roi_report_and_overheads_are_independent(tmp_path: Path) -> None:
    from experiments.bench.rescore_maths import shared_cameras

    manifest = inputs(tmp_path)
    sources, _ = rescore.inventory(
        tmp_path / "runs",
        {"scene": "beach", "runs": [A], "rehearsal": True, "ply": ["trained"]},
        manifest,
    )
    cameras = shared_cameras([[x, y, 0] for x in range(4) for y in range(4)], [], [0, 0, 1])
    original = copy.deepcopy(cameras)
    settings = {
        "frozenSize": [1600, 900],
        "datasetSize": [1599, 899],
        "sources": sources,
        "cameras": cameras,
    }
    rescore.synthetic(settings, tmp_path)
    document = read_json(tmp_path / "scores.json")
    document.update(status="complete", rehearsal=True)
    full = document["rows"][0]["resolutions"]["full"]["scoredAt"]
    half = document["rows"][0]["resolutions"]["half"]["scoredAt"]
    assert full == {"height": 1598, "width": 898}
    assert half == {"height": 799, "width": 449}
    assert cameras == original and cameras["cameras"][0]["width"] == 384
    markdown, report = rescore.table([document])
    assert "1598x898" in markdown and "799x449" in markdown
    assert report["tables"]["full"][0]["scoredAt"] == full
    with pytest.raises(ValueError, match="protocol"):
        rescore.table([document, {**document, "protocol": "rescore-v1"}])


def test_worst_case_generated_pngs_fit_return_budget() -> None:
    # Conservative PNG upper bound: raw RGB/gray scanlines + zlib expansion,
    # 64KiB IDAT chunks with headers, and fixed IHDR/IEND. Covers incompressible
    # 640-square photo previews; actual portrait previews are smaller.
    def encoded(height: int, width: int, channels: int) -> int:
        raw = height * (width * channels + 1)
        zlib = raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + 13
        png = zlib + 12 * ((zlib + 65535) // 65536) + 100
        return 4 * ((png + 2) // 3)

    photos = 4 * 2 * 5 * encoded(640, 640, 3)
    overheads = 4 * 3 * (2 * encoded(384, 384, 3) + encoded(384, 384, 1))
    panels = 3 * encoded(429, 4 * 384, 3)
    masks = 3 * encoded(384, 384, 1)
    assert photos + overheads + panels + masks < rescore.MAX_PNG_BASE64_BYTES
