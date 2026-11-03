import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gsplat_stand_in import write_ply

import gaussians
from experiments.bench.evaluate import full_sh_columns, region_info
from experiments.bench.results import markdown, parse_result
from experiments.bench.scene import write_json
from holdout_error import read_ply


def result(directory: Path, **kwargs: Any) -> dict[str, Any]:
    plan = {
        "runId": "one",
        "scene": "spool",
        "variant": {"name": "baseline", "overrides": {}},
        "seed": 42,
        "schedule": "screen",
        "tier": "l4",
        "benchCommit": "commit",
        "timeoutSeconds": 1800,
    }
    return parse_result(
        plan,
        directory,
        params={"ship_sh_degree": 0},
        split_digest="split",
        manifest_digest="manifest",
        wall=100,
        train_seconds=90,
        gpu_name="NVIDIA L4",
        **kwargs,
    )


def test_parses_complete_reports_and_shipped_count_not_stale_stats(tmp_path: Path) -> None:
    write_json(
        tmp_path / "train_metrics.json", {"psnr": 25, "ssim": 0.9, "lpips": 0.1, "gaussians": 999}
    )
    write_json(tmp_path / "holdout.json", {"meanPsnrFullSh": 26})
    write_json(
        tmp_path / "test.json",
        {
            "psnr": 22,
            "ssim": 0.8,
            "lpips": 0.2,
            "psnrFullSh": 24,
            "psnrFullShGap": 2,
            "frames": 1,
            "perFrame": [{"name": "test.png", "psnr": 22}],
        },
    )
    write_ply(tmp_path / "trained.ply", 7, 42)
    parsed = result(tmp_path)
    assert parsed["schemaVersion"] == 1 and parsed["status"] == "complete"
    assert parsed["val"] == {"psnr": 25, "ssim": 0.9, "lpips": 0.1, "meanPsnrFullSh": 26}
    assert parsed["test"]["psnrFullShGap"] == 2
    assert parsed["splats"] == 7
    assert parsed["outputBytes"] == (tmp_path / "trained.ply").stat().st_size
    assert parsed["costUsd"] > 100 * 0.7992 / 3600


def test_missing_optional_files_and_failed_run_are_null_not_zero(tmp_path: Path) -> None:
    parsed = result(tmp_path, error="trainer failed")
    assert parsed["status"] == "failed" and parsed["error"] == "trainer failed"
    assert parsed["splats"] is None and parsed["outputBytes"] is None
    assert all(value is None for value in parsed["val"].values())
    assert parsed["test"]["psnr"] is None and parsed["test"]["perFrame"] == []
    assert parsed["costUsd"] > 0
    write_json(tmp_path / "train_metrics.json", {"holdout": {"meanPsnrFullSh": 24}})
    assert result(tmp_path)["val"]["meanPsnrFullSh"] == 24


def test_full_sh_diagnostic_refuses_mismatched_geometry(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    write_ply(raw, 7, 42, 3)
    columns = read_ply(raw)
    stripped = {k: v for k, v in columns.items() if not k.startswith("f_rest_")}
    full, reason = full_sh_columns(stripped, raw)
    assert full is not None and reason is None
    assert np.array_equal(full["f_rest_44"], columns["f_rest_44"])
    cropped = {k: v[[0, 2, 6]] for k, v in stripped.items()}
    full, reason = full_sh_columns(cropped, raw)
    assert full is not None and reason is None
    assert np.array_equal(full["f_rest_44"], columns["f_rest_44"][[0, 2, 6]])
    changed = {k: v.copy() for k, v in stripped.items()}
    changed["x"][0] += 1
    assert full_sh_columns(changed, raw)[0] is None
    reason = full_sh_columns(changed, raw)[1]
    assert reason is not None and "geometry differs" in reason
    assert full_sh_columns(stripped, None)[0] is None


def test_region_hook_never_invents_masks(tmp_path: Path) -> None:
    assert region_info(tmp_path)["regions"] is None
    assert "No per-frame" in region_info(tmp_path)["regionsReason"]
    write_json(tmp_path / "frames_meta.json", {"frames": [{"subjectMask": "a.png"}]})
    assert "no annotated-mask adapter" in region_info(tmp_path)["regionsReason"]


def test_full_sh_uses_shipped_degree_three_without_raw_export(tmp_path: Path) -> None:
    ply = tmp_path / "trained.ply"
    write_ply(ply, 7, 42, 3)
    columns = read_ply(ply)
    full, reason = full_sh_columns(columns, None)
    assert full is columns and reason is None


def test_full_sh_refuses_ambiguous_duplicate_geometry(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    write_ply(raw, 3, 42, 3)
    columns = {k: v.copy() for k, v in read_ply(raw).items()}
    for key, values in columns.items():
        if not key.startswith("f_rest_"):
            values[1] = values[0]
    gaussians.write_ply(raw, columns, sh_degree=3)
    cropped = {k: v[:1] for k, v in columns.items() if not k.startswith("f_rest_")}
    full, reason = full_sh_columns(cropped, raw)
    assert full is None and reason is not None


def test_summary_spread_over_seeds_and_separates_refrozen_scene(tmp_path: Path) -> None:
    a = result(tmp_path)
    a["test"].update(psnr=20, ssim=0.8, lpips=0.2)
    b = copy.deepcopy(a)
    b.update(seed=43, runId="two")
    b["test"].update(psnr=24, ssim=0.9, lpips=0.1)
    table = markdown([a, b])
    assert "| psnr | 2 | 22.0000 | 20.0000 | 24.0000 | 4.0000 |" in table
    b["manifestDigest"] = "refrozen"
    assert "| psnr | 2 |" not in markdown([a, b])
    b["manifestDigest"] = a["manifestDigest"]
    b["status"] = "failed"
    assert "| psnr | 2 |" not in markdown([a, b])


def test_table_shows_full_sh_and_size_and_flags_only_rehearsals(tmp_path: Path) -> None:
    a = result(tmp_path)
    a["test"].update(psnr=22.0, psnrFullSh=25.0, psnrFullShGap=3.0)
    a["outputBytes"] = 32_000_000
    table = markdown([a])
    assert "| full-SH PSNR | splats | MB |" in table
    assert "| 25.0000 |" in table and "| 32.0000 |" in table
    assert "| psnrFullShGap | 1 | 3.0000 |" in table
    assert "Rehearsal rows" not in table
    assert "Rehearsal rows" in markdown([{**a, "rehearsal": True}])


def test_non_numeric_optional_metrics_become_null(tmp_path: Path) -> None:
    write_json(tmp_path / "train_metrics.json", {"psnr": "bad", "ssim": True})
    parsed = result(tmp_path)
    assert parsed["val"]["psnr"] is None and parsed["val"]["ssim"] is None
    assert parsed["trainSeconds"] == pytest.approx(90)


def test_degree_reports_and_gpu_metrics_survive_parsing_and_seed_summary(tmp_path: Path) -> None:
    by_degree = {
        str(d): {
            "psnr": 20 + d,
            "ssim": 0.8,
            "lpips": 0.2,
            "frames": 1,
            "perFrame": [{"name": "frame.png", "psnr": 20 + d}],
        }
        for d in range(4)
    }
    write_json(tmp_path / "test.json", {"psnr": 21, "byDegree": by_degree, "shDegree": 1})
    write_json(tmp_path / "train_metrics.json", {"iterations": 3000})
    write_json(tmp_path / "gpu_samples.json", {"samples": ["10, 100", "30, 200"]})
    a = result(tmp_path)
    assert a["test"]["byDegree"] == by_degree and a["test"]["psnr"] == 21
    assert a["test"]["shDegree"] == 1
    assert a["gpuUtilMean"] == a["gpuUtilP50"] == 20 and a["gpuMemMaxMiB"] == 200
    assert a["iterations"] == 3000 and a["trainSecondsPer1kSteps"] == 30
    b = copy.deepcopy(a)
    b.update(seed=43, runId="two")
    b["test"]["byDegree"]["2"]["psnr"] = 24
    table = markdown([a, b])
    assert "| one | 2 | 1 | 22.0000 | 0.8000 | 0.2000 |" in table
    assert "| sh2.psnr | 2 | 23.0000 | 22.0000 | 24.0000 | 2.0000 |" in table
    assert "train s/1k steps" in table and "GPU mem max MiB" in table
    b["status"] = "failed"
    assert "| sh2.psnr | 2 |" not in markdown([a, b])
    b["status"] = "complete"
    b["manifestDigest"] = "other"
    assert "| sh2.psnr | 2 |" not in markdown([a, b])


@pytest.mark.parametrize("iterations", [None, 0, -1, "bad"])
def test_seconds_per_steps_missing_invalid_or_zero_is_null(tmp_path: Path, iterations: Any) -> None:
    write_json(tmp_path / "train_metrics.json", {"iterations": iterations})
    assert result(tmp_path)["trainSecondsPer1kSteps"] is None


def test_incomplete_raw_sh_is_not_mislabelled_as_full(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    write_ply(raw, 4, 42, 2)
    columns = {k: v for k, v in read_ply(raw).items() if not k.startswith("f_rest_")}
    full, reason = full_sh_columns(columns, raw)
    assert full is None and reason is not None and "degree-3" in reason
