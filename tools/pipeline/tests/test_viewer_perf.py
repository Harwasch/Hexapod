from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.bench.viewer import (
    frame_stats,
    gpu_stats,
    markdown,
    pacing_warning,
    relative_costs,
    require_renderer,
)
from experiments.bench.viewer_run import inventory, write_results


def row(degree: int = 0, factor: float = 1) -> dict[str, Any]:
    return {
        "degree": degree,
        **frame_stats([0, 1 * factor, 3 * factor, 6 * factor, 10 * factor]),
        **gpu_stats([]),
        "viewport": {"width": 1600, "height": 900},
        "renderer": "NVIDIA T4",
        "splats": 12000,
        "path": {"orbit": 3, "dolly": 1},
        "software": False,
        "flags": ["--disable-gpu-vsync", "--disable-frame-rate-limit"],
        "tileset_bytes": 200000,
    }


def test_statistics_keep_hitches_and_use_nearest_rank_p95() -> None:
    timestamps = [*range(20), 119]
    result = frame_stats(timestamps)
    assert result["frames"] == 20
    assert result["median_ms"] == 1
    assert result["p95_ms"] == 1
    assert result["mean_ms"] == 119 / 20
    assert result["frame_ms"][-1] == 100
    assert frame_stats([0, 2, 6])["median_ms"] == 3
    assert frame_stats([0, 2, 6])["p95_ms"] == 4


@pytest.mark.parametrize("values", [[], [1], [0, 0], [2, 1], [0, float("nan")]])
def test_invalid_frames_refused(values: list[float]) -> None:
    with pytest.raises(ValueError):
        frame_stats(values)


def test_relative_cost_is_always_against_degree_zero() -> None:
    results = relative_costs([row(3, 4), row(0), row(1, 2), row(2, 3)])
    assert [r["degree"] for r in results] == [0, 1, 2, 3]
    assert [r["median_relative_sh0"] for r in results] == [1, 2, 3, 4]
    assert [r["p95_relative_sh0"] for r in results] == [1, 2, 3, 4]


@pytest.mark.parametrize("key", ["viewport", "renderer", "splats", "path", "software", "flags"])
def test_incomparable_runs_refused(key: str) -> None:
    changed = row(1)
    changed[key] = "different"
    with pytest.raises(ValueError, match=key):
        relative_costs([row(), changed])


def test_missing_baseline_duplicate_degree_and_zero_times_refused() -> None:
    for rows in ([row(1)], [row(), row()], [row(), {**row(1), "median_ms": 0}]):
        with pytest.raises(ValueError):
            relative_costs(rows)


def test_gpu_stats_are_device_wide_and_missing_is_not_zero() -> None:
    assert gpu_stats([])["gpu_mem_max_mib"] is None
    assert gpu_stats(
        [
            {"memory_mib": 100, "utilization_pct": 10},
            {"memory_mib": 250, "utilization_pct": 30},
        ]
    ) == {"gpu_samples": 2, "gpu_mem_max_mib": 250, "gpu_util_mean_pct": 20}
    with pytest.raises(ValueError):
        gpu_stats([{"memory_mib": 10, "utilization_pct": 101}])


def test_no_software_gpu_can_silently_pass_hardware_gate() -> None:
    require_renderer("ANGLE (NVIDIA, Vulkan, Tesla T4)", software=False)
    require_renderer("ANGLE (SwiftShader Device)", software=True)
    for name in ("SwiftShader NVIDIA", "llvmpipe", "", "Intel"):
        with pytest.raises(ValueError):
            require_renderer(name, software=False)
    with pytest.raises(ValueError):
        require_renderer("NVIDIA", software=True)


def test_pacing_plateau_is_flagged_without_discarding_frames() -> None:
    assert pacing_warning([i * 1000 / 60 for i in range(25)])
    assert pacing_warning([i * 1000 / 120 for i in range(25)])
    assert pacing_warning([0, 2, 5, 20, 100]) is None


def test_output_contains_raw_frames_ratios_and_software_label(tmp_path: Path) -> None:
    results = [{**row(), "software": True}]
    out = tmp_path / "run"
    write_results(results, out)
    saved = json.loads((out / "sh0.json").read_text())
    assert saved["frame_ms"] == [1, 2, 3, 4]
    assert saved["median_relative_sh0"] == 1
    assert "software renderer, relative only" in markdown(results)
    assert "Median / SH0" in (out / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        write_results(results, out)


def packed(tmp_path: Path, nested: bool = False) -> tuple[Path, dict[str, Any]]:
    directory = tmp_path / "splat" if nested else tmp_path
    directory.mkdir(exist_ok=True)
    (directory / "parent.glb").write_bytes(b"parent")
    (directory / "leaf.glb").write_bytes(b"leaf")
    root = {
        "content": {"uri": "parent.glb"},
        "extras": {"gaussians": 2},
        "boundingVolume": {"sphere": [0, 0, 0, 1]},
        "children": [{"content": {"uri": "leaf.glb"}, "extras": {"gaussians": 10}}],
    }
    (directory / "tileset.json").write_text(json.dumps({"root": root}))
    return directory, root


@pytest.mark.parametrize("nested", [False, True])
def test_inventory_handles_both_packer_layouts_and_excludes_replace_parents(
    tmp_path: Path, nested: bool
) -> None:
    directory, _ = packed(tmp_path, nested)
    result = inventory(tmp_path)
    assert result["splats"] == 10
    assert result["tiles"] == 1
    assert result["tile_bytes"] == 10
    assert result["tileset_bytes"] == sum(p.stat().st_size for p in directory.iterdir())
    assert len(result["sha256"]["leaf.glb"]) == 64


def test_inventory_add_counts_parents_and_refuses_external_tiles(tmp_path: Path) -> None:
    directory, root = packed(tmp_path)
    root["refine"] = "ADD"
    (directory / "tileset.json").write_text(json.dumps({"root": root}))
    assert inventory(tmp_path)["splats"] == 12
    for uri in ("https://example.com/remote.glb", "../outside.glb"):
        root["content"]["uri"] = uri
        (directory / "tileset.json").write_text(json.dumps({"root": root}))
        with pytest.raises(ValueError, match="local files"):
            inventory(tmp_path)
