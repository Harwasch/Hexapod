"""Synthetic geometry verifies masks, lifting, counts and metric bounds together."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hex_analysis.cli import analyze
from hex_analysis.fixture import FixtureSegmenter, load_fixture
from hex_analysis.lifting import lift_depth

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("bundle", ["yard-small", "yard-limited"])
def test_fixture_probes_count_and_height(bundle: str, tmp_path: Path) -> None:
    directory = ROOT / "apps/ios/Fixtures/bundles" / bundle
    truth = json.loads((directory / "ground_truth.json").read_text())
    # Declared accuracy limit: 0.30 m, roughly two depth rows at 6 m / 40.8 px
    # focal length. Independent probes constrain calibration/rounding separately.
    for index in range(24):
        frame, _ = load_fixture(directory, index)
        assert frame.depth is not None
        masks = {d.label: d.mask for d in FixtureSegmenter(truth).segment(frame, "all")}
        for probe in (p for p in truth["probes"] if p["frame"] == index):
            mask = np.zeros_like(frame.depth, dtype=np.bool_)
            u, v = probe["pixel"]
            mask[v, u] = True
            point = lift_depth(
                frame.depth,
                mask,
                frame.intrinsics,
                frame.transform,
                frame.image_size,
                pixel_centres=frame.pixel_centres,
            )[0]
            np.testing.assert_allclose(point, probe["world"], atol=0.005, rtol=0)
            for label, instance_mask in masks.items():
                expected = label == probe["label"] or (
                    label == "tree" and probe["label"] in ("trunk", "crown")
                )
                assert bool(instance_mask[v, u]) == expected
        entities = analyze("count_measure", directory, "tree", index, tmp_path)
        assert len(entities) == truth["counts"]["tree"]
        tree = entities[0]
        assert tree["attrs"]["instanceCountInReferenceFrame"] == truth["counts"]["tree"]
        assert abs(tree["extent"]["y"] - truth["treeHeight"]) <= 0.30
        assert tree["attrs"]["synthetic"] is True
        assert (tmp_path / tree["masks"][0]["ref"]).is_file()
        sheds = analyze("count_measure", directory, "shed", index, tmp_path)
        assert len(sheds) == truth["counts"]["shed"]
        assert abs(sheds[0]["extent"]["y"] - truth["objects"][1]["extents"][1]) <= 0.30


@pytest.mark.parametrize("module", ["segment", "plant_id", "invasive_lookup", "count_measure"])
def test_each_module_runs_alone(module: str, tmp_path: Path) -> None:
    directory = ROOT / "apps/ios/Fixtures/bundles/yard-small"
    entities = analyze(module, directory, "tree", 2, tmp_path)
    assert len(entities) == 1
    assert entities[0]["source"] == {"module": module, "model": "fixture"}
    if module == "plant_id":
        assert entities[0]["attrs"]["taxon"] is None
    if module == "invasive_lookup":
        assert entities[0]["attrs"]["invasive"]["listed"] is None
    assert analyze(module, directory, "unseen object", 2, tmp_path) == []


def test_real_backends_remain_explicitly_unavailable(tmp_path: Path) -> None:
    from hex_analysis.registry import AnalysisUnavailableError

    with pytest.raises(AnalysisUnavailableError):
        analyze(
            "segment",
            ROOT / "apps/ios/Fixtures/bundles/yard-small",
            "tree",
            2,
            tmp_path,
            backend="real",
        )


@pytest.mark.parametrize("module", ["segment", "plant_id", "invasive_lookup", "count_measure"])
def test_cli_prints_entities_json(module: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hex_analysis.cli",
            module,
            "--bundle",
            "apps/ios/Fixtures/bundles/yard-small",
            "--prompt",
            "tree",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT / "tools/analysis",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr == ""
    assert json.loads(result.stdout)[0]["source"]["module"] == module
