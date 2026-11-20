from typing import Any

import numpy as np
import pytest

from hex_analysis.registry import (
    AnalysisUnavailableError,
    Detection,
    FixtureInvasiveSource,
    Frame,
    Registry,
)


class Segments:
    def segment(self, frame: Any, prompt: Any) -> Any:
        assert prompt == "plant"
        return [
            Detection(
                "Cortaderia jubata",
                np.array([[True, False], [False, True]]),
                0.8,
                "fixture-segmenter",
            )
        ]


class Plants:
    def identify(self, frame: Any, detection: Any) -> Any:
        return {"taxon": detection.label, "identifier": "fixture"}


def frame(depth: Any = None) -> Any:
    return Frame(
        0, b"image", np.ones((2, 2)) if depth is None else depth, np.eye(3), np.eye(4), (2, 2)
    )


def test_registered_modules_share_measured_contract() -> None:
    registry = Registry(Segments(), Plants())
    masks = []

    def write(index: Any, mask: Any) -> Any:
        masks.append(mask)
        return "analysis/mask.png"

    for name in ("segment", "plant_id", "invasive_lookup", "count_measure"):
        result = registry.run(name, {"prompt": "plant"}, [frame()], {"lat": 35, "lon": -120}, write)
        assert len(result) == 1
        entity = result[0]
        assert entity["anchor"] == {"world": {"x": 0.5, "y": -0.5, "z": -1}}
        assert entity["extent"] == {"x": 1, "y": 1, "z": 0}
        assert entity["attrs"]["validDepthPixels"] == 2
        assert entity["masks"] == [{"keyframe": "frames.jsonl#0", "ref": "analysis/mask.png"}]
        if name == "invasive_lookup":
            assert entity["attrs"]["invasive"]["source"] == "fixture"
        if name == "plant_id":
            assert entity["attrs"]["taxon"] == "Cortaderia jubata"
        if name == "count_measure":
            assert entity["attrs"]["instanceCountInReferenceFrame"] == 1
    assert len(masks) == 4


def test_default_backend_fails_explicitly_and_holes_emit_no_entity() -> None:
    with pytest.raises(AnalysisUnavailableError):
        Registry().run("segment", {"prompt": "plant"}, [frame()], None, lambda *_: "")
    assert (
        Registry(Segments()).run(
            "segment", {"prompt": "plant"}, [frame(np.zeros((2, 2)))], None, lambda *_: ""
        )
        == []
    )
    with pytest.raises(AnalysisUnavailableError):
        Registry(Segments()).run("plant_id", {"prompt": "plant"}, [frame()], None, lambda *_: "")


def test_fixture_coverage_and_unknown_tools() -> None:
    assert FixtureInvasiveSource().lookup(0, 0, "Cortaderia jubata")["listed"] is None
    assert FixtureInvasiveSource().lookup(35, -120, "Cortaderia jubata")["listed"] is True
    with pytest.raises(ValueError):
        Registry().run("shell", {}, [], None, lambda *_: "")
    with pytest.raises(ValueError):
        Registry().run(
            "segment", {"prompt": "plant", "bundleUrl": "ignored"}, [], None, lambda *_: ""
        )
    with pytest.raises(AnalysisUnavailableError):
        Registry().run("segment", {"prompt": "plant"}, [], None, lambda *_: "")
