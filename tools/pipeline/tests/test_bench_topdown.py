import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from experiments.bench import __main__ as cli
from experiments.bench import ledger, worker
from experiments.bench.topdown_maths import frame_camera


def test_topdown_central_ninety_percent_fits_and_looks_straight_down() -> None:
    points = np.array([[x, y, 0.2 * x] for x in range(-5, 5) for y in range(-5, 5)])
    points[-10:] = [1000, 1000, 1000]
    camera = frame_camera(points, [], [0, 0, 1])
    view = np.array(camera["viewmat"])
    intrinsic = np.array(camera["K"])
    assert camera["framedCentres"] == 90
    assert np.allclose(view[:3, :3] @ view[:3, :3].T, np.eye(3))
    assert np.linalg.det(view[:3, :3]) == pytest.approx(1)
    assert np.array_equal(view[2, :3], [0, 0, -1])
    projected = points[:90] @ view[:3, :3].T + view[:3, 3]
    pixels = projected @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2, None]
    assert np.all((pixels > 0) & (pixels < 512))
    assert np.all(projected[:, 2] > camera["near"])
    assert np.all(projected[:, 2] < camera["far"])


def test_topdown_camera_plane_fallback_and_degenerate_failure() -> None:
    points = [[-1, -1, 0], [1, 1, 0], [1, -1, 0], [-1, 1, 0]]
    cameras = [[-2, -2, 3], [2, -2, 3], [2, 2, 3], [-2, 2, 3]]
    framed = frame_camera(points, cameras, None)
    assert framed["upMethod"] == "camera-plane"
    assert np.allclose(framed["up"], [0, 0, 1])
    with pytest.raises(ValueError, match="define a plane"):
        frame_camera(points, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], None)
    with pytest.raises(ValueError, match="no projected extent"):
        frame_camera([[0, 0, 0]], [], [0, 0, 1])
    with pytest.raises(ValueError, match="nonfinite"):
        frame_camera([[float("nan"), 0, 0]], [], [0, 0, 1])


def test_topdown_timeout_is_nonfatal(tmp_path: Path) -> None:
    outcome = worker.topdown_preview(tmp_path, tmp_path / "missing.ply", tmp_path, {}, 0)
    assert outcome["status"] == "failed"
    assert "TimeoutError" in outcome["warning"]


def test_local_topdown_png_return_and_optional_invalid_payload(tmp_path: Path) -> None:
    png = io.BytesIO()
    Image.new("RGB", (2, 2)).save(png, format="PNG")
    book = tmp_path / "ledger.jsonl"
    for run, payload in (
        ("good", base64.b64encode(png.getvalue()).decode()),
        ("bad", "not base64"),
    ):
        plan = {"runId": run, "scene": "truck", "schedule": "screen", "tier": "l4"}
        ledger.reserve(book, [plan], 2)
        result: dict[str, Any] = {
            "status": "complete",
            "costUsd": 0.1,
            "topdown": {"status": "complete", "pngBase64": payload, "path": "../../bad"},
        }
        cli.save_outcome(tmp_path, book, plan, result)
        assert result["status"] == "complete"
        assert "pngBase64" not in result["topdown"]
        assert (tmp_path / f"runs/{run}/topdown.png").is_file() == (run == "good")
    assert ledger.summary(ledger.read(book))["totalUsd"] == pytest.approx(0.2)
