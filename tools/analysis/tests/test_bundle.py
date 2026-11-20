from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from hex_analysis.bundle import context, read_frames


def record() -> dict[str, Any]:
    return {
        "index": 0,
        "intrinsics": np.eye(3).ravel().tolist(),
        "transform": np.eye(4).ravel().tolist(),
        "imageWidth": 2,
        "imageHeight": 2,
        "depth": {"file": "depth-0.bin", "width": 2, "height": 2},
    }


def ref() -> dict[str, Any]:
    return {
        "index": 0,
        "image": "frame-000000.jpg",
        "pose": "frames.jsonl#0",
        "depth": "depth-0.bin",
    }


def test_video_keyframe_depth_and_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    np.array([[0.0, 2], [np.nan, 4]], dtype="<f2").tofile(tmp_path / "depth-0.bin")

    def decode(argv: list[str], **kwargs: Any) -> None:
        assert argv[-1] == str(tmp_path / "frame-000000.jpg")
        assert "select=eq(n\\,0)" in argv
        assert kwargs["timeout"] == 90
        Image.new("RGB", (2, 2)).save(argv[-1])

    monkeypatch.setattr("hex_analysis.bundle.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("hex_analysis.bundle.subprocess.run", decode)
    frames = read_frames(tmp_path, [record()], [ref()])
    assert len(frames) == 1
    assert frames[0].depth is not None
    np.testing.assert_array_equal(frames[0].depth, [[0, 2], [np.nan, 4]])
    content = context(frames)
    assert '"validPixels": 2' in content[0]["text"]
    assert '"minM": 2.0' in content[0]["text"]
    assert content[1]["source"]["media_type"] == "image/jpeg"


@pytest.mark.parametrize(
    "change",
    [
        {"index": -1},
        {"pose": "frames.jsonl#1"},
        {"image": "../../escape.jpg"},
        {"depth": "../../escape.bin"},
    ],
)
def test_references_cannot_leave_their_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    decode = Mock()
    monkeypatch.setattr("hex_analysis.bundle.subprocess.run", decode)
    # A wrong depth reference must be checked before decoding anything.
    with pytest.raises(ValueError):
        read_frames(tmp_path, [record()], [{**ref(), **change}])
    decode.assert_not_called()


def test_frame_resource_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eight"):
        read_frames(tmp_path, [record()], [ref()] * 9)
    with pytest.raises(ValueError, match="dimensions"):
        read_frames(tmp_path, [{**record(), "imageWidth": 10_000_000}], [ref()])
