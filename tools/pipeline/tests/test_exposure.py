"""`exposure`: what the kept frames measure, and what the capture's EXIF says it metered."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import exposure
import frame_meta
from drone_frames import dji_m3e_photo


def test_luma_stats_count_clipped_pixels() -> None:
    grey = np.full((10, 100), 128, dtype=np.uint8)
    grey[:1] = 255  # 10% blown
    grey[1:3] = 0  # 20% crushed
    grey[3, :10] = 250  # the clip level counts
    grey[4, :10] = 5

    mean, high, low = exposure.luma_stats(grey)

    assert (high, low) == (pytest.approx(11.0), pytest.approx(21.0))
    assert mean == pytest.approx(float(grey.mean()))
    assert exposure.luma_stats(np.zeros((0, 0), dtype=np.uint8)) == (0.0, 0.0, 0.0)


def test_linear_mean_is_a_stop_per_doubling_of_light() -> None:
    bright = np.full((4, 4), 200, dtype=np.uint8)
    dim = np.full((4, 4), round(200 * 0.5 ** (1 / 2.2)), dtype=np.uint8)

    ratio = exposure.linear_mean(bright) / exposure.linear_mean(dim)

    assert np.log2(ratio) == pytest.approx(1.0, abs=0.02)


def _records(tmp_path: Path, times: list[float]) -> list[frame_meta.FrameRecord]:
    out = []
    for index, seconds in enumerate(times):
        photo = dji_m3e_photo(
            tmp_path / f"DJI_{index}.JPG", index, size=(64, 48), exposure_s=seconds
        )
        out.append(
            frame_meta.read_photo(
                photo,
                frame_name=f"frame_{index:04d}.jpg",
                upload_root=tmp_path,
                frame_size=(64, 48),
            )
        )
    return out


def test_ev_spread_from_exif(tmp_path: Path) -> None:
    records = _records(tmp_path, [1 / 1000, 1 / 250])

    summary = exposure.summarise(records, {}, clip_pct=5.0, ev_spread=1.0, luma_spread_stops=1.0)

    assert summary["evSpread"] == pytest.approx(2.0, abs=1e-6)
    flags = summary["flags"]
    assert isinstance(flags, dict)
    assert flags["evSwing"] is True
    assert flags["brightnessSwing"] is False  # no frame was measured
    same = exposure.summarise(
        _records(tmp_path / "b", [1 / 1000, 1 / 1000]),
        {},
        clip_pct=5.0,
        ev_spread=1.0,
        luma_spread_stops=1.0,
    )
    assert same["evSpread"] == 0.0
    assert same["flags"]["evSwing"] is False  # type: ignore[index]


def test_clipped_frames_are_named_at_either_end() -> None:
    def record(name: str, high: float, low: float) -> frame_meta.FrameRecord:
        return frame_meta.FrameRecord(name=name, original="o").with_luma(100.0, high, low)

    records = [record("a", 6.0, 0.0), record("b", 0.0, 0.0), record("c", 0.0, 50.0)]

    assert exposure.clipped_frames(records, 5.0) == ["a", "c"]
    summary = exposure.summarise(
        records, {"a": 0.5, "b": 0.5, "c": 0.5}, clip_pct=5.0, ev_spread=1, luma_spread_stops=1
    )
    assert summary["flags"] == {
        "clippedHighlights": ["a"],
        "clippedShadows": ["c"],
        "evSwing": False,
        "brightnessSwing": False,
    }
    assert summary["flaggedFrames"] == 2
    assert "first principles" in str(summary["thresholds"])
